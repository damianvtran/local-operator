"""The per-session Saver chain: verify, splice, prefetch, adopt, fail open (§3.2, §6).

WHERE IT RUNS. ``SessionStreamFn._stream_account_request`` calls
:meth:`SaverChain.apply` for a conversation turn, before the request is measured,
admitted, sized and handed to ``stream_with_failover``. The reduced history is a
provider-neutral ``Message`` list, so every failover hop, credential rotation or
OAuth account switch serialises the SAME reduced history for its own provider
(§7). One chain per stream: a fork builds its own (§7 "Subagents").

THE PROTOCOL, in the order ``apply`` runs it:

1. **Verify** — re-digest the current projected source epoch by epoch; at the
   first mismatch (compaction, prune, fork, ``/clear``, resume) truncate from
   that epoch on and bump ``chain_generation``.
2. **Adopt** — a prefetch result finished since the last turn is adopted only
   when its 4-tuple ``(session_id, chain_generation, frontier,
   uploaded_digest)`` still matches the chain AND the uploaded rows are still
   the source's rows at the frontier; then every §6.1 check runs against that
   source (S-30). A due prefetch still in flight is waited on for at most
   250 ms, then the turn goes without it.
3. **Splice** — ``[epoch messages …] + source[frontier:]``; check 9(b) runs
   over the whole spliced list before any wire repair can mask it.
4. **Prefetch** — when ``len(source) − frontier ≥ call_again_after`` start the
   next prepare in the background; zero latency on the critical path.

FAIL OPEN, ALWAYS (§6). No failure here fails the turn: the chain does not
advance, the existing epochs keep being sent with a longer verbatim tail, and a
session with no epochs is sent unreduced. Each reason CLASS produces one keyed,
translated notice per session; the reason is a closed enum and the server's text
is never shown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from local_operator.saver.canonical import CanonicalError, digest_bytes
from local_operator.saver.client import SaverClient, SaverHTTPError, SaverRefused
from local_operator.saver.projection import (
    ProjectionCache,
    Redaction,
    shape_only_redaction,
    span_bytes,
    verbatim_wire,
)
from local_operator.saver.validate import (
    CONTRACT,
    MAX_OVERSIZED_UPLOAD_BYTES,
    MAX_UPLOAD_BYTES,
    Epoch,
    SaverRejected,
    check_spliced_pairing,
    response_cap,
    validate_response,
)

if TYPE_CHECKING:
    from local_operator.harness.types import ChatRequest
    from local_operator.i18n.messages import Msg

logger = logging.getLogger(__name__)

METHOD = "reduce-v2"
FORMAT = "openai.chat"
#: §6: if a due prepare is still in flight, wait at most this long.
ADOPTION_WAIT_S = 0.25
#: §6: 503 backs off 60 s.
BUSY_BACKOFF_S = 60.0
#: Transport failures (timeout, refused connection, blackhole) back off like a 503.
TRANSPORT_BACKOFF_S = 60.0
#: The §0.4 boundary step: a boundary is ≥ this many rows past the previous one.
BOUNDARY_STEP = 12
#: The server's live window (§3 gap 6): the last 6 rows, extended back to
#: 6,000 characters. Nothing inside it freezes.
LIVE_WINDOW_CHARS = 6000
LIVE_MESSAGES = 6
#: Until the server states its cadence, ask once there is this much new source.
DEFAULT_CALL_AGAIN_AFTER = BOUNDARY_STEP
#: Most ``prior`` entries one request carries (§3).
MAX_PRIOR = 20_000

#: Notice reason classes (closed enum, §6). One notice per class per session.
NOTICE_UNAVAILABLE = "unavailable"
NOTICE_REJECTED = "rejected"
NOTICE_SIGNED_OUT = "signed_out"
NOTICE_NO_CREDIT = "no_credit"
NOTICE_RATE_LIMITED = "rate_limited"
NOTICE_DISABLED = "disabled"
#: Contract r5 S3-6: the message at the frontier is too large to ever upload.
NOTICE_OVERSIZE = "oversize"
#: Contract r6 S3-7 (b): a ``Content-Encoding`` rejection pauses Saver for the
#: PROCESS. The closed reason is ``transport``; its keyed copy is the existing
#: ``notice.unavailable`` ("unavailable right now, so this turn sends your full
#: history"), which says exactly what the user sees. A dedicated key would
#: change the English source and re-open every locale's audit for no gain in
#: meaning. It is still its OWN reason class, so it notices once per session
#: even after a transient ``unavailable`` notice in the same session.
NOTICE_TRANSPORT = "transport"

NoticeSink = Callable[["Msg"], Awaitable[None]]
SpendSink = Callable[[Any], None]


class _ProcessState:
    """Process-scoped pause (§6: 404/403 ⇒ pause for the PROCESS; r6 S3-7 b).

    A route that does not exist on this server, or refuses this client
    outright, will not appear on the next turn of another session either. Nor
    will an encoded answer to an identity request: whatever path rewrites the
    encoding (a proxy, an edge Worker) rewrites it for every session.
    """

    paused = False
    #: r6 S3-7 (b), kept apart from ``paused`` so the pause carries its notice:
    #: the notice is "one per session", so a session that never uploaded still
    #: hears once, on its next due prefetch, why it is sending full history.
    #: (A chain IS one session's stream, so its ``_noticed`` set is the
    #: per-session dedupe.)
    transport_paused = False


@dataclass
class _Prefetch:
    """One in-flight prepare and the turn binding it must still match (§3.2)."""

    session_id: str
    chain_generation: int
    frontier: int
    uploaded_digest: str
    rows: list[dict[str, Any]]
    encoded: list[bytes]
    task: asyncio.Task[Any]
    #: Offset (from ``frontier``) of an over-1-MiB unit this upload carried
    #: (S3-6): a 413 or 503 answering it marks that unit instead of backing off.
    oversized_unit: int | None = None
    #: The upload stopped before a marked ``oversize`` unit (S3-6).
    ended_at_barrier: bool = False


@dataclass
class SaverChain:
    """One conversation's Saver state. Never shared across sessions."""

    client: SaverClient
    session_id: str
    notice: NoticeSink | None = None
    spend: SpendSink | None = None
    redaction: Redaction = field(default_factory=shape_only_redaction)
    tier_hint: Callable[["ChatRequest"], str | None] = lambda request: None

    epochs: list[Epoch] = field(default_factory=list)
    frontier: int = 0
    chain_generation: int = 0
    call_again_after: int = DEFAULT_CALL_AGAIN_AFTER
    #: The canonical bytes of the frozen span, per epoch, as last verified. Kept
    #: so the verify step can compare without re-concatenating every turn.
    _epoch_bytes: list[list[bytes]] = field(default_factory=list)
    _prefetch: _Prefetch | None = None
    _cache: ProjectionCache = field(default_factory=ProjectionCache)
    _paused_until: float = 0.0
    _paused_session: bool = False
    _refreshed_after_401: bool = False
    _noticed: set[str] = field(default_factory=set)
    #: The last turn's outcome, for the badge (LO2) and tests.
    failing_open: bool = False
    #: Set on a 402: the stream re-reads the balance once (§8, "on 402 only").
    needs_availability_refresh: bool = False
    #: Ids of the SOURCE messages frozen into epochs as of the last turn. Read by
    #: the session's prune sweep (§7): its history is a different render from
    #: the request's (live-only injections are absent), so it restricts itself
    #: by message identity, never by this chain's index.
    frozen_ids: frozenset[str] = frozenset()
    #: Contract r5 S3-6: the unit marked ``oversize`` as (absolute position, row
    #: count, digest). No upload ever carries it; it clears when its bytes change.
    _oversize_mark: tuple[int, int, str] | None = None
    #: One successful upload has already ended at the marked unit. ``prepare`` is
    #: a pure function of (span, prior) and the server commits every boundary it
    #: can, so a second upload ending at the same barrier would freeze nothing:
    #: it is never sent (S3-6 "no retries").
    _barrier_spent: bool = False
    #: The last raised (multi-MB) upload: ((position, digest) of its unit, the
    #: source length then). If it froze nothing past the unit, another is sent
    #: only after a further boundary step of conversation, never per turn.
    _raised_attempt: tuple[tuple[int, str], int] | None = None
    _oversize_noticed_pending: bool = False
    #: S3-7 (b): the process is paused for ``transport`` and this session has
    #: not been told yet (``_maybe_prefetch`` is synchronous; ``_apply`` sends).
    _transport_noticed_pending: bool = False

    def __post_init__(self) -> None:
        # A stream not yet bound to a session passes ``None``: keep the
        # credential-shape floor rather than uploading unredacted text.
        if self.redaction is None:
            self.redaction = shape_only_redaction()

    # -- the turn ------------------------------------------------------------

    async def apply(self, request: "ChatRequest") -> "ChatRequest":
        """Return ``request`` with its frozen prefix spliced in, or unchanged.

        Never raises: every failure is a notice and the unreduced request.
        """
        try:
            return await self._apply(request)
        except Exception:  # noqa: BLE001 — fail open is the contract (§6)
            logger.debug("saver: apply failed; sending unreduced", exc_info=True)
            self._reset()
            await self._fail_open(NOTICE_REJECTED)
            return request

    async def _apply(self, request: "ChatRequest") -> "ChatRequest":
        source = list(request.messages)
        rows, encoded = self._cache.project(source, self.redaction)
        self._verify(encoded)
        await self._adopt(rows, encoded)
        spliced = request
        if self.epochs:
            wire = [message for epoch in self.epochs for message in epoch.messages]
            frozen = len(wire)
            wire.extend(source[self.frontier :])
            try:
                check_spliced_pairing(wire)
            except SaverRejected:
                self._reset()
                await self._fail_open(NOTICE_REJECTED)
                wire, frozen = source, 0
            if frozen:
                spliced = request.model_copy(
                    update={"messages": wire, "frozen_prefix_messages": frozen}
                )
        self.frozen_ids = frozenset(message.id for message in source[: self.frontier] if message.id)
        self._maybe_prefetch(rows, encoded, request)
        await self._flush_oversize_notice()
        await self._flush_transport_notice()
        return spliced

    # -- §3.2 step 1: verify ------------------------------------------------

    def _verify(self, encoded: Sequence[bytes]) -> None:
        position = 0
        for index, epoch in enumerate(self.epochs):
            end = position + epoch.source_count
            if end > len(encoded) or list(encoded[position:end]) != self._epoch_bytes[index]:
                self._truncate(index)
                return
            position = end

    def _truncate(self, keep: int) -> None:
        del self.epochs[keep:]
        del self._epoch_bytes[keep:]
        self.frontier = sum(epoch.source_count for epoch in self.epochs)
        self.chain_generation += 1

    def _reset(self) -> None:
        if self.epochs or self.frontier:
            self._truncate(0)

    # -- §3.2 step 3 (adoption half) ---------------------------------------

    async def _adopt(self, rows: list[dict[str, Any]], encoded: list[bytes]) -> None:
        prefetch = self._prefetch
        if prefetch is None:
            return
        if not prefetch.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(prefetch.task), ADOPTION_WAIT_S)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001 — judged below
                pass
            if not prefetch.task.done():
                # Still running: this turn goes without it; adoption is retried on
                # the next turn under the same binding check.
                return
        self._prefetch = None
        try:
            result = prefetch.task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001 — classified, never raised
            if prefetch.oversized_unit is not None and _is_oversize_refusal(exc):
                # S3-6: a 413, or 503 ``saver_busy``, answering an oversized-head
                # upload marks that unit instead of backing off, so a slow
                # reducer can never become a per-turn retry loop. r6 S3-9 adds
                # every timeout, reset and transport error on that upload: a
                # slow UPLINK otherwise re-sends megabytes after each back-off.
                # Marked only when the chain has not moved since the upload, so
                # the offset still names the same bytes.
                if (
                    prefetch.chain_generation == self.chain_generation
                    and prefetch.frontier == self.frontier
                ):
                    self._mark_oversize(prefetch.rows, prefetch.encoded, prefetch.oversized_unit)
                await self._flush_oversize_notice()
                return
            await self._on_failure(exc)
            return
        # Turn binding (§6.1 check 7): the 4-tuple must still match the chain.
        # A result computed for a chain that has since been truncated (/clear,
        # compaction, fork, prune) is discarded without a notice — it is not a
        # failure, the conversation simply moved on.
        if (
            prefetch.session_id != self.session_id
            or prefetch.chain_generation != self.chain_generation
            or prefetch.frontier != self.frontier
        ):
            return
        # The 4th member, ``uploaded_digest``, binds the bytes. Every row the
        # server froze must still be byte-identical in the THEN-CURRENT source
        # (S-30): the upload's trailing rows may legitimately differ (the live
        # turn — a user message the next turn has since answered — is rebuilt
        # every turn), so the digest compared is that of the span the response
        # claims, re-taken from the current source. A rewrite inside it (a prune,
        # an edit) changes those bytes and the result is dropped; §6.1 check 2
        # then re-verifies each epoch digest against these same current bytes.
        claimed = _claimed_span(result.payload, len(prefetch.encoded))
        current_rows = encoded[self.frontier : self.frontier + claimed]
        if len(current_rows) != claimed or list(current_rows) != prefetch.encoded[:claimed]:
            return
        try:
            epochs, advanced = validate_response(
                result.payload,
                uploaded_rows=rows[self.frontier : self.frontier + len(prefetch.rows)],
                uploaded_bytes=encoded[self.frontier : self.frontier + len(prefetch.encoded)],
                verbatim=verbatim_wire,
                id_prefix=f"saver:{self.chain_generation}:{self.frontier}",
            )
        except SaverRejected as exc:
            logger.debug("saver: response rejected (%s)", exc.reason)
            await self._fail_open(NOTICE_REJECTED)
            return
        self._refreshed_after_401 = False
        self.failing_open = False
        if prefetch.ended_at_barrier:
            self._barrier_spent = True
        if prefetch.oversized_unit is not None:
            unit_end = prefetch.oversized_unit + head_unit_len(
                prefetch.rows[prefetch.oversized_unit :]
            )
            if advanced < unit_end:
                unit_bytes = span_bytes(prefetch.encoded[prefetch.oversized_unit : unit_end])
                self._raised_attempt = (
                    (prefetch.frontier + prefetch.oversized_unit, digest_bytes(unit_bytes)),
                    prefetch.frontier + len(prefetch.rows),
                )
            else:
                self._raised_attempt = None
        offset = 0
        for epoch in epochs:
            self.epochs.append(epoch)
            self._epoch_bytes.append(list(prefetch.encoded[offset : offset + epoch.source_count]))
            offset += epoch.source_count
        self.frontier += advanced
        call_again = result.payload.get("call_again_after")
        if isinstance(call_again, int) and not isinstance(call_again, bool) and call_again > 0:
            self.call_again_after = call_again
        self._record_spend(result.payload)

    # -- §3.2 step 3 (prefetch half) ---------------------------------------

    def _maybe_prefetch(
        self, rows: list[dict[str, Any]], encoded: list[bytes], request: "ChatRequest"
    ) -> None:
        if self._prefetch is not None or not self._may_call():
            return
        if len(rows) - self.frontier < self.call_again_after:
            return
        if _ProcessState.transport_paused:
            # S3-7 (b): an upload in this process met an encoded answer. Nothing
            # is sent; ``_apply`` delivers this session's one notice.
            self._transport_noticed_pending = True
            return
        suffix_rows = rows[self.frontier :]
        suffix_bytes = encoded[self.frontier :]
        blocked = self._marked_offset(suffix_rows, suffix_bytes)
        plan = plan_upload(suffix_rows, suffix_bytes, blocked=blocked)
        if plan.blocking_new and plan.blocking_unit is not None:
            # S3-6 hard barrier: a unit no upload limit can hold. The server
            # could never accept it, so mark it, notice once, and never send it.
            self._mark_oversize(suffix_rows, suffix_bytes, plan.blocking_unit)
        if plan.oversized_unit is not None and self._raise_deferred(
            suffix_rows, suffix_bytes, plan.oversized_unit
        ):
            # The last raised upload of this unit froze nothing past it (the
            # server's live window disagreed with the prediction): wait for a
            # boundary step of new conversation before sending megabytes again.
            plan = plan_upload(
                suffix_rows, suffix_bytes, blocked=blocked, no_raise_at=plan.oversized_unit
            )
        at_barrier = plan.blocking_unit is not None
        if plan.count < 2 or (
            at_barrier
            and (self._barrier_spent or predicted_boundaries(suffix_rows[: plan.count])[-1] == 0)
        ):
            # Nothing uploadable before a barrier, or nothing that would freeze
            # (§0.4, with the live window). No call is made: an upload that
            # cannot freeze is a fee for nothing and, at a stuck barrier, a
            # per-turn retry loop. ``_barrier_spent`` is the backstop when the
            # prediction is wrong: one upload per barrier, ever.
            return
        upload_rows = list(suffix_rows[: plan.count])
        upload_bytes = list(suffix_bytes[: plan.count])
        prior: list[str] = []
        for epoch in self.epochs:
            prior.extend(epoch.index)
        if len(prior) > MAX_PRIOR:
            # Dedup is an optimisation, never correctness (§3).
            prior = []
        body = {
            "contract": CONTRACT,
            "format": FORMAT,
            "method": METHOD,
            "prior": prior,
            "messages": upload_rows,
        }
        uploaded_digest = digest_bytes(span_bytes(upload_bytes))
        task = asyncio.create_task(
            self.client.prepare(
                body,
                session_id=self.session_id,
                idempotency_key=uuid.uuid4().hex,
                tier_hint=self.tier_hint(request),
                max_response_bytes=response_cap(plan.limit),
            )
        )
        self._prefetch = _Prefetch(
            session_id=self.session_id,
            chain_generation=self.chain_generation,
            frontier=self.frontier,
            uploaded_digest=uploaded_digest,
            rows=upload_rows,
            encoded=upload_bytes,
            task=task,
            oversized_unit=plan.oversized_unit,
            ended_at_barrier=at_barrier,
        )

    def _marked_offset(
        self, suffix_rows: list[dict[str, Any]], suffix_bytes: list[bytes]
    ) -> int | None:
        """Where the ``oversize`` unit sits in the current suffix, or ``None``.

        The mark is the unit's absolute position, its row count and the digest of
        its canonical bytes. "The mark clears only when the source at the frontier
        changes": here, when that unit's bytes change (a prune rewrote it) or it
        is no longer in the unfrozen suffix (a truncation moved the frontier past
        or before it in a way that changes its offset's content).
        """
        mark = self._oversize_mark
        if mark is None:
            return None
        position, length, unit_digest = mark
        offset = position - self.frontier
        if (
            offset < 0
            or offset + length > len(suffix_rows)
            or (digest_bytes(span_bytes(suffix_bytes[offset : offset + length])) != unit_digest)
        ):
            self._oversize_mark = None
            self._barrier_spent = False
            return None
        return offset

    def _unit_key(
        self, suffix_rows: list[dict[str, Any]], suffix_bytes: list[bytes], offset: int
    ) -> tuple[int, str]:
        length = head_unit_len(suffix_rows[offset:])
        data = span_bytes(suffix_bytes[offset : offset + length])
        return (self.frontier + offset, digest_bytes(data))

    def _raise_deferred(
        self, suffix_rows: list[dict[str, Any]], suffix_bytes: list[bytes], offset: int
    ) -> bool:
        attempt = self._raised_attempt
        if attempt is None:
            return False
        key, rows_then = attempt
        if key != self._unit_key(suffix_rows, suffix_bytes, offset):
            self._raised_attempt = None
            return False
        return self.frontier + len(suffix_rows) < rows_then + BOUNDARY_STEP

    def _mark_oversize(
        self, suffix_rows: list[dict[str, Any]], suffix_bytes: list[bytes], offset: int
    ) -> None:
        length = head_unit_len(suffix_rows[offset:])
        unit_digest = digest_bytes(span_bytes(suffix_bytes[offset : offset + length]))
        mark = (self.frontier + offset, length, unit_digest)
        if mark != self._oversize_mark:
            self._oversize_mark = mark
            self._barrier_spent = False
            self._oversize_noticed_pending = True

    async def _flush_oversize_notice(self) -> None:
        if self._oversize_noticed_pending:
            self._oversize_noticed_pending = False
            await self._fail_open(NOTICE_OVERSIZE)

    # -- §6: back-off, notices ------------------------------------------------

    def _may_call(self) -> bool:
        if _ProcessState.paused or self._paused_session:
            return False
        return time.monotonic() >= self._paused_until

    async def _on_failure(self, exc: BaseException) -> None:
        if isinstance(exc, SaverRefused):
            if exc.reason == "signed_out":
                self._paused_session = True
                await self._fail_open(NOTICE_SIGNED_OUT)
            else:
                # A non-canonical base or a credential that equals a provider
                # key: nothing will change within this session.
                self._paused_session = True
                await self._fail_open(NOTICE_DISABLED)
            return
        if isinstance(exc, SaverHTTPError):
            status = exc.status
            if status == 401:
                if not self._refreshed_after_401:
                    # ONE refresh through the shared single-flight store, then
                    # pause for the session (never an interactive prompt from a
                    # background prefetch).
                    self._refreshed_after_401 = True
                    try:
                        await self.client.refresh()
                    except Exception:  # noqa: BLE001 — the pause below covers it
                        pass
                    return
                self._paused_session = True
                await self._fail_open(NOTICE_SIGNED_OUT)
            elif status == 402:
                self._paused_session = True
                self.needs_availability_refresh = True
                await self._fail_open(NOTICE_NO_CREDIT)
            elif status in (403, 404):
                _ProcessState.paused = True
                await self._fail_open(NOTICE_DISABLED)
            elif status == 429:
                self._paused_until = time.monotonic() + (exc.retry_after_s or BUSY_BACKOFF_S)
                await self._fail_open(NOTICE_RATE_LIMITED)
            elif 300 <= status < 400:
                # A redirect is never followed; treat the route as absent here.
                _ProcessState.paused = True
                await self._fail_open(NOTICE_DISABLED)
            else:
                self._paused_until = time.monotonic() + BUSY_BACKOFF_S
                await self._fail_open(NOTICE_UNAVAILABLE)
            return
        if isinstance(exc, SaverRejected):
            if exc.reason == "bad_encoding":
                # r6 S3-7 (b): deterministic, like 404/403 — the path that
                # encoded this answer will encode the next one too.
                _ProcessState.transport_paused = True
                await self._fail_open(NOTICE_TRANSPORT)
                return
            await self._fail_open(NOTICE_REJECTED)
            return
        if isinstance(exc, (httpx.HTTPError, OSError, asyncio.TimeoutError, CanonicalError)):
            self._paused_until = time.monotonic() + TRANSPORT_BACKOFF_S
            await self._fail_open(NOTICE_UNAVAILABLE)
            return
        self._paused_until = time.monotonic() + TRANSPORT_BACKOFF_S
        await self._fail_open(NOTICE_UNAVAILABLE)

    async def _flush_transport_notice(self) -> None:
        if self._transport_noticed_pending:
            self._transport_noticed_pending = False
            await self._fail_open(NOTICE_TRANSPORT)

    async def _fail_open(self, reason: str) -> None:
        self.failing_open = True
        if reason in self._noticed or self.notice is None:
            return
        self._noticed.add(reason)
        try:
            await self.notice(notice_message(reason))
        except Exception:  # noqa: BLE001 — a notice must never fail the turn
            logger.debug("saver: notice delivery failed", exc_info=True)

    # -- spend ----------------------------------------------------------------

    def _record_spend(self, payload: Any) -> None:
        if self.spend is None:
            return
        try:
            record = spend_record(payload, session_id=self.session_id)
        except Exception:  # noqa: BLE001 — accounting must never fail the turn
            logger.debug("saver: spend record failed", exc_info=True)
            return
        if record is not None:
            try:
                self.spend(record)
            except Exception:  # noqa: BLE001 — same
                logger.debug("saver: spend sink failed", exc_info=True)

    async def aclose(self) -> None:
        prefetch, self._prefetch = self._prefetch, None
        if prefetch is not None and not prefetch.task.done():
            prefetch.task.cancel()
            try:
                await prefetch.task
            except BaseException:  # noqa: BLE001 — teardown
                pass


def _claimed_span(payload: Any, uploaded: int) -> int:
    """``advanced`` as the response states it, clamped to the upload; 0 if unusable.

    Read only to choose WHICH current-source bytes to compare at adoption; the
    response is still fully validated (§6.1) against those bytes afterwards.
    """
    advanced = payload.get("advanced") if isinstance(payload, dict) else None
    if not isinstance(advanced, int) or isinstance(advanced, bool) or advanced < 0:
        return 0
    return min(advanced, uploaded)


def _is_oversize_refusal(exc: BaseException) -> bool:
    """A 413 or a 503 (``saver_busy``) — the two answers S3-6 turns into the mark.

    The 503's body is never read (S-31), so a 503 answering an oversized-head
    upload is treated as ``saver_busy``: the contract's rule is that such an
    upload is not retried either way.

    Contract r6 S3-9 widens it to any timeout, reset or other transport error
    (``httpx.TransportError`` covers write/read/pool timeouts, connect errors
    and ``RemoteProtocolError``; ``OSError`` a reset surfacing raw). The
    client's limits are per operation (§6: 3 s, connect 0.5 s): a 2.5 MiB POST
    at 2 Mbit/s raised ``WriteTimeout`` after 3.1 s on a loopback throttle
    (measured), and each 60 s back-off would otherwise end in another
    multi-MiB upload. Only the oversized-head upload is marked; an ordinary
    upload's transport failure keeps its timed back-off.
    """
    if isinstance(exc, SaverHTTPError):
        return exc.status in (413, 503)
    return isinstance(exc, (httpx.TransportError, OSError, asyncio.TimeoutError))


def head_unit_len(rows: Sequence[dict[str, Any]]) -> int:
    """Rows in the pairing-safe unit at the head of ``rows`` (contract r5, S3-6).

    "The message plus its call or result partner": an assistant row with tool
    calls extends through the tool rows that answer them, so the unit is the
    shortest prefix after which no call is still open. A head that is not a
    call is one row. A call never answered within ``rows`` makes the whole
    remainder the unit, which is then never cut mid-pair.
    """
    pending: set[str] = set()
    for index, row in enumerate(rows):
        for call in row.get("tool_calls") or []:
            pending.add(str(call.get("id")))
        if row.get("role") == "tool":
            pending.discard(str(row.get("tool_call_id")))
        if not pending:
            return index + 1
    return len(rows)


@dataclass(frozen=True)
class UploadPlan:
    """How much of the unfrozen suffix one prepare uploads (§6.1 check 1).

    ``count`` rows from the frontier, under the upload limit ``limit`` (U).
    ``oversized_unit`` is the offset of an over-1-MiB pairing-safe unit the
    upload CARRIES (U grew for it, S3-6): a 413 or 503 answering this upload
    marks that unit. ``blocking_unit`` is the offset of an over-1-MiB unit no U
    can hold (or one already marked): the upload stops before it, and
    ``blocking_new`` says the unit was found unfittable by size on this plan
    (the notice's trigger).
    """

    count: int
    limit: int
    oversized_unit: int | None = None
    blocking_unit: int | None = None
    blocking_new: bool = False


def plan_upload(
    rows: Sequence[dict[str, Any]],
    encoded: Sequence[bytes],
    *,
    blocked: int | None = None,
    no_raise_at: int | None = None,
) -> UploadPlan:
    """The rows to upload from the unfrozen suffix (§6.1 check 1, contract r5 S3-6).

    Walked one pairing-safe unit at a time, so a cut never splits a call from
    its result. Normally U = 1 MiB. A unit that is ITSELF over 1 MiB raises U
    to ``min(8 MiB − 64 KiB, bytes before it + unit + 1 MiB)``: it is uploaded
    with up to 1 MiB of following context, still cut at a message boundary.

    TWO RULES THE CONTRACT TEXT IMPLIES BUT DOES NOT SPELL OUT:

    - Any over-1-MiB unit, not only one exactly at ``source[frontier]``. Server
      boundaries fall on the first safe cut ≥ 12 rows past the last (§0.4), so
      the frontier rarely lands exactly on a large unit; a head-only raise would
      cut before it on every turn and the session would stop freezing for good.
      Every bound the contract derives still holds (U ≤ 8 MiB − 64 KiB, R from U).
    - FREEZABLE before it is sent. The unit rides an upload only once the
      server would commit a boundary past it (:func:`predicted_boundaries`, the
      §0.4 rule with the live window). Before that the upload cuts in front of
      it, so a multi-MB upload is sent once, when it freezes the unit —
      "frozen through one oversized-head upload, with no retry".

    A unit no U can hold, or the ``blocked`` offset (a unit marked after a
    413/503), ends the upload before it and is never sent.
    """
    if not rows:
        return UploadPlan(0, MAX_UPLOAD_BYTES)
    limit = MAX_UPLOAD_BYTES
    oversized: int | None = None
    count = 0
    total = 2
    while count < len(rows):
        size = head_unit_len(rows[count:])
        unit_bytes = sum(len(data) for data in encoded[count : count + size]) + size
        if blocked is not None and count <= blocked < count + size:
            return UploadPlan(count, limit, oversized, blocking_unit=count)
        if total + unit_bytes <= limit:
            total += unit_bytes
            count += size
            continue
        if oversized is not None or unit_bytes <= MAX_UPLOAD_BYTES or count == no_raise_at:
            break  # the normal cut: this unit rides a later prefetch
        if unit_bytes + 2 > MAX_OVERSIZED_UPLOAD_BYTES:
            # The S3-6 hard barrier: no upload limit can ever hold this unit.
            return UploadPlan(count, limit, None, blocking_unit=count, blocking_new=True)
        raised = min(MAX_OVERSIZED_UPLOAD_BYTES, total + unit_bytes + MAX_UPLOAD_BYTES)
        # The rows this raised upload would carry: the unit, then whole units
        # of following context up to the raised limit.
        end = count + size
        carried = total + unit_bytes
        while end < len(rows):
            follow = head_unit_len(rows[end:])
            follow_bytes = sum(len(data) for data in encoded[end : end + follow]) + follow
            if carried + follow_bytes > raised:
                break
            carried += follow_bytes
            end += follow
        if predicted_boundaries(rows[:end])[-1] < count + size:
            # Not freezable yet: the server's live window still covers the
            # first boundary past the unit. Cut in front of it; the unit rides
            # an upload once enough conversation follows it.
            break
        limit = raised
        oversized = count
        count = end
        break
    return UploadPlan(count, limit, oversized)


def _live_size(row: dict[str, Any]) -> int:
    """A row's weight in the live-window rule (§3 gap 6; ``saver.live_start`` in
    the reference): its content's compact JSON length, plus each ``image_ref``'s
    ``bytes`` so a window of screenshots cannot stay live as ~100-char refs."""
    content = row.get("content", "")
    size = len(json.dumps(content, ensure_ascii=False, separators=(",", ":")))
    blocks = content if isinstance(content, list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for ref in [
            block,
            *(block.get("content") or [] if isinstance(block.get("content"), list) else []),
        ]:
            if isinstance(ref, dict) and ref.get("type") == "image_ref":
                value = ref.get("bytes")
                if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                    size += value
    return size


def _live_start(rows: Sequence[dict[str, Any]]) -> int:
    """The oldest live row of ``rows`` (§3 gap 6): the last ``LIVE_MESSAGES``,
    extended back until ``LIVE_WINDOW_CHARS``, never starting right after an
    assistant tool call."""
    total = 0
    cut = max(0, len(rows) - LIVE_MESSAGES)
    for index in range(len(rows) - 1, -1, -1):
        total += _live_size(rows[index])
        if total >= LIVE_WINDOW_CHARS:
            cut = min(cut, index)
            break
    while cut > 0 and rows[cut - 1].get("role") == "assistant" and rows[cut - 1].get("tool_calls"):
        cut -= 1
    return cut


def predicted_boundaries(rows: Sequence[dict[str, Any]]) -> list[int]:
    """The boundaries the server commits for an upload of ``rows`` (§0.4).

    b₀ = 0; bₖ₊₁ = the first tool-pairing-safe cut ≥ bₖ + ``BOUNDARY_STEP``,
    committed while ≤ the live-window start. The contract makes this a pure
    function of the source, so the client can predict it; it is used ONLY to
    decide when a multi-megabyte upload is worth sending, never to trust a
    response (§6.1 still checks every epoch against the client's own bytes).
    """
    if len(rows) < 2:
        return [0]
    live = _live_start(rows)
    pending: set[str] = set()
    safe: set[int] = set()
    for index, row in enumerate(rows):
        for call in row.get("tool_calls") or []:
            pending.add(str(call.get("id")))
        if row.get("role") == "tool":
            pending.discard(str(row.get("tool_call_id")))
        if not pending:
            safe.add(index + 1)
    bounds = [0]
    while True:
        nxt = next((i for i in range(bounds[-1] + BOUNDARY_STEP, live + 1) if i in safe), None)
        if nxt is None:
            return bounds
        bounds.append(nxt)


_SHARED: dict[str, SaverClient] = {}


def shared_client() -> SaverClient:
    """The process-wide Saver client for the configured Radient base URL.

    ONE connection pool per process (the classification service's §5a rule:
    a fresh TLS handshake per call would spend the latency budget). Per-session
    state never lives on it — each conversation's chain holds its own epochs,
    frontier and turn binding (§3.2: "a shared client object never routes a
    result to another session's chain"). The base URL comes from
    ``RADIENT_API_BASE_URL`` exactly as the Radient provider's does.
    """
    from local_operator.env import get_env_config
    from local_operator.paths import config_dir
    from local_operator.saver.client import (
        radient_bearer_source,
        stored_provider_secrets,
    )

    base_url = get_env_config().radient_api_base_url
    client = _SHARED.get(base_url)
    if client is None:
        root = config_dir()
        client = SaverClient(
            base_url,
            bearer=radient_bearer_source(root, base_url),
            forbidden=lambda: stored_provider_secrets(root),
        )
        _SHARED[base_url] = client
    return client


def notice_message(reason: str) -> "Msg":
    """The keyed notice for one reason class (closed enum)."""
    from local_operator.i18n.keys import wire_radient_saver as _rs

    return {
        NOTICE_UNAVAILABLE: _rs.notice_unavailable,
        NOTICE_REJECTED: _rs.notice_rejected,
        NOTICE_SIGNED_OUT: _rs.notice_signed_out,
        NOTICE_NO_CREDIT: _rs.notice_no_credit,
        NOTICE_RATE_LIMITED: _rs.notice_rate_limited,
        NOTICE_DISABLED: _rs.notice_disabled,
        NOTICE_OVERSIZE: _rs.notice_oversize,
        # S3-7 (b): reuses the ``unavailable`` copy (see NOTICE_TRANSPORT).
        NOTICE_TRANSPORT: _rs.notice_unavailable,
    }.get(reason, _rs.notice_unavailable)()


def spend_record(payload: Any, *, session_id: str) -> Any:
    """The ``saver`` channel record for one charged prepare, or ``None``.

    From the response's ``cost`` object (§5a "Echo"): a prepare that charged 0
    carries no object and writes no row, so absence here is "nothing charged",
    and no record is made. The record id is the server's ``usage_record_id`` when
    present — the join key that reconciles this line with ``GET /v1/usage`` —
    so a replayed response can never count twice.
    """
    from local_operator.clients.radient_cost import (
        channel_record_from_cost,
        cost_from_payload,
    )
    from local_operator.session.channel_spend import new_record_id

    cost = cost_from_payload(payload)
    if cost is None or cost.amount_micro is None:
        return None
    return channel_record_from_cost(
        cost,
        channel="saver",
        record_id=new_record_id("saver", cost.usage_record_id),
        session_id=session_id,
        unit="tokens",
        provider="radient",
        ts_ms=int(time.time() * 1000),
    )
