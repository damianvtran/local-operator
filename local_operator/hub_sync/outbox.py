"""Durable outbox and the bounded, abortable network dispatcher.

WHY AN OUTBOX. A hub push reaches the network and (for a generalization or a
moderation pass) a paid model. If the process dies between "we decided to
publish" and "the server accepted it", a bare retry could publish the same
change twice or, worse, re-run a paid stage whose result was already accepted.
So the INTENT is written durably (atomically, under the status lock) BEFORE the
first byte leaves, and the terminal outcome is written back only by the worker
that owns the CURRENT generation of that operation.

WHY A GENERATION (FENCING). A worker can outlive its own deadline: the deadline
aborts the AWAIT, but the thread/task may still complete late. Each dispatch
carries the generation it was created at; ``settle`` refuses a write from a
worker whose generation no longer matches the record, so a late worker cannot
overwrite a newer attempt's result (the classic double-settlement race).

WHY ABORTABLE ASYNC HTTP, NOT ``to_thread`` + ``wait_for``. Cancelling a
``to_thread`` await does NOT stop the underlying blocking ``requests`` call: the
socket keeps reading and the thread keeps running after the deadline, so the
"budget" is a fiction. ``httpx.AsyncClient`` is genuinely cancellable, so
``asyncio.wait_for`` bound to :class:`OperationBudget`'s absolute monotonic
deadline actually aborts the transfer.

NO LOCK ACROSS THE NETWORK. The outbox lock is held only for the short
read-modify-write of the record, never across the ``await`` of a request, a
model call, or a human review: a lock held across the network would wedge every
other writer for as long as the provider takes.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import errno
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import httpx

from local_operator.hub_sync.budget import OperationBudget, OperationDeadlineExceeded
from local_operator.hub_sync.provenance import hub_root, now_iso

logger = logging.getLogger(__name__)

SCHEMA = 1
LOCK_WAIT_S = 2.0

#: Push-side retry policy: a failing push is NOT retried every tick (each retry
#: costs a paid preview), so a failed attempt schedules the next one with
#: exponential backoff and the operation is DEAD-LETTERED after this many
#: attempts. A dead-lettered record is skipped with a named status until a
#: deliberate act (a new manual publication) resets it.
PUSH_MAX_ATTEMPTS = 5
PUSH_RETRY_BASE_S = 900.0
PUSH_RETRY_MAX_S = 21600.0


def push_backoff_seconds(attempts: int) -> float:
    """Exponential backoff for attempt N (1-based), capped at six hours."""

    exponent = max(0, int(attempts) - 1)
    return min(PUSH_RETRY_BASE_S * (2**exponent), PUSH_RETRY_MAX_S)


#: The outbox directory name under the hub root.
OUTBOX_DIR = "outbox"


@dataclass(frozen=True)
class OutboxRecord:
    """One durable operation intent plus its (eventual) settled result."""

    operation_id: str
    kind: str
    local_id: str
    direction: str  # "push" | "pull"
    generation: int
    state: str  # "pending" | "pending-reconciliation" | "done" | "failed"
    created_at: str
    updated_at: str
    result: Mapping[str, Any] | None = None
    error_class: str | None = None
    #: How many attempts this operation has STARTED. A terminal ``done`` resets
    #: it on the next begin; a ``failed`` record carries it into the backoff and
    #: the dead-letter decision.
    attempts: int = 0
    #: When the next attempt may start (ISO). None = immediately eligible.
    next_retry_at: str | None = None
    #: The VERBATIM commit intent (document, preview pin, expected revision,
    #: acknowledgements) written after a preview and BEFORE the commit dispatch.
    #: A ``pending-reconciliation`` replay sends exactly these bytes again --
    #: never a re-projection -- so an uncertain commit can never turn into a
    #: fresh create or a second paid preview.
    intent: Mapping[str, Any] | None = None
    #: A DETERMINISTIC refusal (round-2 R12): nothing about retrying the SAME
    #: input can change the outcome (needs-review, an ack-required preview, an
    #: invalid document), so the record waits for the INPUT to change instead of
    #: burning the dead-letter cap tick after tick. ``begin_attempt`` skips it
    #: while ``input_fp`` is unchanged and re-arms it the moment it differs.
    waiting_input: bool = False
    #: Fingerprint of the projection INPUT this record last attempted (kind,
    #: captured/accepted/local/remote documents, expected revision). It is what
    #: makes "wait for input change" and "a deliberate edit re-arms the cap"
    #: decidable without reading the registries.
    input_fp: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "local_id": self.local_id,
            "direction": self.direction,
            "generation": self.generation,
            "state": self.state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": dict(self.result) if self.result is not None else None,
            "error_class": self.error_class,
            "attempts": self.attempts,
            "next_retry_at": self.next_retry_at,
            "intent": dict(self.intent) if self.intent is not None else None,
            "waiting_input": self.waiting_input,
            "input_fp": self.input_fp,
        }


@dataclass(frozen=True)
class Attempt:
    """The right to run ONE push attempt, plus its fencing generation.

    ``replay`` is the stored verbatim intent when the previous attempt ended
    UNCERTAIN (``pending-reconciliation``): the caller must resend exactly it,
    never re-project. ``None`` means a normal (or fresh) attempt.
    """

    generation: int
    attempts: int
    replay: Mapping[str, Any] | None = None


class Outbox:
    """Durable per-operation records on disk, written atomically under an flock."""

    def __init__(self, config_dir: Path) -> None:
        self.config_dir = Path(config_dir)
        self.dir = hub_root(self.config_dir) / OUTBOX_DIR
        self._lock_path = self.dir / ".outbox.lock"

    # -- paths ---------------------------------------------------------------

    def _path(self, operation_id: str) -> Path:
        # The id is a validated opaque token elsewhere; refuse a path-shaped one
        # here so a crafted id can never escape the outbox directory.
        if not operation_id or "/" in operation_id or "\\" in operation_id or ".." in operation_id:
            raise ValueError("invalid outbox operation id")
        return self.dir / f"{operation_id}.json"

    # -- lock ----------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> Iterator[bool]:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            import fcntl

            deadline = time.monotonic() + LOCK_WAIT_S
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                        raise
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
            yield acquired
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _atomic_write(self, record: OutboxRecord) -> None:
        path = self._path(record.operation_id)
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{record.operation_id}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record.to_json(), handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # -- reads ---------------------------------------------------------------

    def read(self, operation_id: str) -> OutboxRecord | None:
        try:
            raw = json.loads(self._path(operation_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            return None
        try:
            return OutboxRecord(
                operation_id=str(raw["operation_id"]),
                kind=str(raw["kind"]),
                local_id=str(raw["local_id"]),
                direction=str(raw["direction"]),
                generation=int(raw["generation"]),
                state=str(raw["state"]),
                created_at=str(raw.get("created_at") or ""),
                updated_at=str(raw.get("updated_at") or ""),
                result=raw.get("result"),
                error_class=raw.get("error_class"),
                attempts=int(raw.get("attempts") or 0),
                next_retry_at=(
                    str(raw["next_retry_at"]) if raw.get("next_retry_at") is not None else None
                ),
                intent=raw.get("intent") if isinstance(raw.get("intent"), Mapping) else None,
                waiting_input=bool(raw.get("waiting_input") or False),
                input_fp=str(raw["input_fp"]) if raw.get("input_fp") is not None else None,
            )
        except (KeyError, TypeError, ValueError):
            return None

    # -- writes --------------------------------------------------------------

    def enqueue(
        self, *, operation_id: str, kind: str, local_id: str, direction: str, generation: int
    ) -> OutboxRecord:
        """Write the INTENT durably before any dispatch. Raises on a failed write.

        A failure to persist the intent must stop the dispatch: publishing
        without a durable record is exactly the double-publish this exists to
        prevent, so the caller lets this raise rather than proceeding.
        """
        at = now_iso()
        record = OutboxRecord(
            operation_id=operation_id,
            kind=kind,
            local_id=local_id,
            direction=direction,
            generation=generation,
            state="pending",
            created_at=at,
            updated_at=at,
        )
        with self._locked() as held:
            if not held:
                raise TimeoutError("hub outbox lock busy; intent not persisted")
            self._atomic_write(record)
        return record

    def settle(
        self,
        operation_id: str,
        generation: int,
        *,
        result: Mapping[str, Any] | None = None,
        error_class: str | None = None,
        next_retry_at: str | None = None,
        refund_attempt: bool = False,
        waiting_input: bool = False,
    ) -> bool:
        """Record the outcome, fenced by generation. False = rejected or skipped.

        FENCING: a worker whose ``generation`` is not the record's current
        generation is STALE (a newer attempt superseded it) and its settlement is
        discarded rather than overwriting the newer result.

        ``next_retry_at`` is the push-side backoff schedule for a failed attempt
        (``None`` = eligible immediately); the attempt count is carried so the
        dead-letter decision has the full history.

        ``refund_attempt`` gives the attempt back (R12): outcomes that are NOT
        the client's failure -- a deterministic refusal that waits for input, or
        a 412 from a concurrent remote edit -- must not walk a healthy link
        toward the dead-letter cap. ``waiting_input`` marks a deterministic
        refusal whose retry needs the projection INPUT to change; with it set the
        record carries no backoff and ``_skip_reason`` answers
        ``awaiting-input-change``.
        """
        with self._locked() as held:
            if not held:
                return False
            record = self.read(operation_id)
            if record is None or record.generation != generation:
                return False
            if record.state not in ("pending", "pending-reconciliation"):
                # Already settled (or never started); a replay is idempotent, not
                # a rewrite. A pending-reconciliation record IS settleable: its
                # replay is the worker that owns that same generation.
                return False
            attempts = max(0, record.attempts - 1) if refund_attempt else record.attempts
            settled = OutboxRecord(
                operation_id=record.operation_id,
                kind=record.kind,
                local_id=record.local_id,
                direction=record.direction,
                generation=record.generation,
                state="failed" if error_class else "done",
                created_at=record.created_at,
                updated_at=now_iso(),
                result=dict(result) if result is not None else None,
                error_class=error_class,
                attempts=attempts,
                next_retry_at=(next_retry_at if error_class and not waiting_input else None),
                intent=None,
                waiting_input=bool(waiting_input),
                input_fp=record.input_fp,
            )
            self._atomic_write(settled)
        return True

    def store_intent(self, operation_id: str, generation: int, intent: Mapping[str, Any]) -> bool:
        """Persist the VERBATIM commit intent before the commit dispatch.

        Written after a preview produced its pin/document and BEFORE the commit
        leaves, so a crash or deadline in between leaves a replayable record: the
        next attempt resends exactly these bytes (same pin, same body, same
        expected revision) instead of re-projecting or paying a second preview.
        """
        with self._locked() as held:
            if not held:
                return False
            record = self.read(operation_id)
            if record is None or record.generation != generation or record.state != "pending":
                return False
            nxt = OutboxRecord(
                operation_id=record.operation_id,
                kind=record.kind,
                local_id=record.local_id,
                direction=record.direction,
                generation=record.generation,
                state=record.state,
                created_at=record.created_at,
                updated_at=now_iso(),
                result=record.result,
                error_class=record.error_class,
                attempts=record.attempts,
                next_retry_at=record.next_retry_at,
                intent=dict(intent),
                input_fp=record.input_fp,
            )
            self._atomic_write(nxt)
        return True

    def begin_attempt(
        self,
        operation_id: str,
        *,
        kind: str,
        local_id: str,
        max_attempts: int = PUSH_MAX_ATTEMPTS,
        now: str | None = None,
        input_fp: str | None = None,
    ) -> "Attempt | None":
        """Start (or resume) an attempt; ``None`` = skip this tick.

        The state machine, in ONE locked read-modify-write (``now`` and the
        stored ``next_retry_at`` are compared as strings in the SAME
        ``%Y-%m-%dT%H:%M:%SZ`` spelling):

        * no record -> fresh enqueue, attempt 1;
        * ``pending-reconciliation`` -> an UNCERTAIN prior attempt: resume with the
          SAME generation and hand back its stored intent for a verbatim replay
          (no new projection, no new preview); an intent-less legacy record waits
          for its INPUT to change (a changed projection may start fresh) rather
          than re-projecting an unchanged one into a fresh create;
        * ``done`` -> a NEW authored change: reset the attempt counter, fence the
          old worker with a new generation;
        * ``failed`` -> a DETERMINISTIC refusal (``waiting_input``) is skipped
          while the input fingerprint is unchanged and re-armed the moment it
          differs (retrying the same bytes cannot change the answer); a
          RETRYABLE failure waits out ``next_retry_at`` and the attempt cap;
          either way a CHANGED fingerprint resets the counter, so a deliberate
          new edit re-arms a capped operation (round-2 R12).
        """
        at = now or now_iso()
        with self._locked() as held:
            if not held:
                raise TimeoutError("hub outbox lock busy")
            record = self.read(operation_id)
            if record is None:
                record = OutboxRecord(
                    operation_id=operation_id,
                    kind=kind,
                    local_id=local_id,
                    direction="push",
                    generation=1,
                    state="pending",
                    created_at=at,
                    updated_at=at,
                    attempts=1,
                )
                self._atomic_write(dataclasses.replace(record, input_fp=input_fp))
                return Attempt(generation=1, attempts=1)
            changed = input_fp is not None and input_fp != record.input_fp
            if record.state == "pending-reconciliation":
                if not isinstance(record.intent, Mapping):
                    if not changed:
                        return None  # nothing to replay; wait for the input to move
                    return self._begin(record, attempts=1, at=at, input_fp=input_fp)
                return Attempt(
                    generation=record.generation,
                    attempts=record.attempts,
                    replay=record.intent,
                )
            if record.state == "pending":
                # An abandoned attempt (crash between begin and settle): fence it.
                attempts = record.attempts + 1
                if attempts > max_attempts and not changed:
                    return None
                return self._begin(
                    record, attempts=1 if changed else attempts, at=at, input_fp=input_fp
                )
            if record.state == "done":
                return self._begin(record, attempts=1, at=at, input_fp=input_fp)
            # failed
            if record.waiting_input and not changed:
                return None  # deterministic refusal: skip until the input changes
            if record.attempts >= max_attempts and not changed:
                return None
            if not changed and record.next_retry_at is not None and at < record.next_retry_at:
                return None  # backoff window: skip this tick without burning an attempt
            return self._begin(
                record, attempts=1 if changed else record.attempts + 1, at=at, input_fp=input_fp
            )

    def _begin(
        self, record: OutboxRecord, *, attempts: int, at: str, input_fp: str | None = None
    ) -> "Attempt":
        nxt = OutboxRecord(
            operation_id=record.operation_id,
            kind=record.kind,
            local_id=record.local_id,
            direction=record.direction,
            generation=record.generation + 1,
            state="pending",
            created_at=record.created_at,
            updated_at=at,
            attempts=attempts,
            input_fp=input_fp if input_fp is not None else record.input_fp,
        )
        self._atomic_write(nxt)
        return Attempt(generation=nxt.generation, attempts=attempts)

    def mark_pending_reconciliation(self, operation_id: str, generation: int) -> bool:
        """Mark an UNCERTAIN outcome (aborted dispatch) for replay, fenced by generation.

        A timeout does not prove the server did or did not accept the write, so it
        is not settled as failed. The record stays replayable with the SAME
        operation id/pin/body; state ``pending-reconciliation`` distinguishes it
        from a fresh ``pending`` so a replay is never treated as a new create.
        The stored ``intent`` (when present) is what makes the replay verbatim.
        """
        with self._locked() as held:
            if not held:
                return False
            record = self.read(operation_id)
            if record is None or record.generation != generation:
                return False
            if record.state not in ("pending", "pending-reconciliation"):
                return False
            nxt = OutboxRecord(
                operation_id=record.operation_id,
                kind=record.kind,
                local_id=record.local_id,
                direction=record.direction,
                generation=record.generation,
                state="pending-reconciliation",
                created_at=record.created_at,
                updated_at=now_iso(),
                result=record.result,
                error_class=record.error_class,
                attempts=record.attempts,
                next_retry_at=record.next_retry_at,
                intent=record.intent,
                input_fp=record.input_fp,
            )
            self._atomic_write(nxt)
        return True

    def bump_generation(self, operation_id: str) -> int:
        """Start a new attempt at the same operation id; returns the new generation.

        The previous worker's settlement is fenced out from this point on.
        """
        with self._locked() as held:
            if not held:
                raise TimeoutError("hub outbox lock busy")
            record = self.read(operation_id)
            if record is None:
                raise KeyError(operation_id)
            nxt = OutboxRecord(
                operation_id=record.operation_id,
                kind=record.kind,
                local_id=record.local_id,
                direction=record.direction,
                generation=record.generation + 1,
                state="pending",
                created_at=record.created_at,
                updated_at=now_iso(),
                attempts=record.attempts + 1,
                input_fp=record.input_fp,
            )
            self._atomic_write(nxt)
        return nxt.generation


@dataclass
class AsyncDispatcher:
    """A bounded, abortable HTTP dispatcher sharing one ``httpx.AsyncClient``.

    The client is INJECTED (a ``MockTransport`` in tests); nothing here reaches
    the network on construction. One client per process, reused across calls so
    connection setup is not paid per request.
    """

    client: httpx.AsyncClient | None = None
    connect_timeout: float = 5.0
    read_timeout: float = 15.0
    _owned: bool = field(default=False, repr=False)

    def _ensure(self) -> httpx.AsyncClient:
        if self.client is None:
            # The absolute deadline comes from the budget, so the transport
            # timeouts are only a belt: they bound a stalled connect/read the
            # budget's coarse wait_for might not notice promptly.
            timeout = httpx.Timeout(self.read_timeout, connect=self.connect_timeout)
            self.client = httpx.AsyncClient(timeout=timeout)
            self._owned = True
        return self.client

    async def aclose(self) -> None:
        if self.client is not None and self._owned:
            await self.client.aclose()
            self.client = None
            self._owned = False

    async def send(
        self,
        method: str,
        url: str,
        *,
        budget: OperationBudget,
        json_body: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        stage: str = "commit",
    ) -> httpx.Response:
        """Dispatch one request under the budget; aborts the transfer on timeout.

        ``asyncio.wait_for`` around a genuinely async client call actually
        cancels the request (unlike cancelling ``to_thread`` over ``requests``),
        so the deadline is real. ``params`` carries the family's target query
        params (visibility/tenant) exactly as the client's own publish helpers
        spell them.
        """
        remaining = budget.require(stage)
        client = self._ensure()
        try:
            return await asyncio.wait_for(
                client.request(method, url, json=json_body, headers=headers, params=params),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            raise OperationDeadlineExceeded(f"{stage} exceeded the attempt budget") from None
        except httpx.HTTPError:
            # A transport error is not a deadline; the caller classifies it.
            raise


def spawn_outbox_worker(
    coro_factory: Callable[[], Any],
    *,
    loop: asyncio.AbstractEventLoop | None = None,
) -> asyncio.Task[Any]:
    """Run a dispatch coroutine on the loop; kept here so callers don't each re-derive it."""

    loop = loop or asyncio.get_event_loop()
    return loop.create_task(coro_factory())
