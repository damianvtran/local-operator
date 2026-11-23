"""The turn-side consumer of a pending conversation BODY reservation.

WHY this is its own module: the frame (``body_frame``) answers what the bytes
look like and ``request_body`` answers whether they may enter THIS request
once. Neither may open a store, hold a lease or decide when a turn ended —
those are authority decisions that belong to the turn, and putting them beside
the pure assembly would make the pure half untestable without a database.

THE INVARIANT (seam design, Revision 5) is enforced here, in one place:

  * Every PHYSICAL provider call renews the lease, re-reads four LOCAL facts
    (owner, designation, policy epoch, consent) and only then re-appends the
    SAME cached bytes. Any drift, or a lost lease, sends that call WITHOUT the
    frame and leaves the row pending with a recorded reason.
  * Settlement completes only when the frame was appended at least once this
    turn, the claim-guarded UPDATE matched, the turn was not aborted, and the
    terminal verdict was successful (``error is None`` and a ``stop_reason`` in
    :data:`SUCCESS_STOP_REASONS`). Everything else releases the row so another
    turn can serve it — never a silent drop and never a false completion.

NOTHING HERE PERSISTS BODY BYTES INTO THE STATE STORE OR THE TRANSCRIPT. The
fetched body and the frame text live in this object's memory for the life of one
turn; the reservation's state transitions carry ids, digests and closed codes
(the store validates every one of them).

THE ONE EXCEPTION, stated precisely: on a ``completed`` settlement — and only
then — the hash-verified body this turn was given is written to the DISPLAY
store (``display_bodies``, a separate 0600 database) so the operator can read
what their agent read. That row is never an admission proof and never re-enters
a request except through ``card_projection``'s bounded excerpt gate.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from .body_frame import BodyFrameRefused
from .request_body import (
    BodyBudget,
    BodyNotFitted,
    InjectedBody,
    PendingBody,
    RequestAuthority,
    TurnFence,
    admission_digest,
    assemble_injection,
)
from .store import OwnerSnapshot, Store, StoreRefused, _owner_key

__all__ = [
    "FetchedBody",
    "PendingTurn",
    "SUCCESS_STOP_REASONS",
]

#: The terminal verdicts that may complete a reservation (S-11). A provider
#: error, an abort or a truncation is NOT proof the answer consumed the body,
#: so the row stays retryable rather than being completed on a bad turn.
logger = logging.getLogger(__name__)

SUCCESS_STOP_REASONS = frozenset({"stop", "length", "toolUse"})

#: DESIGN S-12 says the TTL must be above the max turn so that expiry means the
#: holder is dead. That cannot be proven here — normal turns carry no wall-clock
#: deadline (``LoopConfig.deadline`` is unset on this path) — so the lease is
#: renewed at BOTH observable boundaries instead: every provider call
#: (:meth:`PendingTurn.for_call`) and every tool batch (``Session._on_turn_end``
#: calls :meth:`PendingTurn.renew`). The floor below is chosen well above any
#: single operation we can name: a provider call is bounded by its own client
#: timeout (30 s in ``team_bridge.client``) and a tool batch by its own tool,
#: so 8 hours covers a long build/training run with hours to spare.
#:
#: THE RESIDUAL, stated rather than hidden: an operation that outlives 8 hours
#: between two renewals can still expire mid-turn; a second turn may then claim
#: the row and the body is served twice. A real turn deadline (option (b)) is
#: what would close it.
_LEASE_FLOOR_SECONDS = 8 * 60 * 60.0
#: An UNCAPPED extension would let a crashed turn pin its row indefinitely,
#: which defeats "expiry means dead" in the other direction. One day is above
#: any legitimate run and short enough that a dead claim resurfaces.
_LEASE_CAP_SECONDS = 24 * 60 * 60.0

#: Anchor recorded when Call 1's history was empty: the frame leads the list.
#: A sentinel rather than ``None`` because ``None`` means "no Call 1 yet".
_ANCHOR_AT_START = object()


@dataclass(frozen=True)
class FetchedBody:
    """The hub's answer for one reserved message.

    ``body`` is third-party content and never leaves this process except into
    the request being assembled — and, after a ``completed`` settlement, into
    the display-only store (see the module docstring's one exception). The
    labels are the hub's own display strings; an older backend that omits them
    yields empty strings, which the frame renders as unknown provenance rather
    than as a person's name.
    """

    body: str
    body_sha256: str
    conversation_label: str
    sender_label: str
    #: The hub's CLOSED sender kind (human|agent|unknown). It is the one
    #: provenance fact the frame's trusted section may state; an older backend
    #: that omits it reads as "unknown" rather than being inferred from a name.
    sender_kind: str = "unknown"
    #: The hub's conversation kind (dm|group|channel) from the same body read;
    #: ``None`` when an older hub omits it. It selects the frame's channel
    #: participation line and is never inferred from a label.
    conversation_kind: str | None = None


class _FrameAnchor:
    """Where a turn's frame block goes: after the Call 1 predecessor, or last.

    Shared by the one-body turn and :class:`PendingTurnBatch` (whose K frames
    are inserted as ONE contiguous block at this single index), so the two
    cannot drift on the tool_use/tool_result safety rules below.
    """

    def __init__(self) -> None:
        self._anchor_after: Any = None
        self._anchor_after_id: str | None = None

    def index(self, messages: list[Any]) -> int:
        """Where this call's frame goes: after the Call 1 predecessor, or last.

        The anchor is matched by object identity first (the harness loop
        passes the same message objects from step to step), then by the
        message's own ``id`` in case a caller re-validated or copied them.
        Inserting there is refused -- and the frame appended instead -- when
        the predecessor is gone (compaction summarised it away), when it is an
        assistant message carrying tool calls, or when the message after it is
        a tool result: in all three the frame would sit between a tool_use and
        its tool_result, which neither Anthropic nor OpenAI-compat accepts.
        Appending forfeits the cache saving for that call, never validity.
        """

        if self._anchor_after is None:
            # Call 1: the frame goes last; remember what it follows.
            if messages:
                self._anchor_after = messages[-1]
                self._anchor_after_id = getattr(messages[-1], "id", None) or None
            else:
                self._anchor_after = _ANCHOR_AT_START
            return len(messages)
        if self._anchor_after is _ANCHOR_AT_START:
            pos = 0
        else:
            pos = -1
            # Scan from the end: the anchor sits before this turn's steps.
            for i in range(len(messages) - 1, -1, -1):
                candidate = messages[i]
                if candidate is self._anchor_after or (
                    self._anchor_after_id is not None
                    and getattr(candidate, "id", None) == self._anchor_after_id
                ):
                    pos = i + 1
                    break
            if pos < 0 or getattr(messages[pos - 1], "tool_calls", None):
                return len(messages)
        if pos < len(messages) and getattr(messages[pos], "role", None) == "tool":
            return len(messages)
        return pos


class PendingTurn:
    """One turn's claim on one pending reservation, and its per-call gate."""

    def __init__(
        self,
        owner: OwnerSnapshot,
        session_id: str,
        *,
        config_dir: Path,
        store: Store | None = None,
        lease_ttl_seconds: float | None = None,
        binding_guard: Callable[[], bool] | None = None,
    ) -> None:
        # SQLite proves the captured owner's standing, not that the Session
        # still serves that owner after an await. The host supplies this live
        # identity guard; standalone consumers retain their own store fencing.
        self._binding_guard = binding_guard
        self._owner = owner
        self._session_id = session_id
        self._store = store if store is not None else Store(config_dir)
        #: Never shorter than the stated floor: a caller's deadline may extend
        #: the lease, never shrink it below one long tool batch (S-12).
        requested = float(lease_ttl_seconds) if lease_ttl_seconds else 0.0
        self._ttl = min(max(_LEASE_FLOOR_SECONDS, requested), _LEASE_CAP_SECONDS)
        self._fence = TurnFence()
        self._claim_token: str | None = None
        self._turn_token: str | None = None
        self._conversation_id: str | None = None
        self._lost_lease = False
        self._conversation_id: str | None = None
        self._message_id: str | None = None
        self._policy_epoch: int | None = None
        self._consent_version: str | None = None
        self._drifted = False
        self._drift_reason = "admission_drift"
        #: An ambient-digest member of a batch: claimed/released through the
        #: digest CAS so it returns to ``deferred``, never ``pending``.
        self._digest = False
        self._row: dict[str, Any] | None = None
        self._injected: InjectedBody | None = None
        #: A batch member's RULE-CARRYING variant of the same frame (receiver
        #: rules head + the same VERIFIED SOURCE/body bytes), rendered at open
        #: so whichever member is the first LIVE frame of a call can carry the
        #: rules. ``None`` for a one-body turn, whose single frame already
        #: carries them.
        self._rules_text: str | None = None
        #: The message Call 1 placed the ephemeral frame right after (``None``
        #: until Call 1; :data:`_ANCHOR_AT_START` when Call 1's history was
        #: empty). Call 2+ re-inserts after the SAME message so the pre-turn +
        #: frame prefix stays byte-identical across mid-turn tool steps.
        #: WHY a message and not a list index: mid-turn compaction replaces
        #: the whole list (``[summary, *preserved, *kept]``), so a saved index
        #: points into unrelated history and can split a tool_use from its
        #: tool_result -- a 400 on both wires for the rest of the turn.
        self._anchor = _FrameAnchor()
        self._presentation_ref: Any = None
        #: Held only so a COMPLETED settlement can hand the verified bytes to
        #: the display store; cleared with the claim either way.
        self._fetched: FetchedBody | None = None

    # -- lifecycle ----------------------------------------------------------

    @property
    def reservation_id(self) -> str | None:
        return self._fence.reservation_id

    @property
    def has_claim(self) -> bool:
        return self._claim_token is not None

    @property
    def appended_calls(self) -> int:
        return self._fence.appended_calls

    def binding_intact(self) -> bool:
        """Refuse host rebinding without releasing another claimant's token."""
        try:
            intact = self._binding_guard is None or self._binding_guard()
        except Exception:  # noqa: BLE001 — unavailable host authority fails closed
            intact = False
        if not intact:
            self._drifted = True
            self._drift_reason = "admission_drift"
            self._release("admission_drift")
        return intact

    async def open(
        self,
        *,
        budget: BodyBudget,
        fetch: Callable[[str, str], Awaitable[FetchedBody | None]],
        selected: tuple[str, str] | None = None,
    ) -> bool:
        """Claim this session's oldest pending reservation and cache its frame.

        False means NO transform should be attached: there is nothing to
        carry, another turn holds the row, or the fetched bytes are not the
        reserved ones. The request in that case must be the ordinary one.

        The CAS is synchronous because it must not interleave; the fetch is the
        hub round trip the design permits on this path. A failed fetch or a
        mismatch RELEASES the claim so a later turn can retry — a pending row
        is never converted into a silently lost message.
        """

        if not self.binding_intact():
            return False
        row = self._store.pending_body_reservation_for(
            self._owner, self._session_id, selected=selected
        )
        if row is None:
            return False
        if not self.claim(row):
            return False
        if not await self.hydrate(fetch):
            return False
        return self.render(budget)

    # -- member steps (shared with PendingTurnBatch) ------------------------
    #
    # WHY ``open`` is split into claim / hydrate / render: a multi-frame turn
    # must claim and verify EVERY member before it can decide which ones fit
    # and which frame carries the rules, and only then render. Each step keeps
    # the single-body contract it had inside ``open``: a failed step releases
    # this member's own claim and answers False, never raises.

    @property
    def identity(self) -> tuple[str, str] | None:
        if self._conversation_id is None or self._message_id is None:
            return None
        return (self._conversation_id, self._message_id)

    @property
    def is_digest(self) -> bool:
        return self._digest

    @property
    def conversation_seq(self) -> int:
        return int(self._row["conversation_seq"]) if self._row is not None else 0

    @property
    def conversation_kind(self) -> str | None:
        return self._fetched.conversation_kind if self._fetched is not None else None

    @property
    def frame_size(self) -> int:
        return self._injected.size_bytes if self._injected is not None else 0

    def claim(self, row: dict[str, Any], *, digest: bool = False) -> bool:
        """CAS ``row`` into this member's claim (digest rows by their own CAS)."""

        claim_token = f"pt:{uuid.uuid4().hex}"
        turn_token = f"tt:{uuid.uuid4().hex}"
        conversation_id = str(row["conversation_id"])
        message_id = str(row["message_id"])
        # A digest row is CLAIMED through its own CAS so it stays classed:
        # ``claim_body_reservation`` only takes ``pending`` rows, and a digest
        # row must never become one (see ``Store.claim_digest_reservation``).
        cas = self._store.claim_digest_reservation if digest else self._store.claim_body_reservation
        try:
            won = cas(
                self._owner,
                conversation_id=conversation_id,
                message_id=message_id,
                claim_token=claim_token,
                turn_token=turn_token,
                now=time.time(),
                ttl=self._ttl,
            )
        except StoreRefused:
            return False
        if not won:
            # Another turn won the row; injecting here would serve one body twice.
            return False
        self._digest = digest
        self._row = row
        self._claim_token = claim_token
        self._turn_token = turn_token
        self._conversation_id = conversation_id
        self._message_id = message_id
        self._policy_epoch = int(row["policy_epoch"])
        consent = row["consent_version"]
        self._consent_version = consent if isinstance(consent, str) else None
        return True

    async def hydrate(self, fetch: Callable[[str, str], Awaitable[FetchedBody | None]]) -> bool:
        """Fetch the claimed row's body and verify it is the screened bytes."""

        row = self._row
        if row is None or self._claim_token is None:
            return False
        conversation_id = str(row["conversation_id"])
        message_id = str(row["message_id"])
        try:
            fetched = await fetch(row["conversation_id"], str(row["message_id"]))
        except asyncio.CancelledError:
            # A queued body-only wake may be disposed during hydration, before
            # Session owns its turn object. Release here rather than pinning
            # that reservation for the entire crash-recovery lease.
            self._release("binding_drift")
            raise
        except Exception:  # noqa: BLE001 — one closed code, never a provider echo
            fetched = None
        if not self.binding_intact():
            return False
        if fetched is None:
            self._release("binding_drift")
            return False
        if fetched.body_sha256 != str(row["body_sha256"]):
            # The hub served bytes other than the ones screened and reserved.
            self._release("binding_drift")
            return False
        self._fetched = fetched

        from .types import ConversationFeedRef

        self._presentation_ref = ConversationFeedRef(
            conversation_id=conversation_id,
            message_id=message_id,
            conversation_seq=int(row["conversation_seq"]),
            mention=bool(row["mention"]),
            sender_kind=fetched.sender_kind if fetched.sender_kind in ("human", "agent") else None,
            # Neither the reservation row nor the body route carries the hub's
            # thread root, so this ref must not claim the message IS a root:
            # a reply presented here would then be answered under its own id
            # (refused by the hub as ``invalid_root``). Placement resolves an
            # unknown root from the hub at post time (tool._default_thread_root).
            thread_root_known=False,
        ).validated()
        return True

    def _pending_body(self, **shape: Any) -> PendingBody:
        row, fetched = self._row, self._fetched
        assert row is not None and fetched is not None
        reservation_id = f"{row['conversation_id']}/{row['message_id']}"
        return PendingBody(
            reservation_id=reservation_id,
            source=_source_ref(row, conversation_kind=fetched.conversation_kind),
            conversation_label=fetched.conversation_label,
            sender_label=fetched.sender_label,
            sender_kind=_closed_sender_kind(fetched.sender_kind),
            body=fetched.body,
            authority=RequestAuthority(
                admission_digest=admission_digest(
                    _owner_key(self._owner),
                    str(row["screening_fingerprint"]),
                    fetched.body_sha256,
                    reservation_id,
                ),
                receipt_id=str(row["screening_fingerprint"]),
                policy_digest=str(row["policy_epoch"]),
            ),
            **shape,
        )

    def measure(self, **shape: Any) -> int | None:
        """This member's frame size under ``shape``, WITHOUT caching it.

        The batch sizes every member before it decides who fits; caching here
        would fix the bytes before the rule carrier is known. ``None`` = the
        frame itself is refused (the member is released as drift).
        """

        from .body_frame import build_body_frame, request_size

        if self._row is None or self._fetched is None or self._claim_token is None:
            return None
        try:
            pending = self._pending_body(**shape)
            source = pending.source
            if pending.sender_kind != source.sender_kind:
                from dataclasses import replace

                source = replace(source, sender_kind=pending.sender_kind).validated()
            return request_size(
                build_body_frame(
                    source,
                    conversation_label=pending.conversation_label,
                    sender_label=pending.sender_label,
                    body=pending.body,
                    with_rules=pending.with_rules,
                    participation=pending.participation,
                    delivery=pending.delivery,
                    omitted_ambient=pending.omitted_ambient,
                )
            )
        except BodyFrameRefused:
            self._release("binding_drift")
            return None

    def render(
        self,
        budget: BodyBudget,
        *,
        rules: tuple[BodyBudget, dict[str, Any]] | None = None,
        **shape: Any,
    ) -> bool:
        """Render and cache this member's frame, or release it.

        ``rules`` = (budget, shape) of the rule-carrying variant a batch member
        must ALSO hold (see ``PendingTurnBatch``): both are fixed now so every
        later call re-sends exact admitted bytes, and a member that cannot
        render the variant is not served at all — a live frame that could not
        carry the rules would otherwise reach the model without them.
        """

        if self._row is None or self._fetched is None or self._claim_token is None:
            return False
        try:
            self._injected = assemble_injection(self._pending_body(**shape), budget, self._fence)
            if rules is not None:
                rules_budget, rules_shape = rules
                # A scratch fence: the member's own fence caches ONE text per
                # turn (the plain frame); this variant is the same message
                # with the trusted head prepended, never a second admission.
                self._rules_text = assemble_injection(
                    self._pending_body(**rules_shape), rules_budget, TurnFence()
                ).text
        except BodyNotFitted:
            self._injected = None
            # Capacity is a fact about THIS request, not a verdict on the
            # message: the row goes back to pending so a roomier turn can serve it.
            self._release("body_not_fitted")
            return False
        except BodyFrameRefused:
            self._injected = None
            self._release("binding_drift")
            return False
        return True

    def transform(self, messages: list[Any]) -> list[Any]:
        """The ``transform_context`` callback: inject the frame, or pass through.

        WHY the callback lives here rather than in the session: the session must
        not know how a frame is shaped or where it is inserted, and this is the
        same object that holds the lease the injection depends on. Returning the
        input list unchanged is the ordinary-request answer, not an error.

        ANCHOR INVARIANT: the first admitted call of this turn appends the
        frame and records the message it follows (the wake card / end of
        pre-turn history). Later tool-step calls insert the frame right after
        that same message, which keeps the wire prefix through the frame
        byte-identical across Call 1 -> Call 2+ so mid-turn tool calls hit the
        prompt cache. See :meth:`_insert_index` for when that is not safe and
        the frame goes last instead (the pre-anchor behaviour).

        Asides and clearances never carry the frame; the cache trade-off that
        implies is recorded at ``Session._read_only_prompt``.
        """

        text = self.for_call()
        if text is None:
            return messages
        from .request_body import CustomFrameInjection

        insert_idx = self._anchor.index(messages)
        shaped = CustomFrameInjection.insert(
            messages, text, message_id=self._frame_message_id(), index=insert_idx
        )
        self.mark(shaped[insert_idx])
        # This native frame contains screened source metadata AND body. Raw
        # journal cards never borrow its admission by repeating the same ids;
        # they remain content-free markers even in this very request.
        return shaped

    def mark(self, frame: Any) -> None:
        """Stamp THIS member's admission proof on its own injected frame."""

        if self._presentation_ref is not None:
            from .card_projection import mark_admitted_frame

            mark_admitted_frame(frame, self._presentation_ref, self._card_admission_intact)

    def _card_admission_intact(self) -> bool:
        """Reused ephemeral copies cannot outlive their exact admission lease."""
        if not self.binding_intact() or self._drifted or self._injected is None:
            return False
        if self._conversation_id is None or self._message_id is None:
            return False
        try:
            row = self._store.body_reservation(self._owner, self._conversation_id, self._message_id)
            facts = self._store.per_call_admission_state(
                self._owner,
                self._session_id,
                reservation=(self._conversation_id or "", self._message_id or ""),
            )
            return bool(
                row
                and self._facts_hold(facts)
                and row["state"] == "claimed"
                and row["claim_token"] == self._claim_token
                and row["turn_token"] == self._turn_token
                and self._injected.authority.admission_digest
                == admission_digest(
                    _owner_key(self._owner),
                    str(row["screening_fingerprint"]),
                    str(row["body_sha256"]),
                    f"{self._conversation_id}/{self._message_id}",
                )
            )
        except (StoreRefused, KeyError, TypeError, ValueError):
            return False

    def _frame_message_id(self) -> str:
        """A stable, host-owned id for the injected row.

        Uses the reservation's own ids so a retry of the same call would replace
        rather than duplicate — and so nothing is derived from the body.
        """

        return f"bridge-body:{self._conversation_id}:{self._message_id}"

    def renew(self) -> bool:
        """Extend the lease at a TOOL-BATCH boundary (design S-12).

        WHY a separate method rather than re-calling :meth:`for_call`: the
        session's tool boundary must not append a frame — there is no request
        being built — it only needs to prove the turn is still alive. A False
        answer means the lease was taken or expired; the turn must then stop
        injecting, which `settle` records as `claim_lost`.
        """

        if not self.binding_intact():
            return False
        if self._claim_token is None or self._injected is None or self._drifted:
            return False
        if self._conversation_id is None or self._message_id is None:
            return False
        try:
            held = self._store.renew_body_claim(
                self._owner,
                conversation_id=self._conversation_id,
                message_id=self._message_id,
                claim_token=self._claim_token,
                turn_token=self._turn_token or "",
                now=time.time(),
                ttl=self._ttl,
            )
        except StoreRefused as failure:
            # A busy store at an intermediate boundary is NOT a lost lease.
            self._drift_reason = "admission_drift"
            logger.debug("bridge: lease renewal refused code=%s", failure.code)
            return False
        if not held:
            self._drifted = True
            self._lost_lease = True
            self._drift_reason = "claim_lost"
        return held

    def for_call(self, *, with_rules: bool = False) -> str | None:
        """The bytes to append for THIS provider call, or None to send without.

        Called once per physical request, synchronously, on the loop's own
        thread: renew, four local reads, then the cached bytes. Every early
        return is a call that goes WITHOUT the frame; none of them raises, so a
        store hiccup can never fail a turn.

        ``with_rules`` asks for the rule-carrying variant (batch members only;
        the batch asks it of the first live frame). A member rendered without
        one answers None rather than its bare frame: untrusted bytes must
        never lead a request without the receiver rules ahead of them.
        """

        if with_rules and self._rules_text is None:
            return None

        if not self.binding_intact():
            return None
        if self._injected is None or self._claim_token is None or self._drifted:
            return None
        if self._conversation_id is None or self._message_id is None:
            return None
        now = time.time()
        try:
            held = self._store.renew_body_claim(
                self._owner,
                conversation_id=self._conversation_id,
                message_id=self._message_id,
                claim_token=self._claim_token,
                turn_token=self._turn_token or "",
                now=now,
                ttl=self._ttl,
            )
        except StoreRefused as failure:
            # TRANSIENT (busy store, damaged read) is not a stolen lease: the
            # row may still be ours, so this call goes without the frame and
            # settlement records drift — never `claim_lost`, which would leave
            # the row parked until expiry with no one to blame.
            self._drifted = True
            self._drift_reason = "admission_drift"
            logger.debug("bridge: lease renew refused code=%s", failure.code)
            return None
        if not held:
            # S-12: an expired or stolen lease means this turn no longer owns
            # the row. It must stop injecting rather than race the new holder.
            self._drifted = True
            self._lost_lease = True
            self._drift_reason = "claim_lost"
            return None
        try:
            facts = self._store.per_call_admission_state(
                self._owner,
                self._session_id,
                reservation=(self._conversation_id or "", self._message_id or ""),
            )
        except StoreRefused:
            facts = None
        if not self._facts_hold(facts):
            # S-10/S-21: send THIS call without the frame and stop for the
            # rest of the turn; the row stays pending with its reason.
            self._drifted = True
            self._drift_reason = "admission_drift"
            return None
        self._fence.record_append()
        return self._rules_text if with_rules else self._injected.text

    def settle(
        self,
        *,
        entry_id: str | None,
        error: str | None,
        stop_reason: str | None,
        aborted: bool,
    ) -> str:
        """Complete, release, or do nothing — and say which, in closed terms.

        Order is the contract: a turn that never appended cannot complete even
        if it looks successful (S-16a, the silent-loss class), and a turn that
        lost its claim cannot write over the row (S-16b — the store's UPDATE is
        claim-guarded and its zero-row answer is a no-op here, not an error).
        """

        if self._claim_token is None or self._injected is None:
            return "not_claimed"
        if self._drifted:
            if self._claim_lost:
                # The lease is gone: the row belongs to whoever holds it now,
                # and releasing would be a no-op against their token anyway.
                self._claim_token = None
                return "claim_lost"
            self._release(self._drift_reason)
            return "drifted"
        if self._fence.appended_calls < 1:
            # Nothing was ever appended: a later ordinary turn must not inherit
            # this completion (S-16a).
            self._release("never_appended")
            return "never_appended"
        if aborted or error is not None or stop_reason not in SUCCESS_STOP_REASONS:
            self._release("admission_drift")
            return "unverified"
        if entry_id is None:
            self._release("never_appended")
            return "never_appended"
        try:
            completed = self._store.complete_body_reservation(
                self._owner,
                conversation_id=self._conversation_id or "",
                message_id=self._message_id or "",
                claim_token=self._claim_token,
                entry_id=entry_id,
                now=time.time(),
            )
        except StoreRefused:
            return "claim_lost"
        if not completed:
            return "claim_lost"
        self._record_display_body()
        return "completed"

    def _record_display_body(self) -> None:
        """The SINGLE turn-side writer of the display store.

        Runs only after the claim-guarded completion matched, so the row means
        exactly "this agent was given these bytes". Best effort: a failed write
        leaves the card showing the hub backfill (or today's copy), never a
        changed settlement.
        """

        fetched = self._fetched
        self._fetched = None
        if fetched is None or not self._conversation_id or not self._message_id:
            return
        from .display_bodies import DisplayBodyStore

        DisplayBodyStore(self._store.config_dir).record(
            _owner_key(self._owner),
            message_id=self._message_id,
            conversation_id=self._conversation_id,
            body=fetched.body,
            body_sha256=fetched.body_sha256,
            sender_kind=_closed_sender_kind(fetched.sender_kind),
            source="turn",
        )

    # -- internals ----------------------------------------------------------

    @property
    def _claim_lost(self) -> bool:
        return self._drifted and self._lost_lease

    def _facts_hold(self, facts: dict[str, Any] | None) -> bool:
        if facts is None:
            return False
        if not facts.get("designation_matches"):
            return False
        if facts.get("policy_epoch") != self._policy_epoch:
            return False
        return facts.get("consent_version") == self._consent_version

    def _release(self, reason: str, *, count_attempt: bool = True) -> None:
        if self._claim_token is None:
            return
        try:
            if self._digest:
                # Back to ``deferred`` (or retired after its attempt cap) —
                # never ``pending``, which would make an ambient post nobody
                # was woken for a full turn of its own.
                self._store.release_digest_claim(
                    self._owner,
                    conversation_id=self._conversation_id or "",
                    message_id=self._message_id or "",
                    claim_token=self._claim_token,
                    reason=reason,
                    now=time.time(),
                    count_attempt=count_attempt,
                )
            else:
                self._store.release_body_claim(
                    self._owner,
                    conversation_id=self._conversation_id or "",
                    message_id=self._message_id or "",
                    claim_token=self._claim_token,
                    reason=reason,
                    now=time.time(),
                )
        except StoreRefused as failure:
            # A lost token is a no-op by contract, but a REFUSED release is not
            # nothing: the row would stay claimed with no ``last_error`` until
            # its lease expires, and silence is how that parks unnoticed. One
            # structural code, no content.
            logger.warning("bridge: body release refused code=%s", failure.code)
        self._claim_token = None
        self._turn_token = None
        # A released row was not completed: nothing may reach the display store.
        self._fetched = None


#: Woken rows that may ride ONE turn beside its primary. Bounded so a 47-post
#: burst becomes a handful of turns, not one request whose size and blast
#: radius (one failed verdict releases every member) grow with the burst.
MAX_COMPANIONS = 7
#: Ambient-digest rows per turn, and their combined frame bytes. A digest is
#: context, not work: it must never crowd out a woken body, so it is fitted
#: LAST, newest first, and capped well under one 16 KiB body pair.
DIGEST_MAX = 10
DIGEST_MAX_BYTES = 24 * 1024
#: Rows the digest reader scans to count what it did NOT show (the "N more
#: ambient messages not shown" head line). A count only; no row is claimed.
_DIGEST_SCAN = 200


class PendingTurnBatch:
    """One turn serving SEVERAL admitted bodies, each as its own frame.

    WHY (operator request, 2026-11-21: "if there are multiple messages arriving
    at a time or in close proximity ... respond to all at once"): half of all
    inbound bridge messages arrive <=30 s after the previous one, and the drain
    ran one full-context turn per body — a 5-post burst was 5 consecutive Opus
    turns, mostly ``no_reply``. Here a burst is ONE request carrying K frames.

    K FRAMES, NOT ONE FRAME WITH K OBJECTS: ``build_body_frame`` and
    ``decode_untrusted_object`` stay one-message-per-frame, so the whole-message
    and exact-bytes contracts are untouched. Every member is a full
    :class:`PendingTurn` with its OWN claim/lease CAS, sha verification,
    admission digest, per-call renewal and admission stamp — nothing about one
    member's authority is inferred from another's.

    MEMBERS: a primary (the drain's oldest eligible woken row, with today's
    refusal path), up to :data:`MAX_COMPANIONS` woken rows of the SAME
    conversation admitted no later than the drain's horizon, then up to
    :data:`DIGEST_MAX` deferred ambient rows (newest first, <=
    :data:`DIGEST_MAX_BYTES`). Order on the wire: digest frames first (oldest
    to newest), then the woken frames in ``conversation_seq`` order, so the
    newest woken message reads last (the primary, the drain's OLDEST pending
    row, reads first among the woken frames) — one contiguous block at the
    single anchor :class:`_FrameAnchor` computes. Only the first LIVE frame of
    each call carries the receiver/participation rules.

    RULES FOLLOW THE FIRST LIVE FRAME: every member renders a plain frame AND
    a rule-carrying variant at open, and each call gives the variant to its
    first live member. A member drops out for the rest of the turn on ONE
    transient refusal of its own renewal; were the rules fixed to a member,
    that one busy read would leave every later call carrying untrusted bodies
    with no receiver rules and no config-request boundary. Budget is unchanged:
    the head is reserved once and only one frame per call ever carries it.
    The cost: when the carrier changes, the block's bytes change, so that
    call misses the prompt cache through the block (it changed anyway — a
    frame dropped out).

    CAPACITY ORDER: woken members are fitted against the head THEY need (the
    receiver rules, plus participation only if a woken member is a
    group/channel post). The digest's extra head bytes (participation for its
    channel rows, the "N more ambient messages not shown" line) are charged to
    the digest step and the whole digest is dropped when they do not fit, so
    ambient context can never crowd out a woken body that fits on its own.

    SETTLEMENT: one turn-level verdict applied to every member (all complete
    or all release, as a one-body turn would); a member that lost its claim or
    drifted mid-turn has already dropped out of later calls (its ``for_call``
    answers None) and settles its own way.
    """

    def __init__(
        self,
        owner: OwnerSnapshot,
        session_id: str,
        *,
        config_dir: Path,
        store: Store | None = None,
        lease_ttl_seconds: float | None = None,
        binding_guard: Callable[[], bool] | None = None,
        admissible: Callable[[dict[str, Any]], bool] | None = None,
        admitted_before: float | None = None,
        exclude: frozenset[tuple[str, str]] = frozenset(),
        max_companions: int = MAX_COMPANIONS,
        digest_max: int = DIGEST_MAX,
        digest_max_bytes: int = DIGEST_MAX_BYTES,
    ) -> None:
        self._owner = owner
        self._session_id = session_id
        self._config_dir = config_dir
        self._store = store if store is not None else Store(config_dir)
        self._ttl = lease_ttl_seconds
        self._binding_guard = binding_guard
        #: The session's per-row authority (receipt verdict + sha, designation,
        #: consent, policy, routed scope): the SAME test its drain applies to a
        #: primary, applied to every companion and digest row before claiming.
        self._admissible = admissible
        self._admitted_before = admitted_before
        self._exclude = exclude
        self._max_companions = max(0, int(max_companions))
        self._digest_max = max(0, int(digest_max))
        self._digest_max_bytes = max(0, int(digest_max_bytes))
        self._primary: PendingTurn | None = None
        self._members: list[PendingTurn] = []
        self._anchor = _FrameAnchor()
        #: Companions released for CAPACITY: the drain must NOT count them as
        #: attempted, so the next pass serves one as its primary.
        self.deferred_for_capacity: list[tuple[str, str]] = []
        self.last_outcomes: list[tuple[tuple[str, str] | None, str]] = []
        self._tried: list[tuple[str, str]] = []

    def _member(self) -> PendingTurn:
        return PendingTurn(
            self._owner,
            self._session_id,
            config_dir=self._config_dir,
            store=self._store,
            lease_ttl_seconds=self._ttl,
            binding_guard=self._binding_guard,
        )

    # -- the surface Session calls on a one-body turn ------------------------

    @property
    def reservation_id(self) -> str | None:
        return self._primary.reservation_id if self._primary is not None else None

    @property
    def has_claim(self) -> bool:
        return any(member.has_claim for member in self._members)

    @property
    def appended_calls(self) -> int:
        return self._primary.appended_calls if self._primary is not None else 0

    @property
    def members(self) -> tuple[PendingTurn, ...]:
        return tuple(self._members)

    @property
    def identities(self) -> tuple[tuple[str, str], ...]:
        """Every row the batch CLAIMED (served or not) — the drain's attempted set.

        Minus companions released for capacity (:attr:`deferred_for_capacity`):
        those were never tried, and the next pass must be free to serve one as
        its primary.
        """

        return tuple(i for i in self._tried if i not in self.deferred_for_capacity)

    def binding_intact(self) -> bool:
        # Every member shares the host guard; asking each one lets each release
        # its own claim on a rebind, exactly as a one-body turn does.
        intact = True
        for member in self._members or ([self._primary] if self._primary else []):
            intact = member.binding_intact() and intact
        return intact

    async def open(
        self,
        *,
        budget: BodyBudget,
        fetch: Callable[[str, str], Awaitable[FetchedBody | None]],
        selected: tuple[str, str] | None = None,
    ) -> bool:
        """Claim, verify, fit and render the batch. False = no transform at all."""

        primary = self._member()
        self._primary = primary
        if not primary.binding_intact():
            return False
        row = self._store.pending_body_reservation_for(
            self._owner, self._session_id, selected=selected
        )
        if row is None or not primary.claim(row):
            return False
        self._tried = [primary.identity] if primary.identity is not None else []
        # The PRIMARY is verified BEFORE any companion is claimed: a primary
        # whose fetch or sha fails must cost exactly what it cost a one-body
        # turn (its own release, its own attempt) and must not drag the rows
        # behind it into this pass's attempted set — the next one is then the
        # drain's next primary, as before (test_dm_runtime_transport's
        # retained-failed-body cases pin this).
        if not await primary.hydrate(fetch):
            return False
        claimed: list[PendingTurn] = [primary]
        try:
            companions = self._claim_companions(row)
            claimed.extend(companions)
            digest, scanned = self._claim_digest(
                {*self._exclude, *(m.identity for m in claimed if m.identity)}
            )
            claimed.extend(digest)
            self._tried = [m.identity for m in claimed if m.identity is not None]
            others = await self._hydrate_all(claimed[1:], fetch)
        except BaseException:
            for member in claimed:
                member._release("binding_drift", count_attempt=False)
            raise
        return self._fit_and_render(primary, [primary, *others], budget, scanned)

    def _claim_companions(self, primary_row: dict[str, Any]) -> list[PendingTurn]:
        if self._max_companions == 0:
            return []
        identity = (str(primary_row["conversation_id"]), str(primary_row["message_id"]))
        try:
            rows = self._store.pending_body_reservations_for(
                self._owner,
                self._session_id,
                exclude=frozenset({*self._exclude, identity}),
                admitted_before=self._admitted_before,
                conversation_id=identity[0],
                limit=self._max_companions,
            )
        except StoreRefused as failure:
            # Companions are optional: a busy read must cost them, not the
            # primary the one-body path would have served. Unread rows stay
            # pending and untouched for the next pass.
            logger.debug("bridge: companion read refused code=%s", failure.code)
            return []
        out: list[PendingTurn] = []
        for row in rows:
            if self._admissible is not None and not self._admissible(row):
                # Stale admission: left pending, untouched, exactly as the
                # drain's own skip does for a primary candidate.
                continue
            member = self._member()
            if member.claim(row):
                out.append(member)
        return out

    def _claim_digest(self, taken: set[tuple[str, str]]) -> tuple[list[PendingTurn], int]:
        if self._digest_max == 0:
            return [], 0
        try:
            rows = self._store.deferred_body_reservations_for(
                self._owner, self._session_id, limit=_DIGEST_SCAN, exclude=frozenset(taken)
            )
        except StoreRefused as failure:
            # Ambient context is optional: no digest (and no "not shown"
            # count) rather than no turn. Rows stay ``deferred``.
            logger.debug("bridge: digest read refused code=%s", failure.code)
            return [], 0
        out: list[PendingTurn] = []
        for row in rows[: self._digest_max]:
            if self._admissible is not None and not self._admissible(row):
                continue
            member = self._member()
            if member.claim(row, digest=True):
                out.append(member)
        return out, len(rows)

    async def _hydrate_all(
        self,
        members: list[PendingTurn],
        fetch: Callable[[str, str], Awaitable[FetchedBody | None]],
    ) -> list[PendingTurn]:
        # Sequential on purpose: each fetch is one hub round trip on the
        # runtime's single client, and a failed member must release before the
        # next is fetched (a CancelledError mid-way releases the rest above).
        out: list[PendingTurn] = []
        for member in members:
            if await member.hydrate(fetch):
                out.append(member)
        return out

    def _fit_and_render(
        self,
        primary: PendingTurn,
        hydrated: list[PendingTurn],
        budget: BodyBudget,
        scanned: int,
    ) -> bool:
        from .body_frame import DELIVERY_AMBIENT_DIGEST, frame_rules_head, request_size
        from .request_body import BodyBudget as _Budget

        try:
            budget = budget.validated()
        except BodyFrameRefused:
            for member in hydrated:
                member._release("binding_drift", count_attempt=False)
            return False
        woken = [m for m in hydrated if not m.is_digest]
        digest = [m for m in hydrated if m.is_digest]

        def channel(members: list[PendingTurn]) -> bool:
            return any(m.conversation_kind in ("group", "channel") for m in members)

        def head_size(*, participation: bool, omitted: int) -> int:
            # +1: the newline joining the head to the frame's VERIFIED SOURCE.
            lines = frame_rules_head(participation=participation, omitted_ambient=omitted)
            return request_size("\n".join(lines)) + 1

        # The WOKEN head is reserved first, against the woken members alone:
        # which member carries it varies per call, so it is counted once and
        # every member's size below is measured WITHOUT it.
        woken_participation = channel(woken)
        head_bytes = head_size(participation=woken_participation, omitted=0)
        remaining = budget.available_bytes - head_bytes
        sizes: dict[int, int] = {}

        def fits(member: PendingTurn, *, delivery: str | None) -> int | None:
            # Sized WITHOUT the rules: their bytes are reserved separately.
            return member.measure(with_rules=False, delivery=delivery)

        def drop_digest(members: list[PendingTurn]) -> None:
            # Dropped for the digest's own bounds before any request carried
            # it: un-counted, back to ``deferred``.
            for member in members:
                member._release("body_not_fitted", count_attempt=False)

        # PRIMARY: today's refusal path — if it cannot fit, nothing is served.
        size = fits(primary, delivery=None)
        if size is None or size > remaining:
            if size is not None:
                primary._release("body_not_fitted")
            for member in hydrated:
                if member is not primary:
                    # Never tried on their own merits: keep them out of the
                    # drain's attempted set so the next primary can be one.
                    if member.identity is not None and not member.is_digest:
                        self.deferred_for_capacity.append(member.identity)
                    member._release("body_not_fitted", count_attempt=False)
            return False
        remaining -= size
        sizes[id(primary)] = size
        kept_woken = [primary]
        for member in woken:
            if member is primary:
                continue
            size = fits(member, delivery=None)
            if size is None:
                continue
            if size > remaining:
                # Capacity, not a verdict: released WITHOUT joining the
                # drain's attempted set, so it is the next pass's primary.
                identity = member.identity
                member._release("body_not_fitted")
                if identity is not None:
                    self.deferred_for_capacity.append(identity)
                continue
            remaining -= size
            sizes[id(member)] = size
            kept_woken.append(member)
        # DIGEST: its extra head bytes first (worst case: participation if any
        # candidate is a channel post, every scanned row counted as omitted —
        # the real count is never larger, so never more digits), then rows
        # newest first (the reader's order: the most recent context wins).
        participation = woken_participation
        omitted = 0
        kept_digest: list[PendingTurn] = []
        if digest or scanned:
            digest_head = head_size(
                participation=woken_participation or channel(digest), omitted=scanned
            )
            extra = digest_head - head_bytes
            if extra > remaining:
                drop_digest(digest)
            else:
                remaining -= extra
                head_bytes = digest_head
                digest_bytes = 0
                for member in digest:
                    size = fits(member, delivery=DELIVERY_AMBIENT_DIGEST)
                    if size is None:
                        continue
                    if size > remaining or digest_bytes + size > self._digest_max_bytes:
                        drop_digest([member])
                        continue
                    remaining -= size
                    digest_bytes += size
                    sizes[id(member)] = size
                    kept_digest.append(member)
                # Every scanned row this request does not show — capped by
                # row count, by bytes, or lost to a failed read/claim/fetch.
                omitted = max(0, scanned - len(kept_digest))
                participation = woken_participation or channel(kept_digest)
        kept_digest.sort(key=lambda m: (m._row or {}).get("created_at") or 0.0)
        kept_woken.sort(key=lambda m: m.conversation_seq)
        ordered = [*kept_digest, *kept_woken]
        rendered: list[PendingTurn] = []
        for index, member in enumerate(ordered):
            delivery = DELIVERY_AMBIENT_DIGEST if member.is_digest else None
            plain: dict[str, Any] = {
                "with_rules": False,
                "participation": participation,
                "delivery": delivery,
                "omitted_ambient": 0,
            }
            # EVERY member also holds the rule-carrying variant: whichever is
            # the first live frame of a call carries the rules (class docstring).
            with_rules = {**plain, "with_rules": True, "omitted_ambient": omitted}
            rules = (_Budget(sizes[id(member)] + head_bytes, budget.measured_from), with_rules)
            if member.render(
                _Budget(sizes[id(member)], budget.measured_from), rules=rules, **plain
            ):
                rendered.append(member)
            elif member is primary:
                for other in rendered:
                    other._release("body_not_fitted", count_attempt=False)
                for other in ordered[index + 1 :]:
                    other._release("body_not_fitted", count_attempt=False)
                return False
        self._members = rendered
        return primary in rendered

    def transform(self, messages: list[Any]) -> list[Any]:
        """Insert every live member's frame as ONE contiguous block at the anchor.

        Each member runs its own ``for_call`` (renew + four local facts) every
        physical call; a member that fails drops out for the rest of the turn
        while the others still ride. The block's position is the one-body
        frame's: after the Call 1 predecessor, or last when that is unsafe.

        The FIRST live frame of THIS call carries the receiver rules, so the
        rules precede every untrusted body even after the previous carrier
        dropped out (one transient refusal of its renewal is enough).
        """

        from .request_body import CustomFrameInjection

        frames: list[tuple[PendingTurn, str]] = []
        for member in self._members:
            # Members are asked in wire order, so "first live" is known the
            # moment one answers; each still renews/checks exactly once.
            text = member.for_call(with_rules=not frames)
            if text is not None:
                frames.append((member, text))
        if not frames:
            return messages
        at = self._anchor.index(messages)
        shaped = list(messages)
        for offset, (member, text) in enumerate(frames):
            shaped = CustomFrameInjection.insert(
                shaped, text, message_id=member._frame_message_id(), index=at + offset
            )
            member.mark(shaped[at + offset])
        return shaped

    def renew(self) -> bool:
        """Renew EVERY member (each its own lease); answer for the primary."""

        primary_held = False
        for member in self._members:
            held = member.renew()
            if member is self._primary:
                primary_held = held
        return primary_held

    def settle(
        self,
        *,
        entry_id: str | None,
        error: str | None,
        stop_reason: str | None,
        aborted: bool,
    ) -> str:
        """Apply ONE turn verdict to every member; answer the primary's outcome."""

        outcomes: list[tuple[tuple[str, str] | None, str]] = []
        for member in self._members:
            identity = member.identity
            outcomes.append(
                (
                    identity,
                    member.settle(
                        entry_id=entry_id, error=error, stop_reason=stop_reason, aborted=aborted
                    ),
                )
            )
        self.last_outcomes = outcomes
        if self._primary is None:
            return "not_claimed"
        for member, (_, outcome) in zip(self._members, outcomes):
            if member is self._primary:
                return outcome
        return "not_claimed"


def _closed_sender_kind(value: Any) -> str:
    """Narrow the hub's sender kind to the frame's closed enum.

    WHY not pass it through: the frame validates it, and an unexpected string
    from a newer backend would refuse the whole frame. Mapping an unknown value
    to "unknown" keeps the trusted section honest — the kind is never inferred
    from a label.
    """

    from .body_frame import SENDER_KINDS

    return value if isinstance(value, str) and value in SENDER_KINDS else "unknown"


def _source_ref(row: dict[str, Any], *, conversation_kind: Any = None) -> Any:
    """Rebuild the frame's source locator from the reservation row.

    The row carries the ids, the sequence and the mention flag precisely so a
    turn can build the VERIFIED SOURCE header without re-reading the feed —
    which is what keeps a hydrated frame's provenance identical to the one the
    connector saw.
    """

    from .body_frame import ConversationBodyFrame

    mention = row["mention"]
    # ``sender_kind`` is NOT a column on the reservation: the row is
    # contentless and a label is never authority. The frame therefore renders
    # UNKNOWN provenance rather than guessing a person, which is the honest
    # answer for a hydrated turn — the connector's card already showed the
    # label at admission time.
    return ConversationBodyFrame(
        conversation_id=str(row["conversation_id"]),
        conversation_seq=int(row["conversation_seq"]),
        message_id=str(row["message_id"]),
        mention=bool(mention) if isinstance(mention, (int, bool)) else False,
        tenant_id=str(row["tenant_id"]),
        sender_kind=_closed_sender_kind(row.get("sender_kind")),
        # The row has no kind column (contentless); the fetched body does. An
        # unexpected value degrades to "did not say" rather than refusing the
        # whole frame, the same posture as ``_closed_sender_kind``.
        conversation_kind=(
            conversation_kind if conversation_kind in ("dm", "group", "channel") else None
        ),
    ).validated()
