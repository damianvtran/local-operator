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

NOTHING HERE PERSISTS BODY BYTES. The fetched body and the frame text live in
this object's memory for the life of one turn; the only durable writes are the
reservation's own state transitions, which carry ids, digests and closed codes
(the store validates every one of them).
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class FetchedBody:
    """The hub's answer for one reserved message.

    ``body`` is third-party content and never leaves this process except into
    the request being assembled. The labels are the hub's own display strings;
    an older backend that omits them yields empty strings, which the frame
    renders as unknown provenance rather than as a person's name.
    """

    body: str
    body_sha256: str
    conversation_label: str
    sender_label: str
    #: The hub's CLOSED sender kind (human|agent|unknown). It is the one
    #: provenance fact the frame's trusted section may state; an older backend
    #: that omits it reads as "unknown" rather than being inferred from a name.
    sender_kind: str = "unknown"


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
    ) -> None:
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
        self._injected: InjectedBody | None = None

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

    async def open(
        self,
        *,
        budget: BodyBudget,
        fetch: Callable[[str, str], Awaitable[FetchedBody | None]],
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

        row = self._store.pending_body_reservation_for(self._owner, self._session_id)
        if row is None:
            return False
        claim_token = f"pt:{uuid.uuid4().hex}"
        turn_token = f"tt:{uuid.uuid4().hex}"
        conversation_id = str(row["conversation_id"])
        message_id = str(row["message_id"])
        try:
            won = self._store.claim_body_reservation(
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
        self._claim_token = claim_token
        self._turn_token = turn_token
        self._conversation_id = conversation_id
        self._message_id = message_id
        self._policy_epoch = int(row["policy_epoch"])
        consent = row["consent_version"]
        self._consent_version = consent if isinstance(consent, str) else None

        try:
            fetched = await fetch(row["conversation_id"], str(row["message_id"]))
        except Exception:  # noqa: BLE001 — one closed code, never a provider echo
            fetched = None
        if fetched is None:
            self._release("binding_drift")
            return False
        if fetched.body_sha256 != str(row["body_sha256"]):
            # The hub served bytes other than the ones screened and reserved.
            self._release("binding_drift")
            return False

        reservation_id = f"{conversation_id}/{message_id}"
        try:
            pending = PendingBody(
                reservation_id=reservation_id,
                source=_source_ref(row),
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
            )
            self._injected = assemble_injection(pending, budget, self._fence)
        except BodyNotFitted:
            # Capacity is a fact about THIS request, not a verdict on the
            # message: the row goes back to pending so a roomier turn can serve it.
            self._release("body_not_fitted")
            return False
        except BodyFrameRefused:
            self._release("binding_drift")
            return False
        return True

    def transform(self, messages: list[Any]) -> list[Any]:
        """The ``transform_context`` callback: append the frame, or pass through.

        WHY the callback lives here rather than in the session: the session must
        not know how a frame is shaped or where it is inserted, and this is the
        same object that holds the lease the append depends on. Returning the
        input list unchanged is the ordinary-request answer, not an error.
        """

        text = self.for_call()
        if text is None:
            return messages
        from .request_body import CustomFrameInjection

        return CustomFrameInjection.append(messages, text, message_id=self._frame_message_id())

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

    def for_call(self) -> str | None:
        """The bytes to append for THIS provider call, or None to send without.

        Called once per physical request, synchronously, on the loop's own
        thread: renew, four local reads, then the cached bytes. Every early
        return is a call that goes WITHOUT the frame; none of them raises, so a
        store hiccup can never fail a turn.
        """

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
            facts = self._store.per_call_admission_state(self._owner, self._session_id)
        except StoreRefused:
            facts = None
        if not self._facts_hold(facts):
            # S-10/S-21: send THIS call without the frame and stop for the
            # rest of the turn; the row stays pending with its reason.
            self._drifted = True
            self._drift_reason = "admission_drift"
            return None
        self._fence.record_append()
        return self._injected.text

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
        return "completed" if completed else "claim_lost"

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

    def _release(self, reason: str) -> None:
        if self._claim_token is None:
            return
        try:
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


def _closed_sender_kind(value: Any) -> str:
    """Narrow the hub's sender kind to the frame's closed enum.

    WHY not pass it through: the frame validates it, and an unexpected string
    from a newer backend would refuse the whole frame. Mapping an unknown value
    to "unknown" keeps the trusted section honest — the kind is never inferred
    from a label.
    """

    from .body_frame import SENDER_KINDS

    return value if isinstance(value, str) and value in SENDER_KINDS else "unknown"


def _source_ref(row: dict[str, Any]) -> Any:
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
    ).validated()
