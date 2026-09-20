"""The ONE authorisation chokepoint: ``Authorizer.check``.

One function, one call site (``RelayServer._dispatch``), with an ``OP_CAPABILITY``
table a totality test pins. A missing entry is a refusal AND a named test failure,
so "an op nobody decided the authorisation for" cannot ship — the property a
chokepoint exists for.

THREE VOCABULARIES, AND ONLY TWO REACH THIS TABLE. Session-plane ops
(``ControlOp``, forwarded inside ``net_forward``), peer-scope ops (``NET_OPS``)
and the relay's LOCAL control ops. The last group never reaches a peer link at
all — it is authorised by the control key of the 0600 peers record, exactly as the
session runtime's own control socket is — so a ``LOCAL_OPS`` name appearing in
``OP_CAPABILITY`` is a bug, not a gap, and the test asserts both directions of
that rule.

FAILS CLOSED, EVERYWHERE. An unknown name at any level is ``unknown_op``. A frame
claiming ``locality: "local"`` over a peer link is a ``protocol_error`` — the
spine's "not a softer path". A link in the reconcile phase may do exactly two
things. A session-scoped op must name a session THIS device owns: with forwarding
unimplemented, asking a relay to act on a third device's session is a refusal, not
a silent hop.

WHAT THE PEER IS TOLD. Only ``{"op": "error", "message": <sentence>}``: never
which of membership, epoch or capability failed. The real cause goes to the local
audit record, where the operator can see it and a remote attacker cannot.
"""

from __future__ import annotations

from typing import Any, Protocol

from local_operator.network.audit import AuditEvent
from local_operator.network.types import (
    INNER_OP_CAPABILITY,
    LOCAL_OPS,
    NET_OPS,
    NET_PAIR_OPS,
    OP_CAPABILITY,
    Granted,
    LinkContext,
    LinkPhase,
    NetworkRecord,
    Refusal,
)

#: The pair ceremony's ops, refused on any link that is not in the pair phase.
PAIR_OPS = frozenset(NET_PAIR_OPS)

#: The only ops a link in the reconcile phase may dispatch. Two, and both are
#: read-only: this is what keeps a previous-epoch credential from being a
#: general-purpose one.
RECONCILE_OPS = frozenset({"net_reconcile", "ping"})


class NetworkState(Protocol):
    """What the authoriser may ask about the world.

    Deliberately three questions and no more — the narrower this is, the harder it
    is for an authorisation decision to depend on something nobody reviewed.
    """

    def network(self, network_id: str) -> NetworkRecord | None:
        """The record, or ``None`` when this device is not in that network."""

    def local_session_ids(self) -> set[str]:
        """Sessions that live on THIS device (the relay's read-through cache)."""


class Authorizer:
    """The chokepoint. Construct once per relay; ``check`` per inbound frame."""

    def __init__(self, networks: NetworkState, audit: Any) -> None:
        self._networks = networks
        self._audit = audit

    # -- the one decision ---------------------------------------------------

    def effective_op(self, link: LinkContext, frame: dict[str, Any]) -> str:
        """``net_forward`` resolves to its INNER frame's op; anything else to itself.

        Exposed separately so a test can assert the resolution rule without
        dispatching, and so ``check`` can consult only real op names — never a
        carrier. A ``net_forward`` whose inner op has no entry is refused rather
        than forwarded on the grounds that the outer op is known.
        """
        op = frame.get("op")
        if not isinstance(op, str) or not op:
            raise Refusal("protocol_error", "a frame arrived with no op name")
        if op != "net_forward":
            return op
        inner = frame.get("frame")
        if not isinstance(inner, dict):
            raise Refusal(
                "protocol_error",
                "a net_forward frame must carry the inner frame it is forwarding",
            )
        inner_op = inner.get("op")
        if not isinstance(inner_op, str) or not inner_op:
            raise Refusal("protocol_error", "the forwarded inner frame has no op name")
        return inner_op

    def check(self, link: LinkContext, frame: dict[str, Any]) -> Granted:
        """Raise :class:`Refusal`, or return what was admitted.

        The order below is the design, not an implementation detail: a frame's op
        name is read first (so nothing else in an unauthenticated frame is
        touched), then the phase, then the epoch, then the capability, then the
        session scope.
        """
        op = frame.get("op")
        if not isinstance(op, str) or not op:
            raise Refusal(
                "protocol_error", "a frame with no op name was refused before anything read it"
            )
        outer_op = op

        # Pair-phase ops are phase-`pair` ONLY, and they are not in the capability
        # table at all: a pair link has no member row to authorise against, so its
        # authorisation is the invite validation plus the two human confirmations.
        # Asserting the rule in both directions is what stops a pair op becoming a
        # pre-membership general-purpose op.
        if outer_op in PAIR_OPS:
            if link.phase != "pair":
                raise Refusal(
                    "phase_forbidden",
                    f"{outer_op} is only valid during a pairing, and this link is already a "
                    "member link",
                )
            return Granted(action=outer_op)
        if link.phase == "pair":
            raise Refusal(
                "phase_forbidden",
                f"{outer_op} is not available on a link that is still pairing",
            )

        try:
            effective = self.effective_op(link, frame)
        except Refusal as refusal:
            self._refused(link, outer_op, refusal, frame)
            raise

        if effective in LOCAL_OPS:
            # A local op arriving over a peer link is a bug in the caller's
            # vocabulary, not a privilege escalation to negotiate.
            refusal = Refusal(
                "unknown_op",
                f"{effective} is a local control op and cannot be dispatched over a peer link",
            )
            self._refused(link, effective, refusal, frame)
            raise refusal

        if link.phase == "reconcile" and effective not in RECONCILE_OPS:
            refusal = Refusal(
                "phase_forbidden",
                f"this link authenticated at the previous epoch, so only "
                f"{' and '.join(sorted(RECONCILE_OPS))} are available until it reconciles",
            )
            self._refused(link, effective, refusal, frame)
            raise refusal

        self._check_epoch(link, effective, frame)
        self._check_locality(link, effective, frame)

        capability = self._required_capability(effective, outer_op)
        if capability is not None and capability not in link.capabilities:
            refusal = Refusal(
                "not_authorised",
                f"{link.device_id} may not do that on this device "
                f"(it does not hold the {capability!r} capability)",
            )
            self._refused(link, effective, refusal, frame, capability=capability)
            raise refusal

        session_id = self._session_scope(link, effective, frame)
        return Granted(action=effective, session_id=session_id, capability=capability)

    # -- steps --------------------------------------------------------------

    def _check_epoch(self, link: LinkContext, op: str, frame: dict[str, Any]) -> None:
        record = self._networks.network(link.network_id)
        if record is None:
            refusal = Refusal(
                "not_a_member",
                "this device is not in that network any more, so it cannot act on the link",
            )
            self._refused(link, op, refusal, frame)
            raise refusal
        recorded_epoch = int(frame.get("epoch") or link.epoch)
        # A ROTATION ANNOUNCES THE NEW EPOCH, so its own ``epoch`` is the one we do not
        # have yet. Requiring it to equal the record's epoch refused EVERY rotation
        # that arrived over a live link — measured, on a real one: a rotation to epoch
        # 2 reached a peer at epoch 1 and was refused `epoch_stale` ("the network has
        # rotated (this device is at epoch 1); the link must be re-established"), so no
        # member removal ever propagated over a live link to anybody, and the removed
        # device was left reporting its old epoch, its old member list and
        # `trust: active` while every attempt to reach the network failed (QA round 3,
        # Q-R3-2; the same silence is why peer-b's epoch 3 never reached the Mac in
        # that run's 90 s watch).
        #
        # The GATE'S PURPOSE IS KEPT: the LINK must be at the epoch this record holds,
        # which is what stops a frame from a link established against an older key from
        # acting. What a rotation's own epoch MEANS is settled by the apply path, which
        # is the code that owns the rule (§8.1 step 4: a strictly greater epoch,
        # attributed to the sender, internally consistent, digest-checked) — a second,
        # weaker copy of it here is how the two came to disagree.
        if op == "net_epoch":
            if link.epoch == record.epoch:
                return
            refusal = Refusal(
                "epoch_stale",
                f"this link authenticated at epoch {link.epoch} and the network is at "
                f"{record.epoch}, so a rotation on it is refused before it is read",
            )
            self._refused(link, op, refusal, frame)
            raise refusal
        if link.epoch == record.epoch and recorded_epoch == record.epoch:
            return
        if link.phase == "reconcile" and recorded_epoch == record.epoch - 1:
            return
        refusal = Refusal(
            "epoch_stale",
            f"the network has rotated (this device is at epoch {record.epoch}); the link "
            "must be re-established",
        )
        self._refused(link, op, refusal, frame)
        raise refusal

    def _check_locality(self, link: LinkContext, op: str, frame: dict[str, Any]) -> None:
        """A command that claims ``local`` while arriving over a peer link is a
        protocol error.

        Treating it as a softer path would make the locality field a hint an
        attacker controls; treating it as an error makes it a fact the receiver
        checks. A frame that omits the field is read as ``remote`` BY POSITION —
        it arrived here — never as ``local`` by default.
        """
        if frame.get("locality") == "local":
            refusal = Refusal(
                "protocol_error",
                "a frame arriving over a peer link claimed locality 'local'",
            )
            self._refused(link, op, refusal, frame)
            raise refusal

    def _required_capability(self, op: str, outer_op: str) -> str | None:
        if outer_op == "net_forward":
            capability = INNER_OP_CAPABILITY.get(op)
            if capability is None:
                raise Refusal(
                    "unknown_op",
                    f"the forwarded op {op!r} has no capability decision, so it was refused "
                    "rather than carried",
                )
            return capability
        if op not in OP_CAPABILITY:
            raise Refusal("unknown_op", f"{op!r} is not an operation this build dispatches")
        return OP_CAPABILITY[op]

    def _session_scope(self, link: LinkContext, op: str, frame: dict[str, Any]) -> str | None:
        """A session-scoped op must name a session THIS device owns.

        ``net_forward`` carries the session id in its inner frame; the network-scope
        ops that name one (``net_session_move``, ``net_session_lifecycle``) carry it
        on the outer frame.
        """
        if op == "net_session_create":
            # A create names no EXISTING session: the device that will own it mints
            # the id (§5.3). Any id in such a frame is a protocol error the handler
            # answers by name, not a scope claim to check here — there is nothing
            # yet to own, and checking one would answer "you do not own it" to a
            # frame that is simply malformed.
            return None
        inner = frame.get("frame") if isinstance(frame.get("frame"), dict) else None
        session_id = ""
        if inner is not None:
            session_id = str(inner.get("session_id") or "")
        else:
            session_id = str(frame.get("session_id") or "")
        if not session_id:
            return None
        owned = self._networks.local_session_ids()
        if session_id not in owned:
            refusal = Refusal(
                "not_authorised",
                f"session {session_id} does not live on this device; a relay acts only on "
                "sessions it owns",
            )
            self._refused(link, op, refusal, frame)
            raise refusal
        return session_id

    # -- audit --------------------------------------------------------------

    def _refused(
        self,
        link: LinkContext,
        op: str,
        refusal: Refusal,
        frame: dict[str, Any],
        *,
        capability: str | None = None,
    ) -> None:
        """Every refusal is audited with the actor, subject, op, network and epoch.

        The peer is told a sentence; the operator gets this record. That asymmetry
        is the reason the two live on different objects (``Refusal.sentence`` vs
        the event's ``detail``).
        """
        if self._audit is None:
            return
        self._audit.record(
            AuditEvent(
                event="authorisation_refused",
                actor=link.device_id,
                subject=link.network_id,
                outcome="refused",
                network_id=link.network_id,
                epoch=link.epoch,
                cause=_CAUSE_FOR.get(refusal.code, "not_authorised"),
                detail={
                    "op": op,
                    "capability": capability or "",
                    "phase": link.phase,
                    "link_epoch": link.epoch,
                },
            )
        )


#: The refusal code → the auditor's closed cause enum. Two vocabularies on purpose:
#: the code is what a local caller branches on, the cause is what a forensic query
#: counts, and a code is free to be finer-grained than a cause.
_CAUSE_FOR: dict[str, str] = {
    "not_authorised": "capability_denied",
    "not_a_member": "not_a_member",
    "epoch_stale": "epoch_stale",
    "phase_forbidden": "policy",
    "unknown_op": "policy",
    "protocol_error": "policy",
    "self_link": "policy",
}


def op_tables_are_total() -> list[str]:
    """The names a totality decision is missing — empty is the healthy answer.

    A FUNCTION rather than only a test, so ``lop network doctor`` can report the
    same drift the test would fail on: an op added without a capability decision
    should be visible to the person who added it, not only to CI.
    """
    from typing import get_args

    from local_operator.mobile.types import ControlOp

    missing: list[str] = []
    for name in (*get_args(ControlOp), *NET_OPS):
        if name in INNER_OP_CAPABILITY or name in OP_CAPABILITY:
            continue
        if name in NET_PAIR_OPS:
            # Pair ops are authorised by the ceremony, not by a capability, and are
            # deliberately absent from both tables (see the module docstring).
            continue
        missing.append(name)
    for name in OP_CAPABILITY:
        if name in LOCAL_OPS:
            missing.append(f"{name} (a local op must not be in the peer capability table)")
    return missing


#: The phase a link must be in for a given op, used by the dispatch switch so the
#: relay's handler table and the authoriser cannot disagree about the pair phase.
def phase_for_op(op: str) -> LinkPhase:
    if op in PAIR_OPS:
        return "pair"
    if op in RECONCILE_OPS:
        return "reconcile"
    return "member"
