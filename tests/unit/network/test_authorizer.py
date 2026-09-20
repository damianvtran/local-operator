"""The authorisation chokepoint: the tables' totality and every refusal path."""

from __future__ import annotations

from typing import Any, get_args

import pytest

from local_operator.network import authorizer as az
from local_operator.network import types

NETWORK = "n_0123456789abcdef01234567"


def control_ops() -> tuple[str, ...]:
    """The session plane's op names, read from the ONE declaration of them.

    Imported here rather than re-listed, because a hand-kept copy of the session
    vocabulary would be the second spelling that makes the totality test agree with
    itself while disagreeing with the wire.
    """
    from local_operator.mobile.types import ControlOp

    return get_args(ControlOp)


class FakeState(az.NetworkState):
    def __init__(self, *, epoch: int = 1, sessions: set[str] | None = None, present: bool = True):
        self.epoch = epoch
        self.sessions = sessions or set()
        self.present = present

    def network(self, network_id: str) -> types.NetworkRecord | None:
        if not self.present:
            return None
        return types.NetworkRecord(network_id=network_id, name="home-net", epoch=self.epoch)

    def local_session_ids(self) -> set[str]:
        return self.sessions


class FakeAudit:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)

    def names(self) -> list[str]:
        return [event.event for event in self.events]


def link(
    *,
    capabilities: set[str] | None = None,
    phase: str = "member",
    epoch: int = 1,
    device_id: str = "d_" + "b" * 32,
) -> types.LinkContext:
    return types.LinkContext(
        link_id="l1",
        device_id=device_id,
        instance_id="i_1",
        network_id=NETWORK,
        epoch=epoch,
        capabilities=frozenset(capabilities if capabilities is not None else {"list", "view"}),
        phase=phase,  # type: ignore[arg-type]
        peer_addr="127.0.0.1:1",
    )


def make(
    *, state: FakeState | None = None, audit: FakeAudit | None = None
) -> tuple[az.Authorizer, FakeAudit]:
    audit = audit or FakeAudit()
    return az.Authorizer(state or FakeState(), audit), audit


# ---------------------------------------------------------------------------
# Totality — the guard that makes a chokepoint worth having
# ---------------------------------------------------------------------------


def test_the_op_capability_tables_are_total() -> None:
    """Fails BY NAME, and the failure it prevents is a *permissive default*.

    Every session-plane op (``ControlOp``), every peer-scope op (``NET_OPS``) and
    every pair op must have a decided authorisation. A new op added without one is
    refused at runtime AND named here.
    """
    missing = az.op_tables_are_total()
    assert missing == [], f"ops with no capability decision: {missing}"


def test_the_totality_check_fails_by_name_when_an_entry_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PROVE THE TEST CAN FAIL: drop one entry and the guard names it."""
    monkeypatch.delitem(types.OP_CAPABILITY, "net_catalog")
    assert "net_catalog" in az.op_tables_are_total()


def test_a_local_op_in_the_peer_table_is_a_bug_not_a_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(types.OP_CAPABILITY, "net_status", "view")
    reported = az.op_tables_are_total()
    assert any("net_status" in entry for entry in reported)


def test_the_vocabulary_is_the_one_agreed_set() -> None:
    """Convergence: nine names, and the rejected draft names are absent."""
    assert types.CAPABILITIES == frozenset(
        {
            "broker_credential",
            "admin",
            "list",
            "view",
            "prompt",
            "steer",
            "stop",
            "slash",
            "delete",
            "move",
        }
    )
    for rejected in ("broker:request", "broker:grant", "member:admin", "trust"):
        assert rejected not in types.CAPABILITIES


def test_every_table_value_is_a_real_capability() -> None:
    for op, capability in types.OP_CAPABILITY.items():
        if capability is not None:
            assert capability in types.CAPABILITIES, op
    for op, capability in types.INNER_OP_CAPABILITY.items():
        assert capability in types.CAPABILITIES, op


def test_every_control_op_has_an_inner_capability() -> None:
    for name in control_ops():
        assert name in types.INNER_OP_CAPABILITY, name


def test_roles_are_the_documented_sets() -> None:
    assert types.ROLE_CAPABILITIES["read"] == frozenset({"list", "view"})
    assert "prompt" not in types.ROLE_CAPABILITIES["read"]
    assert "admin" in types.ROLE_CAPABILITIES["admin"]
    assert types.ROLE_CAPABILITIES["admin"] == types.CAPABILITIES


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_read_role_cannot_prompt_steer_stop_or_slash() -> None:
    """Four refusals, four audit records — and the read member never gets past the
    chokepoint, which is the point of having one.

    The ops travel as ``net_forward`` carriers, which is the only shape in which a
    session-plane op can arrive over a link — so this is the real request, not a
    stand-in for it.
    """
    authorizer, audit = make(state=FakeState(sessions={"s_local"}))
    read_link = link(capabilities={"list", "view"})
    for op in ("prompt", "steer", "abort", "slash"):
        with pytest.raises(types.Refusal) as excinfo:
            authorizer.check(
                read_link,
                {"op": "net_forward", "req": 1, "frame": {"op": op, "session_id": "s_local"}},
            )
        assert excinfo.value.code == "not_authorised"
    assert audit.names().count("authorisation_refused") == 4
    assert all(event.cause == "capability_denied" for event in audit.events)


def test_read_role_may_list_and_view() -> None:
    authorizer, _audit = make()
    read_link = link(capabilities={"list", "view"})
    assert authorizer.check(read_link, {"op": "net_catalog", "req": 1}).action == "net_catalog"
    assert authorizer.check(read_link, {"op": "net_member_list", "req": 2}).action == (
        "net_member_list"
    )


def test_net_forward_resolves_to_the_inner_op_capability() -> None:
    """A carrier is not an authorisation bypass: the inner op's capability is what
    is required, and a read member carrying ``prompt`` is refused."""
    authorizer, _audit = make()
    read_link = link(capabilities={"list", "view"})
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(
            read_link,
            {"op": "net_forward", "req": 1, "frame": {"op": "prompt", "session_id": "s1"}},
        )
    assert excinfo.value.code == "not_authorised"
    assert authorizer.effective_op(read_link, {"op": "net_forward", "frame": {"op": "abort"}}) == (
        "abort"
    )


def test_a_forwarded_op_with_no_decision_is_refused_rather_than_carried() -> None:
    authorizer, _audit = make()
    drive_link = link(capabilities=set(types.ROLE_CAPABILITIES["drive"]))
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(drive_link, {"op": "net_forward", "req": 1, "frame": {"op": "teleport"}})
    assert excinfo.value.code == "unknown_op"


def test_an_unknown_op_is_refused() -> None:
    authorizer, _audit = make()
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(), {"op": "net_do_everything", "req": 1})
    assert excinfo.value.code == "unknown_op"


def test_a_frame_with_no_op_is_a_protocol_error() -> None:
    authorizer, _audit = make()
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(), {"req": 1})
    assert excinfo.value.code == "protocol_error"


def test_a_frame_claiming_local_locality_is_a_protocol_error() -> None:
    """The spine's rule: a command that arrives over a peer link claiming ``local``
    is a protocol error, not a softer path."""
    authorizer, _audit = make()
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(), {"op": "net_catalog", "req": 1, "locality": "local"})
    assert excinfo.value.code == "protocol_error"


def test_a_frame_that_omits_locality_is_read_as_remote() -> None:
    authorizer, _audit = make()
    assert authorizer.check(link(), {"op": "net_catalog", "req": 1}).action == "net_catalog"


def test_a_local_op_over_a_peer_link_is_refused() -> None:
    authorizer, _audit = make()
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(), {"op": "net_init", "req": 1})
    assert excinfo.value.code == "unknown_op"


def test_a_stale_epoch_is_refused() -> None:
    authorizer, _audit = make(state=FakeState(epoch=5))
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(epoch=4), {"op": "net_catalog", "req": 1, "epoch": 4})
    assert excinfo.value.code == "epoch_stale"


def test_a_rotation_is_admitted_by_the_epoch_gate() -> None:
    """A ``net_epoch`` frame carries the epoch we do NOT have yet, and that is the point.

    Measured on a real link, before this rule: a rotation to epoch 2 reached a peer at
    epoch 1 and was refused ``epoch_stale`` by this gate — the check required the
    frame's own epoch to equal the receiver's, which no rotation can ever satisfy.
    The consequence was that NO member removal propagated over a live link to anyone,
    so a removed device was never told it had been removed (QA round 3, Q-R3-2).

    The gate keeps its purpose: the LINK must be at the epoch the record holds. What a
    rotation's epoch means is the apply path's rule (§8.1 step 4), and the two used to
    be two copies of one rule that disagreed.
    """
    authorizer, _audit = make(state=FakeState(epoch=1))
    granted = authorizer.check(
        link(epoch=1, capabilities={"admin"}), {"op": "net_epoch", "req": 1, "epoch": 2}
    )
    assert granted.action == "net_epoch"


def test_a_rotation_on_a_link_that_is_not_at_our_epoch_is_still_refused() -> None:
    """The half that must NOT move: a rotation arriving on a link authenticated against
    an epoch this record no longer holds is refused before the frame is read."""
    authorizer, _audit = make(state=FakeState(epoch=5))
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(
            link(epoch=4, capabilities={"admin"}), {"op": "net_epoch", "req": 1, "epoch": 6}
        )
    assert excinfo.value.code == "epoch_stale"


def test_a_network_this_device_left_refuses_the_link() -> None:
    authorizer, _audit = make(state=FakeState(present=False))
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(), {"op": "net_catalog", "req": 1})
    assert excinfo.value.code == "not_a_member"


def test_a_session_on_a_third_device_is_refused_not_forwarded() -> None:
    authorizer, _audit = make(state=FakeState(sessions={"s_local"}))
    admin_link = link(capabilities=set(types.ROLE_CAPABILITIES["admin"]))
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(
            admin_link,
            {"op": "net_forward", "req": 1, "frame": {"op": "prompt", "session_id": "s_other"}},
        )
    assert excinfo.value.code == "not_authorised"
    granted = authorizer.check(
        admin_link,
        {"op": "net_forward", "req": 2, "frame": {"op": "prompt", "session_id": "s_local"}},
    )
    assert granted.session_id == "s_local"


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def test_pair_ops_are_refused_on_a_member_link_and_member_ops_on_a_pair_link() -> None:
    """The rule asserted in BOTH directions, so a pair op cannot become a
    pre-membership general-purpose op."""
    authorizer, _audit = make()
    for op in types.NET_PAIR_OPS:
        with pytest.raises(types.Refusal) as excinfo:
            authorizer.check(link(), {"op": op, "req": 1})
        assert excinfo.value.code == "phase_forbidden"
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(link(phase="pair"), {"op": "net_catalog", "req": 1})
    assert excinfo.value.code == "phase_forbidden"
    # A pair op IS dispatchable in the pair phase, with no capability required.
    granted = authorizer.check(link(phase="pair"), {"op": "net_pair_ready", "req": 1})
    assert granted.capability is None


def test_reconcile_phase_admits_only_two_ops() -> None:
    authorizer, _audit = make(state=FakeState(epoch=8))
    reconcile = link(phase="reconcile", epoch=7, capabilities=set(types.ROLE_CAPABILITIES["admin"]))
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(
            reconcile,
            {
                "op": "net_forward",
                "req": 1,
                "epoch": 7,
                "frame": {"op": "prompt", "session_id": "s"},
            },
        )
    assert excinfo.value.code == "phase_forbidden"
    assert authorizer.check(reconcile, {"op": "net_reconcile", "req": 2, "epoch": 7}).action == (
        "net_reconcile"
    )
    assert authorizer.check(reconcile, {"op": "ping", "req": 3, "epoch": 7}).action == "ping"


def test_net_bye_needs_a_member_but_no_capability() -> None:
    authorizer, _audit = make()
    granted = authorizer.check(link(capabilities=set()), {"op": "net_bye", "req": 1})
    assert granted.action == "net_bye"
    assert granted.capability is None


def test_the_wire_only_ever_gets_a_sentence() -> None:
    """The peer is told that it was refused, never which of membership, epoch or
    capability failed; the real cause is in the local audit record."""
    authorizer, audit = make()
    with pytest.raises(types.Refusal) as excinfo:
        authorizer.check(
            link(capabilities={"view"}), {"op": "net_forward", "req": 1, "frame": {"op": "prompt"}}
        )
    sentence = excinfo.value.sentence
    assert "prompt" in sentence  # the peer may know what IT asked for
    assert NETWORK not in sentence
    assert audit.events[-1].detail["capability"] == "prompt"
