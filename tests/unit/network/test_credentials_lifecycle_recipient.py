"""Recipient half of ``lifecycle_announce`` and the evidence it may record.

Synthetic ``tmp_path`` rig shared with the consent suite. The headline rule: a
lifecycle observation is METADATA, so no announcement of any state can ever make a
credential ``usable`` — only ``note_resolution`` given a real ``Grant`` does.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import effective
from local_operator.network.credentials import lifecycle as L
from local_operator.network.credentials.types import BrokerError
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig

# The consent suite's rig under the name pytest resolves (assignment, not a shadowing
# import, so flake8 stays quiet).
rig = consent_rig
SCOPE = "acct-one"


@pytest.fixture(autouse=True)
def _clean_receipts():
    effective._receipts.clear()
    yield
    effective._receipts.clear()


def _server(rig: Any) -> Any:
    # The RECIPIENT's relay: root + identity are all member_lifecycle_announce reads.
    return SimpleNamespace(root=rig.root, identity=SimpleNamespace(device_id=rig.peer.device_id))


def _announce(rig: Any, *, state="available", chain=1, material=1, sender="", **over: Any):
    sender = sender or rig.own.device_id
    link = SimpleNamespace(device_id=sender, network_id=N, epoch=1)
    frame = {
        "op": "net_broker",
        "kind": "lifecycle_announce",
        "req": 1,
        "from_device": sender,
        "owner_device": sender,
        "recipient_device": rig.peer.device_id,
        "entries": [
            {
                "credential_key": P,
                "account_scope_ref": SCOPE,
                "state": state,
                "material_generation": material,
                "login_chain_revision": chain,
            }
        ],
    }
    frame.update(over)
    return L.member_lifecycle_announce(_server(rig), link, frame)


def _hw(rig: Any):
    return L.LifecycleStore(N, rig.root).load().high_water


def test_owner_announcement_is_applied_and_acked(rig):
    detail = _announce(rig)
    assert detail["kind"] == "ack"
    assert detail["entries"][0]["applied"] is True
    assert _hw(rig) == {(rig.own.device_id, P, SCOPE): (1, 1)}


def test_replay_and_stale_do_not_move_the_high_water(rig):
    assert _announce(rig, chain=2, material=3)["entries"][0]["applied"] is True
    replay = _announce(rig, chain=2, material=3)["entries"][0]
    stale = _announce(rig, chain=1, material=1)["entries"][0]
    assert (replay["applied"], replay["reason"]) == (False, L.OBSERVED_REPLAY)
    assert (stale["applied"], stale["reason"]) == (False, L.OBSERVED_STALE)
    assert _hw(rig) == {(rig.own.device_id, P, SCOPE): (2, 3)}


def test_withdrawn_is_reapplied_at_an_unchanged_counter(rig):
    # A sticky deny moves neither counter; forward-only alone would drop it.
    _announce(rig, chain=1, material=1)
    detail = _announce(rig, state="withdrawn", chain=1, material=1)
    assert detail["entries"][0]["applied"] is True
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"


def test_sender_that_is_not_the_placement_owner_changes_nothing(rig):
    # The recipient itself is a holder, not the owner of P: an announce from it is refused.
    detail = _announce(rig, sender=rig.peer.device_id, recipient_device=rig.peer.device_id)
    assert detail["code"] == "not_owner"
    assert _hw(rig) == {} and effective._receipts == {}


def test_frame_addressed_to_another_device_changes_nothing(rig):
    detail = _announce(rig, recipient_device="d_" + "9" * 32)
    assert detail["code"] == "identity_mismatch"
    assert _hw(rig) == {}


def test_unknown_key_changes_nothing(rig):
    link = SimpleNamespace(device_id=rig.own.device_id, network_id=N, epoch=1)
    frame = {
        "kind": "lifecycle_announce",
        "owner_device": rig.own.device_id,
        "recipient_device": rig.peer.device_id,
        "entries": [
            {
                "credential_key": "not-shared",
                "account_scope_ref": SCOPE,
                "state": "available",
                "material_generation": 1,
                "login_chain_revision": 1,
            }
        ],
    }
    assert L.member_lifecycle_announce(_server(rig), link, frame)["code"] == "unknown_key"
    assert _hw(rig) == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"entries": []},
        {"note": "x" * 17000},
        {"access_token": "synthetic"},
    ],
)
def test_malformed_announcement_changes_nothing(rig, bad):
    detail = _announce(rig, **bad)
    assert detail["code"] == "malformed_announce" or detail["kind"] == "error"
    assert _hw(rig) == {} and effective._receipts == {}


def test_scope_unknown_is_not_a_state_an_owner_pushes(rig):
    assert _announce(rig, state="scope_unknown")["code"] == "malformed_announce"


@pytest.mark.parametrize(
    "state", ["available", "absent", "withdrawn", "unsupported", "non_portable", "bogus", ""]
)
def test_no_lifecycle_observation_can_ever_be_usable(rig, state):
    receipt = effective.note_lifecycle_observation(rig.root, P, state, owner=rig.own.device_id)
    assert receipt is None or (receipt.state != "effective_borrow" and not receipt.usable)
    # And through the whole recipient path, whatever state arrives.
    _announce(rig, state="available" if state == "bogus" else state or "available")
    for stored in effective._receipts.values():
        assert not stored.usable


def test_observation_after_a_refused_resolve_stays_pending(rig):
    # The resolution path (a real Grant) is the consent suite's; here: an observation
    # layered on a refused resolve never turns it usable.
    effective.note_resolution(
        rig.root, P, BrokerError(code="owner_offline", owner_device=rig.own.device_id)
    )
    receipt = effective.note_lifecycle_observation(
        rig.root, P, "available", owner=rig.own.device_id
    )
    assert receipt is not None and receipt.state == "pending" and not receipt.usable


# -- relay routing: a device that owns nothing still receives announcements --------


def _no_broker_handler(rig: Any) -> Any:
    from local_operator.network.credentials.owner import MeshCredentialBroker

    return MeshCredentialBroker.relay_handler(_server(rig))


def _announce_frame(rig: Any, kind: str = "lifecycle_announce") -> dict[str, Any]:
    return {
        "op": "net_broker",
        "kind": kind,
        "req": 7,
        "from_device": rig.own.device_id,
        "owner_device": rig.own.device_id,
        "recipient_device": rig.peer.device_id,
        "entries": [
            {
                "credential_key": P,
                "account_scope_ref": SCOPE,
                "state": "withdrawn",
                "material_generation": 1,
                "login_chain_revision": 1,
            }
        ],
    }


def test_a_device_that_owns_nothing_acks_a_lifecycle_announce(rig):
    # The recipient owns no credential, so ``broker_for_relay`` is None; the
    # announcement must still be applied (it is addressed to exactly this device).
    link = SimpleNamespace(device_id=rig.own.device_id, network_id=N, epoch=1)
    detail = _no_broker_handler(rig)(link, _announce_frame(rig))
    assert detail["kind"] == "ack" and detail["entries"][0]["applied"] is True
    assert _hw(rig) == {(rig.own.device_id, P, SCOPE): (1, 1)}


@pytest.mark.parametrize("kind", ["grant", "lifecycle_read", "lifecycle_ack", "lifecycle_x"])
def test_other_kinds_on_a_device_that_owns_nothing_keep_the_refusal(rig, kind):
    from local_operator.network.types import MeshRefusal

    link = SimpleNamespace(device_id=rig.own.device_id, network_id=N, epoch=1)
    with pytest.raises(MeshRefusal) as excinfo:
        _no_broker_handler(rig)(link, _announce_frame(rig, kind))
    assert excinfo.value.code == "not_implemented"
    assert _hw(rig) == {}


@pytest.mark.parametrize(
    "reply",
    [
        None,
        {"detail": "ack"},
        {"detail": {"kind": "grant", "access_token": "[redacted]"}},
        {"detail": {"kind": "ack", "entries": [{"note": "x" * 17000}]}},
        {"detail": {"kind": "ack", "entries": ["not-an-object"]}},
        {"detail": {"kind": "ack", "entries": [{}] * 9}},
        {"detail": {"kind": "ack", "entries": {"a": 1}}},
        {"detail": {"kind": "something-else"}},
    ],
)
def test_a_malformed_lifecycle_reply_detail_is_refused(reply):
    with pytest.raises(ValueError):
        L.checked_reply_detail(reply)


def test_a_well_formed_lifecycle_reply_detail_is_returned():
    ack = {"kind": "ack", "entries": [{"credential_key": P, "applied": True}]}
    assert L.checked_reply_detail({"op": "ack", "detail": ack}) is ack


def test_the_authorizer_admits_a_lifecycle_announce_only_with_broker_credential():
    # The routing above relies on the transport's gate, so pin that gate: a recipient
    # admits the frame iff the SENDER holds ``broker_credential`` on its member row.
    from tests.unit.network.test_authorizer import link, make

    frame = {"op": "net_broker", "kind": "lifecycle_announce", "req": 1}
    authorizer, _audit = make()
    assert authorizer.check(link(capabilities={"broker_credential"}), frame).capability == (
        "broker_credential"
    )
    from local_operator.network.types import Refusal

    with pytest.raises(Refusal):
        authorizer.check(link(capabilities={"view"}), frame)


# -- the sender: lifecycle_tick_step ---------------------------------------------


class _Wire:
    """A scripted peer link: records frames, answers from ``reply``."""

    def __init__(self, caps: set[str], reply: Any = None) -> None:
        self.capabilities = frozenset(caps)
        self.frames: list[dict[str, Any]] = []
        self.reply = reply

    def request(self, frame: dict[str, Any], timeout: float = 0.0) -> Any:
        self.frames.append(frame)
        if callable(self.reply):
            return self.reply(frame)
        return {"op": "ack", "req": frame["req"], "detail": self.reply}


class _Audit:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


def _owner_server(rig: Any, wire_link: Any) -> Any:
    from local_operator.network import store

    for record in store.list_networks(rig.root):
        record.self_capabilities = ["broker_credential"]
        store.save(record, rig.root)
    return SimpleNamespace(
        root=rig.root,
        identity=SimpleNamespace(device_id=rig.own.device_id, name="owner"),
        audit=_Audit(),
        _ensure_link=lambda device: wire_link,
        _next_relay_req=lambda: 5,
    )


def _seed(rig: Any, rows: int = 1) -> None:
    life = L.LifecycleStore(N, rig.root)
    life.record_native(P, SCOPE, L.NativeFacts(present=True, new_login=True))
    for index in range(rows - 1):
        life.record_native(P, f"scope-{index}", L.NativeFacts(present=True, new_login=True))


def _echo_applied(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "op": "ack",
        "req": frame["req"],
        "detail": {
            "kind": "ack",
            "entries": [
                {
                    "credential_key": e["credential_key"],
                    "account_scope_ref": e["account_scope_ref"],
                    "applied": True,
                    "reason": "applied",
                }
                for e in frame["entries"]
            ],
        },
    }


def test_the_tick_sends_nothing_to_a_peer_that_did_not_negotiate_the_capability(rig):
    _seed(rig)
    old_peer = _Wire({"mesh-net-v1", "credential-broker-v1"}, _echo_applied)
    server = _owner_server(rig, old_peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "skipped:peer_lacks_lifecycle"
    assert old_peer.frames == []


def test_the_tick_announces_to_a_peer_that_negotiated_it(rig):
    _seed(rig)
    new_peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    server = _owner_server(rig, new_peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    (frame,) = new_peer.frames
    assert frame["kind"] == "lifecycle_announce" and frame["recipient_device"] == (
        rig.peer.device_id
    )
    assert frame["entries"][0]["state"] == "available"
    assert [e.event for e in server.audit.events] == ["credential.lifecycle_announced"]
    # Delivered rows are not re-sent while the counters are unchanged.
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "in_sync"
    assert len(new_peer.frames) == 1


def test_the_tick_caps_one_exchange_at_eight_entries(rig):
    _seed(rig, rows=11)
    new_peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    server = _owner_server(rig, new_peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    assert len(new_peer.frames[0]["entries"]) == 8
    # The remainder follows on the next tick, recomputed from the counters.
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    assert len(new_peer.frames[1]["entries"]) == 3


def test_observation_table_full_is_not_counted_as_delivered(rig):
    _seed(rig)

    def full(frame: dict[str, Any]) -> dict[str, Any]:
        reply = _echo_applied(frame)
        for entry in reply["detail"]["entries"]:
            entry.update(applied=False, reason=L.OBSERVED_TABLE_FULL)
        return reply

    new_peer = _Wire({"credential-lifecycle-v1"}, full)
    server = _owner_server(rig, new_peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "retry_later"
    assert server.audit.events == []
    # Not recorded as sent, so the next tick offers the row again.
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "retry_later"
    assert len(new_peer.frames) == 2


def test_a_malformed_reply_is_not_counted_as_delivered(rig):
    _seed(rig)
    new_peer = _Wire({"credential-lifecycle-v1"}, {"kind": "ack", "access_token": "[redacted]"})
    server = _owner_server(rig, new_peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "failed"
    assert server.audit.events == []


def test_install_registers_the_tick_step_once():
    from local_operator.network import definitions

    L.install()
    L.install()
    assert definitions._tick_steps().count(L.lifecycle_tick_step) == 1  # noqa: SLF001
