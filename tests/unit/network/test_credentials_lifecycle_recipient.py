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
SCOPE = f"native:{P}:YWNjdC1vbmU"


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
    link = SimpleNamespace(
        device_id=sender,
        network_id=N,
        epoch=1,
        capabilities={"credential-lifecycle-v1"},
        local_capabilities={"credential-lifecycle-v1"},
    )
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
    link = SimpleNamespace(
        device_id=rig.own.device_id,
        network_id=N,
        epoch=1,
        capabilities={"credential-lifecycle-v1"},
        local_capabilities={"credential-lifecycle-v1"},
    )
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
    link = SimpleNamespace(
        device_id=rig.own.device_id,
        network_id=N,
        epoch=1,
        capabilities={"credential-lifecycle-v1"},
        local_capabilities={"credential-lifecycle-v1"},
    )
    detail = _no_broker_handler(rig)(link, _announce_frame(rig))
    assert detail["kind"] == "ack" and detail["entries"][0]["applied"] is True
    assert _hw(rig) == {(rig.own.device_id, P, SCOPE): (1, 1)}


@pytest.mark.parametrize("kind", ["grant", "lifecycle_read", "lifecycle_ack", "lifecycle_x"])
def test_other_kinds_on_a_device_that_owns_nothing_keep_the_refusal(rig, kind):
    from local_operator.network.types import MeshRefusal

    link = SimpleNamespace(
        device_id=rig.own.device_id,
        network_id=N,
        epoch=1,
        capabilities={"credential-lifecycle-v1"},
        local_capabilities={"credential-lifecycle-v1"},
    )
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
        self.local_capabilities = frozenset({"credential-lifecycle-v1"})
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
    rig.auth.delete_credentials_for_provider(P)
    for identity in ["acct-one"] + [f"scope-{index}" for index in range(rows - 1)]:
        rig.auth.upsert_credential(
            P,
            {
                "type": "oauth",
                "access": "test-access",
                "refresh": "test-refresh",
                "email": identity,
                "account_id": identity,
                "expires": 9999999999999,
            },
        )


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


def test_every_advertised_op_capability_has_a_registered_handler(rig):
    from local_operator.network import relay, wire
    from local_operator.network.credentials import owner
    from local_operator.network.types import OP_CAPABILITY

    # Keep the classification exhaustive: a new advertised op feature cannot silently
    # inherit the pure-feature exception and promise an unimplemented receiver.
    handlers = {
        "mesh-net-v1": "net_catalog",
        "session-mobility-v1": "net_session_move",
        "credential-broker-v1": "net_broker",
        wire.CREDENTIAL_LIFECYCLE_V1: "net_broker",
        "peer-readiness-v1": "net_readiness",
        "mcp-defs-v1": "net_mcp_defs",
        "mesh-update-v1": "net_update",
        "peer-journal-source-v1": "net_session_history",
    }
    pure_features = {
        # Reserved pool feature: no compute-pool op or dispatch contract exists yet.
        "compute-pool-v1",
        # Pair-offer is handled during the handshake, before op dispatch exists.
        "pair-offer-v1",
        # Compression changes record encoding, not the set of dispatchable ops.
        "zlib-records-v1",
    }
    assert set(wire.LINK_CAPABILITIES) == set(handlers) | pure_features
    server = relay.RelayServer(root=rig.root, identity=rig.own)
    for capability, op in handlers.items():
        assert op in OP_CAPABILITY, capability
        assert callable(server._handlers.get(op)), capability  # noqa: SLF001
    assert owner.broker_for_relay(_server(rig)) is None
    L.install()
    from local_operator.network import definitions

    assert L.lifecycle_tick_step in definitions._tick_steps()  # noqa: SLF001


def test_new_and_old_peers_negotiate_lifecycle_without_changing_finite_grants(rig):
    from local_operator.network import wire
    from tests.unit.network.test_credentials_consent import grant

    _seed(rig)
    advertised = list(wire.LINK_CAPABILITIES)
    old_advertised = [cap for cap in advertised if cap != wire.CREDENTIAL_LIFECYCLE_V1]
    old_peer = _Wire(set(wire.negotiate_capabilities(advertised, old_advertised)), _echo_applied)
    old_server = _owner_server(rig, old_peer)
    assert L.lifecycle_tick_step(old_server, rig.peer.device_id) == "skipped:peer_lacks_lifecycle"
    assert old_peer.frames == []
    # The established broker path still mints a finite grant without any lifecycle
    # negotiation: feature absence is not absence of login or permission.
    assert grant(rig)["kind"] == "grant"
    new_peer = _Wire(set(wire.negotiate_capabilities(advertised, advertised)), _echo_applied)
    new_server = _owner_server(rig, new_peer)
    assert L.lifecycle_tick_step(new_server, rig.peer.device_id) == "delivered"
    assert [frame["kind"] for frame in new_peer.frames] == ["lifecycle_announce"]


def test_native_login_refresh_logout_new_login_registered_cadence(rig):
    """Actual native metadata writes, not hand-seeded lifecycle counters."""
    from local_operator.network import definitions

    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    server = _owner_server(rig, peer)
    _seed(rig)
    L.install()
    clock = definitions.DefinitionsSyncer(server)
    assert clock.tick(now=1000) == [(rig.peer.device_id, "skipped:no_admin")]
    assert len(peer.frames) == 1  # sibling ran despite definitions' own admin refusal
    first = peer.frames[-1]["entries"][0]
    assert (first["login_chain_revision"], first["material_generation"]) == (1, 1)
    native = rig.auth.list_credentials(P)[0]
    incarnation = rig.auth.native_login_incarnation(native.id)
    # The actual refresh persistence SQL cannot change the login incarnation.
    rig.auth._conn.execute(
        "UPDATE auth_credentials SET updated_at = updated_at + 1 WHERE id = ?", (native.id,)
    )
    rig.auth._conn.commit()
    clock.tick(now=1061)
    rotation = peer.frames[-1]["entries"][0]
    assert rig.auth.native_login_incarnation(native.id) == incarnation
    assert (rotation["login_chain_revision"], rotation["material_generation"]) == (1, 2)
    rig.auth.delete_credentials_for_provider(P)
    clock.tick(now=1122)
    assert peer.frames[-1]["entries"][0]["state"] == "absent"
    _seed(rig)
    clock.tick(now=1183)
    back = peer.frames[-1]["entries"][0]
    assert (back["login_chain_revision"], back["material_generation"]) == (2, 4)


def test_sender_denial_precedes_all_native_reads(rig, monkeypatch):
    from local_operator.network.credentials import consent, placement

    _seed(rig)
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "test-deny", rig.root
    )
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.grant(P, rig.peer.device_id, scope="session", by=rig.own.device_id)
    monkeypatch.setattr(L, "_native_contexts", lambda *args: pytest.fail("denied native read"))
    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    assert L.lifecycle_tick_step(_owner_server(rig, peer), rig.peer.device_id) == "in_sync"
    assert peer.frames == []


def test_sender_rotates_retry_prefix_and_reannounces_new_link(rig):
    _seed(rig, 11)

    def selective(frame):
        reply = _echo_applied(frame)
        for result, item in zip(reply["detail"]["entries"], frame["entries"]):
            if item["account_scope_ref"] in sorted(L._native_contexts(rig.root, P))[:8]:
                result.update(applied=False, reason=L.OBSERVED_TABLE_FULL)
        return reply

    peer = _Wire({"credential-lifecycle-v1"}, selective)
    server = _owner_server(rig, peer)
    for _ in range(3):
        L.lifecycle_tick_step(server, rig.peer.device_id)
    scopes = {item["account_scope_ref"] for frame in peer.frames for item in frame["entries"]}
    assert len(scopes) == 11
    assert all(len(frame["entries"]) <= 8 for frame in peer.frames)
    # A distinct link is real catch-up evidence, not a manual ledger reset.
    replacement = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    server._ensure_link = lambda device: replacement
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    assert len(replacement.frames[0]["entries"]) == 8


@pytest.mark.parametrize("local,remote", [(False, True), (True, False), (False, False)])
def test_receiver_requires_both_transcript_advertisements(rig, local, remote):
    feature = {"credential-lifecycle-v1"}
    link = SimpleNamespace(
        device_id=rig.own.device_id,
        network_id=N,
        local_capabilities=feature if local else set(),
        capabilities=feature if remote else set(),
    )
    result = L.member_lifecycle_announce(_server(rig), link, _announce_frame(rig))
    assert result["code"] == "unsupported"
    assert not _hw(rig)


def test_capacity_blocks_new_restrictive_identity_and_reserves_other_owner(rig, monkeypatch):
    monkeypatch.setattr(L, "MAX_OWNER_OBSERVATIONS", 1)
    life = L.LifecycleStore(N, rig.root)
    assert life.observe(L.Observation(rig.own.device_id, P, "occupied", 1, 1)).applied
    assert life.observe(L.Observation("other-owner", P, "free", 1, 1)).applied
    result = _announce(rig, state="withdrawn")["entries"][0]
    assert result == dict(
        credential_key=P, account_scope_ref=SCOPE, applied=False, reason=L.OBSERVED_TABLE_FULL
    )
    assert effective.effective_access(P, root=rig.root).state == "withdrawn"
    # Capacity reservation works before overflow; after an omitted restriction,
    # the deliberately coarse network latch requires authoritative recovery.
    assert (
        life.observe(L.Observation("other-owner", P, "free", 2, 2)).reason
        == L.NEEDS_AUTHORITATIVE_BASELINE
    )


def test_unknown_native_login_baseline_is_not_seeded(rig):
    _seed(rig)
    rig.auth._conn.execute("UPDATE auth_credentials SET login_incarnation = NULL")
    rig.auth._conn.commit()
    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    assert L.lifecycle_tick_step(_owner_server(rig, peer), rig.peer.device_id) == "in_sync"
    assert not L.LifecycleStore(N, rig.root).load().rows
    assert not peer.frames


def test_late_denial_withdraws_only_previously_disclosed_evidence(rig, monkeypatch):
    from local_operator.network.credentials import consent, placement

    _seed(rig)
    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    server = _owner_server(rig, peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    original = dict(peer.frames[-1]["entries"][0])
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "test-deny", rig.root
    )
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.grant(P, rig.peer.device_id, scope="session", by=rig.own.device_id)
    monkeypatch.setattr(L, "_native_contexts", lambda *args: pytest.fail("read after denial"))
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    assert peer.frames[-1]["entries"] == [{**original, "state": "withdrawn"}]
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "in_sync"


def test_partial_observation_failure_returns_truthful_per_entry_results(rig, monkeypatch):
    from local_operator.network.types import MeshRefusal

    original = L.LifecycleStore.observe

    def fail_second(target, obs):
        if obs.account_scope_ref == "second":
            raise MeshRefusal("busy", "try later")
        return original(target, obs)

    monkeypatch.setattr(L.LifecycleStore, "observe", fail_second)
    first = _announce_frame(rig)["entries"][0]
    reply = _announce(rig, entries=[first, {**first, "account_scope_ref": "second"}])
    assert reply["kind"] == "ack"
    assert reply["entries"][0]["applied"] is True
    assert reply["entries"][1]["applied"] is False
    assert reply["entries"][1]["reason"] == "busy"
    assert len(_hw(rig)) == 1


def test_account_denial_preserves_other_confirmed_native_accounts(rig):
    from local_operator.network.credentials import consent, placement

    _seed(rig, 2)
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P, "account_ref": "acct-one"},
        "test-deny",
        rig.root,
    )
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.grant(P, rig.peer.device_id, scope="session", by=rig.own.device_id)
    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    assert L.lifecycle_tick_step(_owner_server(rig, peer), rig.peer.device_id) == "delivered"
    assert len(peer.frames[0]["entries"]) == 1
    assert peer.frames[0]["entries"][0]["account_scope_ref"] != SCOPE


def test_sender_preserves_verified_future_permission_without_finite_holder(rig):
    from local_operator.network.credentials import consent, placement
    from tests.unit.network.test_credentials_consent import request

    approval, run = request(rig)
    consent.provision_from_run(approval, run, rig.subject, rig.root)
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.revoke(P, rig.peer.device_id)
    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    assert L.lifecycle_tick_step(_owner_server(rig, peer), rig.peer.device_id) == "delivered"
    assert len(peer.frames[0]["entries"]) == 1
    assert peer.frames[0]["entries"][0]["state"] == "available"


def test_equal_counter_availability_cannot_lift_durable_withdrawal(rig):
    _announce(rig)
    _announce(rig, state="withdrawn")
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    replay = _announce(rig)["entries"][0]
    assert replay == dict(
        credential_key=P, account_scope_ref=SCOPE, applied=False, reason=L.OBSERVED_REPLAY
    )
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    # The process receipt is disposable, the restrictive freshness evidence is not.
    effective._receipts.clear()
    _announce(rig)
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    # A rotation is not new authorization, even at a higher material generation.
    _announce(rig, material=2)
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    effective._receipts.clear()
    stale = _announce(rig)["entries"][0]
    assert stale["reason"] == L.OBSERVED_STALE
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    final = _announce(rig, chain=2, material=3)
    assert effective._receipts[(effective._root(rig.root), P)].state == "pending", (
        final,
        L.LifecycleStore(N, rig.root).load().to_json(),
    )


def test_other_scope_availability_cannot_lift_this_keys_restriction(rig):
    _announce(rig, state="withdrawn")
    entry = {
        **_announce_frame(rig)["entries"][0],
        "state": "available",
        "account_scope_ref": "other-scope",
        "login_chain_revision": 2,
        "material_generation": 2,
    }
    _announce(rig, entries=[entry])
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    doc = L.LifecycleStore(N, rig.root).load()
    assert doc.observation_states[(rig.own.device_id, P, SCOPE)] == "withdrawn"
    assert doc.observation_states[(rig.own.device_id, P, "other-scope")] == "available"


def test_legacy_counter_only_replay_is_unknown_not_receipt_permission(rig):
    L.LifecycleStore(N, rig.root).observe(L.Observation(rig.own.device_id, P, SCOPE, 1, 1))
    _announce(rig)
    assert (effective._root(rig.root), P) not in effective._receipts
    _announce(rig, state="withdrawn")
    effective._receipts.clear()
    _announce(rig)
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"


def test_denied_reconnect_retains_disclosure_not_delivery_ack(rig, monkeypatch):
    from local_operator.network.credentials import consent, placement

    _seed(rig)
    peer = _Wire({"credential-lifecycle-v1"}, _echo_applied)
    server = _owner_server(rig, peer)
    assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
    disclosed = dict(peer.frames[0]["entries"][0])
    consent.restrict_and_revoke(
        N, rig.own.device_id, rig.peer.device_id, {"credential_key": P}, "late-deny", rig.root
    )
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.grant(P, rig.peer.device_id, scope="session", by=rig.own.device_id)
    monkeypatch.setattr(L, "_native_contexts", lambda *args: pytest.fail("denied native read"))
    for _ in range(2):
        replacement = _Wire({"credential-lifecycle-v1"}, _echo_applied)
        server._ensure_link = lambda device: replacement
        assert L.lifecycle_tick_step(server, rig.peer.device_id) == "delivered"
        assert replacement.frames[0]["entries"] == [{**disclosed, "state": "withdrawn"}]
        assert L.lifecycle_tick_step(server, rig.peer.device_id) == "in_sync"
        assert len(L._disclosure_history(server)) == 1
        assert len(L._sent_ledger(server)) == 1
    # Removed owner keys do not become a process-lifetime metadata archive.
    with placement.mutate(N, rig.root, self_device=rig.own.device_id) as doc:
        doc.entries[P].owner_device = rig.peer.device_id
    L._pending_entries(server, rig.peer.device_id)
    assert L._disclosure_history(server) == {}
    assert L._sent_ledger(server) == {}


def test_capacity_restriction_is_durable_bounded_and_not_cleared_by_other_login(rig, monkeypatch):
    monkeypatch.setattr(L, "MAX_OWNER_OBSERVATIONS", 1)
    _announce(rig)
    life = L.LifecycleStore(N, rig.root)
    before = life.path.read_bytes()
    blocked = {**_announce_frame(rig)["entries"][0], "account_scope_ref": "omitted"}
    result = _announce(rig, entries=[blocked])["entries"][0]
    assert result["reason"] == L.OBSERVED_TABLE_FULL and not result["applied"]
    marker = life.path.with_name(L.LIFECYCLE_OVERFLOW_FILENAME)
    assert marker.stat().st_mode & 0o777 == 0o600
    marker_bytes = marker.read_bytes()
    assert len(marker_bytes) < 128
    assert life.path.read_bytes() == before
    for chain, material in [(1, 1), (1, 2), (99, 99)]:
        effective._receipts.clear()
        reply = _announce(rig, chain=chain, material=material)["entries"][0]
        assert not reply["applied"] and reply["reason"] == L.NEEDS_AUTHORITATIVE_BASELINE
        assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    # New scope identities are not recorded in a second overflow table.
    for index in range(8):
        entry = {
            **blocked,
            "account_scope_ref": f"untracked-{index}",
            "state": "available",
            "login_chain_revision": 99,
            "material_generation": 99,
        }
        _announce(rig, entries=[entry])
    assert len(L.LifecycleStore(N, rig.root).load().high_water) == 1
    assert life.path.read_bytes() == before and marker.read_bytes() == marker_bytes
    assert life.status() == L.NEEDS_AUTHORITATIVE_BASELINE
    assert not L.LifecycleStore("n_222222222222222222222222", rig.root).restriction_overflowed()
    assert not L.LifecycleStore(N, rig.root / "another-root").restriction_overflowed()


def test_file_byte_capacity_latches_before_ephemeral_receipt(rig, monkeypatch):
    _announce(rig)
    life = L.LifecycleStore(N, rig.root)
    before = life.path.read_bytes()
    monkeypatch.setattr(L, "MAX_FILE_BYTES", len(before))
    original = effective.note_lifecycle_observation

    def checked(root, provider, state, **kwargs):
        assert life.restriction_overflowed()  # durable marker precedes receipt/ack
        return original(root, provider, state, **kwargs)

    monkeypatch.setattr(effective, "note_lifecycle_observation", checked)
    blocked = {**_announce_frame(rig)["entries"][0], "account_scope_ref": "file-cap"}
    reply = _announce(rig, entries=[blocked])["entries"][0]
    assert not reply["applied"] and reply["reason"] == L.OBSERVED_TABLE_FULL
    assert life.path.read_bytes() == before
    effective._receipts.clear()
    _announce(rig)
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"


@pytest.mark.parametrize("payload", ["", "not-json", '{"reason":false}'])
def test_any_existing_capacity_marker_is_fail_closed(rig, payload):
    _announce(rig)
    life = L.LifecycleStore(N, rig.root)
    marker = life.path.with_name(L.LIFECYCLE_OVERFLOW_FILENAME)
    marker.write_text(payload)
    marker.chmod(0o000)
    try:
        effective._receipts.clear()
        assert _announce(rig)["entries"][0]["reason"] == L.NEEDS_AUTHORITATIVE_BASELINE
        assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
    finally:
        marker.chmod(0o600)
    assert marker.read_text() == payload  # malformed is restrictive, not auto-repaired


def test_capacity_marker_symlink_is_never_overwritten(rig, tmp_path):
    _announce(rig)
    life = L.LifecycleStore(N, rig.root)
    target = tmp_path / "unrelated-marker-target"
    target.write_text("untouched")
    marker = life.path.with_name(L.LIFECYCLE_OVERFLOW_FILENAME)
    marker.symlink_to(target)
    effective._receipts.clear()
    assert _announce(rig)["entries"][0]["reason"] == L.NEEDS_AUTHORITATIVE_BASELINE
    assert marker.is_symlink() and target.read_text() == "untouched"
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"
