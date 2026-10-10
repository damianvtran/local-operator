"""``lop network member grant/revoke`` (mesh build plan §0 finding 3, P0).

The default ``drive`` role cannot move, delete or borrow a login, so the move and
credential slices are unusable between two ordinary members without this verb.
What is pinned: the row write round-trips, it is audited, it takes effect on an
ALREADY-OPEN link (the authoriser re-reads the row per frame), a non-admin cannot
grant anything but the two onboarding scopes it decides for itself (F7: the
``SELF_DECIDED_SCOPES`` block at the end), ``admin`` cannot be granted, and both
the relay path and the relay-down path say the same words.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import cli as net_cli
from local_operator.network import identity, relay, store, types
from local_operator.session.placement import read_stamp
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _init_network,
    _pair,
    devices,
)

PEER = "d_" + "c" * 32


def _server(root: Path) -> relay.RelayServer:
    return relay.RelayServer(
        root=root,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.mint(root, name="device-a"),
        audit=audit_mod.AuditLog(root),
    )


def _with_peer(server: relay.RelayServer, *, role: str = "admin") -> types.NetworkRecord:
    record = _init_network(server, role=role)
    with store.mutate(record.network_id, server.root) as fresh:
        relay.admit(
            fresh,
            device_id=PEER,
            public_key="",
            name="laptop",
            role="drive",
            added_by=server.identity.device_id,
            root=server.root,
            persist=False,
        )
        store.save(fresh, server.root)
    return store.load(record.network_id, server.root)


def _caps(root: Path, network_id: str, device_id: str = PEER) -> list[str]:
    member = store.load(network_id, root).member(device_id)
    assert member is not None
    return sorted(member.capabilities)


def _audit_records(root: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in audit_mod.AuditLog(root).tail(200)
        if row.get("event") == "member_capabilities_changed"
    ]


def _args(
    verb: str, network: str, *caps: str, json_out: bool = False, device: str = PEER
) -> Namespace:
    return Namespace(
        network_command="member",
        member_command=verb,
        network=network,
        device=device,
        capabilities=list(caps),
        json=json_out,
    )


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_grant_and_revoke_round_trip_through_the_relay_and_are_audited(root: Path) -> None:
    server = _server(root)
    try:
        record = _with_peer(server)
        assert "move" not in _caps(root, record.network_id)

        granted = server.control_dispatch(
            "net_member_caps",
            {"req": 1, "network": record.name, "device_id": PEER, "grant": ["move", "delete"]},
        )
        assert granted["op"] == "ack", granted
        assert granted["detail"]["added"] == ["delete", "move"]
        assert {"move", "delete"} <= set(_caps(root, record.network_id))

        revoked = server.control_dispatch(
            "net_member_caps",
            {"req": 2, "network": record.name, "device_id": PEER, "revoke": ["delete"]},
        )
        assert revoked["detail"]["removed"] == ["delete"]
        caps = _caps(root, record.network_id)
        assert "move" in caps and "delete" not in caps

        again = server.control_dispatch(
            "net_member_caps",
            {"req": 3, "network": record.name, "device_id": PEER, "grant": ["move"]},
        )
        assert again["detail"]["changed"] is False

        records = _audit_records(root)
        assert len(records) == 2, "a no-op change is not an event"
        first, second = records
        assert first["subject"] == PEER and first["actor"] == server.identity.device_id
        assert first["detail"]["added"] == ["delete", "move"]
        assert second["detail"]["removed"] == ["delete"]
    finally:
        server.stop()


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"grant": ["admin"]}, "not_grantable"),
        ({"grant": ["teleport"]}, "unknown_capability"),
        ({"grant": ["move"], "revoke": ["move"]}, "conflicting_change"),
    ],
)
def test_a_grant_nobody_decided_is_refused(root: Path, change: dict[str, Any], code: str) -> None:
    server = _server(root)
    try:
        record = _with_peer(server)
        before = _caps(root, record.network_id)
        reply = server.control_dispatch(
            "net_member_caps", {"req": 1, "network": record.name, "device_id": PEER, **change}
        )
        assert reply["op"] == "error" and reply["code"] == code, reply
        assert _caps(root, record.network_id) == before
        assert _audit_records(root) == []
    finally:
        server.stop()


def test_a_non_admin_device_cannot_grant(root: Path) -> None:
    """``broker_credential`` stays admin-only: only an admin may hand it out."""
    server = _server(root)
    try:
        record = _with_peer(server, role="drive")
        reply = server.control_dispatch(
            "net_member_caps",
            {"req": 1, "network": record.name, "device_id": PEER, "grant": ["broker_credential"]},
        )
        assert reply["op"] == "error" and reply["code"] == "not_admin", reply
        assert "only an admin device" in reply["message"]
        assert "broker_credential" not in _caps(root, record.network_id)
        assert _audit_records(root) == []
    finally:
        server.stop()


def test_a_device_cannot_change_its_own_row_or_an_unknown_one(root: Path) -> None:
    server = _server(root)
    try:
        record = _with_peer(server)
        own = server.control_dispatch(
            "net_member_caps",
            {
                "req": 1,
                "network": record.name,
                "device_id": record.self_device_id,
                "grant": ["move"],
            },
        )
        assert own["code"] == "self_capabilities", own
        stranger = server.control_dispatch(
            "net_member_caps",
            {"req": 2, "network": record.name, "device_id": "d_" + "f" * 32, "grant": ["move"]},
        )
        assert stranger["code"] == "unknown_member", stranger
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# The CLI, both paths
# ---------------------------------------------------------------------------


def test_the_cli_grants_locally_when_no_relay_runs(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    record = _with_peer(server)
    server.stop()
    assert store.find_own_relay() is None

    assert net_cli.main(_args("grant", record.name, "move", "broker_credential")) == 0
    out = capsys.readouterr().out
    assert "laptop may now borrow this device's logins; move sessions to or from this device" in out
    assert "(in home-net, on this device only)" in out
    assert {"move", "broker_credential"} <= set(_caps(root, record.network_id))
    assert _audit_records(root)[-1]["detail"]["added"] == ["broker_credential", "move"]

    assert net_cli.main(_args("revoke", record.name, "move", json_out=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True and payload["removed"] == ["move"]
    assert payload["applied"] == "locally (relay not running)"
    assert "move" not in payload["capabilities"]


def test_the_cli_goes_through_a_running_relay(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    try:
        record = _with_peer(server)
        server.bind_control()
        server.start()
        assert net_cli.main(_args("grant", record.name, "delete", json_out=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["applied"] == "relay" and payload["added"] == ["delete"]
        assert "delete" in _caps(root, record.network_id)
        assert len(_audit_records(root)) == 1
    finally:
        server.stop()


def test_the_cli_refusal_carries_code_and_sentence(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    record = _with_peer(server, role="drive")
    server.stop()
    assert net_cli.main(_args("grant", record.name, "move", json_out=True)) == 1
    captured = capsys.readouterr()
    body = json.loads(captured.out)
    assert body == {"ok": False, "code": "not_admin", "message": body["message"]}
    assert "only an admin device" in captured.err


def test_the_member_verb_takes_the_name_every_other_surface_prints(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """AUDIT ROUND 2, F4: ``<device>`` is a NAME here too.

    ``lop network show``/``peers`` print names, and ``credential share --with
    <name>`` resolves one — but this verb handed whatever it was given straight to
    the row writer, so a valid member name came back as
    "laptop is not an active member of home-net": false, and it points the operator
    at the network instead of at the argument. Measured on the two-device rig.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    record = _with_peer(server)  # the peer's name is "laptop"
    server.stop()

    assert net_cli.main(_args("grant", record.name, "move", json_out=True, device="laptop")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["device_id"] == PEER and payload["added"] == ["move"]
    assert "move" in _caps(root, record.network_id)

    # An unambiguous id TAIL still resolves, and the exact id is unchanged.
    assert net_cli.main(_args("revoke", record.name, "move", json_out=True, device=PEER[-6:])) == 0
    assert "move" not in json.loads(capsys.readouterr().out)["capabilities"]

    # An unknown name refuses with the sentence that names the member list, rather
    # than denying a device that exists.
    assert (
        net_cli.main(_args("grant", record.name, "move", json_out=True, device="not-a-device")) == 1
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out)["code"] == "unknown_member"
    assert "is not a device in home-net" in captured.err


def test_member_rm_takes_the_same_name_spelling(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REVIEW ROUND 1, R3: ``member rm`` was the third verb with a raw-id argument.

    ``member grant|revoke`` were fixed in round 1 of the audit; ``rm`` kept handing
    ``args.device`` to the row writer, so the SAME valid name answered
    "laptop is not a member of home-net" (``relay.remove_member``'s refusal) — a false
    statement about a member that is one. One resolver, and the refusal names the
    member list rather than denying the device exists.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    try:
        record = _with_peer(server)  # the peer's name is "laptop"
        server.bind_control()
        server.start()
        args = Namespace(
            network_command="member",
            member_command="rm",
            network=record.name,
            device="laptop",
            capabilities=[],
            json=True,
        )
        assert net_cli.main(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["removed"] == PEER, payload
        # The tombstone is the observable side effect: the member is gone from the
        # active set on this device, and the removal is recorded.
        removed = store.load(record.network_id, root).member(PEER)
        assert removed is not None and not removed.active, removed

        args = Namespace(
            network_command="member",
            member_command="rm",
            network=record.name,
            device="not-a-device",
            capabilities=[],
            json=True,
        )
        assert net_cli.main(args) == 1
        captured = capsys.readouterr()
        assert json.loads(captured.out)["code"] == "unknown_member"
        assert "is not a device in home-net" in captured.err
    finally:
        server.stop()


def test_the_parser_takes_grant_and_revoke_and_the_bare_group_names_all_verbs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="lop")
    net_cli.add_parser(parser.add_subparsers(dest="subcommand"))
    args = parser.parse_args(["network", "member", "grant", "home", "d_x", "move", "delete"])
    assert (args.member_command, args.capabilities) == ("grant", ["move", "delete"])
    assert net_cli.main(Namespace(network_command="member", member_command=None)) == 2
    assert "grant|revoke" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# It takes effect on an open link
# ---------------------------------------------------------------------------


def test_a_grant_opens_the_move_op_on_an_already_open_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding 3 end to end: a ``drive`` peer is refused ``net_session_move`` by the
    chokepoint; after ``member grant … move`` on the owner the SAME link passes the
    capability check and the op reaches MOBILITY, whose own refusal for a frame with
    no conversation id is what comes back (the capability, not the op, was what
    blocked it before)."""
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch, role="drive")
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    try:
        frame = {"op": "net_session_move", "phase": "status", "locality": "remote"}
        before = link.request({**frame, "req": 71})
        assert before is not None and before["op"] == "error"
        # THE CAPABILITY IS WHAT REFUSED IT, ASSERTED BY NAME (review round 1, T1).
        # This cell used to accept any error whose sentence was not the by-name "not
        # implemented" refusal — and once mobility started refusing a frame with no
        # conversation id, that let the cell pass with the capability check DISABLED
        # (mutation-measured: ``OP_CAPABILITY["net_session_move"] = "list"`` kept it
        # green). It read as coverage for the grant without testing the grant.
        assert "does not hold the 'move' capability" in before["message"], before
        assert "not implemented" not in before["message"], "refused by capability first"

        granted = server_a.control_dispatch(
            "net_member_caps",
            {
                "req": 1,
                "network": record.network_id,
                "device_id": server_b.identity.device_id,
                "grant": ["move"],
            },
        )
        assert granted["op"] == "ack", granted

        after = link.request({**frame, "req": 72})
        assert after is not None and after["op"] == "error"
        assert "a move needs a conversation id" in after["message"]
        assert "not implemented" not in after["message"]
    finally:
        link.close("test")


def test_a_peers_rotation_does_not_undo_a_local_grant() -> None:
    """Found while building this: ``apply_epoch`` replaced the member table with
    the ROTATOR's copy, whose row for the peer still carried the admission-time
    set, so any `member rm` anywhere in the network silently revoked a grant made
    here. Rows this device holds now keep their LOCAL authority (the
    ``adopt_members`` rule 2 already stated); a row this device does not hold is
    still adopted as sent."""
    me, rotator, newcomer = "d_" + "a" * 32, "d_" + "b" * 32, "d_" + "e" * 32
    record = types.NetworkRecord(network_id="n_rot", name="rot", self_device_id=me, epoch=1)
    for device, role in ((me, "admin"), (PEER, "drive"), (rotator, "admin")):
        relay.admit(
            record,
            device_id=device,
            public_key="k" + device[-4:],
            name=device[:6],
            role=role,
            added_by=me,
            persist=False,
        )
    relay.set_member_capabilities(record, device_id=PEER, grant=["move"])

    theirs = types.NetworkRecord.from_json(record.to_json())
    peer_row = theirs.member(PEER)
    assert peer_row is not None
    peer_row.capabilities = sorted(types.ROLE_CAPABILITIES["drive"])
    relay.admit(
        theirs, device_id=newcomer, public_key="knew", role="read", added_by=rotator, persist=False
    )
    frame = {
        "epoch": 2,
        "rotation_id": rotator,
        "members": [row.to_json() for row in theirs.members],
        "secret": "new-secret",
    }
    state = types.SecretState(network_id="n_rot", epoch=1, secret="old-secret")
    outcome = relay.apply_epoch(record, state, frame, sender_device_id=rotator, persist=False)
    assert outcome.applied, outcome
    kept = record.member(PEER)
    assert kept is not None and "move" in kept.capabilities
    added = record.member(newcomer)
    assert added is not None and sorted(added.capabilities) == sorted(
        types.ROLE_CAPABILITIES["read"]
    )


# ---------------------------------------------------------------------------
# F7 — the deciding device records its own decision (the onboarding scopes)
# ---------------------------------------------------------------------------


def test_the_self_decided_set_is_exactly_the_onboarding_scopes_and_update() -> None:
    # The carve-out is by NAME, so it cannot silently grow into a role capability:
    # the two onboarding scopes AND `update` (S1 of the mesh rolling updates, whose
    # decision governs this device's own install — RU §2) are grantable-only, and
    # the set is exactly the three.
    assert types.SELF_DECIDED_SCOPES == frozenset({"approve", "unattended", "update"})
    assert types.SELF_DECIDED_SCOPES <= types.GRANTABLE_CAPABILITIES


@pytest.mark.parametrize("role", ["drive", "read"])
def test_the_deciding_device_may_grant_and_revoke_the_onboarding_scopes(
    root: Path, role: str
) -> None:
    """F7: ``approve``/``unattended`` are written by the device that decides.

    The writer is a NON-admin member (``drive``, and ``read`` for symmetry) — the
    caller the blanket gate refused — through the same primitive both writer paths
    share. Pre-cleared first, so the cell can only pass on the WRITE, and audited
    like every other change.
    """
    server = _server(root)
    try:
        record = _with_peer(server, role=role)
        before = _caps(root, record.network_id)
        assert "approve" not in before and "unattended" not in before

        granted = server.control_dispatch(
            "net_member_caps",
            {
                "req": 1,
                "network": record.name,
                "device_id": PEER,
                "grant": ["approve", "unattended"],
            },
        )
        assert granted["op"] == "ack", granted
        assert granted["detail"]["added"] == ["approve", "unattended"]
        assert {"approve", "unattended"} <= set(_caps(root, record.network_id))

        revoked = server.control_dispatch(
            "net_member_caps",
            {"req": 2, "network": record.name, "device_id": PEER, "revoke": ["unattended"]},
        )
        assert revoked["op"] == "ack", revoked
        assert revoked["detail"]["removed"] == ["unattended"]
        caps = _caps(root, record.network_id)
        assert "approve" in caps and "unattended" not in caps

        records = _audit_records(root)
        assert len(records) == 2, records
        assert records[0]["subject"] == PEER
        assert records[0]["actor"] == server.identity.device_id
        assert records[0]["detail"]["added"] == ["approve", "unattended"]
        assert records[1]["detail"]["removed"] == ["unattended"]
    finally:
        server.stop()


def test_the_cli_records_the_onboarding_scopes_with_the_relay_down(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The relay-down writer path says the same words (F7)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    record = _with_peer(server, role="drive")
    server.stop()
    assert store.find_own_relay() is None

    assert net_cli.main(_args("grant", record.name, "approve", "unattended")) == 0
    out = capsys.readouterr().out
    assert "laptop may now answer approval prompts for sessions here" in out
    assert "start sessions here without approval prompts" in out
    assert "(in home-net, on this device only)" in out
    assert {"approve", "unattended"} <= set(_caps(root, record.network_id))
    assert _audit_records(root)[-1]["detail"]["added"] == ["approve", "unattended"]


def test_the_cli_records_the_onboarding_scopes_through_a_running_relay(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The relay writer path covers the same write (F7)."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    server = _server(root)
    try:
        record = _with_peer(server, role="drive")
        server.bind_control()
        server.start()
        assert net_cli.main(_args("grant", record.name, "unattended", json_out=True)) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["applied"] == "relay" and payload["added"] == ["unattended"]
        assert "unattended" in _caps(root, record.network_id)
        assert len(_audit_records(root)) == 1
    finally:
        server.stop()


@pytest.mark.parametrize(
    "change",
    [
        {"grant": ["broker_credential"]},
        {"grant": ["move"]},
        {"grant": ["delete"]},
        {"grant": ["list"]},
        {"grant": ["admin"]},
        {"revoke": ["move"]},
        {"grant": ["approve", "move"]},
        {"revoke": ["unattended", "move"]},  # the union arm is shared; mixed revoke too (NIT-2)
    ],
)
def test_every_other_capability_still_needs_an_admin(root: Path, change: dict[str, Any]) -> None:
    """F7's carve-out is exactly the onboarding scopes; nothing else widens.

    ``broker_credential`` (never in a non-admin ROLE, and only an admin may hand
    it out), the role caps, ``admin`` itself, and a MIXED set containing just one
    non-carved capability all keep the admin gate — and a refused change leaves
    the row and the audit untouched.
    """
    server = _server(root)
    try:
        record = _with_peer(server, role="drive")
        before = _caps(root, record.network_id)
        reply = server.control_dispatch(
            "net_member_caps", {"req": 1, "network": record.name, "device_id": PEER, **change}
        )
        assert reply["op"] == "error" and reply["code"] == "not_admin", reply
        assert _caps(root, record.network_id) == before
        assert _audit_records(root) == []
    finally:
        server.stop()


def test_a_drive_device_still_cannot_change_its_own_row(root: Path) -> None:
    """The carve-out is about a PEER's row in the deciding device's own record.

    A device changing ITSELF keeps the ``self_capabilities`` refusal, including
    for the two onboarding scopes — so the carve-out cannot be read as "a device
    may help itself".
    """
    server = _server(root)
    try:
        record = _with_peer(server, role="drive")
        reply = server.control_dispatch(
            "net_member_caps",
            {
                "req": 1,
                "network": record.name,
                "device_id": record.self_device_id,
                "grant": ["approve", "unattended"],
            },
        )
        assert reply["op"] == "error" and reply["code"] == "self_capabilities", reply
    finally:
        server.stop()


def test_the_onboarding_grant_opens_a_full_auto_create_on_an_already_open_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F7 acceptance — the WRITE takes effect on a LIVE consumer, not just permitted.

    Topology is the drill's: the RECEIVING device (server_a) is the one that
    decides, and it joined as ``drive`` — so the blanket admin gate would refuse
    the write, and this cell fails on it. Pre-cleared, its row for the sender
    gains ``unattended``, and the SAME already-open link flips: a full-auto
    (``yolo``) create the gate refused is now accepted, and the accepted authority
    rides the stamp.
    """
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch, role="drive")
    # ``_pair`` seats the inviter as admin; the drill's node holds ``drive``, so
    # re-seat this device's OWN row to the role the fix is about — with an admin
    # row the cell would pass without the gate change and prove nothing.
    with store.mutate(record.network_id, server_a.root) as fresh:
        me = fresh.self_member()
        assert me is not None and "admin" in me.capabilities
        me.role = "drive"
        me.capabilities = sorted(types.capabilities_for_role("drive"))
        fresh.self_role = "drive"
        fresh.self_capabilities = sorted(types.capabilities_for_role("drive"))
        store.save(fresh, server_a.root)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    try:
        frame = {"op": "net_session_create", "req": 71, "cwd": "", "yolo": True}
        before = link.request(dict(frame))
        assert before is not None and before["op"] == "error", before
        message = str(before["message"])
        assert "grants the requesting member 'unattended'" in message, before
        # F7 design round 1 (D1): the remedy NAMES the deciding device. This
        # sentence is relayed to the REQUESTING side, where "this device" would
        # mean the READER's machine, not the node that refused.
        assert "approve setup for device-a in the Mesh tab" in message, before
        assert "this device" not in message, before
        sender = server_b.identity.device_id
        assert "unattended" not in _caps(server_a.root, record.network_id, sender)

        granted = server_a.control_dispatch(
            "net_member_caps",
            {
                "req": 1,
                "network": record.network_id,
                "device_id": sender,
                "grant": ["unattended"],
            },
        )
        assert granted["op"] == "ack", granted
        assert granted["detail"]["added"] == ["unattended"]
        assert "unattended" in _caps(server_a.root, record.network_id, sender)

        def _no_warm(self: Any, *args: Any, **kwargs: Any) -> None:
            # The accept path's warm-up side effect is not the assertion here;
            # the ack and the stamp are (review round 1, NIT-1).
            return None

        monkeypatch.setattr(relay.RelayServer, "_warm_after_create", _no_warm)
        after = link.request({**frame, "req": 72})
        assert after is not None and after["op"] == "ack", after
        session_id = str(after["detail"]["session_id"])
        assert session_id
        stamp = read_stamp(server_a.root, session_id)
        assert stamp is not None and stamp.unattended is True, stamp
    finally:
        link.close("test")
