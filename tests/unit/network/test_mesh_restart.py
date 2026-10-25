"""Peer-targeted relay recovery over the real handshake and relay dispatch."""

from __future__ import annotations

import argparse
import os
import threading
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import authorizer as authorizer_mod
from local_operator.network import cli as net_cli
from local_operator.network import meshrestart, relay, store, types, wire
from local_operator.session.runtime import registry
from local_operator.session.runtime.types import SessionRecord
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[Any, Any, str, int]


def _set_peer_endpoints(server: Any, record: Any, device_id: str, endpoints: list[str]) -> None:
    with store.mutate(record.network_id, server.root) as copy:
        row = copy.member(device_id)
        assert row is not None
        row.endpoints = list(endpoints)
        store.save(copy, server.root)


def _rig(devices: Devices, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:  # noqa: F811
    server_a, server_b, host, port = devices
    record, _host, _port = _pair(devices, monkeypatch)
    _set_peer_endpoints(server_b, record, server_a.identity.device_id, [f"{host}:{port}"])
    return server_a, server_b, record


def _grant(server_a: Any, record: Any, requester: str) -> None:
    reply = server_a.control_dispatch(
        "net_member_caps",
        {"req": 1, "network": record.network_id, "device_id": requester, "grant": ["update"]},
    )
    assert reply["op"] == "ack", reply
    assert reply["detail"]["added"] == ["update"], reply


def _restart_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _restart(action: str) -> dict[str, Any]:
        calls.append(action)
        return {"ok": True}

    monkeypatch.setattr(relay, "service_action", _restart)
    return calls


def _publish_busy(server: Any) -> SessionRecord:
    record = SessionRecord(
        pid=os.getpid(),
        kind="daemon",
        session_id="bu5y-restart-01",
        conversation_name="offloaded training turn",
        cwd=str(server.root),
        model_label="test/mock",
        control_port=1,
        control_key="k" * 32,
        busy=True,
        started=True,
    )
    registry.publish(record, server.root)
    return record


def test_restart_is_registered_without_widening_roles() -> None:
    assert types.OP_CAPABILITY["net_restart"] == "update"
    assert "net_restart" in types.NET_OPS
    assert "peer_restart" in types.LOCAL_OPS
    assert wire.MESH_RESTART_V1 in wire.LINK_CAPABILITIES
    assert "update" not in types.ROLE_CAPABILITIES["read"]
    assert "update" not in types.ROLE_CAPABILITIES["drive"]
    assert "restart" in types.CAPABILITY_WORDS["update"]
    assert authorizer_mod.op_tables_are_total() == []


def test_granted_peer_restarts_and_receipt_has_fresh_process_evidence(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """The supervisor is the only replaced seam: it stops and starts a REAL relay."""
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    current = {"server": server_a}
    calls: list[str] = []
    port = devices[3]

    def _restart(action: str) -> dict[str, Any]:
        assert action == "restart"
        calls.append(action)
        old = current["server"]
        old.stop()
        # Same identity, store and port; a new RelayServer is a new process
        # instance for protocol purposes, with a fresh instance id and start time.
        new = relay.RelayServer(
            root=old.root,
            identity=old.identity,
            settings=relay.NetworkSettings(port=port, listen_address="127.0.0.1"),
            audit=audit_mod.AuditLog(old.root),
        )
        new.bind()
        new.bind_control()
        new.start()
        current["server"] = new
        return {"ok": True}

    monkeypatch.setattr(relay, "service_action", _restart)
    try:
        detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)
    finally:
        current["server"].stop()

    assert detail["ok"] is True, detail
    assert detail["state"] == detail["code"] == "restarted", detail
    assert calls == ["restart"]
    before = detail["before"]
    after = detail["after"]
    assert before["instance_id"] != after["instance_id"]
    # The test process keeps one pid, so the discriminating process fact is the
    # answering relay's reset start time; the new handshake instance is independent.
    assert before["pid"] == after["pid"] == os.getpid()
    assert after["started_at"] > before["started_at"]
    assert current["server"] is not server_a
    assert "restarted its relay" in detail["message"]
    assert "uptime reset" in detail["message"]
    assert before["instance_id"] in detail["message"]
    assert after["instance_id"] in detail["message"]


def test_busy_peer_names_running_turn_and_is_not_restarted(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    calls = _restart_calls(monkeypatch)
    busy = _publish_busy(server_a)

    detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)

    assert detail["ok"] is False, detail
    assert detail["state"] == detail["code"] == "busy", detail
    assert "offloaded training turn (bu5y-restart-01)" in detail["message"]
    assert calls == [], "a busy peer must not reach its supervisor"
    rows = registry.scan(server_a.root, reap=False, check_zombie=False)
    live = [row for row, _state in rows if row.session_id == busy.session_id]
    assert live and live[0].busy is True


def test_busy_receipt_through_the_real_local_control_socket(
    devices: Devices,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    calls = _restart_calls(monkeypatch)
    _publish_busy(server_a)
    # The pairing fixture needs only B's outbound dial; the CLI path additionally
    # needs B's real loopback control listener.
    server_b.start()
    capsys.readouterr()

    rc = net_cli.main(argparse.Namespace(network_command="restart", peer="device-a", json=False))

    assert rc == 1
    assert capsys.readouterr().out == (
        "device-a: 1 session(s) are busy: offloaded training turn (bu5y-restart-01); "
        "restart waits for idle so the running work is not interrupted\n"
        "re-run `lop network restart --peer device-a` when the work finishes\n"
    )
    assert calls == []


def test_a_turn_admitted_between_final_scan_and_supervisor_is_serialized(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Reviewer R1-B1's exact interleaving, with no mock at the registry/fence seam."""
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    monkeypatch.setattr(meshrestart, "RESTART_CONFIRM_WINDOW_S", 0.4)
    monkeypatch.setattr(meshrestart, "RESTART_POLL_S", 0.02)
    probes = 0
    calls: list[str] = []
    publisher_attempted = threading.Event()
    publisher_done = threading.Event()
    publisher_errors: list[BaseException] = []
    observations: dict[str, Any] = {}
    real_busy_sessions = meshrestart.meshupdate.busy_sessions
    real_publish = registry.publish

    def _observed_publish(record: Any, root: Any = None, dirname: Any = None) -> Any:
        publisher_attempted.set()
        try:
            if dirname is None:
                return real_publish(record, root)
            return real_publish(record, root, dirname)
        except BaseException as exc:
            publisher_errors.append(exc)
            raise
        finally:
            publisher_done.set()

    def _busy_then_admit(root: Any, *, include_names: bool = False) -> tuple[int, str]:
        nonlocal probes
        result = real_busy_sessions(root, include_names=include_names)
        probes += 1
        if probes == 2:
            # The final scan has returned idle. Admission starts NOW, before the
            # supervisor seam, which is the window R1-B1 proved was unguarded.
            publisher.start()
            assert publisher_attempted.wait(2.0), "the admission never reached its registry write"
        return result

    def _supervisor(action: str) -> dict[str, Any]:
        calls.append(action)
        # Fixed fence: the publisher is still excluded, so this wait expires and
        # the action linearizes first. Unfenced code lets publication finish here
        # and fails the busy-at-supervisor assertion below.
        observations["publisher_done_before_action"] = publisher_done.wait(0.5)
        observations["busy_at_action"] = real_busy_sessions(server_a.root)[0]
        return {"ok": True}

    monkeypatch.setattr(registry, "publish", _observed_publish)
    monkeypatch.setattr(meshrestart.meshupdate, "busy_sessions", _busy_then_admit)
    monkeypatch.setattr(relay, "service_action", _supervisor)
    publisher = threading.Thread(target=_publish_busy, args=(server_a,), daemon=True)

    detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)
    publisher.join(2.0)

    assert not publisher.is_alive(), "the admitted turn did not leave the fence"
    assert publisher_errors == []
    assert probes == 2
    assert calls == ["restart"]
    assert observations == {"publisher_done_before_action": False, "busy_at_action": 0}
    assert publisher_done.is_set(), "admission must proceed after the restart decision"
    assert real_busy_sessions(server_a.root)[0] == 1
    # The supervisor reported ok without replacing the instance, so the origin
    # remains honestly unconfirmed; this cell is about admission ordering.
    assert detail["ok"] is False and detail["code"] == "restart_unconfirmed", detail


def test_absent_update_grant_is_refused_by_name(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    server_a, server_b, _record = _rig(devices, monkeypatch)
    calls = _restart_calls(monkeypatch)

    detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)

    assert detail["ok"] is False, detail
    assert detail["state"] == "refused", detail
    assert "does not hold the 'update' capability" in detail["message"], detail
    assert calls == []


def test_older_peer_is_refused_before_an_unknown_op_can_be_sent(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    calls = _restart_calls(monkeypatch)
    link, reason = server_b.dial(
        record.network_id, host=f"{devices[2]}:{devices[3]}", epoch=record.epoch
    )
    assert link is not None, reason
    link.capabilities = frozenset(
        name for name in link.capabilities if name != wire.MESH_RESTART_V1
    )

    detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)

    assert detail["ok"] is False, detail
    assert detail["code"] == "predates_peer_restart", detail
    assert "does not support peer restart" in detail["message"]
    assert "run `lop network restart` on that device" in detail["message"]
    assert calls == []


def test_unknown_restart_op_is_a_clear_error_not_success(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    calls = _restart_calls(monkeypatch)
    # Simulate the over-the-wire answer of a build with no handler while keeping
    # the negotiated capability fixed, so this cell covers the error arm rather
    # than the older-peer preflight above. Dispatch and handshake remain real.
    monkeypatch.setattr(
        server_a,
        "_handlers",
        {name: handler for name, handler in server_a._handlers.items() if name != "net_restart"},
    )

    detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)

    assert detail["ok"] is False, detail
    assert detail["state"] == "refused", detail
    # The authorizer deliberately keeps its internal code off the wire; the
    # remote-facing sentence is the compatibility contract.
    assert detail["code"] == "refused", detail
    assert "net_restart is not implemented in this build" in detail["message"], detail
    assert calls == []


def test_acceptance_without_a_new_instance_is_unconfirmed_not_success(
    devices: Devices, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    server_a, server_b, record = _rig(devices, monkeypatch)
    _grant(server_a, record, server_b.identity.device_id)
    calls = _restart_calls(monkeypatch)  # a supervisor that claims ok but changes nothing
    monkeypatch.setattr(meshrestart, "RESTART_REPLY_GRACE_S", 0.05)
    monkeypatch.setattr(meshrestart, "RESTART_CONFIRM_WINDOW_S", 0.4)
    monkeypatch.setattr(meshrestart, "RESTART_POLL_S", 0.02)

    detail = meshrestart.restart_peer(server_b, server_a.identity.device_id)

    assert detail["ok"] is False, detail
    assert detail["state"] == "unconfirmed", detail
    assert detail["code"] == "restart_unconfirmed", detail
    assert "accepted the restart" in detail["message"]
    assert "restart state is unknown" in detail["message"]
    assert calls == ["restart"]


def test_restart_help_names_the_recovery_gap_and_bare_restart_stays_local(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    parsed = parser.parse_args(["network", "restart"])
    assert parsed.peer == ""
    peer_parsed = parser.parse_args(["network", "restart", "--peer", "device-a", "--json"])
    assert peer_parsed.peer == "device-a" and peer_parsed.json is True

    network_parser = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ).choices["network"]
    restart_parser = next(
        action
        for action in network_parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices["restart"]
    help_text = " ".join(restart_parser.format_help().split())
    assert "wedged remote node" in help_text
    assert "cannot be recovered from the origin" in help_text
    assert "`update` grant" in help_text
    assert "`busy`" in help_text and "no --force" in help_text
    assert "new pid or uptime evidence" in help_text

    calls: list[str] = []
    monkeypatch.setattr(
        relay,
        "service_action",
        lambda action: calls.append(action) or {"ok": True},
    )

    def _no_peer_call(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("bare restart crossed the peer boundary")

    monkeypatch.setattr(net_cli, "_relay_answer", _no_peer_call)
    assert net_cli.main(argparse.Namespace(network_command="restart", peer="", json=False)) == 0
    assert calls == ["restart"]
    assert capsys.readouterr().out == "relay restart ok\n"


def test_cli_receipts_distinguish_busy_refusal_and_no_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(network_command="restart", peer="device-a", json=False)
    outcomes = [
        (
            {
                "ok": False,
                "state": "busy",
                "code": "busy",
                "name": "device-a",
                "message": "1 session(s) are busy: training (abc); restart waits for idle",
            },
            1,
            "device-a: 1 session(s) are busy: training (abc); restart waits for idle\n"
            "re-run `lop network restart --peer device-a` when the work finishes\n",
        ),
        (
            {
                "ok": False,
                "state": "refused",
                "code": "refused",
                "name": "device-a",
                "message": "does not hold the 'update' capability",
            },
            1,
            "device-a: does not hold the 'update' capability\n",
        ),
        (
            {
                "ok": False,
                "state": "no_answer",
                "code": "no_answer",
                "name": "device-a",
                "message": (
                    "device-a did not answer the restart request; nothing is claimed "
                    "about whether its restart began"
                ),
            },
            1,
            "device-a did not answer the restart request; nothing is claimed about "
            "whether its restart began\n",
        ),
    ]
    for payload, expected_rc, expected_out in outcomes:
        monkeypatch.setattr(net_cli, "_relay_answer", lambda *a, _p=payload, **kw: dict(_p))
        assert net_cli.main(args) == expected_rc
        assert capsys.readouterr().out == expected_out
