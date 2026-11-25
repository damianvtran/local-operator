"""The negotiated, idle-only remote stop (``net_session_stop_idle``).

WHY THIS FILE EXISTS. The ``sessions`` tool promises a stop that is idle-only and
never signals. Locally that is ``control.stop_session(graceful_only=True)``; across
devices the only stop on the wire used to be ``net_session_stop`` — the owner's full
kill-switch ladder, which may SIGTERM/SIGKILL — so the tool refused every remote stop.
This change adds a DISTINCT op, negotiated per link (``wire.SESSION_STOP_IDLE_V1``),
that runs only the owner's atomic idle latch, and each cell below pins one promise of
that contract at the layer that keeps it:

* the OWNER op runs ``graceful_only=True`` and never force, and refuses any frame
  that tries to escalate (``mode``/``force``) instead of quietly downgrading it;
* authority is the existing ``stop`` capability: a ``read`` member is refused at the
  authoriser's chokepoint before any handler runs;
* the REQUESTER's relay refuses with NOTHING SENT when the link lacks the feature or
  there is no link, and never falls back to ``net_session_stop`` or retries on a
  refusal, lost reply or exception;
* the CLI refuses with nothing sent when THIS device's relay predates the feature,
  and reports a sent-but-unanswered stop as unconfirmed rather than as nothing.

The human ladder (``--stop`` → ``peer_session_stop`` → ``net_session_stop``) is
deliberately untouched; its own tests (``test_session_plane.py``,
``test_sessions_pilot.py``) still pin it, and one cell here asserts the two ops stay
distinct names with the same capability row.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import relay, types, wire
from local_operator.network.types import MeshRefusal
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)
from tests.unit.network.test_session_plane import _dial_to, _listen, _own_locally

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]
PEER_ID = "d_" + "c" * 32
SESSION = "9f3ac1e0b7d2"


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture under an alias (``test_session_plane``'s
    reason: a parameter named like the imported fixture is an F811)."""
    pair: Devices = request.getfixturevalue("devices")
    return pair


# ---------------------------------------------------------------------------
# The op tables: a distinct op, the existing capability, no new grant
# ---------------------------------------------------------------------------


def test_the_idle_stop_is_its_own_op_on_the_existing_stop_capability() -> None:
    assert types.OP_CAPABILITY["net_session_stop_idle"] == "stop"
    # The legacy op keeps its row; the two are distinct names, never one op's mode.
    assert types.OP_CAPABILITY["net_session_stop"] == "stop"
    assert "peer_session_stop_idle" in types.LOCAL_OPS
    assert "peer_session_stop_idle" not in types.OP_CAPABILITY
    # No new capability or grant: ``drive`` already holds ``stop``, ``read`` does not.
    assert "stop" in types.ROLE_CAPABILITIES["drive"]
    assert "stop" not in types.ROLE_CAPABILITIES["read"]
    assert wire.SESSION_STOP_IDLE_V1 in wire.LINK_CAPABILITIES


# ---------------------------------------------------------------------------
# The OWNER op
# ---------------------------------------------------------------------------


def _owner_calls(
    server: relay.RelayServer, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """Record every ``control.stop_session`` call the owner op makes."""
    from local_operator.session.runtime import control as control_mod
    from tests.unit.session.runtime.test_control import _bare_record

    calls: list[dict[str, Any]] = []
    target = _bare_record(session_id=SESSION)

    async def _stop(record: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return control_mod.StopOutcome(
            pid=record.pid,
            session_id=record.session_id,
            name="stub",
            method="refused",
            reason=control_mod.REASON_BUSY,
            line="refused (English fallback)",
        )

    monkeypatch.setattr(server, "_session_record", lambda _sid: target)
    monkeypatch.setattr(control_mod, "stop_session", _stop)
    return calls


def test_the_owner_op_runs_only_the_idle_latch(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_a, server_b, _host, _port = peer_pair
    calls = _owner_calls(server_b, monkeypatch)
    answer = server_b._op_session_stop_idle(None, {"session_id": SESSION})  # type: ignore[arg-type]
    assert len(calls) == 1
    assert calls[0]["graceful_only"] is True and calls[0]["force"] is False
    assert calls[0]["_command"] == "mesh net_session_stop_idle"
    assert answer["intent"] == "stop_if_idle"
    assert answer["outcome"] == "refused" and answer["reason"] == "busy"


@pytest.mark.parametrize(
    "forged",
    [{"force": True}, {"mode": "immediate"}, {"mode": "anything-else"}],
)
def test_the_owner_op_refuses_a_forged_escalation_without_acting(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, forged: dict[str, Any]
) -> None:
    """A frame asking for more than idle-only is REFUSED, never downgraded."""
    _server_a, server_b, _host, _port = peer_pair
    calls = _owner_calls(server_b, monkeypatch)
    with pytest.raises(MeshRefusal) as refused:
        server_b._op_session_stop_idle(
            None, {"session_id": SESSION, **forged}  # type: ignore[arg-type]
        )
    assert refused.value.code == "stop_idle_force_refused"
    assert calls == []


def test_the_owner_op_answers_not_running_as_an_outcome(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    _server_a, server_b, _host, _port = peer_pair
    monkeypatch.setattr(server_b, "_session_record", lambda _sid: None)
    answer = server_b._op_session_stop_idle(None, {"session_id": SESSION})  # type: ignore[arg-type]
    assert answer["outcome"] == "not_running" and answer["intent"] == "stop_if_idle"


def test_over_a_real_link_a_silent_owner_is_refused_and_never_signalled(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end-to-end promise: busy, socket-silent target, real paired link.

    This is the exact shape on which the legacy ladder reaches its SIGTERM rung
    with ``immediate`` (``test_the_forwarded_stop_mode_is_the_owners_own_force``);
    the idle-only op must refuse it with no signal sent, and a forged
    ``mode=immediate`` on the new op must not change that.
    """
    from local_operator.session.runtime import control as control_mod
    from tests.unit.session.runtime.test_control import _bare_record

    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    link = _dial_to(server_a, record, host_b, port_b)
    assert wire.SESSION_STOP_IDLE_V1 in link.capabilities

    sent: list[tuple[int, int]] = []
    target = _bare_record(busy=True, control_port=1)
    _own_locally(server_b.root, target.session_id, server_b.identity.device_id)
    monkeypatch.setattr(server_b, "_session_record", lambda _sid: target)
    monkeypatch.setattr(control_mod, "_identity_by_record", lambda _r: (True, ""))
    monkeypatch.setattr(control_mod.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(control_mod.registry, "pid_alive", lambda _pid, **_: True)

    reply = link.request(
        {
            "op": "net_session_stop_idle",
            "req": 21,
            "locality": "remote",
            "session_id": target.session_id,
        }
    )
    assert reply is not None and reply["op"] == "ack", reply
    assert reply["detail"]["intent"] == "stop_if_idle"
    assert reply["detail"]["outcome"] == "refused", reply["detail"]
    assert sent == [], "the idle-only stop must never signal"

    forged = link.request(
        {
            "op": "net_session_stop_idle",
            "req": 22,
            "locality": "remote",
            "session_id": target.session_id,
            "mode": "immediate",
        }
    )
    assert forged is not None and forged["op"] == "error", forged
    assert forged["code"] == "stop_idle_force_refused"
    assert sent == []
    link.close("test")


def test_a_read_member_is_refused_at_the_chokepoint(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session.runtime import control as control_mod

    # The ROLE is the JOINER's grant on the inviter (``_pair``): B holds ``read`` on
    # A, so B dials A and asks A to stop a session A owns.
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="read")
    link = _dial_to(server_b, record, host_a, port_a)
    _own_locally(server_a.root, SESSION, server_a.identity.device_id)
    called: list[Any] = []

    async def _stop(*args: Any, **kwargs: Any) -> Any:
        called.append(kwargs)
        raise AssertionError("a read member must never reach the ladder")

    monkeypatch.setattr(control_mod, "stop_session", _stop)
    reply = link.request(
        {"op": "net_session_stop_idle", "req": 31, "locality": "remote", "session_id": SESSION}
    )
    assert reply is not None and reply["op"] == "error", reply
    # The authoriser's own sentence (observed: "<device> may not do that on this
    # device (it does not hold the 'stop' capability)"): refused before any handler.
    assert "'stop' capability" in str(reply.get("message")), reply
    assert called == []
    link.close("test")


# ---------------------------------------------------------------------------
# The REQUESTER's relay (ctl op)
# ---------------------------------------------------------------------------


class _Link:
    """A peer link stub: its negotiated capabilities and a scripted answer."""

    def __init__(self, capabilities: tuple[str, ...], answer: Any) -> None:
        self.capabilities = capabilities
        self.answer = answer
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], *, timeout: float | None = None) -> Any:
        self.frames.append(frame)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


def _ctl(server: relay.RelayServer, **fields: Any) -> dict[str, Any]:
    return server.control_dispatch(
        "peer_session_stop_idle", {"peer": PEER_ID, "session_id": SESSION, "req": 1, **fields}
    )


def _wire_link(
    server: relay.RelayServer, monkeypatch: pytest.MonkeyPatch, link: _Link | None
) -> None:
    monkeypatch.setattr(server, "_resolve_peer", lambda peer: peer)
    monkeypatch.setattr(server, "_ensure_link", lambda _peer: link)


def test_ctl_refuses_a_peer_without_the_feature_and_sends_nothing(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _b, _h, _p = peer_pair
    link = _Link(tuple(c for c in wire.LINK_CAPABILITIES if c != wire.SESSION_STOP_IDLE_V1), None)
    _wire_link(server_a, monkeypatch, link)

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("nothing may be sent to a peer without the feature")

    monkeypatch.setattr(server_a, "_local_peer_call", _forbidden)
    reply = _ctl(server_a)
    assert reply["op"] == "error" and reply["code"] == "stop_idle_unsupported", reply
    assert "nothing was sent" in reply["message"]
    assert link.frames == []


def test_ctl_with_no_link_is_unreachable_and_sends_nothing(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _b, _h, _p = peer_pair
    _wire_link(server_a, monkeypatch, None)
    monkeypatch.setattr(
        server_a,
        "_local_peer_call",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not call")),
    )
    reply = _ctl(server_a)
    assert reply["op"] == "error" and reply["code"] == "peer_unreachable", reply


@pytest.mark.parametrize("forged", [{"force": True}, {"mode": "immediate"}])
def test_ctl_refuses_a_forged_escalation_before_the_link(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, forged: dict[str, Any]
) -> None:
    server_a, _b, _h, _p = peer_pair
    link = _Link(wire.LINK_CAPABILITIES, {"op": "ack", "detail": {}})
    _wire_link(server_a, monkeypatch, link)
    reply = _ctl(server_a, **forged)
    assert reply["op"] == "error" and reply["code"] == "stop_idle_force_refused", reply
    assert link.frames == []


def test_ctl_sends_exactly_one_idle_only_frame_and_passes_the_receipt(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _b, _h, _p = peer_pair
    detail = {"intent": "stop_if_idle", "outcome": "stopped", "rung": "socket", "pid": 7}
    link = _Link(wire.LINK_CAPABILITIES, {"op": "ack", "detail": dict(detail)})
    _wire_link(server_a, monkeypatch, link)
    reply = _ctl(server_a)
    assert reply["op"] == "ack" and reply["detail"] == detail, reply
    assert [frame["op"] for frame in link.frames] == ["net_session_stop_idle"]
    assert "mode" not in link.frames[0] and "force" not in link.frames[0]


@pytest.mark.parametrize(
    ("answer", "code"),
    [
        # Sent, no reply: the owner MAY have acted — never "not asked", never retried.
        (None, "stop_idle_unconfirmed"),
        # An owner that does not know the op ran nothing: unsupported.
        ({"op": "error", "code": "unknown_op", "message": "x"}, "stop_idle_unsupported"),
        # Any other refusal is the owner's own, passed through.
        ({"op": "error", "code": "not_authorised", "message": "view-only"}, "not_authorised"),
    ],
)
def test_ctl_never_falls_back_or_retries(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, answer: Any, code: str
) -> None:
    server_a, _b, _h, _p = peer_pair
    link = _Link(wire.LINK_CAPABILITIES, answer)
    _wire_link(server_a, monkeypatch, link)
    reply = _ctl(server_a)
    assert reply["op"] == "error" and reply["code"] == code, reply
    assert [frame["op"] for frame in link.frames] == ["net_session_stop_idle"]


def test_ctl_does_not_fall_back_when_the_link_raises(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _b, _h, _p = peer_pair
    link = _Link(wire.LINK_CAPABILITIES, OSError("link dropped"))
    _wire_link(server_a, monkeypatch, link)
    with pytest.raises(OSError):
        server_a._ctl_peer_stop_idle({"peer": PEER_ID, "session_id": SESSION})
    assert [frame["op"] for frame in link.frames] == ["net_session_stop_idle"]


# ---------------------------------------------------------------------------
# The CLI: ``lop network sessions --stop-if-idle``
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    return parser


def _run(argv: list[str]) -> int:
    return int(net_cli.main(_parser().parse_args(["network", *argv])))


class _Relay:
    def __init__(self, capabilities: tuple[str, ...]) -> None:
        self.capabilities = capabilities


def _cli_seams(
    monkeypatch: pytest.MonkeyPatch, *, capabilities: tuple[str, ...], answer: Any
) -> list[dict[str, Any]]:
    """This device's relay record and its control answer; records every call."""
    import local_operator.network.store as store_mod

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(store_mod, "find_own_relay", lambda *a, **k: _Relay(capabilities))
    monkeypatch.setattr(net_cli, "_translate_act_target", lambda target, _peer: target)

    def _call(op: str, **fields: Any) -> Any:
        calls.append({"op": op, **fields})
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(net_cli, "_relay_call", _call)
    return calls


def _doc(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    return json.loads(capsys.readouterr().out)


def test_cli_an_old_local_relay_is_refused_with_nothing_sent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    old = tuple(c for c in wire.LINK_CAPABILITIES if c != wire.SESSION_STOP_IDLE_V1)
    calls = _cli_seams(monkeypatch, capabilities=old, answer={})
    assert _run(["sessions", "--json", "--peer", "b", "--stop-if-idle", SESSION]) == 1
    doc = _doc(capsys)
    assert doc["code"] == "relay_restart_required" and "lop network restart" in doc["message"]
    assert calls == []


def test_cli_an_old_relay_answering_unknown_op_reads_as_restart(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _cli_seams(
        monkeypatch,
        capabilities=wire.LINK_CAPABILITIES,
        answer=MeshRefusal("unknown_local_op", "this relay does not know the action"),
    )
    assert _run(["sessions", "--json", "--peer", "b", "--stop-if-idle", SESSION]) == 1
    assert _doc(capsys)["code"] == "relay_restart_required"
    assert [call["op"] for call in calls] == ["peer_session_stop_idle"]


def test_cli_a_lost_answer_is_unconfirmed_and_not_retried(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _cli_seams(monkeypatch, capabilities=wire.LINK_CAPABILITIES, answer=None)
    assert _run(["sessions", "--json", "--peer", "b", "--stop-if-idle", SESSION]) == 1
    assert _doc(capsys)["code"] == "stop_idle_unconfirmed"
    assert [call["op"] for call in calls] == ["peer_session_stop_idle"]
    assert calls[0]["report_unsent"] is True and calls[0]["allow_no_answer"] is True


def test_cli_never_sent_is_relay_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_seams(
        monkeypatch, capabilities=wire.LINK_CAPABILITIES, answer=types.ControlRequestUnsent()
    )
    assert _run(["sessions", "--json", "--peer", "b", "--stop-if-idle", SESSION]) == 1
    assert _doc(capsys)["code"] == net_cli.CODE_RELAY_UNAVAILABLE


def test_cli_a_receipt_from_another_op_is_not_reported_as_this_stop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _cli_seams(
        monkeypatch,
        capabilities=wire.LINK_CAPABILITIES,
        answer={"outcome": "stopped", "rung": "sigkill", "detail": "killed"},
    )
    assert _run(["sessions", "--json", "--peer", "b", "--stop-if-idle", SESSION]) == 1
    assert _doc(capsys)["code"] == "stop_idle_unconfirmed"


def test_cli_a_stopped_receipt_is_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _cli_seams(
        monkeypatch,
        capabilities=wire.LINK_CAPABILITIES,
        answer={"intent": "stop_if_idle", "outcome": "stopped", "detail": "stopped"},
    )
    assert _run(["sessions", "--json", "--peer", "b", "--stop-if-idle", SESSION]) == 0
    doc = _doc(capsys)
    assert doc["ok"] is True and doc["intent"] == "stop_if_idle"
    assert calls[0]["peer"] == "b" and calls[0]["session_id"] == SESSION
    assert "mode" not in calls[0] and "force" not in calls[0]


@pytest.mark.parametrize(
    "extra",
    [["--force"], ["--stop", "other"], ["--engage", "other"], ["--stop", "other", "--force"]],
)
def test_cli_stop_if_idle_is_exclusive_and_never_forced(
    monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    calls = _cli_seams(monkeypatch, capabilities=wire.LINK_CAPABILITIES, answer={})
    assert _run(["sessions", "--peer", "b", "--stop-if-idle", SESSION, *extra]) == 2
    assert calls == []


def test_cli_stop_if_idle_needs_a_peer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _cli_seams(monkeypatch, capabilities=wire.LINK_CAPABILITIES, answer={})
    assert _run(["sessions", "--json", "--stop-if-idle", SESSION]) == 1
    assert _doc(capsys)["code"] == "peer_required"
    assert calls == []
