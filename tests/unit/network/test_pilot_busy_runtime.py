"""A remote pilot act against a runtime that is ALIVE but too busy to answer.

The incident (2026-11-10): ``lop network sessions --peer <mac> --send …`` failed
with "the dial … produced no session (owner could not prepare this view; the
session is alive — reconnect to retry) … `lop network doctor` tells a link that
is down from a runtime that stopped". The owner's relay audit showed every
stream OPENED and none refused: the link was up, the runtime was alive and
loaded, and it missed the viewer's first-sync window. Two defects made that
read as a dead device:

1. **Classification.** The runtime's three "alive but did not answer" outcomes
   (``RuntimeUnresponsiveError``, a ``bind_failed`` announcement, and
   ``OwnerAckTimeout``) are ``ConnectionError``s, so the pilot's dial-refusal
   rung claimed them and pointed at ``network doctor``.
2. **The window.** ``bind_runtime()`` took the TUI's 15 s foreground envelope,
   so the verb's own 120 s bind budget (``PILOT_BIND_TIMEOUT_S``) never reached
   the sync wait: a runtime that would have answered at 16 s lost the message.

These cells drive the REAL ``AttachedSession`` against an in-process
``RuntimeServer`` over its real socket, through the real CLI entry point. Only
the peer's catalogue row and the open seam are stubbed (the relay hop is not
what is under test). Durations are scaled down by patching the module
constants, keeping the incident's ORDERING: foreground envelope < sync delay <
pilot bind budget.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile.attach_client import OwnerAckTimeout
from local_operator.network import cli as net_cli
from local_operator.session import attached as attached_mod
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.network.test_sessions_pilot import (
    PEER,
    SESSION,
    _FakeViewer,
    _patch,
    _remote_row,
    _run,
)
from tests.unit.session.runtime.test_server import FakeHandle

#: The scaled incident. A watched TUI bind keeps 1 s here (15 s in production);
#: the pilot's bind budget is 6 s (120 s); the busy runtime syncs at 3 s (16 s).
FOREGROUND_S = 1.0
PILOT_BIND_S = 6.0
PILOT_SLACK_S = 1.0
BUSY_SYNC_AT_S = 3.0


class _SilentHandle(FakeHandle):
    """Accepts the connection and never produces the canonical sync."""

    def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001, ANN202
        async def never() -> Any:
            await asyncio.Event().wait()

        return never()


class _BusyHandle(FakeHandle):
    """Produces the sync, but only after ``BUSY_SYNC_AT_S`` (a loaded loop)."""

    def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001, ANN202
        base = FakeHandle.subscribe_frontend

        async def later() -> Any:
            await asyncio.sleep(BUSY_SYNC_AT_S)
            return base(self, on_update, display_window=display_window)

        return later()


class _RefusingHandle(FakeHandle):
    """Cannot prepare this connection, so the server announces ``bind_failed``."""

    def subscribe_frontend(self, on_update, *, display_window=False):  # noqa: ANN001, ANN202
        raise RuntimeError("the frontend bind could not complete (test)")


async def _no_engage(*_args: Any, **_kwargs: Any) -> None:
    """A record already exists; engaging must not spawn a second runtime."""
    return None


async def _never() -> Any:
    raise AssertionError("a pilot viewer never takes over")


def _await_record(root: Path) -> None:
    for _ in range(250):
        rows = registry.scan(root)
        if rows and rows[0][1] == "live":
            return
        time.sleep(0.02)
    raise AssertionError("the runtime record did not publish")


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Start ``handle`` as this test's runtime and route the pilot's open to it."""
    servers: list[RuntimeServer] = []

    def start(handle: FakeHandle) -> None:
        import local_operator.network.store as store_mod
        import local_operator.session.remote_open as remote_open

        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
        session_dir = tmp_path / "sessions" / "s1"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
        server = RuntimeServer(handle, kind="tui")
        server.start()
        servers.append(server)
        _await_record(tmp_path)

        monkeypatch.setattr(attached_mod, "FRONTEND_SYNC_FOREGROUND_S", FOREGROUND_S)
        monkeypatch.setattr(attached_mod, "_FOREGROUND_BIND_BUDGET_S", FOREGROUND_S)
        monkeypatch.setattr(net_cli, "PILOT_BIND_TIMEOUT_S", PILOT_BIND_S)
        monkeypatch.setattr(net_cli, "_PILOT_BIND_SLACK_S", PILOT_SLACK_S, raising=False)
        monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", _no_engage)
        # The relay is up here: a dial-refusal would read ``peer_unreachable``.
        monkeypatch.setattr(store_mod, "find_own_relay", lambda root=None: object())

        _patch(monkeypatch, None, _remote_row())

        async def open_real(_session_id: str, **_kwargs: Any) -> AttachedSession:
            return await AttachedSession.cold(
                "s1", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
            )

        monkeypatch.setattr(remote_open, "open_remote_viewer", open_real)

    yield start
    for server in servers:
        server.close()


def _payload(printed: str) -> dict[str, Any]:
    return json.loads(printed.strip().splitlines()[-1])


def test_a_runtime_that_never_syncs_is_busy_not_a_dial_that_produced_no_session(
    runtime: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fix 1 and fix 2 on one real socket: the session is named busy, after the
    verb's own window, and nothing points at ``network doctor``."""
    runtime(_SilentHandle())
    started = time.monotonic()
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    waited = time.monotonic() - started
    payload = _payload(capsys.readouterr().out)
    assert payload["code"] == "session_unreachable", payload
    sentence = payload["message"]
    assert "produced no session" not in sentence, sentence
    assert "doctor" not in sentence, sentence
    assert "too busy" in sentence and "was not delivered" in sentence, sentence
    assert "retry" in sentence, sentence
    # The window is the verb's own (6 s less the slack), not the 1 s envelope.
    assert waited >= PILOT_BIND_S - PILOT_SLACK_S - 0.5, waited


def test_a_runtime_that_syncs_after_the_tui_envelope_is_served_inside_the_pilots(
    runtime: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fix 2: a busy runtime that answers inside the pilot's budget is served."""
    runtime(_BusyHandle())
    rc = _run(["sessions", "--json", "--peer", PEER, "--peek", SESSION])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "session_unreachable" not in printed and "peer_unreachable" not in printed, printed


def test_a_bind_failed_announcement_is_busy_not_a_dial_that_produced_no_session(
    runtime: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """The incident's exact sentence came from this shape: the runtime SPOKE."""
    runtime(_RefusingHandle())
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = _payload(capsys.readouterr().out)
    assert payload["code"] == "session_unreachable", payload
    assert "produced no session" not in payload["message"], payload
    assert "doctor" not in payload["message"], payload
    assert "too busy" in payload["message"], payload


class _AckTimeoutViewer(_FakeViewer):
    async def bind_runtime(self, **_kwargs: Any) -> None:
        raise OwnerAckTimeout("owner did not acknowledge the request in time")


def test_an_owner_ack_timeout_is_busy_not_the_bind_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``OwnerAckTimeout`` is also a ``TimeoutError``: it must not read as the
    120 s budget running out, which never happened."""
    _patch(monkeypatch, _AckTimeoutViewer(), _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--steer", SESSION, "hi"]) == 1
    payload = _payload(capsys.readouterr().out)
    assert payload["code"] == "session_unreachable", payload
    assert "within 120s" not in payload["message"], payload
    assert "doctor" not in payload["message"], payload
    assert "the steer was not delivered" in payload["message"], payload


@pytest.mark.parametrize("welcome", [True, False], ids=["late-welcome", "plain-refusal"])
def test_main_s_owner_alive_mark_is_the_pilot_s_busy_rule(
    welcome: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Merge fold with main: the late welcome (``OwnerWelcomeTimeout``) carries only
    the ``owner_alive`` mark and is raised on the pilot's own bind, so a class list
    called that busy owner a device that "produced no session". The control: a
    plain ``ConnectionError`` (no mark) still takes the dial-refusal rung."""
    from local_operator.mobile.attach_client import OwnerWelcomeTimeout

    error: ConnectionError = (
        OwnerWelcomeTimeout("owner did not welcome the connection in time")
        if welcome
        else ConnectionError("owner socket unreachable")
    )

    class _Viewer(_FakeViewer):
        async def bind_runtime(self, **_kwargs: Any) -> None:
            raise error

    _patch(monkeypatch, _Viewer(), _remote_row())
    assert _run(["sessions", "--json", "--peer", PEER, "--send", SESSION, "hi"]) == 1
    payload = _payload(capsys.readouterr().out)
    if welcome:
        assert payload["code"] == "session_unreachable", payload
        assert "produced no session" not in payload["message"], payload
        assert "too busy" in payload["message"], payload
    else:
        # The dial-refusal rung, which asks this device's relay what happened.
        assert "too busy" not in payload["message"], payload


class _DeadlineViewer(_FakeViewer):
    def __init__(self) -> None:
        super().__init__()
        self.deadlines: list[float | None] = []

    async def bind_runtime(self, *, deadline: float | None = None) -> None:
        self.deadlines.append(deadline)
        self.bound = True


def test_the_pilot_hands_its_own_bind_budget_to_the_viewer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The deadline is this verb's budget less the slack, measured from the bind."""
    viewer = _DeadlineViewer()
    _patch(monkeypatch, viewer, _remote_row())
    before = time.monotonic()
    _run(["sessions", "--json", "--peer", PEER, "--steer", SESSION, "hi"])
    capsys.readouterr()
    (deadline,) = viewer.deadlines
    assert deadline is not None
    expected = net_cli.PILOT_BIND_TIMEOUT_S - net_cli._PILOT_BIND_SLACK_S
    assert before + expected - 1.0 <= deadline <= time.monotonic() + expected
