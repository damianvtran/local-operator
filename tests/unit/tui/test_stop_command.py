"""``/stop`` in the TUI: the in-process branch, the follower branch, the
target branch, and ``/stop all``'s arm-and-repeat — driven through the real
``OperatorApp`` with pilot.

The escalation ladder itself is covered in
``tests/unit/session/runtime/test_control.py``; here the shared module is
stubbed at its seams (``stop_session`` / ``stop_all`` / ``_stop_targets``)
so no socket is dialled and no process is signalled from a TUI test. What is
pinned is what the USER sees: the session ends and the transcript survives,
the receipt names the way back, the arm listing names the targets and the
window, and a repeat inside the window executes while one outside re-arms.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.session.runtime import control
from local_operator.session.runtime.types import SessionRecord
from local_operator.tui import app as app_mod
from local_operator.tui.app import STOP_ALL_WINDOW_S, OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


async def _booted(app: OperatorApp, pilot: Any, session: Any) -> None:
    for _ in range(40):
        await pilot.pause()
        if app._session is session:
            return
    raise AssertionError("session never adopted")


def _record(pid: int, name: str) -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=f"sid-{pid}",
        conversation_name=name,
        cwd="/tmp",
        model_label="test/model",
        control_port=1,
        control_key="k",
    )


@pytest.mark.asyncio
async def test_bare_stop_ends_the_session_and_keeps_the_transcript() -> None:
    """Bare ``/stop`` on a TUI-owned session: disposed, shown cold, the
    transcript still on screen, and the receipt names ``/resume``."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._append_block(UserBlock("hello there"))
        await pilot.pause()
        app._run_slash_command("/stop")
        for _ in range(20):
            await pilot.pause()
            if app._session is None:
                break
        assert session.disposed
        assert app._session is None
        # The reading record survives the session it describes.
        assert any(isinstance(b, UserBlock) for b in app.query_one(TranscriptView).blocks())
        receipt = [n for n in _notices(app) if n.startswith("stopped")]
        assert receipt, _notices(app)
        assert "/resume sess reopens it" in receipt[0]
        # The next prompt reports the ended session rather than starting one.
        app._submit_prompt("again?")
        await pilot.pause()
        assert session.prompts == []


@pytest.mark.asyncio
async def test_stop_with_no_session_is_a_warning() -> None:
    class NeverBoots(FakeSession):
        pass

    app = OperatorApp(lambda: _factory(NeverBoots()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._session = None
        app._run_slash_command("/stop")
        await pilot.pause()
        assert "no session to stop yet" in _notices(app)


@pytest.mark.asyncio
async def test_follower_stop_sends_the_op_to_the_owner() -> None:
    """On a RemoteSession bare ``/stop`` forwards the graceful op and paints
    the owner's ack — never disposes locally, never signals."""
    from local_operator.session.frontend_state import FrontendSessionState

    class Remoteish(FakeSession):
        is_remote = True
        frontend_state: FrontendSessionState

        def __init__(self) -> None:
            super().__init__()
            self.stop_requests = 0

        async def request_stop(self) -> str:
            self.stop_requests += 1
            return "stopping"

    session = Remoteish()
    session.frontend_state = FrontendSessionState(session_id="sess", epoch="owner")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop")
        for _ in range(20):
            await pilot.pause()
            if session.stop_requests:
                break
        assert session.stop_requests == 1
        assert not session.disposed
        assert app._session is session
        assert "stopping" in _notices(app)


@pytest.mark.asyncio
async def test_stop_target_uses_the_send_vocabulary(monkeypatch: pytest.MonkeyPatch) -> None:
    """``/stop <target>`` resolves through ``resolve_peer_target`` and paints
    the shared module's receipt line verbatim."""
    target = _record(77777, "other agent")
    resolved: list[dict[str, Any]] = []

    def fake_resolve(**kwargs: Any):
        resolved.append(kwargs)
        return target, [], ""

    async def fake_stop(record, *, timeout_s=10.0, _root=None):  # noqa: ANN001, ANN202
        return control.StopOutcome(
            record.pid, record.session_id, "other agent", "socket", 'stopped "other agent"'
        )

    monkeypatch.setattr("local_operator.mobile.peer_send.resolve_peer_target", fake_resolve)
    monkeypatch.setattr(control, "stop_session", fake_stop)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop other")
        for _ in range(20):
            await pilot.pause()
            if 'stopped "other agent"' in _notices(app):
                break
        assert resolved and resolved[0]["target"] == "other"
        assert 'stopped "other agent"' in _notices(app)
        # This session is untouched: a target stop never ends the caller.
        assert app._session is session


@pytest.mark.asyncio
async def test_stop_target_ambiguity_lists_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resolve(**kwargs: Any):
        return None, [_record(1, "alpha"), _record(2, "alphabet")], ""

    monkeypatch.setattr("local_operator.mobile.peer_send.resolve_peer_target", fake_resolve)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop alph")
        for _ in range(20):
            await pilot.pause()
            if any("2 sessions match" in n for n in _notices(app)):
                break
        match = [n for n in _notices(app) if "2 sessions match" in n]
        assert match and "pid 1 (alpha)" in match[0] and "pid 2 (alphabet)" in match[0]


@pytest.mark.asyncio
async def test_stop_all_arms_then_a_repeat_inside_the_window_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First ``/stop all`` prints the listing and stops NOTHING; a second
    inside the window runs the shared ``stop_all`` and reports grouped."""
    targets = [_record(101, "alpha"), _record(102, "beta")]
    calls: list[str] = []

    def fake_targets(root, own_pid=None):  # noqa: ANN001, ANN202
        return targets

    async def fake_all(*, own_pid, _root, timeout_s=10.0):  # noqa: ANN001, ANN202
        calls.append("all")
        return [
            control.StopOutcome(101, "sid-101", "alpha", "socket", 'stopped "alpha"'),
            control.StopOutcome(102, "sid-102", "beta", "sigterm", 'stopped "beta"'),
        ]

    monkeypatch.setattr(control, "_stop_targets", fake_targets)
    monkeypatch.setattr(control, "stop_all", fake_all)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        for _ in range(20):
            await pilot.pause()
            if any("will stop" in n for n in _notices(app)):
                break
        listing = [n for n in _notices(app) if "will stop" in n]
        assert listing, _notices(app)
        assert "pid 101  alpha" in listing[0]
        assert "pid 102  beta" in listing[0]
        assert f"repeat /stop all within {STOP_ALL_WINDOW_S:g}s" in listing[0]
        assert calls == []  # the arming press stops nothing
        assert app._session is session

        app._run_slash_command("/stop all")
        for _ in range(30):
            await pilot.pause()
            if app._session is None:
                break
        assert calls == ["all"]
        # Own session last, ended in-process, and the grouped report says so.
        assert app._session is None
        assert session.disposed
        notices = _notices(app)
        # Own session's receipt (the in-process branch) precedes the summary.
        own = [n for n in notices if n.startswith("stopped") and "/resume sess" in n]
        assert own, notices
        assert "1 stopped via socket, 1 stopped via sigterm" in notices[-1]
        # The listing settled from a standing promise into a note.
        listing = [n for n in notices if "will stop" in n]
        assert listing and listing[0].endswith("stopping them all…")


@pytest.mark.asyncio
async def test_stop_all_repeat_outside_the_window_re_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_all(*, own_pid, _root, timeout_s=10.0):  # noqa: ANN001, ANN202
        calls.append("all")
        return []

    monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: [_record(5, "x")])
    monkeypatch.setattr(control, "stop_all", fake_all)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        await pilot.pause()
        # Age the arm past the window without waiting on the clock.
        assert app._stop_all_armed_at is not None
        app._stop_all_armed_at -= STOP_ALL_WINDOW_S + 1
        app._run_slash_command("/stop all")
        for _ in range(10):
            await pilot.pause()
        assert calls == []
        assert sum(1 for n in _notices(app) if "will stop" in n) == 2
        assert app._session is session


@pytest.mark.asyncio
async def test_stop_all_lapse_retires_the_promise(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the window closes untaken, the listing's instruction line is
    replaced by a lapsed note — the facts stay, the promise goes."""
    monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: [_record(5, "x")])
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        for _ in range(10):
            await pilot.pause()
        armed_at = app._stop_all_armed_at
        assert armed_at is not None
        app._expire_stop_all_window(armed_at)
        await pilot.pause()
        assert app._stop_all_armed_at is None
        listing = [n for n in _notices(app) if "will stop" in n]
        assert listing and listing[0].endswith("window lapsed — /stop all again to re-arm")
        assert "repeat /stop all" not in listing[0]


@pytest.mark.asyncio
async def test_stop_all_refusals_get_their_own_line(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grouped count cannot say WHICH agent was refused; that one gets
    its own warning line so the user can act on it."""
    monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: [_record(9, "z")])

    async def fake_all(*, own_pid, _root, timeout_s=10.0):  # noqa: ANN001, ANN202
        return [
            control.StopOutcome(9, "sid-9", "z", "refused", "refused to signal pid 9 — identity"),
        ]

    monkeypatch.setattr(control, "stop_all", fake_all)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        await pilot.pause()
        app._run_slash_command("/stop all")
        for _ in range(30):
            await pilot.pause()
            if app._session is None:
                break
        notices = _notices(app)
        assert "refused to signal pid 9 — identity" in notices
        assert notices[-1] == "1 refused"


@pytest.mark.asyncio
async def test_stop_all_with_only_own_session_lists_it_alone() -> None:
    """No other agents: the listing is honest about the one thing a repeat
    would stop — this session — rather than claiming nothing is running."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(control, "_stop_targets", lambda root, own_pid=None: [])
            app._run_slash_command("/stop all")
            for _ in range(10):
                await pilot.pause()
        listing = [n for n in _notices(app) if "will stop 1 session:" in n]
        assert listing and "(this one, last)" in listing[0]
        assert app._stop_all_armed_at is not None


@pytest.mark.asyncio
async def test_stop_all_with_nothing_at_all_disarms() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._session = None
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(control, "_stop_targets", lambda root, own_pid=None: [])
            app._run_slash_command("/stop all")
            for _ in range(10):
                await pilot.pause()
        assert "no sessions to stop" in _notices(app)
        assert app._stop_all_armed_at is None


def test_stop_is_registered_and_frontend_local() -> None:
    from local_operator.session.frontend_state import CommandScope, _slash_capabilities
    from local_operator.tui.app import slash_command_for

    entry = slash_command_for("/stop all")
    assert entry is not None and entry.name == "stop"
    caps = {c.command: c for c in _slash_capabilities()}
    assert caps["stop"].scope is CommandScope.FRONTEND_LOCAL


def test_stop_all_window_is_longer_than_the_esc_ladder() -> None:
    """The confirmation is a LISTING the user has to read; the window has to
    outlast the reading, which is the argument for it being longer than
    the Esc ladder's single-line offer."""
    assert STOP_ALL_WINDOW_S > app_mod.DOUBLE_STOP_WINDOW_S
