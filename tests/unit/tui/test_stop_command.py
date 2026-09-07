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

import asyncio
import os
import re
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
        # The next prompt names the way back rather than "still starting".
        app._submit_prompt("again?")
        await pilot.pause()
        assert session.prompts == []
        assert _notices(app)[-1] == (
            "this session was stopped — your message was not sent; /resume sess reopens it"
        )


@pytest.mark.asyncio
async def test_stop_with_no_session_is_a_warning() -> None:
    """A session that has genuinely not arrived yet.

    The factory is held open rather than adopting and then dropping
    ``app._session``: those are two DIFFERENT states that used to share one
    answer, and conflating them is what let `/stop` tell a user with a live,
    spending session to wait for a boot that had already finished (QA Q-2).
    The adopted-then-unbound state is pinned separately below.
    """
    release = asyncio.Event()

    async def never_boots() -> Any:
        await release.wait()
        return FakeSession()

    app = OperatorApp(never_boots)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._session is None
        app._run_slash_command("/stop")
        await pilot.pause()
        assert "session is still starting…" in _notices(app)
        release.set()


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
        # Each candidate in the form that RESOLVES when retyped (U4).
        assert match and '/stop 1 ("alpha")' in match[0] and '/stop 2 ("alphabet")' in match[0]


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

    async def fake_all(*, own_pid, _root, only_pids=None, timeout_s=10.0):  # noqa: ANN001, ANN202
        calls.append("all")
        # The execution is restricted to what the listing showed (R1-6).
        assert only_pids == {101, 102}
        assert own_pid == os.getpid()
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
        # Grid: pids right-aligned to the widest pid (this process's), so
        # the names form one column (D1).
        assert re.search(r"pid +101  alpha\n", listing[0]), listing[0]
        assert re.search(r"pid +102  beta\n", listing[0]), listing[0]
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
        # The summary reconciles with the listing's promise: three listed,
        # three accounted for, own session folded in (D2).
        assert notices[-1] == "3 sessions: 2 stopped, 1 stopped via sigterm"
        # The listing settled from a standing promise into a finished note (D4).
        listing = [n for n in notices if "will stop" in n]
        assert listing and listing[0].endswith("stopped them all")


@pytest.mark.asyncio
async def test_stop_all_repeat_outside_the_window_re_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def fake_all(*, own_pid, _root, only_pids=None, timeout_s=10.0):  # noqa: ANN001, ANN202
        calls.append("all")
        # The execution is restricted to what the listing showed (R1-6).
        assert only_pids == {101, 102}
        assert own_pid == os.getpid()
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

    async def fake_all(*, own_pid, _root, only_pids=None, timeout_s=10.0):  # noqa: ANN001, ANN202
        return [
            control.StopOutcome(9, "sid-9", "z", "refused", 'refused "z" (pid 9) — identity'),
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
        assert 'refused "z" (pid 9) — identity' in notices
        assert notices[-1] == "2 sessions: 1 stopped, 1 refused"
        listing = [n for n in notices if "will stop" in n]
        assert listing and listing[0].endswith("stopped 1 of 2")


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


# -- real-ladder tests: the app's OWN registrant is published and the shared
# module runs unstubbed (``os.kill`` spied so a wrong ladder cannot take
# pytest with it). These are the tests the first draft lacked: with
# ``stop_all`` stubbed, the TUI SIGTERMing itself was invisible (R1-1).


@pytest.fixture
def kill_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record every real signal the ladder sends; signal 0 (the liveness
    probe) passes through untouched."""
    sent: list[tuple[int, int]] = []
    real_kill = control.os.kill

    def spy(pid: int, sig: int) -> None:
        if sig == 0:
            real_kill(pid, sig)
            return
        sent.append((pid, sig))

    monkeypatch.setattr(control.os, "kill", spy)
    return sent


async def _own_record(app: OperatorApp, pilot: Any) -> SessionRecord:
    """This app's published record (the registrant runs on its own thread)."""
    for _ in range(100):
        await pilot.pause(0.05)
        for rec, state in control.registry.scan():
            if rec.pid == os.getpid() and state == "live":
                return rec
    raise AssertionError("the app never published its record")


@pytest.mark.asyncio
async def test_stop_all_never_signals_its_own_process(kill_spy) -> None:
    """The real ``stop_all`` with the app's own record published: no signal
    reaches ``os.getpid()``, the session ends in-process, the report paints
    and the process is still here to paint it (R1-1 / U1 / Q1-1)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        own = await _own_record(app, pilot)
        assert own.kind == "tui"
        app._run_slash_command("/stop all")
        for _ in range(20):
            await pilot.pause()
            if any("will stop" in n for n in _notices(app)):
                break
        app._run_slash_command("/stop all")
        for _ in range(60):
            await pilot.pause(0.05)
            if app._session is None and any(n.startswith("1 session") for n in _notices(app)):
                break
        assert [s for s in kill_spy if s[0] == os.getpid()] == []
        assert kill_spy == []
        assert app._session is None and session.disposed
        notices = _notices(app)
        assert any("/resume sess reopens it" in n for n in notices), notices
        assert notices[-1] == "1 session: 1 stopped"
        # The record is unpublished: the process no longer advertises a
        # session it does not have (R1-3).
        assert [r for r, _s in control.registry.scan() if r.pid == os.getpid()] == []


@pytest.mark.asyncio
async def test_another_tui_is_ended_beneath_a_surviving_process(kill_spy) -> None:
    """A TUI targeted by the ladder from OUTSIDE gets the graceful op through
    ``TuiSessionHandle.request_stop``: its session ends, its record goes,
    its process (this one) is never signalled (R1-2 / Q1-2)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        own = await _own_record(app, pilot)
        # Drive the shared ladder as a peer would, concurrently with the app
        # loop — the app must keep running to service the op.
        task = app.run_worker(
            control.stop_session(own, timeout_s=5.0, _root=control.registry.run_dir()),
            thread=False,
        )
        for _ in range(120):
            await pilot.pause(0.05)
            if task.is_finished:
                break
        outcome = task.result
        assert outcome is not None and outcome.method == "socket", outcome
        assert kill_spy == []
        assert app._session is None and session.disposed
        assert any("/resume sess reopens it" in n for n in _notices(app))
        assert [r for r, _s in control.registry.scan() if r.pid == os.getpid()] == []


@pytest.mark.asyncio
async def test_bare_stop_unpublishes_the_record() -> None:
    """After bare ``/stop`` the process stops heartbeating the ended session
    as live (R1-3 / Q1-3)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        await _own_record(app, pilot)
        app._run_slash_command("/stop")
        for _ in range(40):
            await pilot.pause()
            if app._session is None:
                break
        assert app._mobile_registrant is None
        assert [r for r, _s in control.registry.scan() if r.pid == os.getpid()] == []


@pytest.mark.asyncio
async def test_stop_all_re_arms_when_the_listing_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session that appears inside the window changes the terms: the
    repeat re-lists and re-arms instead of executing (Q1-5 / R1-6)."""
    targets = [_record(101, "alpha")]
    calls: list[str] = []

    async def fake_all(*, own_pid, _root, only_pids=None, timeout_s=10.0):  # noqa: ANN001, ANN202
        calls.append("all")
        return []

    monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: list(targets))
    monkeypatch.setattr(control, "stop_all", fake_all)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        for _ in range(20):
            await pilot.pause()
            if any("will stop 2 sessions" in n for n in _notices(app)):
                break
        targets.append(_record(102, "beta"))  # appears inside the window
        app._run_slash_command("/stop all")
        for _ in range(20):
            await pilot.pause()
            if any("will stop 3 sessions" in n for n in _notices(app)):
                break
        assert calls == []
        assert app._session is session
        assert app._stop_all_armed_at is not None
        notices = _notices(app)
        assert any(n.endswith("the sessions changed — re-listing") for n in notices)
        assert any(re.search(r"pid +102  beta", n) for n in notices)
        # Unchanged now: the repeat executes.
        app._run_slash_command("/stop all")
        for _ in range(30):
            await pilot.pause()
            if calls:
                break
        assert calls == ["all"]


@pytest.mark.asyncio
async def test_mid_turn_stop_leaves_the_band_idle() -> None:
    """A ``/stop`` while the session streams must not leave the working
    spinner running: the receipt says the session ended (U3)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        session.streaming = True
        status = app._status
        assert status is not None
        status.update(streaming=True)
        app._on_frontend_update(None)  # a queued paint from the dying session
        app._run_slash_command("/stop")
        for _ in range(40):
            await pilot.pause()
            if app._session is None:
                break
        for _ in range(5):
            await pilot.pause()
        assert status._streaming is False


@pytest.mark.asyncio
async def test_bare_stop_disarms_an_armed_stop_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The armed listing promised "(this one, last)" for the session a bare
    ``/stop`` just ended; the window closes with it (U5)."""
    monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: [_record(5, "x")])
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        for _ in range(20):
            await pilot.pause()
            if app._stop_all_armed_at is not None and app._stop_all_listing is not None:
                break
        app._run_slash_command("/stop")
        for _ in range(40):
            await pilot.pause()
            if app._session is None:
                break
        assert app._stop_all_armed_at is None
        listing = [n for n in _notices(app) if "will stop" in n]
        assert listing and listing[0].endswith(
            "this session was stopped — /stop all again to re-arm"
        )
        # A second /stop names the way back, not a session "yet" (U8).
        app._run_slash_command("/stop")
        await pilot.pause()
        assert _notices(app)[-1] == "this session was stopped; /resume sess reopens it"


@pytest.mark.asyncio
async def test_tui_handle_stop_op_acks_and_ends_the_session(kill_spy) -> None:
    """The ``stop`` control op against a TUI owner acks and the session
    ends beneath the process (the follower's bare ``/stop`` path, Q1-2)."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        own = await _own_record(app, pilot)
        task = app.run_worker(
            control._exchange(own, {"op": "stop"}, reply_timeout_s=5.0), thread=False
        )
        for _ in range(120):
            await pilot.pause(0.05)
            if task.is_finished and app._session is None:
                break
        reply = task.result
        assert reply is not None and reply.get("op") == "ack", reply
        assert app._session is None and session.disposed
        assert kill_spy == []


@pytest.mark.asyncio
async def test_arm_listing_at_80_columns_never_wraps_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """The painted block's height equals its row count — no wrapped
    continuation can clip the instruction off the block (D2-1/U2-2).

    The acceptance test the round-2 design finding shipped: a name longer
    than the row budget at 80 columns truncates inside its own row, every
    painted row is one block row, and the repeat instruction is the block's
    last line.
    """
    long_name = "Investigating the flaky analytics recorder test on main"  # 55 cells
    targets = [_record(101, "Auditing merged MRs"), _record(102, long_name)]
    monkeypatch.setattr(control, "_stop_targets", lambda root, own_pid=None: targets)
    session = FakeSession()
    session.set_conversation_name(long_name)  # the own row is the worst case
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 30)) as pilot:
        await _booted(app, pilot, session)
        app._run_slash_command("/stop all")
        for _ in range(30):
            await pilot.pause()
            if app._stop_all_listing is not None:
                break
        block = app._stop_all_listing
        assert block is not None
        # No row exceeds the block's own body budget: every painted row is
        # one line, so the block's pinned height equals its text rows and the
        # last line (the instruction) is always painted.
        # Assert the PROPERTY, not a budget re-derived from the terminal
        # width: doing the arithmetic here repeats the very off-by-three the
        # test exists to catch, and it passed against the D2-1 bug for
        # exactly that reason (round-3 D3-3, demonstrated by mutation). What
        # matters to the user is that no row wraps, which is precisely
        # "the block is as tall as it has rows".
        from rich.text import Text

        authored = block._text.split("\n")
        built = block._build()
        # `_build` is typed as returning any renderable; this block builds a
        # Text, and the row count is what the assertions below are about.
        assert isinstance(built, Text)
        painted = built.plain.split("\n")
        assert len(painted) == len(authored), (len(painted), len(authored))
        assert block.size.height == len(authored), (block.size.height, len(authored))
        assert authored[-1].startswith("repeat /stop all")
        assert "…" in block._text  # the long name really was truncated
        assert "(this one, last)" in block._text.split("\n")[-2]


@pytest.mark.asyncio
async def test_owner_local_stop_announces_to_viewers_before_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare ``/stop`` in the OWNER's own window announces the deliberate stop.

    Round-3 BLOCKER-1: the control-op dispatch was the only emitter, so an
    owner that stopped itself tore its registrant down without a word and a
    watching follower read a plain EOF as owner death — taking over the
    session the user had just ended (U2-4 on a different route). The
    announcement must be written BEFORE ``_mobile_teardown`` closes those
    sockets, which is why it is written inline rather than scheduled.
    """
    from local_operator.mobile.attach_client import AttachClient
    from local_operator.session.runtime.server import RuntimeServer
    from tests.unit.session.runtime.test_server import FakeHandle, _wait_record

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        # `.start()`, not `start_in_process()`: those are the two branches of
        # `announce_stop`, split on `_on_owner_loop()`, and the TUI hosts its
        # registrant in a THREAD. The in-process branch is a path production
        # never takes (round-4 MINOR-4).
        server = RuntimeServer(FakeHandle(), kind="tui")
        server.start()
        app._mobile_registrant = server
        # A REAL attached viewer, so the frame travels the real socket the
        # teardown is about to close — the ordering under test.
        record = await _wait_record()
        client = AttachClient(
            on_projection=lambda _p: None,
            on_disconnected=lambda _r: None,
        )
        await client.connect(record, record.session_id)
        for _ in range(6):
            await pilot.pause()
        # Spy the WRITE, not the viewer's disconnect: the frame reaching the
        # transport BEFORE the teardown is the property under test and it is
        # deterministic, whereas the EOF that follows crosses a real socket
        # from a real thread and would make this a timing race. Delivery on
        # this path is covered by the real-socket evidence on the PR.
        original_write = RuntimeServer._write_now
        wrote: list[dict[str, object]] = []

        def _spy(self: RuntimeServer, frame: dict[str, object]) -> None:
            wrote.append(frame)
            original_write(self, frame)

        monkeypatch.setattr(RuntimeServer, "_write_now", _spy)
        app._run_slash_command("/stop")
        for _ in range(200):
            await pilot.pause()
            await asyncio.sleep(0.02)
            if wrote:
                break
        # Exactly one announcement, naming the stop, on the production path.
        assert [f["op"] for f in wrote] == ["stopping"]
        client.close()
        # The registrant is torn down by the same path; announcing first is
        # the whole point, so the frame must already be out by now.
        assert app._mobile_registrant is None


@pytest.mark.asyncio
@pytest.mark.parametrize("in_process", [False, True])
async def test_announce_stop_is_safe_without_viewers_or_registrant(in_process: bool) -> None:
    """Announcing never breaks a stop: no registrant and no viewers both pass.

    The stop is the user's instruction; a best-effort courtesy frame that
    could raise would turn a working kill switch into a failing one.

    Parameterised over BOTH hosting modes, which are the two branches of
    ``announce_stop``: the TUI hosts its registrant in a thread, a runtime
    process hosts in-process, and both are production paths. Covering one
    only moves the gap rather than closing it (round-5 MINOR-6).
    """
    from local_operator.session.runtime.server import RuntimeServer
    from tests.unit.session.runtime.test_server import FakeHandle

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._mobile_registrant = None
        app._announce_stop_to_viewers()  # no registrant at all
        server = RuntimeServer(FakeHandle(), kind="tui")
        if in_process:
            await server.start_in_process()
        else:
            server.start()
        app._mobile_registrant = server
        app._announce_stop_to_viewers()  # registrant, zero viewers
        server.close()
        app._announce_stop_to_viewers()  # already closed


@pytest.mark.asyncio
async def test_a_second_stop_after_a_resume_still_notifies() -> None:
    """The notice latch belongs to the session, not to the app.

    Round-5 MAJOR-5: `_watched_stop_notice_shown` was set once and cleared
    only in `__init__`, so a viewer that survived one stop, `/resume`d onto a
    second session and had THAT stopped too saw nothing at all — the
    transcript stopped mid-air, which is the round-3 D3-1 defect returning one
    `/resume` later. The latch exists to keep ONE stop from painting twice, so
    a session transition is exactly where it must be cleared.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)

        app._stopped_session_id = "sess-a"
        app._on_watched_session_stopped()
        await pilot.pause()
        first = [b._text for b in app.query(NoticeBlock) if "was stopped" in b._text]
        assert first, "the first stop painted no notice"

        # The transition BOTH /resume branches go through. The operation
        # itself is irrelevant here — what is under test is the state reset
        # the transition performs before running it.
        async def _swap() -> Any:
            return session

        app._run_session_transition(_swap())
        for _ in range(20):
            await pilot.pause()

        app._on_watched_session_stopped()
        await pilot.pause()
        painted = [b._text for b in app.query(NoticeBlock) if "was stopped" in b._text]
        # The point is that the SECOND stop speaks at all; the id it names is
        # read from the live session, as it is on every other route.
        assert len(painted) == len(first) + 1, painted


@pytest.mark.asyncio
async def test_a_failed_stop_does_not_claim_the_next_one() -> None:
    """A stop that stopped nothing must not be remembered as this app's own.

    Round-5 MINOR-5: `_issued_own_stop` was set before the op was issued and
    never cleared when the owner refused, so the NEXT stop — a stranger's —
    dropped the "from another terminal" clause and implied the user had ended
    their own session. Same latching class as round-3 MAJOR-1.
    """

    class RefusingSession(FakeSession):
        async def request_stop(self) -> str:
            raise ConnectionError("owner refused")

    session = RefusingSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        await app._stop_remote_worker(session)
        assert app._issued_own_stop is False, "a refused stop stayed latched"

        # A later EXTERNAL stop is still attributed to whoever really did it.
        app._stopped_session_id = "sess-x"
        app._on_watched_session_stopped()
        await pilot.pause()
        painted = [b._text for b in app.query(NoticeBlock) if "was stopped" in b._text]
        assert "from another terminal" in painted[-1], painted


@pytest.mark.asyncio
async def test_a_stopped_session_settles_its_ask_picker() -> None:
    """A stop settles the ask picker as well as the approval card.

    Round-6 MAJOR-6: the round-5 fix added `_deny_queued_approvals()` without
    its sibling `_settle_ask_picker()`, so a parked picker survived the stop
    holding key routing. `enter` then recorded a choice nobody made, and — the
    worse half — the message typed afterwards was swallowed entirely, with no
    user block and no refusal, reopening round-4's MAJOR-3 hole for this
    state. Every other turn-ending path in the app settles these together.
    """
    from local_operator.harness.types import AskOption, AskQuestion

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _booted(app, pilot, session)
        question = AskQuestion(
            id="req-ask-1",
            question="ship it?",
            options=[AskOption(label="yes"), AskOption(label="no")],
        )
        pending = asyncio.ensure_future(app.request_user_choice([question]))
        for _ in range(60):
            await pilot.pause()
            if app._ask_screen is not None:
                break
        assert app._ask_screen is not None, "the picker never mounted"

        app._stopped_session_id = "sess"
        app._on_watched_session_stopped()
        for _ in range(6):
            await pilot.pause()
        # No key-routing surface survives the session it belongs to.
        assert app._live_prompt() is None, "a parked prompt outlived the stop"
        await asyncio.wait_for(pending, 2)

        # And the message typed after the stop is answered, not swallowed.
        await pilot.press("enter")
        for _ in range(6):
            await pilot.pause()
        notices = [b._text for b in app.query(NoticeBlock) if "was stopped" in b._text]
        assert notices, "the stop notice never painted"


@pytest.mark.asyncio
async def test_a_mid_turn_stop_does_not_add_a_bare_interrupted_row() -> None:
    """The per-card mark explains the stop; a standalone row restates it.

    Round-6 MINOR-7: D5-1's one-word fix (keeping the retired-card COUNT that
    `_on_agent_end` reads) had no test, so reverting it left the whole suite
    green — which is exactly how the defect was introduced in the first place.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        # A live tool card, mounted the way the app tracks one.
        from local_operator.tui.widgets.tool_card import ToolCard

        card = ToolCard("call-1", "bash", {"command": "sleep 30"})
        app._tool_cards["call-1"] = card
        app._append_block(card)
        for _ in range(4):
            await pilot.pause()

        app._stopped_session_id = "sess"
        app._on_watched_session_stopped()
        for _ in range(4):
            await pilot.pause()
        # The retired card is what the aborted-turn branch reads to decide it
        # has already been explained; dropping the count re-adds the bare row.
        assert app._interrupted_cards >= 1, "the retired card was not counted"

        from local_operator.tui.events import TurnEnded

        app.post_message(TurnEnded(aborted=True, error=None))
        for _ in range(8):
            await pilot.pause()
        bare = [b for b in app.query(NoticeBlock) if b._text == "interrupted"]
        assert not bare, "a bare interrupted row was added beside the per-card mark"


@pytest.mark.asyncio
async def test_stop_reaches_a_live_session_the_viewer_lost_its_binding_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE KILL SWITCH MUST WORK IN THE STATE IT IS NEEDED IN.

    The refusal used to test the VIEWER's binding rather than whether a session
    exists to stop. With the session adopted, alive and undisposed but the
    viewer's reference dropped — a swap, a reload, a reconnect race — `/stop`
    answered "session is still starting…" for a session that was visibly
    burning money, and told the user to WAIT (QA Q-2).
    """
    target = _record(77777, "the runaway")
    stopped: list[str] = []
    selectors: list[dict[str, Any]] = []

    def fake_resolve(**kwargs: Any):
        selectors.append(kwargs)
        return target, [], ""

    async def fake_stop(record, *, timeout_s=10.0, _root=None):  # noqa: ANN001, ANN202
        stopped.append(record.session_id)
        return control.StopOutcome(
            record.pid, record.session_id, "the runaway", "socket", 'stopped "the runaway"'
        )

    monkeypatch.setattr("local_operator.mobile.peer_send.resolve_peer_target", fake_resolve)
    monkeypatch.setattr(control, "stop_session", fake_stop)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        # The incident's shape: the session is ALIVE and spending; only the
        # viewer's reference is gone.
        app._session = None
        assert session.disposed is False
        app._run_slash_command("/stop")
        for _ in range(20):
            await pilot.pause()
            if stopped:
                break
        assert stopped, "an unbound viewer must still be able to stop what it is watching"
        assert "session is still starting…" not in _notices(app)
        # The kill switch addresses the watched session by the EXACT selector,
        # never the substring vocabulary — see the wrong-session test below.
        assert selectors and selectors[0]["session"] == "sess"
        assert not selectors[0]["target"]


@pytest.mark.asyncio
async def test_a_lost_binding_never_tells_the_user_to_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the stop genuinely cannot be routed, the copy must still be honest.

    "Still starting" is false for a session that has been adopted, and its
    advice — wait — is the most expensive thing a user can do when the session
    they cannot reach is spending money. The residual answer names the state
    and points at the lever that works from outside this viewer.
    """

    def fake_resolve(**kwargs: Any):
        return None, [], "no live session matches"

    monkeypatch.setattr("local_operator.mobile.peer_send.resolve_peer_target", fake_resolve)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._session = None
        app._run_slash_command("/stop")
        for _ in range(20):
            await pilot.pause()
            if any("lost its link" in text for text in _notices(app)):
                break
        notices = _notices(app)
        assert any("lost its link" in text for text in notices), notices
        assert any("lop stop" in text for text in notices), notices
        assert "session is still starting…" not in notices


@pytest.mark.asyncio
async def test_a_booting_session_still_says_it_is_starting() -> None:
    """The guard on the copy change: "still starting" is TRUE during boot.

    The fix must narrow that message to the state it describes, not delete it —
    a user who typed `/stop` a beat after launch should still be told the
    session is on its way.
    """
    release = asyncio.Event()

    async def delayed_factory():
        await release.wait()
        return FakeSession()

    app = OperatorApp(delayed_factory)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._session is None
        assert app._adopted_session_id == ""
        app._run_slash_command("/stop")
        await pilot.pause()
        assert "session is still starting…" in _notices(app)
        release.set()


@pytest.mark.asyncio
async def test_an_unbound_stop_does_not_recurse_into_itself() -> None:
    """A regression guard found while building the fallback.

    Routing the unbound case through `_stop_target_worker` looked right, but
    that worker's self-pid branch calls back into `_cmd_stop` — which, with no
    bound session, takes the unbound fallback again and dispatches another
    worker, forever. It painted nothing at all, a quieter version of the defect
    being fixed.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        own_pid = os.getpid()
        calls: list[str] = []
        original = app._stop_target_worker

        async def counting_worker(target: str) -> None:
            calls.append(target)
            await original(target)

        app._stop_target_worker = counting_worker  # type: ignore[assignment]
        # Resolve to THIS process, the branch that used to recurse.
        own = _record(own_pid, "me")
        app._session = None
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "local_operator.mobile.peer_send.resolve_peer_target",
                lambda **kwargs: (own, [], ""),
            )
            app._run_slash_command("/stop")
            for _ in range(20):
                await pilot.pause()
        # The self-pid case is answered here, not bounced back into the
        # command: at most one delegation, and no unbounded growth.
        assert len(calls) == 0, calls
        assert any("lost its link" in text for text in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_the_unbound_stop_never_reaches_a_look_alike_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BLOCKER-1: the kill switch must not stop a session it did not name.

    The unbound fallback used the `target=` SUBSTRING vocabulary, which matches
    against conversation_name, session_id AND the cwd basename of every live
    record. The state this path exists for is precisely the one where the
    watched session is not resolvable, so the needle fell through to whatever
    else contained it — resolving to a different live session with no error and
    no ambiguity candidates, which the ladder then stopped.

    This is the operator's machine with several live sessions on it: stopping
    the wrong one destroys someone's in-flight work while the runaway keeps
    billing. Found independently by both the reviewer and QA.
    """
    watched = "00264921d0d9"
    # The ONLY live record is an unrelated session whose NAME contains the
    # watched id. The watched session itself is gone from the registry.
    decoy = _record(4242, f"notes about {watched}")
    stopped: list[str] = []

    async def fake_stop(record, *, timeout_s=10.0, _root=None):  # noqa: ANN001, ANN202
        stopped.append(record.session_id)
        return control.StopOutcome(
            record.pid, record.session_id, record.conversation_name, "socket", "stopped"
        )

    monkeypatch.setattr(
        "local_operator.session.runtime.registry.scan", lambda *a, **k: [(decoy, "live")]
    )
    monkeypatch.setattr(control, "stop_session", fake_stop)
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._adopted_session_id = watched
        app._session = None
        await app._stop_unbound_worker(watched)
        await pilot.pause()

        assert stopped == [], f"the kill switch stopped a session it never named: {stopped}"
        # And it says so honestly rather than silently doing nothing.
        assert any("lost its link" in text for text in _notices(app)), _notices(app)


@pytest.mark.asyncio
async def test_an_ordinary_reload_does_not_take_the_unbound_branch() -> None:
    """BLOCKER-2: a transition is not a lost binding.

    `_adopted_session_id` was set on adopt but cleared in one place, while four
    paths null `_session`. So for the whole window between "old session
    disposed" and "replacement adopted" — an ordinary `/reload`, `/new`,
    `/resume` or attach — the viewer was sessionless with a stale id, and a
    bare `/stop` both painted "this viewer lost its link…" in the error weight
    AND dispatched the stop worker against the session being replaced. That is
    what made BLOCKER-1 reachable day to day rather than only in the rare
    lost-binding state.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    dispatched: list[str] = []

    async def spy(session_id: str) -> None:
        dispatched.append(session_id)

    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        app._stop_unbound_worker = spy  # type: ignore[assignment]
        # Exactly what `_reload_session` does to the binding mid-transition.
        app._session_transition_pending = True
        app._session = None
        app._run_slash_command("/stop")
        for _ in range(10):
            await pilot.pause()

        assert dispatched == [], "a /reload must not dispatch a stop for the outgoing session"
        notices = _notices(app)
        # "Still starting" is TRUE here: a replacement really is on its way.
        assert any("still starting" in text for text in notices), notices
        assert not any("lost its link" in text for text in notices), notices


@pytest.mark.asyncio
async def test_a_strand_mid_swap_leaves_the_kill_switch_armed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """BLOCKER-3: the watched id must SURVIVE a failed swap.

    `_attach_or_refuse` (and `_reload_session`) null the binding and then adopt
    a replacement. If anything in between raises, the viewer is stranded while
    the session it was watching keeps running — for a remote, in another
    process, still billing. That is the exact state this whole feature exists
    to serve, and the round-2 remediation broke it by clearing the id inside
    that window: `/stop` then dispatched nothing and told the user to WAIT,
    which is the pre-PR failure mode with the pre-PR wording.

    An ordinary transition is kept honest by `_session_transition_pending`
    alone (pinned by the reload test above), so the id-clear bought nothing
    there and cost the lever here.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    dispatched: list[str] = []

    async def spy(session_id: str) -> None:
        dispatched.append(session_id)

    async with app.run_test(size=(100, 30)) as pilot:
        await _booted(app, pilot, session)
        assert app._adopted_session_id == "sess"
        app._stop_unbound_worker = spy  # type: ignore[assignment]

        # Drive the REAL `_attach_or_refuse` rather than hand-rolling its
        # sequence: the defect was one statement inside that window, so a test
        # that reproduces the shape by hand would step straight over it. Its
        # imports are function-local, so the record lookup and the connect are
        # patched at their source modules and everything else is the real path.
        remote_record = _record(90909, "the remote")
        remote_record.protocol = 4

        async def fake_find(_root: Any, _concrete: str):
            return remote_record, 90909

        async def fake_connect(*args: Any, **kwargs: Any):
            return FakeSession()

        # The failure lands in `_reset_ledger_for_swap`, which the source runs
        # BETWEEN nulling the binding and adopting the replacement. That is the
        # actual strand window: a raise later, inside `_adopt_session`, is
        # harmless because it has already re-stamped the id by then.
        def explode() -> Any:
            raise RuntimeError("swap failed before the replacement was adopted")

        monkeypatch.setattr(app, "_reset_ledger_for_swap", explode)

        monkeypatch.setattr(
            "local_operator.mobile.attach_client.find_owner_record",
            lambda root, concrete: (remote_record, 90909),
        )
        monkeypatch.setattr("local_operator.session.remote.RemoteSession.connect", fake_connect)
        app._resume_factory = fake_find  # type: ignore[assignment]

        with pytest.raises(RuntimeError):
            await app._attach_or_refuse(tmp_path, "remote-1", 90909)

        assert app._session is None, "the viewer really is unbound"
        assert app._adopted_session_id == "sess", "the only lever the user has must survive"

        app._run_slash_command("/stop")
        for _ in range(20):
            await pilot.pause()
            if dispatched:
                break

        assert dispatched == ["sess"], "a stranded viewer must still be able to stop what it sees"
        assert "session is still starting…" not in _notices(
            app
        ), "telling a user to wait while a remote bills is the failure this PR removes"
