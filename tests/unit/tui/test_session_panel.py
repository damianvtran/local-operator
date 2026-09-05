"""Current-session diagnostics: real slash dispatch, read-only and stale-safe."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import replace

import pytest

from local_operator.analytics.model import SessionReport, UsageAggregate
from local_operator.analytics.store import AnalyticsStore
from local_operator.session.frontend_state import FrontendSessionState
from local_operator.tui.app import OperatorApp, slash_command_for
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.session_panel import (
    SessionDiagnostics,
    SessionScreen,
    build_session_report,
)
from tests.unit.analytics.test_store import _snap
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_slash_echo import _submit


def runtime():
    return SessionDiagnostics(
        "sess",
        "Investigate request latency",
        "selected/model",
        "effective/model",
        False,
        context_tokens=1200,
        context_window=200000,
        context_is_estimate=True,
        generation=3,
    )


def test_report_empty_unavailable_and_cost_knowledge():
    empty = build_session_report(SessionReport("sess"), runtime()).plain
    unavailable = build_session_report(SessionReport("sess", available=False), runtime()).plain
    assert "No recorded requests" in empty and "Ledger unavailable" not in empty
    assert "Ledger unavailable" in unavailable and "No recorded requests" not in unavailable
    assert "selected/model" in empty and "effective/model" in empty
    assert "~1,200 / 200,000" in empty
    for cost, known, expected in [
        (0, 0, "$—"),
        (0, 1, "$0.0000+"),
        (0, 2, "$0.0000"),
        (1000, 1, "$0.0010+"),
    ]:
        report = SessionReport(
            "sess", aggregate=UsageAggregate(calls=2, cost_micro=cost, cost_known_calls=known)
        )
        text = build_session_report(report, runtime()).plain
        assert expected in text
        assert "Input/output/tool dollars are unavailable separately" in text
        assert "list-price estimate" in text
        assert "pending recorder writes" in text
        assert "not provider compute" in text
        assert "Tool inventory" in text and "Tool schemas" in text and "Tool results" in text
        assert "Conversation" in text


def test_runtime_captures_only_supported_scalars():
    class SessionWithState(FakeSession):
        frontend_state: FrontendSessionState

    session = SessionWithState()
    session.frontend_state = FrontendSessionState(
        session_id="sess", epoch="test", generation=8, context_tokens=90, context_window=100
    )
    captured = SessionDiagnostics.capture(session)
    session.frontend_state = session.frontend_state.model_copy(update={"generation": 9})
    assert captured.generation == 8 and captured.context_tokens == 90
    assert SessionDiagnostics.capture(FakeSession()).context_tokens is None


def test_slash_discovery_and_locality():
    from local_operator.session.frontend_state import CommandScope, _slash_capabilities

    command = slash_command_for("/session")
    assert command is not None and not command.echo and not command.consumes_prompt
    capabilities = _slash_capabilities()
    assert (
        next(c for c in capabilities if c.command == "session").scope == CommandScope.FRONTEND_LOCAL
    )


@pytest.mark.parametrize("size", [(80, 24), (50, 24)])
@pytest.mark.parametrize("close_key", ["escape", "q"])
@pytest.mark.asyncio
async def test_assembled_slash_snapshot_scroll_and_focus(tmp_path, monkeypatch, size, close_key):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: path)
    store = AnalyticsStore(path)
    store.record_batch([replace(_snap(session_id="sess"), request_id="request-" + "a" * 80)])
    store.close()
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        history = list(session.history())
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, SessionScreen)
        screen = app.screen
        assert screen.report is not None
        assert screen.report.aggregate.calls == 1
        assert screen._scroll.max_scroll_y > 0
        assert screen._scroll.max_scroll_x == 0
        await pilot.press("end")
        await pilot.pause()
        assert screen._scroll.scroll_y > 0
        await pilot.press("home")
        await pilot.pause()
        assert screen._scroll.scroll_y == 0
        store.record_batch([_snap(session_id="sess")])
        store.close()
        assert screen.report.aggregate.calls == 1  # pinned until close/reopen
        await pilot.press(close_key)
        await pilot.pause()
        assert not isinstance(app.screen, SessionScreen)
        assert app.focused is app.query_one(Editor)
        assert session.history() == history
        assert not session.prompts
        await _submit(pilot, app, "/session")
        await app.workers.wait_for_complete()
        assert isinstance(app.screen, SessionScreen)
        assert app.screen.report is not None
        assert app.screen.report.aggregate.calls == 2


@pytest.mark.parametrize("replacement_kind", ["new", "resume", "epoch"])
@pytest.mark.asyncio
async def test_disk_worker_is_off_loop_and_drops_stale_session(monkeypatch, replacement_kind):
    started = threading.Event()
    finish = threading.Event()
    ui_thread = threading.get_ident()
    observed = []

    def delayed(self, session_id):
        observed.append(threading.get_ident())
        started.set()
        assert finish.wait(5)
        return SessionReport(session_id)

    monkeypatch.setattr(AnalyticsStore, "session_report", delayed)

    class EpochSession(FakeSession):
        frontend_state: FrontendSessionState

    session = EpochSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test() as pilot:
        await pilot.pause()
        session.frontend_state = FrontendSessionState(session_id="sess", epoch="before")
        app._cmd_session("", lambda body, kind="info": None)
        try:
            assert await asyncio.to_thread(started.wait, 5)

            # This loop-side mutation is possible while SQLite is blocked. A
            # replacement for the same ID models /resume as well as /new.
            class NewSession(FakeSession):
                @property
                def session_id(self) -> str:
                    return "new"

            if replacement_kind == "epoch":
                session.frontend_state = session.frontend_state.model_copy(
                    update={"epoch": "after"}
                )
            else:
                app._session = FakeSession() if replacement_kind == "resume" else NewSession()
            finish.set()
            await app.workers.wait_for_complete()
            assert len(observed) == 1 and observed[0] != ui_thread
            # The result cannot be published, even if the pending modal is
            # still finishing its queued mount before it can safely be popped.
            if isinstance(app.screen, SessionScreen):
                assert app.screen.presentation_cancelled and app.screen.report is None
            await pilot.pause()
            assert not isinstance(app.screen, SessionScreen)
        finally:
            finish.set()
            app._session = session


@pytest.mark.asyncio
async def test_invalid_arguments_never_request_model():
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/session another-id")
        await app.workers.wait_for_complete()
        assert not isinstance(app.screen, SessionScreen)
        assert not session.prompts


@pytest.mark.asyncio
async def test_scroll_hint_tracks_actual_overflow_after_resize():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        screen = SessionScreen(SessionReport("sess"), runtime())
        app.push_screen(screen)
        await pilot.pause()
        assert screen._scroll.max_scroll_y == 0
        assert "scroll" not in str(screen._hint.render())
        await pilot.resize_terminal(50, 24)
        await pilot.pause()
        assert screen._scroll.max_scroll_y > 0
        assert "↑↓ scroll" in str(screen._hint.render())
        assert screen._scroll.max_scroll_x == 0


@pytest.mark.parametrize("cancel_key", ["escape", "q"])
@pytest.mark.asyncio
async def test_real_locked_read_can_be_cancelled_without_interrupting_draft(
    tmp_path, monkeypatch, cancel_key
):
    path = tmp_path / "ledger.db"
    monkeypatch.setattr("local_operator.analytics.store.default_db_path", lambda: path)
    store = AnalyticsStore(path)
    store.record_batch([_snap(session_id="sess")])
    store.close()
    # WAL readers intentionally bypass a writer lock. A real legacy DELETE
    # journal reproduces the reported slow-read path without sleeping a mock.
    locker = sqlite3.connect(path)
    locker.execute("PRAGMA journal_mode=DELETE")
    locker.execute("BEGIN EXCLUSIVE")
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    try:
        async with app.run_test(size=(50, 24)) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "/session")
            try:
                assert isinstance(app.screen, SessionScreen)
                pending = app.screen
                assert pending.report is None
                assert "Loading usage records" in str(pending._body.render())
                assert "esc / q cancel" in str(pending._hint.render())
                await pilot.press(cancel_key)
                await pilot.press(*list("next task"))
                await pilot.pause()
                assert app.focused is app.query_one(Editor)
                assert app.query_one(Editor).text == "next task"
            finally:
                locker.rollback()
                await app.workers.wait_for_complete()
            await pilot.pause()
            assert not isinstance(app.screen, SessionScreen)
            assert pending.presentation_cancelled and pending.report is None
            assert app.focused is app.query_one(Editor)
            assert app.query_one(Editor).text == "next task"
            assert session.prompts == [] and session.aborts == []
    finally:
        locker.close()


@pytest.mark.asyncio
async def test_fast_result_during_mount_replaces_loading_text():
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test() as pilot:
        await pilot.pause()
        screen = SessionScreen(None, runtime())
        app.push_screen(screen)
        # No pause/await: the SQLite worker can settle before the mounted body
        # exists. Losing this update would strand a fast read in loading state.
        screen.set_report(SessionReport("sess"))
        await pilot.pause()
        assert "No recorded requests" in str(screen._body.render())
        assert "Loading usage records" not in str(screen._body.render())
        assert "cancel" not in str(screen._hint.render())


@pytest.mark.asyncio
async def test_invalidating_buried_pending_view_never_pops_newer_modal():
    from local_operator.tui.widgets.analytics_panel import AnalyticsScreen

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test() as pilot:
        await pilot.pause()
        pending = SessionScreen(None, runtime())
        app.push_screen(pending)
        await pilot.pause()
        other = AnalyticsScreen(UsageAggregate())
        app.push_screen(other)
        await pilot.pause()
        pending.invalidate()
        pending.set_report(SessionReport("sess"))
        assert app.screen is other
        assert pending.report is None
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, (SessionScreen, AnalyticsScreen))
        assert app.focused is app.query_one(Editor)
