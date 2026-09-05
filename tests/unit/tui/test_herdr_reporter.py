"""The app's and the band's half of Herdr agent-state reporting.

As with the multiplexer broadcast, the single most important property is that
the SUITE ITSELF must not report: a developer running the tests from inside a
Herdr pane must not have a pilot test overwrite — and then RELEASE — the
Agents row of the session they are running the tests from. Two independent
guards stop that, and this file pins the second one. ``tests/conftest.py``
scrubs ``HERDR_*`` in the autouse isolation fixture, so detection fails
before any gate is consulted; the app's ``is_headless`` check is what still
holds if that scrub is ever narrowed. The first two tests here set the
markers back explicitly and pin the headless gate and its lifted control.

The band tests below use the same ``FakeDock`` host as the terminal-title
tests: the band is a plain object and needs no running app.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, cast

import pytest
from textual.widgets import Static

from local_operator.herdr.reporter import HerdrReporter
from local_operator.session.protocol import SessionProtocol
from local_operator.tui.app import OperatorApp
from local_operator.tui.terminal_title import TerminalTitle
from local_operator.tui.widgets.approval import ApprovalPrompt
from local_operator.tui.widgets.status_line import StatusLine
from tests.unit.test_herdr_reporter import Recorder
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _isolate_tui_settings
from tests.unit.tui.test_status_line import FakeDock


def _fake_reporter(recorder: Recorder, session_id: str | None = "sess") -> HerdrReporter:
    counter = itertools.count(1)
    return HerdrReporter(
        pane_id="w1:p1",
        binary="/opt/herdr",
        session_id=session_id,
        invoker=recorder,
        clock=lambda: next(counter),
    )


@pytest.fixture
def herdr_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Markers set and a binary resolvable, so only the app's gates remain."""
    shim = tmp_path / "herdr"
    shim.write_text("#!/bin/sh\nexit 0\n")
    shim.chmod(0o755)
    monkeypatch.setenv("HERDR_ENV", "1")
    monkeypatch.setenv("HERDR_PANE_ID", "w1:p1")
    monkeypatch.setenv("HERDR_BIN_PATH", str(shim))
    monkeypatch.delenv("LOCAL_OPERATOR_NO_HERDR", raising=False)
    return shim


@pytest.fixture
def user_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Back ``FakeSession``'s id with a real, user-owned session directory.

    ``is_user_owned_session`` reads ``origin.json`` under the config dir; a
    missing directory reads as the user's own, but the directory is created so
    the test is not relying on that default.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "lo-config"))
    (tmp_path / "lo-config" / "sessions" / FakeSession().session_id).mkdir(parents=True)


@pytest.fixture
def spy_start(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Route ``start_reporter`` at a recording reporter instead of the CLI."""
    recorder = Recorder()

    def fake_start(session_id: str | None = None, **kwargs: Any) -> HerdrReporter:
        return _fake_reporter(recorder, session_id)

    monkeypatch.setattr("local_operator.herdr.start_reporter", fake_start)
    return recorder


# ---------------------------------------------------------------------------
# App gates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_headless_app_reports_nothing(
    herdr_env: Path, user_session: None, spy_start: Recorder
) -> None:
    """The guard that keeps this suite off the developer's own Herdr pane."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app.is_headless is True
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        await pilot.pause()
        assert app._herdr_reporter is None
    assert spy_start.calls == []


@pytest.mark.asyncio
async def test_the_same_arrangement_reports_once_the_gate_is_lifted(
    herdr_env: Path, user_session: None, spy_start: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the test above, and the initial-report contract.

    Attaching the reporter is what emits the first report; it carries the
    band's ACTUAL state (idle at a prompt), the session id, and seq 1.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        assert app._herdr_reporter is not None
        (call,) = spy_start.wait_for_calls(1)
        subcommand, argv = call
        assert subcommand == "report-agent"
        assert argv[argv.index("--state") + 1] == "idle"
        assert argv[argv.index("--seq") + 1] == "1"
        assert argv[argv.index("--agent-session-id") + 1] == FakeSession().session_id
    # Leaving the context unmounts, which releases — exactly once, last.
    spy_start.wait_for_calls(2)
    assert [sub for sub, _ in spy_start.calls] == ["report-agent", "release-agent"]


@pytest.mark.asyncio
async def test_a_subagents_session_is_never_reported(
    herdr_env: Path, spy_start: Recorder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A delegated child runs in its parent's pane and must not take its row."""
    from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "lo-config"))
    directory = tmp_path / "lo-config" / "sessions" / FakeSession().session_id
    directory.mkdir(parents=True)
    mark_session_origin(directory, ORIGIN_SUBAGENT)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        assert app._herdr_reporter is None
    assert spy_start.calls == []


@pytest.mark.asyncio
async def test_the_kill_switch_disables_the_app_path(
    herdr_env: Path, user_session: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the REAL `start_reporter`, so the env gate is what is tested."""
    monkeypatch.setenv("LOCAL_OPERATOR_NO_HERDR", "1")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        assert app._herdr_reporter is None


@pytest.mark.asyncio
async def test_outside_herdr_the_app_path_is_a_no_op(
    user_session: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("HERDR_ENV", "HERDR_PANE_ID", "HERDR_BIN_PATH"):
        monkeypatch.delenv(name, raising=False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        assert app._herdr_reporter is None


@pytest.mark.asyncio
async def test_a_start_failure_never_reaches_the_user(
    herdr_env: Path, user_session: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("herdr exploded")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr("local_operator.herdr.start_reporter", boom)
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        try:
            app._start_herdr_reporter()
        except RuntimeError:  # pragma: no cover - the failure this test forbids
            pytest.fail("a herdr start failure escaped into the app")
        assert app._herdr_reporter is None


@pytest.mark.asyncio
async def test_a_session_swap_relabels_the_row_without_releasing_it(
    herdr_env: Path, user_session: None, spy_start: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row belongs to the pane; a `/new` changes only the session-id metadata."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        first = app._herdr_reporter
        assert first is not None
        spy_start.wait_for_calls(1)

        class SwappedSession(FakeSession):
            @property
            def session_id(self) -> str:
                return "sess-two"

        app._session = cast(SessionProtocol, SwappedSession())
        app._start_herdr_reporter()
        assert app._herdr_reporter is first
        spy_start.wait_for_calls(2)
        subs = [sub for sub, _ in spy_start.calls]
        assert subs == ["report-agent", "report-agent"]
        assert "release-agent" not in subs
        _, argv = spy_start.calls[1]
        assert argv[argv.index("--agent-session-id") + 1] == "sess-two"


@pytest.mark.asyncio
async def test_the_full_lifecycle_through_the_real_app(
    herdr_env: Path,
    user_session: None,
    spy_start: Recorder,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """idle → working → blocked → working → idle → release, through the app.

    Drives the same hooks production does: the band's `streaming` from the
    turn worker, the parked approval through `_refresh_working_activity`, and
    the release through unmount. And with the terminal title DISABLED, which
    is the regression this pins: the Herdr report must not be silenced by a
    different feature's kill switch.
    """
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
        assert app._terminal_title is None, "the title switch must be off for this test"
        app._session = cast(SessionProtocol, FakeSession())
        app._start_herdr_reporter()
        assert app._status is not None
        spy_start.wait_for_calls(1)

        app._status.update(streaming=True)
        spy_start.wait_for_calls(2)
        approval = ApprovalPrompt("bash", "run a command")
        app._approval = approval
        app._refresh_working_activity()
        spy_start.wait_for_calls(3)
        app._approval = None
        app._refresh_working_activity()
        spy_start.wait_for_calls(4)
        app._status.update(streaming=False)
        spy_start.wait_for_calls(5)
    spy_start.wait_for_calls(6)
    assert spy_start.states() == ["idle", "working", "blocked", "working", "idle"]
    assert [sub for sub, _ in spy_start.calls][-1] == "release-agent"
    seqs = spy_start.seqs()
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)


# ---------------------------------------------------------------------------
# The band
# ---------------------------------------------------------------------------


def _band(recorder: Recorder) -> tuple[StatusLine, HerdrReporter]:
    status = StatusLine(cast(Static, FakeDock(120)))
    reporter = _fake_reporter(recorder)
    status.set_herdr_reporter(reporter)
    return status, reporter


def test_attaching_reports_the_current_state_once() -> None:
    recorder = Recorder()
    status, _ = _band(recorder)
    (call,) = recorder.wait_for_calls(1)
    assert call[1][call[1].index("--state") + 1] == "idle"
    # Redundant syncs (a repaint, a spinner tick) add nothing.
    status.refresh()
    status.refresh()
    status.update(cwd="/tmp")
    assert len(recorder.calls) == 1


def test_streaming_and_attention_drive_the_states() -> None:
    recorder = Recorder()
    status, reporter = _band(recorder)
    status.update(streaming=True)
    status.set_attention(True)
    status.set_attention(False)
    status.update(streaming=False)
    reporter.release()
    reporter.join()
    assert recorder.states() == ["idle", "working", "blocked", "working", "idle"]


def test_attention_while_idle_recovers_to_idle() -> None:
    """An `ask` outside a streaming turn: blocked, then idle, never working."""
    recorder = Recorder()
    status, reporter = _band(recorder)
    status.set_attention(True)
    status.set_attention(False)
    reporter.release()
    reporter.join()
    assert recorder.states() == ["idle", "blocked", "idle"]


def test_a_failed_turn_reports_idle_not_unknown() -> None:
    recorder = Recorder()
    status, reporter = _band(recorder)
    status.update(streaming=True)
    status.update(streaming=False, failed=True)
    reporter.release()
    reporter.join()
    assert recorder.states() == ["idle", "working", "idle"]


def test_the_report_fires_without_a_terminal_title_attached() -> None:
    """`_sync_terminal_title` early-returns on a missing title; Herdr is before that."""
    recorder = Recorder()
    status = StatusLine(cast(Static, FakeDock(120)))
    assert status._title is None
    status.set_herdr_reporter(_fake_reporter(recorder))
    status.update(streaming=True)
    recorder.wait_for_calls(2)
    assert recorder.states() == ["idle", "working"]


def test_the_band_and_the_title_agree() -> None:
    """One derivation, two sinks: the title's glyph and the row's state move together."""
    recorder = Recorder()
    status, reporter = _band(recorder)
    writes: list[str] = []
    title = TerminalTitle(writes.append)
    title.start()
    status.set_terminal_title(title)
    status.set_attention(True)
    reporter.release()
    reporter.join()
    assert recorder.states()[-1] == "blocked"
    assert writes[-1] == "\x1b]0;lo !\x07"


def test_detaching_stops_reports() -> None:
    recorder = Recorder()
    status, reporter = _band(recorder)
    recorder.wait_for_calls(1)
    status.set_herdr_reporter(None)
    status.update(streaming=True)
    reporter.release()
    reporter.join()
    assert recorder.states() == ["idle"]
