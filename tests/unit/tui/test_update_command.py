"""TUI ``/update`` and the relaunching ``/reload``."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Event
from unittest.mock import patch

import pytest

from local_operator.reexec import REEXEC_CODE, RestartPlan
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from local_operator.tui.widgets.welcome import WelcomeView
from local_operator.update import MobileRefresh, UpdateError, VersionCheck
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _set_editor_line


@pytest.fixture(autouse=True)
def _clear_reexec_plan() -> Iterator[None]:
    from local_operator.reexec import take_plan

    take_plan()
    yield
    take_plan()


@pytest.fixture(autouse=True)
def _no_real_mobile_refresh() -> Iterator[None]:
    """Existing success paths must not probe the live LaunchAgent.

    Tests that care about the bounce patch this target themselves; the
    default is a no-op skip so CI never talks to launchctl.
    """
    with patch(
        "local_operator.update.refresh_mobile_after_upgrade",
        return_value=MobileRefresh(kind="skipped"),
    ):
        yield


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _notice_kinds(app: OperatorApp) -> list[tuple[str, str]]:
    return [
        (block._text, block._token)
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_update_when_current_does_not_reexec() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.27.0", behind=False)
    with patch("local_operator.update.check_latest", return_value=check):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one("Editor")
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if any("already on" in text for text in _notices(app)):
                    break
            assert any("already on v0.27.0" in text for text in _notices(app))
            assert app.return_code != REEXEC_CODE
            assert app._restart_plan is None


@pytest.mark.asyncio
async def test_update_success_exits_75() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", return_value="0.28.0"),
    ):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one("Editor")
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE
            assert isinstance(app._restart_plan, RestartPlan)


@pytest.mark.asyncio
async def test_reload_exits_75_with_resumable_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    session_dir = tmp_path / "sessions" / "sess"
    session_dir.mkdir(parents=True)
    (session_dir / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one("Editor")
        editor.focus()
        _set_editor_line(editor, "/reload")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code == REEXEC_CODE
        plan = app._restart_plan
        assert isinstance(plan, RestartPlan)
        assert plan.resume_id == "sess"
        assert "--resume" in plan.argv
        assert "sess" in plan.argv


@pytest.mark.asyncio
async def test_live_turn_refuses_update_and_reload() -> None:
    session = FakeSession()
    session.streaming = True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one("Editor")
        editor.focus()
        _set_editor_line(editor, "/update")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code != REEXEC_CODE
        _set_editor_line(editor, "/reload")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code != REEXEC_CODE
        assert any("esc first" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_compacting_refuses_update_and_reload() -> None:
    """M1: compaction holds the session lock with is_streaming False."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._compacting = True
        editor = app.query_one("Editor")
        editor.focus()
        _set_editor_line(editor, "/update")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code != REEXEC_CODE
        _set_editor_line(editor, "/reload")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code != REEXEC_CODE
        assert any("esc first" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_loop_refuse_copy_names_the_loop() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        app._loop_running = True
        editor = app.query_one("Editor")
        editor.focus()
        _set_editor_line(editor, "/reload")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code != REEXEC_CODE
        assert any("a loop is still running" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_second_update_is_refused_while_one_is_running() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    started = Event()
    released = Event()

    def _slow_upgrade(*, target: str, **_kwargs: object) -> str:
        started.set()
        released.wait(timeout=5)
        return target

    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", side_effect=_slow_upgrade),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if started.is_set() or app._update_in_progress:
                    break
            assert app._update_in_progress is True
            assert editor.read_only is True
            app._cmd_update(app._notice)
            assert any("already running" in text for text in _notices(app))
            released.set()
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE


@pytest.mark.asyncio
async def test_reload_during_in_flight_upgrade_is_refused() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    released = Event()

    def _slow_upgrade(*, target: str, **_kwargs: object) -> str:
        released.wait(timeout=5)
        return target

    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", side_effect=_slow_upgrade),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if app._update_in_progress:
                    break
            app._cmd_reload(app._notice)
            await pilot.pause()
            assert app.return_code != REEXEC_CODE
            assert any("already running" in text for text in _notices(app))
            released.set()
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE


@pytest.mark.asyncio
async def test_composer_is_not_submittable_during_upgrade() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    released = Event()

    def _slow_upgrade(*, target: str, **_kwargs: object) -> str:
        released.wait(timeout=5)
        return target

    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", side_effect=_slow_upgrade),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if app._update_in_progress:
                    break
            assert editor.read_only is True
            assert editor.can_focus is False
            released.set()
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE
            # Lock held through relaunch so a late submit cannot cancel it.
            assert app._update_in_progress is True
            assert editor.read_only is True


@pytest.mark.asyncio
async def test_successful_upgrade_exits_75_even_if_something_looks_live() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)

    original = app._request_relaunch

    def _force_looks_live(*, force: bool = False) -> None:
        app._compacting = True
        original(force=force)

    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", return_value="0.28.0"),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._request_relaunch = _force_looks_live  # type: ignore[method-assign]
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE
            assert any("restarting" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_git_snapshot_notice_prints_before_installer() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", return_value="0.28.0"),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=True),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            notices = _notices(app)
            assert any("built from git" in text for text in notices)
            assert any("PyPI wheel" in text for text in notices)
            assert app.return_code == REEXEC_CODE


@pytest.mark.asyncio
async def test_reload_prints_relaunching_before_exit() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        _set_editor_line(editor, "/reload")
        await pilot.press("enter")
        await pilot.pause()
        assert app.return_code == REEXEC_CODE
        assert any("relaunching" in text for text in _notices(app))
        assert any("new session" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_failed_update_from_splash_keeps_empty_state() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch(
            "local_operator.update.perform_upgrade",
            side_effect=UpdateError("this interpreter is the repo .venv"),
        ),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.EDITABLE
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if any("repo checkout" in text for text in _notices(app)):
                    break
            notices = _notices(app)
            assert any("checking for updates" in text for text in notices)
            assert any("repo checkout" in text for text in notices)
            assert not any("repo .venv" in text for text in notices)
            assert app.query_one(WelcomeView).display is True
            assert app.return_code != REEXEC_CODE
            assert editor.read_only is False


@pytest.mark.asyncio
async def test_installer_failure_names_a_shell_retry() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch(
            "local_operator.update.perform_upgrade",
            side_effect=UpdateError("installer exited 1"),
        ),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if any("upgrade failed" in text for text in _notices(app)):
                    break
            notices = _notices(app)
            assert any("uv tool upgrade local-operator" in text for text in notices)
            assert app.query_one(WelcomeView).display is True
            assert app.return_code != REEXEC_CODE


@pytest.mark.asyncio
async def test_update_refresh_fail_still_exits_75() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", return_value="0.28.0"),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
        patch(
            "local_operator.update.refresh_mobile_after_upgrade",
            return_value=MobileRefresh(kind="failed", error="kickstart failed"),
        ),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE
            assert isinstance(app._restart_plan, RestartPlan)
            notices = _notices(app)
            assert any(
                "updated, but the mobile daemon did not restart — run lop mobile restart" in text
                for text in notices
            )
            assert any("updated to v0.28.0 — restarting…" in text for text in notices)
            assert app._update_in_progress is True
            assert editor.read_only is True
            kinds = {text: token for text, token in _notice_kinds(app)}
            assert (
                kinds["updated, but the mobile daemon did not restart — run lop mobile restart"]
                == "warning"
            )


@pytest.mark.asyncio
async def test_update_refresh_success_still_exits_75() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.28.0", behind=True)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.perform_upgrade", return_value="0.28.0"),
        patch("local_operator.update.install_kind") as kind,
        patch("local_operator.update.is_git_snapshot", return_value=False),
        patch(
            "local_operator.update.refresh_mobile_after_upgrade",
            return_value=MobileRefresh(kind="restarted"),
        ),
    ):
        from local_operator.update import InstallKind

        kind.return_value = InstallKind.UV_TOOL
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(60):
                await pilot.pause()
                if app.return_code == REEXEC_CODE:
                    break
            assert app.return_code == REEXEC_CODE
            assert isinstance(app._restart_plan, RestartPlan)
            notices = _notices(app)
            assert any("mobile daemon restarted — refresh the phone UI" in text for text in notices)
            assert any("updated to v0.28.0 — restarting…" in text for text in notices)
            kinds = {text: token for text, token in _notice_kinds(app)}
            assert kinds["mobile daemon restarted — refresh the phone UI"] == "dim"


@pytest.mark.asyncio
async def test_update_when_current_does_not_refresh() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    check = VersionCheck(installed="0.27.0", latest="0.27.0", behind=False)
    with (
        patch("local_operator.update.check_latest", return_value=check),
        patch("local_operator.update.refresh_mobile_after_upgrade") as refresh,
    ):
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            editor = app.query_one("Editor")
            editor.focus()
            _set_editor_line(editor, "/update")
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()
                if any("already on" in text for text in _notices(app)):
                    break
            assert any("already on v0.27.0" in text for text in _notices(app))
            refresh.assert_not_called()
            assert app.return_code != REEXEC_CODE
