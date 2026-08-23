"""TUI ``/update`` and the relaunching ``/reload``."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from local_operator.reexec import REEXEC_CODE, RestartPlan
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView
from local_operator.update import VersionCheck
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _set_editor_line


@pytest.fixture(autouse=True)
def _clear_reexec_plan() -> Iterator[None]:
    from local_operator.reexec import take_plan

    take_plan()
    yield
    take_plan()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
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
