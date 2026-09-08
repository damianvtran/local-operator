"""Model receipts through the real app, session and per-key config watcher.

The older default-save pilot used a switchable fake without a session watcher:
it could assert one app receipt while production appended two keeping notices.
"""

from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher
from local_operator.harness.types import ModelSpec
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock
from tests.unit.session.test_config_live import (
    RebindableStream,
    make_session,
    subscribe,
)
from tests.unit.tui.test_app_pilot import _AccessController, _await_session


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [60, 100])
async def test_switch_then_save_prints_one_confirmation_per_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    _reset_for_tests()
    manager = ConfigManager(config_dir)
    manager.set_config_value("hosting", "anthropic")
    manager.set_config_value("model_name", "old")
    session = make_session(tmp_path, RebindableStream({}), model_source="config")
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    ctrl = _AccessController(stored=("openrouter", "anthropic"))
    monkeypatch.setattr(
        ctrl,
        "resolve_model",
        lambda provider, model: ModelSpec(provider=provider, model_id=model),
    )

    async def factory():
        return session

    app = OperatorApp(factory, provider_controller=ctrl)
    try:
        async with app.run_test(size=(width, 32)) as pilot:
            await _await_session(app, pilot)
            app._run_slash_command("/model openrouter/chosen")
            await pilot.pause()
            # The next command starts by opening the same picker. It must not
            # append full help after the switch just because the buffer resyncs.
            app._editor().begin_model_query()
            await pilot.pause()
            app._editor().clear_content()
            app._run_slash_command("/model default")
            await pilot.pause()
            assert watcher.poll_now() is None
            await pilot.pause()
            notices = [block.text() for block in app.query(NoticeBlock)]
            assert len(notices) == 2, notices
            assert notices[0] == "model: test/m → openrouter/chosen (this session)"
            assert notices[1] == "boot default saved: openrouter/chosen (new sessions)"
            assert session.model_label == "openrouter/chosen"
            assert session._explicit_model_choice
            saved = ConfigManager(config_dir)
            assert saved.get_config_value("hosting") == "openrouter"
            assert saved.get_config_value("model_name") == "chosen"
            # The hidden-from-transcript help remains reachable intentionally.
            app._run_slash_command("/model")
            await pilot.pause()
            guidance = [block.text() or "" for block in app.query(NoticeBlock)][-1]
            for route in ("/model default", "/model saved", "/settings"):
                assert route in guidance
    finally:
        await session.dispose()
        await watcher.stop()
        _reset_for_tests()
