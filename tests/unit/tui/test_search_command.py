from __future__ import annotations

import pytest

from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory, _transcript_text


@pytest.mark.asyncio
async def test_search_command_shows_status_and_applies_provider_toggle(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search")
        await pilot.pause()
        first = _transcript_text(app)
        assert "web search" in first
        assert "round_robin" in first
        assert "duckduckgo" in first and "enabled · ready" in first
        assert "local-operator search setup <provider>" in first

        app._run_slash_command("/search disable tavily")
        app._run_slash_command("/search")
        await pilot.pause()
        after = _transcript_text(app)
        assert "tavily disabled; applies to the next search" in after
        assert "tavily" in after and "disabled · ready" in after

    assert session.prompts == []
