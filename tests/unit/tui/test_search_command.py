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

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app._run_slash_command("/search")
        await pilot.pause()
        first = _transcript_text(app)
        assert "Web search" in first
        assert "Round Robin" in first
        assert "DuckDuckGo" in first and "enabled · available" in first
        assert "/search enable|disable <provider>" in first
        assert "/search balance round_robin|ordered" in first
        assert "search setup tavily --oauth|--api-key" in first
        assert "search setup searxng --endpoint <url>" in first

        app._run_slash_command("/search setup tavily")
        app._run_slash_command("/search disable tavily")
        app._run_slash_command("/search")
        await pilot.pause()
        after = _transcript_text(app)
        assert "run in a shell: local-operator search setup tavily --oauth" in after
        assert "tavily disabled; applies to the next search" in after
        assert "Tavily" in after and "disabled · available" in after

    assert session.prompts == []
