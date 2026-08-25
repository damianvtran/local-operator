from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin


@pytest.mark.parametrize(
    ("cmux", "bridge", "advertised"),
    [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
)
def test_builder_selection(monkeypatch, cmux: bool, bridge: bool, advertised: bool) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: cmux)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: bridge)
    assert (builtin.build_browser_tool(None) is not None) is advertised


@pytest.mark.asyncio
async def test_cmux_preferred_when_both_available(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    cmux_open = AsyncMock(return_value=builtin._text("t", "browser", "cmux"))
    bridge_open = AsyncMock(return_value=builtin._text("t", "browser", "bridge"))
    monkeypatch.setattr(builtin, "_browser_open", cmux_open)
    monkeypatch.setattr(builtin, "_bridge_open", bridge_open)
    context = ToolContext(browser=BrowserSurface())
    result = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert result.text == "cmux"
    cmux_open.assert_awaited_once()
    bridge_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_bridge_fallback_and_token_routing(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    bridge_open = AsyncMock(return_value=builtin._text("t", "browser", "bridge"))
    bridge_action = AsyncMock(return_value=builtin._text("t", "browser", "read"))
    monkeypatch.setattr(builtin, "_bridge_open", bridge_open)
    monkeypatch.setattr(builtin, "_bridge_action", bridge_action)
    surface = BrowserSurface()
    context = ToolContext(browser=surface)
    assert (
        await builtin.execute_browser(
            "t", {"action": "open", "url": "https://example.com"}, None, None, context
        )
    ).text == "bridge"
    surface.surface_id = "bridge:12:nonce"
    assert (
        await builtin.execute_browser("t", {"action": "read"}, None, None, context)
    ).text == "read"
    bridge_action.assert_awaited_once()
