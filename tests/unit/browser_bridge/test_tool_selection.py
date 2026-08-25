from __future__ import annotations

from typing import Any
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
async def test_bridge_preferred_when_both_available(monkeypatch) -> None:
    # Precedence flipped (operator decision): a fresh open prefers the paired
    # extension over cmux when both are reachable, because the extension drives
    # the user's real profile and never steals focus. cmux stays a fallback.
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
    assert result.text == "bridge"
    bridge_open.assert_awaited_once()
    cmux_open.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmux_used_when_bridge_absent(monkeypatch) -> None:
    # With no extension connected, cmux is the fallback and drives the open.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: False)
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
async def test_open_cmux_surface_stays_on_cmux(monkeypatch) -> None:
    # An already-open cmux surface pins the transport: even with the bridge
    # available, a second open on that surface must not silently jump backends.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    cmux_open = AsyncMock(return_value=builtin._text("t", "browser", "cmux"))
    bridge_open = AsyncMock(return_value=builtin._text("t", "browser", "bridge"))
    monkeypatch.setattr(builtin, "_browser_open", cmux_open)
    monkeypatch.setattr(builtin, "_bridge_open", bridge_open)
    surface = BrowserSurface()
    surface.surface_id = "surface:already-open"
    context = ToolContext(browser=surface)
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


# ---------------------------------------------------------------------------
# scroll / logs — the two extension-only actions added for viewport control and
# console-log reading. These cover validation, wire-param selection, cmux
# degradation, and result rendering.
# ---------------------------------------------------------------------------


def test_scroll_and_logs_are_advertised_actions() -> None:
    assert "scroll" in builtin.BROWSER_ACTIONS
    assert "logs" in builtin.BROWSER_ACTIONS
    assert builtin.BRIDGE_ONLY_BROWSER_ACTIONS == frozenset({"scroll", "logs"})


@pytest.mark.parametrize(
    ("params", "ok"),
    [
        ({}, True),  # no params -> default one viewport down
        ({"direction": "bottom"}, True),
        ({"direction": "sideways"}, False),  # unknown keyword refused
        ({"x": 100.0, "y": -50.0}, True),
        ({"selector": "#main"}, True),
        ({"selector": "--flag"}, False),  # flag-shaped selector refused
    ],
)
def test_validate_scroll_args(params: dict[str, Any], ok: bool) -> None:
    problem = builtin._validate_browser_args(
        "scroll", builtin.BrowserParams(action="scroll", **params)
    )
    assert (problem == "") is ok


@pytest.mark.parametrize(
    ("level", "ok"),
    [("", True), ("error", True), ("all", True), ("bogus", False)],
)
def test_validate_logs_level(level: str, ok: bool) -> None:
    problem = builtin._validate_browser_args(
        "logs", builtin.BrowserParams(action="logs", level=level)
    )
    assert (problem == "") is ok


@pytest.mark.asyncio
async def test_scroll_wire_params_only_set_fields(monkeypatch) -> None:
    # The extension's precedence (selector > x/y > direction > default) depends
    # on absent params staying ABSENT on the wire, so an unset selector must not
    # arrive as "" and pre-empt an x/y scroll.
    captured: dict[str, Any] = {}

    async def fake_call(tool_call_id, action, params, *, surface=""):
        captured["action"] = action
        captured["params"] = params
        return {"scrollX": 0, "scrollY": 400, "moreBelow": True, "moreRight": False}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:nonce"
    result = await builtin._bridge_action(
        "t",
        surface,
        "scroll",
        builtin.BrowserParams(action="scroll", y=400.0),
        None,
    )
    assert captured["action"] == "scroll"
    assert captured["params"] == {"tab": "bridge:9:nonce", "y": 400.0}
    # Result reports the landing position and that more remains below.
    assert "(0, 400)" in result.text
    assert "more below" in result.text


@pytest.mark.asyncio
async def test_scroll_reports_end_of_page(monkeypatch) -> None:
    async def fake_call(tool_call_id, action, params, *, surface=""):
        return {"scrollX": 0, "scrollY": 999, "moreBelow": False, "moreRight": False}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:nonce"
    result = await builtin._bridge_action(
        "t", surface, "scroll", builtin.BrowserParams(action="scroll", direction="bottom"), None
    )
    assert "at the end (no more content)" in result.text


@pytest.mark.asyncio
async def test_logs_wire_and_rendering(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_call(tool_call_id, action, params, *, surface=""):
        captured["params"] = params
        return {
            "entries": [
                {
                    "level": "log",
                    "text": "hello",
                    "source": "console",
                    "url": "https://x/app.js",
                    "line": 12,
                    "timestamp": 1,
                },
                {
                    "level": "error",
                    "text": "TypeError: boom",
                    "source": "exception",
                    "url": "https://x/app.js",
                    "line": 40,
                    "timestamp": 2,
                },
            ]
        }, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:nonce"
    result = await builtin._bridge_action(
        "t", surface, "logs", builtin.BrowserParams(action="logs", level="all", limit=100), None
    )
    # Level defaults to "all"; the explicit limit rides through.
    assert captured["params"]["level"] == "all"
    assert captured["params"]["limit"] == 100
    # An uncaught exception is tagged distinctly from a plain console line.
    assert "[LOG] hello (https://x/app.js:12)" in result.text
    assert "[ERROR!] TypeError: boom (https://x/app.js:40)" in result.text
    assert "2 log entries" in result.text


@pytest.mark.asyncio
async def test_logs_empty(monkeypatch) -> None:
    async def fake_call(tool_call_id, action, params, *, surface=""):
        return {"entries": []}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:nonce"
    result = await builtin._bridge_action(
        "t", surface, "logs", builtin.BrowserParams(action="logs", level="error"), None
    )
    assert "No console logs at level 'error'" in result.text


@pytest.mark.asyncio
async def test_scroll_and_logs_degrade_on_cmux(monkeypatch) -> None:
    # A cmux-backed surface cannot serve these; the tool must refuse with a
    # message naming the extension rather than dispatching to cmux.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "_stale_surface_error", AsyncMock(return_value=None))
    surface = BrowserSurface()
    surface.surface_id = "surface:cmux-open"
    context = ToolContext(browser=surface)
    for action in ("scroll", "logs"):
        result = await builtin.execute_browser("t", {"action": action}, None, None, context)
        assert "not supported on the cmux backend" in result.text
        assert "Local Operator browser extension" in result.text
