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


def test_scroll_logs_and_tabs_are_advertised_actions() -> None:
    assert "scroll" in builtin.BROWSER_ACTIONS
    assert "logs" in builtin.BROWSER_ACTIONS
    assert "tabs" in builtin.BROWSER_ACTIONS
    assert builtin.BRIDGE_ONLY_BROWSER_ACTIONS == frozenset(
        {"scroll", "logs", "tabs", "request_access", "await_access", "cancel_access"}
    )


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
    # Every action wire now carries the approval-binding identity too; only
    # the scroll params themselves are asserted exactly (see the comment in
    # fake_call above for why absent fields must stay absent).
    assert captured["params"] == {
        "tab": "bridge:9:nonce",
        "requester": "call:t",
        "y": 400.0,
    }
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
    for action in ("scroll", "logs", "tabs"):
        result = await builtin.execute_browser("t", {"action": action}, None, None, context)
        assert "not supported on the cmux backend" in result.text
        assert "Local Operator browser extension" in result.text


# ---------------------------------------------------------------------------
# tabs — multi-surface discovery. Parallel sessions each own a tab, so agents
# need to list what is being driven, spot their own handle, and know which
# handle to close when the surface cap refuses a fresh open.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tabs_lists_surfaces_and_marks_own(monkeypatch) -> None:
    async def fake_call(tool_call_id, action, params, *, surface=""):
        assert action == "tabs"
        assert params == {}
        # The extension REDACTS listed handles (finding M1): truncated nonce
        # plus ellipsis, so the listing cannot hand out drive capabilities.
        return {
            "tabs": [
                {
                    "tab": "bridge:9:aaaaaa\u2026",
                    "url": "https://example.com/a",
                    "title": "Mine",
                    "createdAt": 1000,
                    "lastUsedAt": 2000,
                },
                {
                    "tab": "bridge:12:bbbbbb\u2026",
                    "url": "https://example.com/b",
                    "title": "Theirs",
                    "createdAt": 1000,
                    "lastUsedAt": 1500,
                },
            ],
            "limit": 8,
        }, None

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:aaaaaa0123456789aaaaaa0123456789"
    context = ToolContext(browser=surface)
    result = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not result.is_error
    # The caller's own tab is recognised by PREFIX-matching its full pinned
    # token against the redacted entry; the other session's is listed as
    # awareness only.
    assert "bridge:9:aaaaaa\u2026 (yours)" in result.text
    assert "bridge:12:bbbbbb\u2026:" in result.text
    assert "awareness-only" in result.text
    assert result.details is not None and result.details["tab_count"] == 2


def test_redacted_ownership_prefix_matching() -> None:
    own = "bridge:9:aaaaaa0123456789aaaaaa0123456789"
    assert builtin._owns_redacted_tab(own, "bridge:9:aaaaaa\u2026")
    # Another session's redacted entry does not match.
    assert not builtin._owns_redacted_tab(own, "bridge:9:bbbbbb\u2026")
    # Same tab id alone is not ownership — the nonce prefix must agree.
    assert not builtin._owns_redacted_tab("bridge:9:cccccc\u2026", "bridge:9:aaaaaa\u2026")
    assert not builtin._owns_redacted_tab("", "bridge:9:aaaaaa\u2026")
    # Defensive exact-match path for an unredacted value.
    assert builtin._owns_redacted_tab(own, own)


@pytest.mark.asyncio
async def test_tabs_works_without_an_owned_surface(monkeypatch) -> None:
    # Discovery must not require 'open' first: its main use is deciding
    # whether to resume an existing tab or seeing what fills the cap.
    async def fake_call(tool_call_id, action, params, *, surface=""):
        return {"tabs": []}, None

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(browser=BrowserSurface())
    result = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not result.is_error
    assert "No extension-driven browser tabs" in result.text


@pytest.mark.asyncio
async def test_bridge_open_recovers_from_a_dead_pinned_tab(monkeypatch) -> None:
    # 'open' is the recovery verb: when the session's pinned tab died, the
    # resume attempt fails with tab-gone and the tool must fall back to
    # creating a fresh tab instead of surfacing the error.
    calls: list[dict[str, Any]] = []

    async def fake_call(tool_call_id, action, params, *, surface=""):
        calls.append(params)
        if "tab" in params:
            # Recovery keys on the TYPED wire code carried in details, not on
            # the diagnostic's wording (finding m4).
            problem = builtin._error("t", "browser", "browser tab bridge:9:dead is gone.")
            problem.details = {"error_code": "tab_closed"}
            return None, problem
        return {"tab": "bridge:33:fresh", "url": "https://example.com/", "title": "E"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:dead"
    result = await builtin._bridge_open("t", surface, "https://example.com")
    assert not result.is_error
    assert surface.surface_id == "bridge:33:fresh"
    assert len(calls) == 2 and "tab" in calls[0] and "tab" not in calls[1]


# ---------------------------------------------------------------------------
# Async site-approval flow (request_access / await_access)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_access_works_without_a_surface(monkeypatch) -> None:
    # The whole point of the flow: 'open' just FAILED, so no surface exists.
    # Routing these through the "no browser surface open" guard would send the
    # agent in a circle.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)

    async def fake_call(tool_call_id, action, params, *, surface=""):
        assert action == "request_access"
        # The tool binds the approval to an identity (session when the host
        # provides one, else the tool call id — never anonymous).
        assert params == {"url": "https://example.com", "requester": "call:t"}
        return {"origin": "https://example.com", "state": "pending"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(browser=BrowserSurface())
    result = await builtin.execute_browser(
        "t", {"action": "request_access", "url": "https://example.com"}, None, None, context
    )
    # The result text is the agent's script for the next two steps.
    assert "pending" in result.text
    assert "extension popup" in result.text
    assert "await_access" in result.text


@pytest.mark.asyncio
async def test_request_access_reports_already_allowed(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)

    async def fake_call(tool_call_id, action, params, *, surface=""):
        return {"origin": "https://example.com", "state": "allowed"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    result = await builtin.execute_browser(
        "t",
        {"action": "request_access", "url": "https://example.com"},
        None,
        None,
        ToolContext(browser=BrowserSurface()),
    )
    assert "allowed" in result.text and "'open' or 'goto'" in result.text


@pytest.mark.asyncio
async def test_cancel_access_uses_the_callers_stable_identity(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)

    async def fake_call(tool_call_id, action, params, *, surface=""):
        assert action == "cancel_access"
        assert params == {"url": "https://example.com", "requester": "session:abc"}
        return {"origin": "https://example.com", "state": "cancelled", "pending_count": 2}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(browser=BrowserSurface(), session_id="abc")
    result = await builtin.execute_browser(
        "t", {"action": "cancel_access", "url": "https://example.com"}, None, None, context
    )
    assert "cancelled" in result.text
    assert result.details == {
        "origin": "https://example.com",
        "state": "cancelled",
        "pending_count": 2,
    }


@pytest.mark.asyncio
async def test_await_access_returns_decision_and_denied_warns_off_retry(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)

    async def fake_call(tool_call_id, action, params, *, surface=""):
        assert action == "await_access"
        # The tool slices the wait: each wire call carries a bounded budget.
        assert params["timeout_ms"] <= builtin._BRIDGE_AWAIT_SLICE_MS
        return {"origin": "https://example.com", "state": "denied"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    result = await builtin.execute_browser(
        "t",
        {"action": "await_access", "url": "https://example.com"},
        None,
        None,
        ToolContext(browser=BrowserSurface()),
    )
    assert "denied" in result.text and "Do not retry" in result.text


@pytest.mark.asyncio
async def test_access_actions_degrade_on_cmux_with_typed_error(monkeypatch) -> None:
    # A cmux-pinned surface has no permission model to ask; the answer must be
    # the same honest degrade pattern as scroll/logs, not a fake pending.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: False)
    surface = BrowserSurface()
    surface.surface_id = "surface:3"
    result = await builtin.execute_browser(
        "t",
        {"action": "request_access", "url": "https://example.com"},
        None,
        None,
        ToolContext(browser=surface),
    )
    assert result.is_error
    assert "not supported on the cmux backend" in result.text


@pytest.mark.asyncio
async def test_access_actions_require_a_url(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    result = await builtin.execute_browser(
        "t", {"action": "await_access"}, None, None, ToolContext(browser=BrowserSurface())
    )
    assert result.is_error and "requires a URL" in result.text


@pytest.mark.asyncio
async def test_session_identity_is_stable_across_the_whole_flow(monkeypatch) -> None:
    """Round-2 M4: the REAL plumbing — ToolContext.session_id →
    execute_browser → _browser_requester → the wire params of every access and
    navigation call — must present ONE stable identity across different tool
    call ids, or grants bind to something no later navigation can match.
    Asserted through execute_browser (the public tool path), not by injecting
    requester strings."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    seen: list[tuple[str, str]] = []

    async def fake_call(tool_call_id, action, params, *, surface=""):
        seen.append((action, str(params.get("requester", "<missing>"))))
        if action == "request_access":
            return {"origin": "https://example.com", "state": "pending"}, None
        if action == "await_access":
            return {"origin": "https://example.com", "state": "allowed"}, None
        # open
        return {"tab": "bridge:5:n0nce", "url": "https://example.com/", "title": "x"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(session_id="sess-42", browser=BrowserSurface())
    # Three DIFFERENT tool call ids — the per-command fallback would produce
    # three different identities and silently break grant consumption.
    await builtin.execute_browser(
        "call-1", {"action": "request_access", "url": "https://example.com"}, None, None, context
    )
    await builtin.execute_browser(
        "call-2", {"action": "await_access", "url": "https://example.com"}, None, None, context
    )
    await builtin.execute_browser(
        "call-3", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    assert [entry[0] for entry in seen] == ["request_access", "await_access", "open"]
    identities = {entry[1] for entry in seen}
    assert identities == {"session:sess-42"}, f"identity drifted: {seen}"


@pytest.mark.asyncio
async def test_parallel_contexts_present_distinct_identities(monkeypatch) -> None:
    """Round-2 M4: two ToolContexts (two sessions) must reach the wire as two
    different requesters — the property the extension's fail-closed grant
    check depends on."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    seen: list[str] = []

    async def fake_call(tool_call_id, action, params, *, surface=""):
        seen.append(str(params.get("requester", "<missing>")))
        return {"origin": "https://example.com", "state": "pending"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    for session_id in ("sess-A", "sess-B"):
        context = ToolContext(session_id=session_id, browser=BrowserSurface())
        await builtin.execute_browser(
            "t", {"action": "request_access", "url": "https://example.com"}, None, None, context
        )
    assert seen == ["session:sess-A", "session:sess-B"]


@pytest.mark.asyncio
async def test_goto_carries_the_session_identity(monkeypatch) -> None:
    """Round-2 M4: goto (the other admission-bearing navigation) must carry
    the same session identity; a reversion to per-call identity here would
    strand every grant minted for the session."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    captured: dict[str, Any] = {}

    async def fake_call(tool_call_id, action, params, *, surface=""):
        captured["action"] = action
        captured["requester"] = params.get("requester")
        return {"url": "https://example.com/", "title": "x"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:9:nonce"
    context = ToolContext(session_id="sess-42", browser=surface)
    await builtin.execute_browser(
        "some-other-call-id",
        {"action": "goto", "url": "https://example.com"},
        None,
        None,
        context,
    )
    assert captured["action"] == "goto"
    assert captured["requester"] == "session:sess-42"
