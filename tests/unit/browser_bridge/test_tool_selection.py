from __future__ import annotations

import os
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
    # Gating asks `advertisable`, not `available`: see
    # test_stale_but_alive_daemon_still_advertises_the_browser_tool.
    monkeypatch.setattr(builtin, "bridge_browser_advertisable", lambda: bridge)
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
    # Named for the host that CANNOT serve them. It used to be
    # `BRIDGE_ONLY_BROWSER_ACTIONS`, which on a three-host machine claimed the
    # extension was the only alternative to cmux — false now that the desktop
    # app's browser host serves every one of them.
    assert builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS == frozenset(
        {
            "scroll",
            "logs",
            "tabs",
            "request_access",
            "await_access",
            "cancel_access",
            # File transfer joined the set for a reason of its own: cmux drives a
            # terminal browser panel and has no primitive for handing a file to a
            # page or taking one from it, and the action list and the degrade check
            # are the same set so they cannot drift apart.
            "download",
            "upload",
        }
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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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
        "session_label": "Session",
        "y": 400.0,
    }
    # Result reports the landing position and that more remains below.
    assert "(0, 400)" in result.text
    assert "more below" in result.text


@pytest.mark.asyncio
async def test_scroll_reports_end_of_page(monkeypatch) -> None:
    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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
    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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
        assert "desktop app" in result.text and "browser extension" in result.text


# ---------------------------------------------------------------------------
# tabs — multi-surface discovery. Parallel sessions each own a tab, so agents
# need to list what is being driven, spot their own handle, and know which
# handle to close when the surface cap refuses a fresh open.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tabs_lists_surfaces_and_marks_own(monkeypatch) -> None:
    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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
    assert builtin._BROWSER_TABS_CLEANUP_FOOTER in result.text
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
    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        return {"tabs": []}, None

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(browser=BrowserSurface())
    result = await builtin.execute_browser("t", {"action": "tabs"}, None, None, context)
    assert not result.is_error
    assert "No agent-driven browser tabs" in result.text
    assert builtin._BROWSER_TABS_CLEANUP_FOOTER in result.text


@pytest.mark.asyncio
async def test_bridge_open_reminds_only_when_it_creates_a_new_tab(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        calls.append(params)
        return {
            "tab": params.get("tab", "bridge:33:fresh"),
            "url": "https://example.com/",
            "title": "Example",
        }, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()

    created = await builtin._bridge_open("t1", surface, "https://example.com")
    resumed = await builtin._bridge_open("t2", surface, "https://example.com/next")

    assert builtin._BROWSER_OPEN_CLEANUP_REMINDER in created.text
    assert builtin._BROWSER_OPEN_CLEANUP_REMINDER not in resumed.text
    assert "tab" not in calls[0] and calls[1]["tab"] == "bridge:33:fresh"


@pytest.mark.asyncio
async def test_bridge_open_recovers_from_a_dead_pinned_tab(monkeypatch) -> None:
    # 'open' is the recovery verb: when the session's pinned tab died, the
    # resume attempt fails with tab-gone and the tool must fall back to
    # creating a fresh tab instead of surfacing the error.
    calls: list[dict[str, Any]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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
    # Recovery creates replacement ownership after the stale handle is dropped.
    assert builtin._BROWSER_OPEN_CLEANUP_REMINDER in result.text


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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        assert action == "request_access"
        # The tool binds the approval to an identity (session when the host
        # provides one, else the tool call id — never anonymous).
        assert params == {
            "url": "https://example.com",
            "requester": "call:t",
            "session_label": "Session",
        }
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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        assert action == "cancel_access"
        assert params == {
            "url": "https://example.com",
            "requester": "session:abc",
            "session_label": "Session",
        }
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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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


async def _never_asked(questions):  # pragma: no cover — the hook is never called here
    raise AssertionError("the picker must not be reached by a text assertion")


#: The notify phrase a host that OWNS an ask hook is given (``_notify_channel``).
ASK_PHRASE = "a short message, or `ask`"


@pytest.mark.asyncio
async def test_access_actions_require_a_url(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    result = await builtin.execute_browser(
        "t", {"action": "await_access"}, None, None, ToolContext(browser=BrowserSurface())
    )
    assert result.is_error and "requires a URL" in result.text


@pytest.mark.asyncio
async def test_request_access_reports_the_sessions_attachment(monkeypatch) -> None:
    """The REAL path: the probe reaches the text the agent reads (§5.1).

    ``ToolContext.attached_probe`` is a declared field precisely so a built-in
    tool may look for it; this drives ``execute_browser`` rather than calling the
    renderer directly, so a field that was declared and never wired cannot pass.
    """
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        return {"origin": "https://example.com", "state": "pending"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    unattached = ToolContext(browser=BrowserSurface(), attached_probe=lambda: False)
    result = await builtin.execute_browser(
        "t", {"action": "request_access", "url": "https://example.com"}, None, None, unattached
    )

    assert "No Local Operator pane is attached to this session" in result.text
    assert "then proceed with what you have" in result.text
    # Still notify-first: the request is what makes the prompt visible when a
    # surface does attach.
    assert "Notify them anyway" in result.text
    # ...and it must not deny the surface this same message just named (round 1,
    # D2/U3): the prompt IS on the extension popup, which the operator can click,
    # while the predicate counts Local Operator PANES only.
    assert "nobody can act on it" not in result.text


def test_the_access_flow_reads_the_live_attached_probe() -> None:
    """No probe reads as ATTACHED, and it is re-read per call.

    ``True`` for a bare tool test or an unwired host is the pre-existing default
    and the safe direction: a wrong "attached" costs a wait that is re-checked,
    a wrong "unattached" tells the agent to give up on a question the operator
    was ready to answer.
    """
    assert builtin._attached_here(None) is True
    assert builtin._attached_here(ToolContext()) is True

    state = {"attached": False}
    context = ToolContext(attached_probe=lambda: state["attached"])
    assert builtin._attached_here(context) is False
    state["attached"] = True
    assert builtin._attached_here(context) is True

    def explode() -> bool:
        raise RuntimeError("a broken probe is not a reason to give up")

    assert builtin._attached_here(ToolContext(attached_probe=explode)) is True


def test_a_pending_prompt_says_whether_an_interface_is_attached() -> None:
    """Both variants. The notify-first instruction is shared; the ADVICE differs."""
    attached = builtin._access_result_text("pending", "https://example.com", host="")
    unattached = builtin._access_result_text(
        "pending", "https://example.com", host="", attached=False
    )

    assert "An interface is attached to this session" in attached
    assert "No interface is attached" not in attached
    assert "make sure they are told" in attached

    assert "No Local Operator pane is attached to this session" in unattached
    assert "An interface is attached to this session" not in unattached
    assert "proceed with what you have rather than blocking the turn" in unattached

    # Both name the surface the prompt is on, and both tell the agent to notify.
    for text in (attached, unattached):
        assert "in the Local Operator extension popup" in text
    assert "Notify them anyway" in unattached
    # THE UNATTACHED VARIANT MUST NOT DENY THE SURFACE IT JUST NAMED (round 1,
    # D2/U3). The predicate counts Local Operator PANES — a TUI, a leased desktop
    # renderer — while the prompt can be sitting in the extension popup, which the
    # operator can click. "nobody can act on it until a surface attaches" was that
    # contradiction, and it is the incident's own shape.
    assert "nobody can act on it" not in unattached
    assert "the operator can answer it there" in unattached
    # The host-selected sentence still follows the host argument.
    ui = builtin._access_result_text("pending", "https://example.com", host=builtin.HOST_UI_PREFIX)
    assert "desktop app's browser tab" in ui


def test_the_pending_prompt_names_the_fifteen_minute_wait_and_the_re_request() -> None:
    """The operator's ask, literally: 15 minutes, then re-request to ping again.

    The cap on ONE call does not move (240 s) — the browser tool is
    ``interruptible=False`` and a prompt cannot outlive the extension's
    10-minute TTL — so the text has to say where the rest of the budget comes
    from, and it says REPEATED ``await_access`` CALLS, which are executable. It
    used to name the ``wait`` tool, which awaits a background job and cannot be
    called at all from a session that has none (round 1, MAJOR 2 / U2).
    """
    text = builtin._access_result_text("pending", "https://example.com", host="")

    assert "UP TO 15 MINUTES" in text
    assert "Keep calling action='await_access'" in text
    assert "most 240s per call" in text or "waits at most 240s" in text
    assert "action='request_access'" in text
    assert builtin.BROWSER_AWAIT_ACCESS_MAX_S == 240.0
    # The DEFAULT stays off the cap (round 1, MINOR 4 / U6): an unsized call is
    # the one the pending text tells the model to make, and the cap is an
    # uninterruptible block.
    assert builtin.BROWSER_AWAIT_ACCESS_DEFAULT_S == 120.0
    # The re-request is BOUNDED (round 1, D5/U5): "keep doing that while the
    # origin is still needed" was an unbounded loop whose repeat is a silent
    # no-op until the prompt expires.
    assert "at most once per 15-minute window" in text
    assert "notifies nobody" in text


def test_an_unanswered_prompt_is_not_reported_as_a_refusal() -> None:
    """The claim that cost the incident: silence read as "the origin is unavailable"."""
    for attached in (True, False):
        text = builtin._access_result_text(
            "pending", "https://example.com", host="", attached=attached
        )
        assert "AN UNANSWERED PROMPT IS NOT A REFUSAL" in text
        assert "Do not report it as" in text
        # ...and it must not forbid the honest admission that the agent went on
        # without access, which the previous wording did.
        assert "say plainly if you proceeded without access" in text


def test_the_access_advice_names_only_tools_the_caller_has() -> None:
    """A reader must never be sent to a tool it does not have (round 1, BLOCKER).

    The browser text told subagents to notify through ``ask`` (no child has it)
    and to "ask the user directly" on a deny. The channel is now read off the
    declared ``ask_user`` capability, and the unattached paragraph — whose advice
    is "do not block the turn" — no longer offers ``ask`` at all, because ``ask``
    parks the turn for hours (round 1, D4/U4).
    """
    child = builtin._access_result_text(
        "pending", "https://example.com", host="", notify=ASK_PHRASE
    )
    assert "or `ask`" in child  # the phrase a host with the hook is given

    subagent = builtin._access_result_text(
        "pending", "https://example.com", host="", notify=builtin._notify_channel(ToolContext())
    )
    assert "`ask`" not in subagent
    assert "a short message" in subagent

    denied = builtin._access_result_text(
        "denied", "https://example.com", notify=builtin._notify_channel(ToolContext())
    )
    assert "the operator denied access" in denied
    assert "ask the user directly" not in denied

    # A context with the hook is the only one offered `ask`.
    with_hook = builtin._notify_channel(ToolContext(ask_user=_never_asked))
    assert "`ask`" in with_hook
    assert "`ask`" not in builtin._notify_channel(ToolContext())


@pytest.mark.asyncio
async def test_the_await_timeout_names_re_request_not_an_endless_await(monkeypatch) -> None:
    """The timeout arm must name the re-request, not just another await.

    The retired text ended "then call await_access again", which invites an
    unbounded retry loop against a prompt that has already expired.
    """
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        assert action == "await_access"
        return {"origin": "https://example.com", "state": "pending"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    result = await builtin.execute_browser(
        "t",
        {"action": "await_access", "url": "https://example.com", "timeout_s": 0.05},
        None,
        None,
        ToolContext(browser=BrowserSurface()),
    )

    assert "still pending after" in result.text
    assert "action='request_access'" in result.text
    assert "15 MINUTES" in result.text
    assert "AN UNANSWERED PROMPT IS NOT A REFUSAL" in result.text
    # The timeout arm and the pending arm now agree (round 1, U4): both name the
    # repeated ``await_access`` calls as the budget's mechanism. This context has
    # no probe, and no probe reads as ATTACHED (the fail-open default), so the arm
    # it renders is the attached one.
    assert "action='await_access'" in result.text
    assert "An interface is attached to this session" in result.text

    # ...and the arm IS attachment-aware, where it used to tell every caller to
    # keep waiting.
    detached = ToolContext(browser=BrowserSurface(), attached_probe=lambda: False)
    result = await builtin.execute_browser(
        "t",
        {"action": "await_access", "url": "https://example.com", "timeout_s": 0.05},
        None,
        None,
        detached,
    )
    assert "No pane is attached to this session" in result.text
    # The line is hard-wrapped for the TUI receipt, so the assertion stops at the
    # line break.
    assert "notify the operator and proceed" in result.text


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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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
async def test_browser_session_label_is_sanitized_and_host_derived(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    seen: list[dict[str, Any]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        seen.append(dict(params))
        return {"tab": "bridge:5:n0nce", "url": "https://example.com/", "title": "x"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(
        session_id="secret-session-uuid",
        session_name="  Q1\u202e launch\u200b   notes " + "🙂" * 40,
        browser=BrowserSurface(),
    )
    result = await builtin.execute_browser(
        "call-1",
        {
            "action": "open",
            "url": "https://example.com",
            # BrowserParams forbids these model-controlled overrides entirely.
        },
        None,
        None,
        context,
    )
    assert not result.is_error
    assert seen[0]["requester"] == "session:secret-session-uuid"
    assert seen[0]["session_label"] == "Q1 launch notes…"
    assert "secret-session-uuid" not in seen[0]["session_label"]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", "Session"),
        ("\x00\x7f\x85\u202d\u2066\ufeff", "Session"),
        ("  one\t two\nthree  ", "one two three"),
        # 29 clusters kept plus the ellipsis: the 30-cluster cap is inclusive
        # of the ellipsis the clipper appends, so the RESULT never exceeds it.
        ("e\u0301" * 31, "e\u0301" * 29 + "…"),
    ],
)
def test_browser_session_label_edge_cases(name: str, expected: str) -> None:
    assert builtin._browser_session_label(ToolContext(session_name=name)) == expected


@pytest.mark.asyncio
async def test_rename_changes_label_without_changing_requester(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    seen: list[tuple[str, str]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        seen.append((str(params["requester"]), str(params["session_label"])))
        return {"origin": "https://example.com", "state": "pending"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    context = ToolContext(session_id="same", session_name="Before", browser=BrowserSurface())
    await builtin.execute_browser(
        "t1", {"action": "request_access", "url": "https://example.com"}, None, None, context
    )
    context.session_name = "After"
    await builtin.execute_browser(
        "t2", {"action": "request_access", "url": "https://example.com"}, None, None, context
    )
    assert seen == [("session:same", "Before"), ("session:same", "After")]


@pytest.mark.asyncio
async def test_parallel_contexts_present_distinct_identities(monkeypatch) -> None:
    """Round-2 M4: two ToolContexts (two sessions) must reach the wire as two
    different requesters — the property the extension's fail-closed grant
    check depends on."""
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_available", lambda: True)
    seen: list[str] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
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


# --- The `LO · Session` tab-group bug ------------------------------------
#
# Three concurrent sessions all showed `LO · Session`, `LO · Session (2)` and
# `LO · Session (3)`: the ordinal de-duplication working correctly on top of an
# empty session name. Two independent defects fed it, and both are pinned here.


def test_unnamed_session_labels_itself_by_cwd_not_a_bare_fallback() -> None:
    # Defect 1a: every unnamed session sent the same bare `Session`, so the
    # ordinal was the only thing distinguishing three groups and it named
    # nothing. The cwd basename is the same substitution the TUI's band and
    # terminal title already make in this slot (`lo › <cwd>`).
    assert builtin._browser_session_label(ToolContext(cwd="/Users/x/minervaai")) == "minervaai"
    # A real title still wins over the directory.
    assert (
        builtin._browser_session_label(
            ToolContext(cwd="/Users/x/minervaai", session_name="Fix the tab groups")
        )
        == "Fix the tab groups"
    )
    # A filesystem root has no basename worth showing, so the bare fallback
    # survives exactly there — `LO · /` names a session no better than
    # `LO · Session` does.
    assert builtin._browser_session_label(ToolContext(cwd="/")) == "Session"
    # The cwd goes through the same sanitizer/clipper as a title, ellipsis
    # included in the 30-cluster budget.
    assert builtin._browser_session_label(ToolContext(cwd="/tmp/" + "d" * 40)) == "d" * 29 + "…"


def test_live_session_name_beats_the_per_turn_snapshot() -> None:
    # Defect 1b (the latch race): ToolContext is a SNAPSHOT built once per turn,
    # while the naming errand lands a second or two INTO the first turn. A
    # browse in that turn read the empty name the context was built with even
    # after the title existed, and the group latched it for the tab's life.
    context = ToolContext(cwd="/Users/x/proj", session_name_provider=lambda: "Named mid-turn")
    assert builtin._browser_session_label(context) == "Named mid-turn"

    # Still empty at call time: fall through to the cwd, not to the provider's
    # empty string.
    blank = ToolContext(cwd="/Users/x/proj", session_name_provider=lambda: "")
    assert builtin._browser_session_label(blank) == "proj"


def test_live_session_name_provider_never_breaks_a_browse() -> None:
    # A host callback is best-effort: grouping is presentation, so a provider
    # that raises must degrade to the snapshot rather than fail the command.
    def boom() -> str:
        raise RuntimeError("host went away")

    context = ToolContext(cwd="/Users/x/proj", session_name="Snapshot", session_name_provider=boom)
    assert builtin._browser_session_label(context) == "Snapshot"


@pytest.mark.asyncio
async def test_retitle_pushes_the_late_title_with_trusted_identity(monkeypatch) -> None:
    # Defect 2: nothing propagated a title that arrived AFTER the tab existed.
    # Every ordinary command reconciles the group as a side effect, but an
    # open -> screenshot -> close session issues none, so its group kept the
    # open-time label forever. The session pushes it explicitly instead.
    seen: list[tuple[str, dict[str, Any]]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        seen.append((action, dict(params)))
        return {"title": "LO · Named later"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:5:n0nce"
    await builtin.retitle_browser_surface(
        surface, ToolContext(session_id="sess-7", session_name="Named later")
    )
    assert len(seen) == 1
    action, params = seen[0]
    assert action == "retitle"
    assert params["tab"] == "bridge:5:n0nce"
    assert params["session_label"] == "Named later"
    # The identity boundary is the host's, not the model's: same trusted
    # requester every other command carries.
    assert params["requester"] == "session:sess-7"


@pytest.mark.asyncio
async def test_retitle_is_skipped_without_a_bridge_surface(monkeypatch) -> None:
    # No tab, or a cmux tab (which has no group chrome to rename): nothing on
    # the wire at all, so a non-browsing session pays nothing for this.
    calls: list[str] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        calls.append(action)
        return {}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    for surface_id in ("", "cmux:1"):
        surface = BrowserSurface()
        surface.surface_id = surface_id
        await builtin.retitle_browser_surface(surface, ToolContext(session_id="s"))
    assert calls == []


@pytest.mark.asyncio
async def test_retitle_swallows_a_bridge_failure(monkeypatch) -> None:
    # Renaming tab chrome must never raise into the caller: this runs off a
    # title landing, and a title must not be able to cost a turn.
    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        return None, builtin._error(tool_call_id, "browser", "extension not connected")

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:5:n0nce"
    await builtin.retitle_browser_surface(surface, ToolContext(session_id="s"))


def test_retitle_is_a_wire_method_but_not_a_model_action() -> None:
    # The session pushes the rename; the model has no business renaming browser
    # chrome, and advertising it would tax every request's schema for a
    # capability no agent should hold. The asymmetry is deliberate.
    from local_operator.browser_bridge.protocol import COMMAND_TIMEOUTS, METHODS

    assert "retitle" in METHODS
    assert "retitle" in COMMAND_TIMEOUTS
    assert "retitle" not in builtin.BROWSER_ACTIONS


@pytest.mark.asyncio
async def test_retitle_against_an_old_extension_is_swallowed_end_to_end(monkeypatch) -> None:
    """Mixed-version guard: a NEW runtime pushing ``retitle`` at an OLD extension.

    ``retitle`` ships in a runtime release before every paired extension has
    been updated, so the very first push a user's session makes may land on a
    worker that has no such handler. That worker answers with a typed INTERNAL
    ``unknown method: retitle``, and the whole point of the design is that this
    costs nothing — no raise, no failed turn, no lost title, and the tab stays
    drivable.

    Deliberately NOT a monkeypatched ``_bridge_call``: this drives a REAL
    loopback HTTP daemon speaking the wire protocol, so the client transport,
    the timeout table, the ``Response`` envelope parsing and the ``BridgeError``
    → ``format_error`` mapping are all genuinely exercised. Mocking the client
    would assert only that the mock was called (review round 1, R5).
    """
    import json as _json
    import secrets
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen: list[dict[str, Any]] = []

    class OldExtensionDaemon(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
            body = self.rfile.read(int(self.headers["Content-Length"]))
            request = _json.loads(body)
            seen.append(request)
            # Byte-for-byte what worker.ts emits for an unknown method: a typed
            # INTERNAL error, not a transport failure.
            payload = _json.dumps(
                {
                    "id": request["id"],
                    "ok": False,
                    "error": {
                        "code": "internal",
                        "message": f"unknown method {request['method']}",
                        "data": {},
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            """Silence the stdlib access log so the suite stays readable.

            The parameter is named ``format`` to match
            ``BaseHTTPRequestHandler``'s signature, which pyright checks as a
            keyword parameter; shadowing the builtin is the base class's choice,
            not ours.
            """

    server = HTTPServer(("127.0.0.1", 0), OldExtensionDaemon)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from local_operator.browser_bridge import state as state_store
        from local_operator.browser_bridge.state import BridgeState

        live = BridgeState(
            pid=os.getpid(),
            port=server.server_address[1],
            session_key=secrets.token_hex(24),
            proto=1,
            extension_connected=True,
            paired=True,
        )
        # The store's `read` is addressed by namespace now (the primitives were
        # generalised so the desktop app's browser host could share them), so the
        # stub takes the keywords rather than pinning a signature that no longer
        # exists — it still answers every read the client makes.
        monkeypatch.setattr(state_store, "read", lambda root=None, **_kwargs: live)

        surface = BrowserSurface()
        surface.surface_id = "bridge:5:n0nce"
        context = ToolContext(session_id="mixed-version", session_name="A late title")

        # Must not raise, and must return None rather than a ToolResult.
        assert await builtin.retitle_browser_surface(surface, context) is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    # The push really did reach the wire (so this is not vacuously passing).
    assert [request["method"] for request in seen] == ["retitle"]
    assert seen[0]["params"]["requester"] == "session:mixed-version"
    # The handle is untouched: a rejected rename must never drop the surface,
    # which is what would strand the tab for every later command.
    assert surface.surface_id == "bridge:5:n0nce"


@pytest.mark.parametrize(
    "invisible",
    ["\u200b", "\ufeff", "\u2060", "\u200e", "\x01", "\u202e\u200b"],
    ids=["zwsp", "bom", "word-joiner", "lrm", "c0", "bidi+zwsp"],
)
def test_a_title_of_only_invisible_characters_still_reaches_the_cwd_fallback(
    invisible: str,
) -> None:
    """A name the user cannot see must not beat the cwd fallback.

    ``str.strip()`` removes whitespace but NOT the Cf/Cc classes, so a title of
    nothing but zero-width or bidi characters used to read as a real name,
    survive to the sanitiser, empty there, and land on the bare ``Session`` —
    skipping the very fallback this feature adds, and reproducing the original
    "every group is called Session" symptom for a session that has a perfectly
    good working directory (QA round 1, Q2).
    """
    context = ToolContext(
        session_id="s",
        cwd="/Users/damian/minervaai",
        session_name="",
        session_name_provider=lambda: invisible,
    )
    assert builtin._browser_session_label(context) == "minervaai"


def test_an_invisible_title_at_a_filesystem_root_still_yields_the_bare_fallback() -> None:
    # Both candidates sanitise to nothing, so the last-resort label is correct
    # here — `LO · /` would name a session no better than `LO · Session`.
    context = ToolContext(session_id="s", cwd="/", session_name="\u200b")
    assert builtin._browser_session_label(context) == "Session"


@pytest.mark.asyncio
async def test_retitle_declines_rather_than_sending_a_constant_identity(monkeypatch) -> None:
    """No session id ⇒ no push, instead of a shared ``call:retitle`` requester.

    ``_browser_requester``'s fallback mints ``call:<tool_call_id>``, which is
    unique only because a tool call id is. A rename is not a tool call, so the
    fallback would put the CONSTANT ``call:retitle`` into an identity slot whose
    documented purpose is to be distinct per session — harmless while
    ``trustedOwner`` rejects non-``session:`` requesters, and a trap for whoever
    relaxes that next (review round 1, R3).
    """
    calls: list[str] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        calls.append(str(params.get("requester", "")))
        return {}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    surface = BrowserSurface()
    surface.surface_id = "bridge:5:n0nce"

    await builtin.retitle_browser_surface(surface, ToolContext(session_name="Named"))
    assert calls == [], "a session with no id must not push a rename at all"

    await builtin.retitle_browser_surface(surface, ToolContext(session_id="real", session_name="N"))
    assert calls == ["session:real"]


# ---------------------------------------------------------------------------
# Subagent tab groups: a delegated child has no title and can never grow one.
# ---------------------------------------------------------------------------
# Reported as `LO · Session` on the operator's screen. The cwd substitution
# above fixed that for TOP-LEVEL sessions, but a subagent's context carries its
# PARENT's cwd, so every child of one parent derived the identical label and a
# fleet of them rendered as `local-operator (2)`/`(3)`/… — distinct only by an
# ordinal that names nothing. Title generation lives in the TUI host and the
# owned-session runtime; a one-shot child passes through neither, so it has no
# title of its own and never will (no subagent session directory on disk holds
# a `title.json`). Its identity is the label its parent launched it under.


def test_a_subagent_is_named_by_its_parent_and_its_own_job_label() -> None:
    context = ToolContext(
        session_id="child-1",
        job_id="job-1",
        job_label="zoom-scroll-fix",
        # The PARENT's cwd and the parent's title: a child has neither of its
        # own, which is the whole reason both halves are needed.
        cwd="/Users/damian/local-operator",
        # Short enough that both halves fit whole — the clipping behaviour when
        # they do not has its own test below.
        session_name_provider=lambda: "Tab groups",
    )
    assert builtin._browser_session_label(context) == "Tab groups › zoom-scroll-fix"


def test_two_siblings_of_one_parent_do_not_collide() -> None:
    # The failure the ordinal was papering over: same parent, same cwd, so
    # before the label was carried these two produced identical pills.
    def child(label: str) -> str:
        return builtin._browser_session_label(
            ToolContext(
                session_id=f"child-{label}",
                job_id=f"job-{label}",
                job_label=label,
                cwd="/Users/damian/local-operator",
                session_name_provider=lambda: "Fix the tab groups",
            )
        )

    assert child("bridge-qa") != child("tabgroup-naming")


def test_job_id_not_job_label_is_what_marks_a_context_as_a_child() -> None:
    # A top-level session must never be composed as a subagent. `job_id` is set
    # by the harness on exactly the child contexts; `job_label` is display text
    # that a host could plausibly leave empty.
    parent = ToolContext(session_id="s", cwd="/Users/damian/local-operator", job_id=None)
    assert builtin._browser_session_label(parent) == "local-operator"


def test_a_child_of_an_unnamed_parent_still_beats_the_bare_cwd() -> None:
    # The parent has no title yet (it is named a second or two into its first
    # turn, and children are launched later). The cwd stands in for the parent
    # half, but the child's label is what makes the pill distinguishable.
    context = ToolContext(
        session_id="child-1",
        job_id="job-1",
        job_label="bridge-qa",
        cwd="/Users/damian/local-operator",
        session_name_provider=lambda: "",
    )
    assert builtin._browser_session_label(context) == "local-operator › bridge-qa"


def test_a_child_with_no_usable_label_falls_back_to_its_parents_name() -> None:
    # A label that sanitises away must not drop the child to the bare fallback.
    context = ToolContext(
        session_id="child-1",
        job_id="job-1",
        job_label="\u200b\u202e",
        cwd="/Users/damian/local-operator",
        session_name_provider=lambda: "Fix the tab groups",
    )
    assert builtin._browser_session_label(context) == "Fix the tab groups"


def test_a_renamed_parent_reaches_a_running_childs_next_command() -> None:
    # The rename case, from the child's side: the parent's holder is SHARED
    # with the child, so a title generated or `/rename`d after the child was
    # launched is picked up by its next reconcile rather than latching the
    # launch-time value.
    name = {"text": ""}
    context = ToolContext(
        session_id="child-1",
        job_id="job-1",
        job_label="qa",
        cwd="/Users/damian/proj",
        session_name_provider=lambda: name["text"],
    )
    assert builtin._browser_session_label(context) == "proj › qa"
    name["text"] = "Fix the tab groups"
    assert builtin._browser_session_label(context) == "Fix the tab groups › qa"


def test_the_childs_label_survives_a_parent_title_too_long_to_fit() -> None:
    # Clipping the composed string as ONE unit loses the distinguishing half:
    # measured on a real pair, `Fix Slack-reported UI zoom and overlap bugs` +
    # `zoom-scroll-fix` clipped to `Fix Slack-reported UI zoom…` and dropped
    # the label entirely, re-creating the collision. The parent absorbs the cut.
    label = builtin._browser_session_label(
        ToolContext(
            session_id="child-1",
            job_id="job-1",
            job_label="zoom-scroll-fix",
            session_name_provider=lambda: "Fix Slack-reported UI zoom and overlap bugs",
        )
    )
    assert label.endswith("› zoom-scroll-fix")
    # And the composition still respects the pill budget the extension clips at
    # (the ellipsis the clipper appends is paid out of the parent's room).
    assert len(builtin._browser_clusters(label)) <= builtin._BROWSER_SESSION_LABEL_CLUSTERS


def test_a_label_that_fills_the_pill_alone_drops_the_parent_rather_than_a_stub() -> None:
    # `Fi…` identifies no conversation and only steals room from the half that
    # does, so below the minimum the label stands alone.
    label = builtin._browser_session_label(
        ToolContext(
            session_id="child-1",
            job_id="job-1",
            job_label="an-enormous-child-label-that-fills-it",
            session_name_provider=lambda: "Some Parent Conversation",
        )
    )
    assert "›" not in label
    assert label.startswith("an-enormous-child-label")


def test_a_clipped_label_respects_the_documented_cluster_cap() -> None:
    # The clip counts the ellipsis it appends. It used to keep `budget`
    # clusters and then add one, so every label it touched came back at 31
    # against a documented 30 — harmless (the extension re-clips) but it made
    # the ceiling this module states false, on the plain-title path as much as
    # on the composed one.
    cap = builtin._BROWSER_SESSION_LABEL_CLUSTERS
    for text in (
        "d" * 40,
        "Fix Slack-reported UI zoom and overlap bugs today",
        "e\u0301" * 40,
    ):
        clipped = builtin._browser_clip_label(text)
        assert len(builtin._browser_clusters(clipped)) <= cap, clipped


def test_the_composed_subagent_pill_never_exceeds_the_cap() -> None:
    # Q2's case: parent + separator + label, at the length where the parent is
    # clipped and the ellipsis has to be paid for out of its room.
    for label in ("zoom-scroll-fix", "a-really-very-long-subagent-job-label", "qa"):
        pill = builtin._browser_session_label(
            ToolContext(
                session_id="child-1",
                job_id="job-1",
                job_label=label,
                session_name_provider=lambda: "Fix Slack-reported UI zoom and overlap bugs",
            )
        )
        assert len(builtin._browser_clusters(pill)) <= builtin._BROWSER_SESSION_LABEL_CLUSTERS, pill


def test_a_server_session_carrying_a_job_id_keeps_its_own_title() -> None:
    # `job_id` is NOT unique to subagents — a server-side session carries one
    # too — so the composition requires a `job_label` as well. A server session
    # must render its own conversation title, unchanged.
    context = ToolContext(
        session_id="server-1",
        job_id="queued-job",
        job_label="",
        cwd="/Users/damian/local-operator",
        session_name_provider=lambda: "Reduce agent RAM usage",
    )
    assert builtin._browser_session_label(context) == "Reduce agent RAM usage"


def test_a_peer_message_from_a_child_is_not_signed_with_its_parents_name() -> None:
    # The `send` fallback path reads the ToolContext when no registry record
    # names the process. `session_name` on a child resolves to its PARENT's
    # title, so the card presented one session's name beside another's id.
    child = ToolContext(
        session_id="child-1",
        job_id="job-1",
        job_label="bridge-qa",
        session_name="Fix the tab groups",
    )
    assert builtin._peer_sender_conversation_name(child) == "Fix the tab groups › bridge-qa"

    # A top-level session is untouched, and so is a server session (job_id set,
    # no label): both sign with their own title.
    top = ToolContext(session_id="s", session_name="Fix the tab groups")
    assert builtin._peer_sender_conversation_name(top) == "Fix the tab groups"
    server = ToolContext(session_id="s", job_id="queued-job", session_name="Reduce agent RAM")
    assert builtin._peer_sender_conversation_name(server) == "Reduce agent RAM"

    # An unnamed parent leaves the child its own label rather than a bare
    # separator, and the peer card is NOT clipped to the browser pill's width.
    orphan = ToolContext(session_id="child-2", job_id="job-2", job_label="bridge-qa")
    assert builtin._peer_sender_conversation_name(orphan) == "bridge-qa"
    long_parent = ToolContext(
        session_id="child-3",
        job_id="job-3",
        job_label="zoom-scroll-fix",
        session_name="Fix Slack-reported UI zoom and overlap bugs",
    )
    assert (
        builtin._peer_sender_conversation_name(long_parent)
        == "Fix Slack-reported UI zoom and overlap bugs › zoom-scroll-fix"
    )


# ---------------------------------------------------------------------------
# Three-way host selection, over all eight availability combinations.
#
# The set is small enough to enumerate exhaustively, which is the point: a
# three-way branch's failure mode is a combination nobody tried, and the two
# hosts share one dispatch path precisely so that a combination cannot be handled
# correctly for one and wrongly for the other.
# ---------------------------------------------------------------------------


def _availability(monkeypatch, *, cmux: bool, bridge: bool, ui: bool) -> None:
    """Pin all THREE hosts' answers, and both liveness readings.

    The liveness readings are pinned too (to "no record") so the demotion hints
    and the bridge-absent rescue cannot depend on whatever discovery files exist
    on the machine running the suite — a test whose answer changes with the
    operator's live daemon is not a test.
    """

    async def bridge_reachable(classified: tuple[Any, Any] | None = None) -> bool:
        return bridge

    async def ui_reachable(classified: tuple[Any, Any] | None = None) -> bool:
        return ui

    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: cmux)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", bridge_reachable)
    monkeypatch.setattr(builtin, "ui_browser_reachable", ui_reachable)
    monkeypatch.setattr(builtin, "ui_browser_available", lambda: ui)
    monkeypatch.setattr(builtin, "_bridge_liveness", lambda: (None, None))
    monkeypatch.setattr(builtin, "_ui_liveness", lambda: (None, None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cmux", "bridge", "ui", "expected"),
    [
        (True, True, True, "ui"),
        (True, True, False, "bridge"),
        (True, False, True, "ui"),
        (False, True, True, "ui"),
        (True, False, False, "cmux"),
        (False, True, False, "bridge"),
        (False, False, True, "ui"),
        (False, False, False, None),
    ],
    ids=[
        "all-three",
        "no-ui",
        "no-bridge",
        "no-cmux",
        "cmux-only",
        "bridge-only",
        "ui-only",
        "none",
    ],
)
async def test_fresh_open_precedence_matrix(
    monkeypatch, cmux: bool, bridge: bool, ui: bool, expected: str | None
) -> None:
    _availability(monkeypatch, cmux=cmux, bridge=bridge, ui=ui)
    chosen: list[str] = []

    async def fake_bridge_open(tool_call_id, state, url, context=None, *, client=None, adopt=""):
        # `adopt` is part of the shipped signature (the open-tab handover), so the
        # double takes it even though these cases do not exercise adoption: a
        # double that lags the function it stands in for fails on the call rather
        # than on the behaviour under test.
        chosen.append(builtin._host_of_client(client))
        return builtin._text("t", "browser", "non-cmux")

    async def fake_cmux_open(tool_call_id, state, url):
        chosen.append("cmux")
        return builtin._text("t", "browser", "cmux")

    monkeypatch.setattr(builtin, "_bridge_open", fake_bridge_open)
    monkeypatch.setattr(builtin, "_browser_open", fake_cmux_open)
    context = ToolContext(browser=BrowserSurface())
    result = await builtin.execute_browser(
        "t", {"action": "open", "url": "https://example.com"}, None, None, context
    )
    if expected is None:
        assert chosen == [], "a host was used with none reachable"
        assert result.is_error and "browser not available" in result.text
        # The no-host copy is three-way: it names the desktop app, the extension
        # AND cmux, and it does not send an app user to `lop browser install`.
        assert "desktop app" in result.text and "browser extension" in result.text
        assert "cmux" in result.text
    else:
        assert chosen == [expected], result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("ui:4:aaaaaaaabbbbccccddddeeeeffff0000", "ui"),
        ("bridge:4:aaaaaaaabbbbccccddddeeeeffff0000", "bridge"),
        ("surface:4", "cmux"),
    ],
)
async def test_a_pinned_handle_overrides_the_precedence(
    monkeypatch, surface: str, expected: str
) -> None:
    """A prefixed handle pins the transport for the surface's whole life.

    Checked with ALL THREE hosts reachable, so the pin is doing the work rather
    than the precedence happening to agree with it.
    """
    _availability(monkeypatch, cmux=True, bridge=True, ui=True)
    chosen: list[str] = []

    async def fake_bridge_open(tool_call_id, state, url, context=None, *, client=None, adopt=""):
        # `adopt` is part of the shipped signature (the open-tab handover), so the
        # double takes it even though these cases do not exercise adoption: a
        # double that lags the function it stands in for fails on the call rather
        # than on the behaviour under test.
        chosen.append(builtin._host_of_client(client))
        return builtin._text("t", "browser", "non-cmux")

    async def fake_cmux_open(tool_call_id, state, url):
        chosen.append("cmux")
        return builtin._text("t", "browser", "cmux")

    monkeypatch.setattr(builtin, "_bridge_open", fake_bridge_open)
    monkeypatch.setattr(builtin, "_browser_open", fake_cmux_open)
    holder = BrowserSurface()
    holder.surface_id = surface
    result = await builtin.execute_browser(
        "t",
        {"action": "open", "url": "https://example.com"},
        None,
        None,
        ToolContext(browser=holder),
    )
    assert chosen == [expected], result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bridge", "ui", "named"),
    [
        (False, False, ("desktop app", "browser extension")),
        (True, False, ("browser extension",)),
        (False, True, ("desktop app",)),
        (True, True, ("desktop app", "browser extension")),
    ],
    ids=["neither", "bridge-only", "ui-only", "both"],
)
async def test_the_cmux_degrade_names_the_hosts_that_can_serve_it(
    monkeypatch, bridge: bool, ui: bool, named: tuple[str, ...]
) -> None:
    """`scroll`/`logs`/`tabs` on a cmux surface, with each non-cmux shape.

    The old copy asserted the EXTENSION was the only alternative, which on a host
    with the desktop app running sent the user to install a bridge they did not
    need.
    """
    _availability(monkeypatch, cmux=True, bridge=bridge, ui=ui)
    holder = BrowserSurface()
    holder.surface_id = "surface:3"
    for action in ("scroll", "logs", "tabs"):
        result = await builtin.execute_browser(
            "t", {"action": action}, None, None, ToolContext(browser=holder)
        )
        assert result.is_error and "not supported on the cmux backend" in result.text
        for fragment in named:
            assert fragment in result.text, result.text
        # The one thing the copy must never do again: name a host that cannot
        # serve this and imply it is the only one.
        assert "only exists for the Local Operator browser extension" not in result.text


@pytest.mark.asyncio
async def test_a_non_cmux_surface_routes_to_its_own_host(monkeypatch) -> None:
    """A pinned `ui:` handle must reach the UI client, not the bridge's."""
    _availability(monkeypatch, cmux=True, bridge=True, ui=True)
    seen: list[tuple[str, str]] = []

    async def fake_call(tool_call_id, action, params, *, surface="", client=None):
        seen.append((action, builtin._host_of_client(client)))
        return {"url": "https://example.com/", "title": "Example"}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_call)
    holder = BrowserSurface()
    holder.surface_id = "ui:9:aaaaaaaabbbbccccddddeeeeffff0000"
    result = await builtin.execute_browser(
        "t",
        {"action": "goto", "url": "https://example.com"},
        None,
        None,
        ToolContext(browser=holder),
    )
    assert result.is_error is False, result.text
    assert seen == [("goto", "ui")]
