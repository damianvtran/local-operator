"""``hover`` (issue #3248): the harness half of trusted pointer movement.

No native input is synthesized here. These tests drive the real tool path
(``execute_browser`` / ``_bridge_action`` / ``_bridge_call``) against a fake
desktop-host peer and assert what that peer is *sent* and how its answer is
rendered, plus the typed refusals for hosts that cannot serve it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_operator.browser_bridge.backend import HostCapabilities
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    DESKTOP_ONLY_METHODS,
    METHODS,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin

# The REAL host shape (verbatim from local-operator-ui#1602's `hover #row-1`
# capture): `hit` is the whole hit_test answer, topmost element first. An earlier
# revision of this fake sent a flat descriptor the host never produces, which
# kept the tests green while every real reply said "element under the pointer: ?".
HIT = {
    "count": 2,
    "elements": [
        {
            "tag": "li",
            "id": "row-1",
            "role": "",
            "className": "row",
            "rect": {"x": 16, "y": 147.02, "width": 520, "height": 49.5},
            "styles": {"display": "flex", "position": "static"},
        },
        {"tag": "ul", "id": "rows", "role": "", "className": "list"},
    ],
}


class App:
    host = "ui"

    def __init__(self, methods: tuple[str, ...] = ("hover",), host: str = "ui") -> None:
        self.methods = methods
        self.host = host
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(methods=self.methods, version="0.0.0", capabilities_known=True)

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        # Identity fields (requester/session_label) are stamped on every frame by
        # _bridge_action; the contract under test is the target payload.
        self.sent.append(
            (method, {k: v for k, v in params.items() if k not in ("requester", "session_label")})
        )
        return {
            "url": "https://example.test/",
            "title": "Menu",
            "x": 120.5,
            "y": 40,
            "hit": HIT,
            "hovered": True,
        }


def owned_surface() -> BrowserSurface:
    state = BrowserSurface()
    state.surface_id = "ui:7:synthetic"
    return state


async def hover(app: App, **fields: Any) -> Any:
    return await builtin._bridge_action(
        "t",
        owned_surface(),
        "hover",
        builtin.BrowserParams(action="hover", **fields),
        None,
        client=app,
    )


def test_hover_is_registered_everywhere_a_desktop_only_method_must_be() -> None:
    assert "hover" in METHODS
    assert "hover" in CAPABILITY_GATED_METHODS
    assert "hover" in DESKTOP_ONLY_METHODS
    assert COMMAND_TIMEOUTS["hover"] == 20.0
    assert "hover" in builtin.BROWSER_ACTIONS
    assert "hover" in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS
    offset = METHODS.index("press")
    assert METHODS[offset : offset + 4] == ("press", "hover", "select", "close")


@pytest.mark.asyncio
async def test_selector_hover_sends_only_the_selector_and_names_the_hit() -> None:
    app = App()
    result = await hover(app, selector="  #menu ")
    assert not result.is_error, result.text
    assert app.sent == [("hover", {"tab": "ui:7:synthetic", "selector": "#menu"})]
    assert result.text.startswith("Hovered #menu at (120.5, 40)")
    # Regression (QA Q1 / review R1-1): the topmost of the host's real
    # {count, elements} answer is named, never the "?" fallback.
    assert "element under the pointer: li#row-1.row." in result.text
    assert "pointer: ?" not in result.text
    assert "no :hover" not in result.text
    assert (result.details or {})["hovered"] is True
    assert "Page: " in result.text
    assert (result.details or {})["hit"] == HIT
    assert (result.details or {})["x"] == 120.5


@pytest.mark.asyncio
async def test_coordinate_hover_sends_x_and_y_only() -> None:
    app = App()
    result = await hover(app, x=10, y=0)
    assert not result.is_error, result.text
    assert app.sent == [("hover", {"tab": "ui:7:synthetic", "x": 10.0, "y": 0.0})]
    # The point is stated once (QA Q4): the call already carried it.
    assert result.text.startswith("Hovered (10, 0); element under the pointer: li#row-1")
    assert "at (" not in result.text


@pytest.mark.asyncio
async def test_locator_hover_reports_what_the_host_resolved() -> None:
    class Resolving(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            payload = await super().call(method, params)
            payload["resolved"] = {"by": "locator", "role": "button", "name": "Menu", "matched": 2}
            return payload

    result = await hover(Resolving(), selector='role=button[name="Menu"]')
    assert not result.is_error, result.text
    assert 'button "Menu" (2 matched, first used)' in result.text


@pytest.mark.asyncio
async def test_hover_tolerates_a_host_that_omits_the_hit() -> None:
    class Bare(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.sent.append((method, params))
            return {"url": "https://example.test/", "title": "T"}

    result = await hover(Bare(), selector="#a")
    assert not result.is_error, result.text
    assert result.text.startswith("Hovered #a.")
    assert "element under the pointer" not in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fields", "needle"),
    [
        ({}, "needs a selector"),
        ({"selector": "#a", "x": 1, "y": 2}, "selector OR x and y"),
        ({"selector": "#a", "x": 1}, "selector OR x and y"),
        ({"x": 1}, "needs both x and y"),
        ({"y": 1}, "needs both x and y"),
        ({"x": -1, "y": 2}, "non-negative"),
        ({"x": 1, "y": float("nan")}, "finite"),
        ({"x": float("inf"), "y": 1}, "finite"),
        ({"selector": "--help"}, "flag-shaped selector"),
        ({"selector": "   "}, "needs a selector"),
    ],
)
@pytest.mark.parametrize("entrypoint", ["execute_browser", "_execute_browser"])
async def test_invalid_targets_fail_before_host_discovery(
    monkeypatch, fields: dict[str, Any], needle: str, entrypoint: str
) -> None:
    def unexpected() -> None:
        pytest.fail("invalid hover target reached host discovery")

    monkeypatch.setattr(builtin, "cmux_browser_available", unexpected)
    result = await getattr(builtin, entrypoint)("t", {"action": "hover", **fields})
    assert result.is_error
    assert needle in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("methods", [(), ("click", "type", "press")])
async def test_old_desktop_refusal_is_typed_actionable_and_never_sent(
    methods: tuple[str, ...],
) -> None:
    app = App(methods)
    result = await hover(app, selector="#menu")
    assert result.is_error
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert "update the desktop app, then retry" in result.text
    assert "'hover'" in result.text
    assert app.sent == []


@pytest.mark.asyncio
async def test_extension_host_refuses_hover_as_desktop_only() -> None:
    # The extension cannot deliver trusted pointer input on its inactive tab, so
    # the copy must send the caller to the desktop app, not to a nonexistent
    # extension update or switch.
    app = App((), host="extension")
    result = await hover(app, selector="#menu")
    assert result.is_error
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert "served only by the Local Operator desktop app's browser" in result.text
    assert app.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["hover", "press"])
async def test_cmux_surface_gets_the_typed_not_supported_error(monkeypatch, action: str) -> None:
    state = BrowserSurface()
    state.surface_id = "surface:3"  # a cmux handle: no `ui:`/`ext:` host prefix
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", AsyncMock(return_value=False))
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=False))
    args: dict[str, Any] = (
        {"action": action, "key": "Tab"}
        if action == "press"
        else {
            "action": action,
            "selector": "#a",
        }
    )
    result = await builtin._execute_browser("t", args, None, None, ToolContext(browser=state))
    assert result.is_error
    assert f"'{action}' is not supported on the cmux backend" in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised", [False, True])
async def test_public_tool_routes_hover_to_the_held_desktop_tab(
    monkeypatch, advertised: bool
) -> None:
    app = App(("hover",) if advertised else ("click", "press"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", AsyncMock(return_value=False))
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(builtin, "_client_for", lambda host: app)
    result = await builtin.execute_browser(
        "t",
        {"action": "hover", "selector": "#menu"},
        None,
        None,
        ToolContext(browser=owned_surface()),
    )
    if advertised:
        assert not result.is_error, result.text
        assert app.sent == [("hover", {"tab": "ui:7:synthetic", "selector": "#menu"})]
    else:
        assert result.is_error
        assert (result.details or {})["error_code"] == "capability_unsupported"
        assert app.sent == []


@pytest.mark.asyncio
async def test_host_invalid_arguments_internal_is_rendered_as_a_bad_argument() -> None:
    # The desktop host reports refused hover/press arguments as code `internal`
    # with data.reason == "invalid_arguments" (no new ErrorCode, which would hang
    # a released session). The model must read that as a bad argument, not as a
    # broken bridge.
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode

    class Refusing(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise BridgeError(
                ErrorCode.INTERNAL, "x must be inside the viewport", {"reason": "invalid_arguments"}
            )

    result = await hover(Refusing(), x=99999, y=1)
    assert result.is_error
    assert "rejected this call's arguments: x must be inside the viewport" in result.text
    assert (result.details or {})["error_code"] == "internal"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capabilities", "action_args", "expect_frame"),
    [
        (
            ["hover", "press", "press_chord"],
            {"action": "hover", "selector": "#menu"},
            {"method": "hover", "params": {"selector": "#menu"}},
        ),
        (
            ["hover", "press", "press_chord"],
            {"action": "press", "key": "ctrl+shift+p", "selector": "#q"},
            {"method": "press", "params": {"key": "Control+Shift+p", "selector": "#q"}},
        ),
        (
            ["press"],  # an older app: the original keys only, same frame as before
            {"action": "press", "key": "Shift+Tab"},
            {"method": "press", "params": {"key": "Shift+Tab"}},
        ),
        (["press"], {"action": "press", "key": "Control+K"}, None),
        (["press"], {"action": "hover", "selector": "#menu"}, None),
    ],
)
async def test_real_loopback_wire_capture_against_a_fake_desktop_peer(
    tmp_path,
    capabilities: list[str],
    action_args: dict[str, Any],
    expect_frame: dict[str, Any] | None,
) -> None:
    """The real tool -> real UiHostClient -> real HTTP path, peer faked.

    Only the desktop host is synthetic: the discovery record the capability gate
    reads, the HTTP transport, the auth headers and the JSON-RPC frame are the
    production ones. A gated call must put NOTHING on the wire.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from local_operator.ui_browser import backend as ui_backend
    from local_operator.ui_browser import state as ui_state

    frames: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, body: dict[str, Any]) -> None:
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802 - stdlib hook
            self._reply({"host": "ui", "pid": os.getpid(), "instance_id": "i1"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib hook
            frame = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            frames.append(frame)
            self._reply(
                {
                    "id": frame["id"],
                    "ok": True,
                    "result": {
                        "url": "https://example.test/",
                        "title": "T",
                        "x": 1,
                        "y": 2,
                        "hit": HIT,
                        "hovered": True,
                    },
                }
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            pass

    import os

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        ui_state.publish(
            ui_state.UiHostState.model_validate(
                {
                    "pid": os.getpid(),
                    "port": server.server_address[1],
                    "session_key": "k" * 48,
                    "proto": 1,
                    "app_version": "0.0.0",
                    "instance_id": "i1",
                    "capabilities": capabilities,
                }
            ),
            tmp_path,
        )
        client = ui_backend.UiHostClient(tmp_path)
        params = builtin.BrowserParams(**action_args)
        result = await builtin._bridge_action(
            "t", owned_surface(), params.action, params, None, client=client
        )
    finally:
        server.shutdown()
        server.server_close()
    if expect_frame is None:
        assert result.is_error
        assert (result.details or {})["error_code"] == "capability_unsupported"
        assert frames == []
        print("gated:", result.text)
        return
    assert not result.is_error, result.text
    if expect_frame["method"] == "hover":
        # The real {count, elements} reply must render through the real client.
        assert "element under the pointer: li#row-1.row" in result.text
        print("hover text:", result.text)
    assert len(frames) == 1
    frame = frames[0]
    assert frame["method"] == expect_frame["method"]
    sent = {k: v for k, v in frame["params"].items() if k in expect_frame["params"]}
    assert sent == expect_frame["params"]
    assert set(frame["params"]) <= {"tab", "requester", "session_label", *expect_frame["params"]}
    print("wire frame:", json.dumps(frame, sort_keys=True))


@pytest.mark.asyncio
async def test_flat_hit_descriptor_is_still_tolerated() -> None:
    class Flat(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            payload = await super().call(method, params)
            payload["hit"] = {"tag": "button", "id": "menu", "className": "btn"}
            return payload

    result = await hover(Flat(), selector="#menu")
    assert "element under the pointer: button#menu.btn" in result.text


@pytest.mark.asyncio
async def test_hovered_false_is_reported_not_swallowed() -> None:
    class Ignored(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            payload = await super().call(method, params)
            payload["hovered"] = False
            return payload

    result = await hover(Ignored(), selector="#menu")
    assert not result.is_error
    assert "the page shows no :hover on that element" in result.text
    assert (result.details or {})["hovered"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("host_message", "needle"),
    [
        ("Node does not have a layout object", "Hover the row or parent that reveals it first"),
        (
            "#irow is inside a frame; hover it by position with x and y (take them from hit_test)",
            "'hit_test' inside a frame only reports the frame element itself",
        ),
        ("1 frame(s) could not be entered", "A selector cannot cross an iframe boundary"),
    ],
)
async def test_hover_failures_carry_actionable_guidance(host_message: str, needle: str) -> None:
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode

    class Failing(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise BridgeError(ErrorCode.INTERNAL, host_message, {})

    result = await hover(Failing(), selector="#x")
    assert result.is_error
    assert needle in result.text
    assert "take them from hit_test" not in result.text


@pytest.mark.asyncio
async def test_hover_hint_is_not_added_to_other_actions() -> None:
    from local_operator.browser_bridge.backend import BridgeError
    from local_operator.browser_bridge.protocol import ErrorCode

    class Failing(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            raise BridgeError(ErrorCode.INTERNAL, "Node does not have a layout object", {})

    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "click",
        builtin.BrowserParams(action="click", selector="#x"),
        None,
        client=Failing(),
    )
    assert result.is_error
    assert "reveals it" not in result.text
