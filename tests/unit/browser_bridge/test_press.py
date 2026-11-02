"""Core keyboard contract; desktop implementation pinned to UI PR #1582.

Extended (issue #3248): ``press`` now takes a modifier chord and an optional
``selector``. The original eight keys keep their exact wire frames, and a host
that has not advertised ``press_chord`` is never sent anything wider.

No native input is synthesized here: these tests exercise tool validation,
capability admission and wire result/error decoding with synthetic owned tabs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from local_operator.browser_bridge.backend import BridgeError, HostCapabilities
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    DESKTOP_ONLY_METHODS,
    METHODS,
    ErrorCode,
    Response,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin

KEYS = (
    "Tab",
    "Shift+Tab",
    "Enter",
    "Escape",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
)


# Every shape the grammar must refuse. Whitespace is never trimmed, multi-character
# non-names are text entry, and a chord with a missing/duplicated part is malformed.
REFUSED_KEYS = [
    "",
    "tabb",
    "hello",
    "hello world",
    "Shift+Tab ",
    " Tab",
    "Control+Control+K",
    "Ctrl+Control+K",
    "Control+",
    "Control+Shift+",
    "+Tab",
    "++",
    "Control+hello",
    "Foo+K",
    "F13",
    "F0",
    "Control+K+L",
    "Enter,Enter",
    "Tab Tab",
    "\n",
    " ",
    "Control+ ",
]


class App:
    host = "ui"

    def __init__(self, methods: tuple[str, ...] = ("press", "press_chord")) -> None:
        self.methods = methods
        self.sent: list[tuple[str, dict[str, Any]]] = []

    def capabilities(self) -> HostCapabilities:
        return HostCapabilities(methods=self.methods, version="0.0.0", capabilities_known=True)

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.sent.append((method, params))
        return {"url": "https://example.test/", "title": "Keys", "key": params["key"]}


def owned_surface() -> BrowserSurface:
    state = BrowserSurface()
    state.surface_id = "ui:7:synthetic"
    return state


def test_press_protocol_matches_merged_desktop_order() -> None:
    # UI 023c58964e17:src/main/browser/protocol.ts (PR #1582) inserts
    # press at this exact position. No sibling checkout dependency in CI.
    offset = METHODS.index("click")
    assert METHODS[offset : offset + 5] == ("click", "type", "press", "hover", "close")
    assert "press" in CAPABILITY_GATED_METHODS
    assert "press" in DESKTOP_ONLY_METHODS
    assert COMMAND_TIMEOUTS["press"] == 25.0
    assert builtin._BROWSER_PRESS_LEGACY_KEYS == KEYS
    assert "press" in builtin.BROWSER_ACTIONS
    assert "press" in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS


@pytest.mark.asyncio
@pytest.mark.parametrize("key", REFUSED_KEYS)
@pytest.mark.parametrize("entrypoint", ["execute_browser", "_execute_browser"])
async def test_invalid_key_fails_before_host_discovery(
    monkeypatch, key: str, entrypoint: str
) -> None:
    def unexpected() -> None:
        pytest.fail("invalid keyboard input reached host discovery")

    monkeypatch.setattr(builtin, "cmux_browser_available", unexpected)
    result = await getattr(builtin, entrypoint)("t", {"action": "press", "key": key})
    assert result.is_error
    assert result.details == {"error_code": "unsupported_key", "key": key}
    assert repr(key) in result.text
    assert "'press' takes ONE chord" in result.text
    # Credentials must be steered to `type`, never typed through key presses.
    assert "use 'type'" in result.text and "secret_ref" in result.text


@pytest.mark.asyncio
async def test_internal_dispatch_also_refuses_invalid_key() -> None:
    app = App()
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key="hello"),
        None,
        client=app,
    )
    assert result.is_error
    assert result.details == {"error_code": "unsupported_key", "key": "hello"}
    assert "'hello'" in result.text
    assert app.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", KEYS)
async def test_each_allowed_key_reaches_owned_tab_as_one_command(key: str) -> None:
    app = App()
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key=key),
        None,
        client=app,
    )
    assert not result.is_error
    assert f"Pressed {key!r} in the owned page" in result.text
    assert len(app.sent) == 1
    method, wire = app.sent[0]
    assert method == "press"
    assert wire["key"] == key
    assert wire["tab"] == "ui:7:synthetic"
    assert "selector" not in wire and "text" not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("methods", [(), ("click", "type")])
async def test_old_desktop_refusal_is_typed_actionable_and_never_sent(
    methods: tuple[str, ...],
) -> None:
    app = App(methods)
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key="Tab"),
        None,
        client=app,
    )
    assert result.is_error
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert result.text == (
        "the Local Operator desktop app's browser host does not provide 'press' — "
        "the app that answered was built before this action existed. Please "
        "update the desktop app, then retry; every other browser action still works."
    )
    assert app.sent == []


@pytest.mark.asyncio
async def test_host_unsupported_key_survives_wire_decode_as_typed_error() -> None:
    class RefusingApp(App):
        async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.sent.append((method, params))
            # Same payload emitted by UI PR #1582 input.ts, not an invented
            # exception: Response decoding must recognize its new enum value.
            response = Response.model_validate(
                {
                    "id": "synthetic",
                    "ok": False,
                    "error": {
                        "code": "unsupported_key",
                        "message": "unsupported key: Ctrl+L",
                        "data": {"key": "Ctrl+L"},
                    },
                }
            )
            assert response.error is not None
            error = response.error
            raise BridgeError(error.code, error.message, error.data)

    app = RefusingApp()
    # A lower-level caller that bypassed tool validation still receives the
    # host's typed refusal. Normal browser calls are stopped before this wire.
    payload, result = await builtin._bridge_call(
        "t",
        "press",
        {"tab": "ui:7:synthetic", "key": "Ctrl+L"},
        surface="ui:7:synthetic",
        client=app,
    )
    assert payload is None and result is not None and result.is_error
    assert (result.details or {})["error_code"] == ErrorCode.UNSUPPORTED_KEY.value
    assert "unsupported key: Ctrl+L" in result.text
    assert len(app.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised", [False, True])
async def test_public_tool_routes_press_to_the_held_desktop_tab(
    monkeypatch, advertised: bool
) -> None:
    app = App(("press",) if advertised else ("click", "type"))
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: False)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", AsyncMock(return_value=False))
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=True))
    monkeypatch.setattr(builtin, "_client_for", lambda host: app)
    result = await builtin.execute_browser(
        "t",
        {"action": "press", "key": "Shift+Tab"},
        None,
        None,
        ToolContext(browser=owned_surface()),
    )
    if advertised:
        assert not result.is_error
        assert len(app.sent) == 1
        assert app.sent[0][0] == "press"
        assert app.sent[0][1]["key"] == "Shift+Tab"
        assert app.sent[0][1]["tab"] == "ui:7:synthetic"
    else:
        assert result.is_error
        assert (result.details or {})["error_code"] == "capability_unsupported"
        assert "update the desktop app, then retry" in result.text
        assert app.sent == []


@pytest.mark.parametrize("key", KEYS)
def test_original_eight_keys_canonicalise_to_themselves(key: str) -> None:
    # Back-compat: the pre-chord allowlist is a strict subset of the grammar and
    # must travel the wire byte-for-byte as before.
    assert builtin._canonical_press_chord(key) == (key, "")
    assert builtin._press_is_legacy_shape(key, "")
    assert not builtin._press_is_legacy_shape(key, "#field")


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("Control+K", "Control+K"),
        ("ctrl+k", "Control+k"),
        ("Cmd+Shift+P", "Shift+Meta+P"),
        ("command+option+Delete", "Alt+Meta+Delete"),
        ("Meta+Alt+Control+Shift+Enter", "Shift+Control+Alt+Meta+Enter"),
        ("e", "e"),
        ("E", "E"),
        ("/", "/"),
        ("+", "+"),
        ("Shift++", "Shift++"),
        ("space", "Space"),
        ("PAGEDOWN", "PageDown"),
        ("f5", "F5"),
        ("F12", "F12"),
        ("Alt+ArrowLeft", "Alt+ArrowLeft"),
        ("é", "é"),
    ],
)
def test_chords_are_canonicalised_by_the_harness(raw: str, canonical: str) -> None:
    assert builtin._canonical_press_chord(raw) == (canonical, "")


@pytest.mark.parametrize("key", REFUSED_KEYS)
def test_refusals_carry_a_reason(key: str) -> None:
    chord, reason = builtin._canonical_press_chord(key)
    assert chord == "" and reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "selector", "wire_key"),
    [
        ("ctrl+k", "", "Control+k"),
        ("Cmd+Shift+P", "", "Shift+Meta+P"),
        ("e", "#search", "e"),
        ("Tab", 'role=textbox[name="Search"]', "Tab"),
        ("space", "e5", "Space"),
    ],
)
async def test_extended_press_sends_canonical_key_and_selector(
    raw: str, selector: str, wire_key: str
) -> None:
    app = App()
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key=raw, selector=selector),
        None,
        client=app,
    )
    assert not result.is_error, result.text
    assert f"Pressed {wire_key!r} in the owned page" in result.text
    assert (result.details or {})["key"] == wire_key
    ((method, wire),) = app.sent
    assert method == "press"
    assert wire["key"] == wire_key
    assert wire["tab"] == "ui:7:synthetic"
    if selector:
        assert wire["selector"] == selector
        assert selector in result.text
    else:
        assert "selector" not in wire
    # No field beyond tab/key/selector may leak onto the press frame.
    assert set(wire) <= {"tab", "key", "selector", "requester", "session_label"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "selector"),
    [("Control+K", ""), ("e", ""), ("F5", ""), ("Tab", "#field"), ("Shift+Tab", "e3")],
)
async def test_old_press_host_never_receives_a_widened_press(raw: str, selector: str) -> None:
    # A desktop that advertises `press` but not `press_chord` knows only the
    # original eight keys and ignores `selector`: anything wider would be
    # mis-handled silently, so it is refused locally with a typed error.
    app = App(("press",))
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key=raw, selector=selector),
        None,
        client=app,
    )
    assert result.is_error
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert (result.details or {})["capability"] == "press_chord"
    assert "update the desktop app" in result.text
    assert "Tab, Shift+Tab, Enter, Escape" in result.text  # what still works
    assert app.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("key", KEYS)
async def test_old_press_host_still_serves_the_original_eight(key: str) -> None:
    app = App(("press",))
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key=key),
        None,
        client=app,
    )
    assert not result.is_error, result.text
    ((method, wire),) = app.sent
    assert method == "press"
    assert {k: v for k, v in wire.items() if k in ("tab", "key", "selector")} == {
        "tab": "ui:7:synthetic",
        "key": key,
    }


@pytest.mark.asyncio
async def test_host_without_press_gets_the_method_refusal_not_the_chord_one() -> None:
    app = App(("click",))
    result = await builtin._bridge_action(
        "t",
        owned_surface(),
        "press",
        builtin.BrowserParams(action="press", key="Control+K"),
        None,
        client=app,
    )
    assert result.is_error
    assert "does not provide 'press'" in result.text
    assert app.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["execute_browser", "_execute_browser"])
async def test_press_selector_is_validated_before_discovery(monkeypatch, entrypoint: str) -> None:
    def unexpected() -> None:
        pytest.fail("invalid press target reached host discovery")

    monkeypatch.setattr(builtin, "cmux_browser_available", unexpected)
    result = await getattr(builtin, entrypoint)(
        "t", {"action": "press", "key": "Tab", "selector": "--help"}
    )
    assert result.is_error
    assert "flag-shaped selector" in result.text


def test_press_and_hover_stay_write_tier_like_click(monkeypatch) -> None:
    # Same approval treatment as click/type: no read downgrade for key/pointer input.
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    tool = builtin.build_browser_tool(None)
    assert tool is not None and tool.call_approval_tier is not None
    assert tool.approval_tier == "write"
    for action in ("click", "type", "press", "hover"):
        assert tool.call_approval_tier({"action": action}) == "write"
