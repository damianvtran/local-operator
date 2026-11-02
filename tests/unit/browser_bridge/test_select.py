"""Selection contract through the real tool/transport, with a synthetic host.

Native/ARIA delivery belongs to independent host QA; these checks pin the wire
and result/error shapes so a fake peer cannot silently fork the desktop contract.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock

import pytest

from local_operator.browser_bridge.backend import BridgeError, HostCapabilities
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    METHODS,
    ErrorCode,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin

SELECTED = [{"index": 1, "value": "en-US", "label": "English (U.S.)"}]
RESULT = {
    "selected": SELECTED,
    "kind": "native",
    "url": "https://example.test/",
    "title": "New App",
}


class App:
    host = "ui"

    def __init__(self, methods=("select",), result=None, error=None):
        self.methods = methods
        self.result = RESULT if result is None else result
        self.error = error
        self.sent = []

    def capabilities(self):
        return HostCapabilities(methods=self.methods, version="0.0.0", capabilities_known=True)

    async def call(self, method, params):
        self.sent.append((method, params))
        if self.error:
            raise self.error
        return self.result


def owned_surface():
    state = BrowserSurface()
    state.surface_id = "ui:7:select-test"
    return state


async def select(app, options=None):
    return await builtin._bridge_action(
        "t",
        owned_surface(),
        "select",
        builtin.BrowserParams(
            action="select",
            selector="#language",
            options=["English (U.S.)"] if options is None else options,
        ),
        None,
        client=app,
    )


def test_protocol_and_schema_are_additive():
    assert "select" in METHODS and "select" in CAPABILITY_GATED_METHODS
    assert COMMAND_TIMEOUTS["select"] == 25
    assert (
        "select" in builtin.BROWSER_ACTIONS and "select" in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS
    )
    assert builtin.BrowserParams(action="press", key="Tab").options == []
    assert builtin.BrowserParams(action="select", options=["001", 0]).options == ["001", 0]


@pytest.mark.parametrize("entrypoint", ["execute_browser", "_execute_browser"])
@pytest.mark.parametrize(
    "fields,needle",
    [
        ({"selector": "#s"}, "needs options"),
        ({"options": [0]}, "requires a selector"),
        ({"selector": "--help", "options": [0]}, "flag-shaped"),
        ({"selector": "#s", "options": []}, "options:[] is refused"),
        ({"selector": "#s", "options": [True]}, "booleans"),
        ({"selector": "#s", "options": [1.5]}, "integer indexes"),
        ({"selector": "#s", "options": [-1]}, "non-negative"),
        ({"selector": "#s", "options": "English"}, "list"),
    ],
)
@pytest.mark.asyncio
async def test_bad_arguments_fail_before_discovery(monkeypatch, entrypoint, fields, needle):
    def fail():
        pytest.fail("bad select arguments reached discovery")

    monkeypatch.setattr(builtin, "cmux_browser_available", fail)
    result = await getattr(builtin, entrypoint)("t", {"action": "select", **fields})
    assert result.is_error and needle in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("options", [["English (U.S.)"], [1], ["001", 2]])
async def test_wire_retains_strings_and_indexes(options):
    app = App()
    result = await select(app, options)
    assert not result.is_error, result.text
    method, wire = app.sent[0]
    assert method == "select" and wire["options"] == options
    assert wire["selector"] == "#language" and wire["tab"] == "ui:7:select-test"
    assert set(wire) <= {"tab", "requester", "session_label", "selector", "options"}
    assert (result.details or {})["selected"] == SELECTED
    assert "English (U.S.)" in result.text


@pytest.mark.asyncio
async def test_old_host_is_refused_without_send():
    app = App(methods=("click", "press"))
    result = await select(app)
    assert result.is_error and (result.details or {})["error_code"] == "capability_unsupported"
    assert "update the desktop app" in result.text and app.sent == []


@pytest.mark.asyncio
async def test_no_match_lists_available_options_and_keeps_the_reason():
    result = await select(
        App(
            error=BridgeError(
                ErrorCode.ELEMENT_NOT_FOUND,
                "no option matches",
                {
                    "reason": "option_not_found",
                    "available": SELECTED,
                },
            )
        )
    )
    assert result.is_error and (result.details or {})["reason"] == "option_not_found"
    assert (result.details or {})["available"] == SELECTED
    assert "[1] 'English (U.S.)' (value='en-US')" in result.text


@pytest.mark.asyncio
async def test_available_options_are_bounded_even_if_the_host_forgot():
    entries = [{"index": i, "label": "x" * 10000, "value": "v" * 10000} for i in range(60)]
    result = await select(
        App(
            error=BridgeError(
                ErrorCode.ELEMENT_NOT_FOUND,
                "not found",
                {
                    "reason": "option_not_found",
                    "available": entries,
                },
            )
        )
    )
    assert len((result.details or {})["available"]) == 50
    assert len((result.details or {})["available"][0]["label"]) == 120
    assert "x" * 121 not in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["select_disabled", "option_disabled", "invalid_options"])
async def test_host_refusal_keeps_typed_code_and_reason(reason):
    result = await select(
        App(
            error=BridgeError(
                ErrorCode.ELEMENT_NOT_FOUND, "disabled or invalid", {"reason": reason}
            )
        )
    )
    assert result.is_error and (result.details or {})["reason"] == reason


@pytest.mark.asyncio
async def test_aria_unverified_selection_is_not_claimed_verified():
    result = await select(App(result={**RESULT, "kind": "aria", "verified": False}))
    assert not result.is_error
    assert "selection unverified" in result.text and (result.details or {})["verified"] is False
    assert (result.details or {})["kind"] == "aria"


@pytest.mark.asyncio
async def test_extension_aria_refusal_has_an_actionable_desktop_remedy():
    result = await select(
        App(
            error=BridgeError(
                ErrorCode.CAPABILITY_UNSUPPORTED,
                "not native",
                {"reason": "aria_select_unsupported"},
            )
        )
    )
    assert result.is_error and "backend='ui'" in result.text
    assert "native <select>" in result.text


@pytest.mark.asyncio
async def test_cmux_is_unsupported_and_select_is_write_tier(monkeypatch):
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", AsyncMock(return_value=False))
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=False))
    state = BrowserSurface()
    state.surface_id = "surface:3"
    result = await builtin._execute_browser(
        "t",
        {"action": "select", "selector": "#s", "options": [0]},
        None,
        None,
        ToolContext(browser=state),
    )
    assert result.is_error and "not supported on the cmux backend" in result.text
    tool = builtin.build_browser_tool(None)
    assert tool is not None and tool.call_approval_tier is not None
    assert tool.call_approval_tier({"action": "select"}) == "write"


@pytest.mark.asyncio
async def test_real_loopback_capture_uses_the_actual_selected_result_shape(tmp_path):
    from local_operator.ui_browser.backend import UiHostClient
    from local_operator.ui_browser.state import UiHostState, publish

    frames = []

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            self.reply({"host": "ui", "pid": os.getpid(), "instance_id": "select-test"})

        def do_POST(self):
            frame = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            frames.append(frame)
            self.reply({"id": frame["id"], "ok": True, "result": RESULT})

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        publish(
            UiHostState.model_validate(
                {
                    "pid": os.getpid(),
                    "port": server.server_address[1],
                    "session_key": "k" * 48,
                    "proto": 1,
                    "app_version": "0.0.0",
                    "instance_id": "select-test",
                    "capabilities": ["select"],
                }
            ),
            tmp_path,
        )
        result = await select(UiHostClient(tmp_path), ["English (U.S.)"])
        assert not result.is_error and (result.details or {})["selected"] == SELECTED
        assert len(frames) == 1 and frames[0]["method"] == "select"
        assert frames[0]["params"]["options"] == ["English (U.S.)"]
        print("wire:", json.dumps(frames[0], sort_keys=True))
        print("result:", result.text)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
