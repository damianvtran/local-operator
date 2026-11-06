"""The `setvalue` contract through the real tool/transport, with a fake host.

The page program itself lives in the desktop app's browser host (the other half
of this change, exercised by that repository's browser QA); these checks pin the
half that CAN be pinned here: the wire params, the argument refusals, the
capability gate and the rendering. The split matters because the verb exists to
fix a measured failure — a native `<select>` that refused `select`, a click on
an option node and the keyboard alike — so the tool layer must not be the place
the fix degrades back into an echo of the caller's own value.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from local_operator.browser_bridge.backend import BridgeError, HostCapabilities
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    DESKTOP_ONLY_METHODS,
    METHODS,
    ErrorCode,
)
from local_operator.harness.types import BrowserSurface, ToolContext
from local_operator.tools import builtin

RESULT = {
    # The host's READ-BACK, deliberately different from what the caller asked
    # for: a native select maps a visible label onto its own value, and that
    # mapping is exactly what this verb makes observable.
    "value": "production",
    "kind": "select",
    "url": "https://example.test/",
    "title": "New App",
}


class App:
    """A desktop-app host advertising whatever `methods` it is given."""

    host = "ui"

    def __init__(self, methods=("setvalue",), result=None, error=None):
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


class Extension:
    """The Chrome extension: it neither serves nor advertises `setvalue`."""

    host = "extension"

    def __init__(self):
        self.sent = []

    def capabilities(self):
        return HostCapabilities(
            methods=("open", "click", "type", "select"),
            version="0.1.21",
            capabilities_known=True,
        )

    async def call(self, method, params):  # pragma: no cover - must never be reached
        self.sent.append((method, params))
        raise AssertionError("a refused capability must not reach the wire")


def owned_surface() -> BrowserSurface:
    state = BrowserSurface()
    state.surface_id = "ui:7:setvalue-test"
    return state


async def setvalue(app, selector="#environment", text="Production"):
    return await builtin._bridge_action(
        "t",
        owned_surface(),
        "setvalue",
        builtin.BrowserParams(action="setvalue", selector=selector, text=text),
        None,
        client=app,
    )


def test_protocol_and_schema_are_additive() -> None:
    # One ACTION on tables that already exist: no new tool, no new parameter.
    assert "setvalue" in METHODS
    assert "setvalue" in CAPABILITY_GATED_METHODS
    # Desktop-only: the host owns the page program and can read the value back.
    assert "setvalue" in DESKTOP_ONLY_METHODS
    assert COMMAND_TIMEOUTS["setvalue"] == 25.0
    assert "setvalue" in builtin.BROWSER_ACTIONS
    # Non-cmux by construction: cmux exposes no page-program primitive, so a
    # silent no-op would be the only alternative to a typed refusal.
    assert "setvalue" in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS
    # The value rides the EXISTING `text` param — adding one would have been the
    # tool-surface ladder's rung 2+ for a verb that needs no new field.
    text_help = builtin.BrowserParams.model_fields["text"].description or ""
    assert "'setvalue'" in text_help


@pytest.mark.parametrize("entrypoint", ["execute_browser", "_execute_browser"])
@pytest.mark.parametrize(
    "fields,needle",
    [
        ({"selector": "#environment"}, "needs 'text'"),
        ({"text": "Production"}, "requires a selector"),
        ({"selector": "--help", "text": "Production"}, "flag-shaped selector"),
    ],
)
@pytest.mark.asyncio
async def test_bad_arguments_fail_before_discovery(monkeypatch, entrypoint, fields, needle):
    # Both entry points, because the public one and the internal one each have
    # their own preflight and a caller can reach either.
    def fail():
        pytest.fail("bad setvalue arguments reached discovery")

    monkeypatch.setattr(builtin, "cmux_browser_available", fail)
    result = await getattr(builtin, entrypoint)("t", {"action": "setvalue", **fields})
    assert result.is_error and needle in result.text


@pytest.mark.asyncio
async def test_the_wire_carries_the_target_and_the_value_only() -> None:
    app = App()
    result = await setvalue(app)
    assert not result.is_error, result.text
    method, wire = app.sent[0]
    assert method == "setvalue"
    assert wire["selector"] == "#environment" and wire["text"] == "Production"
    assert wire["tab"] == "ui:7:setvalue-test"
    # Nothing invented on the way out: the host's own vocabulary is the verb, a
    # target and a value.
    assert set(wire) <= {"tab", "requester", "session_label", "selector", "text"}


@pytest.mark.asyncio
async def test_an_empty_value_is_a_legitimate_clear() -> None:
    # Presence is checked by TYPE, never by truthiness: clearing a field with ""
    # is a real operation the host implements, and refusing it here would publish
    # one wire contract with two answers (review R1-1).
    app = App()
    result = await setvalue(app, selector="#notes", text="")
    assert not result.is_error, result.text
    assert app.sent[0][1]["text"] == "", "the clear must reach the host"

    # An OMITTED value is still refused, and refused before any dispatch: a
    # missing argument is a caller mistake, not an instruction to clear.
    missing = await builtin._execute_browser("t", {"action": "setvalue", "selector": "#notes"})
    assert missing.is_error and "needs 'text'" in missing.text


@pytest.mark.asyncio
async def test_a_password_read_back_is_never_rendered() -> None:
    # A password box whose value did not stick would otherwise hand the page's
    # OWN secret to the transcript: this is the one verb that renders a value.
    # `input_type` is the host's echo of the control's type — the harness must
    # not infer it from the selector, and must stay safe against an older host
    # that still sends the value (review R1-2).
    result = await setvalue(
        App(
            result={
                "kind": "input",
                "input_type": "password",
                "value": "hunter2",
                "verified": True,
                "url": "https://example.test/",
                "title": "Sign in",
            }
        )
    )
    assert not result.is_error, result.text
    assert "hunter2" not in result.text, result.text
    assert "hunter2" not in str(result.details or {})
    assert "value withheld: a password field" in result.text
    assert (result.details or {})["input_type"] == "password"
    assert "value" not in (result.details or {})


@pytest.mark.asyncio
async def test_an_ambiguous_option_match_lists_the_controls_own_options() -> None:
    # `option_ambiguous` is the setvalue-specific refusal a MULTIPLE select
    # returns when a label matches twice. It carries the same `available` list as
    # `option_not_found`, so the guide's promise ("names the options") holds for
    # both reasons rather than only the one `select` already emitted (R1-3a).
    available = [
        {"index": 0, "label": "Beta", "value": "beta-1"},
        {"index": 1, "label": "Beta", "value": "beta-2"},
    ]
    result = await setvalue(
        App(
            error=BridgeError(
                ErrorCode.ELEMENT_NOT_FOUND,
                "setvalue refused: option_ambiguous",
                {"reason": "option_ambiguous", "available": available, "matches": 2},
            )
        )
    )
    assert result.is_error and (result.details or {})["reason"] == "option_ambiguous"
    assert (result.details or {})["available"] == available
    assert "[1] 'Beta' (value='beta-2')" in result.text


@pytest.mark.asyncio
async def test_a_dash_leading_value_is_legal_because_it_never_becomes_argv() -> None:
    # `type` refuses flag-shaped text because cmux's `--text` slot parses it as an
    # option. `setvalue` is a non-cmux method whose value travels as a JSON string
    # field, and "-5" is an ordinary value for a numeric select or a spinbox.
    app = App()
    result = await setvalue(app, selector="#count", text="-5")
    assert not result.is_error, result.text
    assert app.sent[0][1]["text"] == "-5"


@pytest.mark.asyncio
async def test_the_rendered_value_is_the_hosts_read_back_not_the_request() -> None:
    # THE point of the verb: echoing "Production" would report a state the page
    # never entered, which is the same false success the incident hit.
    result = await setvalue(App())
    assert not result.is_error
    assert "to 'production'" in result.text
    assert (result.details or {})["value"] == "production"
    assert (result.details or {})["kind"] == "select"


@pytest.mark.asyncio
async def test_an_unread_value_is_not_reported_as_success() -> None:
    # A host that could not read the value back must not let an empty string pass
    # as "the field is now empty".
    result = await setvalue(App(result={"kind": "input", "url": "https://example.test/"}))
    assert not result.is_error
    assert "value unverified" in result.text
    assert (result.details or {})["value"] == ""


@pytest.mark.asyncio
async def test_an_older_desktop_host_is_refused_without_send() -> None:
    app = App(methods=("click", "type", "select"))
    result = await setvalue(app)
    assert result.is_error and (result.details or {})["error_code"] == "capability_unsupported"
    assert "update the desktop app" in result.text
    assert app.sent == []


@pytest.mark.asyncio
async def test_an_extension_host_is_refused_with_the_desktop_sentence() -> None:
    # The documented copy for a DESKTOP_ONLY method: it names the action and the
    # process that serves it, and offers no update/switch remedy because none
    # exists. This is the exact sentence the tool layer must produce.
    extension = Extension()
    result = await setvalue(extension)
    assert result.is_error
    assert (result.details or {})["error_code"] == "capability_unsupported"
    assert "'setvalue' is served only by the Local Operator desktop app's browser" in result.text
    assert "the Chrome extension cannot do it" in result.text
    assert "backend='ui'" in result.text
    assert extension.sent == [], "the refusal must be local, before any socket dial"


@pytest.mark.asyncio
async def test_a_missing_value_match_lists_the_controls_own_options() -> None:
    # The wire spec's typed refusal: an absent or ambiguous match names what IS
    # there, so the caller retries correctly instead of guessing.
    result = await setvalue(
        App(
            error=BridgeError(
                ErrorCode.ELEMENT_NOT_FOUND,
                "no option matches",
                {
                    "reason": "option_not_found",
                    "available": [{"index": 0, "label": "Sandbox", "value": "sandbox"}],
                },
            )
        )
    )
    assert result.is_error and (result.details or {})["reason"] == "option_not_found"
    assert "[0] 'Sandbox' (value='sandbox')" in result.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "disabled",
        "readonly",
        "not_a_form_control",
        "file_input",
        # The statuses the paired host's page program answers with (or is being
        # changed to answer with: `credential_input` for a password target), so a
        # new one cannot reach the model as prose the tool would have to parse.
        "option_ambiguous",
        "credential_input",
        "invalid_arguments",
    ],
)
async def test_a_host_refusal_keeps_its_typed_reason(reason: str) -> None:
    # The refusals the wire spec names arrive as the host's own reason, so the
    # tool never has to parse prose to know which one it met.
    result = await setvalue(
        App(error=BridgeError(ErrorCode.ELEMENT_NOT_FOUND, "cannot set", {"reason": reason}))
    )
    assert result.is_error and (result.details or {})["reason"] == reason


@pytest.mark.asyncio
async def test_cmux_is_unsupported_and_setvalue_is_write_tier(monkeypatch) -> None:
    monkeypatch.setattr(builtin, "cmux_browser_available", lambda: True)
    monkeypatch.setattr(builtin, "bridge_browser_reachable", AsyncMock(return_value=False))
    monkeypatch.setattr(builtin, "ui_browser_reachable", AsyncMock(return_value=False))
    state = BrowserSurface()
    state.surface_id = "surface:3"
    result = await builtin._execute_browser(
        "t",
        {"action": "setvalue", "selector": "#n", "text": "5"},
        None,
        None,
        ToolContext(browser=state),
    )
    assert result.is_error and "not supported on the cmux backend" in result.text
    # Writing a page value is a mutation of the user's page, so it takes the
    # ordinary write tier: no new approval class was invented for it.
    tool = builtin.build_browser_tool(None)
    assert tool is not None and tool.call_approval_tier is not None
    assert tool.call_approval_tier({"action": "setvalue"}) == "write"
