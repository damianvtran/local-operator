"""Security and transport tests for session-owned browser network telemetry."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

import local_operator.tools.builtin as builtin
from local_operator.browser_bridge.protocol import (
    CAPABILITY_GATED_METHODS,
    COMMAND_TIMEOUTS,
    METHODS,
)
from local_operator.harness.types import BrowserSurface, ToolContext


def _params(**kwargs):
    return builtin.BrowserParams(action="network_read", **kwargs)


def _ctx(tmp_path: Path, surface: str = "ui:7:nonce") -> ToolContext:
    ctx = ToolContext(cwd=str(tmp_path))
    ctx.browser = BrowserSurface()
    ctx.browser.surface_id = surface
    ctx.scratchpad_dir = str(tmp_path / "scratch")
    return ctx


@pytest.mark.parametrize(
    "action",
    ["network_start", "network_stop", "network_read", "network_clear"],
)
def test_network_actions_have_additive_protocol_and_capability_contract(action: str):
    assert action in builtin.BROWSER_ACTIONS
    assert action in builtin.CMUX_UNSUPPORTED_BROWSER_ACTIONS
    assert action in METHODS
    assert action in CAPABILITY_GATED_METHODS
    assert COMMAND_TIMEOUTS[action] == 20.0


def test_metadata_start_requires_no_body_approval():
    assert builtin._validate_browser_args("network_start", _params()) == ""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_request_body": True},
        {"approved": True},
        {"include_request_body": True, "approved": True},
        {
            "include_request_body": True,
            "approved": True,
            "url": "https://sso.example.test",
        },
        {
            "include_request_body": True,
            "approved": True,
            "url": "https://sso.example.test/",
        },
        {
            "include_request_body": True,
            "approved": True,
            "url": "https://sso.example.test/acs#private",
        },
        {
            "include_request_body": True,
            "approved": True,
            "url": "https://user@sso.example.test/acs",
        },
    ],
)
def test_sensitive_body_start_requires_exact_scope_and_explicit_approval(kwargs):
    assert builtin._validate_browser_args("network_start", _params(**kwargs))


def test_exact_body_scope_can_contain_query_without_requiring_values():
    error = builtin._validate_browser_args(
        "network_start",
        _params(
            url="https://sso.example.test/auth/saml/acs?transaction=ignored",
            include_request_body=True,
            approved=True,
        ),
    )
    assert error == ""


def test_sensitive_options_cannot_leak_to_other_actions():
    assert "network" in builtin._validate_browser_args("read", _params(approved=True))
    assert "network_start" in builtin._validate_browser_args(
        "network_stop", _params(include_request_body=True)
    )
    assert "limit" in builtin._validate_browser_args("network_read", _params(limit=201))
    assert "status" in builtin._validate_browser_args("network_read", _params(status=99))


def test_sensitive_approval_description_names_body_not_generic_browse():
    description = builtin._describe_browser_approval(
        {
            "action": "network_start",
            "url": "https://sso.example.test/auth/saml/acs",
            "include_request_body": True,
            "approved": True,
        },
        "/tmp",
    )
    assert description == "capture sensitive request body: sso.example.test/auth/saml/acs"


def test_url_metadata_redacts_query_values_credentials_and_fragment():
    raw = "https://user:password@example.test/acs?SAMLResponse=secret&RelayState=private#claim"
    safe = builtin._network_safe_url(raw)
    assert safe == "https://example.test/acs?SAMLResponse=<redacted>&RelayState=<redacted>"
    assert all(secret not in safe for secret in ("secret", "private", "password", "user"))


def test_metadata_read_never_forwards_host_body_or_arbitrary_fields(monkeypatch, tmp_path):
    async def fake_bridge(tool_call_id, action, wire, *, surface="", client=None):
        return {
            "active": True,
            "entries": [
                {
                    "id": 1,
                    "request_id": "req",
                    "url": "https://example.test/acs?SAMLResponse=private",
                    "method": "POST",
                    "request_body_base64": base64.b64encode(b"do-not-render").decode(),
                    "headers": {"Authorization": "private"},
                }
            ],
        }, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        builtin._browser_network("call", ctx.browser, "network_read", _params(), ctx)
    )
    assert not result.is_error
    text = result.text
    assert "private" not in text
    assert "do-not-render" not in text
    assert "headers" not in text
    assert json.loads(text)["entries"][0]["method"] == "POST"


def test_explicit_body_read_writes_mode_0600_without_rendering_secret(monkeypatch, tmp_path):
    # Synthetic non-secret fixture, not a real SAML assertion or user credential.
    body = b"SAMLResponse=synthetic-fixture"
    encoded = base64.b64encode(body).decode()

    async def fake_bridge(tool_call_id, action, wire, *, surface="", client=None):
        assert wire["include_request_body"] is True
        assert wire["approved"] is True
        assert wire["url"] == "https://sso.example.test/auth/saml/acs"
        return {"active": True, "entries": [{"id": 1, "request_body_base64": encoded}]}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        builtin._browser_network(
            "call",
            ctx.browser,
            "network_read",
            _params(
                url="https://sso.example.test/auth/saml/acs",
                include_request_body=True,
                approved=True,
            ),
            ctx,
        )
    )
    assert not result.is_error
    assert "synthetic-fixture" not in result.text
    payload = json.loads(result.text)
    body_path = Path(payload["entries"][0]["request_body_file"])
    assert body_path.parent == tmp_path / "scratch"
    assert body_path.stat().st_mode & 0o777 == 0o600
    assert body_path.read_bytes() == body
    assert payload["entries"][0]["request_body_bytes"] == len(body)


def test_body_oversize_is_not_persisted(monkeypatch, tmp_path):
    async def fake_bridge(tool_call_id, action, wire, *, surface="", client=None):
        return {"entries": [{"request_body_base64": "A" * 100_000}]}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)
    ctx = _ctx(tmp_path)
    result = asyncio.run(
        builtin._browser_network(
            "call",
            ctx.browser,
            "network_read",
            _params(
                url="https://sso.example.test/acs", include_request_body=True, approved=True
            ),
            ctx,
        )
    )
    assert result.is_error
    assert not (tmp_path / "scratch").exists()
    assert "AAAA" not in result.text


def test_network_actions_have_no_new_global_capability_switch():
    from local_operator.browser_bridge.protocol import CAPABILITY_SWITCH_LABEL

    assert not any(action.startswith("network_") for action in CAPABILITY_SWITCH_LABEL)


def test_body_capture_escalates_to_exec_and_metadata_stays_write():
    tool = builtin.build_browser_tool(None)
    assert tool is not None
    tier = tool.call_approval_tier
    assert tier({"action": "network_read"}) == "write"
    assert tier({"action": "network_start"}) == "write"
    assert tier({"action": "network_stop"}) == "write"
    assert tier({"action": "network_clear"}) == "write"
    assert tier({"action": "network_start", "include_request_body": True}) == "exec"
    assert tier({"action": "network_read", "include_request_body": True}) == "exec"


def _allow_body(monkeypatch, entries):
    async def fake_bridge(tool_call_id, action, wire, *, surface="", client=None):
        return {"active": True, "entries": entries}, None

    monkeypatch.setattr(builtin, "_bridge_call", fake_bridge)


def _body_read(ctx):
    return asyncio.run(
        builtin._browser_network(
            "call",
            ctx.browser,
            "network_read",
            _params(
                url="https://sso.example.test/auth/saml/acs",
                include_request_body=True,
                approved=True,
            ),
            ctx,
        )
    )


def test_a_malformed_body_persists_nothing_for_the_whole_read(monkeypatch, tmp_path):
    good = base64.b64encode(b"first-synthetic-body").decode()
    _allow_body(
        monkeypatch,
        [
            {"id": 1, "request_body_base64": good},
            {"id": 2, "request_body_base64": "!!!not-base64!!!"},
        ],
    )
    ctx = _ctx(tmp_path)
    result = _body_read(ctx)
    assert result.is_error
    scratch = tmp_path / "scratch"
    assert not scratch.exists() or not list(scratch.glob("network-request-*.body"))
    assert "first-synthetic-body" not in result.text


def test_an_oversized_second_body_persists_nothing(monkeypatch, tmp_path):
    good = base64.b64encode(b"first-synthetic-body").decode()
    huge = base64.b64encode(b"x" * (64 * 1024 + 1)).decode()
    _allow_body(
        monkeypatch,
        [
            {"id": 1, "request_body_base64": good},
            {"id": 2, "request_body_base64": huge},
        ],
    )
    ctx = _ctx(tmp_path)
    result = _body_read(ctx)
    assert result.is_error
    scratch = tmp_path / "scratch"
    assert not scratch.exists() or not list(scratch.glob("network-request-*.body"))


def test_combined_bodies_over_the_read_cap_persist_nothing(monkeypatch, tmp_path):
    chunk = base64.b64encode(b"y" * (60 * 1024)).decode()
    _allow_body(
        monkeypatch,
        [
            {"id": 1, "request_body_base64": chunk},
            {"id": 2, "request_body_base64": chunk},
            {"id": 3, "request_body_base64": chunk},
            {"id": 4, "request_body_base64": chunk},
            {"id": 5, "request_body_base64": chunk},
        ],
    )
    ctx = _ctx(tmp_path)
    result = _body_read(ctx)
    assert result.is_error
    assert "256" in result.text
    scratch = tmp_path / "scratch"
    assert not scratch.exists() or not list(scratch.glob("network-request-*.body"))
