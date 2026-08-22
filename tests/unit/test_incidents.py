"""Session incident classification and formatting."""

from __future__ import annotations

import pytest

from local_operator.incidents import (
    SESSION_INCIDENT_MESSAGE_TYPE,
    SESSION_MODEL_SWITCH_MESSAGE_TYPE,
    classify_incident,
    format_incident_message,
    format_model_switch_message,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("429 Too Many Requests", "rate-limit"),
        ("RATE LIMIT exceeded for claude-opus-5", "rate-limit"),
        ("usage limit reached on this account", "rate-limit"),
        ("401 Unauthorized: invalid api key", "auth"),
        ("403 permission denied for this model", "auth"),
        ("refresh token expired", "auth"),
        ("402 payment required", "billing"),
        ("503 service unavailable", "provider"),
        ("bad gateway from upstream", "provider"),
        ("read timeout while streaming", "network"),
        ("connection reset by peer", "network"),
        ("SSL certificate verify failed", "network"),
        ("maximum context length is 200000 tokens", "context-length"),
        ("the request was too large", "context-length"),
        ("MCP server 'linear' unavailable", "mcp"),
        ("something completely novel happened", "unknown"),
    ],
)
def test_classification(raw: str, expected: str) -> None:
    assert classify_incident(raw).category == expected


def test_render_carries_category_source_hint_and_raw():
    text = format_incident_message("429 quota exceeded", "anthropic", "claude-opus-5")
    assert text.startswith("[session incident (anthropic/claude-opus-5)] rate-limit:")
    assert "429 quota exceeded" in text
    assert "suggested action:" in text
    assert "previous turn ended" in text


def test_unknown_has_no_invented_hint():
    incident = classify_incident("a novel failure mode")
    assert incident.category == "unknown"
    assert incident.hint == ""


def test_message_type_constant_is_stable():
    # Persisted into transcripts; renaming it would orphan old sessions' replay.
    assert SESSION_INCIDENT_MESSAGE_TYPE == "session_incident"


def test_model_switch_deliberate_names_old_and_new_and_persists():
    text = format_model_switch_message(
        "anthropic/claude-opus-4-8",
        "zai/glm-5.3",
        reason="model switched",
    )
    assert text.startswith("[model switch] You are now running as anthropic/claude-opus-4-8")
    assert "was zai/glm-5.3" in text
    assert "applies from now on" in text
    # A deliberate switch is not transient, so it must not carry the fallback caveat.
    assert "temporary fallback" not in text


def test_model_switch_transient_fallback_is_marked_temporary():
    text = format_model_switch_message(
        "kimi/k3",
        "anthropic/claude-opus-4-8",
        reason="anthropic 429 — falling back",
        transient=True,
    )
    assert "You are now running as kimi/k3" in text
    assert "temporary fallback" in text
    assert "Reason: anthropic 429 — falling back" in text
    assert "applies from now on" not in text


def test_model_switch_without_previous_label_reads_cleanly():
    # The return-to-primary edge passes no previous label.
    text = format_model_switch_message("anthropic/claude-opus-4-8")
    assert text.startswith("[model switch] You are now running as anthropic/claude-opus-4-8.")
    assert "(was" not in text


def test_model_switch_message_type_constant_is_stable():
    # Persisted into transcripts; renaming it would orphan replay of old sessions.
    assert SESSION_MODEL_SWITCH_MESSAGE_TYPE == "session_model_switch"
