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


@pytest.mark.parametrize(
    "raw",
    [
        "This model's maximum context length is 16385 tokens",
        "prompt is too long: 250000 tokens > 200000 maximum",
        "The input token count (1200000) exceeds the maximum number of tokens allowed",
        "The request exceeds the model's maximum context window",
        "Too many tokens in prompt",
        "too many tokens: input is larger than the model's context length",
        "request too large for model",
        "Input is too long for requested model",
    ],
)
def test_every_vendor_overflow_wording_is_recognised(raw: str) -> None:
    """The overflow rule has to cover how vendors ACTUALLY phrase it.

    An audit against real vendor output found the list recognising 6 of 10
    wordings: google/vertex counts input tokens, mistral and bedrock phrase it
    differently again. The gap mattered beyond this classifier, because
    `providers.clients` shares the list to decide that a relayed overflow is
    deterministic and must not be retried as upstream weather.
    """
    assert classify_incident(raw).category == "context-length"


@pytest.mark.parametrize(
    "raw",
    [
        # Verbatim AWS Bedrock ThrottlingException wording.
        "ThrottlingException: Too many tokens, please wait before trying again.",
        # A TPM limit that quotes a token count in passing.
        "Rate limit reached for gpt-4: Limit 90000 token count per min",
        "Number of request tokens has exceeded your per-minute rate limit",
    ],
)
def test_a_throttle_that_mentions_tokens_is_not_an_overflow(raw: str) -> None:
    """Rate limits must not be read as context overflows.

    context-length is the FIRST rule, so it outranks rate-limit: a bare "too
    many tokens" or "token count" marker captures these and tells the user to
    /compact a request whose only problem is that it arrived too soon. The
    markers are therefore qualified to name the INPUT, which a throttle never
    does.
    """
    assert classify_incident(raw).category != "context-length"
