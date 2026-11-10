"""``is_provider_empty``: "the provider sent nothing" vs "the model said nothing".

One case per wire's evidence of generation, so a wire that stops leaving its trace
turns a row red rather than silently turning reasoning stops into provider-empty
retries (the loop's side of this is pinned in ``test_loop.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.provider_empty import is_provider_empty
from local_operator.harness.types import (
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
    Usage,
)

# ---------------------------------------------------------------------------
# is_provider_empty: "the provider sent nothing" vs "the model said nothing"
# ---------------------------------------------------------------------------


def _assistant(**fields) -> Message:
    fields.setdefault("stop_reason", "stop")
    return Message(role="assistant", **fields)


def _native(protocol: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    from local_operator.providers.replay import native_payload

    return native_payload(
        ModelSpec(provider="p", model_id="m"), "https://e", protocol, items, "", []
    )


def test_a_message_with_nothing_at_all_is_provider_empty():
    assert is_provider_empty(_assistant())
    # A zeroed usage block is what a client reports when the wire named no counts.
    assert is_provider_empty(_assistant(usage=Usage()))
    # An empty text part is no content (Anthropic can stream an empty text block).
    assert is_provider_empty(_assistant(content=[TextContent(text="")]))


@pytest.mark.parametrize(
    ("label", "fields"),
    [
        # OpenAI Responses: usage is the ONLY trace of reasoning (no payload is
        # kept unless ``encrypted_content`` was requested).
        ("responses_reasoning_tokens", {"usage": Usage(output_tokens=40, reasoning_tokens=40)}),
        ("reasoning_tokens_alone", {"usage": Usage(reasoning_tokens=5)}),
        # Gemini reports ``thoughtsTokenCount`` as reasoning_tokens AND a thought part.
        (
            "google_thought_part",
            {"provider_payload": _native("google-content", [{"text": "hm", "thought": True}])},
        ),
        # Anthropic: a thinking block, or an opaque redacted one.
        (
            "anthropic_thinking",
            {
                "provider_payload": _native(
                    "anthropic-messages", [{"type": "thinking", "thinking": "hm", "signature": "s"}]
                )
            },
        ),
        (
            "anthropic_redacted_thinking",
            {
                "provider_payload": _native(
                    "anthropic-messages", [{"type": "redacted_thinking", "data": "opaque"}]
                )
            },
        ),
        # Chat Completions: reasoning_content / reasoning fragments.
        (
            "chat_reasoning_content",
            {"provider_payload": _native("openai-chat", [{"reasoning_content": "hm"}])},
        ),
        ("chat_reasoning", {"provider_payload": _native("openai-chat", [{"reasoning": "hm"}])}),
        # Output the client could not show: hidden reasoning, or filtered markup.
        ("output_tokens_only", {"usage": Usage(output_tokens=3)}),
    ],
)
def test_reasoning_evidence_on_every_wire_is_not_provider_empty(label: str, fields: dict[str, Any]):
    assert not is_provider_empty(_assistant(**fields)), label


@pytest.mark.parametrize(
    "fields",
    [
        {"stop_reason": "length"},
        {"stop_reason": "toolUse"},
        {"stop_reason": "error"},
        {"stop_reason": "refusal"},
        {"stop_reason": None},
        {"content": [TextContent(text="an answer")]},
        # Whitespace is something the provider sent (the OSWorld wordless shape).
        {"content": [TextContent(text="\n")]},
        {"tool_calls": [ToolCall(id="c", name="echo", arguments={})]},
    ],
)
def test_other_turn_shapes_are_not_provider_empty(fields: dict[str, Any]):
    assert not is_provider_empty(_assistant(**fields))


def test_only_an_assistant_message_can_be_provider_empty():
    assert not is_provider_empty(Message(role="user", content=[], stop_reason="stop"))


def test_a_payload_with_no_items_is_not_reasoning_evidence():
    """A payload that carries no ``native_replay`` items (the response id and the
    refusal text ride in ``provider_payload`` too) is bookkeeping, not content."""
    assert is_provider_empty(_assistant(provider_payload={"id": "resp_1"}))
    assert is_provider_empty(_assistant(provider_payload=_native("openai-chat", [])))
