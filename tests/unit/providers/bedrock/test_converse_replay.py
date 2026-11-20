"""Native reasoning is bound to the complete preceding rendered conversation."""

import pytest

from local_operator.harness.types import Message, TextContent, ToolCall
from local_operator.providers.bedrock.converse import (
    PROTOCOL,
    _Accumulator,
    build_converse_body,
    project_conversation,
)
from local_operator.providers.bedrock.models import capabilities_for
from local_operator.providers.context import bind_native_context

from .test_converse_body import ENDPOINT, NOVA, TOOL, body, request


def signed_history():
    req = request(tools=[TOOL])
    req.system_blocks = ["system"]
    _, _, _, proof = project_conversation(req, capabilities_for(NOVA), ENDPOINT, "scope")
    acc = _Accumulator(200)
    list(
        acc.json_events(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {
                                        "text": "private",
                                        "signature": "opaque-signature",
                                    }
                                }
                            },
                            {"reasoningContent": {"redactedContent": "c2VjcmV0"}},
                            {"text": "Looking up"},
                            {"toolUse": {"toolUseId": "a", "name": "lookup", "input": {"x": 1}}},
                        ],
                    }
                },
                "stopReason": "tool_use",
            }
        )
    )
    end = acc.end(req, ENDPOINT, "scope", proof)
    req.messages += [
        Message(
            role="assistant",
            content=[TextContent(text="Looking up")],
            tool_calls=[ToolCall(id="a", name="lookup", arguments={"x": 1})],
            stop_reason=end.stop_reason,
            provider_payload=end.provider_payload,
        ),
        Message(role="tool", tool_call_id="a", content=[TextContent(text="result")]),
    ]
    return req


def test_unchanged_native_tool_loop_replays_and_counts_same_blocks():
    req = signed_history()
    system, rows, selected, _ = project_conversation(req, capabilities_for(NOVA), ENDPOINT, "scope")
    assert (
        rows[1]["content"][0]["reasoningContent"]["reasoningText"]["signature"]
        == "opaque-signature"
    )
    assert rows[1]["content"][1]["reasoningContent"]["redactedContent"] == "c2VjcmV0"
    bound = bind_native_context(req, ENDPOINT, PROTOCOL, "scope", 1.0, selected_native=selected)
    assert bound.native_context_tokens > 0
    assert build_converse_body(bound, capabilities_for(NOVA), ENDPOINT, "scope")["messages"] == rows


@pytest.mark.parametrize(
    "change",
    [
        "user",
        "system",
        "assistant",
        "args",
        "aborted",
        "compaction",
        "model",
        "endpoint",
        "credential",
    ],
)
def test_stale_native_state_rejected_by_builder_and_context(change):
    req = signed_history()
    endpoint, scope = ENDPOINT, "scope"
    if change == "user":
        req.messages[0] = Message.user("edited earlier message")
    elif change == "system":
        req.system_blocks = ["changed system"]
    elif change == "assistant":
        req.messages[1].content = [TextContent(text="edited assistant")]
    elif change == "args":
        req.messages[1].tool_calls[0].arguments = {"repaired": True}
    elif change == "aborted":
        req.messages[1].stop_reason = "aborted"
    elif change == "compaction":
        req.messages[0] = Message.user("summary")
    elif change == "model":
        req.model = req.model.model_copy(update={"model_id": "amazon.nova-pro-v1:0"})
    elif change == "endpoint":
        endpoint = "bedrock-runtime:eu-west-1"
    elif change == "credential":
        scope = "different-account"
    _, rows, selected, _ = project_conversation(
        req, capabilities_for(req.model.model_id), endpoint, scope
    )
    assert not any("reasoningContent" in part for part in rows[1]["content"])
    bound = bind_native_context(req, endpoint, PROTOCOL, scope, 1.0, selected_native=selected)
    assert bound.native_context_tokens == 0


def test_cache_only_decorations_do_not_change_signature_proof():
    req = signed_history()
    req.prompt_cache_key = "stable"
    req.frozen_prefix_messages = 1
    rendered = body(req)
    assert any("reasoningContent" in part for part in rendered["messages"][1]["content"])
    assert any("cachePoint" in part for part in rendered["system"])


def test_second_signed_turn_invalidates_after_prior_tool_edit():
    req = signed_history()
    _, _, _, proof = project_conversation(req, capabilities_for(NOVA), ENDPOINT, "scope")
    acc = _Accumulator(100)
    list(
        acc.json_events(
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "reasoningContent": {
                                    "reasoningText": {"text": "next", "signature": "next-sig"}
                                }
                            },
                            {"text": "done"},
                        ],
                    }
                },
                "stopReason": "end_turn",
            }
        )
    )
    end = acc.end(req, ENDPOINT, "scope", proof)
    req.messages.append(
        Message(
            role="assistant",
            content=[TextContent(text="done")],
            stop_reason="stop",
            provider_payload=end.provider_payload,
        )
    )
    req.messages.append(Message.user("continue"))
    assert any("reasoningContent" in part for part in body(req)["messages"][3]["content"])
    req.messages[2].content = [TextContent(text="edited tool output")]
    assert not any("reasoningContent" in part for part in body(req)["messages"][3]["content"])
