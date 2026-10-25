"""Durable, reload, summary, memory, and analytics boundaries for secret results."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import SecretStr

from local_operator.analytics.model import snapshot_component_chars
from local_operator.compaction.api import build_compaction_prompt
from local_operator.harness.types import (
    ONE_TURN_SECRET_RECEIPT,
    ChatRequest,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolResult,
)
from local_operator.memory_capture import snapshot as memory_snapshot
from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.transcript import Transcript

MASK = "[red" + "acted]"


def _canary(label: str = "persist") -> str:
    return "v0rn-" + label + "-c31e"


def _safe_request(canary: str) -> ChatRequest:
    public = ToolResult(
        tool_call_id="call-secret",
        tool_name="native-secret",
        content=[TextContent(text=ONE_TURN_SECRET_RECEIPT)],
    )
    tool_row = Message.tool_result(public)
    tool_row.id = "tool-secret"
    assistant = Message(
        role="assistant",
        content=[TextContent(text=MASK)],
        tool_calls=[
            ToolCall(
                id="call-echo",
                name="echo",
                arguments={"text": MASK},
                raw_arguments=json.dumps({"text": MASK}),
            )
        ],
        provider_payload=None,
    )
    assistant.id = "assistant-safe"
    request = ChatRequest(
        model=ModelSpec(provider="test", model_id="m"),
        messages=[Message.user("go"), assistant, tool_row],
    )
    request.arm_one_turn_secrets({"call-secret": SecretStr(canary)})
    return request


@pytest.mark.asyncio
async def test_transcript_and_reload_hold_only_the_receipt(tmp_path: Path) -> None:
    canary = _canary("jsonl")
    request = _safe_request(canary)
    transcript = Transcript(tmp_path / "session")
    await transcript.append_messages(request.messages)

    raw_jsonl = transcript.path.read_text()
    assert canary not in raw_jsonl
    assert ONE_TURN_SECRET_RECEIPT in raw_jsonl
    assert "_one_turn_secret" not in raw_jsonl
    assert "_one_turn_secrets" not in raw_jsonl

    replayed = Transcript(tmp_path / "session").build_llm_history()
    replay_request = ChatRequest(
        model=ModelSpec(provider="test", model_id="m"),
        messages=[message for message in replayed if isinstance(message, Message)],
    )
    assert not replay_request.has_one_turn_secrets
    assert replay_request.wire_messages() is replay_request.messages
    assert replay_request.messages[-1].text == ONE_TURN_SECRET_RECEIPT
    assert canary not in replay_request.model_dump_json()


@pytest.mark.asyncio
async def test_checkpoint_attach_compaction_memory_and_analytics_are_safe(
    tmp_path: Path,
) -> None:
    canary = _canary("surfaces")
    request = _safe_request(canary)
    tool_row = request.messages[-1]
    result = ToolResult(
        tool_call_id="call-secret",
        tool_name="native-secret",
        content=[TextContent(text=ONE_TURN_SECRET_RECEIPT)],
    )

    store = FrontendStateStore(FrontendSessionState(session_id="s", epoch="e"))
    store._fold_live_event(
        ToolExecutionEndEvent(
            tool_call_id="call-secret",
            tool_name="native-secret",
            result=result,
            is_error=False,
        )
    )
    attach_seed = store.state.model_dump(mode="json")
    assert canary not in json.dumps(attach_seed)
    assert ONE_TURN_SECRET_RECEIPT in json.dumps(attach_seed)

    transcript = Transcript(tmp_path / "frontend")
    await store.checkpoint(transcript)
    checkpoint_jsonl = transcript.path.read_text()
    assert canary not in checkpoint_jsonl
    assert "_one_turn_secret" not in checkpoint_jsonl

    compaction_prompt = build_compaction_prompt(request.messages)
    assert canary not in compaction_prompt
    assert ONE_TURN_SECRET_RECEIPT in compaction_prompt

    capture_input = memory_snapshot(request.messages)
    assert canary not in capture_input
    assert ONE_TURN_SECRET_RECEIPT in capture_input

    analytics_record = snapshot_component_chars(request)
    assert analytics_record["tool_results"] == len(ONE_TURN_SECRET_RECEIPT)
    assert canary not in json.dumps(analytics_record)
    assert tool_row.text == ONE_TURN_SECRET_RECEIPT


def test_ordinary_transcript_payload_has_no_new_serialized_keys() -> None:
    ordinary = ToolResult(
        tool_call_id="call-ordinary",
        tool_name="echo",
        content=[TextContent(text="ordinary output")],
    )
    message = Message.tool_result(ordinary)
    payload = message.model_dump(mode="json")
    assert payload["role"] == "tool"
    assert payload["content"] == [{"type": "text", "text": "ordinary output"}]
    assert payload["tool_call_id"] == "call-ordinary"
    assert payload["tool_name"] == "echo"
    assert "_one_turn_secret" not in payload
    assert "_one_turn_secrets" not in payload
