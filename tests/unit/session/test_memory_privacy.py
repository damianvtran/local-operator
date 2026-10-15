"""Real Session rejection: recognised sensitive memory values never get retained."""

import json

import pytest

from local_operator.harness.types import (
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolCall,
)
from local_operator.memory_privacy import retained_message
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript, encode_message_payload
from local_operator.tools.builtin import build_write_tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["file", "memory", "bad-op", "missing-op", "bad-scope", "missing-scope", "unknown-field"],
)
@pytest.mark.parametrize(
    "label,fragment",
    [
        ("passport", "passport number AB" + "1234567"),
        ("licence", "licence number X" + "9876543"),
        ("ssn", "SSN 1\u200b23-45-6789"),
        ("opaque", "a3bf" * 10),
        ("credential", "mongodb://test:" + "synthetic-password@db.invalid/test"),
    ],
    ids=["passport", "licence", "ssn", "opaque", "credential"],
)
async def test_real_session_refuses_without_journal_index_or_diagnostic_residue(
    tmp_path, caplog, label, fragment, mode
):
    args = {
        "path": "memory://session",
        "content": json.dumps({"title": "Synthetic refusal", "body": fragment}),
    }
    tool_name = "write"
    from local_operator.tools.memory_tools import build_memory_tool

    if mode != "file":
        tool_name = "memory"
        args = {"op": "store", "scope": "session", "text": fragment}
        if mode == "bad-op":
            args["op"] = "invalid"
        elif mode == "missing-op":
            del args["op"]
        elif mode == "bad-scope":
            args["scope"] = "another-owner"
        elif mode == "missing-scope":
            del args["scope"]
        elif mode == "unknown-field":
            args["unknown"] = fragment
    original = json.dumps(args)
    calls = 0

    def stream(request, signal):
        nonlocal calls
        calls += 1

        async def events():
            if calls == 1:
                yield StreamToolCallDelta(
                    index=0, id="sensitive-memory", name=tool_name, argument_delta=original
                )
                yield StreamEndEvent(
                    stop_reason="toolUse",
                    provider_payload={
                        "raw_response": {"function": {"name": "write", "arguments": original}},
                        "alternate": fragment,
                    },
                )
            else:
                yield StreamTextDelta(delta="Value refused; use secret NAME references.")
                yield StreamEndEvent(stop_reason="stop")

        return events()

    transcript = Transcript(tmp_path / "sessions" / "privacy-session")
    session = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[build_write_tool()] if mode == "file" else [build_memory_tool()],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="privacy-session",
        yolo=True,
        cwd=str(tmp_path),
    )
    events = []
    session.subscribe(events.append)
    await session.prompt("Retain the synthetic test item")
    assert calls == 2
    tool_results = [entry for entry in transcript.entries() if entry.payload.get("role") == "tool"]
    assert tool_results
    assert "Sensitive value refused" in json.dumps(tool_results[-1].payload)
    serialized = json.dumps([entry.payload for entry in transcript.entries()])
    assert fragment not in serialized
    assert "memory_arguments_redacted" in serialized
    leaking = [
        event.type
        for event in events
        if fragment in json.dumps(event.model_dump(mode="json"), default=str)
    ]
    assert not leaking, leaking
    assert fragment not in caplog.text
    # In-memory execution must still see and reject the original, not silently
    # write a redaction marker as a remembered fact.
    assert not (session._config_dir / "memory").exists()
    assert not (tmp_path / "memory:").exists()
    for path in transcript.directory.iterdir():
        if path.is_file():
            assert fragment.encode() not in path.read_bytes()
    await session.dispose()


def test_direct_serializer_alternate_raw_carrier_and_original_execution_unchanged():
    fragment = "passport number AB" + "1234567"
    args = {
        "path": "memory://session",
        "content": json.dumps({"title": "Refusal", "body": fragment}),
    }
    raw = json.dumps(args)
    message = Message(
        role="assistant",
        content=[],
        tool_calls=[ToolCall(name="write", arguments=args, raw_arguments=raw)],
        provider_payload={"alternate_tool_json": raw},
    )
    retained = retained_message(message)
    encoded = encode_message_payload(message)
    assert fragment not in json.dumps(encoded)
    assert retained.tool_calls[0].arguments["memory_arguments_redacted"]
    assert message.tool_calls[0].arguments == args
    assert message.tool_calls[0].raw_arguments is not None
    assert fragment in message.tool_calls[0].raw_arguments
    assert fragment in json.dumps(message.provider_payload)


def test_non_memory_message_byte_identity():
    args = {"path": "ordinary.txt", "content": "passport number AB" + "1234567"}
    message = Message(
        role="assistant", content=[], tool_calls=[ToolCall(name="write", arguments=args)]
    )
    assert retained_message(message) is message
