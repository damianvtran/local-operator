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
from local_operator.memory_privacy import retained_arguments, retained_message
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript, encode_message_payload
from local_operator.tools.builtin import build_write_tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["file", "memory", "bad-op", "missing-op", "legacy-scope", "missing-text", "unknown-field"],
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
        args = {"op": "store", "text": fragment}
        if mode == "bad-op":
            args["op"] = "invalid"
        elif mode == "missing-op":
            del args["op"]
        elif mode == "legacy-scope":
            args["scope"] = "session"
        elif mode == "missing-text":
            del args["text"]
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
    refusal = json.dumps(tool_results[-1].payload)
    assert "Sensitive value refused" in refusal or "Invalid memory contract" in refusal
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


@pytest.mark.parametrize(
    "path",
    [
        # The read-path recall exemption is applied to the value the guards
        # inspect at EXECUTION. This is the second carrier of the same call, so
        # it must NOT inherit that exemption: an id-shaped memory argument keeps
        # being refused and the copy fails closed. These bytes are the pre-fix
        # bytes, proved by comparing this same corpus before and after.
        "memory://global?q=d_ef73a4312e5c4686196b30c035c8a05a",
        "memory://global?q=4111111111111111",
        "memory://global?q=GB82WEST12345698765432",
    ],
)
def test_an_id_shaped_memory_read_argument_is_redacted_before_retention(path):
    kept, raw, masked = retained_arguments("read", {"path": path})
    # ``raw`` is the None-able second half of the pair: a masked call always returns
    # the redacted JSON spelling, which is the copy the assertion below pins.
    assert masked is True and raw is not None
    assert kept == {"memory_arguments_redacted": True, "path": "memory://"}
    assert json.loads(raw) == kept


async def _run_memory_store(tmp_path, args):
    """Drive ONE ``memory`` call through a real Session/loop; return (session, transcript)."""
    from local_operator.tools.memory_tools import build_memory_tool

    original = json.dumps(args)
    calls = 0

    def stream(request, signal):
        nonlocal calls
        calls += 1

        async def events():
            if calls == 1:
                yield StreamToolCallDelta(
                    index=0, id="feedback-memory", name="memory", argument_delta=original
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return events()

    transcript = Transcript(tmp_path / "sessions" / "feedback-session")
    session = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[build_memory_tool()],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="feedback-session",
        yolo=True,
        cwd=str(tmp_path),
    )
    await session.prompt("Remember the TestFlight feedback")
    assert calls == 2
    return session, transcript


def _tool_result_text(transcript) -> str:
    results = [e for e in transcript.entries() if e.payload.get("role") == "tool"]
    assert results
    return json.dumps(results[-1].payload)


@pytest.mark.asyncio
async def test_operator_feedback_with_tunnel_host_and_attachment_path_stores(tmp_path):
    """The regression: TestFlight feedback with ids from screenshots is a storable fact."""
    from local_operator.session.attachments import attachments_dir
    from tests.unit.test_memory_guard_shapes import DIGEST64, TUNNEL_HEX, feedback_text

    text = feedback_text(f"~/.local-operator/attachments/{DIGEST64}.bin") + (
        f"\n10. Live blob {attachments_dir() / DIGEST64}.bin"
    )
    session, transcript = await _run_memory_store(tmp_path, {"op": "store", "text": text})
    result = _tool_result_text(transcript)
    assert "refused" not in result, result
    serialized = json.dumps([entry.payload for entry in transcript.entries()])
    assert f"{TUNNEL_HEX}-lop.radienthq.com" in serialized  # retained verbatim, not redacted
    assert "memory_arguments_redacted" not in serialized
    stored = (session._config_dir / "memory").rglob("*")
    assert any(
        p.is_file() and f"{TUNNEL_HEX}-lop.radienthq.com".encode() in p.read_bytes() for p in stored
    ), "the fact must actually land in the session bank"
    await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fragment_index", range(6))
async def test_a_credential_in_a_store_is_refused_with_field_kind_span_and_masked_retention(
    tmp_path, fragment_index
):
    from tests.unit.test_memory_guard_shapes import SECRET_CORPUS

    fragment, kind = SECRET_CORPUS[fragment_index]
    text = f"Feedback item 3: {fragment} (seen in the screenshot)"
    session, transcript = await _run_memory_store(tmp_path, {"op": "store", "text": text})
    result = _tool_result_text(transcript)
    assert f"text: {kind} at chars " in result, result
    from local_operator.memory import SensitiveValueError, guard

    with pytest.raises(SensitiveValueError) as caught:
        guard(text, field="text")
    span = next(s for s in caught.value.spans if s.kind == kind)
    secret = text[span.start : span.end]
    assert f"{span.start}-{span.end}" in result
    serialized = json.dumps([entry.payload for entry in transcript.entries()])
    assert secret not in serialized
    for path in transcript.directory.iterdir():
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    # The agent's own transcript keeps the rest of the call, masked at the span only.
    assert "Feedback item 3: " in serialized and "(seen in the screenshot)" in serialized
    assert f"[redacted:{kind}]" in serialized
    assert not (session._config_dir / "memory").exists()
    await session.dispose()
