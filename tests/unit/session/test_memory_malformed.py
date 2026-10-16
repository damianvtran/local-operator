"""R6/Q9: malformed provider JSON is private execution input, never retained output.

These are real Session runs with locally scripted provider bytes, not paid model
behavior. Instrument parser diagnostics independently of caplog, and check every
retained carrier as well as the private-call identity control.
"""

import io
import json
import logging

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
from local_operator.tools.memory_tools import build_memory_tool

CASES = [
    ("incomplete-escaped", '{"op":"store","scope":"global","text":"SSN 1\\u200b23-45-6789'),
    ("trailing-comma", '{"op":"store","scope":"global","text":"SSN 1\\u200b23-45-6789",}'),
    ("incomplete-opaque", '{"op":"store","scope":"global","text":"' + "ab3f" * 10),
    ("plain-control", '{"text":"SSN 123-45-6789'),
]
# Valid JSON but invalid protocol shapes are equally untrusted raw carriers.
CASES += [
    ("json-string", json.dumps("ab3f" * 10)),
    ("json-list", json.dumps(["ab3f" * 10])),
    ("unknown-value", json.dumps({"op": "invalid", "scope": "invalid", "unknown": "ab3f" * 10})),
    (
        "unknown-key",
        json.dumps({"op": "store", "scope": "global", "text": "benign", "ab3f" * 10: "benign"}),
    ),
    ("opaque-op", json.dumps({"op": "ab3f" * 10, "scope": "global", "text": "benign"})),
    ("opaque-scope", json.dumps({"op": "store", "scope": "ab3f" * 10, "text": "benign"})),
    (
        "bad-intent-type",
        json.dumps({"op": "store", "scope": "global", "text": "benign", "i": ["ab3f" * 10]}),
    ),
]


def test_undecodable_memory_raw_is_fail_closed_but_private_and_nonmemory_bytes_stay_original():
    for _, raw in CASES:
        call = ToolCall(name="memory", arguments={}, raw_arguments=raw)
        message = Message(
            role="assistant", content=[], tool_calls=[call], provider_payload={"alternate": raw}
        )
        safe, kept_raw, masked = retained_arguments("memory", {}, raw)
        assert masked and safe == {"memory_arguments_redacted": True}
        assert kept_raw is not None and raw not in kept_raw
        retained = retained_message(message)
        assert retained.provider_payload == {"memory_arguments_redacted": True}
        assert raw not in json.dumps(encode_message_payload(message))
        assert call.raw_arguments == raw and message.tool_calls[0].arguments == {}
        # Nonmemory malformed raw remains byte-identical; this is not a global
        # argument repair/filter that would change ordinary tool semantics.
        ordinary = Message(
            role="assistant",
            content=[],
            tool_calls=[ToolCall(name="echo", arguments={}, raw_arguments=raw)],
        )
        assert retained_message(ordinary) is ordinary
        assert retained_arguments("echo", {}, raw) == ({}, raw, False)
    valid = {"op": "store", "text": "Uses secret GITHUB_TOKEN"}
    assert retained_arguments("memory", valid, json.dumps(valid)) == (
        valid,
        json.dumps(valid),
        False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("label,raw", CASES, ids=[x[0] for x in CASES])
async def test_real_session_malformed_raw_has_no_journal_event_database_or_warning_residue(
    tmp_path, label, raw
):
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    parser_logger = logging.getLogger("local_operator.harness.loop")
    parser_logger.addHandler(handler)
    events = []
    n = 0

    def stream(request, signal):
        nonlocal n
        n += 1

        async def produce():
            if n == 1:
                yield StreamToolCallDelta(
                    index=0, id="synthetic-malformed", name="memory", argument_delta=raw
                )
                yield StreamEndEvent(
                    stop_reason="toolUse", provider_payload={"raw_response": raw, "alternate": raw}
                )
            else:
                yield StreamTextDelta(delta="Refused safely")
                yield StreamEndEvent(stop_reason="stop")

        return produce()

    transcript = Transcript(tmp_path / "sessions" / "malformed")
    session = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[build_memory_tool()],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="malformed",
        yolo=True,
        cwd=str(tmp_path),
    )
    session.subscribe(events.append)
    try:
        await session.prompt("Retain the synthetic test fact")
        assert n == 2
        try:
            json.loads(raw)
        except json.JSONDecodeError:
            assert "does not parse" in buffer.getvalue(), "loghandler must observe parser warnings"
        retained = json.dumps([entry.payload for entry in transcript.entries()])
        published = json.dumps([event.model_dump(mode="json") for event in events], default=str)
        all_text = retained + published + buffer.getvalue()
        assert "45-6789" not in all_text and "ab3f" * 10 not in all_text
        assert "memory_arguments_redacted" in retained
        assert not (session._config_dir / "memory").exists()
        for path in transcript.directory.iterdir():
            if path.is_file():
                body = path.read_bytes()
                assert b"45-6789" not in body and (b"ab3f" * 10) not in body
    finally:
        parser_logger.removeHandler(handler)
        handler.close()
        await session.dispose()


def test_parser_warning_never_echoes_raw_or_provider_controlled_tool_name(caplog):
    from local_operator.harness.loop import AgentLoop

    raw = '{"text":"SSN 123-45-6789'
    name = "malformed-name-SSN-123-45-6789"
    call = AgentLoop._assemble_tool_call({"arg_parts": [raw], "name": name, "id": "test-call"})
    assert call.raw_arguments == raw and call.name == name
    assert "does not parse" in caplog.text
    assert "123-45-6789" not in caplog.text and "malformed-name" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["global", "session", "team", "agent"])
async def test_real_session_legacy_scope_shape_is_redacted_without_promotion(tmp_path, scope):
    text = "Synthetic legacy personal locator must not be promoted"
    args = {"op": "store", "scope": scope, "text": text}
    raw = json.dumps(args)
    calls = 0
    events = []

    def stream(request, signal):
        nonlocal calls
        calls += 1

        async def produce():
            if calls == 1:
                yield StreamToolCallDelta(index=0, id="legacy", name="memory", argument_delta=raw)
                yield StreamEndEvent(stop_reason="toolUse", provider_payload={"alternate": raw})
            else:
                yield StreamTextDelta(delta="Rejected legacy shape")
                yield StreamEndEvent(stop_reason="stop")

        return produce()

    transcript = Transcript(tmp_path / "sessions" / "legacy")
    session = Session(
        model=ModelSpec(provider="test", model_id="deterministic", context_window=100000),
        stream_fn=stream,
        tools=[build_memory_tool()],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable"],
        session_id="legacy",
        yolo=True,
        cwd=str(tmp_path),
    )
    session.subscribe(events.append)
    try:
        await session.prompt("Exercise synthetic compatibility refusal")
        assert calls == 2
        retained = json.dumps([e.payload for e in transcript.entries()])
        published = json.dumps([e.model_dump(mode="json") for e in events], default=str)
        assert text not in retained + published and "memory_arguments_redacted" in retained
        assert "Invalid memory contract" in retained and "No scope argument" in retained
        assert not (session._config_dir / "memory").exists()
    finally:
        await session.dispose()
