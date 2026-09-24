"""The replay bound: what a later turn re-sends, and what it must never touch.

Two properties are load-bearing and each has its own class of test here:

1. The bound reaches the WIRE. Bounding ``ToolCall.arguments`` while leaving
   ``raw_arguments`` intact is a no-op, because the client replays the raw bytes
   verbatim whenever they parse — so the proof is taken through
   ``_replayable_tool_arguments_json``, the function the wire actually calls,
   rather than against the field a reader happens to look at.
2. The bound never touches the TRANSCRIPT. ``_default_convert_to_llm`` hands out
   the transcript's own ``Message`` objects, so an elision applied in place would
   edit the durable record and poison the compaction cache, the prune pass and
   every later render that reads the same object. Every test that bounds
   something also asserts the input is byte-identical afterwards.

The sizes are chosen to be realistic rather than pathological: a huge value with
one newline in it is retained almost in full by the shared clipper (measured on
the operator's own store: p50 0.99 of the budget), and the pathological shape has
its own test at the end so the behaviour is pinned rather than assumed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.replay_bound import (
    DEFAULT_REPLAY_BOUND_CHARS,
    REPLAY_ARGUMENT_MARKER,
    REPLAY_RESULT_MARKER,
    bound_replay_payloads,
)
from local_operator.harness.types import (
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)
from local_operator.providers.clients import _replayable_tool_arguments_json

MODEL = ModelSpec(provider="test", model_id="test-model", context_window=200_000)
BOUND = 2048


def lines(count: int, prefix: str = "output line") -> str:
    """``count`` newline-terminated lines — the shape real tool output has."""
    return "".join(f"{prefix} {index}: value-{index}\n" for index in range(count))


def tool_row(text: str, *, tool_call_id: str = "c1", name: str = "bash") -> Message:
    return Message(
        role="tool",
        content=[TextContent(text=text)],
        tool_call_id=tool_call_id,
        tool_name=name,
    )


def text_at(message: Message, index: int = 0) -> str:
    """The text of ``message``'s block ``index``, narrowed.

    ``Message.content`` is ``list[TextContent | ImageContent]``, so the text of a
    block is only reachable behind a narrowing check — this is that check, in one
    place rather than repeated as an ``assert isinstance`` at fifteen call sites.
    """
    block = message.content[index]
    assert isinstance(block, TextContent), f"block {index} is not text"
    return block.text


def assistant_call(
    arguments: dict[str, Any], *, call_id: str = "c1", name: str = "write", raw: str | None = None
) -> Message:
    return Message(
        role="assistant",
        content=[TextContent(text="working")],
        tool_calls=[
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                raw_arguments=json.dumps(arguments) if raw is None else raw,
            )
        ],
    )


# --------------------------------------------------------------------------
# Tool results
# --------------------------------------------------------------------------


def test_an_oversized_result_is_elided_on_the_copy_and_kept_in_the_input():
    original = lines(400)  # ~9.6k chars
    row = tool_row(original)

    out = bound_replay_payloads([row], bound_chars=BOUND)

    bounded = out[0]
    assert bounded is not row, "the bounded row must be a copy"
    assert text_at(row, 0) == original, "the input row must be untouched"
    assert bounded.tool_call_id == "c1" and bounded.tool_name == "bash"
    assert isinstance(bounded.content[0], TextContent)
    text = text_at(bounded, 0)
    assert len(text) <= BOUND
    assert "elided on replay" in text
    assert text.startswith("output line 0: value-0\n"), "the head is what the model acts on"
    assert text.rstrip().endswith("output line 399: value-399"), "the tail is the verdict"
    assert bounded.id != row.id, (
        "the token memo is keyed on message.id; a copy sharing it would be "
        "answered with the original's size"
    )


def test_a_result_within_the_bound_is_passed_through_by_identity():
    row = tool_row("short")
    out = bound_replay_payloads([row], bound_chars=BOUND)
    assert out[0] is row
    assert out is not [row]


def test_the_default_bound_is_the_number_the_tools_layer_already_targets():
    """One number, one meaning: 8 KiB is ``builtin.TOOL_OUTPUT_LIMIT_CHARS``."""
    from local_operator.tools.builtin import TOOL_OUTPUT_LIMIT_CHARS

    assert DEFAULT_REPLAY_BOUND_CHARS == TOOL_OUTPUT_LIMIT_CHARS


def test_a_result_the_tools_layer_would_have_capped_is_left_alone_at_the_default():
    """The default changes nothing for a row the tools layer already bounded."""
    from local_operator.tools.builtin import TOOL_OUTPUT_LIMIT_CHARS

    row = tool_row(lines(200))  # ~4.4k chars, under the shared 8 KiB budget
    assert len(text_at(row, 0)) < TOOL_OUTPUT_LIMIT_CHARS
    out = bound_replay_payloads([row])
    assert out[0] is row
    assert TOOL_OUTPUT_LIMIT_CHARS == DEFAULT_REPLAY_BOUND_CHARS


def test_a_leaked_oversized_result_is_caught_at_the_default():
    """The backstop's actual job: rows that escaped the tool-level cap."""
    row = tool_row(lines(40000))  # ~1.2 MB, the shape that leaked in the store
    out = bound_replay_payloads([row])
    assert len(text_at(out[0], 0)) <= DEFAULT_REPLAY_BOUND_CHARS


def test_every_text_block_is_bounded_and_images_are_left_alone():
    frame = ImageContent(data="A" * 64, mime_type="image/png")
    row = Message(
        role="tool",
        content=[TextContent(text=lines(400)), frame, TextContent(text=lines(400))],
        tool_call_id="c1",
    )
    out = bound_replay_payloads([row], bound_chars=BOUND)
    bounded = out[0]
    assert bounded.content[1] is frame
    for index in (0, 2):
        assert len(text_at(bounded, index)) <= BOUND


def test_error_results_keep_their_head_and_tail():
    """An error's first line names the failure and its last names the remedy."""
    text = "Traceback (most recent call last):\n" + lines(500, "frame") + "ValueError: bad input\n"
    row = tool_row(text)
    row.is_error = True
    out = bound_replay_payloads([row], bound_chars=BOUND)
    assert text_at(out[0], 0).startswith("Traceback (most recent call last):")
    assert text_at(out[0], 0).rstrip().endswith("ValueError: bad input")


# --------------------------------------------------------------------------
# Tool-call arguments — the path that had no bound at all
# --------------------------------------------------------------------------


def test_an_oversized_argument_is_bounded_on_the_wire_not_just_in_the_field():
    body = lines(400)
    message = assistant_call({"path": "a.py", "content": body})

    out = bound_replay_payloads([message], bound_chars=BOUND)

    bounded = out[0].tool_calls[0]
    assert message.tool_calls[0].arguments["content"] == body, "input untouched"
    assert bounded.id == "c1" and bounded.name == "write", "pairing fields survive"
    assert bounded.arguments["path"] == "a.py", "a short value is not scarred"
    assert len(bounded.arguments["content"]) <= BOUND
    assert "elided on replay" in bounded.arguments["content"]
    assert "this call already ran" in bounded.arguments["content"]
    # THE proof that matters: the client prefers raw_arguments when they parse,
    # so a bound that left them alone would put the original bytes on the wire.
    wire = json.loads(_replayable_tool_arguments_json(bounded))
    assert wire == bounded.arguments
    assert len(wire["content"]) <= BOUND


def test_an_argument_payload_reachable_only_through_raw_arguments_is_bounded():
    """``arguments`` empty, the payload in ``raw_arguments``: the wire uses raw."""
    payload = {"command": "python - <<'PY'\n" + lines(400) + "PY\n"}
    message = Message(
        role="assistant",
        tool_calls=[
            ToolCall(id="c9", name="bash", arguments={}, raw_arguments=json.dumps(payload))
        ],
    )
    out = bound_replay_payloads([message], bound_chars=BOUND)
    bounded = out[0].tool_calls[0]
    assert len(bounded.arguments["command"]) <= BOUND
    wire = json.loads(_replayable_tool_arguments_json(bounded))
    assert len(wire["command"]) <= BOUND
    assert "elided on replay" in wire["command"]


def test_nested_argument_bulk_is_bounded():
    message = assistant_call({"edits": [{"path": "a.py", "new_text": lines(400)}], "count": 3})
    out = bound_replay_payloads([message], bound_chars=BOUND)
    args = out[0].tool_calls[0].arguments
    assert args["count"] == 3
    assert len(args["edits"][0]["new_text"]) <= BOUND
    assert args["edits"][0]["path"] == "a.py"


def test_a_short_call_is_passed_through_by_identity():
    message = assistant_call({"path": "a.py", "content": "x"})
    out = bound_replay_payloads([message], bound_chars=BOUND)
    assert out[0] is message


def test_an_unparseable_raw_fragment_is_left_exactly_as_it_is():
    """A mid-call abort stores a truncated fragment; salvaging is not this
    module's job and rewriting it would invent an argument list."""
    message = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="bash", arguments={}, raw_arguments='{"command": "ech')],
    )
    out = bound_replay_payloads([message], bound_chars=BOUND)
    assert out[0] is message


def test_no_raw_arguments_stays_none_so_the_client_re_encodes():
    message = assistant_call({"path": "a.py", "content": lines(400)}, raw=None)
    message.tool_calls[0].raw_arguments = None
    out = bound_replay_payloads([message], bound_chars=BOUND)
    assert out[0].tool_calls[0].raw_arguments is None
    assert len(json.dumps(_replayable_tool_arguments_json(out[0].tool_calls[0]))) > 0


# --------------------------------------------------------------------------
# Interaction with the rest of the machinery
# --------------------------------------------------------------------------


def test_the_bounded_copy_is_estimated_at_its_own_size():
    row = tool_row(lines(400))
    assert estimate_tokens(row) > estimate_tokens(
        bound_replay_payloads([row], bound_chars=BOUND)[0]
    )


def test_estimating_the_copy_does_not_change_the_originals_estimate():
    """The memo is a process-wide dict keyed on ``message.id``; the derived id is
    what keeps the two sizes from overwriting each other."""
    row = tool_row(lines(400))
    before = estimate_tokens(row)
    copy = bound_replay_payloads([row], bound_chars=BOUND)[0]
    estimate_tokens(copy)
    assert estimate_tokens(row) == before


def test_zero_disables_the_bound():
    row = tool_row(lines(400))
    assert bound_replay_payloads([row], bound_chars=0)[0] is row


def test_message_order_and_count_are_never_changed():
    """No provider accepts a tool call whose result is missing, so the bound may
    shorten a row but may never remove one."""
    messages = [
        Message.user("go"),
        assistant_call({"path": "a.py", "content": lines(400)}),
        tool_row(lines(400)),
        Message.user("again"),
    ]
    out = bound_replay_payloads(messages, bound_chars=BOUND)
    assert len(out) == len(messages)
    assert [m.role for m in out] == [m.role for m in messages]


@pytest.mark.parametrize("marker", [REPLAY_RESULT_MARKER, REPLAY_ARGUMENT_MARKER])
def test_the_result_marker_is_not_the_tool_layers_marker(marker):
    """``... [output truncated] ...`` promises bytes in a spill store that can be
    expanded by handle. Replay elision promises no such thing — the bytes are in
    the transcript, which the model cannot address — so the two must not read the
    same, or the model spends a turn expanding a handle that does not exist."""
    from local_operator.text_bounds import OUTPUT_TRUNCATION_MARKER

    assert "elided on replay" in marker
    assert OUTPUT_TRUNCATION_MARKER.strip() not in marker


def test_a_single_enormous_line_is_still_bounded_and_says_so():
    """The pathological shape: one line with nothing to snap to. The shared
    clipper keeps less than the budget rather than half a line, and the marker
    reports the real count, so the loss is visible instead of silent."""
    row = tool_row("x" * 50_000)
    out = bound_replay_payloads([row], bound_chars=BOUND)
    text = text_at(out[0], 0)
    assert len(text) <= BOUND
    assert "elided on replay" in text


def test_a_bounded_assistant_row_cannot_replay_its_native_continuation():
    """The provider-native payload carries a fingerprint of the VISIBLE content,
    so bounding the arguments invalidates it and the client rebuilds from the
    bounded text. Without that, an Anthropic/Responses-native row would replay
    the original unbounded arguments alongside the elided ones — the bound would
    be defeated by the very field it cannot see."""
    from local_operator.providers.replay import native_payload, replay_items

    body = lines(400)
    calls = [{"id": "c1", "name": "write", "args": {"path": "a.py", "content": body}}]
    message = assistant_call({"path": "a.py", "content": body})
    message.provider_payload = native_payload(
        MODEL,
        "https://example.invalid/v1",
        "openai_chat",
        [{"type": "text", "text": "working"}],
        message.text,
        calls,
    )

    assert replay_items(
        message, MODEL, "https://example.invalid/v1", "openai_chat"
    ), "the fixture must have a live native payload to lose"
    bounded = bound_replay_payloads([message], bound_chars=BOUND)[0]
    assert replay_items(bounded, MODEL, "https://example.invalid/v1", "openai_chat") is None


def test_a_bound_smaller_than_the_marker_still_states_the_loss():
    """The marker is the one part that may not be elided: a bound honoured by
    hiding the elision would be worse than no bound at all. Such a bound is
    served by the marker plus one character from each end."""
    row = tool_row(lines(400))
    out = bound_replay_payloads([row], bound_chars=8)
    text = text_at(out[0])
    assert "elided on replay" in text
    assert len(text) < 200, "the marker is the result, not the marker plus a budget"
    assert len(text) < len(lines(400))


def test_a_call_under_the_bound_is_not_parsed_at_all(monkeypatch):
    """This runs for every call in the whole history on every request, so the
    common path must be a length check rather than a parse of the model's
    argument JSON. A raw payload no longer than the bound cannot contain a
    string value longer than the bound."""
    from local_operator.harness import replay_bound as module

    parses: list[str] = []
    real_loads = json.loads

    def counting_loads(value, *args, **kwargs):
        parses.append(value)
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(module.json, "loads", counting_loads)

    short = assistant_call({"path": "a.py", "content": "x" * 100})
    assert bound_replay_payloads([short], bound_chars=BOUND)[0] is short
    assert parses == [], "a call within the bound must not be parsed"

    long_args = {"path": "a.py", "content": lines(400)}
    long = assistant_call(long_args)
    out = bound_replay_payloads([long], bound_chars=BOUND)
    assert parses, "an oversized payload is parsed to bound the EFFECTIVE payload"
    assert len(out[0].tool_calls[0].arguments["content"]) <= BOUND
    assert (
        len(json.loads(_replayable_tool_arguments_json(out[0].tool_calls[0]))["content"]) <= BOUND
    )
