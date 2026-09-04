"""The byte budget: the transport ruler, the byte trigger, the shed, and the
byte-side anti-thrash band.

Regression suite for a session that wedged permanently on
``invalid request (HTTP 413): Request exceeds the maximum size``. It carried 42
screenshots totalling 33.9 MB of base64 against Anthropic's 32 MB cap, while
``estimate_messages_tokens`` read 154,690 — 15.5% of a 1M window — so the token
trigger never fired and ``/compact`` was the only escape. Every later turn,
including ``Continue``, failed identically.

The fixture is SYNTHESIZED at the measured size distribution rather than
committed: the real transcript is only 902 KB on disk because images are
attachment references hydrated on replay, so a faithful reproduction needs the
sizes, not a 34 MB blob in the repo.
"""

from __future__ import annotations

import json

import pytest

from local_operator.compaction.pruning import shed_frames_to_wire_budget
from local_operator.compaction.thresholds import (
    DEFAULT_WIRE_BYTES_BUDGET,
    DEFAULT_WIRE_BYTES_TRIGGER,
    WIRE_RECOVERY_BAND,
    CompactionSettings,
    cleared_wire_headroom,
    resolve_wire_bytes_budget,
    resolve_wire_bytes_trigger,
    should_compact,
)
from local_operator.compaction.tokens import (
    estimate_messages_tokens,
    estimate_wire_bytes,
)
from local_operator.harness.types import ImageContent, Message, TextContent, ToolCall

#: Median base64 length of the 42 frames in the wedged session (min 709,616 /
#: median 803,888 / max 981,728). Using the median keeps the fixture's totals
#: within a few percent of the real 34.2 MB at 42 frames.
MEASURED_FRAME_B64 = 803_888

#: What the real session carried.
MEASURED_FRAME_COUNT = 42


def _frame(index: int, size: int = MEASURED_FRAME_B64) -> Message:
    """One screenshot-bearing user message of realistic size."""
    return Message(
        role="user",
        content=[TextContent(text=f"observation {index}"), ImageContent(data="A" * size)],
    )


def _wedged_history(frames: int = MEASURED_FRAME_COUNT) -> list[Message]:
    """A history shaped like the session that 413'd: alternating screenshot
    observations and short replies."""
    messages: list[Message] = [Message.user("start the task")]
    for index in range(frames):
        messages.append(_frame(index))
        messages.append(Message.assistant(f"reply {index}"))
    return messages


# ---------------------------------------------------------------------------
# The ruler
# ---------------------------------------------------------------------------


def test_wire_bytes_counts_text_images_and_tool_arguments_exactly() -> None:
    """Exact, not estimated: the sum of what actually goes on the wire."""
    message = Message(
        role="user",
        content=[TextContent(text="x" * 100), ImageContent(data="d" * 5_000)],
    )
    assert estimate_wire_bytes([message]) == 5_100

    with_call = Message.assistant("hello")
    with_call.tool_calls = [ToolCall(id="1", name="bash", raw_arguments='{"command":"ls"}')]
    # 5 text + 4 name + 18 raw arguments.
    assert estimate_wire_bytes([with_call]) == 5 + 4 + len('{"command":"ls"}')

    assert estimate_wire_bytes([]) == 0


def test_wire_bytes_and_token_estimate_disagree_by_three_orders_of_magnitude() -> None:
    """The defect, pinned: the two rulers answer different questions.

    A flat per-image token charge is CORRECT for billing (providers price by
    pixel area) and useless as a size proxy. This is why the fix adds a third
    number instead of making ``IMAGE_TOKEN_ESTIMATE`` size-aware.
    """
    history = _wedged_history()
    wire = estimate_wire_bytes(history)
    tokens = estimate_messages_tokens(history)

    assert wire > 33_000_000, "fixture must reproduce the ~34 MB payload"
    assert tokens < 200_000, "and the honestly small token estimate that hid it"
    # ~670 real bytes per accounted token — the blindness, quantified.
    assert wire / tokens > 100


# ---------------------------------------------------------------------------
# The trigger (defect 1)
# ---------------------------------------------------------------------------


def test_byte_trigger_fires_where_the_token_trigger_cannot() -> None:
    """THE regression. 154,690 tokens on a 1M window is 15.5% — no token
    threshold can fire — while the request is 34 MB against a 32 MB cap."""
    settings = CompactionSettings()

    assert should_compact(154_690, 1_000_000, settings) is False
    assert should_compact(154_690, 1_000_000, settings, wire_bytes=34_280_000) is True


def test_byte_trigger_respects_disabled_and_off_exactly_like_the_token_trigger() -> None:
    """The byte term is an input to the ONE trigger, not a bypass around it."""
    assert (
        should_compact(0, 1_000_000, CompactionSettings(enabled=False), wire_bytes=10**9) is False
    )
    assert (
        should_compact(0, 1_000_000, CompactionSettings(strategy="off"), wire_bytes=10**9) is False
    )
    # Unknown window still never triggers, byte pressure or not.
    assert should_compact(0, 0, CompactionSettings(), wire_bytes=10**9) is False
    # Explicitly disabled byte trigger.
    settings = CompactionSettings(wire_bytes_trigger=0)
    assert should_compact(0, 1_000_000, settings, wire_bytes=10**9) is False


@pytest.mark.parametrize(
    "context_tokens,window,expected",
    [
        # The resolved trigger is min(threshold_percent * window,
        # threshold_tokens), so 600k binds on a 1M window, not 80%.
        (0, 1_000_000, False),
        (100_000, 1_000_000, False),
        (599_999, 1_000_000, False),
        (600_000, 1_000_000, False),  # strictly greater-than
        (600_001, 1_000_000, True),
        (600_001, 10_000_000, True),
        (79_999, 100_000, False),  # 80% binds on a small window
        (80_001, 100_000, True),
    ],
)
def test_omitting_wire_bytes_reproduces_the_previous_answer(
    context_tokens: int, window: int, expected: bool
) -> None:
    """Backward compatibility: ``wire_bytes`` defaults to 0, and 0 is inert.

    A text-only session must behave byte-identically to one predating the byte
    trigger, both by omitting the argument and by passing the default.
    """
    settings = CompactionSettings()
    assert should_compact(context_tokens, window, settings) is expected
    assert should_compact(context_tokens, window, settings, wire_bytes=0) is expected


def test_byte_trigger_is_monotonic_in_bytes() -> None:
    """The session's cheap pre-gate depends on this: more bytes can only turn
    a False into a True, never the reverse."""
    settings = CompactionSettings()
    trigger = resolve_wire_bytes_trigger(settings)
    seen_true = False
    for wire in range(0, trigger * 2, max(1, trigger // 8)):
        result = should_compact(0, 1_000_000, settings, wire_bytes=wire)
        if seen_true:
            assert result, "trigger went back to False as bytes grew"
        seen_true = seen_true or result
    assert seen_true


# ---------------------------------------------------------------------------
# The resolvers
# ---------------------------------------------------------------------------


def test_resolvers_are_the_single_source_of_the_two_numbers() -> None:
    settings = CompactionSettings()
    assert resolve_wire_bytes_budget(settings) == DEFAULT_WIRE_BYTES_BUDGET == 24_000_000
    assert resolve_wire_bytes_trigger(settings) == DEFAULT_WIRE_BYTES_TRIGGER == 16_000_000

    # Non-positive disables, and normalises to 0 so callers test one way.
    assert resolve_wire_bytes_budget(CompactionSettings(wire_bytes_budget=0)) == 0
    assert resolve_wire_bytes_budget(CompactionSettings(wire_bytes_budget=-5)) == 0
    assert resolve_wire_bytes_trigger(CompactionSettings(wire_bytes_trigger=0)) == 0


def test_soft_trigger_is_clamped_to_the_hard_budget() -> None:
    """A trigger above the ceiling would invert the design: the render seam
    would amputate frames before a proper compaction pass ever fired."""
    settings = CompactionSettings(wire_bytes_budget=10_000_000, wire_bytes_trigger=50_000_000)
    assert resolve_wire_bytes_trigger(settings) == 10_000_000


# ---------------------------------------------------------------------------
# The shed
# ---------------------------------------------------------------------------


def test_shed_keeps_the_maximum_number_of_frames_that_fit() -> None:
    """The measured outcome: 42 frames at 34 MB shed to 28 frames under 24 MB.

    The shed drops the FEWEST frames that fit, because every frame is evidence
    the user may still need — against the 0 a sticky image degrade leaves and
    the 17 a full compaction leaves.
    """
    history = _wedged_history()
    budget = DEFAULT_WIRE_BYTES_BUDGET
    assert estimate_wire_bytes(history) > budget

    out, dropped = shed_frames_to_wire_budget(history, budget=budget)

    assert estimate_wire_bytes(out) <= budget
    remaining = sum(1 for m in out if any(isinstance(b, ImageContent) for b in m.content))
    assert remaining == MEASURED_FRAME_COUNT - dropped
    assert 25 <= remaining <= 30, f"expected ~28 frames kept, got {remaining}"

    # One fewer frame dropped would NOT have fit: the shed is minimal.
    from local_operator.compaction.pruning import prune_stale_frames

    tighter, _ = prune_stale_frames(history, keep_recent_frames=remaining + 1)
    assert estimate_wire_bytes(tighter) > budget


def test_shed_never_removes_a_message_so_pairing_and_alternation_survive() -> None:
    """A transport guard runs on arbitrary history and cannot know what is
    mid-tool-call, so it may only blank images, never drop messages."""
    history = _wedged_history()
    out, dropped = shed_frames_to_wire_budget(history, budget=DEFAULT_WIRE_BYTES_BUDGET)

    assert dropped > 0
    assert len(out) == len(history)
    assert [m.role for m in out] == [m.role for m in history]
    assert [m.id for m in out] == [m.id for m in history]


def test_shed_is_a_no_op_under_budget_and_when_disabled() -> None:
    """The guarantee that every session under budget is unaffected."""
    small = _wedged_history(frames=2)
    assert estimate_wire_bytes(small) < DEFAULT_WIRE_BYTES_BUDGET

    out, dropped = shed_frames_to_wire_budget(small, budget=DEFAULT_WIRE_BYTES_BUDGET)
    assert dropped == 0
    assert all(a is b for a, b in zip(out, small)), "under budget must not copy"

    out, dropped = shed_frames_to_wire_budget(_wedged_history(), budget=0)
    assert dropped == 0


def test_shed_terminates_when_there_is_nothing_left_to_shed() -> None:
    """A text-only history over budget cannot be repaired here; the loop must
    exit rather than spin, and the caller must see 'still over'."""
    huge_text = [Message.user("x" * 30_000_000)]
    out, dropped = shed_frames_to_wire_budget(huge_text, budget=DEFAULT_WIRE_BYTES_BUDGET)

    assert dropped == 0
    assert estimate_wire_bytes(out) > DEFAULT_WIRE_BYTES_BUDGET


def test_shed_is_monotone_down_to_zero_frames() -> None:
    """Tighter budgets shed at least as much, and a budget nothing can satisfy
    ends at zero frames rather than looping."""
    history = _wedged_history()
    previous = -1
    for budget in (24_000_000, 16_000_000, 8_000_000, 1_000_000, 1):
        _, dropped = shed_frames_to_wire_budget(history, budget=budget)
        assert dropped >= previous
        previous = dropped
    assert previous == MEASURED_FRAME_COUNT


# ---------------------------------------------------------------------------
# The byte-side anti-thrash band (risk 4)
# ---------------------------------------------------------------------------


def test_wire_recovery_band_withholds_continuation_when_still_over_budget() -> None:
    """The dead-loop guard, restated in the byte trigger's units.

    ``RECOVERY_BAND`` is defined on TOKENS. A byte-triggered pass can leave the
    token residual far inside the token band (154,690 tokens is 15% of a 1M
    window) while the request is still over the byte budget — continuing on
    that re-fires the byte trigger next turn on a context nothing shrank, which
    is exactly the live dead loop ``RECOVERY_BAND`` was added to prevent.
    """
    settings = CompactionSettings()
    trigger = resolve_wire_bytes_trigger(settings)

    # Residual still above the trigger: no continuation.
    assert cleared_wire_headroom(trigger + 1, settings) is False
    # Residual under the trigger but INSIDE the band: still no continuation —
    # the pass barely helped and would re-fire.
    assert cleared_wire_headroom(int(trigger * 0.95), settings) is False
    # At the band exactly, and below it: real headroom.
    assert cleared_wire_headroom(int(trigger * WIRE_RECOVERY_BAND), settings) is True
    assert cleared_wire_headroom(1_000, settings) is True


def test_wire_recovery_band_is_inert_when_the_byte_trigger_is_off() -> None:
    """It may only ever WITHHOLD a continuation an existing session would have
    scheduled; with bytes disabled it must not become a second veto."""
    settings = CompactionSettings(wire_bytes_trigger=0)
    assert cleared_wire_headroom(10**12, settings) is True


# ---------------------------------------------------------------------------
# Argument sizing must never under-count the real wire (review R7 / QA Q5)
# ---------------------------------------------------------------------------
#
# ``estimate_wire_bytes`` decides whether a request is shed before sending, so
# an UNDER-count is the one error it must never make: it means believing an
# oversize payload fits and sending it, which is the 413 this whole change
# exists to prevent.
#
# The first remediation sized strings with ``len(s)`` — characters, not bytes,
# and blind to escapes — which flipped the bias from +1.2% (over) to -1.9%
# (under) across 487,652 real tool calls, reaching -75% on CJK/emoji and -3%
# to -12% on ordinary ASCII ``write`` calls.
#
# The reference below is the encoding the provider clients ACTUALLY use:
# ``httpx._content.encode_json`` serializes a ``json=`` body with
# ``ensure_ascii=False``, ``separators=(",", ":")`` and encodes UTF-8, and all
# four call sites in ``providers/clients.py`` pass ``json=``. Sizing against
# ``json.dumps`` defaults (``ensure_ascii=True``) would measure a different
# encoding than the one that leaves the machine.


def _wire_json_bytes(value: object) -> int:
    """Exactly what httpx puts on the wire for ``value`` inside a JSON body."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


#: Payload shapes chosen so each one breaks a DIFFERENT wrong assumption:
#: character-counting, escape-blindness, and flat numeric charges.
_ARGUMENT_SHAPES: dict[str, object] = {
    "ascii": {"command": "ls -la /tmp && echo done"},
    # The real-world case that hit ASCII-only users: source text is dense in
    # newlines, tabs and quotes, each of which costs two bytes, not one.
    "write_call": {"path": "/a/b.py", "content": 'def f(x):\n\treturn "%s"\n' * 200},
    "escape_dense": {"s": 'a"b\\c\nd\te\rf\bg\fh' * 300},
    "cjk": {"text": "\u6587\u5b57\u5316" * 500},
    "emoji": {"text": "\U0001f600\U0001f601\U0001f602" * 300},
    "accented": {"text": "\u00e9\u00e8\u00ea\u00eb" * 500},
    # C0 controls with no short form become the six-byte \uXXXX.
    "control_chars": {"s": "".join(chr(code) for code in range(0x20)) * 40},
    "floats": {"values": [3.14159265358979, 1e300, -0.5, 1.0, 2.718281828]},
    "big_ints": {"values": [2**64, -(2**63), 0, 1, 999999999999999999]},
    "literals": {"a": True, "b": False, "c": None},
    "nested": {"a": {"b": [{"c": "\u00e9x"}, [1, 2.5, None], "d\ne"]}},
    "empty_containers": {"a": {}, "b": [], "c": ""},
    "mixed_realistic": {
        "path": "/src/\u6a21\u5757.py",
        "content": 'x = "\u00e9"\nif x:\n\tprint("\U0001f600")\n' * 100,
        "line": 42,
        "ratio": 0.75,
        "flags": [True, None],
    },
}


@pytest.mark.parametrize("shape", sorted(_ARGUMENT_SHAPES))
def test_argument_sizing_never_under_counts_the_real_wire(shape: str) -> None:
    """THE regression for R7/Q5: erring high is fine, erring low is not.

    Fails on 26bf8715 for every non-ASCII, escape-heavy and numeric shape.
    """
    from local_operator.compaction.tokens import _argument_bytes

    arguments = _ARGUMENT_SHAPES[shape]
    estimated = _argument_bytes(arguments)
    actual = _wire_json_bytes(arguments)

    assert estimated >= actual, (
        f"{shape}: estimate {estimated} UNDER the real wire {actual} "
        f"({(estimated - actual) / actual * 100:+.2f}%) — the guard would send "
        "an oversize request believing it fits"
    )


@pytest.mark.parametrize("shape", sorted(_ARGUMENT_SHAPES))
def test_argument_sizing_stays_close_to_the_real_wire(shape: str) -> None:
    """Never-under must not be bought with a wild over-estimate, which would
    shed screenshots no provider asked us to drop.

    The bias is a trailing separator charged per container, so the bound is
    generous only for tiny payloads where one byte is a large fraction.
    """
    from local_operator.compaction.tokens import _argument_bytes

    arguments = _ARGUMENT_SHAPES[shape]
    estimated = _argument_bytes(arguments)
    actual = _wire_json_bytes(arguments)

    assert estimated <= actual + 16 + actual * 0.02, (
        f"{shape}: estimate {estimated} is {(estimated - actual) / actual * 100:+.2f}% "
        f"over the real wire {actual}"
    )


def test_argument_sizing_is_exact_for_the_shapes_that_dominate_real_traffic() -> None:
    """A stronger claim where it can be made: for a flat object of strings the
    estimate matches the encoder byte for byte apart from the one trailing
    separator, so the residual bias is understood rather than merely bounded.
    """
    from local_operator.compaction.tokens import _argument_bytes

    for arguments in (
        {"content": "plain ascii"},
        {"content": "\u6587\u5b57"},
        {"content": 'quotes " and \\ and \n'},
    ):
        estimated = _argument_bytes(arguments)
        actual = _wire_json_bytes(arguments)
        assert estimated - actual == 1, f"{arguments!r}: bias {estimated - actual}, expected 1"


def test_a_cjk_history_is_not_waved_under_the_budget() -> None:
    """QA's minimal reproduction, pinned.

    A 300-call CJK history that the seam believed was 12,007,690 bytes went on
    the wire at 36,007,390 — under the 24 MB budget by the guard's reckoning
    and over Anthropic's 32 MB cap in reality.
    """
    messages: list[Message] = []
    for index in range(300):
        message = Message.assistant("")
        message.tool_calls = [
            ToolCall(id=str(index), name="w", arguments={"text": "\u6587" * 40_000})
        ]
        messages.append(message)

    seam = estimate_wire_bytes(messages)
    actual = sum(
        len(call.name) + _wire_json_bytes(call.arguments)
        for message in messages
        for call in message.tool_calls or ()
    )

    assert seam >= actual, "the seam under-counted a CJK history"
    assert not (
        seam <= DEFAULT_WIRE_BYTES_BUDGET < actual
    ), "the guard believes an over-cap payload fits"


def test_raw_arguments_are_still_preferred_when_present() -> None:
    """The provider's own rendering IS the wire, so it is used verbatim rather
    than re-derived — the structural sizer is only the resumed-session
    fallback, where ``raw_arguments`` has been dropped on the way to disk.
    """
    message = Message.assistant("")
    message.tool_calls = [
        ToolCall(id="1", name="bash", raw_arguments='{"command":"ls"}', arguments={"command": "ls"})
    ]
    # The raw string verbatim, NOT the structural estimate of the parsed dict —
    # which for this payload would be one byte larger.
    assert estimate_wire_bytes([message]) == len("bash") + len('{"command":"ls"}')


# ---------------------------------------------------------------------------
# A lone surrogate must never break sizing (agent review round 3, R10)
# ---------------------------------------------------------------------------
#
# ``\ud800`` is legal in a Python ``str`` AND legal JSON, so a model that emits
# one round-trips it through ``json.dumps``/``json.loads`` and it lands in
# ``transcript.jsonl`` verbatim. A plain ``encode("utf-8")`` raises on it — and
# this sizer runs inside ``_render_history``, which every wire path and
# ``/compact`` go through, so one stray codepoint made every later turn raise
# forever. That is the wedge this whole change exists to delete, reintroduced
# through the sizer meant to prevent it.

#: A lone high surrogate, the shape that reaches a transcript unescaped.
LONE_SURROGATE = "hello \ud800 world"


def test_a_lone_surrogate_is_legal_json_and_survives_a_transcript_round_trip() -> None:
    """The premise, pinned: this is reachable input, not a malformed edge case.

    If this ever stops holding the regression below is moot — but it holds,
    and it is why the sizer must tolerate the codepoint.
    """
    encoded = json.dumps({"text": LONE_SURROGATE})
    assert json.loads(encoded)["text"] == LONE_SURROGATE
    # And the strict encoder — the one the sizer used to call — refuses it.
    with pytest.raises(UnicodeEncodeError):
        json.dumps(LONE_SURROGATE, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize(
    "payload",
    [
        LONE_SURROGATE,
        "\ud800",  # bare, nothing around it
        "\udfff",  # the other end of the surrogate range
        "\ud83d\ude00",  # an unpaired pair, which is NOT the emoji
        "ok \ud800 \u6587 \U0001f600 mixed",  # beside legal non-ASCII
    ],
)
def test_sizing_never_raises_on_a_lone_surrogate(payload: str) -> None:
    """THE R10 regression: a size estimate must always return a number.

    Fails on 2b15c340 with UnicodeEncodeError.
    """
    from local_operator.compaction.tokens import (
        _argument_bytes,
        _string_bytes,
        _utf8_len,
    )

    assert _string_bytes(payload) > 0
    assert _utf8_len(payload) > 0
    assert _argument_bytes({"text": payload}) > 0

    message = Message(role="user", content=[TextContent(text=payload)])
    assert estimate_wire_bytes([message]) > 0


def test_a_surrogate_is_sized_as_a_lenient_encoder_would_emit_it() -> None:
    """``surrogatepass`` is the right NUMBER, not just a non-raising one."""
    from local_operator.compaction.tokens import _string_bytes

    # Three bytes for the surrogate itself, plus the surrounding ASCII.
    expected = len(LONE_SURROGATE.encode("utf-8", "surrogatepass"))
    assert _string_bytes(LONE_SURROGATE) == expected


def test_a_tool_call_carrying_a_surrogate_is_sized_on_both_branches() -> None:
    """Both argument branches must tolerate it — the parsed dict AND the
    pre-serialized ``raw_arguments`` string that live traffic carries."""
    parsed = Message.assistant("")
    parsed.tool_calls = [ToolCall(id="1", name="write", arguments={"t": LONE_SURROGATE})]
    assert estimate_wire_bytes([parsed]) > 0

    raw = Message.assistant("")
    raw.tool_calls = [
        ToolCall(id="1", name="write", raw_arguments=json.dumps({"t": LONE_SURROGATE}))
    ]
    assert estimate_wire_bytes([raw]) > 0


# ---------------------------------------------------------------------------
# raw_arguments is sized in BYTES (agent review round 3, R7 residual)
# ---------------------------------------------------------------------------
#
# ``raw_arguments`` is the branch carrying essentially all LIVE traffic
# (``harness/loop.py`` always populates it), and it kept the character count
# that R7 named: 6.42% of real calls under-counted, worst -29.79%.


@pytest.mark.parametrize(
    "arguments",
    [
        {"text": "plain ascii"},
        {"text": "\u6587\u5b57" * 100},
        {"text": "\U0001f600" * 80},
        {"text": "\u00e9\u00e8" * 100},
        {"path": "/src/\u6a21\u5757.py", "content": 'x = "\u00e9"\n' * 50},
    ],
)
def test_raw_arguments_are_sized_in_bytes_not_characters(arguments: dict[str, str]) -> None:
    """Fails on 2b15c340 for every non-ASCII payload."""
    raw = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    message = Message.assistant("")
    message.tool_calls = [ToolCall(id="1", name="w", raw_arguments=raw)]

    estimated = estimate_wire_bytes([message])
    actual = len("w".encode("utf-8")) + len(raw.encode("utf-8"))

    assert estimated >= actual, (
        f"raw_arguments under-counted by {(estimated - actual) / actual * 100:+.2f}% — "
        "this is the branch live traffic uses"
    )


def test_a_non_ascii_tool_name_is_sized_in_bytes() -> None:
    """The name rides the same wire as its arguments; an MCP server is free to
    use a non-ASCII tool name."""
    message = Message.assistant("")
    message.tool_calls = [ToolCall(id="1", name="\u6587\u5b57", raw_arguments="{}")]
    assert estimate_wire_bytes([message]) >= len("\u6587\u5b57".encode("utf-8")) + 2
