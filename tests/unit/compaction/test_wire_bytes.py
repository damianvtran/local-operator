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
