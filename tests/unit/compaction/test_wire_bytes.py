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

import base64
import io
import json
from typing import Sequence

import pytest
from PIL import Image

from local_operator.compaction.pruning import (
    downscale_stale_frames,
    fit_frames_to_wire_budget,
    shed_frames_to_wire_budget,
)
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
from local_operator.imaging import _CONTEXT_FRAME_CACHE, IMAGE_CONTEXT_EDGES

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
    # An unknown window disables the TOKEN term only: a request over the wire
    # cap is too large to send whether or not a context length is known for
    # the route, and that is the one configuration where the byte trigger is
    # the only trigger there is.
    assert should_compact(0, 0, CompactionSettings(), wire_bytes=10**9) is True
    assert should_compact(0, 0, CompactionSettings(), wire_bytes=0) is False
    # Explicitly disabled byte trigger.
    settings = CompactionSettings(wire_bytes_trigger=0)
    assert should_compact(0, 1_000_000, settings, wire_bytes=10**9) is False


@pytest.mark.parametrize(
    "context_tokens,window,expected",
    [
        # The resolved trigger is min(threshold_percent * window,
        # threshold_tokens), so 400k binds on a 1M window, not 80%.
        (0, 1_000_000, False),
        (100_000, 1_000_000, False),
        (399_999, 1_000_000, False),
        (400_000, 1_000_000, False),  # strictly greater-than
        (400_001, 1_000_000, True),
        (400_001, 10_000_000, True),
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
# The context downscale: the fidelity step of the byte ladder
# ---------------------------------------------------------------------------
#
# The shed tests above run on synthetic non-image bytes on purpose — the shed
# itself never decodes anything. The downscale is a decode, a resize and a
# re-encode, so these tests need REAL pixels; the frame fixture below is a
# deterministic screenshot-shaped PNG (gradient + speckle, which is what keeps
# its size in the hundreds of KB instead of the few KB a flat fill takes).

_FRAME_PNG: dict[tuple[int, int, int], bytes] = {}


def _frame_png(width: int = 1280, height: int = 800, seed: int = 11) -> bytes:
    """A deterministic desktop-screenshot-shaped frame, ~0.5-1 MB as PNG."""
    key = (width, height, seed)
    cached = _FRAME_PNG.get(key)
    if cached is not None:
        return cached
    import random

    rng = random.Random(seed)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    assert pixels is not None
    for y in range(height):
        base = (30 + y * 40 // height, 40 + y * 30 // height, 60 + y * 60 // height)
        for x in range(0, width, 2):
            pixels[x, y] = base
    for _ in range(width * height // 20):
        pixels[rng.randrange(width), rng.randrange(height)] = (
            rng.randrange(256),
            rng.randrange(256),
            rng.randrange(256),
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    _FRAME_PNG[key] = buffer.getvalue()
    return buffer.getvalue()


def _real_frame_messages(count: int) -> list[Message]:
    """``count`` user messages each carrying a real PNG frame."""
    encoded = base64.b64encode(_frame_png()).decode("ascii")
    return [
        Message(
            role="user",
            content=[TextContent(text=f"shot {index}"), ImageContent(data=encoded)],
        )
        for index in range(count)
    ]


def _real_history(count: int = 6) -> tuple[list[Message], list[Message]]:
    """``(history, frame messages)`` in the alternating shape the runner and a
    screen-driving session both build."""
    frames = _real_frame_messages(count)
    history: list[Message] = []
    for index, message in enumerate(frames):
        history.append(message)
        history.append(Message.assistant(f"ok {index}"))
    return history, frames


def _image_blocks_of(messages: Sequence[Message]) -> list[ImageContent]:
    """The image blocks of ``messages``, in order.

    A helper rather than an inline comprehension so every assertion below sees
    ``ImageContent`` (typed), not the message content union.
    """
    return [
        block
        for message in messages
        for block in message.content
        if isinstance(block, ImageContent)
    ]


def test_downscale_stale_frames_keeps_the_newest_and_never_removes_a_message() -> None:
    """The sibling contract to the prune's: the newest frame message is reused
    BY IDENTITY, victims are copied with the smaller bytes, frame-free
    messages are reused, and no message is ever removed."""
    history, frames = _real_history(3)
    _CONTEXT_FRAME_CACHE.clear()

    out, downscaled = downscale_stale_frames(history, keep_recent_frames=1, max_edge=1024)

    assert downscaled == 2
    assert len(out) == len(history)
    assert [m.id for m in out] == [m.id for m in history]
    assert out[4] is frames[2], "the newest frame message was copied or changed"
    assert out[2] is not frames[1]
    blocks = _image_blocks_of(out)
    originals = _image_blocks_of(frames)
    assert blocks[0].data != originals[0].data
    assert blocks[0].mime_type in {"image/png", "image/jpeg"}
    assert out[1] is history[1], "a frame-free message was not reused by identity"


def test_downscale_stale_frames_zero_downscales_everything_and_rejects_negative() -> None:
    history, frames = _real_history(2)
    _CONTEXT_FRAME_CACHE.clear()

    out, downscaled = downscale_stale_frames(history, keep_recent_frames=0, max_edge=1024)

    assert downscaled == 2
    blocks = _image_blocks_of(out)
    originals = _image_blocks_of(frames)
    assert blocks[0].data != originals[0].data
    assert blocks[1].data != originals[1].data
    with pytest.raises(ValueError):
        downscale_stale_frames(history, keep_recent_frames=-1, max_edge=1024)


def test_downscale_stale_frames_leaves_non_images_untouched() -> None:
    """The synthetic 'A' frames the shed tests use are valid base64 that is not
    an image: the downscale must degrade to a no-op rather than raise, so the
    shed below it can still act on exactly the history it always saw."""
    history = [_frame(0), Message.assistant("ok")]
    _CONTEXT_FRAME_CACHE.clear()

    out, downscaled = downscale_stale_frames(history, keep_recent_frames=0, max_edge=1024)

    assert downscaled == 0
    assert all(a is b for a, b in zip(out, history))


def test_fit_downscales_and_keeps_every_frame_when_a_rung_fits() -> None:
    """THE policy change, as one assertion set: over budget, the request comes
    back under it with EVERY frame still present — the newest at full
    fidelity — and ZERO frames dropped."""
    history, frames = _real_history(6)
    original = _image_blocks_of(frames)[-1].data
    total = estimate_wire_bytes(history)
    budget = int(total * 0.75)
    _CONTEXT_FRAME_CACHE.clear()

    out, downscaled, dropped, _rung = fit_frames_to_wire_budget(history, budget=budget)

    assert dropped == 0, "a frame was dropped although a rung fit"
    assert downscaled == 5
    assert estimate_wire_bytes(out) <= budget
    assert len(out) == len(history)
    assert [m.id for m in out] == [m.id for m in history]
    blocks = _image_blocks_of(out)
    assert len(blocks) == 6, "a frame became a notice"
    assert blocks[-1].data == original, "the newest frame lost fidelity"
    assert blocks[0].data != original, "no old frame was re-rendered"


def test_fit_stops_at_the_first_rung_that_fits() -> None:
    """The walk is ordered and stops as soon as the request fits: a budget one
    byte below the widest rung's total lands on the NEXT rung — not on a shed,
    and not on a deeper rung."""
    history, _frames = _real_history(6)
    _CONTEXT_FRAME_CACHE.clear()
    widest, _ = downscale_stale_frames(
        history, keep_recent_frames=1, max_edge=IMAGE_CONTEXT_EDGES[0]
    )
    next_rung, _ = downscale_stale_frames(
        history, keep_recent_frames=1, max_edge=IMAGE_CONTEXT_EDGES[1]
    )
    widest_bytes = estimate_wire_bytes(widest)
    assert estimate_wire_bytes(next_rung) < widest_bytes

    out, downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=widest_bytes - 1)

    assert dropped == 0
    assert downscaled == 5
    assert rung == IMAGE_CONTEXT_EDGES[1], "the reported rung must match the walk"
    assert estimate_wire_bytes(out) <= widest_bytes - 1
    assert estimate_wire_bytes(out) == estimate_wire_bytes(
        next_rung
    ), "the walk did not stop at the first rung that fits"


def test_fit_sheds_from_the_tightest_rung_when_no_rung_fits() -> None:
    """The last resort: when even the tightest rung is over budget — the
    unprotected walk included, so every frame is re-rendered first — the shed
    replaces the oldest frames with notices, terminates, and no message is
    removed."""
    history, _frames = _real_history(6)
    _CONTEXT_FRAME_CACHE.clear()

    out, downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=40_000)

    assert dropped == 6, "every frame must go when even the floor cannot fit"
    assert downscaled == 6, "the rungs were not applied before the shed"
    assert rung == IMAGE_CONTEXT_EDGES[-1], "the shed reports the tightest rung"
    assert _image_blocks_of(out) == [], "a frame survived a shed that cannot fit"
    assert estimate_wire_bytes(out) <= 40_000
    assert len(out) == len(history)
    assert [m.id for m in out] == [m.id for m in history]


def test_fit_descends_the_newest_frame_when_no_protected_rung_fits() -> None:
    """The QA-round boundary: one oversize frame against a tight budget.

    The newest-frame protection must not be the reason a session goes blind.
    When no rung fits with the newest kept full, the walk runs again with the
    newest unprotected and stops at the first rung that fits — so the model
    receives a degraded view of the screen instead of no view at all, and
    nothing is dropped.
    """
    history, _frames = _real_history(1)
    _CONTEXT_FRAME_CACHE.clear()
    widest, _ = downscale_stale_frames(
        history, keep_recent_frames=0, max_edge=IMAGE_CONTEXT_EDGES[0]
    )
    budget = estimate_wire_bytes(widest) - 1

    out, downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=budget)

    assert dropped == 0, "a frame was dropped although the frame's own rung fits"
    assert downscaled == 1, "the newest frame was not re-rendered"
    assert rung == IMAGE_CONTEXT_EDGES[1], "the unprotected walk must stop at its first fit"
    assert len(_image_blocks_of(out)) == 1, "the session lost the only frame it had"
    assert estimate_wire_bytes(out) <= budget
    assert _image_blocks_of(out)[0].data != _image_blocks_of(history)[0].data

    # The prompt-cache property holds on the new path too: re-fitting the same
    # history is byte-stable.
    again, _d, _x, rung_again = fit_frames_to_wire_budget(history, budget=budget)
    assert rung_again == rung
    assert [b.data for b in _image_blocks_of(again)] == [b.data for b in _image_blocks_of(out)]


def test_fit_keeps_every_frame_when_only_the_newest_descent_fits() -> None:
    """Newest over budget AND older frames in context: nothing may be dropped.

    QA round 1's case (b): six oversize frames against a budget where the
    protected walk's tightest rung is still over but the unprotected walk
    fits. The failure this repairs kept one frame and dropped five; every
    frame must survive, with the newest among the re-rendered.
    """
    history, frames = _real_history(6)
    _CONTEXT_FRAME_CACHE.clear()
    floor, _ = downscale_stale_frames(
        history, keep_recent_frames=0, max_edge=IMAGE_CONTEXT_EDGES[-1]
    )
    budget = estimate_wire_bytes(floor)

    out, downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=budget)

    assert dropped == 0, "frames were dropped although rendering all six fits"
    assert downscaled == 6, "not every frame was re-rendered"
    assert rung == IMAGE_CONTEXT_EDGES[-1]
    blocks = _image_blocks_of(out)
    assert len(blocks) == 6
    original = _image_blocks_of(frames)[0].data
    assert all(block.data != original for block in blocks)
    assert estimate_wire_bytes(out) <= budget
    assert [m.id for m in out] == [m.id for m in history]


def test_fit_prefers_the_protected_walk_at_its_boundary() -> None:
    """The second walk engages exactly one byte below the protected walk's
    boundary: at the budget where the protected walk's tightest rung fits, the
    newest frame is untouched; one byte tighter, the newest descends.

    This is the property the common path rests on — an episode where the
    protected walk fits renders exactly as it did before the descent existed —
    stated as a boundary pair rather than as two separate tests.
    """
    history, _frames = _real_history(2)
    _CONTEXT_FRAME_CACHE.clear()
    protected, _ = downscale_stale_frames(
        history, keep_recent_frames=1, max_edge=IMAGE_CONTEXT_EDGES[-1]
    )
    boundary = estimate_wire_bytes(protected)

    out, _downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=boundary)
    assert dropped == 0
    assert rung == IMAGE_CONTEXT_EDGES[-1]
    assert (
        _image_blocks_of(out)[-1].data == _image_blocks_of(history)[-1].data
    ), "the newest frame must keep full fidelity while a protected rung fits"

    out2, _downscaled2, dropped2, rung2 = fit_frames_to_wire_budget(history, budget=boundary - 1)
    assert dropped2 == 0
    assert (
        rung2 == IMAGE_CONTEXT_EDGES[0]
    ), "one byte below the boundary the newest must descend to the widest fitting rung"
    assert _image_blocks_of(out2)[-1].data != _image_blocks_of(history)[-1].data


def test_fit_is_a_no_op_under_budget_and_when_disabled() -> None:
    """The guarantee that every session under budget is byte-identical: the
    input elements come back by identity and both counts are zero."""
    history, _frames = _real_history(2)
    total = estimate_wire_bytes(history)

    out, downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=total + 1)
    assert (downscaled, dropped) == (0, 0)
    assert rung is None, "nothing was walked, so no rung"
    assert all(a is b for a, b in zip(out, history)), "under budget must not copy"

    out, downscaled, dropped, rung = fit_frames_to_wire_budget(history, budget=0)
    assert (downscaled, dropped) == (0, 0)
    assert rung is None
    assert all(a is b for a, b in zip(out, history))


def test_fit_is_deterministic_across_calls() -> None:
    """A frame's downscaled bytes are STABLE across fits — the prompt-cache
    property the policy rests on: the same history re-fitted (as every render
    does) produces identical block bytes for the same frames."""
    history, _frames = _real_history(4)
    total = estimate_wire_bytes(history)
    budget = int(total * 0.8)

    first, _d1, _r1, g1 = fit_frames_to_wire_budget(history, budget=budget)
    second, _d2, _r2, g2 = fit_frames_to_wire_budget(history, budget=budget)

    assert [block.data for block in _image_blocks_of(first)] == [
        block.data for block in _image_blocks_of(second)
    ]
    assert g1 == g2, "the walk lands on the same rung across fits"


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
