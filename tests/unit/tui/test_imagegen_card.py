"""The imagegen card variant: state line, graphic, cancel hint, settle, band.

The variant is the TUI surface of the image-generation programme: for the
detection set's call, the EXPANDED body paints the frozen state vocabulary,
a progress graphic that is determinate exactly when the adapter has a
fraction, the queue position / log tail / provider error when a producer
carried them — and the collapsed row carries the cancel hint while the call
can still be stopped. Absence renders as the reduced state; nothing here may
invent a number.

The card-level tests build a real ``ToolCard`` and read ``_build_content``
(the plain text a user sees), the way ``test_tool_card.py`` does. The
app-level tests drive the REAL app under ``run_test`` for the two things
only it can answer: the ``on_tool_updated`` adapter wiring, and the band
timer's lifecycle (expand/collapse/settle/retirement).
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import (
    FAULT_KEY,
    AgentToolUpdate,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResult,
)
from local_operator.tui.app import OperatorApp
from local_operator.tui.events import ToolEnded, ToolStarted, ToolUpdated
from local_operator.tui.widgets.tool_card import IMAGE_INTERRUPT_HINT, ToolCard

from .test_app_pilot import FakeSession, _factory

SENTENCE = "This generation failed before producing output."


def _running_card(**details: object) -> ToolCard:
    card = ToolCard("g0", "generate_image", {"prompt": "a cat"})
    card.begin_running("generate_image", {"prompt": "a cat"}, None)
    card._expanded = True
    if details:
        card.set_live_details(details)
    return card


def _content(card: ToolCard, width: int = 100) -> str:
    return card._build_content(width).plain


# --- the live body ---------------------------------------------------------


def test_the_running_card_paints_the_state_line_graphic_queue_and_logs() -> None:
    card = _running_card(
        state="running",
        queue_position=3,
        progress_fraction=0.42,
        log_lines=["request 1/1 · model flux-schnell", "step 13/28"],
    )
    body = _content(card)
    assert "⋯ running" in body
    assert "▰▰▰▰▱▱▱▱▱▱ 42%" in body
    assert "queue position 3" in body
    assert "step 13/28" in body
    # The cancel hint has the collapsed row while the call can still be
    # stopped; see `IMAGE_INTERRUPT_HINT`.
    assert IMAGE_INTERRUPT_HINT in card._build_row(100).plain


def test_without_a_fraction_the_graphic_is_the_canvas_not_a_bar() -> None:
    """Absence renders the reduced state: no fraction means no percentage, and
    the canvas is the honest indeterminate read."""
    card = _running_card(state="running")
    body = _content(card)
    assert "▱" * 10 in body
    assert "▰" not in body
    assert "%" not in body


def test_the_provider_word_can_name_cancelling_while_the_call_runs() -> None:
    card = _running_card(state="cancelling")
    assert "⋯ cancelling" in _content(card)


def test_the_queued_card_paints_the_first_word_and_opens_onto_it() -> None:
    """A queued imagegen row is expandable — the state line is its body — and
    paints the vocabulary's first word without a graphic (nothing runs yet)."""
    card = ToolCard("g1", "generate_image")
    card.set_composing(64, "generate_image")
    card.mark_queued()
    assert card.can_expand()
    card.toggle_expanded()
    body = _content(card)
    assert "⋯ queued" in body
    assert "▱" not in body
    assert IMAGE_INTERRUPT_HINT in card._build_row(100).plain


def test_structured_logs_win_and_the_stream_is_the_fallback() -> None:
    """ONE tail, never two: the structured rows when a producer sent them, the
    streamed text otherwise."""
    card = _running_card(state="running", log_lines=["structured row"])
    card.set_partial_detail("streamed row")
    body = _content(card)
    assert "structured row" in body
    assert "streamed row" not in body

    fresh = ToolCard("g2", "generate_image", {"prompt": "p"})
    fresh.begin_running("generate_image", {"prompt": "p"}, None)
    fresh._expanded = True
    fresh.set_partial_detail("streamed row")
    assert "streamed row" in _content(fresh)


def test_the_live_error_line_shows_the_platform_sentence() -> None:
    card = _running_card(state="running", error=SENTENCE, error_type="media_failed")
    assert f"✗ {SENTENCE}" in _content(card)


# --- the settled failure body ---------------------------------------------


def test_the_failure_body_paints_the_platform_sentence_verbatim() -> None:
    """The sentence IS the provider's error in its sanctioned form: painted
    exactly as received, never replaced by a generic sentence of ours."""
    card = ToolCard("g3", "generate_image", {"prompt": "p"})
    card._expanded = True
    card.mark_failed(
        "image generation failed",
        "image generation failed",
        details={"error": SENTENCE, "error_type": "media_rejected"},
    )
    body = _content(card)
    assert SENTENCE in body
    assert f"{'✗'} {SENTENCE}" in body


def test_a_retained_live_error_reaches_the_settled_failure_body() -> None:
    """A provider that reported the fault on one streaming update and then went
    quiet still has its words on the failure body."""
    card = _running_card(state="running", error=SENTENCE, error_type="media_failed")
    card.mark_failed("image generation failed", "image generation failed")
    assert SENTENCE in _content(card)


def test_the_sentence_never_prints_twice() -> None:
    """When the result text already carries it, the standard body is the one
    carrier and the note stays out — the double-print class the reason block
    documents."""
    card = ToolCard("g4", "generate_image", {"prompt": "p"})
    card._expanded = True
    card.mark_failed(SENTENCE, SENTENCE, details={"error": SENTENCE, "error_type": "media_failed"})
    assert _content(card).count(SENTENCE) == 1


def test_the_already_finished_conflict_is_never_an_error() -> None:
    """`media_already_completed` — the cancel conflict against a finished job —
    reads "already finished" and never wears the error's glyph or sentence."""
    card = _running_card(state="running", error_type="media_already_completed")
    body = _content(card)
    assert "already finished" in body
    assert "✗" not in body

    # ... and on the settled FAILURE body the conflict wears no failure
    # furniture in the EXPANSION (reviewer F1): no promoted-reason lead, no
    # ✗-wrapped sentence — the note is the account, and the head line remains
    # only as the row's own status text. (The collapsed row keeps the
    # result's own verdict — a surface must not re-classify a result — so
    # the row's `cancelled ✗` is the harness's receipt, not this variant's
    # dressing.)
    settled = ToolCard("g5", "generate_image", {"prompt": "p"})
    settled._expanded = True
    settled.mark_failed(
        "cancelled",
        "cancelled",
        details={"error": "conflict", "error_type": "media_already_completed"},
    )
    body = _content(settled)
    assert "already finished" in body
    assert "✗ cancelled" not in body
    assert "cancelled" in body


def test_the_finished_note_survives_a_success_settle() -> None:
    """The wire freeze decides which arm the conflict settles on; the note is
    derived from the payload on ALL of them (reviewer F1 / QA Q1) — success
    included, in the same neutral ink."""
    card = _running_card()
    card.mark_done(
        "Generation cancelled before completion.",
        details={"error_type": "media_already_completed"},
    )
    body = _content(card)
    assert "already finished" in body
    assert "✗" not in body


def test_the_finished_note_survives_an_interrupt_settle() -> None:
    """The same rule on the interrupt arm: `mark_interrupted` reads the
    result payload exactly as done and failed do (reviewer F1 / QA Q1)."""
    card = _running_card()
    card.mark_interrupted(
        reason="Stopped before completion.",
        details={"error_type": "media_already_completed"},
    )
    body = _content(card)
    assert "already finished" in body
    assert "✗" not in body


# --- the settled success has NO variant furniture --------------------------


def test_a_settled_card_carries_no_live_furniture() -> None:
    """The variant is live-only: on success the standard receipt is the whole
    card — no state line, no graphic, and no second way to carry the artifact
    (the transcript's image machinery owns that, not this widget)."""
    card = _running_card(state="running", progress_fraction=0.42)
    card.mark_done("Generated 1 image (1024x1024): /tmp/opic/img_04.png")
    body = _content(card)
    assert "Generated 1 image" in body
    assert "▱" not in body and "▰" not in body
    assert "⋯" not in body
    # The kitty placeholder base char would mean the card grew its own image
    # transport; it must not (the attachments lane owns the mount under the
    # settled card).
    assert "\U0010eeee" not in body


# --- the hint sheds like every other slot occupant --------------------------


def test_the_interrupt_hint_sheds_whole_at_narrow_widths() -> None:
    card = _running_card(state="running")
    assert IMAGE_INTERRUPT_HINT in card._build_row(100).plain
    assert IMAGE_INTERRUPT_HINT not in card._build_row(30).plain


def test_the_hint_leaves_once_the_call_is_stopped() -> None:
    card = _running_card(state="running")
    card.mark_interrupted()
    assert IMAGE_INTERRUPT_HINT not in card._build_row(100).plain
    assert "interrupted" in card._build_row(100).plain


# --- non-imagegen rows are untouched ---------------------------------------


def test_a_non_imagegen_card_keeps_the_generic_live_body() -> None:
    card = ToolCard("b0", "bash", {"command": "ls"})
    card.begin_running("bash", {"command": "ls"}, None)
    card._expanded = True
    card.set_live_details({"progress_fraction": 0.42, "queue_position": 1})
    body = _content(card)
    assert "⋯ running" in body  # the generic header, unchanged
    assert "▱" not in body and "%" not in body
    assert card._imagegen_live is None
    assert IMAGE_INTERRUPT_HINT not in card._build_row(100).plain


# --- app wiring ------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_updates_feed_the_adapter_through_the_app() -> None:
    """The one app-side wiring line: `on_tool_updated` hands the update's
    `details` to the card, and the card's adapter decides what they mean."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="g0", tool_name="generate_image", args={"prompt": "p"}
                )
            )
        )
        await pilot.pause()
        app.post_message(
            ToolUpdated(
                ToolExecutionUpdateEvent(
                    tool_call_id="g0",
                    tool_name="generate_image",
                    partial_result=AgentToolUpdate(
                        details={
                            "state": "in_progress",
                            "queue_position": 2,
                            "progress_fraction": 0.25,
                        }
                    ),
                )
            )
        )
        await pilot.pause()
        card = app._tool_cards["g0"]
        assert card._imagegen_live is not None
        assert card._imagegen_live.queue_position == 2
        assert card._imagegen_live.fraction == 0.25
        assert card._imagegen_live.state == "running"


@pytest.mark.asyncio
async def test_the_finished_note_survives_every_settle_arm_through_the_app() -> None:
    """End to end through the real app's ToolEnded path — the seam QA probed:
    the conflict's payload reaches the expansion whether the result settles as
    error, success or interrupt (reviewer F1 / QA Q1)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()
        for call_id in ("a", "b", "c"):
            app.post_message(
                ToolStarted(
                    ToolExecutionStartEvent(
                        tool_call_id=call_id, tool_name="generate_image", args={"prompt": "p"}
                    )
                )
            )
        await pilot.pause()
        # Settled cards leave `_tool_cards` the instant their result lands
        # (the live-registry cleanup), so hold the refs before the ends.
        cards = {call_id: app._tool_cards[call_id] for call_id in ("a", "b", "c")}
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="a",
                    tool_name="generate_image",
                    result=ToolResult(
                        tool_call_id="a",
                        tool_name="generate_image",
                        is_error=True,
                        content=[TextContent(text="image generation failed")],
                        details={"error_type": "media_already_completed"},
                    ),
                )
            )
        )
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="b",
                    tool_name="generate_image",
                    result=ToolResult(
                        tool_call_id="b",
                        tool_name="generate_image",
                        content=[TextContent(text="Generation cancelled before completion.")],
                        details={"error_type": "media_already_completed"},
                    ),
                )
            )
        )
        app.post_message(
            ToolEnded(
                ToolExecutionEndEvent(
                    tool_call_id="c",
                    tool_name="generate_image",
                    result=ToolResult(
                        tool_call_id="c",
                        tool_name="generate_image",
                        content=[TextContent(text="Stopped.")],
                        details={FAULT_KEY: "skipped", "error_type": "media_already_completed"},
                    ),
                )
            )
        )
        await pilot.pause()
        await pilot.pause()
        for call_id, expected_state in (("a", "error"), ("b", "success"), ("c", "interrupted")):
            card = cards[call_id]
            assert card.state == expected_state
            card._expanded = True
            body = _content(card)
            assert "already finished" in body, (call_id, body)


@pytest.mark.asyncio
async def test_an_update_without_details_for_a_plain_tool_is_a_noop() -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(tool_call_id="b0", tool_name="bash", args={"command": "ls"})
            )
        )
        await pilot.pause()
        app.post_message(
            ToolUpdated(
                ToolExecutionUpdateEvent(
                    tool_call_id="b0",
                    tool_name="bash",
                    partial_result=AgentToolUpdate(details=None),
                )
            )
        )
        await pilot.pause()
        card = app._tool_cards["b0"]
        assert card._imagegen_live is None
        card.toggle_expanded()
        body = card._build_content(100).plain
        # The generic live header, unchanged: the variant never engages for
        # a tool outside the detection set.
        assert "⋯ running" in body
        assert "▱" not in body
        assert IMAGE_INTERRUPT_HINT not in card._build_row(100).plain


# --- the band timer --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_band_timer_lives_exactly_where_the_graphic_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Started only for an EXPANDED, RUNNING imagegen card while motion is
    allowed; stopped on collapse and on settle. The collapsed row shows no
    graphic, so it must not pay 30 fps for one."""
    monkeypatch.setattr("local_operator.tui.animation.motion_enabled", lambda: True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="g0", tool_name="generate_image", args={"prompt": "p"}
                )
            )
        )
        await pilot.pause()
        card = app._tool_cards["g0"]
        assert card._frame_timer is None  # collapsed

        card.toggle_expanded()
        await pilot.pause()
        assert card._frame_timer is not None
        before = card._frame_ms
        card._tick_frame()
        assert card._frame_ms == before + 33

        card.toggle_expanded()
        await pilot.pause()
        assert card._frame_timer is None

        card.toggle_expanded()
        await pilot.pause()
        assert card._frame_timer is not None
        app._tool_cards.pop("g0", None)
        card.mark_done("done")
        assert card._frame_timer is None


@pytest.mark.asyncio
async def test_motion_off_holds_the_band_still(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("local_operator.tui.animation.motion_enabled", lambda: False)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="g0", tool_name="generate_image", args={"prompt": "p"}
                )
            )
        )
        await pilot.pause()
        card = app._tool_cards["g0"]
        card.toggle_expanded()
        await pilot.pause()
        assert card._frame_timer is None


# --- retirement (the existing interrupt path) ------------------------------


@pytest.mark.asyncio
async def test_retirement_settles_the_card_and_retires_the_band(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The imagegen card settles through the SAME turn-death path as every
    other live card — `_retire_live_tool_cards` — and the band stops with it
    (no new cancel mechanism, nothing left animating)."""
    monkeypatch.setattr("local_operator.tui.animation.motion_enabled", lambda: True)
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 30)) as pilot:
        await pilot.pause()
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="g0", tool_name="generate_image", args={"prompt": "p"}
                )
            )
        )
        await pilot.pause()
        card = app._tool_cards["g0"]
        card.toggle_expanded()
        await pilot.pause()
        assert card._frame_timer is not None

        retired = app._retire_live_tool_cards()
        assert retired == 1
        assert card._state == "interrupted"
        assert card._frame_timer is None
        row = card._build_row(100).plain
        assert "interrupted" in row
        assert IMAGE_INTERRUPT_HINT not in row


# --- composing keeps its own presentation ----------------------------------


def test_a_composing_imagegen_row_keeps_the_generic_dictation_row() -> None:
    """Pre-vocabulary: the model is still writing the call, so no state line —
    the composing row's byte counter is its honest progress."""
    card = ToolCard("g6", "generate_image")
    card.set_composing(64, "generate_image")
    body = _content(card)
    assert "⋯" not in body
    assert "▱" not in body
