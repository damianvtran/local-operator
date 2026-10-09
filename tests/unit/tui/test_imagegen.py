"""The imagegen adapter: the TUI surface's ONE mapping place, pinned.

Everything asserted here is a contract the card (and any future surface
reader) relies on when the harness lane's wire fields freeze:

- the detection set is exactly ONE tool constant, and the predicate is the
  only reader of it;
- the state vocabulary maps the card's lifecycle onto the frozen words, with
  a provider's word ahead of the card's when it names a LIVE state;
- the adapter reads the structured fields defensively and NEVER invents a
  value: an absent or malformed field renders as the reduced state, because a
  progress bar drawn from a guess is worse than no bar;
- the graphic is determinate exactly when a fraction exists, and it rides the
  shimmer machinery so the existing kill switch flattens it.
"""

from __future__ import annotations

import pytest
from rich.style import Style
from rich.text import Text

from local_operator.tui import imagegen


def _ink_name(style: Style | str | None) -> str | None:
    """The resolved hex of a span's ink, or ``None`` (test-local helper).

    ``Text.spans`` types its style as ``Style | str | None`` even though the
    module always builds ``Style`` objects; the cast is the same one
    ``test_tool_card._style_at`` applies for the same reason.
    """
    resolved = style if isinstance(style, Style) else Style.parse(style) if style else None
    color = None if resolved is None else resolved.color
    return None if color is None else color.name


# --- detection -------------------------------------------------------------


def test_the_detection_set_is_one_tool_and_one_predicate() -> None:
    """One tool, frozen by name: image-to-image is a parameter of it, not a
    second tool, so a set that grows a sibling would silently widen every
    renderer keyed off ``is_image_gen_tool``."""
    assert imagegen.IMAGE_GEN_TOOLS == frozenset({"generate_image"})
    assert imagegen.is_image_gen_tool("generate_image")
    # Case and surrounding space tolerated the way the card's other name tests
    # are, so a producer that shouts the name still reaches the variant.
    assert imagegen.is_image_gen_tool(" Generate_Image ")
    for other in ("generate_altered_image", "bash", "read", "", "generate_image_v2"):
        assert not imagegen.is_image_gen_tool(other), other


# --- state vocabulary ------------------------------------------------------


@pytest.mark.parametrize(
    ("card_state", "expected"),
    [
        ("queued", "queued"),
        ("running", "running"),
        ("success", "done"),
        ("error", "failed"),
        ("interrupted", "cancelled"),
        # Pre-vocabulary states keep their own presentation rather than being
        # forced into a word that would be false: a row still dictating is not
        # yet queued, and an approval wait is parked on the USER.
        ("composing", None),
        ("waiting", None),
    ],
)
def test_card_states_map_onto_the_frozen_vocabulary(card_state: str, expected: str | None) -> None:
    assert imagegen.imagegen_state_word(card_state) == expected


def test_a_provider_word_wins_only_when_it_names_a_live_state() -> None:
    """A LIVE card may say ``queued``/``running``/``cancelling`` because the
    provider knows better than the card does; a TERMINAL word is refused while
    the call still runs, because claiming ``done`` would preempt the settle's
    own receipt. (The provider's RAW words are normalized by
    :func:`live_from_details` — ``pending`` arrives here already ``queued``.)"""
    assert imagegen.imagegen_state_word("running", "queued") == "queued"
    assert imagegen.imagegen_state_word("running", "cancelling") == "cancelling"
    assert imagegen.imagegen_state_word("running", "done") == "running"
    assert imagegen.imagegen_state_word("running", "failed") == "running"
    assert imagegen.imagegen_state_word("running", "cancelled") == "running"
    assert imagegen.imagegen_state_word("running", "not-a-state") == "running"


# --- the adapter -----------------------------------------------------------


def test_the_adapter_reads_every_frozen_field() -> None:
    view = imagegen.live_from_details(
        {
            "state": "in_progress",
            "queue_position": 3,
            "progress_fraction": 0.42,
            "log_lines": ["step 12/28", "step 13/28"],
            "error": "This generation failed before producing output.",
            "error_type": "media_failed",
            "artifact_ref": "opic://artifacts/1",
        }
    )
    assert view.state == "running"
    assert view.queue_position == 3
    assert view.fraction == 0.42
    assert view.log_lines == ("step 12/28", "step 13/28")
    assert view.error == "This generation failed before producing output."
    assert view.error_type == "media_failed"
    assert view.artifact_ref == "opic://artifacts/1"
    assert set(view.keys) == {
        "state",
        "queue_position",
        "progress_fraction",
        "log_lines",
        "error",
        "error_type",
        "artifact_ref",
    }


@pytest.mark.parametrize("details", [None, {}, [], "nope", 7])
def test_the_adapter_never_invents_a_value(details: object) -> None:
    """Absence and malformed shapes both read as the reduced state. The card
    renders what is here and nothing else — no fabricated 0%, no queue
    position guessed from call order."""
    view = imagegen.live_from_details(details)
    assert view.state is None
    assert view.queue_position is None
    assert view.fraction is None
    assert view.log_lines == ()
    assert view.log_dropped == 0
    assert view.error is None
    assert view.artifact_ref is None
    assert view.keys == ()


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.42, 0.42),
        # Out of the fraction's domain: a number that cannot be a 0..1
        # fraction is NOT reinterpreted as a percentage (a producer's
        # ``step: 5`` must not paint as "5%"); it renders as absent.
        (40, None),
        (-0.1, None),
        (1.5, None),
        ("42%", None),
        (True, None),
        (float("nan"), None),
    ],
)
def test_fraction_accepts_only_a_real_fraction(fraction: object, expected: float | None) -> None:
    view = imagegen.live_from_details({"progress_fraction": fraction})
    assert view.fraction == expected


@pytest.mark.parametrize(
    ("position", "expected"),
    [(0, 0), (3, 3), ("3", None), (-1, None), (2.5, None), (True, None)],
)
def test_queue_position_accepts_a_non_negative_int_deliberately(
    position: object, expected: int | None
) -> None:
    """0 is a REAL reading, not a malformed one: FAL documents the field as
    "the number of requests ahead of yours, present only while IN_QUEUE", so
    zero means nothing is ahead (reviewer F2). Negative, non-int and bool
    values still read as absent.
    """
    view = imagegen.live_from_details({"queue_position": position})
    assert view.queue_position == expected


def test_the_log_tail_is_bounded_and_counts_what_it_dropped() -> None:
    rows = [f"row {index}" for index in range(imagegen.LOG_TAIL_LIMIT + 7)]
    view = imagegen.live_from_details({"log_lines": rows})
    assert view.log_lines == tuple(rows[-imagegen.LOG_TAIL_LIMIT :])
    assert view.log_dropped == 7
    # A string payload splits on newlines, the same tail.
    view = imagegen.live_from_details({"log_lines": "one\ntwo"})
    assert view.log_lines == ("one", "two")
    assert view.log_dropped == 0


def test_the_log_tail_and_error_strip_control_sequences() -> None:
    """The provider's words survive; escape sequences that would clear the
    terminal on every repaint do not (the card's boundary rule for names and
    outputs)."""
    view = imagegen.live_from_details(
        {
            "log_lines": ["safe \x1b[2J now"],
            "error": "boom \x1b[31mred\x1b[0m",
        }
    )
    assert view.log_lines[0] == "safe  now"
    assert view.error == "boom red"


def test_error_reads_the_platform_sentence_only() -> None:
    """The frozen shape is a STRING sentence plus a structured ``error_type``;
    a mapping under ``error`` is not the contract and reads as absent rather
    than being fished into."""
    assert (
        imagegen.live_from_details({"error": "This generation failed."}).error
        == "This generation failed."
    )
    assert imagegen.live_from_details({"error": {"message": "no"}}).error is None
    assert imagegen.live_from_details({"error": "  "}).error is None


def test_error_type_is_kept_raw_and_the_conflict_is_not_an_error() -> None:
    view = imagegen.live_from_details(
        {"error": "This generation failed.", "error_type": "media_rate_limited"}
    )
    assert view.error_type == "media_rate_limited"
    assert imagegen.provider_error_note(view.error, view.error_type) == (
        "This generation failed.",
        imagegen.ERROR_KIND_DANGER,
    )

    # The cancel conflict: the generation finished; its words are the frozen
    # note and its kind is NOT the error kind, so no surface can paint it with
    # an error's glyph or ink. The conflict may arrive WITHOUT a sentence.
    conflict = imagegen.live_from_details({"error_type": "media_already_completed"})
    assert conflict.error is None
    assert imagegen.provider_error_note(conflict.error, conflict.error_type) == (
        imagegen.ALREADY_FINISHED_NOTE,
        imagegen.ERROR_KIND_FINISHED,
    )
    assert imagegen.ALREADY_FINISHED_NOTE == "already finished"


# --- the error note's dedupe ------------------------------------------------


def test_provider_error_note_shows_a_new_payload_and_skips_a_carried_one() -> None:
    # No captured output: the payload is the body.
    assert imagegen.provider_error_note("boom", None, "") == ("boom", "error")
    # Already carried by the result text the standard body paints.
    assert imagegen.provider_error_note("boom", None, "prefix boom suffix") is None
    # Whitespace-normalized on both sides: the card stores one collapsed
    # sentence while the output keeps its own spacing.
    assert imagegen.provider_error_note("boom  now", None, "a\nboom now\nb") is None
    assert imagegen.provider_error_note(None, None, "anything") is None
    assert imagegen.provider_error_note("", None, "anything") is None
    # The finished note dedupes on ITS words, not on any vendor text.
    assert imagegen.provider_error_note("ignored", "media_already_completed", "x") == (
        "already finished",
        "finished",
    )
    carried = imagegen.provider_error_note(None, "media_already_completed", "a already finished b")
    assert carried is None


# --- the graphic -----------------------------------------------------------


def test_the_determinate_bar_is_proportional_and_labelled() -> None:
    assert imagegen.progress_graphic(0.42).plain == "▰▰▰▰▱▱▱▱▱▱ 42%"
    assert imagegen.progress_graphic(0.0).plain == "▱" * 10 + " 0%"
    assert imagegen.progress_graphic(1.0).plain == "▰" * 10 + " 100%"
    # The fill floors to the label's own percent, so at 99% the last cell
    # stays hollow — the bar can never read ahead of its number (reviewer F3).
    assert imagegen.progress_graphic(0.99).plain == "▰" * 9 + "▱" + " 99%"


def test_the_canvas_is_the_same_footprint_as_the_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canvas and the bar occupy the same cells, so a fraction freezing in
    never reflows the row.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    canvas = imagegen.progress_graphic(None, 123.0)
    bar = imagegen.progress_graphic(0.42)
    assert canvas.plain == "▱" * imagegen.PROGRESS_CELLS
    assert bar.plain[: imagegen.PROGRESS_CELLS] == "▰▰▰▰" + "▱" * 6


def test_the_canvas_shimmers_under_the_kill_switch_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With shimmer live the band tints the cells it crosses; pinned phases
    make the difference assertable without a clock. At phase 400ms the crest
    sits on the canvas (accent cells + mid shoulders); at phase 0 the band is
    away and every cell is dim.
    """
    from local_operator.tui import theme as theme_mod

    monkeypatch.delenv("LOCAL_OPERATOR_NO_SHIMMER", raising=False)

    def bands(text: Text) -> list[str | None]:
        return [_ink_name(span.style) for span in text.spans]

    lit = imagegen.progress_graphic(None, 400.0)
    dark = imagegen.progress_graphic(None, 0.0)
    assert lit.plain == dark.plain == "▱" * imagegen.PROGRESS_CELLS
    assert bands(dark) == [theme_mod.semantic_color("dim")] * imagegen.PROGRESS_CELLS
    assert theme_mod.semantic_color("accent") in bands(lit)


def test_the_bar_resolves_theme_tokens_not_literals() -> None:
    """The bar's inks come from the theme, so a ramp change moves one file."""
    from local_operator.tui import theme as theme_mod

    bar = imagegen.progress_graphic(0.5)
    assert _ink_name(bar.spans[0].style) == theme_mod.semantic_color("accent")
    assert _ink_name(bar.spans[-2].style) == theme_mod.semantic_color("dim")
    assert _ink_name(bar.spans[-1].style) == theme_mod.semantic_color("muted")
