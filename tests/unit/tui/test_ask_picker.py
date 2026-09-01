"""The ``ask`` picker: real keys and a real click, and what each one answers.

Driven with ``pilot.press`` rather than by calling the actions, because what is
being pinned is the KEYMAP as much as the state machine: the digits, Space, and
``j``/``k`` all change meaning on the free-text row, and a card that "worked"
while its bindings were unreachable would still be an unanswerable question.

The card's own arithmetic (widths, page size) is asserted through
``render_lines_for_test``; the layout it produces under the real stylesheet is
checked against ``OperatorApp`` at the bottom of the file, because the lightweight
host below declares no ``CSS_PATH`` and therefore cannot show a clip or a
scrollbar at all. Where a clip is the thing under test the assertion goes one
step further and reads the COMPOSITED SCREEN (``_painted_rows``): a card that
lays out more lines than its region has still reports the height it wanted, so
its own text agrees with the clip instead of catching it.
"""

from __future__ import annotations

import asyncio
import itertools

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import Container

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.ask_picker import (
    MIN_TRANSCRIPT_ROWS,
    OTHER_LABEL,
    PROMPT_HEIGHT_SHARE,
    RECOMMENDED_TAG,
    SECRET_MASK,
    AskPickerScreen,
)


def _question(
    qid: str = "stale",
    text: str = "What should happen to the stale rows?",
    *,
    labels: tuple[str, ...] = ("Drop them", "Backfill from the audit log"),
    descriptions: tuple[str, ...] = ("nothing reads the column", "slower, keeps history"),
    multi: bool = False,
    recommended: int | None = None,
) -> AskQuestion:
    return AskQuestion(
        id=qid,
        question=text,
        options=[
            AskOption(label=label, description=description)
            for label, description in zip(labels, descriptions + ("",) * len(labels))
        ],
        multi=multi,
        recommended=recommended,
    )


class _AskHost(App[None]):
    """A host whose only job is to own the card under test.

    Mounts it into a plain container rather than pushing a screen: the card is
    dock chrome now, not a modal, and a host that pushed it as a screen would be
    exercising a mounting path the app no longer uses.
    """

    def __init__(self, questions: list[AskQuestion]) -> None:
        super().__init__()
        self._questions = questions
        self.answered: list[dict[str, list[str]] | None] = []

    def compose(self) -> ComposeResult:
        yield Container(id="prompt-host")

    async def open_picker(self) -> AskPickerScreen:
        card = AskPickerScreen(self._questions, self.answered.append)
        await self.query_one("#prompt-host", Container).mount(card)
        return card


class _BareHost(_AskHost):
    """``_AskHost`` without the trailing free-text row.

    Not a stylistic variant: ``allow_free_text=False`` is the configuration the
    APPROVAL gate mounts this widget in (`approval.py`, `ApprovalPrompt`), and
    it is the only one in which the position-row defect below is reachable. With
    the free-text row present ``row_count`` is one greater than the option
    count, so a window of every OPTION is still genuinely short of the list and
    `showing 1-2 of 3` is honest. Take that row away and the same plan reports a
    window over a list it is drawing in full.
    """

    def __init__(self, questions: list[AskQuestion], *, allow_free_text: bool = False) -> None:
        super().__init__(questions)
        self._allow_free_text = allow_free_text

    async def open_picker(self) -> AskPickerScreen:
        card = AskPickerScreen(
            self._questions, self.answered.append, allow_free_text=self._allow_free_text
        )
        await self.query_one("#prompt-host", Container).mount(card)
        return card


#: The reported bug's own question, as `scripts/ask_long_shot.py` asks it.
#:
#: Duplicated from the script rather than imported: the script runs its capture
#: at module scope (`asyncio.run(main())` on its last line), so importing it
#: from a test would execute a screenshot run and then fail on `sys.argv[1]`.
#: The strings are what matter here, and they are the strings the rendered
#: frames in the report were captured from.
_LONG_QUESTION_TEXT = (
    "For the next iteration of the model regression canary battery (v2 currently lives at"
    " ~/workspace/model-canary/canary-eval.md, with the 11-item CORE block fully saturated"
    " at 11/11 for both models across every run logged so far, and essentially all of the"
    " discriminating signal concentrated in the four self-constraint items 13, 16, 17 and 18"
    " of the HARD block), which direction would you like the v3 item-recruitment effort to"
    " take, bearing in mind the calibration result already recorded in the workspace notes"
    " that mechanical difficulty does not discriminate between current frontier models?"
)

_LONG_LABELS = (
    "Double down on self-constraint items",
    "Add an orthogonal instruction-following block",
    "Fix measurement before adding items",
    "Retire the canary in its current form",
)

_LONG_DESCRIPTIONS = (
    "Recruit eight to twelve new v3 items drawn exclusively from the self-monitoring"
    " family that items 13, 16, 17 and 18 already occupy — exact word counts under"
    " simultaneous lexical constraints, sentences that must state their own character"
    " length, paragraphs forbidden from containing a letter that the instruction"
    " itself contains — then pilot each candidate three times per model and keep only"
    " those landing in the partial-failure band.",
    "Keep CORE and HARD byte-stable exactly as the versioning rule requires, and"
    " append a brand-new third scored block with its own denominator that targets"
    " multi-turn instruction adherence and negative constraints, on the theory that"
    " self-constraint failures and delayed-instruction failures are two faces of the"
    " same weakness.",
    "Leave the item set alone entirely for now and spend the effort on statistical"
    " power instead: raise the runner from two core / three hard runs per model to ten"
    " or more, record per-item pass rates rather than only block totals, and add"
    " variance bands to runs.md.",
    "Accept that a coarse hand-graded battery has reached the end of its useful life"
    " now that CORE is saturated and HARD is carried by four items, and either fold"
    " the surviving discriminating items into a proper harness or stop maintaining it.",
)


def _long_description_question(recommended: int | None = 0) -> AskQuestion:
    """The reproduction from the truncation report, as a question."""
    return AskQuestion(
        id="canary_v3_direction",
        question=_LONG_QUESTION_TEXT,
        options=[
            AskOption(label=label, description=description)
            for label, description in zip(_LONG_LABELS, _LONG_DESCRIPTIONS)
        ],
        recommended=recommended,
    )


def _description_lines_of(card: AskPickerScreen, lines: list[str]) -> list[str]:
    """The DESCRIPTION lines among ``lines``, read from the card's own map.

    `_line_rows` rather than arithmetic over the rendered text: it is the map
    the hit-test uses (`_index_at`), it is rebuilt on every paint, and it is
    already multi-line tolerant — so this keeps working when one row's
    description occupies several lines, which is the whole point of the change.
    A row's FIRST line is its label; every further line belonging to the same
    row is description.
    """
    seen: set[int] = set()
    out: list[str] = []
    for index, line in zip(card._line_rows, lines):
        if index is None:
            continue
        if index in seen:
            out.append(line)
        seen.add(index)
    return out


# --- answering --------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_secret_question_masks_the_typed_value_and_returns_it() -> None:
    """A secret question is a paste field, not a picker. The painted label
    must never contain the value; the settled answer must, so the host can
    store it."""
    secret = "ghp_must_not_paint"
    question = AskQuestion(
        id="GITHUB_TOKEN",
        question="Paste the deploy token.",
        options=[],
        secret=True,
    )
    app = _AskHost([question])
    async with app.run_test(size=(100, 30)) as pilot:
        card = await app.open_picker()
        await pilot.pause()
        assert card.row_count == 1
        for char in secret:
            await pilot.press(char)
        await pilot.pause()
        labels = card.visible_rows
        assert secret not in "".join(labels)
        assert SECRET_MASK * len(secret) in "".join(labels)
        # The PAINTED row, not the label alone: `_row_text` re-renders the
        # selected field with the typed tail, and that second path is where a
        # secret leaked to the screen.
        from rich.style import Style

        painted = card._row_text(
            0,
            card._card_width(),
            Style(),
            Style(),
            Style(),
            Style(),
            card._layout(),
        ).plain
        assert secret not in painted
        assert SECRET_MASK * len(secret) in painted
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"GITHUB_TOKEN": [secret]}]


@pytest.mark.asyncio
async def test_enter_answers_with_the_highlighted_option_label() -> None:
    """The whole point of the surface: it hands back the label the model wrote,
    not an index and not free text the agent has to re-parse."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["Backfill from the audit log"]}]


@pytest.mark.asyncio
async def test_a_number_key_jumps_straight_to_a_row() -> None:
    app = _AskHost([_question(labels=("A", "B", "C"), descriptions=("", "", ""))])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("3")
        assert screen.selected_index == 2
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["C"]}]


@pytest.mark.asyncio
async def test_j_and_k_move_the_cursor_like_the_arrows() -> None:
    app = _AskHost([_question(labels=("A", "B", "C"), descriptions=("", "", ""))])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("j")
        await pilot.press("j")
        assert screen.selected_index == 2
        await pilot.press("k")
        assert screen.selected_index == 1


@pytest.mark.asyncio
async def test_the_arrows_wrap_around_the_list() -> None:
    """Arrow movement is a discrete, deliberate press, so it wraps — the
    convention this repo applies everywhere except wheel and page movement."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        rows = screen.row_count  # two options plus the free-text row
        await pilot.press("up")
        assert screen.selected_index == rows - 1
        await pilot.press("down")
        assert screen.selected_index == 0


@pytest.mark.asyncio
async def test_escape_answers_nothing_rather_than_a_guess() -> None:
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.answered == [None]


@pytest.mark.asyncio
async def test_escape_keeps_the_questions_already_answered() -> None:
    """A user who answered two of three has told the agent something; throwing
    it away because the third was never reached would report less than happened.
    """
    app = _AskHost([_question(), _question("timing", "When?", labels=("Now", "Later"))])
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("enter")  # answers the first question
        await pilot.press("escape")  # leaves the second unanswered
        await pilot.pause()
    assert app.answered == [{"stale": ["Drop them"]}]


@pytest.mark.asyncio
async def test_a_click_on_a_row_selects_and_answers_with_it() -> None:
    """The card invites the mouse in with the wheel; a list you can scroll and
    cannot click is a half-built affordance.

    The click target is re-derived from `_line_rows` rather than computed as
    `2 + 1 + 1 + 2`. That arithmetic — header, rule, question, spacer, then two
    lines per row — described this fixture at this size correctly, but only
    because every description happened to wrap to exactly one line. It was an
    assumption about the LAYOUT standing in for the thing under test, and once a
    description can occupy several lines it is false for any card whose rows are
    not uniformly two lines tall.

    `_line_rows` is the map `_index_at` actually resolves a click through
    (`ask_picker.py:870-904`): body-relative line index -> the row it belongs
    to, recorded while painting. Asking it which line belongs to row 1 tests the
    hit-test itself, and it keeps testing the hit-test whatever the rows cost.
    """
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        body = screen.query_one("#ask-picker-body")
        region = body.region
        # The first line the card says belongs to the SECOND option — its label
        # line, whether that is the fourth line of the body or the ninth.
        target = screen._line_rows.index(1)
        await pilot.click(offset=(region.x + 4, region.y + target))
        await pilot.pause()
    assert app.answered == [{"stale": ["Backfill from the audit log"]}]


# --- multi-select -----------------------------------------------------------


@pytest.mark.asyncio
async def test_space_toggles_and_enter_confirms_a_multi_select() -> None:
    app = _AskHost([_question(labels=("A", "B", "C"), descriptions=("", "", ""), multi=True)])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("space")  # A
        await pilot.press("down")
        await pilot.press("down")
        await pilot.press("space")  # C
        assert screen.checked_indexes == [0, 2]
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["A", "C"]}]


@pytest.mark.asyncio
async def test_space_toggles_off_again() -> None:
    app = _AskHost([_question(multi=True)])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("space")
        await pilot.press("space")
        assert screen.checked_indexes == []
        await pilot.press("enter")  # nothing chosen: a no-op, not an answer
        await pilot.pause()
    assert app.answered == []


@pytest.mark.asyncio
async def test_enter_on_a_single_select_with_nothing_typed_in_other_does_nothing() -> None:
    """Advancing here would record a question as answered when the user had
    done nothing, and the model would then act on an answer nobody gave."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("up")  # onto the free-text row, which is empty
        assert screen.selected_index == screen.other_row
        await pilot.press("enter")
        await pilot.pause()
        assert app.answered == []


# --- the free-text row ------------------------------------------------------


@pytest.mark.asyncio
async def test_typing_on_the_other_row_becomes_the_answer() -> None:
    """What the prose surface was reaching for with "(C) You have context I
    don't": an answer that is not on the list."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("3")  # the free-text row
        for char in "archive to s3":
            await pilot.press(char if char != " " else "space")
        assert screen.typed_text == "archive to s3"
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["archive to s3"]}]


@pytest.mark.asyncio
async def test_the_movement_letters_are_text_while_the_field_holds_the_cursor() -> None:
    """``j``/``k`` and the digits move the cursor everywhere else on this card.
    On the field they are letters — a field that silently dropped every ``j``
    would be worse than no field."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("3")
        for char in "jk2":
            await pilot.press(char)
        assert screen.typed_text == "jk2"
        assert screen.selected_index == screen.other_row  # the cursor never moved


@pytest.mark.asyncio
async def test_backspace_edits_the_field() -> None:
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("3")
        for char in "abc":
            await pilot.press(char)
        await pilot.press("backspace")
        assert screen.typed_text == "ab"


@pytest.mark.asyncio
async def test_free_text_rides_along_with_a_multi_select_answer() -> None:
    app = _AskHost([_question(multi=True)])
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("space")  # check the first option
        await pilot.press("3")  # onto the field
        for char in "and vacuum":
            await pilot.press(char if char != " " else "space")
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["Drop them", "and vacuum"]}]


@pytest.mark.asyncio
async def test_the_field_explains_itself_before_it_is_selected() -> None:
    """A row reading only ``Other`` is a dead end; the user has to be able to see
    that it takes typing without selecting it first to find out."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
        assert OTHER_LABEL in text
        assert "type it here" in text


# --- what the card says -----------------------------------------------------


@pytest.mark.asyncio
async def test_the_question_and_every_option_and_description_are_drawn() -> None:
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        lines = screen.render_lines_for_test()
        text = "\n".join(lines)
        assert "What should happen to the stale rows?" in text
        assert "Drop them" in text
        assert "nothing reads the column" in text
        assert "slower, keeps history" in text


@pytest.mark.asyncio
async def test_a_recommendation_is_marked_and_preselected() -> None:
    """Preselected as well as marked: the point of recommending is that Enter
    alone should take it."""
    app = _AskHost([_question(recommended=1)])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert screen.selected_index == 1
        assert RECOMMENDED_TAG in "\n".join(screen.render_lines_for_test())
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["Backfill from the audit log"]}]


@pytest.mark.asyncio
async def test_the_position_is_shown_only_when_there_is_more_than_one_question() -> None:
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert "Question" not in "\n".join(screen.render_lines_for_test())

    app = _AskHost([_question(), _question("timing", "When?", labels=("Now", "Later"))])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert "Question 1 of 2" in "\n".join(screen.render_lines_for_test())
        await pilot.press("enter")
        await pilot.pause()
        assert "Question 2 of 2" in "\n".join(screen.render_lines_for_test())
        assert "When?" in "\n".join(screen.render_lines_for_test())


@pytest.mark.asyncio
async def test_answers_are_collected_across_questions() -> None:
    app = _AskHost(
        [
            _question(),
            _question("timing", "When should this ship?", labels=("Now", "Later")),
        ]
    )
    async with app.run_test(size=(100, 30)) as pilot:
        await app.open_picker()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert app.answered == [{"stale": ["Drop them"], "timing": ["Later"]}]


@pytest.mark.asyncio
async def test_a_control_sequence_in_a_label_never_reaches_the_terminal() -> None:
    """Labels are MODEL-CONTROLLED. One carrying CSI could erase the rows above
    the card and repaint a forged question over them, and would mis-measure every
    width budget on the way (``cell_len`` counts the escape bytes)."""
    app = _AskHost([_question(labels=("\x1b[2JDrop them", "Keep"), descriptions=("", ""))])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
        assert "\x1b" not in text
        assert "Drop them" in text
        await pilot.press("enter")
        await pilot.pause()
    # The ANSWER is the stripped label, so what the model is told matches what
    # the user was shown.
    assert app.answered == [{"stale": ["Drop them"]}]


@pytest.mark.asyncio
async def test_no_row_overflows_the_card_at_any_width() -> None:
    long_labels = ("Drop the whole column and every index that references it",) * 2
    # Three fixtures, because they reach different code. The first is the
    # original one-line-per-description case. The second is the reported bug's
    # own question, whose descriptions want several lines each — without it the
    # sweep never sends a CONTINUATION line through `_fit_row`, which is the new
    # path and the one that can overflow. The third carries grapheme clusters
    # (design §11 risk 5): descriptions are model-authored, and a wrap that
    # measured `👨‍👩‍👧‍👦` or a flag by string length rather than by cell width would
    # overflow here and nowhere else.
    emoji_question = _question(
        text="Which pipeline should absorb the backfill? 🚚",
        labels=("Nightly batch 🌙", "Streaming lane ⚡"),
        descriptions=(
            "🚚📦 moves the whole table in one pass overnight — cheapest per row, and"
            " nothing else can run while it holds the lock 🔒🔒🔒",
            "⚡👨‍👩‍👧‍👦 keeps the tail live for family accounts 🇬🇧🇬🇧 and costs more per row",
        ),
    )
    questions = [
        _question(labels=long_labels, descriptions=("a long consequence " * 6, "")),
        _long_description_question(),
        emoji_question,
    ]
    for width, question in itertools.product((24, 30, 40, 60, 80, 120, 190), questions):
        app = _AskHost([question])
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            # The card's OWN content box, read back from the layout rather than
            # re-derived here: recomputing the column from the terminal width
            # would be a second opinion about the one number the widget is not
            # free to be wrong about.
            #
            # This assertion guards OVERFLOW ONLY, which is all its name claims.
            # Measured (agent review round 1, F4): forcing `_card_width` to
            # over-return by 10 fails it, while UNDER-returning by 10 passes it
            # silently — a card that went back to capping its text would satisfy
            # every line here. `test_the_card_spends_the_whole_column_the_dock_
            # gave_it` is the guard for that half, and it is the one verified to
            # fail against the old 74-cell cap.
            budget = screen.size.width
            for line in screen.render_lines_for_test():
                assert cell_len(line) <= budget, (width, line)


@pytest.mark.asyncio
async def test_the_card_spends_the_whole_column_the_dock_gave_it() -> None:
    """The card is laid out at the composer's width, so its ROWS have to reach
    that width too.

    The defect this pins is the one a clip test cannot see: every row fitted,
    and the card still stopped 42 cells short of its own panel at 160 columns,
    because a modal-era cap (74 cells less a floating margin) survived the move
    into the dock. Under-spending the column is as wrong as overflowing it — the
    rule under the title, the selected row's tint and the footer all ended in
    the middle of a panel whose fill ran to the composer's full width.
    """
    for width in (80, 100, 120, 160, 200):
        app = _AskHost([_question()])
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            column = screen.size.width
            # The selected row is painted in its own ground for its full width
            # (`_fit_row`), so it is the row that states the card's real reach.
            rows = [line for line in screen.render_lines_for_test() if line.strip()]
            assert max(cell_len(line) for line in rows) == column, (width, column)


@pytest.mark.asyncio
async def test_a_short_terminal_drops_descriptions_before_it_drops_options() -> None:
    """A card that shed ROWS to keep prose would hide answers the user is being
    asked to choose between. Descriptions go first; the list goes last.

    Amended for the wrapping change, and the amendment is deliberately narrow:
    "descriptions" here now means a description's FIRST line. C5 — all rows get
    a first line or none do — is unchanged, and it is still what makes `why a`
    absent at 100x20. What the change adds is CONTINUATION lines, which are
    bought after everything else and so can only ever be the first thing lost.

    The ORDER is the contract and the order is untouched: prose is still shed
    before an answer is. The specifics that moved are the vocabulary above and
    the second half of the sweep, which now also pins that the shortfall is not
    circumventable — the ban on drawing prose at a height that cannot afford it
    has to hold for the continuation lines too, or the new rungs would have
    quietly become a way to buy at 100x20 what step 9 refused to sell.
    """
    question = _question(
        labels=("A", "B", "C", "D", "E"),
        descriptions=("why a", "why b", "why c", "why d", "why e"),
    )

    app = _AskHost([question])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        roomy = "\n".join(screen.render_lines_for_test())
        assert "why a" in roomy
        assert len(screen.visible_rows) == screen.row_count

    # 20 rows rather than the 16 this used when the card was a modal. The
    # number is not the contract; the ORDER is, and the order needs a height
    # where the descriptions are unaffordable and the rows are not. An anchored
    # card reserves the conversation's share as well as the composer's, so it
    # reaches that band at a taller terminal than a card that took the screen.
    app = _AskHost([question])
    async with app.run_test(size=(100, 20)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        cramped = "\n".join(screen.render_lines_for_test())
        assert "why a" not in cramped
        # Every option still drawn, and the free-text row with them: the rows
        # the descriptions paid for.
        assert len(screen.visible_rows) == screen.row_count
        assert all(label in cramped for label in ("A", "B", "C", "D", "E"))
        # No row drew prose at all — not "less prose". At this height the
        # budget is exhausted before step 9, so the pool the continuation
        # lines are bought from is empty and the frame is the one this card
        # has always drawn here.
        assert not _description_lines_of(screen, screen.render_lines_for_test())

    # And the order survives a description long enough to WANT several lines.
    # This is the half that would go quiet if continuations were ever bought
    # ahead of the rows: same height, same five options, prose that a wrapping
    # card would love to spend eight lines on.
    app = _AskHost(
        [
            _question(
                labels=("A", "B", "C", "D", "E"),
                descriptions=("why a " * 20, "why b " * 20, "why c", "why d", "why e"),
            )
        ]
    )
    async with app.run_test(size=(100, 20)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        verbose = "\n".join(screen.render_lines_for_test())
        assert len(screen.visible_rows) == screen.row_count
        assert all(label in verbose for label in ("A", "B", "C", "D", "E"))
        assert not _description_lines_of(screen, screen.render_lines_for_test())


@pytest.mark.asyncio
async def test_the_position_row_is_only_drawn_when_the_list_is_windowed() -> None:
    """`showing 1–2 of 2` is a card reporting a window over a list it is
    drawing in full — the count lying in the direction that makes the user look
    for answers that are all already on screen.

    The mechanism is the one-step settle in `_layout`. The position line is
    bought on the `page < row_count` trial; buying it takes a row from
    `remaining`, which drops `remaining` below the two rows the TITLE costs, and
    the two rows the title gives back buy more option rows than the count cost.
    The retry therefore comes back with the WHOLE list drawn while still
    carrying the count that says it is hiding some of it. `_card_text` draws the
    row from `layout.show_position` alone, so it draws it.

    Pinned in the free-text-less configuration because that is where it is
    reachable AND where it matters: `ApprovalPrompt` mounts this widget with
    `allow_free_text=False`, so this is a tool-authorisation card miscounting
    its own answers. With the free-text row present `row_count` is one larger
    than the option count and the same plan's count is honest.

    The inverse of R11 (`ask_picker.py:1749-1753`), where a row nobody paid for
    was drawn; here a row that WAS paid for should not have been.
    """
    question = _question(
        text="Ship it?",
        labels=("Yes", "No"),
        descriptions=("do it", "do not"),
    )
    app = _BareHost([question])
    async with app.run_test(size=(100, 12)) as pilot:
        card = await app.open_picker()
        await pilot.pause()
        layout = card._layout()
        text = "\n".join(card.render_lines_for_test())
        # The list really is drawn in full: every option, nothing hidden.
        assert layout.page >= card.row_count, (layout.page, card.row_count)
        assert len(card.visible_rows) == card.row_count
        # So the card must not claim otherwise. Asserted on the PLAN and on the
        # painted text, because either one alone can be right while the other
        # is wrong: the flag is what the renderer reads, and the string is what
        # the user reads.
        assert not layout.show_position, (layout.page, card.row_count, text)
        assert "showing" not in text, text


@pytest.mark.asyncio
async def test_a_long_description_wraps_instead_of_ending_in_an_ellipsis() -> None:
    """The reported bug. At 190x50 there are rows to spare and every one of the
    four descriptions still ends mid-sentence in `…`, because
    `_description_text` builds each one as a single `no_wrap` line and
    `truncate_cells`-es it into the card's width — one line per row, always,
    however much budget is left over.

    190x50 rather than a tighter size on purpose: it is the size where the
    design's change (A) is expected to fix the bug COMPLETELY, so this test
    states the end condition rather than a partial improvement. The card's
    budget there is 20 rows against the 19 the wrapped card wants.

    Two assertions, and they are different claims. "No description line ends in
    `…`" is the symptom the user reported. "Option 1's full text is present
    across its lines" is the thing they actually wanted — a card could satisfy
    the first by dropping descriptions altogether, which is strictly worse.
    """
    app = _AskHost([_long_description_question()])
    async with app.run_test(size=(190, 50)) as pilot:
        card = await app.open_picker()
        await pilot.pause()
        lines = card.render_lines_for_test()
        descriptions = _description_lines_of(card, lines)
        # The card is drawing prose at all: guards against this test passing
        # because every description was dropped.
        assert descriptions, lines
        ellipsised = [line for line in descriptions if line.rstrip().endswith("…")]
        assert not ellipsised, ellipsised
        # And the first option's consequence is readable IN FULL. Compared on
        # collapsed whitespace, because the text is wrapped across several
        # lines and re-joined with the indent each continuation carries.
        joined = " ".join(" ".join(line.split()) for line in descriptions)
        assert " ".join(_LONG_DESCRIPTIONS[0].split()) in joined, joined


@pytest.mark.asyncio
async def test_a_terminal_too_short_for_the_list_says_what_it_is_hiding() -> None:
    """Once the descriptions are already gone there is nothing left to shed, so
    the list windows. The position line is what keeps that honest — a card
    silently showing two of six answers is a card that hid four of them."""
    question = _question(
        labels=("A", "B", "C", "D", "E"),
        descriptions=("why a", "why b", "why c", "why d", "why e"),
    )
    app = _AskHost([question])
    # 12 rows rather than 10: the question is now bought before the windowing
    # line, so at 10 the card correctly spends its last row on what is being
    # asked and drops the count. The contract under test is "a windowed list
    # says so when it can afford to", which needs a height where it can.
    async with app.run_test(size=(100, 12)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
        assert len(screen.visible_rows) < screen.row_count
        # The count is bought whenever the card can afford it AFTER the
        # question. Where it cannot, the question wins and the count goes: a
        # card that says how many answers it is hiding while hiding what the
        # answers are TO is the worse of the two abbreviations (D1, design
        # round 1). At this size both fit, so both are asserted.
        assert f"of {screen.row_count}" in text
        # And the window follows the cursor, so Enter can never take a row the
        # card did not draw — the `/resume` picker's bug, where the cursor sat
        # on a row past the drawn page and Enter resumed a session the user
        # could not see.
        for _ in range(4):
            await pilot.press("down")
            assert "E" in screen.visible_rows or screen.selected_index < 4
        await pilot.press("down")  # onto the free-text row, the last one
        assert screen.selected_index == screen.other_row
        assert any(row.startswith("Other") for row in screen.visible_rows)


@pytest.mark.asyncio
async def test_the_approval_cards_consequences_are_unchanged_by_the_wrap() -> None:
    """The blast radius. `ApprovalPrompt` subclasses this card, and there the
    description is not a nicety — it is the CONSEQUENCE of authorising a
    possibly destructive tool call, and the difference between "ask again next
    time" and "stop asking for this session".

    Measured on the pre-fix tree: the three consequence strings are 37, 36 and
    28 cells and wrap to one line each at every width down to 44 columns, so the
    approval card never truncates a description today. It is not what the
    wrapping change fixes; it is what the change must not break.

    Pinned as the EXACT frame rather than as "the strings are present". The
    failure mode being guarded is not the text disappearing — it is the
    allocator spending rows differently on a card whose height is already
    correct, which shows up as a moved footer, a dropped spacer or a row gaining
    a line, none of which a substring assertion can see. The three sizes are the
    ones the truncation report was filed against.

    If this test needs relaxing, that is a stop-and-escalate, not an expectation
    to update (design §11 risk 1).
    """
    from local_operator.tui.widgets.approval import ApprovalPrompt

    # The frames as the pre-fix tree draws them, captured at b64eb01c before any
    # allocator change. Written out in full because the point is byte-identity:
    # a golden that was regenerated from the code under test would agree with
    # whatever that code now does.
    expected = {
        (100, 30): [
            "the agent needs your approval",
            "─" * 100,
            "Allow bash? rm -rf /Users/x/project/data",
            "",
            "❯ y. Allow",
            "     run this call and ask again next time",
            "  n. Deny",
            "     refuse this call; the turn continues",
            "  A. Allow all",
            "     stop asking for this session",
            "",
            "↑↓ move · enter answer · esc deny",
        ],
    }
    expected[(130, 30)] = [
        "─" * 130 if line.startswith("─") else line for line in expected[(100, 30)]
    ]
    expected[(150, 40)] = [
        "─" * 150 if line.startswith("─") else line for line in expected[(100, 30)]
    ]

    for size, golden in expected.items():
        app = _AskHost([])
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            card = ApprovalPrompt("bash", "rm -rf /Users/x/project/data")
            await app.query_one("#prompt-host", Container).mount(card)
            await pilot.pause()
            # Right-padding is the row's own ground (`_fit_row` paints the
            # selected row and every description for the card's full width), so
            # it carries no text and comparing it would pin the card's WIDTH a
            # second time — which `test_the_card_spends_the_whole_column_the_
            # dock_gave_it` already owns. The claim here is what the card SAYS
            # and on which line it says it.
            drawn = [line.rstrip() for line in card.render_lines_for_test()]
            assert drawn == golden, (size, drawn)
            # Never a window over a list of three it is drawing in full: the
            # same defect the position-row test pins, on the surface where the
            # miscount would be attached to an authorisation.
            layout = card._layout()
            assert not (layout.show_position and layout.page >= card.row_count), size
            # Settled so the prompt's future is not left pending on teardown.
            card._settled = True


@pytest.mark.asyncio
async def test_the_card_never_draws_more_lines_than_its_budget() -> None:
    """C1, as a property over the plan rather than an example: the card lays out
    exactly what it paid for, and never a line more.

    This is the regression guard for the clipped footer. Textual clips the TAIL
    of an overflowing widget, and the tail is the footer — the one statement of
    how to leave a card the turn is parked on. The card reports the height it
    WANTED, so its own text agrees with the clip instead of catching it; only
    counting the laid-out lines against the budget catches it.

    Swept over the long-description fixture specifically, because that is the
    input that makes the new continuation lines reachable. On the pre-fix tree
    every description is one line, so a sweep with the ordinary fixture never
    exercises the arithmetic the change adds — measured: zero overdraws today
    across this whole sweep, so any failure here is new damage.

    The emoji row is per design §11 risk 5: descriptions are model-authored and
    can carry grapheme clusters, whose cell width is not their string length.
    """
    emoji = _question(
        text="Which pipeline should absorb the backfill? 🚚",
        labels=("Nightly batch 🌙", "Streaming lane ⚡"),
        descriptions=(
            "🚚📦 moves the whole table in one pass overnight — cheapest per row, and"
            " nothing else can run while it holds the lock 🔒🔒🔒",
            "⚡👨‍👩‍👧‍👦 keeps the tail live for family accounts and costs more per row, but the"
            " table stays writable throughout 🇬🇧🇬🇧",
        ),
    )
    for question in (_long_description_question(), emoji):
        for width in (24, 40, 60, 100, 130, 150, 190):
            for height in (12, 20, 30, 36, 40, 50):
                app = _AskHost([question])
                async with app.run_test(size=(width, height)) as pilot:
                    card = await app.open_picker()
                    await pilot.pause()
                    layout = card._layout()
                    budget = card._body_rows(len(card._question_lines(layout.width)))
                    lines = card.render_lines_for_test()
                    assert len(lines) <= budget, (width, height, len(lines), budget, lines)
                    # And no line overflows the card's column either. The
                    # continuation lines are the new path through `_fit_row`,
                    # and a wrap that measured a grapheme cluster by string
                    # length would show up here and nowhere else.
                    for line in lines:
                        assert cell_len(line) <= card.size.width, (width, height, line)


@pytest.mark.asyncio
async def test_a_wrapped_description_never_costs_an_option_row() -> None:
    """Steps 10-11 are ADDITIVE. A description growing a second line must never
    take a row off the list, because a row is an ANSWER and prose about an
    answer ranks strictly below it (`ask_picker.py:1206-1220`).

    The comparison is between the same question asked with LONG descriptions and
    with one-line descriptions: same labels, same question text, same everything
    that costs a row before step 9. If the wrapped card draws fewer option rows
    than the short one at any size, the continuation lines were bought with a
    row's money.

    This is the assertion that would have caught the rejected
    "wrap-everything" design, which needs 12 lines where 5 exist at 150x40 and
    windows the list to 2 of 5 at 100x30 — trading unreadable prose for hidden
    answers.
    """
    long_question = _long_description_question()
    short_question = AskQuestion(
        id=long_question.id,
        question=long_question.question,
        options=[
            AskOption(label=option.label, description="short") for option in long_question.options
        ],
        recommended=long_question.recommended,
    )

    for width in (24, 40, 60, 100, 130, 150, 190):
        for height in (12, 20, 30, 36, 40, 50):
            app = _AskHost([short_question])
            async with app.run_test(size=(width, height)) as pilot:
                card = await app.open_picker()
                await pilot.pause()
                baseline_page = card._layout().page
                baseline_labels = list(card.visible_rows)

            app = _AskHost([long_question])
            async with app.run_test(size=(width, height)) as pilot:
                card = await app.open_picker()
                await pilot.pause()
                assert card._layout().page == baseline_page, (width, height)
                assert list(card.visible_rows) == baseline_labels, (width, height)


@pytest.mark.asyncio
async def test_a_secret_question_still_draws_one_row() -> None:
    """The secret path has `row_count == 1` and a fixed `SECRET_HINT` as its
    description, and neither the wrap nor the reveal may grow it.

    Worth its own test because the secret row is the one place on this card
    where the "description" is chrome the app wrote rather than prose the model
    wrote: it explains the FIELD, and a card that spent extra rows elaborating
    on `hidden as you type` would be padding a paste box. The hint is one line
    at every width it is drawn at, so the row stays two lines.
    """
    question = AskQuestion(
        id="GITHUB_TOKEN",
        question="Paste the deploy token.",
        options=[],
        secret=True,
    )
    for size in ((100, 30), (150, 40), (190, 50)):
        app = _AskHost([question])
        async with app.run_test(size=size) as pilot:
            card = await app.open_picker()
            await pilot.pause()
            assert card.row_count == 1
            lines = card.render_lines_for_test()
            # One label line and at most one description line for the only row.
            row_lines = [index for index in card._line_rows if index is not None]
            assert row_lines.count(0) <= 2, (size, lines)
            assert len(row_lines) == row_lines.count(0), (size, lines)


@pytest.mark.asyncio
async def test_a_recommended_row_keeps_its_prose_when_the_first_word_is_unbreakable() -> None:
    """Review finding (MAJOR): the recommended row drew `· recommended` and NO
    prose at all when its description opened with a word too long to sit beside
    the tag.

    The tag reserves cells on the description's first line, and that line was
    produced by wrapping a filler placeholder and the text in ONE pass, then
    slicing the filler off. A first token longer than `room - tag_cells` does
    not fit beside the filler either, so `wrap_cells` — correctly — put the
    filler on a line of its own, and the slice left line 0 EMPTY. At a grant of
    one line the row then spent its only line on the badge.

    That is worse than the bug this whole branch is fixing: the pre-wrap card at
    least drew one truncated line of the consequence. And it is reached by
    exactly what a model writes into a description — a URL, a path, a hash.

    Measured on the pre-fix tree: 16 frames across 60-100 columns drew an empty
    head for the fixture below. The assertion is on the WRAP rather than on the
    painted line because the grant varies with height; an empty first line is
    the defect at every grant, and at a grant of 1 it is the whole row.
    """
    description = (
        "https://ci.example.internal/pipelines/deploy/production/run/8891234/artifacts/download"
        " rolls the fleet forward and cannot be undone once the canary passes."
    )
    question = AskQuestion(
        id="unbreakable",
        question="Which pipeline should run?",
        options=[
            AskOption(label="Roll forward", description=description),
            AskOption(label="Hold", description="stay on the current build"),
        ],
        recommended=0,
    )
    for width in (60, 70, 80, 100, 130):
        for height in (18, 22, 26, 30, 40):
            app = _AskHost([question])
            async with app.run_test(size=(width, height)) as pilot:
                card = await app.open_picker()
                await pilot.pause()
                layout = card._layout()
                if not layout.description_rows.get(0, 0):
                    continue
                wrapped = card._description_lines(0, layout.width)
                assert wrapped, (width, height)
                # The line the tag shares must carry prose, not just the badge.
                assert wrapped[0].strip(), (width, height, wrapped[:2])
                # And the row's painted description is not blank either.
                drawn = _description_lines_of(card, card.render_lines_for_test())
                assert any(line.strip() for line in drawn), (width, height, drawn)


def test_a_cut_short_description_is_filled_from_the_source_not_a_rejoin() -> None:
    """Review finding (MAJOR): the last kept line of a cut-short description was
    filled by `" ".join(wrapped[position:])`, which INVENTS a space.

    `wrap_cells` breaks a word longer than the row — a URL, a path, a hash; that
    is its documented job — and the pieces either side of such a break never had
    a space between them. Rejoining them with one fabricates a character in text
    the user is being asked to authorise against, which on a path or a URL
    misreads the string itself.

    Pinned as a pure property of the helper rather than through a frame, because
    it is one: whatever `_wrap_tail` returns must be a SUBSTRING of the source.
    A frame test would only see it where `truncate_cells` does not cut before
    the seam, which is why the defect survived a green suite — it needs wide or
    cluster text to become visible.

    The inputs are the four shapes that break a word: an unbreakable URL, a
    path, wide CJK, and keycap clusters whose cell width is not their length.
    """
    from local_operator.tui.widgets.ask_picker import _wrap_tail
    from local_operator.tui.widgets.transcript import wrap_cells

    sources = (
        "1\ufe0f\u20e3" * 5,
        "https://a.example.com/" + "x" * 40,
        "/very/long/path/" + "seg/" * 12,
        "\u4f60\u597d" * 20,
        "a normal sentence that wraps on its spaces and nothing else at all",
    )
    rejoin_was_wrong = 0
    for source in sources:
        for room in (8, 12, 20):
            wrapped = wrap_cells(source, room)
            for position in range(len(wrapped)):
                tail = _wrap_tail(source, wrapped, position)
                assert tail in source, (source[:20], room, position, tail[:20])
                # It is a TAIL: it starts where the wrapped line does, so the
                # fill never re-shows text the lines above already drew.
                assert tail.startswith(wrapped[position]), (source[:20], room, position)
                # The REJECTED expression, evaluated here rather than described
                # in prose. Without this the test would pass on any tree that
                # merely has the helper, and would report an ImportError rather
                # than a defect on the tree that has the bug — proving the
                # helper is new, not that the old fill was wrong.
                rejoin_was_wrong += 0 if " ".join(wrapped[position:]) in source else 1
    # The inputs really do exercise the difference. If a future `wrap_cells`
    # stopped breaking words this drops to zero and the assertions above become
    # a tautology, which should fail loudly rather than pass quietly.
    assert rejoin_was_wrong > 0, rejoin_was_wrong


# --- the reveal (`ctrl+e`) ---------------------------------------------------
#
# Driven against the REAL ``OperatorApp`` through :func:`_real_app_card` and
# :func:`_show`, not against ``_AskHost``, and that is not a stylistic choice —
# it changes the answers. ``_AskHost`` mounts the card into a bare container
# with no stylesheet and no seeded transcript, so the card is not ANCHORED: at
# 150x40 it gets a 25-line budget there against the 20 the real dock leaves it,
# which is enough for (A)'s continuation lines to finish option 1 on their own
# and the reveal is correctly refused. The reported bug lives in the anchored
# budget, so the tests for the key that fixes it have to be measured in it.
#
# ``card.state.selected`` is assigned directly where a test iterates the cursor
# over every row. ``pilot.press("down")`` is the right gesture for pinning the
# KEYMAP and is used for that elsewhere in this file; here the claim is about
# the LAYOUT at each selection, and the free-text row's own key handling would
# otherwise decide which rows the loop can even reach.


@pytest.mark.asyncio
async def test_ctrl_e_reveals_the_selected_rows_full_consequence() -> None:
    """The headline. 150x40 is the size the truncation was reported at, and it
    is the frame where the bug is still visible after slice 1: the card's budget
    is 20 rows, (A) spends its leftovers getting option 1 from one line to three
    of its four, and the sentence that names what the option actually does is
    still off the card.

    So this is the test for the thing the user asked for — reaching the whole
    description — rather than for the mechanism that gets there. Two assertions,
    and they are separate claims:

    - option 1's complete text is present, collapsed on whitespace because the
      block wraps it across lines that each carry the description indent;
    - and the card SAYS the key exists, before and after. A reveal reachable
      only by a user who already knew about it is not reachable.

    The default frame is asserted to be genuinely incomplete first. Without it
    this test would still pass on a card that had simply drawn everything all
    along, which is the state at 190x50 — and then it would be pinning nothing.
    """
    size = (150, 40)
    full = " ".join(_LONG_DESCRIPTIONS[0].split())

    app, card = await _real_app_card(size, [_long_description_question()])
    async with app.run_test(size=size) as pilot:
        await _show(app, pilot, card)

        # The bug, still on screen after slice 1.
        before = card.render_lines_for_test()
        before_prose = " ".join(
            " ".join(line.split()) for line in _description_lines_of(card, before)
        )
        assert full not in before_prose, before_prose
        # ...and the card offers the way out of it.
        assert ("^e", "more") == card._reveal_hint()
        assert "^e more" in _painted_footer(app), _painted_footer(app)

        await pilot.press("ctrl+e")
        await _until(pilot, lambda: card.state.revealed)

        lines = card.render_lines_for_test()
        joined = " ".join(" ".join(line.split()) for line in lines)
        assert full in joined, lines
        # The footer names the way back, on the COMPOSITED frame: a card whose
        # model changed without a repaint would still generate the new hint.
        await _until(pilot, lambda: "^e less" in _painted_footer(app))
        assert "^e less" in _painted_footer(app), _painted_footer(app)
        # And the conversation the question is about is still there. The reveal
        # is bought from the card's own leftovers, never from the transcript.
        assert not app.screen.show_vertical_scrollbar
        assert "answer 5: the audit log still has every row" in "\n".join(_painted_rows(app))


@pytest.mark.asyncio
async def test_the_revealed_card_is_the_same_height_for_every_selection() -> None:
    """The property the whole reveal design rests on: the block reserves the
    height of the TALLEST capped description in the list, not the selected
    row's, and pads the remainder blank.

    A block sized to the cursor's row would be the cheaper implementation and it
    is the one this must not become. Measured under that scheme the card had
    three different heights at 190x50, and a card whose height changes on every
    arrow press moves the footer, re-lays out the dock and shifts the
    conversation under a user who is mid-answer — the `_paint_detail` rule
    (`settings_view.py:3044-3056`) and AGENTS.md's "animated content must
    reserve its row even when it has nothing to show".

    Asserted as ONE distinct line count over every row, which is the strongest
    form: not "close", not "within one", exactly one. The padding is asserted
    too, because a card could hold its height by never drawing anything.
    """
    for size in ((150, 40), (140, 36)):
        app, card = await _real_app_card(size, [_long_description_question()])
        async with app.run_test(size=size) as pilot:
            await _show(app, pilot, card)
            await pilot.press("ctrl+e")
            await _until(pilot, lambda: card.state.revealed)

            heights: set[int] = set()
            reserved: set[int] = set()
            for index in range(card.row_count):
                card.state.selected = index
                card._repaint()
                await pilot.pause()
                heights.add(len(card.render_lines_for_test()))
                reserved.add(card._layout().reveal_rows)

            assert len(heights) == 1, (size, heights)
            # The same reservation at every selection is the MECHANISM behind
            # the height above. Both are asserted because either can be right
            # while the other is wrong: a block that shrank for a short row and
            # a list that grew a line to compensate would hold the total.
            assert len(reserved) == 1, (size, reserved)
            assert reserved.pop() >= 1, size


@pytest.mark.asyncio
async def test_the_reveal_never_takes_the_last_option_row() -> None:
    """Revealing one option's prose must never take another option's LABEL off
    the card. A description is commentary on an answer; a row IS an answer, and
    the priority order (`ask_picker.py:1206-1220`) ranks prose strictly below
    it. On the approval gate a row taken off the card is an authorisation
    choice the user cannot see.

    130x30 and 100x20 are the two shapes the design names: at 130x30 the list
    fits in full and the reveal must not window it, and at 100x20 the list is
    ALREADY windowed, where the failure would be the reveal buying its block out
    of the one row left.

    140x36 and 150x40 are here because those two alone cannot fail. Measured in
    the anchored budget, `reveal_rows` is 0 at both of the design's sizes — the
    plan cannot afford a block there in either state, so a regression that
    bought the block with an option row's money would be INERT at exactly the
    sizes this test names, and the guard would be believed while catching
    nothing. The two sizes added are the ones where the block is genuinely
    bought (`reveal_rows` 4), so the trade the test forbids is reachable.

    Two claims per size, and the second is what keeps the first honest: `page`
    never drops below one drawn row in either state, and wherever the card is
    drawing less than the whole list it BUYS the position line to say so. A card
    that quietly windowed itself to afford prose would satisfy the first alone.

    Asked of `_layout(reveal=...)` at every selection rather than of a single
    keypress, because the block is bought against the SELECTED row's wrap and
    the row with the tallest description is the one that could afford to take a
    row from the list.
    """
    afforded = 0
    for size in ((130, 30), (100, 20), (140, 36), (150, 40)):
        app, card = await _real_app_card(size, [_long_description_question()])
        async with app.run_test(size=size) as pilot:
            await _show(app, pilot, card)

            for index in range(card.row_count):
                card.state.selected = index
                card._repaint()
                await pilot.pause()
                for revealed in (False, True):
                    plan = card._layout(reveal=revealed)
                    assert plan.page >= 1, (size, index, revealed, plan.page)
                    if plan.page < card.row_count:
                        assert plan.show_position, (size, index, revealed, plan.page)
                        assert "of {}".format(card.row_count) in "\n".join(
                            card.render_lines_for_test()
                        ), (size, index, revealed)

                # The claim with teeth: turning the reveal ON never costs a
                # LABEL. Continuation lines are fair game (they are the last
                # thing bought and the first thing lost); an option row is not,
                # because a row is an ANSWER.
                default_plan = card._layout(reveal=False)
                revealed_plan = card._layout(reveal=True)
                assert revealed_plan.page >= default_plan.page, (
                    size,
                    index,
                    default_plan.page,
                    revealed_plan.page,
                )
                afforded += 1 if revealed_plan.reveal_rows >= 1 else 0

    # The sweep above only tests anything at sizes where a block is actually
    # bought. Asserted rather than assumed: if the budgets shift under a future
    # change until no size here affords one, this test degrades to a tautology
    # silently, and that is the failure mode it was rewritten to avoid.
    assert afforded >= 1, afforded


@pytest.mark.asyncio
async def test_the_footer_offers_the_reveal_only_where_it_does_something() -> None:
    """This row already refuses to name dead keys — the digits on a one-row
    window, the whole keymap while the composer holds the caret — and `^e` is
    dead at both ends of the size range for OPPOSITE reasons.

    At 190x50 (A)'s continuation lines have already drawn every description in
    full, so the key would toggle a mode that changes nothing: measured, it adds
    three blank rows and no words. At 100x20 the budget is 6 and step 7a has
    nothing left to buy a reveal line with, so the revealed plan equals the
    default one. Between them, at 150x40, the selected row is genuinely cut and
    the plan can afford the block — so the hint is drawn.

    Both the absent cases are asserted on the COMPOSITED footer as well as on
    the predicate. The hint being derivable is not the claim; what the user is
    told is.
    """
    for size, offered in (((190, 50), False), ((100, 20), False), ((150, 40), True)):
        app, card = await _real_app_card(size, [_long_description_question()])
        async with app.run_test(size=size) as pilot:
            await _show(app, pilot, card)

            assert card._offers_reveal() is offered, (size, card._reveal_hint())
            painted = _painted_footer(app)
            assert ("^e" in painted) is offered, (size, painted)
            # Whatever the footer says, the exit survives — it is the one hint
            # this row defends hardest, and `^e` is inserted immediately before
            # it in the ladder precisely so it sheds first.
            assert "esc" in painted, (size, painted)


@pytest.mark.asyncio
async def test_the_reveal_block_is_drawn_under_the_row_it_explains() -> None:
    """Regression, found in a rendered frame and invisible to the suite that
    was green around it: the block was appended after the LIST rather than under
    the selected row, so at 150x40 the paragraph explaining option 1 sat
    directly beneath `Other (type your own)`, indented exactly as that row's own
    description would be.

    That is misattributed consequence text. On `ApprovalPrompt` it is a user
    reading "stop asking for this session" with Enter still on "Allow" — the
    same misattribution hover was rejected for, arrived at by a different route.

    Pinned positionally, on `_line_rows`, because position is the entire defect:
    the words were all on the card in the broken frame too. The block's own
    lines map to `None` (they are chrome about a row, not the row — a click on
    their blank padding must never answer), so the assertion is that every line
    between the selected row's label and the NEXT row's label belongs to the
    selected row or to the block, and that the block is not sitting after the
    last row.
    """
    size = (150, 40)
    app, card = await _real_app_card(size, [_long_description_question()])
    async with app.run_test(size=size) as pilot:
        await _show(app, pilot, card)
        await pilot.press("ctrl+e")
        await _until(pilot, lambda: card.state.revealed)

        for index in (0, 2):
            card.state.selected = index
            card._repaint()
            await pilot.pause()
            lines = card.render_lines_for_test()
            rows = card._line_rows
            first = " ".join(_LONG_DESCRIPTIONS[index].split()).split(" ")[0:6]
            opening = " ".join(first)

            # Where the block's first line actually is.
            block_at = next(
                (
                    position
                    for position, line in enumerate(lines)
                    if opening in " ".join(line.split()) and rows[position] is None
                ),
                None,
            )
            assert block_at is not None, (index, lines)
            # The selected row's own label line, and the next row's.
            label_at = rows.index(index)
            following = [
                position for position, row in enumerate(rows) if row is not None and row > index
            ]
            assert label_at < block_at, (index, label_at, block_at)
            if following:
                assert block_at < min(following), (index, block_at, min(following), lines)
            # The block's blank padding is click-inert: mapped to a row, empty
            # space under the cursor's option would answer the question.
            assert rows[block_at] is None, (index, lines)


@pytest.mark.asyncio
async def test_the_reveal_is_advertised_from_the_selected_row_not_any_drawn_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression, and the one with the worst blast radius: `^e` must be offered
    on the SELECTED row being cut, never on ANY drawn row being cut.

    The block only ever shows the row under the cursor, but it is bought out of
    the pool the OTHER rows' prose is drawn from. Asked the loose way, the
    approval card at 40x24 advertises `^e` because *Deny*'s consequence is cut
    — and the cursor is on *Allow*, whose consequence is already complete. The
    trade is then pure loss: the block redraws text the user can already read
    and the rows it was bought from lose theirs.

    Reintroduced here by monkeypatching the predicate to its rejected form
    rather than by editing the widget, per AGENTS.md's "prove the test can still
    fail". The patched half is what makes this a guard instead of a restatement
    of current behaviour: it shows the assertion goes red against the exact
    variant that was shipped and reverted.
    """
    from local_operator.tui.widgets.approval import ApprovalPrompt

    def _any_drawn_row_is_cut(self: AskPickerScreen) -> bool:
        """The rejected variant: cut ANYWHERE on the card, not under the caret."""
        plan = self._layout(reveal=False)
        cut = any(
            len(self._reveal_wrap(index, plan.width)) > plan.description_rows.get(index, 0)
            for index in self._window(plan.page)
        )
        return cut and self._layout(reveal=True).reveal_rows >= 1

    size = (40, 24)
    app = _AskHost([])
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        card = ApprovalPrompt("bash", "rm -rf /Users/x/project/data")
        await app.query_one("#prompt-host", Container).mount(card)
        await pilot.pause()

        plan = card._layout(reveal=False)
        selected = card.state.selected
        cut = {
            index: (
                len(card._reveal_wrap(index, plan.width)),
                plan.description_rows.get(index, 0),
            )
            for index in card._window(plan.page)
        }
        # The frame this is about: a NON-selected row is cut, the selected one
        # is not. If the approval strings ever change so that this is no longer
        # true, this test is measuring nothing and must be re-derived rather
        # than deleted.
        assert any(want > got for want, got in cut.values()), cut
        assert cut[selected][0] <= cut[selected][1], (selected, cut)

        # The card is silent about a key that would only cost the user text.
        assert card._reveal_hint() is None, cut

        # And the rejected variant is not silent — the guard can go red.
        monkeypatch.setattr(AskPickerScreen, "_reveal_is_useful", _any_drawn_row_is_cut)
        assert card._reveal_hint() == ("^e", "more"), cut
        card._settled = True


@pytest.mark.asyncio
async def test_ctrl_e_is_inert_on_a_card_too_narrow_to_advertise_it() -> None:
    """Regression: the key must not fire where the footer does not name it.

    `^e` sheds from the footer on a narrow card so `esc deny` stays whole. A key
    that still worked there would be an unadvertised gesture on the surface that
    authorises tool calls — and it is not a harmless one: measured on the
    approval card at 30x24, firing it replaces the three consequence lines with
    a single one, so the user loses two of the three things they are choosing
    between, by a key the card never offered.

    Both halves are asserted. The press does nothing (the state does not flip
    and the frame is byte-identical), and the damage it WOULD do is measured by
    forcing the state directly — otherwise a card that had simply stopped
    drawing a block would look identical to one correctly refusing the key.
    """
    from local_operator.tui.widgets.approval import ApprovalPrompt

    consequences = (
        "run this call and ask again next time",
        "refuse this call; the turn continues",
        "stop asking for this session",
    )
    size = (30, 24)
    app = _AskHost([])
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        card = ApprovalPrompt("bash", "rm -rf /Users/x/project/data")
        await app.query_one("#prompt-host", Container).mount(card)
        await pilot.pause()

        before = [line.rstrip() for line in card.render_lines_for_test()]
        # The frame this is about: the hint has been shed to keep the exit, and
        # the reveal WOULD otherwise buy a line here.
        assert card._reveal_is_useful(), before
        assert not card._offers_reveal(), before
        assert "^e" not in before[-1], before
        assert "esc deny" in before[-1], before

        await pilot.press("ctrl+e")
        await pilot.pause()
        assert not card.state.revealed, card.render_lines_for_test()
        assert [line.rstrip() for line in card.render_lines_for_test()] == before

        # What the ungated key would have cost, on the authorisation frame.
        # Forced past the gate, not pressed: this is the measurement that makes
        # the assertion above a safety property rather than a preference.
        card.state.revealed = True
        card._repaint()
        await pilot.pause()
        forced = " ".join(" ".join(line.split()) for line in card.render_lines_for_test())
        kept = [text for text in consequences if text in forced]
        assert len(kept) < len(consequences), forced
        card._settled = True


@pytest.mark.asyncio
async def test_the_reveal_mode_does_not_follow_the_user_to_the_next_question() -> None:
    """`revealed` is per QUESTION, held in `_QuestionState`, not per card.

    A multi-question ask is several unrelated decisions on one widget. Carrying
    the mode across them would open the second question in a state the user
    never asked for on it — and on a mixed run where one question is an
    approval, that is a frame chosen by a keypress aimed at something else.
    """
    second = AskQuestion(
        id="second",
        question=_LONG_QUESTION_TEXT,
        options=[
            AskOption(label=label, description=description)
            for label, description in zip(_LONG_LABELS, _LONG_DESCRIPTIONS)
        ],
    )
    size = (150, 40)
    app, card = await _real_app_card(size, [_long_description_question(), second])
    async with app.run_test(size=size) as pilot:
        await _show(app, pilot, card)

        await pilot.press("ctrl+e")
        await _until(pilot, lambda: card.state.revealed)
        assert card.state.revealed

        await pilot.press("enter")
        await _until(pilot, lambda: card._index == 1)
        # The next question opens in the card's default state...
        assert not card.state.revealed
        # ...and still says the key is there, so the mode is available rather
        # than lost.
        assert card._reveal_hint() == ("^e", "more")


# --- what survives a card with no room --------------------------------------
#
# Driven against the REAL ``OperatorApp`` rather than ``_AskHost``: the host
# above declares no ``CSS_PATH``, so the card has no ``max-height`` and no
# padding and therefore cannot exhibit a clip at all. These are the sizes the
# card was clipping its own footer at.


SHORT_SIZES = ((100, 14), (100, 16), (54, 14), (30, 12), (24, 10), (20, 8))


def _long_question(recommended: int | None = 1) -> AskQuestion:
    return _question(
        text="The stale rows still reference the email column. What should happen to them?",
        labels=("Drop the column and the rows with it", "Backfill from the audit log", "Leave it"),
        descriptions=("nothing reads it any more", "slower, keeps the history", "cheapest now"),
        recommended=recommended,
    )


async def _real_app_card(size: tuple[int, int], questions: list[AskQuestion]):
    """The card and a real ``OperatorApp``, with the stylesheet applied.

    Returned unmounted; :func:`_show` puts it in the app's real prompt host.
    The pair is kept (rather than mounting here) because every caller wants to
    drive the app's own ``run_test`` context around it.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession

    session = FakeSession()

    async def factory():
        return session

    return OperatorApp(lambda: factory()), AskPickerScreen(questions)


def _baseline_app():  # type: ignore[no-untyped-def]
    """A real app with no prompt raised, for measuring the dock on its own.

    The comparison every overflow assertion in this file needs: at the shortest
    terminals the composer and status band already exceed the screen, so the
    question is never "does anything overflow" but "does raising a question
    make it worse".
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    return OperatorApp(lambda: _factory(FakeSession()))


async def _seed_conversation(app, pilot, turns: int = 6) -> None:  # type: ignore[no-untyped-def]
    """Put a real conversation in the transcript before measuring anything.

    Load-bearing, not scene-setting. An app with an EMPTY transcript is still
    in the BOOT layout, and `Screen.boot TranscriptView` drops the transcript's
    padding from `1 0 1 1` to `0 0 0 1` — one vertical row instead of two. Every
    geometry assertion in this file used to run in that state, so a card that
    under-reserved the transcript by exactly one row measured clean at every
    size, and the error only appeared once a user had said anything at all
    (F2, agent review round 1).

    A prompt over an empty transcript is also not a state worth pinning: the
    agent asks because it is in the middle of something.
    """
    from local_operator.tui.widgets.assistant import AssistantBlock
    from local_operator.tui.widgets.transcript import UserBlock

    for turn in range(turns):
        app._append_block(UserBlock(f"turn {turn}: what should happen to the stale rows?"))
        prose = AssistantBlock()
        prose.update_text(f"answer {turn}: the audit log still has every row, so a backfill works.")
        app._append_block(prose)
    await pilot.pause()


async def _settle(app, pilot) -> None:  # type: ignore[no-untyped-def]
    """Pump until the screen's geometry stops moving.

    Three CONSECUTIVE identical frames, not two: the transient after a mount or
    a boot is itself two frames long (the pre-arrange height repeats once before
    the dock re-arranges), so a two-frame agreement can match ON the transient
    and return early. Every overflow assertion in this file compares two such
    measurements, and both sides have to be settled or the comparison is between
    a settled number and a mid-arrange one.
    """
    # Settles on a RESOLVED frame — one where nothing overflows — and falls
    # back to a merely stable one for the sizes that genuinely cannot fit the
    # composer at all (20x5 and under), where overflow is the settled state and
    # the assertions compare against a no-prompt baseline instead.
    #
    # This used to need a much longer budget, and that was hiding a real defect
    # rather than absorbing jitter: the card under-counted the dock by one row
    # (it summed the dock's children instead of measuring the dock, missing the
    # row the container spends on itself), so at some sizes the overflow was
    # permanent and no amount of pumping cleared it. With that fixed every size
    # in this file resolves on the first frame.
    stable = 0
    previous = None
    for _ in range(30):
        await pilot.pause()
        size = tuple(app.screen.size)
        virtual = tuple(app.screen.virtual_size)
        if size == virtual:
            return
        stable = stable + 1 if (size, virtual) == previous else 0
        previous = (size, virtual)
        if stable >= 4:
            return


async def _show(app, pilot, card) -> None:  # type: ignore[no-untyped-def]
    """Raise ``card`` through the app's OWN mounting path, and let it settle.

    ``app._mount_prompt`` rather than a hand-rolled mount, because these tests
    exist to catch what the real composition clips, and a helper that mounts
    differently measures a layout the app never produces. Written by hand it
    also set ``display`` on the host directly, which skipped the drawability
    sync the app does — so the host kept a row for a card that had hidden
    itself, and the test saw two rows of overflow that the app does not have.

    The settling belongs HERE, not at each call site. The host's visibility is
    resolved through ``call_after_refresh`` plus the card's own repaint, so a
    caller that paused twice read a half-settled dock and measured an overflow
    that does not survive the next frame.

    Seeding the conversation belongs here for a stronger reason: measured
    against an empty transcript, this whole file was blind to a one-row
    under-reservation, because the boot layout's transcript padding cancelled
    it exactly. See :func:`_seed_conversation`.
    """
    await _seed_conversation(app, pilot)
    app._mount_prompt(card)
    # Settle until the screen stops moving, rather than for a fixed number of
    # pauses. Mounting the card takes two frames to reach its final height (the
    # dock re-arranges, then `_sync_prompt_host` resolves the host's row), and a
    # fixed count is a bet on how many pumps that takes on a loaded machine —
    # lost intermittently, where the assertion then read a mid-arrange height
    # and reported an overflow that does not survive the next frame.
    await _settle(app, pilot)


def _painted_footer(app) -> str:  # type: ignore[no-untyped-def]
    """The card's key-hint row AS PAINTED, from the compositor.

    Deliberately not `render_lines_for_test()`: that re-derives the text, so it
    cannot see a card whose model changed and whose body was never repainted —
    which is exactly the defect it once hid (D13, design round 4).
    """
    for row in reversed(_painted_rows(app)):
        if "esc" in row:
            return row
    return ""


def _painted_rows(app) -> list[str]:  # type: ignore[no-untyped-def]
    """The rows the SCREEN is showing, blank ones dropped.

    The compositor rather than the card's own ``Text``, because the clip that
    loses the footer is only observable here: an overflowing child still reports
    the height it WANTED, and ``_card_text`` used to append the footer
    unconditionally, so a card whose keys never reached the terminal could still
    satisfy an assertion on its generated lines.
    """
    strips = app.screen._compositor.render_strips()
    return [text for text in (strip.text.strip() for strip in strips) if text]


async def _until(pilot, predicate, *, ceiling: int = 200) -> None:  # type: ignore[no-untyped-def]
    """Idle-pump until PREDICATE holds, bounded by a deadlock guard.

    Replaces a fixed ``pilot.pause(<seconds>)`` bet on how long a repaint, a
    focus move or an answer takes. ``pilot.pause()`` with no argument returns
    when the event loop next goes idle — as soon as the awaited work is actually
    done — so polling the observable state the next assertion needs is both
    faster and steadier than a wall-clock sleep, which loses its bet and flakes
    under the parallel CPU contention of ``-n``. The ceiling is generous on
    purpose: it exists only to fail a genuine deadlock, never to stand in for a
    timing assumption.
    """
    for _ in range(ceiling):
        await pilot.pause()
        if predicate():
            return


@pytest.mark.asyncio
async def test_the_keys_and_an_option_survive_every_terminal_the_card_fits_in() -> None:
    """The blocker this layout exists for: at 100x14 the card drew a question,
    one option of four and NO keys, and at 30x12 only the title and the
    question — zero options and nothing saying how to leave a card the turn is
    parked on. Chrome the card cannot do without is paid for FIRST now, so a
    short terminal abbreviates the list instead of amputating the footer."""
    for size in SHORT_SIZES:
        # What the dock measures with a conversation but NO prompt. At the
        # shortest of these the composer already exceeds the screen on its own,
        # so "nothing overflows" is not true of this app there and never was;
        # what must be true is that raising a question does not make it worse.
        baseline_app = _baseline_app()
        async with baseline_app.run_test(size=size) as pilot:
            await _seed_conversation(baseline_app, pilot)
            await _settle(baseline_app, pilot)
            baseline = tuple(baseline_app.screen.virtual_size)

        app, screen = await _real_app_card(size, [_long_question()])
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _show(app, pilot, screen)
            lines = screen.render_lines_for_test()
            if not lines:
                # Below the card's minimum the honest card is no card, and the
                # dock must not keep a row for it either. Asserted rather than
                # skipped: a host still reserving space for a prompt painting
                # nothing is what pushed the dock past the screen at 20x8.
                assert not app.query_one("#prompt-host").display, size
                assert tuple(app.screen.virtual_size) == baseline, (size, baseline)
                continue
            card = screen.query_one(".ask-picker")
            # Every line the card lays out is a line the SCREEN actually drew.
            # Measured on the COMPOSITED screen rather than against an
            # arithmetic `room`: a child that overflows its container still
            # reports the height it wanted, so measuring that agreed with the
            # clip instead of catching it — and now that the card shares the
            # screen with the transcript and the composer, a recomputed budget
            # here would be a second, drifting copy of `_body_rows`.
            painted = _painted_rows(app)
            assert lines[-1].strip() in "\n".join(painted), (size, lines[-1], painted)
            assert card.region.height <= screen.size.height, (size, card.region.height)
            assert "esc" in lines[-1] or "enter" in lines[-1], (size, lines[-1])
            if not screen.visible_rows:
                # The collapsed card: the exit, and the question if there is a
                # row for it. The footer must advertise ONLY the exit — `enter`
                # would commit a selection the user cannot see, and the digits
                # would jump within a list that is not drawn.
                assert lines[-1].strip() == "esc skip", (size, lines)
                continue
            if len(screen.visible_rows) < screen.row_count:
                # Rows are hidden, so the card owes the reader an account of
                # that — UNLESS the row it would take is the one carrying the
                # question. The question outranks the count (D1), so what is
                # pinned is that the card is never silent about BOTH: it shows
                # the count, or it shows what is being asked.
                text = "\n".join(lines)
                assert f"of {screen.row_count}" in text or screen.question.question[:20] in text, (
                    size,
                    lines,
                )
            # The anchoring guarantee itself: the dock never grows past the
            # screen, so the transcript is never scrolled out from under the
            # question the user is being asked.
            assert tuple(app.screen.virtual_size) == baseline, (
                size,
                tuple(app.screen.virtual_size),
                baseline,
            )


@pytest.mark.asyncio
async def test_the_footer_is_the_last_line_the_card_gives_up() -> None:
    """A terminal too short to draw the list still has to say how to leave it.

    Asserted on the COMPOSITED SCREEN. Round 2's version of this test read the
    card's own ``render_lines_for_test`` instead, where ``_card_text`` appended
    the footer unconditionally and ``_window`` returned at least one row — so it
    held by construction whatever the terminal painted, and round 3 ran it with
    the fix reverted and watched it pass (R10).

    What is pinned is what round 3 measured. Five and six terminal rows leave
    the body one line and two (two rows to the screen's padding, two to the
    card's), and the card spends them on the footer ALONE: an option row with no
    question above it and no count beside it is an answer to nothing, and those
    two heights used to paint exactly that with no keys under it. Seven rows and
    up draw the list again, keys still on the last painted line. Four rows and
    under leave the body nothing, so the card is not drawn at all rather than
    drawn and clipped — and nothing overflows the screen either way.
    """
    for width in (20, 30, 40, 100):
        for height in (5, 6, 7, 8, 12):
            size = (width, height)
            # What the dock measures with NO prompt raised. At the shortest of
            # these the composer and its status band already exceed the screen,
            # so "nothing overflows" is not true of this app at 20x5 and never
            # was; what must be true is that raising a question does not make it
            # worse. Captured per size so the comparison below is like-for-like.
            baseline_app = _baseline_app()
            async with baseline_app.run_test(size=size) as pilot:
                # Seeded and settled by the SAME rules the measured app is, so
                # the two numbers describe the same layout with and without a
                # prompt. An unseeded baseline is in the BOOT layout, whose
                # transcript padding differs — comparing against it measures the
                # boot/conversation difference as if it were the prompt's cost.
                await _seed_conversation(baseline_app, pilot)
                await _settle(baseline_app, pilot)
                baseline = tuple(baseline_app.screen.virtual_size)

            app, screen = await _real_app_card(size, [_long_question()])
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await _show(app, pilot, screen)
                painted = _painted_rows(app)
                lines = screen.render_lines_for_test()
                if not lines:
                    # Too short for even the exit. The card draws nothing and
                    # the dock keeps no row for it, which is the honest card;
                    # the clip is what happens when it draws anyway.
                    assert not app.query_one("#prompt-host").display, size
                    assert tuple(app.screen.virtual_size) == baseline, (size, baseline)
                    continue
                # The keys reached the TERMINAL. The footer is bought first and
                # drawn last, so a clipped tail would take it and nothing else —
                # this is the assertion the card's own `render_lines_for_test`
                # cannot make, because an overflowing card still reports the
                # lines it WANTED.
                #
                # `painted[-1]` is no longer the card's last row: the composer
                # and the status band are painted below it now that the card is
                # docked rather than covering the screen. So the footer is
                # located in the painted frame instead of assumed to end it,
                # which is a stronger check anyway — it fails both if the footer
                # is missing and if it was clipped to something else.
                assert lines[-1].strip() in "\n".join(painted), (size, lines[-1], painted)
                assert "esc" in lines[-1] or "enter" in lines[-1], (size, lines[-1])
                if not screen.visible_rows:
                    # One line is a line for the exit, and ONLY the exit: with
                    # no option row on screen, `enter` would commit a selection
                    # the user cannot see and the digits would jump within a
                    # list that is not there. Measured before the fix: at a
                    # 5-row terminal `down down down enter` committed an option
                    # nobody had been shown (round 4, R14).
                    assert lines[-1].strip() == "esc skip", (size, lines)
                elif len(screen.visible_rows) == 1:
                    # One row beside the footer is what makes the free-text row
                    # echo what is being typed into it (round 4, R15). The count
                    # is not affordable at that budget.
                    assert f"of {screen.row_count}" not in "".join(lines), (size, lines)
                # Nothing was clipped to get there: raising the question left
                # the dock exactly as tall as it was without one.
                assert tuple(app.screen.virtual_size) == baseline, (
                    size,
                    tuple(app.screen.virtual_size),
                    baseline,
                )

    # Four rows and under: the body has no drawable line, and a card drawn into
    # none of them is the clip itself. Its own padding is rows the screen has
    # not got, so laying it out makes the screen scrollable.
    #
    # The screen is ALREADY scrollable at these sizes with no prompt at all —
    # the composer and its status band do not fit in four rows, which is a
    # pre-existing property of the dock and not something this card can fix.
    # So the assertion is a COMPARISON against that baseline rather than an
    # absolute: raising a question must not make the overflow worse. Measured
    # without this comparison the test asserted `size == virtual_size` at 40x4
    # and failed on a tree where the prompt was never mounted.
    for size in ((40, 4), (40, 3), (30, 2), (20, 1)):
        base_app = _baseline_app()
        async with base_app.run_test(size=size) as pilot:
            await _seed_conversation(base_app, pilot)
            await _settle(base_app, pilot)
            baseline = tuple(base_app.screen.virtual_size)

        app, screen = await _real_app_card(size, [_long_question()])
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _show(app, pilot, screen)
            for _ in range(4):
                await pilot.pause()
            assert screen.render_lines_for_test() == [], size
            assert not app.query_one("#prompt-host").display, size
            assert tuple(app.screen.virtual_size) == baseline, (
                size,
                tuple(app.screen.virtual_size),
                baseline,
            )


@pytest.mark.asyncio
async def test_a_recommended_option_never_widens_the_card_past_the_screen() -> None:
    """`virtual_size` over `size` is the condition AGENTS.md calls always a bug
    on this app. The 15-cell tag used to be appended on top of the label's
    minimum rather than dropped, which bought two cells of overflow at 30x12 —
    and only with a recommendation, which is how it was found."""
    for size in SHORT_SIZES:
        # The no-prompt baseline for this size, seeded and settled identically:
        # the smallest of these terminals cannot fit the composer alone, so the
        # question is whether the RECOMMENDATION costs anything, not whether the
        # app fits.
        baseline_app = _baseline_app()
        async with baseline_app.run_test(size=size) as pilot:
            await _seed_conversation(baseline_app, pilot)
            await _settle(baseline_app, pilot)
            baseline = tuple(baseline_app.screen.virtual_size)

        for recommended in (0, None):
            app, screen = await _real_app_card(size, [_long_question(recommended)])
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                await _show(app, pilot, screen)
                assert tuple(app.screen.virtual_size) == baseline, (
                    size,
                    recommended,
                    tuple(app.screen.virtual_size),
                    baseline,
                )


# --- the footer's ladder ----------------------------------------------------


@pytest.mark.asyncio
async def test_a_narrow_multi_select_keeps_the_only_key_that_can_answer_it() -> None:
    """`space` is the ONLY key that ticks a box, so a multi-select that dropped
    it offered five empty boxes and an Enter that does nothing. It outranks even
    `esc` here, and the words go before any of the keys do."""
    question = _question(
        labels=("Unit suite", "Integration suite", "Visual snapshots", "Load test"),
        descriptions=("2 minutes", "11 minutes", "flaky", "45 minutes"),
        multi=True,
    )
    for width in (46, 40, 34, 30):
        app = _AskHost([question])
        async with app.run_test(size=(width, 24)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            footer = screen.render_lines_for_test()[-1]
            assert "space" in footer, (width, footer)
            assert cell_len(footer) <= width, (width, footer)


@pytest.mark.asyncio
async def test_the_footer_gives_up_words_before_it_gives_up_keys() -> None:
    """A bare key still names a key that exists; a dropped hint is a key nobody
    can discover. Narrowing the card should therefore lose `move`, `jump` and
    `answer` before it loses `↑↓`, `1-9` or `esc` — and `skip` last of the
    words, because it is the one whose meaning is not guessable from its key."""
    question = _question(labels=("A", "B"), descriptions=("", ""))
    app = _AskHost([question])
    async with app.run_test(size=(100, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert screen.render_lines_for_test()[-1].strip() == (
            "↑↓ move · 1-9 jump · enter answer · esc skip"
        )

    app = _AskHost([question])
    async with app.run_test(size=(32, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        # Every key still on the card, and only the escape route still worded.
        assert screen.render_lines_for_test()[-1].strip() == "↑↓ · 1-9 · enter · esc skip"

    app = _AskHost([question])
    async with app.run_test(size=(26, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        assert screen.render_lines_for_test()[-1].strip() == "↑↓ · 1-9 · enter · esc"

    # Narrower than four keys: hints go in the same order, escape route last.
    #
    # 16 rather than the 18 this rung used to need, and the two cells are the
    # point rather than a tolerance: the card spends the column it was given
    # instead of a budget derived from the terminal, so every rung of this
    # ladder now engages two cells LATER — the card reaches this one only when
    # it is genuinely that narrow. The ORDER is what this test pins and it is
    # unchanged; the widths are a consequence of the card's reach, which is why
    # they move when its reach is corrected. This host has no stylesheet
    # (`_AskHost` declares no `CSS_PATH`), so its card is the full terminal
    # width; under the real sheet the same rung lands at 18 columns, measured.
    app = _AskHost([question])
    async with app.run_test(size=(16, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        footer = screen.render_lines_for_test()[-1].strip()
        assert footer == "enter · esc", footer


# --- saying no out loud -----------------------------------------------------


@pytest.mark.asyncio
async def test_a_refused_enter_says_why_and_takes_it_back() -> None:
    """Refusing to advance is right; saying nothing about it is not. The frame
    of the rejected press used to be byte-identical to the frame before it."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("3")  # the free-text row, with nothing typed in it
        before = screen.render_lines_for_test()[-1]
        await pilot.press("enter")
        await pilot.pause()
        assert "type an answer first" in screen.render_lines_for_test()[-1]
        assert app.answered == []
        # And it is gone the moment there is an answer to take.
        await pilot.press("x")
        await pilot.pause()
        assert "type an answer first" not in screen.render_lines_for_test()[-1]
        await pilot.press("backspace")
        await pilot.pause()
        assert screen.render_lines_for_test()[-1] == before


@pytest.mark.asyncio
async def test_a_refused_enter_on_a_multi_select_names_the_key_that_answers_it() -> None:
    app = _AskHost([_question(multi=True)])
    async with app.run_test(size=(100, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert "space toggles" in screen.render_lines_for_test()[-1]
        await pilot.press("space")
        await pilot.pause()
        assert "space toggles" not in screen.render_lines_for_test()[-1]


# --- the accent, the badge and the digits -----------------------------------


@pytest.mark.asyncio
async def test_the_accent_marks_what_enter_takes_and_not_where_the_cursor_is() -> None:
    """One ink, one claim. On a multi-select Enter takes the TICKED rows, so
    painting the cursor's label green pointed at the row Enter would not take
    and spent the accent on two different statements in one frame."""
    app = _AskHost([_question(labels=("A", "B", "C"), descriptions=("", "", ""), multi=True)])
    async with app.run_test(size=(100, 24)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        await pilot.press("space")  # tick row 1
        await pilot.press("down")
        await pilot.press("down")  # cursor now on row 3, unticked
        await pilot.pause()
        accent = theme_mod.semantic_color("accent")

        def inks(line) -> set[str]:
            found = set()
            for span in line.spans:
                colour = span.style.color
                if colour is not None and colour.triplet is not None:
                    found.add(colour.triplet.hex)
            return found

        lines = screen._card_text().split("\n")
        ticked = next(line for line in lines if line.plain.strip().startswith("1."))
        cursored = next(line for line in lines if line.plain.startswith("❯"))
        assert accent in inks(ticked), ticked.plain
        assert accent not in inks(cursored), cursored.plain


@pytest.mark.asyncio
async def test_the_badge_no_longer_shortens_the_option_it_promotes() -> None:
    """The tag was charged to the label's budget, so the recommended row carried
    the shortest label on the card — a badge that truncates what it promotes."""
    label = "Run the whole backfill in one transaction against the primary and eat the lock"
    app = _AskHost(
        [
            _question(
                labels=(label, label),
                descriptions=("holds a lock for forty minutes", "same, behind a flag"),
                recommended=0,
            )
        ]
    )
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        lines = [line.rstrip() for line in screen.render_lines_for_test()]
        promoted = next(line for line in lines if " 1. " in line)
        sibling = next(line for line in lines if " 2. " in line)
        assert cell_len(promoted) == cell_len(sibling), (promoted, sibling)
        assert RECOMMENDED_TAG not in promoted
        # It moved to the row's own second line, where the prose had room.
        assert lines[lines.index(promoted) + 1].strip().startswith(RECOMMENDED_TAG)


@pytest.mark.asyncio
async def test_the_free_text_row_keeps_a_key_when_the_list_outruns_the_digits() -> None:
    """With ten or more options `Other` drew a blank gutter while the footer
    still offered `1-9 jump`, so the one answer that is not on the list was the
    one row no digit reached."""
    labels = tuple(f"Candidate fix number {n}" for n in range(1, 13))
    app = _AskHost([_question(labels=labels, descriptions=("",) * 12)])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
        assert "0. Other" in text, text
        assert "0-9 jump" in text, text
        await pilot.press("0")
        assert screen.selected_index == screen.other_row


# --- the anchoring guarantee -------------------------------------------------
#
# The reason this surface was moved out of a `ModalScreen`. A modal covered the
# conversation the question was about, so a user who needed to re-read the tool
# output, the error, or the plan in order to ANSWER had to dismiss the question
# first — and could not scroll at all while it was up.


@pytest.mark.asyncio
async def test_the_conversation_stays_readable_behind_a_question() -> None:
    """The card is a band above the composer, never a screen over the chat.

    Pinned on the COMPOSITED frame rather than on the widget tree: what matters
    is that conversation text is painted on the terminal at the same time as the
    question, which is exactly what the modal made impossible.
    """
    from local_operator.tui.widgets.transcript import TranscriptView

    app = _baseline_app()
    card = AskPickerScreen([_long_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        # `_show` seeds the conversation (see `_seed_conversation`), so the
        # assertions below read the same text every other test in this file
        # measures against rather than a second, divergent fixture.
        await _show(app, pilot, card)

        painted = "\n".join(_painted_rows(app))
        # The question is on screen...
        assert "the agent needs your decision" in painted
        # ...and so is the conversation it is about. The last exchange is the
        # one a user would be reading to answer, so it is the one pinned.
        assert "answer 5: the audit log still has every row" in painted
        # The transcript keeps a real share of the screen. Asserted as a
        # PROPORTION of the rows the two actually divide, not as "more than the
        # card": a question with four options and a description each is
        # legitimately tall, and the guarantee is that the conversation is still
        # substantially there, not that it always wins. At 100x30 this is 9 rows
        # of conversation against a 13-row card; the modal left zero.
        transcript = app.query_one(TranscriptView)
        divisible = transcript.region.height + card.region.height
        assert transcript.region.height >= MIN_TRANSCRIPT_ROWS
        assert transcript.region.height >= divisible * (1 - PROMPT_HEIGHT_SHARE)
        # And the composer is still there to type into, below the question.
        assert app.query_one("#input-shell").region.y > card.region.y


@pytest.mark.asyncio
async def test_a_long_description_never_costs_the_conversation_its_share() -> None:
    """The anchoring rule, re-asserted against the input that most wants to
    break it: descriptions long enough to fill the screen on their own.

    Separated from the test above rather than folded into it because it is a
    different claim about a different fixture. That one pins the rule for an
    ordinary question; this one pins that WRAPPING cannot become a way around
    it. The rejected "wrap everything" design needs 12 lines where 5 are spent
    at 150x40, and the seven extra rows can only come from the transcript —
    which is the modal behaviour this surface was rewritten to remove.

    The property holds by construction if the continuation lines are bought
    from `remaining` at the bottom of the priority order, because `_body_rows`
    is a CAP the allocator spends within and never a request it can raise. That
    is exactly the reasoning a test should be pinning rather than trusting: the
    single most likely way to ship a regression here is `_body_rows`'s `wanted`
    cap being updated with the new line counts and taking the anchoring share
    with it (design §11 risk 2).
    """
    from local_operator.tui.widgets.transcript import TranscriptView

    for size in ((150, 40), (190, 50)):
        app = _baseline_app()
        card = AskPickerScreen([_long_description_question()])
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _show(app, pilot, card)

            painted = "\n".join(_painted_rows(app))
            assert "the agent needs your decision" in painted
            # The conversation the question is ABOUT is still on screen.
            assert "answer 5: the audit log still has every row" in painted

            transcript = app.query_one(TranscriptView)
            divisible = transcript.region.height + card.region.height
            assert transcript.region.height >= MIN_TRANSCRIPT_ROWS, size
            assert transcript.region.height >= divisible * (1 - PROMPT_HEIGHT_SHARE), size
            # A card that overflowed its share would make the screen scrollable,
            # which AGENTS.md calls always a bug on this app.
            assert not app.screen.show_vertical_scrollbar, size


@pytest.mark.asyncio
async def test_the_question_sits_above_the_dock_band_and_stays_put() -> None:
    """A question outranks status: it is what the turn is parked on.

    The band (subagent jobs, todos) grows and shrinks on its own as work comes
    and goes. Below the question that movement would shift the card under the
    user's cursor mid-answer, so the prompt host is ordered above it.
    """
    app = _baseline_app()
    card = AskPickerScreen([_long_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await _show(app, pilot, card)
        host = app.query_one("#prompt-host")
        band = app.query_one("#band")
        shell = app.query_one("#input-shell")
        # Ordered: question, then status, then the composer.
        assert host.region.y < shell.region.y
        assert band.region.y >= host.region.y + host.region.height


# --- the footer tells the truth about the keyboard the caret is on ----------


@pytest.mark.asyncio
async def test_escape_skips_the_question_from_the_composer() -> None:
    """The card's advertised exit has to work where the caret actually is.

    The caret lives in the COMPOSER while a question is up (answer keys are
    routed rather than focus being moved), so the card's own `escape` binding
    never sees the key — and `esc skip`, which the footer advertises in every
    state and which is the only stated way to leave, did nothing at all.
    Measured on three consecutive presses: question still up, tool still
    waiting (D11, design round 3).

    Whatever was already answered is kept, which is the rule the card's own
    Escape follows: a user who answered two of three questions has told the
    agent something.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(
            app.request_user_choice(
                [
                    _question("first", "First?", labels=("Alpha", "Beta"), descriptions=("", "")),
                    _question(
                        "second", "Second?", labels=("Gamma", "Delta"), descriptions=("", "")
                    ),
                ]
            )
        )
        for _ in range(14):
            await pilot.pause()

        # Answer the first question on the card, then move to the composer.
        await pilot.press("enter")
        await pilot.pause()
        await pilot.click(Editor)
        await _until(pilot, lambda: isinstance(app.screen.focused, Editor))
        assert isinstance(app.screen.focused, Editor)

        await pilot.press("escape")
        # The question is settled, the card is gone, and the answer survived.
        assert await asyncio.wait_for(asked, 2) == {"first": ["Alpha"]}
        await _until(pilot, lambda: not app.query(AskPickerScreen))
        assert not app.query(AskPickerScreen)


@pytest.mark.asyncio
async def test_the_footer_names_only_keys_that_work_where_the_caret_is() -> None:
    """A footer describing one keyboard while the caret sits on another lies.

    With focus in the composer the arrows, Enter and the printable keys are the
    composer's; only the routed ordinals and Escape reach the card. The footer
    said `↑↓ move · 1-9 jump · enter answer` regardless, and none of the first
    three did anything (D13, design round 3).
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(14):
            await pilot.pause()
        card = app._ask_screen
        assert card is not None

        # On the card, the full keymap is real and advertised.
        focused_footer = _painted_footer(app)
        assert "↑↓" in focused_footer and "enter" in focused_footer

        await pilot.click(Editor)
        # The footer repaints when focus lands in the composer; wait for that
        # repaint (the routed keys stand down) rather than betting on its wall
        # time. This is the state the assertions below read from the compositor.
        await _until(pilot, lambda: "↑↓" not in _painted_footer(app))
        # Read what was PAINTED, not what a fresh render would produce.
        #
        # This is the assertion the first version of this test got wrong.
        # `render_lines_for_test` re-derives the card's text on every call, so
        # it reports the intended footer whether or not anything repainted —
        # and nothing did, because `has_focus` is not a reactive. The model was
        # right, the screen still showed `↑↓ move · 1-9 jump · enter answer`,
        # and this test passed anyway (D13, design round 4). A footer claim is
        # a claim about pixels, so it has to be read from the compositor.
        composer_footer = _painted_footer(app)
        # The keys that no longer reach the card are no longer claimed...
        assert "↑↓" not in composer_footer, composer_footer
        assert "enter" not in composer_footer, composer_footer
        # ...and what IS claimed works: the ordinals and the exit.
        assert "answer" in composer_footer, composer_footer
        assert "esc" in composer_footer, composer_footer

        # The advertised range covers the OPTIONS and stops there: the
        # free-text row cannot be answered by a digit (it is answered by typing
        # into it, which needs the card to hold the caret), so naming it would
        # point at a dead end.
        assert "1-2" in composer_footer, composer_footer
        assert str(card.other_row + 1) not in composer_footer.split("answer")[0], composer_footer

        await pilot.press("1")
        assert await asyncio.wait_for(asked, 2) == {"stale": ["Drop them"]}


@pytest.mark.asyncio
async def test_a_held_key_never_answers_a_question_the_card_moved_on_from() -> None:
    """A key aimed at one question must not answer the next one.

    The `ask` picker walks several questions inside ONE widget, so guarding a
    parked keystroke by widget identity is not enough: after the card advances,
    the object is the same and the question is not. Measured before the fix,
    with a two-question ask — press `2` in the composer meaning "Canary" for
    question 1, then answer question 1 on the card inside the 180 ms hold
    window, and the parked key committed `DROP IT` on a question that had never
    been on screen (F4, agent review round 3).

    Reachable by one ordinary mouse click, since a single-select click both
    answers and advances.

    The property under test is the ``still_aimed_at`` guard in
    ``_commit_held_answer_key``, NOT the length of the hold window — so the
    window is stretched here and the commit is fired by hand at the point the
    test cares about. Racing the real 180 ms timer was the flake: under CPU
    contention the timer fired BEFORE ``action_accept()`` advanced the card,
    the key committed while still legitimately aimed at question 1, and the
    test failed reporting the very bug it exists to catch
    (``{'drop_table': ['Keep it'], 'rollout': ['Canary']}``) when nothing was
    wrong. Stretching the hold makes "the key is still parked when the card
    advances" a fact rather than a bet; calling the timer's own callback
    afterwards exercises the identical guard the real timer would reach.
    """
    from local_operator.tui.widgets.editor import Editor

    questions = [
        _question(
            "rollout", "Which rollout?", labels=("Blue-green", "Canary"), descriptions=("", "")
        ),
        _question(
            "drop_table", "Drop the table?", labels=("Keep it", "DROP IT"), descriptions=("", "")
        ),
    ]

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice(questions))
        for _ in range(14):
            await pilot.pause()
        card = app._ask_screen
        assert card is not None

        await pilot.click(Editor)
        # Wait for the click's focus/routing to settle, so the press below is
        # HELD by the composer rather than dropped, instead of betting a fixed
        # 0.1s that it has.
        for _ in range(100):
            if isinstance(app.screen.focused, Editor):
                break
            await pilot.pause()

        await pilot.press("2")  # aimed at question 1: "Canary"
        held = app._held_answer_key
        assert held is not None, "the key was not held"
        # Take the real timer out of the race entirely. Stopping it cannot mask
        # a regression: the commit path is invoked explicitly below, so the
        # guard still runs — it just runs at a moment the test controls.
        held.timer.stop()

        # The card advances to question 2 while the key is still parked.
        card.focus()
        for _ in range(100):
            if app._live_prompt() is card and card.question_index == 1:
                break
            if card.question_index == 0:
                card.action_accept()
            await pilot.pause()
        assert card.question_index == 1, "the card never advanced to question 2"
        assert app._held_answer_key is held, "the key was released before the advance"

        # Now fire exactly what the expired hold would have called.
        app._commit_held_answer_key()
        await pilot.pause()

        # The stale key answered nothing: question 2 is still being asked.
        assert not asked.done(), "a parked key answered a question it was not aimed at"
        assert app._held_answer_key is None

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_multi_select_advertises_no_key_it_cannot_be_answered_by() -> None:
    """A multi-select cannot be answered by one key, so none is offered.

    It is answered by ticking rows with Space and confirming with Enter, and
    both belong to the composer while the caret is there. Advertised as
    `1-2 answer`, pressing a digit only moved the cursor and left the question
    unanswered with `nothing ticked — space toggles` (D15b, design round 4).
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question(multi=True)]))
        for _ in range(16):
            await pilot.pause()

        await pilot.click(Editor)
        # Wait for the composer-mode footer to repaint (the gesture replaces the
        # ordinal range) rather than betting on its wall time.
        await _until(pilot, lambda: "answer here" in _painted_footer(app))
        footer = _painted_footer(app)
        # No ORDINAL range is claimed — a digit cannot answer a multi-select.
        # What is offered instead is the explicit gesture that reaches the card
        # (D18), which is a different promise: "answer here", not "1-2 answer".
        assert "1-" not in footer, footer
        assert "answer here" in footer, footer
        assert "esc" in footer, footer
        # ...and the digit that would have been claimed indeed answers nothing.
        await pilot.press("1")
        await pilot.pause()
        assert not asked.done(), "a digit answered a multi-select"

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_the_composer_footer_keeps_its_exit_on_a_narrow_card() -> None:
    """`esc skip` is the last thing the footer gives up, in BOTH focus states.

    The composer-mode branch returned its hints raw, so `_cut_row` ellipsised
    them and the exit was cut mid-word — `1 answer · esc sk…` at 22 columns.
    `skip` is the one word this row ranks as unsheddable, because a card with
    no stated way out is unusable (D3, and D16 in design round 4). The branch
    now runs through the same ladder, shedding the routed hint's word and then
    the hint itself.
    """
    from local_operator.tui.widgets.editor import Editor

    for width in (26, 22, 20, 18):
        app = _baseline_app()
        async with app.run_test(size=(width, 30)) as pilot:
            await pilot.pause()
            asked = asyncio.create_task(app.request_user_choice([_question()]))
            for _ in range(16):
                await pilot.pause()
            await pilot.click(Editor)
            # Wait for the composer-mode footer to repaint (its exit becomes the
            # last hint standing at these narrow widths) rather than sleeping.
            await _until(pilot, lambda: "esc skip" in _painted_footer(app))

            footer = _painted_footer(app)
            # The exit survives WHOLE — not truncated, not ellipsised.
            assert "esc skip" in footer, (width, footer)
            assert "…" not in footer, (width, footer)

            asked.cancel()
            try:
                await asked
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_the_footer_follows_the_draft_as_it_is_typed() -> None:
    """The footer has to repaint when the BUFFER changes, not just on focus.

    The routed keys stand down on a non-empty composer, so the footer's answer
    changes with every keystroke that opens or closes a draft — and nothing
    else repaints on a keystroke: `_repaint` fires on focus, resize, answer and
    advance, and typing is none of those. So the card went on advertising
    `1-2 answer` while `1` was being typed into the buffer (F7, agent review
    round 5). Same shape as D13 one axis over: the model right, the pixels
    stale.

    Asserted on PAINTED text, because that is the only reader that can see the
    difference — which is the lesson D13 taught this file.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(16):
            await pilot.pause()
        await pilot.click(Editor)
        await _until(pilot, lambda: "answer" in _painted_footer(app))
        assert "answer" in _painted_footer(app), "the routed keys were never offered"

        # Opening a draft withdraws the offer, on the frame the user sees.
        for character in "drop":
            await pilot.press(character)
        await _until(pilot, lambda: "answer" not in _painted_footer(app))
        withdrawn = _painted_footer(app)
        assert "answer" not in withdrawn, withdrawn
        assert "esc" in withdrawn, withdrawn
        # ...and the key it stopped advertising really is a text character now.
        await pilot.press("1")
        await _until(pilot, lambda: app.query_one(Editor).text == "drop1")
        assert not asked.done(), "a routed key answered while a draft was open"
        assert app.query_one(Editor).text == "drop1"

        # Clearing the draft brings the offer back, and it works.
        for _ in range(len("drop1")):
            await pilot.press("backspace")
        await _until(pilot, lambda: "answer" in _painted_footer(app))
        restored = _painted_footer(app)
        assert "answer" in restored, restored
        await pilot.press("1")
        assert await asyncio.wait_for(asked, 2) == {"stale": ["Drop them"]}


@pytest.mark.asyncio
async def test_a_multi_select_is_reachable_by_an_explicit_gesture() -> None:
    """A question the composer cannot answer needs a NAMED way to reach it.

    A multi-select is answered by Space and Enter, both of which the composer
    owns, so it is the one question the routed keys cannot reach. As a
    `ModalScreen` at the merge base it simply held the keyboard; anchored, it
    was answerable only by mouse (D17).

    Two attempts to infer the handover from the buffer both cost the user a
    message — keyed on empty, sending handed over (F9); keyed on
    deleted-to-empty, REWORDING did (D18). There is no better signal: "I have
    finished typing" is not distinguishable from "I am mid-edit" by looking at
    the text. So the gesture is explicit, advertised in the footer, and cannot
    arrive at a moment the user did not choose.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question(multi=True)]))
        for _ in range(16):
            await pilot.pause()
        await pilot.click(Editor)
        await _until(pilot, lambda: "answer here" in _painted_footer(app))

        # The footer NAMES it, or it is a key nobody can discover.
        assert "answer here" in _painted_footer(app), _painted_footer(app)
        # ...and it sheds its WORD before the exit, like every other hint on
        # this row. Asserted at full width above and at a narrow one below,
        # because the full-width assertion passes against any truncation.

        await pilot.press("tab")
        await _until(pilot, lambda: isinstance(app.screen.focused, AskPickerScreen))
        assert isinstance(app.screen.focused, AskPickerScreen)

        await pilot.press("space")
        await pilot.press("enter")
        assert await asyncio.wait_for(asked, 2) == {"stale": ["Drop them"]}


@pytest.mark.asyncio
async def test_the_gesture_sheds_its_word_before_the_exit() -> None:
    """`esc skip` outranks the Tab hint on a narrow card, as it does every hint.

    The composer-mode branch passed an EMPTY ladder whenever the Tab hint was
    showing, so both shed passes iterated over nothing and the row went to
    `_cut_row` raw — the exact case that call exists to prevent. At 18-26
    columns the multi-select painted `⇥ answer here · esc…`, and then
    `⇥ answer here…`: no way out stated at all, on the one surface where
    Escape is the only alternative to the handover (D19, design round 7).
    """
    from local_operator.tui.widgets.editor import Editor

    for width in (26, 22, 20, 18):
        app = _baseline_app()
        async with app.run_test(size=(width, 30)) as pilot:
            await pilot.pause()
            asked = asyncio.create_task(app.request_user_choice([_question(multi=True)]))
            for _ in range(16):
                await pilot.pause()
            await pilot.click(Editor)
            # Wait for the composer-mode footer to repaint with the gesture hint
            # rather than sleeping; the exit and the Tab key are what settle in.
            await _until(pilot, lambda: "esc skip" in _painted_footer(app))

            footer = _painted_footer(app)
            # The exit survives WHOLE, and the Tab key is still named.
            assert "esc skip" in footer, (width, footer)
            assert "\u21e5" in footer or "⇥" in footer, (width, footer)
            assert "…" not in footer, (width, footer)

            asked.cancel()
            try:
                await asked
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.asyncio
async def test_the_gesture_preserves_a_draft_and_leaves_routable_cards_alone() -> None:
    """Tab is offered only where it is needed, and never costs the draft.

    A single-select IS answerable from the composer, so pulling focus there
    would take the caret for nothing — and a user with a half-typed message is
    exactly who needs the gesture, so their text has to survive it.
    """
    from local_operator.tui.widgets.editor import Editor

    # A draft survives the handover.
    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for character in "hold that":
            await pilot.press("space" if character == " " else character)
        await _until(pilot, lambda: app.query_one(Editor).text == "hold that")
        asked = asyncio.create_task(app.request_user_choice([_question(multi=True)]))
        for _ in range(16):
            await pilot.pause()

        await pilot.press("tab")
        await _until(pilot, lambda: isinstance(app.screen.focused, AskPickerScreen))
        assert isinstance(app.screen.focused, AskPickerScreen)
        assert app.query_one(Editor).text == "hold that", "the gesture ate the draft"

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass

    # A routable card does not take the caret.
    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(16):
            await pilot.pause()
        await pilot.click(Editor)
        await _until(pilot, lambda: isinstance(app.screen.focused, Editor))
        await pilot.press("tab")
        # A routable card must NOT pull focus, so the caret stays in the editor;
        # pump the loop to let any (wrong) handover happen — it does not.
        await pilot.pause()
        assert isinstance(app.screen.focused, Editor)
        # ...because it is answerable from where the caret already is.
        await pilot.press("1")
        assert await asyncio.wait_for(asked, 2) == {"stale": ["Drop them"]}


@pytest.mark.asyncio
async def test_rewording_a_draft_never_moves_the_keyboard() -> None:
    """Clearing a line to retype it is mid-edit, not "I am done typing".

    Keyed on the user deleting to empty, the handover fired here: the next
    `space` ticked an option, the retyped message went nowhere, and Enter
    submitted `{'rollout': ['Backfill from the audit log']}` — an answer never
    chosen, with the message lost (D18, design round 6).
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for character in "hmm":
            await pilot.press(character)
        await _until(pilot, lambda: app.query_one(Editor).text == "hmm")
        asked = asyncio.create_task(app.request_user_choice([_question(multi=True)]))
        for _ in range(16):
            await pilot.pause()

        # Delete the line to reword it...
        for _ in range(len("hmm")):
            await pilot.press("backspace")
        # The keyboard must stay in the editor through the delete-to-empty (the
        # handover fires on that condition, wrongly, before the fix); pump the
        # loop to let any handover happen — it does not.
        await _until(pilot, lambda: app.query_one(Editor).text == "")
        assert isinstance(app.screen.focused, Editor), "rewording moved the keyboard"

        # ...and the retyped message lands in the composer, answering nothing.
        for character in "just checking":
            await pilot.press("space" if character == " " else character)
        await _until(pilot, lambda: app.query_one(Editor).text == "just checking")
        assert app.query_one(Editor).text == "just checking"
        assert not asked.done()
        card = app._ask_screen
        assert card is not None
        assert not card.state.checked, "a keystroke ticked an option"

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_the_card_repaints_itself_when_its_inputs_move_untriggered() -> None:
    """The backstop for "correct in the model, stale on screen".

    The footer is derived from state the card does not own — whether it holds
    focus, and whether the composer holds a draft — and neither emits anything
    the card hears. That produced the same defect three review rounds running
    on three different inputs (D13 focus, F7/D14 the buffer), each fixed by
    adding one more explicit trigger: a fix per input, with the next input left
    for a reviewer to find.

    So the card can be ASKED whether what it is showing is still what it would
    draw. What is pinned here is that mechanism working on its own, with the
    explicit triggers taken out of the picture: move an input, confirm the card
    now considers itself stale, tick, and confirm it agrees again — and that
    the repainted footer is the correct one for the new state.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(16):
            await pilot.pause()
        card = app._ask_screen
        assert card is not None
        await pilot.click(Editor)
        await _until(pilot, lambda: "answer" in _painted_footer(app))
        assert "answer" in _painted_footer(app)

        # Move the input WITHOUT any repaint: set the buffer directly on the
        # document, which posts nothing, and forget the recorded fingerprint so
        # the card is in the state a missed trigger leaves it in.
        card._painted_fingerprint = (True, True, (), 99)
        assert card.footer_fingerprint() != card._painted_fingerprint

        # One tick of the app's own poll and the card notices by itself.
        app._refresh_band()
        await _until(pilot, lambda: card.footer_fingerprint() == card._painted_fingerprint)
        assert (
            card.footer_fingerprint() == card._painted_fingerprint
        ), "the card did not repaint itself when its inputs had moved"
        # And what it repainted is right for the state it is actually in.
        assert "answer" in _painted_footer(app), _painted_footer(app)

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


def test_the_footer_fingerprint_covers_everything_the_footer_says() -> None:
    """A fingerprint that misses an input is the bug wearing the fix's clothes.

    `repaint_if_stale` is only as good as the fingerprint it compares: any
    input the footer reads but the fingerprint does not is a state where the
    card believes it is current while showing something else. The first
    version tracked focus, the draft, the drawn window and the question index —
    and missed the refused-Enter complaint and the answer state a multi-select's
    complaint is derived from, both of which REPLACE the key hints entirely.

    Exhaustive rather than sampled, over every combination of the state the
    footer reads: two equal fingerprints must produce the same footer. The
    incomplete version has 14 collisions against this; the current one has none.
    """
    from rich.style import Style

    for multi in (False, True):
        question = _question(multi=multi, labels=("A", "B", "C"), descriptions=("", "", ""))
        card = AskPickerScreen([question])
        seen: dict[tuple[object, ...], str] = {}
        for selected in range(card.row_count):
            for rejected in (False, True):
                for checked in ((), (0,), (0, 1)):
                    for typed in ("", "abc"):
                        card.state.selected = selected
                        card._rejected = rejected
                        card.state.checked = set(checked)
                        card.state.typed = typed
                        fingerprint = card.footer_fingerprint()
                        footer = card._footer_row(60, Style(), Style()).plain
                        if fingerprint in seen:
                            assert seen[fingerprint] == footer, (
                                multi,
                                fingerprint,
                                seen[fingerprint],
                                footer,
                            )
                        seen[fingerprint] = footer


@pytest.mark.asyncio
async def test_sending_a_message_does_not_hand_the_keyboard_to_the_question() -> None:
    """The user's next message must never become the answer to a question.

    The focus hand-back (D17) has to key on the user DELETING their way to
    empty, not on the buffer being empty: `on_text_area_changed` fires for any
    document change, and sending a message empties the buffer too. Keyed on
    emptiness, sending handed the caret to the card — and the user's next line
    was typed into a question they had stopped looking at, with the space
    ticking a row and Enter answering it.

    Measured end to end before the fix: `please check the schema` sent,
    `ok next` typed, Enter, and the ask resolved `{'s': ['next']}` from a line
    meant for the agent (F9, agent review round 6).
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.editor import Editor
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for character in "please check the schema":
            await pilot.press("space" if character == " " else character)
        await _until(pilot, lambda: app.query_one(Editor).text == "please check the schema")

        asked = asyncio.create_task(
            app.request_user_choice([_question(multi=True, labels=("Drop", "next"))])
        )
        for _ in range(16):
            await pilot.pause()
        assert isinstance(app.screen.focused, Editor)

        # Send it. The send empties the buffer, which must NOT be read as the
        # user finishing with the composer.
        await pilot.press("enter")
        await _until(pilot, lambda: session.prompts[-1:] == ["please check the schema"])
        assert session.prompts[-1:] == ["please check the schema"], session.prompts
        assert isinstance(app.screen.focused, Editor), "sending handed over the keyboard"

        # The next message is typed, not answered.
        for character in "ok next":
            await pilot.press("space" if character == " " else character)
        await _until(pilot, lambda: app.query_one(Editor).text == "ok next")
        assert app.query_one(Editor).text == "ok next"
        await pilot.press("enter")
        # Pump the loop so the send fully drains; the assertion is that this
        # chat message did NOT answer the agent's question.
        await _until(pilot, lambda: app.query_one(Editor).text == "")
        assert not asked.done(), "a chat message answered the agent's question"

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_collapsed_card_never_takes_the_keyboard() -> None:
    """A card drawing no options has nothing to do with the keyboard.

    `not answer_keys()` is true for two different reasons and only one wants
    focus: a multi-select, whose answers the composer cannot reach, and a
    COLLAPSED card, which is drawing nothing. Focus on the latter buys nothing
    — there is no cursor to move and the permissive keys are refused there
    anyway (D9) — so taking it is pure theft (F9, agent review round 6).
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 13)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        for character in "hmm":
            await pilot.press(character)
        await _until(pilot, lambda: app.query_one(Editor).text == "hmm")

        asked = asyncio.create_task(app.request_user_choice([_long_question()]))
        for _ in range(16):
            await pilot.pause()
        card = app._ask_screen
        assert card is not None
        assert not card.visible_rows, "this size is meant to draw no options"

        for _ in range(len("hmm")):
            await pilot.press("backspace")
        # A collapsed card must never take the keyboard, so focus stays in the
        # editor through the delete-to-empty; pump the loop to let any (wrong)
        # handover fire — it does not.
        await _until(pilot, lambda: app.query_one(Editor).text == "")
        assert isinstance(app.screen.focused, Editor)

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_a_live_question_does_not_break_slash_completion() -> None:
    """The pickers own their keys while they are open, question or no question.

    A live prompt swallows Tab so that a stray one cannot make the buffer
    non-empty and stand the routing down. Routed ahead of the composer's own
    pickers, that swallow silently broke slash completion for as long as any
    question was up: `/mod` then Tab left `/mod` instead of completing to
    `/model `. Found by driving the combination rather than reasoning about it.
    """
    from local_operator.tui.widgets.editor import Editor

    app = _baseline_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        asked = asyncio.create_task(app.request_user_choice([_question()]))
        for _ in range(16):
            await pilot.pause()

        editor = app.query_one(Editor)
        editor.focus()
        await _until(pilot, lambda: isinstance(app.screen.focused, Editor))
        for character in "/mod":
            await pilot.press(character)
        await _until(pilot, lambda: app.query_one(Editor).text == "/mod")

        await pilot.press("tab")
        await _until(pilot, lambda: app.query_one(Editor).text == "/model ")
        assert app.query_one(Editor).text == "/model ", app.query_one(Editor).text

        asked.cancel()
        try:
            await asked
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_the_card_reaches_the_composer_at_every_width() -> None:
    """Under the REAL stylesheet, the question's rows span the composer's column.

    The regression: the card kept a modal-era text budget (74 cells, less a
    margin held off the terminal edge) after it moved from a floating modal into
    the dock, where the panel is `width: 1fr`. Nothing re-derived the cap, so
    the wider the terminal the worse the disagreement — at 160 columns the ink
    stopped at 74 cells inside a 156-cell panel, and the title's rule, the
    selected row's tint and the footer all ended mid-slab while the composer
    directly beneath ran the full width.

    Asserted against `#input-shell` rather than against the terminal, because
    "extends with the composer" is the actual contract: the two are stacked
    surfaces of one dock, and the composer's own width already accounts for the
    screen inset and the shell's padding. A test written against the terminal
    width would have to restate that arithmetic and would then pass while the
    two surfaces disagreed.
    """
    for size in ((60, 24), (80, 30), (100, 30), (120, 30), (160, 40), (200, 50)):
        app, card = await _real_app_card(size, [_long_question()])
        async with app.run_test(size=size) as pilot:
            await _show(app, pilot, card)
            composer = app.screen.query_one("#input-shell")
            # The card's content box is the composer's: same column, same edges.
            assert card.size.width == composer.size.width, (size, card.size.width)
            # And the card SPENDS it. The selected row is drawn in its own
            # ground for the full width (`_fit_row`), so the widest painted row
            # is the card's real reach — the number the cap used to hold at 74.
            rows = [row for row in card.render_lines_for_test() if row.strip()]
            assert max(cell_len(row) for row in rows) == card.size.width, size
            # Reaching further must not have cost the screen its geometry.
            assert not app.screen.show_vertical_scrollbar, size
