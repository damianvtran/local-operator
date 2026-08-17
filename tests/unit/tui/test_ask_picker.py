"""The ``ask`` picker: real keys and a real click, and what each one answers.

Driven with ``pilot.press`` rather than by calling the actions, because what is
being pinned is the KEYMAP as much as the state machine: the digits, Space, and
``j``/``k`` all change meaning on the free-text row, and a card that "worked"
while its bindings were unreachable would still be an unanswerable question.

The card's own arithmetic (widths, page size) is asserted through
``render_lines_for_test``; the layout it produces under the real stylesheet is
checked against ``OperatorApp`` at the bottom of the file, because the lightweight
host below declares no ``CSS_PATH`` and therefore cannot show a clip or a
scrollbar at all.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult

from local_operator.harness.types import AskOption, AskQuestion
from local_operator.tui.widgets.ask_picker import (
    ASK_MAX_WIDTH,
    OTHER_LABEL,
    RECOMMENDED_TAG,
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
    """A host whose only job is to own the modal under test."""

    def __init__(self, questions: list[AskQuestion]) -> None:
        super().__init__()
        self._questions = questions
        self.answered: list[dict[str, list[str]] | None] = []

    def compose(self) -> ComposeResult:
        return iter(())

    async def open_picker(self) -> AskPickerScreen:
        screen = AskPickerScreen(self._questions)
        self.push_screen(screen, self.answered.append)
        return screen


# --- answering --------------------------------------------------------------


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
    cannot click is a half-built affordance."""
    app = _AskHost([_question()])
    async with app.run_test(size=(100, 30)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        body = screen.query_one("#ask-picker-body")
        region = body.region
        # The second option's row. Rows start after the header, the rule and the
        # wrapped question plus its blank spacer, and each row carries a
        # description line at this height.
        first_row_line = 2 + 1 + 1
        await pilot.click(offset=(region.x + 4, region.y + first_row_line + 2))
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
        assert "Question 1/2" in "\n".join(screen.render_lines_for_test())
        await pilot.press("enter")
        await pilot.pause()
        assert "Question 2/2" in "\n".join(screen.render_lines_for_test())
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
    for width in (24, 30, 40, 60, 80, 120, 190):
        app = _AskHost(
            [_question(labels=long_labels, descriptions=("a long consequence " * 6, ""))]
        )
        async with app.run_test(size=(width, 30)) as pilot:
            screen = await app.open_picker()
            await pilot.pause()
            budget = min(ASK_MAX_WIDTH, max(1, width - 4))
            for line in screen.render_lines_for_test():
                assert cell_len(line) <= budget, (width, line)


@pytest.mark.asyncio
async def test_a_short_terminal_drops_descriptions_before_it_drops_options() -> None:
    """A card that shed ROWS to keep prose would hide answers the user is being
    asked to choose between. Descriptions go first; the list goes last."""
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
    async with app.run_test(size=(100, 14)) as pilot:
        screen = await app.open_picker()
        await pilot.pause()
        text = "\n".join(screen.render_lines_for_test())
        assert len(screen.visible_rows) < screen.row_count
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
