"""Slash-command picker — trigger rules, keyboard, mouse, and rendering.

The keyboard and mouse paths run through a real ``App.run_test()`` pilot
against the shipped stylesheet (``local_operator.tcss``) and the real
``SLASH_COMMANDS`` registry, in a harness app that composes exactly what the
product ships — chevron, editor, picker, status band — so the layout claims
(under the editor, above the band, aligned to the text column) are exercised
by the same CSS the app runs. Pure layout questions that need no event loop
(one row per suggestion at widths 20/40/80/200, the collapse, wide-character
overflow) call the render API directly.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import SLASH_COMMANDS
from local_operator.tui.autocomplete import SlashCommand
from local_operator.tui.widgets.command_picker import (
    CommandPicker,
    command_suggestions,
    slash_context,
)
from local_operator.tui.widgets.editor import Editor, EditorSubmitted

from tests.unit.tui.conftest import TCSS_PATH

# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class PickerHarnessApp(App[None]):
    """The shipped input dock, nothing else.

    A bespoke app instead of ``OperatorApp`` so these tests cannot break when
    unrelated app responsibilities (session boot, status poll, splash) move —
    and so every assertion here answers exactly one question: does the picker
    behave as specified on the keys and the mouse?
    """

    CSS_PATH = TCSS_PATH

    def __init__(self) -> None:
        super().__init__()
        self.submissions: list[str] = []

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables

    def compose(self) -> ComposeResult:
        with Container(id="input-dock"):
            yield Static(id="status-band")
            self.editor = Editor(commands=SLASH_COMMANDS)
            with Horizontal(id="input-row"):
                yield Static("❯", id="prompt-chevron")
                yield self.editor
            yield self.editor.picker

    def on_editor_submitted(self, message: EditorSubmitted) -> None:
        self.submissions.append(message.text)


# ---------------------------------------------------------------------------
# trigger rules (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # Bare slash opens the full menu.
        ("/", (0, "")),
        # A word being typed.
        ("/mo", (0, "mo")),
        # Leading whitespace is fine: the first NON-BLANK line counts, and
        # the completion must not discard that whitespace (start=2).
        ("  /mo", (2, "mo")),
        ("\n\n  /x", (4, "x")),
    ],
)
def test_trigger_cases(text: str, expected: tuple[int, str]) -> None:
    context = slash_context(text)
    assert context is not None
    assert (context.start, context.query) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # nothing typed
        "hello",  # not a slash
        "hello /mo",  # text before the slash on the first non-blank line
        "/model ",  # whitespace terminates the command word
        "/model gpt",  # an argument means the command is already chosen
        "/mo\nrest",  # a newline terminates the word exactly like a space
        "/mo\n",
    ],
)
def test_non_trigger_cases(text: str) -> None:
    assert slash_context(text) is None


def test_bare_slash_suggests_the_full_registry_in_order() -> None:
    """``/`` scores nothing under match_commands (empty prefix = 0), so the
    picker supplies the registry itself — registration order is the ranking."""
    assert command_suggestions("", SLASH_COMMANDS) == [
        (command.name, command) for command in SLASH_COMMANDS
    ]


def test_typed_query_goes_through_match_commands_verbatim() -> None:
    from local_operator.tui.autocomplete import match_commands

    commands = [SlashCommand("help"), SlashCommand("history"), SlashCommand("exit")]
    assert command_suggestions("h", commands) == match_commands("/h", commands)


# ---------------------------------------------------------------------------
# render contract (pure)
# ---------------------------------------------------------------------------


def _picker(commands: list[SlashCommand]) -> CommandPicker:
    picker = CommandPicker(lambda _name: None)
    picker.set_commands(commands)
    return picker


@pytest.mark.parametrize("width", [20, 40, 80, 200])
def test_one_row_per_suggestion_at_width(width: int) -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    rows = picker.render_rows(width)
    # Eight visible out of fifteen: the budget caps the height.
    assert len(rows) == 8 == picker._row_budget()
    for row in rows:
        plain = row.plain
        # ONE line per suggestion: no newlines, and the row occupies EXACTLY
        # the render width — nothing to wrap, nothing to overdraw.
        assert "\n" not in plain
        assert cell_len(plain) == width


@pytest.mark.parametrize("width", [20, 40])
def test_collapses_to_name_only_under_41_cells(width: int) -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    text = "\n".join(row.plain for row in picker.render_rows(width))
    assert "/help" in text and "/exit" in text
    # Descriptions are dropped, not squeezed.
    assert "Show available commands" not in text


def test_descriptions_come_back_above_the_collapse_width() -> None:
    """Above the collapse the description column exists again — at the
    narrowest width that earns it, the name column may truncate it."""
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    text_41 = "\n".join(row.plain for row in picker.render_rows(41))
    assert "Show available comm" in text_41  # the column is back
    text_80 = "\n".join(row.plain for row in picker.render_rows(80))
    assert "Show available commands" in text_80  # room for the whole thing


def test_primary_column_aligns_descriptions() -> None:
    """Every description starts at the same cell, regardless of name length."""
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    starts = set()
    for row in picker.render_rows(80):
        for candidate in ("Show available commands", "Quit the app", "List MCP servers"):
            if candidate in row.plain:
                starts.add(row.plain.index(candidate))
    assert len(starts) == 1


@pytest.mark.parametrize("width", [20, 40, 80, 200])
def test_wide_character_descriptions_never_overflow(width: int) -> None:
    """CJK and emoji account for their cell cost, not their code-point count:
    a description of forty 2-cell characters still renders on exactly one row."""
    commands = [
        SlashCommand("cjk", "日本語テキスト" * 40),
        SlashCommand("emoji", "🎉" * 40),
        SlashCommand("mixed", "日本語🎉" * 20),
    ]
    picker = _picker(commands)
    picker.sync("/")
    rows = picker.render_rows(width)
    assert len(rows) == 3
    for row in rows:
        assert "\n" not in row.plain
        assert cell_len(row.plain) == width


def test_overflow_marker_shows_the_hidden_rows() -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    start, end, total = picker.visible_window()
    assert (start, end, total) == (0, 8, 15)
    marker = picker._overflow_row(80)
    assert marker is not None
    assert "… 7 more" in marker.plain
    assert cell_len(marker.plain) == 80

    # At the bottom of the list the window hides rows at BOTH ends.
    picker._selected = total - 1
    picker._scroll_to_selection()
    start, end, _ = picker.visible_window()
    assert (start, end) == (7, 15)

    # Once everything fits the marker disappears.
    picker.sync("/clear")
    assert picker._overflow_row(80) is None


def test_sync_preserves_highlight_for_unchanged_candidates() -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    picker.move(+3)
    assert picker.selected_index == 3
    picker.sync("/")  # same candidate set: no jump under the finger
    assert picker.selected_index == 3
    picker.sync("/m")  # a different set resets to the top
    assert picker.selected_index == 0


# ---------------------------------------------------------------------------
# keyboard (pilot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bare_slash_opens_the_picker() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        picker = app.editor.picker
        assert picker.is_open()
        assert [name for name, _ in picker.suggestions()] == [c.name for c in SLASH_COMMANDS]
        assert picker.highlighted_name() == "help"
        # Drawn: the pinned height matches the visible rows.
        assert picker.styles.height.value == 9  # 8 suggestions + the overflow count
        assert app.editor.has_focus  # opening the list never steals the caret


@pytest.mark.asyncio
async def test_typing_filters_to_the_matches() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o")
        await pilot.pause()
        picker = app.editor.picker
        assert picker.is_open()
        assert [name for name, _ in picker.suggestions()] == ["model"]


@pytest.mark.asyncio
async def test_completed_word_closes_the_picker() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o")
        assert app.editor.picker.is_open()
        await pilot.press("space")  # "/mo " terminates the word
        await pilot.pause()
        assert not app.editor.picker.is_open()
        assert app.editor.text == "/mo "


@pytest.mark.asyncio
async def test_text_before_the_slash_keeps_it_hidden() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("h", "i", "space", "slash", "m", "o")
        await pilot.pause()
        assert not app.editor.picker.is_open()


@pytest.mark.asyncio
async def test_escape_dismisses_without_touching_the_text() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o")
        assert app.editor.picker.is_open()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        assert app.editor.text == "/mo"  # the typed text survives


def test_esc_stays_dismissed_for_the_same_word_until_it_changes() -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/mo")
    assert picker.is_open()
    picker.dismiss()
    assert not picker.is_open()
    picker.sync("/mo")  # same word: still dismissed
    assert not picker.is_open()
    picker.sync("/mod")  # the word changed: the dismissal expired
    assert picker.is_open()
    picker.dismiss()
    picker.sync("hello")  # leaving slash context forgets the dismissal
    picker.sync("/mo")
    assert picker.is_open()


@pytest.mark.asyncio
async def test_up_down_move_and_wrap() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        picker = app.editor.picker
        total = len(picker.suggestions())
        assert total == len(SLASH_COMMANDS)
        for _ in range(total):
            await pilot.press("down")
        await pilot.pause()
        assert picker.selected_index == 0  # wrapped through the whole list
        assert picker.visible_window()[:2] == (0, 8)  # window rode back up
        await pilot.press("up")
        await pilot.pause()
        assert picker.selected_index == total - 1  # wraps the other way
        start, end, _ = picker.visible_window()
        assert end == total  # the window followed the selection down
        assert app.editor.has_focus  # navigation never steals the caret


@pytest.mark.asyncio
async def test_tab_completes_without_submitting() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "h", "e")
        await pilot.press("tab")
        await pilot.pause()
        # Trailing space: the argument slot is already open, and the closed
        # word is what hides the picker again.
        assert app.editor.text == "/help "
        assert not app.editor.picker.is_open()
        assert app.submissions == []
        assert app.editor.has_focus  # completion never moves focus


@pytest.mark.asyncio
async def test_tab_preserves_leading_whitespace() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("space", "space", "slash", "h", "e")
        await pilot.press("tab")
        await pilot.pause()
        assert app.editor.text == "  /help "


@pytest.mark.asyncio
async def test_enter_completes_then_submits() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o")
        assert app.editor.picker.highlighted_name() == "model"
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/model "]
        assert app.editor.text == ""
        assert not app.editor.picker.is_open()


@pytest.mark.asyncio
async def test_enter_with_the_picker_closed_submits_as_today() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("h", "i")
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["hi"]
        assert not app.editor.picker.is_open()


# ---------------------------------------------------------------------------
# mouse (pilot)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_selects_and_completes_without_submitting() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        picker = app.editor.picker
        # Row 2 is `/clear`. Offset x lands inside the name column.
        await pilot.click(CommandPicker, offset=(4, 2))
        await pilot.pause()
        assert app.editor.text == "/clear "
        assert app.submissions == []
        assert not picker.is_open()


@pytest.mark.asyncio
async def test_hover_highlights_the_row() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        picker = app.editor.picker
        assert picker.hovered_index is None
        await pilot.hover(CommandPicker, offset=(4, 1))
        await pilot.pause()
        assert picker.hovered_index == 1
        # The keyboard highlight and the hover are independent: hovering row
        # one did not move the selected row.
        assert picker.selected_index == 0


@pytest.mark.asyncio
async def test_click_on_the_overflow_row_does_nothing() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash")
        await pilot.pause()
        picker = app.editor.picker
        # Row 8 is the `… 7 more` marker: a count, not a command.
        await pilot.click(CommandPicker, offset=(4, 8))
        await pilot.pause()
        assert app.editor.text == "/"  # no completion: the marker is a count
        assert picker.is_open()
        assert picker.highlighted_name() == "help"
