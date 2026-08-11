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
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.app import SLASH_COMMANDS
from local_operator.tui.autocomplete import ArgumentChoice, SlashCommand
from local_operator.tui.widgets.command_picker import (
    MAX_VISIBLE_ROWS,
    CommandPicker,
    PickerMode,
    argument_suggestions,
    command_suggestions,
    slash_context,
)
from local_operator.tui.widgets.editor import (
    ArgumentQueryOpened,
    Editor,
    EditorSubmitted,
)
from tests.unit.tui.conftest import TCSS_PATH

# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

#: Stand-in provider catalogue. Shaped like the real one where it matters: ids
#: the user types, display names, model-family aliases that must FIND a row
#: without ever being completed into the buffer, and the three credential
#: states the detail column reports.
PROVIDER_CHOICES = [
    ArgumentChoice(
        "anthropic",
        "Anthropic",
        aliases=("claude", "sonnet", "opus"),
        detail="logged in",
    ),
    ArgumentChoice("alibaba", "Alibaba", aliases=("qwen", "dashscope"), detail="logged in"),
    ArgumentChoice("openai", "OpenAI", aliases=("gpt", "chatgpt"), detail="env key"),
    ArgumentChoice("deepseek", "DeepSeek", aliases=("ds",), detail="needs login"),
]


def _logout_choices() -> list[ArgumentChoice]:
    """What `/logout` offers, shaped as the app builds it.

    Only providers holding a stored credential, and each row states the KIND it
    will remove rather than the "logged in" that is true of every row on this
    list by construction — in the danger tint, because every row here destroys
    something.
    """
    return [
        ArgumentChoice(choice.name, choice.description, choice.aliases, "remove api key", True)
        for choice in PROVIDER_CHOICES
        if choice.detail == "logged in"
    ]


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
        self.provider_queries: list[str] = []

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

    def on_argument_query_opened(self, message: ArgumentQueryOpened) -> None:
        # Stands in for the app's controller-backed answer. The harness keeps the
        # two sets distinct so `/logout` can be shown to offer strictly less.
        message.stop()
        self.provider_queries.append(message.command)
        choices = _logout_choices() if message.command == "logout" else PROVIDER_CHOICES
        self.editor.picker.set_choices(list(choices))


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
    assert "List all comm" in text_41  # the column is back
    text_80 = "\n".join(row.plain for row in picker.render_rows(80))
    assert "List all commands" in text_80  # room for the whole thing


def test_primary_column_aligns_descriptions() -> None:
    """Every description starts at the same cell, regardless of name length.

    The probes are asserted PRESENT before their columns are compared. Collecting
    positions only for probes that happen to appear lets a regression that
    truncates or drops two of the three pass on the survivor — `len(starts) == 1`
    is trivially true for a single element, and for an empty set the loop never
    runs at all. Two tests of exactly that shape shipped in this same commit.
    """
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    # Probes MUST be inside the visible window and short enough to survive at
    # this width. "List MCP servers" was neither — `mcp` is the 13th of 15
    # commands and never rendered, which the old subset-tolerant assertion
    # silently accepted.
    #
    # Taken from the command table rather than written out: descriptions get
    # reworded whenever a command gains a capability (``/model`` and
    # ``/resume`` both changed in one afternoon), and an alignment test that
    # is not about wording must not fail for it. The first three commands are
    # always inside the window.
    probes = tuple(command.description for command in SLASH_COMMANDS[:3])
    rows = [row.plain for row in picker.render_rows(200)]
    text = "\n".join(rows)
    missing = [probe for probe in probes if probe not in text]
    assert missing == [], f"probe strings absent, so the comparison below is vacuous: {missing}"

    starts = {row.index(probe) for row in rows for probe in probes if probe in row}
    assert len(starts) == 1, f"descriptions start at different columns: {sorted(starts)}"


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
    # Derived from the registry, not transcribed from it: this test is about the
    # WINDOWING maths, and a hardcoded census broke it every time a command was
    # added — which is not a windowing regression.
    total_commands = len(SLASH_COMMANDS)
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    start, end, total = picker.visible_window()
    visible = end - start
    assert (start, total) == (0, total_commands)
    assert visible < total_commands, "the fixture must overflow for this to mean anything"
    marker = picker._overflow_row(80)
    assert marker is not None
    assert f"… {total_commands - visible} more" in marker.plain
    assert cell_len(marker.plain) == 80

    # At the bottom of the list the window hides rows at BOTH ends.
    picker._selected = total - 1
    picker._scroll_to_selection()
    start, end, _ = picker.visible_window()
    assert (start, end) == (total_commands - visible, total_commands)

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
        assert picker.styles.height is not None
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
    """An unambiguous Enter completes the word AND runs it.

    `/us` is used rather than `/mo`: `model` is a command whose ARGUMENT drives its
    own list, so completing it deliberately stops there — the trailing space opens
    the model picker, and submitting as well would run a command whose whole
    outcome the keystroke had already produced.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "u", "s")
        assert app.editor.picker.highlighted_name() == "usage"
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/usage "]
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


# -- short-query noise and Enter safety --------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("u", ["usage"]),
        ("g", ["goal"]),
        ("s", ["skills"]),
        ("c", ["clear", "compact"]),
        ("lo", ["loop", "login", "logout"]),
        ("mo", ["model"]),
    ],
)
def test_short_queries_keep_only_prefix_matches(query: str, expected: list[str]) -> None:
    """One and two letters matched an arbitrary-looking set by subsequence:
    ``/u`` offered usage, quit, accounts, logout and ``/g`` offered goal, usage,
    login, logout. The right command ranked first every time, but rows 2+ taught
    the user the list is unreliable — the fastest way to make them stop reading
    it. Before the picker existed this tail was never rendered.
    """
    names = [name for name, _ in command_suggestions(query, SLASH_COMMANDS)]
    assert names == expected


@pytest.mark.parametrize(
    "query,expected",
    [("cmpct", "compact"), ("lgout", "logout"), ("qit", "quit"), ("skl", "skills")],
)
def test_typo_tolerance_still_works_from_three_characters(query: str, expected: str) -> None:
    """The gate must not cost the feature its reason to exist: real typos are
    three characters or more, which is where the fuzzy band still applies."""
    names = [name for name, _ in command_suggestions(query, SLASH_COMMANDS)]
    assert names[0] == expected


def test_a_bare_slash_still_lists_everything() -> None:
    """The gate keys off a NON-EMPTY query; `/` is the "show me the commands"
    keystroke and must stay exhaustive."""
    assert len(command_suggestions("", SLASH_COMMANDS)) == len(SLASH_COMMANDS)


@pytest.mark.asyncio
async def test_an_ambiguous_enter_grows_the_common_prefix_and_never_a_command() -> None:
    """Enter on an ambiguous query extends to the matches' longest COMMON prefix
    and leaves the list open. It never sends, and it never inserts a command name.

    `/lo` highlights `loop` while `login` and `logout` also match. An earlier
    design completed to the highlighted row, which put the most destructive
    candidate in the buffer ready to run — so a reflex double-Enter started
    autonomous work for a user reaching for `/login`. The common prefix cannot be
    the wrong command by construction: it is the part every candidate agrees on.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o")
        await pilot.pause()
        assert len(app.editor.picker.suggestions()) > 1, "premise: /lo is ambiguous"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == [], "an ambiguous pick must not be sent"
        assert app.editor.text == "/lo", "the common prefix of loop/login/logout is 'lo'"
        assert app.editor.picker.is_open(), "the list must stay up to keep narrowing"

        # Narrowing by hand still resolves it: `g` leaves login and logout, whose
        # common prefix is `log`, and one more character settles it outright.
        await pilot.press("g", "i")
        await pilot.pause()
        assert app.editor.picker.suggestions()[0][0] == "login"
        await pilot.press("enter")
        await pilot.pause()
        # `login` is a list-opening command like `model`: the resolved Enter
        # completes the word and the trailing space opens the provider list, so
        # the bare command is never run.
        assert app.submissions == []
        assert app.editor.text == "/login "
        assert app.editor.picker.mode is PickerMode.ARGUMENT


@pytest.mark.asyncio
async def test_enter_sends_immediately_when_there_is_only_one_match() -> None:
    """The common case must not cost a second keystroke.

    `/us` rather than `/mo`, because `model` completes without running — its
    argument drives its own list, and the completion is what opens it.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "u", "s")
        await pilot.pause()
        assert len(app.editor.picker.suggestions()) == 1, "premise: /us is unambiguous"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/usage "]


@pytest.mark.asyncio
async def test_a_fully_typed_list_command_opens_its_list_instead_of_running() -> None:
    """A user who typed `logout` in full named it — the ambiguity gate is
    satisfied — and what that resolves to is the PROVIDER list, not a bare
    `/logout` echoed into the transcript with no provider to remove."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "l", "o", "g", "o", "u", "t")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == []
        assert app.editor.text == "/logout "
        assert app.provider_queries == ["logout"]
        assert [name for name, _ in app.editor.picker.suggestions()] == ["anthropic", "alibaba"]


@pytest.mark.asyncio
async def test_tab_never_sends_however_unambiguous() -> None:
    """Tab's contract is unchanged by the Enter rule."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o")
        await pilot.press("tab")
        await pilot.pause()
        assert app.submissions == []
        assert app.editor.text == "/model "


def test_a_short_query_with_no_prefix_match_still_offers_the_fuzzy_tail() -> None:
    """The short-query gate is a PREFERENCE, not a filter.

    An empty suggestion list closes the picker, and a closed picker takes the
    editor's Tab and Enter guards down with it — Tab then indents the user's
    message and Enter submits the raw text. So a two-letter query that no command
    starts with must still answer with its fuzzy matches, which is exactly the
    set of natural abbreviations the matcher exists for.
    """
    for query, expected in (("lg", "login"), ("qt", "quit"), ("md", "model")):
        names = [name for name, _ in command_suggestions(query, SLASH_COMMANDS)]
        assert names, f"/{query} left the picker with nothing to show"
        assert expected in names, f"/{query} should still reach {expected}: {names}"


def test_a_short_query_still_prefers_its_prefix_matches() -> None:
    """When prefix matches DO exist the gate holds: the arbitrary fuzzy tail is
    what it was added to suppress."""
    names = [name for name, _ in command_suggestions("mo", SLASH_COMMANDS)]
    assert names == [name for name in names if name.startswith("mo")], names


@pytest.mark.asyncio
async def test_tab_still_completes_a_short_fuzzy_query() -> None:
    """The gate must not cost Tab its completion. Tab never sends, so it carries
    none of the blast-radius risk the ambiguity rules manage."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.press("slash", "l", "g")
        assert app.editor._picker.is_open(), "the picker closed on a fuzzy-only query"
        await pilot.press("tab")
        assert app.editor.text in ("/login ", "/logout "), app.editor.text
        assert app.submissions == [], "Tab must never send"


@pytest.mark.asyncio
async def test_arrowing_onto_a_row_lets_enter_send_it() -> None:
    """An explicit move answers the very ambiguity the two-Enter rule guards.

    The rule exists because the MATCHER may have chosen the row; a user who
    arrowed onto it chose it themselves, which is also the muscle memory every
    comparable picker has taught (move, Enter, done).

    `/c` (clear, compact) rather than `/lo`: every candidate under `/lo` opens a
    list instead of running, so the send this test is about would never happen
    there for a reason that has nothing to do with the arrow key.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.press("slash", "c")
        assert len(app.editor._picker.suggestions()) > 1
        await pilot.press("down")
        chosen = app.editor._picker.highlighted_name()
        await pilot.press("enter")
        assert app.submissions == [f"/{chosen} "], app.submissions


@pytest.mark.asyncio
async def test_a_matcher_chosen_row_still_needs_the_second_enter() -> None:
    """The protection stays where it was argued for: no explicit move, no send."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.press("slash", "l", "o")
        assert len(app.editor._picker.suggestions()) > 1
        await pilot.press("enter")
        assert app.submissions == [], "a fuzzy pick must not run on one keystroke"


@pytest.mark.asyncio
async def test_completing_a_list_opening_command_does_not_run_it() -> None:
    """`/model` is completed, not submitted.

    Its trailing space is exactly what opens the model picker, so submitting as
    well produced a round trip with three visible costs: the transcript echoed a
    `/model` that did nothing, the buffer was cleared, and the app had to put the
    query BACK to reopen a list the same keystroke had already opened.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "m", "o", "d", "e", "l")
        assert app.editor.picker.highlighted_name() == "model"
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == [], "completing a list-opening command must not run it"
        assert app.editor.text == "/model ", app.editor.text
        assert not app.editor.picker.is_open(), "the command list gives way"


# ---------------------------------------------------------------------------
# argument mode — the SAME picker over a command's provider argument
# ---------------------------------------------------------------------------


def _argument_picker(choices: list[ArgumentChoice], query: str = "") -> CommandPicker:
    picker = CommandPicker(lambda _name: None)
    picker.set_choices(choices)
    picker.sync_argument(query)
    return picker


def test_an_argument_row_shows_the_bare_id_not_a_slash_command() -> None:
    """`/login /anthropic` is not typeable, so the row must not offer it.

    The slash is command vocabulary. Printing it on an argument row would teach a
    keystroke that the parser rejects.
    """
    picker = _argument_picker(list(PROVIDER_CHOICES))
    assert picker.mode is PickerMode.ARGUMENT
    text = "\n".join(row.plain for row in picker.render_rows(80))
    assert "anthropic" in text
    assert "/anthropic" not in text
    assert "/" not in text


@pytest.mark.parametrize("width", [16, 20, 40, 80, 200])
def test_an_argument_row_fills_exactly_the_render_width(width: int) -> None:
    """The one-row rule is structural for argument rows too: exact width means
    Textual has nothing to wrap, and the detail column cannot push past the edge
    however long the state string is."""
    picker = _argument_picker(list(PROVIDER_CHOICES))
    rows = picker.render_rows(width)
    assert len(rows) == len(PROVIDER_CHOICES)
    for row in rows:
        assert "\n" not in row.plain
        assert cell_len(row.plain) == width


def test_detail_is_right_aligned_and_survives_the_description_collapse() -> None:
    """`detail` is what `/logout` is chosen BY, so it outranks the description.

    At 41 cells both columns fit; at 40 the description collapses and the state
    is what stays, because a row that says only "anthropic" cannot answer the
    question the user opened the list to ask.
    """
    picker = _argument_picker(list(PROVIDER_CHOICES))
    wide = picker.render_rows(41)[0].plain
    assert "Anthropic" in wide
    assert wide.rstrip().endswith("logged in")

    narrow = picker.render_rows(40)[0].plain
    assert "Anthropic" not in narrow
    assert "anthropic" in narrow
    assert narrow.rstrip().endswith("logged in")


def test_a_row_too_narrow_for_both_keeps_the_name() -> None:
    """Below the point where the id itself would be squeezed, the detail goes.

    The name is the text Tab types into the buffer; a truncated one is not a
    choice the user can act on, whatever it says about their credentials.
    """
    row = _argument_picker(list(PROVIDER_CHOICES)).render_rows(20)[0].plain
    assert "anthropic" in row
    assert "logged in" not in row


def test_an_alias_finds_the_row_but_never_becomes_the_completion() -> None:
    """`claude` reaches anthropic; the row still says `anthropic`, because that
    is the only text `/login` accepts."""
    picker = _argument_picker(list(PROVIDER_CHOICES), "clau")
    assert [name for name, _ in picker.suggestions()] == ["anthropic"]
    text = "\n".join(row.plain for row in picker.render_rows(80))
    assert "claude" not in text


def test_choices_arriving_after_the_open_still_fill_the_list() -> None:
    """The app answers the opening message a tick later, so `set_choices` has to
    re-derive the rows — otherwise the list sits empty until the next keystroke."""
    picker = _argument_picker([], "")
    assert not picker.is_open()
    picker.set_choices(list(PROVIDER_CHOICES))
    assert picker.is_open()
    assert [name for name, _ in picker.suggestions()] == [c.name for c in PROVIDER_CHOICES]


def test_an_empty_argument_query_offers_everything() -> None:
    """`/login ` with nothing typed is the "show me the providers" keystroke."""
    assert argument_suggestions("", list(PROVIDER_CHOICES)) == [
        (choice.name, choice) for choice in PROVIDER_CHOICES
    ]


@pytest.mark.asyncio
async def test_typing_past_the_command_opens_the_provider_list() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for key in ("slash", "l", "o", "g", "i", "n", "space"):
            await pilot.press(key)
        await pilot.pause()
        picker = app.editor.picker
        assert app.provider_queries == ["login"]
        assert picker.mode is PickerMode.ARGUMENT
        assert [name for name, _ in picker.suggestions()] == [c.name for c in PROVIDER_CHOICES]
        assert app.editor.has_focus

        # And it narrows by model family, not just by id.
        for key in ("c", "l", "a", "u"):
            await pilot.press(key)
        await pilot.pause()
        assert [name for name, _ in picker.suggestions()] == ["anthropic"]
        # Still ONE query: the message rides the transition, not the keystroke.
        assert app.provider_queries == ["login"]


@pytest.mark.asyncio
async def test_logout_offers_only_what_can_be_removed() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for key in ("slash", "l", "o", "g", "o", "u", "t", "space"):
            await pilot.press(key)
        await pilot.pause()
        assert app.provider_queries == ["logout"]
        names = [name for name, _ in app.editor.picker.suggestions()]
        assert names == ["anthropic", "alibaba"], "only stored credentials"


@pytest.mark.asyncio
async def test_switching_between_login_and_logout_reasks_for_the_rows() -> None:
    """The two commands offer different sets, so the rows cannot be carried over —
    `/logout` must never inherit a provider the user was never logged into."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login "
        await pilot.pause()
        assert [name for name, _ in app.editor.picker.suggestions()] == [
            c.name for c in PROVIDER_CHOICES
        ]
        app.editor.text = "/logout "
        await pilot.pause()
        assert app.provider_queries == ["login", "logout"]
        assert [name for name, _ in app.editor.picker.suggestions()] == ["anthropic", "alibaba"]


@pytest.mark.asyncio
async def test_tab_completes_the_provider_id_without_running_it() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login clau"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        # The ID, not the alias that found it — and no trailing space, which
        # would terminate the argument and close the list Tab just used.
        assert app.editor.text == "/login anthropic"
        assert app.submissions == []
        assert app.editor.has_focus


@pytest.mark.asyncio
async def test_enter_runs_an_unambiguous_provider() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/logout anthropic"
        await pilot.pause()
        assert len(app.editor.picker.suggestions()) == 1, "premise: one match"
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/logout anthropic"]
        assert app.editor.text == ""


@pytest.mark.asyncio
async def test_an_ambiguous_enter_never_logs_anyone_out() -> None:
    """`/logout` DELETES a credential, so a row the matcher guessed at must not
    run on one keystroke. The first Enter completes; the buffer then names one
    provider exactly, and the second Enter acts."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/logout a"
        await pilot.pause()
        assert len(app.editor.picker.suggestions()) > 1, "premise: /logout a is ambiguous"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == [], "a fuzzy pick must not remove a credential"
        assert app.editor.text == "/logout anthropic"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/logout anthropic"]


@pytest.mark.asyncio
async def test_arrowing_onto_a_provider_lets_enter_run_it() -> None:
    """An explicit move is the answer to "did the matcher choose this row?" — the
    same rule the command list applies, so the two lists cannot drift."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/logout a"
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        assert app.editor.picker.highlighted_name() == "alibaba"
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/logout alibaba"]


@pytest.mark.asyncio
async def test_arrows_move_the_provider_highlight() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login "
        await pilot.pause()
        picker = app.editor.picker
        assert picker.selected_index == 0
        await pilot.press("down", "down")
        await pilot.pause()
        assert picker.selected_index == 2
        await pilot.press("up")
        await pilot.pause()
        assert picker.selected_index == 1
        assert app.editor.has_focus, "navigation never steals the caret"


@pytest.mark.asyncio
async def test_escape_closes_the_provider_list_and_leaves_the_text() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login clau"
        await pilot.pause()
        assert app.editor.picker.is_open()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        assert app.editor.text == "/login clau", "the typed text survives"


@pytest.mark.asyncio
async def test_clicking_a_login_row_runs_it() -> None:
    """A click names one exact row with a pointer — that is not the matcher's
    guess the keyboard gate protects against, and a click that only filled the
    field would leave the user finishing a choice they already made."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login "
        await pilot.pause()
        await pilot.click(CommandPicker, offset=(4, 1))
        await pilot.pause()
        assert app.submissions == ["/login alibaba"]


@pytest.mark.asyncio
async def test_clicking_a_logout_row_fills_the_field_and_waits_for_enter() -> None:
    """The one list where a click must not act on its own.

    The picker sits directly above the input row — the row a user clicks to put
    the caret in the field — so a click that ran the row put "one misclick, one
    credential gone" a single pixel away, with no gate at all: the keyboard path
    asks `/logout` for the id in full, and the mouse path went straight to the
    handler. An OAuth credential costs another browser login to get back.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/logout "
        await pilot.pause()
        await pilot.click(CommandPicker, offset=(4, 1))
        await pilot.pause()
        assert app.submissions == [], "a click must not remove a credential"
        assert app.editor.text == "/logout alibaba", "the row is filled in, not run"

        # And the confirmation is one keystroke: the buffer now names the id in
        # full, which is exactly what the destructive gate asks for.
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/logout alibaba"]


# ---------------------------------------------------------------------------
# the state COLUMN — the thing the list is scanned by
# ---------------------------------------------------------------------------

#: What `/logout` offers: the credential KIND being removed, in the danger tint.
#: Two widths and a row with no description, which is the shape the real
#: catalogue produces (`openrouter`'s name says nothing its id does not).
LOGOUT_CHOICES = [
    ArgumentChoice("anthropic", "Claude Pro/Max", ("claude",), "remove oauth", True),
    ArgumentChoice("xai", "Grok API key", ("grok",), "remove api key", True),
    ArgumentChoice("openrouter", "", ("router",), "remove api key", True),
]


def _column_starts(rows: list[str], probes: tuple[str, ...]) -> set[int]:
    """Where each probe begins, asserted PRESENT first so the set is not vacuous.

    A comparison over probes that never rendered is trivially satisfied — one
    surviving string makes ``len(starts) == 1`` true and an empty set makes the
    loop never run at all.
    """
    text = "\n".join(rows)
    missing = [probe for probe in probes if probe not in text]
    assert missing == [], f"probe strings absent, so the comparison is vacuous: {missing}"
    return {row.rindex(probe) for row in rows for probe in probes if probe in row}


@pytest.mark.parametrize("width", [41, 60, 80, 200])
def test_the_state_column_starts_at_one_x_for_every_login_row(width: int) -> None:
    """Three states of three different lengths, one left edge to scan.

    Right-aligning each state to its OWN row's trailing edge started them at
    three different columns, so answering "which of these am I logged into"
    meant reading every string instead of running an eye down one edge. The
    column is right-aligned; the strings inside it are not.
    """
    rows = [row.plain for row in _argument_picker(list(PROVIDER_CHOICES)).render_rows(width)]
    starts = _column_starts(rows, ("logged in", "env key", "needs login"))
    assert len(starts) == 1, f"states start at different columns: {sorted(starts)}"
    # And the COLUMN is still pinned to the trailing edge: the widest state ends
    # exactly one edge margin short of the row's last cell.
    assert starts.pop() + cell_len("needs login") == width - 2


@pytest.mark.parametrize("width", [41, 60, 80, 200])
def test_the_state_column_starts_at_one_x_for_every_logout_row(width: int) -> None:
    """Same for `/logout`, whose two kinds differ in length by two cells."""
    rows = [row.plain for row in _argument_picker(list(LOGOUT_CHOICES)).render_rows(width)]
    starts = _column_starts(rows, ("remove oauth", "remove api key"))
    assert len(starts) == 1, f"kinds start at different columns: {sorted(starts)}"
    assert starts.pop() + cell_len("remove api key") == width - 2


def test_the_column_survives_the_description_collapse_and_stays_aligned() -> None:
    """Below 41 cells the description goes and the state stays — still a column.

    This is the width where the rag was worst: with no description left, the
    states were the only thing on the row and they still did not line up.
    """
    rows = [row.plain for row in _argument_picker(list(PROVIDER_CHOICES)).render_rows(40)]
    assert "Anthropic" not in "\n".join(rows), "premise: the description has collapsed"
    starts = _column_starts(rows, ("logged in", "env key", "needs login"))
    assert len(starts) == 1, f"states start at different columns: {sorted(starts)}"


@pytest.mark.parametrize("width", [16, 20, 40, 80, 200])
def test_a_logout_row_fills_exactly_the_render_width(width: int) -> None:
    """The exact-width rule holds for the reserved column and the alert tint too:
    a column sized from the WIDEST detail must not push the longest row past the
    edge, and the danger style must not change the cell count."""
    rows = _argument_picker(list(LOGOUT_CHOICES)).render_rows(width)
    assert len(rows) == len(LOGOUT_CHOICES)
    for row in rows:
        assert "\n" not in row.plain
        assert cell_len(row.plain) == width


def test_the_alert_tint_paints_the_detail_in_danger() -> None:
    """`/logout`'s rows are destructive by construction, so the state column is
    the tool card's outcome red — not a second accent, and never on `/login`."""
    danger = theme_mod.semantic_color("danger").lower()
    muted = theme_mod.semantic_color("muted").lower()

    def detail_colour(row, detail: str) -> str:
        spans = [s for s in row.spans if row.plain[s.start : s.end] == detail]
        assert spans, f"no span covers {detail!r} exactly"
        return spans[-1].style.color.triplet.hex.lower()

    alerted = _argument_picker(list(LOGOUT_CHOICES)).render_rows(80)
    assert detail_colour(alerted[1], "remove api key") == danger

    ordinary = _argument_picker(list(PROVIDER_CHOICES)).render_rows(80)
    assert detail_colour(ordinary[3], "needs login") == muted


@pytest.mark.parametrize("width", range(20, 121))
def test_no_render_width_ever_clips_a_provider_id(width: int) -> None:
    """The detail yields before the NAME does, at every width.

    A fixed twelve-cell floor answered "would a twelve-cell name fit" for a list
    whose longest id is thirteen, so exactly one width rendered
    `openai-devi…` while keeping an intact `needs login` beside it — the state
    column taking cells from the only text on the row the user can type.
    """
    choices = [
        ArgumentChoice("openai", "ChatGPT Plus/Pro", (), "needs login"),
        ArgumentChoice("openai-device", "ChatGPT device code", (), "needs login"),
        ArgumentChoice("xai-oauth", "Grok OAuth", (), "logged in"),
    ]
    rows = [row.plain for row in _argument_picker(choices).render_rows(width)]
    for choice, row in zip(choices, rows):
        assert choice.name in row, f"id clipped at width {width}: {row!r}"


# ---------------------------------------------------------------------------
# the destructive gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("query,row", [("aib", "alibaba"), ("anthrpic", "anthropic")])
async def test_a_single_fuzzy_survivor_never_removes_a_credential(query: str, row: str) -> None:
    """One match is not evidence on a list that DELETES.

    The matcher is a subsequence matcher, so a query that spells nothing can
    still leave exactly one row standing — and "there is only one match" then
    read as "the user must have meant it". A typo one letter off a real id lands
    in this shape, and the outcome was an unrecoverable OAuth credential gone on
    a single Enter.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = f"/logout {query}"
        await pilot.pause()
        assert [name for name, _ in app.editor.picker.suggestions()] == [row], "premise: one match"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == [], "a query the user never spelled must not act"
        assert app.editor.text == f"/logout {row}", "it completes instead"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == [f"/logout {row}"], "and the named id runs"


@pytest.mark.asyncio
async def test_a_single_fuzzy_survivor_still_runs_on_a_login_list() -> None:
    """The harder rule is `/logout`'s alone: `/login` puts nothing at risk, and
    making every list ask twice would spend a keystroke to protect nothing."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login aib"
        await pilot.pause()
        assert [name for name, _ in app.editor.picker.suggestions()] == ["alibaba"]
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/login alibaba"]


@pytest.mark.asyncio
async def test_arrowing_onto_a_logout_row_still_runs_it_on_one_enter() -> None:
    """Deliberately kept, not overlooked.

    Arrowing IS the user reading the list and choosing, and requiring a second
    Enter after an explicit move would break the symmetry the shared picker
    leans on — the same keys mean the same thing on both lists. What makes it
    safe is that the arrowed row states its own consequence: the detail column
    names the credential kind about to be removed, in the danger tint, so the
    row the highlight lands on says what Enter will do.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/logout "
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        picker = app.editor.picker
        assert picker.highlighted_name() == "alibaba"

        _name, choice = picker.suggestions()[picker.selected_index]
        assert isinstance(choice, ArgumentChoice)
        assert choice.detail, "the arrowed row must state what Enter destroys"
        assert choice.alert, "and state it in the destructive tint"

        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/logout alibaba"]


@pytest.mark.asyncio
async def test_escape_lands_in_the_tick_before_the_rows_arrive() -> None:
    """The rows are one message-loop tick behind the keystroke that opens the
    list, and Esc inside that window was dropped: the user dismissed a list and
    then watched it appear anyway."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/logout "
        picker = app.editor.picker
        assert not picker.is_open(), "premise: the rows have not arrived yet"
        assert picker.is_pending()

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not picker.is_open(), "the dismissal survives the rows landing"
        assert app.editor.text == "/logout ", "and the typed text is untouched"


# ---------------------------------------------------------------------------
# how many rows an argument list is worth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "height,visible",
    [(28, 10), (20, 6), (12, 3)],
)
async def test_an_argument_list_is_budgeted_from_the_rows_available(
    height: int, visible: int
) -> None:
    """A COMMAND row is a described one-liner you read; an argument row is one
    item in a set you SCAN. Eight was reasoned from the first and applied to the
    second, which hid four of twelve providers while seven rows above the list
    sat empty — on the one surface whose whole job is "what is supported".

    The floor still holds at 12 rows, where there genuinely is no room.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, height)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login "
        await pilot.pause()
        # AFTER the buffer opened the list: the harness answers the opening
        # message with its own four rows, so a set made before would be replaced.
        app.editor.picker.set_choices(
            [ArgumentChoice(f"provider-{index:02d}", "", (), "needs login") for index in range(12)]
        )
        await pilot.pause()
        start, end, total = app.editor.picker.visible_window()
        assert (start, end, total) == (0, visible, 12)


@pytest.mark.asyncio
async def test_the_command_list_keeps_its_own_smaller_budget() -> None:
    """Unchanged: the command list's cap is reasoned from the editor's own
    max-height, and a picker that towered over the field would be the trade the
    argument budget is careful not to make everywhere."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 28)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/"
        await pilot.pause()
        start, end, total = app.editor.picker.visible_window()
        assert (start, end) == (0, MAX_VISIBLE_ROWS)
        assert total > MAX_VISIBLE_ROWS, "premise: the registry overflows the cap"


@pytest.mark.asyncio
async def test_the_names_align_with_the_text_the_editor_is_completing() -> None:
    """The picker's gutter is the prompt's own indent, so a name sits under the
    text it completes rather than one cell to its left.

    Both columns are MEASURED off the rendered frame — nothing here is keyed to
    a hard-coded x, because the dock's position depends on the layout around it.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/login "
        await pilot.pause()
        await pilot.pause()
        lines = [
            "".join(segment.text for segment in strip)
            for strip in app.screen._compositor.render_strips()
        ]
        typed = next(line for line in lines if "/login" in line)
        row = next(line for line in lines if "alibaba" in line)
        assert typed.index("/login") == row.index("alibaba")


# -- mouse wheel -------------------------------------------------------------


def _wheel_down() -> events.MouseScrollDown:
    """A real wheel-down event.

    The real class rather than a stand-in: the handlers are typed against
    Textual's events, and a duck-typed fake would keep passing while the
    signature it defends drifted. Two helpers rather than one parameterised
    one so each returns a single concrete type the handlers accept.
    """
    return events.MouseScrollDown(
        widget=None, x=1, y=1, delta_x=0, delta_y=1, button=0, shift=False, meta=False, ctrl=False
    )


def _wheel_up() -> events.MouseScrollUp:
    """A real wheel-up event; see :func:`_wheel_down`."""
    return events.MouseScrollUp(
        widget=None, x=1, y=1, delta_x=0, delta_y=-1, button=0, shift=False, meta=False, ctrl=False
    )


def test_the_wheel_moves_the_highlight_one_row_at_a_time() -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    assert picker.selected_index == 0
    picker.on_mouse_scroll_down(_wheel_down())
    assert picker.selected_index == 1
    picker.on_mouse_scroll_up(_wheel_up())
    assert picker.selected_index == 0


def test_the_wheel_clamps_where_the_arrows_wrap() -> None:
    """``move`` wraps, which suits a discrete arrow press. A wheel gesture that
    jumped from the last command back to the first reads as the menu having
    reset itself."""
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    last = len(picker.suggestions()) - 1
    for _ in range(last + 10):
        picker.on_mouse_scroll_down(_wheel_down())
    assert picker.selected_index == last
    for _ in range(last + 10):
        picker.on_mouse_scroll_up(_wheel_up())
    assert picker.selected_index == 0
    # The arrow key still wraps: the wheel path must not have changed it.
    picker.move(-1)
    assert picker.selected_index == last


def test_the_wheel_scrolls_the_argument_submenu_too() -> None:
    """The same widget renders a command's ARGUMENT choices, so the submenu
    has to scroll by the same gesture — it is the surface the user is looking
    at when they reach for the wheel."""
    picker = _argument_picker(PROVIDER_CHOICES)
    assert len(picker.suggestions()) > 1
    picker.on_mouse_scroll_down(_wheel_down())
    assert picker.selected_index == 1


def test_the_wheel_is_stopped_so_the_transcript_behind_stays_put() -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.sync("/")
    down, up = _wheel_down(), _wheel_up()
    picker.on_mouse_scroll_down(down)
    picker.on_mouse_scroll_up(up)
    assert down._stop_propagation and up._stop_propagation


def test_the_wheel_on_a_closed_picker_is_a_no_op() -> None:
    picker = _picker(SLASH_COMMANDS)
    picker.on_mouse_scroll_down(_wheel_down())  # must not raise
    assert picker.suggestions() == []
