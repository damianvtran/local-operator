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
    InlineCommandRequested,
    RefreshArgumentChoices,
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


#: Team rows the `/team` argument list offers. NAME+message commands: choosing
#: a row fills the name and a space and waits for the message, and the typed
#: name is highlighted once it exactly matches one of these.
TEAM_CHOICES = [
    ArgumentChoice("frontend-guild", "Ship the web surface", detail="3 roles"),
    ArgumentChoice("release-crew", "Cut releases", detail="2 roles"),
]

#: Agent rows the `/agent` argument list offers — the mirror of TEAM_CHOICES.
AGENT_CHOICES = [
    ArgumentChoice("auditor", "Audit changes for risk", detail="role"),
    ArgumentChoice("dashboard-sme", "Knows the dashboard", detail="specialist"),
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
        #: Inline slash commands the editor spliced out of a draft (the
        #: ``InlineCommandRequested`` path), captured to prove a mid-text run
        #: dispatches without submitting the surviving draft.
        self.inline_commands: list[str] = []
        self.argument_refreshes: list[str] = []

    def set_editor_text(self, text: str) -> None:
        """Set the buffer AND park the caret at the end, as typing would.

        ``editor.text = x`` leaves the caret at ``(0, 0)``; a user who typed
        ``x`` has it at the end. Slash detection is caret-anchored — which slash
        token is active depends on where the caret is — so a test that sets the
        text without moving the caret is asserting about a state the UI never
        produces. This is the faithful shortcut for "the user typed this".
        """
        self.editor.text = text
        self.editor.move_cursor(self.editor._end_of_buffer())
        # The ``text`` setter re-syncs the picker with the caret still at the
        # origin (Textual moves it after load), so the sync above saw no active
        # token. Re-sync now that the caret sits where a typist left it — this
        # is the resync a real keystroke does on every press.
        self.editor._sync_picker()

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

    def on_inline_command_requested(self, message: InlineCommandRequested) -> None:
        self.inline_commands.append(message.command_text)

    def on_refresh_argument_choices(self, message: RefreshArgumentChoices) -> None:
        message.stop()
        self.argument_refreshes.append(message.command)

    def on_argument_query_opened(self, message: ArgumentQueryOpened) -> None:
        # Stands in for the app's controller-backed answer. The harness keeps the
        # two sets distinct so `/logout` can be shown to offer strictly less.
        message.stop()
        self.provider_queries.append(message.command)
        if message.command in ("team", "teams", "agent", "agents"):
            # Mirror the app: fill the rows AND hand the editor the name snapshot
            # its highlighter reads, so the NAME+message completion and the
            # argument-name highlight are exercised over the real registry seam.
            choices = TEAM_CHOICES if message.command in ("team", "teams") else AGENT_CHOICES
            self.editor.picker.set_choices(list(choices))
            self.editor.set_name_choices(frozenset(c.name.lower() for c in choices))
            return
        choices = _logout_choices() if message.command == "logout" else PROVIDER_CHOICES
        self.editor.picker.set_choices(list(choices))


# ---------------------------------------------------------------------------
# trigger rules (pure)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # Bare slash opens the full menu.
        ("/", (0, "", 1)),
        # A word being typed.
        ("/mo", (0, "mo", 3)),
        # Leading whitespace is fine: the first NON-BLANK line counts, and
        # the completion must not discard that whitespace (start=2).
        ("  /mo", (2, "mo", 5)),
        ("\n\n  /x", (4, "x", 6)),
        # INLINE: a command typed after a message, at a word boundary, opens the
        # list too — the whole point of the feature. ``start`` indexes the slash,
        # ``end`` the cell past the word, so a completion rebuilds only that span.
        ("fix this /mo", (9, "mo", 12)),
        # INLINE on a later line — a command dropped under a multi-line draft.
        ("line one\n/te", (9, "te", 12)),
    ],
)
def test_trigger_cases(text: str, expected: tuple[int, str, int]) -> None:
    context = slash_context(text)
    assert context is not None
    assert (context.start, context.query, context.end) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # nothing typed
        "hello",  # not a slash
        "/model ",  # whitespace terminates the command word
        "/model gpt",  # an argument means the command is already chosen
        "/mo\nrest",  # the caret defaults to the buffer end, on the second line
        "/mo\n",
        "src/foo",  # a glued slash is punctuation inside a word, not a command
        "and/or",
    ],
)
def test_non_trigger_cases(text: str) -> None:
    assert slash_context(text) is None


@pytest.mark.parametrize(
    "text, cursor, expected",
    [
        # Caret still on the first line's command word: the picker shows for the
        # command even though a message follows on the next line.
        ("/te\nfix this", 3, (0, "te", 3)),
        # Caret between two inline slash tokens picks the one it is editing.
        ("a /foo /ba", 10, (7, "ba", 10)),
        ("a /foo /ba", 6, (2, "foo", 6)),
    ],
)
def test_caret_anchored_trigger_cases(
    text: str, cursor: int, expected: tuple[int, str, int]
) -> None:
    context = slash_context(text, cursor)
    assert context is not None
    assert (context.start, context.query, context.end) == expected


@pytest.mark.parametrize(
    "text, cursor",
    [
        # Caret out on the message line: the command word above is terminated.
        ("/te\nfix this", 8),
        # Caret before every slash on the line: nothing is being edited yet.
        ("a /foo", 1),
    ],
)
def test_caret_anchored_non_trigger_cases(text: str, cursor: int) -> None:
    assert slash_context(text, cursor) is None


def test_a_nested_slash_inside_an_engaged_command_is_plain_text() -> None:
    """Once a recognised command owns the line (``/team a …``), a second slash
    inside its argument is plain text — the command claims to the line end, so
    the nested ``/team`` never becomes an active token to highlight or run."""
    known = frozenset({"team", "teams", "goal"})
    text = "/team alpha improve the /team command"
    # Caret right after the nested "/team" (index 29).
    assert slash_context(text, 29, known) is None
    # Without the vocabulary the pure parser cannot know a command claimed the
    # line, so the nested slash is the active token — which is exactly why the
    # editor threads its command set in.
    assert slash_context(text, 29) is not None
    # The first command word is still recognised at its own position.
    ctx = slash_context(text, 5, known)
    assert ctx is not None and ctx.query == "team"


def test_a_crlf_buffer_does_not_leak_a_carriage_return_into_the_word() -> None:
    """A draft restored with CRLF line endings must still parse: the trailing
    ``\\r`` on a non-final line is stripped so the word matches the command set
    and an argument value never carries the control character (round 1, minor-1)."""
    from local_operator.tui.widgets.command_picker import slash_argument, slash_word

    # Caret at the end of "/team" on the first CRLF line (index 5).
    assert slash_word("/team\r\nfix", 5, frozenset({"team"})) == "team"
    # And the argument value on a CRLF line carries no trailing "\r".
    assert slash_argument("/team ops\r\nmsg", ("team",), 9) == "ops"


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
async def test_text_before_the_slash_now_opens_it_inline() -> None:
    """The reported gesture: text typed, then a command remembered mid-draft.

    This USED to assert the picker stayed hidden — the old rule was "the slash
    must be the first character of the first non-blank line". Inline detection is
    exactly the reversal of that rule: a boundary slash anywhere in the draft
    opens the list.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("h", "i", "space", "slash", "m", "o")
        await pilot.pause()
        assert app.editor.picker.is_open()
        assert app.editor.text == "hi /mo"


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


@pytest.mark.asyncio
async def test_inline_command_opens_the_picker_mid_draft() -> None:
    """The reported gesture: a message typed, then a command remembered."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for char in "fix this ":
            await pilot.press("space" if char == " " else char)
        await pilot.press("slash", "t", "e")
        await pilot.pause()
        assert app.editor.picker.is_open(), "an inline /te must open the list"
        assert app.editor.text == "fix this /te"


@pytest.mark.asyncio
async def test_a_glued_slash_never_opens_the_picker() -> None:
    """``src/foo`` is a path; its slash is not at a word boundary."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for char in "src":
            await pilot.press(char)
        await pilot.press("slash", "f", "o", "o")
        await pilot.pause()
        assert not app.editor.picker.is_open()


@pytest.mark.asyncio
async def test_inline_command_runs_and_is_spliced_out_keeping_the_draft() -> None:
    """The whole reported gesture end to end: type a message, append a command,
    run it — the command leaves the buffer, the message stays, and the command is
    dispatched through the inline path (not submitted as a prompt)."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for char in "fix the bug ":
            await pilot.press("space" if char == " " else char)
        await pilot.press("slash", "u", "s", "a", "g", "e")
        await pilot.pause()
        assert app.editor.picker.highlighted_name() == "usage"
        await pilot.press("enter")
        await pilot.pause()
        # The command ran through the inline path, not the submit path.
        assert app.inline_commands == ["/usage"]
        assert app.submissions == []
        # The token is gone and the message survives, with the trailing space the
        # inline gesture added removed too.
        assert app.editor.text == "fix the bug"
        assert app.editor.picker.is_open() is False


@pytest.mark.asyncio
async def test_inline_command_on_its_own_line_keeps_the_message_below() -> None:
    """A command dropped on its OWN line above a multi-line draft runs and is
    spliced out, leaving the message. This is the unambiguous way to route a
    draft whose message should stay when the command comes first — the command's
    argument runs to its line end, so nothing on the next line is absorbed."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "u", "s", "a", "g", "e")
        # A hard newline, then the message on the line below.
        await pilot.press("shift+enter")
        for char in "and then this":
            await pilot.press("space" if char == " " else char)
        # Run the command on line one: move the caret back onto its word.
        app.editor.move_cursor(app.editor._location_at_offset(6))
        app.editor._sync_picker()
        await pilot.pause()
        assert app.editor.picker.is_open()
        await pilot.press("enter")
        await pilot.pause()
        assert app.inline_commands == ["/usage"]
        assert app.editor.text == "and then this"


@pytest.mark.asyncio
async def test_a_prompt_command_reassembles_to_the_front_staged_not_run() -> None:
    """A PROMPT command (``/goal``) engaged inline moves to the front with the
    draft as its argument and is STAGED — never auto-run, so the draft is never
    consumed as a name (the D1 data-loss the naive end-of-line argument caused)."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for char in "land the oauth fix ":
            await pilot.press("space" if char == " " else char)
        for char in "/goal":
            await pilot.press("slash" if char == "/" else char)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        # Reassembled to the front, staged. Nothing ran or submitted.
        assert app.editor.text == "/goal land the oauth fix"
        assert app.inline_commands == []
        assert app.submissions == []


@pytest.mark.asyncio
async def test_a_whole_buffer_command_still_submits_not_inline() -> None:
    """A command that IS the whole draft goes through the ordinary submit path —
    inline splicing is only for a command sharing the buffer with a message."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        await pilot.press("slash", "u", "s", "a", "g", "e")
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/usage "]
        assert app.inline_commands == []


@pytest.mark.asyncio
async def test_type_through_dismiss_leaves_the_text_when_the_word_stops_matching() -> None:
    """Typing PAST a command word that no longer matches anything closes the list
    but keeps every character — the draft is never touched by a dismissal."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # `/zzz` matches no command, so the list closes; the text is still there.
        await pilot.press("slash", "z", "z", "z")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        assert app.editor.text == "/zzz"
        # Terminating the word with a space also dismisses and keeps the text —
        # the user typed on past the command into a message.
        await pilot.press("space", "h", "i")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        assert app.editor.text == "/zzz hi"


@pytest.mark.asyncio
async def test_esc_dismiss_leaves_an_inline_command_in_the_text() -> None:
    """Esc on the picker leaves the text exactly as typed, even inline and even
    when it matches — the same 'not now' the start-of-line list already honours."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        for char in "ship it ":
            await pilot.press("space" if char == " " else char)
        await pilot.press("slash", "u", "s")
        await pilot.pause()
        assert app.editor.picker.is_open()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        assert app.editor.text == "ship it /us"
        assert app.inline_commands == []
        assert app.submissions == []


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
        ("u", ["update", "usage"]),
        ("g", ["goal"]),
        # `settings` and its `config` alias join these two prefixes; both are
        # ranked ahead of their neighbours because a prefix match on a longer
        # word still beats one that starts later in the name.
        ("s", ["settings", "search", "skills"]),
        # Every one of these is a flat 900 prefix match (verified, not assumed:
        # `score_command_text_match("/c", …)` returns 900 for all six), so
        # `match_commands` sorts them on `(-score, registry_index)` and the
        # order here IS registration order. `copy` is second because its
        # registry entry sits directly after `clear`, not because it is
        # shorter — there is no length tiebreak to appeal to, and inventing one
        # would send the next person editing this list when the real cause is
        # a registry move.
        ("c", ["clear", "copy", "config", "context", "compact", "credential"]),
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


def test_an_empty_detail_row_reclaims_the_reserved_column_for_its_description() -> None:
    """D2: a row whose OWN detail is empty pays nothing for the detail column.

    The column is reserved from the widest detail in the SET so the states scan
    down one edge (the /login, /theme, /mcp behaviour the other tests pin). But
    a row that puts nothing at that edge — most /theme rows, /mcp subcommands,
    /agent specialists — was still charged the column's width, clipping its
    description to make room for a column it never fills. Now it reclaims that
    slack while detail-bearing rows keep the shared edge.
    """
    choices = [
        ArgumentChoice(
            "dark",
            "A description long enough to be clipped by the reserved detail column",
            (),
            "current",
        ),
        ArgumentChoice(
            "monokai",
            "A description long enough to be clipped by the reserved detail column",
            (),
            "",
        ),
    ]
    picker = _argument_picker(choices)
    rows = [row.plain for row in picker.render_rows(60)]
    # The detail-bearing row still ends at the shared column edge.
    assert rows[0].rstrip().endswith("current")

    # The empty-detail row inks more of its (identical) description than the
    # detail-bearing row does, because it reclaimed the reserved column.
    def desc_len(row: str, name: str) -> int:
        body = row.split(name, 1)[1]
        # Strip the trailing state word so only the description ink is compared.
        return len(body.replace("current", "").rstrip())

    assert desc_len(rows[1], "monokai") > desc_len(rows[0], "dark")


def test_the_empty_detail_reclaim_does_not_move_the_scannable_state_edge() -> None:
    """D2 must not regress the shared-edge scan the state column exists for.

    Every row that HAS a detail still starts it at one x — the reclaim only
    frees rows that place nothing there, so a /login-style list where every row
    carries a state is untouched.
    """
    rows = [row.plain for row in _argument_picker(list(PROVIDER_CHOICES)).render_rows(80)]
    starts = _column_starts(rows, ("logged in", "env key", "needs login"))
    assert len(starts) == 1, f"states start at different columns: {sorted(starts)}"


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
        app.set_editor_text("/login ")
        await pilot.pause()
        assert [name for name, _ in app.editor.picker.suggestions()] == [
            c.name for c in PROVIDER_CHOICES
        ]
        app.set_editor_text("/logout ")
        await pilot.pause()
        assert app.provider_queries == ["login", "logout"]
        assert [name for name, _ in app.editor.picker.suggestions()] == ["anthropic", "alibaba"]


@pytest.mark.asyncio
async def test_tab_completes_the_provider_id_without_running_it() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/login clau")
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
        app.set_editor_text("/logout anthropic")
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
        app.set_editor_text("/logout a")
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
        app.set_editor_text("/logout a")
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
        app.set_editor_text("/login ")
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
        app.set_editor_text("/login clau")
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
        app.set_editor_text("/login ")
        await pilot.pause()
        await pilot.click(CommandPicker, offset=(4, 1))
        # Wait for the SUBMISSION, not for one tick. A click travels through the
        # picker's row handler and is submitted by a posted message, so the
        # single ``pause()`` this replaced was a bet on that landing within one
        # idle tick — a bet the census watched lose on a contended xdist worker
        # (``assert [] == ['/login alibaba']``). The ceiling is a deadlock
        # guard: a click that never runs the row still fails the assertion below.
        for _ in range(200):
            if app.submissions:
                break
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
        app.set_editor_text("/logout ")
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
        app.set_editor_text(f"/logout {query}")
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
        app.set_editor_text("/login aib")
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
        app.set_editor_text("/logout ")
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
        app.set_editor_text("/logout ")
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
        app.set_editor_text("/login ")
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
        app.set_editor_text("/")
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
        app.set_editor_text("/login ")
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


# ---------------------------------------------------------------------------
# NAME+message completion (`/team`, `/agent`) — fill a name and a space, never
# submit; the message tail and the blank-attach are the existing submit paths.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enter_on_a_team_row_fills_the_name_and_a_space_without_submitting() -> None:
    """Enter on a team row is "chosen, now type the message", not "switch now".

    The key difference from a provider row (which Enter RUNS when unambiguous):
    for a NAME+message command neither Tab nor Enter ever submits — the name is
    filled with a trailing space and the caret parks after it.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team frontend-guild")
        await pilot.pause()
        assert app.editor.picker.highlighted_name() == "frontend-guild"
        await pilot.press("enter")
        await pilot.pause()
        assert app.editor.text == "/team frontend-guild "
        assert app.editor.cursor_location == app.editor._end_of_buffer()
        assert app.submissions == []
        assert not app.editor.picker.is_open()
        assert app.editor.has_focus


@pytest.mark.asyncio
async def test_tab_on_a_team_row_fills_name_and_space() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team front")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        # Tab fills the name AND a space here — the opposite of the provider
        # list, where Tab leaves no space so the matcher keeps matching.
        assert app.editor.text == "/team frontend-guild "
        assert app.submissions == []


@pytest.mark.asyncio
async def test_click_on_a_team_row_fills_name_and_space_without_submitting() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team ")
        await pilot.pause()
        await pilot.click(CommandPicker, offset=(4, 0))
        await pilot.pause()
        assert app.editor.text == "/team frontend-guild "
        assert app.submissions == []
        assert not app.editor.picker.is_open()


@pytest.mark.asyncio
async def test_arrowing_onto_a_team_row_still_does_not_submit() -> None:
    """The unambiguous branch that RUNS a provider must not run a team row: for
    a NAME+message command "one match / arrowed" is still only "ready"."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team ")
        await pilot.pause()
        await pilot.press("down")  # arrow onto row 1 — an explicit move
        await pilot.pause()
        assert app.editor.picker.highlighted_name() == "release-crew"
        await pilot.press("enter")
        await pilot.pause()
        assert app.editor.text == "/team release-crew "
        assert app.submissions == []


@pytest.mark.asyncio
async def test_blank_enter_after_a_completed_team_name_submits_attach_only() -> None:
    """The blank tail after the space is not a picker gesture — the list is
    closed, so Enter submits, and the app's dispatch collapses the bare name to
    an attach-only switch."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team frontend-guild ")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        await pilot.press("enter")
        await pilot.pause()
        # The whole line submits; the arg is `frontend-guild ` which dispatch
        # strips to the bare name (attach-only). We assert the submitted text.
        assert app.submissions == ["/team frontend-guild "]


@pytest.mark.asyncio
async def test_typing_a_message_then_enter_sends_the_whole_line() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team frontend-guild fix the flaky test")
        await pilot.pause()
        assert not app.editor.picker.is_open()
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/team frontend-guild fix the flaky test"]


@pytest.mark.asyncio
async def test_agent_row_behaves_like_team() -> None:
    """`/agent` is the mirror of `/team`: complete the name + space, no submit."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/agent audit")
        await pilot.pause()
        assert app.editor.picker.highlighted_name() == "auditor"
        await pilot.press("enter")
        await pilot.pause()
        assert app.editor.text == "/agent auditor "
        assert app.submissions == []


@pytest.mark.asyncio
async def test_enum_tail_login_row_still_runs_on_enter() -> None:
    """Regression guard: the enum-tail path is untouched — an unambiguous
    provider Enter still completes AND runs, unlike a name command."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/login anthropic")
        await pilot.pause()
        assert len(app.editor.picker.suggestions()) == 1
        await pilot.press("enter")
        await pilot.pause()
        assert app.submissions == ["/login anthropic"]


# ---------------------------------------------------------------------------
# autofill discoverability hint (ux round 1, U1/U2) — at the parked-caret
# moment, name both outcomes: blank Enter switches, a typed message sends.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_autofill_shows_the_switch_or_send_hint() -> None:
    """The moment a name is autofilled with an empty tail, the picker's notice
    row names both outcomes — the fix for U1/U2, where nothing on screen told
    the user that Enter-now switches while type-then-Enter sends."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team frontend-guild")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.editor.text == "/team frontend-guild "
        assert app.editor.picker._notice == Editor.NAME_SWITCH_HINT
        # It shows in the picker's own notice place, not the transcript.
        assert app.editor.picker.display


@pytest.mark.asyncio
async def test_the_hint_is_withdrawn_once_a_message_is_typed() -> None:
    """Typing the first message character means the user chose "send" — the
    hint must clear so it never sits over a live message."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team frontend-guild ")
        app.editor.move_cursor(app.editor._end_of_buffer())  # as the autofill leaves it
        await pilot.pause()
        assert app.editor.picker._notice == Editor.NAME_SWITCH_HINT
        await pilot.press("f")
        await pilot.pause()
        assert app.editor.text == "/team frontend-guild f"
        assert app.editor.picker._notice == ""


@pytest.mark.asyncio
async def test_a_bare_name_list_shows_no_hint() -> None:
    """While the row list is still up (`/team ` with no name chosen), the rows
    answer the question — the hint would compete with them, so it stays away."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team ")
        await pilot.pause()
        assert app.editor.picker.is_open()  # rows are showing
        assert app.editor.picker._notice == ""


@pytest.mark.asyncio
async def test_enum_tail_completion_shows_no_switch_hint() -> None:
    """The hint is for NAME+message commands only — an enum-tail argument like
    `/login anthropic ` must never surface it."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/login anthropic ")
        await pilot.pause()
        assert app.editor.picker._notice == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command, notice",
    [
        ("logout", "no stored credentials — nothing to log out of."),
        ("effort", "this model takes no effort setting"),
    ],
)
async def test_app_owned_notice_survives_a_same_command_keystroke(
    command: str, notice: str
) -> None:
    """CR4 regression (round 2): the U1/U2 hint write must NOT erase the app's
    notices for the OTHER argument commands.

    `/logout` (empty credential store) and `/effort` (a model with no effort
    key) set an informational notice ONCE on the command-word change and never
    re-set it per keystroke — for those empty-by-construction lists the notice
    IS the whole content the user reads. The hint's `set_notice` runs on every
    keystroke of the argument, so before the fix it called `set_notice("")` for
    these non-name commands on the first query character and wiped the notice
    (picker closed). The fix gates the write to NAME+message commands only, so a
    same-command keystroke must leave the app-owned notice intact.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # Open the argument list, then set the notice the app would set on the
        # command-word change (the harness does not model the empty-list case,
        # so we set it directly — the regression is purely in the editor's
        # per-keystroke resync, not in how the notice first arrives).
        app.set_editor_text(f"/{command} ")
        await pilot.pause()
        app.editor.picker.set_notice(notice)
        await pilot.pause()
        assert app.editor.picker._notice == notice, "premise: the notice is shown"
        # Type a query character on the SAME command — this re-runs _sync_picker.
        # A real same-command keystroke (the caret is already parked at the end
        # of `/<command> `), which is the per-keystroke resync the CR4 fix is
        # about — not a whole-buffer reset.
        await pilot.press("x")
        await pilot.pause()
        assert (
            app.editor.picker._notice == notice
        ), "the app-owned notice must survive a same-command keystroke (CR4)"


# ---------------------------------------------------------------------------
# slash-command syntax highlighting — the render pass paints the recognized
# command word, the recognized name, and a muted unknown word, and nothing else.
# ---------------------------------------------------------------------------


def _slash_ink(editor: Editor, y: int = 0) -> list[tuple[str, str | None]]:
    """(text, fg hex) per segment of the editor's rendered row ``y``.

    Reads the FINISHED strip ``render_line`` produces — the same thing the
    terminal is sent — so the assertions probe what the user actually sees, not
    an intention. Foreground only, because the highlight is foreground-only.
    """
    strip = editor.render_line(y)
    cells: list[tuple[str, str | None]] = []
    for segment in strip._segments:
        style = segment.style
        fg = style.color.get_truecolor().hex.lower() if style and style.color else None
        cells.append((segment.text, fg))
    return cells


def _ink_of(cells: list[tuple[str, str | None]], text: str) -> str | None:
    """The fg of the first segment whose text is exactly ``text``."""
    return next(fg for seg_text, fg in cells if seg_text == text)


@pytest.mark.asyncio
async def test_recognized_command_word_gets_the_slash_command_style() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # A space closes the command picker so the highlight is not suppressed.
        app.editor.text = "/usage "
        await pilot.pause()
        cells = _slash_ink(app.editor)
        signal = theme_mod.semantic_color("signal").lower()
        assert _ink_of(cells, "/") == signal
        assert _ink_of(cells, "usage") == signal


@pytest.mark.asyncio
@pytest.mark.parametrize("text, token", [("/goal improve recall", "/goal"), ("/loop 3", "/loop")])
async def test_goal_and_loop_get_the_recognized_command_style(text: str, token: str) -> None:
    """Prompt-like commands must give the same pre-submit recognition cue as /team."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text(text)
        await pilot.pause()
        cells = _slash_ink(app.editor)
        signal = theme_mod.semantic_color("signal").lower()
        assert _ink_of(cells, token) == signal


@pytest.mark.asyncio
async def test_unknown_command_word_gets_the_unknown_style_when_picker_closed() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/notacommand hello"
        await pilot.pause()
        assert not app.editor.picker.is_open()
        cells = _slash_ink(app.editor)
        # `dim`, darkened from `muted` in design round 1 (D1) so the inert
        # "will be sent as text" read lands against the prose tail.
        dim = theme_mod.semantic_color("dim").lower()
        assert _ink_of(cells, "/") == dim
        assert _ink_of(cells, "notacommand") == dim
        # The message tail after the space stays prose.
        fg = theme_mod.semantic_color("fg").lower()
        assert _ink_of(cells, " hello ") == fg


@pytest.mark.asyncio
async def test_command_word_is_not_flagged_unknown_while_the_picker_is_open() -> None:
    """A prefix under an open command list is in progress, not a typo — the
    unknown treatment is suppressed so it does not flash on every keystroke."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/te"
        await pilot.pause()
        assert app.editor.picker.is_open()
        assert app.editor.picker.mode is PickerMode.COMMAND
        cells = _slash_ink(app.editor)
        dim = theme_mod.semantic_color("dim").lower()
        # `/te` is not a full command, but no /-token cell is painted the unknown
        # colour while the picker is choosing. Restrict to the command token so a
        # coincidental `dim` elsewhere on the row (e.g. an unfocused chevron)
        # cannot mask a real regression.
        token = [fg for text, fg in cells if text.strip() in ("/", "/te", "te")]
        assert token and all(fg != dim for fg in token)


@pytest.mark.asyncio
async def test_empty_name_list_catches_up_against_the_current_query() -> None:
    """A late roster reopens `/team lop` without deleting or retyping the word."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team ")
        await pilot.pause()
        # Reproduce the boot race after the one transition fill: the session did
        # not exist yet, so the picker entered ARGUMENT mode with no rows.
        app.editor.picker.set_choices([])
        app.argument_refreshes.clear()
        app.set_editor_text("/team lop")
        await pilot.pause()
        assert app.editor.picker._query == "lop"
        assert app.editor.picker.is_pending()
        # `set_editor_text` mirrors the text setter and the final caret-resync a
        # real keystroke performs; both see the same still-empty list and retry.
        assert app.argument_refreshes and set(app.argument_refreshes) == {"team"}

        # The session/registry arrives later. `set_choices` must rank against the
        # live query it already holds, not reset to the opening empty string.
        choices = [ArgumentChoice("lopdev", "Build Local Operator", detail="3 roles")]
        app.editor.picker.set_choices(choices)
        app.editor.set_name_choices(frozenset({"lopdev"}))
        await pilot.pause()
        assert app.editor.text == "/team lop"
        assert app.editor.picker.is_open()
        assert app.editor.picker.highlighted_name() == "lopdev"


@pytest.mark.asyncio
async def test_empty_name_list_does_not_refresh_after_completed_name() -> None:
    """The parked switch/send state is past catch-up even if rows are absent."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("/team frontend-guild ")
        await pilot.pause()
        app.editor.picker.set_choices([])
        app.argument_refreshes.clear()
        app.editor._sync_picker()
        await pilot.pause()
        assert app.argument_refreshes == []
        assert "switch" in app.editor.picker._notice
        assert "send" in app.editor.picker._notice


@pytest.mark.asyncio
async def test_recognized_team_name_gets_the_argument_style() -> None:
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # The highlighter is caret-independent (it parses the first content
        # line), so this sets the text directly and pushes the name snapshot the
        # highlight reads — mirroring the app's "list opens, then names arrive"
        # without depending on where the caret lands.
        app.editor.text = "/team frontend-guild fix it"
        await pilot.pause()
        app.editor.set_name_choices(frozenset({"frontend-guild"}))
        await pilot.pause()
        cells = _slash_ink(app.editor)
        signal = theme_mod.semantic_color("signal").lower()
        green = theme_mod.semantic_color("string").lower()
        fg = theme_mod.semantic_color("fg").lower()
        assert _ink_of(cells, "team") == signal
        assert _ink_of(cells, "frontend-guild") == green
        # The message tail is NOT highlighted — the whole point of the feature.
        assert _ink_of(cells, " fix it ") == fg


@pytest.mark.asyncio
async def test_a_name_token_that_soft_wraps_is_highlighted_across_both_rows() -> None:
    """The wrap-boundary branch of ``_slash_cells`` (the intricate new math):
    a name longer than the composer's content width breaks mid-token across two
    screen rows, and BOTH halves must carry the argument-name style — otherwise
    a long team/agent name would light up only until the wrap and read as broken.

    Driven at a narrow width so the name (not just name+message) straddles a wrap
    offset, which is exactly the ``wraps_on``/``section_end`` case the straight
    single-row tests never reach. The snapshot is pushed AFTER the text settles
    because the harness's ``on_argument_query_opened`` re-pushes its fixed team
    rows on the space; the app's real ordering (list opens, THEN names arrive) is
    unaffected — this only mirrors it deterministically for a bespoke name.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(24, 20)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # 29 chars, wider than the ~18-cell content box, so it must break inside
        # the token rather than at the space before the message.
        long_name = "frontend-platform-guild-alpha"
        app.editor.text = f"/team {long_name} go now"
        await pilot.pause()
        app.editor.set_name_choices(frozenset({long_name}))
        await pilot.pause()
        wrapped = app.editor.wrapped_document
        assert len(wrapped.get_offsets(0)) >= 2, "premise: the name itself wraps"
        green = theme_mod.semantic_color("string").lower()
        # Gather every green run across all the wrapped rows of document line 0.
        painted = ""
        for y in range(wrapped.height):
            for text, fg in _slash_ink(app.editor, y):
                if fg == green:
                    painted += text
        # The whole name is painted, contiguously, and nothing more — the message
        # tail (`go now`) never picks up the argument-name colour.
        assert painted == long_name


@pytest.mark.asyncio
async def test_partial_name_is_not_highlighted() -> None:
    """A half-typed name is normal in-progress state, not an error — only an
    exact snapshot hit paints the name."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.editor.text = "/team front"
        await pilot.pause()
        cells = _slash_ink(app.editor)
        green = theme_mod.semantic_color("string").lower()
        assert all(fg != green for _, fg in cells)


@pytest.mark.asyncio
async def test_ordinary_command_highlight_dies_on_a_newline() -> None:
    """An ordinary command lives on line 0 only; a newline makes the rest a
    message body and the user has abandoned the command, so no command highlight
    may leak onto either row."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # `/usage` is not a NAME+message command, so the single-line discipline
        # still applies: the newline turns it into prose and the highlight dies.
        app.set_editor_text("/usage some\nmore text")
        await pilot.pause()
        signal = theme_mod.semantic_color("signal").lower()
        green = theme_mod.semantic_color("string").lower()
        dim = theme_mod.semantic_color("dim").lower()
        for y in (0, 1):
            cells = _slash_ink(app.editor, y)
            assert all(fg not in (signal, green, dim) for _, fg in cells)


@pytest.mark.asyncio
async def test_name_command_keeps_highlight_across_a_multiline_message() -> None:
    """The reported bug: `/team <known-name>` highlights the command and name
    tokens on a single line, then loses ALL highlight the instant a newline is
    added to the message.

    NAME+message commands (`/team`, `/agent`) are DEFINED as
    ``/<cmd> <name> <free-text message>`` where the message is expected to span
    lines, and the leading command still dispatches as that command across the
    newline. So the command and name tokens must stay lit over a multi-line body,
    while the message tail — on line 0 or the wrapped lines — stays prose.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # Open the list on a single line so the app pushes the name snapshot...
        app.set_editor_text("/team frontend-guild fix it")
        app.editor.set_name_choices(frozenset({"frontend-guild"}))
        await pilot.pause()
        signal = theme_mod.semantic_color("signal").lower()
        green = theme_mod.semantic_color("string").lower()
        # ``set_editor_text`` parks the caret at the end, so the command renders
        # as one ``/team`` segment (not a split ``/`` + ``team``).
        single = _slash_ink(app.editor)
        assert _ink_of(single, "/team") == signal
        assert _ink_of(single, "frontend-guild") == green
        # ...now add a newline and keep typing the message. BEFORE the fix every
        # token went dark here; the command and name must stay lit.
        app.set_editor_text("/team frontend-guild\nfix it across\nmany lines")
        await pilot.pause()
        row0 = _slash_ink(app.editor, 0)
        assert _ink_of(row0, "/team") == signal
        assert _ink_of(row0, "frontend-guild") == green
        # The message body lines carry no command/name highlight — they are prose.
        for y in (1, 2):
            body = _slash_ink(app.editor, y)
            assert all(c not in (signal, green) for _, c in body)


@pytest.mark.asyncio
async def test_multiline_name_command_with_partial_name_stays_prose() -> None:
    """A half-typed name is in-progress state even on a multi-line body: only an
    exact snapshot hit paints the name, so a newline must not suddenly light up
    an unrecognized name."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # Open the list so the snapshot is pushed, then break the name and go
        # multi-line: `front` is not a known team.
        app.set_editor_text("/team front")
        app.editor.set_name_choices(frozenset({"frontend-guild"}))
        await pilot.pause()
        app.set_editor_text("/team front\nmore")
        await pilot.pause()
        green = theme_mod.semantic_color("string").lower()
        for y in (0, 1):
            cells = _slash_ink(app.editor, y)
            assert all(fg != green for _, fg in cells)


@pytest.mark.asyncio
async def test_cross_family_word_swap_while_multiline_drops_the_name() -> None:
    """A team name must NOT paint under `/agent` after an atomic word-swap.

    The rosters are disjoint, so a name valid for one family is prose for the
    other. Because a multi-line buffer never re-opens the argument list (the
    picker is suppressed once a newline follows the leading word), an atomic
    replacement `/team <team-name>\\n…` -> `/agent <team-name>\\n…` cannot rely on
    the list re-deriving the right roster — the preserved snapshot has to be
    rejected on the family switch or the team name paints green under `/agent`.
    This locks that: the command word still highlights (it is a recognized
    command), but the inherited team name falls back to prose.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        # Fill the TEAM snapshot on a single line, then go multi-line so it is
        # preserved across the newline.
        app.set_editor_text("/team frontend-guild fix it")
        app.editor.set_name_choices(frozenset({"frontend-guild"}))
        await pilot.pause()
        app.set_editor_text("/team frontend-guild\nfix it")
        await pilot.pause()
        signal = theme_mod.semantic_color("signal").lower()
        green = theme_mod.semantic_color("string").lower()
        row0 = _slash_ink(app.editor)
        assert _ink_of(row0, "frontend-guild") == green  # premise: it paints here
        # Atomic word-swap to /agent while STILL multiline (no list re-opens).
        app.set_editor_text("/agent frontend-guild\nfix it")
        await pilot.pause()
        row0 = _slash_ink(app.editor)
        # The command word is recognized and still highlighted (one ``/agent``
        # segment, caret parked at the end by ``set_editor_text``)...
        assert _ink_of(row0, "/agent") == signal
        # ...but the inherited team name must NOT paint under the wrong family.
        assert all(fg != green for _, fg in row0)


@pytest.mark.asyncio
async def test_name_token_soft_wraps_across_rows_on_a_multiline_body() -> None:
    """The soft-wrap row mapping (`_slash_cells`) must place the name token on
    the right SCREEN rows even when the buffer also has a multi-line message.

    The single-row multiline tests never exercise the wrap-boundary math, and
    the pre-existing soft-wrap test uses a single-line buffer. This is the
    intersection: a name long enough to wrap AND a newline-terminated body, so
    the command line's own wrapped rows carry the tokens while the body lines
    below are left untouched.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(24, 20)) as pilot:
        app.editor.focus()
        await pilot.pause()
        long_name = "frontend-platform-guild-alpha"
        # Make the message multi-line first, THEN push the bespoke snapshot: on a
        # multi-line buffer ``_sync_picker`` does not re-open the argument list
        # (so the harness's ``on_argument_query_opened`` will not overwrite the
        # snapshot with its fixed rows), and the family gate keeps a snapshot
        # whose family matches the leading ``/team``. This mirrors the app's real
        # ordering deterministically for a name not in the fixed roster.
        app.set_editor_text(f"/team {long_name}\nsecond line\nthird line")
        app.editor.set_name_choices(frozenset({long_name}))
        await pilot.pause()
        wrapped = app.editor.wrapped_document
        assert len(wrapped.get_offsets(0)) >= 2, "premise: the name itself wraps"
        green = theme_mod.semantic_color("string").lower()
        # Gather every green run across ALL wrapped rows of the buffer: the whole
        # name is painted, contiguously, and nothing on the body lines is.
        painted = ""
        for y in range(wrapped.height):
            for text, fg in _slash_ink(app.editor, y):
                if fg == green:
                    painted += text
        assert painted == long_name


@pytest.mark.asyncio
async def test_inline_team_reassembly_keeps_name_highlight_with_multiline_draft() -> None:
    """Choosing a late inline `/team` keeps the roster snapshot after reassembly."""
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        app.set_editor_text("review\nthis /team ")
        await pilot.pause()
        app.set_editor_text("review\nthis /team front")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert app.editor.text == "/team frontend-guild review\nthis"
        signal = theme_mod.semantic_color("signal").lower()
        green = theme_mod.semantic_color("string").lower()
        fg = theme_mod.semantic_color("fg").lower()
        row0 = _slash_ink(app.editor)
        assert _ink_of(row0, "/team") == signal
        assert _ink_of(row0, "frontend-guild") == green
        assert _ink_of(row0, " review ") == fg
        assert all(color not in (signal, green) for _, color in _slash_ink(app.editor, 1))


@pytest.mark.asyncio
async def test_team_chart_subcommand_never_paints_a_name() -> None:
    """`/team chart <name>` paints ONLY the `/team` token — never `chart`, never
    the name (single-line AND multiline).

    Guards the #258 integration: `/team` is two-level — `chart` is a reserved
    SUBCOMMAND in the first argument slot, and the team NAME the chart wants
    lives in the SECOND slot after `chart `. The highlighter reads the name as
    the FIRST argument token, which for a chart request is `chart` — not a roster
    member — so no name run is emitted and `chart` stays prose. The test injects
    `chart` AND the second-slot name into `_name_choices` adversarially: even if
    both were roster members, the first-token rule must keep the whole argument
    prose, so a future refactor that reads the name via a caret-anchored helper
    (which would resolve the second-slot token) is caught here.
    """
    app = PickerHarnessApp()
    async with app.run_test(size=(100, 30)) as pilot:
        app.editor.focus()
        await pilot.pause()
        signal = theme_mod.semantic_color("signal").lower()
        green = theme_mod.semantic_color("string").lower()
        # Adversarial snapshot: both the reserved subcommand word and the
        # second-slot name are present, so only the first-token rule prevents a
        # mispaint.
        adversarial = frozenset({"chart", "frontend-guild"})
        for text in ("/team chart frontend-guild", "/team chart frontend-guild\nbody text"):
            app.set_editor_text(text)
            app.editor.set_name_choices(adversarial)
            await pilot.pause()
            cells = _slash_ink(app.editor)
            # `/team` command token is lit...
            assert _ink_of(cells, "/team") == signal
            # ...and nothing on the row is painted with the argument-name colour:
            # neither `chart` nor the second-slot name is highlighted.
            assert all(fg != green for _, fg in cells)
