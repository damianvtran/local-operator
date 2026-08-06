"""Slash-command picker — the suggestion list under the input editor.

Typing ``/`` opens a list of commands; the soft (fuzzy) matching and its
score tiers are NOT reimplemented here. They live in
:mod:`local_operator.tui.autocomplete` and this widget calls
:func:`~local_operator.tui.autocomplete.match_commands` verbatim, because the
ranking is the part users build muscle memory on: a second scoring function
would drift from the one Tab/Enter already apply, and the picker would then
highlight a different command than the one that gets run.

Layout — a borderless two-column list, one row per suggestion (D4):

    ❯ /model  /models    Show or switch model (provider/id)
      /mcp               List MCP servers

* The 2-cell selection gutter lines up with ``#prompt-chevron``, so the
  highlighted ``❯`` sits directly under the prompt's own ``❯`` and every
  command name starts in the same column as the editor's text.
* The primary column fits its content, clamped to 12..32 cells, then two
  cells of gap, then the description fills what is left.
* Under 41 cells the description is dropped entirely — a description squeezed
  into a handful of cells is noise, and the command name is the part the user
  is actually choosing between.

The SAME widget also presents a command's ARGUMENT (``/login <provider>``) in
:attr:`PickerMode.ARGUMENT`: bare names instead of ``/name``, and a
right-aligned ``detail`` column carrying the state the user is choosing by. A
second widget was the alternative, which is how a codebase ends up with two
lists that look almost the same and behave almost the same.

ONE ROW PER SUGGESTION is enforced structurally, not hopefully: every row is
padded/truncated to EXACTLY the render width (so Textual has nothing to wrap)
and the widget's height is pinned to the row count on every repaint. Textual's
``Content.from_rich_text`` discards Rich's ``no_wrap``/``overflow`` flags when
a ``Text`` crosses into a widget — see ``tool_card._row_text`` — so those flags
cannot be relied on and the pinned height is what actually holds the contract.
Widths are measured with ``rich.cells.cell_len`` only, so CJK and emoji
descriptions account for their real cell cost instead of their code-point
count.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, NamedTuple, Sequence

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual import events
from textual.dom import NoScreen
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.autocomplete import (
    ArgumentChoice,
    SlashCommand,
    match_choices,
    match_commands,
)

# The project has exactly ONE cell-accurate truncator and it lives in
# tool_card. Importing it across widget modules is deliberate: a local copy is
# how the len()/cell_len split that this helper exists to prevent gets
# reintroduced, one module at a time.
from local_operator.tui.widgets.tool_card import truncate_cells

#: Row budget. Eight rows matches the editor's own ``max-height: 8``, so a full
#: picker never towers over the field that opened it, and it still shows more
#: than half of the command registry at once.
MAX_VISIBLE_ROWS = 8

#: On a short terminal the budget shrinks to a third of the screen, so the
#: picker can never squeeze the transcript to nothing. At the standard 24 rows
#: this resolves to exactly ``MAX_VISIBLE_ROWS``; at 10 rows it resolves to 3,
#: which is the floor omp clamps its own picker to.
_SCREEN_HEIGHT_DIVISOR = 3

#: The selection mark. The app's prompt and user blocks already speak ``❯``
#: (SPINE_INDENT is 2 cells for exactly this reason); the picker reuses that
#: vocabulary rather than introducing a second cursor glyph.
_CURSOR = "❯"
_GUTTER_CELLS = 2

#: Primary column: fit-to-content, clamped. Below 12 the names of short
#: commands stop forming a column at all; above 32 a single long name pushes
#: every description off the row.
_PRIMARY_COLUMN_MIN = 12
_PRIMARY_COLUMN_MAX = 32
_PRIMARY_COLUMN_GAP = 2

#: A description narrower than this is dropped rather than shown as three
#: characters and an ellipsis.
_MIN_DESCRIPTION_CELLS = 10

#: At or below this width the row collapses to the command name only.
DESCRIPTION_COLLAPSE_WIDTH = 40

#: Right-edge breathing room, so no row ever paints into the last cell.
_EDGE_MARGIN = 2

#: Width assumed for the one repaint that can happen before layout has
#: measured the widget. Height is pinned to the ROW COUNT, which is
#: width-independent, so the worst case is a single frame of narrow rows that
#: the following Resize corrects — never a list that silently doubles height.
_MIN_RENDER_WIDTH = 20

#: Floor for the NAME column of an argument row before its ``detail`` is
#: dropped. The name is the text Tab types into the buffer, so a truncated one
#: is unusable — the user cannot read what to complete to. ``detail`` is worth
#: a lot (it is what `/logout` is chosen by) but never worth that.
_MIN_NAME_CELLS = _PRIMARY_COLUMN_MIN


class PickerMode(Enum):
    """Which kind of list the picker is currently showing.

    Read by the editor, which has to know whether Tab is completing a command
    WORD (rewrite everything from the slash, add a trailing space) or a command's
    ARGUMENT (replace the tail, no trailing space — the space would terminate the
    argument and close the very list Tab just used).
    """

    COMMAND = "command"
    ARGUMENT = "argument"


#: One rendered row: its display name and the thing it stands for. A UNION
#: rather than the :class:`~local_operator.tui.autocomplete.Completable`
#: protocol the matcher is generic over, because rendering needs the concrete
#: fields (a command's aliases, a choice's ``detail``) and dispatching on the
#: item's type makes "argument rows carry a detail column" impossible to get
#: wrong — there is no mode flag to fall out of step with the payload.
_Suggestion = tuple[str, "SlashCommand | ArgumentChoice"]


class _RowStyles(NamedTuple):
    """The styles one row paints with, resolved once from its selection state."""

    selected: bool
    ground: Style
    name: Style
    alias: Style
    description: Style
    cursor: Style


class SlashContext(NamedTuple):
    """Where the command word starts, and the word typed so far.

    ``start`` indexes the ``/`` itself so a completion can rebuild the buffer
    without discarding whatever whitespace the user typed in front of it.
    """

    start: int
    query: str


def slash_context(text: str) -> SlashContext | None:
    """The command word being typed, or ``None`` when the picker must hide.

    The picker is for choosing a command, so it shows only while the command
    WORD is still open: ``/`` first on the first non-blank line, and no
    whitespace yet terminating the token. Once the user types
    ``/model `` they have chosen ``model`` and are typing its argument — a
    list of commands there is stale advice covering the transcript.
    """
    lines = text.split("\n")
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return None
    if len(lines) > first + 1:
        # A newline after the word terminates it exactly like a space does.
        return None
    line = lines[first]
    stripped = line.lstrip()
    if not stripped.startswith("/"):
        return None
    if any(char.isspace() for char in stripped):
        return None
    start = sum(len(other) + 1 for other in lines[:first]) + (len(line) - len(stripped))
    return SlashContext(start, stripped[1:])


def slash_argument(text: str, commands: tuple[str, ...]) -> str | None:
    """The ARGUMENT being typed after one of ``commands``, else ``None``.

    The mirror image of :func:`slash_context`: that one is live while the command
    word is still open, this one takes over the instant the word is terminated by
    a space. Together they mean a single buffer drives two different lists without
    either having to know about the other — ``/mo`` offers commands, ``/model ``
    offers models, and the handover happens on the space the user was going to
    type anyway.

    Returns the argument text, which may be ``""`` (the command word is complete
    but nothing has been typed after it — the state that should show the whole
    catalogue). ``None`` means this is not one of ``commands``.

    Single-line only, matching ``slash_context``: a newline means the user is
    composing a message, not picking from a list.
    """
    lines = text.split("\n")
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None or len(lines) > first + 1:
        return None
    stripped = lines[first].lstrip()
    if not stripped.startswith("/"):
        return None
    word, sep, argument = stripped[1:].partition(" ")
    if not sep or word.lower() not in commands:
        return None
    return argument


#: Below this many typed characters the fuzzy tail is suppressed. A one- or
#: two-letter query matches an arbitrary-looking set by subsequence — `/u`
#: offered `usage, quit, accounts, logout` and `/g` offered
#: `goal, usage, login, logout`. The correct command ranked first every time,
#: but rows 2+ taught the user that the list is unreliable, which is the
#: fastest way to make them stop reading it. Typo tolerance lives at three
#: characters and up (`/cmpct`, `/lgout`), so nothing the feature exists for
#: is affected.
FUZZY_MIN_QUERY_CHARS = 3


def command_suggestions(query: str, commands: list[SlashCommand]) -> list[tuple[str, SlashCommand]]:
    """``(display_name, command)`` suggestions for a typed command word.

    A bare ``/`` cannot go through :func:`match_commands`:
    ``score_command_text_match`` scores an empty prefix at 0 by contract (it is
    what makes "no match" and "nothing typed" distinguishable for the completion
    path, and it is pinned by test), so asking it would answer "no commands" for
    the keystroke whose entire purpose is "show me the commands". The full
    registry, in registration order, IS the answer to ``/``.

    Short queries PREFER prefix matches — see :data:`FUZZY_MIN_QUERY_CHARS` — but
    only when there are some. An empty return closes the picker, and a closed
    picker takes the Tab and Enter guards down with it: Tab falls through to
    stock TextArea behaviour and indents the user's message, Enter submits the
    raw text to the agent. So the gate is a preference, not a filter. It has to
    be, because the queries with no prefix match at all are exactly the natural
    abbreviations the fuzzy matcher exists for — `/lg` for login and logout,
    `/ls` for models and skills, `/qt` for quit, `/md` for model.
    """
    if not query:
        return [(command.name, command) for command in commands]
    matches = match_commands(f"/{query}", commands)
    if len(query) >= FUZZY_MIN_QUERY_CHARS:
        return matches
    lowered = query.lower()
    prefixed = [pair for pair in matches if pair[0].lower().startswith(lowered)]
    return prefixed or matches


def argument_suggestions(
    query: str, choices: list[ArgumentChoice]
) -> list[tuple[str, ArgumentChoice]]:
    """``(display_name, choice)`` suggestions for a command's typed ARGUMENT.

    The same shape and the same short-query preference as
    :func:`command_suggestions`, deliberately: the two lists appear in the same
    place, are driven by the same keys and are ranked by the same scorer, so a
    user cannot tell which one they are in — and does not need to.

    The one behavioural difference is that an argument list may legitimately be
    EMPTY when the set itself is empty (``/logout`` with nothing stored), which
    is a real answer rather than a failed match. The caller distinguishes those
    two cases, because only one of them is worth saying out loud.
    """
    if not query:
        return [(choice.name, choice) for choice in choices]
    matches = match_choices(query, choices)
    if len(query) >= FUZZY_MIN_QUERY_CHARS:
        return matches
    lowered = query.lower()
    prefixed = [pair for pair in matches if pair[0].lower().startswith(lowered)]
    return prefixed or matches


def _pad_to(row: Text, width: int, style: Style) -> Text:
    """Pad ``row`` out to exactly ``width`` cells under ``style``.

    Exact width is what makes the one-row rule structural: a row that already
    fills the render width leaves Textual nothing to wrap, and a row-wide
    background tint needs the full span or the highlight reads as a ragged
    smear instead of a selected row.
    """
    missing = width - cell_len(row.plain)
    if missing > 0:
        row.append(" " * missing, style=style)
    return row


class CommandPicker(Static):
    """The suggestion list for the slash command being typed.

    Driven entirely from outside for the keyboard (the editor keeps focus and
    routes Up/Down/Tab/Enter/Esc in, so the caret never leaves the text) and
    from its own mouse handlers for click/hover. Choosing a row calls the
    ``on_choose`` callback with the display name to insert; the picker itself
    never touches the buffer.
    """

    def __init__(self, on_choose: Callable[[str], None]) -> None:
        super().__init__()
        self._on_choose = on_choose
        self._commands: list[SlashCommand] = []
        self._choices: list[ArgumentChoice] = []
        self._mode = PickerMode.COMMAND
        self._matches: list[_Suggestion] = []
        self._selected = 0
        self._window_start = 0
        self._hovered: int | None = None
        self._query = ""
        self._dismissed_query: str | None = None
        # Set by an arrow press, cleared whenever the candidate set changes: the
        # difference between "the matcher put the highlight here" and "I moved it
        # here", which is what the editor's Enter gate needs to know.
        self._chosen_by_hand = False
        # Closed picker takes no layout space at all — `visible: hidden` would
        # still reserve the rows and leave a hole above the status band.
        self.display = False

    # -- public API ---------------------------------------------------------
    def set_commands(self, commands: list[SlashCommand]) -> None:
        """Replace the offered command registry."""
        self._commands = list(commands)

    def set_choices(self, choices: list[ArgumentChoice]) -> None:
        """Replace the values offered for the current command's ARGUMENT.

        Re-derives the visible rows immediately, because the app fills these in
        answer to a posted message — one message-loop tick after the keystroke
        that opened the list. Without the resync the picker would sit closed on
        the empty set it was opened with until the user typed another character.
        """
        self._choices = list(choices)
        if self._mode is PickerMode.ARGUMENT:
            matches = argument_suggestions(self._query, self._choices)
            self._apply(PickerMode.ARGUMENT, self._query, matches)

    @property
    def mode(self) -> PickerMode:
        """Whether the rows are commands or one command's argument values."""
        return self._mode

    def is_open(self) -> bool:
        """True when suggestions are showing."""
        return bool(self._matches)

    def suggestions(self) -> list[_Suggestion]:
        """All current matches, best first (not just the visible window)."""
        return list(self._matches)

    def highlighted_name(self) -> str | None:
        """Display name of the highlighted row, or ``None`` when closed."""
        if not self._matches:
            return None
        return self._matches[self._selected][0]

    @property
    def selected_index(self) -> int:
        """Index of the highlighted row within :meth:`suggestions`."""
        return self._selected

    @property
    def chosen_by_hand(self) -> bool:
        """True when the user arrowed onto the current row themselves."""
        return self._chosen_by_hand

    @property
    def hovered_index(self) -> int | None:
        """Index under the mouse, or ``None``."""
        return self._hovered

    def visible_window(self) -> tuple[int, int, int]:
        """``(start, end, total)`` — which suggestions the rows are showing.

        Exposed because the rendered rows alone cannot tell a caller whether
        anything is hidden, and "the list is longer than it looks" is exactly
        what a capped picker must be able to answer.
        """
        total = len(self._matches)
        end = min(total, self._window_start + self._row_budget())
        return self._window_start, end, total

    def sync(self, text: str) -> None:
        """Re-derive the COMMAND suggestions from the editor's current ``text``."""
        context = slash_context(text)
        if context is None:
            # Left slash context entirely: forget the dismissal, so the next
            # `/` opens a fresh picker.
            self._dismissed_query = None
            self._mode = PickerMode.COMMAND
            self._close()
            return
        matches = command_suggestions(context.query, self._commands)
        self._apply(PickerMode.COMMAND, context.query, matches)

    def sync_argument(self, query: str) -> None:
        """Re-derive the ARGUMENT suggestions for the current command.

        The editor calls this INSTEAD of :meth:`sync` while the buffer holds a
        command whose argument drives a list, so the two can never both be
        showing: which list is up is a property of the buffer parse, not of two
        widgets agreeing to take turns.
        """
        self._apply(PickerMode.ARGUMENT, query, argument_suggestions(query, self._choices))

    def _apply(self, mode: PickerMode, query: str, matches: Sequence[_Suggestion]) -> None:
        """Adopt a freshly derived candidate set, whichever list produced it.

        ``Sequence``, not ``list``: a list is invariant, so the concrete
        ``list[tuple[str, SlashCommand]]`` the command matcher returns is not a
        ``list[_Suggestion]`` — only a read-only view of one.
        """
        if mode is not self._mode:
            # A mode change is a different list of different things. Carrying the
            # highlight, the window or Esc's "not now" across it would point them
            # at rows that no longer exist.
            self._mode = mode
            self._dismissed_query = None
            self._selected = 0
            self._window_start = 0
            self._chosen_by_hand = False
        self._query = query
        if query == self._dismissed_query:
            self._close()
            return
        # The token changed, so Esc's "not now" has expired. Latching the
        # dismissal until the slash is deleted would leave a user who pressed
        # Esc once with no way to get the list back while still typing.
        self._dismissed_query = None
        if not matches:
            self._close()
            return
        if [name for name, _ in matches] != [name for name, _ in self._matches]:
            # A different candidate set means the old highlight pointed at a
            # different command; keeping the index would silently move the
            # selection under the user's fingers. It also retires an explicit
            # choice — the row the user arrowed onto is gone.
            self._selected = 0
            self._window_start = 0
            self._chosen_by_hand = False
        self._matches = list(matches)
        self.display = True
        self._scroll_to_selection()
        self._repaint()

    def move(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows, wrapping at both ends."""
        if not self._matches:
            return
        self._selected = (self._selected + delta) % len(self._matches)
        # An arrow press is the user reading the list and picking a row, which
        # is the whole of what the ambiguity check is worried about. Recording it
        # is what lets Enter send on the first press after a deliberate move
        # while still requiring two on a word the matcher chose alone.
        self._chosen_by_hand = True
        self._scroll_to_selection()
        self._repaint()

    def dismiss(self) -> None:
        """Hide the picker for the CURRENT word without touching the text."""
        self._dismissed_query = self._query
        self._close()

    def close(self) -> None:
        """Hide the picker (submission, completion — not a dismissal)."""
        self._close()

    def choose(self, index: int) -> None:
        """Highlight ``index`` and hand its command to the editor."""
        if not 0 <= index < len(self._matches):
            return
        self._selected = index
        self._on_choose(self._matches[index][0])

    # -- mouse --------------------------------------------------------------
    # Public handler names on purpose: Textual dispatches `_on_<event>` and
    # then `on_<event>`, so the base Widget keeps its own click/leave
    # bookkeeping (`mouse_hover`, which drives every `:hover` rule) instead of
    # being shadowed by an override that would silently latch it on.
    def on_click(self, event: events.Click) -> None:
        # Stop the click here: the input dock below is not a click target, and
        # letting it bubble hands the event to a parent mid-completion.
        event.stop()
        index = self._index_at(event.y)
        if index is not None:
            self.choose(index)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        index = self._index_at(event.y)
        if index != self._hovered:
            self._hovered = index
            self._repaint()

    def on_leave(self, event: events.Leave) -> None:
        if self._hovered is not None:
            self._hovered = None
            self._repaint()

    def on_resize(self, event: events.Resize) -> None:
        """Re-truncate every row against the new width."""
        if self._matches:
            self._repaint()

    # -- rendering ----------------------------------------------------------
    def render_rows(self, width: int) -> list[Text]:
        """One row per VISIBLE suggestion, each exactly ``width`` cells."""
        start, end, _total = self.visible_window()
        return [self._row(index, width) for index in range(start, end)]

    def render_text(self, width: int) -> Text:
        """The full renderable: the visible rows plus the overflow marker."""
        rows = self.render_rows(width)
        overflow = self._overflow_row(width)
        if overflow is not None:
            rows.append(overflow)
        out = Text()
        for index, row in enumerate(rows):
            if index:
                out.append("\n")
            out.append_text(row)
        return out

    def _repaint(self) -> None:
        # Matching, selection and the visible window are all resolved without
        # a screen, so the state machine is fully exercisable (and testable)
        # off-app; only PAINTING needs one, because Static.update has to reach
        # the app console to build its visual.
        if not self._matches or not self.is_mounted:
            return
        width = max(self.size.width, _MIN_RENDER_WIDTH)
        rows = self.render_rows(width)
        overflow = self._overflow_row(width)
        row_count = len(rows) + (0 if overflow is None else 1)
        # Pin the height: `auto` would measure content before layout knows the
        # real width and settle one row too tall per suggestion, exactly the
        # trap ToolCard documents.
        self.styles.height = row_count
        self.update(self.render_text(width))

    def _row(self, index: int, width: int) -> Text:
        """The row for suggestion ``index``, dispatched on what it stands for."""
        name, item = self._matches[index]
        styles = self._row_styles(index)
        if isinstance(item, ArgumentChoice):
            return self._argument_row(name, item, width, styles)
        return self._command_row(name, item, width, styles)

    def _row_styles(self, index: int) -> _RowStyles:
        """Ground and text styles for row ``index`` — shared by both kinds."""
        selected = index == self._selected
        hovered = index == self._hovered

        # ONE green: the accent is spent on the highlighted command NAME.
        #
        # Selection is carried by HUE, not elevation. Pure luminance steps could
        # not do it — surface->raised measures 1.096:1 and surface->overlay
        # 1.218:1, both imperceptible — so the highlight rested entirely on the
        # accent and hover (which has no accent) gave a mouse user almost no
        # feedback about which row a click would run. `tint-select` is the same
        # move `tint-danger` already makes on a failed tool row: elevation says
        # "this is a row", hue says "this is its state" (D8).
        # Hover is ADDITIVE and selection stays dominant. Written the other way
        # round — hover overwriting the ground — pointing at the selected row
        # swapped its clearly-tinted ground for the faintest step in the ramp, so
        # the highlight vanished under the pointer and the row read as LESS
        # selected than its neighbours. A mouse user arrowing to a row and then
        # reaching for the mouse watched the picker appear to lose its place.
        ground = theme_mod.semantic_color("surface")
        if hovered:
            # `overlay`, not `raised`: raised measures dE2000 3.06 against
            # surface, which is the very step the comment above rejects as
            # imperceptible. Every row here is a click target that RUNS a command
            # and some of them are destructive, so "which row will this click
            # hit" has to be answerable.
            ground = theme_mod.semantic_color("overlay")
        if selected:
            ground = theme_mod.semantic_color("tint-select-hi" if hovered else "tint-select")
        row_bg = Style(bgcolor=ground)
        name_style = row_bg + Style(color=theme_mod.semantic_color("accent" if selected else "fg"))
        # `dim`, not `faint`. An alias is a typeable command, and the picker is
        # where a user DISCOVERS that `/quit` and `/models` exist — at `faint`
        # that discovery rendered at 1.7:1 against its own ground, so the row
        # promised two names and hid one. `faint` stays what it is: chrome, for
        # the band's separators (D2).
        #
        # One step up on the selected row: `dim` over the green-tinted ground
        # falls to 3.97:1, just under AA, and the selected row is the one the user
        # is actually reading. The three-tier hierarchy holds everywhere else.
        alias_style = row_bg + Style(color=theme_mod.semantic_color("muted" if selected else "dim"))
        description_style = row_bg + Style(color=theme_mod.semantic_color("muted"))
        # The cursor is MUTED, not the accent name style: the input's focused
        # chevron is already accent at the same column on the adjacent row, so
        # two identical green chevrons a row apart read as a duplicated caret
        # exactly when the user is mid-keystroke (D17).
        cursor_style = row_bg + Style(color=theme_mod.semantic_color("muted"))
        return _RowStyles(
            selected=selected,
            ground=row_bg,
            name=name_style,
            alias=alias_style,
            description=description_style,
            cursor=cursor_style,
        )

    def _gutter(self, styles: _RowStyles) -> Text:
        row = Text()
        mark = f"{_CURSOR} " if styles.selected else " " * _GUTTER_CELLS
        row.append(mark, style=styles.cursor)
        return row

    def _command_row(self, name: str, command: SlashCommand, width: int, s: _RowStyles) -> Text:
        row = self._gutter(s)
        row_bg = s.ground

        primary = f"/{name}"
        aliases = tuple(other for other in command.names if other != name)
        alias_run = "  " + " ".join(f"/{alias}" for alias in aliases) if aliases else ""

        description = command.description.strip()
        if description and width > DESCRIPTION_COLLAPSE_WIDTH:
            column = max(1, min(self._primary_column(), width - _GUTTER_CELLS - _EDGE_MARGIN * 2))
            budget = max(1, column - _PRIMARY_COLUMN_GAP)
            used = self._append_primary(row, primary, alias_run, budget, s.name, s.alias)
            gap = max(_PRIMARY_COLUMN_GAP, column - used)
            row.append(" " * gap, style=row_bg)
            remaining = width - _GUTTER_CELLS - used - gap - _EDGE_MARGIN
            if remaining > _MIN_DESCRIPTION_CELLS:
                row.append(truncate_cells(description, remaining), style=s.description)
                return _pad_to(row, width, row_bg)
            # Not enough room after the name column to say anything useful:
            # rebuild as a name-only row rather than ship a stub description.
            row = self._gutter(s)

        budget = max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)
        self._append_primary(row, primary, alias_run, budget, s.name, s.alias)
        return _pad_to(row, width, row_bg)

    def _argument_row(self, name: str, choice: ArgumentChoice, width: int, s: _RowStyles) -> Text:
        """``name  description                     detail`` — no leading slash.

        The slash is COMMAND vocabulary. Prefixing an argument with it would read
        as `/login /anthropic`, which is not something the user can type. Aliases
        are absent for the same reason: `claude` makes anthropic FINDABLE, but the
        only text that completes into the buffer is the provider id, so listing
        the alias would advertise input the command does not accept.
        """
        row = self._gutter(s)
        row_bg = s.ground
        # `danger` only when the state is a problem; an unfinished login is not
        # one. Tinting every un-logged-in provider red would make the ordinary
        # `/login` list read as a wall of failures.
        detail_style = row_bg + Style(
            color=theme_mod.semantic_color("danger" if choice.alert else "muted")
        )

        span = max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)
        detail = choice.detail.strip()
        # Reserve `detail` BEFORE the description, and drop the description first
        # when only one of the two fits: for `/logout` the state is the thing being
        # chosen by ("which of these am I logged into"), while the description only
        # restates a provider name the id already carries.
        reserved = cell_len(detail) + _PRIMARY_COLUMN_GAP if detail else 0
        if reserved and span - reserved < _MIN_NAME_CELLS:
            detail = ""
            reserved = 0
        body = span - reserved

        description = choice.description.strip()
        if description and width > DESCRIPTION_COLLAPSE_WIDTH:
            column = max(1, min(self._primary_column(), body))
            clipped = truncate_cells(name, max(1, column - _PRIMARY_COLUMN_GAP))
            row.append(clipped, style=s.name)
            used = cell_len(clipped)
            gap = max(_PRIMARY_COLUMN_GAP, column - used)
            remaining = body - used - gap
            if remaining > _MIN_DESCRIPTION_CELLS:
                row.append(" " * gap, style=row_bg)
                row.append(truncate_cells(description, remaining), style=s.description)
                return self._append_detail(row, width, detail, detail_style, row_bg)
            # Not enough room after the name column to say anything useful:
            # rebuild as a name-only row rather than ship a stub description.
            row = self._gutter(s)

        row.append(truncate_cells(name, max(1, body)), style=s.name)
        return self._append_detail(row, width, detail, detail_style, row_bg)

    def _append_detail(
        self, row: Text, width: int, detail: str, detail_style: Style, row_bg: Style
    ) -> Text:
        """Right-align ``detail`` at the row's trailing edge and pad to ``width``.

        Its cells were reserved out of the body budget above, so padding to the
        detail's own start can only ever ADD space — never truncate content that
        has already been appended.
        """
        if detail:
            _pad_to(row, width - _EDGE_MARGIN - cell_len(detail), row_bg)
            row.append(detail, style=detail_style)
        return _pad_to(row, width, row_bg)

    def _append_primary(
        self,
        row: Text,
        primary: str,
        alias_run: str,
        budget: int,
        name_style: Style,
        alias_style: Style,
    ) -> int:
        """Append the name (plus aliases when they fit); return cells used.

        Aliases are all-or-nothing: half an alias list is worse than none,
        and the name is the part being chosen, so it gets the whole budget
        when the two cannot both fit.
        """
        if alias_run and cell_len(primary) + cell_len(alias_run) <= budget:
            row.append(primary, style=name_style)
            row.append(alias_run, style=alias_style)
            return cell_len(primary) + cell_len(alias_run)
        clipped = truncate_cells(primary, budget)
        row.append(clipped, style=name_style)
        return cell_len(clipped)

    def _overflow_row(self, width: int) -> Text | None:
        start, end, total = self.visible_window()
        hidden = total - (end - start)
        if hidden <= 0:
            return None
        dim = Style(
            color=theme_mod.semantic_color("dim"),
            bgcolor=theme_mod.semantic_color("surface"),
        )
        row = Text()
        row.append(" " * _GUTTER_CELLS, style=dim)
        row.append(
            truncate_cells(f"… {hidden} more", max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)),
            style=dim,
        )
        return _pad_to(row, width, dim)

    def _primary_column(self) -> int:
        """Fit-to-content name column, clamped to the 12..32 cell band."""
        widest = 0
        for name, item in self._matches:
            if isinstance(item, ArgumentChoice):
                # No slash and no alias run: an argument row's primary column is
                # the bare value, which is all that is typeable.
                cells = cell_len(name)
            else:
                aliases = tuple(other for other in item.names if other != name)
                cells = cell_len(f"/{name}")
                if aliases:
                    cells += cell_len("  " + " ".join(f"/{alias}" for alias in aliases))
            widest = max(widest, cells + _PRIMARY_COLUMN_GAP)
        return max(_PRIMARY_COLUMN_MIN, min(_PRIMARY_COLUMN_MAX, widest))

    # -- window -------------------------------------------------------------
    def _row_budget(self) -> int:
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            screen_height = 0
        if screen_height <= 0:
            return MAX_VISIBLE_ROWS
        return max(1, min(MAX_VISIBLE_ROWS, screen_height // _SCREEN_HEIGHT_DIVISOR))

    def _scroll_to_selection(self) -> None:
        budget = self._row_budget()
        if self._selected < self._window_start:
            self._window_start = self._selected
        elif self._selected >= self._window_start + budget:
            self._window_start = self._selected - budget + 1
        self._window_start = max(0, min(self._window_start, max(0, len(self._matches) - budget)))

    def _index_at(self, y: int) -> int | None:
        """Suggestion index at content row ``y``, or ``None``.

        Returns ``None`` for the overflow marker row: it is a count, not a
        choice, and clicking a count must not run a command.
        """
        start, end, _total = self.visible_window()
        index = self._window_start + y
        if not start <= index < end:
            return None
        return index

    def _close(self) -> None:
        self._matches = []
        self._selected = 0
        self._window_start = 0
        self._hovered = None
        self.display = False
