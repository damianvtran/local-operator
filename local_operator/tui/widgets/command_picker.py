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
      /mcp               List MCP servers (login/logout/reauth <name> to manage OAuth)

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

#: An ARGUMENT list gets its own, larger budget: half the screen, less the rows
#: that are never the list's to take.
#:
#: ``MAX_VISIBLE_ROWS`` is reasoned from the COMMAND list, where every row is a
#: described one-liner the user READS. An argument list is a set they SCAN, and
#: `/login` is the one surface whose entire job is answering "what is
#: supported" — capping the twelve providers at eight hid a third of the answer
#: (openrouter, which this app's catalogue is built around, among it) while
#: seven rows of the region above the list sat empty. The splash degrades to
#: make the room, which is the right trade for as long as the list is open.
#:
#: The three subtracted rows are the ones BELOW the list inside the screen box
#: (``Screen.size`` already excludes the app's one-cell edge padding): the
#: prompt row the picker hangs off, the status band, and the blank line between
#: them. At the 28-row default this resolves to 10 of the 12 providers, and at
#: 20 rows to 6.
_ARGUMENT_HEIGHT_DIVISOR = 2
_ARGUMENT_CHROME_ROWS = 3

#: Floor for the argument budget on a short terminal, matching the floor the
#: command list clamps to.
_ARGUMENT_ROWS_MIN = 3

#: The selection mark. The app's prompt and user blocks already speak ``❯``
#: (SPINE_INDENT is 2 cells for exactly this reason); the picker reuses that
#: vocabulary rather than introducing a second cursor glyph.
_CURSOR = "❯"

#: Gutter width. THREE, not two: the prompt occupies ``❯`` plus a space and the
#: editor's own text starts in the third cell, so a two-cell gutter left every
#: suggestion one cell to the LEFT of the text it completes into — while the
#: tcss beside it claimed the two columns agreed. On the boot card, where the
#: prompt rail is the only structure on screen, that one cell is the whole
#: composition. The cursor still lands in the gutter's first cell, directly
#: under the prompt chevron.
_GUTTER_CELLS = 3

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

#: FLOOR for the NAME column of an argument row before its ``detail`` is
#: dropped — the minimum, not the answer: see :meth:`CommandPicker._name_floor`,
#: which raises it to the widest id actually offered. The name is the text Tab
#: types into the buffer, so a truncated one is unusable — the user cannot read
#: what to complete to. ``detail`` is worth a lot (at `/logout` it names the
#: credential being removed) but never worth that.
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

    def __init__(
        self,
        on_choose: Callable[[str], None],
        on_highlight: Callable[[str | None], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_choose = on_choose
        #: Observer for the row the user is CONSIDERING — the hover target when
        #: the mouse is over a row, else the keyboard highlight — called with
        #: ``None`` when an argument list stops showing rows. It exists for
        #: live preview (``/theme``): the preview has to track what the eye is
        #: on, which is not always what Enter would choose. Only ARGUMENT rows
        #: report; a command-word list has nothing to preview.
        self._on_highlight = on_highlight
        #: Last name reported to ``_on_highlight``, so the observer hears each
        #: change once — mouse-move events arrive per cell, not per row.
        self._reported_highlight: str | None = None
        #: True only while ``set_choices`` seeds a highlight: the interim
        #: row-0 state must not reach the observer (see ``set_choices``).
        self._suppress_report = False
        self._commands: list[SlashCommand] = []
        self._choices: list[ArgumentChoice] = []
        self._mode = PickerMode.COMMAND
        self._matches: list[_Suggestion] = []
        self._selected = 0
        self._window_start = 0
        self._hovered: int | None = None
        self._query = ""
        self._dismissed_query: str | None = None
        # Set when an ARGUMENT list has nothing to offer AND that is worth saying.
        # Not a match: it is never selectable, so it lives beside the rows rather
        # than among them (see set_notice).
        self._notice = ""
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

    def set_choices(self, choices: list[ArgumentChoice], highlight: str | None = None) -> None:
        """Replace the values offered for the current command's ARGUMENT.

        Re-derives the visible rows immediately, because the app fills these in
        answer to a posted message — one message-loop tick after the keystroke
        that opened the list. Without the resync the picker would sit closed on
        the empty set it was opened with until the user typed another character.

        ``highlight`` seeds the selection onto the named row when the list
        opens bare (empty query, nothing chosen by hand). It exists for lists
        where the highlight has a SIDE EFFECT: ``/theme`` previews the
        highlighted row live, so a list that opened on row 0 flashed every
        non-default user to the default theme before they touched a key
        (review round 1, F2). Seeding the row where the user already IS makes
        the first report a no-op — and is where a browse should start anyway.
        """
        self._choices = list(choices)
        if self._mode is PickerMode.ARGUMENT:
            matches = argument_suggestions(self._query, self._choices)
            seeding = highlight is not None and not self._query and not self._chosen_by_hand
            if seeding:
                # Silence `_apply`'s own report: it fires for row 0 before the
                # seed lands, and for a previewing list that one report IS the
                # flash — the observer would try row 0 on and take it off
                # again one message later.
                self._suppress_report = True
            try:
                self._apply(PickerMode.ARGUMENT, self._query, matches)
            finally:
                self._suppress_report = False
            if seeding:
                names = [name for name, _ in self._matches]
                if highlight in names and names.index(highlight) != self._selected:
                    self._selected = names.index(highlight)
                    self._scroll_to_selection()
                    self._repaint()
                self._report_highlight()

    def set_notice(self, text: str) -> None:
        """Say why an ARGUMENT list is empty, IN THE LIST'S OWN PLACE.

        One dim row where the rows would have been, in the overflow marker's
        vocabulary. The alternative — reporting it into the transcript — repeats
        without bound: the message answers a UI event, so every re-entry into the
        argument state (type `/logout `, backspace, space again) appended another
        identical line to what is supposed to be a record of the conversation.
        Said here it is in the user's eye-line, self-clearing, unrepeatable, and it
        costs the transcript nothing.

        NOT a match. ``_matches`` stays empty, so ``is_open()`` is False,
        ``_index_at`` returns None for the row — a click or a hover cannot action
        it, exactly as for the overflow count — and every key the editor routes at
        an open picker still goes to the buffer. Passing ``""`` withdraws it.
        """
        text = text.strip()
        if text == self._notice:
            return
        self._notice = text
        if self._matches:
            # Rows are showing: they answer the question the notice would.
            return
        if text:
            self.display = True
            self._repaint()
        else:
            self._close()

    @property
    def mode(self) -> PickerMode:
        """Whether the rows are commands or one command's argument values."""
        return self._mode

    def is_open(self) -> bool:
        """True when suggestions are showing."""
        return bool(self._matches)

    def is_pending(self) -> bool:
        """True for an ARGUMENT list that is open in principle but has no rows yet.

        The app fills an argument list in answer to a posted message, so for one
        message-loop tick the picker is in argument mode holding nothing —
        showing as closed while being, from the user's point of view, a list they
        just opened. A key that only reaches an ``is_open()`` picker is silently
        dropped in that window.

        False once :meth:`dismiss` has recorded the query, so a dismissed list
        stops swallowing the key that dismissed it.
        """
        return (
            self._mode is PickerMode.ARGUMENT
            and not self._matches
            and self._dismissed_query is None
        )

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
            # at rows that no longer exist. The notice goes with them: it was about
            # THAT list.
            self._mode = mode
            self._dismissed_query = None
            self._selected = 0
            self._window_start = 0
            self._chosen_by_hand = False
            self._notice = ""
        self._query = query
        if query == self._dismissed_query:
            self._close()
            return
        # The token changed, so Esc's "not now" has expired. Latching the
        # dismissal until the slash is deleted would leave a user who pressed
        # Esc once with no way to get the list back while still typing.
        self._dismissed_query = None
        if not matches:
            if mode is PickerMode.ARGUMENT and self._notice:
                # No rows, but something to say in their place. The list stays up
                # holding the one informational row, and holds it across every
                # re-derivation — the user editing the argument of a command with
                # nothing to offer does not make the answer any less true.
                self._reset_rows()
                self.display = True
                self._repaint()
                return
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
        self._report_highlight()

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
        self._report_highlight()

    def scroll_rows(self, delta: int) -> None:
        """Move the highlight by ``delta`` rows for a WHEEL notch, clamped.

        Deliberately not :meth:`move`: that wraps, which suits a discrete
        arrow press and not a scroll gesture — a wheel that teleports from the
        last row back to the first reads as the menu having reset itself.

        ``_chosen_by_hand`` is set for the same reason :meth:`move` sets it: a
        wheel notch is the user reading the list and landing on a row, which
        is exactly the deliberate choice the ambiguity check looks for.
        """
        if not self._matches:
            return
        target = max(0, min(len(self._matches) - 1, self._selected + delta))
        if target == self._selected:
            return
        self._selected = target
        self._chosen_by_hand = True
        self._scroll_to_selection()
        self._repaint()
        self._report_highlight()

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

    # Stopped for the same reason the click is: the menu floats over the
    # transcript, so a wheel left to bubble scrolls the conversation behind
    # it as well — two surfaces moving for one gesture.
    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        event.stop()
        self.scroll_rows(1)

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        self.scroll_rows(-1)

    def on_mouse_move(self, event: events.MouseMove) -> None:
        index = self._index_at(event.y)
        if index != self._hovered:
            self._hovered = index
            self._repaint()
            self._report_highlight()
        # The hand pointer only over a ROW: the picker's padding and notice
        # rows are not click targets, and a static `pointer` rule on the
        # widget would promise the click the empty rows cannot keep. Setting
        # the inline rule is what makes the shape follow the hover — the
        # property's own observer re-runs `Screen.update_pointer_shape()`,
        # and no-ops when the value did not change.
        self.styles.pointer = "pointer" if index is not None else "default"

    def on_leave(self, event: events.Leave) -> None:
        if self._hovered is not None:
            self._hovered = None
            self._repaint()
            self._report_highlight()
        self.styles.pointer = "default"

    def on_resize(self, event: events.Resize) -> None:
        """Re-truncate every row against the new width."""
        if self._matches or self._notice:
            self._repaint()

    # -- rendering ----------------------------------------------------------
    def render_rows(self, width: int) -> list[Text]:
        """One row per VISIBLE suggestion, each exactly ``width`` cells."""
        start, end, _total = self.visible_window()
        return [self._row(index, width) for index in range(start, end)]

    def render_text(self, width: int) -> Text:
        """The full renderable: the visible rows plus the overflow marker.

        With no rows at all it is the informational row, or nothing — the two
        states the picker can be VISIBLE in without a single suggestion.
        """
        if not self._matches:
            return self._notice_row(width) if self._notice else Text()
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
        if not self.is_mounted or not (self._matches or self._notice):
            return
        width = max(self.size.width, _MIN_RENDER_WIDTH)
        if not self._matches:
            # The informational row stands alone: one row, no window and no
            # overflow count, because there is nothing to count.
            self.styles.height = 1
            self.update(self._notice_row(width))
            return
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
        # Padded from the constant rather than written out: a hard-coded two-cell
        # mark under a three-cell gutter would shift only the SELECTED row, which
        # is the one row a misalignment is guaranteed to be noticed on.
        row = Text()
        mark = (_CURSOR if styles.selected else "").ljust(_GUTTER_CELLS)
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
        # The NAME carries the danger too: on `/logout`-style lists the detail
        # column may hold an innocuous state word ("connected") or nothing at
        # all, so tinting only the detail could leave a destructive row
        # visually identical to a benign one — which is what happened on the
        # `/mcp logout` list. The name is the cell every destructive row has.
        name_style = s.name
        if choice.alert and not s.selected:
            name_style = row_bg + Style(color=theme_mod.semantic_color("danger"))

        span = max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)
        detail = choice.detail.strip()
        # Reserve the widest detail in the MATCH SET, not this row's own. The
        # state is a column, and a column the user can scan has one left edge;
        # right-aligning each string to its own row's trailing edge started the
        # three credential states at three different x, so "which of these am I
        # logged into" meant reading eight strings instead of scanning an edge.
        #
        # Reserved BEFORE the description, and the description dropped first when
        # only one of the two fits: at `/logout` the detail names the credential
        # about to be REMOVED, which no other column on the row says.
        column_cells = self._detail_column()
        reserved = column_cells + _PRIMARY_COLUMN_GAP if column_cells else 0
        if reserved and span - reserved < self._name_floor():
            # Uniform by construction: every row reserves the same width against
            # the same floor, so the column is dropped for the whole list at once
            # and can never leave a ragged half of one behind.
            column_cells = 0
            reserved = 0
        body = span - reserved

        description = choice.description.strip()
        if description and width > DESCRIPTION_COLLAPSE_WIDTH:
            column = max(1, min(self._primary_column(), body))
            clipped = truncate_cells(name, max(1, column - _PRIMARY_COLUMN_GAP))
            row.append(clipped, style=name_style)
            used = cell_len(clipped)
            gap = max(_PRIMARY_COLUMN_GAP, column - used)
            remaining = body - used - gap
            if remaining > _MIN_DESCRIPTION_CELLS:
                row.append(" " * gap, style=row_bg)
                row.append(truncate_cells(description, remaining), style=s.description)
                return self._append_detail(row, width, detail, column_cells, detail_style, row_bg)
            # Not enough room after the name column to say anything useful:
            # rebuild as a name-only row rather than ship a stub description.
            row = self._gutter(s)

        row.append(truncate_cells(name, max(1, body)), style=name_style)
        return self._append_detail(row, width, detail, column_cells, detail_style, row_bg)

    def _append_detail(
        self,
        row: Text,
        width: int,
        detail: str,
        column_cells: int,
        detail_style: Style,
        row_bg: Style,
    ) -> Text:
        """Left-align ``detail`` inside the reserved column and pad to ``width``.

        The COLUMN is right-aligned to the row's trailing edge; the string inside
        it is not, so every row's detail begins at the same x. Its cells were
        reserved out of the body budget above, so padding to the column's start
        can only ever ADD space — never truncate content already appended.
        """
        if column_cells and detail:
            _pad_to(row, width - _EDGE_MARGIN - column_cells, row_bg)
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

    def _notice_row(self, width: int) -> Text:
        """The informational row: why this list is empty, in the marker's voice.

        Deliberately the overflow marker's exact treatment — dim on the dock's own
        surface, text starting at the name column — because the two say the same
        KIND of thing: a fact about the list rather than a row in it. A second
        style here would advertise it as something the user can act on.
        """
        dim = Style(
            color=theme_mod.semantic_color("dim"),
            bgcolor=theme_mod.semantic_color("surface"),
        )
        row = Text()
        row.append(" " * _GUTTER_CELLS, style=dim)
        row.append(
            truncate_cells(self._notice, max(1, width - _GUTTER_CELLS - _EDGE_MARGIN)),
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

    def _detail_column(self) -> int:
        """Fit-to-content DETAIL column: the widest detail in the match set.

        Unclamped, unlike :meth:`_primary_column`. The detail is generated by the
        app from a closed vocabulary (three credential states, two credential
        kinds), not typed by a user, and the row already drops the column whole
        when reserving it would squeeze the name past :meth:`_name_floor`.
        """
        widest = 0
        for _name, item in self._matches:
            if isinstance(item, ArgumentChoice):
                widest = max(widest, cell_len(item.detail.strip()))
        return widest

    def _name_floor(self) -> int:
        """Cells the NAME column keeps before ``detail`` is dropped.

        The widest id ACTUALLY offered, not a constant. A fixed floor of twelve
        answers "would a twelve-cell name fit" for a list whose longest name is
        thirteen, which is how `openai-device` rendered as `openai-devi…` beside
        an intact `needs login` at one exact render width — the detail column
        keeping cells from the only text on the row the user can type.

        Clamped at ``_PRIMARY_COLUMN_MAX``: past that the name column truncates
        anyway, so letting the floor run away would drop the detail at every
        width and buy nothing.
        """
        widest = max((cell_len(name) for name, _ in self._matches), default=0)
        return max(_MIN_NAME_CELLS, min(_PRIMARY_COLUMN_MAX, widest))

    # -- window -------------------------------------------------------------
    def _row_budget(self) -> int:
        try:
            screen_height = self.screen.size.height
        except NoScreen:
            screen_height = 0
        if screen_height <= 0:
            return MAX_VISIBLE_ROWS
        if self._mode is PickerMode.ARGUMENT:
            # The screen-height guard stays — a picker that squeezed the
            # transcript to nothing is the failure this whole method exists for —
            # but the ceiling does not: see ``_ARGUMENT_HEIGHT_DIVISOR``.
            return max(
                _ARGUMENT_ROWS_MIN,
                screen_height // _ARGUMENT_HEIGHT_DIVISOR - _ARGUMENT_CHROME_ROWS,
            )
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

        Returns ``None`` for the overflow marker row and for the informational row
        of an empty list: both are facts about the list, not choices in it, and
        clicking a fact must not run a command. The informational case falls out of
        the window being empty — there is no index for any ``y``.
        """
        start, end, _total = self.visible_window()
        index = self._window_start + y
        if not start <= index < end:
            return None
        return index

    def _report_highlight(self) -> None:
        """Tell the observer which ARGUMENT row the user's eye is on now.

        The reported row is the HOVER target when the pointer is over one,
        else the keyboard highlight — the same precedence the row grounds
        paint, so what previews is always the row that reads as active.
        De-duplicated on the name: a mouse crossing five cells of one row and
        a repaint that reproduced the same set both say nothing new.
        """
        if self._on_highlight is None or self._suppress_report:
            return
        name: str | None = None
        if self._mode is PickerMode.ARGUMENT and self._matches:
            index = self._hovered if self._hovered is not None else self._selected
            if 0 <= index < len(self._matches):
                name = self._matches[index][0]
        if name != self._reported_highlight:
            self._reported_highlight = name
            self._on_highlight(name)

    def _reset_rows(self) -> None:
        """Drop every row and the state that pointed into them, but not the
        notice: a list can be showing its informational row with no rows at all."""
        self._matches = []
        self._selected = 0
        self._window_start = 0
        self._hovered = None

    def _close(self) -> None:
        self._reset_rows()
        # Release the hand BEFORE the surface disappears. Textual only
        # re-evaluates the pointer on a mouse/style event; removing a picker
        # under a stationary pointer otherwise leaves OSC 22 at `pointer`
        # until the person moves again.
        self.styles.pointer = "default"
        # The notice belonged to the list that is now gone. Esc, a completion and a
        # submission all arrive here, and each one is the user done with it.
        self._notice = ""
        self.display = False
        self._report_highlight()
