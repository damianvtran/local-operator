"""The ``ask`` tool's picker: the agent's question as a real answerable list.

Why this exists: a model that needs a decision has, until now, had exactly one
shape available to it — prose. Observed verbatim in a live session:

    (A) Drop email … (B) Escalate it properly … (C) You have context I don't

Three options printed into the transcript, none of them clickable, none of them
selectable, and the user's answer arriving as free text the agent then has to
re-parse. This screen is the other half of the ``ask`` tool: one question at a
time, keyboard and mouse, answering with the label the model wrote.

It is built on the ``/resume`` picker's frame (``ModalScreen`` + one ``Static``,
content-sized, measured against the screen every paint) because that surface
already solved the parts that are easy to get wrong here: the card clipping its
own footer, the cursor sitting on a row that was never drawn, and a click on the
backdrop resolving to a row.

Two things it does that no other picker in this app does:

- **Every question offers a free-text answer.** The prose surface this replaces
  needed one constantly — "(C) You have context I don't" is a model asking for
  an answer it could not enumerate. Selecting the ``Other`` row turns it into a
  text field, so the options never have to be exhaustive.
- **It answers with TEXT, not an index.** The free-text row hands back a string
  that was never in ``options``, which an index cannot express.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.harness.types import AskQuestion
from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.tool_card import truncate_cells
from local_operator.tui.widgets.transcript import wrap_cells

#: Width the card takes when the terminal allows it, and the floor it prefers.
#: Both are CELL counts of the card's content, inside its padding, and are
#: resolved against the screen every paint (see :meth:`AskPickerScreen._card_width`).
ASK_MAX_WIDTH = 74
ASK_MIN_WIDTH = 30

#: Cells left between the card and the terminal edges, on top of the card's own
#: padding, so it reads as floating rather than bolted to the side.
ASK_WIDTH_MARGIN = 6

#: Cells of the card's own padding on EACH side, mirroring the horizontal half
#: of ``padding: 1 2`` in the stylesheet.
ASK_PADDING_CELLS = 2

#: The card's padding ROWS, mirroring the vertical half of the same rule. Kept
#: separate from the cell budget above: they are the same number by coincidence,
#: and spending one for the other is a bug waiting for the stylesheet to change.
CARD_PADDING_ROWS = 2

#: Share of the terminal the card may occupy, mirroring ``max-height: 80%``.
#: Tracked by hand because Textual clips SILENTLY — rows past the cap are simply
#: not drawn and nothing reads back that it happened.
CARD_MAX_HEIGHT_FRACTION = 0.8

#: The cursor glyph, matching the ``/resume`` and command pickers. A caret plus
#: a tinted label rather than a reversed row: an inverted block reads as a
#: selection the user made rather than the position they are on.
CURSOR = "❯"

#: Cells the cursor gutter always occupies, so labels start at one column
#: whether or not the row is the current one.
GUTTER_CELLS = 2

#: Cells the ``1``..``9`` jump number occupies, including its trailing space.
#: Rows past nine get the same indent with no number: the digits are a shortcut,
#: and re-flowing the list at ten options to reclaim two cells would move every
#: row under the cursor.
NUMBER_CELLS = 3

#: The multi-select checkbox, including its trailing space. ``[x]``/``[ ]`` and
#: not a filled glyph pair: this app already says "done/not done" that way in
#: the todo panel, and a box reads as toggleable where ``◉`` reads as decoration.
CHECK_ON = "[x] "
CHECK_OFF = "[ ] "

#: The free-text row's label while it holds nothing and is not selected.
OTHER_LABEL = "Other (type your own)"
#: Its label once it is selected or carries text; the typed string follows.
OTHER_PREFIX = "Other: "
#: The text caret drawn at the end of the field while the row is selected.
FIELD_CARET = "▌"
#: What the free-text row's second line says, so the row explains itself before
#: it is selected rather than after.
OTHER_HINT = "an answer that is not on the list — type it here"

#: The tag marking the option the model recommends. Words, not a glyph: this is
#: the one row the user is being nudged toward, and a nudge nobody can read is
#: just an unexplained difference in colour.
RECOMMENDED_TAG = "recommended"

#: Cells a label must keep for its row to say anything. Below this the row is
#: more honest showing its number alone than a one-character stub, which names a
#: category and hides the instance — on this card that means two different
#: answers painting the same text.
LABEL_MIN_CELLS = 6


@dataclass
class _QuestionState:
    """One question's live answer, kept while the user moves between questions.

    Held per question rather than reset on advance so going back over a
    multi-question ask (or re-drawing after a resize) cannot silently lose a
    selection the user already made.
    """

    selected: int = 0
    checked: set[int] = field(default_factory=set)
    typed: str = ""


class AskPickerScreen(ModalScreen["dict[str, list[str]] | None"]):
    """Put the ``ask`` tool's questions to the user; dismiss with the answers.

    Dismisses with ``question id -> chosen strings``, or ``None`` when nothing
    at all was answered. A PARTIAL mapping is deliberate: escaping out of the
    third question does not throw away the first two, because a user who
    answered and then stopped answering has still told the agent something, and
    the tool reports the rest as not answered.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("enter", "accept", "Answer", show=False),
        Binding("up", "move(-1)", "Up", show=False),
        Binding("down", "move(1)", "Down", show=False),
        # j/k as well as the arrows: this card has no filter to type into, so
        # the vi pair is free and is the movement a hand on the home row
        # reaches for. They are swallowed by :meth:`on_key` while the free-text
        # row is selected, where they are letters the user is typing.
        Binding("k", "move(-1)", "Up", show=False),
        Binding("j", "move(1)", "Down", show=False),
        Binding("ctrl+p", "move(-1)", "Up", show=False),
        Binding("ctrl+n", "move(1)", "Down", show=False),
        # ``toggle_row``, not ``toggle``: ``DOMNode.action_toggle`` already
        # exists and takes an attribute name, so an ``action_toggle(self)``
        # here would override a live Textual action with an incompatible
        # signature — the shadowing class that breaks a widget from the inside
        # and reports the traceback somewhere else entirely.
        Binding("space", "toggle_row", "Toggle", show=False),
        Binding("backspace", "backspace", "Edit answer", show=False),
        *[Binding(str(digit), f"jump({digit})", "Jump", show=False) for digit in range(1, 10)],
    ]

    def __init__(self, questions: Sequence[AskQuestion]) -> None:
        super().__init__()
        # Both the question and every label are MODEL-CONTROLLED and reach a
        # real terminal, so both are stripped on the way in — the discipline the
        # approval prompt and the tool cards already apply. A label carrying CSI
        # could erase the rows above it and repaint a forged question over them,
        # and would mis-measure every width budget below (``cell_len`` counts the
        # escape bytes) before being cut mid-sequence by truncation.
        self._questions = [_sanitize(question) for question in questions]
        self._index = 0
        self._answers: dict[str, list[str]] = {}
        self._states = [_QuestionState() for _ in self._questions]
        for state, question in zip(self._states, self._questions):
            # A recommendation is preselected as well as marked: the whole point
            # of recommending is that Enter alone should take it.
            if question.recommended is not None:
                state.selected = question.recommended
        self._offset = 0
        self._hovered: int | None = None
        #: Body-relative line index -> the row it belongs to. Recorded while the
        #: card is built rather than recomputed as arithmetic, because rows are
        #: one OR two lines tall and the header's height depends on how the
        #: question wrapped; a click resolved by arithmetic landed on the row
        #: below whenever a description wrapped or the question did not.
        self._line_rows: list[int | None] = []
        self._body: Static

    # -- state ---------------------------------------------------------------
    # ``visible_rows``/``_card_text``, not ``visible``/``_render``: both short
    # names are already Textual's (``Widget.visible``, ``Widget._render``) and
    # shadowing them breaks focus, layout or paint from inside the screen, with
    # a traceback that points somewhere else entirely.
    @property
    def question(self) -> AskQuestion:
        return self._questions[self._index]

    @property
    def state(self) -> _QuestionState:
        return self._states[self._index]

    @property
    def question_index(self) -> int:
        return self._index

    @property
    def row_count(self) -> int:
        """Options plus the free-text row, which every question carries."""
        return len(self.question.options) + 1

    @property
    def other_row(self) -> int:
        """Index of the free-text row: always last, so its number never moves."""
        return self.row_count - 1

    @property
    def selected_index(self) -> int:
        return self.state.selected

    @property
    def typed_text(self) -> str:
        return self.state.typed

    @property
    def checked_indexes(self) -> list[int]:
        return sorted(self.state.checked)

    @property
    def visible_rows(self) -> list[str]:
        """The row labels the card is currently drawing, in order.

        Named for what it is — the DRAWN window, not the whole list — because on
        a short terminal those differ, and a test that asserted the full list
        would agree with a card that clipped half of it.
        """
        window = self._window()
        return [self._row_label(index) for index in window]

    def answers_so_far(self) -> dict[str, list[str]] | None:
        """What has been answered up to now, for a host tearing the card down.

        The screen's own Escape path settles through ``dismiss``; this is for the
        app killing the picker from OUTSIDE it (a stop, a cancelled tool call,
        teardown), which has to answer the waiting tool call without going
        through the dismiss callback.
        """
        return dict(self._answers) or None

    # -- actions -------------------------------------------------------------
    def action_cancel(self) -> None:
        """Escape: stop answering. Whatever was already answered still counts."""
        self.dismiss(self._answers or None)

    def action_move(self, delta: int) -> None:
        """Arrow/vi movement WRAPS: a discrete, deliberate keypress."""
        self._move_to((self.state.selected + delta) % self.row_count)

    def action_jump(self, number: int) -> None:
        """``1``..``9`` jumps straight to a row, the free-text row included."""
        if 1 <= number <= self.row_count:
            self._move_to(number - 1)

    def action_toggle_row(self) -> None:
        """Space toggles a multi-select row; on a single-select it does nothing.

        A single-select question has exactly one answer, so a "toggle" there
        would either be Enter under another name or a selection the user cannot
        see the effect of.
        """
        if not self.question.multi or self.state.selected == self.other_row:
            return
        index = self.state.selected
        if index in self.state.checked:
            self.state.checked.discard(index)
        else:
            self.state.checked.add(index)
        self._repaint()

    def action_backspace(self) -> None:
        if self.state.selected == self.other_row and self.state.typed:
            self.state.typed = self.state.typed[:-1]
            self._repaint()

    def action_accept(self) -> None:
        """Enter: answer this question and move to the next, or dismiss.

        An empty answer is a NO-OP rather than a skip. Advancing on Enter with
        nothing chosen would record a question as asked-and-answered when the
        user had done nothing, and the model would then act on an answer nobody
        gave; Escape is how a question is left unanswered, and it says so in the
        footer.
        """
        chosen = self._chosen()
        if not chosen:
            return
        self._answers[self.question.id] = chosen
        if self._index + 1 >= len(self._questions):
            self.dismiss(self._answers)
            return
        self._index += 1
        self._offset = 0
        self._hovered = None
        self._repaint()

    def _chosen(self) -> list[str]:
        """This question's answer as text, or ``[]`` when there is none yet.

        The free-text row's content counts in BOTH modes and is appended last,
        so a multi-select answer reads in the order the card drew it.
        """
        state = self.state
        typed = state.typed.strip()
        if not self.question.multi:
            if state.selected == self.other_row:
                return [typed] if typed else []
            return [self.question.options[state.selected].label]
        labels = [
            self.question.options[index].label
            for index in sorted(state.checked)
            if index < len(self.question.options)
        ]
        if typed:
            labels.append(typed)
        return labels

    # -- keys ----------------------------------------------------------------
    def on_key(self, event) -> None:  # type: ignore[no-untyped-def]
        """Printable keys type into the free-text row while it is selected.

        Handled here rather than as bindings because the field accepts every
        character; a binding per key would be a table of ninety-five entries
        that still missed the ninety-sixth. Textual dispatches the focused
        widget's handlers before its bindings, which is what lets this take
        ``j``, ``k``, the digits and Space back from the movement keys — on this
        row they are letters the user is typing, and a field that silently
        dropped every ``j`` would be worse than no field.

        Everywhere else the key is left alone: this card has no filter, its
        rows are numbered, and swallowing letters would cost the vi movement
        for nothing.
        """
        if self.state.selected != self.other_row:
            return
        char = event.character
        if char is not None and char.isprintable() and len(char) == 1:
            event.stop()
            event.prevent_default()
            self.state.typed += char
            self._repaint()

    # -- mouse ---------------------------------------------------------------
    # The wheel moves the cursor a row at a time, which scrolls the window with
    # it. CLAMPED, unlike the arrows: a scroll gesture that wrapped to the other
    # end of the list reads as the card resetting itself. Every handler stops
    # the event so one gesture does not also scroll the transcript behind.
    def on_mouse_scroll_down(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._move_to(min(self.row_count - 1, self.state.selected + 1))

    def on_mouse_scroll_up(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._move_to(max(0, self.state.selected - 1))

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """A click on a row selects it, and on a single-select answers with it.

        Button 1 only, for the reason the ``/resume`` picker gives: this commits
        an answer the agent will act on, which is not something a right-click
        asking for a context menu or a stray middle-click paste should do.

        A multi-select click TOGGLES instead of answering — the gesture has to
        be repeatable there, and a click that confirmed the list would make the
        first pick the last one.
        """
        if getattr(event, "button", 1) != 1:
            return
        index = self._index_at(event)
        if index is None:
            return
        event.stop()
        self._move_to(index)
        if self.question.multi:
            self.action_toggle_row()
        elif index != self.other_row:
            self.action_accept()

    def on_mouse_move(self, event) -> None:  # type: ignore[no-untyped-def]
        index = self._index_at(event)
        if index != self._hovered:
            self._hovered = index
            self._repaint()

    def on_leave(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._hovered is not None:
            self._hovered = None
            self._repaint()

    def _index_at(self, event) -> int | None:  # type: ignore[no-untyped-def]
        """Row under a mouse event, or ``None`` anywhere else on the screen.

        Three guards, all load-bearing, because a false positive here answers
        the agent's question on the user's behalf:

        - the point must be inside the BODY's region — the modal's backdrop
          covers the whole screen and bubbles clicks from well outside the card,
          including columns beside it where ``y`` alone still looks valid;
        - the line must map to a row in ``_line_rows``, which is recorded while
          painting, so the header, the blank spacers and the footer resolve to
          nothing rather than to whichever row the arithmetic lands on;
        - and the row must exist in THIS question, since the map is rebuilt per
          paint and a click can race an advance.
        """
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return None
        region = body.region
        if not region.contains(event.screen_x, event.screen_y):
            return None
        line = event.screen_y - region.y
        if not 0 <= line < len(self._line_rows):
            return None
        index = self._line_rows[line]
        if index is None or not 0 <= index < self.row_count:
            return None
        return index

    # -- geometry ------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        """The box this card's ``max-width``/``max-height`` resolve in.

        ``self.size`` (this Screen's CONTENT box), not ``self.app.size`` (the
        terminal): ``Screen { padding: 1 }`` insets the content box, and the
        stylesheet's percentage cap resolves against the content box, so
        measuring the terminal asks for more rows than the container will draw
        and Textual clips the difference silently — off the bottom, taking the
        footer with it. The ``/resume`` picker and the usage panel measure the
        screen for the same reason.
        """
        try:
            size = self.size
            if not size.width or not size.height:  # not laid out yet
                size = self.app.size
        except Exception:  # pragma: no cover - only before the app has a screen
            return 80, 24
        return max(1, size.width), max(8, size.height)

    def _card_width(self) -> int:
        """Content cells the card may use, measured against the terminal.

        The preferred floor applies only while it FITS: a minimum width is a
        preference and the terminal is not, so on a very narrow screen the card
        gives up its breathing margin and then the floor rather than overflowing.
        """
        width, _ = self._screen_size()
        padding = ASK_PADDING_CELLS * 2
        room = width - ASK_WIDTH_MARGIN - padding
        if room < ASK_MIN_WIDTH:
            return max(1, width - padding)
        return min(ASK_MAX_WIDTH, room)

    def _question_lines(self, width: int) -> list[str]:
        """The question, wrapped. Never truncated: it is what is being asked.

        Wrapping makes the header's height depend on content, which is why the
        row budget below is computed from this rather than from a constant.
        """
        return wrap_cells(self.question.question, width) or [""]

    def _chrome_rows(self, width: int) -> int:
        """Rows the card spends on everything that is not an option row.

        Header, rule, the wrapped question, a blank above the rows, a blank
        below them, the position line and the footer. The position line is
        reserved UNCONDITIONALLY, even when the whole list fits: a footer that
        appeared and vanished as the window scrolled would move the card under
        the cursor.
        """
        return 2 + len(self._question_lines(width)) + 2 + 2

    def _row_budget(self) -> int:
        """Lines the option rows may actually use, after chrome is reserved.

        Chrome first and the list second, the discipline the ``/resume`` picker
        arrived at the hard way: a fixed page let the cursor sit on a row the
        card never rendered, so Enter answered with an option the user could not
        see, and let the clip eat the footer — the only statement of how to
        leave.
        """
        _, height = self._screen_size()
        budget = int(height * CARD_MAX_HEIGHT_FRACTION) - CARD_PADDING_ROWS
        return max(1, budget - self._chrome_rows(self._card_width()))

    def _show_descriptions(self) -> bool:
        """Whether every row can afford its second line.

        Descriptions are the first thing given up on a short terminal and the
        list is the last: a card that dropped ROWS to keep prose would hide
        answers the user is being asked to choose between. All or nothing, so
        the rows keep one rhythm — a list where only some entries have their
        second line reads as broken rather than as abbreviated.
        """
        return self.row_count * 2 <= self._row_budget()

    def _rows_per_page(self) -> int:
        lines_each = 2 if self._show_descriptions() else 1
        return max(1, self._row_budget() // lines_each)

    def _window(self) -> list[int]:
        """The row indexes currently drawn, after clamping the scroll offset."""
        page = self._rows_per_page()
        offset = max(0, min(self._offset, max(0, self.row_count - page)))
        self._offset = offset
        return list(range(offset, min(self.row_count, offset + page)))

    # -- internals -----------------------------------------------------------
    def _move_to(self, index: int) -> None:
        self.state.selected = max(0, min(self.row_count - 1, index))
        # Scroll only far enough to keep the cursor drawn, so the list is stable
        # while moving through the middle of it.
        page = self._rows_per_page()
        if self.state.selected < self._offset:
            self._offset = self.state.selected
        elif self.state.selected >= self._offset + page:
            self._offset = self.state.selected - page + 1
        self._repaint()

    # -- rendering -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Container(classes="ask-picker"):
            self._body = Static(self._card_text(), id="ask-picker-body")
            yield self._body

    def on_mount(self) -> None:
        self._repaint()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: the width, the page size and the descriptions all come
        from the screen."""
        self._move_to(self.state.selected)

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        body.update(self._card_text())

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads."""
        return [line.plain for line in self._card_text().split("\n")]

    def _row_label(self, index: int) -> str:
        """One row's label text, free-text row included, without styling."""
        if index != self.other_row:
            return self.question.options[index].label
        typed = self.state.typed
        if typed or self.state.selected == self.other_row:
            return f"{OTHER_PREFIX}{typed}"
        return OTHER_LABEL

    def _row_description(self, index: int) -> str:
        if index == self.other_row:
            return OTHER_HINT
        return self.question.options[index].description

    def _card_text(self) -> Text:
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        width = self._card_width()

        out = Text()
        lines: list[int | None] = []

        def newline(row: int | None) -> None:
            """Start a line that belongs to ``row`` (``None`` = chrome)."""
            if lines:
                out.append("\n")
            lines.append(row)

        newline(None)
        out.append_text(self._header(width, fg, dim))
        newline(None)
        # The raised card ground needs the raised hairline: `edge` is tuned
        # against the app background and nearly vanishes on `overlay`.
        out.append("─" * width, style=faint)
        for line in self._question_lines(width):
            newline(None)
            out.append(line, style=fg)
        newline(None)

        show_descriptions = self._show_descriptions()
        window = self._window()
        for index in window:
            ground = self._row_ground(index)
            newline(index)
            out.append_text(self._row_text(index, width, ground, fg, dim, faint))
            if show_descriptions:
                newline(index)
                out.append_text(self._description_text(index, width, ground, dim))
        newline(None)

        # The position line is EMITTED only when the list windows. Its row is
        # reserved in the height budget either way (so the card cannot start
        # clipping the moment it appears), but printing an empty line in its
        # place left two blank rows above the footer and pushed the keys away
        # from the block they belong to — visible in the first captured frame.
        if len(window) < self.row_count:
            newline(None)
            out.append("showing ", style=faint)
            out.append(f"{window[0] + 1}–{window[-1] + 1}", style=dim)
            out.append(" of ", style=faint)
            out.append(str(self.row_count), style=dim)
        newline(None)
        for position, (key, what) in enumerate(self._footer_hints(width)):
            if position:
                out.append(" · ", style=faint)
            out.append(key, style=dim)
            if what:
                out.append(f" {what}", style=faint)
        self._line_rows = lines
        return out

    def _header(self, width: int, fg: Style, dim: Style) -> Text:
        """Title on the left, ``Question n/m`` on the right when there is more
        than one.

        The counter is dropped before the title when the card is narrow: it says
        how much is left, and the title says what the card IS.
        """
        header = Text(no_wrap=True, overflow="ellipsis")
        title = "the agent needs your decision"
        counter = (
            f"Question {self._index + 1}/{len(self._questions)}" if len(self._questions) > 1 else ""
        )
        gap = 2
        if counter and cell_len(title) + gap + cell_len(counter) <= width:
            header.append(title, style=fg)
            header.append(" " * (width - cell_len(title) - cell_len(counter)))
            header.append(counter, style=dim)
        else:
            header.append(truncate_cells(title, width), style=fg)
        return header

    def _row_ground(self, index: int) -> Style:
        """The row's background: selection by HUE, hover additive on top of it.

        The same three steps the ``/resume`` picker paints on the same ``overlay``
        card, and for the reason recorded on ``tint-select`` in ``theme.py``: pure
        elevation cannot carry selection here (surface->raised measures 1.096:1),
        so a bare caret left a mouse user with almost nothing saying which row a
        click would take. Selection stays dominant under the pointer — written the
        other way round, hovering the selected row swapped its tinted ground for
        the faintest step in the ramp and the highlight vanished exactly when the
        user reached for the mouse.
        """
        hovered = index == self._hovered
        if index == self.state.selected:
            token = "tint-select-hi" if hovered else "tint-select"
        elif hovered:
            token = "tint-select"
        else:
            token = "overlay"
        return Style(bgcolor=theme_mod.semantic_color(token))

    def _row_text(
        self,
        index: int,
        width: int,
        ground: Style,
        fg: Style,
        dim: Style,
        faint: Style,
    ) -> Text:
        """One option row: cursor, number, checkbox, label, recommendation tag.

        The highlighted LABEL is the accent green — the same site the command
        picker spends it on (a picker's highlighted name), and the same meaning.
        It was the violet ``label`` token in the first captured frame, which read
        as another product's theme rather than this one's.

        The caret is ``muted``, not the accent, for the reason the command picker
        records: the accent already says "this row is the one" on the label two
        columns to the right, and a second green glyph beside it reads as a
        duplicated caret.
        """
        selected = index == self.state.selected
        accent = ground + Style(color=theme_mod.semantic_color("accent"))
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(
            f"{CURSOR} " if selected else " " * GUTTER_CELLS,
            style=ground + Style(color=theme_mod.semantic_color("muted")),
        )
        number = f"{index + 1}." if index < 9 else ""
        row.append(f"{number:<{NUMBER_CELLS}}", style=ground + (dim if selected else faint))
        spent = GUTTER_CELLS + NUMBER_CELLS
        if self.question.multi:
            checked = index in self.state.checked or (
                index == self.other_row and bool(self.state.typed.strip())
            )
            # A ticked box is the accent too: it is the same statement the
            # highlighted label makes — "this one is chosen" — and a multi-select
            # confirms several rows at once, so the ticks have to be readable
            # without moving the cursor over each of them.
            row.append(
                CHECK_ON if checked else CHECK_OFF,
                style=accent if checked else ground + faint,
            )
            spent += cell_len(CHECK_ON)

        tag = ""
        if self.question.recommended == index:
            tag = f"  · {RECOMMENDED_TAG}"
        budget = max(LABEL_MIN_CELLS, width - spent - cell_len(tag))
        text = self._row_label(index)
        if index == self.other_row and selected:
            # The typed string keeps its TAIL: a field that truncated the end
            # would hide the characters being typed, which is the only part of
            # it the user is looking at.
            budget -= cell_len(FIELD_CARET)
            row.append(OTHER_PREFIX, style=accent)
            row.append(
                _tail_cells(self.state.typed, max(1, budget - cell_len(OTHER_PREFIX))),
                style=ground + fg,
            )
            row.append(FIELD_CARET, style=accent)
        else:
            row.append(truncate_cells(text, budget), style=accent if selected else ground + fg)
        if tag:
            row.append(tag, style=ground + dim)
        return _pad_row(row, width, ground)

    def _description_text(self, index: int, width: int, ground: Style, ink: Style) -> Text:
        """The row's second line, indented under its label.

        Drawn at ``dim`` and NOT at ``faint`` like the footer's grammar words:
        this line is the CONSEQUENCE of choosing the row, which is what the user
        is reading the card to compare. The first captured frame had it at
        ``faint`` on the ``overlay`` ground, where "nothing in the app reads it
        any more" was barely present — the same contrast failure the approval
        prompt's key hints were fixed for.
        """
        indent = GUTTER_CELLS + NUMBER_CELLS + (cell_len(CHECK_ON) if self.question.multi else 0)
        body = Text(no_wrap=True, overflow="ellipsis")
        body.append(" " * indent, style=ground)
        description = self._row_description(index)
        if description:
            body.append(truncate_cells(description, max(1, width - indent)), style=ground + ink)
        return _pad_row(body, width, ground)

    def _footer_hints(self, width: int) -> list[tuple[str, str]]:
        """The key hints that fit, dropping the least needed first.

        Three ladders rather than one, because the keys genuinely differ by
        state: a multi-select confirms rather than answers, and while the
        free-text row is selected the digits and Space have become letters, so
        advertising them there would be a lie. ``enter``/``esc`` are never
        dropped — between them they are how the card is used and how it is left.
        """
        if self.state.selected == self.other_row:
            hints = [("type", "your answer"), ("↑↓", "move"), ("enter", "accept")]
            droppable = ["↑↓", "type"]
        elif self.question.multi:
            hints = [("↑↓", "move"), ("space", "toggle"), ("enter", "confirm")]
            droppable = ["↑↓", "space"]
        else:
            hints = [("↑↓", "move"), ("1-9", "jump"), ("enter", "answer")]
            droppable = ["1-9", "↑↓"]
        hints.append(("esc", "skip"))

        def cells(pairs: list[tuple[str, str]]) -> int:
            return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
                0, len(pairs) - 1
            )

        for key in droppable:
            if cells(hints) <= width:
                return hints
            hints = [pair for pair in hints if pair[0] != key]
        if cells(hints) <= width:
            return hints
        # Even the two survivors will not fit with their labels (a card under
        # about 20 cells): keep the KEYS. Two bare keys still say which keys
        # exist, which is more than a clipped row says.
        return [(key, "") for key, _ in hints]


def _pad_row(row: Text, width: int, ground: Style) -> Text:
    """Extend ``row`` to the full card width in its own ground.

    Without this the tinted selection ends where the label ends, so the
    highlight is a ragged patch behind the text rather than a row — and the
    shorter of two adjacent options looks less selected than the longer one.
    """
    remaining = width - cell_len(row.plain)
    if remaining > 0:
        row.append(" " * remaining, style=ground)
    return row


def _sanitize(question: AskQuestion) -> AskQuestion:
    """A copy of ``question`` with every model-authored string stripped of
    control sequences, so nothing reaching the terminal can move the cursor."""
    return question.model_copy(
        update={
            "question": strip_control_sequences(question.question),
            "options": [
                option.model_copy(
                    update={
                        "label": strip_control_sequences(option.label),
                        "description": strip_control_sequences(option.description),
                    }
                )
                for option in question.options
            ],
        }
    )


def _tail_cells(text: str, width: int) -> str:
    """``text`` reduced to ``width`` cells, keeping its END.

    The mirror of :func:`truncate_cells`, for the one string on this card whose
    tail is the part being read: the characters the user is typing right now.
    """
    if cell_len(text) <= width:
        return text
    kept: list[str] = []
    spent = 1  # the leading ellipsis
    for char in reversed(text):
        size = cell_len(char)
        if spent + size > width:
            break
        kept.append(char)
        spent += size
    return "…" + "".join(reversed(kept))
