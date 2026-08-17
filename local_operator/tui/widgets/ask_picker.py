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

#: The card leaves a margin of conversation visible behind it, so it reads as
#: floating over the turn rather than replacing it: one row in
#: ``CARD_FLOAT_MARGIN_SHARE``, which is the ``max-height: 80%`` the ``/resume``
#: card uses, written as the part it gives back.
#:
#: It is written that way round because the margin has to YIELD, row by row, to
#: what the card actually has to show (:meth:`AskPickerScreen._body_rows`). A
#: flat share is a band worth having on a tall terminal and two rows of nothing
#: on a short one, bought with the answers the user is being asked to choose
#: between: at 100x14 it left three unusable rows under a card showing one
#: option of four, and at 30x12 the card had no room for its footer at all (D1).
#: Written instead as a threshold ("take the whole screen below N rows") the
#: card GREW as the terminal shrank, showing ten options at 100x20 and nine at
#: 100x24 — so the margin yields by the row, and never in a step.
CARD_FLOAT_MARGIN_SHARE = 5

#: The smallest body :meth:`AskPickerScreen._allocate` can say anything about
#: the QUESTION in: one option row, the windowing line that admits the rest of
#: the list is hidden, and the footer. Under it the plan collapses to the footer
#: alone, because one option row with no question above it and no count beside
#: it is an answer to nothing, while the keys are still how the user leaves.
#:
#: It is a THRESHOLD in the allocator, not a floor on the budget. As a floor it
#: changed no observable at all: the card went on laying out these three lines
#: into a body with room for one or two, and Textual clips the tail — so what a
#: five- or six-row terminal painted was an option row and no footer (round 3,
#: R9-R11, measured on the composited screen rather than the card's own text).
MIN_BODY_ROWS = 3

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
#: row under the cursor. The free-text row is the exception — see
#: :data:`OTHER_JUMP_KEY`.
NUMBER_CELLS = 3

#: The digit that always reaches the free-text row. ``0`` because it is the one
#: digit the ordinals never claim, and because the row it reaches is the row a
#: long list pushes past nine — where a blank gutter left the only answer that
#: is not on the list unreachable by any key (D13).
OTHER_JUMP_KEY = "0"

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


@dataclass(frozen=True)
class _CardLayout:
    """How one paint divides the card's rows, decided before anything is drawn.

    Every renderer below reads this rather than measuring the screen for itself.
    The header, the question, the spacers, the window and the footer each used
    to decide independently, and their sum came to more lines than the region
    had — so the card drew chrome it could not show and Textual clipped whatever
    was laid out last (D1). One division, one budget, one answer.
    """

    #: Content cells the card is drawing in, padding excluded.
    width: int
    #: The question's wrapped lines that fit, the last marked ``…`` if any were
    #: cut off.
    question: tuple[str, ...]
    #: The title and its rule, which are shown or dropped together.
    show_title: bool
    space_above: bool
    space_below: bool
    show_descriptions: bool
    #: Option ROWS the window may draw, whatever each of them costs in lines.
    #: Zero when the body is too short to say anything about the question.
    page: int
    #: Whether the windowing line was BOUGHT. The renderer reads this instead of
    #: re-deriving "the window is short of the list", which is how a row nobody
    #: had paid for got drawn and took the footer off the tail (round 3, R11).
    show_position: bool
    #: False only when the body has no drawable line at all. Everywhere else the
    #: footer is the first row bought, so it is the last thing that can go.
    show_footer: bool


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
        # ``0`` is not an ordinal: it reaches the free-text row, which is the
        # row a list of ten or more pushes past the digits (D13).
        Binding(OTHER_JUMP_KEY, "jump_other", "Other", show=False),
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
        #: Set when Enter was pressed on a state that cannot answer, so the
        #: footer can say WHY rather than the card doing nothing at all (D4).
        #: Cleared by every key that changes the answer, so the complaint can
        #: never outlive the state it describes.
        self._rejected = False
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
        """Index of the free-text row: always last, so its POSITION never moves.

        Its NUMBER does move, and past nine it has none of its own, which is
        why :data:`OTHER_JUMP_KEY` is bound to it separately.
        """
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
        would agree with a card that clipped half of it. Empty on a body with no
        room for a row at all, where the card is the footer or nothing.
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

    def action_jump_other(self) -> None:
        """``0`` reaches the free-text row from anywhere in the list.

        Its ordinal is past the digits on a list of ten or more, so without this
        the one row that can express an answer nobody enumerated is the one row
        no key reaches — on exactly the questions where scanning the list is
        hardest (D13).
        """
        self._move_to(self.other_row)

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
        self._rejected = False
        self._repaint()

    def action_backspace(self) -> None:
        if self.state.selected == self.other_row and self.state.typed:
            self.state.typed = self.state.typed[:-1]
            self._rejected = False
            self._repaint()

    def action_accept(self) -> None:
        """Enter: answer this question and move to the next, or dismiss.

        An empty answer is a NO-OP rather than a skip. Advancing on Enter with
        nothing chosen would record a question as asked-and-answered when the
        user had done nothing, and the model would then act on an answer nobody
        gave; Escape is how a question is left unanswered, and it says so in the
        footer.

        Refusing is only half of the job. The card NAMES this key in its footer,
        and a named key that does nothing and says nothing leaves the user
        pressing it again: the exported frame of the rejected press was
        BYTE-IDENTICAL to the frame before it, in two reachable states (D4). The
        refusal now answers in the footer's own row — see :meth:`_rejection`,
        which reverts as soon as there is something to take.
        """
        chosen = self._chosen()
        if not chosen:
            self._rejected = True
            self._repaint()
            return
        self._rejected = False
        self._answers[self.question.id] = chosen
        if self._index + 1 >= len(self._questions):
            self.dismiss(self._answers)
            return
        self._index += 1
        self._offset = 0
        self._hovered = None
        self._repaint()

    def _rejection(self) -> str:
        """What the footer says instead of the keys, or ``""`` to keep them.

        Derived from the state rather than stored as prose, so the complaint
        cannot contradict the card it sits on: the moment something is typed or
        ticked there is an answer, and the keys come back on the next paint.

        Which complaint depends on where the cursor is, not on the mode: a
        multi-select whose cursor is parked on the free-text row is answered by
        typing, and telling that user about Space would point at a key that is a
        letter while the field holds the cursor.
        """
        if not self._rejected or self._chosen():
            return ""
        if self.state.selected == self.other_row:
            return "type an answer first"
        return "nothing ticked — space toggles"

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
            self._rejected = False
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
        # No floor on the HEIGHT. Reporting at least eight rows on a six-row
        # screen is the same mistake the row budget used to make one level up:
        # every caller then divides rows the region does not have, and the
        # difference comes off the bottom silently. At 20x8 that floor alone
        # clipped the windowing line and the whole footer back off a card that
        # had budgeted for both (D1). The allocator handles a two-row budget.
        return max(1, size.width), max(1, size.height)

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

    def _body_rows(self, question_lines: int) -> int:
        """Lines the card's BODY may draw before the container clips them.

        The screen, less the margin the card floats in — and the margin yields
        to what the card has to show, a row at a time, down to nothing. What it
        yields to is the card WITHOUT its comforts: title, question, one line
        per option, and the footer. The spacers and the descriptions are left
        out on purpose, because they are the first things :meth:`_allocate`
        gives up, and a card should not push the conversation off the screen to
        keep a blank row.

        A CAP and not a fill: the body is ``height: auto``, so a card with
        little to say still draws short and still floats.

        The result is what the card can DRAW, with no floor under it: two rows
        go to ``Screen { padding: 1 }`` before this is measured
        (:meth:`_screen_size`) and two more to the card's own padding, so a
        terminal five rows tall leaves the body exactly one line and one four
        rows tall leaves it none. Zero is the honest answer there, and a plan of
        no lines is the honest card: there is nowhere to paint.

        Returning more than this is not a bigger card but a clipped one, and
        Textual clips the TAIL — the footer, the row :meth:`_allocate` buys
        first precisely because it is the only statement of how to leave. That
        is what a floor here bought: at heights 5 and 6 the card laid out three
        lines into one or two and painted an option row with no keys under it
        (round 3, R9-R11). :meth:`_allocate` spends this budget line by line and
        never overdraws it, so nothing is clipped rather than the wrong thing
        being.
        """
        _, height = self._screen_size()
        room = max(0, height - CARD_PADDING_ROWS)
        wanted = 2 + question_lines + self.row_count + 1
        margin = min(height // CARD_FLOAT_MARGIN_SHARE, max(0, room - wanted))
        return room - margin

    def _layout(self) -> _CardLayout:
        """Divide the body's rows BEFORE anything is drawn into them.

        The old arithmetic reserved chrome and then handed the options
        ``max(1, budget - chrome)`` rows — a floor the region did not have. The
        card then laid out more lines than it could draw and Textual dropped
        whatever came last, which is the footer: at 100x14 the frame showed a
        question, one option of four and NO keys, and at 30x12 only the title
        and the question — zero options and nothing saying how to leave a card
        the turn is parked on (D1, round 1).

        The fix is an order, not a bigger floor. Nothing is drawn that was not
        paid for, and it is paid for in the order the card cannot do without it:

        1. the footer — the only statement of how to leave;
        2. one option row — a question with no answers is not a question;
        3. the windowing line, whenever the window is short of the list, because
           a card quietly showing one of four has hidden three;
        4. the question, every wrapped line of it, marked ``…`` if even that
           cannot fit;
        5. the title and its rule, which travel together — a rule under a title
           is a caption, a rule under nothing is the edge of a box;
        6. the rest of the option rows;
        7. the blank spacers, which are rhythm and nothing else;
        8. the descriptions, all of them or none.

        A budget under :data:`MIN_BODY_ROWS` cannot buy even the first three, so
        the plan collapses to the footer alone — and at a budget of zero, which
        is every terminal four rows tall and under, to nothing at all. Laying
        out a card the screen cannot paint is how the keys went missing in the
        first place.
        """
        width = self._card_width()
        question = self._question_lines(width)
        budget = self._body_rows(len(question))
        plan = self._allocate(width, question, budget, position=False)
        if 0 < plan.page < self.row_count:
            # The list windows after all, so the line saying how much is hidden
            # has to be bought. Taking a row back can only shrink the page, so
            # this settles in one step rather than looping — and this branch is
            # the only place the row can be bought, which is the only place the
            # renderer will draw it from.
            plan = self._allocate(width, question, budget, position=True)
        return plan

    def _allocate(
        self,
        width: int,
        question: list[str],
        budget: int,
        *,
        position: bool,
    ) -> _CardLayout:
        """One trial division of ``budget`` body rows, in the order above.

        Past the collapse below, every line the returned plan implies is bought
        from ``remaining`` and nothing is bought that ``remaining`` cannot pay
        for, so the plan is exactly ``budget`` less whatever is left of it —
        never more than the budget, whichever branches are taken. It could be
        more before: handed 1 or 2 rows this ran ``remaining`` negative, still
        returned ``page=1``, and the renderer drew three lines into a body with
        room for one (round 3, R11).
        """
        if budget < MIN_BODY_ROWS:
            # Nothing about the question fits. See :data:`MIN_BODY_ROWS`.
            #
            # Two rows buy the selected row as well as the footer, and that row
            # is worth having even with no question above it and no count
            # beside it: on the free-text row it is the ONLY echo of what the
            # user is typing, and a card that accepts a typed answer without
            # showing it is worse than one showing a bare option. One row buys
            # the exit alone; none is a card that cannot be drawn, and drawing
            # it anyway is the clip itself (round 4, R15).
            return _CardLayout(
                width=width,
                question=(),
                show_title=False,
                space_above=False,
                space_below=False,
                show_descriptions=False,
                page=1 if budget >= 2 else 0,
                show_position=False,
                show_footer=budget >= 1,
            )
        remaining = budget - 2  # the footer, and one option row
        if position:
            remaining -= 1
        kept = list(question[:remaining])
        if len(kept) < len(question) and kept:
            # Say that the question continues. A silently halved question is
            # the one clip on this card the reader cannot detect: every other
            # abbreviation leaves a count, a caret or an empty gutter behind.
            tail = truncate_cells(kept[-1], max(1, width - 2))
            # `truncate_cells` marks its OWN cut, and two ellipses in a row read
            # as a rendering fault rather than as "there is more question".
            kept[-1] = f"{tail[:-1].rstrip() if tail.endswith('…') else tail} …"
        remaining -= len(kept)
        show_title = remaining >= 2
        if show_title:
            remaining -= 2
        extra = max(0, min(self.row_count - 1, remaining))
        remaining -= extra
        space_above = remaining >= 1
        if space_above:
            remaining -= 1
        space_below = remaining >= 1
        if space_below:
            remaining -= 1
        rows = 1 + extra
        # Descriptions are bought last and all at once: they cost one line per
        # row, and a list where only some entries have their second line reads
        # as broken rather than as abbreviated.
        descriptions = rows >= self.row_count and remaining >= self.row_count
        return _CardLayout(
            width=width,
            question=tuple(kept),
            show_title=show_title,
            space_above=space_above,
            space_below=space_below,
            show_descriptions=descriptions,
            page=self.row_count if descriptions else rows,
            show_position=position,
            show_footer=True,
        )

    def _window(self, page: int | None = None) -> list[int]:
        """The row indexes currently drawn, after clamping the scroll offset."""
        if page is None:
            page = self._layout().page
        offset = max(0, min(self._offset, max(0, self.row_count - page)))
        self._offset = offset
        return list(range(offset, min(self.row_count, offset + page)))

    # -- internals -----------------------------------------------------------
    def _move_to(self, index: int) -> None:
        self.state.selected = max(0, min(self.row_count - 1, index))
        # Any movement is a new state, so a refused Enter stops describing it.
        self._rejected = False
        # Scroll only far enough to keep the cursor drawn, so the list is stable
        # while moving through the middle of it. A page of zero draws no rows at
        # all (a body under :data:`MIN_BODY_ROWS`), and there is no keeping a
        # cursor drawn on a list that is not: scrolling it would only leave an
        # offset behind for the terminal to grow back into.
        page = self._layout().page
        if self.state.selected < self._offset:
            self._offset = self.state.selected
        elif page and self.state.selected >= self._offset + page:
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
        text = self._card_text()
        body.update(text)
        card = body.parent
        if card is not None:
            # A card with no drawable line is not a smaller card: its own
            # ``padding: 1 2`` is two rows the screen has not got, so at three
            # terminal rows and under the container pushed the screen's virtual
            # height past its size — a scrollable screen, which AGENTS.md calls
            # always a bug here, over a card painting nothing. Nothing to draw,
            # nothing laid out; the next resize brings it back.
            card.display = bool(text.plain)

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads.

        Empty when the body has no drawable line at all, because ``Text``
        splits an empty card into ONE empty line and a caller counting that
        against the room the screen has would be told the card overflows a
        terminal it is painting nothing into.
        """
        text = self._card_text()
        if not text.plain:
            return []
        return [line.plain for line in text.split("\n")]

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
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        layout = self._layout()
        width = layout.width

        out = Text()
        lines: list[int | None] = []

        def newline(row: int | None) -> None:
            """Start a line that belongs to ``row`` (``None`` = chrome)."""
            if lines:
                out.append("\n")
            lines.append(row)

        if layout.show_title:
            newline(None)
            out.append_text(self._header(width, fg, muted))
            newline(None)
            # The raised card ground needs the raised hairline: `edge` is tuned
            # against the app background and nearly vanishes on `overlay`.
            out.append("─" * width, style=faint)
        for line in layout.question:
            newline(None)
            out.append(line, style=fg)
        if layout.space_above:
            newline(None)

        window = self._window(layout.page)
        for index in window:
            ground = self._row_ground(index)
            newline(index)
            out.append_text(self._row_text(index, width, ground, fg, dim, faint, layout))
            if layout.show_descriptions:
                newline(index)
                out.append_text(self._description_text(index, width, ground, muted, dim))
        if layout.space_below:
            newline(None)

        # Both rows are drawn only where the plan BOUGHT them. The position line
        # used to be emitted on `len(window) < self.row_count` alone, with no
        # reference to the budget: at 1 or 2 rows the allocator never paid for
        # it, the renderer drew it regardless, and the footer went off the tail
        # (round 3, R11). The allocator decides; this only reads the decision.
        if layout.show_position:
            newline(None)
            out.append_text(self._position_row(width, window, muted, dim))
        if layout.show_footer:
            newline(None)
            out.append_text(self._footer_row(width, muted, dim, drawn=bool(window)))
        self._line_rows = lines
        return out

    def _position_row(self, width: int, window: list[int], muted: Style, dim: Style) -> Text:
        """``showing 2–3 of 6`` — how much of the list is not on screen.

        Numerals at `muted` and the grammar at `dim`, both a step up: at
        `faint`, `showing`/`of` measured 1.49:1 on this card's own ground, so
        the row that says how much is hidden was itself hidden (D5).

        The counts outlive the word: at 20 columns `showing 1–1 of 4` is wider
        than the whole card, and the card is exactly where the row matters most.
        `1–1 of 4` says the same thing in half the cells, so the word is what
        goes rather than the row.
        """
        span = f"{window[0] + 1}–{window[-1] + 1}"
        total = str(self.row_count)
        row = Text(no_wrap=True, overflow="ellipsis")
        if cell_len(f"showing {span} of {total}") <= width:
            row.append("showing ", style=dim)
        row.append(span, style=muted)
        row.append(" of ", style=dim)
        row.append(total, style=muted)
        return _cut_row(row, width)

    def _footer_row(self, width: int, muted: Style, dim: Style, *, drawn: bool = True) -> Text:
        """The key hints — or what the last refused Enter has to say instead.

        One row either way, so a refusal never moves the card under the cursor,
        and the keys come back the moment the state can answer.

        ``drawn`` is False on the collapsed card, where the footer is the only
        line and no option row is on screen. Then the exit is the ONLY honest
        hint: `enter` would commit a selection the user cannot see, and the
        digits would jump within a list that is not there. Measured on the
        previous revision at a 5-row terminal: `down down down enter` committed
        an option nobody had been shown (round 4, R14). This file already
        refuses to advertise the digits on the free-text row for the same
        reason — a key offered where it does not do what it says is a lie.
        """
        row = Text(no_wrap=True, overflow="ellipsis")
        rejection = self._rejection()
        if rejection:
            row.append(rejection, style=muted)
            return _cut_row(row, width)
        hints = self._footer_hints(width) if drawn else [("esc", "skip")]
        for position, (key, what) in enumerate(hints):
            if position:
                row.append(" · ", style=dim)
            # Keys at `muted` (6.51:1) and their grammar at `dim` (3.43:1), each
            # a step up from `dim`/`faint`. The row that explains how to use and
            # how to leave the card was its least legible text, its words at
            # 1.49:1 on the `overlay` ground — the same failure this file
            # diagnosed one method down for the descriptions and left standing
            # here (D5).
            row.append(key, style=muted)
            if what:
                row.append(f" {what}", style=dim)
        return _cut_row(row, width)

    def _header(self, width: int, fg: Style, ink: Style) -> Text:
        """Title on the left, ``Question 1 of 2`` on the right when there is
        more than one.

        Worded and at ``muted``, not ``1/2`` at ``dim``: this is the only thing
        on the card saying that more questions follow, and right-aligned at
        3.43:1 it was the faintest text in the header — two consecutive frames
        of a two-question run differed in exactly one character (D10).

        The counter is dropped before the title when the card is narrow: it says
        how much is left, and the title says what the card IS.
        """
        header = Text(no_wrap=True, overflow="ellipsis")
        title = "the agent needs your decision"
        counter = (
            f"Question {self._index + 1} of {len(self._questions)}"
            if len(self._questions) > 1
            else ""
        )
        gap = 2
        if counter and cell_len(title) + gap + cell_len(counter) <= width:
            header.append(title, style=fg)
            header.append(" " * (width - cell_len(title) - cell_len(counter)))
            header.append(counter, style=ink)
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

    def _row_number(self, index: int) -> str:
        """The digit gutter's contents for one row.

        Rows past nine keep the indent and lose the number: the digits are a
        shortcut, and re-flowing the list at ten options to reclaim two cells
        would move every row under the cursor. The free-text row is the
        exception, because it is the one row a long list ALWAYS pushes past the
        digits and the one row that can express an answer nobody enumerated —
        with twelve options it drew a blank gutter while the footer still
        offered `1-9 jump`, so `Other` was unreachable by digit exactly where
        scanning the list is hardest (D13).
        """
        if index < 9:
            return f"{index + 1}."
        if index == self.other_row:
            return f"{OTHER_JUMP_KEY}."
        return ""

    def _row_text(
        self,
        index: int,
        width: int,
        ground: Style,
        fg: Style,
        dim: Style,
        faint: Style,
        layout: _CardLayout,
    ) -> Text:
        """One option row: cursor, number, checkbox, label.

        The accent green marks WHAT ENTER WILL TAKE — the cursor's row on a
        single-select, the ticked rows on a multi-select, and neither on a
        free-text row with nothing typed into it. It used to be the cursor's
        label in every mode, so one multi-select frame spent the ink on two
        different claims at once: `[x]` on row 1 (chosen) and the label of row 3
        (merely under the cursor, its own box empty). The accent's one-thing
        rule in `local_operator.tcss` was then not what the frame said (D11),
        and on a multi-select the ink was pointing at the row Enter does NOT
        take.

        The cursor stays unmistakable without it: `tint-select` is a HUE step
        rather than an elevation one (see theme.py — elevation alone measures
        1.096:1), and the caret sits two columns to its left. The caret is
        ``muted`` and not the accent for the reason the command picker records:
        a second green glyph beside a green label reads as a duplicated caret.
        """
        selected = index == self.state.selected
        accent = ground + Style(color=theme_mod.semantic_color("accent"))
        row = Text(no_wrap=True, overflow="ellipsis")
        row.append(
            f"{CURSOR} " if selected else " " * GUTTER_CELLS,
            style=ground + Style(color=theme_mod.semantic_color("muted")),
        )
        row.append(
            f"{self._row_number(index):<{NUMBER_CELLS}}",
            style=ground + (dim if selected else faint),
        )
        spent = GUTTER_CELLS + NUMBER_CELLS
        typed = bool(self.state.typed.strip())
        if self.question.multi:
            checked = index in self.state.checked or (index == self.other_row and typed)
            # A ticked box is the accent: it is the statement the ink is spent
            # on — "Enter takes this one" — and a multi-select confirms several
            # rows at once, so the ticks have to be readable without moving the
            # cursor over each of them.
            row.append(
                CHECK_ON if checked else CHECK_OFF,
                style=accent if checked else ground + faint,
            )
            spent += cell_len(CHECK_ON)
            taken = checked
        else:
            taken = selected and (index != self.other_row or typed)

        budget = max(LABEL_MIN_CELLS, width - spent)
        text = self._row_label(index)
        if index == self.other_row and selected:
            # The typed string keeps its TAIL: a field that truncated the end
            # would hide the characters being typed, which is the only part of
            # it the user is looking at.
            budget -= cell_len(FIELD_CARET)
            row.append(OTHER_PREFIX, style=accent if taken else ground + fg)
            row.append(
                _tail_cells(self.state.typed, max(1, budget - cell_len(OTHER_PREFIX))),
                style=ground + fg,
            )
            row.append(FIELD_CARET, style=accent if taken else ground + dim)
        else:
            row.append(truncate_cells(text, budget), style=accent if taken else ground + fg)
        if self.question.recommended == index and not layout.show_descriptions:
            # No description line to carry the tag, so it rides here — but only
            # when it fits AFTER the label, rather than out of the label's own
            # budget. Charged to the budget it made the promoted option the
            # shortest label on the card, and on a 28-cell screen the label
            # floor won the arithmetic and the 15-cell tag was appended on top
            # of the floor anyway, pushing the card two cells past the terminal
            # (D2/D6). A badge that truncates what it promotes, or that
            # overflows the screen to fit, is worth less than no badge: the
            # recommendation is PRESELECTED too, so the cursor is already there.
            tag = f"  · {RECOMMENDED_TAG}"
            if cell_len(row.plain) + cell_len(tag) <= width:
                row.append(tag, style=ground + dim)
        return _fit_row(row, width, ground)

    def _description_text(
        self, index: int, width: int, ground: Style, tag_ink: Style, ink: Style
    ) -> Text:
        """The row's second line: the recommendation tag, then the consequence.

        Drawn at ``dim`` and NOT at ``faint`` like the footer's grammar words:
        this line is the CONSEQUENCE of choosing the row, which is what the user
        is reading the card to compare. The first captured frame had it at
        ``faint`` on the ``overlay`` ground, where "nothing in the app reads it
        any more" was barely present — the same contrast failure the approval
        prompt's key hints were fixed for.

        The tag lives HERE, ahead of the prose, rather than after the label: on
        the label line it was paid for out of the label, so the one row the
        model is pointing at carried the shortest text on the card (D6). At
        ``muted`` it is the loudest thing on a line of ``dim`` prose and still
        quieter than the label above it, which is the ranking a hint wants.
        """
        indent = GUTTER_CELLS + NUMBER_CELLS + (cell_len(CHECK_ON) if self.question.multi else 0)
        body = Text(no_wrap=True, overflow="ellipsis")
        body.append(" " * indent, style=ground)
        description = self._row_description(index)
        room = max(1, width - indent)
        if self.question.recommended == index:
            body.append(RECOMMENDED_TAG, style=ground + tag_ink)
            room -= cell_len(RECOMMENDED_TAG)
            if description and room > 3:
                body.append(" · ", style=ground + ink)
                room -= 3
            else:
                description = ""
        if description:
            body.append(truncate_cells(description, room), style=ground + ink)
        return _fit_row(body, width, ground)

    def _footer_hints(self, width: int) -> list[tuple[str, str]]:
        """The key hints that fit, shedding WORDS before it sheds KEYS.

        Two passes, and the order between them is the point: a key with no word
        beside it still names a key that exists, while a dropped hint is a key
        nobody can discover. Shedding whole hints first took `space toggle` off
        a 46-column multi-select and kept `esc skip`, leaving five empty boxes
        and one offered key that does nothing until one of them is ticked (D3).

        Three ladders rather than one, because the keys genuinely differ by
        state: a multi-select confirms rather than answers, and while the
        free-text row is selected the digits and Space have become letters, so
        advertising them there would be a lie. Each ladder is ordered LEAST
        defended first and drives both passes, so the last word standing and the
        last key standing belong to the same hint.

        What sits at the end differs by state, and that is the whole ranking.
        Normally it is ``esc``: a card with no stated way out is unusable where
        a card with fewer keys is merely terse, and `skip` is the one word here
        nobody can guess (it leaves THIS question unanswered, it does not cancel
        the ones already answered). On a multi-select it is ``space``, which
        outranks even that: it is the ONLY key that can answer the question, and
        it used to be dropped while `esc skip` survived.
        """
        if self.state.selected == self.other_row:
            hints = [("type", "your answer"), ("↑↓", "move"), ("enter", "accept")]
            ladder = ["↑↓", "type", "enter", "esc"]
        elif self.question.multi:
            hints = [("↑↓", "move"), ("space", "toggle"), ("enter", "confirm")]
            ladder = ["↑↓", "enter", "esc", "space"]
        else:
            # `0-9` once the free-text row has taken `0`, because `1-9` then
            # advertised a range that stopped short of the list (D13).
            jump = f"{OTHER_JUMP_KEY}-9" if self.other_row >= 9 else "1-9"
            hints = [("↑↓", "move"), (jump, "jump"), ("enter", "answer")]
            ladder = ["↑↓", jump, "enter", "esc"]
        hints.append(("esc", "skip"))

        def cells(pairs: list[tuple[str, str]]) -> int:
            return sum(cell_len(f"{key} {what}".strip()) for key, what in pairs) + 3 * max(
                0, len(pairs) - 1
            )

        shown = list(hints)
        if cells(shown) <= width:
            return shown
        for key in ladder:
            shown = [(name, "" if name == key else what) for name, what in shown]
            if cells(shown) <= width:
                return shown
        for key in ladder[:-1]:
            shown = [pair for pair in shown if pair[0] != key]
            if cells(shown) <= width:
                return shown
        # One bare key left, on a card too narrow for two. It is the one the
        # ladder defends hardest, and :func:`_cut_row` keeps even that inside
        # the card rather than letting it widen the screen.
        return shown


def _cut_row(row: Text, width: int) -> Text:
    """``row`` cut to the card's width, in the card's own width model.

    Load-bearing rather than defensive: every one of these Texts is
    ``append_text``-ed into ONE Text whose overflow governs, so the
    ``no_wrap``/``ellipsis`` pair they were built with never fires. The body is
    ``width: auto``, so a line that overshoots simply widens the card — which is
    how a 15-cell tag pushed the card two cells past a 28-cell screen and gave
    the app the one condition AGENTS.md calls always a bug (D2).
    """
    if cell_len(row.plain) > width:
        row.truncate(width, overflow="ellipsis")
    return row


def _fit_row(row: Text, width: int, ground: Style) -> Text:
    """``row`` at exactly the card's width, in its own ground.

    Without the padding the tinted selection ends where the label ends, so the
    highlight is a ragged patch behind the text rather than a row — and the
    shorter of two adjacent options looks less selected than the longer one.
    """
    _cut_row(row, width)
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
