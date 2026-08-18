"""The ``ask`` tool's picker: the agent's question as a real answerable list.

Why this exists: a model that needs a decision has, until now, had exactly one
shape available to it — prose. Observed verbatim in a live session:

    (A) Drop email … (B) Escalate it properly … (C) You have context I don't

Three options printed into the transcript, none of them clickable, none of them
selectable, and the user's answer arriving as free text the agent then has to
re-parse. This screen is the other half of the ``ask`` tool: one question at a
time, keyboard and mouse, answering with the label the model wrote.

It is ANCHORED IN THE DOCK, not pushed as a modal screen, and that is the
central design decision. A ``ModalScreen`` covers the terminal: the question
arrived and the conversation it is about — the tool output, the error, the plan
the agent is asking about — went behind it, unreadable and unscrollable. A user
who needed to look something up to answer had to abandon the question to do it.
Mounted in ``#prompt-host`` instead, the card is a row band at the top of the
input dock: the transcript keeps the rest of the screen, keeps its scrollback,
and the card stays put while the user scrolls up for the context they need.

It sits ABOVE the dock band (subagents, todos) rather than below it, because
those panels are status and this is a question the turn is parked on: the thing
being waited on belongs closest to the eye's resting place, and a status list
that pushed the question around as jobs came and went would move the card under
the user's cursor mid-answer.

Its frame is the ``/resume`` picker's (one ``Static``, content-sized, measured
every paint) because that surface already solved the parts that are easy to get
wrong here: the card clipping its own footer, the cursor sitting on a row that
was never drawn, and a click resolving to a row that was never painted.

Two things it does that no other picker in this app does:

- **Every question offers a free-text answer.** The prose surface this replaces
  needed one constantly — "(C) You have context I don't" is a model asking for
  an answer it could not enumerate. Selecting the ``Other`` row turns it into a
  text field, so the options never have to be exhaustive.
- **It answers with TEXT, not an index.** The free-text row hands back a string
  that was never in ``options``, which an index cannot express.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.message import Message
from textual.widget import Widget
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
#: of ``padding: 1 1`` in the stylesheet. One cell rather than the modal's two:
#: docked, this is the same rail every other panel in the dock uses (the
#: composer's ``❯`` column, ``.band-body { padding: 0 1 }``), so the question
#: starts in the column the rest of the dock starts in. Chrome that sets its
#: own left edge is what makes a docked panel look detached from the input.
ASK_PADDING_CELLS = 1


#: Rows the transcript claims even when it is showing nothing at all.
#:
#: It is ``height: 1fr`` and a sibling of the dock under the screen, so it takes
#: its rows whatever the dock does with the rest. A card that budgeted the
#: screen without them produced a dock taller than the screen — a scrollable
#: screen with the top of the question cut off. Distinct from
#: :data:`MIN_TRANSCRIPT_ROWS`, which is the anchoring PREFERENCE: this is the
#: layout engine's floor, and it applies even where the preference has yielded.
#:
#: TWO, not one: ``TranscriptView`` carries ``padding: 1 0 1 1`` — a row of the
#: conversation's own ground above and below — and padding is inside the
#: scrollable region, so both survive. Written as 1 the budget was short by
#: exactly one row and `virtual_size` exceeded `size` at 100x14, 54x14, 80x13,
#: 30x12 and 40x12 (F2, agent review round 1).
#:
#: It was invisible to every geometry test in this file because they all mount
#: into an app with an EMPTY transcript, which is still in the boot layout —
#: and ``Screen.boot TranscriptView`` drops the padding to ``0 0 0 1``,
#: cancelling the error exactly. The tests now seed a conversation, which is
#: also the only state in which this surface is ever actually used.
_TRANSCRIPT_MIN_ROWS = 2

#: Rows to assume the rest of the dock needs before it has been laid out — a
#: one-row composer, its chevron row's padding, and the status band. Used only
#: on the first paint of a session's first question, and deliberately an
#: OVER-estimate of the minimum dock: reserving rows the dock does not need
#: costs one windowed row for one frame, while reserving too few lets the card
#: lay out over the composer and lose its footer to the clip.
_DOCK_ROWS_FALLBACK = 4

#: Rows the card costs the dock beyond its own text: one padding row above, one
#: below (the vertical half of ``padding: 1 1``), plus the one rhythm row its
#: slot owns underneath (``.prompt-slot { padding: 0 0 1 0 }``), which separates
#: the question from the band or the composer below it.
#:
#: Kept separate from the cell budget above rather than shared with it: the two
#: were equal by coincidence under the old ``padding: 1 2`` and spending one for
#: the other is a bug waiting for the stylesheet to move, which it since has.
CARD_PADDING_ROWS = 3

#: The most of the terminal the card may ever claim, as a share of its rows.
#:
#: This is the anchoring rule expressed as a number, and it is the whole reason
#: the surface was moved out of a ``ModalScreen``. A modal took the screen, so
#: the conversation the question is ABOUT went behind it: the tool output that
#: prompted the ask, the error being asked about, the plan under discussion. A
#: user who needed to re-read any of that to answer had to dismiss the question
#: first. Capped here, the card is always a BAND and the transcript is always
#: the rest of the screen.
#:
#: A share rather than a row count because the thing being protected is
#: proportional: two rows of transcript under a card is not "some conversation
#: visible", it is a sliver that answers nothing. The floor below is what keeps
#: the rule meaningful on a short terminal, where a share alone rounds to almost
#: everything.
#:
#: 0.7 is the same ratio omp's ask dialog settled on (``DIALOG_HEIGHT_RATIO``),
#: and the agreement is not a coincidence: it is the point where a six-option
#: question can still show the CONSEQUENCE line under each option on an
#: ordinary 30-row terminal. Measured at 0.6 the card had 15 body rows against
#: the 17 that question needs, so every description was dropped — and the
#: descriptions are what the user is comparing when they choose. At 0.7 the same
#: question draws in full and still leaves six rows of conversation.
PROMPT_HEIGHT_SHARE = 0.7

#: Conversation rows the card will not take, whatever the share above allows.
#:
#: The share is a ceiling on the CARD; this is a floor under the TRANSCRIPT, and
#: both are needed because they bind at opposite ends of the size range. On a
#: tall terminal the share is what stops a twelve-option question from filling
#: the screen; on a short one the share alone would still leave the conversation
#: with a row or two, so this takes over and the card windows its list instead.
#:
#: Four rows is the smallest number that shows an exchange rather than a
#: fragment: a user block, its first line of reply, and the beginning of what
#: came next. Below that the card is better off windowing — it can say "3 of 9"
#: about its own list, and the transcript cannot say anything about what it is
#: cut off from.
#:
#: It is a FLOOR and not the target. On a terminal with room to spare the share
#: above is what governs, and it leaves considerably more; this only takes over
#: where the share alone would leave the conversation a sliver. Both are needed
#: because they bind at opposite ends of the size range — see
#: :meth:`AskPickerScreen._body_rows`.
MIN_TRANSCRIPT_ROWS = 4

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


class AskPickerScreen(Container):
    """Put the ``ask`` tool's questions to the user; settle with the answers.

    Settles with ``question id -> chosen strings``, or ``None`` when nothing at
    all was answered. A PARTIAL mapping is deliberate: escaping out of the third
    question does not throw away the first two, because a user who answered and
    then stopped answering has still told the agent something, and the tool
    reports the rest as not answered.

    A ``Container`` mounted in the dock, not a ``ModalScreen``: see the module
    docstring. The name is kept — it is what the app, the tests and the harness
    already call this surface, and renaming a class to describe its parent
    widget rather than its job would churn every call site to say nothing new.

    It TAKES FOCUS, and that is what makes the keys work. The alternative is
    app-level bindings while the composer keeps focus, and that cannot work
    here: the answer keys are digits, letters and Space, and the composer is a
    text buffer that would swallow every one of them as input. Focus is handed
    back on settle (:meth:`restore_focus`), so answering never silently leaves
    the user somewhere other than the composer.
    """

    #: Focusable so the answer keys reach the card rather than the composer's
    #: buffer. Same rule the approval prompt follows, for the same reason.
    can_focus = True

    class DrawableChanged(Message):
        """The card started or stopped having anything to paint.

        Raised on the edge only, so the host is not told the same thing on
        every keystroke. The host reserves a separation row for the prompt and
        must give it back when the terminal gets too short for the card to draw
        at all — otherwise the dock keeps a row for a question painting nothing
        and pushes itself past the screen.
        """

        def __init__(self, card: "AskPickerScreen") -> None:
            super().__init__()
            self.card = card

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

    def __init__(
        self,
        questions: Sequence[AskQuestion],
        on_settle: Callable[[dict[str, list[str]] | None], None] | None = None,
        *,
        allow_free_text: bool = True,
        # Defaulted per INSTANCE rather than to a shared literal: two prompts
        # can be attached at once for the few pump hops between one settling
        # and the awaiting frame unmounting it, and a fixed id makes Textual
        # refuse the second mount outright (`DuplicateIds`). Callers that want
        # a stable handle pass their own.
        widget_id: str | None = None,
        title: str = "the agent needs your decision",
        exit_hint: tuple[str, str] = ("esc", "skip"),
    ) -> None:
        super().__init__(id=widget_id or f"ask-picker-{id(self):x}", classes="prompt-slot")
        #: Whether the list carries the trailing "Other (type your own)" row.
        #:
        #: A property of the SURFACE rather than of ``AskQuestion``, so the
        #: model-facing tool schema is unchanged: the ``ask`` tool's contract is
        #: that every question it asks accepts an answer nobody enumerated, and
        #: that stays true. What varies is the host — the approval prompt reuses
        #: this widget for a question with exactly three answers (allow, deny,
        #: allow all), where a free-text row would offer to send prose to a
        #: gate that can only return a boolean.
        self._allow_free_text = allow_free_text
        #: The card's title row. Varies by host because the two surfaces are
        #: asking different kinds of question: ``ask`` wants a decision the
        #: agent cannot make, and the approval gate wants permission to act.
        self._title = title
        #: The last hint in the footer, and the one the width ladder defends
        #: hardest — a card with no stated way out is unusable. Its WORD differs
        #: by host and the difference is load-bearing: "skip" is honest for a
        #: question the agent can proceed without, and a lie for an approval,
        #: where escaping denies the tool and stops the turn.
        self._exit_hint = exit_hint
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
        #: Called once with the answers when the card settles. The app resolves
        #: the waiting tool call from it.
        self._on_settle = on_settle
        #: Guards the callback to EXACTLY ONE call. Several paths end this card
        #: — Enter on the last question, Escape, and the app tearing it down on
        #: a stop or an abort — and the tool call behind it is parked on a
        #: future that raises if it is resolved twice. The same idempotence the
        #: approval prompt's :meth:`resolve` has, for the same reason.
        self._settled = False
        #: What held focus when the card appeared, so answering hands it back
        #: rather than leaving the user out of the composer.
        self._restore_target: object | None = None
        #: Whether the last paint produced any line at all. Starts True so the
        #: first undrawable paint is an EDGE and is announced: starting False
        #: would make "nothing to draw" the unremarkable case and leave the host
        #: holding a row nobody asked it to give back.
        self._drawable = True
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
        """Options, plus the free-text row where this surface offers one."""
        return len(self.question.options) + (1 if self._allow_free_text else 0)

    @property
    def other_row(self) -> int:
        """Index of the free-text row: always last, so its POSITION never moves.

        Its NUMBER does move, and past nine it has none of its own, which is
        why :data:`OTHER_JUMP_KEY` is bound to it separately.

        ``-1`` where the surface has no free-text row, which is deliberately an
        index no row can equal: every ``index == self.other_row`` test in this
        file then answers False without a second condition beside it, and the
        one place the value is used as a destination (:meth:`action_jump_other`)
        guards on the flag instead.
        """
        return self.row_count - 1 if self._allow_free_text else -1

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

    # -- settling ------------------------------------------------------------
    def settle(self, answers: dict[str, list[str]] | None) -> None:
        """Hand ``answers`` to the waiting tool call, exactly once.

        Idempotent, because this card has several ends and they can race: the
        user's Enter on the last question, their Escape, and the app tearing the
        card down for a stop, an abort, or teardown. The tool call is parked on
        a future, and resolving that twice raises out of whichever path lost.
        """
        if self._settled:
            return
        self._settled = True
        callback = self._on_settle
        self._on_settle = None
        if callback is not None:
            callback(answers)

    @property
    def settled(self) -> bool:
        """Whether the answers have already been handed back."""
        return self._settled

    def restore_focus(self) -> None:
        """Return focus to whatever held it when the question appeared."""
        widget = self._restore_target
        self._restore_target = None
        if widget is not None and getattr(widget, "is_attached", False):
            widget.focus()  # type: ignore[attr-defined]

    # -- actions -------------------------------------------------------------
    def action_cancel(self) -> None:
        """Escape: stop answering. Whatever was already answered still counts."""
        self.settle(self._answers or None)

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

        A no-op where the surface has no such row, rather than a jump to
        ``-1``: the key is bound unconditionally, and the approval prompt that
        reuses this widget has three answers and no fourth to reach.
        """
        if self._allow_free_text:
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
        # Nothing is committed from a card that drew no options.
        #
        # The cursor still sits on a row — often the RECOMMENDED one, which is
        # preselected — so Enter on a collapsed card would take an answer the
        # user was never shown. The footer already refuses to advertise `enter`
        # there for exactly this reason; this makes the key agree with the hint
        # rather than merely going unadvertised (D9, design round 2).
        if not self.visible_rows:
            self._rejected = True
            self._repaint()
            return
        chosen = self._chosen()
        if not chosen:
            self._rejected = True
            self._repaint()
            return
        self._rejected = False
        self._answers[self.question.id] = chosen
        if self._index + 1 >= len(self._questions):
            self.settle(self._answers)
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

        Any button-1 click on the card also takes FOCUS back, before the row
        hit-test and whether or not it landed on a row. The card can lose focus
        without being answered — the user clicks into the composer to look
        something up while deciding — and without this the question sits there
        advertising keys with nothing to receive them. Clicking the thing you
        are being asked is the discoverable way back, and it is the affordance
        the approval prompt already offers.
        """
        if getattr(event, "button", 1) != 1:
            return
        if not self.has_focus:
            self.focus()
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

        - the point must be inside the BODY's region. The card is full-width in
          the dock while its TEXT is not, so a click in the empty columns beside
          a row still lands on this widget with a ``y`` that looks valid; and
          the padding rows above and below the body do the same with an ``x``
          that does. Region containment is what separates the card's ink from
          the container it is painted in;
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
        """The box the card budgets itself against: the SCREEN's content box.

        Deliberately not ``self.size``, which is the inverse of what the modal
        version did and is forced by the move into the dock. A docked container
        is sized BY its content: ``height: auto`` means ``self.size`` is
        whatever the card asked for on the last paint, so budgeting against it
        would be the card measuring its own shadow — it could never shrink,
        because every measurement would confirm the height it already had.

        And deliberately not ``self.app.size`` either, which is the raw
        TERMINAL. ``Screen { padding: 1 }`` insets the box the dock is actually
        laid out in, so the two differ by two rows: measured at an 80x12
        terminal the screen's content box is 78x10, and a card budgeting
        against 12 handed the dock two rows that did not exist. The dock came
        out 11 rows tall inside a 10-row screen, ``virtual_size`` exceeded
        ``size``, and the screen went scrollable with the top of the question
        pushed off the frame — the condition AGENTS.md calls always a bug here.
        Falls back to the app's own size only when the screen is not available,
        which is the pre-layout case where nothing is drawn yet anyway.
        """
        try:
            screen = self.screen if self.is_attached else None
            size = screen.size if screen is not None else self.app.size
        except Exception:  # pragma: no cover - only before the app has a screen
            return 80, 24
        if not size.width:  # pragma: no cover - pre-layout
            return 80, 24
        if not size.height:
            # A screen with ZERO content rows is a real state, not an
            # unmeasured one: `Screen { padding: 1 }` consumes two rows, so a
            # terminal two rows tall leaves nothing at all. Falling back to the
            # pre-layout default here (24 rows) was a defect — the card budgeted
            # against a screen four times the terminal, laid out its full
            # thirteen-line layout, and drove `virtual_size` to 20 rows on a
            # screen of 0. Zero rows in, zero rows out, and the card is not
            # drawn; the next resize brings it back.
            return max(1, size.width), 0
        # No floor on the HEIGHT. Reporting at least eight rows on a six-row
        # screen is the same mistake the row budget used to make one level up:
        # every caller then divides rows the region does not have, and the
        # difference comes off the bottom silently. At 20x8 that floor alone
        # clipped the windowing line and the whole footer back off a card that
        # had budgeted for both (D1). The allocator handles a two-row budget.
        return max(1, size.width), max(1, size.height)

    def _dock_reserved_rows(self) -> int:
        """Rows the rest of the dock needs below this card, measured live.

        The composer, the status band, and whatever the dock band is showing
        (subagent list, todos) all sit under the question, and the card may not
        push any of them off: the composer is where the answer to a free-text
        row is typed, and the status band carries the working line that says the
        turn is parked. Measured from the live widgets rather than assumed as a
        constant, because the band's height is genuinely variable — it is zero
        on an idle session and several rows with two subagents and a todo list —
        and a constant would be wrong in whichever direction the session was not
        in when it was written.

        Falls back to a conservative estimate when the dock has not been laid
        out yet (the first paint of the first question in a session), which
        errs toward a SHORTER card: budgeting rows the dock turns out to need
        is the failure that clips a footer, and budgeting too few only windows
        the list one row earlier for one frame.
        """
        screen = self.screen if self.is_attached else None
        if screen is None:
            return _DOCK_ROWS_FALLBACK
        # A host with no composer at all reserves nothing, and saying otherwise
        # is not conservative, it is wrong: the reduced hosts (the widget tests,
        # and any embedder mounting this card beside a transcript) have no dock,
        # so charging them for one shrinks the card by four rows it was given.
        # Distinguished from "the dock exists but has not been laid out yet" by
        # asking whether the composer is THERE, not how tall it currently is.
        if not screen.query("#input-shell"):
            return 0
        # The DOCK's own outer height, less whatever this card is currently
        # contributing to it, rather than the sum of its other children.
        #
        # Summing the siblings misses what the dock spends on itself: measured
        # at 30x12, `#prompt-host` (4) plus `#input-shell` (5) came to 9 while
        # `#input-dock` measured 10, and the missing row put the whole dock past
        # the screen — `virtual_size` 12 against `size` 10, permanently, because
        # the card kept re-deriving the same one-row-too-many budget. Reading
        # the container is also robust to a fourth child appearing later, which
        # a hardcoded list of siblings is not.
        dock = None
        try:
            dock = screen.query_one("#input-dock")
        except Exception:  # pragma: no cover - only before the dock is composed
            dock = None
        if dock is not None and dock.display and dock.outer_size.height:
            # This card's own contribution comes OUT of the reservation: what
            # is being computed is the room left FOR it, and counting the rows
            # it currently occupies would shrink the budget by whatever the card
            # already claimed — a feedback loop that can only ratchet downward.
            host = self.parent
            mine = 0
            if isinstance(host, Widget) and host.display:
                mine = host.outer_size.height
            return max(0, dock.outer_size.height - mine)
        reserved = 0
        seen = False
        for selector in ("#band", "#input-shell"):
            try:
                widget = screen.query_one(selector)
            except Exception:
                continue
            if not widget.display:
                continue
            # ``region.height``, NOT ``size.height``. Textual sizes border-box:
            # ``size`` is the CONTENT box, and the rows the dock actually
            # reserves are the outer ones. ``#input-shell`` carries
            # ``padding: 1``, so the two differ by two rows there — and reading
            # the content box let the card claim those two rows twice. Measured
            # at 80x12: the dock came out one row taller than the screen, its
            # region started at y=-1, and ``virtual_size`` (78, 12) exceeded
            # ``size`` (78, 10) — the scrollable screen AGENTS.md calls always a
            # bug on this app, with the top of the question scrolled off.
            height = widget.region.height
            if height:
                reserved += height
                seen = True
        return reserved if seen else _DOCK_ROWS_FALLBACK

    def _card_width(self) -> int:
        """Content cells the card may use, measured against the terminal.

        The preferred floor applies only while it FITS: a minimum width is a
        preference and the terminal is not, so on a very narrow screen the card
        gives up its breathing margin and then the floor rather than overflowing.

        ``ASK_PADDING_CELLS`` must stay equal to the horizontal half of this
        card's ``padding`` in the stylesheet. They are two statements of one
        measurement, and when they disagree the card either wastes cells it was
        given or lays out lines wider than the panel can draw — which the panel
        resolves by clipping, silently.
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
        """Lines the card's BODY may draw, with the transcript's share reserved.

        Three limits, and the card takes the smallest. Each one exists because
        the other two do not cover its case:

        1. **What is left of the terminal** once the rest of the dock (composer,
           status band, subagent/todo panels) and this card's own padding are
           paid for. This is a hard limit — exceeding it does not make a taller
           card, it makes a clipped one, and Textual clips the TAIL, which is
           the footer the allocator buys first precisely because it is the only
           statement of how to leave.
        2. **The anchoring share** (:data:`PROMPT_HEIGHT_SHARE`): the card is a
           band over a conversation, never a replacement for it. This is what
           binds on a tall terminal, where limit 1 would happily hand a
           twelve-option question the whole screen — which is the modal
           behaviour this rework exists to remove.
        3. **The transcript floor** (:data:`MIN_TRANSCRIPT_ROWS`): a share of a
           short terminal still rounds to nearly all of it, so below roughly
           fifteen rows the share stops protecting anything and this takes over.

        A CAP and not a fill: the body is ``height: auto``, so a card with
        little to say still draws short, and a two-option question does not
        reserve room for twelve.

        There is no floor under the result. Zero is a legitimate answer on a
        terminal with nothing to spare, and it is the honest one: a plan of no
        lines draws no card, where a floor would lay out lines the region cannot
        paint and let the compositor decide which to lose (round 3, R9-R11).
        :meth:`_allocate` spends this budget line by line and never overdraws
        it, so nothing is clipped rather than the wrong thing being.
        """
        _, height = self._screen_size()
        # 1. What physically remains for this card, inside its own padding.
        #
        # ``MIN_TRANSCRIPT_ROWS`` is NOT what is subtracted here — that is the
        # anchoring rule, applied below. This subtracts ONE row, because the
        # transcript is ``height: 1fr`` and a flexible child still claims a row:
        # the dock and the transcript are siblings under the screen, so a dock
        # sized to the whole screen leaves the transcript to overflow it. Left
        # out, the dock came to 11 rows on a 10-row screen and the screen went
        # scrollable, which took the top of the question off the frame.
        available = max(
            0, height - self._dock_reserved_rows() - CARD_PADDING_ROWS - _TRANSCRIPT_MIN_ROWS
        )
        # 2. and 3. Both anchoring rules are CEILINGS on the card, so the card
        # takes the tighter of the two rather than the looser: they bind at
        # opposite ends of the size range (the share on a tall terminal, the
        # transcript floor on a short one), and taking the looser would let
        # whichever is slack at this size cancel the one that is doing the work.
        #
        # The share is taken over the rows the card and the transcript actually
        # DIVIDE — the screen less the composer and the dock band — rather than
        # over the whole screen. Over the whole screen the dock's rows come out
        # of the transcript's side of the split alone: measured at 100x30, a
        # four-option question took 19 of 28 rows and left the conversation 4,
        # which is the sliver this rework exists to prevent. Over the divisible
        # rows the same question takes 16 and leaves 7.
        divisible = max(0, height - self._dock_reserved_rows())
        share = int(divisible * PROMPT_HEIGHT_SHARE) - CARD_PADDING_ROWS
        floor = height - MIN_TRANSCRIPT_ROWS - self._dock_reserved_rows() - CARD_PADDING_ROWS
        # The anchoring caps are ceilings on a card that has room to spare. They
        # must not become a gag: on a short terminal both go to zero or below,
        # and a question the agent is parked on that draws NO LINES AT ALL is a
        # worse outcome than a thin strip of conversation. Measured at 40x10
        # before this floor: the card was not drawn at all and the turn waited
        # on an answer with nothing on screen to give it. So the two caps yield
        # to :data:`MIN_BODY_ROWS` — the question, one option, and the footer.
        #
        # ``available`` is applied LAST and outside that floor, because it is
        # the only one of the three that is not a preference. The floor says
        # what the card would like to keep; ``available`` says what the screen
        # physically has, and a floor allowed to win over it is not a taller
        # card but a clipped one — measured at 80x12, where a 3-row floor over
        # 2 rows of room put the dock one row past the screen and made the
        # screen scrollable, cutting the top off the question.
        anchored = max(min(share, floor), MIN_BODY_ROWS)
        room = max(0, min(available, anchored))
        # Never taller than it needs to be: the caps above are ceilings, and a
        # card that padded itself out to its allowance would push the
        # conversation up to show blank rows.
        #
        # `wanted` is the card at its FULL natural height — title and rule, the
        # wrapped question, both spacer rows, every option with its description
        # line, and the footer — and getting that wrong in the other direction
        # is a real defect rather than a rounding difference. Computed as the
        # MINIMAL card (one line per option, no descriptions, no spacers) this
        # cap sat below what `_allocate` needed to buy the comforts, so the
        # allocator was handed a budget that could never afford a description
        # and the card silently lost every second line: measured as options
        # drawn with no consequences under them, and a list that windowed at 13
        # options on a terminal with room for all of them.
        wanted = (
            2  # title and its rule
            + question_lines
            + 2  # the spacer above the list and below it
            + self.row_count * 2  # each option, plus its description line
            + 1  # the windowing line, where the list turns out to need one
            + 1  # the footer
        )
        return min(room, wanted)

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
        2. the FIRST LINE OF THE QUESTION — what is being asked;
        3. one option row — a question with no answers is not a question;
        4. the windowing line, whenever the window is short of the list, because
           a card quietly showing one of four has hidden three;
        5. the rest of the question, every wrapped line of it, marked ``…`` if
           even that cannot fit;
        6. the title and its rule, which travel together — a rule under a title
           is a caption, a rule under nothing is the edge of a box;
        7. the rest of the option rows;
        8. the blank spacers, which are rhythm and nothing else;
        9. the descriptions, all of them or none.

        **The question outranks the options, and that ordering is a safety
        property rather than a preference.** It used to sit below them, which
        was defensible while this card only ever asked the ``ask`` tool's
        questions: an option row at least says what one of the answers is. It
        stopped being defensible when the approval gate started using this same
        surface. Measured at 60x16 before this change, the approval card
        rendered exactly three lines:

            ❯ 1. Allow
            showing 1–1 of 3
            ↑↓ move · 1-9 jump · enter answer · esc deny

        — an authorisation prompt for ``rm -rf /Users/damian/project/data``
        that never names the tool, the command, or the target, with the cursor
        parked on *Allow*. A user cannot consent to something the card declines
        to state, and "Allow" without its object is worse than no card at all,
        because it looks answerable. The question is now bought immediately
        after the exit, so the last thing to go before the card gives up is
        what it is asking about (D1, design round 1).

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
            windowed = self._allocate(width, question, budget, position=True)
            # ...unless paying for it costs the QUESTION. The count is a
            # refinement of the answers on offer; the question is what the card
            # is for. Measured at 60x16 with a 3-row budget: buying the count
            # took the last row the question had, leaving `❯ 1. Allow` over
            # `showing 1–1 of 3` — a card that says how many answers it is
            # hiding while hiding what the answers are TO (D1). Where the two
            # compete, the question wins and the list stays honest by other
            # means: the option rows it did draw are still numbered.
            if windowed.question or not plan.question:
                plan = windowed
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
            # Too little for the question, one answer and the exit together.
            #
            # Two rows go to the QUESTION and the exit, not to an option row and
            # the exit. A bare `❯ 1. Allow` over `esc deny` is a prompt asking
            # for authorisation while refusing to say what it would authorise —
            # and it looks answerable, so the cursor sits on the permissive
            # option with nothing on screen to weigh it against (D1). Naming the
            # thing and stating how to leave is the honest minimum; the answers
            # come back the moment there is a third row to put them on.
            #
            # ONE row goes to the question, not to the exit.
            #
            # This is the same rule as the two-row case and it took a second
            # round to get right. A single `esc deny` row was reachable at every
            # width on a 13-row terminal, and it is the worst frame this card
            # can draw: it names nothing, and the answer letters still work — so
            # `y` approved `rm -rf /Users/x/project/data` from a card whose
            # entire content was the word for refusing (D9, design round 2).
            #
            # The exit is the one thing a user can always guess, and Escape is
            # the app's stop key everywhere regardless of what this footer says.
            # What they cannot guess is what they are being asked. So on a
            # single row the question wins and the footer goes.
            #
            # No rows at all is a card that cannot be drawn, and drawing it
            # anyway is the clip itself (round 4, R15).
            first = question[:1] if budget >= 1 else ()
            return _CardLayout(
                width=width,
                question=tuple(first),
                show_title=False,
                space_above=False,
                space_below=False,
                show_descriptions=False,
                page=0,
                show_position=False,
                show_footer=budget >= 2,
            )
        # The footer, the first line of the question, and one option row: the
        # three lines the card cannot say anything useful without. The question
        # line is charged here rather than out of `remaining` below, which is
        # what puts it ahead of the option rows in the priority order.
        remaining = budget - 3
        if position:
            remaining -= 1
        # The first line is already paid for above; the rest competes at step 5.
        kept = list(question[: remaining + 1])
        if len(kept) < len(question) and kept:
            # Say that the question continues. A silently halved question is
            # the one clip on this card the reader cannot detect: every other
            # abbreviation leaves a count, a caret or an empty gutter behind.
            tail = truncate_cells(kept[-1], max(1, width - 2))
            # `truncate_cells` marks its OWN cut, and two ellipses in a row read
            # as a rendering fault rather than as "there is more question".
            kept[-1] = f"{tail[:-1].rstrip() if tail.endswith('…') else tail} …"
        # Less ONE, because the question's first line was already bought above
        # with the footer and the option row. Charging `len(kept)` here would
        # bill it twice and hand the rest of the plan a budget short by a row.
        remaining -= len(kept) - 1
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
        """Take focus — unless the user is in the middle of typing.

        The card takes focus by default so its full keymap (arrows, Enter, the
        digits) works without a click, and remembers what held it so answering
        hands it back.

        It yields to a NON-EMPTY composer, because a question is raised by the
        AGENT and can land at any moment, including mid-sentence. Taking the
        caret then is not merely rude: the answer keys are ordinary characters,
        so the rest of the user's sentence starts landing on the card, and on an
        approval the first `y` AUTHORISES the call. Measured before this guard:
        typing `please ` and then `yes do it` through the mount approved
        `rm -rf /Users/x/project/data` and left `please es do it` in the buffer
        (D12, design round 3).

        Nothing is lost by yielding. The keys the card advertises are ROUTED
        from the composer (`OperatorApp.route_key_to_live_prompt`), the footer
        names only the ones that work from there, and clicking the card takes
        focus deliberately.
        """
        screen = self.screen
        self._restore_target = screen.focused if screen is not None else None
        if not self._composer_has_draft():
            self.focus()
        self._repaint()

    def _composer_has_draft(self) -> bool:
        """Whether the user has text in the composer right now.

        Defensive on purpose: this decides whether to take the caret, and a
        host with no composer (the reduced test harnesses) must fall through to
        the ordinary "take focus" path rather than raise out of a mount.
        """
        from local_operator.tui.widgets.editor import Editor

        screen = self.screen
        if screen is None:
            return False
        try:
            return bool(screen.query_one(Editor).text)
        except Exception:
            return False

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-measure: the width, the page size and the descriptions all come
        from the screen."""
        self._move_to(self.state.selected)

    def remeasure(self) -> None:
        """Re-run the layout against the CURRENT screen, from outside.

        A hidden widget is not laid out, so it receives no ``Resize`` event —
        which means a card that hid itself on a terminal too short to draw it
        could never learn that the terminal had grown back. The question stayed
        invisible for the rest of the turn while the tool went on waiting: a
        shrink was a one-way door onto a permanently unanswerable prompt (D10,
        design round 2).

        The app calls this on every terminal resize instead, because the app
        still gets the event when this widget does not.
        """
        self._repaint()

    def _repaint(self) -> None:
        body = getattr(self, "_body", None)
        if body is None or not body.is_mounted:
            return
        text = self._card_text()
        body.update(text)
        drawable = bool(text.plain)
        card = body.parent
        if card is not None:
            # A card with no drawable line is not a smaller card: its padding is
            # rows the screen has not got, so at three terminal rows and under
            # the container pushed the screen's virtual height past its size — a
            # scrollable screen, which AGENTS.md calls always a bug here, over a
            # card painting nothing. Nothing to draw, nothing laid out; the next
            # resize brings it back.
            card.display = drawable
        # And the same for THIS widget, which is the one that carries the
        # padding now that the card is docked rather than modal. Hiding only the
        # inner holder left the outer one claiming its own two padding rows
        # around zero rows of content: measured at 40x10, that was two rows of
        # pure chrome for a card drawing nothing, and `virtual_size` (38, 9)
        # over `size` (38, 8) — a scrollable screen, which AGENTS.md calls
        # always a bug here.
        #
        # The HOST's own row is not this widget's to write, and an earlier
        # revision that wrote it directly was a bug: the app shows the host when
        # it mounts a prompt, so two owners disagreed and which won depended on
        # ordering. The card ANNOUNCES instead, and the app (the single writer)
        # decides. A message rather than a direct call because the answer
        # changes on any repaint — a resize, a question advancing, a typed
        # character rewrapping the card — and each of those has to be able to
        # bring the host's row back as well as take it away.
        was_drawable = self._drawable
        self._drawable = drawable
        self.display = drawable
        if was_drawable != drawable and self.is_attached:
            self.post_message(self.DrawableChanged(self))

    def answer_keys(self) -> frozenset[str]:
        """Keys that answer this card directly from the COMPOSER.

        Deliberately narrow. The card holds focus in the ordinary case and its
        full keymap works there; this is only the set worth intercepting while
        the caret is in the composer, where every character is otherwise the
        user's text. On the ``ask`` picker that is the row ordinals — a digit
        with an empty composer is unambiguous — and nothing else: Enter is left
        to the composer because it SUBMITS a prompt there, and Escape is the
        app's stop key.

        The FREE-TEXT row is excluded, because a digit cannot answer it: it is
        answered by typing into it, which needs the card to hold the caret. Left
        in, the footer advertised `1-3 answer` on a three-row card whose third
        row was `Other` — and pressing `3` selected it and then refused, since
        there is nothing typed to accept. A hint that names a key which lands on
        a dead end is the same defect as one that names a key that does nothing.

        Empty while the card is drawing no rows, so a key can never commit an
        answer the user was not shown (the rule :meth:`action_accept` follows).
        """
        if not self.visible_rows:
            return frozenset()
        return frozenset(
            str(index + 1) for index in self._window() if index < 9 and index != self.other_row
        )

    def routed_hint(self) -> tuple[str, str] | None:
        """How the footer names the keys that still work from the composer.

        A range for the ordinals (`1-3 answer`), because they are contiguous
        and a list of them would cost more width than it explains.
        """
        digits = sorted(key for key in self.answer_keys() if key.isdigit())
        if not digits:
            return None
        span = digits[0] if len(digits) == 1 else f"{digits[0]}-{digits[-1]}"
        return (span, "answer")

    def answer_from_key(self, character: str) -> None:
        """Take ``character`` as an answer routed from the composer."""
        if character.isdigit():
            self.action_jump(int(character))
            self.action_accept()

    @property
    def is_drawable(self) -> bool:
        """Whether the card has any line to paint at this terminal size.

        False on a terminal too short to show even the footer, where the card
        hides itself rather than laying out rows the screen cannot draw. The
        host reads this to decide whether to keep its own separation row, so
        that row cannot outlive the card it separates.
        """
        return self._drawable

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings — what a user reads.

        Empty when the body has no drawable line at all, because ``Text``
        splits an empty card into ONE empty line and a caller counting that
        against the room the screen has would be told the card overflows a
        terminal it is painting nothing into.

        Also empty while the card is HIDDEN. This method re-derives the text
        rather than reading back what was painted, so on a terminal too short
        to draw the card it would otherwise report the line the card WOULD
        have drawn — and a caller then asserts that line reached the terminal,
        which it never did. Measured as an intermittent failure at 30x12 where
        the card reported `esc skip` against a frame that contained only the
        composer. Whether the card is drawn is `display`'s answer, so this
        defers to it rather than keeping a second opinion.
        """
        if self.is_mounted and not self.display:
            return []
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
                out.append_text(self._description_text(index, width, ground, muted, muted))
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
        hints = self._footer_hints(width) if drawn else [self._exit_hint]
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
        title = self._title
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

    def row_key(self, index: int) -> str:
        """The letter that answers row ``index`` directly, or ``""``.

        Empty on this class: the ``ask`` picker's rows are addressed by ordinal
        and have no letters of their own. :class:`ApprovalPrompt` overrides it,
        because its three rows DO answer to letters (`y`/`n`/`A`) that predate
        the list and that people have in their fingers.

        It exists here so the gutter has one place to ask, rather than the
        renderer growing a branch on which subclass it is drawing.
        """
        return ""

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

        A row with a LETTER of its own shows that instead. On the approval card
        the letters are the older interface — `y`/`n`/`A` predate the list and
        people have them in their fingers — and after the rework they were live
        bindings rendered nowhere, so "allow all" had lost its only discoverable
        shortcut (D4, design round 1). Shown in the gutter the ordinal would
        occupy, they cost no width, and the ordinals still work for anyone who
        counts rows instead.
        """
        key = self.row_key(index)
        if key:
            return f"{key}."
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

        Drawn at ``muted``, which measures 6.51:1 on this card's ``overlay``
        ground. It has been walked up this ramp twice for the same reason: the
        first captured frame had it at ``faint`` (1.49:1, barely present), and
        it then sat at ``dim`` for a release — 3.43:1, which is under the 4.5:1
        WCAG AA floor for body text (D7, design round 1).

        The floor is the right test rather than a nicety, because of what this
        line CARRIES: it is the consequence of choosing the row, which is the
        thing the user is reading the card to compare — and on the approval
        prompt it is the difference between "ask again next time" and "stop
        asking for this session". Text that decides an authorisation cannot be
        the least legible text on the card.

        It stays a step below the LABEL (``fg``, 11.30:1), so the ranking
        between a row's name and its explanation is intact; what changed is
        that the explanation is now readable in absolute terms and not only
        relative to the label above it.

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
        # While the CARD does not hold the caret, most of this keymap is not
        # reachable: the composer has focus and takes the arrows, Enter and the
        # printable keys as text. Only what the app routes still works — the
        # row ordinals and Escape — so those are all the footer claims.
        #
        # A footer describing one keyboard while the caret sits on another is
        # the same lie this row already refuses to tell about `enter` on a
        # collapsed card. Measured from the composer: `↑↓` moved nothing and
        # `enter` answered nothing, while both were advertised (D13, design
        # round 3).
        if not self.has_focus:
            hints = []
            # ...and not even those while the composer holds a DRAFT. The
            # routing stands down whenever there is text in the buffer, so
            # every character is the user's again — which is exactly the state
            # the mount-time focus yield creates (D12), so the keys would be
            # advertised precisely where they are dead (F6, agent review round
            # 4). The exit survives, because Escape works from anywhere.
            routed = None if self._composer_has_draft() else self.routed_hint()
            if routed is not None:
                hints.append(routed)
            hints.append(self._exit_hint)
            return hints

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
            # Where the rows carry LETTERS of their own, the footer names those
            # instead of the digit range: the gutter is showing `y`/`n`/`A`, and
            # a hint reading `1-9 jump` beside it describes a different keyboard
            # from the one on screen.
            hints = [("↑↓", "move"), (jump, "jump"), ("enter", "answer")]
            ladder = ["↑↓", jump, "enter", "esc"]
            # Rows that carry their own LETTER need no range hint at all: the
            # letter is printed in each row's gutter, next to the label it
            # answers, which says it better than a footer can. Repeating it here
            # produced `y/n/A answer · enter answer` — two hints claiming the
            # same verb, and the digit range would have been a claim about a
            # keyboard the card is not showing.
            if any(self.row_key(index) for index in self._window()):
                hints = [pair for pair in hints if pair[0] != jump]
                ladder = [key for key in ladder if key != jump]
            # ...and only where there is somewhere to jump TO. The range was
            # advertised unconditionally, so a card windowed down to a single
            # row still offered `1-9 jump` — verified live on a 3-row approval
            # card, where `5`, `7` and `9` did nothing at all (D3, design round
            # 1). A key offered where it does nothing is the same lie the
            # collapsed card's footer already refuses to tell about `enter`.
            #
            # Keyed on the DRAWN page rather than on `row_count`: the digits
            # address rows by ordinal, and a row that is not on screen is one
            # the user cannot see they are committing to.
            drawn = len(self._window())
            if drawn < 2:
                hints = [pair for pair in hints if pair[0] != jump]
                ladder = [key for key in ladder if key != jump]
        hints.append(self._exit_hint)

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
