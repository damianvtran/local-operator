"""The ``/btw`` aside — a side conversation that leaves no trace.

**What an aside IS.** A question you can ask the agent about the session
without it becoming part of the session: "why did you pick that model", "are
those subagents stuck", "what does snapcompact mean". The agent answers from
the context it already has — the same system blocks and the same message list a
real turn is built from — and then the question, the answer and the whole
exchange are thrown away. Esc puts you back in the main chat with the
conversation, the model's view of it, and your half-typed prompt exactly as you
left them.

**Why it has to be that way.** The gesture the surface promises is "step out of
the room, ask, come back". If the exchange were quietly appended to history,
Esc would have committed words to the model's context that the user never
decided to send — and a mid-turn "wait, what are you doing?" would become a
permanent instruction the agent has to reconcile with the actual task on every
later turn. It would also cost tokens forever, on a question whose whole point
was that it was not the work. So the aside READS the conversation and never
writes to it (:meth:`SessionProtocol.complete_aside` enforces the write half).

**The one door out is the user's.** ``^F`` forks the exchange into the chat:
every settled turn is appended as an ordinary user/assistant pair, to the live
context and the transcript both, and the aside closes. That is the "forking"
half of the feature — the user decides when a side thread becomes part of the
record, rather than the app deciding for them in either direction. The
reference implementation (``omp``'s ``/btw`` with ``b branch to chat``) draws
the line in the same place; it branches the session where this appends,
because this app has no session branch tree to cut.

There is a second door, and it opens somewhere else. The copy key
(:data:`ASIDE_COPY_KEY`) lifts the exchange to the CLIPBOARD, which is outside
the session entirely — nothing joins the context and nothing joins the
transcript, so the claim above ("reads the conversation and never writes to
it") is untouched by it. It is stated here because a reader who finds two keys
under a sentence saying "the one door" will assume the sentence is stale. The
distinction is the whole point of having both: fork is how the user chooses to
put the exchange ON the record, copy is how they keep a sentence without doing
that, and copy is the only one of the two that works while the answer is still
streaming.

**The card scrolls in ROWS.** It used to scroll in whole turns, which meant a
turn was the smallest thing any gesture could address and an answer taller than
the card had a middle no gesture could reach at all — 191 of 200 rows at 80x24,
with nothing on screen admitting the text had been cut. The wheel is still the
card's own gesture; the keyboard path is a chord the app binds
(:meth:`AsidePanel.scroll_page`), because this card takes no focus and so can
hold no bindings of its own.

**No trace means no trace on dismissal either.** Esc discards the exchange, so
reopening the aside opens an empty one. A surface that says "off the record"
and then hands back what you discarded is lying about which of the two it is.

**No second composer.** The card is not focusable and owns no input. The app
points the ONE editor at it while it is open (placeholder and all) and stashes
the main draft, which is what makes "keep chatting inside the aside" and "Esc
gives me my prompt back" the same mechanism instead of two.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import overlay
from local_operator.tui.widgets.assistant import flatten
from local_operator.tui.widgets.tool_card import truncate_cells

#: The card takes the COMPOSER's width and the composer's left and right
#: edges (``overlay.composer_column``), not a measure of its own. It used to be
#: an 88-cell card centred over the transcript, and that was the wrong
#: relationship drawn correctly: the aside's input IS the composer directly
#: below it, so two elements that are one unit read as a floating dialog above
#: an unrelated dock. Sharing a column is what says they belong together.
#:
#: Prose therefore wraps at whatever the composer is wide, exactly as the
#: transcript's own blocks do — measured at 120 columns the card's answer rows
#: and the transcript's assistant rows end on the same cell. A narrower measure
#: inside a full-width card would put a ragged right edge in the middle of a
#: surface whose edges are the whole point. Capping the MEASURE is the
#: composer's job, not this widget's: ``#input-shell`` already carries a
#: ``max-width`` in the boot layout and the card inherits it for free through
#: ``overlay.composer_column`` (82 cells at 120 columns, 100 at 180).
PANEL_MIN_WIDTH = 24

#: Ground the card keeps above the screen's top inset, and the inner padding —
#: declared here as well as in the stylesheet because the widget sizes ITSELF
#: and Textual's width/height are border-box. A change to ``AsidePanel``'s
#: ``padding`` in the tcss MUST change these.
#:
#: One cell of horizontal padding matches ``#input-shell`` exactly, which puts
#: the card's ``▌`` spine in the same cell as the composer's ``❯`` at every
#: width and in both layouts.
#:
#: ONE padding row on every side (``padding: 1``). The card briefly had no
#: bottom row, on a measurement that was right and a conclusion one level too
#: coarse: the seam between the hint row and the ``❯`` does measure 2 rows,
#: but the fill change falls BETWEEN them, so the card contributes one and the
#: composer contributes one and neither surface contains a 2-row interval. The
#: unit that has to stay on a 1-row rhythm is each SURFACE, not the cell column
#: across a fill boundary.
#:
#: What the missing row actually cost, measured at 120x40: the hint row was
#: simultaneously the last text row and the card's LAST row, so the fill
#: terminated on an inked line while every other row in the card had fill above
#: and below it — and the row denied that air was the one telling the user how
#: to leave. With the dock band populated it was worse: the hint row abutted
#: the band's own heading with zero rows between two different widgets.
PANEL_HEIGHT_MARGIN = 2
PANEL_PADDING_CELLS = 2
PANEL_PADDING_ROWS = 2

#: Rows the pinned chrome costs: title, the rule under it, the blank above the
#: footer, and the footer. Pinned rather than scrolled because the footer
#: carries the way out, and a card that scrolled its own exit away would trap a
#: reader exactly when the exchange got long enough to need it.
CHROME_ROWS = 4

#: Rows above the dock at which the card can still afford its vertical gutter.
#: Below it the gutter is spent rather than the prompt covered — and
#: ``-squeezed`` now drops BOTH padding rows symmetrically rather than the
#: asymmetric 1/0 the missing bottom row used to produce.
SQUEEZE_ROWS = PANEL_HEIGHT_MARGIN + CHROME_ROWS + PANEL_PADDING_ROWS + 2

#: The prompt the side question is wrapped in. Three instructions, each earning
#: its line: OFF THE RECORD so the model does not treat the question as a new
#: task and start narrating a plan; TEXT ONLY because the request does carry
#: the live tool catalogue (it has to, to stay on the working turn's cache
#: prefix) and on Anthropic even ``tool_choice`` reads ``auto`` on the wire
#: (see ``Session.complete_aside``), so the prompt is the model-facing half of
#: "calls nothing" and a call it makes anyway is discarded unread; and
#: answer-from-context because the whole reason to ask here rather than in the
#: chat is that the agent already knows.
ASIDE_PROMPT = """<aside>
The user has stepped aside to ask you something about this session. This is OFF
THE RECORD: neither their question nor your answer joins the conversation, and
no work is being asked for. Answer from the context you already have, briefly
and directly, in prose. Answer in text only: do not call any tool (a tool call
here is discarded unread), do not propose a plan, and do not ask a follow-up
question. If your context does not answer it, say so plainly.
Question:
{question}
</aside>"""

#: Marker on a question row, and the indent every line of prose sits at.
#:
#: ``▌`` at two cells is the TRANSCRIPT's mark for a row the user SENT
#: (``UserBlock.RULE`` / ``SPINE_INDENT``), not the composer's ``❯``, which
#: marks a row being typed. An aside question is sent the moment Enter lands,
#: so the card uses the same spine at the same column — and a forked exchange
#: then renders in the transcript as it looked on the card, which is what
#: "keep this" ought to look like. The ANSWER holds up its half of that by
#: rendering as markdown here too (:meth:`AsidePanel._markdown_rows`); it used
#: to arrive as literal ``**bold**``, so identical words changed shape the
#: moment ``^f`` moved them. The card is already told apart from the
#: transcript by the two things the design system reserves for that (the
#: ``$lo-overlay`` elevation step and the title row); a third differentiator
#: spelled in a glyph that already means something else would be a second
#: grammar for one idea.
QUESTION_MARK = "▌"

#: Prose indent, matching the mark's width so question and answer share one
#: text column with the gutter reserved for the speaker.
ANSWER_INDENT = "  "

#: The key the FOOTER advertises for copying the exchange out. The binding
#: itself is the app's (the card is ``can_focus = False`` and holds none), so
#: this constant exists to stop the label and the binding drifting apart — a
#: footer naming a key that does nothing is worse than no footer, because the
#: user stops trusting the row that also tells them how to leave.
#:
#: ``ctrl+r`` and not the reference implementation's bare ``c``: a printable
#: character can never reach app level here, because the composer holds focus
#: by contract and Textual's ``TextArea`` consumes the character first — the
#: "only act on an empty draft" guard that makes a bare letter work there is
#: unreachable once the letter is already in the buffer. ``ctrl+y`` was the
#: other candidate and is taken by ``TextArea``'s Redo.
ASIDE_COPY_KEY = "ctrl+r"

#: Failure marker. The app-wide warning glyph rather than the words "aside
#: failed" — the card is titled ``Aside``, the row is in danger ink, and the
#: position is under the question, so a fourteen-cell prefix restates three
#: things the reader already has and pushes the actual cause right.
ERROR_MARK = "!"


@dataclass
class AsideTurn:
    """One question and the answer to it. The unit the card renders and forks."""

    question: str
    answer: str = ""
    state: Literal["running", "done", "error", "cancelled"] = "running"
    #: Set only for ``error``; kept apart from ``answer`` so a failed turn can
    #: never be forked into the chat as if the model had said it.
    error: str = ""

    @property
    def forkable(self) -> bool:
        """Whether this turn is a real exchange, worth promoting to the chat."""
        return self.state == "done" and bool(self.answer.strip())


@dataclass
class AsideBody:
    """The visible rows, and how many whole QUESTIONS sit above them.

    Questions, not lines. A user remembers asking three things; they never
    counted the rows an answer wrapped to, so a line count names a quantity
    they cannot check against anything.

    That reasoning covers whole turns the window has scrolled past. It does
    NOT extend to rows cut out of the middle of ONE answer: there is no
    question to count there (the count is zero, which is why the card used to
    say nothing at all), and the reader's question is "how much of this answer
    am I missing", which only a row count answers. ``_body`` therefore names
    lines in that case and questions in this one; see the two markers there.

    Counted from the top of the window only. The old count also included turns
    hidden BELOW it, so a card scrolled to the oldest question announced "5
    earlier questions" under an arrow pointing up at nothing.
    """

    lines: list[Text] = field(default_factory=list)
    hidden_turns: int = 0


@dataclass
class _FlatBody:
    """Every row of the exchange in one list, plus who owns each row.

    The card scrolls in ROWS, so the row list is the thing it windows over and
    the turn structure survives only as an index beside it. It has to survive:
    a window that opens inside an answer must still show the question that
    produced it (see :meth:`AsidePanel._window`), and after the rows are
    flattened there is nothing in a row itself that says which question that
    was.
    """

    lines: list[Text] = field(default_factory=list)
    #: Per ROW, the index of the turn it belongs to.
    owners: list[int] = field(default_factory=list)
    #: Per TURN, the half-open row span of its QUESTION — the rows pinned when
    #: a window opens mid-answer. Excludes the blank that separates turns:
    #: that blank belongs between two turns, and leading the card with it reads
    #: as the exchange having started and then said nothing.
    heads: list[tuple[int, int]] = field(default_factory=list)


class AsidePanel(Static):
    """The floating card: the exchange so far, and the keys that leave it.

    Holds no focus and no input (see the module docstring), so it posts no
    messages — every key that acts on it is the app's, because every one of
    them also acts on the composer or the session. What it owns is the
    exchange, and one question at a time: a ``generation`` counter retires a
    request whose answer is no longer wanted, which a Textual worker
    cancellation alone cannot do (workers and messages are separate queues).
    """

    can_focus = False

    def __init__(self) -> None:
        super().__init__(id="aside-panel")
        self._turns: list[AsideTurn] = []
        self._generation = 0
        #: Told the card's settled height on every paint, so the host can keep
        #: the ground under the conversation clear. Installed while the card is
        #: open and cleared with it; the app owns the transcript, not the card.
        self._on_height: Callable[[int], None] | None = None
        #: Whether the footer may advertise ``^f``. The APP's answer, not the
        #: card's: forking needs a session and is refused mid-turn, and a card
        #: that decided for itself would advertise a key that fails.
        self._can_fork = False
        #: A refusal the card has to state — ``^f`` pressed with nothing to
        #: fork, or into a running turn. It lives HERE and not in the
        #: transcript: a warning row appended to the conversation by a card
        #: whose title says nothing here joins it is exactly the trace the
        #: aside promises not to leave, and while the card is up the row would
        #: be drawn behind it, so the user finds it only after dismissing.
        self._notice = ""
        #: How many ROWS back from the tail the reader has walked. The wheel
        #: moves it; ASKING snaps it back to 0, because the user's own new
        #: question is a re-acquire. Streaming does NOT: the same three-state
        #: rule the transcript follows (:class:`TailAnchor`), in this card's
        #: units — a reader who walked back mid-answer is reading, and the
        #: deltas must not drag them forward (see :meth:`append_answer`).
        #:
        #: Rows and not turns, which is the fix for the defect this card
        #: shipped with: a turn is the coarsest possible unit and an answer
        #: taller than the card has no unit smaller than itself, so the middle
        #: of one was addressable by no gesture at all — measured at 191 of 200
        #: rows at 80x24. The name carries the unit because getting it wrong
        #: is precisely what went wrong.
        self._scroll_back_rows = 0
        #: Screen size plus the live dock ceiling at the last paint. The dock
        #: grows as the composer wraps without resizing this card, so Textual
        #: emits no resize event for it and the app has to ask.
        self._layout_shown: tuple[int, int, int] | None = None
        #: Rendered answer rows, keyed by the three things they depend on:
        #: ``(text, width, theme epoch)``. See :meth:`_markdown_rows` for why
        #: it is here and :meth:`_body` for why it cannot grow.
        self._answer_cache: dict[tuple[str, int, int], list[Text]] = {}
        #: Keys :meth:`_markdown_rows` reached during the paint in progress.
        self._used_answers: set[tuple[str, int, int]] = set()
        #: Bumped by every edit to the exchange — a question appended, a delta
        #: streamed, an answer settled or failed, the card opened or closed.
        #: It is what tells :meth:`_flat_body` its memo is stale, so the
        #: counter has to move on EVERY mutation; a missed bump paints the old
        #: rows. Mutating ``_turns`` without going through one of those methods
        #: is therefore not supported, and the tests that assign ``_turns``
        #: directly go through :meth:`_invalidate_flat` for the same reason.
        self._revision = 0
        #: The last :meth:`_flat_body` result and the inputs it was built from,
        #: ``(width, theme epoch, revision)``. Flattening renders every turn,
        #: so without this a scroll — which changes none of those three — pays
        #: to re-render the whole exchange to learn row counts it already had.
        self._flat_memo: tuple[tuple[int, int, int], _FlatBody] | None = None
        self.display = False

    def _invalidate_flat(self) -> None:
        """Mark the flattened exchange stale. Cheap, and called on every edit."""
        self._revision += 1
        self._flat_memo = None

    # -- state ---------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return bool(self.display)

    @property
    def turns(self) -> list[AsideTurn]:
        return list(self._turns)

    def set_notice(self, text: str) -> None:
        """State a refusal on the card. ``""`` clears it."""
        if text == self._notice:
            return
        self._notice = text
        self._repaint()

    def open(self) -> None:
        """Show an EMPTY aside. Reopening never restores a dismissed exchange."""
        self._turns = []
        self._notice = ""
        self._scroll_back_rows = 0
        self._answer_cache.clear()
        self._invalidate_flat()
        self._generation += 1
        self.display = True
        self._repaint()

    def close(self) -> None:
        """Hide the card and discard the exchange — that is what dismiss means.

        The rendered-answer cache is discarded WITH it. A dismissed exchange
        that lives on in a cache is the trace this surface promises not to
        leave, and the paint-scoped prune cannot reach it: ``_repaint`` returns
        early on a hidden card, so nothing would run to drop it.
        """
        self._generation += 1
        self._turns = []
        self._notice = ""
        self._scroll_back_rows = 0
        self._answer_cache.clear()
        self._used_answers.clear()
        self._invalidate_flat()
        self.display = False

    def ask(self, question: str) -> int:
        """Append a question in its pending state; returns the request identity.

        The previous question is not waited for. A user who asks again while an
        answer is still streaming has moved on, so the older turn is retired
        (``cancelled``) and stays on the card as the record of what they asked.
        """
        self._generation += 1
        for turn in self._turns:
            if turn.state == "running":
                turn.state = "cancelled"
        self._notice = ""
        # Zero is the TAIL in either unit, which is why all three reset sites
        # kept their line through the turns-to-rows change: "snap to the
        # newest" is what they mean and the offset counts back FROM the newest.
        self._scroll_back_rows = 0
        self._turns.append(AsideTurn(question=question))
        self._invalidate_flat()
        self.display = True
        self._repaint()
        return self._generation

    def accepts(self, generation: int) -> bool:
        """Whether a worker's result still belongs to the visible question."""
        return self.is_open and generation == self._generation

    def append_answer(self, generation: int, delta: str) -> None:
        if not self.accepts(generation) or not delta:
            return
        # Measured BEFORE the delta lands, and only while the reader is parked
        # away from the tail. The offset counts back FROM the tail, so rows
        # arriving at the tail slide the window forward under a reader who is
        # holding still: measured, a reader parked 120 rows back watched their
        # top row walk from ANSWER-ROW-067 to ANSWER-ROW-127 across 60 deltas
        # while the offset itself never changed. Holding the NUMBER still is
        # not the rule — holding the ROWS still is.
        anchored = len(self._flat_body().lines) if self._scroll_back_rows else 0
        self._turns[-1].answer += delta
        # BETWEEN the two measurements, and that placement is the whole point:
        # `anchored` is the pre-delta height and the count below is the
        # post-delta one, so the memo has to be dropped here or the second
        # `_flat_body()` returns the first one's rows and the difference is
        # always zero — which is exactly the drift this branch exists to undo.
        self._invalidate_flat()
        if anchored:
            # NOT reset, and now not drifted either. `ask` already put the
            # reader on the new question, so the only way the offset is
            # non-zero mid-answer is that they wheeled back on purpose — and
            # dragging them forward on every delta is the same bug the
            # transcript had, in this card's units. They re-acquire by
            # wheeling back down to the tail.
            #
            # Under the old TURN-index model this whole branch was dead within
            # one answer: a streaming answer is one turn, so max scroll-back
            # was 0 and the offset could not be non-zero to begin with. In rows
            # the state is real for the first time, which is why the rule the
            # comment above protects needed code behind it and not just a
            # comment. Clamped by `_window`, so a shrinking settle cannot strand
            # the offset past the top.
            self._scroll_back_rows += len(self._flat_body().lines) - anchored
        self._repaint()

    def settle_answer(self, generation: int, answer: str) -> None:
        """Adopt the authoritative text and mark the turn done.

        An empty answer is not an instruction to erase what streamed — the same
        rule ``AssistantBlock`` applies at ``message_end`` — but an exchange
        with nothing in it is reported rather than left looking answered.
        """
        if not self.accepts(generation):
            return
        turn = self._turns[-1]
        if answer.strip():
            turn.answer = answer
        turn.state = "done" if turn.answer.strip() else "error"
        if turn.state == "error":
            turn.error = "the model returned nothing"
        # The settled text routinely differs from what streamed (and is often
        # shorter), and the state row under it changes with `turn.state`, so
        # both the row COUNT and the rows themselves can move here.
        self._invalidate_flat()
        self._repaint()

    def fail_answer(self, generation: int, message: str) -> None:
        """A failure stays IN the card, next to the question it belongs to."""
        if not self.accepts(generation):
            return
        turn = self._turns[-1]
        turn.state = "error"
        turn.error = message
        # The error rows are part of the answer block, so this changes height.
        self._invalidate_flat()
        self._repaint()

    def fork_messages(self) -> list[tuple[str, str]]:
        """``(question, answer)`` for every turn worth promoting to the chat.

        Failed and cancelled turns are dropped: forking is "keep this exchange",
        and half an exchange is not one.
        """
        return [(turn.question, turn.answer) for turn in self._turns if turn.forkable]

    def copy_text(self) -> str:
        """The exchange as plain text, for the clipboard. ``""`` if empty.

        Off the DATACLASS, exactly as :meth:`fork_messages` is, and never off
        the painted rows — the rows are the windowed subset the reader can
        already see, and they carry the card's chrome, which is the thing
        ``Chrome.ALLOW_SELECT`` exists to keep out of the clipboard. A copy key
        that returned the screen would fail on precisely the long answer that
        makes someone reach for it.

        Includes a RUNNING turn's partial answer, unlike ``fork_messages``.
        Forking refuses a half exchange because it writes to the record and
        half an exchange is not one; the clipboard is the user's own scratch
        space, they can see the answer is still arriving, and this is the one
        way out that works while ``^f`` is refused mid-stream.

        Filtered on STATE, not on whether text happens to be present. A turn
        that streamed a few sentences and then failed still HOLDS those
        sentences, and copying them hands the user text the model never stood
        behind, formatted exactly like an answer it did — the reason
        :meth:`fork_messages` drops the same turns, and the reason ``error``
        keeps its cause in its own field rather than in ``answer``. Cancelled
        turns go for the same reason: the user moved on, and the card marks
        them ``(superseded)`` on screen while the clipboard could not.
        """
        blocks: list[str] = []
        for turn in self._turns:
            if turn.state not in ("done", "running"):
                continue
            answer = turn.answer.strip()
            if not answer:
                continue
            blocks.append(f"{turn.question.strip()}\n\n{answer}")
        return "\n\n".join(blocks)

    # -- mouse ----------------------------------------------------------------
    # Every gesture is STOPPED, because the card floats over the transcript:
    # left to bubble, one scroll would move both the aside and the chat behind
    # it and a click would focus whatever happens to sit underneath. The same
    # stop the toast and the usage card make.
    #
    # A click is stopped and nothing more. The card owns no input by contract
    # (the one composer is pointed at it), so there is no hit area to offer.
    #
    # The WHEEL is the one gesture the CARD itself binds. Scroll KEYS were
    # rejected here because ↑/↓ belong to the focused composer's prompt history
    # and the aside's whole premise is that the user keeps typing there — that
    # still holds, and it is why the keyboard path is a chord bound at APP level
    # (:meth:`scroll_page`) rather than a binding on this card, which is
    # ``can_focus = False`` and would never receive one anyway.
    #
    # One row per wheel event, because a row is now the unit: the card used to
    # step whole TURNS, which is why an answer taller than the card had a middle
    # no gesture could reach.
    def on_click(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()

    def on_mouse_scroll_down(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(-1)

    def on_mouse_scroll_up(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(1)

    def scroll_page(self, *, down: bool) -> bool:
        """Page the body from the KEYBOARD; ``True`` if the window moved.

        ``TodoPanel.scroll_expanded``'s shape, for its reason: the card is
        non-focusable, so a focus-then-arrow gesture cannot reach it and the
        content its own ``↑ … · scroll`` marker names would be MOUSE-ONLY — the
        wheel is not delivered under ``tmux set -g mouse off``, on terminals
        with mouse reporting disabled, under ``screen(1)`` or on non-SGR
        terminals. It drives the same :meth:`_scroll_by` the wheel drives so the
        two gestures cannot diverge.

        ``down`` is toward the NEWEST rows, matching the wheel's sense of the
        word and ``TodoPanel``'s.

        A page is the rows the card is SHOWING, not the rows it budgeted for.
        The two differ here: the window's top rows are OVERLAID by the marker
        and the pinned question (:meth:`_window`), so paging by ``_fit()[2]``
        would step over exactly the rows the overlay was covering — which are
        the rows this gesture exists to reach, and measured through the real
        app it left 42 of 200 rows visible to the wheel and not to the key.
        ``usage_panel.py:797-801`` hit the same trap on its block headings and
        wrote the rule down; do not "simplify" this to the budget.
        """
        if not self.is_open:
            return False
        flat = self._flat_body()
        budget = self._fit()[2]
        if len(flat.lines) <= budget:
            return False
        before = self._scroll_back_rows
        if down:
            # FORWARD steps by what the DESTINATION will show, not by what the
            # current window shows. The two differ: the overlay covers rows at
            # the window's TOP, so paging back leaves the covered rows below
            # the new window's top and the next step re-reads them — backward
            # self-corrects. Forward moves the top the other way, so a step
            # measured here lands past rows the destination will cover, and
            # they are never painted at any offset. Measured at budget 8 the
            # jump 51 -> 43 stepped 8 while 6 rows were visible, and rows 8-9
            # fell in the gap. Paging must be reversible or the two directions
            # disagree about which rows exist.
            self._scroll_by(-self._step_to(flat, budget))
        else:
            self._scroll_by(self._visible(flat, budget))
        return self._scroll_back_rows != before

    def _step_to(self, flat: _FlatBody, budget: int) -> int:
        """Rows to move FORWARD so the destination window abuts this one.

        Solved by trying the candidate step and asking what the window there
        would show, rather than by inverting the arithmetic: the overlay's size
        depends on which turn the destination's top lands in, so the step and
        its own consequence are mutually defined. Two passes settle it — the
        first guesses with this window's overlay, the second corrects with the
        destination's — and the result is clamped so a step can never exceed
        the window's own span and skip rows outright.
        """
        step = self._visible(flat, budget)
        for _ in range(2):
            probe = max(0, self._scroll_back_rows - step)
            candidate = self._visible(flat, budget, back=probe)
            if candidate >= step:
                break
            step = candidate
        return max(1, step)

    def _visible(self, flat: _FlatBody, budget: int, back: int | None = None) -> int:
        """Rows of the exchange a window actually shows to the reader.

        The window spans ``budget`` rows, but its first rows are covered by the
        overlay, and a covered row has not been read. Paging by this number is
        what makes the keyboard reach every row the wheel reaches.

        ``back`` measures a window the reader is not at yet, which is how the
        forward step is sized against its own destination.
        """
        rows, first, end, _ = self._window(flat, budget, back=back)
        covered = sum(1 for row, source in zip(rows, flat.lines[first:end]) if row is not source)
        return max(1, (end - first) - covered)

    def _scroll_by(self, delta: int) -> None:
        """Move back from the tail by ROWS, CLAMPED — the newest row is home."""
        target = max(0, min(self._max_scroll_back(), self._scroll_back_rows + delta))
        if target == self._scroll_back_rows:
            return
        self._scroll_back_rows = target
        self._repaint()

    def _max_scroll_back(self) -> int:
        """Rows the window can walk back before its top IS the exchange's top.

        Rows and not turns. ``len(turn_groups) - 1`` was the defect: with one
        turn it is 0, so a 200-row answer in a 16-row card had every gesture
        clamped at home and 184 rows addressable by nothing. Total rows minus
        the budget is the offset at which the window's top is row 0, which is
        what "as far back as there is anything to go" actually means.
        """
        return max(0, len(self._flat_body().lines) - self._fit()[2])

    # -- geometry -------------------------------------------------------------
    def _screen_size(self) -> tuple[int, int]:
        return overlay.screen_size(self)

    def _rows_above_dock(self) -> int:
        return overlay.rows_above_dock(self)

    def _composer_column(self) -> tuple[int, int]:
        x, width = overlay.composer_column(self)
        return x, max(PANEL_MIN_WIDTH, width)

    def panel_width(self) -> int:
        return self._composer_column()[1]

    def _content_width(self) -> int:
        return max(1, self.panel_width() - PANEL_PADDING_CELLS)

    def _fit(self) -> tuple[int, int, int]:
        """``(rows above the dock, gutter rows, body budget)`` — one measurement.

        One method because they are one sum: the card settles at exactly
        ``budget + chrome + gutter`` rows, and that must not exceed the ground
        above the dock. Splitting the terms across methods that each re-measure
        is how a card ends up a row taller than the space it was sized against,
        with that row on top of the prompt.

        A visible notice is chrome — it survives the tail-cut and sits with the
        keys — so it comes out of the budget, not out of the card's height.
        """
        rows = self._rows_above_dock()
        gutter = PANEL_PADDING_ROWS if rows >= SQUEEZE_ROWS else 0
        chrome = CHROME_ROWS + (1 if self._notice else 0)
        return rows, gutter, max(1, rows - PANEL_HEIGHT_MARGIN - chrome - gutter)

    def sync_layout(self, *, force: bool = False) -> None:
        """Repaint when the screen, the dock, or the composer's column moved.

        The composer's column is in the fingerprint because the card is sized
        AND placed from it: a resize that changes the panel's width without
        changing the rows above it would otherwise leave the card at the old
        width, breaking the shared edges that are the whole composition.

        ``force`` bypasses the guard. The resize path needs it — measured, the
        fingerprint read one refresh after a resize still holds the pre-resize
        dock, so the guard compares two stale numbers and agrees with itself.
        """
        if not self.display or not self.is_mounted:
            return
        if force or self._layout_fingerprint() != self._layout_shown:
            self._repaint()

    def _layout_fingerprint(self) -> tuple[int, int, int]:
        x, width = self._composer_column()
        return x, width, self._rows_above_dock()

    def on_resize(self, event) -> None:  # type: ignore[no-untyped-def]
        self._repaint()

    # -- rendering ------------------------------------------------------------
    def _flat_body(self) -> _FlatBody:
        """Every row of the exchange, newest last, with its turn beside it.

        Flat rather than grouped by turn, which is the change this card needed.
        Grouping existed because the card shed whole TURNS when it overflowed,
        and that was chosen over a row cut for a real reason: a row cut "left
        the top of the card showing a mid-sentence continuation at the answer
        indent with no question above it, which reads as the start of a new
        answer". But shedding whole turns made a turn the smallest addressable
        unit, so an answer taller than the card had a middle no gesture could
        reach at all.

        The reason the turn cut was chosen is honoured WITHOUT the turn being
        the unit: :meth:`_window` cuts by row and pins the owning question, so
        a fragment still never appears without the question that produced it.
        That is why ``heads`` is carried here — it is the only thing left that
        knows which question a row belongs to once the rows are one list.

        MEMOISED, because this renders every turn and the row cut cannot ask
        for fewer: to know where a row window starts you need the row counts of
        everything above it. That is a real cost the turn walk did not pay
        (it stopped at the first turn that did not fit), and left uncached it
        lands on gestures that change nothing about the rows — a wheel step, a
        page chord, the clamp in ``_scroll_by`` — each of which would re-render
        the whole exchange to learn numbers it just computed. Measured at 120
        turns, flattening cold is 34.9 ms against 0.5 ms for the memo, and the
        exchange is uncapped: 300 turns is 72.8 ms, which is past the 30 ms
        ``STALL_MS`` bar in ``tests/unit/test_tui_responsiveness.py``.

        The key is every input the rows are built from. ``_revision`` covers
        the exchange itself (see :meth:`_invalidate_flat`); width and the theme
        epoch are the two the rows are folded and coloured for, and they are
        the same pair :meth:`_markdown_rows` keys its own cache on — a resize
        or a theme change misses both, which is correct, because both change
        what a row looks like. The two caches are layers of one thing: this one
        skips the assembly, that one skips the markdown render underneath it.
        """
        width = self._content_width()
        key = (width, theme_mod.get_theme_epoch(), self._revision)
        memo = self._flat_memo
        if memo is not None and memo[0] == key:
            return memo[1]
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))
        flat = _FlatBody()
        for index, turn in enumerate(self._turns):
            # The blank that separates turns belongs BETWEEN them, so it leads
            # every turn but the first. A window CAN open on one, and it is
            # left in place: that only happens when a turn is above the window,
            # which is exactly when the marker is drawn over the top row, so
            # the blank separates the marker from the exchange instead of
            # leading it. Stripping it would cost a budget row the window
            # cannot win back, which is what made the card's height vary with
            # scroll position.
            rows: list[Text] = [] if index == 0 else [Text()]
            question = self._question_rows(turn.question, width, dim, fg)
            head_start = len(flat.lines) + len(rows)
            rows.extend(question)
            flat.heads.append((head_start, head_start + len(question)))
            # Status/error rows are answer-side too. Keep the separator in the
            # measured body so it consumes scroll budget instead of extra height.
            # Markdown lists/code may already supply that breathing row.
            answer = self._answer_rows(turn, width, dim, danger)
            if not answer or answer[0].plain.strip():
                rows.append(Text())
            rows.extend(answer)
            flat.lines.extend(rows)
            flat.owners.extend([index] * len(rows))
        self._flat_memo = (key, flat)
        return flat

    def _window(
        self, flat: _FlatBody, budget: int, back: int | None = None
    ) -> tuple[list[Text], int, int, int]:
        """``(rows to paint, first content row, one past the last, questions above)``.

        The whole scroll model in one place, so the painter, the clamp and the
        keyboard page can never disagree about what is on screen.

        Two things come out of the BUDGET rather than out of the card's height,
        because the card may never grow past ``_fit()`` (see its docstring):
        the overflow marker, and the owning question pinned above a window that
        opens mid-answer.

        They OVERLAY the window's top rows rather than pushing them down, and
        that is deliberate. Reserving budget for them instead — moving the top
        down by as many rows as they cost — cannot fill the budget exactly at
        every offset: stepping the top past a turn boundary changes the number
        of pinned rows at the same time as the content count, so the total
        jumps from under to over with no offset in between. Measured on 8 short
        turns, that left the card alternating between 21 and 22 rows as the
        reader scrolled, and the card is sized to its content (``_repaint``),
        so a card that changes height with scroll position moves the text being
        read. Overlaying is exactly ``budget`` rows at every offset by
        construction.

        Overlaying costs nothing in reachability, which is the property this
        whole change exists for: a covered row is one the window has not
        finished passing, and one more step of the gesture moves it down out of
        the overlay. That holds ONLY while the overlay is strictly smaller than
        the budget, which is why it is clamped to ``budget - 1`` below rather
        than left to the arithmetic. An overlay that filled the budget would
        paint a marker over every row it names, at every offset — measured at
        budget 1 that is a card showing ``↑ 60 earlier lines · scroll`` and
        nothing else, forever, which is a sharper version of the defect this
        whole change exists to remove. At the last step the overlay is empty:
        the window starts at row 0, so no question sits above it to pin and no
        marker is drawn.

        Budget 1 and 2 are REACHABLE, not theoretical — ``_fit()`` yields 1 for
        4 to 7 rows above the dock, and for 8 or 10 with a notice showing. A
        short terminal or a full dock lands there, and the card must still show
        the reader a row of their answer.
        """
        total = len(flat.lines)
        # Clamped HERE and not only in `_scroll_by`, because the exchange can
        # shrink under a parked reader: `settle_answer` replaces a streamed
        # answer with the authoritative text, which is routinely shorter.
        offset = self._scroll_back_rows if back is None else back
        back = min(offset, max(0, total - budget))
        end = total - back
        first = max(0, end - budget)
        # A separator alone is not a readable one-row window. Show its owning
        # question instead; the next scroll step still reaches the answer.
        if budget == 1 and first == flat.heads[flat.owners[first]][1]:
            first -= 1
            end = first + 1

        owner = flat.owners[first]
        head_start, head_end = flat.heads[owner]
        pinned = flat.lines[head_start : min(head_end, first)]
        # Questions above the window are whole questions the reader has walked
        # past; the one being pinned is not among them, because part of it is
        # on screen. Counted from the TOP only — the old count added the turns
        # hidden BELOW the window too, so a card scrolled to the oldest
        # question announced "5 earlier questions" under an arrow pointing up
        # at nothing.
        hidden_turns = owner
        width = self._content_width()
        dim = Style(color=theme_mod.semantic_color("dim"))

        marker: list[Text] = []
        if hidden_turns:
            noun = "question" if hidden_turns == 1 else "questions"
            label = f"↑ {hidden_turns} earlier {noun} · scroll"
        else:
            label = ""
            # Rows above the window, LESS the question rows the overlay
            # re-shows (those are on screen), PLUS the rows the overlay covers
            # — which cancels to ``first + 1``, the whole overlay being one
            # marker row plus exactly the pinned rows it accounts for. Only
            # when the window has somewhere above it to be: at ``first == 0``
            # the exchange starts on screen, the overlay is empty, and a marker
            # there would hide the question it exists to keep visible while
            # claiming a row was withheld that the reader is looking at.
            withheld = first + 1 if first else 0
            if withheld > 0:
                # LINES, where the multi-turn marker says questions. The
                # question count is the better unit when whole questions are
                # above (see `AsideBody`), but here it is zero, which is
                # exactly why this card used to say nothing at all and let a
                # truncated answer read as a complete one. What the reader
                # wants to know inside one answer is how much of it they are
                # missing, and only a row count answers that. It states the
                # quantity for the reason the subagent page states its own:
                # "⟨expand⟩ alone does not distinguish two more lines from
                # fifty, and that is the whole difference between clicking and
                # not bothering" (`subagent_view.py`).
                noun = "line" if withheld == 1 else "lines"
                label = f"↑ {withheld} earlier {noun} · scroll"
        if label:
            marker = [Text(truncate_cells(label, width), style=dim)]

        # Shed rather than overflow, in the order that keeps the card readable:
        # a wrapped question loses its tail before the content loses a row, and
        # the content always keeps at least one. The pinned rows kept are the
        # FIRST ones, which carry the `▌` mark and the start of the question.
        #
        # CONTENT WINS THE LAST ROW. The marker goes before the pin, because a
        # marker with nothing under it is a card that describes its content
        # instead of showing any, while a pinned question with nothing under it
        # at least paints a row the reader asked for. Below budget 2 neither
        # fits and the card is bare content — which is what it did before this
        # change and the right answer at that size.
        if len(marker) >= budget:
            marker = []
        pinned = pinned[: max(0, budget - len(marker) - 1)]
        # A pinned question needs the same breathing row as an uncut turn. It
        # costs overlay budget, never height: shorten a wrapped pin first, but
        # at tiny budgets retain the existing question/content priority. Do not
        # insert a second gap when the window still shows the question or its
        # original separator (head_end), or cover the next turn's question.
        if pinned and budget - len(marker) >= 3:
            pinned = pinned[: budget - len(marker) - 2]
            content_start = first + len(marker) + len(pinned)
            if (
                content_start > head_end
                and flat.owners[content_start] == owner
                and flat.lines[content_start].plain.strip()
            ):
                pinned.append(Text())
        # Pinned UNCONDITIONALLY while the window opens inside a turn, even at
        # the offsets where the overlay then covers the last of that turn's own
        # rows and the question is left with nothing under it. That frame is
        # imperfect and it is the better of the two available: the alternative
        # is dropping the pin, which puts a bare continuation at the answer
        # indent under the marker — the exact misreading ("the start of a new
        # answer") the turn-grouped cut was chosen to prevent. A question whose
        # answer is one scroll step below it is a frame in transit; an
        # unattributed fragment is a frame that lies about whose words it is.
        overlay_rows = [*marker, *pinned]
        rows = [*overlay_rows, *flat.lines[first:end][len(overlay_rows) :]]
        return rows, first, end, hidden_turns

    def _body(self) -> AsideBody:
        """The visible rows, and the paint-scoped bound on the answer cache.

        The cache is rebuilt to exactly what THIS paint reached. Keying on the
        text means a streaming answer mints a fresh key per delta, so a cache
        that only ever inserted would hold every prefix of every answer for as
        long as the card is open; dropping whatever the paint did not touch
        bounds it without a policy.

        What it bounds it AT changed with the row window. The turn walk it was
        written for rendered only the turns it could show, so the cache held a
        card's worth; :meth:`_flat_body` renders the whole exchange because a
        row cut cannot know where the window starts without the row counts.
        The cache is what keeps that affordable — a turn's rows are rendered
        once and read back on every later paint — so the prune still runs, and
        still drops answers no longer on the card, but the paint it is scoped
        to is now the exchange rather than the window.
        """
        self._used_answers = set()
        memo = self._flat_memo
        body = self._visible_rows()
        # ONLY when this paint actually flattened. On a memo hit nothing calls
        # `_markdown_rows`, so `_used_answers` is empty and pruning against it
        # would evict the entire cache — and the next real flatten would then
        # re-render every answer, which is the cost both caches exist to avoid.
        # `is not memo` is the test for "rebuilt": `_flat_body` stores a fresh
        # `_FlatBody` when it misses, so identity changes exactly then.
        if self._flat_memo is not memo:
            self._answer_cache = {
                key: rows for key, rows in self._answer_cache.items() if key in self._used_answers
            }
        return body

    def _visible_rows(self) -> AsideBody:
        """The visible rows: a ROW window onto the tail of the exchange.

        Tail-anchored rather than paged, and that is the difference between
        this card and the usage card. A quota table is a reference document the
        reader navigates; an aside is a conversation, whose interesting end is
        always the newest turn. The wheel and the app's chord walk
        ``_scroll_back_rows`` back from that tail without taking ↑/↓ from the
        composer, which is holding focus so the user can keep talking.
        """
        width = self._content_width()
        dim = Style(color=theme_mod.semantic_color("dim"))
        if not self._turns:
            return AsideBody(
                [Text(truncate_cells("Ask anything about this session.", width), style=dim)]
            )
        rows, _, _, hidden_turns = self._window(self._flat_body(), self._fit()[2])
        return AsideBody(rows, hidden_turns)

    def _question_rows(
        self, question: str, width: int, mark_style: Style, text_style: Style
    ) -> list[Text]:
        """The question on the transcript's spine: dim mark, ``fg`` prose.

        NOT the accent. The stylesheet enumerates the five sites the one green
        is spent on and ends "Before adding a sixth, take one away" — and this
        would be by far the largest of them, three wrapped questions being some
        hundreds of cells. The accent also MEANS "a turn is live", which a
        settled question sitting beside a dim ``thinking…`` is not. The
        reference (``omp``'s btw panel) paints the question accent because it
        has no accent budget; this app does, so this is a place the port has to
        diverge. ``UserBlock``'s own pairing is what it diverges to.
        """
        body = max(1, width - len(ANSWER_INDENT))
        wrapped = _wrap(question, body) or [""]
        gutter = QUESTION_MARK + " " * (len(ANSWER_INDENT) - cell_len(QUESTION_MARK))
        rows: list[Text] = []
        # Extent, not colour, makes this the same sent-question spine as
        # UserBlock; continuation rows must keep the mark and prose styles apart.
        for line in wrapped:
            row = Text(gutter, style=mark_style)
            row.append(line, style=text_style)
            rows.append(row)
        return rows

    def _answer_rows(self, turn: AsideTurn, width: int, dim: Style, danger: Style) -> list[Text]:
        body = max(1, width - len(ANSWER_INDENT))
        if turn.state == "error" and not turn.answer.strip():
            return self._error_rows(turn.error or "the aside failed", body, danger)
        text = turn.answer.strip()
        if not text:
            waiting = "thinking…" if turn.state == "running" else "no answer"
            return [Text(f"{ANSWER_INDENT}{waiting}", style=dim)]
        # Copied because the renderer hands back its CACHED list, and the
        # status rows below are appended per state — mutating it in place would
        # leave a stale "…" welded onto the answer the next paint reads back.
        rows = list(self._markdown_rows(text, body))
        if turn.state == "running":
            rows.append(Text(f"{ANSWER_INDENT}…", style=dim))
        elif turn.state == "cancelled":
            rows.append(Text(f"{ANSWER_INDENT}(superseded)", style=dim))
        elif turn.state == "error":
            rows.extend(self._error_rows(turn.error, body, danger))
        return rows

    def _markdown_rows(self, text: str, body: int) -> list[Text]:
        """The answer as MARKDOWN rows, indented onto the card's text column.

        The answer is model prose, and every other surface that shows model
        prose renders it (``AssistantBlock`` at
        :meth:`AssistantBlock._flat_rows`). Rendered here through the SAME
        :func:`flatten` rather than a second renderer, so a code span, a bullet
        and a heading are the one shape everywhere — and so ``^f``, which hands
        this exact string to an ``AssistantBlock``, cannot change how the words
        look on their way into the chat.

        The QUESTION deliberately stays plain (see :meth:`_question_rows`).

        Width is the body column, not the card: :func:`flatten` bakes the width
        in AND pads every row out to it, so rendering at ``_content_width()``
        and then adding ``ANSWER_INDENT`` would push each row two cells past
        the card's own edge.

        The pad is KEPT. It is what makes a fenced code block a rectangle: rich
        paints the block's fill onto the pad, so cropping it leaves a ragged
        dark band with the card's ``$lo-overlay`` showing through the notch at
        the end of every short line. On prose rows the pad carries no
        background, so against the card fill it is invisible — the cost is
        trailing spaces on a copied row, which
        ``TranscriptBlock.get_selection`` already drops for the transcript's
        own padded rows.

        CACHED, because ``_repaint`` runs on every streamed delta and repaints
        the whole card: the streaming turn is one render, but the settled turns
        beside it would be re-rendered from text that cannot have changed.
        Measured at a 120-column card over a 636-character answer in 106
        deltas, the median repaint went 0.21 ms plain to 0.75 ms markdown; with
        the cache the settled turns are dictionary hits and only the live
        answer renders.

        The cache is a warm-path optimisation and NOT the bound on this cost.
        It cannot be: width is in the key, so a resize misses every key at
        once. What bounds the work is :meth:`_visible_rows` rendering only the
        turns the card shows — read its note for the measurements, which is
        where a paint that rendered 120 answers to show one question was
        costing 117 ms.

        Keyed on ``(text, width, theme epoch)`` — the three inputs
        :func:`flatten` bakes in. Width because it is folded into the rows
        (``sync_layout``/``on_resize`` repaint, and a stale key would paint the
        old fold at the new width); the theme epoch for the reason
        ``AssistantBlock`` drops its frozen renderable on one (TUI-016), the
        ramp the code spans were coloured from having moved.
        """
        key = (text, body, theme_mod.get_theme_epoch())
        self._used_answers.add(key)
        cached = self._answer_cache.get(key)
        if cached is not None:
            return cached
        indent = Text(ANSWER_INDENT)
        rows: list[Text] = []
        for line in flatten(Markdown(text), body, self._flat_console()).split("\n"):
            row = indent.copy()
            row.append_text(line)
            rows.append(row)
        self._answer_cache[key] = rows
        return rows

    def _flat_console(self) -> Console | None:
        """The app's console, or ``None`` when the card is detached.

        Same bargain ``AssistantBlock._flat_console`` makes, for the same
        reason: the app's console carries the brand markdown theme and the
        terminal's encoding, so the card's code spans match the transcript's.
        A card built before mount, or held directly by a test, has no app —
        :func:`flatten` then falls back to a private console carrying the same
        theme, so the rows are the same either way.
        """
        try:
            return self.app.console
        except Exception:
            return None

    @staticmethod
    def _error_rows(message: str, body: int, danger: Style) -> list[Text]:
        """One shape for every failure: the warning glyph, then the cause."""
        wrapped = _wrap(message, max(1, body - 2)) or [""]
        rows = [Text(f"{ANSWER_INDENT}{ERROR_MARK} {wrapped[0]}", style=danger)]
        rows.extend(Text(f"{ANSWER_INDENT}  {line}", style=danger) for line in wrapped[1:])
        return rows

    def _title_row(self) -> Text:
        """The card's name AND its contract, because the contract is the feature.

        The title states WHAT the surface is; the footer states what the key
        does. They used to overlap — the title ended "esc discards it" while
        the footer two rows below said "esc back to the chat", so the card
        named the same key twice in two verbs, and at 60 columns the title's
        copy was the half that truncated mid-word. The discard moved into the
        footer's own hint, where the key it qualifies already lives, and each
        fact is now stated exactly once.

        ``^f`` is the exception to "nothing here joins the chat", which is why
        the clause says "nothing HERE" — the fork is a deliberate act, not
        something the card does on its own.

        ``muted``, not ``dim``: this is the sentence the whole feature turns
        on, and it is the same rank the usage card gives its second title slot.
        """
        row = Text()
        row.append("Aside", style=Style(color=theme_mod.semantic_color("fg")))
        row.append(
            "  off the record — nothing here joins the chat",
            style=Style(color=theme_mod.semantic_color("muted")),
        )
        row.truncate(max(1, self._content_width()), overflow="ellipsis")
        return row

    def _notice_row(self) -> Text:
        """A refusal, stated on the card that refused rather than in the ledger."""
        row = Text(f"{ERROR_MARK} ", style=Style(color=theme_mod.semantic_color("warning")))
        row.append(self._notice, style=Style(color=theme_mod.semantic_color("warning")))
        row.truncate(max(1, self._content_width()), overflow="ellipsis")
        return row

    def _hint_row(self) -> Text:
        """The keys, shed right-to-left until the row fits.

        ``esc`` is never dropped: it is the only one that is not a convenience,
        and a floating card whose exit is invisible is a trap.

        The ink is one rank up from the usage card's footer, and it has to be.
        Descriptions in ``faint`` measure 1.48:1 against ``$lo-overlay`` —
        unreadable — and this footer is the only stated way out of a mode that
        has taken the Enter key and is holding the user's main draft. Keys
        ``fg`` (11.24:1), descriptions ``dim`` (3.39:1), separators ``faint``.
        """
        key_style = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        faint = Style(color=theme_mod.semantic_color("faint"))
        # "discard, back to the chat" carries BOTH jobs of the key, which is
        # why the title no longer mentions esc at all: the fact that dismissing
        # throws the exchange away belongs on the key that does it, not in a
        # sentence about what the surface is. It also stops the two rows naming
        # the same key in two verbs, which at 60 columns truncated mid-word.
        #
        # Not "close, keep the chat": the fork hint beside it also ends in "the
        # chat" and means the opposite. This row is likewise the ONLY statement
        # of the exit — the composer's placeholder used to repeat it, which read
        # as repetition once the two surfaces became one column a row apart.
        hints = [("esc", "discard, back to the chat")]
        if self._turns:
            # Copy sits NEXT TO fork, and that adjacency is the point: the two
            # keys are the only ways text leaves this card, and they differ in
            # exactly the thing the card's title promises. `^f` writes the
            # exchange into the context and the transcript; `ctrl+r` writes it
            # to the clipboard, which is outside the session, so the off-the-
            # record contract survives. A user who can only see `^f` has to
            # break that contract to keep a sentence. Advertised rather than
            # left as a hidden chord for the same reason the reference
            # implementation puts `c copy` in its own footer, beside branch.
            #
            # Shown whenever there is a turn, NOT gated on `_can_fork`: copy
            # works while the answer is still streaming, which is precisely
            # when `^f` is refused, so borrowing fork's condition would hide
            # the key in the window where it is the only one that works.
            hints.append((ASIDE_COPY_KEY, "copy"))
        if self._can_fork:
            hints.append(("^f", "fork into the chat"))
        # "again" only once there IS a first time. On a card the user has just
        # opened with a bare `/btw`, it names a history that does not exist.
        hints.append(("enter", "ask again" if self._turns else "ask"))
        width = self._content_width()
        while len(hints) > 1 and _hint_width(hints) > width:
            hints.pop()
        row = Text()
        for key, what in hints:
            if row.plain:
                row.append(" · ", style=faint)
            row.append(key, style=key_style)
            row.append(f" {what}", style=dim)
        row.truncate(max(1, width), overflow="ellipsis")
        return row

    def render_lines_for_test(self) -> list[str]:
        """The card as plain strings, chrome included — what a user reads."""
        return [line.plain for line in self._compose_rows()]

    def _compose_rows(self) -> list[Text]:
        body = self._body()
        faint = Style(color=theme_mod.semantic_color("faint"))
        rows = [
            self._title_row(),
            Text("─" * self._content_width(), style=faint),
            *body.lines,
            # One quiet row between the exchange and the keys, and only one.
            # The card is sized to its CONTENT and rests on the composer (see
            # ``_repaint``), so unspent budget is not padding to be printed —
            # a two-line answer in a card padded to the full budget is thirty
            # rows of empty overlay covering the conversation it is about.
            Text(),
        ]
        # The notice sits with the keys rather than in the body: it is a
        # statement ABOUT the card (a key that will not work yet), not a turn
        # in the exchange, and it must survive the tail-cut that drops turns.
        if self._notice:
            rows.append(self._notice_row())
        rows.append(self._hint_row())
        return rows

    def set_fork_available(self, can_fork: bool) -> None:
        """Tell the card whether ``^f`` is live right now, and repaint if so.

        Called by the app on every state change that could move the answer —
        a turn starting or ending, an aside settling — because the two inputs
        are the session's streaming flag and this card's own turns, and only
        the app can see both.
        """
        if can_fork == self._can_fork:
            return
        self._can_fork = can_fork
        self._repaint()

    def _repaint(self) -> None:
        if not self.display or not self.is_mounted:
            return
        rows = self._compose_rows()
        x, width = self._composer_column()
        self.styles.width = width
        # Pinned rather than ``auto``: ``auto`` measures content against a
        # guessed width before layout and settles a row out, and this card is
        # repainted on every streamed delta. It is pinned to what the content
        # ACTUALLY needs, which is what keeps a short exchange a short card.
        _, gutter, _ = self._fit()
        self.set_class(gutter == 0, "-squeezed")
        height = len(rows) + gutter
        self.styles.height = height
        # Stacked ON the composer, sharing its column: same left edge, same
        # right edge, no gap. The card and the composer are one unit — you ask
        # in the card and you type in the field directly below it — and a
        # floating dialog above an unrelated full-width dock said the opposite.
        # Distinct fills (`$lo-overlay` over `$lo-surface`) keep them legible as
        # two surfaces, which is the elevation step the kit uses instead of a
        # rule.
        #
        # Growth still goes UPWARD, which is why the anchor is the dock and not
        # the middle of the ground: a centred card moves half a row per line it
        # gains and drags the text being read with it.
        overlay.stack_on_dock(self, width, height, x)
        self._layout_shown = self._layout_fingerprint()
        # The card COVERS the rows it occupies, and they are the newest turns —
        # measured at 120x24, the last thing the user asked disappeared behind
        # it, which is the context the aside exists to ask about. The host is
        # what owns the transcript (there can be two of them while the subagent
        # page is open), so the card reports its settled height and the app
        # reserves the ground rather than reaching across.
        if self._on_height is not None:
            self._on_height(height)
        out = Text()
        for index, row in enumerate(rows):
            if index:
                out.append("\n")
            out.append_text(row)
        self.update(out)

    def on_mount(self) -> None:
        if self.display:
            self._repaint()


def _wrap(text: str, width: int) -> list[str]:
    """Wrap prose to ``width``, preserving the blank lines between paragraphs.

    ``textwrap`` on the whole string would collapse the paragraph breaks a
    writer uses for structure, which is most of the shape text carries when
    nothing is interpreting it. Lists survive for the same reason: each source
    line is wrapped on its own, so a ``- item`` keeps its own row.

    This is the VERBATIM path, and the two callers left on it are the ones
    whose text must not be reinterpreted: the user's own question (an asterisk
    they typed is an asterisk they meant) and an error message (a provider's
    stack trace is not markup). The ANSWER is model prose and renders as
    markdown — see :meth:`AsidePanel._markdown_rows`.
    """
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out.append("")
            continue
        out.extend(textwrap.wrap(line, width=max(1, width)) or [""])
    return out


def _hint_width(hints: list[tuple[str, str]]) -> int:
    """Cells, not characters — the shed threshold has to agree with the
    cell-aware ``Text.truncate`` that renders the row it guards."""
    return sum(cell_len(key) + 1 + cell_len(what) for key, what in hints) + 3 * max(
        0, len(hints) - 1
    )
