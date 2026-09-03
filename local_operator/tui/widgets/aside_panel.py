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
    """The visible rows, and how many whole QUESTIONS were dropped above them.

    Questions, not lines. A user remembers asking three things; they never
    counted the rows an answer wrapped to, so a line count names a quantity
    they cannot check against anything.
    """

    lines: list[Text] = field(default_factory=list)
    hidden_turns: int = 0


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
        #: How many turns back from the tail the reader has walked. The wheel
        #: moves it; ASKING snaps it back to 0, because the user's own new
        #: question is a re-acquire. Streaming does NOT: the same three-state
        #: rule the transcript follows (:class:`TailAnchor`), in this card's
        #: units — a reader who walked back mid-answer is reading, and the
        #: deltas must not drag them forward.
        self._scroll_back = 0
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
        self.display = False

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
        self._scroll_back = 0
        self._answer_cache.clear()
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
        self._scroll_back = 0
        self._answer_cache.clear()
        self._used_answers.clear()
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
        self._scroll_back = 0
        self._turns.append(AsideTurn(question=question))
        self.display = True
        self._repaint()
        return self._generation

    def accepts(self, generation: int) -> bool:
        """Whether a worker's result still belongs to the visible question."""
        return self.is_open and generation == self._generation

    def append_answer(self, generation: int, delta: str) -> None:
        if not self.accepts(generation) or not delta:
            return
        self._turns[-1].answer += delta
        # NOT reset here. `ask` already put the reader on the new question, so
        # the only way `_scroll_back` is non-zero mid-answer is that they
        # wheeled back on purpose — and snapping them forward on every delta is
        # the same bug the transcript had, in this card's units. They re-acquire
        # by wheeling back down to the tail.
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
        self._repaint()

    def fail_answer(self, generation: int, message: str) -> None:
        """A failure stays IN the card, next to the question it belongs to."""
        if not self.accepts(generation):
            return
        turn = self._turns[-1]
        turn.state = "error"
        turn.error = message
        self._repaint()

    def fork_messages(self) -> list[tuple[str, str]]:
        """``(question, answer)`` for every turn worth promoting to the chat.

        Failed and cancelled turns are dropped: forking is "keep this exchange",
        and half an exchange is not one.
        """
        return [(turn.question, turn.answer) for turn in self._turns if turn.forkable]

    # -- mouse ----------------------------------------------------------------
    # Every gesture is STOPPED, because the card floats over the transcript:
    # left to bubble, one scroll would move both the aside and the chat behind
    # it and a click would focus whatever happens to sit underneath. The same
    # stop the toast and the usage card make.
    #
    # A click is stopped and nothing more. The card owns no input by contract
    # (the one composer is pointed at it), so there is no hit area to offer.
    #
    # The WHEEL is the one gesture that acts. Scroll KEYS were rejected because
    # ↑/↓ belong to the focused composer's prompt history and the aside's whole
    # premise is that the user keeps typing there — but the wheel costs no key,
    # and without it the ``↑ N earlier questions`` marker names content with no
    # way to reach it.
    def on_click(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()

    def on_mouse_scroll_down(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(-1)

    def on_mouse_scroll_up(self, event) -> None:  # noqa: ANN001 - Textual event type
        event.stop()
        self._scroll_by(1)

    def _scroll_by(self, delta: int) -> None:
        """Move back from the tail, CLAMPED — the newest turn is home."""
        target = max(0, min(self._max_scroll_back(), self._scroll_back + delta))
        if target == self._scroll_back:
            return
        self._scroll_back = target
        self._repaint()

    def _max_scroll_back(self) -> int:
        """How many whole turns are droppable off the head at this size.

        Counted off ``_turns`` rather than off built groups: the grouping is one
        group per turn, so building them to take their length rendered the whole
        exchange on every wheel event to learn a number the list already had.
        """
        return max(0, len(self._turns) - 1)

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
    def _turn_group(
        self, index: int, width: int, fg: Style, dim: Style, danger: Style
    ) -> list[Text]:
        """One turn as a row group: its question, then its answer.

        Grouped rather than flattened because the card sheds whole TURNS when
        it overflows (see :meth:`_body`) — cutting by row left the top of the
        card showing a mid-sentence continuation at the answer indent with no
        question above it, which reads as the start of a new answer.

        One turn at a time, and not the whole exchange, because rendering is
        what :meth:`_body` is trying to avoid spending on turns it then drops.
        The leading blank separates this turn from the one above it, so the
        oldest turn does not get one.
        """
        turn = self._turns[index]
        rows: list[Text] = [] if index == 0 else [Text()]
        rows.extend(self._question_rows(turn.question, width, dim, fg))
        rows.extend(self._answer_rows(turn, width, dim, danger))
        return rows

    def _body(self) -> AsideBody:
        """The visible rows, and the paint-scoped bound on the answer cache.

        The cache is rebuilt to exactly what THIS paint reached. Keying on the
        text means a streaming answer mints a fresh key per delta, so a cache
        that only ever inserted would hold every prefix of every answer for as
        long as the card is open; dropping whatever the paint did not touch
        bounds it without a policy. What it bounds it AT is what
        :meth:`_visible_rows` renders — the turns on the card, plus at most the
        one turn it had to build to find out did not fit.
        """
        self._used_answers = set()
        body = self._visible_rows()
        self._answer_cache = {
            key: rows for key, rows in self._answer_cache.items() if key in self._used_answers
        }
        return body

    def _visible_rows(self) -> AsideBody:
        """The tail of the exchange, cut on turn boundaries.

        Tail-anchored rather than paged, and that is the difference between
        this card and the usage card. A quota table is a reference document the
        reader navigates; an aside is a conversation, whose interesting end is
        always the newest turn. ``_scroll_back`` lets the WHEEL walk back
        through the earlier ones without taking ↑/↓ from the composer, which is
        holding focus so the user can keep talking.

        Turns are built LAZILY, newest first, and the walk stops as soon as one
        does not fit. The cut used to read the lengths of a fully-built
        exchange, which looked necessary — the budget is spent backwards, so
        the sizes have to come from somewhere — but every group above the cut
        was rendered and then thrown away. Measured at a 120x40 card over
        realistic answers, a cold paint cost 9.7 ms at 10 turns and 117 ms at
        120 while painting ONE question, against 1.1/8.7 ms for the same card
        before the answer was markdown at all. The turn count is uncapped and
        the card repaints on every streamed delta, so that is a stall the user
        feels while typing. Only a group's own length gates the walk, so a
        group nobody will see never has to exist.
        """
        width = self._content_width()
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))
        count = len(self._turns)
        if not count:
            return AsideBody(
                [Text(truncate_cells("Ask anything about this session.", width), style=dim)]
            )

        budget = self._fit()[2]
        # Walk back from the newest turn, taking whole turns while they fit.
        # One row of the budget is held back for the marker whenever anything
        # is dropped, so the reader is told what is above rather than shown a
        # conversation that begins nowhere.
        end = count - min(self._scroll_back, max(0, count - 1))
        first = end - 1
        window = [self._turn_group(first, width, fg, dim, danger)]
        total = len(window[0])
        while first > 0:
            # Built to be measured. It is kept only if it fits, which is the
            # one turn's worth of render this cut cannot do without.
            above = self._turn_group(first - 1, width, fg, dim, danger)
            if total + len(above) > budget - 1:
                break
            first -= 1
            total += len(above)
            window.append(above)
        window.reverse()
        # The blank row that separates turns belongs BETWEEN them. On the first
        # visible group it is a blank leading the card, which reads as the
        # exchange having started and then said nothing.
        if window and window[0] and not window[0][0].plain:
            window[0] = window[0][1:]
        lines = [row for group in window for row in group]
        hidden = first + (count - end)
        if hidden == 0:
            return AsideBody(lines[-budget:] if len(lines) > budget else lines)
        noun = "question" if hidden == 1 else "questions"
        head = Text(truncate_cells(f"↑ {hidden} earlier {noun} · scroll", width), style=dim)
        # A single turn taller than the whole card is the one case a turn
        # boundary cannot resolve: keep its question pinned as the first row so
        # the fragment underneath has an owner, then take the newest rows.
        keep = max(1, budget - 1)
        if len(lines) > keep:
            lines = [lines[0], *lines[-(keep - 1) :]] if keep > 1 else [lines[0]]
        return AsideBody([head, *lines], hidden)

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
        head = Text(gutter, style=mark_style)
        head.append(wrapped[0], style=text_style)
        rows.append(head)
        rows.extend(Text(f"{ANSWER_INDENT}{line}", style=text_style) for line in wrapped[1:])
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
