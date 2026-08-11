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
from rich.style import Style
from rich.text import Text
from textual.widgets import Static

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets import overlay
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
#: ONE padding row, not two: the card has a top padding row and NO bottom one
#: (``padding: 1 1 0 1``). Measured at 120x40 with a bottom row, the seam
#: between the card and the composer was the widest interval in the whole
#: stack — turn to turn 1 row, exchange to keys 1 row, keys to ``❯`` 2 rows —
#: because both halves of one unit pushed a padding row into the one join that
#: should be tightest. A reader groups by spacing before fill, so the frame
#: said "one column" at the edges and "two panels" at the seam. The composer's
#: own top padding row is now the single row of breathing for both.
PANEL_HEIGHT_MARGIN = 2
PANEL_PADDING_CELLS = 2
PANEL_PADDING_ROWS = 1

#: Rows the pinned chrome costs: title, the rule under it, the blank above the
#: footer, and the footer. Pinned rather than scrolled because the footer
#: carries the way out, and a card that scrolled its own exit away would trap a
#: reader exactly when the exchange got long enough to need it.
CHROME_ROWS = 4

#: Rows above the dock at which the card can still afford its vertical gutter.
#: Below it the gutter is spent rather than the prompt covered. One row lower
#: than it used to be, which is the right consequence of the gutter costing one
#: row less: the card can afford it on a shorter terminal.
SQUEEZE_ROWS = PANEL_HEIGHT_MARGIN + CHROME_ROWS + PANEL_PADDING_ROWS + 2

#: The prompt the side question is wrapped in. Three instructions, each earning
#: its line: OFF THE RECORD so the model does not treat the question as a new
#: task and start narrating a plan; no tools because none are sent (the request
#: carries an empty catalogue and ``tool_choice="none"``) and a model that
#: tried would produce a tool call nobody executes; and answer-from-context
#: because the whole reason to ask here rather than in the chat is that the
#: agent already knows.
ASIDE_PROMPT = """<aside>
The user has stepped aside to ask you something about this session. This is OFF
THE RECORD: neither their question nor your answer joins the conversation, and
no work is being asked for. Answer from the context you already have, briefly
and directly, in prose. Do not use tools, do not propose a plan, and do not ask
a follow-up question. If your context does not answer it, say so plainly.
Question:
{question}
</aside>"""

#: Marker on a question row, and the indent every line of prose sits at.
#:
#: ``▌`` at two cells is the TRANSCRIPT's mark for a row the user SENT
#: (``UserBlock.RULE`` / ``SPINE_INDENT``), not the composer's ``❯``, which
#: marks a row being typed. An aside question is sent the moment Enter lands,
#: so the card uses the same spine at the same column — and a forked exchange
#: then renders in the transcript exactly as it looked on the card, which is
#: what "keep this" ought to look like. The card is already told apart from the
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
        #: How many rows back from the tail the reader has scrolled. The wheel
        #: moves it; asking or streaming snaps it back to 0, because a card
        #: that stayed parked in history while a new answer arrived would hide
        #: the thing the user just asked for.
        self._scroll_back = 0
        #: Screen size plus the live dock ceiling at the last paint. The dock
        #: grows as the composer wraps without resizing this card, so Textual
        #: emits no resize event for it and the app has to ask.
        self._layout_shown: tuple[int, int, int] | None = None
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
        self._generation += 1
        self.display = True
        self._repaint()

    def close(self) -> None:
        """Hide the card and discard the exchange — that is what dismiss means."""
        self._generation += 1
        self._turns = []
        self._notice = ""
        self._scroll_back = 0
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
        # Streaming snaps the view back to the tail: a reader parked in the
        # history while a new answer arrived would watch the card refuse to
        # show them the thing they just asked for.
        self._scroll_back = 0
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
        """How many whole turns are droppable off the head at this size."""
        return max(0, len(self._turn_groups()) - 1)

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
    def _turn_groups(self) -> list[list[Text]]:
        """The exchange as one row group per turn, newest last.

        Grouped rather than flattened because the card sheds whole TURNS when
        it overflows (see :meth:`_body`) — cutting by row left the top of the
        card showing a mid-sentence continuation at the answer indent with no
        question above it, which reads as the start of a new answer.
        """
        width = self._content_width()
        fg = Style(color=theme_mod.semantic_color("fg"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        danger = Style(color=theme_mod.semantic_color("danger"))
        groups: list[list[Text]] = []
        for index, turn in enumerate(self._turns):
            rows: list[Text] = [] if index == 0 else [Text()]
            rows.extend(self._question_rows(turn.question, width, dim, fg))
            rows.extend(self._answer_rows(turn, width, fg, dim, danger))
            groups.append(rows)
        return groups

    def _body(self) -> AsideBody:
        """The visible rows: the tail of the exchange, cut on turn boundaries.

        Tail-anchored rather than paged, and that is the difference between
        this card and the usage card. A quota table is a reference document the
        reader navigates; an aside is a conversation, whose interesting end is
        always the newest turn. ``_scroll_back`` lets the WHEEL walk back
        through the earlier ones without taking ↑/↓ from the composer, which is
        holding focus so the user can keep talking.
        """
        width = self._content_width()
        dim = Style(color=theme_mod.semantic_color("dim"))
        if not self._turns:
            return AsideBody(
                [Text(truncate_cells("Ask anything about this session.", width), style=dim)]
            )

        groups = self._turn_groups()
        budget = self._fit()[2]
        # Walk back from the newest turn, taking whole turns while they fit.
        # One row of the budget is held back for the marker whenever anything
        # is dropped, so the reader is told what is above rather than shown a
        # conversation that begins nowhere.
        end = len(groups) - min(self._scroll_back, max(0, len(groups) - 1))
        first = end - 1
        total = len(groups[first])
        while first > 0 and total + len(groups[first - 1]) <= budget - 1:
            first -= 1
            total += len(groups[first])
        window = [list(group) for group in groups[first:end]]
        # The blank row that separates turns belongs BETWEEN them. On the first
        # visible group it is a blank leading the card, which reads as the
        # exchange having started and then said nothing.
        if window and window[0] and not window[0][0].plain:
            window[0] = window[0][1:]
        lines = [row for group in window for row in group]
        hidden = first + (len(groups) - end)
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

    def _answer_rows(
        self, turn: AsideTurn, width: int, fg: Style, dim: Style, danger: Style
    ) -> list[Text]:
        body = max(1, width - len(ANSWER_INDENT))
        if turn.state == "error" and not turn.answer.strip():
            return self._error_rows(turn.error or "the aside failed", body, danger)
        text = turn.answer.strip()
        if not text:
            waiting = "thinking…" if turn.state == "running" else "no answer"
            return [Text(f"{ANSWER_INDENT}{waiting}", style=dim)]
        rows = [Text(f"{ANSWER_INDENT}{line}", style=fg) for line in _wrap(text, body)]
        if turn.state == "running":
            rows.append(Text(f"{ANSWER_INDENT}…", style=dim))
        elif turn.state == "cancelled":
            rows.append(Text(f"{ANSWER_INDENT}(superseded)", style=dim))
        elif turn.state == "error":
            rows.extend(self._error_rows(turn.error, body, danger))
        return rows

    @staticmethod
    def _error_rows(message: str, body: int, danger: Style) -> list[Text]:
        """One shape for every failure: the warning glyph, then the cause."""
        wrapped = _wrap(message, max(1, body - 2)) or [""]
        rows = [Text(f"{ANSWER_INDENT}{ERROR_MARK} {wrapped[0]}", style=danger)]
        rows.extend(Text(f"{ANSWER_INDENT}  {line}", style=danger) for line in wrapped[1:])
        return rows

    def _title_row(self) -> Text:
        """The card's name AND its contract, because the contract is the feature.

        Both halves of the contract, not the flattering half. "nothing here
        joins the chat" is what the user wants to hear; "esc discards it" is
        what costs them something, and a card that states only the first lets
        someone get a good answer, dismiss it, and reach back for something
        that is gone. ``^f`` in the footer is the exception to the first
        clause, which is why the clause says "nothing HERE" — the fork is a
        deliberate act, not something the card does on its own.

        ``muted``, not ``dim``: this is the sentence the whole feature turns
        on, and it is the same rank the usage card gives its second title slot.
        """
        row = Text()
        row.append("Aside", style=Style(color=theme_mod.semantic_color("fg")))
        row.append(
            "  off the record — nothing here joins the chat, esc discards it",
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
        # "back to the chat", not "close, keep the chat": the fork hint beside
        # it also ends in "the chat" and means the opposite. This row is now the
        # ONLY statement of the exit — the composer's placeholder used to repeat
        # it, which read as repetition once the two surfaces became one column
        # a row apart, so the field went back to saying what it does.
        hints = [("esc", "back to the chat")]
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
    model uses for structure, which is most of the shape a plain-text answer
    has. Lists survive for the same reason: each source line is wrapped on its
    own, so a ``- item`` keeps its own row.
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
