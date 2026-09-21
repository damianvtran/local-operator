"""The model's private reasoning, streamed while it thinks.

WHY THIS BLOCK EXISTS, AND WHY IT IS NOT AN ``AssistantBlock``
-------------------------------------------------------------
The operator's report was a 3-5 s wait before anything appeared, and a measured
part of that wait was the model's reasoning phase, which the harness dropped
outright (``harness/types.ReasoningDeltaEvent``). It is now an agent event, and
this is its TUI surface: a dim, transient block that streams the model's thinking
above the answer it is about to write.

Three properties are deliberate:

* **It is a block of its own, not an ``AssistantBlock``.** The assistant block is
  markdown-rendered, freeze-cached, link-scanned and rail-managed because its
  content is the ANSWER: it is copied by ``/copy``, re-projected on resume, and
  carries narration classification. Reasoning is none of those things — it is
  display-only, it never joins the transcript, and rendering it through the
  assistant path would make it answer-shaped. Sharing the class would have been
  one line and would have made every one of those behaviours a question about
  whether it now applied to thinking.

* **It shows the TAIL of the stream, bounded in both rows and characters.** A
  reasoning model emits thousands of tokens before the first answer token; a
  block that grew with them would push the answer off the top of the viewport and
  re-wrap the whole text on every 30 Hz flush. The live tail is what "watching it
  think" means, and the bound makes each flush O(``REASONING_TAIL_CHARS``)
  regardless of how long the model has been thinking.

* **It is TRANSIENT, and it COLLAPSES when the phase ends.**
  ``SPACING_TRANSIENT`` is exactly this case (a block that appears and vanishes
  within a turn, taking no gap and anchoring none), and the app retires it when
  the answer starts. Retiring collapses it to its header row rather than leaving
  its rows in place: one ~7-row block per model call, dozens per session, with
  nothing to compare them against after a resume, was the review's U1. The
  reasoning of a finished turn is not re-shown on resume — it is not in the
  transcript, and pretending otherwise would be a second, divergent source for a
  phase the ledger now times (``first_reasoning_ms``). ``display.reasoning =
  false`` suppresses the block entirely, which is the one setting under which the
  live transcript and the resumed one agree exactly.

* **It yields to a squeezed viewport.** ``REASONING_VISIBLE_ROWS`` is a ceiling
  the block budgets against the transcript's own height, not a constant: six rows
  plus a header does not fit a 14-row terminal, where the constant pushed the
  user's question off the top (design round 1, D3).
"""

from __future__ import annotations

from rich.style import Style
from rich.text import Text

from local_operator.tui import theme as theme_mod
from local_operator.tui.widgets.transcript import (
    SPINE_INDENT,
    TranscriptBlock,
    TranscriptView,
    wrap_cells,
)

#: Rows of live reasoning painted at most. Six is the working line's own
#: neighbourhood on a 30-row terminal: enough to read a sentence of thinking,
#: short enough that the answer's first row stays in the frame as it arrives.
#: A CEILING rather than a constant: :meth:`ReasoningBlock._visible_rows`
#: budgets it against the transcript's own height, because six rows is taller
#: than the whole transcript on a 14-row terminal (design round 1, D3).
REASONING_VISIBLE_ROWS = 6

#: The floor that budget may shrink to. Two rows is still a sentence of
#: thinking; one is a fragment, and zero would make the block a bare header.
REASONING_MIN_ROWS = 2

#: Rows the transcript keeps for everything else when budgeting: the block's own
#: header, the blank row above it and the question. Subtracted from the
#: transcript's height before the ceiling is applied, so a squeezed viewport
#: costs the block its rows rather than expelling the question (D3: at 100x14
#: the constant bound pushed the user's prompt to y=-3 and raised a scrollbar).
REASONING_VIEWPORT_RESERVE = 3

#: Characters of the reasoning tail the block paints. The bound is what keeps a
#: flush O(1) in the length of the reasoning: the tail is sliced BEFORE wrapping,
#: so a model that has been thinking for two minutes costs the same per flush as
#: one that started a second ago.
REASONING_TAIL_CHARS = 2000

#: Characters of RAW reasoning the block RETAINS. Larger than the painted budget
#: because painting collapses whitespace and then slices, so the retained prefix
#: has to survive that squeeze and still fill :data:`REASONING_TAIL_CHARS`; four
#: times is a comfortable margin over any real fragment stream. Before this bound
#: the block kept the whole phase for the life of the session while painting
#: 2,000 characters of it (review round 1, MINOR-2).
REASONING_RETAINED_CHARS = REASONING_TAIL_CHARS * 4

#: Marks the first painted row when earlier rows were dropped. Without it a
#: reader who looks away returns mid-sentence with no way to tell whether the
#: model started mid-thought or the block shed rows (UX review round 1, U3).
REASONING_TAIL_MARKER = "… "

#: The header row's word. Deliberately NOT "thinking": the working line already
#: says that one row below, and two rows answering different questions with the
#: same word is the redundancy the UX review flagged (U2, and design D1 for the
#: same pair). This block answers "here is what the model is producing"; the
#: working line answers "here is the state of the call".
REASONING_LABEL = "reasoning"

#: Default for ``display.reasoning`` (the ``/settings`` key), kept beside the
#: code that reads it: ON, because the phase this block shows is the thing the
#: operator could not see at all before, and a display flag whose default hides
#: the feature is one nobody discovers. It is an escape hatch, not a preference
#: to opt into.
DEFAULT_REASONING = True

#: The header's glyph, drawn in the same field the notice family uses
#: (``SPINE_INDENT`` + glyph + space). Deliberately from the plain repertoire
#: rather than a Nerd/Font-Awesome codepoint: this row is not a tool row and must
#: not depend on the icon gate (``glyphs.py``).
REASONING_GLYPH = "·"

#: The header field width: the glyph and the space after it, so wrapped
#: reasoning hangs under the first character of the text rather than under the
#: glyph.
GLYPH_COLS = SPINE_INDENT + 2


class ReasoningBlock(TranscriptBlock):
    """One streaming reasoning phase, dim, above the answer.

    :meth:`update_text` takes the ACCUMULATED reasoning (the controller's flush
    contract for ``AssistantDelta``, mirrored for the reasoning channel) and
    re-authoring is skipped when the visible tail did not change, so a flush that
    adds nothing visible costs nothing.

    The height is PINNED to the authored rows, for the reason ``UserBlock._build``
    records: under ``auto`` the engine measures this block and caches that
    measurement on the width alone, so the growing block would reserve rows it
    never paints (or paint into a hole it did not reserve).
    """

    SPACING_KIND = "reasoning"

    #: Appears and vanishes within a turn: takes no gap above itself and anchors
    #: none, so the answer rising as the thinking retires does not flicker a gap
    #: in and out of the transcript.
    SPACING_TRANSIENT = True

    def __init__(self) -> None:
        super().__init__()
        self.add_class("reasoning-block")
        self._text: str = ""
        #: Whether the phase has been closed and collapsed to its header. Set by
        #: :meth:`retire`; the block stops accepting text and sheds its rows.
        self._collapsed: bool = False
        #: The rows last authored, so an update whose VISIBLE tail is unchanged
        #: (a fragment arriving below the fold, a re-flush of the same text) is a
        #: no-op instead of a rebuild.
        self._authored_rows: list[str] = []
        #: The lane the rows were wrapped at; ``on_resize`` and the container's
        #: lane walk both compare against it so one rebuild serves both triggers.
        self._built_width: int = -1
        #: The row count last written to ``styles.height`` — see
        #: :meth:`TranscriptBlock._set_authored_height`. ``-1`` means nothing is
        #: pinned yet, so the first apply always lays out.
        self._pinned_rows: int = -1

    # -- content -------------------------------------------------------------

    def update_text(self, text: str) -> None:
        """Adopt the accumulated reasoning text from the controller's flush."""
        if self._collapsed:
            # A closed phase takes no more text: the controller's final flush can
            # land after ``reasoning_end``, and re-authoring here would grow a
            # block the user has already watched settle.
            return
        # BOUNDED RETENTION. The tail is what the block paints, so holding the
        # whole phase bought nothing and cost the session its bytes: reasoning is
        # not copyable and not persisted, so the text was never read again after
        # the flush that painted it (review round 1, MINOR-2). The comparison
        # below is made on the same bound, so an update whose tail is unchanged
        # is still free.
        retained = text[-REASONING_RETAINED_CHARS:]
        if retained == self._text:
            return
        self._text = retained
        lane = self._built_width if self._built_width > 0 else self.fold_width(80)
        rows = self._rows(self._body_budget(lane))
        if rows == self._authored_rows:
            # Nothing visible moved: the fragment landed below the pinned tail.
            # Skipping the rebuild is what keeps a long reasoning phase from
            # paying a wrap per token for rows nobody can see.
            return
        self._authored_rows = rows
        self._built_width = lane
        self.set_content(self._build(lane), layout=len(rows) != self._pinned_rows)

    def retire(self) -> None:
        """Close the phase: stop accepting text and COLLAPSE to the header row.

        Called when the answer's own block mounts (the model has stopped
        thinking) and on every terminal path (message/turn/agent end, abort).

        Removing the block would delete rows from under the reader's cursor, and
        keeping its rows was the other extreme: one ~7-row block per model call,
        dozens per session, with no way to dismiss them and nothing to compare
        them against after a resume (UX review round 1, U1). Collapsing is the
        middle the tool cards already use — the phase keeps ONE row saying this
        call reasoned, and the transcript stays readable. It is display-only
        either way: the reasoning is never durable, so the collapsed row is a
        marker of the phase, not a handle on content that survives a resume, and
        ``display.reasoning = false`` is the escape hatch for a reader who wants
        none of it at all.
        """
        if self._collapsed:
            return
        self._collapsed = True
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build(), layout=True)
        finally:
            self._finalized = was_finalized
        self.finalize()

    def text(self) -> str:
        """The reasoning the block holds, for tests and inspection."""
        return self._text

    # -- geometry ------------------------------------------------------------

    @staticmethod
    def _body_budget(lane: int) -> int:
        """The text-column width the tail wraps at inside a ``lane``-wide box."""
        return max(lane - GLYPH_COLS - 1, 12)

    def refit_width(self, width: int) -> None:
        """Re-wrap the tail when the lane the block authors at has moved."""
        lane = width if width > 0 else self.fold_width(80)
        if lane == self._built_width:
            return
        self._built_width = lane
        self._authored_rows = self._rows(self._body_budget(lane))
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(
                self._build(lane), layout=len(self._authored_rows) != self._pinned_rows
            )
        finally:
            self._finalized = was_finalized
        parent = self.parent
        if isinstance(parent, TranscriptView):
            parent.refresh_gap_around(self)

    def on_resize(self, event: object) -> None:
        """Re-wrap at the width the layout pass reports (see ``UserBlock``)."""
        self.refit_width(self.fold_width(80))

    def retheme(self) -> None:
        """Re-ink the block from the current ramp."""
        was_finalized = self._finalized
        self._finalized = False
        try:
            self.set_content(self._build(), layout=False)
        finally:
            self._finalized = was_finalized

    # -- authoring -----------------------------------------------------------

    def _tail(self) -> str:
        """The reasoning tail the block paints, bounded in characters."""
        text = self._text
        if len(text) <= REASONING_TAIL_CHARS:
            return text
        return text[-REASONING_TAIL_CHARS:]

    def _rows(self, body: int) -> list[str]:
        """The reasoning rows at ``body`` cells, newest rows that fit the viewport.

        Wrapped with :func:`wrap_cells` rather than handed to Rich, because the
        block authors its own rows: a row folded at paint time would wrap at the
        terminal's left edge and break the single text column every other block
        in the transcript keeps.
        """
        if self._collapsed:
            return []
        rows: list[str] = []
        for paragraph in self._tail().split("\n"):
            if paragraph:
                rows.extend(wrap_cells(paragraph, body))
            else:
                rows.append("")
        budget = self._visible_rows()
        dropped = len(rows) > budget
        tail = rows[-budget:]
        # Blank rows are meaningful only BETWEEN rows: at the edge they reserve a
        # row the block then paints with nothing.
        while tail and not tail[-1].strip():
            tail.pop()
        while len(tail) > 1 and not tail[0].strip():
            tail.pop(0)
        if dropped and tail:
            # One marker on the first painted row. It REPLACES its first two
            # cells rather than being prepended, so the row still fits the lane
            # the wrap was computed for; a prepended marker would overflow into
            # the ellipsis and eat the row's own tail instead.
            head = tail[0]
            tail[0] = (REASONING_TAIL_MARKER + head)[:body] if head.strip() else "…"
        return tail

    def _visible_rows(self) -> int:
        """How many body rows this block may paint in the viewport it is in.

        The ceiling is a viewport budget, not a constant: six rows plus a header
        is taller than the whole transcript on a 14-row terminal, where it pushed
        the user's own question off the top and raised a scrollbar (design round
        1, D3 — the one place this change regressed against the base). The
        reserve is what the rest of the transcript needs on screen: the block's
        header, the blank separation row, and the question it belongs to.
        """
        parent = self.parent
        if isinstance(parent, TranscriptView) and parent.size.height > 0:
            budget = parent.size.height - REASONING_VIEWPORT_RESERVE
            return max(REASONING_MIN_ROWS, min(REASONING_VISIBLE_ROWS, budget))
        # No viewport to measure (a detached block, a unit test): the ceiling.
        return REASONING_VISIBLE_ROWS

    def _build(self, width: int | None = None) -> Text:
        """The header and the hanging reasoning rows, above the answer, height pinned."""
        # BOTH rows use `muted`. The body was `dim`, which measures 4.55:1 on the
        # dark ramp and 3.77:1 on the light one -- under AA for prose that a
        # reader is meant to read (design round 1, D2). `muted` clears both
        # (8.62:1 and 7.18:1), and the header keeps its distinctness from the
        # glyph, the label and the hanging indent rather than from being
        # brighter than the text under it.
        label_style = Style(color=theme_mod.semantic_color("muted"))
        body_style = Style(color=theme_mod.semantic_color("muted"))
        lane = width if width is not None and width > 0 else self.fold_width(80)
        rows = self._rows(self._body_budget(lane))
        self._built_width = lane
        self._authored_rows = rows
        # The authored height counts the header row too.
        self._pinned_rows = len(rows) + 1
        self._set_authored_height(self._pinned_rows)
        line = Text(no_wrap=True, overflow="ellipsis")
        line.append(" " * SPINE_INDENT + REASONING_GLYPH + " ", style=label_style)
        line.append(REASONING_LABEL, style=label_style)
        for row in rows:
            line.append("\n")
            line.append(" " * GLYPH_COLS, style=body_style)
            if row:
                line.append(row, style=body_style)
        return line


__all__ = [
    "DEFAULT_REASONING",
    "REASONING_LABEL",
    "REASONING_MIN_ROWS",
    "REASONING_RETAINED_CHARS",
    "REASONING_TAIL_CHARS",
    "REASONING_TAIL_MARKER",
    "REASONING_VIEWPORT_RESERVE",
    "REASONING_VISIBLE_ROWS",
    "ReasoningBlock",
]
