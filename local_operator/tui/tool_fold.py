"""The transcript's multi-level fold over tool calls: turn > run > sub-group.

ONE LAYER PER VIEW, and one visibility authority. The fold is DERIVED, never
accumulated: every derive reads the rows on hand through
``transcript.outline_row_of``, asks ``tool_outline`` for the tree, and then
reconciles the bars and the rows they hide against what is already mounted. That
is what makes a live transcript and a replayed one agree — the same rows give the
same tree, whichever event stream built them — and it is why a ToolCard is never
remounted to change what is folded over it.

WHAT THIS OWNS

* The :class:`~local_operator.tui.widgets.transcript.FoldBarBlock` for every turn
  bar, run bar and same-tool sub-group, mounted as a sibling immediately before
  the first row it covers.
* Every covered row's veto: ``TranscriptBlock.fold_vetoes`` (the quiet fold's
  mechanism, shared, so two folds over one transcript cannot write each other's
  ``display``) plus the ONE ``sync_display`` call that paints the answer. Rows are
  never written directly.
* The reader's open/closed choices (``_open``, keyed by the outline's stable bar
  key) and the defaults behind them (a settled run folds; the run a live turn is
  still growing stays open until it finishes, unless the reader touched it).
* The INDENT level of every row the fold draws over, and the By-response foot
  line. Both are properties of the step's SHAPE, so they belong to the one
  owner that knows which step a row belongs to — a row asked to indent itself
  would need a second copy of the outline. ``fold_depth`` on the row widget is
  the seam (see ``tool_card.row_indent``), never a CSS padding: the copy gutter
  reads the same number, so a covered row still copies without eating a
  character of its tool name.
* The visible reading order that Up/Down walk (``focus_neighbour``), because this
  layer is the only thing that knows which rows a closed bar is hiding.
* The ``f``/``F`` failed-call jump and the ``+``/``-`` whole-turn open/collapse:
  three gestures that each need the document order AND the bar tree, which is
  exactly what the layer holds. They live here rather than on the bar because a
  bar knows only itself.

WHAT THIS DELIBERATELY DOES NOT OWN

* The receipt fold. Peer/wake/monitor groups keep ``QuietGroupBlock`` and its
  contract untouched; a ToolCard inside one of those bars is SKIPPED by this
  layer (it folds with the receipt, not with its neighbours), which is also why a
  bar is never shown over a receipt's tools while the receipt bar stands.
* Muted narration's CLASSIFICATION. The block already knows whether it is
  narration (``AssistantBlock.is_narration``, set from ``tui.narration``); this
  layer only decides that the By-response layout paints those blocks in the step
  ink. An answer and a remark are never told apart by position here.
* Any persistence. Open state lives for the view's lifetime (and ``/clear``
  resets it), exactly like the receipt fold's.

RE-DERIVATION is once per dirty frame, scheduled with ``call_after_refresh``: a
replay of five hundred rows marks dirty five hundred times and derives once per
painted frame. A closed turn's plan cannot move again (rows only append, and a
state change that moves a tally re-derives it), so an unchanged plan's restate is
a no-op and its visibility diff is empty: only the bars that really moved paint,
and no ToolCard is touched.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from local_operator.i18n.keys import tui_transcript_fold
from local_operator.i18n.text import tr
from local_operator.tui.tool_outline import (
    Outline,
    OutlineRow,
    Run,
    ToolFacts,
    Turn,
    outline_of,
)
from local_operator.tui.widgets.transcript import (
    ExpandableActionBlock,
    FoldBarBlock,
    NoticeBlock,
    QuietGroupBlock,
    TranscriptBlock,
    TranscriptView,
    TurnFootBlock,
    outline_row_is_exempt,
    outline_row_of,
)

if TYPE_CHECKING:
    from local_operator.tui.widgets.tool_card import ToolCard

#: The cells a turn bar must hide before it pays for its own row. The prototype's
#: ``hidden >= 3``: a bar over one tool row and one line of narration costs more
#: than it saves, and the design keeps 1-2 call steps as plain rows for the same
#: reason.
TURN_BAR_MIN_ROWS = 3

#: A bar's veto name, one per bar key. The prefix keeps this fold's names from
#: colliding with any other fold writing ``fold_vetoes`` (the receipt fold's
#: ``QuietGroupBlock.FOLD_VETO`` is a bare name).
VETO_PREFIX = "tool-fold:"

#: The veto a By-response foot writes over the turn-failed notice it replaces.
#: The NOTICE stays in the transcript (it is also the row the model is shown, and
#: stored rows are not rewritten for a layout); only its paint is withheld, so
#: the reason is stated once, on the foot, with the incident head and the
#: model-facing trailer dropped.
FOOT_VETO_PREFIX = "tool-foot:"

_fold_t = tr("tui.transcript_fold")


@dataclass(frozen=True)
class _BarPlan:
    """One bar as the last derive derived it — the unit the screen is diffed on.

    Frozen and comparable: an unchanged plan (the same key, facts and coverage)
    means the bar and its vetoes are already right, which is what lets a long
    transcript re-derive without restating every row it contains.
    """

    key: str
    #: ``turn`` | ``run`` | ``sub`` — the row grammar, and the mount order among
    #: bars that share an anchor.
    kind: str
    tool_name: str
    facts: ToolFacts
    #: The row ids this bar hides while closed, in reading order. Rows exempt
    #: from every fold (an ask, an image, a waiting call) are never in here.
    covered: tuple[str, ...]
    #: The bar keys this bar also hides while closed: its whole subtree, so a
    #: closed turn cannot leave an open run's sub-bars floating on their own.
    covered_bars: tuple[str, ...]
    #: The first row under the bar — where it mounts.
    anchor: str
    #: The bar's open state before the reader has said anything.
    default_open: bool
    parent: str | None
    #: Direct child bar keys, in reading order (``right`` steps into the first).
    children: tuple[str, ...]
    #: The turn produced no answer, so its failure tally is an alarm.
    noanswer: bool

    @property
    def depth(self) -> int:
        return {"turn": 0, "run": 1, "sub": 2}[self.kind]


@dataclass(frozen=True)
class _FootPlan:
    """One By-response foot line: the whole turn's tally, after it settled.

    The foot is not a bar — it hides nothing but the failed notice it restates
    — but it is planned, diffed and reconciled the same way, because it has the
    same two properties a bar needs to stay stable: a key that cannot move
    under appends (the user row's id) and a restate that is a no-op when its
    facts did not move.
    """

    key: str
    #: The turn's LAST row: the answer, or the failed notice (which is in
    #: ``nodes`` even though ``final_id`` skips it when an answer came first).
    anchor: str
    facts: ToolFacts
    died: bool
    #: The raw failure notice text; the foot cleans it at paint time.
    reason: str
    #: The notice row whose paint the foot replaces (empty when none did).
    hides: str


def _turn_row_ids(turn: Turn) -> list[str]:
    """Every call/loose row id inside a turn, in reading order, minus its answer.

    ``Turn.nodes`` is exactly the pre-answer body (the outline keeps the final
    answer and the user row out of it), so this is the span a turn bar hides.
    """
    ids: list[str] = []
    for node in turn.nodes:
        if isinstance(node, Run):
            ids.extend(node.tool_ids)
        else:
            ids.append(node.id)
    return ids


class ToolFoldLayer:
    """Owns the per-view fold: its bars, their state, and every covered row's veto."""

    def __init__(self, view: TranscriptView) -> None:
        self._view = view
        #: Mounted bars, by outline key.
        self._bars: dict[str, FoldBarBlock] = {}
        #: The reader's own choices, by bar key. Only a toggle writes here — a
        #: derive never does, so a mode switch or an append cannot undo one.
        self._open: dict[str, bool] = {}
        #: Bar key -> the (rows, bars) it currently hides. The reconcile reads
        #: this, so a row that leaves a bar's coverage keeps no stale veto.
        self._applied: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
        #: The last derived plans, by key, in document order.
        self._plans: dict[str, _BarPlan] = {}
        #: Row id -> the block it named at the last derive.
        self._rows: dict[str, TranscriptBlock] = {}
        #: The visible reading order: bars and uncovered rows, as they paint.
        self._order: list[TranscriptBlock] = []
        #: Call ids a dead turn leaves visible under its folds (the design pins
        #: only the LAST failure of such a turn).
        self._pinned: frozenset[str] = frozenset()
        #: Bar key -> the turn its part of the tree belongs to. The turn bar is
        #: absent in By response, so the outline's parent pointers cannot answer
        #: "which turn is this run in" there; this map can, and `+`/`-` need it.
        self._bar_turn: dict[str, str] = {}
        #: Row id -> the display level `row_indent` takes (0 outside the fold's
        #: steps, 1 in a step, 2 revealed under a bar or pinned below one).
        self._row_depth: dict[str, int] = {}
        #: Row id -> widget, inverted (for the gesture that has a focused CARD
        #: and needs its document id).
        self._id_of: dict[int, str] = {}
        #: Every failed call, in document order — the `f`/`F` ladder.
        self._failed: tuple[str, ...] = ()
        #: The failed call the last jump landed on. The jump hands the keyboard
        #: to the CALL (BINDINGS.md's contract: that is where the reader acts),
        #: and a ToolCard deliberately does not bind `f`, so the next press
        #: arrives from a BAR. This cursor is what makes that press continue the
        #: ladder instead of starting over from the bar's own seat.
        self._failed_cursor: str | None = None
        #: Row ids in document order, for "what is after the bar I am on".
        self._row_order: list[str] = []
        #: Mounted feet, by key, and their last plans.
        self._feet: dict[str, TurnFootBlock] = {}
        self._foot_plans: dict[str, _FootPlan] = {}
        #: Foot key -> the notice row id whose paint it currently withholds.
        self._foot_applied: dict[str, str] = {}
        #: Set when the gap classes are stale (a mode switch, a bar mount or a
        #: visibility change); consumed by the next derive, which walks the
        #: child order once. Never set per append — append-time gaps are already
        #: decided by the container.
        self._gaps_dirty = False
        self._scheduled = False

    # -- the dirty seam -------------------------------------------------------

    def invalidate(self) -> None:
        """Mark the fold stale; the next painted frame re-derives it ONCE.

        Every structural change in ``TranscriptView`` calls this: an append (via
        ``note_quiet_append``), a source part, a replacement, a removal, a page
        seam, a positional insert and a settled row's ``finalize``. Marking is
        free and idempotent — the derive is what costs — so no caller has to
        decide whether its change could matter.
        """
        if self._scheduled:
            return
        # A detached view has no frame to schedule against; the mount that
        # follows will mark it again. `call_after_refresh` returns whether the
        # message was posted, so a closed pump cannot leave the flag stuck on.
        if not self._view.call_after_refresh(self._derive):
            return
        self._scheduled = True

    def note_mode_change(self) -> None:
        """A layout switch: re-derive, keeping every bar and every choice.

        The brief's contract for this switch is that no ToolCard is remounted and
        a run bar keeps its key (and so its open state) — both fall out of
        diffing by key instead of clearing the layer. The GAP grammar changes
        with the mode too (By response is steps; By turn is adaptive air), so
        the switch marks the gaps stale for the derive to re-apply.
        """
        self._gaps_dirty = True
        self.invalidate()

    def reset(self) -> None:
        """Drop every bar and every choice: the transcript under them is gone."""
        for key in list(self._bars):
            self._remove_bar(key)
        # Feet before rows: a foot releases the notice veto through `_rows`, so
        # clearing the map first would leave a hidden notice behind for the
        # blocks that survive a partial reset.
        for key in list(self._feet):
            self._remove_foot(key)
        self._open.clear()
        self._applied.clear()
        self._plans.clear()
        self._rows.clear()
        self._order = []
        self._pinned = frozenset()
        self._bar_turn.clear()
        self._row_depth.clear()
        self._id_of.clear()
        self._failed = ()
        self._failed_cursor = None
        self._row_order = []
        self._foot_plans.clear()
        self._gaps_dirty = False

    # -- the reader's gestures ------------------------------------------------

    def note_bar_toggled(self, bar: FoldBarBlock) -> None:
        """A bar was opened or closed by the reader: record it and paint it.

        Applied in the SAME frame as the keypress rather than on the next derive
        — a fold that unfolds a beat later reads as a dropped key.
        """
        self._open[bar.key] = bar.expanded
        plan = self._plans.get(bar.key)
        if plan is None:
            return
        if self._apply_visibility(plan, bar, bar.expanded):
            self._gaps_dirty = False
            self._view.reapply_gaps()
        self._rebuild_order()

    def parent_bar(self, key: str) -> FoldBarBlock | None:
        plan = self._plans.get(key)
        if plan is None or plan.parent is None:
            return None
        return self._bars.get(plan.parent)

    def first_child(self, key: str) -> TranscriptBlock | None:
        """The first thing under an OPEN bar: a child bar, else its first row."""
        plan = self._plans.get(key)
        if plan is None or not plan.covered:
            return None
        first = plan.covered[0]
        for child_key in plan.children:
            child = self._bars.get(child_key)
            if child is not None and self._plans.get(child_key) is not None:
                if self._plans[child_key].anchor == first:
                    return child
        return self._rows.get(first)

    def jump_failed(self, delta: int) -> None:
        """``f``/``F``: walk the transcript's failed calls, in document order.

        The target is opened to before it is focused: every bar that hides it
        takes the reader's open choice, in the same frame, so the keyboard lands
        on a row that is already painted. Wrapping at the ends is the
        prototype's behaviour and the reason there is no "no more" notice on a
        transcript that HAS failures — the ladder is circular, which is how one
        key reaches them all.

        A transcript with no failed call at all gets the one sentence the design
        names, through the app's notice row. It is said in the layer rather than
        painted as a transcript row: the reader pressed a NAVIGATION key, and a
        row appended per miss would grow the conversation every time they
        retried it.
        """
        if not self._failed:
            self._view.notice(_fold_t(tui_transcript_fold.no_failed()))
            return

        target = self._next_failed(delta)
        if target is None:
            return
        self._failed_cursor = target
        for key, plan in self._plans.items():
            if target not in plan.covered:
                continue
            bar = self._bars.get(key)
            if bar is None:
                continue
            self._open[key] = True
            bar.set_open(True)
            if self._apply_visibility(plan, bar, True):
                self._gaps_dirty = True
        self._rebuild_order()
        if self._gaps_dirty:
            self._gaps_dirty = False
            self._view.reapply_gaps()
        block = self._rows.get(target)
        if block is not None:
            block.focus()

    def _next_failed(self, delta: int) -> str | None:
        """The failed id ``delta`` steps from the focus, wrapping at the ends."""
        failed = list(self._failed)
        focused = self._focused()
        current = self._id_of.get(id(focused)) if focused is not None else None
        if isinstance(focused, FoldBarBlock):
            plan = self._plans.get(focused.key)
            if plan is not None and self._covers_cursor(plan):
                # Standing on a bar over the last jump's target: continue the
                # ladder from the CALL, not from the bar's own first row.
                current = self._failed_cursor
            elif plan is not None:
                current = plan.anchor
        if current in failed:
            index = (failed.index(current) + delta) % len(failed)
            return failed[index]
        # Standing on a row that is not a failure (or on a bar): take the first
        # failure after its seat for `f`, the last before it for `F`, wrapping.
        try:
            position = self._row_order.index(current) if current is not None else -1
        except ValueError:
            position = -1
        if delta > 0:
            for row_id in failed:
                if self._row_index(row_id) > position:
                    return row_id
            return failed[0]
        for row_id in reversed(failed):
            if self._row_index(row_id) < position:
                return row_id
        return failed[-1]

    def _covers_cursor(self, plan: _BarPlan) -> bool:
        """Whether ``plan`` is the bar over the reader's last failed-call seat."""
        cursor = self._failed_cursor
        return cursor is not None and cursor in self._failed and cursor in plan.covered

    def _row_index(self, row_id: str) -> int:
        try:
            return self._row_order.index(row_id)
        except ValueError:
            return -1

    def set_turn_open(
        self, bar: FoldBarBlock, open: bool
    ) -> None:  # noqa: A002 - the reader's state
        """``+``/``-``: open every level of this bar's turn, or restore defaults.

        ``-`` DROPS the turn's explicit choices rather than forcing everything
        closed: the design's word is "back to defaults", and a turn whose live
        run defaulted open would mean something else under a blanket close. The
        choices are per session (the layer's dict), never persisted — the
        transcript reopens with the shipped defaults, like the receipt fold.
        """
        turn_key = self._bar_turn.get(bar.key)
        if turn_key is None:
            return
        for key, owning_turn in self._bar_turn.items():
            if owning_turn != turn_key:
                continue
            plan = self._plans.get(key)
            widget = self._bars.get(key)
            if plan is None or widget is None:
                continue
            if open:
                self._open[key] = True
            else:
                self._open.pop(key, None)
            want = self._resolved_open(plan)
            widget.set_open(want)
            if self._apply_visibility(plan, widget, want):
                self._gaps_dirty = True
        self._rebuild_order()
        if self._gaps_dirty:
            self._gaps_dirty = False
            self._view.reapply_gaps()

    def focus_neighbour(self, block: TranscriptBlock, delta: int) -> bool:
        """Walk the VISIBLE order — bars included — and focus the next row.

        Answers False when ``block`` is not in the order at all (a view with no
        fold, or a row inside an open receipt bar), which is the container's cue
        to walk its own ledger instead.
        """
        order = self._order
        if not order:
            return False
        try:
            index = order.index(block)
        except ValueError:
            return False
        step = 1 if delta > 0 else -1
        cursor = index + step
        while 0 <= cursor < len(order):
            candidate = order[cursor]
            if candidate.focusable:
                candidate.focus()
                return True
            cursor += step
        return False

    # -- the derive -----------------------------------------------------------

    def _derive(self) -> None:
        """Rebuild the outline from the view's rows and reconcile the screen."""
        self._scheduled = False
        blocks = self._view.blocks()
        receipt_members = self._receipt_members(blocks)
        live_block = self._view.pinned_tail()

        rows: list[OutlineRow] = []
        by_id: dict[str, TranscriptBlock] = {}
        for block in blocks:
            if id(block) in receipt_members or isinstance(block, FoldBarBlock):
                continue
            if block is not live_block:
                # A transient row is chrome (the working line, a reasoning
                # tail): it appears and vanishes inside a turn, so it must
                # neither split a run nor anchor one. The PINNED one is the
                # exception — it is the only marker a live turn has, and
                # `outline_of` needs its id to know which turn is in flight.
                if getattr(block, "SPACING_TRANSIENT", False):
                    continue
                if getattr(block, "work_dormant", False):
                    continue
            row = outline_row_of(block)
            if row is None:
                continue
            rows.append(row)
            by_id[row.id] = block

        live_id = None
        if live_block is not None:
            candidate = outline_row_of(live_block)
            if candidate is not None and candidate.id in by_id:
                live_id = candidate.id
        outline = outline_of(rows, live_id)
        plans, pinned, bar_turn, feet = self._plans_of(outline, rows, by_id)
        self._pinned = pinned
        self._rows = by_id
        self._bar_turn = bar_turn
        self._failed = tuple(row.id for row in rows if row.kind == "tool" and row.state == "error")
        self._row_order = [row.id for row in rows]
        self._id_of = {id(block): row_id for row_id, block in by_id.items()}
        depths: dict[str, int] = {}
        bars = {plan.key for plan in plans}
        # Which rows a bar really sits over — its own children, including the
        # pin it leaves visible (a pin is deliberately not in `covered`, so it
        # needs this answer separately).
        barred: set[str] = set()
        for turn in outline.turns:
            if turn.key in bars:
                barred.update(_turn_row_ids(turn))
            for node in turn.nodes:
                if isinstance(node, Run) and node.key in bars:
                    barred.update(node.tool_ids)
            # Ben's "tool rows one indent deeper than prose" applies to every
            # turn that holds a tool call, in BOTH modes. A turn that grew a bar
            # has its rows under one; a By-response turn closes with a foot; and
            # a turn with too few calls for a bar keeps the same column anyway,
            # because the bar column is where a tool row lives — two insets for
            # the same kind of row in one frame (col 3 here, col 4 beside a bar)
            # read as two different kinds of row. A turn with NO call keeps the
            # column it always drew: there is no step to belong to.
            if turn.key not in bars and not turn.facts.count:
                continue
            for row_id in _turn_row_ids(turn):
                depths[row_id] = 1
        for plan in plans:
            # Revealed under an open bar, or hidden under a closed one — either
            # way it is one level below the bar's column when painted.
            for row_id in plan.covered:
                depths[row_id] = 2
        for row_id in pinned:
            # A pin is the failure a collapsed bar leaves visible, so it belongs
            # to that bar's group and takes the revealed column — but only when a
            # bar really covers its run. With no bar over it there is nothing for
            # it to sit under, and it keeps the tool-row column rather than
            # claiming a level nothing gave it.
            if row_id in barred:
                depths[row_id] = 2
        self._row_depth = depths
        self._reconcile(plans, feet)

    def _receipt_members(self, blocks: list[TranscriptBlock]) -> set[int]:
        """The blocks a receipt bar stands for: their tools fold with it."""
        members: set[int] = set()
        for block in blocks:
            if isinstance(block, QuietGroupBlock):
                members.update(id(member) for member in block.members())
        return members

    def _plans_of(
        self,
        outline: Outline,
        rows: list[OutlineRow],
        by_id: dict[str, TranscriptBlock],
    ) -> tuple[list[_BarPlan], frozenset[str], dict[str, str], list[_FootPlan]]:
        """Every bar this outline calls for, in document order, plus the pins.

        Also answers two questions the GESTURES need and the bars alone cannot:
        which TURN each bar belongs to (``+``/``-`` are turn-scoped, and By
        response draws no turn bar to carry the parent link) and the By-response
        foot line each settled turn earns.
        """
        row_of = {row.id: row for row in rows}
        exempt = {row.id for row in rows if outline_row_is_exempt(by_id[row.id])}
        pinned = self._pins(outline, row_of)
        mode = self._view.display_mode

        plans: list[_BarPlan] = []
        bar_turn: dict[str, str] = {}
        parents: dict[str, str | None] = {}
        children: dict[str, list[str]] = {}
        descendants: dict[str, list[str]] = {}

        def add(plan: _BarPlan) -> None:
            plans.append(plan)
            parents[plan.key] = plan.parent
            if plan.parent is not None:
                children.setdefault(plan.parent, []).append(plan.key)
                cursor: str | None = plan.parent
                while cursor is not None:
                    descendants.setdefault(cursor, []).append(plan.key)
                    cursor = parents.get(cursor)

        for turn in outline.turns:

            def add_in_turn(plan: _BarPlan, _turn_key: str = turn.key) -> None:
                """Record the plan AND the turn it belongs to (the gesture map)."""
                add(plan)
                bar_turn[plan.key] = _turn_key

            body = _turn_row_ids(turn)
            covered = tuple(
                row_id for row_id in body if row_id not in exempt and row_id not in pinned
            )
            turn_bar: _BarPlan | None = None
            # A PAGE-SEAM TURN GETS NO TURN BAR. Its prompt row is in the part of
            # the conversation this window has not loaded, so the bar would label
            # a turn the reader cannot see the start of and hide the rows they
            # resumed to read; the seam is a FRAGMENT of a turn, and By turn
            # condenses turns. Its runs still fold — that is the useful part on a
            # resumed window. (Caught by the synthetic paging corpus, whose rows
            # carry no user turns at all: the whole 600-row window read as one
            # seam turn and collapsed to a single bar.)
            seam = turn.user_id is None
            if (
                mode == "by-turn"
                and turn.has_turn_bar
                and not seam
                and len(covered) >= TURN_BAR_MIN_ROWS
            ):
                turn_bar = _BarPlan(
                    key=turn.key,
                    kind="turn",
                    tool_name="",
                    facts=turn.facts,
                    covered=covered,
                    covered_bars=(),
                    anchor=covered[0],
                    default_open=False,
                    parent=None,
                    children=(),
                    noanswer=turn.died,
                )
                add_in_turn(turn_bar)

            flat = turn_bar is not None and turn.flat
            # The newest RUN of a live turn stays open. Positions are not enough
            # to say which one that is: the working line is a loose node AFTER
            # the run it reports on, so "the last node" is that marker and never
            # the run. The last run is what the reader is watching grow.
            run_positions = [
                index for index, node in enumerate(turn.nodes) if isinstance(node, Run)
            ]
            last_run = run_positions[-1] if run_positions else None
            for node_index, node in enumerate(turn.nodes):
                if not isinstance(node, Run):
                    continue
                run_parent = turn_bar.key if turn_bar is not None else None
                tail = node_index == last_run
                # A run with a running call, or the run a live turn is still
                # growing, must NOT fold under itself: the reader is watching it.
                run_open = turn.live and (tail or node.facts.running_id is not None)
                if flat and turn_bar is not None:
                    # One run and a turn bar: the bar's children are the run's
                    # children. A second bar saying the same thing is the defect
                    # the design's "flat" rule exists for.
                    self._add_sub_bars(node, turn, turn_bar.key, pinned, add_in_turn)
                    continue
                run_covered = tuple(row_id for row_id in node.tool_ids if row_id not in pinned)
                if not run_covered:
                    continue
                add_in_turn(
                    _BarPlan(
                        key=node.key,
                        kind="run",
                        tool_name="",
                        facts=node.facts,
                        covered=run_covered,
                        covered_bars=(),
                        anchor=run_covered[0],
                        default_open=run_open,
                        parent=run_parent,
                        children=(),
                        noanswer=turn.died,
                    )
                )
                self._add_sub_bars(node, turn, node.key, pinned, add_in_turn)

        final: list[_BarPlan] = []
        for plan in plans:
            final.append(
                replace(
                    plan,
                    children=tuple(children.get(plan.key, ())),
                    covered_bars=tuple(descendants.get(plan.key, ())),
                )
            )
        return final, pinned, bar_turn, self._feet_of(outline, rows, by_id)

    def _feet_of(
        self,
        outline: Outline,
        rows: list[OutlineRow],
        by_id: dict[str, TranscriptBlock],
    ) -> list[_FootPlan]:
        """The By-response foot each settled, action-bearing turn earns.

        NO FOOT IN BY TURN: the turn bar already closes that turn with the same
        facts, and the bar carries them where the eye expects the summary — on
        the row that hid the work. NO FOOT FOR A LIVE TURN: the count and clock
        are still moving, and a settled-looking total under a growing run reads
        as a bug. NO FOOT WITHOUT ACTIONS: a plain answer has nothing to total.

        A PAGE-SEAM TURN gets none either. Its key would have no user row to
        name (the brief's ``tf:foot:<user id>``), and it has no visible prompt to
        close — the same reason it draws no turn bar.
        """
        if self._view.display_mode != "by-response":
            return []

        # The LAST row of each turn, in document order. Read from the rows
        # rather than from ``Turn`` alone: an answered turn's final row is
        # ``final_id``, a dead turn's is its notice, and ``nodes`` holds the
        # notice even when the answer came first — the rows are the one place
        # the whole span is visible.
        last_of_turn: dict[str, str] = {}
        current: str | None = None
        for row in rows:
            if row.kind == "user":
                current = f"tf:turn:{row.id}"
            if current is not None:
                last_of_turn[current] = row.id

        feet: list[_FootPlan] = []
        for turn in outline.turns:
            if turn.live or turn.user_id is None or not turn.facts.count:
                continue
            # A call awaiting approval has not SETTLED: the turn is paused on a
            # question, not dead, and a `✗ turn failed` foot under it would state
            # an outcome the reader is being asked to decide.
            if self._turn_is_paused(turn, by_id):
                continue
            anchor = last_of_turn.get(turn.key)
            if anchor is None:
                continue
            reason = ""
            hides = ""
            # The reason is the turn's closing ERROR notice, whatever head it
            # arrived under: `turn failed` is the harness's plain arm, and a
            # provider incident arrives as `[session incident …] category: raw`
            # (which the outline's `notice_failed` classification does not
            # match). The foot cleans either form for display.
            anchor_block = by_id.get(anchor)
            if isinstance(anchor_block, NoticeBlock) and anchor_block.notice_kind == "error":
                reason = anchor_block.text()
                hides = anchor
            feet.append(
                _FootPlan(
                    key=f"tf:foot:{turn.user_id}",
                    anchor=anchor,
                    facts=turn.facts,
                    died=turn.died,
                    reason=reason,
                    hides=hides,
                )
            )
        return feet

    @staticmethod
    def _turn_is_paused(turn: Turn, by_id: dict[str, TranscriptBlock]) -> bool:
        """True when a call in ``turn`` is waiting for the reader's approval."""
        from local_operator.tui.widgets.tool_card import ToolCard

        for row_id in _turn_row_ids(turn):
            block = by_id.get(row_id)
            if isinstance(block, ToolCard) and block.state == "waiting":
                return True
        return False

    def _add_sub_bars(
        self,
        node: Run,
        turn: Turn,
        parent: str,
        pinned: frozenset[str],
        add,
    ) -> None:
        """Add the run's same-tool sub-groups as children of ``parent``."""
        tail_id = node.tool_ids[-1] if node.tool_ids else None
        for sub in node.subgroups:
            covered = tuple(row_id for row_id in sub.tool_ids if row_id not in pinned)
            if not covered:
                continue
            add(
                _BarPlan(
                    key=sub.key,
                    kind="sub",
                    tool_name=sub.tool_name,
                    facts=sub.facts,
                    covered=covered,
                    covered_bars=(),
                    anchor=covered[0],
                    # The live tail stays open for the same reason its run does:
                    # the newest call must not vanish as it lands.
                    default_open=turn.live and tail_id is not None and tail_id in sub.tool_ids,
                    parent=parent,
                    children=(),
                    noanswer=turn.died,
                )
            )

    def _pins(self, outline: Outline, row_of: dict[str, OutlineRow]) -> frozenset[str]:
        """The LAST failed call of each dead turn: the design's one exemption.

        A dead turn has no bar of its own (a turn bar needs an answer), so the
        pin is what keeps the reason its run failed reachable without unfolding
        every level. Only the last: pinning all sixteen failures of a
        browser-batch turn would rebuild the screenshot problem the fold exists
        to answer.
        """
        pinned: set[str] = set()
        for turn in outline.turns:
            if not turn.died or not turn.facts.failed:
                continue
            failed = [row_id for row_id in _turn_row_ids(turn) if row_of[row_id].state == "error"]
            if failed:
                pinned.add(failed[-1])
        return frozenset(pinned)

    # -- reconcile with the screen -------------------------------------------

    def _reconcile(self, plans: list[_BarPlan], feet: list[_FootPlan]) -> None:
        """Mount, restate, retire and hide so the screen matches ``plans``."""
        wanted = {plan.key: plan for plan in plans}
        # A bar that is about to go can be the one the keyboard is standing on —
        # a mode switch replaces the run bar of a now-flat turn with the turn
        # bar. Note the seat BEFORE the widgets go: the nearest surviving bar
        # takes focus after the reconcile, which is the brief's "mode switch
        # keeps focus on the same entry or the nearest visible bar".
        lost_focus = self._bar_focus_anchor(wanted)
        for key in list(self._bars):
            if key not in wanted:
                self._remove_bar(key)

        retry = False
        for plan in plans:
            anchor = self._rows.get(plan.anchor)
            if anchor is None:
                # The anchor row is gone (a re-derive will pick the new first
                # row). Never mount against nothing: a bar with no seat would
                # paint at the tail, over rows it does not stand for.
                self._remove_bar(plan.key)
                continue
            if not anchor.is_mounted:
                # A HELD batch's row: it is in ``_blocks`` but has no seat yet,
                # so a bar cannot be mounted before it. Skip this bar and ask
                # for one more derive — `_mount_batch` invalidates on landing,
                # and this branch covers any other path that derives in between.
                self._remove_bar(plan.key)
                retry = True
                continue
            bar = self._bars.get(plan.key)
            if bar is not None:
                previous = self._plans.get(plan.key)
                if previous is not None and previous.anchor != plan.anchor:
                    # The seat moved (the row it stood before was removed and
                    # the span re-derived). Re-seat the widget; its open state
                    # is keyed, so it comes back exactly as the reader left it.
                    self._remove_bar(plan.key)
                    bar = None
            created = bar is None
            if bar is None:
                bar = FoldBarBlock(
                    plan.key, self, fold_width=self._view.scrollable_content_region.width
                )
                self._bars[plan.key] = bar
            live = self._live_card(plan, row_of=self._rows)
            bar.restate(
                kind=plan.kind,
                depth=plan.depth,
                tool_name=plan.tool_name,
                facts=plan.facts,
                noanswer=plan.noanswer,
                live=live[0],
                live_summary=live[1],
            )
            if created:
                self._view.mount(bar, before=anchor)
                self._gaps_dirty = True
            open_state = self._resolved_open(plan)
            bar.set_open(open_state)
            self._plans[plan.key] = plan

        for plan in plans:
            bar = self._bars.get(plan.key)
            if bar is None:
                continue
            if self._apply_visibility(plan, bar, bar.expanded):
                self._gaps_dirty = True
        self._apply_depths()
        self._apply_ink()
        self._reconcile_feet(feet)
        self._rebuild_order()
        if lost_focus is not None:
            self._restore_bar_focus(lost_focus, plans)
        if self._gaps_dirty:
            # Once, in child order — see `TranscriptView.reapply_gaps` for why
            # this cannot be a per-block decision: a bar is not in the ledger's
            # `_blocks`, and the By-response grammar is a property of the whole
            # frame rather than of one append.
            self._gaps_dirty = False
            self._view.reapply_gaps()
        if retry:
            self.invalidate()

    def _apply_depths(self) -> None:
        """Give every ledger row the inset the step it belongs to earned.

        Every expandable action row steps TOGETHER — a tool card, a scheduled
        wake receipt and an inbound peer receipt are one ledger, and a receipt
        left on the plain spine while the tool rows beside it stepped in is the
        column defect this repo fixed once already ("the peer message doesn't
        have the proper left padding like the other tool cards"); in By
        response the tool rows have been one step in from the start, so the
        receipts have to follow them there too.

        Prose, notices and the fold's own bars are not ledger rows and keep the
        column they always drew, and a bar's own depth is fixed at its bar
        column when it is constructed. ``set_fold_depth`` is a no-op when the
        level has not moved, so a long transcript re-derives without rebuilding
        a row.
        """
        for row_id, block in self._rows.items():
            if isinstance(block, ExpandableActionBlock):
                block.set_fold_depth(self._row_depth.get(row_id, 0))

    def _apply_ink(self) -> None:
        """Mute a turn's remarks in By response; the answer keeps text ink."""
        from local_operator.tui.widgets.assistant import AssistantBlock

        muted = self._view.display_mode == "by-response"
        for block in self._rows.values():
            if isinstance(block, AssistantBlock):
                block.set_fold_muted(muted and block.is_narration())

    def _reconcile_feet(self, feet: list[_FootPlan]) -> None:
        """Mount, restate and retire the By-response foot lines."""
        wanted = {plan.key: plan for plan in feet}
        for key in list(self._feet):
            if key not in wanted:
                self._remove_foot(key)

        for plan in feet:
            anchor = self._rows.get(plan.anchor)
            if anchor is None or not anchor.is_mounted:
                # The seat is gone (or not laid out yet): drop this foot and let
                # a later derive seat it. Never mount against nothing.
                self._remove_foot(plan.key)
                continue
            foot = self._feet.get(plan.key)
            previous = self._foot_plans.get(plan.key)
            if foot is not None and previous is not None and previous.anchor != plan.anchor:
                self._remove_foot(plan.key)
                foot = None
            created = foot is None
            if foot is None:
                foot = TurnFootBlock(plan.key)
                self._feet[plan.key] = foot
            foot.restate(facts=plan.facts, died=plan.died, reason=plan.reason)
            if created:
                self._view.mount(foot, after=anchor)
                self._gaps_dirty = True
            self._foot_plans[plan.key] = plan
            self._apply_foot_veto(plan, foot)

    def _apply_foot_veto(self, plan: _FootPlan, foot: TurnFootBlock) -> None:
        """Hide the turn-failed notice the foot restates (By response only)."""
        veto = f"{FOOT_VETO_PREFIX}{plan.key}"
        applied = self._foot_applied.get(plan.key, "")
        if applied and applied != plan.hides:
            block = self._rows.get(applied)
            if block is not None:
                block.fold_vetoes.discard(veto)
                block.sync_display()
            self._foot_applied.pop(plan.key, None)
            self._gaps_dirty = True
        if not plan.hides or applied == plan.hides:
            return
        block = self._rows.get(plan.hides)
        if block is None:
            return
        block.fold_vetoes.add(veto)
        block.sync_display()
        self._foot_applied[plan.key] = plan.hides
        self._gaps_dirty = True

    def _remove_foot(self, key: str) -> None:
        """Release the notice a foot hid, then unmount it."""
        foot = self._feet.pop(key, None)
        hides = self._foot_applied.pop(key, "")
        if hides:
            block = self._rows.get(hides)
            if block is not None:
                block.fold_vetoes.discard(f"{FOOT_VETO_PREFIX}{key}")
                block.sync_display()
            self._gaps_dirty = True
        self._foot_plans.pop(key, None)
        if foot is not None:
            foot.remove()

    def _bar_focus_anchor(self, wanted: dict[str, _BarPlan]) -> str | None:
        """The seat of the bar the keyboard is on, when that bar is leaving."""
        focused = self._focused()
        if not isinstance(focused, FoldBarBlock):
            return None
        plan = self._plans.get(focused.key)
        if plan is None or focused.key in wanted:
            return None
        return plan.anchor

    def _restore_bar_focus(self, anchor: str, plans: list[_BarPlan]) -> None:
        """Focus the nearest surviving bar to ``anchor`` (same turn first)."""
        if not plans:
            return
        order = {id(block): index for index, block in enumerate(self._view.blocks())}
        target = order.get(id(self._rows.get(anchor)), 0)
        best: tuple[int, str] | None = None
        for plan in plans:
            index = order.get(id(self._rows.get(plan.anchor)))
            if index is None:
                continue
            if best is None or abs(index - target) < abs(best[0] - target):
                best = (index, plan.key)
        if best is None:
            return
        bar = self._bars.get(best[1])
        if bar is not None:
            bar.focus()

    def _live_card(
        self, plan: _BarPlan, *, row_of: dict[str, TranscriptBlock]
    ) -> tuple["ToolCard | None", str]:
        """The running call a bar reports, and its summary text."""
        from local_operator.tui.widgets.tool_card import ToolCard

        running = plan.facts.running_id
        if running is None:
            return None, ""
        block = row_of.get(running)
        if not isinstance(block, ToolCard):
            # ``running`` names a row the adapter saw as a call; anything else
            # here is a re-derived id looking at a row that already changed. No
            # live reading is the honest answer, never a guessed label.
            return None, ""
        return block, block.summary

    def _resolved_open(self, plan: _BarPlan) -> bool:
        """Open when the reader said so, else the default, else where focus is."""
        explicit = self._open.get(plan.key)
        if explicit is not None:
            return explicit
        if plan.default_open:
            return True
        # Never hide the row the keyboard is standing on: a fold that swallowed
        # focus would leave the reader pressing keys into nothing.
        return self._focus_inside(plan)

    def _focus_inside(self, plan: _BarPlan) -> bool:
        """Whether the keyboard stands on a row or child bar this one would hide.

        The bar ITSELF is deliberately not counted. Focusing a closed bar is a
        normal state — the reader tabbed to it and has not opened it — and
        counting it made a closed bar impossible to keep: every re-derive read
        "focus is inside" and reopened it, and ``-`` (back to defaults) could
        never close the turn it was standing on. The rule that matters is the
        one :meth:`_guard_focus` enforces at the moment of collapse: if the
        keyboard is on a CHILD, focus moves to the bar before the child goes.
        """
        focused = self._focused()
        if focused is None:
            return False
        if any(focused is self._rows.get(row_id) for row_id in plan.covered):
            return True
        return any(focused is self._bars.get(key) for key in plan.covered_bars)

    def _focused(self):
        try:
            return self._view.app.focused
        except Exception:  # noqa: BLE001 - a detached view has no screen to ask
            return None

    def _apply_visibility(self, plan: _BarPlan, bar: FoldBarBlock, open_state: bool) -> bool:
        """Add or drop this bar's veto, focusing the bar before it hides focus.

        Returns whether any row or child bar actually changed visibility: the
        gap grammar depends on what is SHOWING, so the reconcile only walks the
        frame when a veto landed or lifted.
        """
        veto = f"{VETO_PREFIX}{plan.key}"
        old_rows, old_bars = self._applied.get(plan.key, ((), ()))
        rows = () if open_state else plan.covered
        bars = () if open_state else plan.covered_bars
        changed = set(old_rows) != set(rows) or set(old_bars) != set(bars)

        for row_id in set(old_rows) - set(rows):
            block = self._rows.get(row_id)
            if block is not None:
                block.fold_vetoes.discard(veto)
                block.sync_display()
        for child_key in set(old_bars) - set(bars):
            child = self._bars.get(child_key)
            if child is not None:
                child.fold_vetoes.discard(veto)
                child.sync_display()

        if not open_state:
            self._guard_focus(bar, rows, bars)

        for row_id in set(rows) - set(old_rows):
            block = self._rows.get(row_id)
            if block is not None:
                block.fold_vetoes.add(veto)
                block.sync_display()
        for child_key in set(bars) - set(old_bars):
            child = self._bars.get(child_key)
            if child is not None:
                child.fold_vetoes.add(veto)
                child.sync_display()
        self._applied[plan.key] = (rows, bars)
        return changed

    def _guard_focus(self, bar: FoldBarBlock, rows: tuple[str, ...], bars: tuple[str, ...]) -> None:
        """Move focus to ``bar`` FIRST when this collapse would hide it.

        The rule is absolute — a fold never leaves the keyboard on an invisible
        row — and the ORDER is the whole point: focusing after the veto would
        paint one frame with focus nowhere, and Textual's ``focus()`` is deferred
        behind the layout, so the bar has to be told before the rows go.
        """
        focused = self._focused()
        if focused is None or focused is bar:
            return
        hidden = {id(self._rows[row_id]) for row_id in rows if row_id in self._rows}
        hidden.update(id(self._bars[key]) for key in bars if key in self._bars)
        if any(id(widget) in hidden for widget in focused.ancestors_with_self):
            bar.focus()

    def _remove_bar(self, key: str) -> None:
        """Release everything a bar hides, then unmount it."""
        bar = self._bars.pop(key, None)
        rows, bars = self._applied.pop(key, ((), ()))
        veto = f"{VETO_PREFIX}{key}"
        for row_id in rows:
            block = self._rows.get(row_id)
            if block is not None:
                block.fold_vetoes.discard(veto)
                block.sync_display()
        for child_key in bars:
            child = self._bars.get(child_key)
            if child is not None:
                child.fold_vetoes.discard(veto)
                child.sync_display()
        self._plans.pop(key, None)
        self._gaps_dirty = True
        if bar is not None:
            bar.retire()
            bar.remove()

    def _rebuild_order(self) -> None:
        """Recompute the visible reading order Up/Down walk."""
        bars_at: dict[int, list[FoldBarBlock]] = {}
        for key, plan in self._plans.items():
            bar = self._bars.get(key)
            anchor = self._rows.get(plan.anchor)
            if bar is not None and anchor is not None:
                bars_at.setdefault(id(anchor), []).append(bar)
        for grouped in bars_at.values():
            grouped.sort(key=lambda candidate: candidate.depth)

        order: list[TranscriptBlock] = []
        for block in self._view.blocks():
            for bar in bars_at.get(id(block), ()):
                if bar.display:
                    order.append(bar)
            if block.display:
                order.append(block)
        self._order = order
