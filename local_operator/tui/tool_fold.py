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
* The visible reading order that Up/Down walk (``focus_neighbour``), because this
  layer is the only thing that knows which rows a closed bar is hiding.

WHAT THIS DELIBERATELY DOES NOT OWN

* The receipt fold. Peer/wake/monitor groups keep ``QuietGroupBlock`` and its
  contract untouched; a ToolCard inside one of those bars is SKIPPED by this
  layer (it folds with the receipt, not with its neighbours), which is also why a
  bar is never shown over a receipt's tools while the receipt bar stands.
* Row indentation, muted narration, the By-response foot line and the gap rules:
  slice 5's work. Every bar uses the semantic ink tokens that exist today, so that
  slice only adds colour-table entries.
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

from local_operator.tui.tool_outline import (
    Outline,
    OutlineRow,
    Run,
    ToolFacts,
    Turn,
    outline_of,
)
from local_operator.tui.widgets.transcript import (
    FoldBarBlock,
    QuietGroupBlock,
    TranscriptBlock,
    TranscriptView,
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
        diffing by key instead of clearing the layer.
        """
        self.invalidate()

    def reset(self) -> None:
        """Drop every bar and every choice: the transcript under them is gone."""
        for key in list(self._bars):
            self._remove_bar(key)
        self._open.clear()
        self._applied.clear()
        self._plans.clear()
        self._rows.clear()
        self._order = []
        self._pinned = frozenset()

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
        self._apply_visibility(plan, bar, bar.expanded)
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
        plans, pinned = self._plans_of(outline, rows, by_id)
        self._pinned = pinned
        self._rows = by_id
        self._reconcile(plans)

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
    ) -> tuple[list[_BarPlan], frozenset[str]]:
        """Every bar this outline calls for, in document order, plus the pins."""
        row_of = {row.id: row for row in rows}
        exempt = {row.id for row in rows if outline_row_is_exempt(by_id[row.id])}
        pinned = self._pins(outline, row_of)
        mode = self._view.display_mode

        plans: list[_BarPlan] = []
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
                add(turn_bar)

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
                    self._add_sub_bars(node, turn, turn_bar.key, pinned, add)
                    continue
                run_covered = tuple(row_id for row_id in node.tool_ids if row_id not in pinned)
                if not run_covered:
                    continue
                add(
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
                self._add_sub_bars(node, turn, node.key, pinned, add)

        final: list[_BarPlan] = []
        for plan in plans:
            final.append(
                replace(
                    plan,
                    children=tuple(children.get(plan.key, ())),
                    covered_bars=tuple(descendants.get(plan.key, ())),
                )
            )
        return final, pinned

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

    def _reconcile(self, plans: list[_BarPlan]) -> None:
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
            open_state = self._resolved_open(plan)
            bar.set_open(open_state)
            self._plans[plan.key] = plan

        for plan in plans:
            bar = self._bars.get(plan.key)
            if bar is None:
                continue
            self._apply_visibility(plan, bar, bar.expanded)
        self._rebuild_order()
        if lost_focus is not None:
            self._restore_bar_focus(lost_focus, plans)
        if retry:
            self.invalidate()

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
        """Whether the keyboard is standing on this bar or a row it would hide."""
        focused = self._focused()
        if focused is None:
            return False
        if focused is self._bars.get(plan.key):
            return True
        if any(focused is self._rows.get(row_id) for row_id in plan.covered):
            return True
        return any(focused is self._bars.get(key) for key in plan.covered_bars)

    def _focused(self):
        try:
            return self._view.app.focused
        except Exception:  # noqa: BLE001 - a detached view has no screen to ask
            return None

    def _apply_visibility(self, plan: _BarPlan, bar: FoldBarBlock, open_state: bool) -> None:
        """Add or drop this bar's veto, focusing the bar before it hides focus."""
        veto = f"{VETO_PREFIX}{plan.key}"
        old_rows, old_bars = self._applied.get(plan.key, ((), ()))
        rows = () if open_state else plan.covered
        bars = () if open_state else plan.covered_bars

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
