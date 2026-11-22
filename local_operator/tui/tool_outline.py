"""The TUI's multi-level tool-call outline: turn -> run -> same-tool sub-group.

WHAT THIS IS. The transcript's tools can be read at four depths (design note
``docs/design/tui-tool-fold.md``; prototype ``tui-tool-collapse-prototype.html``,
functions ``buildTurn``/``subGroups``/``barSummaryST``): a turn bar over
everything before its answer, a run bar over 3+ back-to-back calls, a same-tool
sub-group inside a run, and the call card itself. This module derives the
SHAPE and the SUMMARY FACTS of that tree from an ordered list of rows; the
widgets paint it and own every decision about open state, focus and layout.

WHY IT IS ITS OWN MODULE, and why it imports nothing from Textual: the tree has
to be identical for the live path and for replay, for By turn and By response,
and it is what the open-state keys are derived from. A second copy of the
thresholds or the key scheme is how a resumed session comes to disagree with
the one that produced it about which calls folded together. Pure functions here
mean the shape can be tested without mounting a TUI.

THE ROW VOCABULARY IS THE ADAPTER'S CONTRACT. A row is one visible thing in one
transcript, in reading order. ``kind`` is already classified by the caller
(``prose_narration`` vs ``prose_final`` follows ``narration.py``'s
``is_intermediate_narration``; ``notice_failed`` is the turn-failed notice).
Rows that must never be hidden -- an ask, an image, an approval prompt's own
rows -- arrive as ``other_visible`` and split runs. A tool row's ``state`` is
``ToolCard``'s own settled/live vocabulary (``success``/``error``/
``interrupted``/``waiting``/``queued``/``composing``/``running``); this module
does not re-derive it.

WHAT FOLDS. A run is a maximal span of ``MIN_RUN_CALLS`` or more tool rows with
no splitter between them. A splitter is every row the reader must be able to
see: any prose, a notice, an ask or image (``other_visible``), the next user
row, and a call AWAITING APPROVAL (``waiting``) -- approval-pending work is
never hidden behind a fold. One or two back-to-back calls stay loose rows. A
failed call stays inside its run: it is counted on the bar, not released
(``error`` only -- a call the user stopped is ``interrupted``, which is not a
failure, the same distinction ``quiet_groups.is_failed_call`` makes). Inside a
run, ``MIN_SUB_CALLS`` or more consecutive calls of one tool become a
sub-group, because 16 browser rows say less than ``browser x16``.

KEYS ARE STABLE UNDER APPENDS. Rows only append at the tail, so a key derived
from the FIRST row of what it names never moves while that thing grows:
``tf:turn:<user row id>``, ``tf:run:<first tool row id>``,
``tf:sub:<first tool row id>``. A closed turn is therefore prefix-stable: a
later append cannot change it, which is what lets the view memoise closed turns
and keep a reader's open state on a key across appends and mode switches.

LIVE. ``live_from`` is the id of the first row of the transcript's live tail
(``TranscriptView.pinned_tail()``); the turn CONTAINING that row, and anything
after it, is live. A live turn draws no turn bar and is never ``died``: the
answer simply has not arrived yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

from rich.cells import cell_len

#: One row's visible role in a transcript. ``other_visible`` is the catch-all
#: for a row a reader must see that the outline has no rule about (an ask, a
#: generated image, an approval prompt): it splits a run exactly like prose.
RowKind = Literal[
    "user",
    "tool",
    "prose_narration",
    "prose_final",
    "notice_failed",
    "other_visible",
]

#: ``ToolCard``'s own state vocabulary (``tool_card.py``). ``error`` is the
#: failure; ``interrupted`` is the user stopping a call, not a failure.
ToolState = Literal[
    "success",
    "error",
    "interrupted",
    "running",
    "waiting",
    "queued",
    "composing",
]

#: Desktop's ``FOLD_MIN_ACTIONS``: three back-to-back actions are worth a bar.
MIN_RUN_CALLS = 3

#: Consecutive calls of ONE tool inside a run before they get their own level
#: (``browser x16``). Three of a tool reads as three rows; four is a rhythm.
MIN_SUB_CALLS = 4

#: Kinds named on a bar before the rest collapse into ``+N``.
MAX_SUMMARY_KINDS = 3

#: The reader-facing noun for a tool whose own name says less than its effect.
#: ``hub`` and ``send`` share one word so a turn that did both names it once.
KIND_WORDS: dict[str, str] = {
    "bash": "shell",
    "task": "subagents",
    "hub": "messages",
    "send": "messages",
}

#: The ink a summary clause asks for. ``danger`` marks a failure count only on
#: a turn that died; on a turn that answered, failures are history, not alarm —
#: ``failed_muted`` is that quieter role, kept distinct from ``muted`` so the
#: colour table can name the tally's own element (and a theme can ink it
#: differently from the kinds it sits beside).
SegmentRole = Literal["muted", "failed_muted", "dim", "danger"]


class Clause(NamedTuple):
    """One summary segment: the text and the role the bar styles it with."""

    text: str
    role: SegmentRole


@dataclass(frozen=True)
class OutlineRow:
    """One visible row, as the adapter saw it. Fields not meaningful for a
    ``kind`` stay at their defaults."""

    id: str
    kind: RowKind
    tool_name: str = ""
    #: Only a tool row carries a state. The default keeps non-tool rows
    #: harmless; a tool row must pass the card's real state, because ``error``
    #: is the only thing that counts a failure.
    state: ToolState = "success"
    #: The call's measured interval when one is known (replay restores it from
    #: the persisted ``provider_payload``); ``None`` for a legacy row, a call
    #: still running, or a row whose result never reached the transcript.
    duration_s: float | None = None
    summary: str = ""
    #: The visible head of a notice's text, lower-cased, for the ONE question
    #: the shape cannot answer: whether the notice says the turn FAILED or was
    #: merely interrupted/cut off. An error notice is a failure signal; a
    #: `turn cut off` row arrives in error ink too but is the user stopping or
    #: the provider truncating, and the design keeps those neutral.
    notice_head: str = ""


@dataclass(frozen=True)
class ToolFacts:
    """What a bar says about the calls under it. Nothing here is ever sent."""

    count: int
    #: ``(tool name, count)`` in first-appearance order -- the bar's kind list
    #: is ordered by what the reader met first, not alphabetically.
    kinds: tuple[tuple[str, int], ...]
    #: ``state == "error"`` calls only: an interruption is not a failure.
    failed: int
    #: The sum of the calls' KNOWN durations, or None when no call knows one.
    #: A running call contributes nothing: its true interval is not over, and
    #: the bar shows its live elapsed instead of a total.
    duration_s: float | None
    #: The call still running, if the span holds one.
    running_id: str | None


@dataclass(frozen=True)
class SubGroup:
    """``MIN_SUB_CALLS``+ consecutive calls of one tool inside a run."""

    key: str
    tool_name: str
    tool_ids: tuple[str, ...]
    facts: ToolFacts


@dataclass(frozen=True)
class LooseRow:
    """A row the reader sees on its own: a 1-2 call step, a narration line, a
    notice, an ask or image. It is in the outline to keep the reading ORDER of
    a turn's nodes, not to be folded."""

    id: str
    kind: RowKind


@dataclass(frozen=True)
class Run:
    """A maximal span of 3+ calls with nothing visible between them."""

    key: str
    #: Display order: a call row id, or a sub-group standing in for its calls.
    children: tuple[str | SubGroup, ...]
    facts: ToolFacts

    @property
    def tool_ids(self) -> tuple[str, ...]:
        """Every call in the run, flat, in reading order."""
        ids: list[str] = []
        for child in self.children:
            if isinstance(child, SubGroup):
                ids.extend(child.tool_ids)
            else:
                ids.append(child)
        return tuple(ids)

    @property
    def subgroups(self) -> tuple[SubGroup, ...]:
        """The sub-groups this run exposes, in reading order."""
        return tuple(child for child in self.children if isinstance(child, SubGroup))


@dataclass(frozen=True)
class Turn:
    """One user prompt and everything that answered it."""

    key: str
    #: The user row that opened the turn, or None for a page seam whose user
    #: row is above the loaded window (a resumed transcript can start mid-turn).
    user_id: str | None
    #: The assistant row that was the turn's answer, if one arrived.
    final_id: str | None
    #: The turn ended with no answer, or with a turn-failed notice. A live turn
    #: has not ended, so it never reads as dead.
    died: bool
    live: bool
    #: Facts over EVERY call in the turn, which is what the turn bar and the
    #: By-response foot line summarise -- runs alone would miss loose calls.
    facts: ToolFacts
    nodes: tuple[Run | LooseRow, ...]

    @property
    def has_turn_bar(self) -> bool:
        """Whether a turn bar may exist at all: an answered, settled turn.

        Necessary, not sufficient -- the view still draws no bar for a turn
        that would hide too few rows for the bar to pay for its own row.
        """
        return self.final_id is not None and not self.live

    @property
    def flat(self) -> bool:
        """The turn holds exactly ONE run, so its bar's children are that run's
        children: opening it steps straight to the calls instead of stacking a
        second bar that says the same thing.

        Loose rows that a turn bar cannot cover (an error notice, a receipt)
        do not earn the run a second bar of its own: the reader would see the
        turn bar and, one row under it, a run bar with the identical tally.
        """
        return sum(isinstance(node, Run) for node in self.nodes) == 1


@dataclass(frozen=True)
class Outline:
    """One transcript's turns, in order."""

    turns: tuple[Turn, ...]


def tool_facts(rows: Sequence[OutlineRow]) -> ToolFacts:
    """Tally a span of tool rows. Total over the span: a row with no duration
    simply does not contribute, and a span of none contributes None."""
    counts: dict[str, int] = {}
    failed = 0
    running: str | None = None
    known: list[float] = []
    for row in rows:
        counts[row.tool_name] = counts.get(row.tool_name, 0) + 1
        if row.state == "error":
            failed += 1
        if row.state == "running":
            # The last one wins: a settled run holds at most one call still
            # going, and that is always the tail.
            running = row.id
            continue
        if row.duration_s is not None:
            known.append(row.duration_s)
    return ToolFacts(
        count=len(rows),
        kinds=tuple(counts.items()),
        failed=failed,
        duration_s=sum(known) if known else None,
        running_id=running,
    )


def outline_of(records: Sequence[OutlineRow], live_from: str | None = None) -> Outline:
    """Derive the outline. Pure: the same rows always give the same outline.

    ``live_from`` names the first row of the live tail (see the module doc); an
    id that is no longer among ``records`` means nothing is live, which is the
    safe reading for a view whose live turn has just been replaced.
    """
    rows = tuple(records)
    live_index = next((i for i, row in enumerate(rows) if row.id == live_from), None)
    return Outline(
        turns=tuple(_turn_of(rows, start, end, live_index) for start, end in _turn_spans(rows))
    )


def summary_clauses(
    facts: ToolFacts,
    budget_cells: int,
    noanswer: bool,
    *,
    plural: Callable[[int], str] | None = None,
    failed: Callable[[int], str] | None = None,
    kind_word: Callable[[str], str] | None = None,
) -> tuple[Clause, ...]:
    """The bar's left-hand summary, per the prototype's ``barSummaryST``.

    ``N actions`` first, then the kinds it covered (``read, grep, glob +1``),
    then the failure count. Kinds shorten to ``KIND_WORDS`` and repeat once:
    a turn that sent two messages names ``messages`` once.

    ``budget_cells`` is the width left for this text after the row's own
    columns -- indent, glyph, and the RIGHT-ALIGNED time (or the live
    ``running`` label), which the bar draws itself. That is why shrinking
    drops kinds and never the time: the time is not in this text at all, and
    it has already been taken out of the budget. Nor is the failure count
    dropped: past a budget too small even for the count, the count is still
    what returns.

    ``plural`` renders the count clause so the caller can plug the catalogue's
    ICU plural (``tui.transcript.group.actions``); without it the English
    singular/plural pair is used. ``failed`` renders the tally the same way
    (``✗ K failed``) and defaults to the English literal, so this module stays
    runnable without a catalogue while the widget paints keyed copy.
    ``kind_word`` localizes a tool's short noun (``bash`` -> ``shell``) and
    folds an MCP name through the same display-name path a tool row uses;
    without it this module keeps its English fallback.
    ``noanswer`` marks a turn that died, which
    is the only state where the failure count earns danger ink.
    """
    count_clause = (plural or _english_actions)(facts.count)
    names: list[str] = []
    for name, _count in facts.kinds:
        # The caller owns localization (``kind_word`` maps a tool onto the
        # catalogue's short noun); without it this module keeps its English
        # fallback so it stays runnable with no catalogue, like ``plural``.
        word = kind_word(name) if kind_word is not None else KIND_WORDS.get(name, name)
        if word not in names:
            names.append(word)

    def build(shown: int) -> list[Clause]:
        clauses = [Clause(count_clause, "muted")]
        if shown > 0:
            more = len(names) - shown
            kinds = ", ".join(names[:shown]) + (f" +{more}" if more > 0 else "")
            clauses.append(Clause(" · ", "dim"))
            clauses.append(Clause(kinds, "muted"))
        if facts.failed:
            clauses.append(Clause(" · ", "dim"))
            tally = failed(facts.failed) if failed is not None else f"✗ {facts.failed} failed"
            clauses.append(Clause(tally, "danger" if noanswer else "failed_muted"))
        return clauses

    for shown in range(min(MAX_SUMMARY_KINDS, len(names)), -1, -1):
        clauses = build(shown)
        if _cells(clauses) <= budget_cells:
            return tuple(clauses)
    return tuple(build(0))


def _cells(clauses: Sequence[Clause]) -> int:
    """The clauses' painted width, in TERMINAL CELLS.

    ``cell_len``, never ``len``: a clause can be a translated count ("十六项
    操作" is 10 cells in 5 characters) or a tool's own name, so counting code
    points under-reports the row's length. The under-count is not cosmetic — it
    lets :func:`summary_clauses` believe a set of kinds fits when it does not,
    and the row then reaches the bar's ellipsis cut with kinds still in front of
    the failure tally, which is the one clause the design never drops.
    """
    return sum(cell_len(clause.text) for clause in clauses)


def _english_actions(count: int) -> str:
    return f"{count} action" if count == 1 else f"{count} actions"


def _turn_spans(rows: Sequence[OutlineRow]) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` index ranges, one per turn.

    A turn starts at its user row; anything before the first user row is a page
    seam's own turn (a resumed window can open mid-turn).
    """
    if not rows:
        return []
    starts = [0]
    starts.extend(i for i, row in enumerate(rows) if row.kind == "user" and i > 0)
    return [
        (start, starts[i + 1] if i + 1 < len(starts) else len(rows))
        for i, start in enumerate(starts)
    ]


def _turn_of(rows: Sequence[OutlineRow], start: int, end: int, live_index: int | None) -> Turn:
    span = rows[start:end]
    anchor = span[0] if span[0].kind == "user" else None
    body = span[1:] if anchor is not None else span
    # The answer is the LAST final-prose row: an earlier one is prose the turn
    # went on after, so it is a node (and a run splitter) like any other.
    final_index = next(
        (i for i in range(len(body) - 1, -1, -1) if body[i].kind == "prose_final"),
        None,
    )
    live = live_index is not None and start <= live_index < end
    # THE TURN BAR ENDS AT THE ANSWER. A row after it is not work the answer
    # condenses: a slash receipt, a system notice, a peer message or a wake
    # that arrives while the agent is idle must stay visible, and opening the
    # bar must not reveal a second bar over the same calls (the flat rule).
    cover_end = final_index if final_index is not None else len(body)
    pre_answer = body[:cover_end]

    nodes: list[Run | LooseRow] = []
    buffered: list[OutlineRow] = []

    def flush() -> None:
        if not buffered:
            return
        if len(buffered) >= MIN_RUN_CALLS:
            nodes.append(_run_of(buffered))
        else:
            nodes.extend(LooseRow(row.id, row.kind) for row in buffered)
        buffered.clear()

    for row in pre_answer:
        # A waiting call is approval-pending: it splits the run and stays a
        # readable row, because a fold must never hide a decision from the user.
        if row.kind == "tool" and row.state != "waiting":
            buffered.append(row)
            continue
        flush()
        nodes.append(LooseRow(row.id, row.kind))
    flush()

    return Turn(
        key=f"tf:turn:{anchor.id if anchor is not None else span[0].id}",
        user_id=anchor.id if anchor is not None else None,
        final_id=body[final_index].id if final_index is not None else None,
        died=not live and _ended_in_failure(span),
        live=live,
        facts=tool_facts([row for row in pre_answer if row.kind == "tool"]),
        nodes=tuple(nodes),
    )


#: Notice heads that name a turn the harness ENDED in failure (the harness's
#: error arm, in both its live and replay spellings).
_FAILED_NOTICE_HEADS = ("turn failed", "stopped with an error")

#: Heads that name a turn which stopped WITHOUT failing: the user interrupted
#: it, the provider cut it off, or it spent its output budget. The design keeps
#: these neutral — a danger tally and a "turn failed" foot would state an
#: outcome nobody diagnosed.
_NEUTRAL_NOTICE_HEADS = (
    "turn cut off",
    "interrupted",
    "no answer: the model spent",
)


def _ended_in_failure(span: Sequence[OutlineRow]) -> bool:
    """Whether a settled turn's rows end in a FAILURE notice.

    "No answer" alone is not a failure: a steer, a stop and a cut-off all leave
    a turn without an answer, and only the harness's own failed notice says the
    turn died. A failed CALL is not enough either — the run folds and counts it.

    Two arms, and the ANSWERED one is ordered on purpose. A turn that produced
    prose answered; a failure notice before that prose is history the turn went
    on from, and it cannot make the turn read as dead. For a turn with no answer
    the failed head may sit before the closing incident row (a provider failure
    replays as the raw `[session incident …]` record first and the stop notice
    last), so there the head test is unordered; the last-row test catches the
    other spelling, `Stopped with an error — …`, which has no such head.
    """
    if not span:
        return False
    last = span[-1]
    if last.kind == "notice_failed" and not last.notice_head.startswith(_NEUTRAL_NOTICE_HEADS):
        # The turn ENDED on an error: `turn failed`, `Stopped with an error`, or
        # a provider error that produced nothing. Cut-off/interrupted are
        # excluded by their neutral heads above.
        return True
    answered = any(row.kind == "prose_final" for row in span)
    if answered:
        return False
    return any(
        row.kind == "notice_failed" and row.notice_head.startswith(_FAILED_NOTICE_HEADS)
        for row in span
    )


def _run_of(rows: Sequence[OutlineRow]) -> Run:
    children: list[str | SubGroup] = []
    index = 0
    while index < len(rows):
        end = index
        while end < len(rows) and rows[end].tool_name == rows[index].tool_name:
            end += 1
        same_tool = rows[index:end]
        if len(same_tool) >= MIN_SUB_CALLS:
            children.append(
                SubGroup(
                    key=f"tf:sub:{same_tool[0].id}",
                    tool_name=same_tool[0].tool_name,
                    tool_ids=tuple(row.id for row in same_tool),
                    facts=tool_facts(same_tool),
                )
            )
        else:
            children.extend(row.id for row in same_tool)
        index = end
    return Run(key=f"tf:run:{rows[0].id}", children=tuple(children), facts=tool_facts(rows))
