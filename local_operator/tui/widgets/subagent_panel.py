"""The dock-band subagent panel (item 6-TUI).

The status band already carries the ◍/▣ counters; this panel is the DETAIL
surface — one row per task job in the session's job manager: label, state
glyph (a moving spinner while running, ✓/✗ once settled), elapsed time, and
the latest progress the engine relayed. A row is the click/Enter target for
the full-page subagent view (``widgets/subagent_view.py``), which renders the
child session's retained events as a transcript.

Rows are live by construction: the app repaints the panel on every Subagent*
event AND on the 1 Hz job poll, and the panel advances its own spinner while
anything is running — motion, not colour, says "alive": the accent green is
spent at exactly four sites (see the tcss preamble) and a fifth spinner is
not one of them. Settled rows follow the tool ledger's ink law: ✓ dim,
✗ danger, nothing else.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.costs import job_cost
from local_operator.tui.widgets.status_line import context_spelling, format_cost
from local_operator.tui.widgets.tool_card import (
    clean_intent,
    format_duration,
    truncate_cells,
)

#: Spinner cadence shared with the status band: 12.5 fps is the app's one
#: notion of "this is moving", and two different speeds on one screen would
#: read as two different states. PUBLIC because the full-page subagent view
#: (``widgets/subagent_view.py``) animates the same job's state in its title
#: — the band row and the page header must not be able to drift apart.
SPINNER_FRAMES = ("⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷")
SPINNER_INTERVAL_S = 0.08

#: Status glyphs reuse the tool ledger's vocabulary: the outcome is readable
#: in a colourless frame (✓/✗), and failure is the only state that gets colour.
GLYPH_DONE = "✓"
GLYPH_FAILED = "✗"
#: Cancelled reuses the ledger's interrupted mark, and a queued job gets a
#: WAITING mark rather than a settled one: it has not run, so ✓ would lie.
GLYPH_CANCELLED = "⊘"
GLYPH_QUEUED = "⏳"
#: A child that was RUNNING when a previous process exited, rehydrated from the
#: persisted roster on resume (see ``AsyncJobManager`` status ``interrupted``).
#: Its own mark — neither ✓ (it did not finish) nor ✗ (it did not fail): the
#: run was cut off, and if its transcript survived it can be resumed.
#:
#: ``↺`` (U+21BA) over the earlier ``⇥``: the tab-arrow read as a Tab key once
#: the accompanying word dropped under the panel's narrow-width reduction
#: (design review round 1, D1), whereas the open circle-arrow says "suspended,
#: pick it back up" — which is exactly the resumability the state carries, so
#: it doubles as the D2 cue. Chosen for width too: it is ``east_asian_width=N``
#: (locked to one cell) and outside the emoji-presentation block, so unlike the
#: media-control marks (``⏸``/``⏯``, siblings of the wide ``⏳`` queued glyph) it
#: cannot balloon to a two-cell colour emoji and shear the time column.
GLYPH_INTERRUPTED = "↺"

#: Rows the panel spends on chrome rather than on jobs: the ``Subagents``
#: caption. Named because :meth:`SubagentPanel.predicted_rows` adds it to the
#: job count, and a reader of that sum should not have to count `compose`.
_HEADER_ROWS = 1

#: Seam between the numbers on a row. The same ` · ` the full-page view's
#: title uses between its own facts, one tone under them — a row and the page
#: it opens must punctuate the same way.
STATS_SEAM = " · "

#: Cells between the numbers and the activity. FOUR, and wider than the
#: 3-cell seam on purpose: this is the break between the row's numbers and its
#: one sentence, and a break narrower than the separators inside each group
#: loses to proximity and dissolves the row into one run. It was two, and the
#: band four rows below had already litigated the same point and reached the
#: same answer for the same reason (``status_line._MIN_GROUP_GAP``).
ACTIVITY_GAP = 4

#: Below this the activity is not worth its cells. Sized to hold ``running 3
#: tools`` (15) whole, because that phrase truncated is a phrase that has lost
#: the only number it carried.
ACTIVITY_FLOOR = 16

#: The narrowest a label may be squeezed. It is the LAST thing to yield and it
#: never yields past this: `MR review agent round 1` and `… round 2` are the
#: same twelve cells, so a shorter floor buys width by making rows
#: indistinguishable — which is the one thing a list of rows may not do.
LABEL_FLOOR = 12

#: Width the ladder assumes before the first layout has measured a row. Wide
#: rather than narrow on purpose: an over-generous guess degrades to Rich's
#: own ellipsis at the row's real edge, while an under-generous one would shed
#: a segment the terminal had room for and then flicker it back on the resize.
_DEFAULT_ROW_WIDTH = 120


def status_glyph(
    status: str, *, queued: bool = False, spinner_glyph: str = ""
) -> tuple[str, str, str]:
    """``(glyph, word, semantic colour token)`` for one task job's state.

    One function rather than two branch ladders, because the band row and the
    full-page view render the SAME job side by side — a user who leaves the
    page and reads the row must not meet a second vocabulary for the state
    they were just looking at.

    The WORD comes back with the glyph rather than being read off the job,
    and that is not tidiness: a queued job's ``status`` is still ``running``,
    so the page's title paired ``⏳`` — the mark that means "has not started"
    — with the word ``running``. Two facts derived in two places will
    eventually disagree; derived here they cannot.

    A running job's glyph is the caller's spinner frame: motion says "alive"
    and the ink stays neutral, so the accent green is not spent a sixth time.
    """
    if queued:
        return GLYPH_QUEUED, "queued", "dim"
    if status == "running":
        return spinner_glyph or SPINNER_FRAMES[0], "running", "muted"
    if status == "failed":
        return GLYPH_FAILED, "failed", "danger"
    if status == "cancelled":
        return GLYPH_CANCELLED, "cancelled", "dim"
    if status == "interrupted":
        # Rehydrated from a previous process's roster; the run was cut off, not
        # finished or failed. Muted rather than danger — nothing went wrong, the
        # process simply ended — and its own word so a reader can tell it apart
        # from a clean cancel and know it may be resumable.
        return GLYPH_INTERRUPTED, "interrupted", "muted"
    return GLYPH_DONE, status or "completed", "dim"


def job_seconds(job: Any) -> float:
    """How long the job has run, in seconds; ``0.0`` when it cannot be known.

    A running job is measured against now; a settled one against its settle
    time — so a row read five minutes after completion still reports the job's
    own duration, not how long ago the user glanced at it. Missing clocks
    degrade to zero rather than a negative or a crash: an observability
    surface must not be able to take the app down.
    """
    try:
        start = float(getattr(job, "start_time", 0.0) or 0.0)
        settled = getattr(job, "settled_at", None)
        end = float(settled) if settled else time.time()
        return max(end - start, 0.0)
    except Exception:
        return 0.0


def job_elapsed(job: Any) -> str:
    """:func:`job_seconds` in the ledger's own grammar (``7m49s``).

    The seconds are exposed separately because the status band needs the
    NUMBER — its duration segment does its own formatting and compares against
    zero to decide whether the segment exists at all.
    """
    return format_duration(job_seconds(job))


@dataclass(frozen=True)
class JobStats:
    """The numbers one child session has reported about ITSELF.

    Never invented. A child whose provider has not reported usage yet has no
    context reading and no cost, and every surface below OMITS those segments
    rather than printing a zero — ``$0.0000`` on a row that has spent money
    reads as "this was free", which is exactly why ``costs.job_cost`` answers
    ``None`` instead of a confident number.
    """

    #: The CHILD's model, which is not always the parent's: ``run_subagent``
    #: takes a ``model_spec`` override, and a child on a different model is
    #: the case these stats exist to make visible.
    model_label: str = ""
    context_tokens: int = 0
    context_window: int = 0
    cost: float | None = None
    #: Whether the child has reported ANY usage. ``cost is None`` conflates
    #: two facts — "no price exists for this model" and "nothing has been
    #: reported yet" — and only the first of them is worth an ``$—``.
    billed: bool = False


def job_stats(job: Any, *, default_model_label: str = "") -> JobStats:
    """One task job's model/usage/cost facts, read defensively.

    ``default_model_label`` is the PARENT's label. A child launched with no
    ``model_spec`` override inherits the parent's model and records nothing on
    the job, so falling back to it is the NORMAL path and not a guess — the
    engine writes ``AsyncJob.model_label`` only when it knows the child's own.

    Context is the child's last point-in-time reading (``Usage.context_tokens``
    is REPLACED per turn by the engine, never summed, because window occupancy
    is a level and not a total), falling back to ``input_tokens`` — which IS a
    running sum across the child's turns and therefore only an approximation
    of occupancy, used because a provider that reports no context size still
    reports what it was billed for. Same precedence the parent's band uses
    (``tui/events.py``), so one number cannot mean two things four rows apart.

    Never raises, and the money is never computed here: :func:`costs.job_cost`
    is the app's ONE pricing path and this function only reads it.

    Neither is the context WINDOW resolved here, and that is a correctness
    requirement rather than tidiness. This runs on the paint path, off a
    12.5 fps timer, for every child the panel has not cached — and
    ``resolve_model_info`` is memoized but not free on a miss: the current
    Claude registry rows carry ``limits_from_listing``, so a miss enters
    provider discovery, which is an HTTP listing fetch behind a disk cache.
    Measured: 0.007 ms warm, 45 ms on a warm disk cache, 222 ms cold, and 13 s
    against a slow host for a model the shipped registry does not describe
    (a 10 s listing budget, then 3 s of aggregator catalogue) — thirteen
    seconds of a TUI that does not read the keyboard, on the panel whose
    docstring already records having dropped keystrokes once. That model is
    not an edge case here: ``run_subagent`` takes a ``model_spec`` override
    and a child on an unlisted model is precisely what these stats exist to
    surface. A slow call is not an exception, so the guard below would not
    have caught it, and lowering a timeout would not fix it — a repaint must
    do no I/O at any budget. The window is therefore read off the JOB, where
    the engine records it at launch from the child's own ModelSpec; a job that
    carries none renders its context as a token count rather than a
    percentage, which is the honest degradation.
    """
    model_label = default_model_label or ""
    tokens = 0
    window = 0
    cost: float | None = None
    billed = False
    try:
        model_label = str(getattr(job, "model_label", "") or default_model_label or "")
        usage = getattr(job, "usage", None)
        billed = usage is not None
        if billed:
            tokens = int(
                getattr(usage, "context_tokens", None) or getattr(usage, "input_tokens", 0)
            )
        window = int(getattr(job, "context_window", 0) or 0)
        cost = job_cost(job, default_model_label=default_model_label or None)
    except Exception:
        # A job the panel cannot read is a row with fewer numbers on it, never
        # a broken frame: this is observability, and it runs against embedder
        # hosts and replayed ledgers whose job objects are not ours. The whole
        # read is inside the guard, including the priced part: `job_cost`
        # promises not to raise on an unpriceable model, which is not the same
        # promise as not raising on a job that refuses to be read.
        pass
    return JobStats(
        model_label=model_label,
        context_tokens=max(tokens, 0),
        context_window=max(window, 0),
        cost=cost,
        billed=billed,
    )


@dataclass(frozen=True)
class RowFacts:
    """What one row SAYS, before anything decides how much of it fits.

    Derived away from the widget so the panel can measure every row before
    any of them paints — the reduction below is a decision about the LIST,
    not about a row, and a list cannot be measured one widget at a time.
    """

    label: str
    status: str
    queued: bool
    running: bool
    elapsed: str
    activity: str
    #: Whether the full-page view is showing THIS child. The row then stops
    #: restating what the page's title says three rows above it — see
    #: :func:`compose_row`.
    current: bool = False
    #: The child's ROLE (``AsyncJob.agent_role``, recorded at registration by
    #: ``run_subagent``), or ``""`` when there is nothing worth saying.
    #:
    #: Empty for the ``"task"`` default, matching the full-page view's title
    #: (``subagent_view._title_row``): every child is a task unless told
    #: otherwise, so the word says nothing a reader did not already assume and
    #: would cost eight cells on every ordinary row. Without this the list
    #: names a child only by the label its parent happened to choose, so
    #: whether ``review-301-r2`` is a reviewer or a coder that was asked to
    #: look at a review was a guess.
    agent_role: str = ""


def row_facts(job: Any, *, fallback_id: str, current: bool) -> RowFacts:
    """Read one job into the strings a row paints. Never raises.

    Guarded whole, and not merely with ``getattr`` defaults: a default only
    covers a MISSING attribute, while the hosts this widget has to survive —
    embedders and replayed ledgers whose job objects are not ``AsyncJob`` —
    hand it properties that raise. This is called from the 1 Hz poll and from
    the ``Subagent*`` handlers, so an exception here is an unhandled Textual
    message-handler exception, i.e. the whole app, for a status row.
    """
    try:
        return _read_row(job, fallback_id=fallback_id, current=current)
    except Exception:
        return RowFacts(
            label=fallback_id,
            status="running",
            queued=False,
            running=False,
            elapsed="0s",
            activity="",
        )


def _read_row(job: Any, *, fallback_id: str, current: bool) -> RowFacts:
    """:func:`row_facts` without the net. Everything here may raise."""
    status = str(getattr(job, "status", "running"))
    queued = bool(getattr(job, "queued", False))
    running = status == "running" and not queued
    details = getattr(job, "latest_details", None)
    if not isinstance(details, dict):
        details = {}
    if queued:
        # The word ``status_glyph`` already derived for this state, rather
        # than a second one: a queued row otherwise spends its whole width on
        # a bullet, a label and a clock, and ``⏳`` alone is the only thing
        # saying the child has not started.
        activity = status_glyph(status, queued=True)[1]
    elif running:
        # The relay's progress string, which is the child's ACTIVITY in the
        # working line's own vocabulary — the model's intent while a tool
        # runs, `running N tools` for a batch, `responding`, `thinking`.
        # `clean_intent` is the same boundary re-check the tool ledger runs on
        # model-written text before painting it, so `Auditing merged MRs.`
        # reads as `auditing merged MRs` here and in the card above.
        activity = clean_intent(str(details.get("progress") or "")) or ""
    elif status == "failed":
        # Settled rows carry the outcome's first line instead: the band row is
        # the summary, the full-page view is the detail, and the row between
        # them says which side of that split it is on.
        activity = " ".join(
            strip_control_sequences(str(getattr(job, "error_text", "") or "")).split()
        )
    else:
        activity = " ".join(
            strip_control_sequences(str(getattr(job, "result_text", "") or "")).split()
        )
        if not activity and status == "cancelled":
            # ``cancelled`` was the only state that painted no word: a job
            # cancelled mid-run records no ``result_text``, so the row's whole
            # state rested on a 1-cell ``⊘`` while the page it opens prints
            # ``⊘ cancelled`` and the sibling ``queued`` branch already spells
            # itself. Cancelled-while-PARKED does not reach here empty — the
            # manager stamps ``CANCELLED_BEFORE_START`` on it, which is the
            # more specific sentence and wins on its own.
            activity = status_glyph(status)[1]
        elif not activity and status == "interrupted":
            # A restored ``interrupted`` row carries no ``result_text`` (its run
            # never settled), so like the cancelled-mid-run case it would rest
            # on a 1-cell glyph alone. Spell the word so the row says WHY it is
            # not a clean outcome — the process ended under it — and reads the
            # same as the page it opens.
            activity = status_glyph(status)[1]
    if current and not running:
        # The page IS this row's detail, three rows above it. Repeating a
        # SETTLED outcome here printed a failed child's error string twice in
        # one frame, 14 rows apart, in the same danger ink. A RUNNING child's
        # activity is a short phrase and the one thing on the row that moves,
        # and glancing at the dock instead of scanning the page is why the
        # dock stayed visible — so that one is kept.
        activity = ""
    # ``strip_control_sequences`` for the same reason the label gets it: a
    # role name reaches this from an operator-authored registry entry, which
    # is not a trusted source of terminal-safe text.
    agent_role = strip_control_sequences(str(getattr(job, "agent_role", "") or "")).strip()
    if agent_role == "task":
        # The no-role default carries no information (see ``RowFacts``).
        agent_role = ""
    return RowFacts(
        label=strip_control_sequences(str(getattr(job, "label", "") or fallback_id)),
        status=status,
        queued=queued,
        running=running,
        elapsed=job_elapsed(job),
        activity=activity,
        current=current,
        agent_role=agent_role,
    )


#: The reduction ladder, widest first. Each rung names which context spelling
#: and whether the cost survives, and how many cells the label may keep.
#:
#: **What degrades, in order.** A full row does not fit 60 columns, so this is
#: load-bearing rather than defensive, and it follows the status band's own
#: reduction order (``status_line._DROP_LADDER``) because the two surfaces are
#: read in one glance:
#:
#: 0. The ROLE sheds first, and it is the only rung above the ladder as it
#:    stood: the widest rung is the previous widest rung PLUS the role, so a
#:    terminal that showed a given set of fields before this column existed
#:    still shows exactly that set at exactly that width, and only a terminal
#:    with spare cells gains the new one. That is what makes this column
#:    additive rather than a re-litigation of every width below it. It leads
#:    for the same reason the full-page view's title makes it the most
#:    disposable field (``subagent_view._title_row``): the role is identity
#:    sugar the label already half-carries, and it is re-derivable by opening
#:    the page, where a state or a number is not. The two surfaces are read in
#:    one glance and must not rank the same field differently.
#: 1. COST sheds next — it is monotonic and slow, and nothing an operator
#:    acts on inside a second.
#: 2. CONTEXT then SHORTENS before it drops: ``5%`` keeps the segment for two
#:    cells instead of thirteen, and a bare percentage still answers "is this
#:    child about to compact". Shortening before dropping is the band's
#:    principle too.
#: 3. CONTEXT drops.
#: 4. The LABEL yields, last, and only to :data:`LABEL_FLOOR`.
#:
#: The ACTIVITY is never traded for a number. It is the row's only statement
#: of what the child is DOING — every number here is re-derivable by opening
#: the page — so it truncates against :data:`ACTIVITY_FLOOR` and disappears
#: only when even the floor cannot be paid.
#: ``(context spelling, keep cost, keep role, label budget)``; ``None`` means
#: "a third of the width", which is only known at measure time.
#:
#: The context has THREE spellings, not two, because the middle one is what
#: keeps a mixed-model fan-out comparable: ``24%`` and ``31%`` beside each
#: other read as similar loads when one child is holding 48k of 200k and the
#: other 311k of 1M. Dropping the decimal saves two cells and keeps the
#: denominator; dropping the denominator saves seven and keeps only a ratio
#: whose base the reader would have to open the page to learn.
#:
#: The reduction is monotone in width down to the point where the label budget
#: reaches :data:`LABEL_FLOOR` and stops tracking the row (about 54 cells for a
#: 23-cell label). Below that the row narrows and its label does not, so a
#: dropped segment can become affordable again for a column or two. Accepted:
#: it is beneath any width this dock is read at, and the alternative — freezing
#: the budget once it stops shrinking — buys monotonicity by holding cells the
#: narrowest rows most need.
_RUNGS: tuple[tuple[str, bool, bool, int | None], ...] = (
    ("full", True, True, None),
    ("full", True, False, None),
    ("full", False, False, None),
    ("nodec", False, False, None),
    ("short", False, False, None),
    ("none", False, False, None),
    ("none", False, False, LABEL_FLOOR),
)


def _row_rung(facts: RowFacts, stats: JobStats, width: int, column: int, clock: int = 0) -> int:
    """The LEAST-REDUCED rung this row still reads correctly at.

    Walks :data:`_RUNGS` widest-first and takes the first that fits, so a
    smaller return is a richer row and :func:`panel_layout`'s ``max`` picks
    the reduction every row can live with. Well-founded because both the
    numbers and the label budget are non-increasing in the index, so
    acceptance is monotone and the first hit is the answer.
    """
    for index in range(len(_RUNGS)):
        label, role, context, cost, activity = _lay_out(facts, stats, width, index, column, clock)
        head = (
            _CHROME_CELLS
            + max(cell_len(label), column)
            + max(cell_len(facts.elapsed), clock)
            + _glyph_cells(facts)
        )
        numbers = sum(len(STATS_SEAM) + cell_len(part) for part in (role, context, cost) if part)
        if not facts.activity:
            if head + numbers <= width:
                return index
        elif activity and cell_len(activity) >= min(ACTIVITY_FLOOR, cell_len(facts.activity)):
            return index
    return len(_RUNGS) - 1


#: Everything on a row that is not the label, the glyph, the numbers or the
#: activity: ``• `` + the two spaces after the label + the space after the
#: glyph. The elapsed reading and the GLYPH COLUMN are added by the caller,
#: being the two chrome fields whose width varies with the row.
_CHROME_CELLS = 2 + 2 + 1

#: Cells the state mark is padded out to, so the clock, the numbers and the
#: activity start in the same column on every row. ``⏳`` (U+23F3) is
#: ``East_Asian_Width=W`` and draws two cells where every other mark draws
#: one, so an unpadded queued row pushed everything after it one cell right —
#: its clock alone out of the column the other rows right-align into.
_GLYPH_COL = 2


def _glyph_cells(facts: RowFacts) -> int:
    """Cells the state mark's COLUMN occupies — measured, never assumed.

    A running row's spinner frame and ``✓``/``✗``/``⊘`` are one cell each and
    ``⏳`` is two, and the set is shared with the full-page view's title, so
    this widget does not get to pick. Any frame will do for the measurement:
    the spinner glyphs are the same width as each other by construction (the
    band's arithmetic already depends on that).

    The answer is the padded column rather than the raw glyph, because that is
    what :func:`compose_row` spends — a budget measured one cell short of what
    the row draws is what lets ``compose_row``'s own truncate eat the last
    cell of a dollar figure.
    """
    glyph, _word, _token = status_glyph(facts.status, queued=facts.queued)
    return max(_GLYPH_COL, cell_len(glyph))


def _lay_out(
    facts: RowFacts, stats: JobStats, width: int, rung: int, column: int = 0, clock: int = 0
) -> tuple[str, str, str, str, str]:
    """``(label, role, context, cost, activity)`` exactly as ``rung`` paints them.

    ``column`` is the shared label width the row will be padded to; the
    activity is measured against it, not against this row's own label, or a
    short-labelled row would be handed slack the padding is about to spend.
    """
    spelling, keep_cost, keep_role, budget = _RUNGS[min(rung, len(_RUNGS) - 1)]
    # The last rungs carry no reading at all; every other spelling is one the
    # band names too (`status_line.CONTEXT_FORMS`), so the same child's
    # occupancy cannot be written two ways four rows apart.
    context = (
        ""
        if spelling == "none"
        else context_spelling(stats.context_tokens, stats.context_window, form=spelling)
    )
    # A row with no role to show pays nothing for the column at any rung: the
    # segment is per-ROW, unlike cost, which sheds for the whole list at once
    # so a blank cell there cannot be read as "did not fit". Here a missing
    # role means the child is a plain task, which is the same thing the
    # full-page title says by omitting it.
    role = facts.agent_role if keep_role else ""
    if not keep_cost:
        cost = ""
    elif stats.cost is not None:
        cost = format_cost(stats.cost)
    else:
        # The band's own vocabulary for "billed, and nobody can price it".
        # Safe on a row now that the whole column sheds at one rung: at a rung
        # that keeps cost it is kept for EVERY row, so a blank cell there
        # means unpriced and cannot also mean "did not fit". A child that has
        # reported no usage at all still prints nothing — it has not spent
        # anything anyone knows of, which is a different fact again.
        cost = "$—" if stats.billed else ""
    label = truncate_cells(
        facts.label, budget if budget is not None else max(LABEL_FLOOR, width // 3)
    )
    head = (
        _CHROME_CELLS
        + max(cell_len(label), column)
        + max(cell_len(facts.elapsed), clock)
        + _glyph_cells(facts)
    )
    numbers = sum(len(STATS_SEAM) + cell_len(part) for part in (role, context, cost) if part)
    room = width - head - numbers - ACTIVITY_GAP
    activity = truncate_cells(facts.activity, room) if facts.activity and room > 0 else ""
    return label, role, context, cost, activity


def panel_layout(rows: Sequence[tuple[RowFacts, JobStats]], width: int) -> tuple[int, int, int]:
    """``(rung, label column)`` for the whole list — the panel's one layout.

    Two decisions that cannot be made apart. The rung says which fields every
    row carries; the column says where they sit, so that two children's spend
    can be compared by looking down instead of reading across. Padding costs
    the shorter rows cells they were spending on activity slack, which can
    push a row under its floor — so the rung is re-checked WITH the column
    applied and reduced until the pair fits, rather than each being solved
    against a layout the other has not happened yet.

    Terminates: the column is non-increasing in the rung (the last rung caps
    the label at :data:`LABEL_FLOOR`) and the final rung carries no numbers at
    all, so the loop is bounded by ``len(_RUNGS)``.
    """
    # The clock column is width-independent — a duration is as wide as it is —
    # so it is settled once, before the rung search that consumes it.
    clock = max((cell_len(facts.elapsed) for facts, _ in rows), default=0)
    rung = max((_row_rung(facts, stats, width, 0, clock) for facts, stats in rows), default=0)
    while rung < len(_RUNGS) - 1:
        column = _label_column(rows, width, rung, clock)
        if all(_row_rung(facts, stats, width, column, clock) <= rung for facts, stats in rows):
            return rung, column, clock
        rung += 1
    return rung, _label_column(rows, width, rung, clock), clock


def _label_column(
    rows: Sequence[tuple[RowFacts, JobStats]], width: int, rung: int, clock: int = 0
) -> int:
    """Cells the label field occupies on EVERY row, so the rest aligns.

    One rung gave the column the same fields; this gives them the same
    POSITIONS, which is the half that makes them comparable. With the label
    inline and variable-length, one child's cost sat at column 63 and its
    neighbour's at 17, so reading two children's spend meant scanning both
    rows left to right hunting for a ``$`` — the expensive half of the
    promise with the cheap half of the payoff.

    Costs nothing on the widest row, which already occupies the column, and
    converts trailing activity slack into leading whitespace on the others.
    """
    return max(
        (cell_len(_lay_out(facts, stats, width, rung, 0, clock)[0]) for facts, stats in rows),
        default=0,
    )


def compose_row(
    *,
    facts: RowFacts,
    stats: JobStats,
    spinner_glyph: str,
    width: int,
    rung: int,
    column: int = 0,
    clock: int = 0,
) -> Text:
    """One row at the panel's chosen ``rung``: identity, state, numbers, activity.

    ``• <label>  <glyph> <elapsed> · <context> · <cost>    <activity>``

    The numbers carry NO icons, which departs from the status band four rows
    below. The band packs eight segments and needs ``▦``/``◈`` to say which
    number is which; here a percentage, a dollar sign and a duration each name
    themselves, and two icons would cost four of the cells this row is
    fighting over. Nor do they take the band's ``signal``/``warning`` inks:
    this panel's ink law spends colour on failure and on nothing else (see the
    module docstring), and N rows of warning-coloured money in the dock would
    be N false alarms.

    They are ``muted`` and not ``dim``, which is a separate decision from the
    colour one and was got wrong first time round: measured against this
    panel's ground, ``dim`` is 4.18:1 — under WCAG AA — and it was carrying
    every number this surface exists to add, while the label the reader
    already knows sat at 13.76:1. ``muted`` is 7.93:1 and spends no colour.

    ``column`` pads the label to the panel's shared width; zero means "lay
    this row out alone", which is what a test measuring one row wants.
    """
    fg = Style(color=theme_mod.semantic_color("fg"))
    muted = Style(color=theme_mod.semantic_color("muted"))
    dim = Style(color=theme_mod.semantic_color("dim"))
    faint = Style(color=theme_mod.semantic_color("faint"))

    label, role, context, cost, activity = _lay_out(facts, stats, width, rung, column, clock)
    glyph, _word, token = status_glyph(
        facts.status, queued=facts.queued, spinner_glyph=spinner_glyph
    )

    row = Text(no_wrap=True, overflow="ellipsis")
    row.append("• ", style=dim)
    row.append(label, style=fg)
    row.append(" " * max(0, column - cell_len(label)), style=dim)
    row.append("  ", style=dim)
    if facts.current:
        # The page's own title says the state and the age, three rows above
        # this. Repeating them here made one child's state four separate
        # statements inside twenty-five rows, in three vocabularies; the row's
        # job in this mode is WHICH subagent, and the numbers are the one
        # thing the title does not carry. Non-current rows are unchanged.
        for index, value in enumerate(part for part in (role, context, cost) if part):
            if index:
                row.append(STATS_SEAM, style=faint)
            row.append(value, style=muted)
        row.truncate(width, overflow="ellipsis")
        return row
    # Shared with the full-page view's title so one job cannot read as two
    # different states on two surfaces. A running row's glyph advances on the
    # panel's timer, so a stopped timer means a frozen row — visible at once.
    row.append(glyph, style=Style(color=theme_mod.semantic_color(token)))
    # Padded to one shared column so what follows starts in the same cell on
    # every row. Without it a `⏳` row — two cells where every other mark is
    # one — carried its clock, numbers and activity one cell right of the
    # column the rest of the list right-aligns into, which is visible as a
    # single stepped row in a list whose whole job is comparison.
    row.append(" " * max(0, _GLYPH_COL - cell_len(glyph)), style=dim)
    # RIGHT-aligned into the panel's clock column, which is the one field D2's
    # label column did not settle: `22s` is two cells narrower than `7m49s`,
    # and that row's context and cost then sat two cells left of everybody
    # else's. Durations right-align by convention and the pad abuts the glyph,
    # so no seam moves.
    row.append(f" {facts.elapsed.rjust(max(clock, cell_len(facts.elapsed)))}", style=muted)
    # The role leads the numbers run rather than trailing it: it qualifies WHAT
    # the child is, the way the page title reads ``Subagent · reviewer ·
    # <label>``, and reading it after the clock and the percentage would put
    # the row's identity behind its measurements. It takes the same ` · ` seam
    # and ``muted`` ink as every other segment here — a second visual idiom
    # beside an established one is a defect — which also means it inherits the
    # contrast decision recorded above (``dim`` is under WCAG AA on this
    # ground; ``muted`` is 7.93:1).
    for value in (role, context, cost):
        if value:
            row.append(STATS_SEAM, style=faint)
            row.append(value, style=muted)
    if activity:
        row.append(" " * ACTIVITY_GAP, style=dim)
        row.append(activity, style=muted)
    # Below the narrowest rung — a dock a dozen cells wide — even the label's
    # floor plus the clock overruns, and the ladder has nothing left to give.
    # Cut here rather than leaving it to the renderer, so the string this
    # function returns is the string a reader sees and can be measured as one
    # (the full-page view's title row does the same at its own edge).
    row.truncate(width, overflow="ellipsis")
    return row


class SubagentRow(Static):
    """One task job: bullet, label, state glyph, elapsed, numbers, activity.

    The whole row is the click target for the full-page subagent view, and
    Enter on a focused row does the same — the keyboard and the mouse must
    agree, or one of them is a guess.

    The layout is :func:`compose_row`'s, kept OUT of the widget so the width
    ladder can be exercised at any width without a running app — the widget's
    own width is a fact only a mounted layout has, and "what does this row
    shed at 60 columns" is a question the tests have to be able to ask.
    """

    can_focus = True
    BINDINGS = [
        Binding("enter", "open_subagent", "Open subagent", show=False),
    ]

    def __init__(self, job_id: str, on_open: Callable[[str], None]) -> None:
        super().__init__(classes="subagent-row")
        self._job_id = job_id
        self._on_open = on_open
        #: Painted fingerprint, so a repaint happens only when the row's
        #: inputs move — the spinner tick would otherwise redraw every row
        #: eight times a second.
        self._fingerprint: tuple[Any, ...] | None = None
        self._running = False
        #: True while the full-page view is showing THIS job. The row is the
        #: only interactive thing left in the dock in that mode, and a column
        #: of identical bullets cannot answer "which one am I reading" — the
        #: page title and the row truncate a model-written sentence by
        #: different rules, so comparing them is not an answer either.
        self._current = False
        #: Whether this row is the one the page is showing is a property of
        #: the row; the WIDTH is not — the panel measures it once for the
        #: whole list, so it arrives as an argument and lives in the
        #: fingerprint rather than in a field here.

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def running(self) -> bool:
        return self._running

    @property
    def current(self) -> bool:
        """Whether the full-page view is showing THIS job."""
        return self._current

    def set_current(self, current: bool) -> None:
        """Mark this row as the one the full-page view is showing.

        The tint is the app's existing word for "this row is the one" (the
        same ``tint-select`` step a focused approval carries), so no token and
        no colour is spent on the mode.
        """
        if current == self._current:
            return
        self._current = current
        self.set_class(current, "current")
        self._fingerprint = None  # the row's content changes with the mark

    def paint(
        self,
        facts: RowFacts,
        *,
        stats: JobStats,
        spinner_glyph: str,
        width: int,
        rung: int,
        column: int,
        clock: int,
    ) -> bool:
        """Repaint from already-derived state; returns the row's running-ness.

        Everything is handed in rather than read here, and that is what makes
        the panel-wide ladder possible: the reduction is a decision about the
        whole list, so the list has to be measured before any row of it is
        painted (see :func:`panel_layout`). Deriving the numbers upstream also
        lets the panel decide how OFTEN they are read — the spinner calls this
        eight times a second, and a model resolution per row per frame is work
        nobody can see.
        """
        self._running = facts.running
        fingerprint = (
            facts,
            stats,
            spinner_glyph if facts.running else "",
            self._current,
            width,
            rung,
            column,
            clock,
        )
        if fingerprint == self._fingerprint:
            return facts.running
        self._fingerprint = fingerprint
        # ``layout=False``: the sheet fixes a row at ``height: 1`` and the row
        # is built ``no_wrap``/``ellipsis`` to that width, so its content can
        # never move the box. Textual's default would reflow the whole screen
        # — 7.8 ms across 173 widgets on a 161-block transcript — and this
        # runs 12.5 times a second per running child for as long as the child
        # is alive. Measured with three running rows and 161 blocks: 5.2% of a
        # core with the default, 2.1% with this.
        self.update(
            compose_row(
                facts=facts,
                stats=stats,
                spinner_glyph=spinner_glyph,
                width=width,
                rung=rung,
                column=column,
                clock=clock,
            ),
            layout=False,
        )
        return facts.running

    # No `on_resize` here: the rows are full-width children of one panel, so
    # the container's own resize covers every one of them in a single dirty
    # mark (:meth:`SubagentPanel.on_resize`) instead of N widgets each
    # scheduling their own repaint through a window drag.

    def action_open_subagent(self) -> None:
        self._on_open(self._job_id)

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._on_open(self._job_id)


class SubagentPanel(Container):
    """The task-job list in the dock band.

    Visibility follows the ledger: shown while the manager has ANY task job
    (running or settled within retention — the manager's own sweep evicts the
    rest), hidden when it has none. Rows are keyed by job id and reused
    across repaints, so focus survives the 1 Hz refresh.

    **One timer paints this panel.** :meth:`sync` is called on every
    ``Subagent*`` event as well as on the 1 Hz poll, and a child emits an
    event per tool start, per tool end and per message boundary — so with N
    children the event-driven path was N rows repainted per event, i.e. work
    quadratic in the fan-out on the same thread that reads the keyboard. That
    exact shape has already dropped keystrokes on this branch once. So a sync
    now marks the panel DIRTY and the tick below is the only thing that
    paints: bursts between two ticks collapse into one repaint, and the
    ceiling is the tick rate no matter how loud the children are. Measured
    over one simulated second of a busy fan-out (each running child relaying
    four events, plus the 1 Hz poll, plus thirteen ticks), app-thread time
    spent in this panel, per-event shape → coalesced: 3 children 2.61 ms →
    1.66 ms, 6 children 5.55 ms → 3.22 ms, 12 children 14.13 ms → 6.18 ms,
    24 children 39.87 ms → 12.28 ms. The old shape grows with the SQUARE of
    the fan-out and the new one does not, which is the whole point; what
    remains is the spinner's own 12.5 fps over the running rows, which is
    where 224 of the 24-child repaints go and is not this diff's to spend.

    The one exception is a row APPEARING or leaving, which is painted at once:
    that changes the dock's height, and a height that changes 80 ms after the
    row it belongs to is a visible jump. It is also rare and bounded by the
    fan-out, which is what makes it affordable.
    """

    #: Ticks between two readings of the per-child numbers. The tick itself is
    #: the spinner's 12.5 fps — one cadence for "this is moving" across the
    #: app — but tokens and dollars are not animation: they move when the
    #: child reports a turn, which the dirty flag already catches. This is the
    #: belt for the slow ways they change (a model's context window arriving
    #: from the catalogue after the row was first drawn), and one second is
    #: the resolution those have. 13 ticks is 1.04 s. Measured: re-reading
    #: the numbers for six children costs 61 µs, so at every tick it would be
    #: 0.76 ms/s and here it is 0.06 ms/s — small either way, which is the
    #: honest reason this is a ceiling and not an optimisation.
    STATS_EVERY_TICKS = 13

    def __init__(self, on_open: Callable[[str], None]) -> None:
        super().__init__(id="subagent-panel", classes="band-slot")
        self._on_open = on_open
        self._header = Static(id="subagent-header", classes="band-body")
        # `.band-body`, the same class the header carries and the todo panel's
        # single body carries: the panel's ROWS are the panel, so they take the
        # dock's fill and the dock's one-cell inset rather than sitting on bare
        # ground one cell to the left of every other glyph in the column. Without
        # it this panel read as a filled caption over a floating list — and only
        # while a `/btw` card happened to be open, since `Screen.aside #band`
        # filled the band underneath them and hid it (design round 12, D1/D5).
        self._list = Vertical(id="subagent-rows", classes="band-body")
        self._rows: dict[str, SubagentRow] = {}
        #: Last ledger read, keyed by job id. Refresh repopulates it; the
        #: spinner tick repaints from it between refreshes rather than
        #: re-querying the manager eight times a second.
        self._jobs_by_id: dict[str, Any] = {}
        self._spinner_index = 0
        self._spinner_timer = None
        #: Whether the header has been painted once. It is a static label —
        #: the running-count that would once have changed it moved to the
        #: status band — so a second paint can only ever redraw the same row.
        self._header_shown: bool = False
        #: The ledger moved since the last paint. Set by `sync`, cleared by
        #: the tick that acts on it — the coalescing buffer, one bit wide,
        #: because "something changed" is all a full repaint needs to know.
        self._dirty = False
        self._tick_count = 0
        #: Job ids whose numbers are being read off-thread right now, so a
        #: slow provider cannot collect a queue of identical reads behind it.
        self._stats_pending: set[str] = set()
        #: Per-child numbers, cached between readings (see STATS_EVERY_TICKS).
        self._stats: dict[str, JobStats] = {}
        #: The PARENT's model label, which is what a child that recorded none
        #: of its own is running on. Read from the session at sync time rather
        #: than from `self.app` at paint time: the panel is handed the session
        #: already and a widget reaching into the app for a fact it was given
        #: is how the two get to disagree.
        self._model_label = ""
        #: The reduction rung the LIST is currently laid out at, settled by
        #: the last full paint and reused by the spinner's cheap path. One
        #: pair for every row, which is the point (:func:`panel_layout`).
        self._rung = 0
        self._column = 0
        self._clock = 0
        self.display = False

    def compose(self):  # type: ignore[override]
        yield self._header
        yield self._list

    def predicted_rows(self) -> int:
        """Content rows this panel will paint, for a caller that cannot measure.

        The dock's inset check runs at the moment a panel appears — a
        ``SubagentStarted`` event — when the slot has not been arranged yet and
        measures zero (see ``app.slot_rows``). This panel's height is simply
        its header plus one row per job, both of which are known as soon as
        ``sync`` has run, so answering here is what lets the dock paint its
        settled frame first instead of jumping a row at the next 1 Hz poll.

        Never raises and never returns less than one: a displayed panel is at
        least a row, and under-counting hands the transcript a row the dock is
        about to take.
        """
        return max(1, len(self._rows) + _HEADER_ROWS)

    def on_unmount(self) -> None:
        self._stop_spinner()

    # -- sync -------------------------------------------------------------
    def sync(self, session: Any) -> None:
        """Re-read ``session.jobs`` and schedule a repaint.

        Called on every Subagent* event (immediate) and on the 1 Hz poll (the
        belt to the events' suspenders — elapsed time moves with no event at
        all). Never raises: this is a status surface.

        The rows are NOT painted here. This marks the panel dirty and lets
        :meth:`_tick` do it, so a burst of child events costs one repaint
        instead of one per event per row; a row appearing or leaving is the
        exception and paints at once (see the class docstring).
        """
        try:
            manager = getattr(session, "jobs", None)
            jobs = manager.list() if manager is not None else []
        except Exception:
            jobs = []
        self._model_label = str(getattr(session, "model_label", "") or "")
        task_jobs = [job for job in jobs if getattr(job, "type", "") == "task"]
        if not task_jobs:
            self._jobs_by_id = {}
            self._stats = {}
            self._sync_rows([])
            self.display = False
            self._dirty = False
            self._stop_spinner()
            return
        self.display = True
        self._jobs_by_id = {str(getattr(job, "id", "") or ""): job for job in task_jobs}
        changed = self._sync_rows(task_jobs)
        if changed:
            self._paint_all()
        # Dirty AFTER the arrival paint, not before it. That paint happens the
        # instant a row is mounted, which is before the layout has measured
        # anything, so it necessarily lays out against the guessed width — it
        # exists to stop the dock's height arriving 80 ms late, not to be
        # right. The tick that follows re-measures and is what settles the
        # rung; without this line a panel whose first sync created its rows
        # kept the guess until something else moved.
        self._dirty = True
        self._start_spinner()

    def stats_for(self, job_id: str) -> JobStats:
        """This child's last completed reading, for a surface outside the panel.

        The status band shows the same child's numbers while its page is open,
        and it must not derive them for itself: :func:`job_stats` prices the
        child, and pricing resolves the model, which for one the shipped
        registry does not describe is a 10 s provider listing plus a 3 s
        aggregator catalogue. The panel already takes that reading off-thread
        on a bounded cadence, so the band reads the ANSWER rather than
        repeating the question. Empty until the first reading lands, which is
        the same "no number yet" the rows show.
        """
        return self._stats.get(job_id) or JobStats(model_label=self._model_label)

    def mark_current(self, job_id: str | None) -> None:
        """Tint the row the full-page view is showing (``None`` = no page).

        Kept on the panel rather than pushed from the app per row, because
        only the panel knows which rows exist after a refresh has added and
        dropped some.

        Marks dirty; does NOT paint. It used to call ``_paint_all`` whenever
        a page was open, and the app calls this from ``_refresh_subagent_view``
        — which every relayed child event reaches — so the whole list was
        repainted and every child's numbers re-resolved per event, on the
        thread that reads the keyboard. That is exactly the shape the tick
        exists to remove, reinstated by the one code path that only runs while
        a reader is watching a child work. Measured: 12 relayed events over
        three children produced 8 immediate row repaints with a page open and
        0 with it closed.

        Dirty UNCONDITIONALLY, including for ``None``. Clearing the mark
        changes what the row says — a settled row gets its outcome back — and
        the old ``job_id is not None`` guard left that row blank until the
        next 1 Hz sync, i.e. for about a second after Esc.
        """
        for row_id, row in self._rows.items():
            row.set_current(row_id == job_id)
        self._dirty = True
        if self._rows:
            self._start_spinner()

    def _sync_rows(self, jobs: list[Any]) -> bool:
        """Bring the row set into agreement with the ledger (add/drop/order).

        Returns whether the SET moved, which is what buys a row appearing an
        immediate paint instead of a tick's wait.
        """
        changed = False
        seen: set[str] = set()
        order: list[str] = []
        for job in jobs:
            job_id = str(getattr(job, "id", "") or "")
            if not job_id:
                continue
            seen.add(job_id)
            order.append(job_id)
            if job_id not in self._rows:
                changed = True
                self._rows[job_id] = SubagentRow(job_id, self._on_open)
                self._list.mount(self._rows[job_id])
        for job_id in list(self._rows):
            if job_id not in seen:
                changed = True
                self._rows.pop(job_id).remove()
                self._stats.pop(job_id, None)
        # The manager hands jobs back in start order; keep the DOM in the
        # same order so a stable ledger paints a stable list.
        #
        # Only rows the list ALREADY owns: `mount` and `remove` are deferred in
        # Textual, so between two syncs closer together than the DOM's apply
        # tick, `self._rows` can name a widget that is not a child yet - and
        # `move_child` raises `WidgetError` on it rather than ignoring it. That
        # surfaced as a hard crash in the band refresh, roughly one full-suite
        # run in ten and never reproducibly in isolation. A row that misses
        # this pass is ordered by the next one, which is already how a row that
        # arrives between syncs gets placed.
        mounted = set(self._list.children)
        children = [self._rows[job_id] for job_id in order if self._rows.get(job_id) in mounted]
        if list(self._list.children) != children:
            for index, row in enumerate(children):
                self._list.move_child(row, before=index)
        return changed

    def _paint_all(self, *, reread_stats: bool = True) -> None:
        """Measure the whole list, then repaint every row at one rung.

        Self-sufficient by design: ``mark_current`` and the tick both land
        here, so the numbers and the rung are worked out HERE rather than
        threaded in by each caller — a caller that forgot would paint a row
        with somebody else's cost, or at a width nobody else agreed to.
        """
        glyph = SPINNER_FRAMES[self._spinner_index]
        width = self._row_width()
        measured: list[tuple[str, SubagentRow, RowFacts, JobStats]] = []
        for job_id, row in self._rows.items():
            job = self._jobs_by_id.get(job_id)
            if job is None:
                continue
            facts = row_facts(job, fallback_id=job_id, current=row.current)
            measured.append((job_id, row, facts, self._stats_for(job_id, job, reread_stats)))
        self._rung, self._column, self._clock = panel_layout(
            [(facts, stats) for _, _, facts, stats in measured], width
        )
        for _job_id, row, facts, stats in measured:
            row.paint(
                facts,
                stats=stats,
                spinner_glyph=glyph,
                width=width,
                rung=self._rung,
                column=self._column,
                clock=self._clock,
            )
        self._paint_header()
        self._dirty = False

    def _row_width(self) -> int:
        """Cells a row actually has, which is not simply this widget's width.

        ``#band`` is content-sized on purpose (``width: auto``, so it does not
        defeat the input shell's centering) and ``.band-slot`` is ``1fr``
        against it, so a long row makes the panel WIDER than the dock it sits
        in and the surplus is clipped by the screen. Measured at a 58-cell
        dock: 58 for a short row and 65 for ``MR review agent round 2``. Fed
        that 65, the ladder keeps a rung the terminal cannot show and the row
        loses the end of its sentence to a hard cut — the ladder defeated by
        the very row it exists for. So the screen is the ceiling: it is exact
        when the panel has not over-grown, and it is the truth when it has.

        Measured on ``self._list``, not on the panel: the rows live inside the
        ``.band-body`` container, whose ``padding: 0 1`` is what puts their
        bullets on the dock's rail with the header and the composer's chevron.
        Textual is border-box, so that padding comes out of the row's cells and
        the panel's own width is two too many — the ladder would keep a rung
        that does not fit and hand the overflow to a hard ellipsis, which is the
        exact failure the ceiling below exists to prevent.

        Before the first arrange nothing is measured and the ladder needs a
        number regardless. That fallback is deliberately WIDE — an
        over-generous guess degrades to Rich's own ellipsis at the row's real
        edge, while an under-generous one sheds a segment the terminal had
        room for and flickers it back one tick later.
        """
        try:
            ceiling = self.screen.size.width or 0
        except Exception:
            ceiling = 0  # not on a screen yet; the arrange that follows fixes it
        if ceiling:
            # The screen ceiling has to be charged the SAME padding the rows
            # already pay. ``.band-body`` is ``padding: 0 1``, and Textual is
            # border-box, so a row inside it never has more than
            # ``screen - 2`` cells even though the screen is ``screen`` wide.
            # Comparing an unpadded ceiling against a padded ``own`` only
            # matters once the panel over-grows its dock (``#band`` is
            # ``width: auto``, so a long row widens it past the screen and the
            # surplus is clipped) — and there the ladder was handed two cells
            # that do not exist, kept a rung the terminal could not show, and
            # the row lost its tail to a hard cut with no ellipsis: precisely
            # the failure the ceiling exists to prevent, two cells short of
            # preventing it. Read off the container rather than hardcoding 2,
            # so a stylesheet change to the rail cannot silently desync this.
            padding = self._list.styles.padding
            ceiling = max(0, ceiling - padding.left - padding.right)
        own = self._list.size.width or self.size.width or 0
        if own and ceiling:
            return min(own, ceiling)
        return own or ceiling or _DEFAULT_ROW_WIDTH

    def _stats_for(self, job_id: str, job: Any, reread: bool) -> JobStats:
        """This child's numbers, from the cache — never computed on this thread.

        A miss returns an EMPTY reading and schedules the real one off-thread,
        so the row paints without its numbers for a frame or two rather than
        the app pausing to earn them. That is not caution about arithmetic:
        :func:`costs.job_cost` resolves the child's model to price it, and for
        a model the shipped registry does not describe that resolution is a
        10 s provider listing followed by a 3 s aggregator catalogue. A child
        on a ``model_spec`` override is the case these stats exist to surface,
        so the slow path is the interesting path, and a repaint may do no I/O
        at any budget. ``job_stats`` itself no longer resolves anything for
        the window — the engine records that on the job at launch — and this
        is the same rule applied to the leg that remained.

        The cache is pruned in :meth:`_sync_rows` alongside the rows: retention
        sweeps children out one at a time, so a session with a large fan-out
        otherwise carried a ``JobStats`` for every child it had ever run.
        """
        cached = self._stats.get(job_id)
        if cached is not None and not reread:
            return cached
        self._read_stats(job_id, job)
        return cached if cached is not None else JobStats(model_label=self._model_label)

    def _read_stats(self, job_id: str, job: Any) -> None:
        """Compute one child's numbers off the UI thread, then mark dirty.

        Coalesced by ``_stats_pending``: the tick asks once a second and the
        answer takes as long as it takes, so without this a slow provider
        would have a queue of identical reads behind it.
        """
        if job_id in self._stats_pending:
            return
        self._stats_pending.add(job_id)
        label = self._model_label

        def read() -> None:
            stats = job_stats(job, default_model_label=label)
            self.app.call_from_thread(self._stats_read, job_id, stats)

        try:
            self.run_worker(
                read, thread=True, group=f"subagent-stats-{job_id}", exit_on_error=False
            )
        except Exception:
            # No app to run a worker on (an unmounted panel in a unit test).
            # Read inline: there is no UI thread to protect.
            self._stats_pending.discard(job_id)
            self._stats[job_id] = job_stats(job, default_model_label=label)

    def _stats_read(self, job_id: str, stats: JobStats) -> None:
        """Land an off-thread reading and let the next tick paint it."""
        self._stats_pending.discard(job_id)
        if self._stats.get(job_id) != stats:
            self._stats[job_id] = stats
            self._dirty = True
            self._start_spinner()

    def _paint_header(self) -> None:
        """Paint the panel's identifying label.

        Deliberately a bare ``Subagents`` word and NOT a ``running/total``
        counter (D-04): the running-count is already on the status band, where
        the same jobs are tallied, and a second copy would add a redundant
        counter vocabulary to a symbol-driven screen. The label exists to
        distinguish the two independent panels that share the band — the user
        needs to know which list they are reading.
        """
        if self._header_shown:
            return
        self._header_shown = True
        muted = Style(color=theme_mod.semantic_color("muted"))
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Subagents", style=muted)
        self._header.update(header)

    def on_resize(self) -> None:
        """A narrower panel is a different row layout, for every row at once.

        Marked dirty rather than painted: a window drag emits a resize per
        column, and the tick is what turns that stream into one repaint.
        """
        self._dirty = True
        if self._rows:
            self._start_spinner()

    # -- tick ----------------------------------------------------------------
    def _start_spinner(self) -> None:
        if self._spinner_timer is None and self.is_mounted:
            self._spinner_timer = self.set_interval(SPINNER_INTERVAL_S, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _tick(self) -> None:
        """The panel's ONE repaint point: spinner, coalesced syncs, numbers.

        Three cadences ride one timer. The glyph advances every tick (12.5 fps
        is the app's single notion of "this is moving"); a dirty panel is
        repainted whole on the next tick, which is the coalescing; the numbers
        are re-read every :data:`STATS_EVERY_TICKS`.

        The timer stops itself once there is nothing left to animate and
        nothing pending, so a dock full of settled children costs nothing.
        """
        self._tick_count += 1
        self._spinner_index = (self._spinner_index + 1) % len(SPINNER_FRAMES)
        due = self._tick_count % self.STATS_EVERY_TICKS == 0
        if self._dirty or due:
            self._paint_all(reread_stats=self._dirty or due)
        elif any(row.running for row in self._rows.values()):
            # The cheap path, and the common one: only the glyph moved, so
            # only the rows carrying a glyph are touched, each reusing the
            # numbers already read for it and the rung the last full paint
            # settled on. Re-measuring the list to advance a spinner would be
            # the whole panel's work for one cell of animation.
            glyph = SPINNER_FRAMES[self._spinner_index]
            width = self._row_width()
            for job_id, row in self._rows.items():
                job = self._jobs_by_id.get(job_id)
                if job is not None and row.running:
                    row.paint(
                        row_facts(job, fallback_id=job_id, current=row.current),
                        stats=self._stats_for(job_id, job, False),
                        spinner_glyph=glyph,
                        width=width,
                        rung=self._rung,
                        column=self._column,
                        clock=self._clock,
                    )
        # Sampled AFTER the paint, from the rows this tick has just produced.
        # Read beforehand it said False for a queued child that started in
        # this very tick, and the timer stopped under a row that had just come
        # alive — frozen spinner and frozen clock until the next 1 Hz sync.
        running = any(row.running for row in self._rows.values())
        if not running and not self._dirty:
            self._stop_spinner()
