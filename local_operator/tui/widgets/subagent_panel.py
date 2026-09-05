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
spent at exactly five sites (see the tcss preamble) and a sixth spinner is
not one of them. Settled rows do NOT follow the tool ledger's "✓ success"
ink law — that overturn (see `tool_card.py`) was deliberately bounded to the
TOOL LEDGER; `status_glyph` keeps this panel's ✓ dim on completion, the same
as ✗ and every other settled state, because a subagent's own transcript
already carries the colour and this row is a job-manager summary, not the
ledger.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, NamedTuple

from rich.cells import cell_len
from rich.style import Style
from rich.text import Text
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widgets import Static

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.animation import BLURRED_SPINNER_INTERVAL_S, animation_focused
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

#: Default dock footprint, including the caption and pinned disclosure row.
#: Six jobs plus those two chrome rows keep the common panel near the todo
#: panel's eight-row preview while preserving the bottom of the start-ordered
#: ledger, where active and newly-launched children arrive.
MAX_SUBAGENT_ROWS = 8
_PREVIEW_JOB_ROWS = MAX_SUBAGENT_ROWS - _HEADER_ROWS - 1

#: Screen rows the column spends around this panel, subtracted from the screen
#: height to get the rows the collapsed preview may paint. The mirror of
#: ``todo_panel._DOCK_ROWS`` and of ``app._SUBAGENT_DOCK_ROWS``, which gates the
#: band inset against the same arithmetic: five for ``#input-shell``, two for
#: the transcript's padding, one for this panel's caption, one for its slot
#: rhythm row, and one row of conversation left over.
#:
#: ``_PREVIEW_JOB_ROWS`` used to be ABSOLUTE, which was invisible while a
#: resumed session opened with an EMPTY dock: the six-row preview only ever
#: appeared once children were running, by which time the user had chosen to
#: start them. Restoring the roster on the first frame (this PR) makes a cold
#: resume paint the full preview immediately, and on a short terminal the dock
#: then took the whole column — measured on the reference session at 100x24,
#: the operator's own height: ZERO conversation rows, with ``ctrl+g`` unable to
#: recover them because the COLLAPSED dock alone was already eight rows
#: (UX round 2, U7).
#:
#: Counts CHROME only, so the conversation floor below can be reasoned about
#: separately: five for ``#input-shell``, two for the transcript's padding, and
#: three for the panel itself (caption, pinned affordance, slot rhythm row).
#: ``app._SUBAGENT_DOCK_ROWS`` is the same sum with one conversation row folded
#: in, which is why that gate reads 10 against this 10 plus a floor.
_COLLAPSED_DOCK_ROWS = 10

#: Conversation rows the collapsed dock may never take. The dock is chrome
#: around a transcript, so the transcript keeps enough rows to read a reply in
#: place. Applies to the COLLAPSED state only: expanding is an explicit
#: ``ctrl+g`` request to see the roster, and that state has always been allowed
#: to borrow from the transcript (which is ``1fr`` and scrolls, so it can yield
#: rows — the composer cannot).
#:
#: Six, swept rather than guessed. Measured on the reference session (19
#: children, 4 todo phases, 1 wake) across screen heights 22-50 at floors 4/5/6,
#: reading the SETTLED transcript height at two pump depths so a value that
#: still moved would show as a reflow rather than average away:
#:
#:   height   22  24  26  28  30  32  34  36  40  50
#:   floor 4   3   3   3   3   4   4   6   8  12  22
#:   floor 5   3   4   4   4   4   4   6   8  12  22
#:   floor 6   3   4   4   5   5   5   6   8  12  22
#:
#: Six dominates: it is never worse, and it is the only value that lifts the
#: 28-32 band — where the collapsed dock alone used to exceed the screen, so
#: ``ctrl+g`` could not recover the conversation. Above 34 rows every floor
#: converges because the panels reach their content ceilings and stop asking.
#: Below 24 the band's irreducible chrome binds instead (see the module note on
#: ``_MIN_PREVIEW_JOB_ROWS``), so a larger floor buys nothing there.
_TRANSCRIPT_FLOOR_ROWS = 6

#: Floor for the collapsed preview, in JOB rows. Zero is deliberate and is the
#: COMPACT state: on a terminal too short to afford both the roster and a
#: readable conversation, the panel drops to its caption plus the affordance
#: and lets the header carry the count ("Subagents · 19"), so the session still
#: says what it has and ``ctrl+g`` still reaches it.
#:
#: A one-row floor was the obvious choice and is worse. Three panels share this
#: column; each holding one row back costs three rows plus three lots of chrome,
#: which at the operator's 100x24 left the conversation at two rows — the dock
#: still owning the screen, only less honestly, because a single arbitrary child
#: beside a silent "+18" says less than the total does.
_MIN_PREVIEW_JOB_ROWS = 0

#: Rows outside an expanded roster that may never be taken from the terminal:
#: transcript breathing room plus the composer/status shell and slot rhythm.
#: Expanded means every child is reachable, not that the dock may push the
#: composer off-screen; overflow therefore scrolls inside the roster.
_EXPANDED_DOCK_ROWS = 14


class Density(str, Enum):
    """How much of the dock the panel takes; ``ctrl+g`` cycles it (#525).

    A SECOND axis beside ``_expanded``, not a replacement for it. ``full`` is
    everything the panel did before: the budgeted preview, and inside it the
    expanded roster the overflow disclosure opens. ``summary`` is one row of
    counts in the caption; ``hidden`` paints nothing. Kept orthogonal so the
    overflow contract (``+N earlier``, navigation, ``collapse_for_child_view``)
    is untouched — the density only decides whether that machinery is shown.

    Before this the key was an OVERFLOW disclosure and nothing else: with six
    or fewer children — the common fan-out — ``toggle_expanded`` returned at
    its budget gate and the press did nothing, so the panel was a fixed claim
    on 2-8 rows above the composer for the rest of the session (#525). The
    only zero-row shape was the height-forced compact fallback, which the user
    could not ask for.

    Values are the ``display.dock`` setting's vocabulary: ``compact`` already
    names the height-forced state in this module and ``off`` would read as
    "the feature is disabled", so the setting, the enum, the ``/help`` row and
    the tests all say ``full | summary | hidden``.
    """

    FULL = "full"
    SUMMARY = "summary"
    HIDDEN = "hidden"


#: The density a session starts at when ``display.dock`` is unset. Read via
#: ``tui/settings.settings_get`` — whose defaults derive from the ``settings_io``
#: registry, so the registry row is the guarded source and this constant is
#: the panel's own fallback for a reader that cannot reach it.
DEFAULT_DOCK_DENSITY = Density.FULL

#: Order the summary row sheds its segments at a narrow width, first to go
#: first. Whole segments, never a truncation into the hotkey: the ``ctrl+g``
#: token is the one thing on the row that must always survive, because it is
#: the ONLY cue the panel can be brought back (``todo_panel._footer`` rule).
#: Counts of settled-quietly states go before the running count, and the
#: failed count goes last because it is the reason the row re-emerged from
#: ``hidden`` at all. The ``Subagents`` word goes before the numbers: on a
#: 50-column terminal the reader can tell a row of ✗/⣾ counts from the todo
#: list without the label, and cannot tell ``1 failed`` from nothing.
_SUMMARY_SHED_ORDER = ("cancelled", "interrupted", "queued", "done", "label", "running", "failed")

#: Eviction rank for the collapsed preview, lowest kept first. Running and
#: queued children are the ones the user is waiting on; a failure is the one
#: outcome that needs acting on; an interrupted child may be resumable; a
#: completed or cancelled one has said everything a row can say. Ties break
#: by start order, newest kept, which is the rule the slice had before.
_EVICTION_RANK: dict[str, int] = {
    "running": 0,
    "failed": 1,
    "interrupted": 2,
}

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

#: The widest a ROLE segment may be before it is truncated with the usual `…`.
#:
#: Role names are operator-authored (`agent op=create`) and have no length cap,
#: and the `agent`/`team` tooling actively encourages descriptive specialist
#: names — `user-dashboard-agent` is 20 cells, and nothing stops a 60-cell one.
#: Left unbounded the segment is a per-LIST cost paid by every row: the longest
#: role in the roster decides whether ANYBODY gets one, so a single verbose
#: specialist silently strips the role from its short-named peers and the
#: feature reads as "the role column is gone again" (design review round 1, D2).
#: Fourteen holds every packaged role whole (`ux-reviewer` is the longest at 11)
#: and `architect`, `designer`, `reviewer` and `coder` with room to spare, while
#: keeping the shared column below in the same order of magnitude as the label's
#: own floor. A truncated role still identifies the child: it is a disambiguator
#: beside the label, not the label itself.
ROLE_CEILING = 14

#: Maximum cells the row gives the two live numeric readings. Both are
#: explicitly truncated to these budgets before they paint, and the role-rung
#: gate reserves the SAME maxima whether a reading currently exists or not.
#: That coupling is load-bearing: usage/context/cost arrive live and their raw
#: formatters interpolate uncapped magnitudes, so measuring the current strings
#: made the roster-wide role column appear/disappear at a fixed width after the
#: first usage report or as spend grew (agent review round 3, R5).
#:
#: Context's normal full form is ``24.1%/200k`` (11 cells); fourteen preserves
#: even ``999.9%/1000M`` whole and visibly ellipsises pathological magnitudes.
#: Cost's ordinary largest precision is ``$999.99`` (7); ten preserves nine
#: digits of useful magnitude before the ellipsis. The exact values matter less
#: than sharing them between paint and acceptance: no live string may consume
#: cells the width-only gate did not reserve.
CONTEXT_CEILING = 14
COST_CEILING = 10

#: :func:`format_duration` is bounded at six cells by construction (``100d+``
#: is the terminal spelling), independently tested in ``test_status_line``.
#: Name the contract here because the role gate reserves this maximum rather
#: than the current elapsed spelling, so a clock transition cannot toggle the
#: whole column (R5).
CLOCK_CEILING = 6

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
        if start <= 0:
            return 0.0
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
    # Cold presentation-only rows have identity/outcome but no trustworthy
    # clock. Epoch zero is not a launch time, and "0s" would invent a duration.
    if not getattr(job, "start_time", None):
        return ""
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
    cost_partial: bool = False
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
    partial = False
    try:
        model_label = str(getattr(job, "model_label", "") or default_model_label or "")
        usage = getattr(job, "usage", None)
        billed = usage is not None
        if billed:
            tokens = int(
                getattr(usage, "context_tokens", None) or getattr(usage, "input_tokens", 0)
            )
        window = int(getattr(job, "context_window", 0) or 0)
        knowledge = getattr(job, "direct_cost_knowledge", None)
        if knowledge is not None:
            # The runtime resolved the serving model; this background viewer
            # must not redo discovery with its unrelated cache/credentials.
            cost = getattr(job, "direct_cost", None)
            partial = knowledge in {"partial", "floor"}
        else:
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
        cost_partial=partial,
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
    if not isinstance(details, Mapping):
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
        # runs, `running N tools` for a batch, `responding` once prose is
        # actually streaming, `thinking` for a model call in flight with
        # nothing streamed yet (see `harness.intent`).
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
#:    one glance and must not rank the same field differently. It is also the
#:    only rung whose acceptance is CONDITIONAL: :func:`_row_rung` rejects it
#:    whenever keeping the shared column would drive ANY row's activity below
#:    :data:`ACTIVITY_FLOOR`, including a roleless row that reserves blank
#:    cells for alignment. The threshold is a WIDTH CAPACITY whose other inputs
#:    are explicit maxima, not current activity/elapsed/context/cost strings,
#:    so no live update can make the whole column flicker in and out (design
#:    review round 3 D8, agent review round 3 R5).
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
class _Rung(NamedTuple):
    """One monotone reduction step, named so its policy is not positional.

    This table is read in the painter, both measurement passes and the shared
    role-column gate. Positional ``[2]`` meant ``keep_role`` only to a reader
    who counted four unlike tuple fields correctly every time; a named row
    makes adding or reordering a field a type-checked change instead of a
    silent policy swap (agent review round 2, N1).
    """

    context_spelling: str
    keep_cost: bool
    keep_role: bool
    label_budget: int | None


_RUNGS: tuple[_Rung, ...] = (
    _Rung("full", True, True, None),
    _Rung("full", True, False, None),
    _Rung("full", False, False, None),
    _Rung("nodec", False, False, None),
    _Rung("short", False, False, None),
    _Rung("none", False, False, None),
    _Rung("none", False, False, LABEL_FLOOR),
)


def _row_rung(
    facts: RowFacts,
    stats: JobStats,
    width: int,
    column: int,
    clock: int = 0,
    role_column: int = 0,
) -> int:
    """The LEAST-REDUCED rung this row still reads correctly at.

    Walks :data:`_RUNGS` widest-first and takes the first that fits, so a
    smaller return is a richer row and :func:`panel_layout`'s ``max`` picks
    the reduction every row can live with. Well-founded because both the
    numbers and the label budget are non-increasing in the index, so
    acceptance is monotone and the first hit is the answer.

    The ROLE rung is additionally rejected when keeping the shared role COLUMN
    would leave ANY row fewer than :data:`ACTIVITY_FLOOR` cells for activity
    (D3/R4/D8/R5). The column is roster-wide, so this capacity check is
    symmetric: a roleless row still reserves the blank column for D1 alignment.

    The veto consumes only WIDTH and bounded inputs. It does not measure the
    current activity (D8), elapsed spelling, or live context/cost strings (R5):
    all three change without a resize, and raw numeric formatters can grow with
    magnitude. Instead it reserves ``CLOCK_CEILING`` and the maximum cells of
    every numeric segment this rung can paint. `_lay_out` truncates those
    segments to the same ceilings, so acceptance and rendering cannot diverge.
    The result may conservatively shed the role earlier than today's short
    values require, but it cannot flicker as stats arrive or counters grow.
    """
    for index in range(len(_RUNGS)):
        label, role, context, cost, activity = _lay_out(
            facts, stats, width, index, column, clock, role_column
        )
        head = (
            _CHROME_CELLS
            + max(cell_len(label), column)
            + max(cell_len(facts.elapsed), clock)
            + _glyph_cells(facts)
            + _role_cells(index, role, role_column)
        )
        numbers = sum(len(STATS_SEAM) + cell_len(part) for part in (context, cost) if part)
        if role_column and _RUNGS[index].keep_role:
            # Reserve MAXIMUM rendered widths, not today's strings. Elapsed,
            # context and cost all update live; using ``head``/``numbers`` here
            # let the first usage report or a magnitude transition toggle the
            # whole column without a resize (R5). The painter applies the same
            # ceilings in `_lay_out`, making this conservative but truthful.
            fixed_head = (
                _CHROME_CELLS
                + max(cell_len(label), column)
                + CLOCK_CEILING
                + _glyph_cells(facts)
                + _role_cells(index, role, role_column)
            )
            fixed_numbers = len(STATS_SEAM) + CONTEXT_CEILING
            if _RUNGS[index].keep_cost:
                fixed_numbers += len(STATS_SEAM) + COST_CEILING
            activity_room = width - fixed_head - fixed_numbers - ACTIVITY_GAP
            if activity_room < ACTIVITY_FLOOR:
                continue
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


def _role_width(rung: int, role: str, role_column: int) -> int:
    """Cells the role FIELD occupies on one row, seam EXCLUDED.

    One definition shared by the two measurement sites and the painter, because
    a budget computed one way and spent another is exactly how a row ends up
    one cell over its width. The width is the shared ``role_column`` rather
    than this row's own role, so a roleless row pays the same as its peers and
    the seam run after it stays put (D1).

    Zero at any rung that has already SHED the role: a rung with no role at all
    must charge nothing, or the blank column would be paid for at every width
    below the role's own rung and the ladder would hand the activity fewer
    cells than it had before this field existed.
    """
    if not _RUNGS[min(rung, len(_RUNGS) - 1)].keep_role:
        return 0
    return max(role_column, cell_len(role))


def _role_cells(rung: int, role: str, role_column: int) -> int:
    """:func:`_role_width` plus the seam that precedes it, or zero for neither."""
    width = _role_width(rung, role, role_column)
    return len(STATS_SEAM) + width if width else 0


def _lay_out(
    facts: RowFacts,
    stats: JobStats,
    width: int,
    rung: int,
    column: int = 0,
    clock: int = 0,
    role_column: int = 0,
) -> tuple[str, str, str, str, str]:
    """``(label, role, context, cost, activity)`` exactly as ``rung`` paints them.

    ``column`` is the shared label width the row will be padded to; the
    activity is measured against it, not against this row's own label, or a
    short-labelled row would be handed slack the padding is about to spend.

    ``role_column`` is the same idea for the role: the width EVERY row's role
    field occupies, so what follows it starts in one vertical line instead of
    being shoved sideways by each row's own role length. Zero means "lay this
    row out alone", which is what a caller measuring a single row wants.
    """
    policy = _RUNGS[min(rung, len(_RUNGS) - 1)]
    spelling = policy.context_spelling
    keep_cost = policy.keep_cost
    keep_role = policy.keep_role
    budget = policy.label_budget
    # The last rungs carry no reading at all; every other spelling is one the
    # band names too (`status_line.CONTEXT_FORMS`), so the same child's
    # occupancy cannot be written two ways four rows apart.
    context = (
        ""
        if spelling == "none"
        else truncate_cells(
            context_spelling(stats.context_tokens, stats.context_window, form=spelling),
            CONTEXT_CEILING,
        )
    )
    # Bounded before it is measured, so one verbose specialist cannot make the
    # shared column below 34 cells wide and evict its peers' roles (D2).
    role = truncate_cells(facts.agent_role, ROLE_CEILING) if keep_role else ""
    if not keep_cost:
        cost = ""
    elif stats.cost is not None:
        cost = truncate_cells(
            format_cost(stats.cost) + ("+" if stats.cost_partial else ""), COST_CEILING
        )
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
        # The role is charged as a COLUMN, not as a number: every row pays the
        # shared width whether or not it has a role to put in it, which is what
        # keeps the context, the cost and the activity on one vertical line
        # (D1). A row that suppresses its role therefore pays for the blank and
        # stays flush with its peers, instead of pulling its whole tail left.
        + _role_cells(rung, role, role_column)
    )
    numbers = sum(len(STATS_SEAM) + cell_len(part) for part in (context, cost) if part)
    room = width - head - numbers - ACTIVITY_GAP
    activity = truncate_cells(facts.activity, room) if facts.activity and room > 0 else ""
    return label, role, context, cost, activity


def panel_layout(
    rows: Sequence[tuple[RowFacts, JobStats]], width: int
) -> tuple[int, int, int, int]:
    """``(rung, label column, clock, role column)`` — the panel's one layout.

    Decisions that cannot be made apart. The rung says which fields every
    row carries; the columns say where they sit, so that two children's spend
    can be compared by looking down instead of reading across. Padding costs
    the shorter rows cells they were spending on activity slack, which can
    push a row under its floor — so the rung is re-checked WITH the columns
    applied and reduced until they fit, rather than each being solved
    against a layout the other has not happened yet.

    The ROLE column is settled the same way and for the same reason as the
    label's: an inline role of a different length on every row shoved that
    row's context, cost and activity sideways by a different amount, turning a
    column of four ``$`` signs into four x positions (design review round 1,
    D1). Bounded by :data:`ROLE_CEILING` before it is measured, so the column
    cannot be widened without limit by one operator-authored specialist name.

    Terminates: the columns are non-increasing in the rung (the last rung caps
    the label at :data:`LABEL_FLOOR` and no rung past the first carries a role)
    and the final rung carries no numbers at all, so the loop is bounded by
    ``len(_RUNGS)``.
    """
    # The clock column is width-independent — a duration is as wide as it is —
    # so it is settled once, before the rung search that consumes it.
    clock = max((cell_len(facts.elapsed) for facts, _ in rows), default=0)
    rung = max(
        (_row_rung(facts, stats, width, 0, clock, _role_column(rows)) for facts, stats in rows),
        default=0,
    )
    while rung < len(_RUNGS) - 1:
        column = _label_column(rows, width, rung, clock)
        role_column = _role_column(rows) if _RUNGS[rung].keep_role else 0
        if all(
            _row_rung(facts, stats, width, column, clock, role_column) <= rung
            for facts, stats in rows
        ):
            return rung, column, clock, role_column
        rung += 1
    role_column = _role_column(rows) if _RUNGS[rung].keep_role else 0
    return rung, _label_column(rows, width, rung, clock), clock, role_column


def _role_column(rows: Sequence[tuple[RowFacts, JobStats]]) -> int:
    """Cells the role field occupies on EVERY row that shows one.

    Width-independent: the roles are already bounded by :data:`ROLE_CEILING`,
    so this is a property of the roster rather than of the terminal, and a
    roster with no roles at all yields zero and costs nothing.
    """
    return max(
        (cell_len(truncate_cells(facts.agent_role, ROLE_CEILING)) for facts, _ in rows),
        default=0,
    )


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
    role_column: int = 0,
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

    label, role, context, cost, activity = _lay_out(
        facts, stats, width, rung, column, clock, role_column
    )
    # The cells the role field occupies on THIS row, seam included. A row with
    # no role pays the same width as its peers so the segments after it stay in
    # column (D1), but pays it as pure whitespace: emitting the ` · ` seam with
    # nothing after it would paint a separator separating nothing, which reads
    # as a missing value rather than as a child with nothing to say.
    role_width = _role_width(rung, role, role_column)
    role_pad = " " * max(0, role_width - cell_len(role))
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
        if role:
            row.append(role, style=muted)
            row.append(role_pad, style=dim)
        elif role_width:
            row.append(" " * (role_width + len(STATS_SEAM)), style=dim)
        for index, value in enumerate(part for part in (context, cost) if part):
            if index or role:
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
    if role:
        row.append(STATS_SEAM, style=faint)
        row.append(role, style=muted)
        row.append(role_pad, style=dim)
    elif role_width:
        row.append(" " * (role_width + len(STATS_SEAM)), style=dim)
    for value in (context, cost):
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


class SummaryCounts(NamedTuple):
    """Per-state child counts the summary row paints.

    A tuple rather than a dict so it can be the paint-once guard's key: the
    summary caption repaints only when one of these moves or the spinner
    frame advances, the same equality gate every other coalesced paint on
    this panel sits behind.
    """

    running: int = 0
    queued: int = 0
    done: int = 0
    failed: int = 0
    cancelled: int = 0
    interrupted: int = 0

    @property
    def total(self) -> int:
        return sum(self)


def summary_counts(jobs: Sequence[Any]) -> SummaryCounts:
    """Bucket task jobs by the state the row vocabulary already names.

    Reads through :func:`row_facts` rather than ``job.status`` directly so a
    queued child counts as queued here exactly as its row would glyph it (a
    queued job's ``status`` is still ``running``; see :func:`status_glyph`).
    Restored rows arrive as ``interrupted``/``completed`` and bucket there.
    """
    counts = dict.fromkeys(SummaryCounts._fields, 0)
    for job in jobs:
        facts = row_facts(job, fallback_id="", current=False)
        if facts.queued:
            counts["queued"] += 1
        elif facts.status == "running":
            counts["running"] += 1
        elif facts.status == "failed":
            counts["failed"] += 1
        elif facts.status == "cancelled":
            counts["cancelled"] += 1
        elif facts.status == "interrupted":
            counts["interrupted"] += 1
        else:
            counts["done"] += 1
    return SummaryCounts(**counts)


def _summary_segments(
    counts: SummaryCounts, spinner_glyph: str, *, dropped: frozenset[str]
) -> list[tuple[str, str, str]]:
    """``(name, text, semantic ink)`` for every segment that survives ``dropped``.

    Zero-count segments are omitted outright (``format_agents`` says nothing
    at zero, and ``0 failed`` would spend the danger ink on good news). The
    spinner frame follows the running count so the count is what the eye
    reads first and the motion is what says it is still moving. When the
    running COUNT is shed but children are still running, the frame stays as
    a bare glyph: at the width that sheds it, the reader still needs to know
    the panel is not describing a finished session. A count that is shed
    while nonzero compresses to its glyph (``✗2``) for the same reason the
    label is shed before it: the number carries the meaning, the word only
    spells it.
    """
    segments: list[tuple[str, str, str]] = []
    if "label" not in dropped:
        segments.append(("label", "Subagents", "dim"))
    if counts.running:
        if "running" in dropped:
            segments.append(("running", spinner_glyph, "muted"))
        else:
            segments.append(("running", f"{counts.running} running {spinner_glyph}", "muted"))
    if counts.done and "done" not in dropped:
        segments.append(("done", f"{counts.done} done", "dim"))
    if counts.failed:
        if "failed" in dropped:
            segments.append(("failed", f"{GLYPH_FAILED}{counts.failed}", "danger"))
        else:
            segments.append(("failed", f"{counts.failed} failed", "danger"))
    if counts.queued and "queued" not in dropped:
        segments.append(("queued", f"{counts.queued} queued", "dim"))
    if counts.cancelled and "cancelled" not in dropped:
        segments.append(("cancelled", f"{counts.cancelled} cancelled", "dim"))
    if counts.interrupted and "interrupted" not in dropped:
        segments.append(("interrupted", f"{counts.interrupted} interrupted", "dim"))
    segments.append(("hotkey", "ctrl+g", "muted"))
    return segments


def compose_summary(counts: SummaryCounts, *, spinner_glyph: str, width: int) -> Text:
    """The one-row summary the panel paints in :attr:`Density.SUMMARY`.

    ``Subagents · 1 running ⣾ · 3 done · 1 failed · ctrl+g``

    Ink follows the row law: ``failed`` is the only coloured segment, matching
    the ``✗`` a full row would carry; everything else is ``dim`` chrome except
    the running count and the hotkey at ``muted``, because the hotkey is the
    only signal the panel can grow again and has to be the loudest token
    (``todo_panel`` D3/U3).

    Fit is decided against the WIDEST rendering the roster can produce (every
    count at its current digit width, spinner present) rather than the string
    being painted, so a segment does not flicker in and out as the spinner
    frame or a count's last digit changes (``todo_panel`` U1). Segments are
    shed whole in :data:`_SUMMARY_SHED_ORDER`; the row is never truncated into
    the hotkey, and if even ``✗N · ctrl+g`` cannot fit the hotkey alone is
    painted and clipped by the renderer, which is the one case with nothing
    left to shed.
    """
    inks = {
        name: Style(color=theme_mod.semantic_color(name)) for name in ("dim", "muted", "danger")
    }
    dropped: set[str] = set()
    # The widest frame is the reference so the choice is stable across frames.
    widest = max(SPINNER_FRAMES, key=cell_len)
    for _ in range(len(_SUMMARY_SHED_ORDER) + 1):
        segments = _summary_segments(counts, widest, dropped=frozenset(dropped))
        needed = sum(cell_len(text) for _name, text, _ink in segments)
        needed += cell_len(STATS_SEAM) * (len(segments) - 1)
        if needed <= width:
            break
        remaining = [name for name in _SUMMARY_SHED_ORDER if name not in dropped]
        if not remaining:
            break
        dropped.add(remaining[0])
    segments = _summary_segments(counts, spinner_glyph, dropped=frozenset(dropped))
    row = Text(no_wrap=True, overflow="ellipsis")
    for index, (_name, text, ink) in enumerate(segments):
        if index:
            row.append(STATS_SEAM, style=inks["dim"])
        row.append(text, style=inks[ink])
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
        Binding("up", "move_roster(-1)", "Previous subagent", show=False),
        Binding("down", "move_roster(1)", "Next subagent", show=False),
        # Priority keeps the app-level double-Esc stop ladder from taking the
        # first press while roster navigation owns focus.
        Binding("escape", "exit_roster", "Return to composer", show=False, priority=True),
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
        role_column: int = 0,
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
            # In the fingerprint because the role column is a property of the
            # ROSTER: a child arriving or leaving can change it without
            # changing anything else about this row, and a row that skipped
            # that repaint would keep a stale pad and sit out of column.
            role_column,
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
                role_column=role_column,
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

    def action_move_roster(self, direction: int) -> None:
        panel = self.parent
        if isinstance(panel, VerticalScroll) and isinstance(panel.parent, SubagentPanel):
            panel.parent.move_focus(self._job_id, direction)

    def action_exit_roster(self) -> None:
        panel = self.parent
        if isinstance(panel, VerticalScroll) and isinstance(panel.parent, SubagentPanel):
            panel.parent.exit_navigation()

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        self._on_open(self._job_id)


class SubagentAffordance(Static):
    """Pinned collapse/expand control, separate so only the control clicks."""

    def __init__(self) -> None:
        super().__init__(classes="band-body", id="subagent-affordance")

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        # Stop before toggling so one click cannot also reach and scroll the
        # transcript behind the dock, matching the todo disclosure contract.
        event.stop()
        panel = self.parent
        if isinstance(panel, SubagentPanel):
            panel.request_toggle()


class SubagentHeader(Static):
    """The caption, clickable in every density (#525 design §3).

    The affordance row only exists with overflow, so before this a pointer
    user had no target at all in the common ≤ 6-child case — the same gap
    the key had. The caption is the one widget painted in every non-hidden
    density (it IS the summary row), so it is the one that can carry the
    click. Same contract as the affordance: stop the event so the transcript
    behind the dock does not also scroll, and cycle through the pointer path
    so the composer keeps focus.
    """

    def __init__(self) -> None:
        super().__init__(id="subagent-header", classes="band-body")

    def on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        event.stop()
        panel = self.parent
        if isinstance(panel, SubagentPanel):
            panel.request_toggle()


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
        self._header = SubagentHeader()
        # `.band-body`, the same class the header carries and the todo panel's
        # single body carries: the panel's ROWS are the panel, so they take the
        # dock's fill and the dock's one-cell inset rather than sitting on bare
        # ground one cell to the left of every other glyph in the column. Without
        # it this panel read as a filled caption over a floating list — and only
        # while a `/btw` card happened to be open, since `Screen.aside #band`
        # filled the band underneath them and hid it (design round 12, D1/D5).
        self._list = VerticalScroll(id="subagent-rows", classes="band-body")
        # A status surface must not steal composer focus. Mouse-wheel scrolling
        # still works when the expanded roster exceeds its screen budget.
        self._list.can_focus = False
        self._affordance = SubagentAffordance()
        self._rows: dict[str, SubagentRow] = {}
        self._expanded = False
        #: How much of the dock the panel takes (#525). VIEW state, like
        #: `_expanded`: `ctrl+g` and the header click cycle it, the setting
        #: only seeds it, and nothing writes it back to disk — a hard override
        #: would make the key a no-op again, the exact defect being fixed.
        self._density: Density = DEFAULT_DOCK_DENSITY
        #: Whether the user chose the current density THIS session (plandex's
        #: `userToggledBuild`). An explicit choice holds against a live edit of
        #: `display.dock` and against new children starting; only a child
        #: FAILING breaks through, and that clears it so the next press reads
        #: naturally (summary → hidden re-hides what the failure surfaced).
        self._user_density: bool = False
        #: Whether the seed from `display.dock` has been read. Deferred to the
        #: first non-empty `sync` rather than `__init__` because the panel is
        #: built at compose time, before the config watcher the reader prefers
        #: exists, and a session swap (`/new`, `/resume`) re-seeds through
        #: `reset_density` anyway.
        self._density_seeded: bool = False
        #: What the summary caption last painted, for its paint-once guard
        #: (counts + spinner frame). None until the summary has painted once.
        self._summary_key: tuple[SummaryCounts, str] | None = None
        # Logical focus within the full roster. The expanded DOM displays only
        # one screenful around it; every job remains keyboard-reachable without
        # making Textual reflow 100 hidden/off-screen widgets per arrow press.
        self._navigation_index = 0
        self._painted_rows = 0
        #: Last ledger read, keyed by job id. Refresh repopulates it; the
        #: spinner tick repaints from it between refreshes rather than
        #: re-querying the manager eight times a second.
        self._jobs_by_id: dict[str, Any] = {}
        self._selected_job_id = ""
        self._spinner_index = 0
        self._spinner_timer = None
        #: Interval the live timer was created with; a focus change compares
        #: against it to decide whether the timer must be replaced.
        self._spinner_rate: float = SPINNER_INTERVAL_S
        #: Whether the header has been painted once. It is a static label —
        #: the running-count that would once have changed it moved to the
        #: status band — so a second paint can only ever redraw the same row.
        self._header_shown: bool = False
        #: Whether the caption is currently carrying the roster count, which it
        #: does only in the compact state. Tracked so `_paint_header`'s
        #: paint-once guard can still repaint when the state flips (a resize
        #: across the threshold), without repainting on every unrelated tick.
        self._header_compact: bool = False
        #: Whether the full-density caption is carrying the `ctrl+g` hint (no
        #: overflow, so no affordance row to say it). Same guard purpose as
        #: `_header_compact`: repaint when the hint moves, not per tick.
        self._header_hint: bool = False
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
        #: Shared width of the role field, so the segments after it line up on
        #: every row (:func:`panel_layout`). Zero when no child on the roster
        #: carries a role, which is the ordinary fan-out and costs nothing.
        self._role_column = 0
        self.display = False

    def compose(self):  # type: ignore[override]
        yield self._header
        yield self._list
        yield self._affordance

    @property
    def density(self) -> Density:
        """The panel's current :class:`Density` (view state; see `_density`)."""
        return self._density

    @property
    def has_overflow(self) -> bool:
        """Whether the collapsed preview hides rows, i.e. expanding is a step.

        Gated on the BUDGET, not the flat ceiling: on a short terminal the
        preview shows fewer rows than `_PREVIEW_JOB_ROWS`, so a roster of
        four children can already have rows hidden. Keying the refusal on the
        constant would make `ctrl+g` a silent no-op on exactly the screens
        where it is the only way to reach them (UX round 2, U7).
        """
        return len(self._rows) > self._preview_job_rows()

    def toggle_expanded(self, *, enter_navigation: bool = False) -> None:
        """Advance the density cycle one step, optionally entering navigation.

        The name predates the cycle and is kept because both callers (the
        ``ctrl+g`` action and the header/affordance click) already go through
        it; what it does is now the ``ctrl+g`` table from the #525 design:

        * full, no overflow → summary (was a silent no-op: the bug)
        * full, overflow, collapsed → expanded roster, as before
        * full, overflow, expanded → collapse AND shrink to summary in one
          press — the roster's own collapse is folded into the forward cycle
          rather than being a fourth stop the user has to press through
        * summary → hidden
        * hidden → full, collapsed

        Any press is the USER's choice, so it pins `_user_density` (decision
        2): from here a live `display.dock` edit and new children starting
        leave the density alone; only a failure moves it, and that unpins.
        """
        self._user_density = True
        if self._density is Density.FULL and self.has_overflow and not self._expanded:
            self._expanded = True
            self._apply_visibility()
            # Existing row content remains valid across disclosure/navigation.
            # A full 100-row repaint here delayed the key's visible focus
            # feedback; the ordinary spinner tick refreshes facts afterwards.
            if enter_navigation:
                # ``ctrl+g`` is the explicit keyboard-navigation gesture: start
                # at the oldest row so arrows can traverse every retained child.
                # Pointer disclosure only changes visibility and must leave the
                # draft-bearing composer's focus and next keystroke untouched.
                self._navigation_index = 0
                self._apply_visibility()
                self.call_after_refresh(self._focus_navigation_row)
            return
        if self._density is Density.FULL:
            next_density = Density.SUMMARY
        elif self._density is Density.SUMMARY:
            next_density = Density.HIDDEN
        else:
            next_density = Density.FULL
        # Leaving the expanded roster from the keyboard hands focus back to
        # the composer exactly as the old collapse did; the pointer path never
        # took it. Unconditional on `_expanded` because navigation may have
        # been entered by the key and the collapse by the click.
        if self._expanded and enter_navigation:
            self.exit_navigation()
        self._set_density(next_density)

    def _set_density(self, density: Density) -> None:
        """Apply ``density`` and settle the panel's own geometry for it.

        Every density change funnels through here so the three things that
        must move together always do: the expanded flag (only meaningful in
        `full`, and a hidden roster that came back expanded would re-take the
        screen), the row visibility and the caption. The band inset and the
        todo budget are the app's to settle — callers go through
        `request_toggle`/`action_toggle_subagents`, which `_refresh_band` in
        the same frame (design §8, dock height jumps).
        """
        if density is not Density.FULL:
            self._expanded = False
        changed = density is not self._density
        self._density = density
        # `display` is settled HERE and re-asserted by `sync`, not only there:
        # the key press must change the frame it lands on, and `sync` only
        # runs on the following band refresh.
        if self._rows:
            self.display = density is not Density.HIDDEN
        self._apply_visibility()
        if changed:
            # A caption keyed on the old density would stand for a repaint.
            self._header_shown = False
            self._summary_key = None
            self._paint_header()
            self._sync_spinner_for_density()

    def _sync_spinner_for_density(self) -> None:
        """Stop the spinner while hidden; let `sync_animation_rate` restart it.

        A hidden panel has nothing to animate, and without this it would keep
        ticking at 12.5 fps for the rest of the session (design §8). The
        restart is not done here: `_tick`'s own start/stop already answers to
        whether any row is running, and it is called from the sync that paints
        the re-emerged panel.
        """
        if self._density is Density.HIDDEN:
            self._stop_spinner()
        elif any(row.running for row in self._rows.values()):
            self._start_spinner()

    def seed_density(self, density: Density, *, force: bool = False) -> None:
        """Take the configured initial density without claiming it as a choice.

        The setting's job (decision 5): it is where a session STARTS, so a
        `ctrl+g` afterwards still cycles from it and never writes it back. The
        live-apply path uses the same entry with the same rule — a file edit
        applies only while the user has not chosen a density this session
        (`force` is for the session swap, where the previous choice belonged to
        a conversation that is gone).
        """
        if force:
            self._user_density = False
        if self._user_density:
            return
        self._set_density(density)

    def reset_density(self) -> None:
        """Session swap: forget the user's choice and re-read `display.dock`."""
        self._density_seeded = False
        self._user_density = False

    def note_child_failed(self) -> None:
        """A child failed: a hidden panel re-emerges as ONE row, never more.

        The clig.dev rule the issue quotes ("if there is an error, print the
        logs") scaled to a row: a failure should not re-take 2-8 rows of the
        user's screen, it should make one row appear that says ``1 failed``.
        Regardless of `_user_density` — a failure outranks the preference —
        and it CLEARS the pin so the user's next press re-hides it rather than
        stepping to `full` (decision 2). Cancelled and interrupted children
        are not failures and do not come through here; neither do restored
        rows, which never end in this process.
        """
        if self._density is not Density.HIDDEN:
            return
        self._user_density = False
        self._set_density(Density.SUMMARY)

    def _configured_density(self) -> Density:
        """The `display.dock` value, mapped to the enum; unknown strings are full.

        Function-local import for the reason `tui/settings.py` gives: this
        module is on the band paint path and the reader is cheap, but the
        registry it consults on first read is not, so it is not paid at import.
        """
        try:
            from local_operator.tui.settings import settings_get

            raw = settings_get("display.dock", DEFAULT_DOCK_DENSITY.value)
            return Density(str(raw).strip().lower())
        except Exception:
            return DEFAULT_DOCK_DENSITY

    def _focus_navigation_row(self) -> None:
        rows = list(self._rows.values())
        if rows:
            rows[self._navigation_index].focus(scroll_visible=True)

    def move_focus(self, job_id: str, direction: int) -> None:
        """Move through the complete start-ordered roster without wrapping."""
        job_ids = list(self._rows)
        try:
            current = job_ids.index(job_id)
        except ValueError:
            current = self._navigation_index
        target = max(0, min(len(job_ids) - 1, current + direction))
        self._navigation_index = target
        # Only shift the visible window at its edge. Most arrow presses now move
        # focus without touching layout; crossing a page boundary swaps one row.
        if not self._rows[job_ids[target]].display:
            self._apply_visibility()
        self._focus_navigation_row()

    def collapse_for_child_view(self) -> None:
        """Give the child page the rows the expanded roster was temporarily using.

        Touches `_expanded` only: the density is the user's and survives opening
        a child page and coming back (design §8).
        """
        if not self._expanded:
            return
        self._expanded = False
        self._apply_visibility()

    def exit_navigation(self) -> None:
        """Return keyboard ownership to the composer without touching its draft."""
        try:
            editor = self.app.query_one("Editor")
            editor.focus()
        except Exception:
            pass

    def request_toggle(self) -> None:
        """Cycle from the pointer path and settle the dock in the same frame.

        The SAME cycle as ``ctrl+g`` (the header and the affordance both land
        here) but with ``enter_navigation=False``: a click is a visibility
        gesture and must never take focus from a draft-bearing composer.
        """
        self.toggle_expanded()
        app = getattr(self, "app", None)
        refresh = getattr(app, "_refresh_band", None)
        if callable(refresh):
            refresh()

    def _apply_visibility(self) -> None:
        """Show the preview slice, or make every start-ordered row reachable.

        In `summary` and `hidden` the list and the affordance are switched off
        and the panel is header-only (one row, or nothing); the row budget
        arithmetic already counts the caption, so `_painted_rows` is
        ``_HEADER_ROWS`` and the app's inset and todo budget follow from it.
        """
        job_ids = list(self._rows)
        if self._density is not Density.FULL:
            for row in self._rows.values():
                row.display = False
            self._list.display = False
            self._affordance.display = False
            self._painted_rows = _HEADER_ROWS if self._density is Density.SUMMARY else 0
            self._paint_header()
            return
        self._list.display = True
        # The COLLAPSED budget is computed in both states: the expanded branch
        # sizes its own viewport, but the affordance below is keyed on this one
        # so the way back stays on screen (see `has_overflow`).
        preview_budget = self._preview_job_rows()
        if self._expanded:
            try:
                budget = (
                    max(1, preview_budget)
                    if self.screen.has_class("subagent")
                    else max(1, int(self.screen.size.height) - _EXPANDED_DOCK_ROWS)
                )
            except Exception:
                budget = len(job_ids)
            self._navigation_index = max(0, min(len(job_ids) - 1, self._navigation_index))
            start = max(0, min(self._navigation_index, len(job_ids) - budget))
            visible_ids = set(job_ids[start : start + budget])
        else:
            budget = preview_budget
            # A zero budget is the COMPACT state, and the slice has to be
            # written as a guard rather than `job_ids[-0:]` — which is the whole
            # list, the exact inversion of what a zero budget asks for.
            visible_ids = self._priority_slice(job_ids, budget) if budget > 0 else set()
        for job_id, row in self._rows.items():
            row.display = job_id in visible_ids
        visible_count = len(visible_ids)
        # Overflow is measured against the COLLAPSED budget in both states, so
        # an expanded roster showing every child still carries the row that says
        # `ctrl+g to collapse`. Keying it on what is hidden RIGHT NOW would hide
        # the affordance exactly when the user needs it to get back.
        has_overflow = len(job_ids) > preview_budget
        self._affordance.display = has_overflow
        list_rows = visible_count
        self._list.styles.max_height = budget
        self._painted_rows = _HEADER_ROWS + list_rows + int(has_overflow)
        if has_overflow:
            # `+N earlier` is only TRUE when the hidden set is the roster's
            # prefix, which the collapsed priority pick no longer guarantees:
            # a failed child from the first batch can outrank a completed one
            # from the last. `+N more` when the hidden rows are mixed in. The
            # expanded roster keeps the wording it had: its window scrolls and
            # the count there is "not in view", the same as before.
            hidden_ids = [job_id for job_id in job_ids if job_id not in visible_ids]
            prefix = job_ids[: len(hidden_ids)]
            self._paint_affordance(len(hidden_ids), earlier=self._expanded or hidden_ids == prefix)
        # The caption changes shape with the compact state, and that flips on a
        # RESIZE — which moves no row content, so `_paint_all` does not run.
        # Repainting here keeps the count truthful across a resize; the guard
        # inside makes the call free when the state has not moved.
        self._paint_header()

    def _preview_job_rows(self) -> int:
        """Job rows the COLLAPSED preview may paint at the current height.

        A budget rather than the flat ``_PREVIEW_JOB_ROWS`` because the dock is
        chrome ABOVE the composer: it may shorten itself, but it may never take
        the conversation. On a short terminal the fixed six-row preview did
        exactly that once the roster was restored on the first frame — the
        transcript went to zero rows at 100x24 (UX round 2, U7).

        Ceiling stays ``_PREVIEW_JOB_ROWS`` so nothing changes on a normal
        terminal, and the floor is :data:`_MIN_PREVIEW_JOB_ROWS` so the panel
        keeps saying that children exist even where it cannot list them. What is
        cut is never lost: the affordance's ``+N earlier`` counts every hidden
        row and ``ctrl+g`` still reaches them, which is the same
        bounded-preview-plus-counter contract the panel already had — this only
        makes the bound depend on the screen instead of assuming one.

        Subtracts the band's other slots for the reason
        ``TodoPanel._band_sibling_rows`` does: a budget blind to a docked todo
        list re-opens the clip the moment both panels are up on a short screen.
        """
        try:
            screen_height = int(self.screen.size.height)
        except Exception:
            # No screen yet (compose-time sync): the flat ceiling is the honest
            # answer, and the first real paint re-runs this with a height.
            return _PREVIEW_JOB_ROWS
        if screen_height <= 0:
            return _PREVIEW_JOB_ROWS
        floor = _TRANSCRIPT_FLOOR_ROWS
        if self.screen.has_class("subagent") and not self.screen.has_class("subagent-compact"):
            # The detail page has title/breadcrumb/footer rows the root transcript
            # does not. Compact mode recovers those from its disabled composer;
            # normal mode must reserve them before allocating auxiliary rows.
            floor += 6
        available = screen_height - _COLLAPSED_DOCK_ROWS - floor - self._band_sibling_rows()
        return max(_MIN_PREVIEW_JOB_ROWS, min(_PREVIEW_JOB_ROWS, available))

    def _band_sibling_rows(self) -> int:
        """Rows the band's OTHER visible slots occupy, outer size included.

        Predicted through ``app.slot_rows`` rather than measured, the same call
        ``TodoPanel._band_sibling_rows`` and the band's own inset check use — one
        answer to "how tall is that slot", so the three cannot disagree about the
        same frame. A sibling that has just been un-hidden measures zero until
        Textual re-arranges, which is precisely when this runs.

        Imported lazily: the app imports this module, so a module-level import
        would close the cycle.
        """
        parent = self.parent
        if parent is None:
            return 0
        try:
            from local_operator.tui.app import slot_rows

            return sum(
                slot_rows(slot) for slot in parent.children if slot is not self and slot.display
            )
        except Exception:
            return 0

    def _priority_slice(self, job_ids: list[str], budget: int) -> set[str]:
        """The ``budget`` rows the collapsed preview shows, by what needs attention.

        Was ``job_ids[-budget:]`` — newest by start order — which evicted a
        failed child the moment enough later ones completed, so the one row
        that asked for action was the one the preview dropped (#525 design
        §4). Rank by :data:`_EVICTION_RANK` (running/queued, then failed, then
        interrupted, then everything settled quietly), ties to the newest, and
        the picked set keeps its start order because `_sync_rows` owns the DOM
        order and this only decides which rows are displayed.

        Only the collapsed preview: the expanded roster shows every row, so
        there is nothing to pick.
        """
        if len(job_ids) <= budget:
            return set(job_ids)

        def rank(item: tuple[int, str]) -> tuple[int, int]:
            index, job_id = item
            row = self._rows.get(job_id)
            # The row's `running` flag is the last-painted state, not the job's
            # — a row mounted this sync has not been painted yet, so reading it
            # here would rank a failed child as running and lose it to the
            # budget. Fall back to the job's own status, which `_sync_rows`
            # has just refreshed.
            if row is not None and row.running:
                status = "running"
            else:
                facts = row_facts(self._jobs_by_id.get(job_id), fallback_id=job_id, current=False)
                status = "running" if facts.queued else facts.status
            # Newest first within a rank: a negative index sorts later starts
            # ahead, which is the `[-budget:]` rule the slice had before.
            return (_EVICTION_RANK.get(status, 3), -index)

        picked = sorted(enumerate(job_ids), key=rank)[:budget]
        return {job_id for _index, job_id in picked}

    def _paint_affordance(self, hidden: int, *, earlier: bool = True) -> None:
        dim = Style(color=theme_mod.semantic_color("dim"))
        muted = Style(color=theme_mod.semantic_color("muted"))
        row = Text(no_wrap=True, overflow="ellipsis")
        if hidden:
            row.append(f"+{hidden} {'earlier' if earlier else 'more'} · ", style=dim)
        row.append("ctrl+g to collapse" if self._expanded else "ctrl+g to expand", style=muted)
        self._affordance.update(row)

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
        about to take. In `summary` that row IS the panel; a hidden panel is
        not displayed, so the app never asks.
        """
        return max(1, self._painted_rows)

    def on_unmount(self) -> None:
        self._stop_spinner()

    # -- sync -------------------------------------------------------------
    def sync(
        self, session: Any, *, jobs: Sequence[Any] | None = None, selected_job: Any = None
    ) -> None:
        """Re-read ``session.jobs`` and schedule a repaint.

        Called on every Subagent* event (immediate) and on the 1 Hz poll (the
        belt to the events' suspenders — elapsed time moves with no event at
        all). Never raises: this is a status surface.

        The rows are NOT painted here. This marks the panel dirty and lets
        :meth:`_tick` do it, so a burst of child events costs one repaint
        instead of one per event per row; a row appearing or leaving is the
        exception and paints at once (see the class docstring).
        """
        if jobs is None:
            try:
                manager = getattr(session, "jobs", None)
                jobs = manager.list() if manager is not None else []
            except Exception:
                jobs = []
        job_rows = jobs or []
        self._model_label = str(getattr(session, "model_label", "") or "")
        task_jobs = [job for job in job_rows if getattr(job, "type", "") == "task"]
        selected_id = str(getattr(selected_job, "id", "") or "")
        self._selected_job_id = selected_id
        if selected_id:
            # The viewed manager is not its own child row, but its status band
            # still needs the same off-thread stats cache as visible children.
            self._stats_for(selected_id, selected_job, True)
        if not task_jobs:
            self._jobs_by_id = {}
            self._stats = {key: value for key, value in self._stats.items() if key == selected_id}
            self._sync_rows([])
            self.display = False
            self._dirty = False
            self._stop_spinner()
            return
        if not self._density_seeded:
            # First non-empty sync of this session: the setting is the INITIAL
            # density (decision 5). Read here rather than at construction so
            # the config watcher's snapshot, which the reader prefers, exists.
            self._density_seeded = True
            self.seed_density(self._configured_density())
        # NOT an unconditional `True`: the 1 Hz poll lands here every second,
        # and un-hiding on each tick would make `hidden` last one second.
        self.display = self._density is not Density.HIDDEN
        self._jobs_by_id = {str(getattr(job, "id", "") or ""): job for job in task_jobs}
        changed = self._sync_rows(task_jobs)
        self._apply_visibility()
        if changed:
            self._paint_all()
        if self._density is Density.HIDDEN:
            # Nothing is painted, so nothing should animate; the rows keep
            # their facts through `_sync_rows` for the frame the panel returns.
            self._dirty = False
            self._stop_spinner()
            return
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
                # The selected reader still consumes its owner's band stats
                # after that job stops being a visible child-list row.
                if job_id != self._selected_job_id:
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
            if not row.display:
                continue
            job = self._jobs_by_id.get(job_id)
            if job is None:
                continue
            facts = row_facts(job, fallback_id=job_id, current=row.current)
            measured.append((job_id, row, facts, self._stats_for(job_id, job, reread_stats)))
        self._rung, self._column, self._clock, self._role_column = panel_layout(
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
                role_column=self._role_column,
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
        # In the COMPACT state the caption is the only thing left, so it has to
        # carry the total the hidden rows would otherwise have shown. That is
        # not a second copy of the band's RUNNING tally (the D-04 objection):
        # this counts every child the session HAS, which on a resumed session
        # with nothing running is precisely the number the band cannot show.
        #
        # The SUMMARY density is the same zero-row shape asked for by the user
        # rather than forced by the height, so it paints the same row (design
        # §8: one vocabulary, not two) — the counts caption replaced the bare
        # `Subagents · 19` when the summary row arrived (#525).
        compact = self._painted_rows > 0 and not self._list_shows_any_row()
        if compact:
            self._paint_summary()
            return
        # In full with no overflow the affordance row is not painted, so the
        # hint moves into the caption: a `ctrl+g` the user only ever saw with
        # seven or more children was the discoverability half of #525, and
        # the caption already has the row. With overflow the affordance says
        # it and the caption does not repeat it.
        hint = not self._affordance.display
        if self._header_shown and not self._header_compact and hint == self._header_hint:
            return
        self._header_shown = True
        self._header_compact = False
        self._header_hint = hint
        self._summary_key = None
        muted = Style(color=theme_mod.semantic_color("muted"))
        dim = Style(color=theme_mod.semantic_color("dim"))
        header = Text(no_wrap=True, overflow="ellipsis")
        header.append("Subagents", style=muted)
        if hint:
            header.append(STATS_SEAM, style=dim)
            header.append("ctrl+g", style=muted)
        self._header.update(header)

    def _paint_summary(self) -> None:
        """Paint the one-row summary, once per (counts, spinner frame).

        The guard is keyed on the counts tuple and the glyph, not on the
        density flag the old caption keyed on: a summary keyed on the flag
        alone painted once and then froze while children settled (design §8).
        """
        counts = summary_counts(list(self._jobs_by_id.values()))
        glyph = SPINNER_FRAMES[self._spinner_index] if counts.running else ""
        key = (counts, glyph)
        if self._header_shown and self._header_compact and self._summary_key == key:
            return
        self._header_shown = True
        self._header_compact = True
        self._summary_key = key
        self._header.update(compose_summary(counts, spinner_glyph=glyph, width=self._row_width()))

    def summary_text(self) -> str:
        """The plain string the caption reads right now, for a test to assert."""
        return str(self._header.content)

    def _list_shows_any_row(self) -> bool:
        """Whether the roster is painting a job row right now."""
        return any(row.display for row in self._rows.values())

    def on_resize(self) -> None:
        """A narrower panel is a different row layout, for every row at once.

        Marked dirty rather than painted: a window drag emits a resize per
        column, and the tick is what turns that stream into one repaint.
        """
        self._apply_visibility()
        # Resize arrives before the new geometry has fully settled. Reapplying
        # after layout prevents a short terminal's max-height surviving the
        # grow-back frame and leaving the screen virtually one row too tall.
        self.call_after_refresh(self._apply_visibility)
        self._dirty = True
        if self._rows:
            self._start_spinner()

    # -- tick ----------------------------------------------------------------
    def _spinner_interval(self) -> float:
        """Full cadence when the terminal is focused, reduced when it is not."""
        return SPINNER_INTERVAL_S if animation_focused() else BLURRED_SPINNER_INTERVAL_S

    def _start_spinner(self) -> None:
        if self._spinner_timer is None and self.is_mounted:
            self._spinner_rate = self._spinner_interval()
            self._spinner_timer = self.set_interval(self._spinner_rate, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def sync_animation_rate(self) -> None:
        """Re-rate the tick after a focus change.

        A Textual timer's interval is fixed at creation, so the timer is
        replaced. Only a panel that already HAS one is touched: this must never
        start a tick under a dock of settled children, which is the state
        `_tick` deliberately stops itself in.

        Coming back to the fast rate marks the panel dirty rather than painting
        here, because `_tick` is this panel's one repaint point and a second
        paint path is exactly the duplication its docstring warns about. The
        next tick is then a FULL repaint, so the numbers, clocks and row set a
        user sees on refocus are current rather than whatever the throttled
        window last painted.
        """
        if self._spinner_timer is None:
            return
        wanted = self._spinner_interval()
        if wanted == self._spinner_rate:
            return
        self._stop_spinner()
        self._dirty = True
        self._start_spinner()

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
        if self._density is not Density.FULL:
            # Header-only: the rows are not displayed, so the cheap path is the
            # caption — its own guard keeps a settled roster from repainting.
            # `_dirty` is consumed here too, or the first tick after the panel
            # returns to `full` would still owe a full paint (which it does,
            # from the sync that un-hid it).
            self._paint_header()
            self._dirty = False
            if not any(row.running for row in self._rows.values()):
                self._stop_spinner()
            return
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
                        role_column=self._role_column,
                    )
        # Sampled AFTER the paint, from the rows this tick has just produced.
        # Read beforehand it said False for a queued child that started in
        # this very tick, and the timer stopped under a row that had just come
        # alive — frozen spinner and frozen clock until the next 1 Hz sync.
        running = any(row.running for row in self._rows.values())
        if not running and not self._dirty:
            self._stop_spinner()
