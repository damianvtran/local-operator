"""Metric vocabulary, percentile reduction, and the budget verdicts for the TTFT bench.

WHY THIS IS A SEPARATE MODULE FROM THE DRIVERS
==============================================
The numbers, the way they are reduced, and the rule that decides PASS/FAIL are the
part of this harness that a reviewer has to be able to trust without running it —
they are pure functions over dictionaries, so they are unit-testable with no
subprocess, no daemon and no provider. The drivers (``channels.py``) are the part
that needs a machine. Keeping them apart is what lets ``tests/unit/scripts`` pin
the arithmetic and the budget table in CI while the six-channel run happens on a
loaded host.

THE METRIC CONTRACT
===================
Every metric is milliseconds measured from the moment the turn was SUBMITTED, and
``-1.0`` means "this channel cannot observe that here". ``-1`` is deliberately not
``0``: a missing observation and a zero-latency observation are opposite findings,
and the reasoning column below is the whole reason this harness exists — on the
tree this was written against, every channel reports ``-1`` for it, and that is
the measurement, not a gap.

:data:`FIRST_PAINT` is the headline: submit -> the first event the FRONT END can
paint for this turn. It is the only metric with a budget, and the budget is 300 ms
because that is the achievable contract the design doc settled on.

IT IS A PAINT MOMENT, NOT A THINKING MOMENT, and the name says so on purpose
(QA round 1, Q4). A front end can paint a frame before any assistant content
arrives: on the phone arm a frame was painted at 26 ms while the first text reached
the handset-facing path at 157 ms, a 5.6x gap, so a reader who takes the headline
for "the operator saw thinking" is reading it wrong. The thinking moment is
:data:`FIRST_REASONING` and the content moment is :data:`FIRST_TEXT`; the table
prints all three side by side for exactly this reason. The metric was called
``first_event_ms`` before round 1 and is ``first_paint_ms`` now.

THE LARGEST CLAIM THIS HARNESS SUPPORTS, in one sentence: *with the provider's
first byte stubbed out, the first frame of a warm single turn reaches the front end
within 300 ms on this repository's own code, and it did so on every run of the
final instrument taken in one comparable window* — everything else it prints (real
wire, cold engage, 4 and 8 concurrent turns, the provider's own floor) is a
measured number with its load, not a guarantee.

:data:`FIRST_REASONING` and :data:`FIRST_TEXT` split :data:`FIRST_PAINT` by what
the content IS, which is what makes the operator's complaint legible: the model
reasoning for 0.45-1.75 s while the front end shows nothing, because the runtime
drops ``StreamReasoningDelta`` (``harness/loop.py``) before any consumer sees it.

:data:`PROVIDER_REASONING` / :data:`PROVIDER_TEXT` are the PROVIDER's own clock —
when the model put the token on the wire — and they are REPORTED, NEVER ASSERTED.
See :data:`BUDGET_MS` for why.

:data:`RUNTIME_EMIT` is the runtime-side (or daemon-side) emission of the event
:data:`FIRST_PAINT` measured, so transport cost separates from runtime cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------

#: Submit -> the first event the front end can PAINT for this turn. The headline.
#: The name is the contract: this is the first FRAME, not the first thinking
#: content — see the module docstring for why (QA Q4) and for the one-sentence
#: claim this number supports.
FIRST_PAINT = "first_paint_ms"
#: Submit -> the first streaming event carrying model REASONING at the front end.
#: ``-1`` on every channel until the runtime stops dropping reasoning deltas.
FIRST_REASONING = "first_reasoning_ms"
#: Submit -> the first streaming event carrying assistant TEXT at the front end.
FIRST_TEXT = "first_text_ms"
#: Submit -> the submit request RETURNED. Admission is not visible output, and on
#: the cold desktop arm it is the number that separates "the engage happened" from
#: "the user saw something" — reported for every channel that has a request.
ADMITTED = "admitted_ms"
#: Submit -> the RUNTIME produced a reasoning delta. Observed BELOW the front end
#: (in-process channels only), so it is the evidence that reasoning happens while
#: :data:`FIRST_REASONING` says no front end can show it.
RUNTIME_REASONING = "runtime_reasoning_ms"
#: Submit -> the runtime/daemon emitted that first event. ``-1`` where the harness
#: does not own the emitting process or the wire carries no stamp; see each
#: driver's ``emit`` note for which is which.
RUNTIME_EMIT = "runtime_emit_ms"
#: Submit -> the PROVIDER had the first reasoning token on the wire. Reported only.
PROVIDER_REASONING = "provider_reasoning_ms"
#: Submit -> the PROVIDER had the first text token on the wire. Reported only.
PROVIDER_TEXT = "provider_text_ms"
#: Submit -> the runtime entered its stream function, i.e. all local work before
#: the provider call. In-process channels only.
STREAM_ENTERED = "stream_entered_ms"
#: Submit -> the turn finished. Context for the row, never a budget.
TURN = "turn_ms"

#: Every metric a driver may record, in the order the table prints them.
METRICS: tuple[str, ...] = (
    FIRST_PAINT,
    FIRST_REASONING,
    FIRST_TEXT,
    RUNTIME_REASONING,
    RUNTIME_EMIT,
    PROVIDER_REASONING,
    PROVIDER_TEXT,
    STREAM_ENTERED,
    ADMITTED,
    TURN,
)

#: The three the summary table shows: what the operator waits for, split by what
#: the content is.
TABLE_METRICS: tuple[str, ...] = (FIRST_PAINT, FIRST_REASONING, FIRST_TEXT)

#: Sentinel for "not observable on this channel". NOT zero — see the module
#: docstring.
UNAVAILABLE = -1.0

#: Percentiles the summary reports. Reported, and only the p50 is ever asserted
#: (see :data:`BUDGET_MS`).
PERCENTILES: tuple[int, ...] = (50, 95, 99)


# ---------------------------------------------------------------------------
# The budget, and the two things it must never become
# ---------------------------------------------------------------------------

#: The achievable contract, from the design doc: < 300 ms from submit to the
#: first event the RUNTIME emits and the front end paints.
#:
#: IT IS NOT, AND MUST NEVER BECOME, A BUDGET ON THE FIRST PROVIDER TOKEN. That
#: was measured and it is impossible: the provider floor on the operator's own
#: machine is 355-514 ms on a 102-token prompt and 1.55 s on their 227k-token
#: cached p50 prompt (both against real deepseek, two independent clients — see
#: ``scripts/probe_provider_ttfb.py``). A future reader who "fixes" this file by
#: adding ``PROVIDER_REASONING < 300`` would be asserting something the wire
#: cannot deliver, on every run, forever. The provider numbers stay REPORTED and
#: BUDGETED (they are what the remaining wait consists of) but never asserted.
BUDGET_MS = 300.0

#: The concurrency levels the budget is ASSERTED at, and the measurement that fixed
#: it. On this host (14 CPUs, ~25 concurrent agent sessions, load 130-180 during the
#: baseline) every channel is far over 300 ms at 8 concurrent turns — exec 26.9 s,
#: sse-jobs 22.4 s, desktop 19.8 s (cold) and 733 ms (warm), tui 1.2-1.6 s — because
#: the host's own queue is in the number. A gate that fails on every run for a reason
#: the diff cannot control is not a gate, it is a red light wired to the weather, so
#: the assertion covers the cells whose latency this repository OWNS end to end
#: (one turn, one process) while every higher concurrency is still measured and
#: PRINTED against the same budget. The over-budget cells are the evidence for the
#: per-turn local work this ticket is about — they are reported, not deleted.
ASSERTED_CONCURRENCY: tuple[int, ...] = (1,)


#: Cells whose budget is ENFORCED at :data:`ASSERTED_CONCURRENCY`: the harness exits
#: non-zero when their p50 meets or exceeds :data:`BUDGET_MS`. Keyed by
#: ``(channel, warmth)``.
#:
#: ``tui`` warm is the floor this repository owns end to end — no process spawn, no
#: daemon, no bridge — and ``desktop`` warm is the same turn with the daemon plane on
#: top of it; both were measured well inside the budget before this harness existed
#: (21.8 ms and 52.7 ms p50), so enforcing them is a regression gate, not a target.
ENFORCED_CELLS: frozenset[tuple[str, str]] = frozenset(
    {
        ("tui", "warm"),
        ("desktop", "warm"),
    }
)

#: The provider arm an ENFORCED cell must have been measured on, and the reason the
#: enforced population is scoped by provider rather than by channel (QA round 1, Q2).
#:
#: The reviewer's finding was that the enforced cells were red or green with the
#: weather: ``desktop/warm@1`` PASSED at 143 ms on a load-86 box and FAILED at 624 ms
#: on a load-111-131 box, and that single cell decided the exit code. The cause is
#: which hop sits inside the measured window. On the ``test`` provider the turn is
#: entirely local — the mock answers from a canned list, so what is left in
#: submit -> first paint is local-operator's own work plus this process being
#: scheduled, and that measured 26 ms (tui/warm@1, load 100-145) against a 300 ms
#: budget: an order of magnitude of headroom, which is what makes it a REGRESSION
#: gate rather than a reading of the box. On the real-wire arm the same cell measured
#: 210-450 ms in the SAME windows, because a genuine TCP+HTTP+SSE round trip to the
#: loopback server is inside the number and a loaded scheduler stretches it.
#:
#: So enforcement is scoped to the arm whose budget this repository OWNS end to end,
#: and the real-wire cells stay BUDGETED AND PRINTED against the same 300 ms — their
#: number is the evidence about the hop, and it is not deleted or loosened to make a
#: gate pass. The threshold did not move for this: :data:`BUDGET_MS` is still 300.
ENFORCED_PROVIDER = "test"

#: The load-per-CPU band inside which an enforced cell may be read as PASS/FAIL.
#:
#: Outside the band the harness prints ``OUT-OF-BAND`` and exits 3 rather than
#: claiming a result, because a p50 taken while the box is swamped is a measurement of
#: the box. The band is CALIBRATED FROM THIS HARNESS'S OWN RUNS, and the calibration is
#: recorded here because the first attempt got it wrong in a way worth remembering:
#:
#: * enforced cells observed at 11-52 ms (tui/warm@1 11.1, desktop/warm@1 52.2) in a
#:   window at **12.2 load/CPU** — twenty-seven times inside the budget;
#: * enforced cells observed at 26 ms in the mock arm's first window at 7.1-10.4/CPU;
#: * the swamped windows of the same evening, where this box sat at 407-418 on 14 CPUs
#:   = **29-30/CPU** and a Python process is descheduled for hundreds of ms.
#:
#: The first value shipped was 12.0, and the re-take immediately showed why that was
#: wrong: a normal evening at 12.18/CPU was refused, so the gate answered `OUT-OF-BAND`
#: with its cells at 11 ms — an instrument that refuses to answer on an ordinary night
#: is as unusable as one that answers with the weather. 20.0 sits above every window in
#: which an enforced cell has actually been measured (<=12.2/CPU) and below the
#: swamped ones (29-30/CPU), which is the only boundary that matters here.
BUDGET_LOAD_PER_CPU_MAX = 20.0

#: Statuses that mean "this cell missed the budget, or lost every sample it had".
#: One definition, read by both :meth:`Verdict.failed` and the harness's exit code,
#: so the two cannot drift into a gate that fails in the table and exits 0.
FAILURE_STATUSES: frozenset[str] = frozenset({"FAIL", "FAILED"})

#: The submission ACKNOWLEDGEMENT ceilings, carried here rather than dropped with the
#: file they lived in. The ack-before-engage branch (PR #1343) asserted these on its
#: own desktop scenario; this instrument measures the same mark off the same frame
#: (``admission.accepted``, correlated on the submit's request id), so the gate moves
#: with the work instead of disappearing when that file was superseded.
#:
#: TWO CEILINGS, AND THE SECOND IS NOT A LOOSENED FIRST. The operator's requirement is
#: 300 ms and it is met where it is measured — the MEDIAN, 60-75 ms on this box even
#: under load. The tail does not stay inside 300 ms, and asserting that it does would
#: be this harness asserting a bound its own measurement contradicts: on that branch's
#: passes the ack's p95 was 148 ms, 219 ms and 326 ms at load 100-220, and QA's
#: independent matrix saw one cold submit of 333 ms in seven at load 140-190. 500 ms
#: is the bound those numbers support, stated with the load they were taken at.
ADMISSION_CEILING_MS = 300.0
ADMISSION_P95_CEILING_MS = 500.0

#: The channels whose admission acknowledgement is enforced. The frame is
#: session-scoped and delivered to an attached viewer, which is the desktop plane;
#: the other channels have no acknowledgement frame to score.
ADMISSION_CELLS: frozenset[str] = frozenset({"desktop"})

#: Verdicts that are neither a pass nor a failure, and the exit code they produce.
#: ``OUT-OF-BAND`` is the box's fault and ``NO-DATA`` is the gate having nothing to
#: judge (QA round 1, minor: an enforced cell with ``n=0`` used to read REPORTED,
#: which ``_amain`` treated as success — a gate that passes on no data is worse than
#: no gate). Both are non-zero on purpose: an instrument that cannot answer must not
#: let a pipeline read "green" out of that.
INDETERMINATE_STATUSES: frozenset[str] = frozenset({"OUT-OF-BAND", "NO-DATA"})
INDETERMINATE_EXIT_CODE = 3

#: Cells measured against the budget that the tree does NOT meet yet, each named
#: with the change that closes it. Reported as ``UNMET`` and never silently
#: dropped: the number is the evidence for that change, so losing it once the
#: harness exists would be the regression.
#:
#: ``desktop`` cold is the one that matters — the cold engage (spawn, import,
#: bind) happens INSIDE the submit POST, so no frame can reach the client first
#: and the first emitted event is 2.6 s (p50) away. The ack-before-engage change
#: (O2) is what moves it; when it lands, move this entry into
#: :data:`ENFORCED_CELLS` and delete the note.
PENDING_CELLS: Mapping[tuple[str, str], str] = {
    ("desktop", "cold"): "ack-before-engage (O2, sibling workstream)",
}


# ---------------------------------------------------------------------------
# Reduction
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], percent: int) -> float:
    """Nearest-rank percentile — the smallest observed value at or above ``percent``.

    Nearest-rank rather than interpolation, deliberately: with the run counts
    this harness uses (7 runs, so 7-56 samples per cell) a p99 IS the maximum,
    and interpolation would manufacture a number between two real observations
    and present it as a measurement. A p99 that equals the max is a weaker
    claim than an interpolated one and a TRUE one, which is the trade this
    harness takes everywhere.
    """
    if not values:
        raise ValueError("percentile of an empty sample set")
    if not 0 < percent <= 100:
        raise ValueError(f"percent must be in (0, 100], got {percent}")
    ordered = sorted(values)
    # ceil(percent / 100 * n) - 1, without importing math for one call.
    index = max(0, -(-percent * len(ordered) // 100) - 1)
    return ordered[min(index, len(ordered) - 1)]


def reduce_samples(samples: Iterable[float]) -> dict[str, Any]:
    """p50/p95/p99 plus the supporting facts, over the AVAILABLE samples only.

    ``UNAVAILABLE`` sentinels are excluded from the percentile arithmetic and
    counted separately, so a cell that observed the metric once out of eight
    turns reports ``n=1`` and an explicit ``unavailable=7`` rather than a
    percentile computed as though the seven misses were fast turns.
    """
    available = [value for value in samples if value != UNAVAILABLE]
    unavailable = sum(1 for value in samples if value == UNAVAILABLE)
    result: dict[str, Any] = {
        "n": len(available),
        "unavailable": unavailable,
        "min": round(min(available), 1) if available else UNAVAILABLE,
        "max": round(max(available), 1) if available else UNAVAILABLE,
    }
    for percent in PERCENTILES:
        key = f"p{percent}"
        result[key] = round(percentile(available, percent), 1) if available else UNAVAILABLE
    return result


@dataclass(frozen=True)
class Verdict:
    """One cell's budget verdict, with the reason a reader needs to act on it."""

    cell: tuple[str, str]
    status: str  # "PASS" | "FAIL" | "UNMET" | "WARN" | "REPORTED"
    reason: str

    @property
    def failed(self) -> bool:
        """A budget miss, or a cell whose every sample died instead of being measured.

        ``FAILED`` is the second case and it is a failure on purpose (round 1, Q1):
        the sweep continues past it, but a run that lost a cell must not exit 0.
        """
        return self.status in FAILURE_STATUSES

    @property
    def indeterminate(self) -> bool:
        """Neither a pass nor a failure: the instrument could not answer.

        Carried as its own exit code (see :data:`INDETERMINATE_EXIT_CODE`) so a
        pipeline cannot read "green" out of a run that never judged anything.
        """
        return self.status in INDETERMINATE_STATUSES


def judge(
    cell: tuple[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    concurrency: int = 1,
    provider_kind: str = "",
    load_per_cpu: float | None = None,
    errors: int = 0,
) -> Verdict:
    """Decide the cell's budget verdict from its reduced :data:`FIRST_PAINT` stats.

    The assertion is on the **p50**, and that is a deliberate ceiling on what this
    gate claims. The operator's ask is "under 300 ms on every channel even under
    load", and a loaded shared host cannot deliver that as a *guarantee*: this
    machine runs ~25 concurrent sessions and swings between load 20 and 600, so a
    p99 assertion here would be asserting the machine's queue, not the code's
    latency. A p50 that meets the budget is the honest gate; a p95 that does not is
    reported as ``WARN`` — visible, never fatal, and never silently promoted into an
    assertion. For the same reason the assertion applies at
    :data:`ASSERTED_CONCURRENCY` only; see that constant for the measured numbers
    that fixed it.

    FOUR THINGS DECIDE A VERDICT, and each of them answers a finding from round 1:

    * the cell must be in :data:`ENFORCED_CELLS` at :data:`ASSERTED_CONCURRENCY`
      (Q3: the enforced population is 2 of 30 cells and the output now says so);
    * it must have been measured on :data:`ENFORCED_PROVIDER`, because the budget is
      only this repository's to own where no real hop is inside the window (Q2);
    * the host must be inside :data:`BUDGET_LOAD_PER_CPU_MAX`, or the verdict is
      ``OUT-OF-BAND`` and the run exits 3 rather than reading the box (Q2);
    * it must have samples at all — an enforced cell with ``n=0`` is ``NO-DATA`` and
      also exits 3, because passing on no data is worse than not running (minor).

    ``errors`` is the number of samples that died instead of being measured. A cell
    where EVERY sample died is ``FAILED``: the sweep continues past it (that is the
    point of a matrix) but the run must not come out green.
    """
    channel, warmth = cell
    enforced = cell in ENFORCED_CELLS and concurrency in ASSERTED_CONCURRENCY
    # ``or {}`` rather than a None check: the mapping is read below under a guard that
    # already returned unless it exists and has samples, and this keeps that narrowing
    # true for the type checker as well as for a reader.
    stats_for_metric: Mapping[str, Any] = stats.get(FIRST_PAINT) or {}
    sample_count = int(stats_for_metric.get("n") or 0)
    if not sample_count:
        if errors:
            return Verdict(
                cell,
                "FAILED",
                f"every one of {errors} sample(s) died instead of being measured — "
                "the cell is a failure, not an absence (see the run's error lines)",
            )
        if enforced:
            return Verdict(
                cell,
                "NO-DATA",
                "enforced cell produced no samples; the gate has nothing to judge and "
                "will not read that as a pass (exit 3) — see judge()",
            )
        return Verdict(cell, "REPORTED", f"no {FIRST_PAINT} samples (channel unavailable here)")

    median = float(stats_for_metric["p50"])
    p95 = float(stats_for_metric.get("p95", UNAVAILABLE))
    errored = f" [{errors} sample(s) died]" if errors else ""
    if enforced:
        if provider_kind != ENFORCED_PROVIDER:
            return Verdict(
                cell,
                "REPORTED",
                f"p50 {median:.0f} ms on the {provider_kind} provider, printed against "
                f"the {BUDGET_MS:.0f} ms budget but NOT asserted: a real network hop is "
                f"inside this window and this repository does not own its timing "
                f"(see ENFORCED_PROVIDER — {ENFORCED_PROVIDER} owns it end to "
                f"end){errored}",
            )
        if load_per_cpu is not None and load_per_cpu > BUDGET_LOAD_PER_CPU_MAX:
            return Verdict(
                cell,
                "OUT-OF-BAND",
                f"p50 {median:.0f} ms measured at load {load_per_cpu:.1f}/CPU, outside "
                f"the {BUDGET_LOAD_PER_CPU_MAX:.1f}/CPU band this bound was calibrated "
                f"in: NOT judged (exit 3) — the number is the box's, re-run on a "
                f"quieter host or read it as load context{errored}",
            )
        if median >= BUDGET_MS:
            band = f" at load {load_per_cpu:.1f}/CPU" if load_per_cpu is not None else ""
            return Verdict(
                cell,
                "FAIL",
                f"p50 {median:.0f} ms >= budget {BUDGET_MS:.0f} ms over "
                f"{sample_count} samples{band}{errored}",
            )
        if p95 != UNAVAILABLE and p95 >= BUDGET_MS:
            return Verdict(
                cell,
                "WARN",
                f"p50 {median:.0f} ms inside budget; p95 {p95:.0f} ms is over it "
                f"(host contention; not asserted — see judge()){errored}",
            )
        return Verdict(
            cell,
            "PASS",
            f"p50 {median:.0f} ms < {BUDGET_MS:.0f} ms over {sample_count}"
            + (f" at load {load_per_cpu:.1f}/CPU" if load_per_cpu is not None else "")
            + errored,
        )
    if cell in PENDING_CELLS and concurrency in ASSERTED_CONCURRENCY:
        if median < BUDGET_MS:
            return Verdict(
                cell,
                "PASS",
                f"p50 {median:.0f} ms < {BUDGET_MS:.0f} ms — the pending change "
                f"({PENDING_CELLS[cell]}) has landed; move this cell into ENFORCED_CELLS",
            )
        return Verdict(
            cell,
            "UNMET",
            f"p50 {median:.0f} ms >= {BUDGET_MS:.0f} ms; closes with {PENDING_CELLS[cell]}",
        )
    if cell in ENFORCED_CELLS or cell in PENDING_CELLS:
        return Verdict(
            cell,
            "REPORTED",
            f"p50 {median:.0f} ms at {concurrency} concurrent, measured against the "
            f"{BUDGET_MS:.0f} ms budget but not asserted: the assertion is scoped to "
            f"concurrency {list(ASSERTED_CONCURRENCY)} (see ASSERTED_CONCURRENCY for "
            "the measured numbers that fixed it)",
        )
    return Verdict(cell, "REPORTED", "no budget — measured for the record")


def judge_admission(
    cell: tuple[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    concurrency: int = 1,
    provider_kind: str = "",
    load_per_cpu: float | None = None,
    errors: int = 0,
) -> Verdict | None:
    """The ADMISSION verdict for a cell, or ``None`` where it is not scored.

    Same three scope rules as :func:`judge` — the channel must acknowledge at all, the
    concurrency must be the asserted one, the arm must be the one whose window this
    repository owns, and the box must be inside the calibrated band — applied to
    :data:`ADMITTED` instead of :data:`FIRST_PAINT`. ``None`` rather than a REPORTED
    verdict for a channel with no acknowledgement frame: a table where every row says
    "not scored" for a thing that does not exist on that channel is noise, and the
    absence is a property of the channel rather than a reading of it.

    The p50 is held to :data:`ADMISSION_CEILING_MS` and the p95 to
    :data:`ADMISSION_P95_CEILING_MS`; BOTH are failures, because that is what the
    branch that measured them asserted. See those constants for why there are two.
    """
    channel, _warmth = cell
    if channel not in ADMISSION_CELLS:
        return None
    if concurrency not in ASSERTED_CONCURRENCY:
        return None
    if provider_kind != ENFORCED_PROVIDER:
        return None
    stats_for_metric = stats.get(ADMITTED) or {}
    sample_count = int(stats_for_metric.get("n") or 0)
    if not sample_count:
        if errors:
            return Verdict(
                cell,
                "FAILED",
                f"every one of {errors} acknowledgement sample(s) died instead of "
                "being measured",
            )
        return Verdict(
            cell,
            "NO-DATA",
            "no acknowledgement samples: the frame was never observed. On the desktop "
            "channel that means either the daemon does not emit it or the reader never "
            "matched this submit's request id — the gate has nothing to judge (exit 3)",
        )
    if load_per_cpu is not None and load_per_cpu > BUDGET_LOAD_PER_CPU_MAX:
        return Verdict(
            cell,
            "OUT-OF-BAND",
            f"acknowledgement p50 {float(stats_for_metric['p50']):.0f} ms measured at "
            f"load {load_per_cpu:.1f}/CPU, outside the {BUDGET_LOAD_PER_CPU_MAX:.1f}/CPU "
            "band: not judged (exit 3)",
        )
    median = float(stats_for_metric["p50"])
    p95 = float(stats_for_metric.get("p95", UNAVAILABLE))
    if median >= ADMISSION_CEILING_MS:
        return Verdict(
            cell,
            "FAIL",
            f"acknowledgement p50 {median:.0f} ms >= {ADMISSION_CEILING_MS:.0f} ms over "
            f"{sample_count} samples",
        )
    if p95 != UNAVAILABLE and p95 >= ADMISSION_P95_CEILING_MS:
        return Verdict(
            cell,
            "FAIL",
            f"acknowledgement p95 {p95:.0f} ms >= {ADMISSION_P95_CEILING_MS:.0f} ms "
            f"(tail bound) over {sample_count} samples; p50 {median:.0f} ms",
        )
    return Verdict(
        cell,
        "PASS",
        f"acknowledgement p50 {median:.0f} ms < {ADMISSION_CEILING_MS:.0f} ms and p95 "
        f"{p95:.0f} ms < {ADMISSION_P95_CEILING_MS:.0f} ms over {sample_count}",
    )
