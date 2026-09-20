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

:data:`FIRST_EVENT` is the headline: submit -> the first event the FRONT END can
paint for this turn. It is the only metric with a budget, and the budget is 300 ms
because that is the achievable contract the design doc settled on.

:data:`FIRST_REASONING` and :data:`FIRST_TEXT` split :data:`FIRST_EVENT` by what
the content IS, which is what makes the operator's complaint legible: the model
reasoning for 0.45-1.75 s while the front end shows nothing, because the runtime
drops ``StreamReasoningDelta`` (``harness/loop.py``) before any consumer sees it.

:data:`PROVIDER_REASONING` / :data:`PROVIDER_TEXT` are the PROVIDER's own clock —
when the model put the token on the wire — and they are REPORTED, NEVER ASSERTED.
See :data:`BUDGET_MS` for why.

:data:`RUNTIME_EMIT` is the runtime-side (or daemon-side) emission of the event
:data:`FIRST_EVENT` measured, so transport cost separates from runtime cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------------

#: Submit -> the first event the front end can PAINT for this turn. The headline.
FIRST_EVENT = "first_event_ms"
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
    FIRST_EVENT,
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
TABLE_METRICS: tuple[str, ...] = (FIRST_EVENT, FIRST_REASONING, FIRST_TEXT)

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
        return self.status == "FAIL"


def judge(
    cell: tuple[str, str],
    stats: Mapping[str, Mapping[str, Any]],
    *,
    concurrency: int = 1,
) -> Verdict:
    """Decide the cell's budget verdict from its reduced :data:`FIRST_EVENT` stats.

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
    """
    channel, warmth = cell
    stats_for_metric = stats.get(FIRST_EVENT)
    if not stats_for_metric or not stats_for_metric.get("n"):
        return Verdict(cell, "REPORTED", f"no {FIRST_EVENT} samples (channel unavailable here)")

    median = float(stats_for_metric["p50"])
    p95 = float(stats_for_metric.get("p95", UNAVAILABLE))
    if cell in ENFORCED_CELLS and concurrency in ASSERTED_CONCURRENCY:
        if median >= BUDGET_MS:
            return Verdict(
                cell,
                "FAIL",
                f"p50 {median:.0f} ms >= budget {BUDGET_MS:.0f} ms over "
                f"{stats_for_metric['n']} samples",
            )
        if p95 != UNAVAILABLE and p95 >= BUDGET_MS:
            return Verdict(
                cell,
                "WARN",
                f"p50 {median:.0f} ms inside budget; p95 {p95:.0f} ms is over it "
                "(host contention; not asserted — see judge())",
            )
        return Verdict(
            cell, "PASS", f"p50 {median:.0f} ms < {BUDGET_MS:.0f} ms over {stats_for_metric['n']}"
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
