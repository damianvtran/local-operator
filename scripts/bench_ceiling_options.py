#!/usr/bin/env python3
"""Should we ship the advisor, a lower ceiling, both, or nothing?

``bench_advisor_tokens.py`` answered "is the advisor cheaper than leaving the
ceiling alone" (yes, -41%) and then found the answer that matters more: a
STATIC 300k trigger is cheaper still, and needs no new code, no beta flag and
no provider calls. That turns the question from "does the advisor work" into
"which option should ship, including the option of shipping nothing".

This script benchmarks the whole option space on one axis set:

    A. do nothing            -- the shipped 600k ceiling, advisor off
    B. static lower ceiling  -- threshold_tokens at 300k/400k/500k, advisor off
    C. advisor on            -- shipped defaults and best configuration
    D. combinations          -- advisor on top of a lower ceiling

reusing the cache-aware accounting from ``bench_advisor_tokens.py`` (read /
write / fresh split, the advisor's own calls, the extra summarisation calls)
rather than re-deriving it, so the two benchmarks cannot disagree.

THE NON-TOKEN AXES ARE NOT DECORATION. The operator's stated requirement is
that capacity to 600k stays available when a task genuinely needs it, so a
cheaper option that silently caps the product is not automatically better.
Three consequences are reported per option:

- **peak context** actually reached;
- **severance**, the passes whose cut landed inside the task in flight;
- **headroom**, the margin between the ceiling and the largest single task
  this session ran. That last one is the capability cost of a static ceiling,
  and it is measured rather than asserted: a ceiling below a task's own span
  cannot hold that task whole no matter how the cut is chosen.

The distinction the headroom number makes, and which a raw cost column hides:
a task's OWN span (its user turn through the end of its work) is bounded here
at 271k, while the LIVE context a task runs against -- its span plus whatever
earlier history is still resident -- routinely passes 500k. A ceiling truncates
the second, not the first. So a low ceiling does not sever tasks mid-flight; it
shortens the cross-task memory the session carries, and the margin above the
largest single task is what is left before it starts doing the former too.

Run:
    .venv/bin/python scripts/bench_ceiling_options.py <transcript.jsonl>
    .venv/bin/python scripts/bench_ceiling_options.py <transcript.jsonl> --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from bench_advisor_replay import Turn, load_turns  # noqa: E402
from bench_advisor_tokens import (  # noqa: E402
    Calibration,
    Config,
    Ledger,
    Prices,
    calibrate,
    simulate,
)

#: Ceilings swept for the static option. 600k is what ships today; the lower
#: three are the "just lower the default" alternatives the operator named.
CEILINGS = (300_000, 400_000, 500_000, 600_000)


@dataclass(frozen=True)
class TaskShape:
    """What this session's tasks actually demanded, independent of any ceiling.

    ``own_spans`` is each task's own token weight (a genuine, non-continuation
    user turn through the turn before the next one). ``live_peaks`` is the
    context each task actually ran against under the shipped 600k ceiling.
    The two answer different questions and are kept apart deliberately.
    """

    own_spans: tuple[int, ...]
    live_peaks: tuple[int, ...]

    @property
    def largest_task(self) -> int:
        return max(self.own_spans) if self.own_spans else 0

    def tasks_over(self, ceiling: int) -> int:
        """Tasks whose LIVE context passed ``ceiling`` -- i.e. how often this
        session would have been cut short by that ceiling."""
        return sum(1 for peak in self.live_peaks if peak > ceiling)


def measure_tasks(turns: Sequence[Turn], *, keep_recent: int, summary: int) -> TaskShape:
    """Task spans and the live context each task reached under 600k.

    A task starts at a genuine user turn that is NOT a continuation: a
    "Continue" is the same task resuming, and counting it as a new one would
    understate how much context a real request accumulates.
    """
    starts = [i for i, t in enumerate(turns) if t.is_user and not t.is_continuation]
    own: list[int] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(turns)
        own.append(sum(t.tokens for t in turns[start:end]))

    # Live peaks are replayed under the SHIPPED ceiling, because that is the
    # world whose demand we are measuring; under a lower ceiling the peaks are
    # clipped by construction and would prove only that the clip happened.
    context = 0
    current: int | None = None
    peaks: dict[int, int] = {}
    for i, turn in enumerate(turns):
        if turn.is_user and not turn.is_continuation:
            current = i
        context += turn.tokens
        if current is not None:
            peaks[current] = max(peaks.get(current, 0), context)
        if context > 600_000:
            accumulated = 0
            cut = 0
            for j in range(i, -1, -1):
                accumulated += turns[j].tokens
                if accumulated >= keep_recent:
                    cut = j
                    break
            context = summary + sum(t.tokens for t in turns[cut : i + 1])
    return TaskShape(own_spans=tuple(own), live_peaks=tuple(peaks.values()))


@dataclass
class Option:
    """One shippable option and the evidence for or against it."""

    name: str
    ledger: Ledger
    config: Config
    cost: float
    #: True when this option needs code that does not exist in a release today.
    needs_new_code: bool

    def headroom(self, shape: TaskShape) -> int:
        return self.config.trigger_tokens - shape.largest_task


_HEADER = (
    f"{'option':<30}{'passes':>7}{'adv':>6}{'sever':>7}{'peak ctx':>11}"
    f"{'cost $':>11}{'vs base':>9}{'headroom':>10}{'new code':>10}"
)


def _row(option: Option, baseline: float, shape: TaskShape) -> str:
    delta = (option.cost - baseline) / baseline * 100 if baseline else 0.0
    return (
        f"{option.name:<30}{option.ledger.passes:>7}{option.ledger.advisor_calls:>6}"
        f"{option.ledger.severed:>7}{option.ledger.peak_context:>11,}"
        f"{option.cost:>11,.2f}{delta:>8.1f}%"
        f"{option.headroom(shape):>10,}{'yes' if option.needs_new_code else 'no':>10}"
    )


def _table(options: Sequence[Option], baseline: float, shape: TaskShape) -> str:
    return "\n".join([_HEADER, "-" * len(_HEADER), *(_row(o, baseline, shape) for o in options)])


def build_options(
    turns: Sequence[Turn], cal: Calibration, prices: Prices, strategy: str
) -> list[Option]:
    """Every option worth reporting, in the order a decision would consider them."""
    options: list[Option] = []

    def add(name: str, config: Config, *, new_code: bool) -> Option:
        ledger = simulate(turns, config, cal)
        option = Option(
            name=name,
            ledger=ledger,
            config=config,
            cost=ledger.total_cost(prices),
            needs_new_code=new_code,
        )
        options.append(option)
        return option

    # A. do nothing -- the baseline everything must beat. ``task_aware=False``
    # is deliberate and load-bearing: the task-aware preserve window is part of
    # THIS PR, so a baseline carrying it would not be "do nothing", and every
    # delta below would be measured against a product that does not exist yet.
    add(
        "A. do nothing (600k, today)",
        Config(
            advisor_enabled=False,
            trigger_tokens=600_000,
            task_aware=False,
            strategy=strategy,
        ),
        new_code=False,
    )
    # B. static lower ceilings -- no new code, no beta flag, no provider calls.
    # Also on the recency cut, since lowering a config value is all a user does.
    for ceiling in CEILINGS:
        if ceiling == 600_000:
            continue
        add(
            f"B. static ceiling {ceiling // 1000}k",
            Config(
                advisor_enabled=False,
                trigger_tokens=ceiling,
                task_aware=False,
                strategy=strategy,
            ),
            new_code=False,
        )
    # B'. the task-aware preserve window on its own: unconditional code from
    #     this PR, no beta flag, no extra provider call. This is the arm that
    #     turns out to fix severance, which is what the PR set out to do.
    add(
        "B'. task-aware cut only (600k)",
        Config(
            advisor_enabled=False,
            trigger_tokens=600_000,
            task_aware=True,
            strategy=strategy,
        ),
        new_code=True,
    )
    # C. advisor at the shipped defaults, and at its best configuration
    # (cadence 10 was the cheapest point in the earlier cadence sweep).
    add(
        "C. advisor (defaults, 600k)",
        Config(advisor_enabled=True, trigger_tokens=600_000, strategy=strategy),
        new_code=True,
    )
    add(
        "C. advisor (best cfg, 600k)",
        Config(
            advisor_enabled=True,
            trigger_tokens=600_000,
            advisor_every_n_turns=10,
            strategy=strategy,
        ),
        new_code=True,
    )
    # D. combinations -- the advisor riding a lower ceiling.
    for ceiling in (400_000, 500_000):
        add(
            f"D. advisor + {ceiling // 1000}k ceiling",
            Config(
                advisor_enabled=True,
                trigger_tokens=ceiling,
                advisor_floor_tokens=min(200_000, ceiling // 2),
                advisor_trigger_tokens=min(300_000, ceiling // 2),
                strategy=strategy,
            ),
            new_code=True,
        )
    return options


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.transcript.exists():
        print(f"no such transcript: {args.transcript}", file=sys.stderr)
        return 2

    turns = load_turns(args.transcript)
    if not turns:
        print("transcript contained no replayable messages", file=sys.stderr)
        return 2

    prices = Prices.for_model()
    cal = calibrate(args.transcript)
    shape = measure_tasks(turns, keep_recent=20_000, summary=146)

    print(f"transcript: {args.transcript}")
    print(f"entries: {len(turns):,}")
    print(
        f"prices $/Mtok: cache_read={prices.cache_read} cache_write={prices.cache_write} "
        f"(write/read = {prices.cache_write / prices.cache_read:.1f}x)"
    )
    print(cal.describe())

    print("\n\n== WHAT THIS SESSION'S TASKS ACTUALLY DEMANDED ==")
    own = sorted(shape.own_spans)
    live = sorted(shape.live_peaks)
    print(f"tasks: {len(own)} (genuine, non-continuation user turns)")
    print(
        f"  task's OWN span:   p50 {statistics.median(own):>9,.0f}  "
        f"p90 {own[int(0.9 * len(own))]:>9,.0f}  max {max(own):>9,.0f}"
    )
    print(
        f"  LIVE context run:  p50 {statistics.median(live):>9,.0f}  "
        f"p90 {live[int(0.9 * len(live))]:>9,.0f}  max {max(live):>9,.0f}"
    )
    print(
        "\nA ceiling truncates the LIVE context, not the task's own span. How often\n"
        "this session would have been cut short by each candidate ceiling:"
    )
    for ceiling in CEILINGS:
        over = shape.tasks_over(ceiling)
        print(
            f"  ceiling {ceiling:>7,}: {over:>2}/{len(live)} tasks "
            f"({over / len(live) * 100:>4.1f}%) ran past it   "
            f"headroom above the largest single task: {ceiling - shape.largest_task:>8,}"
        )
    print(
        f"\nLargest single task: {shape.largest_task:,} tokens. A ceiling below that\n"
        "cannot hold that task whole regardless of cut policy."
    )

    results: dict[str, list[Option]] = {}
    for strategy in ("snapcompact", "context-full"):
        print(f"\n\n== OPTION SPACE: {strategy} ==")
        options = build_options(turns, cal, prices, strategy)
        results[strategy] = options
        baseline = options[0].cost
        print(_table(options, baseline, shape))
        cheapest = min(options, key=lambda o: o.cost)
        print(f"\n  cheapest: {cheapest.name} at ${cheapest.cost:,.2f}")
        advisor_best = min((o for o in options if o.config.advisor_enabled), key=lambda o: o.cost)
        static_best = min(
            (o for o in options if not o.config.advisor_enabled), key=lambda o: o.cost
        )
        gap = (advisor_best.cost - static_best.cost) / static_best.cost * 100
        print(
            f"  best advisor option ({advisor_best.name}) vs best no-code option "
            f"({static_best.name}): {gap:+.1f}%"
        )

    # --- the decision table: advisor on/off at MATCHED capability ---------
    #
    # The earlier benchmark reported "a static 300k ceiling beats the advisor
    # by 3.2%" and that comparison was not apples to apples: those two options
    # differ in headroom by 300k, so it priced a cheaper product against a more
    # capable one and credited the difference to the advisor's absence.
    #
    # Holding the ceiling FIXED is the comparison that answers "does the
    # advisor earn its place", because both arms then retain exactly the same
    # capacity and differ only by the feature under test.
    print("\n\n== DECISION TABLE: advisor on/off at MATCHED ceiling ==")
    print("(same ceiling both sides, so capability is held constant)\n")
    header = (
        f"{'strategy':<14}{'ceiling':>9}{'off $':>11}{'on $':>11}"
        f"{'advisor delta':>15}{'sever off':>11}{'sever on':>10}"
    )
    print(header)
    print("-" * len(header))
    matched: dict[str, list[dict[str, float]]] = {}
    for strategy in ("snapcompact", "context-full"):
        rows: list[dict[str, float]] = []
        for ceiling in CEILINGS:
            off = simulate(
                turns,
                Config(advisor_enabled=False, trigger_tokens=ceiling, strategy=strategy),
                cal,
            )
            on = simulate(
                turns,
                Config(
                    advisor_enabled=True,
                    trigger_tokens=ceiling,
                    advisor_floor_tokens=min(200_000, ceiling // 2),
                    advisor_trigger_tokens=min(300_000, ceiling // 2),
                    strategy=strategy,
                ),
                cal,
            )
            off_cost, on_cost = off.total_cost(prices), on.total_cost(prices)
            delta = (on_cost - off_cost) / off_cost * 100 if off_cost else 0.0
            print(
                f"{strategy:<14}{ceiling:>9,}{off_cost:>11,.2f}{on_cost:>11,.2f}"
                f"{delta:>14.1f}%{off.severed:>11}{on.severed:>10}"
            )
            rows.append(
                {
                    "ceiling": ceiling,
                    "off": off_cost,
                    "on": on_cost,
                    "delta_pct": delta,
                    "severed_off": off.severed,
                    "severed_on": on.severed,
                }
            )
        matched[strategy] = rows
    print(
        "\nRead this table, not the raw cheapest-option row: it is the only one\n"
        "where both sides retain the same capacity, so the delta column is the\n"
        "advisor's own contribution rather than the cost of a smaller product."
    )

    # --- what is actually IN this PR, separated -----------------------------
    #
    # The PR bundles two independent changes and the option table above cannot
    # tell them apart:
    #
    #   1. a TASK-AWARE preserve window (keep max(keep_recent, task span)),
    #      which is unconditional code on the normal compaction path;
    #   2. the speculative ADVISOR, behind the beta flag.
    #
    # Severance is the problem the PR set out to solve, so the question that
    # decides what ships is which of the two actually solves it. Separating
    # them is the difference between shipping a flag-gated provider-call
    # feature and shipping a cut-policy change.
    print("\n\n== WHAT FIXES SEVERANCE: task-aware cut vs the advisor ==")
    print("(snapcompact; severance is passes whose cut landed inside a live task)\n")
    header = (
        f"{'ceiling':>9}{'recency sever':>15}{'task-aware sever':>18}"
        f"{'recency $':>12}{'task-aware $':>14}{'cost of the fix':>17}"
    )
    print(header)
    print("-" * len(header))
    decomposition: list[dict[str, float]] = []
    for ceiling in CEILINGS:
        recency = simulate(
            turns,
            Config(advisor_enabled=False, trigger_tokens=ceiling, task_aware=False),
            cal,
        )
        aware = simulate(
            turns,
            Config(advisor_enabled=False, trigger_tokens=ceiling, task_aware=True),
            cal,
        )
        rc, ac = recency.total_cost(prices), aware.total_cost(prices)
        pct = (ac - rc) / rc * 100 if rc else 0.0
        print(
            f"{ceiling:>9,}{recency.severed:>15}{aware.severed:>18}"
            f"{rc:>12,.2f}{ac:>14,.2f}{pct:>16.1f}%"
        )
        decomposition.append(
            {
                "ceiling": ceiling,
                "severed_recency": recency.severed,
                "severed_task_aware": aware.severed,
                "cost_recency": rc,
                "cost_task_aware": ac,
                "cost_pct": pct,
            }
        )
    print(
        "\nThe task-aware preserve window alone takes severance to zero at every\n"
        "ceiling, for a few percent. It needs no beta flag and makes no extra\n"
        "provider call, so the advisor cannot claim the severance win as its own."
    )

    if args.json:
        print(
            "\n"
            + json.dumps(
                {
                    "entries": len(turns),
                    "calibration": asdict(cal),
                    "largest_task_tokens": shape.largest_task,
                    "matched_ceiling": matched,
                    "severance_decomposition": decomposition,
                    "task_own_spans": list(shape.own_spans),
                    "task_live_peaks": list(shape.live_peaks),
                    "options": {
                        strategy: [
                            {
                                "name": o.name,
                                "cost": o.cost,
                                "passes": o.ledger.passes,
                                "advisor_calls": o.ledger.advisor_calls,
                                "severed": o.ledger.severed,
                                "peak_context": o.ledger.peak_context,
                                "trigger_tokens": o.config.trigger_tokens,
                                "needs_new_code": o.needs_new_code,
                                "headroom": o.headroom(shape),
                            }
                            for o in options
                        ]
                        for strategy, options in results.items()
                    },
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
