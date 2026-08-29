#!/usr/bin/env python3
"""Does the compaction advisor spend FEWER tokens end to end, or more?

``bench_advisor_replay.py`` answers "where does the cut land" (severance). It
deliberately does not answer the money question, and its ``replay Mtok`` column
must not be read as one: it counts context tokens re-sent per turn as if every
token cost the same. They do not. On the session that motivated this feature
92.9% of Anthropic prompt-side tokens were cache READS at 0.5 $/Mtok, while a
cache WRITE costs 6.25 $/Mtok — **12.5x more**. Any claim of the form "the
advisor keeps the context smaller, therefore it is cheaper" is unsound until
the re-write is priced, because a compaction pass INVALIDATES the prefix and
forces the next turn to re-write nearly all of what survived.

That invalidation is measured here, not assumed. Reading the reference
session's own provider counters at the turn immediately after each of its
compaction passes:

    pass    before      after     read     write   write%
    1050   440,417    117,946   22,224    95,722   81.2%
    2123   596,569    149,998   22,294   127,704   85.1%
    3998   511,300    157,832   23,105   134,725   85.4%
    4660   600,031    157,562   23,105   134,455   85.3%
    5439   600,050    161,098   23,904   137,192   85.2%
    7979   600,035    130,702   24,448   106,252   81.3%
    8982   586,520    127,835   24,715   103,118   80.7%

So a pass leaves ~25.6% of the context standing and re-writes ~83.4% of it at
write price; only a ~23.4k head (system blocks + tool schemas, which sit ahead
of the cut) stays warm. Those three constants drive the model below and are
derived from the transcript at runtime rather than hardcoded, so the numbers
move with the evidence instead of with an assumption.

WHAT IS COUNTED. Three terms, all of them, because dropping any one of them
produces the flattering answer:

1. **Turn prompt tokens**, split read / write / fresh. Every turn re-sends the
   whole context; the split is what makes it cheap or expensive.
2. **The advisor's own calls.** Extra provider calls that would not otherwise
   happen. Priced at the MEASURED shape (96.1% cache read, ~568 tokens of
   appended prompt written, ~500 output) from
   ``docs/evidence/compaction-advisor/cache-measurement.txt``.
3. **The compaction passes the advisor causes to happen EARLIER and MORE
   OFTEN.** Each one pays a summarisation call (``context-full``) or zero LLM
   cost (``snapcompact``, which rasterises locally) PLUS the cache re-write in
   term 1.

THE SNAPCOMPACT CAVEAT, and it is the single biggest lever in the whole model:
all nine passes in the reference session were ``snapcompact`` (their
``preserve_data`` carries a ``snapcompact`` archive and their summary text is
the 146-token reading-instructions digest, not a model-written summary). Under
``strategy: auto`` a vision model gets snapcompact, so the marginal LLM cost of
an extra pass is ZERO and only the cache re-write is paid. Under
``context-full`` an extra pass also buys a full summarisation call over the
discarded span. Both are reported; they disagree, and which one a user is in
decides whether the feature pays.

The advisor's decision is simulated, not guessed at: a pass fires early when
the replay is at a genuine task boundary (a non-continuation user turn) inside
the advisor's operating band. ``--advisor-accuracy`` degrades that oracle so
the result is not quoted for a perfect advisor only; 1.0 is an UPPER BOUND on
the benefit and is labelled as such.

Run:
    .venv/bin/python scripts/bench_advisor_tokens.py <transcript.jsonl>
    .venv/bin/python scripts/bench_advisor_tokens.py <transcript.jsonl> --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from bench_advisor_replay import (  # noqa: E402
    Turn,
    _cut_index,
    _iter_records,
    _last_user_before,
    _task_span,
    load_turns,
)

from local_operator.model.registry import anthropic_models  # noqa: E402

#: Priced on ``claude-opus-5`` from the shipped registry rather than a literal,
#: so a price revision reaches this benchmark. The reference session ran mostly
#: on this family (claude-opus-4-8 / claude-opus-5, identical pricing).
_MODEL_ID = "claude-opus-5"

#: Advisor request shape, MEASURED live (arm 2 of measure_advisor_cache.py):
#: cache_read=14024 cache_write=568 output=569 on a 14.6k context, i.e. 96.1%
#: of the prompt read from the turn's warm prefix. Only the appended question
#: is written, and that cost does not scale with the context — which is the
#: whole reason the advisor is affordable at 600k.
ADVISOR_APPENDED_TOKENS = 568
ADVISOR_OUTPUT_TOKENS = 569

#: Fallback constants, used only when a transcript carries no measurable pass.
#: The real run derives all three from the transcript's own counters.
FALLBACK_WARM_CARRY = 23_400
FALLBACK_RESIDUAL_FRACTION = 0.256

#: A compaction summary's own size. Measured at 146 tokens for the nine
#: snapcompact passes here (a reading-instructions digest); a context-full
#: summary is capped at MAX_SUMMARY_TOKENS and runs 4-8k in practice.
SNAPCOMPACT_SUMMARY_TOKENS = 146
CONTEXT_FULL_SUMMARY_TOKENS = 6_000


@dataclass(frozen=True)
class Prices:
    """Per-million-token rates for one model, in dollars."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    @classmethod
    def for_model(cls, model_id: str = _MODEL_ID) -> "Prices":
        info = anthropic_models[model_id]
        # calculate_cost's own fallback rule: an absent cache rate is the base
        # input rate, never zero -- those tokens were billed at something.
        return cls(
            input=info.input_price,
            output=info.output_price,
            cache_read=info.cache_reads_price or info.input_price,
            cache_write=info.cache_writes_price or info.input_price,
        )

    def cost(self, *, read: int, write: int, fresh: int, output: int) -> float:
        return (
            read * self.cache_read
            + write * self.cache_write
            + fresh * self.input
            + output * self.output
        ) / 1e6


@dataclass
class Ledger:
    """Token and dollar accumulation for one simulated arm."""

    read: int = 0
    write: int = 0
    fresh: int = 0
    output: int = 0
    #: Same buckets, attributed to the advisor's own calls.
    advisor_read: int = 0
    advisor_write: int = 0
    advisor_output: int = 0
    #: ...and to compaction summarisation calls.
    summary_fresh: int = 0
    summary_output: int = 0

    passes: int = 0
    advisor_calls: int = 0
    advisor_passes: int = 0
    severed: int = 0
    peak_context: int = 0

    def turn_cost(self, prices: Prices) -> float:
        return prices.cost(read=self.read, write=self.write, fresh=self.fresh, output=self.output)

    def advisor_cost(self, prices: Prices) -> float:
        return prices.cost(
            read=self.advisor_read,
            write=self.advisor_write,
            fresh=0,
            output=self.advisor_output,
        )

    def summary_cost(self, prices: Prices) -> float:
        return prices.cost(read=0, write=0, fresh=self.summary_fresh, output=self.summary_output)

    def total_cost(self, prices: Prices) -> float:
        return self.turn_cost(prices) + self.advisor_cost(prices) + self.summary_cost(prices)

    @property
    def prompt_tokens(self) -> int:
        """Every prompt-side token, advisor and summariser included."""
        return (
            self.read
            + self.write
            + self.fresh
            + self.advisor_read
            + self.advisor_write
            + self.summary_fresh
        )


@dataclass(frozen=True)
class Calibration:
    """Cache behaviour measured from the transcript's own provider counters."""

    warm_carry_tokens: int
    residual_fraction: float
    write_fraction: float
    samples: int
    source: str

    def describe(self) -> str:
        return (
            f"calibration: warm_carry={self.warm_carry_tokens:,} tok  "
            f"residual={self.residual_fraction * 100:.1f}% of pre-pass  "
            f"re-write={self.write_fraction * 100:.1f}% of post-pass  "
            f"(n={self.samples}, {self.source})"
        )


def _normalized_usage(payload: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """``(fresh, output, cache_read, cache_write)`` for one usage record.

    Providers disagree about whether ``input_tokens`` already contains the
    cached tokens; ``configure._cache_tokens_are_inside_input`` is the
    authority and is consulted rather than re-implemented, because charging an
    OpenAI-shaped turn for ``input + cache_read`` double-counts the prefix at
    11x its real rate.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    from local_operator.model.configure import _cache_tokens_are_inside_input

    provider = str(usage.get("provider") or "")
    read = int(usage.get("cache_read_tokens") or 0)
    write = int(usage.get("cache_write_tokens") or 0)
    fresh = int(usage.get("input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    if not provider:
        return None
    try:
        if _cache_tokens_are_inside_input(provider):
            fresh = max(0, fresh - read - write)
    except Exception:  # noqa: BLE001 -- an unknown provider is not a reason to die
        fresh = max(0, fresh - read - write)
    return fresh, output, read, write


def calibrate(path: Path) -> Calibration:
    """Derive the post-compaction cache model from REAL provider counters.

    A compaction pass is the one legitimate cache-invalidating event, and its
    cost is the crux of this whole benchmark, so it is measured on the
    transcript rather than assumed: for each pass, find the first turn after it
    that reported usage, and read how much that turn re-read versus re-wrote.
    Only Anthropic-wire passes carry a ``cache_write`` counter at all, so the
    sample is those; OpenAI-shaped rows report cached tokens without a separate
    write bucket and cannot calibrate a write fraction.
    """
    records = [r for r in _iter_records(path)]
    warm: list[int] = []
    residual: list[float] = []
    writef: list[float] = []
    for index, record in enumerate(records):
        if record.get("type") != "compaction":
            continue
        before = record.get("payload", {}).get("tokens_before")
        if not isinstance(before, int) or before <= 0:
            continue
        for follow in records[index + 1 : index + 40]:
            payload = follow.get("payload")
            if not isinstance(payload, dict):
                continue
            parsed = _normalized_usage(payload)
            if parsed is None:
                continue
            fresh, _out, read, write = parsed
            total = fresh + read + write
            # A pass with no write bucket is an OpenAI-shaped row; it cannot
            # tell us what fraction was re-written, so it is skipped rather
            # than folded in as a zero (which would understate the penalty).
            if total <= 0 or write <= 0:
                break
            warm.append(read)
            residual.append(total / before)
            writef.append(write / total)
            break
    if not warm:
        return Calibration(
            warm_carry_tokens=FALLBACK_WARM_CARRY,
            residual_fraction=FALLBACK_RESIDUAL_FRACTION,
            write_fraction=0.834,
            samples=0,
            source="defaults (no measurable pass in this transcript)",
        )
    return Calibration(
        warm_carry_tokens=int(statistics.mean(warm)),
        residual_fraction=statistics.mean(residual),
        write_fraction=statistics.mean(writef),
        samples=len(warm),
        source="measured from this transcript's post-pass turns",
    )


@dataclass
class Config:
    """One point in the swept configuration space."""

    advisor_enabled: bool
    trigger_tokens: int = 600_000
    keep_recent_tokens: int = 20_000
    advisor_every_n_turns: int = 20
    advisor_floor_tokens: int = 200_000
    advisor_trigger_tokens: int = 300_000
    advisor_cooldown_turns: int = 60
    advisor_max_calls: int = 200
    task_aware: bool = True
    #: Fraction of genuine task boundaries the advisor correctly identifies.
    #: 1.0 is a PERFECT advisor and therefore an upper bound on the benefit.
    accuracy: float = 1.0
    #: ``snapcompact`` passes cost no LLM call; ``context-full`` pays a
    #: summarisation call per pass over the discarded span.
    strategy: str = "snapcompact"
    #: Fraction of turns whose prefix is still warm at the provider. Anthropic's
    #: cache TTL is minutes, so a session with long human gaps re-writes a
    #: prefix nobody compacted. 1.0 is the measured steady state (92.9% reads).
    cache_hit_rate: float = 1.0

    def label(self) -> str:
        if not self.advisor_enabled:
            return f"OFF  trig={self.trigger_tokens // 1000}k"
        return (
            f"ON   trig={self.trigger_tokens // 1000}k n={self.advisor_every_n_turns} "
            f"floor={self.advisor_floor_tokens // 1000}k"
        )


def _deterministic_miss(index: int, accuracy: float) -> bool:
    """Whether probe ``index`` is a MISS at ``accuracy``, without an RNG.

    A seeded RNG would make the sweep irreproducible under a different Python;
    a fixed stride is deterministic, spreads misses evenly across the session,
    and is honest about being a model rather than a sample.
    """
    if accuracy >= 1.0:
        return False
    if accuracy <= 0.0:
        return True
    stride = 1.0 / (1.0 - accuracy)
    return int(index % stride) == 0


def simulate(turns: Sequence[Turn], config: Config, cal: Calibration) -> Ledger:
    """Walk the session once under ``config`` and account for every token.

    The per-turn cache split is the whole point:

    - Steady state, prefix warm: the previous context is a cache READ and only
      this turn's own delta is a cache WRITE. This is the 92.9%-read regime the
      real session actually ran in.
    - Immediately after a compaction pass: the prefix diverged at the cut, so
      only the ~23k system+tools head stays warm and the rest of the residual
      is re-written at 12.5x the read rate. THIS is the term a naive
      "smaller context is cheaper" argument omits.
    - Cache expiry (``cache_hit_rate`` < 1): a turn whose prefix aged out pays
      the same full re-write with no compaction involved.
    """
    ledger = Ledger()
    context = 0
    # Tokens the NEXT turn can read warm. After a pass this collapses to the
    # system+tools head, which is what makes an early pass expensive.
    warm = 0
    turns_since_advisor = -(10**9)
    cooldown_until = -1
    probe_index = 0
    cap = max(config.keep_recent_tokens, config.trigger_tokens // 2)

    for i, turn in enumerate(turns):
        context += turn.tokens

        # --- the turn's own request -------------------------------------
        # Deterministic expiry: every Nth turn goes cold when the hit rate is
        # below 1, so a session with human-scale gaps is representable.
        expired = False
        if config.cache_hit_rate < 1.0:
            stride = 1.0 / (1.0 - config.cache_hit_rate)
            expired = stride > 0 and int(i % stride) == 0
        readable = 0 if expired else min(warm, context)
        ledger.read += readable
        ledger.write += context - readable
        ledger.output += turn.tokens if turn.role == "assistant" else 0
        warm = context
        ledger.peak_context = max(ledger.peak_context, context)

        # --- advisor call, off the critical path ------------------------
        advisory_ready = False
        if config.advisor_enabled and context >= config.advisor_trigger_tokens:
            due = i - turns_since_advisor >= config.advisor_every_n_turns
            if due and i >= cooldown_until and ledger.advisor_calls < config.advisor_max_calls:
                turns_since_advisor = i
                ledger.advisor_calls += 1
                # The advisor reads the turn's warm prefix and writes only its
                # own appended question -- measured, and independent of context
                # size. It does NOT disturb ``warm``: the request is
                # append-only, so the next real turn still reads the same
                # prefix. See PART B of the cache-integrity check.
                ledger.advisor_read += context
                ledger.advisor_write += ADVISOR_APPENDED_TOKENS
                ledger.advisor_output += ADVISOR_OUTPUT_TOKENS

                task_start = _last_user_before(turns, i)
                at_boundary = (
                    task_start is not None
                    and turns[task_start].is_user
                    and not turns[task_start].is_continuation
                )
                probe_index += 1
                if at_boundary and not _deterministic_miss(probe_index, config.accuracy):
                    advisory_ready = True

        # --- would a pass fire? -----------------------------------------
        # An advisory may only pull the trigger EARLIER, and never below the
        # floor -- the same one-directional rule the production gate applies.
        effective_trigger = config.trigger_tokens
        advisory_pass = False
        if advisory_ready and context > config.advisor_floor_tokens:
            effective_trigger = max(config.advisor_floor_tokens, config.advisor_trigger_tokens)
            advisory_pass = context > effective_trigger

        if context <= config.trigger_tokens and not advisory_pass:
            continue

        # --- the pass ---------------------------------------------------
        keep = config.keep_recent_tokens
        if config.task_aware:
            keep = max(keep, _task_span(turns, i, cap))
        cut = _cut_index(turns, i, keep)
        task_start = _last_user_before(turns, i)

        ledger.passes += 1
        if advisory_pass:
            ledger.advisor_passes += 1
            cooldown_until = i + config.advisor_cooldown_turns
        if task_start is not None and cut > task_start:
            ledger.severed += 1

        discarded = sum(t.tokens for t in turns[:cut])
        summary_tokens = (
            SNAPCOMPACT_SUMMARY_TOKENS
            if config.strategy == "snapcompact"
            else CONTEXT_FULL_SUMMARY_TOKENS
        )
        if config.strategy != "snapcompact":
            # context-full ships the discarded span to a summariser: a fresh,
            # uncached prompt plus the summary it writes back. snapcompact
            # rasterises locally and makes NO provider call, which is why the
            # strategy dominates whether an extra pass is affordable.
            ledger.summary_fresh += discarded
            ledger.summary_output += summary_tokens

        context = summary_tokens + sum(t.tokens for t in turns[cut : i + 1])
        # The cut diverges the prefix: only the head ahead of it survives.
        warm = min(cal.warm_carry_tokens, context)

    return ledger


def _row(label: str, ledger: Ledger, prices: Prices, baseline: Ledger | None) -> str:
    total = ledger.total_cost(prices)
    delta = ""
    if baseline is not None:
        base = baseline.total_cost(prices)
        if base:
            pct = (total - base) / base * 100
            delta = f"{pct:+7.1f}%"
    return (
        f"{label:<28}{ledger.passes:>7}{ledger.advisor_calls:>9}"
        f"{ledger.severed:>8}{ledger.prompt_tokens / 1e6:>12,.0f}"
        f"{ledger.read / 1e6:>11,.0f}{ledger.write / 1e6:>11,.0f}"
        f"{total:>11,.2f}{delta:>9}"
    )


_HEADER = (
    f"{'config':<28}{'passes':>7}{'adv':>9}{'sever':>8}{'prompt Mtok':>12}"
    f"{'read M':>11}{'write M':>11}{'cost $':>11}{'vs off':>9}"
)


def _table(rows: Sequence[str]) -> str:
    return "\n".join([_HEADER, "-" * len(_HEADER), *rows])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--model", default=_MODEL_ID)
    args = parser.parse_args(argv)

    if not args.transcript.exists():
        print(f"no such transcript: {args.transcript}", file=sys.stderr)
        return 2

    turns = load_turns(args.transcript)
    if not turns:
        print("transcript contained no replayable messages", file=sys.stderr)
        return 2

    prices = Prices.for_model(args.model)
    cal = calibrate(args.transcript)

    print(f"transcript: {args.transcript}")
    print(f"entries: {len(turns):,}   model: {args.model}")
    print(
        f"prices $/Mtok: input={prices.input} output={prices.output} "
        f"cache_read={prices.cache_read} cache_write={prices.cache_write} "
        f"(write/read = {prices.cache_write / prices.cache_read:.1f}x)"
    )
    print(cal.describe())

    results: dict[str, Ledger] = {}

    def run(name: str, config: Config) -> Ledger:
        ledger = simulate(turns, config, cal)
        results[name] = ledger
        return ledger

    # --- 1. headline: default config, both strategies -------------------
    print("\n\n== 1. HEADLINE: shipped defaults (trigger 600k, n=20, floor 200k) ==")
    for strategy in ("snapcompact", "context-full"):
        print(f"\n-- strategy: {strategy} --")
        off = run(f"off/{strategy}", Config(advisor_enabled=False, strategy=strategy))
        on = run(f"on/{strategy}", Config(advisor_enabled=True, strategy=strategy))
        print(
            _table(
                [
                    _row("advisor OFF", off, prices, None),
                    _row("advisor ON", on, prices, off),
                ]
            )
        )
        print(
            f"  advisor calls cost ${on.advisor_cost(prices):,.2f}; "
            f"summarisation ${on.summary_cost(prices):,.2f} "
            f"(off: ${off.summary_cost(prices):,.2f})"
        )

    # --- 2. cadence sweep ----------------------------------------------
    print("\n\n== 2. CADENCE SWEEP (advisor_every_n_turns), snapcompact ==")
    off = results["off/snapcompact"]
    rows = [_row("advisor OFF", off, prices, None)]
    for every in (5, 10, 20, 40, 80):
        led = run(
            f"cadence/{every}",
            Config(advisor_enabled=True, advisor_every_n_turns=every),
        )
        rows.append(_row(f"ON n={every}", led, prices, off))
    print(_table(rows))

    # --- 3. floor sweep -------------------------------------------------
    print("\n\n== 3. FLOOR SWEEP (advisor_floor_tokens), snapcompact ==")
    rows = [_row("advisor OFF", off, prices, None)]
    for floor in (100_000, 150_000, 200_000, 300_000, 400_000):
        led = run(
            f"floor/{floor}",
            Config(advisor_enabled=True, advisor_floor_tokens=floor),
        )
        rows.append(_row(f"ON floor={floor // 1000}k", led, prices, off))
    print(_table(rows))

    # --- 4. trigger sweep ------------------------------------------------
    print("\n\n== 4. TRIGGER SWEEP (values.compaction.threshold_tokens) ==")
    rows = []
    for trigger in (300_000, 400_000, 600_000):
        base = run(
            f"trig/{trigger}/off",
            Config(advisor_enabled=False, trigger_tokens=trigger),
        )
        led = run(
            f"trig/{trigger}/on",
            Config(
                advisor_enabled=True,
                trigger_tokens=trigger,
                advisor_floor_tokens=min(200_000, trigger // 2),
                advisor_trigger_tokens=min(300_000, trigger // 2),
            ),
        )
        rows.append(_row(f"OFF trig={trigger // 1000}k", base, prices, None))
        rows.append(_row(f"ON  trig={trigger // 1000}k", led, prices, base))
    print(_table(rows))

    # --- 5. accuracy sweep ----------------------------------------------
    print("\n\n== 5. ADVISOR ACCURACY (1.0 is a PERFECT advisor: an upper bound) ==")
    rows = [_row("advisor OFF", off, prices, None)]
    for accuracy in (1.0, 0.75, 0.5, 0.25, 0.0):
        led = run(
            f"acc/{accuracy}",
            Config(advisor_enabled=True, accuracy=accuracy),
        )
        rows.append(_row(f"ON acc={accuracy:.2f}", led, prices, off))
    print(_table(rows))

    # --- 6. cache-hit sweep: THE regime that decides it ------------------
    print("\n\n== 6. CACHE HIT RATE (the term that decides the verdict) ==")
    rows = []
    for hit in (1.0, 0.95, 0.9, 0.8, 0.5):
        base = run(
            f"hit/{hit}/off",
            Config(advisor_enabled=False, cache_hit_rate=hit),
        )
        led = run(
            f"hit/{hit}/on",
            Config(advisor_enabled=True, cache_hit_rate=hit),
        )
        rows.append(_row(f"OFF hit={hit:.0%}", base, prices, None))
        rows.append(_row(f"ON  hit={hit:.0%}", led, prices, base))
    print(_table(rows))

    # --- 7. context-full cadence: where it LOSES -------------------------
    print("\n\n== 7. context-full CADENCE (extra passes buy a summarisation call) ==")
    cf_off = results["off/context-full"]
    rows = [_row("advisor OFF", cf_off, prices, None)]
    for every in (5, 10, 20, 40):
        led = run(
            f"cf-cadence/{every}",
            Config(advisor_enabled=True, advisor_every_n_turns=every, strategy="context-full"),
        )
        rows.append(_row(f"ON n={every}", led, prices, cf_off))
    print(_table(rows))

    # --- 8. THE CONFOUND: is this just a lower threshold? ----------------
    # The operator explicitly rejected "lower threshold_tokens" as a
    # substitute, because they want to RETAIN the ability to run to 600k when
    # a task genuinely needs it. So the honest question is not "does ON beat
    # OFF at 600k" (it does, by compacting sooner) but "does ON at 600k beat
    # simply configuring the lower trigger". If a static 300k trigger is
    # cheaper, then the token argument for the advisor collapses and its case
    # has to rest on severance and retained headroom instead. Reporting this
    # is the difference between a benchmark and an advertisement.
    print("\n\n== 8. CONFOUND CHECK: advisor vs simply lowering the trigger ==")
    static_low = results["trig/300000/off"]
    advisor_high = results["on/snapcompact"]
    baseline_high = results["off/snapcompact"]
    print(
        _table(
            [
                _row("OFF trig=600k (baseline)", baseline_high, prices, None),
                _row("OFF trig=300k (static)", static_low, prices, baseline_high),
                _row("ON  trig=600k (advisor)", advisor_high, prices, baseline_high),
            ]
        )
    )
    static_cost = static_low.total_cost(prices)
    advisor_cost = advisor_high.total_cost(prices)
    gap = (advisor_cost - static_cost) / static_cost * 100 if static_cost else 0.0
    print(
        f"\n  advisor costs {gap:+.1f}% vs the static 300k trigger."
        f"\n  Peak context retained: advisor {advisor_high.peak_context:,} tok"
        f" vs static {static_low.peak_context:,} tok."
        "\n  The advisor's token saving is therefore NOT a unique capability:"
        "\n  most of it is available from threshold_tokens alone. What the"
        "\n  advisor adds is that the ceiling STAYS at 600k for the turns that"
        "\n  need it, which a static trigger cannot express."
    )

    # --- 9. WHERE IT DOES NOT PAY: session length ------------------------
    # An advisor call is charged the moment the context crosses
    # ``advisor_trigger_tokens``; the saving only arrives once a pass it
    # caused has been amortised over enough subsequent turns. A session that
    # ends shortly after crossing that line pays for advice it never banks.
    # This arm walks prefixes of the real session to find the crossover.
    print("\n\n== 9. SESSION LENGTH: where the advisor has not yet repaid ==")
    rows = []
    for fraction in (0.15, 0.25, 0.35, 0.5, 0.75, 1.0):
        prefix = turns[: max(1, int(len(turns) * fraction))]
        base = simulate(prefix, Config(advisor_enabled=False), cal)
        led = simulate(prefix, Config(advisor_enabled=True), cal)
        base_cost = base.total_cost(prices)
        led_cost = led.total_cost(prices)
        pct = (led_cost - base_cost) / base_cost * 100 if base_cost else 0.0
        verdict = "LOSS" if pct > 0 else "win"
        rows.append(
            f"{fraction:>7.0%}{len(prefix):>9,}{base.passes:>8}{led.passes:>7}"
            f"{led.advisor_calls:>7}{base_cost:>12,.2f}{led_cost:>12,.2f}"
            f"{pct:>9.1f}%  {verdict}"
        )
    header = (
        f"{'of run':>7}{'turns':>9}{'p.off':>8}{'p.on':>7}{'adv':>7}"
        f"{'off $':>12}{'on $':>12}{'delta':>10}"
    )
    print("\n".join([header, "-" * len(header), *rows]))

    if args.json:
        print(
            "\n"
            + json.dumps(
                {
                    "entries": len(turns),
                    "model": args.model,
                    "prices": asdict(prices),
                    "calibration": asdict(cal),
                    "results": {
                        name: {
                            **asdict(led),
                            "cost_total": led.total_cost(prices),
                            "cost_turns": led.turn_cost(prices),
                            "cost_advisor": led.advisor_cost(prices),
                            "cost_summary": led.summary_cost(prices),
                            "prompt_tokens": led.prompt_tokens,
                        }
                        for name, led in results.items()
                    },
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
