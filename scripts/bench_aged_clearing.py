#!/usr/bin/env python3
"""Replay benchmark for aged tool-result clearing.

Replays real session transcripts request by request and prices what each
model call would have sent under two arms:

- ``baseline`` — today's pruning only: :func:`prune_tool_outputs` at every
  user-turn boundary (where the session's post-turn pass runs it).
- ``aged``     — the same, plus the REAL :func:`clear_aged_tool_results` before
  every model call (the session runs it at every tool-loop boundary and after
  each turn).

Both arms replay the transcript's own compaction entries identically (summary
plus the kept window), so the comparison isolates the clearing pass. One
simplification is stated rather than hidden: an arm that sheds context would in
reality reach the compaction trigger LATER than the recorded session did; this
replay compacts where the recording did, which under-states the aged arm's
benefit on sessions that compacted.

Per arm it reports: total context tokens sent; modeled billed cost with cache
reads at 0.1x and uncached/written input at 1.0x and 1.25x (the cached prefix
of a request ends at the first message index changed since the previous
request, so a clear pays the write rate on everything after it); the number
of clear events; and historical later same-key calls (read path, bash command or grep
pattern+scope). The recorded models NEVER saw the stubs: these are historical
repeats, NOT induced re-fetches. Only matched live runs measure the latter.
An additional conservative Anthropic model credits only previously WRITTEN
markers within 20 blocks and charges the configured 1h/2x write tier.
Per-length buckets and per-session deltas expose short-run regressions.

Read-only by construction: point ``--corpus`` at a COPY of the transcripts
(``cp -c`` clones on APFS cost nothing), never at the live session store.

Run:
    .venv/bin/python scripts/bench_aged_clearing.py --corpus "$PAD/corpus/*.jsonl"
    .venv/bin/python scripts/bench_aged_clearing.py --corpus ... --json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.compaction.pruning import (  # noqa: E402
    AGED_CLEAR_MIN_SAVING_RATIO,
    DEFAULT_AGED_CLEAR_AFTER_CALLS,
    DEFAULT_AGED_CLEAR_BATCH_TOKENS,
    DEFAULT_AGED_CLEAR_MIN_TOKENS,
    clear_aged_tool_results,
    prune_tool_outputs,
)
from local_operator.compaction.tokens import (  # noqa: E402
    clear_estimate_cache,
    estimate_tokens,
)
from local_operator.harness.replay_bound import bound_replay_payloads  # noqa: E402
from local_operator.harness.types import Message  # noqa: E402

#: Price multipliers, relative to one uncached input token.
CACHE_READ = 0.1
WRITE_RATES = (1.0, 1.25)

#: Stand-in for the system prompt + tool schemas every request carries. The
#: same constant in both arms; it dilutes the relative saving honestly rather
#: than pretending a request is only its messages.
DEFAULT_SYSTEM_TOKENS = 25_000


@dataclass
class ArmResult:
    requests: int = 0
    context_tokens: int = 0
    cost: dict[str, float] = field(default_factory=lambda: {str(w): 0.0 for w in WRITE_RATES})
    clear_events: int = 0
    cleared_items: int = 0
    cleared_tokens: int = 0
    historical_repeat_items: int = 0
    historical_repeat_tokens: int = 0
    anthropic_cost: float = 0.0


@dataclass
class BreakpointCache:
    """Conservative text-history cache simulator, not arbitrary-prefix reuse.

    Only prefixes a previous request explicitly WROTE can be read. Lookup is
    limited to the 20 content blocks preceding either current marker (tail
    and previous-user; production's text-only marker layout). We use message
    ends as potential hits: a real intra-message block hit is ignored rather
    than invented. Images/re-rendering, provider minimum-cache sizes, wall
    clock expiry, schemas and system-block splits are not reconstructible
    here; the fixed system head is assumed cacheable after its first write.
    These are assumptions, NOT provider-accurate invoice claims. Long-context
    writes cost 2x (1h), smaller writes 1.25x (5m), cache reads 0.1x. The TTL
    choice uses the previous request's size, like the live provider hint.
    """

    written: set[int] = field(default_factory=set)
    system_written: bool = False
    previous_size: int = 0

    def price(self, messages, tokens, system_tokens, threshold) -> float:
        ends, fingerprints, prefix_tokens = [], [], []
        blocks, fingerprint, subtotal = 0, 0, 0
        for message, size in zip(messages, tokens):
            blocks += max(1, len(message.content) + len(message.tool_calls))
            fingerprint = hash(
                (fingerprint, message.id, bool((message.provider_payload or {}).get("pruned")))
            )
            subtotal += size
            ends.append(blocks)
            fingerprints.append(fingerprint)
            prefix_tokens.append(subtotal)
        targets = [len(messages) - 1] if messages else []
        # Anthropic maps tool results to user/tool_result blocks before
        # selecting its previous-user marker (not just typed user messages).
        users = [i for i, m in enumerate(messages) if m.role in ("user", "tool")]
        if len(users) >= 2:
            targets.append(users[-2])
        cached = system_tokens if self.system_written else 0
        for target in targets:
            for index in range(target, -1, -1):
                if ends[target] - ends[index] >= 20:
                    break
                if fingerprints[index] in self.written:
                    cached = max(cached, system_tokens + prefix_tokens[index])
                    break
        total = system_tokens + subtotal
        write_rate = 2.0 if threshold > 0 and self.previous_size >= threshold else 1.25
        cost = cached * CACHE_READ + (total - cached) * write_rate
        for target in targets:
            self.written.add(fingerprints[target])
        self.system_written = True
        self.previous_size = total
        return cost


def distribution(pairs) -> dict[str, Any]:
    """Report every session and length buckets; an aggregate can hide losers."""
    rows = [
        {
            "session_index": index,
            "requests": b.requests,
            "clear_events": a.clear_events,
            "delta": a.anthropic_cost / b.anthropic_cost - 1 if b.anthropic_cost else 0.0,
            "baseline_cost": b.anthropic_cost,
            "aged_cost": a.anthropic_cost,
        }
        for index, (b, a) in enumerate(pairs)
    ]
    buckets = {}
    for label, lo, hi in (
        ("1-40", 1, 40),
        ("41-80", 41, 80),
        ("81-160", 81, 160),
        ("161+", 161, float("inf")),
    ):
        subset = [r for r in rows if lo <= r["requests"] <= hi]
        old = sum(r["baseline_cost"] for r in subset)
        new = sum(r["aged_cost"] for r in subset)
        buckets[label] = {
            "sessions": len(subset),
            "clearing": sum(r["clear_events"] > 0 for r in subset),
            "regressions": sum(r["delta"] > 1e-9 for r in subset),
            "cost_delta": new / old - 1 if old else 0.0,
        }
    deltas = sorted(r["delta"] for r in rows)
    quantiles = {
        name: deltas[min(len(deltas) - 1, int(q * (len(deltas) - 1)))] if deltas else 0
        for name, q in (("min", 0), ("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("max", 1))
    }
    return {
        "buckets": buckets,
        "quantiles": quantiles,
        "regressions": sum(r["delta"] > 1e-9 for r in rows),
        "sessions": rows,
    }


def _refetch_key(name: str | None, arguments: dict[str, Any]) -> str | None:
    """What makes a later call the SAME re-fetch of a cleared result."""
    if name == "read" and isinstance(arguments.get("path"), str):
        return f"read:{arguments['path']}"
    if name == "bash" and isinstance(arguments.get("command"), str):
        return f"bash:{arguments['command'].strip()}"
    if name == "grep" and isinstance(arguments.get("pattern"), str):
        return f"grep:{arguments['pattern']}:{arguments.get('path') or '.'}"
    return None


def _load(path: str) -> list[tuple[str, Any]]:
    """The transcript as ordered events: ('msg', payload-dict) / ('compact', payload)."""
    events: list[tuple[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            kind = entry.get("type")
            payload = entry.get("payload") or {}
            if kind == "compaction":
                events.append(("compact", payload))
            elif kind == "message" and payload.get("kind", "message") == "message":
                if payload.get("role") in ("user", "assistant", "tool"):
                    data = {
                        k: v for k, v in payload.items() if k not in ("kind", "producer_command_id")
                    }
                    data["id"] = entry.get("id")
                    events.append(("msg", data))
    return events


def _replay(events: list[tuple[str, Any]], aged: bool, args: argparse.Namespace) -> ArmResult:
    clear_estimate_cache()  # IDs repeat across arms; their content must not share estimates.
    result = ArmResult()
    cache = BreakpointCache()
    context: list[Message] = []
    previous: list[tuple[str, bool]] = []
    cleared_keys: dict[str, list[int]] = {}  # same-key historical results, not induced re-fetches
    call_keys: dict[str, str] = {}  # tool_call_id -> refetch key

    # Per-arm memo keyed (id, pruned): within one arm a message is blanked at
    # most once (every pass skips an already-pruned result), so the pair names
    # its content exactly. The production estimate cache is an LRU of 4096
    # entries, which a long transcript priced at every request would thrash.
    memo: dict[tuple[str, bool], int] = {}

    def tokens_of(message: Message) -> int:
        key = (message.id, bool((message.provider_payload or {}).get("pruned")))
        if key not in memo:
            memo[key] = estimate_tokens(message)
        return memo[key]

    def prices(request: list[Message]) -> None:
        # Same non-mutating, derived-ID wire bound as AgentLoop._stream_once.
        # Clearing still sees the durable unbounded history, as in Session.
        request = bound_replay_payloads(request)
        tokens = [tokens_of(m) for m in request]
        signature = [(m.id, bool((m.provider_payload or {}).get("pruned"))) for m in request]
        first_change = next(
            (i for i, (a, b) in enumerate(zip(signature, previous)) if a != b),
            min(len(signature), len(previous)),
        )
        total = args.system_tokens + sum(tokens)
        cached = (args.system_tokens + sum(tokens[:first_change])) if previous else 0
        result.requests += 1
        result.context_tokens += total
        for rate in WRITE_RATES:
            result.cost[str(rate)] += cached * CACHE_READ + (total - cached) * rate
        result.anthropic_cost += cache.price(
            request, tokens, args.system_tokens, args.cache_1h_threshold
        )
        previous[:] = signature

    for kind, data in events:
        if kind == "compact":
            keep = data.get("first_kept_entry_id")
            index = next((i for i, m in enumerate(context) if m.id == keep), len(context))
            context = [Message.user(str(data.get("summary", ""))), *context[index:]]
            continue
        try:
            message = Message.model_validate(data)
        except Exception:  # noqa: BLE001 — a malformed row drops alone, as on resume
            continue
        if message.role == "user":
            prune_tool_outputs(context, 0, 0)
            if aged:
                _clear(context, result, args, call_keys, cleared_keys)
        if message.role == "assistant":
            if aged:
                _clear(context, result, args, call_keys, cleared_keys)
            prices(context)
            for call in message.tool_calls:
                key = _refetch_key(call.name, call.arguments)
                if key is None:
                    continue
                call_keys[call.id] = key
                if key in cleared_keys:
                    sizes = cleared_keys.pop(key)
                    result.historical_repeat_items += len(sizes)
                    result.historical_repeat_tokens += sum(sizes)
        context.append(message)
    return result


def _clear(
    context: list[Message],
    result: ArmResult,
    args: argparse.Namespace,
    call_keys: dict[str, str],
    cleared_keys: dict[str, list[int]],
) -> None:
    before = {m.id: estimate_tokens(m) for m in context if m.role == "tool"}
    pruned = {m.id for m in context if (m.provider_payload or {}).get("pruned")}
    _, cleared = clear_aged_tool_results(
        context,
        after_calls=args.after_calls,
        min_tokens=args.min_tokens,
        batch_tokens=args.batch_tokens,
        min_saving_ratio=args.min_saving_ratio,
    )
    if not cleared:
        return
    result.clear_events += 1
    for message in context:
        if message.id in pruned or not (message.provider_payload or {}).get("pruned"):
            continue
        if message.role != "tool" or message.id not in before:
            continue
        freed = before[message.id] - estimate_tokens(message)
        result.cleared_items += 1
        result.cleared_tokens += freed
        key = call_keys.get(message.tool_call_id or "")
        if key is not None:
            cleared_keys.setdefault(key, []).append(freed)


def _one(path: str, args: argparse.Namespace) -> tuple[ArmResult, ArmResult] | None:
    events = _load(path)
    if not any(kind == "msg" and data.get("role") == "assistant" for kind, data in events):
        return None
    return _replay(events, False, args), _replay(events, True, args)


def _sum(results: list[ArmResult]) -> ArmResult:
    total = ArmResult()
    for r in results:
        total.anthropic_cost += r.anthropic_cost
        total.requests += r.requests
        total.context_tokens += r.context_tokens
        for rate in total.cost:
            total.cost[rate] += r.cost[rate]
        total.clear_events += r.clear_events
        total.cleared_items += r.cleared_items
        total.cleared_tokens += r.cleared_tokens
        total.historical_repeat_items += r.historical_repeat_items
        total.historical_repeat_tokens += r.historical_repeat_tokens
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", required=True, help="glob of COPIED transcript.jsonl files")
    parser.add_argument("--after-calls", type=int, default=DEFAULT_AGED_CLEAR_AFTER_CALLS)
    parser.add_argument("--min-saving-ratio", type=float, default=AGED_CLEAR_MIN_SAVING_RATIO)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_AGED_CLEAR_MIN_TOKENS)
    parser.add_argument("--batch-tokens", type=int, default=DEFAULT_AGED_CLEAR_BATCH_TOKENS)
    parser.add_argument("--system-tokens", type=int, default=DEFAULT_SYSTEM_TOKENS)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--cache-1h-threshold", type=int, default=150_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.corpus))
    if not paths:
        parser.error(f"no transcripts match {args.corpus!r}")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pairs = [p for p in pool.map(_one, paths, [args] * len(paths)) if p is not None]
    base = _sum([b for b, _ in pairs])
    aged = _sum([a for _, a in pairs])

    def delta(new: float, old: float) -> float:
        return (new / old - 1.0) if old else 0.0

    report = {
        "transcripts": len(pairs),
        "params": {
            "after_calls": args.after_calls,
            "min_saving_ratio": args.min_saving_ratio,
            "min_tokens": args.min_tokens,
            "batch_tokens": args.batch_tokens,
            "system_tokens": args.system_tokens,
            "cache_1h_threshold": args.cache_1h_threshold,
        },
        "baseline": asdict(base),
        "aged": asdict(aged),
        "anthropic_cost_delta": delta(aged.anthropic_cost, base.anthropic_cost),
        "distribution": distribution(pairs),
        "context_delta": delta(aged.context_tokens, base.context_tokens),
        "cost_delta": {r: delta(aged.cost[r], base.cost[r]) for r in base.cost},
        "historical_repeat_share_items": (
            (aged.historical_repeat_items / aged.cleared_items) if aged.cleared_items else 0.0
        ),
        "historical_repeat_share_tokens": (
            (aged.historical_repeat_tokens / aged.cleared_tokens) if aged.cleared_tokens else 0.0
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    p = report["params"]
    cost = report["cost_delta"]
    print(
        f"transcripts={report['transcripts']} requests={base.requests} K={p['after_calls']}"
        f" MIN={p['min_tokens']} T={p['batch_tokens']} system={p['system_tokens']}"
    )
    print(f"{'arm':<9}{'context tokens':>18}{'cost w=1.0':>16}{'cost w=1.25':>16}{'clears':>9}")
    for name, arm in (("baseline", base), ("aged", aged)):
        print(
            f"{name:<9}{arm.context_tokens:>18,}{arm.cost['1.0']:>16,.0f}"
            f"{arm.cost['1.25']:>16,.0f}{arm.clear_events:>9,}"
        )
    print(
        f"delta    context {report['context_delta']:+.1%}  cost w=1.0 {cost['1.0']:+.1%}"
        f"  cost w=1.25 {cost['1.25']:+.1%}"
    )
    print(
        f"cleared  {aged.cleared_items:,} results / {aged.cleared_tokens:,} tokens;"
        f" historical later same-key calls: {report['historical_repeat_share_items']:.1%} of items,"
        f" {report['historical_repeat_share_tokens']:.1%} of tokens"
    )
    print(f"Anthropic breakpoint/20-block/TTL model delta: {report['anthropic_cost_delta']:+.1%}")
    print(json.dumps(report["distribution"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
