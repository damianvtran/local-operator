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
of clear events; and the share of cleared results the transcript later
re-fetched (the same read path, bash command or grep pattern+scope issued
again after the clear).

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
    DEFAULT_AGED_CLEAR_AFTER_CALLS,
    DEFAULT_AGED_CLEAR_BATCH_TOKENS,
    DEFAULT_AGED_CLEAR_MIN_TOKENS,
    clear_aged_tool_results,
    prune_tool_outputs,
)
from local_operator.compaction.tokens import estimate_tokens  # noqa: E402
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
    refetched_items: int = 0
    refetched_tokens: int = 0


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
    result = ArmResult()
    context: list[Message] = []
    previous: list[tuple[str, bool]] = []
    cleared_keys: dict[str, int] = {}  # refetch key -> tokens cleared under it
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
        tokens = [tokens_of(m) for m in request]
        signature = [(m.id, bool((m.provider_payload or {}).get("pruned"))) for m in request]
        first_change = next(
            (i for i, (a, b) in enumerate(zip(signature, previous)) if a != b),
            min(len(signature), len(previous)),
        )
        total = args.system_tokens + sum(tokens)
        cached = args.system_tokens + sum(tokens[:first_change])
        result.requests += 1
        result.context_tokens += total
        for rate in WRITE_RATES:
            result.cost[str(rate)] += cached * CACHE_READ + (total - cached) * rate
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
                    result.refetched_items += 1
                    result.refetched_tokens += cleared_keys.pop(key)
        context.append(message)
    return result


def _clear(
    context: list[Message],
    result: ArmResult,
    args: argparse.Namespace,
    call_keys: dict[str, str],
    cleared_keys: dict[str, int],
) -> None:
    before = {m.id: estimate_tokens(m) for m in context if m.role == "tool"}
    pruned = {m.id for m in context if (m.provider_payload or {}).get("pruned")}
    _, cleared = clear_aged_tool_results(
        context,
        after_calls=args.after_calls,
        min_tokens=args.min_tokens,
        batch_tokens=args.batch_tokens,
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
            cleared_keys[key] = cleared_keys.get(key, 0) + before[message.id]


def _one(path: str, args: argparse.Namespace) -> tuple[ArmResult, ArmResult] | None:
    events = _load(path)
    if not any(kind == "msg" and data.get("role") == "assistant" for kind, data in events):
        return None
    return _replay(events, False, args), _replay(events, True, args)


def _sum(results: list[ArmResult]) -> ArmResult:
    total = ArmResult()
    for r in results:
        total.requests += r.requests
        total.context_tokens += r.context_tokens
        for rate in total.cost:
            total.cost[rate] += r.cost[rate]
        total.clear_events += r.clear_events
        total.cleared_items += r.cleared_items
        total.cleared_tokens += r.cleared_tokens
        total.refetched_items += r.refetched_items
        total.refetched_tokens += r.refetched_tokens
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--corpus", required=True, help="glob of COPIED transcript.jsonl files")
    parser.add_argument("--after-calls", type=int, default=DEFAULT_AGED_CLEAR_AFTER_CALLS)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_AGED_CLEAR_MIN_TOKENS)
    parser.add_argument("--batch-tokens", type=int, default=DEFAULT_AGED_CLEAR_BATCH_TOKENS)
    parser.add_argument("--system-tokens", type=int, default=DEFAULT_SYSTEM_TOKENS)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
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
            "min_tokens": args.min_tokens,
            "batch_tokens": args.batch_tokens,
            "system_tokens": args.system_tokens,
        },
        "baseline": asdict(base),
        "aged": asdict(aged),
        "context_delta": delta(aged.context_tokens, base.context_tokens),
        "cost_delta": {r: delta(aged.cost[r], base.cost[r]) for r in base.cost},
        "refetch_share_items": (
            (aged.refetched_items / aged.cleared_items) if aged.cleared_items else 0.0
        ),
        "refetch_share_tokens": (
            (aged.refetched_tokens / aged.cleared_tokens) if aged.cleared_tokens else 0.0
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
        f" re-fetched later: {report['refetch_share_items']:.1%} of items,"
        f" {report['refetch_share_tokens']:.1%} of tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
