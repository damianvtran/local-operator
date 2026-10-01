#!/usr/bin/env python3
"""Offline benchmark: projects soft search — cold build and warm per-query cost.

WHY THIS EXISTS. The Projects views search on every keystroke, so the question
this script answers is "what does one more request cost the server", split into
its two regimes: the COLD first search in a process (fold + tokenise + vocab
build, i.e. the in-process index's one-time price) and the WARM per-query cost
(one string comparison per field per row against the raw-first cache, then
resolve + rank — see ``local_operator/projects_search.py``). Those are the
numbers the architecture note budgets against (warm p95 <= 40 ms @ ~121 rows,
<= 150 ms @ an 800-row simulation; cold <= 250 ms); the PR carries the runs —
including the plain verdict on those budgets, which this script helps the
reader reach but never asserts (wall figures are observations on a shared
host, never CI assertions).

Synthetic corpus by default — no live data, deterministic seed — sized to the
store the design measured (avg ~4.7 KB of updates per row). ``--store`` ADDS a
real ``projects/`` directory to the same run (READ-ONLY; aggregate numbers
only, never row text), so the simulated and live figures print side by side.

**Host load moves these figures by 2-5x**, and every run prints the load it
was taken at: the same 146-row store arm has measured warm p95 ~33-65 ms
across this fleet's ordinary band, and a load-42 spike during review round 1
read 2-5x the quieter figures. Compare arms back to back; never quote a
figure without its load. p95 is a nearest-rank percentile over the warm
sample count the run prints (the first revision's 7-sample set made "p95"
the max; that is what the sample count is here to make visible).

Run it from a worktree with that worktree's interpreter, isolated:

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \\
      .venv/bin/python scripts/bench_projects_search.py [--store /abs/path/to/projects]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import local_operator.projects_search as search_mod  # noqa: E402
from local_operator.projects import Project  # noqa: E402

#: A base vocabulary plus generated pseudo-words, because the fuzzy tiers' cost
#: depends on VOCABULARY SIZE (the DP runs against deduplicated words, not per
#: row token): a 40-word pool made every resolve free and understated the live
#: store 10x. Sized so a corpus carries a few thousand distinct words, like a
#: real store; the base words are the queries' exact/typo targets.
_BASE_WORDS = (
    "release process dashboard classifier kafka deploy migration audit billing "
    "pipeline rollout cache index search timeline quota invoice ledger export "
    "tenant gateway session runtime project milestone regression incident "
    "postmortem cutover backfill schema replica partition"
).split()


def _vocabulary(size: int = 2_500) -> tuple[str, ...]:
    rng = random.Random(7)
    consonants, vowels = "bcdfghjklmnpqrstvwz", "aeiou"
    words: list[str] = list(_BASE_WORDS)
    while len(words) < size:
        syllables = "".join(
            rng.choice(consonants) + rng.choice(vowels) for _ in range(rng.randint(2, 4))
        )
        words.append(syllables)
    return tuple(words)


WORDS = _vocabulary()

#: The warm query mix: single tokens (exact/prefix — the cheap end), typos
#: (bounded edit distance — the dear end), and multi-term phrases, all drawn
#: from the vocabulary above. ~50 samples, so p95 is a real nearest-rank
#: percentile rather than the max a 7-query set silently made it.
QUERIES = (
    # single tokens, exact/prefix
    "release",
    "process",
    "dashboard",
    "classifier",
    "kafka",
    "deploy",
    "migration",
    "audit",
    "billing",
    "pipeline",
    "rollout",
    "cache",
    "index",
    "search",
    "timeline",
    "quota",
    "invoice",
    "ledger",
    "export",
    "tenant",
    "gateway",
    "session",
    "runtime",
    "project",
    "milestone",
    "regression",
    "incident",
    "postmortem",
    "cutover",
    "backfill",
    "schema",
    "replica",
    "partition",
    # typos (bounded edit distance; the dear ones)
    "dashbord",
    "classifer",
    "migratio",
    "deply",
    "invoce",
    "milestne",
    # multi-term phrases
    "release process",
    "kube deploy",
    "audit ledger",
    "quota invoice",
    "cache index",
    "search timeline",
    "cutover backfill",
    "schema replica",
    "incident postmortem",
    "gateway session",
    "billing pipeline",
    "tenant export",
)

#: Updates per row / words per update entry, sized so a row's history is the
#: ~4.7 KB the live store averages (12 x 60 words x ~6 chars).
UPDATES_PER_ROW = 12
WORDS_PER_UPDATE = 60


def _payloads(count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    payloads = []
    for index in range(count):
        payloads.append(
            {
                "id": f"{seed:04x}{index:08x}",
                "name": f"proj-{index:03d}-{rng.choice(WORDS)}",
                "description": " ".join(rng.choices(WORDS, k=8)),
                "tags": rng.sample(WORDS, k=2),
                "owner": rng.choice(WORDS),
                "team": rng.choice(WORDS),
                "progress": " ".join(rng.choices(WORDS, k=12)),
                "progress_reported_by": "operator",
                "updated_at": float(count - index),
                "updates": [
                    {
                        "at": "2026-01-01T00:00:00Z",
                        "text": " ".join(rng.choices(WORDS, k=WORDS_PER_UPDATE)),
                        "by": "operator",
                    }
                    for _ in range(UPDATES_PER_ROW)
                ],
            }
        )
    return payloads


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile: the ``ceil(fraction * n)``-th smallest sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(fraction * len(ordered))
    return ordered[max(rank, 1) - 1]


def _arm(label: str, payloads: list[dict[str, Any]], queries: tuple[str, ...]) -> None:
    """Cold + warm timings for one corpus size.

    Two MATERIALISED copies of the same payloads alternate across the warm
    queries: the caches are content-keyed, so identical text must NOT re-fold
    even when it arrives as different objects — alternating keeps the string
    comparison on the memcmp path a fresh registry produces per request,
    instead of the pointer-equality fast path a reused object would get.
    """
    rows_a = [Project.model_validate(payload) for payload in payloads]
    rows_b = [Project.model_validate(payload) for payload in payloads]
    corpus_chars = sum(len(text) for row in rows_a for text in (row.name, row.description))
    corpus_chars += sum(len(row.progress) + sum(len(u.text) for u in row.updates) for row in rows_a)

    # Each arm starts from empty singletons (the module's process-wide cache is
    # what a cold process pays once).
    search_mod._INDEXES = {field: search_mod._FieldIndex() for field in search_mod._FIELDS}

    start = time.perf_counter()
    first = search_mod.search_projects(rows_a, queries[0])
    cold_ms = (time.perf_counter() - start) * 1000

    warm_ms: list[float] = []
    for position, query in enumerate(queries[1:]):
        rows = rows_a if position % 2 == 0 else rows_b
        start = time.perf_counter()
        search_mod.search_projects(rows, query)
        warm_ms.append((time.perf_counter() - start) * 1000)

    p50 = _percentile(warm_ms, 0.50)
    p95 = _percentile(warm_ms, 0.95)
    print(
        f"{label}: rows={len(rows_a)} corpus_chars={corpus_chars:,}\n"
        f"  cold (fold+tokenise+vocab, first search '{queries[0]}'): {cold_ms:.1f} ms "
        f"({len(first)} hits)\n"
        f"  warm (per query, {len(warm_ms)} samples, alternating row objects): "
        f"p50 {p50:.2f} ms  p95 {p95:.2f} ms  max {max(warm_ms):.2f} ms"
    )


def _store_payloads(store: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(store.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("name"):
            payloads.append(data)
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[146, 800],
        help="synthetic corpus sizes to time (default: 146 800)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=None,
        help=(
            "ALSO time a real projects/ directory (read-only): the live figures "
            "print beside the synthetic ones"
        ),
    )
    args = parser.parse_args()

    print(
        "note budgets: cold <= 250 ms; warm p95 <= 40 ms @ ~121 rows / <= 150 ms @ 800-sim\n"
        "host load (1/5/15m): " + ", ".join(f"{value:.1f}" for value in os.getloadavg())
    )
    for index, count in enumerate(args.rows):
        _arm(f"N={count} (synthetic)", _payloads(count, seed=20261001 + index), QUERIES)
    if args.store is not None:
        payloads = _store_payloads(args.store)
        if not payloads:
            print(f"no readable rows under {args.store}", file=sys.stderr)
            return 1
        _arm(f"store={args.store}", payloads, QUERIES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
