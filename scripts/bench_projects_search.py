#!/usr/bin/env python3
"""Offline benchmark: projects soft search — cold build and warm per-query cost.

WHY THIS EXISTS. The Projects views search on every keystroke, so the question
this script answers is "what does one more request cost the server", split into
its two regimes: the COLD first search in a process (fold + tokenise + vocab
build, i.e. the in-process index's one-time price) and the WARM per-query cost
(one string comparison per field per row against the raw-first cache, then
resolve + rank — see ``local_operator/projects_search.py``). Those are the
numbers the architecture note budgets against (warm p95 <= 40 ms @ ~121 rows,
<= 150 ms @ an 800-row simulation; cold <= 250 ms); the PR carries the runs.

Synthetic corpus by default — no live data, deterministic seed — sized to the
store the design measured (avg ~4.7 KB of updates per row). ``--store`` points
it at a real ``projects/`` directory instead (READ-ONLY; aggregate numbers
only, never row text).

Wall figures are observations on a shared host, never CI assertions. Run it
from a worktree with that worktree's interpreter, isolated:

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" \\
      .venv/bin/python scripts/bench_projects_search.py
"""

from __future__ import annotations

import argparse
import json
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

#: Worst-case-ish queries: prefix, typo, multi-term, single short word — the
#: mix the architecture note measured (typo-heavy queries are the dear ones).
QUERIES = (
    "release",
    "dashbord",
    "release process",
    "classifer",
    "kube deploy",
    "migratio",
    "audit ledger",
    "quota",
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

    warm_sorted = sorted(warm_ms)
    p50 = warm_sorted[len(warm_sorted) // 2]
    p95 = warm_sorted[min(len(warm_sorted) - 1, round(len(warm_sorted) * 0.95) - 1)]
    print(
        f"{label}: rows={len(rows_a)} corpus_chars={corpus_chars:,}\n"
        f"  cold (fold+tokenise+vocab, first search '{queries[0]}'): {cold_ms:.1f} ms "
        f"({len(first)} hits)\n"
        f"  warm (per query, alternating row objects): p50 {p50:.2f} ms  "
        f"p95 {p95:.2f} ms  max {max(warm_ms):.2f} ms  ({len(warm_ms)} queries)"
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
        help="a real projects/ directory to time instead of the synthetic arms (read-only)",
    )
    args = parser.parse_args()

    if args.store is not None:
        payloads = _store_payloads(args.store)
        if not payloads:
            print(f"no readable rows under {args.store}", file=sys.stderr)
            return 1
        _arm(f"store={args.store.name or args.store}", payloads, QUERIES)
        return 0

    for index, count in enumerate(args.rows):
        _arm(f"N={count} (synthetic)", _payloads(count, seed=20261001 + index), QUERIES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
