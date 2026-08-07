#!/usr/bin/env python3
"""Measure the skill-selection threshold against a real skills corpus.

The shipped ``LocalEmbedder.default_threshold`` is a NUMBER OF RECORD: it
decides whether the agent gets its playbook or silently proceeds without one.
A number of record has to be reproducible by anyone, on any corpus — otherwise
the next person to touch it has to take the docstring on faith.

Usage:

    python scripts/calibrate_skill_threshold.py --skills-dir ~/.omp/agent/skills

Supply labelled queries to measure recall (a JSON file of
``[["query", "expected-skill-name"], ...]``); without them the script still
reports the off-corpus noise floor, which is the half that bounds the threshold
from below.

Prints, for a sweep of candidate thresholds, recall over the labelled queries
and the false-positive rate over the off-corpus ones, plus the score gap and
its midpoint. A threshold is defensible when it sits inside the gap of EVERY
corpus you care about, not just the one it was tuned on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from local_operator.skills.api import (  # noqa: E402
    SkillIndex,
    default_backend_from_env,
    discover_skills,
)

#: Queries with no matching skill in any sane corpus. These bound the threshold
#: from BELOW: whatever the best score among them is, the cut must be above it.
DEFAULT_OFF_CORPUS = [
    "translate this poem into french",
    "what is the capital of peru",
    "write a haiku about rain",
    "aaaaaa bbbbbb cccccc",
    "the airspeed velocity of an unladen swallow",
    "sort this list of integers in python",
    "explain quantum entanglement",
    "who won the 1998 world cup",
]

SWEEP = [0.10, 0.12, 0.15, 0.18, 0.19, 0.20, 0.22, 0.25, 0.27, 0.30]


async def run(skills_dir: Path, labelled: list[tuple[str, str]], off_corpus: list[str]) -> int:
    skills, warnings = discover_skills([skills_dir])
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not skills:
        print(f"no skills found under {skills_dir}", file=sys.stderr)
        return 1

    backend = default_backend_from_env(lambda _key: None, None)
    index = SkillIndex(skills, backend)
    await index.build(backend)
    names = [skill.name for skill in index.skills]
    print(f"corpus: {len(skills)} skills from {skills_dir}")
    print(f"backend: {type(backend).__name__} (dim {backend.dim})")
    print(f"shipped default_threshold: {backend.default_threshold}")

    async def scores_for(query: str) -> dict[str, float]:
        vector = (await backend.embed([query]))[0]
        return dict(zip(names, index._scores(vector)))

    relevant: list[float] = []
    misranked: list[str] = []
    for query, expected in labelled:
        scored = await scores_for(query)
        if expected not in scored:
            print(f"warning: {expected!r} is not in this corpus; skipping", file=sys.stderr)
            continue
        relevant.append(scored[expected])
        top = max(scored, key=lambda name: scored[name])
        if top != expected:
            misranked.append(f"{query!r} -> {top} (expected {expected})")

    unrelated = [max((await scores_for(query)).values()) for query in off_corpus]

    if relevant:
        print(
            f"\nrelevant  : min {min(relevant):.4f}  median {median(relevant):.4f}  "
            f"max {max(relevant):.4f}  (n={len(relevant)})"
        )
    print(
        f"unrelated : max {max(unrelated):.4f}  median {median(unrelated):.4f}  "
        f"(n={len(unrelated)})"
    )
    if relevant:
        low, high = max(unrelated), min(relevant)
        if low < high:
            print(f"gap       : ({low:.4f}, {high:.4f})  midpoint {(low + high) / 2:.4f}")
        else:
            print(f"gap       : NONE — best false {low:.4f} >= worst true {high:.4f}")

    if misranked:
        print("\nranking failures (no threshold can fix these):")
        for line in misranked:
            print(f"  {line}")

    print(f"\n{'thresh':>7} {'recall':>8} {'false-pos':>10}")
    for threshold in SWEEP:
        recall = (
            sum(1 for score in relevant if score >= threshold) / len(relevant)
            if relevant
            else float("nan")
        )
        false_pos = sum(1 for score in unrelated if score >= threshold) / len(unrelated)
        marker = "  <- shipped" if abs(threshold - backend.default_threshold) < 1e-9 else ""
        print(f"{threshold:7.2f} {recall:8.0%} {false_pos:10.0%}{marker}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills-dir", type=Path, required=True)
    parser.add_argument(
        "--labelled",
        type=Path,
        help='JSON file of [["query", "expected-skill"], ...] for recall',
    )
    parser.add_argument(
        "--off-corpus",
        type=Path,
        help="JSON file of query strings with no matching skill",
    )
    args = parser.parse_args(argv)

    labelled: list[tuple[str, str]] = []
    if args.labelled:
        labelled = [(q, e) for q, e in json.loads(args.labelled.read_text())]
    off_corpus = (
        json.loads(args.off_corpus.read_text()) if args.off_corpus else list(DEFAULT_OFF_CORPUS)
    )
    return asyncio.run(run(args.skills_dir, labelled, off_corpus))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
