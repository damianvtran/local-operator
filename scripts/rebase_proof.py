#!/usr/bin/env python3
"""Prove a rebase moved a PR without changing it — or name what needs review.

Why this exists: over the last 120 merged PRs, 45% of agent-review rounds
followed an already-CLEAN round, and the largest single class (35 rounds) was
rebase/convergence. Of those with a resolvable scope, 85% had ZERO overlap
between the files upstream moved and the files the PR changes — a full review
round re-reading an unchanged patch. This script turns that zero-overlap case
into a mechanical, re-runnable proof instead of a round, and keeps every other
case a convergence review.

The verdict is MECHANICAL only when ALL three hold:

1. ``git range-diff <old-mb>..<old> <new-mb>..<new>`` pairs every commit with
   ``=`` (no commit added, dropped or altered);
2. the PR's own patch is identical across the rebase — compared as the
   multiset of its non-context lines (``+``/``-`` lines, file headers, mode,
   rename and binary-literal lines; hunk positions, context and ``index`` lines
   are excluded because they legitimately move);
3. no file the PR changes (before or after) is among the files upstream
   changed between the two merge-bases.

(3) is what (1) and (2) cannot see: an identical patch over a moved file can
still be a semantic conflict (a renamed callee, a changed default), so any
overlap is reported by name and stays a review. Exit 0 = MECHANICAL, 1 = review
needed, 2 = the proof could not be computed (unknown rev, git failure) — a
missing proof is never a pass.

The old head is usually gone from the branch after a force-push; fetch it by
SHA first (``git fetch origin <old-sha>``) — GitHub serves PR heads by SHA.

Run (from the repository, after fetching both heads and the base):
    python scripts/rebase_proof.py <old-head> <new-head> [--base origin/main]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

#: One ``git range-diff`` pair line: ``1:  abc1234 = 1:  def5678 subject``.
#: Unpaired commits use ``-`` for the missing side, and the status character
#: is one of ``=`` (identical), ``!`` (changed), ``<`` (dropped), ``>`` (added).
_RANGE_DIFF_PAIR = re.compile(r"^\s*(?:\d+|-+):\s+\S+\s+([=!<>])\s+(?:\d+|-+):\s+\S+")


class ProofError(RuntimeError):
    """The proof could not be computed; the caller exits 2, never 0."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProofError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _patch_lines(repo: Path, base: str, head: str) -> Counter[str]:
    """The patch's identity-bearing lines, as a multiset.

    ``--full-index`` is deliberately NOT used and ``index`` lines are dropped:
    a pre-image blob id changes whenever upstream touched the file, which (3)
    already reports by name — counting it here would double-report the same
    fact as "patch changed".
    """
    diff = _git(repo, "diff", "--no-color", "--no-ext-diff", "--binary", base, head)
    kept: Counter[str] = Counter()
    for line in diff.splitlines():
        if line.startswith((" ", "@@", "index ")) or line == "":
            continue
        kept[line] += 1
    return kept


def _changed_files(repo: Path, base: str, head: str) -> set[str]:
    # --no-renames: a rename's two paths must BOTH count as touched, or an
    # upstream rename of a PR file would slip past the overlap check.
    out = _git(repo, "diff", "--name-only", "--no-renames", base, head)
    return {line for line in out.splitlines() if line}


def prove(repo: Path, old: str, new: str, base: str) -> tuple[bool, str]:
    """(mechanical, markdown block) for the rebase ``old`` -> ``new``."""
    old_sha = _git(repo, "rev-parse", "--verify", f"{old}^{{commit}}").strip()
    new_sha = _git(repo, "rev-parse", "--verify", f"{new}^{{commit}}").strip()
    old_mb = _git(repo, "merge-base", base, old_sha).strip()
    new_mb = _git(repo, "merge-base", base, new_sha).strip()

    range_out = _git(
        repo, "range-diff", "--no-color", f"{old_mb}..{old_sha}", f"{new_mb}..{new_sha}"
    )
    statuses = [m.group(1) for m in map(_RANGE_DIFF_PAIR.match, range_out.splitlines()) if m]
    # An empty pairing is not "all equal": it means the ranges were not
    # comparable (or the PR has no commits), and that proves nothing.
    range_equal = bool(statuses) and all(status == "=" for status in statuses)
    range_summary = ", ".join(f"{s}×{n}" for s, n in sorted(Counter(statuses).items())) or "none"

    old_patch = _patch_lines(repo, old_mb, old_sha)
    new_patch = _patch_lines(repo, new_mb, new_sha)
    patch_identical = old_patch == new_patch

    pr_files = _changed_files(repo, old_mb, old_sha) | _changed_files(repo, new_mb, new_sha)
    upstream_files = _changed_files(repo, old_mb, new_mb) if old_mb != new_mb else set()
    overlap = sorted(pr_files & upstream_files)

    mechanical = range_equal and patch_identical and not overlap
    if mechanical:
        verdict = "MECHANICAL — no re-review needed"
    else:
        reasons = []
        if not range_equal:
            reasons.append(f"range-diff not all '=' ({range_summary})")
        if not patch_identical:
            delta = (old_patch - new_patch) + (new_patch - old_patch)
            reasons.append(f"patch lines differ ({sum(delta.values())} lines)")
        files = ", ".join(f"`{f}`" for f in overlap) or "none overlapping"
        verdict = f"CONVERGENCE REVIEW NEEDED — files: {files}; " + "; ".join(reasons or ["overlap"])

    short = lambda sha: sha[:10]  # noqa: E731 — display only
    block = "\n".join(
        [
            f"### Rebase proof — {short(old_sha)}..{short(new_sha)}",
            "",
            f"- Base: `{base}`; merge-bases `{short(old_mb)}` → `{short(new_mb)}`",
            f"- `git range-diff`: {len(statuses)} commit pair(s), {range_summary}"
            f" → {'all =' if range_equal else 'NOT all ='}",
            f"- PR patch (+/- and header lines): "
            f"{'byte-identical' if patch_identical else 'CHANGED'}"
            f" ({sum(new_patch.values())} lines)",
            f"- Upstream files moved between merge-bases: {len(upstream_files)}; "
            f"PR files: {len(pr_files)}; overlap: "
            + (", ".join(f"`{f}`" for f in overlap) if overlap else "none"),
            f"- Verdict: **{verdict}**",
            "",
            f"Re-run: `python scripts/rebase_proof.py {old_sha} {new_sha} --base {base}`",
        ]
    )
    return mechanical, block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old_head", help="PR head before the rebase (fetch it by SHA if gone)")
    parser.add_argument("new_head", help="PR head after the rebase")
    parser.add_argument("--base", default="origin/main", help="upstream ref (default origin/main)")
    parser.add_argument("--repo", default=".", help="repository path (default: cwd)")
    args = parser.parse_args(argv)
    try:
        mechanical, block = prove(Path(args.repo), args.old_head, args.new_head, args.base)
    except ProofError as exc:
        print(f"rebase proof could not be computed: {exc}", file=sys.stderr)
        return 2
    print(block)
    return 0 if mechanical else 1


if __name__ == "__main__":
    raise SystemExit(main())
