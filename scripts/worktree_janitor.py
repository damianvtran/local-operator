#!/usr/bin/env python3
"""List git worktrees that are safe to remove: clean, fully pushed, PR merged/closed.

Why this exists: worktrees created outside ``<repo>/.worktrees`` (sibling dirs,
``$HOME`` top level, ``$TMPDIR/tmp.*`` release cuts, session scratchpads) piled
up to ~140 merged/closed checkouts (~9 GiB) because nothing removed them when the
PR merged. The policy is "site at ``<repo>/.worktrees/<name>``, remove when the
PR merges"; this is the safety net for the ones that were missed. It is a
helper you run, not a daemon, and it never deletes by default.

Usage (from any directory)::

    scripts/worktree_janitor.py [--repo ~/local-operator] [--no-gh]   # dry run
    scripts/worktree_janitor.py --apply                               # remove REMOVE rows

A worktree is ``REMOVE`` only when ALL hold; otherwise it is ``KEEP`` (or
``STALE``, see below) with the reason, so unpushed or in-flight work is never
offered up:

* not the primary checkout, not locked, directory exists;
* ``git status --porcelain`` is empty (no modified or untracked files);
* EITHER (PR evidence, needs ``gh``) every PR for its branch is MERGED/CLOSED and
  HEAD equals one PR's ``headRefOid`` - checked BEFORE the unpushed-commit test
  because ``delete_branch_on_merge`` removes the remote branch, which makes a
  squash-merged branch's own commits read as "on no remote". Matching the exact
  head sha (not just the branch name) keeps a reused name such as
  ``release-next`` from inheriting an old PR's verdict;
* OR (ancestry, no PR evidence) no commit reachable from HEAD is absent from
  every remote ref, HEAD is an ancestor of ``origin/main`` itself (a detached QA
  tree at an open PR head is contained in a remote ref but is NOT merged), and
  HEAD last moved at least ``--min-age-hours`` ago. The age gate exists because a
  freshly cut lane with no commits yet is an ancestor of ``origin/main`` and
  indistinguishable from a merged one by ancestry alone. Age is the mtime of the
  worktree's own ``HEAD`` file or its HEAD reflog, whichever is newer (creation,
  checkout and every commit touch one of them).

``STALE`` rows are registrations git cannot clean up by itself (a ``locked``
entry whose reason contains "initializing" left by an interrupted
``worktree add``, or a prunable entry whose directory vanished). They are never
auto-applied; the row prints the exact manual remedy.

``--apply`` runs plain ``git worktree remove`` (never ``--force``), so git itself
refuses a tree that turned dirty in the meantime. Ignored build output
(``node_modules``, ``.venv``) goes with the tree; check the KEEP/REMOVE note
before applying if a tree holds ignored non-regenerable files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import sys
from pathlib import Path


DEFAULT_MIN_AGE_HOURS = 24.0


def _run(args: list[str], cwd: str | Path) -> tuple[int, str, str]:
    """Run ``args``; return ``(code, stdout, stderr)`` (git reports refusals on stderr)."""
    try:
        proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", str(exc)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _worktrees(repo: str) -> list[dict[str, str]]:
    """Parse ``git worktree list --porcelain`` into one dict per worktree."""
    code, out, _ = _run(["git", "worktree", "list", "--porcelain"], repo)
    if code != 0:
        sys.exit(f"not a git repository: {repo}")
    trees: list[dict[str, str]] = []
    for block in out.split("\n\n"):
        entry: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            entry[key] = value
        if "worktree" in entry:
            trees.append(entry)
    return trees


def _pr_records(path: str, branch: str) -> list[tuple[str, str]]:
    """``(state, headRefOid)`` for every PR whose head branch is ``branch``."""
    code, out, _ = _run(
        ["gh", "pr", "list", "--head", branch, "--state", "all", "--json", "state,headRefOid"],
        path,
    )
    if code != 0:
        return []
    try:
        return [(row["state"], row.get("headRefOid", "")) for row in json.loads(out)]
    except (ValueError, KeyError, TypeError, AttributeError):
        return []


def _head_age_hours(path: str) -> float:
    """Hours since this worktree's HEAD last moved (newest of its HEAD file / HEAD reflog)."""
    code, gitdir, _ = _run(["git", "rev-parse", "--absolute-git-dir"], path)
    if code != 0:
        return 0.0  # unknown age reads as fresh: fail toward KEEP
    mtimes = []
    for name in ("HEAD", "logs/HEAD"):
        try:
            mtimes.append(os.stat(Path(gitdir) / name).st_mtime)
        except OSError:
            continue
    return (time.time() - max(mtimes)) / 3600 if mtimes else 0.0


def classify(
    tree: dict[str, str],
    primary: str,
    use_gh: bool,
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
) -> tuple[str, str]:
    """Return ``("REMOVE"|"KEEP"|"STALE", reason)`` for one worktree."""
    path = tree["worktree"]
    if path == primary:
        return "KEEP", "primary checkout"
    if "locked" in tree:
        if "initializing" in tree["locked"]:
            return (
                "STALE",
                "locked by an interrupted `worktree add`; manual: "
                f"`git worktree remove -f -f {path}`",
            )
        return "KEEP", "locked"
    if "prunable" in tree or not Path(path).is_dir():
        return "STALE", "registration points at a missing directory; manual: `git worktree prune`"
    code, status, _ = _run(["git", "status", "--porcelain"], path)
    if code != 0:
        return "KEEP", "cannot inspect"
    if status:
        return "KEEP", f"dirty ({len(status.splitlines())} paths)"
    branch = tree.get("branch", "").removeprefix("refs/heads/")
    head = tree.get("HEAD", "")
    if branch and use_gh:
        records = _pr_records(path, branch)
        if any(state not in ("MERGED", "CLOSED") for state, _ in records):
            return "KEEP", "PR still open"
        if head and any(oid == head for _, oid in records):
            return "REMOVE", f"PR {'/'.join(sorted({s for s, _ in records}))} at this head"
    _, unpushed, _ = _run(["git", "rev-list", "--count", "HEAD", "--not", "--remotes"], path)
    if unpushed != "0":
        return "KEEP", f"{unpushed} commit(s) on no remote"
    code, _, _ = _run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], path)
    if code != 0:
        return "KEEP", "no merged/closed PR at this head"
    if _head_age_hours(path) < min_age_hours:
        return "KEEP", f"on origin/main but HEAD moved <{min_age_hours:g}h ago (fresh lane?)"
    return "REMOVE", "HEAD is on origin/main"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=".", help="any checkout or worktree of the repo")
    parser.add_argument("--no-gh", action="store_true", help="skip PR lookups (ancestry only)")
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=DEFAULT_MIN_AGE_HOURS,
        help="ancestry-only removals need HEAD unchanged this long (default %(default)s)",
    )
    parser.add_argument("--apply", action="store_true", help="run `git worktree remove` on REMOVE rows")
    args = parser.parse_args(argv)
    repo = str(Path(args.repo).expanduser().resolve())
    trees = _worktrees(repo)
    primary = trees[0]["worktree"]  # git lists the main working tree first
    exit_code = 0
    for tree in trees:
        verdict, reason = classify(tree, primary, not args.no_gh, args.min_age_hours)
        if tree["worktree"] == primary:
            continue
        stray = "" if "/.worktrees/" in tree["worktree"] else "  [outside .worktrees]"
        print(f"{verdict:6} {tree['worktree']}  ({reason}){stray}")
        if verdict == "REMOVE" and args.apply:
            code, out, err = _run(["git", "worktree", "remove", tree["worktree"]], primary)
            if code != 0:
                print(f"       remove failed: {err or out}", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
