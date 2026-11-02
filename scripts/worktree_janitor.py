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

A worktree is ``REMOVE`` only when ALL hold; otherwise it is ``KEEP`` with the
reason, so unpushed or in-flight work is never offered up:

* not the primary checkout, not locked, directory exists;
* ``git status --porcelain`` is empty (no modified or untracked files);
* no commit reachable from HEAD that is absent from every remote ref;
* its branch's PR is MERGED/CLOSED (via ``gh``; squash merges are invisible to
  ancestry), or - for a detached HEAD / without ``gh`` - HEAD is contained in a
  remote ref such as ``origin/main``.

``--apply`` runs plain ``git worktree remove`` (never ``--force``), so git itself
refuses a tree that turned dirty in the meantime. Ignored build output
(``node_modules``, ``.venv``) goes with the tree; check the KEEP/REMOVE note
before applying if a tree holds ignored non-regenerable files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(args: list[str], cwd: str | Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout.strip()


def _worktrees(repo: str) -> list[dict[str, str]]:
    """Parse ``git worktree list --porcelain`` into one dict per worktree."""
    code, out = _run(["git", "worktree", "list", "--porcelain"], repo)
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


def _pr_states(path: str, branch: str) -> list[str]:
    code, out = _run(
        ["gh", "pr", "list", "--head", branch, "--state", "all", "--json", "state"], path
    )
    if code != 0:
        return []
    try:
        return [row["state"] for row in json.loads(out)]
    except (ValueError, KeyError, TypeError):
        return []


def classify(tree: dict[str, str], primary: str, use_gh: bool) -> tuple[str, str]:
    """Return ``("REMOVE"|"KEEP", reason)`` for one worktree."""
    path = tree["worktree"]
    if path == primary:
        return "KEEP", "primary checkout"
    if "locked" in tree:
        return "KEEP", "locked"
    if not Path(path).is_dir():
        return "KEEP", "directory missing (run `git worktree prune`)"
    code, status = _run(["git", "status", "--porcelain"], path)
    if code != 0:
        return "KEEP", "cannot inspect"
    if status:
        return "KEEP", f"dirty ({len(status.splitlines())} paths)"
    _, unpushed = _run(["git", "rev-list", "--count", "HEAD", "--not", "--remotes"], path)
    if unpushed != "0":
        return "KEEP", f"{unpushed} commit(s) on no remote"
    branch = tree.get("branch", "").removeprefix("refs/heads/")
    if branch and use_gh:
        states = _pr_states(path, branch)
        if states and all(s in ("MERGED", "CLOSED") for s in states):
            return "REMOVE", f"PR {'/'.join(sorted(set(states)))}"
        if states:
            return "KEEP", "PR still open"
    code, _ = _run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"], path)
    if code == 0:
        return "REMOVE", "HEAD is on origin/main"
    if not branch:
        _, contained = _run(["git", "branch", "-r", "--contains", "HEAD"], path)
        if contained:
            return "REMOVE", "detached HEAD contained in a remote ref"
    return "KEEP", "no merged/closed PR found"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default=".", help="any checkout or worktree of the repo")
    parser.add_argument("--no-gh", action="store_true", help="skip PR lookups (ancestry only)")
    parser.add_argument("--apply", action="store_true", help="run `git worktree remove` on REMOVE rows")
    args = parser.parse_args()
    repo = str(Path(args.repo).expanduser().resolve())
    trees = _worktrees(repo)
    primary = trees[0]["worktree"]  # git lists the main working tree first
    exit_code = 0
    for tree in trees:
        verdict, reason = classify(tree, primary, not args.no_gh)
        if tree["worktree"] == primary:
            continue
        stray = "" if "/.worktrees/" in tree["worktree"] else "  [outside .worktrees]"
        print(f"{verdict:6} {tree['worktree']}  ({reason}){stray}")
        if verdict == "REMOVE" and args.apply:
            code, out = _run(["git", "worktree", "remove", tree["worktree"]], primary)
            if code != 0:
                print(f"       remove failed: {out}", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
