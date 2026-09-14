"""Which tree did an arm actually measure? The provenance field, mechanised.

WHY THIS EXISTS. ``git rev-parse HEAD`` names the WORKTREE, not the subtree under
test, and a benchmark arm is routinely measured with ``local_operator/`` checked
out of a different commit — that is the whole before/after design these scripts
exist for. A bare HEAD stamp therefore named the AFTER commit in six of six
before-arm artefacts of the campaign this module was written for (review round 2,
R2-1: `--label` carried the truth in prose while `rev` contradicted it), which is
exactly the hole the field was added to close in round 1.

So the tree is DECLARED by the caller (``--measured-tree``) and VERIFIED here, and
a script refuses to measure when ``local_operator/`` does not match what it was
asked to name. An arm can no longer record a commit it did not run: the failure
mode is a refusal before any measurement, not a plausible-looking wrong hash.

VERIFYING AGAINST THE WORKTREE, NOT THE INDEX, is deliberate: the arm's whole
point is the source the child imports, which is what is on disk. It also makes
this the guard for the sibling finding (R2-2) — a subtree checkout from a parent
commit leaves the child commit's ADDED files in place, and those show up here as
a non-empty diff, so the same check refuses a before arm that is not exactly the
tree it names.

RUNNING FROM A TREE WITH NO GIT is supported rather than refused: there is no
commit to name, so the rev is recorded as ``unknown`` and the warning says so. A
dirty-but-committed-elsewhere tree cannot be named by any rev, so the caller
commits first — this is a reproducibility tool, and "which commit produced this
number" is the one thing it must never guess.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The subtree these benchmarks measure (and therefore the subtree whose state
#: has to match the declared rev). Kept in one place: a benchmark that grows a
#: second source root must add it here rather than invent a second check.
SUBTREE = "local_operator"


class MeasuredTreeError(RuntimeError):
    """The subtree on disk does not match the tree the caller asked to name."""


def _git(*args: str) -> str | None:
    """Run git in the repo root. ``None`` when git or the repo is unavailable."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def worktree_head() -> str | None:
    """The worktree's HEAD commit, or ``None`` outside a git checkout."""
    return _git("rev-parse", "HEAD") or None


def describe(declared: str = "") -> dict[str, Any]:
    """Resolve and verify the measured tree, or raise :class:`MeasuredTreeError`.

    ``declared`` is the rev the caller names (``--measured-tree``); empty means
    "the worktree HEAD", which is the right answer for a normal run and the wrong
    one for a before arm — hence the verification rather than trust.

    Returns the fields the benchmark artefacts record:
    ``rev`` (the verified measured rev), ``worktree_head``, ``subtree``,
    ``verified`` (True, or None when there is no git to verify with).
    """
    head = worktree_head()
    if head is None:
        return {
            "rev": "unknown",
            "worktree_head": "unknown",
            "subtree": SUBTREE,
            "verified": None,
        }

    rev = declared.strip() or head
    resolved = _git("rev-parse", f"{rev}^{{commit}}")
    if resolved is None:
        raise MeasuredTreeError(f"--measured-tree names a commit git cannot resolve: {rev!r}")

    diff = _git("diff", resolved, "--", SUBTREE)
    if diff is None:
        raise MeasuredTreeError(
            f"could not diff {SUBTREE}/ against {resolved[:9]} to verify the " "measured tree"
        )
    if diff:
        stat = _git("diff", "--stat", resolved, "--", SUBTREE) or diff
        raise MeasuredTreeError(
            f"{SUBTREE}/ on disk does not match --measured-tree {resolved[:9]} "
            f"(worktree HEAD is {head[:9]}). The artefact would name a commit "
            "this run did not measure, so this run is refused; pass the rev you "
            f"actually checked out, or commit first. Diff:\n{stat}"
        )

    return {
        "rev": resolved,
        "worktree_head": head,
        "subtree": SUBTREE,
        "verified": True,
    }


def format_banner(tree: dict[str, Any]) -> str:
    """One line for the campaign header, so a human sees the tree it measured."""
    rev = str(tree["rev"])
    head = str(tree["worktree_head"])
    if tree["verified"] is None:
        return f"  measured tree: unknown (no git in {REPO_ROOT}) — provenance unverified"
    if rev == head:
        return f"  measured tree: {rev[:9]} (worktree HEAD, verified clean)"
    return (
        f"  measured tree: {rev[:9]} (VERIFIED against {SUBTREE}/ on disk; "
        f"worktree HEAD is {head[:9]})"
    )
