#!/usr/bin/env python3
"""Make a merge-gate run test the PR head merged into the CURRENT main tip.

WHY THIS EXISTS
---------------
`full-suite.yml` runs on a pull request only for a `labeled` event (the
`make merge-gate` ritual toggles `ready-to-merge`). A plain
`actions/checkout` on that event checks out `refs/pull/<n>/merge`, and GitHub
recomputes that ref only when the PR itself is pushed — NOT when `main` moves.
A re-armed gate therefore tests the PR against whatever `main` was at the last
push to the PR. Measured on 2026-11-10:

* #3723 gate 42881345745 checked out `15f31d1 = Merge 5cfa1bc4 into
  065ab3c8a9f`, 36 commits behind main;
* #3720 gate 42881290246 checked out a merge into `ce220b9216c`, 38 behind;
* #3732 gate 42881232605 checked out `ff0c557 = Merge 44e32ff6 into
  1799972999c`, 22 behind, and re-failed a membership_change_push node that
  main had already fixed (#3685, ad16110f);
* `gh api repos/.../pulls/3732` kept returning the stale `merge_commit_sha`
  for over a minute; only a push (`gh pr update-branch`) refreshed it.

So a gate went red on failures main had already fixed, and could go green
without ever testing the PR against the main it would merge into.

WHAT IT DOES
------------
Two subcommands, used together by `full-suite.yml`:

``resolve``  (once per run, in the `changes` job): on a `pull_request` event,
    read the live tip of the PR's base branch with `git ls-remote` and write it
    as the `gate_base` step output. Resolving ONCE and handing the SHA to every
    job through job outputs is what keeps the shards of one run on the SAME
    base even when main moves mid-run (a per-job `ls-remote` would let shard 0
    and shard 5 test different trees). Off `pull_request` it writes nothing.

``apply``  (in every job that checks out code, right after the checkout):
    fetch that base and the PR head SHA, build the merge locally with
    `git merge --no-ff`, check it out detached, and VERIFY `HEAD^1 == base` and
    `HEAD^2 == head`. A conflict, a missing base, or a failed verification
    exits non-zero with both SHAs in a `::error` annotation — never a silent
    fall back to the stale checkout. Off `pull_request` it changes nothing.

Either way `apply` prints exactly one greppable line per job::

    full-suite tested-base=<sha|none> head=<sha> tree=<sha> merge=<sha> job=<id> event=<name>

How to verify a gate log: grep every job's log for `full-suite tested-base=`.
`tested-base` must equal `git rev-parse origin/main` at the moment the run's
`changes` job ran (or a later main if you check late — it must at least be an
ancestor of today's main), `head` must equal the PR head, and `tree` must be
IDENTICAL across all jobs of the run (merge SHAs differ by commit timestamp;
the tree does not).

CONSTRAINTS
-----------
* Stdlib only, and it runs before `setup-python`: the `changes` job has none,
  and an install that can fail must never decide what gets tested.
* Runs on ubuntu and macOS runners (`tui-e2e` uses both), so it shells out to
  `git` and nothing platform specific.
* The workflow's first step in `test` stays a bare `actions/checkout@v7`:
  `scripts/gen_test_durations.py` binds timing sources to exactly that shape,
  so this runs as a separate step after it rather than replacing it.
* The fetch deepens step by step instead of unshallowing a 12k-commit
  history: a PR is usually a few hundred commits from main at most, and a
  merge base found in the shallow window is the true one (shallow cuts hide
  older commits, never newer ones).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence

#: The line prefix a log reader greps for; pinned by tests/unit/test_ci_hygiene.py.
LOG_PREFIX = "full-suite tested-base="

#: Commits fetched per side on the first try, then extra depth per retry.
#: Six retries reach 50 + 6 * 400 = 2450 commits before the final unshallow.
INITIAL_DEPTH = 50
DEEPEN_STEP = 400
DEEPEN_TRIES = 6

#: Identity for the throwaway local merge commit (never pushed).
MERGE_IDENTITY = ("-c", "user.name=full-suite gate", "-c", "user.email=full-suite-gate@invalid")


class GateError(RuntimeError):
    """A condition under which the gate must fail rather than test something else."""


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc


def _rev(ref: str) -> str:
    return _git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").stdout.strip()


def _has_commit(sha: str) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def _is_shallow() -> bool:
    return _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"


def _merge_base(base: str, head: str) -> str | None:
    proc = _git("merge-base", base, head, check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _fetch(remote: str, base: str, head: str) -> None:
    """Fetch both SHAs with enough history to find their merge base."""
    shallow = _is_shallow()
    if not shallow:
        # A full clone (fetch-depth: 0, or a local test repo) only needs the
        # objects; history is already complete.
        if not (_has_commit(base) and _has_commit(head)):
            _git("fetch", "--no-tags", "--quiet", remote, base, head)
        return
    _git("fetch", "--no-tags", "--quiet", f"--depth={INITIAL_DEPTH}", remote, base, head)
    for _ in range(DEEPEN_TRIES):
        if _merge_base(base, head):
            return
        _git("fetch", "--no-tags", "--quiet", f"--deepen={DEEPEN_STEP}", remote, base, head)
    if not _merge_base(base, head):
        _git("fetch", "--no-tags", "--quiet", "--unshallow", remote, base, head)


def _emit(line: str) -> None:
    print(line, flush=True)


def resolve(*, event: str, remote: str, base_ref: str, output: str | None) -> str | None:
    """Return (and export as `gate_base`) the live tip of `base_ref` on a PR event."""
    if event != "pull_request":
        _emit(
            f"gate-merge-base: event={event}: no PR base to resolve; the checkout is tested as-is"
        )
        return None
    if not base_ref:
        raise GateError("pull_request event without a base ref (GITHUB_BASE_REF is empty)")
    proc = _git("ls-remote", "--exit-code", remote, f"refs/heads/{base_ref}")
    lines = [line.split() for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) != 1 or len(lines[0][0]) != 40:
        raise GateError(f"could not resolve one tip for refs/heads/{base_ref}: {proc.stdout!r}")
    sha = lines[0][0]
    _emit(f"gate-merge-base: resolved {base_ref} tip {sha} once for this run")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"gate_base={sha}\n")
    return sha


def apply(*, event: str, remote: str, base: str, head: str, job: str) -> str:
    """Rebuild HEAD as `head` merged into `base` and print the tested-base line."""
    if event != "pull_request":
        # Push/schedule/dispatch: the checkout IS the commit under test. Report
        # it in the same greppable shape and touch nothing.
        sha = _rev("HEAD")
        tree = _git("rev-parse", "HEAD^{tree}").stdout.strip()
        line = f"{LOG_PREFIX}none head={sha} tree={tree} merge={sha} job={job} event={event}"
        _emit(line)
        return line
    if not base:
        raise GateError(
            "pull_request gate without a resolved base: the `changes` job's `gate_base` "
            f"output is empty (head {head or '<empty>'}); refusing to test the stale merge ref"
        )
    if not head:
        raise GateError(f"pull_request gate without a head SHA (base {base})")
    checked_out = _rev("HEAD")
    # The checkout's own first parent is the base GitHub last computed the
    # merge ref against; logged below so a reader sees how stale it was.
    parent = _git("rev-parse", "--verify", "--quiet", "HEAD^1", check=False)
    stale_base = parent.stdout.strip() if parent.returncode == 0 else ""
    _fetch(remote, base, head)
    if not _merge_base(base, head):
        raise GateError(
            f"no merge base between base {base} and head {head} even after unshallowing"
        )
    _git("checkout", "--quiet", "--detach", base)
    merge = _git(
        *MERGE_IDENTITY,
        "merge",
        "--no-ff",
        "--no-edit",
        "--quiet",
        "-m",
        f"full-suite gate: merge {head} into {base}",
        head,
        check=False,
    )
    if merge.returncode != 0:
        conflicts = _git("diff", "--name-only", "--diff-filter=U", check=False).stdout.split()
        _git("merge", "--abort", check=False)
        detail = (
            f"conflicting paths: {' '.join(conflicts)}"
            if conflicts
            else f"{merge.stdout.strip()} {merge.stderr.strip()}".strip()
        )
        raise GateError(
            f"head {head} does not merge cleanly into current base {base} ({detail}). "
            "Rebase or `gh pr update-branch`, then re-arm the gate."
        )
    sha = _rev("HEAD")
    parents = _git("rev-list", "--parents", "-n", "1", "HEAD").stdout.split()[1:]
    if parents != [base, head]:
        raise GateError(
            f"rebuilt merge {sha} has parents {parents}, expected base {base} and head {head}"
        )
    tree = _git("rev-parse", "HEAD^{tree}").stdout.strip()
    _emit(
        f"gate-merge-base: checkout was {checked_out} (first parent {stale_base or 'none'}); "
        f"rebuilt as {sha} on base {base}"
        + (" — the merge ref was STALE" if stale_base and stale_base != base else "")
    )
    line = f"{LOG_PREFIX}{base} head={head} tree={tree} merge={sha} job={job} event={event}"
    _emit(line)
    return line


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("resolve", "apply"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", ""))
        cmd.add_argument("--remote", default="origin")
    sub.choices["resolve"].add_argument("--base-ref", default=os.environ.get("GITHUB_BASE_REF", ""))
    sub.choices["resolve"].add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))
    sub.choices["apply"].add_argument("--base", default=os.environ.get("GATE_BASE", ""))
    sub.choices["apply"].add_argument("--head", default=os.environ.get("GATE_HEAD", ""))
    sub.choices["apply"].add_argument("--job", default=os.environ.get("GITHUB_JOB", ""))
    args = parser.parse_args(argv)
    try:
        if args.command == "resolve":
            resolve(
                event=args.event,
                remote=args.remote,
                base_ref=args.base_ref,
                output=args.github_output,
            )
        else:
            apply(
                event=args.event, remote=args.remote, base=args.base, head=args.head, job=args.job
            )
    except GateError as exc:
        # One annotation a reader sees on the run summary, naming the SHAs.
        print(f"::error title=Merge gate base::{exc}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
