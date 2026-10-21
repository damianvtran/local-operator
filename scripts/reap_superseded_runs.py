#!/usr/bin/env python3
"""Cancel QUEUED runs whose commit is no longer the tip that would run again.

Why this exists
---------------
`concurrency: cancel-in-progress: true` supersedes a run when a NEW run starts
on the same group. It cannot help the case this script covers: a run that is
still QUEUED when its branch moves, because `cancel-in-progress` only fires
when the replacement run is created, and a queued run can sit for a long time
under a saturated runner pool. Those queued-but-superseded runs are the ones
that starve a release gate: they hold a place in the queue for work whose
result nobody will read.

What it will NEVER touch -- the safety rules, in order of how much damage each
one prevents:

1. Only runs in status `queued`. A run that has STARTED is doing work whose
   artifacts or logs someone may still need, and cancelling it wastes the work
   already done. The bug this addresses is a QUEUE problem, so the queue is
   the only place it acts.
2. Never a protected branch. `main` and `release-next-*` are release-bearing:
   a gate that can be cancelled is not a gate. (The release gate workflow
   already uses a per-SHA group precisely so a later run cannot evict it.)
3. A PR run is cancelled only when BOTH independent readings agree that it is
   superseded: `run.head_sha != pull_request.head.sha` AND
   `run.head_sha != branch_tip`. MEASURED on this repository's own run history
   (2026-10-20, four sampled pull_request runs): `head_sha` tracks the PR head
   and the branch tip, NOT a synthetic merge commit -- three sampled runs had
   head == tip == PR head, and one genuinely superseded run
   (39514179639, head b54a5b775 against tip/head e7499dbaa) differed from both.
   The earlier assumption that `head_sha` is always the `refs/pull/<n>/merge`
   commit is NOT what this API view returns, and writing the guard on that
   assumption alone would have been a rationale nobody had checked. Requiring
   both readings costs nothing when they agree (they did in every sample) and
   refuses to act when either is unresolvable.
4. Push runs are compared against the branch tip, and only when the branch
   still exists. A deleted branch means the run cannot be superseded by a
   newer one on that ref; it is left alone rather than guessed at.
5. A grace period. A run younger than --min-age-minutes is never touched,
   because a head move and the run it triggers arrive seconds apart and a
   sweep racing that gap would cancel the run that is about to be current.
6. ABANDONMENT, bounded hard. A run that has been QUEUED for longer than
   --max-queue-hours (default 24) is cancelled regardless of whether its
   branch or PR can still be resolved, because nothing queues for a day and
   then runs. This rule exists because a live dry run found exactly that:
   two `pull_request` runs from 2026-09-13, still `queued` 37 days later, on
   branches that no longer exist (404) and with no `pull_requests` entry in
   either the list or the single-run view -- so the supersession rules could
   never resolve them and they would have sat in the queue forever. The
   threshold is deliberately far above any observed queue time (the worst
   measured pending age at the 2026-10-19 peak was a median of 25.8 minutes),
   so it cannot fire on a live fork PR waiting behind a saturated pool.

Idempotence: the only write is a cancel of a run this same invocation decided
is superseded. Cancelling an already-terminal run is a no-op, and the script
re-derives every decision from live state, so running it twice changes
nothing the second time.

Usage:
    python scripts/reap_superseded_runs.py --repo OWNER/NAME [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from typing import Any

PROTECTED_PREFIXES = ("main", "release-next")
SEARCH_LIMIT = 100


def gh_api(path: str, *, method: str = "GET") -> Any:
    """One GitHub API call through the authenticated `gh` CLI."""
    cmd = ["gh", "api", path]
    if method != "GET":
        cmd += ["-X", method]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout or "null")
    except json.JSONDecodeError:
        return None


def queued_runs(repo: str) -> list[dict[str, Any]]:
    data = gh_api(f"repos/{repo}/actions/runs?status=queued&per_page={SEARCH_LIMIT}")
    return list((data or {}).get("workflow_runs") or [])


def pr_head_sha(repo: str, number: int) -> str | None:
    data = gh_api(f"repos/{repo}/pulls/{number}")
    return ((data or {}).get("head") or {}).get("sha")


def branch_tip(repo: str, branch: str) -> str | None:
    data = gh_api(f"repos/{repo}/branches/{branch}")
    return ((data or {}).get("commit") or {}).get("sha")


def age_minutes(created_at: str) -> float:
    stamp = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    now = datetime.datetime.now(datetime.timezone.utc)
    return (now - stamp).total_seconds() / 60.0


def is_protected(branch: str) -> bool:
    return branch == "main" or branch.startswith(PROTECTED_PREFIXES[1])


def decide(repo: str, run: dict[str, Any], min_age: float,
           max_queue_hours: float = 24.0) -> tuple[bool, str]:
    """(should_cancel, reason) for one queued run. Pure: makes no writes."""
    run_id = run.get("id")
    branch = run.get("head_branch") or ""
    event = run.get("event") or ""
    created = run.get("created_at") or ""

    if run.get("status") != "queued":
        return False, "not queued"
    if is_protected(branch):
        return False, f"protected branch {branch!r}"
    age = age_minutes(created)
    if age < min_age:
        return False, f"inside the {min_age:g}-minute grace period"
    if age > max_queue_hours * 60:
        return True, (f"queued {age / 60:.1f} h, past the {max_queue_hours:g}-h "
                      f"abandonment bound -- nothing queues for a day and then runs")

    head = run.get("head_sha")

    if event == "pull_request":
        prs = run.get("pull_requests") or []
        if not prs:
            return False, "PR run with no pull_requests entry — cannot verify the head"
        number = prs[0].get("number")
        current = pr_head_sha(repo, number) if isinstance(number, int) else None
        if not current:
            return False, f"cannot read PR #{number}'s current head"
        tip = branch_tip(repo, branch) if branch else None
        if not tip:
            return False, f"cannot read branch tip for {branch!r} — refusing to act on one reading"
        if head != current and head != tip:
            return True, (f"PR #{number} moved past {str(head)[:9]} "
                          f"(head {current[:9]}, tip {tip[:9]})")
        return False, f"PR #{number} head is current"

    if event == "push":
        tip = branch_tip(repo, branch) if branch else None
        if not tip:
            return False, f"branch {branch!r} is gone — not guessing"
        if head != tip:
            return True, f"branch {branch!r} tip moved {str(head)[:9]} -> {tip[:9]}"
        return False, f"branch {branch!r} tip is current"

    return False, f"event {event!r} is not superseded by definition"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="OWNER/NAME")
    ap.add_argument("--dry-run", action="store_true", help="report decisions, cancel nothing")
    ap.add_argument("--min-age-minutes", type=float, default=5.0)
    ap.add_argument("--max-queue-hours", type=float, default=24.0,
                    help="cancel a run queued longer than this, unresolvable or not")
    args = ap.parse_args()

    runs = queued_runs(args.repo)
    print(f"{len(runs)} queued run(s) on {args.repo}", flush=True)
    if not runs:
        return 0

    cancelled = kept = 0
    for run in runs:
        run_id = run.get("id")
        should, reason = decide(args.repo, run, args.min_age_minutes, args.max_queue_hours)
        label = f"run {run_id} ({run.get('name', '?')[:40]}, {run.get('head_branch')})"
        if not should:
            kept += 1
            print(f"  KEEP   {label}: {reason}", flush=True)
            continue
        if args.dry_run:
            cancelled += 1
            print(f"  WOULD  {label}: {reason}", flush=True)
            continue
        ok = gh_api(f"repos/{args.repo}/actions/runs/{run_id}/cancel", method="POST") is not None
        cancelled += 1
        print(f"  CANCEL {label}: {reason} -> {'ok' if ok else 'refused'}", flush=True)

    verb = "would cancel" if args.dry_run else "cancelled"
    print(f"{verb} {cancelled}, kept {kept}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
