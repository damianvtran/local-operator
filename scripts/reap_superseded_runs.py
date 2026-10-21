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
3. For a pull_request run the PULL REQUEST's head is authoritative.
   `head_sha == pull_request.head.sha` means KEEP, at any age, whether or not
   the branch resolves -- that is the fork-PR case, where the fork's branch
   404s here forever (a review round found an earlier version let the
   abandonment bound cancel a live fork PR for exactly that reason).
   `head_sha != pull_request.head.sha` means the run has been superseded, and
   the branch tip then only corroborates: if the tip still points at the run's
   commit, the run is kept on that one reading.
   MEASURED, not assumed: across 19 sampled pull_request runs, `head_sha` is
   never a synthetic `refs/pull/<n>/merge` commit -- it tracks the PR head. An
   earlier draft of this guard was written on the opposite belief and would
   have been a rationale nobody had checked.
4. Push runs are compared against the branch tip, and only when the branch
   still exists. A deleted branch means the run cannot be superseded by a
   newer one on that ref; it is left alone rather than guessed at.
5. A grace period. A run younger than --min-age-minutes is never touched,
   because a head move and the run it triggers arrive seconds apart and a
   sweep racing that gap would cancel the run that is about to be current.
6. ABANDONMENT, bounded hard, and ONLY for a run no reading can resolve. A
   run queued longer than --max-queue-hours (default 24) whose branch, PR head
   and pull_requests entry are all unavailable is cancelled, because nothing
   queues for a day and then runs. Age NEVER overrules a resolved run: a
   current head wins at 37 days. This rule exists because a live dry run found exactly that:
   two `pull_request` runs from 2026-09-13, still `queued` 37 days later, on
   branches that no longer exist (404) and with no `pull_requests` entry in
   either the list or the single-run view -- so the supersession rules could
   never resolve them and they would have sat in the queue forever. The
   threshold is deliberately far above any observed queue time (the worst
   measured pending age at the 2026-10-19 peak was a median of 25.8 minutes).
   AGE IS READ FROM `updated_at`, NOT `created_at`: a re-run inherits the run
   object's `created_at` (measured on 39498542825 -- object 21:13:02Z, attempt
   2 at 22:02:55Z), so an age taken from it would put every re-run of an old
   run past the bound on its first sweep.

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


class ApiError(RuntimeError):
    """A read the sweep depends on could not be performed."""


def gh_api(path: str, *, method: str = "GET") -> Any:
    """One GitHub API call. RAISES on failure; an empty body is a valid null.

    C-3/C-4: success must be judged by the process exit status, not by whether
    the body parses. A cancel returns 202 with an EMPTY body, so a helper that
    parsed the body and returned None on failure reported every successful
    cancel as "refused"; and a failed LIST read looked identical to an empty
    queue, so a dead sweep would have exited 0 and looked healthy. Reads that
    matter now raise, and the caller decides whether that is fatal.
    """
    cmd = ["gh", "api", path]
    if method != "GET":
        cmd += ["-X", method]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:200]
        raise ApiError(f"{method} {path} failed rc={proc.returncode}: {detail}")
    body = (proc.stdout or "").strip()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def queued_runs(repo: str) -> list[dict[str, Any]]:
    """Every queued run, across pages (Q6: one page silently capped at 100)."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        data = gh_api(
            f"repos/{repo}/actions/runs?status=queued&per_page={SEARCH_LIMIT}&page={page}"
        )
        if data is None:
            raise ApiError(
                "the queued-run listing returned no body; refusing to read that as an empty queue"
            )
        batch = list(data.get("workflow_runs") or [])
        out.extend(batch)
        if len(batch) < SEARCH_LIMIT:
            return out
        page += 1
        if page > 20:          # a queue deeper than 2000 is not this script's problem
            return out


def pr_head_sha(repo: str, number: int) -> str | None:
    data = gh_api(f"repos/{repo}/pulls/{number}")
    return ((data or {}).get("head") or {}).get("sha")


def branch_tip(repo: str, branch: str) -> str | None:
    data = gh_api(f"repos/{repo}/branches/{branch}")
    return ((data or {}).get("commit") or {}).get("sha")


def age_minutes(run: dict[str, Any]) -> float:
    """How long this run has been in its CURRENT queue residency.

    Q3: `created_at` is the RUN OBJECT's birth time, which a re-run inherits.
    Measured on run 39498542825: the object reports created_at 21:13:02Z while
    attempt 2 was created at 22:02:55Z, so a re-run of an old run carries the
    old age -- and would have been past the abandonment bound on its first
    sweep, bypassing both the grace period and the current-tip keep, on every
    retry. `updated_at` tracks the latest queue entry (the re-queue), so it is
    the residency proxy this needs; the API exposes no queued_at and
    run_started_at is null while queued.

    Raises ValueError on an unreadable stamp; the caller keeps that run rather
    than aborting the sweep (Q5).
    """
    for key in ("updated_at", "created_at"):
        raw = run.get(key)
        if isinstance(raw, str) and raw:
            try:
                stamp = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now - stamp).total_seconds() / 60.0
    raise ValueError(f"no readable timestamp: updated_at={run.get('updated_at')!r} "
                     f"created_at={run.get('created_at')!r}")


def is_protected(branch: str) -> bool:
    return branch == "main" or branch.startswith(PROTECTED_PREFIXES[1])


def _abandoned_or(age: float, max_queue_hours: float, unresolved: str) -> tuple[bool, str]:
    """The abandonment bound applies ONLY to a run no rule can resolve.

    C-5: the bound used to be evaluated before the event rules, which meant it
    cancelled `schedule`, `workflow_dispatch`, `issue_comment` and
    `pull_request_target` runs that the docstring promises to leave alone, and
    it cancelled a CURRENT pull-request run that had merely been queued for a
    day before either head reading was consulted. Resolve first; if the run is
    genuinely unresolvable, then age decides.
    """
    if age > max_queue_hours * 60:
        return True, (f"queued {age / 60:.1f} h past the {max_queue_hours:g}-h abandonment "
                      f"bound and unresolvable ({unresolved}) -- nothing queues for a day "
                      f"and then runs")
    return False, f"unresolvable ({unresolved}) -- kept, inside the abandonment bound"


def decide(repo: str, run: dict[str, Any], min_age: float,
           max_queue_hours: float = 24.0) -> tuple[bool, str]:
    """(should_cancel, reason) for one queued run. Pure: makes no writes.

    The order is the safety argument. Status, protection and the grace period
    exclude a run outright. An event that cannot be superseded is excluded
    outright. Only then is the run RESOLVED, and the resolution is authoritative:

    * a readable PR head that equals the run's commit means KEEP, full stop.
      This is the fork-PR hole the QA round found: a fork's branch 404s in this
      repository, and an earlier version treated "cannot read the tip" as
      unresolvable and then let the abandonment bound cancel a run whose PR
      head was readable and CURRENT.
    * a readable branch tip that equals the run's commit means KEEP for a push
      run, and for a PR run it corroborates the PR head -- one reading saying
      current is enough to keep.

    The abandonment bound applies ONLY to a run no reading can resolve. Age
    never overrules a resolved run.
    """
    branch = run.get("head_branch") or ""
    event = run.get("event") or ""
    head = run.get("head_sha")

    if run.get("status") != "queued":
        return False, "not queued"
    if is_protected(branch):
        return False, f"protected branch {branch!r}"

    try:
        age = age_minutes(run)
    except ValueError:
        return False, "unreadable timestamps -- kept rather than guessed (Q5)"

    if age < min_age:
        return False, f"inside the {min_age:g}-minute grace period"

    if event == "pull_request":
        prs = run.get("pull_requests") or []
        if not prs:
            return _abandoned_or(age, max_queue_hours,
                                 "PR run carries no pull_requests entry")
        number = prs[0].get("number")
        try:
            current = pr_head_sha(repo, number) if isinstance(number, int) else None
        except ApiError as exc:
            return _abandoned_or(age, max_queue_hours, f"API read failed: {exc}")
        if not current:
            return _abandoned_or(age, max_queue_hours,
                                 f"cannot read PR #{number}'s current head")
        if head == current:
            # Authoritative: the run is about to be current. Never cancellable,
            # at any age, and independent of whether the branch resolves.
            return False, f"PR #{number} head is current"
        try:
            tip = branch_tip(repo, branch) if branch else None
        except ApiError:
            tip = None
        if tip == head:
            return False, (f"branch tip still points at {str(head)[:9]} -- "
                           f"one reading says current")
        return True, f"PR #{number} moved past {str(head)[:9]} (head {current[:9]})"

    if event == "push":
        try:
            tip = branch_tip(repo, branch) if branch else None
        except ApiError as exc:
            return _abandoned_or(age, max_queue_hours, f"API read failed: {exc}")
        if not tip:
            return _abandoned_or(age, max_queue_hours, f"branch {branch!r} is gone")
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

    try:
        runs = queued_runs(args.repo)
    except ApiError as exc:
        print(f"FAILED to read the queue: {exc}", file=sys.stderr, flush=True)
        return 2
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
        # C-6: re-read the status immediately before writing. A run can start
        # between the listing and this call, and cancelling work that has
        # already begun is the one thing this script must never do.
        try:
            fresh = gh_api(f"repos/{args.repo}/actions/runs/{run_id}")
        except ApiError as exc:
            print(f"  KEEP   {label}: re-read failed ({exc})", flush=True)
            kept += 1
            continue
        if (fresh or {}).get("status") != "queued":
            print(f"  KEEP   {label}: no longer queued ({(fresh or {}).get('status')})", flush=True)
            kept += 1
            continue
        try:
            gh_api(f"repos/{args.repo}/actions/runs/{run_id}/cancel", method="POST")
            ok = True
        except ApiError as exc:
            ok = False
            print(f"         cancel refused: {exc}", flush=True)
        cancelled += 1
        print(f"  CANCEL {label}: {reason} -> {'ok' if ok else 'refused'}", flush=True)

    verb = "would cancel" if args.dry_run else "cancelled"
    print(f"{verb} {cancelled}, kept {kept}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
