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
2. Never `release-next-*`. A release branch's gate is the verdict the release is
   cut on, and a gate that can be cancelled is not a gate. (The release gate
   workflow already uses a per-SHA group precisely so a later run cannot evict
   it.) This is unchanged.

2a. `main` is NOT covered by rule 2 any more, and the reversal is narrow: only a
   QUEUED `push` run whose commit is an ANCESTOR of the current `main` tip, and
   which NO tag points at, may be cancelled. The reason rule 2 protected `main`
   at all was that a `main` run is the post-merge CI record for a merge commit;
   the reason it no longer can be absolute is measured: `ci.yml`'s concurrency
   group is per-SHA for `main`, so a queued run for an ancestor is never evicted
   by the newer one -- it just waits, holding a runner slot, in a pool that was
   observed at 20 concurrent jobs with 38 runs queued. On 2026-10-29 sixteen
   such runs were cancelled by hand and every one was proven an ancestor of the
   newest with `git merge-base --is-ancestor`. The record for the TIP is the one
   anybody reads; the record for an ancestor is superseded by definition.

2b. NEVER a tag target. Two of those sixteen hand-cancelled runs were the
   commits `v0.91.0` and `v0.91.1` were tagged on (`20d8c7c3fc`, `85f3b9097b`).
   "Older than the newest queued run" alone would have destroyed the CI evidence
   for a PUBLISHED release, and it is not recoverable: a cancelled run's jobs
   cannot be un-cancelled, and re-running one on a tag target re-runs code that
   may no longer be the branch tip. The tag list is read once per sweep from the
   API; if it cannot be read, EVERY `main` run is kept (rule 6's fail-closed
   shape, applied to a read this rule depends on).

2c. Ancestry is PROVEN, not inferred. `head_sha != tip` is not enough: a
   force-push or a rebase onto `main` can leave a queued run's commit outside
   the tip's history, and such a commit is not "superseded post-merge CI", it is
   a commit the branch no longer contains. The comparison is the API's
   `compare/{run_sha}...{tip}`, whose `ahead`/`identical` status is the same
   reading `git merge-base --is-ancestor` gives. A transient failure on that read
   keeps the run.

2d. Only `push`. A `workflow_dispatch` run on `main` was asked for by a person,
   not produced by the merge, and is kept whatever its age.
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
   run queued longer than --max-queue-hours (default 24) is cancelled when the
   reading that would supersede it is CONFIRMED absent: a 404 on the branch or
   on the PR head, or no `pull_requests` entry to ask about. Note the scope --
   a PR run whose PR head 404s is bounded without the branch ever being read,
   so "all three unavailable" is the common shape, not a requirement. Any
   UNKNOWN failure keeps the run instead. Age NEVER overrules a resolved run:
   a current head wins at 37 days. This rule exists because a live dry run
   found exactly that:
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

Outcomes
--------
Each queued run ends in exactly one of:

  KEEP    a rule protected it (protected branch, current head, grace period)
  CANCEL  cancelled, confirmed by a successful POST
  WOULD   selected in --dry-run
  REFUSE  the write failed in a way the NEXT sweep might fix (403, 5xx,
          rate limit) -- retrying is meaningful
  GHOST   the write failed with 409. Terminal: GitHub reports the run as
          `queued` but refuses ever to cancel it. Measured on runs
          34749070299 and 34747986252 -- 37 days old, branches 404, and both
          with `jobs: []`, i.e. never scheduled, so they hold no runner slot.
          Retrying every 20 minutes forever would be noise, not progress.

`cancelled + kept + refused + ghosts == the number of queued runs examined`.

Usage:
    python scripts/reap_superseded_runs.py --repo OWNER/NAME [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from typing import Any

PROTECTED_PREFIXES = ("release-next",)
#: The post-merge branch. Its QUEUED push runs are eligible under rule 2a-2d;
#: everything else about it is still protected.
PROTECTED_MAIN = "main"
SEARCH_LIMIT = 100


class ApiError(RuntimeError):
    """A read the sweep depends on could not be performed.

    `status` distinguishes a CONFIRMED absence (404 -- the branch or PR really
    is gone, which is what makes a run unresolvable) from an UNKNOWN failure
    (5xx, network, rate limit). The abandonment bound may act on the first and
    must never act on the second, because a transient outage is not evidence
    that anything is gone.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


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
        # Parse whatever status `gh` reports rather than matching a fixed list.
        # A hardcoded tuple silently dropped 409, which made the GHOST branch
        # in main() unreachable in production while its unit tests passed --
        # they injected ApiError(status=409) directly and so never exercised
        # this line. `gh` prints "... (HTTP 409)" and some subcommands "...
        # (409)", so both shapes are accepted.
        match = re.search(r"HTTP (\d{3})", detail) or re.search(r"\((\d{3})\)", detail)
        status = int(match.group(1)) if match else None
        raise ApiError(f"{method} {path} failed rc={proc.returncode}: {detail}", status)
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
        if page > 20:  # a queue deeper than 2000 is not this script's problem
            return out


def pr_head_sha(repo: str, number: int) -> str | None:
    """The PR's current head, or None when the PR CONFIRMED does not exist."""
    try:
        data = gh_api(f"repos/{repo}/pulls/{number}")
    except ApiError as exc:
        if exc.status == 404:
            return None
        raise
    return ((data or {}).get("head") or {}).get("sha")


def branch_tip(repo: str, branch: str) -> str | None:
    """The branch's tip, or None when the branch CONFIRMED does not exist.

    A 404 is the only thing that means "gone". Any other failure raises, so the
    caller can keep the run instead of treating an outage as a deletion (Q2).
    """
    try:
        data = gh_api(f"repos/{repo}/branches/{branch}")
    except ApiError as exc:
        if exc.status == 404:
            return None
        raise
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
    raise ValueError(
        f"no readable timestamp: updated_at={run.get('updated_at')!r} "
        f"created_at={run.get('created_at')!r}"
    )


def tag_targets(repo: str) -> frozenset[str]:
    """Every commit a tag points at, across pages (rule 2b).

    `repos/{repo}/tags` reports the COMMIT an annotated tag resolves to (not the
    tag object), which is exactly the question rule 2b asks: is this SHA the
    published record of a release? A paginated listing is used rather than
    `git ls-remote --tags` because this script is stdlib + `gh` only and must
    work without a checkout.

    RAISES on failure. The caller keeps every `main` run rather than treating an
    unreadable tag list as an empty one -- the difference between the two is
    whether a release's CI record can be cancelled.
    """
    out: set[str] = set()
    page = 1
    while True:
        data = gh_api(f"repos/{repo}/tags?per_page={SEARCH_LIMIT}&page={page}")
        if data is None:
            raise ApiError("the tag listing returned no body")
        if not isinstance(data, list):
            # Not a shape this endpoint produces; treat it as unreadable rather
            # than iterating it and cancelling what the caller cannot vouch for.
            raise ApiError(f"the tag listing returned {type(data).__name__}, not a list")
        batch = list(data)
        for entry in batch:
            sha = (entry.get("commit") or {}).get("sha")
            if sha:
                out.add(sha)
        page += 1
        if len(batch) < SEARCH_LIMIT:
            return frozenset(out)
        if page > 20:  # more than 2000 tags is not this script's problem
            return frozenset(out)


def is_ancestor(repo: str, older: str, newer: str) -> bool:
    """Whether `older` is an ancestor of `newer` (rule 2c).

    `compare/{base}...{head}` answers this without walking the graph locally:
    `ahead` means head contains base, `identical` means they are the same
    commit. `behind` and `diverged` both mean the queued run's commit is not in
    the tip's history, which keeps the run.

    RAISES on failure; the caller keeps the run.
    """
    data = gh_api(f"repos/{repo}/compare/{older}...{newer}")
    return ((data or {}).get("status")) in ("ahead", "identical")


def is_protected(branch: str) -> bool:
    return branch.startswith(PROTECTED_PREFIXES)


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
        return True, (
            f"queued {age / 60:.1f} h past the {max_queue_hours:g}-h abandonment "
            f"bound and unresolvable ({unresolved}) -- nothing queues for a day "
            f"and then runs"
        )
    return False, f"unresolvable ({unresolved}) -- kept, inside the abandonment bound"


def _post_merge_decision(
    repo: str,
    run: dict[str, Any],
    head: str | None,
    tag_targets: frozenset[str] | None,
) -> tuple[bool, str]:
    """The narrow reversal of "never touch `main`" (rules 2a-2d).

    Every guard here is a separate reason the old blanket rule existed, and each
    one returns KEEP on its own. The only path to CANCEL is: a queued push run,
    a readable tag list that does not name this commit, a readable `main` tip
    that is not this commit, and a proof that this commit is in that tip's
    history.
    """
    if run.get("event") != "push":
        return False, f"{PROTECTED_MAIN} run from {run.get('event')!r} -- kept (rule 2d)"
    if tag_targets is None:
        return False, (
            "tag targets unreadable -- every "
            f"{PROTECTED_MAIN} commit could be a release (rule 2b)"
        )
    if head and head in tag_targets:
        return False, (
            f"{str(head)[:9]} is a release tag target -- its CI record is the "
            "published evidence (rule 2b)"
        )
    try:
        tip = branch_tip(repo, PROTECTED_MAIN)
    except ApiError as exc:
        return False, f"transient read failure ({exc.status or 'unknown'}) -- kept, not guessed"
    if not tip:
        return False, f"{PROTECTED_MAIN} has no readable tip -- kept"
    if tip == head:
        return False, f"{PROTECTED_MAIN} tip is this commit"
    try:
        ancestor = is_ancestor(repo, str(head), tip)
    except ApiError as exc:
        return False, f"ancestry unreadable ({exc.status or 'unknown'}) -- kept (rule 2c)"
    if not ancestor:
        return False, (
            f"not in the history of {PROTECTED_MAIN}'s tip ({tip[:9]}) -- a "
            "force-push or rebase, kept (rule 2c)"
        )
    return True, (
        f"superseded post-merge CI: {PROTECTED_MAIN} moved {str(head)[:9]} -> "
        f"{tip[:9]} and no tag points at {str(head)[:9]}"
    )


def decide(
    repo: str,
    run: dict[str, Any],
    min_age: float,
    max_queue_hours: float = 24.0,
    tag_targets: frozenset[str] | None = None,
) -> tuple[bool, str]:
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

    `main` is the one branch whose verdict is not an event rule: its queued push
    runs are decided by `_post_merge_decision` (rules 2a-2d), which is where the
    tag-target exclusion lives.
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

    if branch == PROTECTED_MAIN:
        return _post_merge_decision(repo, run, head, tag_targets)

    if event == "pull_request":
        prs = run.get("pull_requests") or []
        if not prs:
            return _abandoned_or(age, max_queue_hours, "PR run carries no pull_requests entry")
        number = prs[0].get("number")
        try:
            current = pr_head_sha(repo, number) if isinstance(number, int) else None
        except ApiError as exc:
            return False, f"transient read failure ({exc.status or 'unknown'}) -- kept, not guessed"
        if not current:
            return _abandoned_or(age, max_queue_hours, f"cannot read PR #{number}'s current head")
        if head == current:
            # Authoritative: the run is about to be current. Never cancellable,
            # at any age, and independent of whether the branch resolves.
            return False, f"PR #{number} head is current"
        try:
            tip = branch_tip(repo, branch) if branch else None
        except ApiError as exc:
            # The PR head already said "moved", but a transient failure on the
            # corroborating read is not permission to cancel on one signal.
            return False, f"transient tip read failure ({exc.status or 'unknown'}) -- kept"
        if tip == head:
            return False, (
                f"branch tip still points at {str(head)[:9]} -- " f"one reading says current"
            )
        return True, f"PR #{number} moved past {str(head)[:9]} (head {current[:9]})"

    if event == "push":
        try:
            tip = branch_tip(repo, branch) if branch else None
        except ApiError as exc:
            return False, f"transient read failure ({exc.status or 'unknown'}) -- kept, not guessed"
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
    ap.add_argument(
        "--max-queue-hours",
        type=float,
        default=24.0,
        help="cancel a run queued longer than this that no reading can resolve",
    )
    args = ap.parse_args()

    try:
        runs = queued_runs(args.repo)
    except ApiError as exc:
        print(f"FAILED to read the queue: {exc}", file=sys.stderr, flush=True)
        return 2
    print(f"{len(runs)} queued run(s) on {args.repo}", flush=True)
    if not runs:
        return 0

    # One read for the whole sweep (rule 2b), and only when a post-merge run is
    # actually queued: a sweep with nothing on `main` has no use for a paginated
    # listing. A failure is NOT an empty list: passing None keeps every `main`
    # run, because "we cannot see the tags" and "no commit carries a tag" are
    # the two readings that decide whether a published release's CI record can be
    # cancelled.
    needs_tags = any(
        run.get("head_branch") == PROTECTED_MAIN and run.get("event") == "push" for run in runs
    )
    tags: frozenset[str] | None = frozenset()
    if needs_tags:
        try:
            tags = tag_targets(args.repo)
        except ApiError as exc:
            tags = None
            print(
                f"WARNING: tag targets unreadable ({exc}); every {PROTECTED_MAIN} run "
                "will be kept this sweep",
                file=sys.stderr,
                flush=True,
            )

    cancelled = kept = refused = ghosts = 0
    #: (run id, sha) for every post-merge run this sweep acted on, so the
    #: decision is auditable from the log rather than from the queue's absence.
    post_merge: list[tuple[Any, str]] = []
    for run in runs:
        run_id = run.get("id")
        should, reason = decide(args.repo, run, args.min_age_minutes, args.max_queue_hours, tags)
        label = f"run {run_id} ({run.get('name', '?')[:40]}, {run.get('head_branch')})"
        if not should:
            kept += 1
            print(f"  KEEP   {label}: {reason}", flush=True)
            continue
        if args.dry_run:
            cancelled += 1
            if run.get("head_branch") == PROTECTED_MAIN:
                post_merge.append((run_id, str(run.get("head_sha"))[:9]))
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
        except ApiError as exc:
            if exc.status == 409 and "has not been queued" in str(exc):
                # THIS 409 is TERMINAL, not retryable. The message matters:
                # cancel also returns 409 "Cannot cancel a workflow run that
                # is completed", which is a different situation, so the ghost
                # label is gated on the never-queued wording rather than on
                # the bare status. Measured on runs 34749070299 and 34747986252 --
                # 37 days old, branches 404, reported `queued` by the API, and
                # both with `jobs: []`. They were never scheduled, so they hold
                # no runner slot; retrying every 20 minutes forever would be
                # noise with no possible success. Counted separately from a
                # refusal (403/5xx), which may well succeed next sweep.
                ghosts += 1
                print(
                    f"  GHOST  {label}: {reason} -> uncancellable ({exc}); "
                    f"holds no runner slot",
                    flush=True,
                )
                continue
            refused += 1
            print(f"  REFUSE {label}: {reason} -> {exc}", flush=True)
            continue
        cancelled += 1
        if run.get("head_branch") == PROTECTED_MAIN:
            post_merge.append((run_id, str(run.get("head_sha"))[:9]))
        print(f"  CANCEL {label}: {reason} -> ok", flush=True)

    verb = "would cancel" if args.dry_run else "cancelled"
    print(
        f"{verb} {cancelled}, kept {kept}, refused {refused}, ghosts {ghosts}",
        flush=True,
    )
    # The audit list for the ONE reversal this script makes (rules 2a-2d). It is
    # printed even when empty, so a reader of a quiet sweep can tell "nothing
    # was eligible" from "this rule did not run".
    print(
        f"post-merge {PROTECTED_MAIN} runs {verb}: {len(post_merge)} -- "
        + (", ".join(f"run {rid} {sha}" for rid, sha in post_merge) or "none"),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
