# Full-suite gate ownership and duplicate runs

## When to arm (and when not to)

- **Re-arm only when BOTH:** the required success is **absent**, AND **no viable queued or running run can supply it** (none exists, or it is definitively dead/invalid rather than merely slow).
- **OR** a specifically diagnosed **stuck or invalid** run needs **owner-controlled replacement** — and in that case the override is that path, named as such.
- **An older failure alongside a newer success is never a reason to re-arm.** Neither is API list order, and neither is `filter=latest` — **the check-run list is not gate authority**; the merge state is.
- A pending run's correct response is to **wait**, and to inspect only if it looks stuck.

GitHub's [required-check documentation](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
requires success (or neutral/skipped) on the latest relevant commit, not success
for every historical check-run object. Pending is not success. The test merge
commit takes precedence when it has checks; otherwise checks on the head apply.
A check and a commit status with the same name must **both** pass, and the
configured source app matters. A `workflow_dispatch` check is not an eligible
replacement for a pull-request required check. Preserve the required names and
use the existing label trigger, never a skipped aggregate or a bypass.

### Repository evidence: PR #3654

Read on 2026-11-09 from GitHub, head
`4e9fe6e6bc70cbe959737348cb8fe080565bb2a7`:

| Full-suite run | Merge-gate check ID | Conclusion | Completed (UTC) |
| --- | --- | --- | --- |
| [42559874441](https://github.com/damianvtran/local-operator/actions/runs/42559874441) | 128984385438 | failure | 04:43:20 |
| [42563549134](https://github.com/damianvtran/local-operator/actions/runs/42563549134) | 128998420845 | success | 05:41:03 |
| [42569246334](https://github.com/damianvtran/local-operator/actions/runs/42569246334) | 129020789767 | success | 06:51:20 |

[PR #3654](https://github.com/damianvtran/local-operator/pull/3654) merged at
06:54:43 UTC as `3b65500045a48e16f5a20233e2a7558745dddc79` with the older
failure still present. Reproduction reads (read-only, no gate is armed):

```sh
gh api 'repos/damianvtran/local-operator/commits/4e9fe6e6/check-runs?filter=all&per_page=100' \
  --jq '.check_runs[] | select(.name == "Merge gate (full suite)") | {id,status,conclusion,started_at,completed_at,details_url}'
gh pr view 3654 --json headRefOid,mergedAt,mergeCommit
```

The incident observer reported `mergeStateStatus=BLOCKED` while the newest
aggregate was queued (06:45:52 UTC), then a successful merge after its success.
The same observer reported `failed=4 pending=0` immediately before the merge:
those retained historical failures did not require another re-arm. That
historical pending snapshot and count are supplied incident evidence, not recoverable
from today's completed-run API. The API confirms all three final conclusions
and the merge time; it does not retrospectively expose the earlier merge state.
The rule "re-arm because a stale failure coexists with green" was wrong.
Together these support **newest relevant gate pending**, not **stale failure
beside green**, as the cause in this incident. They do not prove that pending is
the only possible merge blocker (reviews, conflicts, and other required checks
still matter).

Do not turn this into a universal "take the last REST row by name" algorithm.
GitHub's [Checks REST API](https://docs.github.com/en/rest/checks/runs#list-check-runs-for-a-git-reference)
describes `filter=latest` as returning the most recent check runs, but in this
case it still returns all three, in distinct check suites. The GraphQL status
rollup also retains all three. Raw counts such as `failed=4 pending=0` count
historical objects, not unsatisfied required contexts. GitHub also
[warns against duplicate job names across workflows](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches):
they can produce ambiguous required-check results. The documented latest-commit
rule and this repository's observation must not be overstated as a documented
cross-suite tie-breaking guarantee.

## Pre-arm guard and bounded concurrency

`make merge-gate PR=<n>` now refuses before **any label write** when an eligible
full-suite PR run on the same branch/head is `queued`, `in_progress`, `pending`,
`waiting` or `requested`. It prints the existing run ID/URL. Each lookup filters
by workflow, PR event, branch, head and status; fetching one matching result is
an existence check, not a potentially incomplete slice of historical runs. This
is **advisory and non-atomic**: it reduces redundant queued runs but does not
guarantee their absence, and it is not a lock. Completed historical successes or failures
do not trigger this active-run guard.

Recovery is ordered: **retry (bounded) → proceed with warning on transient →
override if the lookup is unusable or a diagnosed stuck run needs replacement →
refuse only on positive authentication/permission failure**. Each status lookup
gets at most **3 attempts**, with **1 second** between transient failures, each
request capped at **5 seconds**, within a **17-second total advisory budget**
(shared across all statuses). These are a small operator-wait policy budget,
not a measured GitHub latency percentile: 3 × 5 + 2 × 1 = 17 seconds allows two
short recovery opportunities without making an advisory a prolonged outage gate.
The remaining budget caps each request and wait; exhausted budget follows the
transient warning-and-proceed path. Network errors, timeouts, 5xx and rate limits
are transient. Positive authentication/permission failures (401/403, except an
explicit rate-limit response) stop immediately without label changes.

Unusable/malformed or unrecognised responses stop automatic arming without label
changes, but are not a permanent inability to gate: the lane owner inspects the
failure and uses the explicit manual override below. That override is the **only**
path through an unusable advisory response; there is no silent fallback. The
warning on exhausted transient retries also names this manual recovery path.

The lane owner uses this explicit, visible manual override when the advisory
cannot answer, or for owner-controlled replacement of a specifically diagnosed
stuck or invalid run (not one that is merely pending or slow):

```sh
make merge-gate PR=<n> I_KNOW_A_GATE_RUN_EXISTS=1
```

The override is evaluated before any existing-gate lookup and prints a warning
that the lookup did not run (was skipped), rather than implying it passed. It
therefore works even when that API is unreachable. It does not cancel anything
or waive any required check. This is **not an atomic lock**: concurrent callers
can both pass before either label is added, and simultaneous direct label writes
can still produce two runs. Labels added outside the target bypass the guard.
Native concurrency remains the backstop. Nothing new is manually enforced at
merge time.

The existing concurrency block is unchanged, including the fixed main group
`full-suite-main`, per-ref PR/dispatch groups and `cancel-in-progress: false`.
Same-head repeat labels already share a group; adding a head-SHA key would not
improve that bound and would increase parallelism across heads. The improvement
is refusing redundant arms before another suite is requested, not changing
GitHub's scheduling rules. Shards and other refs are untouched.

GitHub's [concurrency semantics](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
allow **one running plus one concurrency-pending run** per group by default:

- An arrival replaces (cancels) the existing **concurrency-pending** run. The
  newest request for that same head gets the pending slot.
- With `cancel-in-progress: false`, an **in-progress** run is never canceled;
  another arrival waits behind it. Further arrivals replace that waiting run.
- A workflow whose *jobs* are waiting for runners may already hold the running
  concurrency slot. A generic `queued` display is not proof it can be replaced.
- A pending replacement is not a successful check. This mechanism does not
  manufacture green or cancel a running gate, and it does not promise one run
  total. At most one running plus one pending duplicate remains.

## Owner-reserved queue convention

**The owning lane is `lane-<PR number>` for a PR, or `lane-<ref name>` otherwise;
`owner=<actor>` is the account that armed it.** All sessions working on that PR
share the lane. A lane owns the run it armed, not a global runner position; main
and other refs retain independent queues. The workflow's `run-name` exposes
owner, lane, `arm/re-arm:<label>` (or the dispatch/push/schedule event), head SHA
and attempt before a runner starts:

```sh
gh run list --workflow full-suite.yml --limit 30 \
  --json databaseId,displayTitle,event,headSha,status,conclusion,url
```

A labeled payload cannot tell first-arm from re-arm without fetching mutable
history, so `arm/re-arm` deliberately names both rather than claiming a false
classification. `attempt` distinguishes a GitHub rerun. For a human-readable
session mapping, record `Gate owner: <lane name> / <session ID>` in the PR's
validation evidence; the stable PR-number lane remains visible in run lists.

**Arm only the final reviewed head; reuse the lane's queued/running gate; do not
re-arm a green head to erase historical failures; never cancel another lane's
run or bypass a required check.** This is an arming convention, not a new manual
merge-time approval. Existing required checks remain the automatic merge guard.

The guard lives in the existing Makefile target rather than new persistent
coordination machinery. Ownership is diagnostic, not a promise of FIFO position
or priority. Its refusal is bypassable only through the explicit flag above;
it is never a silent skipped required check.

## Validation boundary

Local tests execute the real Makefile target with a stubbed GitHub boundary:
active statuses refuse before any label mutation, no active run arms normally,
another head does not match, completed-success-only history arms, and the explicit
override skips an unreachable or unusable advisory lookup with a warning.
Transient API failures retry before warning and arming; a recovered active match
still prevents label mutation. A synthetic monotonic clock verifies request,
retry and shared-deadline bounds. Auth/permission or malformed responses stop
automatic arming with no label write; unusable responses name the manual override.
Persisted before/after label state and exit codes distinguish these paths.
Topology tests retain the original concurrency contract
and pin ownership fields.
Workflow syntax/expression lint cannot simulate GitHub's scheduler. A real
`pull_request:labeled` run is still needed to prove the generated run title and
group on GitHub, and controlled repeated labels are needed to demonstrate pending
replacement. No duplicate live suites are armed just to obtain this evidence;
CI must still prove the unchanged full matrix and required aggregate on the
final head. Never report a dispatch or local YAML assertion as that proof.
