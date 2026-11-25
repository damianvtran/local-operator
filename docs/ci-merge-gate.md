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

## Red gate on the same head: rerun only the failed shards (GH #4074, cut 2)

Measured 2026-11-17..23: 190 of 539 gate runs failed, and every red gate was
re-armed with a whole new full run (both e2e OSes, six shards, coverage), even
when one shard failed. `make merge-gate PR=<n>` now reads the newest
`full-suite.yml` `pull_request` run for the PR's **current** head SHA, after the
fast-CI pre-check and the duplicate-arm advisory above (both unchanged):

| Newest run on this head | Path printed | Action |
|---|---|---|
| completed `failure` / `cancelled` / `timed_out` | `path: RERUN-FAILED` | `gh run rerun <id> --failed`; no label change |
| none (new head), `success`, `startup_failure`, still active | `path: FRESH: ...` | remove + re-add `ready-to-merge` (unchanged) |
| rerun refused by GitHub (e.g. older than its 30-day rerun window) | `path: FRESH (fallback)` | label arm |
| any, with `MERGE_GATE_FRESH=1` | `path: FRESH: MERGE_GATE_FRESH=1` | label arm |

A rerun that GitHub accepts but whose attempt does not advance within 60 s exits
non-zero with no label write, rather than also starting a full run.

**Why the verdict stays "every shard green on this head".** A failed-only rerun
is a new *attempt* of the same run: same `head_sha`, same event payload, and the
carried `changes`/`precondition` outputs, so the re-run shards rebuild the same
`gate_base` merge tree as the kept ones. GitHub re-runs failed **and cancelled**
jobs plus their dependents. Read from the API for run
[45325871121](https://github.com/damianvtran/local-operator/actions/runs/45325871121)
(PR #4508, `attempts/1/jobs` vs `attempts/2/jobs`): attempt 1 had shard 3
`failure` and shards 0/2/4 `cancelled` (fail-fast); attempt 2 re-ran exactly
those four plus `coverage-report` and the aggregate, and listed shards 1/5,
both `tui-e2e` legs, `changes` and `precondition` as carried `success` jobs with
their original start/finish times. The aggregate's `needs` therefore covers the
whole matrix of that run, and its existing allowlist (every need `success`, or
`skipped` for path-filtered jobs; `precondition` only `success`) is unchanged:
a shard that fails again keeps `needs.test.result` at `failure`. No job from
another run or head can enter `needs`, so no extra API cross-check was added.

**Base freshness.** The rerun keeps the original run's `gate_base` (the main tip
`changes` resolved when the gate armed). That matches the existing rule above:
the gate is evidence about the PR's head, the branch is not forced up to date,
and `main`'s own full-suite run is the backstop for a semantic conflict with a
moved base. An owner who wants today's main tip uses `MERGE_GATE_FRESH=1`.

**Artifacts across attempts.** Artifacts are run-scoped and immutable, so:
`coverage-data-<py>-<shard>` stays unsuffixed (coverage-report's
`coverage-data-<py>-*` pattern must find the carried shards' attempt-1 uploads;
run 45325871121 attempt 2 combined them) and uploads with `overwrite: true` for
the race where a shard is cancelled after uploading; every `always()` upload a
failed job can leave behind carries `-attempt-<n>` (the e2e junit did not, and
would have 409'd on its own rerun); `scripts/gen_test_durations.py` accepts a
carried shard's earlier-attempt junit, bound to the job's execution interval.

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

## Cost gate: fast CI first, main batched hourly (GH #4074)

The `precondition` job in `full-suite.yml` runs `scripts/fast_gate_check.py`
(stdlib only, fetched from the PR's **base** commit, never the PR checkout)
before any heavy job. Every heavy job needs it and tests `outputs.ok == 'true'`.

| Event | Rule |
|---|---|
| `pull_request` (label arm) | newest `ci.yml` `pull_request` run bound to this PR number, head SHA and head repo; latest attempt; exactly one `CI Gate` job `success`. Pending, red, missing, fork/unbound, API error: the check goes **red in seconds**, no test job starts. |
| `workflow_dispatch` (`train/*`, WIDE route) | newest `ci.yml` **push/dispatch** run on the same SHA, branch and repo, `CI Gate` green. A carrier PR's run tests `refs/pull/<n>/merge`, a different tree, and never counts. Train route: `gh workflow run ci.yml --ref train/<t>` -> wait for `CI Gate` on that SHA -> `gh workflow run full-suite.yml --ref train/<t>`. Always the **full** suite. |
| `push` to `main` | **BATCHED** (cut 1): an ordinary merge runs two tiny jobs and no heavy job; `Merge gate (full suite)` shows grey with `BATCHED`. A release commit (`chore(release)` subject, or a `Merge pull request` commit whose PR title is `chore(release)`) or a tagged commit runs in **full** on its exact SHA. Each push has its own concurrency group (`full-suite-main-push-<sha>`), so a pending release run is never replaced. An unreadable commit or tag lookup runs in full. |
| `schedule` `41 * * * *` (hourly batch) | the full matrix on the newest main tip, in `full-suite-main`. Skipped (grey, `HOURLY SKIP`) only when that exact SHA already has a full verdict (green **or red**: an unchanged red tip is not re-proved every hour; fix forward moves the tip), a run with heavy jobs in flight, or an OLDER scheduled/release run still deciding. Unknown runs in full. |
| `schedule` `17 7 * * *` (nightly) | never gated or coalesced; own concurrency group `full-suite-nightly`. |

There is no bypass input. The runner never waits for CI; use
`make fast-gate PR=<n>` (one-shot client check of the same helper via `--local`;
exits immediately) or `gh pr checks <n> --watch`, then `make merge-gate PR=<n>`
to arm the full suite.

**`make merge-gate` checks first (GH #4194/#4215/#4145).** The ritual runs the
same helper with `--local` before any label write and refuses to arm on exit 1
(pending, red, absent): no label changes, and it prints the `FAST-CI` line and
how to retry. Exit 3 (API or token unavailable locally) warns and arms, since
the runner re-decides fail-closed. Before this, arming right after a push found
`ci.yml` still `queued` (21 of 92 red gate runs, 2026-11-15..18), and the gate
went red with no test run. A label added by hand still bypasses the client
check; that run's `Merge gate (full suite)` now says `PRECONDITION NOT MET (not
a test failure)` so it is not mistaken for a red suite. The runner deliberately
does not poll: a waiting precondition would hold a runner from the same
saturated pool the queued `ci.yml` run is waiting for (ci.yml `pull_request`
runs, n=140, 2026-11-17..18: median 46, p90 137, max 210 min created->completed).

**Trusted helper.** For a PR the helper is fetched from `gate_base`, the live
main tip `changes` resolved and the heavy jobs test against, not from the
event's stale base SHA. The step summary logs the full trusted commit and the
helper's git blob id. A trusted tree without the helper fails the job red.

**Rollout is two steps; do not change flags in between.** (1) A tests-only PR
lands `scripts/fast_gate_check.py` (+ its tests) on main with no workflow
change. (2) Only then can the PR carrying the `full-suite.yml` change pass its
own gate. Until step 1 is on main that PR's required check is red by design.

**Coverage claims.** Every labeled PR head (unchanged), the main tip once per
hour when it moved, every release/tagged commit and the nightly still run every
test. A tag on a
commit is found whatever its name or age: every tag ref is listed
(`git/matching-refs/tags/`, `Link` followed on the same API origin only) and
annotated tag objects are peeled to their commit; any cap, cycle, timeout or API
error means UNKNOWN, which runs in full.

**Red batch: the merge range.** A red main run (hourly, release push or main
dispatch) has a last `Merge gate (full suite)` step that names
`<last green full main SHA>..<this SHA>` and every merged PR in it (first-parent:
`#N title (sha7)`), in the step summary and log, plus the `git log
--first-parent` command to reproduce it. "Last green" reads the jobs: a batched
or skipped run is run-level success with nothing run and never counts. The
step is fail-soft three ways (`if: failure()`, `continue-on-error`, the helper's
`--red-range` mode always exits 0) and runs after the verdict step has already
failed the job, so it can neither turn a green run red nor hide a red one. On a
PR it is skipped (the gate names its own head).

**Release pre-cut check.** `make release-check SHA=<40-hex>` (the helper's
`--release-check` mode) exits 0 only when a full-suite run on EXACTLY that SHA
on main concluded `success` with the full matrix: both Python legs of `test`
and `coverage-report`, linux and macOS `tui-e2e`, a green aggregate. The
release owner runs it before `gh release create` (AGENTS.md runbook step 4); the
runbook pipes `origin/main`'s copy of the helper into `python3 -` rather than
running `make` in the shared checkout, whose branch or dirty state could
otherwise supply an older helper. A malformed SHA prints the usage line (exit
2). It is deliberately not a step in `publish.yml`: that workflow starts after the tag
and GitHub Release exist, so a refusal there would strand a half-made release.

**Limits.** A main commit between two hourly ticks gets no verdict of its own;
the range report is how a red batch is attributed. A run cancelled by hand or
lost to a startup failure leaves the tip uncovered until the next hourly tick
(it is not a verdict, so the hourly re-runs it). A release bump now waits for
its own full run (~40 min) before `make release-check` passes.

## Matrix axes: PR merge gate vs `main` / nightly / `workflow_dispatch`

`full-suite.yml` keeps the `shard` lists literal (`[0..5]` for `test`, `[0, 1]`
for `tui-e2e`) and varies `python-version`, `os`, and `fail-fast` by
`github.event_name`:

| Job | `pull_request` (merge gate) | `push` (`main` release commits), `schedule` (hourly/nightly), `workflow_dispatch` |
|---|---|---|
| `test` | Python `3.13` × 6 shards (`fail-fast: true`) | Python `3.12` + `3.13` × 6 shards (`fail-fast: false`) |
| `coverage-report` | Python `3.13` (`fail-fast: true`, skipped on `affected` lane) | Python `3.12` + `3.13` (`fail-fast: false`) |
| `tui-e2e` | `ubuntu-latest` × 2 shards (`fail-fast: true`) | `ubuntu-latest` + `macos-latest` × 2 shards (`fail-fast: false`) |

Why this split: Python 3.12 compatibility is already verified on every PR push
by `ci.yml` (`guards`, `install-smoke`, `pyright`) and before every release by
`release-gate.yml`, while Python 3.13 is where strict upper-version regressions
surface; dropping `macos-latest` from PR merge gates removes the 10× macOS cost
multiplier and the 5-job account-wide macOS concurrency bottleneck while keeping
the Darwin `#401` freeze guard active post-merge on `main`, nightly, and on
demand via `workflow_dispatch` (including `WIDE:` branch dispatches).
`tui-e2e` still runs on every non-inert PR (no `github.*` job `if:` condition).
