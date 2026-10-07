# Slice-0 POC results — remote cloud agents

What was built, what was run, and what each item of §9.3's acceptance test actually
returned. Design and rationale: `remote-cloud-agents.md` (§4 cold start, §9 plan).
The long form of every divergence from the spec, with the measured error behind it, is
in `../../infra/remote-agents-poc/README.md`.

**Provenance.** Account `325492156725`, region `ca-central-1`,
`AWS_PROFILE=minerva_sandbox`; the account is asserted with `sts get-caller-identity`
immediately before every `pulumi up` and before every run. Branch
`poc/remote-cloud-agents-slice0` in the `lo-poc-slice0` worktree; the commits that
produced this revision are `02d9d4747`, `4f2114baf`, `732433330`, `b8f1ac04e`,
`6427a92be`, `fe76f09f1`, `242cc2f0e` plus the review-remediation commit (not pushed).
Pulumi state is a file backend at `$HOME/.lop-poc-pulumi-state`; `pulumi login` is
never run.

**The thing that was measured**

| | |
| --- | --- |
| Image | `325492156725.dkr.ecr.ca-central-1.amazonaws.com/lop-poc-agent@sha256:f9ce54119b392cf66b05661c2278c8be0404223e20982a04d214af084b255a3d` (tag `0.68.3-58c69256037c`), built by CodeBuild `lop-poc-image-build`, in-image smoke `docker run --entrypoint lop <img> --version` → `v0.68.3` before the push |
| Task definition | `lop-poc-agent:11` — FARGATE, ARM64, 2 vCPU / 4 GiB, read-only root filesystem, `user: 10001:10001`, empty task role, both IAM roles wired, **no `initProcessEnabled`** (divergence 22) |
| Fixture | https://github.com/olafagbemi/lop-poc-fixture @ `69db7e55fc14f918cccdf2fea62894fc37f1f642`, prompt "make the failing test `test_add` pass" |
| Cluster | `lop-poc`; app log group `/lop-poc/agent`; results bucket `lop-poc-results-325492156725` |
| Run population | **the five runs on the image digest above, and no others**; see "Which runs this measures" |

Every run record, probe file, `describe-tasks` dump and timing table is under the
driver's `--out-dir` (`run.json` + `describe-tasks.json` + the extracted `results/`),
regenerated with:

```sh
export AWS_PROFILE=minerva_sandbox AWS_REGION=ca-central-1
export PULUMI_BACKEND_URL="file://$HOME/.lop-poc-pulumi-state"
.venv/bin/python scripts/remote_agents_poc.py run --mock --runs 5 --out-dir /tmp/poc-runs
.venv/bin/python scripts/remote_agents_poc.py status
.venv/bin/python scripts/remote_agents_poc.py verify /tmp/poc-runs/<run-id> \
  --fixture-url https://github.com/olafagbemi/lop-poc-fixture.git \
  --fixture-sha 69db7e55fc14f918cccdf2fea62894fc37f1f642
```

**EVERY TABLE BELOW IS GENERATED, NOT TYPED.** The per-run cells, the summary row and
the digest population come from `run.json` and `describe-tasks.json` alone:

```sh
.venv/bin/python scripts/remote_agents_poc.py report /tmp/poc-runs \
  --digest sha256:f9ce54119b392cf66b05661c2278c8be0404223e20982a04d214af084b255a3d
```

That subcommand exists because the first version of the cold-start table was hand-copied
and three of its fifteen per-run cells were values that occur nowhere in the records
(agent review round 2, finding 1). A reviewer's sweep either matches these cells or the
tool is wrong; the next drift is one command to fix.

The two watcher self-tests (4e's red case and the coexistence case) are recorded in the
same out-dir, each as its own directory with its `describe-tasks.json`, `selftest.json`
and uploaded `results/`:

```sh
.venv/bin/python scripts/remote_agents_poc.py run --mock --runs 5 --out-dir /tmp/poc-runs
# the self-tests are RunTask calls with one extra environment entry, never set by the driver:
#   POC_ENVIRON_WATCH_SELFTEST=1   -> the red proof
#   POC_ENVIRON_WATCH_COEXIST=1    -> the coexistence proof
```

## §9.3 acceptance test

| # | Test | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | `describe-tasks` shows one task, ARM64, `lastStatus: STOPPED`, `stopCode: EssentialContainerExited`, exit 0; **no task tagged `lop-poc` running** afterwards | **PASS** | 5/5 runs: `stopCode: EssentialContainerExited`, `stoppedReason: Essential container in task exited`, container `exitCode: 0`. `status` → `active_tasks: []`, `active_count: 0`. ARM64 comes from the task definition, not `describe-tasks` (divergence 2) |
| 2 | Bundle verifies; `git log` shows one commit on `lop/<id>` whose parent is the fixture SHA; `test_add` fails at that SHA and passes on the branch | **PENDING** | Needs a run whose model actually edits the fixture, i.e. `LOP_POC_MODEL_KEY`. A mock run makes no commit by design, so `verify` reports it BLOCKED naming that reason — not FAIL |
| 3 | Session directory copied into an isolated config root opens and shows the full transcript | **PASS** | `verify` on `ct_2bc804c9`, and the OPEN half is now the real load path: after transplanting `session.tar.gz` into a fresh `mktemp` root, `lop exec --resume <id> --hosting test --model test-model --json ping` (stdin `/dev/null`) exits 0, reports the **same session id**, grows the transcript **8 → 16 lines**, and the first 8 lines are **byte-identical**. `lop sessions --all --json` in the same root lists it `state: stored`. `lop --resume` is the TUI form of the same store load; the headless surface of that load is what was driven. The transcript is the mock provider's, so a *real* turn is still PENDING |
| 4 | Probe file shows 4a (creds endpoint, no policies), 4b (non-443 egress fails), 4c (no key on the filesystem) | **PASS** | `probes.json["failed"] == []` in all 5 runs; 4a/4b/4c plus 4c-env, 4d and **4f** (no `ps` reachable) verbatim below; 4e (the launch-environment watcher) separately below |
| 5 | Cold start measured for 5 runs, recorded as evidence, replacing §4's estimates | **PASS** | Tables below, generated by `report`; 5 runs on the image digest in the header |
| 6 | Cost Explorer for the tag reconciles within 20% of `Σ wall_seconds × $0.0869/3600` | **COMPUTED, not reconciled** | `Σ wall = 2574.9 s` over all 35 recorded tasks → **$0.0622** (the five acceptance runs are 366.1 s → **$0.0088**; the two self-test tasks add 157.3 s, ~$0.0038); reconciliation is impossible today for the two reasons below |
| 7 | Teardown leaves zero tagged resources | **NOT RUN** | Awaiting approval; the commands and three caveats are below |

### Test 4 — the probes, verbatim (`ct_67f1c6f2`, trimmed)

```json
{"failed": [],
 "probes": [
  {"name": "4a_creds_endpoint", "pass": true, "detail": {
     "role_arn": "arn:aws:iam::325492156725:role/lop-poc-task",
     "caller_identity": "arn:aws:sts::325492156725:assumed-role/lop-poc-task/<task-id>",
     "s3_list_buckets": "denied: AccessDenied",
     "ecs_list_clusters": "denied: AccessDeniedException",
     "secretsmanager_get_secret_value": "denied: AccessDeniedException"}},
  {"name": "4b_egress", "pass": true, "detail": {
     "must_be_reachable": {"github.com:443": "connected in 15 ms"},
     "must_be_unreachable": {
       "example.com:80": "failed in 5005 ms: deadline exhausted",
       "github.com:22": "failed in 5006 ms: TimeoutError",
       "1.1.1.1:53": "failed in 5005 ms: TimeoutError",
       "portquiz.net:8080": "failed in 5003 ms: TimeoutError",
       "169.254.169.254:80": "failed in 0 ms: OSError"}}},
  {"name": "4c_no_secret_on_disk", "pass": true, "detail": {
     "files_scanned": 16708, "matches_by_needle": {"value": 0},
     "key_length_bytes": 26, "key_sha256_first8": "5a71e522"}},
  {"name": "4c_env_no_key_in_child_env", "pass": true, "detail": {
     "child_environment_entries": 36, "entries_containing_key": 0, "entry_names": []}},
  {"name": "4d_platform", "pass": true, "detail": {"checks": {
     "uname_m_is_aarch64": true, "uid_is_10001": true, "root_is_read_only": true,
     "usr_is_read_only": true, "workspace_is_writable": true}}},
   {"name": "4f_no_ps", "pass": true, "detail": {
      "which_ps": null, "paths_present": [], "multiplexers_present": [],
      "checked_paths": ["/bin/ps", "/usr/bin/ps", "/sbin/ps", "/usr/sbin/ps",
                         "/usr/local/bin/ps", "/usr/local/sbin/ps"]}}]}
```

Four things this establishes that §7.1 asserted and the design could not show: the
task-credentials endpoint serves the **empty** role's credentials and every AWS call it
makes is denied; **443 is the only egress** that works (the positive control is what
makes "everything failed" mean something); the key is **not on disk** (0 of 16 708
files) and **not in any child's environment**; and the work phase is **uid 10001** with
a read-only root and a writable workspace. The 4c counts are real verdicts on a real
value — the run's injected value is a placeholder, and the probe searches for it
exactly, so "0 of 16 708 files" is a measurement rather than a skip.

**4f exists because the environ claim is conditional, and the condition is a property of
the image.** Two product spawn sites hand the CALLER's environment to `ps` —
`local_operator/tools/group_reaper.py:229` (`env={**os.environ, "LC_ALL": "C"}`) and
`local_operator/memory_guard.py`'s `_default_runner` (no `env=` at all, on every tick of
every guarded command) — so either child would carry the key in its own
`/proc/<pid>/environ` and the model's same-uid bash child could read that file. The
closure therefore holds only while the image ships no `ps`, and this is the probe that
says so: `which_ps: null`, the six knowable paths absent, and no `busybox`/`toybox` `ps`
either. The Dockerfile fails the build if one appears, because adding procps for any
reason (a debugging tool, a base-image change) would reopen the path silently. 4e cannot
catch it: a `ps` lasts tens of milliseconds against 4e's one-second samples.

### Test 4e — the launch-environment watcher, and the exposure it closes (SEC-1)

Agent review round 1 raised this as SEC-1: the model's own bash child could recover the
provider key from the **agent process's launch environment** (`cat /proc/$PPID/environ`,
the read that `local_operator/tools/shell_env.py` documents and that unsetting in place
cannot close). The delivery was changed so that it cannot, and the change is measured
rather than argued:

| what | where | reading |
| --- | --- | --- |
| **GREEN, in the container** | 5/5 acceptance runs (`report`'s 4e column, all `P`) | `{"matches_by_needle": {"value": 0}, "matching_processes": [], "pass": true, "processes_scanned_max": 5, "samples": 3}` — 0 processes, over 3–5 samples of every readable `/proc/<pid>/environ`. **These five runs are mock, so the agent spawns no tool child at all** (`exec.jsonl` is `message_start(user)` → one assistant text with no `tool_use`), which is why the coexistence case below exists rather than being assumed from these readings |
| **COEXISTENCE, in the container** | one run with `POC_ENVIRON_WATCH_COEXIST=1` (task `f88619b98cd6…`, out-dir `coexist_r2_fa29af2b7a9d`), where the real launcher holds the key in-process and spawns a bash child through the product's own filter while 4e samples | 4e stays **GREEN** (`pass: true`, 0 matches, 3 samples) and the launcher reports `{"child_argv": ["sh", "-c", "sleep 2"], "child_env_entries": 2, "child_exit": 0, "key_value_anywhere_in_child_env": false, "parent_self_environ_clean": true, "pass": true, "provider_var_in_child_env": false}` after `lop-launch: set LOP_POC_MODEL_KEY in-process (26 bytes); /proc/self/environ clean: True`. A parent holding the key, a child that inherited neither its name nor its value |
| **RED, in the container** | one run with `POC_ENVIRON_WATCH_SELFTEST=1` (task `d0009b1a95c8…`, out-dir `selftest_r2_d0009b1a95c8`), which launches a child the OLD way — key exported into its environment | `{"matches_by_needle": {"value": 1}, "matching_processes": [{"comm": "sleep", "needle": "value", "pid": 42}], "pass": false, ...}`, task exit code **1**. The probe detects the delivery path SEC-1 replaced. (The round-1 instance of this run, task `813531fe54e2…`, uploaded no artifact before the self-test branch learned to tar `$OUT`; its reading is in CloudWatch stream `lop-poc/agent/813531fe54e2443c9fc2a5b706b3b4f9` and nowhere else, which is the traceability gap agent review round 2 called finding 6.) |
| **RED/GREEN, hermetic** | `tests/unit/test_remote_agents_poc_probes.py` (5 cases: red, green, prefix-only, blocked-without-procfs, rescan exit codes) | passes; the environ source is stubbed because a real one needs procfs |
| **no re-exec in the launcher** | every run's `agent_stderr.txt` | `lop-launch: branding re-exec plan: None` — `reexec_branded` is a no-op for a launch through `lop_launch.py`, so nothing re-execs with the key in its environment |
| **key arrived over the fd** | every run's `agent_stderr.txt` | `lop-launch: read 26 key byte(s) from fd 3; provider_env=''; set=False` (a mock run sets no provider variable, which is why `set=False` is correct rather than a failure) |
| **the entrypoint scrubs itself** | the key is read and unset at the top of the script, then `exec "$0"` re-execs it | the entrypoint's own `/proc/<pid>/environ` is clean from that exec on, and the two children that run before it — `$(id -u)` became `$EUID`, a builtin, and `stamp` moved below the unset (SEC-12) — are gone, which is why the watcher's 0 includes PID 1 |

**The residual, as measured, not as asserted.** `yama_ptrace_scope: "1"` and, for every
run, `mem_target: {"comm": "Local Operator", "pid": 50}` with `mem_openable: false`,
`mem_error: "PermissionError: Permission denied"` — a same-uid **non-descendant** (the
watcher is a sibling of the agent, not its parent) could **not** open the agent's
memory. The key is still in the agent's memory and every process is uid 10001; that
exposure is bounded by yama on this AMI, not by anything this POC did, and the probe
records both facts per run. The ECS task definition also still carries the secret
reference and the ECS agent's view of the container environment, outside the
container's `/proc`.

**The one condition that is not a residual but a dependency:** 4f, above. "No process's
initial environment carries the key" holds while the image ships no `ps`, because
`group_reaper.py:229` and `memory_guard._default_runner` are the two product spawns that
receive the caller's environment. Closing it product-side — a filtered environment at
those two sites, or a delivery that never enters `os.environ` — is **deferred product
work**: this POC runs the released `local-operator==0.68.3` wheel and patches no product
code, which is recorded in the PR thread rather than as an issue.

### Test 5 — cold start, 5 runs on the accepted digest

| run id | task arn (suffix) | cpuArch | stopCode | exit | RunTask→RUNNING s | scheduling/ENI s | image pull s | container start s | probes s | RunTask→1st model event s | minus probes s | probes | 4e |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ct_54c56e7a | `…f5a686cff8f8` | ARM64 | EssentialContainerExited | 0 | 22.61 | 14.02 | 4.61 | 3.17 | 21.18 | 45.34 | 24.15 | P/P/P/P/P/P | P |
| ct_31c28ef9 | `…0b27fdf6d837` | ARM64 | EssentialContainerExited | 0 | 22.61 | 13.35 | 4.64 | 3.88 | 21.17 | 45.16 | 23.99 | P/P/P/P/P/P | P |
| ct_ed98df28 | `…3fdf12613952` | ARM64 | EssentialContainerExited | 0 | 21.07 | 10.72 | 5.78 | 3.61 | 21.62 | 44.52 | 22.91 | P/P/P/P/P/P | P |
| ct_286e4fba | `…1c027b743f65` | ARM64 | EssentialContainerExited | 0 | 22.14 | 13.50 | 4.75 | 3.05 | 21.17 | 44.60 | 23.43 | P/P/P/P/P/P | P |
| ct_2bc804c9 | `…12ba118b345b` | ARM64 | EssentialContainerExited | 0 | 22.04 | 10.87 | 5.97 | 3.73 | 21.59 | 44.73 | 23.14 | P/P/P/P/P/P | P |

Min / median / max in seconds, n = 5 — the same `report` output, not a second rendering:

| phase | min | median | max |
| --- | --- | --- | --- |
| RunTask call → first RUNNING (driver wall clock) | 21.07 | 22.14 | 22.61 |
| **scheduling + ENI attach** (`createdAt` → `pullStartedAt`) | 10.72 | 13.35 | 14.02 |
| **image pull** (`pullStartedAt` → `pullStoppedAt`) | 4.61 | 4.75 | 5.97 |
| **container start** (`pullStoppedAt` → `startedAt`) | 3.05 | 3.61 | 3.88 |
| the three phases above as one span (`createdAt` → `startedAt`) | 20.11 | 21.29 | 21.87 |
| container's first stamp → probes start | 0.23 | 0.24 | 0.25 |
| **probes (4a–4f, incl. the whole-filesystem key scan)** | 21.17 | 21.18 | 21.62 |
| probes done → first model event | 1.37 | 1.39 | 1.91 |
| container start → first model event | 22.77 | 22.82 | 23.78 |
| RunTask → first model event (raw) | 44.52 | 44.73 | 45.34 |
| **RunTask → first model event, minus the probes** | 22.91 | 23.43 | 24.15 |

**What dominates.** Three phases, kept apart because they answer different questions and
one of them was mislabelled in the first draft of §4: **scheduling and ENI attachment**
is the largest single phase (13.4 s median), the **image pull is 4.75 s**, and the
**container start 3.61 s** — so image pull is ~21% of the 22.1 s arrival, not the ~57% an
earlier revision claimed, and SOCI (whose case the pull is) has a ~4.8 s ceiling here
rather than a 12 s one. The published "cold start" of ~44.7 s is then **~21 s of
instrument**: probe 4c reads every regular file (16 708) before the agent starts, by
design. Subtracting it, a task is ready for a model turn **23.4 s (median) after
`RunTask`**, and the agent's own first token lands 1.4 s after the probes finish. The
three ECS phases span `createdAt`→`startedAt` (**21.3 s median**); the ~0.8 s between
that and the RunTask figure is the control plane's own round trip, not a phase.

**Which runs this measures.** The five above are **every** run on
`sha256:f9ce54119b39…`, and the population is complete by construction: every artifact
directory records the image digest its task actually ran, and grouping **all 35 recorded
tasks** by that digest gives the table below. Earlier cycles of the build-out are visible
as their own populations rather than mixed in or omitted — the first accepted digest
(`sha256:121c9926…`) carries 5 and the one before it (`sha256:bba757f3…`) carries **6**,
which is the omission agent review round 1 caught in this document's first revision.

| image digest | runs | run ids |
| --- | --- | --- |
| `sha256:bba757f3554e8…` | **6** | ct_7bbae11f, ct_7340a069, ct_a018fd2a, ct_a63a986e, ct_631d1b0d, ct_18b58959 |
| `sha256:29b799e52a792…` | **5** | ct_ca76cdfa, ct_c551ae58, ct_1844f480, ct_ede01329, ct_c45e05e8 |
| `sha256:2df1ce6592ac7…` | **5** | ct_bdc59259, ct_7ffbeac5, ct_65646837, ct_e85c8d08, ct_36d88058 |
| `sha256:ccbb2ba7e0cad…` | **5** | ct_23145198, ct_59c2709f, ct_7fbc159c, ct_97b7b1db, ct_e16d3bea |
| `sha256:121c99262662a…` | **5** | ct_b9d45be9, ct_67f1c6f2, ct_bb54411f, ct_d51a664b, ct_f95de1c9 |
| `sha256:f9ce54119b392…` **← this table** | **5** | ct_54c56e7a, ct_31c28ef9, ct_ed98df28, ct_286e4fba, ct_2bc804c9 |
| `sha256:133e1f0ea1e78…` | **1** | ct_17ececef |
| `sha256:094c8b0aec83e…` | **1** | ct_6146eed7 |
| `sha256:3c2efb1e536d5…` | **1** | ct_a7a20b4d |
| `sha256:9ddccaa6cf879…` | **1** | ct_32c487ba |

total recorded runs: 35
| `sha256:bba757f3…` | **6** | the previous revision (non-root via VOLUME) |
| `sha256:29b799e5…` | 5 | the revision before that (timings + connect-budget fixes) |
| `sha256:3c2efb1e…`, `094c8b0a…`, `133e1f0e…` | 1 each | the three pre-artifact build-out runs (permissions, presign, first artifacts) |

### Test 6 — cost

`Σ wall_seconds × $0.0869/3600` over **all 35 recorded tasks**: `Σ wall = 2574.9 s =
0.7153 h` → **$0.0622**; the five acceptance runs are 366.1 s → **$0.0088**. The two
watcher self-test tasks (the red proof and the coexistence proof, each its own RunTask)
add 157.3 s → ~$0.0038, which is in the same table's class but not in the 35, because
they carry no `run.json`. Per-task wall times were 46.3 s to 89.8 s. Nothing exceeded the
2 h + 10 min client deadline, so no `StopTask` was ever issued.

Reconciliation is **not possible today**, for two reasons that are properties of this
account rather than of the POC:

1. **Cost Explorer lags ~24 h**, and these runs are hours old, so a query now returns
   nothing to compare against.
2. **`lop-poc` cannot be an activated cost-allocation tag here.** The account is a
   LINKED account in `o-rxz2wexmo9`, and `ListCostAllocationTags` answers
   `AccessDenied` ("Linked account doesn't have access"), so a tag-filtered query
   tracks nothing until the management account activates `lop-poc`. The working
   substitute is a usage-type/service filter — `Amazon Elastic Container Service` for
   Fargate, ARM, `ca-central-1` — over the run dates, which is sound only because no
   other Fargate task runs in this account (§9.1 records zero). The stack ships both:
   budget `lop-poc` (tag-filtered, inert until activation) and `lop-poc-fargate`
   (service-filtered, the working backstop).

```sh
aws ce get-cost-and-usage --time-period Start=2026-10-07,End=2026-10-09 \
  --granularity DAILY --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Container Service"]}}'
```

### Test 7 — teardown

**NOT RUN**: §9.3 item 7 and the POC spec both require the operator's approval first,
and nothing has been destroyed.

```sh
cd infra/remote-agents-poc
export AWS_PROFILE=minerva_sandbox AWS_REGION=ca-central-1
export PULUMI_BACKEND_URL="file://$HOME/.lop-poc-pulumi-state"
PUL="lop secret run --secret [redacted]=PULUMI_CONFIG_PASSPHRASE -- pulumi"
aws sts get-caller-identity          # MUST be 325492156725 before a destroy
$PUL destroy                         # S3 forceDestroy, ECR forceDelete, secret recovery window 0
$PUL stack rm poc
aws resourcegroupstaggingapi get-resources --tag-filters Key=[redacted]
```

Three caveats, all measured while assembling the inventory below — each would otherwise
make that last command look like a failed teardown:

1. **Superseded task-definition revisions are not removed by `pulumi destroy`.** The
   family now carries `lop-poc-agent:1` … `:10` (one per image digest pinned during the
   build-out) and Pulumi owns only the current one; `aws ecs
   list-task-definitions --family-prefix lop-poc-agent` then `aws ecs
   delete-task-definitions --task-definition …:N` per stale revision is the cleanup, and
   it is not something the stack can do for revisions it has already replaced.
2. **STOPPED task records stay visible to the tagging API for about an hour** after they
   stop, and ECS reclaims them on its own. The "zero tagged resources" check therefore
   has to run more than an hour after the last run, or filter `ecs:task/` rows out.
3. **The VPC's default security group is adopted but not deletable** (divergence 23).
   `pulumi destroy` revokes the rules it manages and leaves the group, which AWS keeps
   for the life of the VPC — and it is untagged, so it does not appear in the inventory.

A fourth item is deliberate, not a leak: the bucket's SSE key is the **AWS-managed**
`aws/s3` key, which is untagged and does not appear in the inventory — no CMK was
created precisely so teardown cannot leave a pending-deletion key behind.

### Tagged inventory (51 ARNs at report time)

| service | count | what they are |
| --- | --- | --- |
| ecs | 38 | `cluster/lop-poc`, `task-definition/lop-poc-agent:1`…`:11`, and 26 STOPPED `task/lop-poc/*` records (caveat 2: they age out ~1 h after each stop, which is why this count moves between readings — the round-1 head read 45, then 42, then 51 as runs were added and records aged out) |
| ec2 | 7 | the VPC, 2 subnets, IGW, route table, SG, plus the adopted default SG |
| logs | 2 | `/lop-poc/agent`, `/lop-poc/codebuild` |
| codebuild / ecr / s3 / secretsmanager | 1 each | `lop-poc-image-build`, `lop-poc-agent`, `lop-poc-results-325492156725`, `lop-poc/model-key` |

Obtained with `aws resourcegroupstaggingapi get-resources --tag-filters
Key=[redacted]` on 2026-10-07; the same call is what `status` prints, whose verdict line
was `OK: no RUNNING or PENDING tasks in lop-poc; 45 tagged lop-poc resources`.

## Divergences from the spec and the design doc (short form)

Full text, with the error each one caused, is in `infra/remote-agents-poc/README.md`
§ Divergences (32 items). The ones that change the design document rather than just the
POC:

1. **A non-root container needs the image to declare `VOLUME ["/workspace"]`** over a
   `/workspace` it already owns; without it the task dies on its first `mkdir`.
2. **`describe-tasks` does not return `runtimePlatform` for Fargate**, so §9.3 item 1's
   "ARM64" is read from the task definition.
3. **§9.2 item 10's resource list omits `executionRoleArn` and `taskRoleArn`.**
4. **SSE-KMS with the AWS-managed key is `sseAlgorithm: aws:kms` with no
   `kmsMasterKeyId`** — naming `aws/s3` fails every `PutObject`.
5. **A value-less Secrets Manager secret cannot back a task-level secret**, so §9.2's
   mock-run step is unreachable as written; a placeholder is written instead.
6. **The key is delivered over a file descriptor through `lop_launch.py`, and
   `initProcessEnabled` is dropped** (SEC-1): the ECS init shim would be a process the
   entrypoint cannot re-exec, whose `/proc/1/environ` keeps carrying the key.
7. **"No process's initial environment carries the key" is conditional on the image
   shipping no `ps`** (SEC-11), because `tools/group_reaper.py:229` and
   `memory_guard._default_runner` hand the caller's environment to `ps`. Probe **4f**
   asserts it at runtime and the Dockerfile **fails the build** if a `ps` ever appears;
   the product-side fix is deferred work recorded in the PR thread. The claim also
   covers children started under the container's `allowlist` shell-environment policy.
8. **The mock runs spawn no tool child at all, so their 4e reading is not the
   coexistence case** (SEC-13). That case is measured by its own run
   (`POC_ENVIRON_WATCH_COEXIST=1`: the launcher holds the key and spawns a bash child
   through `shell_env.child_environment` while 4e samples), and the red case by
   `POC_ENVIRON_WATCH_SELFTEST=1`. Both upload their artifacts into the same out-dir.
9. **The cold-start tables are generated by `report`, not typed.** Every per-run cell,
   the summary row and the digest population come from `run.json` /
   `describe-tasks.json` in one command (finding 1 of agent review round 2).
10. **The watcher requires procfs** (divergence 29): a Darwin `ps -Eww` fallback was
    written, measured, and removed — it printed only ARGV, so it false-positived on
    command lines and missed an environment.
11. **Presigned PUTs must be signed for the regional S3 host, and `curl` reads a 307 as
    success** — both silent; the first artifact-bearing run exited 0 with nothing
    uploaded.
12. Smaller ones, each measured: the account id must come from `fn::invoke`
    `getCallerIdentity`; CodeBuild reads an unquoted `echo "a: b"` as a mapping and fails
    the phase; `lop sessions` needs `--all` to list stored sessions; `ephemeralStorage`
    is omitted rather than declared as 20 GiB; probe 4b's 5 s budget is per target, not
    per address; `--exclude` URNs must use the Pulumi resource keys; the public IPv4 and
    what 443-to-anywhere still permits; the CodeBuild builder image is tag-pinned
    because AWS publishes no digest for it; probe 4b dials two third-party hosts as
    negative controls.

**Not addressed here.** §4's "with and without SOCI" is half done: SOCI/Seekable OCI
lazy loading was not tested, so the 4.75 s pull stands un-improved. The two `ps` spawn
sites in SEC-11 are product-side work this POC cannot do (it runs the released wheel);
that item is deferred in the PR thread. Nothing in this document used
`LOP_POC_MODEL_KEY`, which does not exist yet.
