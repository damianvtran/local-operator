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
| Image | `325492156725.dkr.ecr.ca-central-1.amazonaws.com/lop-poc-agent@sha256:121c99262662a59d1aa260ed2f8b2ce90044065f22267c19e045ce10bb78f6d9` (tag `0.68.3-0999101d8244`), built by CodeBuild `lop-poc-image-build`, in-image smoke `docker run --entrypoint lop <img> --version` → `v0.68.3` before the push |
| Task definition | `lop-poc-agent:10` — FARGATE, ARM64, 2 vCPU / 4 GiB, read-only root filesystem, `user: 10001:10001`, empty task role, both IAM roles wired, **no `initProcessEnabled`** (divergence 22) |
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

## §9.3 acceptance test

| # | Test | Verdict | Evidence |
| --- | --- | --- | --- |
| 1 | `describe-tasks` shows one task, ARM64, `lastStatus: STOPPED`, `stopCode: EssentialContainerExited`, exit 0; **no task tagged `lop-poc` running** afterwards | **PASS** | 5/5 runs: `stopCode: EssentialContainerExited`, `stoppedReason: Essential container in task exited`, container `exitCode: 0`. `status` → `active_tasks: []`, `active_count: 0`. ARM64 comes from the task definition, not `describe-tasks` (divergence 2) |
| 2 | Bundle verifies; `git log` shows one commit on `lop/<id>` whose parent is the fixture SHA; `test_add` fails at that SHA and passes on the branch | **PENDING** | Needs a run whose model actually edits the fixture, i.e. `LOP_POC_MODEL_KEY`. A mock run makes no commit by design, so `verify` reports it BLOCKED naming that reason — not FAIL |
| 3 | Session directory copied into an isolated config root opens and shows the full transcript | **PASS** | `verify` on `ct_67f1c6f2`, and the OPEN half is now the real load path: after transplanting `session.tar.gz` into a fresh `mktemp` root, `lop exec --resume <id> --hosting test --model test-model --json ping` (stdin `/dev/null`) exits 0, reports the **same session id**, grows the transcript **8 → 16 lines**, and the first 8 lines are **byte-identical**. `lop sessions --all --json` in the same root lists it `state: stored`. `lop --resume` is the TUI form of the same store load; the headless surface of that load is what was driven. The transcript is the mock provider's, so a *real* turn is still PENDING |
| 4 | Probe file shows 4a (creds endpoint, no policies), 4b (non-443 egress fails), 4c (no key on the filesystem) | **PASS** | `probes.json["failed"] == []` in all 5 runs; 4a/4b/4c plus 4c-env and 4d verbatim below; 4e (the launch-environment watcher) separately below |
| 5 | Cold start measured for 5 runs, recorded as evidence, replacing §4's estimates | **PASS** | Table below; 5 runs on the image digest in the header |
| 6 | Cost Explorer for the tag reconciles within 20% of `Σ wall_seconds × $0.0869/3600` | **COMPUTED, not reconciled** | `Σ wall = 2208.8 s` over all 30 recorded tasks → **$0.0533** (the five acceptance runs are 379.1 s → **$0.0092**); reconciliation is impossible today for the two reasons below |
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
     "usr_is_read_only": true, "workspace_is_writable": true}}}]}
```

Four things this establishes that §7.1 asserted and the design could not show: the
task-credentials endpoint serves the **empty** role's credentials and every AWS call it
makes is denied; **443 is the only egress** that works (the positive control is what
makes "everything failed" mean something); the key is **not on disk** (0 of 16 708
files) and **not in any child's environment**; and the work phase is **uid 10001** with
a read-only root and a writable workspace. The 4c counts are real verdicts on a real
value — the run's injected value is a placeholder, and the probe searches for it
exactly, so "0 of 16 708 files" is a measurement rather than a skip.

### Test 4e — the launch-environment watcher, and the exposure it closes (SEC-1)

Agent review round 1 raised this as SEC-1: the model's own bash child could recover the
provider key from the **agent process's launch environment** (`cat /proc/$PPID/environ`,
the read that `local_operator/tools/shell_env.py` documents and that unsetting in place
cannot close). The delivery was changed so that it cannot, and the change is measured
rather than argued:

| what | where | reading |
| --- | --- | --- |
| **GREEN, in the container** | 5/5 acceptance runs, sampled while the agent ran | `{"matches_by_needle": {"value": 0}, "matching_processes": [], "pass": true, "processes_scanned_max": 5, "samples": 3}` — 0 processes, over 3–5 samples of every readable `/proc/<pid>/environ` |
| **RED, in the container** | one run with `POC_ENVIRON_WATCH_SELFTEST=1` (task `813531fe54e2…`), which launches a child the OLD way — key exported into its environment | `{"matches_by_needle": {"value": 1}, "matching_processes": [{"comm": "sleep", "needle": "value", "pid": 46}], "pass": false, ...}`, task exit code **1**. The probe detects the delivery path SEC-1 replaced |
| **RED/GREEN, hermetic** | `tests/unit/test_remote_agents_poc_probes.py` (5 cases: red, green, prefix-only, blocked-without-procfs, rescan exit codes) | passes; the environ source is stubbed because a real one needs procfs |
| **no re-exec in the launcher** | every run's `agent_stderr.txt` | `lop-launch: branding re-exec plan: None` — `reexec_branded` is a no-op for a launch through `lop_launch.py`, so nothing re-execs with the key in its environment |
| **key arrived over the fd** | every run's `agent_stderr.txt` | `lop-launch: read 26 key byte(s) from fd 3; provider_env=''; set=False` (a mock run sets no provider variable, which is why `set=False` is correct rather than a failure) |
| **the entrypoint scrubs itself** | `exec "$0"` after taking the key out of the environment | the only process that ever held it in its *initial* environment is replaced by that exec, which is why the watcher's 0 includes PID 1 |

**The residual, as measured, not as asserted.** `yama_ptrace_scope: "1"` and, for every
run, `mem_target: {"comm": "Local Operator", "pid": 53}` with `mem_openable: false`,
`mem_error: "PermissionError: Permission denied"` — a same-uid **non-descendant** (the
watcher is a sibling of the agent, not its parent) could **not** open the agent's
memory. The key is still in the agent's memory and every process is uid 10001; that
exposure is bounded by yama on this AMI, not by anything this POC did, and the probe
records both facts per run. The ECS task definition also still carries the secret
reference and the ECS agent's view of the container environment, outside the
container's `/proc`.

### Test 5 — cold start, 5 runs on the accepted digest

| run id | task arn (suffix) | cpuArch | stopCode | exit | RunTask→RUNNING s | scheduling/ENI s | image pull s | container start s | probes s | RunTask→1st model event s | **minus probes s** | 4a/4b/4c/4c-env/4d | 4e |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ct_67f1c6f2 | `…ec0de95626d4` | ARM64 | EssentialContainerExited | 0 | 26.31 | 16.27 | 4.52 | 3.51 | 21.17 | 47.71 | 26.54 | P/P/P/P/P | P |
| ct_b9d45be9 | `…5dad73c4a2d7` | ARM64 | EssentialContainerExited | 0 | 27.75 | 18.34 | 4.67 | 3.92 | 21.18 | 51.22 | 30.03 | P/P/P/P/P | P |
| ct_bb54411f | `…c164da834784` | ARM64 | EssentialContainerExited | 0 | 23.88 | 12.52 | 4.41 | 3.21 | 21.19 | 45.90 | 24.71 | P/P/P/P/P | P |
| ct_d51a664b | `…0de3dbb3f265` | ARM64 | EssentialContainerExited | 0 | 24.87 | 15.26 | 4.53 | 3.73 | 21.23 | 47.61 | 26.38 | P/P/P/P/P | P |
| ct_f95de1c9 | `…c4dcfaad1958` | ARM64 | EssentialContainerExited | 0 | 22.37 | 13.70 | 4.48 | 3.74 | 21.15 | 44.03 | 22.89 | P/P/P/P/P | P |

min / median / max in seconds, n = 5:

| phase | min | median | max |
| --- | --- | --- | --- |
| RunTask call → first RUNNING (driver wall clock) | 22.37 | 24.87 | 27.75 |
| **scheduling + ENI attach** (`createdAt` → `pullStartedAt`) | 12.52 | 15.86 | 18.34 |
| **image pull** (`pullStartedAt` → `pullStoppedAt`) | 4.41 | 4.53 | 4.67 |
| **container start** (`pullStoppedAt` → `startedAt`) | 3.21 | 3.73 | 3.92 |
| ECS `startedAt` → container's first stamp | ≈ 0 (−0.01) | | |
| container start → probes start | 0.21 | 0.22 | 1.24 |
| **probes (4a–4d, incl. the whole-filesystem key scan)** | 21.15 | 21.18 | 21.23 |
| probes done → first model event | 1.36 | 1.38 | 1.45 |
| container start → first model event | 22.73 | 22.78 | 23.83 |
| RunTask → first model event (raw) | 44.03 | 47.61 | 51.22 |
| **RunTask → first model event, minus the probes** | 22.89 | 26.38 | 30.03 |

**What dominates.** Three phases, kept apart because they answer different questions
and one of them was mislabelled in the first draft of §4: **scheduling and ENI
attachment** is the largest single phase (15.9 s median), the **image pull is only
4.5 s**, and the **container start 3.7 s** — so image pull is ~18% of the 24.9 s arrival,
not the ~57% an earlier revision of this document claimed, and SOCI (which the pull
dominates the case for) has a 4.5 s ceiling here rather than a 12 s one. The published
"cold start" of ~48 s is then **~21 s of instrument**: probe 4c reads every regular file
(16 708) before the agent starts, by design. Subtracting it, a task is ready for a model
turn **26.4 s (median) after `RunTask`**, and the agent's own first token lands 1.4 s
after the probes finish.

**Which runs this measures.** The five above are **every** run on
`sha256:121c9926…`, and the population is complete by construction: the artifact store
records the image digest each task actually ran, and grouping all 30 recorded tasks by
that digest gives the table below. Earlier cycles of the build-out are visible as their
own populations rather than mixed in or omitted — the previous accepted digest
(`sha256:bba757f3…`) carries **six** runs, which is the omission agent review round 1
caught in this document's first revision.

| image digest | runs | note |
| --- | --- | --- |
| `sha256:121c9926…` | **5** | **the acceptance population: this revision** |
| `sha256:ccbb2ba7…`, `sha256:2df1ce65…` | 5 + 5 | the two intermediate rebuilds of the remediation round (probe fixes) |
| `sha256:9ddccaa6…` | 1 | first run of the launcher/fd delivery (found the misleading log line) |
| `sha256:bba757f3…` | **6** | the previous revision (non-root via VOLUME) |
| `sha256:29b799e5…` | 5 | the revision before that (timings + connect-budget fixes) |
| `sha256:3c2efb1e…`, `094c8b0a…`, `133e1f0e…` | 1 each | the three pre-artifact build-out runs (permissions, presign, first artifacts) |

### Test 6 — cost

`Σ wall_seconds × $0.0869/3600` over **all 30 recorded tasks**: `Σ wall = 2208.8 s =
0.6136 h` → **$0.0533**; the five acceptance runs are 379.1 s → **$0.0092**. Per-task
wall times were 46.3 s to 89.8 s. Nothing exceeded the 2 h + 10 min client deadline, so
no `StopTask` was ever issued.

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

### Tagged inventory (45 ARNs at report time)

| service | count | what they are |
| --- | --- | --- |
| ecs | 27 | `cluster/lop-poc`, `task-definition/lop-poc-agent:1`…`:10`, and 16 STOPPED `task/lop-poc/*` records (caveat 2: they age out ~1 h after each stop, which is why this count moves between readings) |
| ec2 | 7 | the VPC, 2 subnets, IGW, route table, SG, plus the adopted default SG |
| logs | 2 | `/lop-poc/agent`, `/lop-poc/codebuild` |
| codebuild / ecr / s3 / secretsmanager | 1 each | `lop-poc-image-build`, `lop-poc-agent`, `lop-poc-results-325492156725`, `lop-poc/model-key` |

Obtained with `aws resourcegroupstaggingapi get-resources --tag-filters
Key=[redacted]` on 2026-10-07; the same call is what `status` prints, whose verdict line
was `OK: no RUNNING or PENDING tasks in lop-poc; 45 tagged lop-poc resources`.

## Divergences from the spec and the design doc (short form)

Full text, with the error each one caused, is in `infra/remote-agents-poc/README.md`
§ Divergences (29 items). The ones that change the design document rather than just the
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
7. **Presigned PUTs must be signed for the regional S3 host, and `curl` reads a 307 as
   success** — both silent; the first artifact-bearing run exited 0 with nothing
   uploaded.
8. Smaller ones, each measured: the account id must come from `fn::invoke`
   `getCallerIdentity`; CodeBuild reads an unquoted `echo "a: b"` as a mapping and fails
   the phase; `lop sessions` needs `--all` to list stored sessions; `ephemeralStorage`
   is omitted rather than declared as 20 GiB; probe 4b's 5 s budget is per target, not
   per address; `--exclude` URNs must use the Pulumi resource keys; the public IPv4 and
   what 443-to-anywhere still permits; the CodeBuild builder image is tag-pinned
   because AWS publishes no digest for it; probe 4b dials two third-party hosts as
   negative controls.

**Not addressed here.** §4's "with and without SOCI" is half done: SOCI/Seekable OCI
lazy loading was not tested, so the 4.5 s pull stands un-improved. Nothing in this
document used `LOP_POC_MODEL_KEY`, which does not exist yet.
