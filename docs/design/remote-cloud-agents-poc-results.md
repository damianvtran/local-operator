# Slice-0 POC results — remote cloud agents

What was built, what was run, and what each item of §9.3's acceptance test actually
returned. Design and rationale: `remote-cloud-agents.md` (§4 cold start, §9 plan).
The long form of every divergence from the spec, with the measured error behind it,
is in `../../infra/remote-agents-poc/README.md`.

**Provenance.** Account `325492156725`, region `ca-central-1`,
`AWS_PROFILE=minerva_sandbox`; the account is asserted with `sts
get-caller-identity` immediately before every `pulumi up` and before every run.
Branch `poc/remote-cloud-agents-slice0` in the `lo-poc-slice0` worktree, commits
`02d9d4747`, `4f2114baf`, `732433330`, `b8f1ac04e`, `6427a92be`, `fe76f09f1` plus
this round's commit (not pushed). Pulumi state is a file backend at
`$HOME/.lop-poc-pulumi-state`; `pulumi login` is never run.

**The thing that was measured**

| | |
| --- | --- |
| Image | `325492156725.dkr.ecr.ca-central-1.amazonaws.com/lop-poc-agent@sha256:bba757f3554e87bbf3321cbd2c5ee879d0ed37c628e3ab4a23154519c0aaf3c5` (tag `0.68.3-f56db669d976`), built by CodeBuild `lop-poc-image-build`, in-image smoke `docker run --entrypoint lop <img> --version` → `v0.68.3` before the push |
| Task definition | `lop-poc-agent:5` — FARGATE, ARM64, 2 vCPU / 4 GiB, read-only root filesystem, `user: 10001:10001`, empty task role, both IAM roles wired |
| Fixture | https://github.com/olafagbemi/lop-poc-fixture @ `69db7e55fc14f918cccdf2fea62894fc37f1f642`, prompt "make the failing test `test_add` pass" |
| Cluster | `lop-poc`; app log group `/lop-poc/agent`; results bucket `lop-poc-results-325492156725` |
| Recorded tasks | 14 across the build-out (4 image digests, 5 task-definition revisions); the acceptance numbers below are the **final 5 runs on the final digest** |

Every run record, probe file, `describe-tasks` dump and timing table is under the
driver's `--out-dir` (`run.json` + `describe-tasks.json` + the extracted
`results/`), regenerated with:

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
| 3 | Session directory copied into an isolated config root opens and shows the full transcript | **PASS (transplant half)**; the real-model transcript is **PENDING** | `verify` on `ct_7340a069`: `session.tar.gz` extracted into a fresh `mktemp` root, `lop sessions --all --json` there lists the session `state: stored`, transcript 8 lines. The transcript is the mock provider's, so this proves the transplant and the store, not a real turn |
| 4 | Probe file shows 4a (creds endpoint, no policies), 4b (non-443 egress fails), 4c (no key on the filesystem) | **PASS** | `probes.json["failed"] == []` in all 5 runs; 4a/4b/4c plus the two added probes, verbatim below |
| 5 | Cold start measured for 5 runs, recorded as evidence, replacing §4's estimates | **PASS** | Table below; 5 runs on the final digest |
| 6 | Cost Explorer for the tag reconciles within 20% of `Σ wall_seconds × $0.0869/3600` | **COMPUTED, not reconciled** | `Σ wall = 1004.0 s` over all 14 tasks → **$0.0242**; reconciliation is impossible today for two measured reasons, below |
| 7 | Teardown leaves zero tagged resources | **NOT RUN** | Awaiting approval; the exact commands and two caveats are below |

### Test 4 — the probes, verbatim (`ct_7340a069`, trimmed)

```json
{"failed": [],
 "probes": [
  {"name": "4a_creds_endpoint", "pass": true, "detail": {
     "endpoint_url": "http://169.254.170.2/v2/credentials/ec0a0e7d-5230-45fe-9b8a-88715276bcb7",
     "role_arn": "arn:aws:iam::325492156725:role/lop-poc-task",
     "caller_identity": "arn:aws:sts::325492156725:assumed-role/lop-poc-task/d86f1b56765442fcabe0b726a96f8c70",
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
     "files_scanned": 16707, "matches_by_needle": {"value": 0},
     "key_length_bytes": 26, "key_sha256_first8": "5a71e522"}},
  {"name": "4c_env_no_key_in_child_env", "pass": true, "detail": {
     "child_environment_entries": 36, "entries_containing_key": 0, "entry_names": []}},
  {"name": "4d_platform", "pass": true, "detail": {"checks": {
     "uname_m_is_aarch64": true, "uid_is_10001": true, "root_is_read_only": true,
     "usr_is_read_only": true, "workspace_is_writable": true}}}]}
```

Three things this establishes that §7.1 asserted and the design could not show:
the task-credentials endpoint serves the **empty** role's credentials and every
AWS call it makes is denied; **443 is the only egress** that works (and the
positive control is what makes "everything failed" mean something); and the key
is **neither on disk nor in any child's environment**. The 4c reported here is a
real verdict on a real value: the run's injected value is a placeholder, and the
probe searches for it exactly, so "0 of 16 707 files" is a measurement rather
than a skip.

### Test 5 — cold start, 5 runs on the final digest

| run id | task arn (suffix) | cpuArch | stopCode | exit | RunTask→RUNNING s | probes s | RunTask→1st model event s | **minus probes s** | 4a/4b/4c/4c-env/4d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ct_7340a069 | `…b726a96f8c70` | ARM64 | EssentialContainerExited | 0 | 24.46 | 21.59 | 46.96 | 25.36 | P/P/P/P/P |
| ct_a018fd2a | `…603ee824cf38` | ARM64 | EssentialContainerExited | 0 | 21.05 | 21.48 | 44.12 | 22.64 | P/P/P/P/P |
| ct_a63a986e | `…7b43694786e0` | ARM64 | EssentialContainerExited | 0 | 21.63 | 21.17 | 43.80 | 22.63 | P/P/P/P/P |
| ct_631d1b0d | `…010878d8acd3` | ARM64 | EssentialContainerExited | 0 | 21.10 | 21.15 | 43.16 | 22.00 | P/P/P/P/P |
| ct_18b58959 | `…43b4a086bf72` | ARM64 | EssentialContainerExited | 0 | 22.78 | 21.46 | 45.98 | 24.52 | P/P/P/P/P |

min / median / max, seconds:

| span | min | median | max |
| --- | --- | --- | --- |
| RunTask call → first RUNNING (driver wall clock) | 21.05 | 21.63 | 24.46 |
| ECS `createdAt` → `pullStartedAt` (schedule + image pull) | 11.04 | 12.36 | 13.45 |
| ECS `pullStartedAt` → `startedAt` | 8.02 | 8.53 | 9.82 |
| ECS `startedAt` → container's first stamp | ≈ 0 (−0.01) | | |
| container start → probes start | 0.23 | 0.23 | 0.26 |
| **probes (4a–4d, incl. the whole-filesystem key scan)** | 21.15 | 21.46 | 21.59 |
| probes done → first model event | 1.29 | 1.62 | 1.85 |
| container start → first model event | 22.68 | 23.31 | 23.68 |
| RunTask → first model event (raw) | 43.16 | 44.12 | 46.96 |
| **RunTask → first model event, minus the probes** | 22.00 | 22.64 | 25.36 |

**What dominates, and what the raw number hides.** The published "cold start" is
~44 s to a first model event, and ~21 s of that is the instrument: probe 4c reads
every regular file on the filesystem (16 707 of them) looking for the key, and it
runs before the agent by design. Subtracting it, a task is ready for a model turn
**22.6 s (median) after `RunTask`**, of which ~12 s is image pull and scheduling
and ~2 s is the whole rest of the entrypoint. The agent's own first token arrives
1.3–1.9 s after the probes finish. `ECS startedAt → container's first stamp` is
−10 ms: that is clock skew between the control plane and the task, not a negative
duration.

### Test 6 — cost

`Σ wall_seconds × $0.0869/3600` over **all 14 recorded tasks** (every RunTask
issued during the build-out, failures included): `Σ wall = 1004.0 s = 0.2789 h` →
**$0.0242**. Per-run wall times were 46.3 s to 89.8 s; the five acceptance runs are
365.0 s of that, i.e. **$0.0088**. Nothing exceeded the 2 h + 10 min client
deadline, so no `StopTask` was ever issued.

Reconciliation is **not possible today**, for two reasons that are properties of
this account rather than of the POC:

1. **Cost Explorer lags ~24 h**, and these runs are hours old, so a query now
   returns nothing to compare against.
2. **`lop-poc` cannot be an activated cost-allocation tag here.** The account is a
   LINKED account in `o-rxz2wexmo9`, and `ListCostAllocationTags` answers
   `AccessDenied` ("Linked account doesn't have access"), so a tag-filtered query
   tracks nothing until the management account activates `lop-poc`. The working
   substitute is a usage-type/service filter — `Amazon Elastic Container Service`
   for Fargate, ARM, `ca-central-1` — over the run dates, which is sound only
   because no other Fargate task runs in this account (§9.1 records zero). The
   stack ships both: budget `lop-poc` (tag-filtered, inert until activation) and
   `lop-poc-fargate` (service-filtered, the working backstop).

Run this after the lag has passed:

```sh
aws ce get-cost-and-usage --time-period Start=2026-10-07,End=2026-10-09 \
  --granularity DAILY --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Container Service"]}}'
```

### Test 7 — teardown

**NOT RUN**: §9.3 item 7 and the POC spec both require the operator's approval
first, and nothing has been destroyed.

```sh
cd infra/remote-agents-poc
export AWS_PROFILE=minerva_sandbox AWS_REGION=ca-central-1
export PULUMI_BACKEND_URL="file://$HOME/.lop-poc-pulumi-state"
PUL="lop secret run --secret LOP_POC_PULUMI_PASSPHRASE=PULUMI_CONFIG_PASSPHRASE -- pulumi"
aws sts get-caller-identity          # MUST be 325492156725 before a destroy
$PUL destroy                         # S3 forceDestroy, ECR forceDelete, secret recovery window 0
$PUL stack rm poc
aws resourcegroupstaggingapi get-resources --tag-filters Key=lop-poc,Values=true
```

Two caveats measured while assembling the inventory below, both of which would
otherwise make that last command look like a failed teardown:

1. **Superseded task-definition revisions are not removed by `pulumi destroy`.**
   The family carries `lop-poc-agent:1` … `:5` (one per image digest pinned during
   the build-out) and Pulumi owns only the current one; `aws ecs
   list-task-definitions --family-prefix lop-poc-agent` then `aws ecs
   delete-task-definitions --task-definition …:N` per stale revision is the
   cleanup, and it is not something the stack can do for revisions it has already
   replaced.
2. **STOPPED task records stay visible to the tagging API for about an hour** after
   they stop, and ECS reclaims them on its own. The "zero tagged resources" check
   therefore has to run more than an hour after the last run, or filter
   `ecs:task/` rows out of the response.

A third item is deliberate, not a leak: the bucket's SSE key is the **AWS-managed**
`aws/s3` key, which is untagged and does not appear in the inventory — no CMK was
created precisely so teardown cannot leave a pending-deletion key behind.

### Tagged inventory (32 ARNs at report time)

| service | count | what they are |
| --- | --- | --- |
| ecs | 19 | `cluster/lop-poc`, `task-definition/lop-poc-agent:1`…`:5`, and 13 STOPPED `task/lop-poc/*` records (caveat 2 above; the 14th had already aged out) |
| ec2 | 6 | `vpc-02d4e704cc79e084e`, 2 subnets, `igw-07c639b7e920fed46`, `rtb-05bc0f8a367ea0635`, `sg-0af56e0923c4fce18` |
| logs | 2 | `/lop-poc/agent`, `/lop-poc/codebuild` |
| codebuild / ecr / s3 / secretsmanager | 1 each | `lop-poc-image-build`, `lop-poc-agent`, `lop-poc-results-325492156725`, `lop-poc/model-key` |

Obtained with `aws resourcegroupstaggingapi get-resources --tag-filters
Key=lop-poc,Values=true` on 2026-10-07; the same call is what `status` prints.

## Divergences from the spec and the design doc (short form)

Full text, with the error each one caused, is in `infra/remote-agents-poc/README.md`
§ Divergences. The ones that change the design document rather than just the POC:

1. **A non-root container works, and the mechanism is the image's `VOLUME`.** The
   image chowns `/workspace` then declares `VOLUME ["/workspace"]`, and the task
   definition sets `user: "10001:10001"`; because the VOLUME path equals the task
   volume's `containerPath`, ECS copies the image's data **and ownership** into the
   mount. Verified: probe 4d reads uid 10001 and writes `/workspace` on all 5 runs,
   with no privilege drop anywhere in the entrypoint. An earlier revision ran a
   root phase-0 that chowned the volume and re-exec'd itself; it worked and it is
   gone.
2. **`describe-tasks` does not return `runtimePlatform` for Fargate**, so §9.3 item
   1's "ARM64" is read from the task definition. Recorded beside it, as null.
3. **§9.2 item 10's resource list omits `executionRoleArn` and `taskRoleArn`.**
   ECS refuses a registration with container secrets and no execution role, and
   without a task role the credentials endpoint has no role to serve — probe 4a's
   subject.
4. **SSE-KMS with the AWS-managed key is `sseAlgorithm: aws:kms` with NO
   `kmsMasterKeyId`.** `aws/s3` as a key id passes `preview` and then fails every
   `PutObject` with `KMS.NotFoundException`; the provider's read normalises it and
   reports no diff, and `--replace` on that singleton sub-resource deletes the
   config it just created.
5. **A value-less Secrets Manager secret cannot back a task-level secret** — ECS
   fails the task before the entrypoint runs, so §9.2's mock-run step is
   unreachable as written. A placeholder value is written instead, and probe 4c
   searches for it exactly, so a mock run yields a real verdict rather than `null`.
6. **Presigned PUTs must be signed for the regional S3 host, and `curl` treats a
   307 as success.** Both were silent: the first artifact-bearing run exited 0 with
   nothing uploaded. The driver pins `s3v4` + virtual addressing; the entrypoint
   asserts a 2xx.
7. Smaller ones, each measured: the account id must come from `fn::invoke`
   `getCallerIdentity` (a numeric string config default reaches an interpolation as
   a float, `3.25492156725e+11`); CodeBuild reads an unquoted `echo "a: b"` as a
   mapping and fails the phase; `lop sessions` needs `--all` to list stored
   sessions; `ephemeralStorage` is omitted rather than declared as 20 GiB (the
   provider's range is 21–200); probe 4b's 5 s budget is per target, not per
   resolved address; `--exclude` URNs must use the Pulumi resource keys, not the
   AWS-side names.

**Not addressed here.** §4's "with and without SOCI" is half done: SOCI/Seekable
OCI lazy loading was not tested, so the 11–13 s image-pull share stands
un-improved. Nothing in this document used `LOP_POC_MODEL_KEY`, which does not
exist yet.


