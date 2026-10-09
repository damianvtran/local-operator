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
`6427a92be`, `fe76f09f1`, `242cc2f0e` plus the review-remediation commits up to
`18a55d8e5`. Pulumi state is a file backend at `$HOME/.lop-poc-pulumi-state`;
`pulumi login` is never run.

**The mock sections below and the real-key run.** Everything down to "Divergences" except
the "Real-key acceptance run" section was measured on **mock** runs (lop's built-in mock
provider, an injected placeholder in place of the model key), and those numbers are kept as
recorded. The run that turned items 2, 3 and the real half of 4 into PASSes is
`ct_92161251` (2026-10-08), a real OpenRouter turn on the same image digest; it has its own
section below and its own out-dir, `$HOME/.lop-poc-runs/real-20261008T025635Z`.

**The thing that was measured**

| | |
| --- | --- |
| Image | `325492156725.dkr.ecr.ca-central-1.amazonaws.com/lop-poc-agent@sha256:d70840cf1efb5fc9323946c3f8fcbc9cab838ac2c7680bf50c848b997d1b4055` (tag `0.68.3-…`), built by CodeBuild `lop-poc-image-build`, in-image smoke `docker run --entrypoint lop <img> --version` → `v0.68.3` before the push |
| Task definition | `lop-poc-agent:12` — FARGATE, ARM64, 2 vCPU / 4 GiB, read-only root filesystem, `user: 10001:10001`, empty task role, both IAM roles wired, **no `initProcessEnabled`** (divergence 22) |
| Fixture | https://github.com/olafagbemi/lop-poc-fixture @ `69db7e55fc14f918cccdf2fea62894fc37f1f642`, prompt "make the failing test `test_add` pass" |
| Cluster | `lop-poc`; app log group `/lop-poc/agent`; results bucket `lop-poc-results-325492156725` |
| Run population | **the five mock runs on the image digest above**; the real-key acceptance run below shares that digest and is tabulated in its own section — see "Which runs this measures" |

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
  --digest sha256:d70840cf1efb5fc9323946c3f8fcbc9cab838ac2c7680bf50c848b997d1b4055
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
| 2 | Bundle verifies; `git log` shows one commit on `lop/<id>` whose parent is the fixture SHA; `test_add` fails at that SHA and passes on the branch | **PASS** (PENDING until the real-key run) | `ct_92161251`, section below: `bundle_verify` → the bundle contains `a212a907f9e313766700c1154a4d890f826bbb4c` and requires `69db7e55fc14f918cccdf2fea62894fc37f1f642`, "is okay"; `one_commit` → 1 commit on `lop/ct_92161251`; `parent_is_fixture_sha` → `69db7e55…`; the fixture runner at the SHA **1/3 passed** (`FAIL test_add: -1 != 5`, `FAIL test_add_identity: -7 != 7`, `PASS test_sub`), on the branch **3/3 passed** (`PASS test_add: 5 == 5`). A *mock* run still makes no commit by design, so for one `verify` reports item 2 BLOCKED naming that reason — not FAIL |
| 3 | Session directory copied into an isolated config root opens and shows the full transcript | **PASS** | `verify` on `ct_69617f00`, and the OPEN half is now the real load path: after transplanting `session.tar.gz` into a fresh `mktemp` root, `lop exec --resume <id> --hosting test --model test-model --json ping` (stdin `/dev/null`) exits 0, reports the **same session id**, grows the transcript **8 → 16 lines**, and the first 8 lines are **byte-identical**. `lop sessions --all --json` in the same root lists it `state: stored`. `lop --resume` is the TUI form of the same store load; the headless surface of that load is what was driven. The transcript in that row is the mock provider's; the real-key run below re-ran **the same load path** on a real session and grew it **38 → 46 lines** with the first 38 byte-identical |
| 4 | Probe file shows 4a (creds endpoint, no policies), 4b (non-443 egress fails), 4c (no key on the filesystem) | **PASS** | `probes.json["failed"] == []` in all 5 runs; 4a/4b/4c plus 4c-env, 4d and **4f** (no `ps` reachable) verbatim below; 4e (the launch-environment watcher) separately below. The real run re-ran the same six with the real **73-byte** key and reports `failed: []` too (section below) |
| 5 | Cold start measured for 5 runs, recorded as evidence, replacing §4's estimates | **PASS** | Tables below, generated by `report`; 5 mock runs on the image digest in the header, plus the real run's own `report` table in its section below (same digest) |
| 6 | Cost Explorer for the tag reconciles within 20% of `Σ wall_seconds × $0.0869/3600` | **PARTIAL** — service proxy reconciled (−1.7%); tag half BLOCKED | Re-verified read-only 2026-10-09 at 05:36Z, account `325492156725`, `ca-central-1`, `UnblendedCost`, every bucket still `Estimated: true`: the POC's whole billed Fargate usage is the **two UTC days 2026-10-07 and 2026-10-08** — 1.5666666651 vCPU-h + 3.1333333349 GB-h = **0.7833 task-hours**, and 0.0388888889 vCPU-h + 0.0777777778 GB-h = **0.0194 task-hours**, together **0.8028 task-hours**; on-demand-equivalent **$0.0680403302 + $0.0016889445 = $0.0697292747** (`SavingsPlanCoveredUsage`), negated to a **net unblended $0.0001256602 + $0.0000373094 = $0.0001629696**; 10-05/10-06/10-09 read $0. The measurable half now reconciles: the two-day gross is **−1.7%** against the doc's own 40-task computed `$0.0709` (0.8161 h), inside the criterion's 20% band, and 0.8028 task-hours is above the 0.6667 task-h that 40 tasks' one-minute minimum billing requires. The tag half still cannot be evaluated: `Tags lop-poc=true` returns **$0 on every day**, because that key is not an activated cost-allocation tag in this linked account. The criterion names the tag, so the item cannot be a PASS. An earlier revision of this row recorded a single day at 0.5333 task-hours / $0.0463253312 — a mid-revision read of the same in-flight bucket, superseded (revision note under "Test 6 — cost", which also carries the commands, the `owner=lopdev` figure, the budget consequence and the applied criterion) |
| 7 | Teardown leaves zero tagged resources | **NOT RUN** | Awaiting approval; the commands and three caveats are below |

### Real-key acceptance run — `ct_92161251` (2026-10-08)

The run whose verdicts rows 2 and 3 above cite, and the only run in this document driven by a
real model: the placeholder key the mock sections use was replaced by the real
`LOP_POC_MODEL_KEY` (Secrets Manager → the task's secret, in memory), and the model was
`anthropic/claude-sonnet-4.5` on OpenRouter.

| | |
| --- | --- |
| Command class | `.venv/bin/python scripts/remote_agents_poc.py run --runs 1 --out-dir "$HOME/.lop-poc-runs/real-20261008T025635Z"` — **no `--mock`** |
| Run / task | `ct_92161251`; `arn:aws:ecs:ca-central-1:325492156725:task/lop-poc/1b878de50efd4f56b34dcd23a7fd53dc` |
| Task definition / image | `lop-poc-agent:12`; `325492156725.dkr.ecr.ca-central-1.amazonaws.com/lop-poc-agent@sha256:d70840cf1efb…` — **the same digest the five mock runs above ran** |
| Outcome (item 1) | `lastStatus: STOPPED`, `desiredStatus: STOPPED`, `stopCode: EssentialContainerExited`, `stoppedReason: Essential container in task exited`, container `agent` `exitCode: 0`; task tags `lop-poc=true`, `owner=lopdev`, `run-id=ct_92161251` |
| Fixture / prompt | `https://github.com/olafagbemi/lop-poc-fixture.git` @ `69db7e55…`, "make the failing test `test_add` pass" |
| Session / branch | session `7c840687bebe`; branch `lop/ct_92161251`, commit `a212a907f9e313766700c1154a4d890f826bbb4c` |
| The model's turn | `exec.jsonl`: 191 events, `agent_start` → `agent_end`, 11 turns, **10 tool executions** (`bash`, `read`, `edit`), `anthropic/claude-sonnet-4.5` on all 11 provider turns |

**`verify` on this run dir: 22 checks, 0 FAIL, 0 BLOCKED.** (Every key-independent row — 21 of
the 22 — was re-run on this host while writing this section and all 21 PASSed; the
twenty-second, `acceptance4.key_scan`, scans *for* the key and prints only the verdict, so it
needs the key present in the store and is quoted from the run's own environment.) The re-run was
made with the ambient `PYTHONDONTWRITEBYTECODE` and any `PYTHONPYCACHEPREFIX` redirect explicitly
removed from the environment, which is the state a fresh checkout or a CI host presents:
**22 checks, 0 FAIL, 0 BLOCKED** there too. That is the point rather than a coincidence — the
driver sets the bytecode switch for the fixture runner itself, so `verify` answers about the
recorded run instead of about the host that runs it (the note under item 2). The check
set is 11 acceptance-2 checks (a mock run carries one BLOCKED row there instead), 8
acceptance-3 checks, `acceptance4.no_probe_failed`,
`acceptance4e.no_key_in_any_process_environ` and `acceptance4.key_scan`:

```sh
export AWS_PROFILE=minerva_sandbox AWS_REGION=ca-central-1
.venv/bin/python scripts/remote_agents_poc.py verify \
  "$HOME/.lop-poc-runs/real-20261008T025635Z/ct_92161251" \
  --fixture-url https://github.com/olafagbemi/lop-poc-fixture.git \
  --fixture-sha 69db7e55fc14f918cccdf2fea62894fc37f1f642
```

Item 2, as that run's `verify` reports it (each row's multi-line detail folded onto one
line, otherwise as printed):

```
PASS  acceptance2.bundle_verify: The bundle contains this ref: a212a907… refs/heads/lop/ct_92161251
      The bundle requires this ref: 69db7e55…  … is okay
PASS  acceptance2.one_commit: 1 commit(s) on refs/heads/lop/ct_92161251
PASS  acceptance2.parent_is_fixture_sha: parent 69db7e55fc14f918cccdf2fea62894fc37f1f642
PASS  acceptance2.test_fails_at_fixture_sha: exit 1: FAIL test_add: -1 != 5 / FAIL test_add_identity: -7 != 7 / PASS test_sub: 2 == 2 / 1/3 passed
PASS  acceptance2.test_passes_on_branch: exit 0: PASS test_add: 5 == 5 / PASS test_add_identity: 7 == 7 / PASS test_sub: 2 == 2 / 3/3 passed
```

The commit fixes `calc.add` (`return a - b` → `return a + b`) and carries, beside it,
`__pycache__/calc.cpython-312.pyc` (442 B) — the container's Python 3.12 bytecode, which the
agent committed along with the fix. It changes no test, but it means the BRANCH that commit
produced tracks a bytecode file the fixture runner regenerates the moment it imports `calc` (the
merge base, `69db7e55…`, tracks none): the fixture-SHA run writes that path as an **untracked**
file, and the checkout of the branch then refuses to overwrite an untracked file. `verify`
answers that itself rather than leaving it to whoever runs it — it gives the fixture runner an
environment it BUILDS (PATH and the locale family, plus `PYTHONDONTWRITEBYTECODE=1`, and nothing
else, so the caller's credentials and `LOP_*` store variables never cross) — which is what makes
the verdict a statement about the recorded run and not about the host.

With the ambient switch unset that is the difference between a PASS and a FAIL: the pre-fix
driver's fixture-SHA run left the untracked bytecode, the branch checkout refused it, and
`acceptance2.checkout_branch` was the one row that FAILed — **20 PASS / 1 FAIL / 0 BLOCKED of the
21 rows printed**, every other row unchanged — on an artifact that is itself correct. The table
is one row short of the 22 a passing run prints because the failing row RETURNS EARLY, so
`acceptance2.test_passes_on_branch` is never recorded: a FAIL here does not just mark a row, it
drops the rows after it. Both directions are pinned by
`tests/unit/test_remote_agents_poc_verify_isolation.py`.

Item 3, on the transplanted **real** session (the session listing elided):

```
PASS  acceptance3.transcript_non_empty: 38 transcript line(s) before the resume
PASS  acceptance3.lop_sessions_lists_id: exit 0: [{ "state": "stored", … "session_id": "7c840687bebe" …}]
PASS  acceptance3.resume_reuses_session: exit 0; same session id in the event stream: True
PASS  acceptance3.resume_grew_transcript: 38 -> 46 transcript line(s)
PASS  acceptance3.resume_kept_prior_lines: the first 38 line(s) are byte-identical after the resume
```

Item 4, the same six probes with the real key (`failed: []`, every probe `pass: true`):

| probe | reading from this run |
| --- | --- |
| 4a creds endpoint | `caller_identity: arn:aws:sts::325492156725:assumed-role/lop-poc-task/1b878de5…`; `s3_list_buckets`, `ecs_list_clusters`, `secretsmanager_get_secret_value` all `denied` |
| 4b egress | `github.com:443` "connected in 15 ms"; `1.1.1.1:53`, `example.com:80`, `github.com:22` and `portquiz.net:8080` time out, `169.254.169.254:80` is `OSError` |
| 4c no secret on disk | `files_scanned: 16708`, `key_length_bytes: 73`, `key_sha256_first8: 6ade1e4f`, `matches_by_needle {prefix: 0, value: 0}` |
| 4c-env | `child_environment_entries: 38`, `entries_containing_key: 0`, `entry_names: []` |
| 4d platform | `uname_m_is_aarch64`, `uid_is_10001`, `root_is_read_only`, `usr_is_read_only`, `workspace_is_writable` — all true |
| 4f no `ps` | `which_ps: null`, `paths_present: []`, `multiplexers_present: []` |

4e (the launch-environment watcher) over this real run: `samples: 37`,
`processes_scanned_max: 5`, `matches_by_needle {prefix: 0, value: 0}`,
`observed_processes: [{pid 1, bash}, {pid 46, python}, {pid 50, bash}, {pid 51, timeout},
{pid 53, Local Operator}]`, `pass: true`; the residual is the same as the mock runs' (the
target's `/proc/<pid>/mem` is unreadable, `yama_ptrace_scope: 1`). This is the run the
coexistence self-test below exists for: the agent **spawned real tool children** here, and the
scan found the key in none of them.

The key's own trail, from this run: `agent_stderr.txt` reads `lop-launch: read 73 key byte(s)
from fd 3; provider_env='OPENROUTER_API_KEY'; set=True`, then `lop-launch: set
OPENROUTER_API_KEY in-process (73 bytes); /proc/self/environ clean: True`. The driver's rescan
of the download, `results/key_scan.json`, is `{"refused_upload": false, "match_count": 0,
"scanned_files": 12}`. **No key value is recorded in this document, in the driver's argv (the
key is piped on an fd) or in the run's artifacts.**

Item 5 — this run's own cold start, generated by `report` on its out-dir (one run, so every
summary cell equals its per-run cell):

```sh
.venv/bin/python scripts/remote_agents_poc.py report \
  "$HOME/.lop-poc-runs/real-20261008T025635Z" --digest sha256:d70840cf1efb5fc9323946c3f8fcbc9cab838ac2c7680bf50c848b997d1b4055
```

| run id | task arn (suffix) | cpuArch | stopCode | exit | RunTask→RUNNING s | scheduling/ENI s | image pull s | container start s | probes s | RunTask→1st model event s | minus probes s | probes | 4e |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ct_92161251 | `…cd23a7fd53dc` | ARM64 | EssentialContainerExited | 0 | 24.78 | 14.65 | 5.23 | 3.76 | 21.77 | 50.07 | 28.31 | P/P/P/P/P/P | P |

Wall clock for the run, `createdAt` → `stoppedAt`: **109.353 s** (20:56:37.560 → 20:58:26.913
−06:00).

After the run, `status` verbatim:

```
OK: no RUNNING or PENDING tasks in lop-poc; 27 tagged lop-poc resources at 2026-10-08T02:58:40Z
```

(a re-read at 2026-10-08T03:02:18Z returned the same 27 — the count moves with record ageing,
caveat 2 below.)

**Not closed by this run.** Item 6 moves from **computed** to **measured, not reconciled** (Test 6
below): the Cost Explorer lag blocking it has cleared for the 2026-10-07 billing day, so the
billed figures now exist — but the tag-filtered query the criterion names still cannot work in
this account and the usable slice does not reconcile within 20%, so the verdict is a PARTIAL, not
a PASS, and this run's 109.353 s is still not inside the measured day. Item 7 stays **NOT RUN** — the stack is still up and teardown still awaits
approval. Nothing here relaxes the isolation findings, the `ps` residual (SEC-11) or the
security caveats in the divergences and the design doc's §7.

### Test 4 — the probes, verbatim (`ct_69617f00`, trimmed)

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
     "must_be_reachable": {"github.com:443": "connected in 17 ms"},
     "must_be_unreachable": {
       "1.1.1.1:53": "failed in 5004 ms: TimeoutError",
       "169.254.169.254:80": "failed in 0 ms: OSError",
       "example.com:80": "failed in 5004 ms: deadline exhausted",
       "github.com:22": "failed in 5005 ms: TimeoutError",
       "portquiz.net:8080": "failed in 5000 ms: TimeoutError"}}},
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
| **COEXISTENCE, in the container** | one run with `POC_ENVIRON_WATCH_COEXIST=1` (task `abdac194ab13…`, out-dir `coexist_r2_abdac194ab13`): the real launcher holds the key in-process and spawns `sh -c 'sleep 20'` **through the product's own filter** while 4e samples | `samples: 6`, `processes_scanned_max: 5`, `observed_processes: [{pid 1, bash}, {pid 43, python}, {pid 46, sh}, {pid 47, sleep}, {pid 48, python}]`, `pass: true`, 0 matches — the child the row is about (`pid 46`, `sh`, equal to the launcher's own `child_pid`) is **in the scanned set**, so the window is the window. The launcher's half: `{"child_argv": ["sh", "-c", "sleep 20"], "child_env_entries": 2, "child_exit": 0, "child_pid": 46, "child_sleep_seconds": 20, "expected_key_in_child_env": false, "key_value_in_child_env": false, "mode": "filtered", "parent_self_environ_clean": true, "provider_var_in_child_env": false, "pass": true}` after `lop-launch: set LOP_POC_MODEL_KEY in-process (26 bytes); /proc/self/environ clean: True`. A parent holding the key and a child that inherited neither its name nor its value — **and the by-NAME half of that is asserted by the DRIVER, not by the launcher's `pass` predicate**: the launcher checks the value only, and tightening it would mean changing the image the accepted runs were made with, so `verify <self-test dir>` fails when `provider_var_in_child_env != expected_key_in_child_env`. That check, and the coverage column beside it, are in `report`'s self-test panel below. |
| **round-2 reading, SUPERSEDED** | out-dir `coexist_r2_fa29af2b7a9d` (task `f88619b98cd6…`) | **the child exited before the first sample**: a 2-second child behind a hard-coded 3-second pre-watch sleep left `processes_scanned_max` at 2 — the container's baseline of PID 1 plus the watcher — so that run never observed the coexistence window it was quoted for (agent review round 3, finding 1). Kept as one line because the correction under it is only checkable against what it corrected; its record also predates the `mode`/`expected_key_in_child_env` fields, which is why `report` does not list it. |
| **RED, in the container, in the SAME shape** | one run with `POC_ENVIRON_WATCH_SELFTEST=1` (task `b8dc22d28d7c…`, out-dir `selftest_r2_22a95113061a`): the same launcher, the same `sh -c 'sleep 20'`, with the environment **inherited** rather than filtered — `{**os.environ, "LC_ALL": "C"}`, which is what `tools/group_reaper.py:229` passes to `ps` and what `memory_guard._default_runner` does by passing no `env=` at all | `samples: 6`, `processes_scanned_max: 5`, `observed_processes` the same five, `pass: false`, `matches_by_needle: {"value": 2}`, `matching_processes: [{"comm": "sh", "pid": 46, "needle": "value"}, {"comm": "sleep", "pid": 47, "needle": "value"}]` — the watcher sees the key in the child AND in its grandchild — and task exit code **1**. The launcher's own half reports `mode: "inherited"`, `expected_key_in_child_env: true`, `key_value_in_child_env: true`, `pass: true`: the shape is as leaky as it is designed to be. Same window, same child argv, one difference — the environment. (The round-1 instance, task `813531fe54e2…`, uploaded no artifact before the branch learned to tar `$OUT`; its reading is in CloudWatch stream `lop-poc/agent/813531fe54e2443c9fc2a5b706b3b4f9` and nowhere else.) |
| **RED/GREEN, hermetic** | `tests/unit/test_remote_agents_poc_probes.py` (six cases: red, green, prefix-only, blocked-without-procfs, rescan exit codes, and 4f by name / absolute path / multiplexer) | passes; the environ source is stubbed because a real one needs procfs |
| **no re-exec in the launcher** | every run's `agent_stderr.txt` | `lop-launch: branding re-exec plan: None` — `reexec_branded` is a no-op for a launch through `lop_launch.py`, so nothing re-execs with the key in its environment |
| **key arrived over the fd** | every run's `agent_stderr.txt` | `lop-launch: read 26 key byte(s) from fd 3; provider_env=''; set=False` (a mock run sets no provider variable, which is why `set=False` is correct rather than a failure) |
| **the entrypoint scrubs itself** | the key is read and unset at the top of the script, then `exec "$0"` re-execs it | the entrypoint's own `/proc/<pid>/environ` is clean from that exec on, and the two children that run before it — `$(id -u)` became `$EUID`, a builtin, and `stamp` moved below the unset (SEC-12) — are gone, which is why the watcher's 0 includes PID 1 |

#### The same three properties, as `report` renders them from the recorded fields

```
.venv/bin/python scripts/remote_agents_poc.py report <out-dir> \
  --digest sha256:d70840cf1efb5fc9323946c3f8fcbc9cab838ac2c7680bf50c848b997d1b4055
```

| out-dir | mode | child pid | child argv | child env entries | variable name | key value | watcher | samples / scanned_max | child in observed set | driver verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `coexist_r2_abdac194ab13` | filtered | 46 | `sh -c sleep 20` | 2 | False (expected False) | False (expected False) | GREEN | 6 / 5 | yes | **PASS** — mode=filtered: the variable name and the value are both absent, as this mode expects |
| `selftest_r2_22a95113061a` | inherited | 46 | `sh -c sleep 20` | 39 | True (expected True) | True (expected True) | RED | 6 / 5 | yes | **PASS** — mode=inherited: the variable name and the value are both present, as this mode expects |

The by-NAME column is the driver's check, not the launcher's `pass` predicate: the
launcher asserts the VALUE only, and tightening it would mean changing the image the
accepted runs were made with. `verify <self-test dir>` runs the same three checks
(`selftest.child_env_matches_mode`, `selftest.watcher_agrees_with_mode`,
`selftest.child_was_observed`) and exits non-zero if any of them fails; a self-test
directory needs no `--fixture-sha`, because it is not a run.

**The residual, as measured, not as asserted.** `yama_ptrace_scope: "1"` and, across the
five accepted runs, `mem_target: {"comm": "Local Operator", "pid": 50–52}` (each run's
own agent pid; not one value quoted for the set) with `mem_openable: false`,
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
| ct_e9958415 | `…4d73c1b32cd0` | ARM64 | EssentialContainerExited | 0 | 22.59 | 14.13 | 4.50 | 3.18 | 21.21 | 45.31 | 24.10 | P/P/P/P/P/P | P |
| ct_ba2a8ccc | `…31abb31b1d42` | ARM64 | EssentialContainerExited | 0 | 24.78 | 15.43 | 7.27 | 0.96 | 21.51 | 47.79 | 26.28 | P/P/P/P/P/P | P |
| ct_5932eceb | `…fcb940a2205f` | ARM64 | EssentialContainerExited | 0 | 25.90 | 17.34 | 4.35 | 3.10 | 21.17 | 48.15 | 26.97 | P/P/P/P/P/P | P |
| ct_69617f00 | `…7fc4ee04c4c4` | ARM64 | EssentialContainerExited | 0 | 19.96 | 10.82 | 4.53 | 3.13 | 21.14 | 41.74 | 20.60 | P/P/P/P/P/P | P |
| ct_58feb16e | `…922f47e96496` | ARM64 | EssentialContainerExited | 0 | 25.86 | 16.24 | 6.99 | 1.19 | 21.47 | 48.48 | 27.01 | P/P/P/P/P/P | P |

Min / median / max in seconds, n = 5 — the rest of the same `report` output:

| phase | min | median | max |
| --- | --- | --- | --- |
| RunTask call → first RUNNING (driver wall clock) | 19.96 | 24.78 | 25.90 |
| **scheduling + ENI attach** (`createdAt` → `pullStartedAt`) | 10.82 | 15.43 | 17.34 |
| **image pull** (`pullStartedAt` → `pullStoppedAt`) | 4.35 | 4.53 | 7.27 |
| **container start** (`pullStoppedAt` → `startedAt`) | 0.96 | 3.10 | 3.18 |
| the three phases above as one span (`createdAt` → `startedAt`) | 18.48 | 23.66 | 24.78 |
| container's first stamp → probes start | 0.23 | 0.24 | 0.35 |
| **probes (4a–4f, incl. the whole-filesystem key scan)** | 21.14 | 21.21 | 21.51 |
| probes done → first model event | 1.35 | 1.48 | 1.75 |
| container start → first model event | 22.72 | 22.92 | 23.50 |
| RunTask → first model event (raw) | 41.74 | 47.79 | 48.48 |
| **RunTask → first model event, minus the probes** | 20.60 | 26.28 | 27.01 |

**What dominates.** Three phases, kept apart because they answer different questions and
one of them was mislabelled in the first draft of §4: **scheduling and ENI attachment**
is the largest single phase (15.4 s median), the **image pull is 4.53 s**, and the
**container start 3.10 s** — so image pull is ~18% of the 24.8 s arrival, not the ~57% an
earlier revision claimed, and SOCI (whose case the pull is) has a ~4.5 s ceiling here
rather than a 12 s one. The published "cold start" of ~47.8 s is then **~21 s of
instrument**: probe 4c reads every regular file (16 708) before the agent starts, by
design. Subtracting it, a task is ready for a model turn **26.3 s (median) after
`RunTask`**, and the agent's own first token lands 1.5 s after the probes finish. The
three ECS phases span `createdAt`→`startedAt` (**23.7 s median**); the ~1.1 s between
that and the RunTask figure is the control plane's own round trip, not a phase.

**Which runs this measures.** The five above are **every mock** run on
`sha256:d70840cf1efb…`, and that population is complete by construction: every artifact
directory records the image digest its task actually ran, and grouping **the mock out-dir's
40 recorded tasks** by that digest gives the table below. The real-key acceptance run below
is a **sixth** run on the same digest; it is tabulated in its own section rather than merged
into these, because the mock out-dir these tables aggregate (`/tmp/poc-runs`) is no longer
on this host, so the 40-task sums cannot be recomputed here. Earlier cycles of the build-out are visible
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
| `sha256:f9ce54119b392…` | **5** | ct_54c56e7a, ct_31c28ef9, ct_ed98df28, ct_286e4fba, ct_2bc804c9 |
| `sha256:d70840cf1efb5…` **← this table** | **5** | ct_e9958415, ct_ba2a8ccc, ct_5932eceb, ct_69617f00, ct_58feb16e |
| `sha256:133e1f0ea1e78…` | **1** | ct_17ececef |
| `sha256:094c8b0aec83e…` | **1** | ct_6146eed7 |
| `sha256:3c2efb1e536d5…` | **1** | ct_a7a20b4d |
| `sha256:9ddccaa6cf879…` | **1** | ct_32c487ba |

total recorded runs: 40

### Test 6 — cost

Two numbers, two different claims. `Σ wall_seconds × $0.0869/3600` over **all 40 recorded
tasks** is the **computed** figure: `Σ wall = 2937.8 s = 0.8161 h` → **$0.0709**; the five
acceptance runs are 362.9 s → **$0.0088**. The two watcher self-test tasks (the coexistence
proof and the red proof, each its own RunTask) add 179.0 s → ~$0.0043, which is in the same
table's class but not in the 40, because they carry no `run.json`. Per-task wall times were
46.3 s to 89.8 s. Nothing exceeded the 2 h + 10 min client deadline, so no `StopTask` was ever
issued. The real-key run is a 41st task outside that sum: 109.353 s → **$0.0026** computed.

**Measured**, read-only on 2026-10-09 at 05:36Z with `AWS_PROFILE=minerva_sandbox`,
`AWS_REGION=ca-central-1`; `aws sts get-caller-identity` → account `325492156725`. Every figure
below was re-queried for this revision rather than copied from the earlier note, and is Cost
Explorer `UnblendedCost` / `UsageQuantity` over the whole window 2026-10-05→10-09:

```sh
export AWS_PROFILE=minerva_sandbox AWS_REGION=ca-central-1
aws sts get-caller-identity
aws ce get-cost-and-usage --time-period Start=2026-10-05,End=2026-10-10 \
  --granularity DAILY --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Container Service"]}}' \
  --group-by Type=DIMENSION,Key=RECORD_TYPE
aws ce get-cost-and-usage --time-period Start=2026-10-05,End=2026-10-10 \
  --granularity DAILY --metrics UsageQuantity UnblendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Container Service"]}}' \
  --group-by Type=DIMENSION,Key=USAGE_TYPE
```

The POC's **whole billed Fargate usage is the two UTC days 2026-10-07 and 2026-10-08**;
2026-10-05, 10-06 and 10-09 each return a $0 ECS total. Every day comes back `Estimated: true`,
so neither billed day is final and a further small movement is possible — which is exactly what
happened between revisions (see the revision note below).

**2026-10-07**

| usage type | quantity | unblended |
| --- | --- | --- |
| `CAN1-Fargate-ARM-vCPU-Hours:perCPU` | 1.5666666651 Hrs | covered by the Savings Plan (below) |
| `CAN1-Fargate-ARM-GB-Hours` | 3.1333333349 Hrs | covered by the Savings Plan (below) |
| `CAN1-DataTransfer-Out-Bytes` | 0.0003739087 GB | $0.0000336528 |
| `CAN1-DataTransfer-Regional-Bytes` | 0.0091578598 GB | $0.0000915783 |
| `CAN1-DataTransfer-In-Bytes` | 0.0623753794 GB | $0 |
| `CAN1-USE1-AWS-Out-Bytes` | 0.0000216216 GB | $0.0000004324 |
| `CAN1-USE1-AWS-In-Bytes` | 0.0000173962 GB | $0 |

**2026-10-08**

| usage type | quantity | unblended |
| --- | --- | --- |
| `CAN1-Fargate-ARM-vCPU-Hours:perCPU` | 0.0388888889 Hrs | covered by the Savings Plan (below) |
| `CAN1-Fargate-ARM-GB-Hours` | 0.0777777778 Hrs | covered by the Savings Plan (below) |
| `CAN1-DataTransfer-Out-Bytes` | 0.0003927666 GB | $0.000035349 |
| `CAN1-DataTransfer-Regional-Bytes` | 0.0001960369 GB | $0.0000019604 |
| `CAN1-DataTransfer-In-Bytes` | 0.0020409618 GB | $0 |

The two `CAN1-USE1-AWS-*` rows are the task's cross-region traffic (`USE1` is `us-east-1`, so
this is not the intra-region `CAN1-DataTransfer-*` pair); being non-Fargate usage they sit in the
`Usage` record type alongside the three `CAN1-DataTransfer-*` rows. The five data-transfer rows
sum to the day's `Usage` line exactly — $0.0001256635 on 10-07 and $0.0000373094 on 10-08:

```sh
aws ce get-cost-and-usage --time-period Start=2026-10-05,End=2026-10-10 \
  --granularity DAILY --metrics UnblendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon Elastic Container Service"]}}'
# 10-05 $0 / 10-06 $0 / 10-07 $0.0001256602 / 10-08 $0.0000373094 / 10-09 $0
```

1.5666666651 vCPU-h + 3.1333333349 GB-h is **0.7833 task-hours** of a 2 vCPU / 4 GiB task, and
at §8.1's rate it prices at 1.5666666651 × $0.03565 + 3.1333333349 × $0.00389 = **$0.0680403** —
exactly the day's `SavingsPlanCoveredUsage` record. The same holds on the second day:
0.0388888889 × $0.03565 + 0.0777777778 × $0.00389 = **$0.0016889**, the 10-08
`SavingsPlanCoveredUsage`, so the rate and the billed hours agree on both days:

| record type | 2026-10-07 | 2026-10-08 |
| --- | --- | --- |
| `SavingsPlanCoveredUsage` | **$0.0680403302** | **$0.0016889445** |
| `SavingsPlanNegation` | **−$0.0680403335** | **−$0.0016889445** |
| `Usage` (the data-transfer rows above) | $0.0001256635 | $0.0000373094 |
| **ECS service total** | **$0.0001256602** | **$0.0000373094** |
| **two-day ECS total** | **$0.0001629696** | |

**An existing Savings Plan paid for the Fargate charge.** `SavingsPlanCoveredUsage` is the
on-demand-equivalent value of what the plan covered ($0.0697292747 over the two days);
`SavingsPlanNegation` cancels it back to the plan's already-paid rate. So the POC's **net cash**
for the two days' Fargate compute was **~$0.0002** while the **on-demand-equivalent value was
$0.0697**. There is no `SavingsPlanRecurringFee` line on either day — the plan had spare
commitment and the POC did not add to it.

**Attribution by tag does not work here, and the tag is what the criterion names.**

| query, daily (2026-10-05→10-09) | 10-05 | 10-06 | **10-07** | **10-08** | 10-09 |
| --- | --- | --- | --- | --- | --- |
| `--filter '{"Tags":{"Key":"owner","Values":["lopdev"]}}'` | $0 | $0 | **$0.0978767782** | **$0.0116441516** | $0 |
| `--filter '{"Tags":{"Key":"lop-poc","Values":["true"]}}'` | $0 | $0 | **$0** | **$0** | $0 |

`lop-poc=true` reads $0 on every day in the window while `owner=lopdev` reads $0.0979 and $0.0116
on the two billed days, so the POC's own key is **not an activated cost-allocation tag**: this
account is a LINKED account, and activation is the management account's (`212841448981`) action,
not this one's.

```sh
aws ce list-cost-allocation-tags --status Active
# An error occurred (AccessDeniedException) when calling the ListCostAllocationTags operation:
# Failed to list Cost Allocation Tags: Linked account doesn't have access to cost allocation tags.
```

The `owner=lopdev` two-day total, $0.1095209298, breaks down as ECS $0.0681659937 + $0.0017262539,
ECR $0.0219794251 + $0.0027079442, Secrets Manager $0.0023578518 + $0.0070142472, VPC $0.00410001 +
$0.000131945, S3 $0.0008757084 + $0.0000167096, CloudWatch $0.0003977892 + $0.0000470517, CodeBuild
$0 on both days. The ECS lines are the on-demand-equivalent Fargate plus the data transfer above
(the tag view carries no `SavingsPlanNegation`), so adding the account's two-day negation back
(0.0680403335 + 0.0016889445) gives a **POC net cash of ≈$0.0398** over the two days, only
≈$0.0002 of which is Fargate compute; the rest is ECR $0.0247, Secrets Manager $0.0094, NAT
$0.0042, S3 $0.0009 and logs $0.0004.

The ECS filter is sound *for these days* because nothing else in the account ran Fargate: the
account-wide ECS totals are the same $0.0001256602 and $0.0000373094 the POC's usage produced.

**Consequence for the budgets — plainly.** Both are $25/month COST budgets (`describe-budget`
returns `BudgetLimit.Amount "25.0"`, `Unit USD`, for each):

* `lop-poc` filters on `CostFilters.TagKeyValue = ["user:lop-poc$true"]` — the same key that
  attributes nothing — and reads `ActualSpend 0.0`. **It cannot fire on this spend, and it cannot
  be relied on at all** until the management account activates the tag. This is not softened by
  the money being small; it is inert.
* `lop-poc-fargate` filters on `CostFilters.Service = ["Amazon Elastic Container Service"]`,
  needs no activation, and is **the only working backstop** — account-wide for ECS rather than
  POC-scoped, so it is sound only while no other Fargate task runs here. It too reads
  `ActualSpend 0.0` today, because the covered charge nets below a cent.

**The criterion applied, and the verdict.** §9.3 item 6 asks for *Cost Explorer for the tag* to
reconcile within 20% of `Σ wall_seconds × $0.0869/3600` — both a tag-filtered query and the 20%
band. One half now holds; the other still cannot be evaluated:

1. **The tag half cannot be evaluated at all.** The POC's key attributes $0 and the working
   `owner=lopdev` filter is a superset of the compute the criterion is about. **BLOCKED** on a
   management-account action.
2. **The measurable proxy now reconciles.** The two billed days carry 0.8028 task-hours, an
   on-demand-equivalent **$0.0697292747** gross against the doc's own 40-task computed $0.0709
   (2937.8 s = 0.8161 h) — **−1.7%**, inside the criterion's 20% band. (Net cash is −99.8% of the
   computed figure only because the Savings Plan paid for the compute.) The earlier revision's
   objection — that 0.5333 h sat below the 0.6667 task-h that 40 tasks of Fargate's one-minute
   minimum billing require, so the published slice covered only part of the run population — is
   removed by the settled figures: 0.7833 h on 10-07 is ≈2,820 s of the 2,937.8 s recorded run
   total, and the two-day 0.8028 h is above that floor. CE's ~24 h lag has cleared for both billed
   days; both are still `Estimated: true`, so a further small movement remains possible.

Verdict: **PARTIAL — proxy reconciled, tag half BLOCKED**, applied as written. Not a PASS: the
criterion names a tag-filtered query, and this linked account cannot produce one — the spend is
selectable only by service or by the `owner` tag. Not a FAIL of the POC either: the settled bill
agrees with the model's own 40-task estimate to within 2%, and its on-demand-equivalent value
matches §8.1's rate to four significant figures. What is missing is a tag that can select the
spend; the compute itself now reconciles.

**Revision note — the earlier figures were a mid-revision read.** An earlier revision of this
section recorded a single billed day, 2026-10-07, as 1.0666666656 vCPU-h + 2.1333333344 GB-h =
0.5333 task-hours, `SavingsPlanCoveredUsage` $0.0463253312, `SavingsPlanNegation` −$0.0463253335,
net $0.0001256431, with an `owner=lopdev` day of $0.0514468804. That was the *same* in-flight
billing day read while Cost Explorer was still revising the bucket, and the settled figures above
(re-read 2026-10-09 at 05:36Z) supersede it: the day grew to 0.7833 task-hours and $0.0680403302
covered, and the `owner=lopdev` view grew with it to $0.0978767782 — ECR $0.0219794251 and
Secrets Manager $0.0023578518 had not posted at the earlier read, and the account-wide ECS total
is not the $0 that revision reported for 10-08. The superseded numbers are not wrong so much as
incomplete, and none of them should be quoted as the POC's cost. Both days remain
`Estimated: true`, so treat the totals above as the current settled reading, not a frozen final
bill (agent review round 7, MINOR 1).

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

### Tagged inventory (46 ARNs at report time)

| service | count | what they are |
| --- | --- | --- |
| ecs | 33 | `cluster/lop-poc`, `task-definition/lop-poc-agent:1`…`:12`, and 20 STOPPED `task/lop-poc/*` records (caveat 2: they age out ~1 h after each stop, which is why this count moves between readings — it has read 45, 42, 51, 46, then **27** at 2026-10-08T02:58:40Z after the real run below and the ageing-out of the mock era's STOPPED records) |
| ec2 | 7 | the VPC, 2 subnets, IGW, route table, SG, plus the adopted default SG |
| logs | 2 | `/lop-poc/agent`, `/lop-poc/codebuild` |
| codebuild / ecr / s3 / secretsmanager | 1 each | `lop-poc-image-build`, `lop-poc-agent`, `lop-poc-results-325492156725`, `lop-poc/model-key` |

Obtained with `aws resourcegroupstaggingapi get-resources --tag-filters
Key=[redacted]` on 2026-10-07; the same call is what `status` prints, whose verdict line
was, verbatim and with its own timestamp:

```
OK: no RUNNING or PENDING tasks in lop-poc; 46 tagged lop-poc resources at 2026-10-07T21:45:44Z
```

The timestamp is in the line because the count moves as task records age out, which is
how a "45" ended up quoted under a "51 ARNs" heading (agent review round 3, finding 3).

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
   through `shell_env.child_environment` while 4e samples), with `--child-sleep 20` so
   the child outlives the sampling window, and the watcher records `observed_processes`
   (PIDs and comms only) so a reader can check the child was in the scanned set rather
   than take the coverage on trust. The red case runs the SAME launcher with the same
   child argv and the environment **inherited** instead of filtered, so green and red
   differ in one thing. Both upload their artifacts into the same out-dir.
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
that item is deferred in the PR thread. The mock sections of this document used the
injected placeholder, never a real key; the real-key acceptance run in its own section did
use one — delivered over an fd, never on disk and never logged (probes 4c, 4c-env and 4e,
and the driver's own rescan, below).
