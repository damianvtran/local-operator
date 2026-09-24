# OSWorld 2.0: apparatus and methods

How this harness runs an OSWorld 2.0 episode, what is pinned, what is
recorded, and what a reader must know before treating a number produced here
as a result. It is the methods section for the OSWorld 2.0 work, not a
runbook: the operator commands it names are the ones under
[Reproduction](#reproduction), and the build recipe itself lives in
[`benchmarks/osworld_v2_adapter/README.md`](../../../benchmarks/osworld_v2_adapter/README.md),
which this document cross-references rather than duplicates. For the harness's
own cost and throughput measurements — a different subject — see
[`docs/BENCHMARKS.md`](../../BENCHMARKS.md).

**Historical baseline (2026-09-02):** one paid episode had failed on its first
decision and no score existed at that point. The dated observations under
[Status and honest limitations](#status-and-honest-limitations) preserve that
baseline, not the current episode count. See [the matched-model pilot ledger](MODEL_PILOT.md)
for subsequent attempts, current apparatus validation and reference caveats.
See the [ten-task developmental pilot](PILOT10_2026_09_05.md) for the frozen
sample, all ten first attempts, confirmed completion, costs, and apparatus
amendments. Neither an individual score nor the offline checks below establish
a full-suite benchmark result.

## 1. The benchmark

OSWorld 2.0 (`osworld-v2-2026.08.08`) is a computer-use benchmark of **108
long-horizon workflows** on a real Ubuntu desktop. Upstream reports a median
human completion time of about 1.6 hours per task and an average of ~318 tool
calls for a frontier model, against about 30 in OSWorld 1.0; its headline
figures in the original paper are quoted **under a binary-completion metric at
500 steps**, with 20.6% completion (54.8% partial) for its best reported
configuration. That historical paper result is not a current frontier claim. Tasks are
scored by per-task Python evaluators shipped with the corpus, averaging 27.25
scoring checkpoints per task.

**It is not OSWorld-Verified, and the two must never be conflated.** The
original OSWorld (v1) is a 369-task benchmark of short, mostly single-app
tasks (a 361-task variant excludes 8 Google Drive tasks needing manual setup).
A number from one says nothing about the other: different corpus, different
task lengths, different step budgets, different evaluators. Nothing in this
apparatus runs OSWorld 1.0 or OSWorld-Verified, and no comparison to their
leaderboards is licensed by anything here.

### What is pinned, and by what

The pin is the file `benchmarks/osworld_v2_adapter/config/release-v2026.08.08.json`,
committed in-tree. Every value below is read from it, not from prose:

| Pin | Value |
| --- | --- |
| Release name | `osworld-v2-2026.08.08` |
| Release manifest sha256 | `afe4f61ba6f4e4dce6c9f5815578e41e084fb6b61ee96b7118d9055e5d339aab` |
| Upstream code | `xlang-ai/OSWorld-V2` tag `v2026.08.08`, commit `d578d2d4e0dc82b43e270fdaa7fa89d9708cd154` |
| Tasks dataset | `xlangai/osworld_v2_tasks` rev `3736efa55d9d5dc78f57e873ef78886663e41200`, **108 task modules** |
| Task-hash manifest sha256 | `42f8f6f8939b8712997d5891456a575f8a2a5f53465e9e3e6747af5d6efd0915` |
| Assets dataset | `xlangai/osworld_v2_assets_gated` rev `acad110ef3136405f95434b54862bf9066176c2a`, manifest required |
| Guest image | `ami-01017272139e01feb` (us-east-1, 1920x1080) |

The tasks and assets are **gated** Hugging Face datasets: a human accepts the
terms once and fetches them into an inputs root (default `~/worktrees/osworld`).
Nothing in the apparatus downloads them; the build script verifies what is on
disk against the pin above and refuses, naming the path, on any mismatch.

### Scoring

An episode's score comes from the task's own upstream `evaluate()` and is
mapped by `scoring.score_to_artifact` on a **scored-or-raise** contract:

- every valid score records both metrics: `binary=1` only for exact raw `1.0`,
  otherwise `binary=0`, and `partial_ppm = round(v * 1_000_000)`;
- a near miss can round to `partial_ppm=1000000` while remaining `binary=0`;
  partial reward rounding never promotes binary completion;
- the full upstream return value is retained as bounded canonical JSON in
  `score.details`, including any returned checkpoint, safety or error fields.
  A scalar return cannot recover checkpoint data upstream already discarded;
- the evaluator's OWN scoring-path output is retained in the same artifact, so a
  `0.00%` row can be told apart from one whose evaluator bailed out before
  checking a checkpoint. See "Evaluator diagnostics" below;
- NaN, infinity, out-of-range, non-numeric, missing evaluator, or invalid or
  over-budget detail data → **raise**.

#### Evaluator diagnostics

Upstream evaluators compute the values that decide a score and then discard
almost all of them: `task_002` computes four checkpoint booleans and returns
their mean, `task_016` computes `email_avg`/`linkedin_avg` and returns the
pairing's mean, `task_098` writes its normalised results into the task's cache
directory and returns a scalar. `score.details` used to hold only the return
value, so the archived bundles could not answer "which checkpoint failed, or did
the evaluator return early" — the single question that separates an apparatus
bug from an agent that genuinely failed.

The adapter now captures, around the one call that runs the evaluator:

- the evaluator's `sys.stdout` (its `print` lines) and `sys.stderr`;
- its log records, for the namespaces its modules use (`desktopenv.*`,
  `desktop_env.*`, `llm_metrics`, and the `osworld_task_<id>` names the adapter
  registers task modules under). Their level is raised to `INFO` for the window
  only, because OSWorld never configures logging and the default `WARNING` gate
  drops a task's `logger.info` partials before any handler can see them;
- a bounded manifest of the state the evaluator fetched, from the task's own
  cache directory (`cache_dir_base/<task_id>`, `desktop_env.py:471`), with the
  text of its small files inline and a reported reason for every file whose
  content is withheld.

The retained block rides inside the SAME detail artifact, under
`evaluator_diagnostics`, with the evaluator's return value preserved verbatim
under `evaluator_result`. It is therefore bounded by the same limits, scanned by
the same redaction pass, and referenced by the same digest — no second artifact
and no new protocol field. The capture is additive: an evaluator that emitted
nothing and fetched nothing produces no block, and the staged detail bytes are
byte-identical to what they were before it existed. Nothing an evaluator
returns changes, no task file is touched, and the worker's own stdout/stderr
still carry what they carried (the capture tees rather than diverts).

Retention is bounded, and every cut is reported rather than absorbed: each
captured stream keeps its last 32k characters and counts what it dropped, the
per-file text budget is 8k with a 64k aggregate, and at most 128 cache entries
and 256 directories are walked. A diagnostics block that could not be attached
within the score-detail limits is replaced by a one-field `refused` marker — a
report, not a silent drop — and the score is never affected either way.

The worker stages the raw detail bytes. The runner verifies them, then the
writer publishes them through its existing confinement, redaction, media and
fsync checks before committing the exactly-once scoring receipt. Ordinary
artifact publication remains closed during finalization; only the validated
score receipt authorizes this narrow exception. Publication failure triggers
resource rescue and leaves the run unsealed, without evaluating a second time.
No score-detail bytes enter model context.

This evidence-publication correction requires the matching harness **and**
adapter builds. Development wheels from the same reviewed source commit suffice;
there is no need to publish or release the harness to test them. The helper
packaging correction by itself remains adapter-only.

The raise matters. Upstream's `evaluate()` swallows metric exceptions into
`0.0` and logs a missing evaluator rather than failing. Mapping "could not
evaluate" to zero reports a failure the agent did not commit, which is score
deflation; the harness expresses "we could not score this" as an `unscored`
outcome decided by the runner, never as an adapter-returned zero.

### Step budget

The step budget here is a harness parameter, not the benchmark's. Upstream's
headline metric is measured at 500 steps. `EpisodeConfig.max_steps` defaults
to **50** and `scripts/run_episode.py --max-steps` defaults to **25**; the one
paid episode ran at 25. **A run at 25 steps is not comparable to a published
500-step number**, and any suite run intended for comparison must state its
step budget alongside its score.

## 2. The sandbox

One episode owns one EC2 instance for its lifetime.

| Property | Value | Source |
| --- | --- | --- |
| Region | `us-east-1` (infra `AWS_REGION`) | the release AMI exists only there |
| AMI | `ami-01017272139e01feb` | release pin, `provider_images.aws` |
| Instance type | `t3.xlarge` unless the task overrides it | `provisioning._DEFAULT_INSTANCE_TYPE` |
| Root volume | gp3, 4000 IOPS, 1000 MB/s; **40 GiB** — a floor, raised only if the AMI's own block-device mapping is larger. The release AMI declares 30 GiB, so the request is 40 | `providers/aws.py` |
| Screen | 1920x1080, headless, `action_space="pyautogui"` | `providers/aws.py` |
| Subnet / SG | operator-supplied via infra `AWS_SUBNET_ID`, `AWS_SECURITY_GROUP_ID` | not created by the adapter |

### Ports, and why they are restricted to one address

The security group is a **pre-existing operator-owned group** — the adapter
neither creates nor repairs one — and it must allow inbound TCP from the
**controller host's current public `/32`** only:

- **5000** — the OSWorld guest control service. Every action, every
  screenshot, and every setup upload crosses this port, and the readiness
  probe polls `http://<public-ip>:5000/terminal` until it answers 200 (600 s
  timeout, 5 s interval).
- **9222** — Chrome DevTools. Upstream's setup controller and its Chrome
  evaluators talk to the guest's browser on this port **from the controller**
  (`chromium_port`, default 9222).
- **5910** — noVNC web access, optional, for a human watching a guest.
  Upstream logs a `http://<ip>:5910/vnc.html` URL for AWS instances.
- **3000 and 8000** — V2 task services, explicitly required in the pinned
  upstream README in addition to the standard backend/control ports. The
  ten-task pilot's dated report records when the staged group was corrected;
  do not imply those ports were open for every earlier attempt.

These are unauthenticated services on a public IP. A `0.0.0.0/0` rule on 5000
hands anyone on the internet full control of a desktop that is executing a
scored run, and on 9222 a remote debugger on its browser; the `/32` is the
whole access control. The operator's own address is deliberately not recorded
in this repository — supply it at group-creation time and re-check it before
each run.

**Caveat, verified against upstream:** `DesktopEnv` itself defaults
`vnc_port = 8006` and only AWS provider *log lines* use 5910. 5910 is the port
the operator's staged group opens and the one upstream advertises for AWS web
VNC; 8006 is what the env object carries. Neither is on the episode's critical
path (nothing the harness does reads VNC), so this is recorded as an
unresolved inconsistency rather than a claim either way.

### The TTL lease

An unattended cloud instance that outlives its controller is the failure mode
that costs real money, so the lease is structural rather than best-effort. The
provider creates an EventBridge Scheduler schedule named `lop-ttl-<episode-id>`
targeting `ec2:terminateInstances`, **immediately after `run_instances` and
before waiting for readiness**. A failure to create the lease terminates the
instance and fails the episode — it is never downgraded to a warning.

The schedule's role (infra `AWS_SCHEDULER_ROLE_ARN`) is an operator-created
IAM role trusted by `scheduler.amazonaws.com` whose only permission is
`ec2:TerminateInstances` on instances tagged `lop:adapter = osworld-v2`.
Because the role is tag-scoped, the tag must exist at instance creation; it is
applied inside `run_instances` via `TagSpecifications`, atomically. A
follow-up `create_tags` would leave a window in which the instance exists
untagged and therefore outside both the lease's authority and the leak audit.

Lease length: `OSWORLD_TTL_SECONDS` if the operator sets it, otherwise
derived by `run_episode.py` as **wall budget + 900 s**
(`_ensure_lease_outlasts_wall`), floored at `TTL_SLACK_SECONDS = 900` by the
provider.

The derivation happens in the runner rather than the provider because **the
wall budget is not on the adapter wire**: `ttl_seconds_for` receives `None`
and would otherwise fall back to `DEFAULT_TTL_SECONDS = 7200`. That fallback
was harmless while the wall default was 1800 s and became a defect when it
rose to 18000 s — a lease shorter than the wall means an episode past two
hours dies on a terminated instance rather than at a budget boundary, losing
the episode instead of ending it. An explicit `OSWORLD_TTL_SECONDS` still
wins and is never shortened; it is what bounds a leaked instance's worst-case
cost.

### Burstable credit exhaustion, and `AWS_INSTANCE_TYPE`

The default `t3.xlarge` is a **burstable** instance, and that silently
destroyed five paid episodes with
`ObservationError: environment returned no screenshot frame`. An instrumented
run identified the cause: at the failure moment CloudWatch reported
`CPUCreditBalance 4.2` with `CPUSurplusCreditBalance 0.0` and CPU dipping to
**10.3%** — the guest was throttled to its baseline, not idle. A starved guest
cannot answer its screenshot HTTP server, so the episode dies at step 9–32
having already spent $0.12–$0.32. AWS status checks stay `ok` throughout,
because from the hypervisor's side nothing is wrong; the failure is only
visible as credit metrics plus a suspiciously low CPU floor.

The escape hatch is the optional infra value `AWS_INSTANCE_TYPE`, which
replaces the instance type for the benchmark VM:

```sh
--infra AWS_INSTANCE_TYPE=m5.xlarge
```

It is infra rather than a task field because the task files are **content-hash
verified** against the release pin — editing one to change `instance_type`
invalidates the digest that makes a score reproducible. For the same reason
the override **beats a task's own pinned `instance_type`**: the task author
chose a size against hardware they could reach, while the operator is working
around an infrastructure failure they never saw and cannot fix from inside a
hash-pinned file. A malformed value is refused at `prepare`, before anything
is allocated, rather than surfacing as an opaque botocore error midway through
a paid run.

Omitting it reproduces the previous behaviour exactly, which is what keeps a
default run comparable. When it **is** set, `scripts/run_episode.py` stamps
`aws_instance_type_override` into the evidence manifest's metadata, so a score
produced on non-default hardware is disclosable from the sealed bundle alone
rather than from operator memory. A run on non-default hardware is not
directly comparable to one on the release default and should be reported as
such.

That key records the value **requested** on the command line. It is only
honest because the runner refuses the one case that would make it a lie — see
below.

#### Expanded-task dependency packaging

The expanded pilot exposed a packaging failure, not a model failure: task 010
imports `evaluation_examples.task_class.generated_task_utils`, which the upstream
`osworld` wheel excludes. The exception occurred after VM allocation with zero
model spend. Task 001 did not exercise that import subset. Sixteen release tasks
need the same helper; the adapter now packages the complete three-file runtime
helper closure under its upstream namespace, with unchanged upstream bytes,
license and SHA-256 provenance covered by the adapter wheel RECORD. No gated
task/answer files are added to the wheel or model context.

The pre-allocation static check runs only for the selected AWS task and helper
closure, without importing tasks or executing setup. The offline acceptance
census parsed and loaded all 108 task modules in an isolated interpreter, with
network/process execution prohibited and no setup/evaluate calls. This proves
import packaging, not environment setup, evaluator behavior or task scores.
Optional `lpips`/`torch` imports in task 057 retain upstream's guarded fallback.

Rebuild the adapter wheel and a **new** workspace/selector via the
[adapter setup recipe](../../../benchmarks/osworld_v2_adapter/README.md#runtime-helper-packaging-and-pre-allocation-checks).
The changed `package_digest`, not the reused `0.1.1` version string, identifies
this artifact. Existing pilot evidence and installed environments remain intact;
no harness publication is needed to test this adapter packaging correction.

#### The adapter source and the pinned wheel both call themselves `0.1.1`

**Read this before running with `AWS_INSTANCE_TYPE`.** The adapter source in
this repository gained the override *without* a distribution version bump. The
bump was withheld on purpose: the adapter version feeds `_release_digest`, so
bumping it would have falsified the committed attestation of what the paid
pilot actually ran on. The consequence is that **two materially different
adapter code bodies now both report version `0.1.1`**, and the wheel installed
in the pilot interpreter (`~/worktrees/osworld/venvs/0.1.1/`) is the one
*without* override support. The committed selectors still pin that artifact by
`package_digest`.

The digest pin catches a *mismatched* wheel. It cannot catch a *correctly
pinned old* one — so version alone cannot tell an operator which build they
are about to run.

Because a stale build would silently ignore the override while the manifest
recorded it as applied — a false statement sealed inside a `verify_bundle`-valid
bundle, which is worse than no disclosure at all — the runner fails closed.
`EpisodeRunner._refuse_undeclared_disclosed_infra` compares the value against
the adapter's own `inspect_requirements` response, which distinguishes the two
builds exactly, and refuses **before `prepare`**, so nothing is allocated and no
bundle exists to mislead a reader:

```
UndeclaredDisclosedInfra: adapter 'osworld-v2' version '0.1.1' does not
declare ['AWS_INSTANCE_TYPE'], so the value would be silently ignored while
the evidence bundle recorded it as applied; rebuild the adapter workspace and
selector, or drop the value
```

Seeing that error means the workspace and selector need rebuilding against the
current adapter source (§ the build recipe in
`benchmarks/osworld_v2_adapter/README.md`), not that the flag is wrong.

### Disk exhaustion by the guest's own snapd, and `AWS_ROOT_VOLUME_SIZE`

`AWS_INSTANCE_TYPE` fixed a *starved* guest. A second failure presents
**identically** — `ObservationPhaseError: environment returned no screenshot
frame` — and has nothing to do with CPU. Episodes died at roughly the same
**wall-clock time** regardless of how much work the agent had done: 7 of 8 runs
first failed in a **424–466s window**, at 16–32 steps, on both `t3.xlarge`
*and* `m5.xlarge`. That the instance-type fix changed nothing was the clue.

Instrumenting the guest's own control server showed the root filesystem
filling on a clock rather than on workload:

| time | root filesystem |
|------|-----------------|
| t+54s … t+342s | 93% used, 2.2 GB free (stable) |
| t+363s | 95% |
| t+383s | **100% used, 0 bytes free** |
| t+424s | first `ObservationPhaseError` |

> **Correction.** This was first diagnosed as OSWorld's `x11grab` **ffmpeg
> screen recorder**, and that was **wrong**. `pgrep -af ffmpeg` on a failing
> guest showed **no ffmpeg process at all** — the only match was the probe's own
> `pgrep` command line, which is what made the theory look confirmed from
> outside. The `~6.8 MB/s` fill rate quoted in the 0.46.11 release notes was
> inferred from the disk series, not measured at a process. Do not re-derive it.

> **Two corrections, both measured 2026-09-21.** (1) `/var/lib/snapd/cache`
> is **4096 bytes** on this image and clearing it reclaims nothing; the ~10 GB
> is `*.partial` downloads in `/var/lib/snapd/snaps`. The bullets below and the
> fix section further down carry the correction; the earlier revision of both
> named the cache directory as the consumer, and a reclamation built on it
> reclaimed nothing while reporting success. (2) `AWS_ROOT_VOLUME_SIZE` is
> **INERT** — the guest's root filesystem measured **30,993,747,968 bytes** in
> both a 40 GiB and a 120 GiB volume, because the adapter builds
> `DesktopEnv(...)` without `volume_size=`, so upstream's `expand_guest_volume`
> never runs.

The measured consumer is **snapd**, inside the guest:

- `/var` is 15G of the 29G disk, and the space is what snapd's **downloader**
  writes: `/var/lib/snapd/snaps` went 7.9 GB → 9.9 GB → 10.6 GB in ~50 s while
  free space went 2.2 GB → 0.2 GB → **0**, and the files growing there were
  `<name>_<rev>.snap.xdelta3-<old>-to-<new>.partial` — a pending refresh
  revision being downloaded beside the installed ones (`kf6-core24_64`
  209 MB → 1.17 GB, `audacity_1239` 128 MB → 399 MB).
- `/var/lib/snapd/cache`, named here as the consumer until 2026-09-21, measured
  **4096 bytes**. It *is* documented as snapd's working download cache, which is
  exactly why it looks like the right target and is not: this snapd streams its
  refreshes, deltas included, straight into the `snaps` directory as `*.partial`
  files and leaves the cache directory empty.
- `snap changes` shows `Auto-refresh 9 snaps` and `Pre-download novnc`, both
  fired at boot
- the AMI ships ~93% full, so a few GB of snap downloads exhausts it

A disk at 0 bytes cannot write a screenshot, which is exactly the observed
failure. Snapd's auto-refresh starts at boot and downloads at its own pace,
entirely independent of what the agent is doing — which is precisely why the
wall looked like a clock rather than a workload, and why the volume being
identical on either instance type meant changing the hardware family did not
move it.

The escape hatch is the optional infra value `AWS_ROOT_VOLUME_SIZE`, a whole
number of GiB:

```sh
--infra AWS_ROOT_VOLUME_SIZE=120
```

Measured effect — and **measured 2026-09-21 to be INERT**: the guest's root
filesystem is **30,993,747,968 bytes (28.9 GiB)** in a 40 GiB volume *and* in a
120 GiB volume, episode after episode. `guest-preparation.json` carries both
numbers, so this is checkable from any bundle: `filesystem_bytes` is the same
number every time, while `disk_bytes` is 42,949,672,960 (40 GiB) and
128,849,018,880 (120 GiB) respectively. The partition is never grown because
the adapter constructs `DesktopEnv(...)` **without `volume_size=`**, so
upstream's `expand_guest_volume` (`desktop_env/providers/volume.py`:
`growpart`, `partprobe`, `resize2fs`) is never called — a bigger volume, the
same filesystem inside it, and nothing to download into once that filesystem is
full.

An earlier revision of this section credited a 100 GiB volume with moving the
first failure from t+424s to **t+1936s**. Whatever that run measured, it was not
a partition growing: the reason geometry was suspected — "the root partition
stays 29.5G inside that 100 GiB disk, with ~70 GiB unallocated, because the AMI
carries no `growpart` and `apt-get install cloud-guest-utils` cannot run on a
disk with no free space" — is now the measured END state of every run, at every
volume size. Do not read a longer run as the volume buying room.

The value is still accepted (validated at `prepare`, stamped into the evidence
manifest when set) and setting it is harmless. What it is not is a lever for
this failure. The one-line change that would make it a lever — passing
`volume_size=` to the `DesktopEnv` construction — is NOT made here: it changes
the guest's hardware contract inside the benchmark, and the reclamation below
addresses the cause instead. See "Growing the partition is deliberately not
done".

It is infra rather than a task field for the same reason as the instance type:
task files are **content-hash verified**, so editing `volume_size` invalidates
the digest that makes a score reproducible. For the same reason it **beats a
task's own pinned `volume_size`** — the task author sized a volume against the
workload they could see, not against a recorder filling the disk on a clock.

Validation runs in two places, deliberately:

- **At `prepare`, before anything is allocated** — non-integers (`40.5`,
  `1e3`, `+40`, `4_0`, whitespace), zero, negatives, and anything outside
  1–16384 GiB (the gp3 maximum, since the provider pins gp3).
- **At launch, before `run_instances`** — a size smaller than the AMI's own
  snapshot. EBS cannot restore a snapshot into a smaller volume, and AWS
  refuses it with an `InvalidBlockDeviceMapping` naming neither size; the
  adapter's message names both plus the knob to change. This check cannot move
  to `prepare`, which by contract issues no I/O at all — not even a read-only
  `describe_images` — because that is what lets it run before allocation.

Omitting the value reproduces the previous behaviour exactly: the task's pin,
else the AMI's own root size resolved at launch (OSWorld's 40 GiB floor). When
it **is** set, `scripts/run_episode.py` stamps `aws_root_volume_size_override`
into the evidence manifest's metadata, so a run that survived past the
exhaustion wall is not silently compared against truncated ones.

Like `AWS_INSTANCE_TYPE`, this value is **gated**: supplying it to an adapter
build that does not declare it fails the episode before `prepare` with
`UndeclaredDisclosedInfra`, because a silently dropped override plus a stamped
disclosure is a false statement sealed in a `verify_bundle`-valid bundle. Both
the gate and the stamp derive from a single table
(`DISCLOSED_INFRA_METADATA_KEYS` in `local_operator/evaluation/runner/episode.py`),
so a future value cannot be gated without being disclosed or vice versa.

### Explicit system proxy policy

`--infra OSWORLD_ENABLE_PROXY=false` in `benchmark_compute` scope selects
upstream's supported system-disabled mode. Only the exact lowercase strings
`true` and `false` are accepted; malformed values or the wrong scope fail at
`prepare`, before provider construction or allocation, even before a task is
loaded. This is apparatus policy, not a task edit or a task-ID exception.

For episodes that also need simulator configuration, the generic CLI accepts
an optional per-value purpose prefix, without changing the global default:

```sh
--infra-purpose benchmark_compute \
  --infra AWS_REGION=us-east-1 \
  --infra benchmark_compute:OSWORLD_ENABLE_PROXY=false \
  --infra benchmark_user_simulator:OSWORLD_USER_SIM_MODEL=<simulator-model>
```

Legacy `NAME=VALUE` entries still use `--infra-purpose` (default
`benchmark_compute`). A prefix applies only to that entry. Unknown purposes,
empty names/values, and conflicting entries for the same name (including across
scopes) fail cleanly before selector loading/allocation without printing their
values; identical duplicates coalesce. Purpose-prefixed policy and hardware
inputs receive the same manifest disclosures as unprefixed inputs. Values stay
non-secret; simulator API keys still travel through the existing secret path.

Omitting the value preserves the adapter's existing behavior:
`enable_proxy=bool(task.proxy)`. Explicit `true` sets the upstream system switch
on; upstream still combines it with the task's proxy hint. Explicit `false`
sets the switch off regardless of that hint, and post-policy requirements no
longer demand `OSWORLD_PROXY_CREDENTIALS` or `OSWORLD_PROXY_ENDPOINT`.

**Enabled setup now works, through `PROXY_CONFIG_FILE`.** This was the
"separately reviewed follow-up" this section previously called for.

Upstream builds its pool at *module import*: `desktop_env/controllers/setup.py`
calls `init_proxy_pool(PROXY_CONFIG_FILE)` at the top level, reading that name
via `os.getenv` with a default of the **CWD-relative**
`evaluation_examples/settings/proxy/dataimpulse.json`. The adapter worker is
spawned `-I` from an arbitrary CWD, so that default never resolved, and
`load_proxies_from_file` swallows the error into a log line — leaving an empty
pool that only failed later, at `reset_start`, with the VM already billed. Two
of the ten frozen pilot tasks died that way on every model tried.

Pass an absolute path to an upstream proxy-pool JSON file:

```sh
--infra PROXY_CONFIG_FILE=/abs/path/to/proxies.json
```

The file is a JSON list of objects with at least `host` and `port`
(`username`, `password`, `protocol` optional), exactly as
`ProxyPool.load_proxies_from_file` parses it. It is validated **before
allocation** — absolute, a regular file, readable, size-bounded, and shaped the
way upstream actually accepts — so a bad path is a free refusal naming both
ways out rather than a paid crash. Because upstream re-reads the path at import,
validation cannot guarantee the bytes it will load; a file edited in between can
still yield a degraded pool.

`PROXY_CONFIG_FILE` is **required** for a task whose descriptor sets
`proxy = True` unless `OSWORLD_ENABLE_PROXY=false` selects system-disabled mode.
The requirement is enforced in `reset_start` rather than `prepare` because
`PrepareParams` carries no `task_id`: at `prepare` the task's proxy hint is
structurally unknowable, so a check there would pass every task it was meant to
protect.

`OSWORLD_PROXY_CREDENTIALS` is **no longer declared** — nothing in the package
ever consumed it, and demanding a secret the apparatus cannot use trains an
operator to fabricate one. `OSWORLD_PROXY_ENDPOINT` remains declared but
optional, so existing invocations are accepted unchanged.

The requested switch is sealed as `osworld_enable_proxy_override` via the same
manifest disclosure table as the compute overrides. An older adapter that does
not declare the optional `OSWORLD_ENABLE_PROXY` requirement is refused before
`prepare`, rather than silently applying its old behavior under a false disclosure.
Omission adds no override metadata and remains compatible with older selectors;
no adapter wire-schema change is required.

Disabling proxy does **not** establish direct-network adequacy or benchmark
comparability. Record the changed apparatus before outcomes, verify necessary
guest network access separately, and classify blocked access as an infrastructure
failure rather than inventing credentials or silently changing policy. Local
validation here uses captured provider constructor arguments and fake-provider
CLI episodes, not cloud/network validation.

### Guest disk reclamation at episode start

Growing the volume treats the symptom; the cause is that snapd downloads
gigabytes into a disk that ships ~93% full. So `AwsProvider.allocate` runs one
**guest preparation** step between guest readiness and upstream's `reset` —
before the episode's first observation, because hygiene that ran after the
reset would be too late for the frame that reset captures.

It drives the guest's own HTTP control server (`POST /execute`, argv with
`shell: false` — the same endpoint and contract upstream's `SetupController`
uses), and does three things, in an order that is load-bearing:

1. **Abort the in-flight auto-refresh** — `snap changes` showed
   `Auto-refresh 9 snaps` and `Pre-download novnc` already running at boot.
   Only `Doing` changes with those two summaries are aborted; a seeding hook or
   any other change is left alone. Abort goes first because it is immediate,
   whereas a hold is a `configure core` hook change the CLI waits on, and snapd
   runs one hook per snap at a time — behind a live refresh of core/snapd the
   hold could queue past the per-command ceiling. Snapd's 20-minute retry delay
   means no new refresh can start in the gap before the hold lands. Aborting a
   partly-done change undoes its completed tasks, which returns those snaps to
   the revision the AMI shipped — the benchmark's own baseline.
2. **Hold snap auto-refresh** (`snap refresh --hold=forever`, falling back to
   `snap set system refresh.hold=<far future>` on snapd older than 2.58). This
   stops a *new* refresh starting.
3. **Clear the download scratch** — snapd's `*.partial` incomplete downloads in
   `/var/lib/snapd/snaps`, and the contents of `/var/lib/snapd/cache`.

   The second directory is the one this step used to target alone, and it is the
   correction that matters most here: `/var/lib/snapd/cache` is documented by
   Canonical as the working cache "used to minimise download size and speed-up
   refreshes", and it measured **4096 bytes** on this image. The bytes are in
   `/var/lib/snapd/snaps`, as `*.partial` files beside the installed revisions
   (measured: that directory going 7.9 GB → 9.9 GB → 10.6 GB while free space
   went to **0**). Both are pure download scratch — deleting a partial costs a
   re-download and nothing else — and the installed `.snap` revisions are left
   alone: one of those is an application, mounted through a loop device, and
   deleting it is an uninstall rather than housekeeping.

Every privileged step runs one fragment inside one privileged `bash -c`, reached
through the same **candidate ladder** upstream's own `expand_guest_volume` uses
(`desktop_env/providers/volume.py`): `sudo -n` first, then the value of
`OSWORLD_CLIENT_PASSWORD`, then — only when that value is itself one of
upstream's two documented development defaults — the other one. Each rung's
outcome is recorded in the step's detail as a chain
(`escalation=agentless>supplied`, or `escalation=agentless>supplied>upstream-default`),
so a reader can tell "the operator's value was refused and the image's own
documented default opened the guest" from "the operator's value worked". On a
step that succeeded the chain ends on the rung that authenticated; on one that
failed it ends on the last rung tried, and the collected output says why — a
`sudo: …` line means the candidates were refused, anything else is the
fragment's own failure. No password value is recorded anywhere.

**`OSWORLD_CLIENT_PASSWORD` must be a value the image accepts, and getting this
wrong is not a warning.** Measured 2026-09-21: with a rejected value, *every*
privileged step above fails (`sudo: no password was provided` / `sudo: 1
incorrect password attempt`), snapd keeps downloading, the guest's root
filesystem reaches 0 bytes free at ~t+383 s, the control server dies with
`OSError: [Errno 28] No space left on device` writing a screenshot, and the
episode ends minutes later on an opaque transport error. Four paid episodes went
that way. The adapter therefore **refuses the episode at preparation time** when
a reclamation step did not land — before upstream's environment is constructed
and before any model spend — naming the failing steps and this knob. The
partial `guest-preparation.json` is still written, so the evidence of why the
episode ended survives the refusal.

With the correct value the identical episode completed, was scored, and served
200 on all 50 screenshots.

Every privileged step's shell shape is not cosmetic: the control server runs as
an unprivileged user, so a glob like `/var/lib/snapd/cache/*` expanded by the
*outer* shell matches nothing against a `drwx------ root:root` directory and
`rm -rf` of the literal name exits 0, and an `xargs … echo pw | sudo -S snap
abort` pipeline parses as `xargs echo` piped into one id-less `sudo`. Both were
real defects that reported `ok` while doing nothing, caught only by re-running
the E2E with the cache directory genuinely owned by root. A glob is therefore
used only INSIDE the privileged shell (the `*.partial` suffix), where it is the
privileged shell that expands it.

The design constraints are worth stating explicitly, because each is a line a
future change could cross without noticing:

- **It is environment preparation, not benchmark semantics.** Nothing here
  changes the task, the scoring, the applications available, or anything the
  model observes. Clearing a package manager's download cache is housekeeping;
  **uninstalling** an application a task might need is not, so no command may
  ever `snap remove`, `apt-get purge`, or delete an installed `.snap` revision.
  A test asserts that.
- **It is fail-soft per step and fail-LOUD about an unprepared guest.** A
  missing binary, a denied `sudo`, an unreachable control server, or a guest
  that answers slowly are each recorded as a step outcome rather than thrown
  out of the pass, and a whole-pass budget keeps a wedged guest from eating the
  reset timeout. But a reclamation step that did NOT land is the measured
  signature of a guest that will die mid-episode, so the caller refuses the
  episode on it (see above). The distinction is deliberate: recording without
  acting is what the four dead episodes were.
- **It is conditional, and "still short of space afterwards" is not a failure.**
  Free space is measured on every episode, but the reclamation only runs below
  **12 GiB free** — set above snapd's largest measured appetite (the ~9.9 GB of
  delta downloads), so a guest that can already absorb a full auto-refresh is
  left untouched. A guest whose free space cannot be measured *is* reclaimed:
  the protection must not go missing exactly when the guest is least healthy.
  After a SUCCESSFUL reclamation the guest still sits at ~2.2 GB free (measured
  on the run that completed and scored) — far below the threshold — so the
  guard cannot be the threshold. What is checked is whether the hold and the
  clear landed; a guest that is healthy but short of space warns, it does not
  abort, because aborting on it would refuse every healthy episode.
- **It is observable.** Free space before and after, the filesystem and
  whole-disk sizes, every step's outcome, and the `blocking_steps` the refusal
  was raised on are written to
  `<run-root>/osworld-cache/<episode-id>/guest-preparation.json`. "The guest had
  N bytes free at the start" is the fact a later
  `environment returned no screenshot frame` has to be read against, and it must
  not depend on anyone having probed the guest by hand. It sits in the episode's
  own cache root rather than the artifact root because the bundle verifier
  refuses any artifact-root entry that is not a digest-named artifact; it is not
  on the observation because `Observation.metadata` feeds
  `observation_content_id`, and a content-addressed observation id must be a
  function of what the model saw, not of the guest's filesystem.

**Growing the partition is deliberately not done.** `growpart` is absent, and
the in-place `sfdisk` alternative rewrites the root partition table where a
wrong start sector destroys the guest. A hygiene step that can fail *hard* is
exactly what this must not be. The disk-vs-filesystem geometry is **reported**
instead, read-only — and that pair of numbers is what made the
`AWS_ROOT_VOLUME_SIZE` claim above checkable from any bundle: 30,993,747,968 in
a 40 GiB volume says the override never reached the filesystem.

### Upstream is sealed after the first reset

One `reset` per episode is the contract. Upstream's own allocation paths — a
second `reset` on a used env (`_revert_to_snapshot` →
`AWSProvider.revert_to_snapshot`), `manager.get_vm_path`, `_save_state`,
`close`/`stop_emulator` — would launch or release an instance with no client
token, no `lop:adapter` tag, and no TTL lease: invisible to the audit and
unreachable by rescue. The provider replaces each of those methods on the live
env with a raiser before any boto3 call, and a test statically scans the
pinned upstream for any method reaching `run_instances`/`terminate_instances`/
`create_image` and asserts it is sealed. `DesktopEnv.close()` is likewise never
called: it terminates without confirming.

## 3. Adapter architecture

The adapter is a **separately distributed wheel**
(`lop-osworld-v2-adapter`, currently 0.1.1) whose source lives in-tree at
`benchmarks/osworld_v2_adapter/`. It is separate because
`discovery.distribution_digest` pins the adapter by verifying every RECORD row
of the installed wheel: if adapter code shipped inside the harness
distribution, every harness release would invalidate the pin. Isolation comes
from the wheel plus the digest plus the isolated worker, not from where the
source sits.

**The host never imports adapter code.** The adapter runs in a subprocess
under a *different*, locked interpreter, and the parent's only contact with it
is a JSONL RPC channel. That is what makes the pin meaningful: importing
adapter code into the host would make the harness's own dependency resolution
part of the benchmark's environment, and a benchmark whose environment moves
with its harness cannot be reproduced.

The pieces, and what each guarantees:

- **A dedicated interpreter.** `python_executable` must be a real file, not a
  symlink (`discovery._symlink_free`), so the venv is built with the stdlib
  `venv --copies`. It holds the adapter wheel, its locked dependency set, a
  pinned `local-operator`, and — for paid runs — the `osworld` extra
  (upstream `desktop_env` and its ~380-package dependency tree; the committed
  lock resolves 424 packages in total).

  Three construction details are load-bearing and cost about twenty minutes
  to rediscover, because each fails *after* the previous gate passes:

  - **`uv venv` cannot build this venv, at any flag combination.** It always
    writes `bin/python3.12` as a symlink to `bin/python` (and that to the
    managed toolchain), so discovery rejects the launch path with
    `AdapterDiscoveryError: adapter launch path has a symlink or lexical
    alias` before any spend. `--link-mode copy` does not change it (that
    governs the *package* store, not the interpreter shims), nor do
    `--managed-python`, `--relocatable`, or the now-undocumented
    `--python-preference only-managed` — which is still parsed and honoured
    despite being absent from `uv venv --help` since `--managed-python`
    superseded it. Use `<base-python> -m venv --copies --without-pip <venv>`.
  - **`venv --copies` copies the interpreter but not its runtime library.**
    On the uv-managed macOS toolchain the copied binary resolves
    `libpython3.12.dylib` through `@rpath` relative to the venv, so it dies at
    startup with `dyld: Library not loaded: @rpath/libpython3.12.dylib`.
    Copy `<toolchain>/lib/libpython3.12.dylib` into `<venv>/lib/` after
    creating the venv. A venv whose interpreter cannot start is
    indistinguishable, from the batch log, from a harness bug.

  - **The adapter must be installed as a wheel, never `-e`.** An editable
    install *does* write a `RECORD` — and `distribution_digest` hashes it
    happily — but that RECORD covers only the `.pth` shim and the metadata
    directory: it has **no rows under the package source**. So
    `_resolve_module_artifact` finds no artifact for the entry module and
    `discovery.py` raises `adapter entry module is not uniquely
    RECORD-covered`. The error names RECORD coverage, not the editable install,
    so read it as "the entry module's source is not listed", which is also what
    it means in the rarer case of a genuinely malformed wheel. Build with
    `uv build --wheel` and install the artifact.

  `local-operator` itself may be editable without breaking discovery — only
  the adapter distribution is resolved this way — but installing it as a
  wheel too keeps one interpreter's provenance uniform, and a wheel is what
  the committed lock describes. The harness version in evidence is read from
  the **running install**, never from this checkout: `_harness_version`
  (`scripts/run_episode.py`) resolves it through
  `local_operator.update.installed_build`, and `harness_git_revision` comes from
  the same install's own ``.lop-source`` record, falling back to a digest of
  that version when no commit was recorded. Install metadata is the *right*
  answer and a working tree is the wrong one: a campaign that ran a 0.61.11
  interpreter over a copy of this tree sealed every one of its bundles as
  **0.61.9**, the version in the shared checkout's working tree — which is not
  the build that ran — so a checkout sitting beside the script may not answer
  for the build under any install mode.
- **Exact-distribution discovery.** Before launch, `worker_argv` re-resolves
  both spawn boundaries symlink-free, verifies the release manifest, and
  re-hashes the workspace. At load, `distribution_digest` hashes every RECORD
  row *after verifying each installed file against it*, and the entry point's
  module is loaded by a source loader that executes **the exact bytes that
  were hashed** — never a cached artifact.
- **Isolation flags.** The worker is spawned as
  `<python> -I -s -E -B -m local_operator.evaluation.adapters.worker`. `-I`
  and `-s`/`-E` keep user site directories, `PYTHON*` variables and the
  current directory from changing which wheel is verified; `-B` keeps the
  worker from writing bytecode (see §4).
- **Canonical JSONL over dedicated pipes.** RPC rides two inherited one-way
  protocol descriptors — not stdout/stderr, which upstream libraries write to
  freely — with a 1 MiB frame cap and a closed error-code vocabulary
  (`adapter_error`, `cancelled`, `invalid_request`, `invalid_state`,
  `protocol_error`, `timeout`). Protocol schema version is `1.2`.
- **A supervised process group.** The supervisor owns the worker's process
  group and reaps it; upstream spawns subprocesses of its own, and a leaked
  process tree holding a cloud handle is the same class of problem as a
  leaked instance.
- **Secrets travel only on the pipe.** The worker's environment is built from
  a closed allowlist, so nothing ambient carries a credential. Resolved
  secrets ride `reset_start` (the side-effect boundary) and `begin_rescue`
  (a fresh worker tearing down from a descriptor) and nothing else — never
  `prepare`, never `rescue.json`. AWS values never touch `os.environ`; the
  provider builds its boto3 session from the values directly, and installs
  that session as the process default only because upstream builds its own
  clients ambiently with no credentials argument.
- **Host-side resolution before the bundle opens.** The runner resolves
  `EpisodeSpec.secret_refs` after the handshake and **before the evidence
  writer opens**, so every resolved value is a redaction canary from the
  bundle's first byte. A missing ref fails the episode `failed_pre_bundle`
  naming only the ref — before `prepare`, before any resource exists.

### The three digests

A selector names the adapter with three hashes, all recomputed at launch:

- `package_digest` — every RECORD row of the installed wheel, each verified
  against the file on disk.
- `workspace_digest` — every immutable workspace file (§4).
- `release_digest` — the build attestation:
  `sha256("lop-osworld-v2-adapter" || version || package_digest ||
  benchmark_release_name || task_hash_manifest_sha256)`. It ties a specific
  harness build to a specific benchmark release, which is the claim a
  leaderboard number has to carry.

## 4. The content-pinned workspace

The workspace is a read-only directory the build script materialises from the
verified inputs root: `adapter-release.json`, `benchmark_release.json`,
`task_hashes.json`, `adapter-provider.json`, `inputs.json`, and `tasks/`
(108 modules). It is hashed in full, and the hash is re-checked by the worker
at launch **and** by the rescue sweep before it will spawn anything. A
mismatch is refused with `adapter workspace content digest differs`.

The 4.2 GB of gated assets are **not** in the workspace: `MAX_WORKSPACE_BYTES`
is 4 GiB (and `MAX_WORKSPACE_FILES` 100 000). Instead `inputs.json` records
the asset manifest's sha256 and the prepared checkout commit, and the adapter
re-verifies the live inputs root against those pins at every `reset_start`.

### What invalidates the pin — and the two things that wrongly did

Anything that changes bytes under the workspace invalidates it. That is the
point, and it is also how the apparatus caught two defects on its first paid
episode. Both were fixed at the source rather than by loosening the digest.

**Bytecode caches (#542, harness 0.44.36).** Upstream's `instantiate_task`
imports a task module *from the pinned workspace*, which left
`tasks/__pycache__/task_001.cpython-312.pyc` behind. The rescue sweep then
refused to act on an instance that was in fact already terminated. Two
independent fixes: the worker and supervisor are spawned with `-B` (and
`worker.main` sets `sys.dont_write_bytecode` as the in-process guarantee), and
`workspace_digest` excludes `__pycache__/` and bytecode suffixes **by rule** —
bytecode is never verified content, since the loader refuses to load a `.pyc`
whether RECORD covers it or not, so a cache written by any tool can never
invalidate a pin. Symlinks inside a cache directory are still refused like any
other. The same episode also exposed the frame-id contract defect that ended
it (§8).

**Upstream's own scratch writes (#543, harness 0.44.39).** `DesktopEnv`'s
default `cache_dir="cache"` resolves against the worker's cwd — and the cwd
*was* the pinned workspace, so `_download_setup` wrote
`workspace/cache/001/…` during the episode: real content the digest correctly
refuses to ignore. The adapter now hands `DesktopEnv` an **absolute,
per-episode** cache root at `<artifact_root>/../osworld-cache/<episode_id>`:
beside the artifact root, because the bundle verifier walks the artifact
directory and refuses any non-digest entry; per-episode, because upstream's
`reset_cache_dir` only reassigns the attribute and clears nothing, so two
episodes on the same task must not share a cache; durable, because the run
root is put through `refuse_volatile_root`.

Routing `cache_dir` fixed the write path we observed but not the class:
several upstream helpers open hard-coded relative names with the builtin
`open` at module scope (`temp.pdf` in `evaluators/metrics/vscode.py`,
`temp_extracted_<n>.jpeg` in `slides.py`, an epub's `<name>.dir` in
`others.py`) and consult no env object, so no attribute the adapter installs
can intercept them. `reset_start` therefore also moves the worker's **cwd** to
the episode cache root and `close` restores it, so any relative write by any
upstream path, known or unknown, lands in episode scratch. This is safe by
construction and each leg was checked: task modules are loaded by absolute
location (`spec_from_file_location`), the worker runs with `-I` so the cwd is
not on `sys.path`, and both the digest re-check and the rescue sweep hash the
selector's absolute workspace path.

A genuine drift is still a refusal, by design: restore the workspace from the
verified inputs root and sweep again.

## 5. Teardown, rescue, and the leak audit

`cleanup` reports `succeeded` only on **positive** evidence. `terminate` polls
`describe_instances` until the state is `terminated` (up to 55 s) and
otherwise reports `terminate-unconfirmed`, which keeps `rescue_required` set.
The evidence-code vocabulary is closed and greppable:

| Code | Meaning |
| --- | --- |
| `instance-terminated` | terminal state observed |
| `instance-absent` | the tag query found nothing — nothing to release |
| `terminate-unconfirmed` | we asked and could not confirm; rescue stays required |
| `terminate-denied` | the API refused the termination |
| `schedule-deleted` / `schedule-absent` | the TTL lease is gone / was already gone |
| `schedule-delete-failed` | the lease still exists and will still fire (safe) but was not retired |
| `session-closed` | the upstream env session was closed |
| `kind-unsupported` | this build cannot execute that action kind — teardown was never attempted |

The mapping to statuses is deliberately asymmetric: `not_needed` clears
`rescue_required` and `attempted`/`failed` keep it set, so a code meaning "we
could not look" can never read as clean.

**If the parent dies**, `<rescue-root>/<episode-id>/rescue.json` names the
episode's refs. The refs are minted deterministically from the episode id
(`lop-ep-<id>`, `lop-ttl-<id>`, the id itself) precisely because the natural
identifier — the `i-…` instance id — does not exist until `run_instances`
returns, which is *after* the descriptor has been persisted. Teardown resolves
id-from-tag with `describe_instances(Filters=[tag:lop:episode=<id>])`, which
needs nothing but the episode id. The sweep spawns the exact pinned worker per
descriptor, re-resolves the descriptor's secret refs from the credential store
(in the parent — the values ride the pipe and never reach disk, environment,
or stdout), reconciles every action, and unlinks the descriptor **only** when
the aggregate is complete. It takes no `--region`: each descriptor already
carries its own.

### Rescue was inert before 0.44.41 — state this plainly

**Every rescue before harness 0.44.41 was a no-op, and this is the single most
important honesty item in this document.** `AdapterWorker._dispatch` handled
`begin_rescue` entirely itself — validating pins, storing the descriptor,
returning an `AckResult` — and never called the adapter. `params.secrets`, the
only credential a rescue worker ever receives, were dropped. The provider
therefore stayed `None`, every cleanup action took the honest "could not look"
branch, and a genuinely leaked instance would never have been terminated by a
sweep. The visible symptom was only a sweep that never confirmed teardown;
the real consequence was that the leak backstop did not exist.

**#548 (harness 0.44.41)** forwards `begin_rescue` to the adapter after the
worker's own pin validation and descriptor storage (which is the security
boundary and is unchanged), and an adapter that cannot accept the handoff now
fails loudly rather than returning a clean-looking `Ack` for a resource
nothing can release. It was proven against a **copy** of the real stranded
descriptor from the paid episode: before the fix the sweep reported
`complete: false` with codes `["session-closed", "terminate-unconfirmed",
"schedule-absent"]`; after it, the descriptor reconciles.

Note what this does *not* prove: the fix was verified against a descriptor
whose instance was already gone, not against a live leaked instance. The
live kill-and-rescue drill is still outstanding (§8).

The `provider is None` → `attempted`/`terminate-unconfirmed` fallback is
deliberately kept — it remains correct for genuinely ambiguous cases.

### Leak audit

Every instance and volume carries `lop:adapter=osworld-v2` and every lease is
a schedule named `lop-ttl-<episode>`, so one tag-filtered read-only query per
resource kind is a complete inventory. `scripts/osworld_tag_audit.py --region
us-east-1` prints `[]` and exits 0 when clean, otherwise lists what it found
and exits 1. **It terminates nothing, ever** — teardown happens only through a
descriptor-driven rescue, so every termination has a receipt. Run it before
and after every paid episode; it must print `[]` both times.

## 6. The evidence bundle

Each episode writes one bundle under `<run-root>/evidence/<episode-id>/`:
`manifest.json`, an append-only `events.jsonl` journal, `state.json`, exactly
one terminal file (`outcome.json` **or** `abandonment.json` — both present is
an error), a `.lock`, and a content-addressed `artifacts/` directory.

The manifest is provenance, fixed at creation: episode id, harness version and
git revision, adapter id/version, benchmark id and release, task id, task
digest, input digest, requested route, fallback policy, environment digest and
release, provider image digest, and the dependency/budget/cleanup plan ids. It
is self-identifying — `manifest_digest` is the canonical digest of its own
unsigned contents and `bundle_id` derives from that plus the episode id, so a
manifest cannot disagree with itself. Metadata is portable by construction:
digests and public pins, never credentials, provider request bodies, or raw
prompts.

The journal's event kinds are a closed set: `preflight`,
`lifecycle_transition`, `model_request`, `model_response`, `usage_cost`,
`context_compaction`, `budget_commitment`, `reconciliation`, `observation`,
`action_batch`, `environment_step`, `user_simulator_exchange`,
`finalization_start`, `scoring_start`, `scoring_result`, `cleanup`, `error`,
`cancel`. Screenshots are artifacts referenced by digest, written through one
reader that verifies size and digest with `O_NOFOLLOW` — the same reader that
supplies the model, so a frame the runner would refuse to publish is a frame
the model cannot be shown.

**Sealing and verification.** `EvidenceWriter` is the sole append and
finalization authority for a bundle; a redaction scanner streams every byte
written against the resolved secret values, matching not only plaintext but
base64, hex and percent-encoded projections of them, so a credential cannot
slip through in an encoded artifact. `verify_bundle` then
recomputes the bundle **without trusting writer memory or the recorded
status**: it re-reads under `O_NOFOLLOW`, checks owner and mode, refuses any
unknown root entry, re-parses each model canonically (a non-canonical
encoding is an error in itself), re-walks the journal, recomputes counters,
and re-derives the terminal state. Independent verification is the point — a
bundle nobody can recheck is a claim, not evidence.

### Reportable vs unscored

Two orthogonal labels ride the outcome, and both are computed, not asserted.

**Reportability** picks the single most severe of
`cleanup_incomplete > budget_unreconciled > cancelled > unscored >
synthetic_model > reportable`. The ordering is a statement about honesty: a
leaked resource is a worse claim about a run than an unclosed budget, which
matters more than a missing score. A run is `reportable` only when every one
of those is clear. Two labels deserve naming:

- `synthetic_model` — the decisions came from a scripted client (`--model-client
  scripted-finish`), so the score grades nothing. Such a bundle verifies like
  any other and carries `model_client` in its manifest metadata; it is
  **never** `reportable` and cannot be mistaken for a result.
- `cleanup_incomplete` — what the one paid episode sealed as, because its
  rescue descriptor was still outstanding.

**Comparability** is separate: `comparable`, or `route_changed`,
`environment_unpinned`, `input_mismatch`, `adapter_mismatch`,
`benchmark_mismatch`. A silent provider fallback to another model invalidates
a comparison however well the run scored, so a run whose served route left its
pin cannot seal as `comparable`.

A number is reportable only if its bundle verifies, its reportability label is
`reportable`, and its comparability label is `comparable`. Anything else is
evidence of an attempt, not a result.

## 7. Run parameters and cost

### Parameters

Flags on `scripts/run_episode.py`, with their defaults:

| Parameter | Default | Notes |
| --- | --- | --- |
| `--route` | required | `<provider>/<model>`; the paid episode used `openrouter/deepseek/deepseek-v4-flash-vision-exp` |
| `--max-steps` | 25 | bounds the step loop; `EpisodeConfig.max_steps` itself defaults to 50. Stating it also takes the cost-rate ratio out of the picture for that episode: the step budget is the authority about how long the run lasts, so cycle prices are not judged as a rate (see `--max-cycle-usd`) |
| `--max-usd` | 0.50 | hard provider spend cap; reaching it is a scored truncation (`budget-cap`), and it is the only COST authority a step-budgeted episode has |
| `--max-wall-s` | 18000 | runaway guard only; the 500-step budget binds first. The TTL lease is derived from it (`_ensure_lease_outlasts_wall`), see BUDGETS_AND_LATENCY.md |
| `--max-cycle-usd` | none | ADDS an absolute per-cycle cap, the operator's own number: it truncates (`cost-spike`) whatever the step budget says, and a series crossing it is worth reading. Unlike the prorated per-cycle ceiling this used to be paired with, it does not mistake context growth for waste |
| `--keep-recent-frames` | 3 | frame retention |
| `--benchmark-release` | `osworld-v2-2026.08.08` | |
| `--run-root` | required | must be durable; `/tmp` and `$TMPDIR` are refused |
| `--config-dir` | harness default | credential store location |

Derived reservations: `guest_actions = max_steps * 8`,
`model_cycles = max_steps * 2`.

`keep_recent_frames = 3` is a behavioural constant, not a benchmark tuning
knob — it is what an interactive screen-driving session sets, on the reasoning
that a screen is *state*: the current frame is what the agent acts on, the
last couple are what it compares against, and older ones are views the surface
has since replaced. Pruning is batched into context rebuilds every 8 frames
rather than done per turn.

Retries: a billed reply that fails strict decision parsing is a **model**
error, not a provider outage. The client raises `DecisionRejected` carrying
the call's full billing provenance and appends both the bad reply and a
correction naming the observation and its valid frame ids, so the runner's
re-call is corrective by construction. `max_decision_retries` defaults to 2
(`0` restores one-strike behaviour). Every attempt writes its own
`model_request`/`model_response`/`usage_cost` triple and counts as a model
cycle; a rejected attempt also writes a retryable `error`. Exhausting the
bound seals as `model_failure` — distinct from `provider` (nothing was down)
and `crash` (nothing broke).

Route identity is folded losslessly into the seal (`RouteIdentity` fields
cannot carry `/`): `_`→`__`, `/`→`_s`, anything else outside
`[A-Za-z0-9.:-]` → `_x<hh>` per UTF-8 byte, so
`deepseek/deepseek-v4-flash-vision-exp` seals as
`deepseek_sdeepseek-v4-flash-vision-exp`. The manifest metadata also carries
the raw id as `route_model_id`, so a reader never has to decode by hand.

### Cost accounting

Model cost is **the provider's own figure**, not a local reconstruction:
`usage.cost` from the stream is accumulated as `provider_cost_micros` in
integer micro-USD and written to a `usage_cost` event per call, then sealed
into the reconciliation. Micro-USD integers avoid float drift across
thousands of calls, and taking the provider's number avoids re-deriving a
price from a catalogue that may have moved since the call. What cannot be
measured is marked, not guessed: a budget that cannot be reconciled against
real usage seals `budget_unreconciled` rather than silently reportable.

Infrastructure cost is **not** in the bundle and must be accounted separately
from AWS billing. The staged estimate for one 30-minute t3.xlarge episode in
us-east-1, at the on-demand rate of $0.1664/h plus a prorated gp3 root volume
and one schedule, is **$0.05–0.10 of AWS per episode**; the model at the
route's catalogue prices ($0.22/M in, $0.66/M out) was estimated at
$0.05–0.30, capped at $0.50. For scale on a full suite: prior uncapped
500-step samples measured ~$0.27–0.32 of AWS per episode.

**These are estimates from a staging document, not measurements**, and the one
paid episode cost $0.00 in model spend (it ended before any call was billed),
so per-task cost has never actually been measured end to end.

## 8. Status and honest limitations

### What has been done

- The cloud-free slice — the complete adapter plus a `FakeProvider`, driven
  through the real `EpisodeRunner` — is proved end to end with zero AWS spend
  and is what CI exercises.
- A scripted-finish twin run (`--model-client scripted-finish --no-store`,
  the real paid infra values, canary AWS secrets) sealed
  `status=completed`, `reportability=synthetic_model`,
  `comparability=comparable`, `verify_bundle: valid=True, issues=[]`, with the
  rescue inbox empty and the canary values absent from every byte under the
  run root.
- **One paid episode** (`~/worktrees/osworld/runs/proof-20260902-081937`,
  bundle `evidence/ep-6ea01a117eee`), task_001, 25 steps.

### What that one paid episode actually showed

The AWS side worked: the instance launched, the TTL schedule was created and
later deleted, the instance was terminated, and the post-episode tag audit
printed `[]`. The episode **failed on its first decision** — the adapter
publishes its frame as `frame_id="screen"`, the model answered `frame_id "1"`,
and nothing in the prompt or the rendered observation had ever listed the
valid ids. It sealed `status=failed`, `reportability_label=cleanup_incomplete`,
`score.status=unscored` (`reason: infrastructure_failure`),
`rescue_required=true`. **Zero `model_request` events**: the one billed call
was never recorded, and model spend was $0.00.

It produced #542 (the frame-id contract, plus the bytecode-proof digest) and
#543 (cache routing), and its stranded descriptor is what exposed #548.

### What has NOT been done

- **No full 108-task run.** Not once, not partially. There is no aggregate
  score, no per-task score, and no measured suite cost or wall time.
- **No successful paid episode.** One paid episode exists and it failed on
  turn one. No episode has ever reached `scoring_result` against a real model.
- **No live kill-and-rescue drill.** The drill — SIGKILL the parent mid-episode,
  confirm the audit is non-empty, sweep, confirm it returns to `[]` — was
  staged and still pending at the time of writing. #548's fix was verified
  against a copy of the real descriptor whose instance was already terminated,
  which proves the code path executes but not that it terminates a live
  leaked instance.
- **No judged tasks.** Tasks whose evaluator imports the LLM judge are refused
  at preflight and again at `reset_start` without the judge credential, rather
  than silently scoring zero (upstream's `llm_metrics` returns `0.0` on any
  exception; a previous pilot scored ~17% of its suite as silent zeros that
  way). Wiring the judge with a receipt is later work.
- **No infeasible-task support.** The runner returns on a `finish` batch
  without calling `execute`, so the adapter never sees the terminal action and
  cannot push `DONE`/`FAIL` into OSWorld's `action_history`, which
  `evaluate()` reads to score `infeasible` tasks. An agent correctly declaring
  such a task infeasible would score 0. Fabricating a `FAIL` the agent never
  sent would be score fraud, so `reset_start` instead raises
  `InfeasibleTaskExcluded` before allocating anything.

### Known limitations and residual risks

- **Episode scratch has no reaper.** Each episode's cache root under
  `<run-root>/../osworld-cache/<episode-id>` is created and never collected.
  The one paid episode's run root is 3.2 MB, but earlier full-length samples
  under `~/worktrees/osworld/run-r13/` measure ~340 MB each, so a 108-task
  suite accumulates tens of gigabytes with nothing to clean it up.
  *(The oft-quoted ~59 MB/episode figure could not be reproduced from the
  artifacts on disk; the measured figures above are what this doc reports.)*
- **The adapter's lock predates several harness versions.** The committed
  `uv.lock` pins `local-operator==0.44.26`, which is older than the schema-1.2
  worker it must speak to; the staged venv was built with a manual
  substitution to the then-current harness version. The lock must be bumped
  (`uv lock --upgrade-package local-operator`) rather than relied on.
- **A documented `requests` override.** OSWorld pins `requests~=2.31.0` while
  local-operator requires `requests>=2.32.0` (a security floor). The two
  cannot both be satisfied; the adapter carries
  `override-dependencies = ["requests>=2.32.0"]`, so the installed set
  deliberately violates upstream's pin. Upstream's use is plain GET/POST to
  the guest, which is why this is judged safe — but it is an override, and
  `uv pip check` reports it.
- **Screenshot-only observations.** The accessibility tree is not shipped as a
  frame (a geometry for an XML document is a fiction); its presence is
  recorded in observation metadata.
- **`user_simulator` is one-sided.** The harness's own responder supplies the
  answer the model sees; the benchmark's simulator is only notified.
- **Static parse coverage is 108/108, with five caveats.** Task fields are read
  by AST, never by import. Five tasks (015, 050, 056, 057, 072) interpolate an
  imported name into their `instruction` and keep the literal skeleton with
  `instruction_static=False` — honest because no harness decision reads the
  instruction text; OSWorld's live object supplies the real one at `reset`.
- **Security groups and IP drift are manual.** The operator supplies a
  pre-existing group; the adapter does not create or repair one. A residential
  or VPN address that changes mid-run makes the guest unreachable.
- **A fourth pre-existing test flake.** `test_redundant_roster_events_skip_the_sidecar_write`
  flakes on clean `main` (7/60 measured) by a different mechanism from the
  three fixed in #559; it reproduces only under load and is unfixed. It is
  unrelated to this apparatus but will show up in a full suite run made
  alongside benchmark work.
- **Never place anything under `/tmp`.** macOS purges `/private/tmp` without
  warning; a purge mid-run once destroyed a pilot's prepared checkout, its
  assets and its output directory, and left an EC2 instance running.
  `run_episode.py` refuses a volatile run root for this reason.

### Permanently out of scope

Multi-task run orchestration, leaderboard reporting and aggregation, and any
CLI or TUI surface for benchmark running.

## Reproduction

Prerequisites: the gated inputs fetched into a durable inputs root, the
adapter built and installed per
[`benchmarks/osworld_v2_adapter/README.md`](../../../benchmarks/osworld_v2_adapter/README.md),
and the one-time AWS objects (TTL role, security group) created by hand. The
encrypted secret store (`~/.local-operator/secrets/store.db`, the one
`lop secret set` writes) must carry `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` as agent secrets — the store resolver reads the
encrypted store first and deliberately does **not** fall back to the process
environment or to `~/.aws`, so a name absent from the store is missing even if
the shell exports it. (A legacy `credentials.env` is still read as a transition
fallback until it is migrated with `lop secret migrate-env`.) Use a scoped key: `ec2:RunInstances, DescribeInstances,
DescribeImages, DescribeVolumes, TerminateInstances, CreateTags`,
`scheduler:CreateSchedule, DeleteSchedule, ListSchedules`, and `iam:PassRole`
on the TTL role. Presence-only check, never printing a value:

```sh
lop secret list | grep -cE '(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)'  # expect 2
```

**1. Pre-audit — must print `[]`.**

```sh
export AWS_PROFILE=<operator profile>          # audit only; the episode reads the store
PY=~/worktrees/osworld/venvs/<version>/bin/python3.12
$PY ~/local-operator/scripts/osworld_tag_audit.py --region us-east-1   # expect [] and exit 0
curl -s https://checkip.amazonaws.com          # must equal the /32 in the security group
find ~/worktrees/osworld/runs -name rescue.json                        # expect nothing
```

If the controller's address has drifted from the group's `/32`, update the
three rules by hand first — the adapter does not repair groups.

**2. One episode.**

```sh
RUN=~/worktrees/osworld/runs/proof-$(date +%Y%m%d-%H%M%S); mkdir -p "$RUN"

$PY ~/local-operator/scripts/run_episode.py \
    --selector ~/worktrees/osworld/workspaces/<version>/selector.json \
    --task-id task_001 \
    --route openrouter/deepseek/deepseek-v4-flash-vision-exp \
    --run-root "$RUN" \
    --infra AWS_REGION=us-east-1 \
    --infra AWS_SUBNET_ID=<subnet id> \
    --infra AWS_SECURITY_GROUP_ID=<security group id> \
    --infra AWS_SCHEDULER_ROLE_ARN=<TTL role arn> \
    --infra OSWORLD_CLIENT_PASSWORD=osworld-public-evaluation \
    --infra OSWORLD_FILE_BASE_URL=~/worktrees/osworld/gated/assets \
    --infra OSWORLD_TTL_SECONDS=2700 \
    --max-steps 25 --max-usd 0.50 --max-wall-s 1800 --keep-recent-frames 3 \
    | tee "$RUN/outcome.json"
```

This is a SMOKE command: 25 steps, $0.50, and a matching short wall and lease.
A scored run uses the standard 500-step budget and the 18000 s default wall,
and lets the lease derive from it -- see `BUDGETS_AND_LATENCY.md`. Do not copy
these caps into a run whose numbers you intend to report.

Exit 0 means `completed`; 1 is any other terminal state; 2 is a missing or
unusable secret (named on stderr, value never printed) or a volatile run root.
`OSWORLD_FILE_BASE_URL` is a **host-side local path** — the setup controller
runs on the controller host and copies assets into its cache before POSTing
them to the guest, so no S3, HTTP server, or HF token is involved.
`osworld-public-evaluation` is upstream's documented default password for
every OSWorld 2.0 image. `OSWORLD_TTL_SECONDS=2700` is the 1800 s wall budget
plus the 900 s slack, which is the manual step §2 explains.

**`OSWORLD_CLIENT_PASSWORD` must be a value the image accepts, and it is not
optional bookkeeping.** Every privileged preparation step reaches `sudo` with it
(the candidate ladder above), and a value this image rejects fails all of them —
measured 2026-09-21, four paid episodes: `sudo: no password was provided` on
every step, the guest's root filesystem reaching 0 bytes free,
`OSError: [Errno 28] No space left on device` on the next screenshot, and a
transport error minutes later. Such an episode is now REFUSED at preparation,
naming the step and this knob, so a wrong value costs an instance launch and a
re-run rather than a billed episode. If a run stops with that refusal, read
`<run-root>/osworld-cache/<episode-id>/guest-preparation.json` — the failing
steps, their status and the escalation chain are in it.

**3. Post-audit and bundle verification.**

```sh
$PY ~/local-operator/scripts/osworld_tag_audit.py --region us-east-1   # MUST print []
find "$RUN/rescue" -name rescue.json                                   # MUST be empty
$PY - "$RUN" <<'PY'
import json, sys
from pathlib import Path
from local_operator.evaluation.evidence.verify import verify_bundle
run = Path(sys.argv[1]); o = json.loads((run / "outcome.json").read_text())
r = verify_bundle(Path(o["bundle_root"]))
print("valid:", r.valid, [i.code for i in r.issues],
      "| status:", o["status"], "| score:", o["score"],
      "| reportability:", o["reportability_label"])
PY
```

If the audit is non-empty, run the sweep and audit again. Never terminate by
hand first — the sweep is what produces the receipt:

```sh
$PY ~/local-operator/scripts/osworld_rescue_sweep.py --rescue-root "$RUN/rescue"
```

**4. Kill-and-rescue drill (a second paid episode, ~$0.05–0.10).** Start the
same episode in the background; wait until both the descriptor exists and the
tag audit is non-empty; `kill -KILL` the parent and any orphaned worker; the
audit must then be non-empty and `rescue.json` present. Run the sweep, and
confirm the audit returns to `[]` with an empty rescue inbox. Even if the
sweep were never run, the `lop-ttl-<episode>` schedule terminates the instance
at `OSWORLD_TTL_SECONDS` — belt and braces. **This drill has not yet been
executed** (§8).

**Cloud-free rehearsal.** The whole plumbing — spawn, secrets, frames, seal —
runs with no provider call and no AWS spend against a workspace whose
`adapter-provider.json` names the fake provider:

```sh
$PY ~/local-operator/scripts/run_episode.py \
    --selector <fake-provider workspace>/selector.json \
    --task-id task_001 --route <any route> --run-root "$RUN" \
    --model-client scripted-finish --no-store \
    --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
    ...
```

Such a bundle verifies like any other but seals `synthetic_model` and can
never be mistaken for a result.
