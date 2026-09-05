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

**Status at time of writing (2026-09-02): no score exists.** One paid episode
has been run; it failed on its first decision, and the two suite-wide numbers
this apparatus is designed to produce (per-task binary completion, aggregate
completion over the 108 tasks) have never been measured. Everything below
describes an apparatus that is built and partly exercised, not a result.
[Status and honest limitations](#status-and-honest-limitations) states exactly
what has and has not been done, and it is the section to read first if the
question is "can I trust a number from this?".

## 1. The benchmark

OSWorld 2.0 (`osworld-v2-2026.08.08`) is a computer-use benchmark of **108
long-horizon workflows** on a real Ubuntu desktop. Upstream reports a median
human completion time of about 1.6 hours per task and an average of ~318 tool
calls for a frontier model, against about 30 in OSWorld 1.0; its headline
figures are quoted **under a binary-completion metric at 500 steps**, where
the best reported model completes 20.6% of tasks (54.8% partial). Tasks are
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

- exact `0.0` → `binary=0`; exact `1.0` → `binary=1`;
- any other in-range value → `partial_ppm = round(v * 1_000_000)` (the
  protocol's portable-metadata subset excludes floats; V2's `conj: "avg"`
  and `"sum"` evaluators produce genuine fractions, so this path is real);
- NaN, infinity, out-of-range, non-numeric, or a missing evaluator → **raise**.

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
`DEFAULT_TTL_SECONDS = 7200`, floored at `TTL_SLACK_SECONDS = 900`.
`ttl_seconds_for` can derive `wall_budget + 900 s` from a capped wall budget,
but **the wall budget is not on the adapter wire today**: the adapter passes
`None` for it and only the override or the 7200 s default is ever in force.
Setting `OSWORLD_TTL_SECONDS` to the wall budget plus 900 is therefore the
operator's job, and is what bounds a leaked instance's worst-case cost.

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

The measured consumer is **snapd**, inside the guest:

- `/var` is 15G of the 29G disk, and `/var/lib/snapd/cache` **alone is 9.7 GB**
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

Measured effect: a 100 GiB volume moved the first failure from t+424s to
**t+1936s**. It does not fully fix the problem, and the reason is geometry —
the root **partition** stays 29.5G inside that 100 GiB disk, with ~70 GiB
unallocated, because the AMI carries no `growpart` and
`apt-get install cloud-guest-utils` cannot run on a disk with no free space to
download into. The extra GiB bought time only because a larger volume's
filesystem happens to start with more slack, not because the partition grew.

Since 0.46.12 the adapter reclaims the space itself at episode start, which
addresses the cause rather than the symptom — see below. `AWS_ROOT_VOLUME_SIZE`
remains available as an independent lever.

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
so a future third value cannot be gated without being disclosed or vice versa.

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
3. **Clear `/var/lib/snapd/cache`** — pure download scratch, documented by
   Canonical as "the working cache … used to minimise download size and speed-up
   refreshes". Deleting it costs a re-download and nothing else.

Every privileged step is exactly one `echo <password> | sudo -S bash -c
'<fragment>'`, with all of the work — the `snap changes` loop, the
`find /var/lib/snapd/cache -mindepth 1 -delete` — inside the privileged inner
shell. That shape is not cosmetic: the control server runs as an unprivileged
user, so a glob like `/var/lib/snapd/cache/*` expanded by the *outer* shell
matches nothing against a `drwx------ root:root` directory and `rm -rf` of the
literal name exits 0, and an `xargs … echo pw | sudo -S snap abort` pipeline
parses as `xargs echo` piped into one id-less `sudo`. Both were real defects
that reported `ok` while doing nothing, caught only by re-running the E2E with
the cache directory genuinely owned by root.

The design constraints are worth stating explicitly, because each is a line a
future change could cross without noticing:

- **It is environment preparation, not benchmark semantics.** Nothing here
  changes the task, the scoring, the applications available, or anything the
  model observes. Clearing a package manager's download cache is housekeeping;
  **uninstalling** an application a task might need is not, so no command may
  ever `snap remove` or `apt-get purge` anything. A test asserts that.
- **It fails soft, per step.** A missing binary, a denied `sudo`, an
  unreachable control server, or a guest that answers slowly are each recorded
  and stepped over. An episode that would otherwise have worked must never be
  destroyed by a hygiene step, and a whole-pass budget keeps a wedged guest from
  eating the reset timeout.
- **It is conditional.** Free space is measured on every episode, but the
  reclamation only runs below **12 GiB free** — set above snapd's largest
  measured appetite (the 9.7 GB cache), so a guest that can already absorb a
  full auto-refresh is left untouched. A guest whose free space cannot be
  measured *is* reclaimed: the protection must not go missing exactly when the
  guest is least healthy.
- **It is observable.** Free space before and after, the filesystem and
  whole-disk sizes, and every step's outcome are written to
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
instead, read-only, since that pair of numbers is what tells the next reader
whether `AWS_ROOT_VOLUME_SIZE` bought anything.

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
| `--max-steps` | 25 | bounds the step loop; `EpisodeConfig.max_steps` itself defaults to 50 |
| `--max-usd` | 0.50 | hard provider spend cap |
| `--max-wall-s` | 1800 | wall clock; **not** propagated to the TTL lease (§2) |
| `--max-cycle-usd` | none | per-cycle cost-rate guard |
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
credential store (`~/.local-operator/credentials.env`, mode 600) must carry
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` — the store resolver
deliberately does **not** fall back to the process environment or to
`~/.aws`, so a name absent from that file is missing even if the shell
exports it. Use a scoped key: `ec2:RunInstances, DescribeInstances,
DescribeImages, DescribeVolumes, TerminateInstances, CreateTags`,
`scheduler:CreateSchedule, DeleteSchedule, ListSchedules`, and `iam:PassRole`
on the TTL role. Presence-only check, never printing a value:

```sh
grep -cE '^(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY)=.' ~/.local-operator/credentials.env  # expect 2
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

Exit 0 means `completed`; 1 is any other terminal state; 2 is a missing or
unusable secret (named on stderr, value never printed) or a volatile run root.
`OSWORLD_FILE_BASE_URL` is a **host-side local path** — the setup controller
runs on the controller host and copies assets into its cache before POSTing
them to the guest, so no S3, HTTP server, or HF token is involved.
`osworld-public-evaluation` is upstream's documented default password for
every OSWorld 2.0 image. `OSWORLD_TTL_SECONDS=2700` is the 1800 s wall budget
plus the 900 s slack, which is the manual step §2 explains.

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
