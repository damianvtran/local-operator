# OSWorld 2.0 adapter for local-operator

A separately-distributed evaluation adapter that runs OSWorld 2.0
(`osworld-v2-2026.08.08`) episodes behind local-operator's verified adapter
boundary. The cloud-free slice — the complete adapter plus a `FakeProvider`,
proved end to end through the real `EpisodeRunner` with zero AWS spend — is
what CI exercises. `providers/aws.py` is the production backend: it launches
the guest on EC2 with boto3, leases it with an EventBridge TTL schedule, and
hands the instance to upstream's `DesktopEnv`.

## Why it is its own distribution

The adapter imports `local_operator.evaluation.adapters.api`, so it is coupled
to the harness protocol version and lives in-tree. But
`discovery.distribution_digest` pins the adapter by verifying every RECORD row
of the installed wheel — so it must be a *separate* distribution, or every
harness release would invalidate the adapter pin. Isolation comes from the
wheel + digest + isolated worker, not from the source's location.

## Prerequisite: the gated inputs, in a DURABLE root

OSWorld 2.0's 108 task classes and its 4.2 GB of assets are **gated** Hugging
Face datasets (`xlangai/osworld_v2_tasks`, `xlangai/osworld_v2_assets_gated`).
A human accepts the terms once and fetches them, together with the OSWorld
checkout at the pinned commit, into an **inputs root**:

```
$OSWORLD_INPUTS_ROOT/            (default ~/worktrees/osworld)
├── prepared/                    xlang-ai/OSWorld-V2 at d578d2d4 (git checkout)
│   └── benchmark_releases/osworld-v2-2026.08.08.json
└── gated/
    ├── tasks/task_*.py          108 task modules
    ├── tasks/manifests/task_hashes.json
    ├── manifests/assets.json    per-file sha256 of the asset snapshot
    └── assets/                  4.2 GB, served to the guest via OSWORLD_FILE_BASE_URL
```

**Never put the inputs root, the workspace, or a run's output under `/tmp`.**
macOS purges `/private/tmp` on disk pressure and on a periodic sweep with no
warning; a purge mid-run destroyed a previous paid pilot's prepared checkout,
its assets, its output directory, and left an EC2 instance running.

The build script (below) verifies every input against the committed pin
`config/release-v2026.08.08.json` — release manifest sha, task-hash manifest
sha, every task file sha, task count, prepared checkout commit — and refuses
(exit 4, naming the path) on any mismatch. It never downloads. The assets are
**not** copied into the workspace (the workspace cap is 4 GiB); the workspace
records their manifest sha in `inputs.json`, and the adapter re-verifies the
live root against it at every `reset_start`.

## Build, lock, install

```sh
cd benchmarks/osworld_v2_adapter

# 1. lock (in the source tree). The committed uv.lock pins the harness; bump
#    it so the locked local-operator is not the stale 0.44.26 (which predates
#    schema 1.2 the worker speaks).
uv lock --upgrade-package local-operator   # writes uv.lock (committed)

# 2. build the wheel
uv build --wheel --out-dir dist/

# 3. create the dedicated interpreter + install the locked set.
#    --copies is REQUIRED (python_executable must be a real file, not a
#    symlink — discovery._symlink_free), but current uv's `venv` has NO
#    `--copies` flag (that is uv pip's install link-mode). Use the stdlib
#    venv, which DOES copy. `--without-pip` because uv pip installs the rest
#    and the uv-managed interpreter's ensurepip can SIGABRT on macOS.
#
#    Invoke the venv module through the interpreter's REAL path, never through
#    a ~/.local/bin shim: venv records the invoking argv[0]'s directory as
#    `home` in pyvenv.cfg, and step 3a resolves libpython relative to it. A
#    shim directory holds no lib/, so the copy silently finds nothing.
VENV=/opt/lop-adapters/osworld-v2/0.1.1/venv
WS=~/worktrees/osworld/workspaces/0.1.1/workspace   # the digest-pinned dir itself
BASE=$(python3.12 -c 'import sys; print(sys.base_prefix)')   # e.g. ~/.local/share/uv/python/cpython-3.12.13-...
"$BASE/bin/python3.12" -m venv --copies --without-pip "$VENV"

# 3a. copy libpython beside the interpreter. REQUIRED on macOS and easy to
#     miss: `--copies` copies the python binary but NOT the shared library it
#     links, and the copied binary keeps `@rpath/libpython3.12.dylib` with an
#     rpath of `<venv>/lib`. Without this the interpreter aborts on every
#     invocation with "Library not loaded" (dyld) / SIGABRT, and uv reports
#     the opaque "Failed to inspect Python interpreter". A framework or
#     system python that links libpython by absolute path does not need it;
#     the uv-managed CPython this recipe uses does.
#     The copy is allowed to fail (a python that needs no dylib has none to
#     copy); the -V beside it is the real check and catches a genuine miss.
cp "$BASE/lib/libpython3.12.dylib" "$VENV/lib/" 2>/dev/null || true   # macOS only
"$VENV/bin/python3.12" -V   # must print the version, not abort
uv pip install --python "$VENV/bin/python3.12" \
    --no-deps dist/lop_osworld_v2_adapter-0.1.1-py3-none-any.whl

# 3b. install the locked set. For the PAID path export WITH the osworld extra
#     (~380 packages the cloud-free wheel does not need); omit --extra for a
#     cloud-free venv. Export to a FILE and keep it beside the venv: it is the
#     record of what was actually installed, and the lock's own
#     local-operator pin lags the harness (schema 1.2 and host_secrets only
#     exist from 0.44.30+), so substitute the harness version you are running
#     and note the substitution in the build record.
uv export --frozen --no-emit-project --extra osworld --no-hashes \
    -o "$VENV/build/requirements.locked.txt"
sed -i '' 's/^local-operator==.*/local-operator==<harness version>/' \
    "$VENV/build/requirements.locked.txt"   # macOS sed; GNU sed wants -i without ''
uv pip install --python "$VENV/bin/python3.12" -r "$VENV/build/requirements.locked.txt"

# The one expected `uv pip check` incompatibility is the documented
# requests>=2.32 override onto OSWorld's ~=2.31 pin (see [tool.uv] above).

# 4. compute package_digest FIRST: it digests the installed wheel's RECORD and
#    does not depend on the workspace, while step 5's --package-digest does
#    depend on it. (Computing it after the build is what forces a second
#    build with the real value.)
PKG=$("$VENV/bin/python3.12" -c '
from pathlib import Path
from importlib.metadata import PathDistribution
from local_operator.evaluation.adapters.discovery import distribution_digest
import sysconfig
sp = Path(sysconfig.get_paths()["purelib"])
di = next(sp.glob("lop_osworld_v2_adapter-*.dist-info"))
print(distribution_digest(PathDistribution(di)))
')
echo "package_digest   $PKG"

# 5. materialise the workspace from the verified inputs root (no download;
#    writes adapter-release.json, benchmark_release.json, task_hashes.json,
#    adapter-provider.json, inputs.json, tasks/, all read-only).
#    --out is the WORKSPACE directory itself, not its parent: it is what the
#    selector's `workspace` field and workspace_digest address. Keep the
#    selector and build-output.json in the parent so they are not inside the
#    digest they describe.
#    Run it with the venv's interpreter (it imports the harness). --version
#    defaults to the adapter's own pyproject version and REFUSES a value that
#    disagrees with it unless --allow-version-mismatch is passed.
"$VENV/bin/python3.12" ~/local-operator/scripts/build_osworld_adapter.py \
    --benchmark-release osworld-v2-2026.08.08 \
    --inputs-root ~/worktrees/osworld \
    --package-digest "$PKG" \
    --out "$WS"

# 6. the remaining digest the AdapterSelector needs (release_digest is printed
#    by step 5 and written into adapter-release.json).
"$VENV/bin/python3.12" -c "
from local_operator.evaluation.adapters.discovery import workspace_digest
print('workspace_digest', workspace_digest('$WS'))
"
```

`release_digest` is our attestation of the build:
`sha256("lop-osworld-v2-adapter" || version || package_digest ||
benchmark_release_name || task_hash_manifest_sha256)`, written into both
`adapter-release.json` and the selector. It ties the harness build to the
benchmark release — the claim a leaderboard number must carry.

## One-time AWS prerequisites (operator, by hand, never automated)

The provider creates instances, volumes and one TTL schedule per episode. It
does NOT create IAM roles or security groups; those are one-time human steps.

1. **TTL role** → infra `AWS_SCHEDULER_ROLE_ARN`. A role trusted by
   `scheduler.amazonaws.com` (with an `aws:SourceAccount` condition) whose only
   permission is `ec2:TerminateInstances` on instances where
   `aws:ResourceTag/lop:adapter = osworld-v2`. The provider creates the schedule
   **immediately after `run_instances`, before waiting for readiness**, and a
   failure to create it terminates the instance and fails the episode — never a
   warning. Because the role is tag-scoped, the adapter tag MUST be on the
   instance at creation (it is, via `TagSpecifications`).
2. **Security group** → infra `AWS_SECURITY_GROUP_ID`. A pre-existing group in
   the default VPC allowing inbound TCP **5000** (guest control), **5910** (VNC,
   optional) and **9222** (Chrome DevTools) from the controller's current
   `/32`. The adapter does not create or repair groups yet; a residential or
   VPN address that changes mid-run makes the guest unreachable (see the
   previous pilot's "controller IP drift"). Re-check your address before a
   run.
3. **Subnet / region** → infra `AWS_SUBNET_ID`, `AWS_REGION=us-east-1`. The
   release AMI exists only in us-east-1; every client is built with an
   explicit region, so a profile whose default region differs is fine.
4. **Credentials** → secrets `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   (optional `AWS_SESSION_TOKEN`) in the harness credential store. Use a
   **scoped key**: `ec2:RunInstances, DescribeInstances, DescribeImages,
   DescribeVolumes, TerminateInstances, CreateTags`, `scheduler:CreateSchedule,
   DeleteSchedule, ListSchedules`, and `iam:PassRole` on the TTL role. An admin
   key works but should not be the one a worker subprocess holds.
5. **Guest settings** → infra `OSWORLD_CLIENT_PASSWORD`, `OSWORLD_FILE_BASE_URL`;
   optional `OSWORLD_INPUTS_ROOT` (default `~/worktrees/osworld`) and
   `OSWORLD_TTL_SECONDS` (default 7200). Optional `AWS_INSTANCE_TYPE`
   (purpose `benchmark_compute`) replaces the EC2 instance type for the
   benchmark VM; it **overrides a task's own `instance_type`**, because the
   task files are content-hash verified and cannot be edited to escape an
   infrastructure problem their author never saw. Omitting it reproduces the
   previous behaviour exactly. A malformed value is refused at `prepare`,
   before anything is allocated (shape only — a well-formed but nonexistent
   type still reaches `run_instances`). Reach for it when the default
   burstable `t3.xlarge` runs out of CPU credits — see "Burstable credit
   exhaustion" in `docs/benchmarks/osworld_2/README.md`.

   Optional `AWS_ROOT_VOLUME_SIZE` (purpose `benchmark_compute`, a whole
   number of GiB) replaces the root volume size for the benchmark VM. It
   follows `AWS_INSTANCE_TYPE` exactly: it **overrides a task's own
   `volume_size`**, for the same content-hash reason, and omitting it
   reproduces the previous behaviour exactly (the task's pin, else the AMI's
   own size resolved at launch). A non-integer, zero, negative, or
   out-of-range value is refused at `prepare` before anything is allocated;
   a size smaller than the AMI's own snapshot is refused at launch, with both
   numbers named, because establishing that floor needs a `describe_images`
   call `prepare` may not make. Reach for it when episodes die at a
   consistent wall-clock time — see "Disk exhaustion" in
   `docs/benchmarks/osworld_2/README.md`.

   **Both values require a workspace built from adapter source that supports
   them.** The source here gained them without a version bump (the bump would
   falsify the pilot's pinned attestation), so the shipped source and the
   digest-pinned `0.1.1` wheel are different code under one version string.
   Supplying one to a build that does not declare it is refused by the runner
   before `prepare` rather than silently ignored — an ignored override would
   put a false hardware claim into a sealed bundle.
6. **Judged tasks only** → secret `OSWORLD_EVAL_MODEL_API_KEY` plus infra
   `OSWORLD_EVAL_MODEL_PROVIDER` / `OSWORLD_EVAL_MODEL_NAME` (purpose
   `benchmark_judge`). A task whose source imports the judge client is
   **refused at preflight and again at `reset_start`** without them: OSWorld's
   `llm_metrics` returns `0.0` on any exception, and the previous pilot scored
   ~17% of its suite as silent zeros that way. The key is the one secret the
   worker writes into its own environment (OSWorld reads it from nowhere
   else); it is scrubbed on `close`.

## How secrets reach the worker

The worker is spawned with an environment built from a closed allowlist, so
nothing ambient carries a credential. Resolved secrets travel **only** on the
private RPC pipe, on `reset_start` (the side-effect boundary) and on
`begin_rescue` (a fresh worker tearing down from a persisted descriptor). They
are never on `prepare`, never in `rescue.json`, and the AWS values never touch
`os.environ`; the provider builds its boto3 session from them directly.

On the parent side the runner resolves `EpisodeSpec.secret_refs` through an
injected `SecretResolver` (`runner/secrets.py`) **after the handshake and
before the evidence writer opens**, so every resolved value is a redaction
canary from the bundle's first byte. A missing ref fails the episode
`failed_pre_bundle` with a diagnostic naming only the ref — before `prepare`,
before any descriptor is persisted, before anything is allocated. The
credential-store resolver (`runner/host_secrets.py`) is the only runner
module besides `provider_client.py` allowed near the store.

## Running one episode

`scripts/run_episode.py` runs ONE episode end to end — real spawned worker,
real provider-backed model client, real credential store, sealed bundle — and
prints the `EpisodeOutcome` as JSON. It is an operator script, not a CLI
surface. Its `--run-root` must be durable (it refuses `/tmp`, `$TMPDIR` and
friends); `evidence/`, `artifacts/` and `rescue/<episode-id>/` are created
under it, so `<run-root>/rescue` is the inbox the sweep command below reads.

Secrets: every ref the task needs is either on `--secret-env NAME` (read from
the script's own environment, and only the names listed) or resolved from the
credential store's file (`~/.local-operator/credentials.env`, or
`--config-dir`) — the store resolver deliberately does NOT fall back to the
process environment, so a name absent from the file is missing even if the
shell happens to export it.
Both the AWS pair and the model client come from that same store, so a paid
episode needs no environment variables at all:

```sh
python ~/local-operator/scripts/run_episode.py \
    --selector ~/worktrees/osworld/workspaces/0.1.1/selector.json \
    --task-id task_001 \
    --route openrouter/google/gemini-3.8-flash \
    --run-root ~/worktrees/osworld/runs/$(date +%Y%m%d-%H%M%S) \
    --infra AWS_REGION=us-east-1 \
    --infra AWS_SUBNET_ID=subnet-f2f9adad \
    --infra AWS_SECURITY_GROUP_ID=<sg-id> \
    --infra AWS_SCHEDULER_ROLE_ARN=<role arn> \
    --infra OSWORLD_CLIENT_PASSWORD=<guest password> \
    --infra OSWORLD_FILE_BASE_URL=<asset mirror> \
    --max-steps 25 --max-usd 0.50 --max-wall-s 1800 --keep-recent-frames 3
```

Exit 0 only on `completed`; 1 on any other terminal; 2 when a secret is
missing or unusable (named on stderr, value never printed) or the run root is
volatile. Run the leak audit before and after.

`--model-client scripted-finish` (with `--no-store`) exercises the plumbing —
spawn, secrets, frames, seal — with no provider call. Such a bundle verifies
like any other but seals `reportability_label: synthetic_model` (never
`reportable`) and carries `model_client` in its manifest metadata; it is not
a result and cannot be mistaken for one.

The sealed `requested_route.model_id` is a **lossless fold** of the model id
(`RouteIdentity` fields cannot carry `/`): `_` → `__`, `/` → `_s`, anything
else outside `[A-Za-z0-9.:-]` → `_x<hh>` per UTF-8 byte, so
`google/gemini-3.8-flash` seals as
`google_sgemini-3.8-flash` and `runner.route_ids.unfold_model_id`
recovers it exactly. The manifest metadata also carries the raw id as
`route_model_id`, so a reader never has to decode by hand.

## Teardown and rescue

`cleanup` reports `succeeded` only on **positive** evidence: `terminate`
polls `describe_instances` until the state is `terminated` (up to 55 s) and
otherwise reports `terminate-unconfirmed`, which keeps `rescue_required` set.
`DesktopEnv.close()` is never called — it terminates without confirming.

**Upstream is sealed after the first `reset`.** OSWorld's own allocation
paths — a second `reset` on a used env (`_revert_to_snapshot` →
`AWSProvider.revert_to_snapshot`), `manager.get_vm_path`, `_save_state`,
`close`/`stop_emulator` — would launch or release an instance with no
`ClientToken`, no `lop:adapter` tag and no TTL lease: invisible to the audit,
unreachable by rescue. The provider replaces every one of those methods on
the live env with a raiser (`UpstreamAllocationRefused`) before any boto3
call. One `reset` per episode is the contract; the harness runner only ever
issues one (`reset_start` is PREPARED→RUNNING once), and the seal is what
makes that a guarantee rather than a convention. A test statically scans
the pinned upstream for any method that reaches `run_instances`/
`terminate_instances`/`create_image` and asserts it is sealed.

If the parent dies mid-episode, `<rescue root>/<episode-id>/rescue.json`
names the episode's refs (for `run_episode.py` the rescue root is
`<run-root>/rescue`). Sweep them all:

```sh
python ~/local-operator/scripts/osworld_rescue_sweep.py --rescue-root <run-root>/rescue
```

It spawns the exact pinned worker per descriptor, re-resolves the descriptor's
secret refs from the credential store, reconciles every action, and unlinks
the descriptor **only** when the aggregate is complete.

The sweep takes no `--region`, unlike the audit below: each descriptor
already carries the episode's `AWS_REGION` in its `infra_values`, and the
worker reconciles in that region. Passing one would only invite a mismatch
between where the operator thinks the instance is and where the descriptor
says it is.

The sweep re-hashes the pinned workspace before it will spawn the worker, so
the workspace must still match the selector's `workspace_digest`. Bytecode
caches never count: `__pycache__/` and `*.pyc` are excluded from the digest
by rule, and the worker is launched with `-B` so it writes none — upstream's
`instantiate_task` imports the task module from the workspace, and in the
first paid episode the cache that import left behind made the sweep refuse
("adapter workspace content digest differs") for an instance that was in
fact already terminated. Anything else that changes under the workspace is a
genuine drift and the sweep will (rightly) refuse; restore the workspace from
the verified inputs root and sweep again.

**Upstream's cache lives OUTSIDE the workspace for the same reason.**
`DesktopEnv`'s default `cache_dir="cache"` is resolved against the worker's
cwd — and the cwd IS the pinned workspace. The first paid episode's
`_download_setup` wrote `cache/001/…calendar.ics` and `…thunderbird-profile.tar.gz`
there, which the digest (correctly) does NOT ignore, so the rescue sweep
refused even though the instance was already terminated. The adapter now hands
`DesktopEnv` an ABSOLUTE, per-episode cache root at
`<artifact_root>/../osworld-cache/<episode_id>` — beside the artifact root, not
inside it (the bundle verifier walks the artifact directory and refuses any
non-digest entry), and per-episode so two episodes running the same task never
share a cache — upstream's `reset_cache_dir` only reassigns the attribute
(`controllers/setup.py:55-56`) and clears nothing, so the isolation has to come
from the path. The cache root is durable (it is a sibling under the run root,
which `run_episode.py` puts through `refuse_volatile_root`, so it can never be
`/tmp`) and is minted at `reset_start`, never in `prepare`.

**The episode's cache root is also its working directory.** Routing `cache_dir`
fixes the one write path we observed; it does not fix the class. Several
upstream helpers open a hard-coded RELATIVE name with the builtin `open` at
module scope — `temp.pdf` (`evaluators/metrics/vscode.py:210`),
`temp_extracted_<n>.jpeg` (`slides.py:2051`), an epub's `<name>.dir`
(`others.py:64-72`) — and they never consult the env object, so no attribute
the adapter installs on it can intercept them. Since a relative write resolves
against the cwd, `reset_start` moves the worker's cwd to the episode cache root
and `close` puts it back. Any relative write by any upstream path, known or
not, therefore lands in the episode's scratch dir rather than the pin. This is
safe because the workspace is never reached via the cwd: `instantiate_task`
loads task modules by absolute location, the worker runs with `-I` so the cwd
is not on `sys.path`, and both the worker's digest re-check and the rescue
sweep hash `selector.workspace`, an absolute path.

If a future upstream path still manages to write into the workspace, the
digest re-check is what surfaces it: the worker (and `sweep-rescue`) refuse
with `adapter workspace content digest differs`. Restore the workspace from
the verified inputs root and sweep again.

## Leak audit (operator command)

Every instance and volume this adapter creates carries `lop:adapter=osworld-v2`
and every lease is a schedule named `lop-ttl-<episode>`, so one read-only
script is a complete inventory:

```sh
python ~/local-operator/scripts/osworld_tag_audit.py --region us-east-1
```

It prints `[]` and exits 0 when clean, otherwise lists what it found and exits
1. It never terminates anything. **Run it before and after every paid episode;
it must print `[]`.**

## Known scope limitations

- **Infeasible tasks are refused, not merely undocumented.** The runner returns
  on a `finish` batch without calling `execute` (episode.py:531-534), so the
  adapter never sees the terminal action and cannot push `DONE`/`FAIL` into
  OSWorld's `action_history`, which `evaluate()` reads to score `infeasible`
  tasks. An agent correctly declaring such a task infeasible would score 0. We
  will not fabricate a `FAIL` the agent never sent — that would be score fraud.
  `reset_start` therefore raises `InfeasibleTaskExcluded` **before allocating
  anything** when a task's evaluator declares `func: "infeasible"`, so an
  operator-built workspace containing one fails loudly instead of silently
  grading an honest refusal as a failure. Support lands when the terminal-claim
  gap is resolved.
- **Static parse coverage: 108 of 108 pinned tasks.** Task fields are read
  by AST, never by import, so a field bound through code is a refusal, not an
  execution. The closed set of shapes the parser folds — a module-level
  constant, an earlier class attribute, a parenthesised f-string over those,
  and `"...".strip()` — covers every task in the `v2026.08.08` corpus. Five
  tasks (015, 050, 056, 057, 072) interpolate an *imported* name into their
  `instruction`; their descriptor keeps the literal skeleton with
  `instruction_static=False`, which is honest because no harness decision
  reads the instruction text (OSWorld's live object supplies the real one at
  `reset`). `id` and `user_simulator` must resolve completely or the task is
  refused with the field named.
- **Guest disk reclamation runs before every episode's first observation.**
  The released AMI ships ~93% full and the guest's own snapd fills the rest from
  boot (a 9.7 GB `/var/lib/snapd/cache` plus an `Auto-refresh 9 snaps`), which
  took the root filesystem to 0 bytes at ~t+383s and destroyed 7 of 8 episodes
  in a 424-466s window. `allocate` therefore holds snap auto-refresh, aborts
  in-flight snap changes, and clears the snapd download cache between guest
  readiness and upstream's `reset`. It is conditional (below 12 GiB free), it
  fails soft step by step, it never removes an installed snap or anything a task
  could need, and it writes `guest-preparation.json` — free space before and
  after, the disk geometry, every step's outcome — into the episode's cache
  root. See "Guest disk reclamation at episode start" in
  `docs/benchmarks/osworld_2/README.md`, which also records that the earlier
  `x11grab`/ffmpeg diagnosis was wrong.
- **Screenshot-only observations.** The a11y tree is not shipped as a frame
  (a geometry for an XML document is a fiction). Its presence is recorded in
  observation metadata; shipping it is a protocol addition, not a fake frame.
- **`user_simulator` is one-sided.** The harness's own responder supplies the
  answer the model sees; the benchmark's simulator is notified for the record.
  Faithful two-sided wiring is a later PR. Pilot tasks declare no simulator.
- **Temporary security groups and IP-drift repair are not automated.** The
  operator supplies a pre-existing group (`AWS_SECURITY_GROUP_ID`); a
  per-episode group with a `CleanupActionKind` of its own is a later PR.
- **The judge is refused, not wired.** Judged tasks fail preflight without the
  judge credential; running them through the judge with a receipt, and wiring
  `LLMUserSimulator` as the user responder, is a later PR.
- **The TTL lease is not yet derived from the wall budget.** The budget lives
  in the runner's `BudgetAuthorization` and is not on the adapter wire; the
  lease is `OSWORLD_TTL_SECONDS` or the 7200 s default.

## Out of scope (permanently)

Multi-task run orchestration, leaderboard reporting/aggregation, and any CLI
or TUI surface.
