# Two-tier cloud compute provisioning — Fargate tasks and a dedicated EKS workload tier

Status: **implementation design, pre-code. Nothing here is built and nothing here
authorises provisioning.** Every AWS resource described below is behind a disabled
gate; the only account named is read-only-verified.

Base: branch `poc/remote-cloud-agents-slice0` @ `3d2bcf3c8` (a merge of
`origin/main` @ `6891c13d5` with `docs/remote-cloud-agents` @ `ac846d4d7`), read in
the worktree `~/src/lo-cloud-tiers`. **Line numbers drift: every code reference below
is by SYMBOL.** Siblings, in their order of authority here:
`remote-cloud-agents.md` (the product design this implements), `mesh-compute-pool.md`
(the pool contract), `mesh-session-mobility.md` (placement, sync, `exec --peer`),
`remote-cloud-agents-poc-results.md` (what Slice 0 actually measured).

## 0. The ask, and the answer in one page

**The operator asked for two kinds of auto-provisioning:**

1. **Short-lived tasks on ECS Fargate** — one task per run. Slice 0 (`scripts/remote_agents_poc.py`
   + `infra/remote-agents-poc/`) is the seed and it is *already proven* on a real-key run.
2. **Dedicated workloads on a Kubernetes cluster with two node pools** — one **static**
   (on-demand) pool and one **spot** pool.

**What this document adds to the two existing designs** — nothing in
`remote-cloud-agents.md` is contradicted; it is made concrete and cut down to a
first slice:

| New thing | Where | Why it is not already there |
|---|---|---|
| `ComputeProvider` Protocol + `PoolRequest` + `ComputeTask` | §3 | `remote-cloud-agents.md` §5.1 maps the *contract* onto a **remote control plane**; there is no local, testable seam today and no `PoolRequest` in code |
| A local **task ledger** under the config dir | §3.4 | mesh-compute-pool §4.1's `pool-requests/` dir assumes C4's remote reader; our slice provisions locally, so the record needs a local store with the house write discipline |
| `FargateProvider` | §4 | Slice 0's driver is a POC script (`scripts/`), not product code: no Protocol, no status record, no cancel, no injected client |
| `K8sWorkloadProvider` + the two-pool scheduling contract | §5 | Tier 2 does not exist in any form — `remote-cloud-agents.md` §4 **rejected** EKS as the *per-task* substrate; this tier is for *dedicated* workloads, which is a different question and is where the rejected reasons do not apply |
| IaC for tier 2 in the POC's own Pulumi YAML style | §6 | `infra/remote-agents-poc/` is Fargate-only |
| `lop cloud run\|status\|cancel\|list`, `lop exec --cloud` | §7 | Designed (`remote-cloud-agents.md` §1.1, D6) but not built |
| An explicit scope cut and a 4-slice plan | §9, §10 | Neither design doc has one for *this* work |

**The one substitution this document makes:** tier 2's IaC and the *tier-1*
substrate both target `ca-central-1` in account `325492156725`
(`minerva_sandbox`), because that is where Slice 0's resources and its reconciled
bill live. `remote-cloud-agents.md` §11 **D1** still recommends a dedicated
Radient-owned account, and that decision is **not** made here: nothing in this
design deploys, and the IaC is a separate program with `enableEks: false` (§6).

**The two decisions a reader is most likely to want to argue with:**

* Tier 2 is a **separate Pulumi program with its own stack**, not a flag inside
  `infra/remote-agents-poc/Pulumi.yaml` — because Pulumi YAML has no conditionals
  (§6.4) and because the two stacks must be destructible independently (§6.2).
* `placement.mode: "pool"` is **NOT produced in this slice** (§9.3). A safe minimal
  admission story for a hosted pod is design item **C2** (attested pool handshake,
  the pod never holds the epoch secret), which is new code in `handshake.py` /
  `relay.py` and is the largest single piece in the whole design. Shipping
  `mode: "pool"` without it means either a pod holding the network's epoch secret —
  which *is* `admin` (`relay.py` `_ctl_invite` mints any role from it) — or a
  placement pointing at a member row nothing authenticated. This slice therefore
  provisions **unbound** tasks (`network_id: ""`) and returns their work the way
  Slice 0 already does: a branch, a session directory and a bundle.

---

## 1. What exists today, verified at this revision

Read at `3d2bcf3c8`, in this worktree. These are the facts the design builds on, each
one checked rather than recalled.

### 1.1 The mesh half (real, and deliberately not reused yet)

| Claim | Where it actually is | Verdict |
|---|---|---|
| `MemberKind` = `device \| pool` | `local_operator/network/types.py`: `MemberKind` | **Present** |
| `MemberLifecycle` = `active \| provisioning \| draining \| expired` | `local_operator/network/types.py`: `MemberLifecycle` | **Present** |
| Pool members get no credential rows | `local_operator/network/onboard.py`: `OnboardRun.step_provision` — `grants_credentials = kind != "pool" and role in ("drive","admin")`, and the `pool` arm records `"no credential rows were written: this member is a pool member (§1.3)"` | **Present, exactly as described** |
| Rollouts skip pool members | `local_operator/network/rollouts.py`: `_member_snapshot` sets `state: "skipped"`, `code: "unsupported_kind"`; `UNROLLABLE_SKIP_CODES` | **Present** |
| `PlacementMode` includes `pool`, documented as reserved | `local_operator/session/placement.py`: `PlacementMode`, the comment above it ("`pool` is reserved and never produced in this pass"), `SessionPlacement` | **Present** |
| GitHub App tokens narrowed to repositories | `local_operator/network/credentials/github.py`: `REPOSITORIES_PATH`, `grant_narrowing`, `mint_repository_names`; a GitLab arm exists beside it at `credentials/gitlab.py` | **Present** |
| Session mobility and the move queue | `local_operator/network/mobility.py` (`MoveDevice`, `MovePhaseStamp`, `SessionMoveResult`, `move_bound_s`), `local_operator/network/move_queue.py`, `local_operator/network/sync.py` (`SyncRefused`, `promote_replica`) | **Present** |

### 1.2 What is NOT on main (confirmed by reading, not asserted)

* **`PoolRequest` exists nowhere in code.** No match in `local_operator/**`,
  `tests/**` or `scripts/**`. It is a *design* artifact only
  (`mesh-compute-pool.md` §4.1). So this document is free to define its shape, and
  must define the whole of it.
* **No producer of `placement.mode: "pool"`** — confirmed the literal appears only in
  the `PlacementMode` literal and `SessionPlacement.from_json`'s validator.
* **`MemberRecord` does not carry compute-pool §3.3's "seven additive keys".**
  `network/types.py`'s `MemberRecord` has no `ephemeral`, `size_class`, `provider`,
  `provider_ref`, `grant_id`, `expires_at`, `max_session_seconds` or `max_sessions`.
  These are `mesh-compute-pool.md` §3.3's delta, attributed there to
  `mesh-transport-identity.md` §4.2 — neither is built. This is consistent with
  "no pool producer", and it means the member-side half of this design is
  genuinely future work (§9.3).
* **No `local_operator/cloud/` module and no name collision.** The directory does not
  exist; the only near-misses in the tree are `local_operator/providers/qwencloud_console.py`
  and `local_operator/guides/qwencloud`. `local_operator/cloud/` is free.
* **No attested pool admission, no provisioning code, no metering.** Confirmed: the
  `network/` package has no `provision`, no `pool` op, and `local_operator/network/audit.py`'s
  `EVENT_KINDS` — a *closed* `frozenset` — carries no `pool_*` or `cloud_*` event kind.

### 1.3 Contradictions found while verifying (four here; §13 collects these and two more)

1. **`lop exec --peer` is NOT built.** The ask cites it as on main. It is not: the
   `exec` parser in `local_operator/cli.py` (built around `subparsers.add_parser("exec", …)`
   and its `exec_parser.add_argument` calls) has no `--peer`, `local_operator/exec_startup.py`
   has none, and no `--peer` exists outside the `sessions`/`network` verbs
   (`local_operator/network/cli.py`'s `net_sessions.add_argument("--peer", …)` with
   `--create`, and `cli.py`'s `sessions --peer`). It is *designed*
   (`mesh-session-mobility.md` §5.3) and *documented* — `tests/unit/guides/test_guides.py`
   asserts the string `"lop exec --peer"` appears in a guide body — but the flag is
   absent. **Consequence for this design:** §7 must not hang `lop exec --cloud` off
   `exec --peer`; remote session creation on main is
   `lop network sessions --create --peer <device>` only.
2. **`mesh-compute-pool.md` §4 names a module that does not exist.** It cites
   `SessionPlacement` at `local_operator/mesh/placement.py`; the real path is
   `local_operator/session/placement.py`. Worse for a reader who greps: there is a
   *different, unrelated* `placement` module at
   `local_operator/network/credentials/placement.py` (the credential *sharing list*,
   imported as `placement_mod` by `network/onboard.py`), so the wrong path is a
   plausible-looking hit. Doc-hygiene item, noted not fixed here.
3. **`remote-cloud-agents.md` §9.3's §9.2 step 1 list is incomplete as written** — it
   omits `executionRoleArn`/`taskRoleArn`, which it then corrects in the same bullet.
   The POC's `Pulumi.yaml` is authoritative and does carry both
   (`aws:iam/role:Role` for the execution role and an empty-policy task role). No
   action; the code is right.
4. **`boto3` is deliberately kept out of the default install set.** `pyproject.toml`'s
   `dev` extra carries `"boto3>=1.34"` with the comment that it is there *only* because
   the OSWorld adapter's tests need it, that the harness itself never imports it, and
   that keeping it out of `all` is deliberate: "an end user who never runs a benchmark
   should not install an AWS SDK". This is not a contradiction of the ask, but it **is**
   a constraint on §3.7: tier 1's SDK must arrive as an **optional extra**, and
   `require_extra` (`local_operator/optional.py`) is the route.

---

## 2. The problem, stated as the code will meet it

Two provisioning shapes, one lifecycle vocabulary, one set of verbs, no new
membership semantics yet:

* **Tier 1 — `fargate`.** One ECS task per run, ARM64, 2 vCPU / 4 GiB for size `m`,
  ephemeral, 2 h client deadline, results returned as a branch + a session directory +
  a bundle. Proven end-to-end by Slice 0 on a real key (`remote-cloud-agents-poc-results.md`,
  `ct_92161251`): RunTask→first `RUNNING` 24.8 s median, cost reconciled to −1.7% over
  two billed days, an empty task role, no key on disk.
* **Tier 2 — `k8s`.** A *dedicated workload* for a run that wants a persistent
  workspace and the option of spot pricing: `StatefulSet` (1 replica) + PVC, on a
  cluster with one static on-demand node pool and one Karpenter-managed spot pool.
  Long-lived enough that interruption is a normal event rather than a fault.

Both are asked for the same user-visible things: `run`, `status`, `cancel`, `list`;
a bounded budget; a hard maximum duration; and a durable record of what was asked for
and what happened. That is one Protocol and one record, twice implemented.

**Non-goals, stated once so §9 can be short:** no control plane, no metering/billing,
no attested admission, no blind relay, no egress proxy, no PR creation, no mobile/web
launch. All deferred with reasons in §9.

---

## 3. The seam: `ComputeProvider`, `PoolRequest`, `ComputeTask`, the ledger

### 3.1 Module layout

```
local_operator/cloud/__init__.py         # re-exports; no logic
local_operator/cloud/types.py            # Tier, SizeClass, Size, SIZES, PoolRequest, TaskState, ComputeTask
local_operator/cloud/provider.py         # ComputeProvider, ProviderCapacity, ComputeStatus, DrainOutcome, CloudRefused
local_operator/cloud/ledger.py           # CloudLedger: requests/ + tasks/, transitions, sweep, reconcile
local_operator/cloud/aws.py              # AwsCredentials, FargateClients, build_clients, verify_account
local_operator/cloud/fargate.py          # FargateProvider
local_operator/cloud/kube.py             # KubeRunner (kubectl) + KubeResult
local_operator/cloud/k8s.py              # K8sWorkloadProvider
local_operator/cloud/workload.py         # StatefulSet/PVC/ServiceAccount/StorageClass builders (plain dicts)
local_operator/cloud/fake.py             # FakeComputeProvider (tests and `--mock`)
local_operator/cloud/cli.py              # the `lop cloud` verbs' implementations
```

`local_operator/cloud/` is a *new leaf package*: it imports `local_operator.paths`
and (later) `local_operator.network.*`, and **nothing in the boot closure imports it**,
so it never widens `local_operator/cli.py`'s boot-closure barrier beyond the one
dispatch arm §7 adds.

### 3.2 `ComputeProvider` — the Protocol

`local_operator/cloud/provider.py`. A structural `Protocol` (not a base class), so
`FargateProvider`, `K8sWorkloadProvider` and `FakeComputeProvider` are interchangeable
and the ledger never learns which it holds. Every method is **async**, matching the
repo's existing provider seam — `benchmarks/osworld_v2_adapter/src/lop_osworld_v2_adapter/providers/base.py`'s
`EnvironmentProvider` (`"Every method is async because the AWS path is I/O-bound and the
fake must drive the same code path"`) — and every blocking call inside an
implementation runs under `asyncio.to_thread`, which is the discipline that adapter's
`providers/aws.py` module docstring states ("Every boto3 / HTTP call is blocking and runs
under `asyncio.to_thread` so the event loop stays free"). A `FakeComputeProvider` that
shares the code path is then the highest-value test, exactly as `providers/fake.py`
(`FakeProvider`) is for the OSWorld adapter.

```python
@runtime_checkable
class ComputeProvider(Protocol):
    """One provisioning backend. Implemented by FargateProvider, K8sWorkloadProvider, FakeComputeProvider."""

    name: str  # class attribute: "fargate" | "k8s" | "fake"

    async def describe(self) -> ProviderCapacity:
        """The capacity/describe call: what this provider can accept right now."""

    async def submit(self, request: PoolRequest) -> str:
        """Create the workload. The ONLY method that creates anything. Returns the provider_ref."""

    async def status(self, provider_ref: str) -> ComputeStatus:
        """One reading of the provider's own state. Never mutates it."""

    async def cancel(self, provider_ref: str, *, reason: str) -> None:
        """Stop it now, without waiting for a checkpoint."""

    async def drain(self, provider_ref: str, *, deadline_s: float) -> DrainOutcome:
        """Ask it to stop, wait up to deadline_s for a checkpoint, then stop it."""
```

Four records cross the seam, and each one is deliberately *not* the ledger's:

```python
@dataclass(frozen=True)
class ProviderCapacity:
    provider: str                 # "fargate" | "k8s"
    tier: Tier
    ready: bool
    detail: dict[str, str]        # fargate: {"cluster": ..., "task_definition": "<family>:<rev>"}
                                  # k8s:     {"cluster": ..., "static_nodes": "1", "spot_nodes": "0"}
    limits: dict[str, float]      # {"max_hours": 2.0, "max_concurrent": 8}
    checked_at: float

ProviderPhase = Literal["absent", "pending", "running", "stopped"]

@dataclass(frozen=True)
class ComputeStatus:
    provider: str
    provider_ref: str
    phase: ProviderPhase
    exit_code: int | None = None
    stop_reason: str = ""
    started_at: float | None = None
    stopped_at: float | None = None
    node: str = ""                # k8s only; the node the pod landed on ("" for fargate)
    detail: dict[str, str] = field(default_factory=dict)

@dataclass(frozen=True)
class DrainOutcome:
    drained: bool
    waited_s: float
    reason: str                   # "checkpointed" | "deadline" | "already_stopped" | "provider_refused"
    checkpoint: dict[str, str]    # {"branch": ..., "commit": ..., "session_id": ..., "artifacts": ...}

class CloudRefused(Exception):
    code: str       # not_enabled | budget_missing | size_unknown | tier_unknown | wrong_account
                    # provider_unavailable | provider_ref_unknown | drain_timeout | budget_exceeded
    sentence: str
```

`CloudRefused` follows the shape of `local_operator/network/types.py`'s `MeshRefusal`
(a `code` plus one human sentence), because §7 wants the same CLI refusal style
already used across the mesh verbs.

**Why the split matters.** `status()` reports what AWS or the cluster says and nothing
more; the *ledger* owns the state machine (§3.4). One writer per record is the rule
`move_queue.py` states for its own file ("its writers are the source RELAY's
transitions and `--cancel-queued` — never the requester's memory"), and a provider that
could also write task state would break it.

### 3.3 `PoolRequest` — the intent record

`local_operator/cloud/types.py`. This is `mesh-compute-pool.md` §4.1's `PoolRequest`
**plus** the cloud delta the operator asked for. Both are in one dataclass on purpose:
a second shape that agrees today is how two readers come to disagree, which is the
argument `SessionPlacement` makes for itself in `local_operator/session/placement.py`
("ONE dataclass serialised in both places rather than two shapes that agree today").

```python
Tier = Literal["fargate", "k8s"]
SizeClass = Literal["s", "m", "l"]
Hosting = Literal["hosted", "byo"]

@dataclass(frozen=True)
class Size:
    vcpu: float          # 1.0 / 2.0 / 4.0
    memory_gib: float    # 2.0 / 4.0 / 8.0

#: The size classes, which mesh-compute-pool §3.3 leaves opaque ('compute size class, opaque to
#: the mesh'). The ladder is the POC's `m` (2 vCPU / 4 GiB, the only class Slice 0 ran) with one
#: step either side. The PRICE of a class is a rate and lives in §11's cost table, not here.
SIZES: dict[SizeClass, Size] = {"s": Size(1.0, 2.0), "m": Size(2.0, 4.0), "l": Size(4.0, 8.0)}

@dataclass(frozen=True)
class PoolRequest:
    # --- mesh-compute-pool §4.1 keys, kept verbatim ---------------------------------
    version: int = 2                      # 1 was §4.1's; 2 adds the cloud delta (a superset, see below)
    request_id: str = ""                  # "req_<8 hex>"; the ledger filename stem
    network_id: str = ""                   # "" = unbound (this slice, §9.3)
    size: SizeClass = "m"
    max_hours: float = 2.0
    requested_at: float = 0.0
    requested_by: str = ""                 # device_id of the requester ("" when no network)
    # --- the cloud delta -----------------------------------------------------------
    tier: Tier = "fargate"
    budget_usd: float = 0.0                # model+compute cap; submit refuses 0
    spot_ok: bool = False
    repo_url: str = ""
    repo_sha: str = ""                     # "" = clone the default branch tip
    prompt: str = ""
    hosting: Hosting = "hosted"
    model: str = ""                        # provider model id; "" = profile default
    labels: dict[str, str] = field(default_factory=dict)
    max_sessions: int = 1                  # one member per task (remote-cloud-agents.md §3.3)
    include_local: bool = False            # upload a git bundle of unpushed commits (§1.1 of the product design)
    mock: bool = False                     # the POC's --mock, for a key-free end-to-end run
```

**Three deliberate departures from `mesh-compute-pool.md` §4.1, each recorded so the
convergence round can rule on them:**

1. `size_class` is spelled `size` and typed `SizeClass`. The operator's ask names the
   field `size`; the doc's `size_class` is prose. One spelling, the ask's.
2. **`state` and `grant_id` move to `ComputeTask`,** not the request. §4.1 puts both on
   the request because its reader is a remote control plane polling a directory; our
   request is *immutable intent* and the task is *mutable lifecycle*. Two mutable
   fields on an immutable record is how a record gets two writers (§3.2's rule).
3. `version: 2`, and `from_json` is **lenient**: an absent key is not a claim, so a
   `version: 1` file (the doc's shape) parses with the cloud delta defaulted, and an
   unknown future key is dropped rather than refused. That is the discipline
   `SessionPlacement.from_json` already applies ("Absent is not a claim").

`repo_url` + `repo_sha` are the repo mutation; `prompt`, `hosting`, `model` are the
run. `hosting`/`model` are not invented names: Slice 0's container contract already
carries `POC_HOSTING` and `POC_MODEL` (`scripts/remote_agents_poc.py`'s
`build_environment`), and §4.1 keeps those exact env names.

### 3.4 `ComputeTask` — the status record and its state machine

```python
TaskState = Literal[
    "queued", "provisioning", "running", "draining",   # live
    "expired", "failed", "cancelled", "lost",          # terminal
]
TERMINAL_STATES: frozenset[str] = frozenset({"expired", "failed", "cancelled", "lost"})

@dataclass
class ComputeTask:
    version: int = 1
    request_id: str = ""
    tier: Tier = "fargate"
    provider: str = ""                     # "" until submit
    provider_ref: str = ""                 # ECS task ARN, or "<namespace>/<statefulset>"
    state: TaskState = "queued"
    grant_id: str = ""                     # the pool grant, once there is one (absent this slice)
    size: SizeClass = "m"
    spot: bool = False                     # what we asked for
    spot_fallback: bool = False            # whether the static fallback fired (§5.4)
    created_at: float = 0.0
    submitted_at: float | None = None
    started_at: float | None = None
    stopped_at: float | None = None
    deadline_at: float | None = None       # created_at + max_hours*3600
    expires_at: float | None = None        # the member-side hard stop (mesh-compute-pool §3.3)
    last_seen_at: float | None = None
    exit_code: int | None = None
    stop_reason: str = ""                  # drained | lost | cancelled | failed | over_deadline
    budget_usd: float = 0.0
    repo_url: str = ""
    repo_sha: str = ""
    branch: str = ""                       # "lop/<request_id>"
    session_id: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)   # name -> local path or presigned URL
    checkpoint: dict[str, str] = field(default_factory=dict)  # what drain() returned
    events: list[dict[str, Any]] = field(default_factory=list)  # [{"at": float, "phase": str, "detail": {...}}]

    def transition(self, state: TaskState, *, detail: str = "") -> None: ...
    @property
    def member_state(self) -> MemberLifecycle | None: ...
```

**The state machine.** One writer (`CloudLedger.transition`), terminal states
**write-once** — the same rule `move_queue.py` states for `cancelled`/`failed`/`resumed`
("each win exactly once and no later transition may overwrite them, which is what makes
'cancel vs a relay transition' race-safe without either side re-reading a stale copy").

```
  queued ──submit()──► provisioning ──provider:running──► running ──drain()──► draining ──checkpoint──► expired
    │                      │                                │                      │
    │                      │                                │                      └── deadline, no checkpoint ─► expired (reason=drained_unconfirmed)
    └──cancel()────────────┴────────────────────────────────┴── provider:stopped + exit!=0 ─► failed
    └──cancel()──► cancelled          provider ref vanished with no stop record ─► lost
```

| From | To | Trigger (all observed, never inferred) |
|---|---|---|
| `queued` | `provisioning` | `submit()` returned a non-empty ref |
| `queued` | `cancelled` | the user's `lop cloud cancel` before submit |
| `queued`/`provisioning` | `failed` | no `running` within `PROVISION_DEADLINE_S` (default 600 s), or `submit`/`status` raised `provider_unavailable` |
| `provisioning` | `running` | `status().phase == "running"` |
| `running` | `draining` | `drain()` entered (an explicit cancel that checkpoints, an expiry, or an idle reap) |
| `draining` | `expired` | `DrainOutcome.drained` is true |
| `draining` | `expired` | deadline passed — still `expired`, with `reason: "drained_unconfirmed"`, because the provider *was* asked and the record must not read as success |
| `running`/`draining` | `failed` | stopped with a non-zero exit code |
| `running` | `cancelled` | `cancel()` while running (no checkpoint promised) |
| any live | `lost` | `status().phase == "absent"` with no stop record and no prior `stopped_at` |

**Mapping onto `MemberLifecycle`** — one function, `member_lifecycle_for(state)`,
because the mesh vocabulary has four states and this one has eight:

| `TaskState` | `MemberLifecycle` | Note |
|---|---|---|
| `queued` | `None` | no member row exists; §3.4 of mesh-compute-pool puts this in the *request's* state |
| `provisioning` | `"provisioning"` | a member row may exist and no session may be placed on it |
| `running` | `"active"` | |
| `draining` | `"draining"` | |
| `expired` | `"expired"` | |
| `failed`, `cancelled`, `lost` | `"expired"` if the task ever reached `running`, else `None` | the mesh's four-value lifecycle collapses these; the *reason* distinguishes them (`pool_member_expired {reason: drained\|lost\|revoked}` is already the design's vocabulary, mesh-compute-pool §3.4) |

A task that never reached `running` never produced a member row, so there is nothing to
expire — that is why the mapping is `None`, not `"expired"`.

### 3.5 Serialisation and JSON shape

Hand-written `to_json()` / `from_json()` on all three records. **Not `asdict`** — the
house pattern for a record that must survive a version skew is the lenient reader
(`SessionPlacement.from_json`, `MemberRecord.from_json`'s `_known`), and `asdict`
cannot be lenient. Rules, stated so two slices cannot diverge:

* keys are `snake_case`, exactly the dataclass field names;
* absent → the field's default, never an error; a key of the wrong type is coerced by
  the same reader that owns the field (`str(...)`, `_as_int`, `float(...)`), matching
  `SessionPlacement.from_json`;
* `version` is bumped only for a *shape change an older reader could misread*, and a
  reader that does not know a newer `version` treats the record as unusable and refuses
  by name rather than guessing — the rule `local_operator/session/placement.py` states
  for its stamp ("A reader that does not know a version treats the stamp as absent
  (fail closed to 'unplaced', never to a guessed owner)");
* no secret ever appears: no token, no key, no ARN-with-credentials. `AwsCredentials`
  must follow `providers/aws.py`'s `AwsCredentials.__repr__` discipline (never logged).

### 3.6 Where state is persisted — the ledger, and the pattern it reuses

```
<config>/cloud/requests/<request_id>.json   # the intent, written once, 0600
<config>/cloud/tasks/<request_id>.json      # the ledger task, mutable, 0600
```

`<config>` is `local_operator/paths.py`: `config_dir()` — read from the environment on
every call, so an isolated run is isolated (`LOCAL_OPERATOR_CONFIG_DIR`), which is what
AGENTS.md's isolation rule needs and what every existing store relies on.

**The pattern reused, named exactly: `local_operator/network/move_queue.py`.** Not a
new store discipline — the same one, for the same reasons:

| move_queue element | what the ledger does |
|---|---|
| `<config>/network/queue/move-<session_id>.json`, one small JSON per record, "beside the mesh store's other durable objects" (`queue_dir`, `record_path`) | `<config>/cloud/tasks/<request_id>.json` (`LEDGER_DIRNAME = "cloud"`, `tasks_dir()`, `task_path()`) |
| temp-file + `os.replace` + 0600, cross-process `flock` through `WakeWriteLock` (`lock_for`, `_write_record_raw`) | same, through `local_operator/wakes/lock.py`'s `WakeWriteLock` — the bundled implementation move_queue already cites |
| terminal states write-once (`QUEUE_TERMINAL_PHASES`) | `TERMINAL_STATES`, enforced in `transition()` |
| `transition()` stamps a phase + detail | `ComputeTask.transition()` appends `{"at", "phase", "detail"}` to `events` |
| `sweep_terminal(max_age_s=QUEUE_TERMINAL_TTL_S)` | `sweep_terminal(max_age_s=TASK_TTL_S)` on `list` |
| `reconcile_on_start(server)` restarts drivers for live records after a crash | `reconcile()` runs on `lop cloud status`/`list`: for a live record, re-read `status()` and drive the transition; refuse to guess when the provider is unreachable (the record stays live and says so) |
| `enqueue` / `cancel` / `payload` public verbs | `CloudLedger.create` / `cancel` / `payload` |

The stronger, more general store — `local_operator/network/store.py`'s
`_write_private_json` / `_write_lock` / `mutate` — is named as the *convention of
record* (0600, staged write, `os.replace`), but the ledger does **not** import its
privates: a cross-package import of a `_`-prefixed helper is a coupling nothing
tests. `move_queue.py` is the closest *public* shape and it is the one copied.

Bounds, mirroring §4.1's "bounded to the last MAX_POOL_REQUESTS (recommend 32)":
`MAX_CLOUD_TASKS = 64` records retained, `TASK_TTL_S = 30 * 86400` for terminal ones,
pruned on `list`. A request and its task share the `request_id`, so a prune takes both.

### 3.7 The dependency seam: an optional extra, not a new hard dependency

Tier 1 needs `boto3`. It must **not** become a base dependency: `pyproject.toml`'s
`dev` extra carries `boto3>=1.34` today with an explicit note that the harness never
imports it and that keeping it out of `all` is deliberate ("an end user who never runs
a benchmark should not install an AWS SDK"). So:

* `pyproject.toml`'s `[project.optional-dependencies]` gains **`cloud = ["boto3>=1.34"]`**;
* `all` does **not** gain it, for the reason above;
* `local_operator/optional.py`'s `EXTRAS` gains
  `"cloud": "cloud compute provisioning on ECS Fargate (boto3)"`, and the import site
  calls `require_extra("boto3", "cloud", "Cloud task provisioning")`;
* `tests/unit/test_optional.py` enforces the two-way sync between `EXTRAS` and
  pyproject (`_declared_extras`), and carries a per-extra parametrised case — so this
  is three touch points, all named, and CI fails loudly if one is missed.

**Tier 2 adds no Python dependency at all** (§5.1), so `cloud` stays boto3-only.

---

## 4. Tier 1 — `FargateProvider`

### 4.1 The extraction map, symbol by symbol

`scripts/remote_agents_poc.py` is the source. It stays in `scripts/` unchanged as the
POC's own evidence tooling (its `report`/`verify`/`selftest` half is not product code
and never will be). What is *lifted* is the run/status/stop logic:

| POC symbol | becomes | note |
|---|---|---|
| `assert_expected_account(session)` | `local_operator/cloud/aws.py`: `verify_account(clients, expected) -> str` | called **first, before any mutating call**, by `FargateProvider.submit()` through a cached `_ensure_account()`; refusal code `wrong_account`. The POC's rule ("Runs FIRST, before a single mutating call, because every hard rule about this sandbox starts with 'the account is 325492156725'") is a property of the *provider*, not of a script, or a second caller can skip it |
| `assume_controller(operator, role_arn)` | `aws.py`: `assume_role(clients, role_arn, *, session_name, duration_s)` | returns an `AwsCredentials`; the CLI does this once, not the provider |
| `_driver_session(args)` | `cli.py`'s `cloud run`/`status`/`cancel`: build credentials → provider | the CLI owns session/profile resolution; the provider never reads `args` |
| `read_stack_outputs` / `load_outputs` | `cli.py`: `read_stack_outputs(infra_dir, backend_url)` (unchanged, moved) | stack outputs are the CLI's business; the provider takes `cluster` + `task_definition` |
| `active_task_definition(operator, outputs)` | `FargateProvider.describe()` | resolves and requires `ACTIVE`; a stale ARN is refused by name (the POC's reason: "a `pulumi up` that changes the task definition REPLACES it") |
| `run_once(...)`'s RunTask block | `FargateProvider.submit(request)` | `launchType=FARGATE`, `platformVersion=LATEST`, `count=1`, `enableECSManagedTags=True`, `propagateTags="TASK_DEFINITION"`, tags `{**POC_TAGS, "run-id": request.request_id}` |
| `build_environment(outputs, run_id, url, sha, prompt, probes_url, results_url, hosting, model, mock)` | `local_operator/cloud/workload.py`: `task_environment(request, *, probes_url, results_url)` | **the env names are kept verbatim** (`POC_RUN_ID`, `POC_REPO_URL`, `POC_SHA`, `POC_PROMPT`, `POC_PROBES_URL`, `POC_RESULTS_URL`, `POC_MODEL_SECRET_ARN`, `POC_MOCK`, `POC_HOSTING`, `POC_MODEL`), because `infra/remote-agents-poc/image/entrypoint.sh` and `image/lop_launch.py` read them; renaming is a second change in the image, and this slice does not need it |
| `OVERRIDES_LIMIT_BYTES` check | same check in `submit()` | ECS rejects >8 KiB with a 400 naming no field (the POC's measured reason); keep the 8192 assertion and its refusal |
| `presign_put` / `s3_client` | `aws.py`: `CloudBlobs` (injectable: `presign_put`, `download_tree`) | result hand-off is not the provider's, but it is on the same injected seam |
| `poll_to_stopped(ecs, outputs, task_arn)` | split: `status()` is ONE `describe_tasks`; `drain()`/`run`-loop owns the 1 s grid and `RUN_DEADLINE_SECONDS` | the 1 s poll grid is a *client* concern; a `status()` that blocks for two hours is not a status call |
| `describe_tasks`' "not describable for the first fractions of a second" | `status()` returns `phase: "pending"` with `detail={"reason": "not_yet_describable"}` | the POC's `sleep`-and-continue becomes an honest reading, not a hidden wait |
| `cmd_status`'s `list_tasks` | `describe()`'s `detail["active_tasks"]` | the tag inventory (`resourcegroupstaggingapi`) stays out of the provider: the POC notes the controller role deliberately cannot make that call, so it is an operator-identity tool, and `lop cloud list` reads the ledger instead |
| `cmd_stop_all` | `cancel(provider_ref, reason=...)` | `ecs.stop_task`; `stop_all` semantics are expressed by the CLI iterating the ledger's live records, not by a cluster-wide sweep |
| `cold_start_numbers(...)` | `ComputeTask.events` entries (`phase: "cold_start"`) | diagnostics, not a required field |

### 4.2 The shapes

```python
@dataclass(frozen=True)
class FargateClients:
    """The injectable I/O surface, so tests use botocore's Stubber."""
    ecs: Any                                   # boto3 ecs client
    sts: Any                                   # boto3 sts client (the account assertion)
    s3: Any                                    # boto3 s3 client (presigning)
    # build_clients(credentials, region) -> FargateClients  constructs the real three.
```

`FargateClients` is `providers/aws.py`'s `_Clients` shape by name (that dataclass is
literally documented as "The injectable I/O surface, so tests use botocore's Stubber"),
and the tests are the same kind: `tests/unit/evaluation/adapters/osworld/test_aws_provider.py`
drives `AwsProvider` against `botocore.stub.Stubber`. **`FargateProvider` is unit-testable
with no AWS account, no network and no `enableEcs` anything** — which is the acceptance
criterion for slice 1 (§10).

```python
class FargateProvider:
    name = "fargate"

    def __init__(
        self,
        credentials: AwsCredentials,
        *,
        region: str,
        cluster: str,
        task_definition: str,
        expected_account: str = EXPECTED_POC_ACCOUNT,   # "325492156725"
        run_deadline_s: float = RUN_DEADLINE_SECONDS,   # 2 h + 10 min, the POC's bound
        poll_interval_s: float = DESCRIBE_POLL_SECONDS, # 1.0
        clients: FargateClients | None = None,
        sleep: Callable[[float], None] = time.sleep,    # the `AwsProvider._sleep` seam
    ) -> None: ...
```

* `queue_arn`/`log_group`/`subnets`/`security_group` come from the **stack outputs**,
  passed in as a frozen `FargateTargets` record built by the CLI from
  `pulumi stack output --json`. The provider never shells out to `pulumi`; the CLI does.
* Every `ecs`/`sts`/`s3` call runs under `asyncio.to_thread`. The test for that is the
  house pattern — AGENTS.md's "put the synchronous call back on the loop" recipe
  (a probe that asserts the loop is not blocked, then the change reverted).
* `drain()` semantics on Fargate are honest and thin: the ECS agent sends SIGTERM to
  PID 1 and SIGKILLs after the task definition's `stopTimeout`. So `drain()` waits for
  the container's own checkpoint marker (the results object appearing at the presigned
  URL) up to `deadline_s`, then `stop_task(reason="lop drain deadline")` — the POC's
  `poll_to_stopped` overdue path, which is "a stop, not a surrender". `stopTimeout`
  rises from the default 30 s to 120 s in the task definition (§6.3), because a drain
  that is killed at 30 s cannot flush a bundle.
* `describe()` returns `ready=False` with `detail["reason"] = "task_definition_inactive"`
  or `"cluster_not_found"` rather than raising: `lop cloud status` must be able to *say*
  the stack is gone.

---

## 5. Tier 2 — `K8sWorkloadProvider`

### 5.1 The client decision: `kubectl` subprocess, behind an injected runner

**Decision: talk to the cluster by invoking `kubectl` as a subprocess, behind an
injected `KubeRunner` seam. Do not add the `kubernetes` Python client in this slice.**

Justification, in the order that settles it:

1. **Dependency weight.** The `kubernetes` client is not one package: it pulls
   `urllib3`, `websocket-client`, `python-dateutil`, `six`, `pyyaml`, `requests-oauthlib`,
   `certifi` and (for auth) `google-auth` — ~10 distributions, one of which
   (`google-auth`) is a whole auth stack. `pyproject.toml`'s existing posture is
   explicit about exactly this class of cost: the harness already carries `pyyaml` but
   deliberately does *not* carry `boto3`, and `local_operator/optional.py` documents the
   rule ("Everything with a heavy or platform-fragile install footprint … sits behind a
   named extra so that a plain install stays small and fast"). `kubectl` costs **zero**
   new Python distributions, and it is already a binary this tree knows about: it is
   allow-listed by subcommand in `local_operator/monitors/readonly.py` (via
   `_kubectl_reason`, "every flag that retargets the cluster, the identity or an
   arbitrary API path is denied by name") and classified as an infra command in
   `local_operator/tools/query_budget.py`.
2. **The workload shape is static.** A `StatefulSet` + `PVC` + `ServiceAccount` is one
   declarative manifest created once and read back. There is no watch, no informer, no
   dynamic discovery, no client-side apply logic — i.e. none of what the typed client is
   *for*.
3. **Everything the provider needs is a stable `kubectl -o json` shape.**
   `get sts <name> -o json`, `get pod -l app=<name> -o json` (phase, `nodeName`,
   container statuses, `reason: FailedScheduling`), `logs`, `patch`, `delete`,
   `apply -f -`. All of it is the same JSON `kubectl` emits today and has emitted for
   years.
4. **Testability is better, not worse.** The seam is a `KubeRunner` that returns
   `KubeResult(returncode, stdout, stderr)`; a fake runner replays recorded JSON and
   asserts argv. That is stronger than mocking a typed client, because it bounds exactly
   what the provider may ask for — and it is what makes the scheduling contract (§5.3)
   a pure unit test.
5. **argv, never a shell.** `asyncio.create_subprocess_exec(*argv)` with a list — no
   `shell=True`, no string interpolation — so there is no quoting hazard on a
   repo URL, a branch name or a prompt. Prompts do not reach kubectl at all: they reach
   the pod as a `Secret` (§5.2), never as argv (argv is world-readable in `/proc` on the
   node, which is the same class of leak as the environment one the POC's SEC-1 closed).

**The honest costs of this choice**, so a reviewer can price them:

* the operator's host needs `kubectl` and a kubeconfig for the tier-2 stack
  (`aws eks update-kubeconfig`). If it is missing, `describe()` refuses with
  `provider_unavailable`, `detail={"binary": "kubectl", "hint": "aws eks update-kubeconfig --name …"}` — a named refusal, not a `FileNotFoundError`;
* channel/attribute errors surface as `stderr` text we must classify rather than as a
  typed exception, so the provider classifies on `returncode` and a small named set of
  strings, and anything unrecognised becomes `provider_unavailable` with the stderr
  tail in `detail` (never swallowed);
* **reversibility:** the seam is a Protocol, so a `PythonClientRunner` can be added
  later — behind the same `KubeRunner` — without touching `K8sWorkloadProvider`, the
  ledger, or the CLI. That is the property that makes this the smallest change rather
  than a corner.

### 5.2 The workload shape

One `StatefulSet` per task, in **one shared namespace `lop-tasks`**:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: lop-task-<request_id>
  namespace: lop-tasks
  labels: {app: lop-task, lop.dev/task: <request_id>, lop.dev/tier: k8s, lop.dev/owner: lopdev}
spec:
  serviceName: lop-task-<request_id>
  replicas: 1
  persistentVolumeClaimRetentionPolicy: {whenDeleted: Retain, whenScaled: Retain}
  template:
    spec:
      serviceAccountName: lop-task          # no cloud permissions at all (§8)
      automountServiceAccountToken: false   # ...and no token to spend
      terminationGracePeriodSeconds: 120    # the drain window, §5.5
      securityContext: {runAsNonRoot: true, runAsUser: 10001, fsGroup: 10001, seccompProfile: {type: RuntimeDefault}}
      nodeSelector: {lop.dev/pool: static}  # spot_ok=False; §5.3 for spot_ok=True
      containers:
        - name: agent
          image: <ECR repo>@sha256:<digest>     # pinned by DIGEST, never a tag (POC rule)
          envFrom: [{secretRef: {name: lop-task-<request_id>}}]
          resources: {requests: {cpu: 2, memory: 4Gi}, limits: {cpu: 2, memory: 4Gi}}
          volumeMounts: [{name: workspace, mountPath: /workspace}]
          securityContext: {allowPrivilegeEscalation: false, readOnlyRootFilesystem: true, capabilities: {drop: [ALL]}}
  volumeClaimTemplates:
    - metadata: {name: workspace}
      spec: {accessModes: [ReadWriteOnce], storageClassName: lop-gp3, resources: {requests: {storage: 30Gi}}}
```

* **StatefulSet, not Deployment** — because the workspace is a PVC identity: a
  `StatefulSet`'s `volumeClaimTemplates` binds exactly one PVC per replica and keeps its
  name stable across a restart, which is what makes "the pod came back and found its
  workspace" true without any code. A `Deployment` would need the PVC created and
  referenced separately, for no gain.
* **One namespace, not one per task.** The product design already settles the principle
  for the per-task substrate: "A namespace is not a kernel boundary"
  (`remote-cloud-agents.md` §4). Per-task namespaces would add RBAC, quota and
  `ServiceAccount` surface for zero isolation gain. The isolation boundary here is the
  pod + the empty node instance role + the taint.
* **The prompt and repo go in a `Secret`,** mounted as env (`envFrom`), created
  alongside the workload and deleted with it: an `envFrom` secret keeps the prompt out
  of the manifest, out of `kubectl apply`'s argv, and out of the pod spec that anyone
  with `get sts` can read. (`Secret` data is base64 in etcd — the same posture as the
  POC's ECS secret, whose whole point is "in memory, not on disk"; the pod must
  `unset` it after the launcher reads it, exactly as `entrypoint.sh` already does.)
* **`PVC` + `serviceAccount`** are created by the same `apply` from `workload.py`'s
  builders, so the manifest set is one function call and one test.
* **`storageClassName: lop-gp3`** is a cluster-scoped `StorageClass` created once by the
  IaC programme's phase 2 (§6.3): `ebs.csi.aws.com`, `gp3`, `volumeBindingMode: WaitForFirstConsumer`
  (mandatory with node-local volumes), `allowVolumeExpansion: true`, **not** the default
  class (the cluster's default must stay unset so nothing else silently claims gp3
  volumes).
* The EBS CSI driver itself is an EKS managed addon (§6.2) — without it the PVC never
  binds, and that is a hard prerequisite of this tier, not a nicety.

### 5.3 The two-pool scheduling contract (the decision the ask names)

Node labels and taints, fixed here and asserted by a unit test:

| Pool | Node label | Taint | Managed by |
|---|---|---|---|
| **static** (on-demand) | `lop.dev/pool=static` | none | EKS managed node group (ASG) |
| **spot** | `lop.dev/pool=spot`, `karpenter.sh/capacity-type=spot` | `lop.dev/spot=true:NoSchedule` | Karpenter `NodePool` (`capacity-type: spot`) |

The spot taint is **load-bearing beyond payloads**: it is what keeps `kube-system`
(CoreDNS, the EBS CSI controller, the Karpenter controller itself) off interruptible
capacity. A cluster whose DNS dies because a spot node was reclaimed is a cluster that
looks like it lost a task and actually lost its name resolution.

| `spot_ok` | tolerations | nodeAffinity | result |
|---|---|---|---|
| `false` | none | `required: lop.dev/pool In [static]` | cannot schedule on spot at all (affinity *and* the taint); capacity grows through the managed node group |
| `true` | `lop.dev/spot=true:NoSchedule` | `required: lop.dev/pool In [spot]` | only spot; the static pool is not a silent fallback — the fallback is explicit and timed (§5.4) |

**Why `required` for `spot_ok=True` rather than `preferred`.** A soft
`preferredDuringSchedulingIgnoredDuringExecution` preference applies only *among feasible
nodes*, so a spot-wanting pod lands on a static node whenever static has room — a
"prefer spot" that silently never provisions spot capacity. `required` makes the
preference real, and moves the fallback question to where it can be answered honestly
and tested: the provider, with a clock (§5.4). The alternative — Karpenter's own
fallback, a second on-demand `NodePool` at lower `weight` — is the phase-2 answer if many
tasks must fall back simultaneously; it is deliberately *not* built now, because it is a
second always-on provisioner whose cost is a permanent extra node-class contract for one
behaviour we can get from one patch (§5.4, and §12 R2).

### 5.4 Spot fallback, and spot interruption

**Fallback (capacity never arrives):** `K8sWorkloadProvider.status()` reads the pod's
conditions. If the pod has been `Pending` with the `PodScheduled=False` /
`reason: FailedScheduling` condition continuously for `SPOT_FALLBACK_S = 120.0`, `status()`
patches the `StatefulSet` to the `spot_ok=False` shape (drop the toleration, set
`nodeSelector: {lop.dev/pool: static}`), sets `ComputeTask.spot_fallback = True`, and
appends a `spot_fallback` event. The workload has not started, so there is no workspace
to lose and nothing to checkpoint. The patch is a manifest re-derivation through
`workload.statefulset(request, spot=False)` — one code path, two manifests, no
hand-edited YAML. Acceptance: a unit test drives a fake runner whose `get pod` returns
`Pending/FailedScheduling` with a controlled clock, and asserts the patch argv plus
`spot_fallback = True`.

**Interruption (capacity arrives, then is reclaimed):** Karpenter's interruption
handling consumes the SQS queue the IaC creates (§6.2) and, for a spot interruption
notice, cordons and drains the node. The pod then gets SIGTERM and has
`terminationGracePeriodSeconds: 120` to act. What the container does on SIGTERM:

1. refuse new turns (the runtime's own admission switch — the "no new turns are
   admitted" arm of `draining` in mesh-compute-pool §3.4);
2. finish or cut the in-flight turn per the existing session-stop semantics;
3. **checkpoint**: `git commit` to `lop/<request_id>` and upload the session directory
   plus `lop exec --status <id> --json`, **exactly what `infra/remote-agents-poc/image/entrypoint.sh`
   already does at its own exit** (`put_file` to the presigned URLs, with the "a redirect
   is not a successful upload" check);
4. exit 0.

**`checkpoint`, stated precisely, and what is deliberately NOT in it.** In this slice
checkpoint means *the branch commit plus the uploaded session directory and status JSON* —
the artifacts Slice 0 already produces and the results doc already verifies. It does
**not** mean resuming a partially-complete task on a replacement node: the PVC is
`ReadWriteOnce` and AZ-bound (an EBS volume attaches only in its own AZ), and Karpenter
is free to replace the node in a different AZ, so a workspace-level resume is not
promised by this design and must not be claimed in the UI. **Phase 2** options, named so
they are not re-derived: `volumeBindingMode` + a single-AZ spot `NodePool`, an EFS-backed
`ReadWriteMany` workspace, or a workspace snapshot to S3 on SIGTERM. The mesh-level
session sync (`local_operator/network/sync.py`'s replica/cursor machinery, and
`promote_replica`'s deliberately-different-id rule) is the *right* mechanism for that
day, and it is **not** needed for the artifacts-only checkpoint this slice ships.

### 5.5 `drain()` on tier 2

1. annotate/patch the `StatefulSet` to stop admitting (`lop.dev/draining: "true"`, and
   `replicas: 1` unchanged — the workload stops itself);
2. wait for the pod's `terminationGracePeriodSeconds` window to run its checkpoint, up to
   `deadline_s` (default `DRAIN_DEADLINE_S = 150.0`, i.e. the 120 s grace plus margin),
   by polling for the checkpoint artifacts and `get pod` reaching `Succeeded`/`Failed`;
3. if the deadline passes: `delete sts <name> --wait=true` (and the `Secret`), record
   `reason: "deadline"`, `drained: False`. The workspace PVC is retained
   (`persistentVolumeClaimRetentionPolicy: Retain`) so a human can still get the
   branch out of it — and `cancel --keep-workspace` keeps it deliberately.

`cancel()` is the same minus the wait: `delete sts`, delete the `Secret`, then delete the
PVC unless `--keep-workspace`. It is the `running → cancelled` edge of §3.4, i.e. no
checkpoint is promised.

---

## 6. IaC for tier 2

### 6.1 Where it lives, and why it is a separate program

`infra/cloud-tiers/Pulumi.yaml` — a **new program with its own stack** (`lop-tiers`),
not a flag in `infra/remote-agents-poc/Pulumi.yaml`. Three reasons, in order of weight:

1. the POC stack is a *different environment with a different lifetime*, and its teardown
   is still awaiting approval (`remote-cloud-agents-poc-results.md` § Test 7); a shared
   stack means a `pulumi destroy` intended for one takes the other;
2. the POC VPC is `10.77.0.0/24` — public subnets, no NAT, sized for one Fargate task
   invocation. EKS needs ≥2 AZs with private subnets, NAT for image pulls, and EKS-role
   subnet tags. That is a new VPC, so it is a new stack;
3. every other property of the style (below) is identical, so a reader who knows one
   file knows the other.

The style is copied deliberately from `infra/remote-agents-poc/Pulumi.yaml`, including
its own stated rationale ("the whole stack is a flat list of resources with no control
flow, so a typed program would add a second language and a build step for nothing"):

* `runtime: yaml`, `name: cloud-tiers`;
* **the account and region are never in the file** — the provider takes them from the
  environment (`AWS_PROFILE`, `AWS_REGION`), and `variables:` resolves
  `aws:index/getCallerIdentity:getCallerIdentity` once so the deployed state names the
  account it landed in (the POC's own defence: a string config holding a 12-digit id
  reaches an interpolation as a float — `325492156725` → `3.25492156725e+11` — which
  `preview` accepts and create rejects);
* **one `aws` provider resource with `defaultTags: {lop-poc: "true", owner: lopdev}`**
  and `options: {provider: ${aws}}` on every resource, because the tags are what
  `resourcegroupstaggingapi` proves teardown with, and a budget filter reads them;
* typed `config:` with defaults, `outputs:` at the end.

### 6.2 Resources

The AWS-shaped half (Pulumi YAML, phase 1):

```
aws:ec2/vpc:Vpc                     10.78.0.0/16, DNS support + hostnames
aws:ec2/subnet:Subnet               × 3 private (one per AZ) with kubernetes.io/role/internal-elb=1
aws:ec2/subnet:Subnet               × 3 public  (NAT only)   with kubernetes.io/role/elb=1
aws:ec2/natGateway:NatGateway       × 1  + aws:ec2/eip:Eip     ($36.50/mo, §11)
aws:ec2/routeTable:RouteTable + aws:ec2/route:Route (0.0.0.0/0 → NAT) per private subnet
aws:ec2/securityGroup:SecurityGroup cluster SG (control-plane ↔ node 443/10250, self only)
aws:eks/cluster:Cluster             kubernetesVersion: the latest EKS offers in ca-central-1
                                    at implementation time (pin it then; do NOT copy a number from
                                    this doc — EKS retires versions on a rolling schedule), private
                                    endpoint access ENABLED,
                                    public access ENABLED but restricted to
                                    clusterPublicAccessCidrs — a config key with NO default, so an
                                    unconfigured stack fails at preview instead of exposing an API
                                    server to 0.0.0.0/0 (the operator sets their current egress CIDR,
                                    and `aws eks update-kubeconfig` is then the whole access story)
aws:eks/managedNodeGroup:ManagedNodeGroup
                                    name lop-tiers-static, capacityType ON_DEMAND,
                                    amiType AL2023_ARM_64_STANDARD, instanceTypes [m7g.xlarge],
                                    scalingConfig {minSize: 1, desiredSize: 1, maxSize: 3},
                                    labels {lop.dev/pool: "static"}, taints []        ← THE STATIC POOL
aws:eks/addon:Addon                 aws-ebs-csi-driver  (PVCs do not bind without it)
aws:eks/addon:Addon                 coredns / kube-proxy / vpc-cni (resolve_conflicts_on_update: OVERWRITE)
aws:eks/accessEntry:AccessEntry     EC2_LINUX for the node role; the operator's principal as ADMIN
aws:iam/role:Role + rolePolicy      Karpenter controller IRSA role (the upstream policy, scoped to
                                    the cluster's name/id tags)
aws:eks/podIdentityAssociation:PodIdentityAssociation
                                    karpenter → the IRSA role (Pod Identity, not the aws-auth ConfigMap)
aws:sqs/queue:Queue                 lop-tiers-interruption  (message retention 300 s)
aws:sqs/queuePolicy:QueuePolicy     allow events.amazonaws.com to SendMessage
aws:cloudwatch/eventRule:EventRule  × 4 sources (EC2 Spot Interruption Warning, EC2 Instance Rebalance
                                    Recommendation, EC2 Instance State-change Notification, AWS Health)
                                    + aws:cloudwatch/eventTarget:EventTarget → the SQS queue
kubernetes:helm.sh/v3:Release       karpenter-provider-aws (chart), crds.install: true,
                                    settings: {clusterName, interruptionQueue: lop-tiers-interruption},
                                    and its own nodeSelector {lop.dev/pool: static}
```

The cluster-shaped half (phase 2, applied by the driver — §6.4): namespace `lop-tasks`,
`ServiceAccount` `lop-task`, `StorageClass` `lop-gp3`, and Karpenter's
`EC2NodeClass` + `NodePool`.

The two Karpenter CRs, in the Karpenter 1.x API (`karpenter.sh/v1` for `NodePool`,
`karpenter.k8s.aws/v1` for `EC2NodeClass` — the v1 GA schema; `apiVersion` verified
against Karpenter's v1 migration guide, and the chart's interruption setting is
`settings.interruptionQueue`, which "disables interruption handling if not specified"):

```yaml
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata: {name: lop-tiers-spot}
spec:
  amiSelectorTerms: [{alias: "al2023@latest"}]
  role: <the node role from phase 1>
  subnetSelectorTerms: [{tags: {"karpenter.sh/discovery": "lop-tiers"}}]
  securityGroupSelectorTerms: [{tags: {"karpenter.sh/discovery": "lop-tiers"}}]
  metadataOptions: {httpTokens: required, httpPutResponseHopLimit: 1}   # IMDSv2 only
  blockDeviceMappings:
    - {deviceName: /dev/xvda, ebs: {volumeSize: 30Gi, volumeType: gp3, encrypted: true, deleteOnTermination: true}}
  tags: {lop-poc: "true", owner: lopdev}
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata: {name: lop-tiers-spot}
spec:
  weight: 50
  template:
    metadata:
      labels: {lop.dev/pool: spot}
    spec:
      nodeClassRef: {group: karpenter.k8s.aws, kind: EC2NodeClass, name: lop-tiers-spot}
      taints: [{key: lop.dev/spot, value: "true", effect: NoSchedule}]
      terminationGracePeriod: 24h
      expireAfter: 168h
      requirements:
        - {key: karpenter.sh/capacity-type, operator: In, values: [spot]}
        - {key: kubernetes.io/arch, operator: In, values: [arm64]}
        - {key: kubernetes.io/os, operator: In, values: [linux]}
        - {key: karpenter.k8s.aws/instance-family, operator: In, values: [m7g, m6g, m8g, c7g, c6g, c8g, r7g, r6g]}
        - {key: karpenter.k8s.aws/instance-size, operator: In, values: [large, xlarge, 2xlarge]}
  disruption: {consolidationPolicy: WhenEmpty, consolidateAfter: 60s, budgets: [{nodes: "10%"}]}
  limits: {cpu: "32", memory: 128Gi}
```

**Static pool: managed node group — decided, with the reason.** The alternative is a
second Karpenter `NodePool` with `capacity-type: on-demand`, and it buys exactly one
thing: one provisioner instead of two. It costs three: (a) the static pool — which
carries `kube-system`, the EBS CSI controller and the Karpenter controller itself —
would then depend on Karpenter's admission webhook and its own provider being healthy,
i.e. the always-on pool would depend on the component whose failure it exists to
tolerate; (b) the node group's `minSize`/`desiredSize`/`maxSize` is a declarative,
reviewable number in the stack's config, whereas a Karpenter on-demand pool's floor is a
`limits` block plus a NodeClass assumption; (c) Karpenter's `consolidationPolicy` would
have to be constrained per-pool to keep it from consolidating the static floor away,
which is a second behaviour to test for no user-visible gain. The managed node group also
inherits the EKS-managed AL2023 AMI lifecycle, so nobody owns an AMI contract here.

**Instance families, and why those.** `m7g`/`m6g`/`m8g` (4 GiB per vCPU) are the general
shape; `c7g`/`c6g`/`c8g` (2 GiB per vCPU) are for CPU-bound runs; `r7g`/`r6g` for memory
headroom. All arm64, all `large`–`2xlarge`, i.e. 2–8 vCPU — deliberately no `metal` or
GPU families, because a diversified pool that can hand back a $3/hour node for a $0.09
task is a cost guard failing open. Spot diversity in ≥8 families across ≥3 AZs is what
makes "spot capacity was unavailable" rare enough for §5.4's 120 s fallback to be a
backstop rather than the common path.

### 6.3 The pieces Pulumi YAML cannot express cleanly — and the fallbacks

1. **Conditionals. Pulumi YAML has none.** If/else and count-style conditionals exist
   only in general-purpose languages (Pulumi's own docs place conditionals in "language
   essentials" for the general-purpose SDKs; there is a long-open request for
   `If`/`For` combinators in the core). This is the same fact
   `infra/remote-agents-poc/Pulumi.yaml` states in its own header from the other
   direction ("The one thing a real language would buy — conditionals — is exactly what
   the two-phase deploy below does NOT need"). **Consequences, all accepted:**
   * the `enableEks` gate lives where branching exists: in the *driver* (§6.4), which
     reads `pulumi config get enableEks` and refuses to run the apply when it is false.
     The program still declares the key (`config: {enableEks: {type: boolean, default: false}}`)
     and echoes it in `outputs:`, so the value is in the stack's own state and a
     `pulumi stack output enableEks` answers "was this stack meant to be on?";
   * the second key, `clusterPublicAccessCidrs`, is declared with **no default**, which is
     the one gate Pulumi YAML can enforce by itself: an unconfigured stack cannot preview,
     so "nothing deploys by default" is true even for someone who sets `enableEks: true`;
   * **the structural gate is stronger than the flag** and is the real answer to
     "nothing deploys by default": a separate program + a separate stack, not referenced
     by any script, `Makefile` target or CI job, means no existing path can reach it even
     with the flag mistyped.
2. **Applying a custom resource in the same update as its CRD is a known ordering hazard**
   in `pulumi-kubernetes` (upstream issue #1446: "creation failed: no matches for kind
   …; Retry", and #1735 for the same failure from a Helm chart's CRDs). **Fallback:
   the CRs are applied by the driver, after the stack, with `kubectl apply -f`** — which
   is also where the gate's branch lives and where the `kubectl` binary is already a
   prerequisite (§5.1). The Helm release itself installs the Karpenter CRDs
   (`crds.install: true`), which is fine: a *chart* gets the retry behaviour, a
   hand-declared CR does not.
3. **No loops over a computed list.** The four EventBridge rules and the six subnet
   route-table associations are written out longhand, exactly as the POC writes its two
   subnets and two route-table associations longhand. Acceptable at this size; if the
   resource count triples, the fallback is `runtime: python` for this one program.
4. **The two-phase apply is the POC's own pattern, reused on purpose.** Slice 0 already
   deploys in two phases because the image digest does not exist until CodeBuild has
   pushed it (`config: imageDigest` with an empty default, and README's "Deploy order").
   Phase 2 for tier 2 is the same shape: phase 1 = cluster + pools + Karpenter + queue
   (Pulumi), phase 2 = CRs + namespace + ServiceAccount + StorageClass (kubectl). A
   reader who has done one has done the other.

### 6.4 The driver, and the gate

`scripts/cloud_tiers_eks.py` (a sibling of `scripts/remote_agents_poc.py`, same
conventions: `argparse`, `--json`, `aws`-profile via environment only):

```
lop-tiers deploy --phase 1      # pulumi up   (refuses unless `pulumi config get enableEks` == true)
lop-tiers deploy --phase 2      # kubectl apply of infra/cloud-tiers/k8s/*.yaml
lop-tiers status                # pulumi stack output + kubectl get nodepools,ec2nodeclasses,nodes
```

`assert_expected_account` semantics apply here too (account `325492156725`, verified
first, before any mutating call), and the driver is **read-only in this slice**: it is
documented, not run. Nothing in this PR runs `pulumi up` or `kubectl apply` (§9.1).

---

## 7. CLI surface

Both live behind the hidden `cloud` subparser, and behind **one** registered
configuration key.

### 7.1 The subparser, and how it is hidden

`local_operator/cli.py` already states the rule: for a *subparser*,
`help=argparse.SUPPRESS` is not the hide idiom — argparse renders the sentinel verbatim
and `lop --help` lists `resume-click  ==SUPPRESS==`. **Omitting `help=` entirely is what
keeps it off the list**, which is what the `resume-click` parser does and says.

```python
cloud_parser = subparsers.add_parser("cloud", parents=[parent_parser])   # no help= : hidden
cloud_actions = cloud_parser.add_subparsers(dest="cloud_command")
# run | status | cancel | list, each with parents=[parent_parser]
cloud_run.add_argument("prompt", nargs="?")
cloud_run.add_argument("--tier", choices=("fargate", "k8s"), default="fargate")
cloud_run.add_argument("--size", choices=("s", "m", "l"), default="m")
cloud_run.add_argument("--budget", type=float, default=0.0, metavar="USD")
cloud_run.add_argument("--max-hours", type=float, default=2.0)
cloud_run.add_argument("--spot", action="store_true")
cloud_run.add_argument("--repo", default=""); cloud_run.add_argument("--sha", default="")
cloud_run.add_argument("--hosting", default=""); cloud_run.add_argument("--model", default="")
cloud_run.add_argument("--label", action="append", default=[], metavar="K=V")
cloud_run.add_argument("--mock", action="store_true")          # the POC's key-free path
cloud_run.add_argument("--yes", action="store_true")           # spending confirmation (see below)
```

Dispatch is one new arm in `main()`'s subcommand chain, beside
`elif args.subcommand == "exec":` — and `local_operator/cli.py` is a **boot-closure
barrier** in `scripts/ci_scope.py`'s terms, so this PR runs the whole-tree `test` and
`type-check` jobs whatever else it touches (AGENTS.md's scoping table). Plan for it;
do not be surprised by it.

`exec --cloud` (slice 3) is additive flags on the existing `exec_parser`:
`--cloud`, `--tier`, `--spot`, `--budget`, `--max-hours`. It constructs the same
`PoolRequest` and calls the same `cloud.cli.run_request(...)` — an alias, not a second
implementation. **It does not use `exec --peer`**, which is not built (§1.3).

### 7.2 The one configuration key

`cloud.enabled`, default **false**:

* it is what makes every `lop cloud …` verb refuse with a named, actionable sentence
  while the feature is experimental — including a hidden verb, because "hidden from
  `--help`" is not "unreachable";
* it is the kill switch for a feature that spends money, and it must not depend on a
  user remembering a flag;
* registering it follows AGENTS.md's "Adding a configuration key" exactly, all four
  touch points named: a `Setting` in `local_operator/settings_io.py`'s `SETTINGS`, a
  `Section` in `SECTIONS` if the section is new (with an honest `Scope` from `Scope`,
  which is uniform within a section by construction), a **real module-level default
  constant next to the code that reads it**
  (`local_operator/cloud/cli.py::DEFAULT_CLOUD_ENABLED = False`) mapped in
  `tests/unit/test_settings_io.py`'s `_consumer_defaults()`, and the key's `path=`
  spelled as a genuinely nested tuple (`("cloud", "enabled")`, not a dotted key —
  dotted keys are the `display.*` exception and getting this backwards writes a key
  nothing reads).

A second key is deliberately **not** added: no account id, region, cluster name or
task-definition ARN becomes configuration. They are stack outputs (§4.2), because a
hand-copied ARN in `config.yml` is a value that drifts from the thing that owns it.

### 7.3 Which verbs ship when

| Surface | Slice | Note |
|---|---|---|
| `lop cloud run --tier fargate …`, `status`, `list`, `cancel` | 3 | the four verbs; `--tier k8s` refuses with `tier_unknown` until slice 4 |
| `lop exec --cloud [--tier fargate]` | 3 | alias onto the same `PoolRequest` path |
| `--spot`, `--tier k8s` | 4 | needs the cluster |
| `--include-local` (git bundle of unpushed commits) | later | product design §1.1; not in the four slices |
| `/new cloud` in the TUI, the desktop/mobile placement picker | later | needs the member slice (§9.3) |

---

## 8. Security posture of this slice

`remote-cloud-agents.md` §7 is the authority; this slice must not weaken it. The four
properties a reviewer should check against that section:

1. **No credential on the pod's disk, and none in the workspace.** Tier 1: unchanged
   from Slice 0 (ECS secret → memory, `entrypoint.sh` unsets it before the uid check,
   the residual SEC-11 `ps`-environment caveat is still open and still documented).
   Tier 2: the same discipline with a `Secret` mounted via `envFrom`, `unset` on read,
   and the pod's `automountServiceAccountToken: false`.
2. **No cloud permissions on the pod.** Tier 1: the empty task role (no policy, verified
   by probe 4a). Tier 2: a `ServiceAccount` with **no** IRSA/Pod-Identity annotation and
   `automountServiceAccountToken: false` — so the pod has no AWS identity at all, which
   is strictly stronger than "an empty role", and the node's instance role is the only
   credential in the path (it is not reachable from inside a pod without the token the
   pod does not mount, and IMDS is `httpTokens: required` with
   `httpPutResponseHopLimit: 1`).
3. **No epoch secret on the pod, by construction, because the pod is not a member.**
   This slice's tasks are unbound (`network_id: ""`), so there is no invite token, no
   member row, and no `epoch_key` MAC to replay. That is the honest way to be safe about
   C2 until C2 exists.
4. **Egress is unchanged and still not narrowed.** Neither tier ships the egress proxy
   (`remote-cloud-agents.md` §7.2 / §9.2's P3). Tier 1's SG is egress-443-only and no
   inbound (unchanged); tier 2's is the same posture, and the cluster's API endpoint is
   private-only (§6.2), so nothing about the *control plane* is publicly reachable even
   though the pod's egress is broad. Stated in the doc rather than implied by a green
   test: this is the residual a reviewer should weigh first (`§12 R3`).

---

## 9. SCOPE CUT

### 9.1 In scope for THIS PR

* `local_operator/cloud/types.py`, `provider.py`, `ledger.py` — the protocol, both
  records, the state machine, the ledger.
* `local_operator/cloud/aws.py` + `fargate.py` — `FargateProvider` with an injected
  `FargateClients`, unit-tested against `botocore.stub.Stubber`.
* `local_operator/cloud/kube.py` + `k8s.py` + `workload.py` — `K8sWorkloadProvider` with
  an injected `KubeRunner`, unit-tested against a fake runner; the manifests are pure
  builders, tested as data (the scheduling contract of §5.3 is a unit test).
* `local_operator/cloud/fake.py` — `FakeComputeProvider`, used by the tests and by
  `--mock`.
* `infra/cloud-tiers/Pulumi.yaml` + `k8s/*.yaml` + `scripts/cloud_tiers_eks.py` —
  written, documented, **not run**; `enableEks: false` default; never referenced by CI,
  `Makefile` or any existing script.
* `local_operator/cloud/cli.py` + the hidden `cloud` subparser + `cloud.enabled` —
  and `exec --cloud` as an alias.
* `pyproject.toml`'s `cloud` extra + `local_operator/optional.py`'s `EXTRAS`.
* A PR body line: `Release: patch — hidden experimental command surface; no default
  install gains a dependency` (the window owner picks the bump; §11 of AGENTS.md's
  versioning rules, and a patch is the honest call for a hidden experimental path).

### 9.2 Deferred, with the reason each one is deferred

| Deferred | Reason |
|---|---|
| **C2 attested pool admission** (`pool` handshake mode, pod never holds the epoch secret) | New code in `handshake.py`/`relay.py` with the rotation carve-outs (epoch-frame withholding, number-only epoch advance) and a bidirectional auth story. It is the largest single piece in the product design and it is *not* required to run compute. Without it, a pod in a network holds either the epoch secret (→ `admin`) or nothing to authenticate with. |
| **`placement.mode: "pool"` as a producer** | Follows directly from C2, and from `MemberRecord` having none of compute-pool §3.3's seven keys (§1.2). See §9.3. |
| **The control plane** (hosted service, Radient login, grants, TTL reaper, EventBridge meter) | `remote-cloud-agents.md` §3.1/§5. This slice provisions from the user's own machine with their own AWS profile, which is the POC's shape. |
| **Metering / billing / Radient credits** | `mesh-compute-pool.md` §5 and product §8.4. The ledger records `created_at`/`started_at`/`stopped_at` and the provider's own timestamps, which is the *input* metering will need; it computes nothing. |
| **Blind relay** (both ends dial out) | §3.2 option A. Tier 2's pods are never members, so there is nothing to relay. |
| **Egress proxy / allowlist** | Product §7.2. Residual, stated in §8.4 above and §12 R3. |
| **Git proxy and per-task scoped push token** | Product §6/D4. Slice 0 clones a public fixture and uploads a bundle; this slice keeps that. When the proxy lands, `network/credentials/github.py`'s `grant_narrowing` is the existing mechanism to narrow it (and `credentials/gitlab.py` the second arm). |
| **`/new cloud`, desktop/mobile launch, artifacts, PR creation, cost meter in the status line** | Product §10 P2–P3. |
| **Workspace resume after spot interruption** (§5.4) | Needs AZ-pinned spot capacity, EFS, or snapshotting; honestly out of reach and dishonest to imply. |
| **`reconcile()`'s automatic driver (re-arming a task after a crash)** | `move_queue.py` needs it because a move is *the* operation; a cloud task is observable through the provider, so `reconcile()` re-reads and reports instead of re-driving. Narrower and testable. |

### 9.3 `placement.mode: "pool"`: explicitly deferred, and why that is safe

**Decision: this slice produces no `mode: "pool"` session.** `PoolRequest.network_id`
defaults to `""` and `lop cloud run` refuses a non-empty `--network` with
`network_unbound` ("cloud membership is not built; run without a network"), so there is
no code path that could write one.

Why this is the safe choice rather than a cheat:

* The *only* safe admission story is C2, because admission is where authority enters.
  Today an invite token **carries the network's epoch secret** and holding that secret is
  enough to mint invites at any role including `admin`
  (`local_operator/network/relay.py`'s `_ctl_invite` calling `mint_invite(record, state.secret, role=…)`).
  A pod that gets it is a pod that can admit itself and anything else. Nothing in this
  slice needs that door open.
* Producing `mode: "pool"` without C2 would mean choosing between two dishonest options:
  a pod holding the secret, or a placement whose member row nothing authenticated
  (`verify_auth`'s member-mode step is an epoch-key MAC, and there is no key to check
  against without the row C2 writes before step 6).
* The **user-visible** capability is not lost. `lop cloud run --tier fargate` returns a
  branch, a session directory and a bundle, exactly as Slice 0 does — the transcript is
  resumable locally with `lop --resume <id>` (Slice 0's acceptance item 3). What is
  missing is *live* view/steer of a running cloud task, which is Slice 1's own promise
  and its own gate (product §9.3 items 8–12).
* The ledger is designed so the member-facing half is *additive* when C2 lands: the
  `member_state` mapping (§3.4) already exists, `grant_id`/`expires_at` are already
  fields, and `MemberLifecycle` is already imported. Nothing here has to be re-cut.

---

## 10. Slice plan

Four slices, ordered so each one is independently reviewable and shippable, and so the
expensive-and-uncertain work (a real cluster) comes last. Every slice runs the gates
below; **the whole-tree jobs are unavoidable for slice 3** because `local_operator/cli.py`
is a boot-closure barrier.

**Gates, every slice** (from AGENTS.md, CI's pins named):
`flake8 .` · `black --check .` (26.1.0) · `isort --check .` (5.13.2) · `pyright` ·
the isolated unit run —
`env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" PATH="$PATH" TERM=xterm-256color .venv/bin/python -m pytest tests/unit -q`,
with `ISO=$(mktemp -d)` **in the same block that uses it** (a block that borrows another's
expands to an empty `HOME`, which Python 3.14 reads as `/`). Every feature worktree owns
its **own** `.venv` — never a symlink to another worktree's.

---

### Slice 1 — the seam and the ledger (no cloud, no CLI)

**Files:** `local_operator/cloud/__init__.py`, `types.py`, `provider.py`, `ledger.py`,
`fake.py`; `tests/unit/cloud/test_types.py`, `test_ledger.py`, `test_provider_contract.py`.

**Acceptance criteria**

1. `tests/unit/cloud/test_types.py::test_pool_request_round_trips` — every field, through
   `to_json`/`from_json`, equality preserved.
2. `...::test_a_version_1_request_from_the_pool_contract_still_parses` — a literal
   `mesh-compute-pool.md` §4.1 JSON body parses with the cloud delta defaulted
   (`tier == "fargate"`, `budget_usd == 0.0`, `spot_ok is False`), i.e. absent is not a
   claim.
3. `...::test_unknown_keys_are_dropped_not_refused`, and
   `...::test_a_version_we_cannot_read_is_refused_by_name`.
4. `test_ledger.py::test_terminal_states_are_write_once` — `expired` after `failed` is a
   no-op with the reason preserved (the `move_queue` rule, proved on this record).
5. `...::test_transition_journal_is_append_only` — `events` grows by one per accepted
   transition and never mutates an earlier entry.
6. `...::test_member_lifecycle_mapping_matches_the_table_in_the_design` — parametrised
   over all eight `TaskState`s, including `failed`/`cancelled`/`lost` → `None` when the
   task never ran.
7. `...::test_two_processes_cannot_interleave_a_transition` — two `CloudLedger` objects,
   one record: the `WakeWriteLock` is genuinely held across the read-modify-write (drive
   it with a lock the test holds, and assert the second writer's refusal, not a sleep).
8. `test_provider_contract.py` — `FakeComputeProvider` and every provider class satisfy
   `runtime_checkable` `ComputeProvider`; `submit` is the only method whose fake
   mutates `describe()`'s reading.
9. **Prove each test can still fail** (AGENTS.md's rule): revert `TERMINAL_STATES`'s
   guard and watch 4 go red; make `from_json` strict and watch 2/3 go red.

### Slice 2 — `FargateProvider`, extracted from the POC

**Files:** `local_operator/cloud/aws.py`, `fargate.py`, `workload.py` (the
`task_environment` builder); `pyproject.toml`'s `cloud` extra; `local_operator/optional.py`;
`tests/unit/cloud/test_fargate_provider.py`, `test_aws_account_guard.py`,
`test_task_environment.py`, `tests/unit/test_optional.py` (the existing sync test, one new
parametrised case).

**Acceptance criteria**

1. `test_fargate_provider.py::test_submit_shapes_runtask_exactly_as_the_poc_does` — a
   `Stubber` on `run_task` asserts the exact kwargs the POC sends (`launchType`,
   `platformVersion`, `count`, `networkConfiguration`, `enableECSManagedTags`,
   `propagateTags`, and tags `{lop-poc: "true", owner: lopdev, run-id: <id>}`).
2. `...::test_the_overrides_limit_is_refused_by_name_at_8192_bytes` — one byte under
   passes, exactly 8192 refuses with `CloudRefused(code="overrides_too_large")`.
3. `...::test_status_returns_pending_while_describe_tasks_is_empty` — no sleep, no loop:
   `phase == "pending"`.
4. `...::test_drain_stops_the_task_at_the_deadline_and_says_so` — controlled `sleep`,
   `deadline_s=2`, asserts one `stop_task` with `reason` naming the deadline and
   `DrainOutcome.drained is False`.
5. `test_aws_account_guard.py::test_the_account_is_asserted_before_any_mutating_call` —
   a `Stubber` whose `sts.get_caller_identity` returns a different account, and an
   `ecs` stub with **no** expectations: any mutating call is a test failure by
   construction, so "first" is proved rather than asserted.
6. `test_task_environment.py::test_env_names_are_the_ones_the_image_reads` — the exact
   ten names `infra/remote-agents-poc/image/entrypoint.sh` + `image/lop_launch.py` read,
   bound so an image change and a provider change cannot drift silently.
7. `test_optional.py` gains `cloud` and stays green (the `EXTRAS` ↔ pyproject sync).
8. **No network, no AWS account, no `boto3` in the base install:** the slice's tests run
   in a venv with the `cloud` extra and pass under `-p no:randomly` and in any order;
   `pip install local-operator` (no extra) still imports the app and every existing test
   passes unchanged — i.e. `local_operator/cloud/fargate.py` is imported *only* behind
   `require_extra`, and a test proves the refusal sentence names the extra and the
   exact `pip install` command (the `local_operator/optional.py` contract).

### Slice 3 — the CLI, the ledger in the user's hands

**Files:** `local_operator/cloud/cli.py`; `local_operator/cli.py` (one dispatch arm + the
hidden subparser + `exec --cloud` flags); `local_operator/settings_io.py` (the key);
`tests/unit/test_settings_io.py` (`_consumer_defaults`); `tests/unit/cloud/test_cloud_cli.py`;
plus the `test_console`/help-surface tests that pin subcommand listing.

**Acceptance criteria**

1. `test_cloud_cli.py::test_verb_is_hidden_from_help` — `lop --help` output contains no
   `cloud`; `lop cloud --help` works. (The `resume-click` precedent, asserted rather than
   assumed; `help=argparse.SUPPRESS` on a subparser would print `cloud  ==SUPPRESS==` and
   this test catches exactly that.)
2. `...::test_disabled_key_refuses_every_verb_by_name` — with `cloud.enabled` false (the
   default), all four verbs exit non-zero with the sentence naming the key and how to set
   it; asserted for the *hidden* verb, not just the listed one.
3. `...::test_run_refuses_a_zero_budget_before_touching_a_provider` — `budget_usd == 0`
   raises `budget_missing` with the fake provider's `submit` never called (the
   spending guard is proved by call counting, not by reading code).
4. `...::test_run_refuses_network_id_by_name` — §9.3's `network_unbound` refusal, so a
   later slice that adds membership has to delete a test rather than notice a gap.
5. `...::test_status_reads_the_ledger_when_the_provider_is_unreachable` — the record is
   still reported, with the provider's `detail`, and nothing is guessed.
6. `...::test_cancel_on_a_queued_task_writes_cancelled_and_never_submits`.
7. `...::test_exec_cloud_and_cloud_run_build_the_same_request` — the alias is proved to
   be an alias (same `PoolRequest` JSON), which is what stops a second implementation
   growing.
8. The whole-tree `test` and `type-check` jobs are green (boot-closure barrier), and
   `make lint`/`flake8 .`/`black --check .`/`isort --check .` are green over the whole
   tree because `local_operator/cli.py` is in the barrier set.

### Slice 4 — the EKS tier (code + IaC written; cluster only if the operator asks)

**Files:** `local_operator/cloud/kube.py`, `k8s.py`, `workload.py`;
`infra/cloud-tiers/Pulumi.yaml`, `infra/cloud-tiers/k8s/*.yaml`,
`infra/cloud-tiers/README.md`, `scripts/cloud_tiers_eks.py`;
`tests/unit/cloud/test_k8s_workloads.py`, `test_k8s_provider.py`,
`tests/unit/cloud/test_pulumi_program.py`.

**Acceptance criteria**

1. `test_k8s_workloads.py::test_the_scheduling_contract_is_the_table_in_the_design` —
   parametrised over `spot_ok ∈ {False, True}`: the exact `nodeSelector` / `tolerations` /
   `affinity` terms from §5.3, asserted as data.
2. `...::test_a_spot_workload_is_impossible_on_a_static_node` — the taint/affinity pair
   is checked *both* ways (no toleration for `spot_ok=False`, and the required affinity
   excludes the static label for `spot_ok=True`), so neither half can be dropped alone.
3. `...::test_the_prompt_never_appears_in_the_workload_spec_or_argv` — the prompt is in
   the `Secret` only; asserted by scanning the serialised manifest **and** every argv the
   fake runner recorded.
4. `test_k8s_provider.py::test_failed_scheduling_for_120s_patches_to_the_static_shape` —
   controlled clock, asserting the patch argv, `spot_fallback = True`, and the
   `spot_fallback` event (§5.4).
5. `...::test_status_maps_pod_phase_running_to_running_and_absent_to_lost`.
6. `...::test_drain_at_the_deadline_deletes_and_reports_deadline` — the `drained: False`
   arm of §5.5, and the PVC retention policy is present in the manifest.
7. `...::test_a_missing_kubectl_refuses_with_the_binary_and_the_hint` — no
   `FileNotFoundError` escapes; `provider_unavailable` + the `aws eks update-kubeconfig`
   hint.
8. `test_k8s_provider.py` asserts **every** `KubeRunner` call is an argv list (no
   `shell=True`, no interpolated string) — a structural invariant, not a numeric one.
9. `test_pulumi_program.py` — the YAML parses; `enableEks`'s default is `false`; every
   resource carries `options.provider: ${aws}`; the provider's `defaultTags` are
   `{lop-poc: "true", owner: lopdev}`; the static node group's `taints` are empty and its
   `labels` are `{lop.dev/pool: static}`; and `infra/cloud-tiers/` is referenced by **no**
   script, `Makefile` target or workflow file (the "nothing deploys by default" gate,
   asserted).
10. **No AWS call, no cluster**, unless the operator separately approves a deploy: every
    test above runs against the fake runner and the parsed YAML.

---

## 11. Cost

Prices below were fetched **read-only on 2026-10-14** with `AWS_PROFILE=minerva_sandbox`
(account asserted first: `aws sts get-caller-identity` → `325492156725`), AWS Price List API
at the `us-east-1` endpoint filtered to `location = "Canada (Central)"`, and
`ec2:DescribeSpotPriceHistory` in `ca-central-1`. No mutating call was made.

| Item | Price | Monthly at 730 h |
|---|---|---|
| EKS control plane (`CAN1-AmazonEKS-Hours:perCluster`) | $0.1000/h | **$73.00** |
| Static node, `m7g.xlarge` on-demand (`CAN1-BoxUsage:m7g.xlarge`) | $0.1819/h | **$132.79** |
| `m7g.large` on-demand | $0.0910/h | $66.43 |
| `m6g.large` on-demand | $0.0856/h | $62.49 |
| gp3 EBS (`AmazonEC2`, gp3) | $0.088/GB-mo | $2.64 per 30 GB node volume |
| NAT gateway hour (`CAN1-NatGateway-Hours`) | $0.0500/h | **$36.50** + $0.05/GB |
| Fargate ARM, size `m`, 2 vCPU / 4 GiB | $0.0869/task-hour | — (per-task; product §8.1's figure, confirmed against Slice 0's settled bill) |

**Tier 2 monthly floor** (the smallest cluster that can host one size-`m` workload plus
the system pods):

| | $/month |
|---|---|
| EKS control plane | 73.00 |
| 1 × `m7g.xlarge` static (min 1 / desired 1 / max 3) | 132.79 |
| 1 × 30 GB gp3 (`/workspace`) | 2.64 |
| **Floor with no NAT** (public subnets for the nodes) | **208.43** |
| NAT gateway (private subnets — the shipped default) | +36.50 |
| **Floor as designed** | **244.93** |
| Optional: 3 interface endpoints (ECR api + dkr, Logs) for image pulls that skip NAT | ~+8 each per AZ |

**Spot discount, measured rather than assumed.** Ratios from
`DescribeSpotPriceHistory` (`Linux/UNIX`, `ca-central-1`, latest six hourly points):

| Type | Spot range | On-demand | Spot as a share of on-demand |
|---|---|---|---|
| `m7g.xlarge` | $0.0885–0.0954 | $0.1819 | 49–52% |
| `m7g.large` | $0.0470–0.0573 | $0.0910 | 52–63% |
| `m6g.large` | $0.0434–0.0498 | $0.0856 | **51–58%** |

So the honest assumption is **spot ≈ 50–60% of on-demand in `ca-central-1`**, i.e. a
**40–50% discount**, *not* the "up to 70% off" this design's sibling quotes for **Fargate
Spot** (a different product and a different number). `remote-cloud-agents.md` §8.1's
`0.026` for Fargate Spot is an estimate on that other product; it is not evidence for
EC2 spot.

**Per-task-hour, the three substrates side by side** (the number a user-facing quote
would print):

| Substrate | $/size-`m`-task-hour | Note |
|---|---|---|
| Fargate ARM (tier 1) | **0.0869** | no floor; scale to zero; per-task isolation |
| EKS static (tier 2, 2 payloads on an `m7g.xlarge`) | ≈0.091 | plus the $208.43/mo floor |
| EKS spot (tier 2, 1 payload on `m6g.large`) | ≈0.043 | plus the floor; interruptible |

**What dominates, and the honest conclusion:** the tier-2 *floor* is
**≈$209–245/month for idle capacity**, which is ~2,400–2,800 size-`m` Fargate
task-hours — i.e. tier 2 does **not** pay for itself as a way to run short tasks. It is
justified only by what tier 1 cannot do: a persistent workspace, in-cluster neighbours
(sidecars, services, a database), and spot pricing on long runs. Model spend still
dominates a real run (product §8.4); compute is the small line item and the *floor* is
the large one.

---

## 12. Risks to watch during rollout

* **R1 — the ledger becomes a second source of truth about a provider.** The provider is
  the authority on whether a workload is running (`remote-cloud-agents.md` §3.1: "AWS is
  the authority"). Mitigation: `status()` is the only writer of live state, `reconcile()`
  re-reads rather than re-drives (§9.2), and no timer transitions a record without a
  provider reading.
* **R2 — the spot fallback patch races Karpenter.** A patch that flips the workload to
  the static shape while Karpenter is mid-provision of a spot node leaves an empty spot
  node that consolidation reclaims in `consolidateAfter: 60s`. Watched by: the
  `spot_fallback` event, and a `describe()` that reports both node pools' node counts, so
  a fallback spike is visible rather than inferred.
* **R3 — egress is not narrowed (unchanged from the product design's §7.2).** A
  prompt-injected workload can exfiltrate whatever egress allows. This is the largest
  open security gap in both tiers and it is *not* closed by this slice; the mitigation
  is that the workload holds no credential of value (no cloud role, no forge token) and
  its results leave by an upload the user can inspect. Named here so no green test reads
  as a fix.
* **R4 — `kubectl` on the operator's host drifts from the cluster's version.** The
  provider must refuse a version skew that matters (`--output=json` shape changes) by
  name rather than mis-parsing. Watched by `describe()` reporting the client version in
  `detail`.
* **R5 — the `cloud` extra is a new install surface.** `tests/unit/test_optional.py`'s
  sync guard protects the declaration, but the *behavioural* risk is a base install that
  refuses at runtime with a confusing message. Watched by slice 2's criterion 8.
* **R6 — the tier-2 IaC exists but has never been applied.** The YAML's type names,
  the Karpenter chart's value keys (`settings.interruptionQueue`) and the EKS addon
  versions are verified against the docs but not against a real `pulumi preview`. The
  first `preview` is a required step before any `up`, and the driver refuses phase 2 when
  phase 1's outputs are missing. Watched by: `scripts/cloud_tiers_eks.py status` printing
  the stack's own outputs, and §6.3's two-phase discipline.
* **R7 — `local_operator/cli.py` is a boot-closure barrier**, so slice 3's local loop is
  the whole-tree suite (40–55 min under fleet load, AGENTS.md). Sequencing risk, not
  correctness: slice 3 should be the *only* in-flight change in its window if the host is
  busy.

---

## 13. Divergences and contradictions found (the reviewer's list)

| # | Finding | Evidence | Action |
|---|---|---|---|
| 1 | `lop exec --peer` is **not built**, though the ask cites it as on main and a guide promises it | `exec_parser` has no `--peer`; no `--peer` outside the `sessions`/`network` verbs; `tests/unit/guides/test_guides.py` asserts the guide *string* only | §7 does not depend on it; remote creation stays `lop network sessions --create --peer` |
| 2 | `mesh-compute-pool.md` §4 cites `local_operator/mesh/placement.py`, which does not exist (real: `local_operator/session/placement.py`), and a *different* `placement` module exists at `local_operator/network/credentials/placement.py` | `ls` on all three paths; `network/onboard.py` imports the credentials one as `placement_mod` | recorded here; the sibling doc's path is stale |
| 3 | `MemberRecord` does **not** carry compute-pool §3.3's seven pool-delta keys, which that doc attributes to transport §4.2 | `network/types.py`'s `MemberRecord` field list | confirms §9.3's deferral; named so the member slice starts from the real baseline |
| 4 | `boto3` is deliberately **not** in the default install set | `pyproject.toml` `dev` extra's comment | §3.7: a `cloud` extra, not a base dependency |
| 5 | Product-design §4 **rejects** EKS for the *per-task* substrate on isolation grounds; the ask's tier 2 is a *dedicated-workload* tier | `remote-cloud-agents.md` §4 ("A namespace is not a kernel boundary… The $73/month control plane buys nothing at this scale") | both are coherent: tier 1 stays Fargate, and tier 2's justification is §11 — tier 2 is *not* a replacement for tier 1, and must not be sold as one |
| 6 | The POC's cost criterion is still PARTIAL: the `lop-poc` tag the §9.3 acceptance criterion names is not an activated cost-allocation tag in that linked account | `remote-cloud-agents-poc-results.md` § Test 6 (`lop-poc=true` returns $0; `lop-poc-fargate` is the working backstop) | noted for whoever runs the first tier-2 cost check: budget on `owner=lopdev` + the service filter, exactly as `lop-poc-fargate` does |

---

## 14. Open questions for the operator (each with my recommendation)

| # | Question | Recommendation | What would change it |
|---|---|---|---|
| Q1 | Does tier 2 deploy in `325492156725` (`minerva_sandbox`, recorded as Pergamon's sandbox destination and in the Minerva-managed org), or wait for the dedicated Radient-owned account `remote-cloud-agents.md` D1 recommends? | **Write the IaC, apply nothing** (this slice). Decide the account before the first `pulumi up`. | The operator naming an account for lop's own infrastructure |
| Q2 | Tier 2's static pool size: 1 × `m7g.xlarge` (floor $208/mo) or start at `min 0`? | **`min 1`**, because `kube-system` + the EBS CSI controller + Karpenter itself need a non-spot home, and a 0-floor cluster cannot schedule its own DNS | A cheaper design that taints `kube-system` onto Fargate profiles (possible, more moving parts, and it re-opens the "namespace is not a boundary" question) |
| Q3 | Is a spot tier wanted for *short* tasks at all, given §11's ≈$0.043 vs $0.0869 per task-hour against a $209/mo floor? | **No, not on cost.** Spot belongs to long-running dedicated workloads (multi-hour) where the floor is already paid | An operator use case where a task runs for hours and the ~45% discount is material |
| Q4 | `cloud.enabled` — the one registered key (§7.2), or no kill switch and rely on the hidden subparser? | **The key.** A spend-capable surface needs an off switch that is not "remember not to type it" | A house preference against adding keys for experimental surfaces |
