# Remote cloud agents — delegating a task to compute we host in AWS

Status: **research and design proposal, pre-implementation. Nothing here is built, and
nothing here authorises building or provisioning anything.** The POC in §9 is a separate
step that needs its own approval.
Base: `origin/main` @ `f47893a22` (v0.68.1). Every `file:line` below was read at that
revision in a worktree of `origin/main`. Line numbers drift, so cite by symbol.
Siblings, in their order of authority for this document: `mesh-network.md` (spine,
R20–R22, A8), `mesh-compute-pool.md` (the pool contract this document builds on),
`mesh-credentials.md` (the broker), `mesh-transport-identity.md` (links, pairing,
capabilities), `mesh-session-mobility.md` (placement, sync, `exec --peer`),
`mesh-incident-response.md` (audit and revocation), `mesh-consent-provisioning.md`.
External facts were accessed **2026-10-06**. Vendor prices are **list-price estimates**
unless the row says the price came from the AWS Price List API.

---

## 0. The answer in one page

**What the operator asked.** The user runs `lop exec` (or the TUI, desktop or mobile
equivalent) and the task runs on agent compute that **we** host in AWS. The compute is
provisioned on demand, runs in isolation, and is torn down afterwards. This is the
Cursor Cloud Agents / Codex cloud / Claude Code cloud-sessions product shape.

**Where lop stands.** The *mesh half* is real. Device identity, pairing, peer session
create/view/steer/stop, the credential broker, R22 session sync, revocation and the audit
log are all built (§2). The *pool half* is designed and not built. Nothing produces
`placement.mode: "pool"`. There is no `PoolRequest`, no A8.1 attested admission, no
metering, and no provisioning code anywhere in the product. Separately, six assumptions
in the pool contract do not hold for a *hosted cloud agent*, and one addition is needed
(§5.2). Four are assumptions that "home is online" (C2, C3, C5, C6), which is the
opposite of the product's main promise ("close your laptop and it keeps working"). One
(C1) is a contradiction between the docs and the as-built capability model that would
stop a pool member from borrowing a model key. One (C4) assumes the control plane can
read a local file. C7 is additive. The hardest of them is admission (C2): today an invite
token carries the network's epoch secret and the joiner must dial the inviter. Both
facts collide with a NATed laptop, and the secret also lets its holder mint `admin`
invites. So the proposal gives pool members their own handshake mode in which **the pod
never holds the epoch secret**.

**Recommendation.**

1. **Substrate:**
   - **POC:** ECS on Fargate, one task per agent run.
   - **Production v1:** ECS on Fargate as well.
   - **Production v2 trigger:** move to Firecracker microVMs when tasks need Docker inside
     the sandbox, snapshot fast-start or suspend/resume. The choice between AWS Lambda
     MicroVMs (once offered in `ca-central-1`) and self-run Firecracker on nested-virt or
     metal hosts is decided then.

   Why Fargate: per AWS, each Fargate task has its own isolation boundary and shares no
   kernel with other tasks. It is the cheapest option at low concurrency (ARM 2 vCPU /
   4 GiB ≈ **$0.087 per task-hour** in `ca-central-1`, from the Price List API). It scales
   to zero, has no cluster to operate, and fits "one pool member per task".

   Rejected as the primary design: EKS with one namespace per task. A namespace is not a
   kernel boundary. EKS Pod Identity and VPC-CNI NetworkPolicy do not work on Fargate
   pods. Cold Karpenter nodes take minutes. The $73/month control plane buys nothing at
   this scale (§4).
2. **Control plane:** a small Radient-hosted service. The user authenticates with their
   **Radient account**, which is already the billing identity. Their **mesh device
   identity** is bound to each request. The service turns a `PoolRequest` into one
   Fargate task, signs an attestation binding the pod's self-minted key to the user's
   home device and relay route, enforces TTL, idle and budget limits, records start and
   stop times from AWS, and charges Radient credits. It also runs a **blind relay** so a
   NATed laptop and a private pod can link without either accepting inbound connections.
   The mesh still holds no capacity credential (`mesh-compute-pool.md` §7), and the
   control plane never holds a network's epoch secret: the pod is admitted by a new
   `pool` handshake mode, not an invite token (§3.2, §5).
3. **Credentials:** the default is **hosted model access through the Radient gateway**
   with a per-task, budget-capped token. Radient already fronts OpenRouter as a provider
   (`providers/clients.py` "Radient fronts OpenRouter") and already holds the credit
   balance. BYO keys through the mesh broker remain an option, but it **requires home to
   be online**, because broker grants last at most 900 s (`GRANT_TTL_S`,
   `network/credentials/__init__.py`). Git goes through a **control-plane git proxy**.
   The pod holds only a scoped credential that allows pushing to its one branch (the
   Claude Code cloud pattern [A1]). The proxy holds the real forge token; making that a
   GitHub App installation token is this doc's choice (Cursor uses a GitHub App [C4]). No
   secret is stored on the pod's disk (§6, §7).
4. **POC:** one task in, one Fargate run, a branch and a transcript back, teardown
   verified. The run happens in an AWS account the operator confirms (§9, D1). Expected
   cost is **under $5 of AWS compute and storage plus a capped model budget (~$20)**.

**Decisions only the operator can make** are listed in §11. The first one blocks
everything else: **which AWS account.** `minerva_sandbox` resolves to 325492156725, which
is a member of the AWS Organization whose management account is 212841448981 (the
account `minerva_nprod` resolves to). The same account is documented as **Pergamon's
sandbox destination**, where infrastructure changes go only through a Pulumi
GitHub-Actions path. lop itself is neither project: the repository is
`github.com/damianvtran/local-operator`, MIT, © Radient Inc.

---

## 1. What a user should experience

### 1.1 The flow, compared with Cursor

| Step | Cursor Cloud Agents (docs, 2026-10-06) | Proposed for lop |
|---|---|---|
| Launch | Desktop "Cloud" dropdown, web, iOS, Slack, GitHub/Linear `@cursor`, API [C1] | `lop cloud run "…"` (alias `lop exec --cloud`), `/new cloud [prompt]` in the TUI (beside the designed `/new remote <peer>`), "Cloud" in the desktop/mobile placement picker. All of them write a `PoolRequest` and a session with `placement.mode: "pool"` |
| Repo handoff | Clones from GitHub/GitLab/Bitbucket/Azure DevOps, works on a separate branch, pushes back [C1] | The default is **clone at the pushed SHA** of the current branch. `--include-local` uploads a `git bundle` of unpushed commits plus a patch of uncommitted changes (Claude's `CCR_FORCE_BUNDLE` analogue [A2]). **Never sync the whole worktree** (it is too large, may contain secrets, and nothing could promise it stays consistent) |
| Environment | Agent-led setup, Dockerfile, or `.cursor/environment.json`; "Builds" prepare snapshots [C2] | v1: one lop base image (Python, uv, git, the common toolchains) plus an optional repo setup script (`.lop/cloud-setup.sh`) read **at the start SHA**. Snapshots and builds are a v2 item that depends on the Firecracker substrate (§4) |
| Watch live | Web/desktop/mobile view; remote desktop take-over [C3] | **The same viewer as a peer session**: the pod is a mesh member and the session appears in the sidebar as a remote row (built: `RemoteSessionClient`, `network/projection.py`). Today the phone could reach it only through the existing tunnel *via home*; with the blind relay (§3.2 A, P2) the phone views it directly while the laptop is closed (§3.4) |
| Steer / follow up | Follow-ups from any surface [C3] | Built inner ops `steer`/`prompt`, carried by `net_forward`/`net_stream` (`INNER_OP_CAPABILITY`, `network/types.py`) |
| Approvals while headless | Isolation and an egress allowlist instead of per-call prompts [C4] | **The sandbox is the approval boundary.** Inside the pod the run auto-approves tools (an `--tools` allow-set or full auto). Actions that leave the box (push outside the task branch, egress outside the allowlist, spend above budget) are **refused by infrastructure**, not prompted. Prompts that remain (e.g. `ask`) park on `--control` and route to any attached viewer (built: `approval_answer` → `prompt` capability) |
| Notifications | Slack, mobile [C1] | Existing desktop/mobile notification paths, fed by control-plane state transitions (`provisioning → running → needs-input → finished/failed`) |
| Results | Branch plus draft PR, artifacts on the PR [C3][C4] | Branch `lop/<session-short-id>` pushed through the git proxy. A draft PR is optional (`--pr`). The transcript syncs home (R22). Artifacts (logs, screenshots) go to a per-task object prefix with a TTL |
| Resume | Follow-up on the same agent | While the pod is alive: an ordinary follow-up. After teardown: the session is cold at home and `/resume` continues **locally**, or `lop exec --cloud --resume <id>` rehydrates a fresh pod from the synced transcript plus the pushed branch |
| Cancel | Stop in UI | `lop cloud cancel <task>` / Stop in the UI → `net_session_stop` (built) **and** control-plane `StopTask`. The second one is what guarantees teardown even if the pod is wedged |
| Cost | API pricing for the model; spend limit required [C1] | A live meter in the status line and `lop cloud status`: compute seconds × size rate plus model spend, against a **required per-task budget** (`--budget`). A final receipt is written into the session (`session_spend.v1` already exists in `session/spend.py`) |

### 1.2 The one-line CLI

```sh
lop exec --cloud --size m --max-hours 2 --budget 10 \
  "Fix the flaky test in tests/unit/foo; open a draft PR"
# → cloud task ct_8c31 provisioning (size m, ≤2h, ≤$10 incl. model) …
#   session 3f9a… running on cloud member radient-m-8c31 — `lop --resume 3f9a…` to watch
lop cloud status ct_8c31   # provisioning|running|needs-input|succeeded|failed|cancelled + spend
lop cloud cancel ct_8c31
```

Cloud tasks get their own `lop cloud` verbs rather than reusing `lop exec --status`.
That flag is built and reads the **local** background-job ledger (`cli.py`, "does not
start a session"); one verb answering from two stores under look-alike ids would be a
trap. `lop exec --cloud` stays as the launch spelling because it is the same request as
a local `exec`, run elsewhere.

The spine already designs `lop exec --peer <peer>` as "the same relay op with a headless
viewer" (`mesh-session-mobility.md` §5.3). `--cloud` is `--peer` with a pod that the
control plane creates first. **Flag naming is decision D6.** `--cloud` matches Claude
Code's `claude --cloud` [A2]. `--remote` is ambiguous with `/new remote <peer>`.

---

## 2. What exists today: built vs. designed

Status key: **BUILT** (code on `main`), **PARTIAL**, **DESIGNED** (docs only), **ABSENT**
(neither).

| Piece a cloud agent needs | Status | Evidence (symbol, file) | Design reference |
|---|---|---|---|
| Device identity (Ed25519) + authenticated handshake + SAS | BUILT | `DeviceIdentity`, `mint` (`network/identity.py`); `verify_auth`, `establish` (`network/handshake.py`); `sas_code`, `LinkCrypto.seal` AES-256-GCM (`network/wire.py`) | spine A3; transport §3, §5, §6 |
| Transport | BUILT, **direct TCP only** | `RelayServer.bind` (`AF_INET, SOCK_STREAM`, `network/relay.py`); dial `socket.create_connection` | transport §6.1. **Two NATed devices cannot link**; a blind-relay hub is "not built here" (transport §10.4) |
| Pairing with no human at the joining end | BUILT | `join --automated` (`network/cli.py`, "the third spelling … no human at THIS end"); the inviter still compares the SAS | transport §12.4 |
| Remote onboarding over SSH, one-transaction provisioning (S1) | BUILT | `OnboardRun`, `step_join`, `step_provision` (`network/onboard.py`); approval store `network/approvals.py` | `mesh-remote-onboarding.md`; `mesh-consent-provisioning.md` S1 (#2025) |
| `MemberRecord.kind = "pool"`, lifecycle `provisioning/draining/expired` | PARTIAL: the types exist, nothing writes them | `MemberKind`, `MemberLifecycle`, `MemberRecord` (`network/types.py`); defensive `kind == "pool"` branches in `onboard.py` | compute-pool §3.3 (the seven extra keys `ephemeral`, `size_class`, `provider`, `provider_ref`, `grant_id`, `expires_at`, `max_session_seconds`/`max_sessions` are **not** in code) |
| A8.1 provider-attested admission | DESIGNED | no attestation in the admission path. The `attest` hits in `network/` (`sync.py` digests, `audit.py`, `readiness.py`, `mobility.py` comments) are unrelated. `relay.py` `admit(..., kind="device")` is never called with `"pool"`, and `--automated` only supplies the SAS at the joining end (`cli.py` `_join_one`), so compute-pool §3.2's "`--automated` … marks the admitted row `kind: "pool"`" is not true of the code | compute-pool §3.2 |
| Placement (A8) | PARTIAL | `SessionPlacement`, `PlacementMode = Literal["local","peer","pool"]` with the comment "`pool` is reserved and never produced in this pass" (`session/placement.py`); `peer` is produced on move/create | mobility §5; compute-pool §4 |
| `PoolRequest`, `lop network pool request/ls/cancel` | DESIGNED | no symbol; not among `lop network` subcommands | compute-pool §4.1 |
| Metering `meter_interval`/`meter_close`, `meter_push`/`meter_ack` | DESIGNED | no symbol (`meter_push` appears only in the design doc) | compute-pool §5 |
| Peer session create / view / steer / stop | BUILT | `_op_session_create`, `_op_session_engage`, `_op_session_stop`, `_op_forward`, `_op_stream` (`network/relay.py`); `RemoteSessionClient` (`network/projection.py`); `lop network sessions --peer … --create` | mobility §2–§5; `sessions-remote-tools.md` |
| Approvals answered from another device | BUILT | `INNER_OP_CAPABILITY["approval_answer"] = "prompt"` (`network/types.py`) | `mesh-remote-onboarding.md` §2.8 |
| `lop exec` headless, background worker, `--control`, `--status` | BUILT, **local only** | `exec_mode.py`, `exec_worker.py`, `ExecControl` (`session/runtime/exec_control.py`); no `--peer`/`--remote`/`--cloud` flag | `docs/EXEC.md`; `exec --peer` designed in mobility §5.3 |
| R22 session sync (replicas) | BUILT for device↔device | `build_manifest`, `copy_set`, replica cursor (`network/sync.py`); `net_sync` → `view` | mobility §7. **As-built deviation:** a replica is promoted **as a fork with a new id**, never the original id (`sync.py` module docstring, `promote_replica`). compute-pool §3.6 step 5 expects "the same id" at home |
| Pool drain barrier, `SYNC_DRAIN_DEADLINE_S` | DESIGNED | no symbol | compute-pool §6.2 |
| Credential broker (A5), grants ≤900 s, in memory | BUILT | `net_broker` → `broker_credential` (`OP_CAPABILITY`); `GRANT_TTL_S = 900.0` (`network/credentials/__init__.py`); provider logins and static API keys are brokered (`network/credentials/offers.py`); GitHub source ladder (`credentials/github.py`, #2026) | `mesh-credentials.md` §3, §6 |
| Pool members excluded from credentials | DESIGNED as "structural"; no code path writes a pool row, so it holds vacuously | `offers.py` docstring ("Pool exclusion is structural") | consent-provisioning §1 table ("`pool` member — **Nothing**") vs. credentials §6.2 ("it borrows; it never holds"). See §5.2 C1 |
| Revocation: tombstone + secret rotation + epoch bump; panic; audit JSONL | BUILT | `remove_member`, `rotate_epoch`, `_op_panic` (`network/relay.py`); `AuditLog`, `AuditEvent` (`network/audit.py`) | spine A6/A7; incident-response §4–§5 |
| Remote kill of a device / quarantine | ABSENT | panic "is **not** a remote kill switch" (`mesh-incident-response.md`) | — |
| Rolling updates / version skew | DESIGNED; node install pin PARTIAL | `_pinned_tag`, `step_install` (`onboard.py`); exact-version `compare_builds` (`network/readiness.py`) | `mesh-rolling-updates.md` |
| Radient tunnels (phone → harness) | BUILT | `RadientTunnels` (`tunnels/api.py`), `Gateway` (`tunnels/gateway.py`); `docs/tunnels.md` | transport §10.4 names it as the WAN path for a peer port, but **no code tunnels a peer port** |
| Radient billing | BUILT **for tunnels only** | `tunnels/cli.py` billing/positive-credit checks; desktop Radient proxy route | compute-pool §7: "the mesh writes an append-only stream; it never posts a charge" |
| Compute provisioning (any cloud API) | ABSENT | no `boto3`/`RunInstances`/`run_task` in `local_operator/` (the only AWS instance code is the evaluation harness) | compute-pool §7 defers it to a control plane |
| Bedrock / Vertex / Azure model providers | ABSENT | `model/discovery.py`: "Vertex, Bedrock and Azure, none of which exist in this tree today" | — |

**Reading.** The session plane (create, view, steer, stop, approve, sync), identity and
brokering are done and exercised: the `cloud-node-1` EC2 drill in
`mesh-remote-onboarding.md` §7 used an EC2 instance as a peer. A cloud agent therefore
needs **no new session protocol**. What is missing is everything around the pod:
provisioning, admission, the network path, credentials that work without home, a durable
copy of results while home is offline, metering and billing.

---

## 3. Architecture: how a task flows

```
 user device (home)                     Radient control plane (AWS)                Pod (Fargate task)
 ──────────────────                     ───────────────────────────                ──────────────────
 lop exec --cloud ──PoolRequest+Radient auth──► POST /v1/pool/tasks
   (CLI, not the relay)                    authn (Radient), quota/budget hold
                                           RunTask(size, image@digest, ttl) ─────► boot: clean lop install
                                                                                    mint device key (pod-side, OQ4)
                                           ◄──────── pod key + task metadata ────── report key to control plane
                                           sign attestation(pod key, account,
                                             network, grant, size, provider_ref,
                                             relay_route, home_device_id,
                                             home_pubkey, expires_at)
 home relay: verify attestation ◄── poll / push (home dials out, HTTPS) ──
 home relay ── dials ──────► blind relay (control plane) ◄── dials ── pod relay
            ══════ `pool`-mode handshake (no epoch secret on the pod), end-to-end ══════►
                (POC Slice 1 shortcut: pod public endpoint, home dials it directly)
 net_session_create(prompt, workspace spec) ══════════════════════════════════► clone@SHA / bundle; setup script
                                                                                    run session (owner = pod)
   viewer / steer / approvals ◄══════════ net_forward / net_stream (built) ═════►
                                           model tokens ◄──── Radient gateway (task token, budget) ── model calls
                                           git proxy ◄────── scoped push (one repo, one branch) ─── git push
 R22 flush ◄═══════════════════════ net_sync (built) ══════════════════════════ every tick + final
                                           custodian copy ◄── final transcript + artifacts (if home offline)
                                           EventBridge task-state → meter, TTL/idle reaper, StopTask
 session cold at home (resumable)          charge Radient credits; audit
```

### 3.1 Who does what

| Concern | Owner | Why there |
|---|---|---|
| Authenticate the user | Control plane, using the user's **Radient login** (the same identity `lop tunnel` and the desktop Radient route use) | Radient is already the billing identity, so a second account system adds nothing |
| Bind the request to a mesh network | Control plane stores `(radient_account, network_id, requesting device_id)`. The relay verifies the attestation with a **locally configured** control-plane public key (compute-pool §3.2) | Keeps A3's self-certifying identity: the control plane never holds a mesh private key (compute-pool OQ4) **or the network's epoch secret** (§5.2 C2) |
| Provision | Control plane: `ecs:RunTask` with a pinned image digest, size class and tags | "Nothing in the mesh holds a capacity credential" (compute-pool §7) |
| Lifecycle (TTL, idle reap, budget stop) | Control plane, with EventBridge ECS task-state events and a scheduled reaper. The pod enforces `expires_at`/`max_session_seconds` itself as defence in depth | AWS is the authority on whether a task is running. A wedged pod cannot be trusted to stop itself |
| Meter | Control plane uses the **AWS-observed** task start/stop and size as the billable record. The pod's signed `meter_interval` stream (compute-pool §5) becomes a **cross-check**, not the source | This removes compute-pool §5.5's main residual risk (a compromised pod inflating its own `cpu_ms`) for billing purposes. It also means the drain barrier must stop waiting on `meter_push` acks (C5) |
| Sync results home | The mesh (built R22) when home is reachable, plus a **custodian copy** (§5.2, C3) | R22 assumes home is reachable at drain time |

### 3.2 Network path: the unbuilt piece that is easiest to miss

Links are plain TCP, and "a link is established when either side can dial the other's
advertised endpoint" (transport §10.4). A laptop behind NAT cannot be dialled. Two
built facts constrain admission further:

- **The joiner dials the inviter.** The join block rides the *dialer's* `hello`
  (`handshake.py` `build_hello`/`send_hello` attach `join` only when this side dials), and
  the joiner's dial targets come from the token (`invite.py` `host_candidates`).
- **An invite token carries the network's current epoch secret.** `InviteEnvelope.material`
  is "the transfer of the network secret" (`invite.py`). That one secret keys the
  member-mode auth MAC (`wire.py` `epoch_key`, checked in `handshake.py` `verify_auth`)
  and the invite key (`wire.py` `invite_key`); the link's encryption keys come from the
  X25519 exchange plus the transcript (`wire.py` `link_keys`). Holding the secret is also
  enough to **mint invites**, any role including `admin`: the relay's local invite op
  calls `mint_invite(record, state.secret, role=…)` (`relay.py` `_ctl_invite`). Whoever
  holds it can therefore admit devices.

So the obvious shortcut, "the control plane hands the pod an invite token", makes the
control plane a bearer of every user's network secret, able to admit itself. This doc
rejects that. Options for the link:

| Option | Built? | Verdict |
|---|---|---|
| **A. Blind relay we host; both ends dial out.** Home and pod each dial a DERP-shaped forwarder in the control plane; the mesh handshake runs end-to-end through it, so the relay sees ciphertext only | DESIGNED as "the shape a future hub must be built to" (transport §10.4); not built | **Recommended for v1.** It is the only option that works with home behind NAT *and* the pod with no inbound port, and the same relay lets the **phone reach the pod while the laptop is closed** (Cursor's headline UX). The pod sits in a private subnet with egress via the proxy (§7.2) |
| B. Pod listens on a public endpoint; home dials the pod | The link is built; pool admission is new (C2) | **POC Slice 1 only.** Needs a public subnet, a public IPv4 ($0.005/h [P]) and an inbound security-group rule on the peer port, which conflicts with §7.2's "private subnet, proxy-only egress". Fine for proving admission, not for production |
| C. Expose home's peer port through the Radient tunnel | Named in transport §10.4; not built for raw TCP (tunnels serve HTTP/WS through `cloudflared`) | No: the tunnel path is HTTP-shaped and the peer protocol is raw TCP |

Whichever path carries the bytes, **admission is a new `pool` handshake mode (C2)**, not
a join with a token. The pod proves possession of its attested key, and **the pod never
receives the epoch secret**, not from the control plane and not from home. Handing it
over after admission would be mechanically easy (the join's `pair_result` and a
rotation's `epoch_frame` already carry the material), but it would let a
prompt-injected pod mint an `admin` invite. That contradicts compute-pool §3.5's
"blast radius is the sessions placed there" and this doc's §7.1.

### 3.3 One pod per task

compute-pool lets a member hold several sessions (`max_sessions`, OQ5). For hosted
agents the recommendation is **`max_sessions: 1`, one member per task**. Isolation
between a user's own tasks is then the substrate's, and teardown is per task. The
"pool member" abstraction stays in place for a later warm-pool optimisation.

### 3.4 What "laptop closed" means in each phase

| Phase | Home online | Home offline |
|---|---|---|
| Launch | required (the CLI submits) | — (v2: launch from web/mobile through the control plane) |
| Run, hosted model key | works | **works** (Radient gateway, no broker) |
| Run, BYO key via broker | works | **stalls within ≤15 min** (grant TTL 900 s, re-ask at expiry − 120 s; `mesh-credentials.md` §3.6) |
| View / steer | works | Through the blind relay (§3.2 A, P2): the phone or web viewer dials the relay directly. Before P2 (POC Slice 1, direct path): not possible |
| Results | R22 flush home | **custodian copy** in the control plane, pulled on the next connect (C3) |

---

## 4. Substrate options

All prices are `ca-central-1` on-demand USD for a **2 vCPU / 4 GiB** sandbox (ARM where
available).

| | (a) EKS, namespace per task | (b) ECS Fargate task per agent | (c) Firecracker microVMs | (d) EC2 instance per task |
|---|---|---|---|---|
| Isolation | Namespace + NetworkPolicy + ResourceQuota: **shared kernel** unless gVisor (syscall interposition) or Kata (VM per pod) is added. A namespace is a policy boundary, not a security boundary | "Each Fargate task has its own isolation boundary and does not share the underlying kernel, CPU resources, memory resources, or elastic network interface" [F1]. Fargate runs on Firecracker [F3] | A VM per sandbox on KVM [F4]. Strongest at the highest density | Full VM per task (Nitro) |
| Cold start (estimate; POC must measure) | Warm node: seconds. **Cold Karpenter node: ~2–4 min** [K1] | Unmeasured here; image pull dominates container start (76% in the study AWS cites) [F2]; SOCI lazy loading cuts image-pull-dominated starts 40–60% [F2] | <125 ms VMM boot [F4]; seconds with a snapshot. Lambda MicroVMs: snapshot launch | ~30–90 s boot plus pull |
| Compute $/task-hour (Price List API) | m7g.xlarge $0.1819/h ÷ 2 = **$0.091** + CP $0.10/h amortised (≈$0.101 at 10 concurrent, ≈$0.092 at 100) | 2×$0.03565 + 4×$0.00389 = **$0.0869**; Fargate Spot up to 70% off (ECS only, interruptible) | c6g.metal $2.3808/h ÷ 24 ≈ **$0.099** (÷32 ≈ $0.074); nested-virt c8i.xlarge $0.2051/h ÷ 2 ≈ **$0.103** | m7g.large (8 GiB) **$0.091**, per-second billing, 60 s minimum |
| Fixed monthly floor | EKS CP **$73** + NAT/endpoints; +$133 per idle warm m7g.xlarge | **$0** compute at idle; NAT/endpoints only | +$1,738 for an always-on metal host (or scale-to-zero hosts with slower starts) | $0 compute at idle |
| Ops burden | Highest: cluster upgrades, Karpenter, CNI policy, admission control, runtime classes | Lowest: task definition, IAM, security groups | High: own scheduler, image/snapshot pipeline, host fleet, jailer. **Or** managed Lambda MicroVMs (see below) | Medium: AMI pipeline, boot scripts, instance reaping |
| Per-task egress identity | Pod Identity / IRSA on EC2 nodes. **EKS Pod Identity does not support Fargate pods** [K2]; **VPC-CNI NetworkPolicy does not apply to Fargate pods** [K3] | Task role + per-task ENI + security group | Host-level tap/iptables per VM | Instance profile + SG |
| Docker inside the sandbox | Yes on EC2 nodes with privileged pods (weakens isolation); gVisor supports some | **No**: "No privileged containers … Docker in Docker" [F5] | Yes (a real VM) | Yes |
| Fit with the pool contract | Fine (pod = member) | **Best**: task = member, `provider_ref` = task ARN, AWS task state = lifecycle | Fine; best for v2 snapshots/suspend | Fine but slow and coarse |

**Where gVisor and Kata fit.** Both are *EKS runtime classes*. gVisor (`runsc`) adds a
user-space kernel on ordinary nodes, giving strong syscall filtering at container density
without snapshots. Kata + Firecracker gives a VM per pod but needs KVM. Nested
virtualization (KVM on **non-metal** instances) launched on C8i/M8i/R8i on 2026-02-16
[N2], and AWS's documentation now lists C7i/C8i/M7i/M8i/R7i/R8i/X8i/I7i families [N1].
A read-only `describe-instance-types` in `ca-central-1` (2026-10-06) returned `c7i c8i
c8id i7i i7ie m7i m8i m8id r7i r8i r8id` (+flex), so Kata or self-run Firecracker no
longer require `*.metal` here. Both make sense **only if we already operate EKS**, and
for one sandbox per task they are a heavier path to what Fargate gives by default.

**Managed microVMs.** AWS Lambda MicroVMs (announced 2026-06-22 [L5]) is the managed answer: a
Firecracker VM per session, snapshot launch, suspend/resume, at most 8 hours, and AWS
documents it as a sandbox for Cursor self-hosted cloud agents and for Claude Managed
Agents [L1][L2]. **It is not offered in `ca-central-1`.** It runs in 10 regions: the five
at launch (us-east-1, us-east-2, us-west-2, eu-west-1, ap-northeast-1) plus Mumbai,
Singapore, Sydney, Frankfurt and Stockholm added 2026-08-19 [L3][L4]. One practitioner
estimate puts the minimum 1 vCPU / 2 GB at $3.03/day, "9x+ Fargate spot" (secondary,
[L4]). Managed sandbox vendors (E2B: $0.000014/vCPU-s + $0.0000045/GiB-s → ≈ **$0.166/h**
for 2 vCPU/4 GiB [E1]; Daytona, Modal, Fly) are faster to adopt but put code and
transcripts outside our AWS account and region, which conflicts with data residency
(§7.6).

**Recommendation.**

- **POC:** (b) Fargate ARM.
- **Production v1:** (b) Fargate ARM. Optionally use Fargate Spot for tasks that accept
  restarts, since R22 sync makes a reclaim cost at most one tick.
- **Production v2 trigger:** move to (c) when one of these becomes a requirement: Docker
  inside the sandbox, a measured cold start that hurts, or suspend/resume across user
  idle time. Use Lambda MicroVMs if it has reached `ca-central-1` by then, else self-run
  Firecracker on nested-virt c8i/m8i.
- **Not chosen:** (a) unless Radient already runs EKS for other reasons. (d) costs the
  same order as (b) ($0.091 vs $0.087 per task-hour) but boots in minutes and needs an AMI
  pipeline and a reaper. Keep (d) as the **fallback if Docker-in-sandbox becomes a v1
  requirement before (c) is ready**: it supplies Docker and an EBS workspace today with
  no new substrate. As written, Docker-in-sandbox is **out of v1 scope** (D5).

---

## 5. Control plane, mapped onto the pool contract

### 5.1 What maps cleanly

| Contract element (`mesh-compute-pool.md`) | Cloud-agent meaning |
|---|---|
| `PoolRequest` (§4.1) | The client-side record of a submitted cloud task. Its `request_id` = the control plane's task id; `size_class`, `max_hours` come from flags |
| Pool grant + A8.1 attestation (§3.2) | The control plane signs `{pod_pubkey, home_device_id, home_pubkey, radient_account, network_id, grant_id, size_class, provider_ref, relay_route, expires_at}` (`relay_route` = the blind-relay rendezvous id, or the public endpoint in the POC). Home verifies it against the configured control-plane key and writes the member row **from the signed fields**. The signature means "the control plane provisioned this pod for this account and grant, and the pod reported this key". It does **not** mean the provider minted the key; per compute-pool OQ4 the pod mints it, and A8.1's §3.2 wording should be aligned to that |
| `MemberRecord` delta (§3.3) | `provider: "radient"`, `provider_ref: <ECS task ARN>`, `size_class`, `expires_at`, `max_session_seconds`, `max_sessions: 1`, `ephemeral: true` |
| Placement `mode: "pool"`, `home_device` (§4) | Produced by `exec --cloud` / `/new cloud`; `home_device` = the requesting device |
| Drain → flush → `expired` (§3.4, §6.2) | ECS task stop after the barrier; the control plane's `StopTask` is the backstop |
| `meter_interval`/`meter_close` (§5.2) | Kept as the pod-attested cross-check and for per-session attribution (`sessions[]`) |

### 5.2 Where the contract has to change

| # | Contract assumption | Why it fails for a hosted agent | Proposed change |
|---|---|---|---|
| **C1** | A pool member's capabilities "never" include `broker_credential` (compute-pool §3.5), yet it "borrows" credentials through the broker (credentials §6.2) | As built, `broker_credential` **is the borrower's capability**: `OP_CAPABILITY["net_broker"] = "broker_credential"`, described as "borrow this device's logins" (`network/types.py`; transport §7 table "ask this device's credential broker for a token"). A pool member built exactly to §3.5 cannot borrow anything, so it cannot call a model with a BYO key. Capabilities are flat names (`CAPABILITIES`, `ROLE_CAPABILITIES` = `read`/`drive`/`admin`; `drive` is exactly §3.5's pool set), so "broker_credential for these keys only" has nowhere to live in the capability itself | **Default: keep §3.5 as written** and use the hosted key (§6). **BYO opt-in:** add `broker_credential` to the pool row's *admit-time capability list* (the row stores resolved capabilities, so no new role is needed), and put the key restriction where restrictions already live: the owner's grant machinery (`holders` entries with `scope: "session"`, `GRANT_TTL_S`; credentials §6.2) plus an explicit key list on the pool grant. No scoped capability name is invented |
| **C2** | Admission is a join: the joiner dials the inviter, proving possession of an invite token that **carries the epoch secret** (§3.2). Every later link is member-mode, authenticated by a MAC under that secret | Home is usually NATed. Any path that puts the epoch secret on the pod, whether via the control plane or from home after admission, lets the pod mint invites (`_ctl_invite`), including `admin` | A **`pool` handshake mode**, used for admission *and* for every later link of a pool member. The pod **never holds the epoch secret**. **Reused:** the hello/challenge/auth transcript order; step 7's signature against the stored row key (`verify_auth`); the epoch-number, tombstone and trust checks; the capability chokepoint. **Replaced:** the epoch-key MAC (`verify_auth`'s member-mode step) becomes "signature by the attested key, plus the attestation's digest bound into the transcript" (otherwise one attestation could be replayed onto another link; `link_keys` binds only hello/challenge/auth today). **Added:** (a) home verifies the control-plane attestation and writes the row with `kind="pool"` and compute-pool §3.3's seven keys **before** `verify_auth` step 6 (which resolves the peer key from the member row); the pod learns home's public key from the attestation (`home_pubkey`, which home supplied when it minted the grant), because `welcome` deliberately omits the listener's key and the join path's `pair_result` member list is not used. **A `pool`-mode handshake must authenticate both ends.** Today only the dialer signs (`auth` is dialer → listener), and the dialer trusts the listener because the listener could check a MAC under the epoch key; pool mode removes that MAC, so the listener must also sign the transcript (a signed `welcome`, or a bidirectional `auth`), and the pod checks that signature against `home_pubkey`. Without it, a pod that dials the blind relay would accept commands from whoever answers; the auth frame becomes signature-only in this mode (today the MAC is mandatory). (b) **Rotation withholds material from pool rows:** the built `epoch_frame` sends `secret` to every active member, and `_broadcast_epoch` queues one for every active member, so without a carve-out the next rotation hands the pod the secret. Add `member.kind == "pool"` to the withhold rule and the outbox path. (c) **Number-only epoch advance:** the pod's own authorizer refuses frames unless `link.epoch == record.epoch` (`_check_epoch`), and `apply_epoch` refuses a frame with no `secret` (`secret_missing`). A pool row therefore needs a path that adopts the new epoch *number* without material. A pool member links **only** to its home device (through the blind relay), never to other members; a user's second device views the session through home, not directly. Either side may dial. This is new code in `handshake.py`/`relay.py`. Tests: no frame or outbox entry addressed to a pool row ever carries `secret`; a pool install has no secrets file, so `store.require_secrets` refuses any invite mint; `net_epoch` is refused by capability (`OP_CAPABILITY["net_epoch"] = "admin"`); an attestation does not verify on a second link; a rotation leaves the pool link working. **Sibling amendments this requires:** compute-pool §3.1/§3.2 ("the identical code path", "an invite token with `kind: \"pool\"`", "the same epoch check every member passes") and transport §12.4 ("when pools arrive they will pass `--kind pool`") |
| **C3** | "`home_device` … is never the member itself — … a 'home' that can die is not a home" (§4). R22 flushes to home, and if the drain barrier times out after 900 s the result is `reason: "lost"` (§6.2) | The product promise is that the laptop can be closed. A pod finishing at 3 a.m. with home asleep would lose its tail | Add a **custodian sink**: on drain, the final flush also goes to a control-plane object store (per-account prefix, KMS, TTL), and home pulls it on the next connect. The custodian is storage, not a member, so it holds no session authority |
| **C4** | The control plane reads `PoolRequest` from `<config>/network/pool-requests/` (§4.1) | The control plane is remote | The file stays as the client's record. Submission is an **HTTPS call from the CLI with the Radient login** (the `lop tunnel` precedent), never from the relay |
| **C5** | Metering is pod-emitted and home-verified (§5.5); the drain barrier waits for every flush **and every `meter_push`** ack, up to `SYNC_DRAIN_DEADLINE_S` = 900 s (§6.2) | The control plane has AWS's own timestamps, and a barrier that waits on a sleeping laptop bills ~15 minutes per run for nothing | The billable record is AWS-observed `wall × size`. Pod events are the cross-check and the per-session attribution; `meter_dispute` stays. **Amend §6.2 together:** power off once the final flush is acked by home *or* the custodian (C3), plus a short fixed best-effort window for the last `meter_push` (seconds, not minutes) |
| **C6** | The synced session at home keeps "the same id" (§3.6 step 5) | As built, R22 promotes a replica **as a fork with a new id**, and this is code, not prose: `sync.py` `promote_replica` raises `SyncRefused("in_progress")` when `new_id == session_id`, because "an EC2 instance restarts" (module docstring) | A **code change with a test**, not a policy note: allow same-id adoption only when the control plane has observed the task in terminal state (`STOPPED`, keyed to `provider_ref`) and that fact reaches home as a signed close event. Keep compute-pool §6.5 rule 2 (refuse to engage a cold copy while the member is alive) unchanged. The test proves two writers cannot arise |
| **C7** | The contract moves *sessions* only | A cloud task also needs a **workspace** (repo, SHA, bundle, setup script) and produces **results** (branch, PR, artifacts) | The workspace spec travels **once**, in the control plane's `RunTask` override; the pod records what it *actually* cloned into the session (one writer, derived). Add a `results` record (branch, commit SHA, PR URL, artifact refs) on the session. Additive |

---

## 6. Model and provider credentials on the pod

| | **Hosted key via the Radient gateway** (recommended default) | BYO key via the mesh broker |
|---|---|---|
| Built today | Radient is a provider (`providers/registry.py` `radient`, `radient-key`) fronting OpenRouter (`providers/clients.py`). There is no per-task token mint | Broker built (`net_broker`); provider logins and API keys are offered to device members (`offers.py`) |
| Home offline | Works | Stalls within ≤15 min (§3.4) |
| Secret on the pod | A per-task Radient token, **budget-capped and expiring at `expires_at`**, injected as an ECS task-level secret. ECS delivers that as a container environment variable, so it is **not on disk but is readable by the agent and anything it spawns**; v1 should have the lop entrypoint read it into the process and unset it before tools run. A leaked token is worth at most the task's remaining budget | A bearer valid ≤900 s, in memory only (credentials §6.2) |
| Metering | Exact. The gateway already meters tokens in Radient credits | Tokens are billed to the user's own provider account. Radient meters compute only |
| Data path | Model traffic goes Radient → OpenRouter → upstream provider. **Not Canada-resident** (§7.6) | User's chosen provider |
| Work needed | Control-plane endpoint to mint and revoke task-scoped gateway tokens with a spend cap | C1, plus documenting the home-online requirement |

**Bedrock as a hosted path** (for residency or enterprise): a read-only
`bedrock list-foundation-models --region ca-central-1` (2026-10-06) lists current Claude
models, but `claude-sonnet-4-6`, `claude-opus-4-8` and `claude-haiku-4-5` report
`inferenceTypesSupported: INFERENCE_PROFILE` only. They are served through cross-region
inference profiles, so **calling from `ca-central-1` does not guarantee in-Canada
inference** [B1]. lop has no Bedrock provider today (§2).

---

## 7. Security

### 7.1 What sits on the pod, and what a prompt injection can reach

On the pod: a clone of one repo at one SHA (plus an optional bundle); a lop install; the
pod's own device key; a per-task model token (budget-capped); a **scoped git credential
that only the git proxy honours**; nothing else. The ECS **task role has no AWS
permissions**. The execution role (image pull and logs) is not reachable from inside the
container.

| Attack | Blast radius | Control |
|---|---|---|
| Exfiltrate repo contents | Whatever egress allows | Egress allowlist (§7.2). Git push only to the task branch through the proxy (Claude pattern [A1]) |
| Burn money | The task's remaining budget | Gateway cap + `expires_at` + control-plane budget stop |
| Push malicious code | The task branch only; a human merges | Proxy enforces repo + branch. Branch protection on the default branch |
| Pivot into the user's mesh | Sessions on this member only (compute-pool §3.5) | Pool capabilities: no `admin`, `trust`, `delete`, `move`. No visibility of other sessions. **No epoch secret on the pod** (C2), so it cannot mint invites or open member-mode links to other devices. Row removal is built |
| Pivot into AWS | None by design | Empty task role: the ECS container-credentials endpoint (169.254.170.2) does serve the task role's credentials, but that role has no policies. Fargate exposes no EC2 IMDS. Security group denies VPC-internal destinations |
| Compromise of the control plane | **Highest-value target.** It can launch pods into any user's network as admitted pool members (it signs attestations) and can mint gateway tokens. It cannot read the mesh epoch secret, a device's private key, or a home device's own credentials | Attestation key in KMS (sign-only, no export), short attestation TTL bound to one `grant_id`, the human mints the grant on their own device (A8.1), per-user `pool_cap`, audit of every signature; a user can remove the control-plane key from their relay config to refuse all pool admission |
| Persist | None | Ephemeral task, nothing written outside the task's storage, image pinned by digest |

### 7.2 Egress

v1 (blind-relay path, §3.2 option A): tasks in private subnets with **only** a security
group to an egress proxy (an Envoy/Squid task with a domain allowlist: package
registries, the Radient gateway, the git proxy) plus VPC endpoints (ECR, S3, Logs). The
default allowlist follows Codex's "Common dependencies" idea [O1] and Copilot's default
firewall [G2]. An agent that tries an unlisted domain gets a clear refusal recorded in
the transcript. AWS Network Firewall is the managed alternative at
**$0.395/endpoint-hour + $0.065/GB** in `ca-central-1` (Price List API) ≈ $288/month per AZ. That is overkill
for the POC and worth evaluating for production. A sidecar proxy inside the same task is
**not** a boundary, because the agent can bypass it.

### 7.3 Tenant isolation

One Fargate task per run, per user. No shared volumes. Per-account object prefixes with
KMS keys. Control-plane authorisation is always `(radient_account, network_id)`, never
the task id alone.

### 7.4 Audit

Three streams, each with an owner: the mesh JSONL audit on home and pod (built; pool
lifecycle events reserved in incident-response §Q7); the control-plane event log (task
created, attested, admitted, stopped, charged); CloudTrail for every AWS API call.
Pod logs go to CloudWatch with 7–30 day retention.

### 7.5 Kill switches, from narrowest to widest

1. Session: `net_session_stop` (built).
2. Task: control-plane `StopTask`.
3. Member: `remove_member` (built). For a pool member the row removal alone refuses its
   next link, since it holds no epoch secret (C2); the built epoch rotation still runs.
   Broker grants die within ≤900 s.
4. User: revoke the gateway tokens and stop all of the user's tasks.
5. Global: a feature flag in the control plane that refuses `RunTask`, plus an IAM
   deny on the controller role. The mesh's panic is not a remote kill
   (incident-response) and should not be stretched into one.

### 7.6 Data residency

Compute, storage and logs stay in `ca-central-1`. **Model inference does not**, unless a
Canada-resident model path is chosen (§6). The user-facing claim must say so. Managed
sandbox vendors (E2B, Daytona, Modal, Fly) and Lambda MicroVMs (no `ca-central-1`) would
move compute out of Canada as well.

---

## 8. Cost model

### 8.1 Compute per task-hour (2 vCPU / 4 GiB)

| Option | $/task-hour | Source |
|---|---|---|
| ECS Fargate ARM | **0.0869** | Price List API `AmazonECS` ca-central-1 (SKUs `MA9ZGYPG5E2CH64A` vCPU $0.03565, `C96MZB6RGQHW2XHW` GB $0.00389) |
| ECS Fargate Spot ARM | ≈0.026 (up to 70% off; estimate) | ECS pricing page |
| EKS on shared m7g.xlarge, 2/node | 0.092–0.101 + disk ≈ 0.094–0.103 | Price List API (m7g.xlarge $0.1819; EKS $0.10/cluster-h) |
| Firecracker on c6g.metal (24–32/host) | 0.074–0.099 (+ our own control plane; host must run) | Price List API ($2.3808/h) |
| Firecracker nested-virt c8i.xlarge (2/host) | 0.103 | Price List API ($0.20506/h) |
| EC2 m7g.large per task | 0.091 | Price List API |
| E2B (for comparison; hosted outside our account) | ≈0.166 | e2b.dev/pricing [E1] (estimate) |

### 8.2 Fixed and idle (monthly, 730 h)

| Item | $/month | Applies |
|---|---|---|
| NAT gateway | 36.50 + $0.05/GB | all private-subnet designs (Price List API: $0.05/h, $0.05/GB) |
| Interface endpoints (ECR api+dkr, Logs, S3 gateway free) | ~8 per endpoint per AZ ($0.011/h) | recommended so image pulls skip NAT |
| Public IPv4 for pod endpoints (§3.2 option B, POC Slice 1 only) | $0.005/h per running task | per task-hour, ≈ +$0.005 |
| Blind relay (§3.2 A; e.g. 2× small Fargate tasks behind an NLB) | ~$30–50 (estimate) | v1 production |
| EKS control plane | 73 | (a) only |
| Egress proxy (2× t4g.nano / Fargate 0.25 vCPU) | ~7–20 | all; Network Firewall alternative ≈ $288 per AZ |
| Control plane (API Gateway + Lambda + DynamoDB, low volume) | single-digit dollars (estimate) | all |

### 8.3 Storage per task

ECR image ~1.5 GB → $0.15/month total (`$0.10/GB-mo`). Fargate ephemeral storage: 20 GB
included, then $0.000122/GB-h. Custodian copy and artifacts: S3 Standard
$0.025/GB-month. A typical transcript is under a megabyte (compute-pool §6.3 cites
216 KB as the largest in the operator's store), so storage is effectively free at a 30-day
TTL.

### 8.4 What dominates

Compute is **cents per hour**. Model spend for an hour of agentic coding is commonly
**dollars** (illustrative: at a $3 per million input-token list price, one million
tokens costs the same as ~35 Fargate task-hours). Pricing and budget UX should therefore
be designed around model spend, with compute as a small line item. Billing in **Radient
credits** follows the tunnel precedent (`docs/tunnels.md`): positive balance required to
start, a quote shown before acceptance, actual infrastructure cost plus the configured
margin (the tunnel doc states an 80% gross margin). For a cloud task:
`charge = wall_seconds × size_rate + gateway model spend`. A **pre-authorised hold** of
`max_hours × size_rate + budget` is taken at submit and released at close.

---

## 9. POC plan (requires separate approval; nothing here has been run)

### 9.1 Account, first

Read-only facts (2026-10-06, `AWS_PROFILE=minerva_sandbox`, identity verified):

- Account 325492156725, IAM user, `ca-central-1`.
- It belongs to the AWS Organization `o-rxz2wexmo9`, whose management account is
  212841448981. That is the account `minerva_nprod` resolves to.
- It has a Pulumi-managed `sbx-vpc` (10.10.0.0/16) with one NAT gateway.
- It has two Pulumi-managed ECS clusters (`discovery-initial-agent`,
  `discovery-initial-index-agent`, 0 running tasks).
- There are no EKS clusters, but leftover `eksctl`/Karpenter CloudFormation stacks
  exist.
- There are no running EC2 instances.

The same account is documented as **Pergamon's sandbox destination**: changes go only
through `pergamon-infra-services` → `sandbox-aggregate` Pulumi preview → reviewed merge →
manually dispatched "Pulumi Apply". Ad-hoc mutation from a laptop "is not the normal
path". The typed capabilities listed there (S3, ECR, RDS, Redis, Kafka) **do not include
ECS task definitions or IAM roles**, so a POC there likely needs a contract extension.
**Decision D1** picks the account. This doc does not assume one.

### 9.2 The smallest slice that ends at both ends

- **Slice 0 — cloud run, no mesh.** Proves AWS lifecycle, isolation, credentials and
  results. Needs no product code.
- **Slice 1 — mesh-attached.** Proves live view and steer, sync and admission. It needs
  the C2 `pool` handshake mode, a minimal attestation verifier, and `exec --cloud`
  placement. That is product code, specified here, built later. To stay small it uses
  §3.2 option B (public subnet, public IPv4, inbound SG on the peer port only, home
  dials the pod) and resolves the task ENI's public IP with `ec2:DescribeNetworkInterfaces`
  in the driver, which passes it to home inside the signed attestation. The blind relay
  (option A) is P2 work.

**Slice 0 steps:**

1. IaC (one stack, D7) creates:
   - an ECR repo;
   - a task definition: ARM64, 2 vCPU / 4 GiB, read-only root filesystem except the
     workspace, non-root user;
   - an execution role (ECR pull, Logs) and an **empty task role**;
   - a log group with 14-day retention;
   - a security group: egress 443 only, no inbound. The ECS container-credentials
     endpoint (169.254.170.2) is link-local and served by the Fargate agent, so this rule
     does not block it; probe 4a below checks it;
   - an S3 bucket for results: private, KMS, versioned, 7-day lifecycle;
   - a controller IAM role allowing only `ecs:RunTask`/`StopTask`/`DescribeTasks` on
     this one task definition, `iam:PassRole` for those two roles, and `s3:PutObject`
     presign. Everything is tagged `lop-poc=true`.
2. Image: `python:3.12-slim` + git + uv + `local-operator==<released version>` from PyPI,
   pinned by digest.
3. A POC driver script (outside the product, under `scripts/`, run with the controller
   role) does the following:
   - uses a **pinned fixture**: a small public repo created for the POC, a fixed SHA,
     and a fixed prompt ("make the failing test `test_add` pass"), so the outcome is
     checkable;
   - mints presigned S3 PUT URLs;
   - fetches a POC model key from Secrets Manager, **injected as a task-level secret
     (in memory, not on disk)**. This is a POC stand-in for the gateway token;
   - calls `RunTask` with `--overrides` carrying the repo URL, SHA, prompt and the URLs.
4. Container entrypoint:
   - clones the public test repo at the SHA;
   - runs the isolation probes (4a–4c below) **from the entrypoint script, before the
     agent starts**, and writes their results to a JSON file (a script is a deterministic
     instrument; a prompt is not);
   - runs `lop exec --json --tools read,write,edit,bash "<prompt>"` with an isolated
     config dir;
   - commits to `lop/<id>`;
   - uploads `git bundle` + the session directory + `exec --status` JSON to the presigned
     URLs;
   - exits.
5. The driver waits for `STOPPED`, downloads, verifies, and confirms teardown.

### 9.3 Acceptance test (all must pass on one recorded run)

1. `aws ecs describe-tasks` shows one task, ARM64, `lastStatus: STOPPED`,
   `stopCode: EssentialContainerExited`, exit 0. **No task with tag `lop-poc` is
   running** afterwards.
2. The bundle verifies (`git bundle verify`). `git log` shows one commit on `lop/<id>`
   whose parent is the fixture SHA. `test_add` fails at the fixture SHA and passes on the
   fetched branch when run locally.
3. The session directory, copied into an isolated local config root, opens with
   `lop --resume <id>` and shows the full transcript.
4. The entrypoint's probe file shows:
   - (4a) `curl http://169.254.170.2$AWS_CONTAINER_CREDENTIALS_RELATIVE_URI` returns
     credentials whose role has **no** policies (verified by `sts get-caller-identity`
     succeeding and `s3 ls` being denied);
   - (4b) outbound to a non-443 port fails;
   - (4c) no secret is present on the filesystem (`grep -r` for the key prefix finds
     nothing).
5. Cold start (RunTask → `RUNNING`, and → first model call) measured for 5 runs; results
   recorded as evidence and replacing the estimates in §4.
6. Cost: Cost Explorer for the tag (after the 24 h lag) reconciles within 20% of
   `Σ wall_seconds × $0.0869/3600`.
7. Teardown of the POC stack (only after approval): `pulumi destroy`/`cdk destroy`
   leaves zero tagged resources.

**Slice 1 adds:**

8. The pod appears in `lop network peers` as `kind: pool`, `lifecycle: active`.
9. The session appears in the TUI sidebar as a remote row and can be steered mid-turn.
10. After drain the transcript is at home and the member is `expired`, with
    `pool_member_expired` in the audit log.
11. `lop network member rm` during a run refuses the pod's next frame (built revocation
    path).
12. The control plane never saw the epoch secret: the driver's logs, its stored state and
    the task overrides contain no invite token. A pod presenting a valid attestation for a
    **different** `network_id` is refused, and so is one whose attestation has expired.

### 9.4 POC cost estimate

| Item | Estimate |
|---|---|
| Fargate ARM, 20 runs × 0.5 h | 20 × 0.5 × $0.0869 ≈ **$0.87** |
| ECR storage (1.5 GB, 1 month) | $0.15 |
| NAT data (if private subnet, ~1.5 GB pull × 20 via NAT) | ≈ $1.50; $0 hourly if the existing `sbx-vpc` NAT is reused |
| Logs, S3, Secrets Manager | < $1 |
| **AWS total** | **< $5** |
| Model spend | **capped at ~$20** (POC key with a hard spend limit) |

---

## 10. Phasing after the POC

| Phase | Adds | Depends on |
|---|---|---|
| P1 | Slice 1 (mesh-attached): C2 `pool` handshake mode, attestation verify, `exec --cloud` → `PoolRequest`, `MemberRecord` delta, `lop cloud status/cancel` | POC pass |
| P2 | Control-plane service: Radient auth, quota and budget holds, EventBridge lifecycle, reaper, task-token mint at the gateway, git proxy (GitHub App), custodian sink (C3), C5 barrier change, C6 adoption, credits charging, **blind relay** (§3.2 A; the production link path, which also lets the phone view with the laptop closed) | D2, D3, D4, D9 |
| P3 | Egress proxy tier and allowlist UX; artifacts; draft PR; notifications; cost meter in the status line; web/mobile launch | P2 |
| P4 | Firecracker/Lambda MicroVMs for snapshots, suspend/resume and Docker | demand + D5 |

---

## 11. Decisions for the operator (each with my recommended default)

| # | Decision | Recommended default | What would change it |
|---|---|---|---|
| **D1** | **Which AWS account hosts the POC (and later production)?** 325492156725 is in the Minerva-managed org and is documented as Pergamon's sandbox | **A dedicated Radient-owned account** (lop is Radient's product). If the POC must use 325492156725, get the owner's explicit OK and go through its Pulumi `sandbox-aggregate` path with an ECS/IAM capability extension | The operator confirming 325492156725 is intended for lop work |
| D2 | Who runs the control plane, and with which identity | Radient, behind the user's Radient login; the mesh device id is bound per request | lop staying self-hosted only (then a per-user "bring your own AWS account" control plane) |
| D3 | Model credentials default | Hosted via the Radient gateway, task-scoped, budget-capped; BYO through the broker as an opt-in documented as "home must stay online" | A residency requirement (then Bedrock with in-region-only models, where available) |
| D4 | Git write path | A control-plane git proxy holding a GitHub App installation token, enforcing one repo and one branch | A requirement to support GitLab first (proxy works the same; App → project token) |
| D5 | Substrate | Fargate ARM for POC and v1; Firecracker (Lambda MicroVMs if it reaches `ca-central-1`, else self-run on nested-virt) for v2 | Docker-in-sandbox being a v1 requirement |
| D6 | Surface naming | `lop exec --cloud`, `/new cloud`, "Cloud" in placement pickers | — |
| D7 | IaC | Pulumi (the sandbox account's mandated tool) for the POC; the same for production unless Radient standardises elsewhere | The chosen account's own IaC rules |
| D8 | Region and residency claim | `ca-central-1` for compute, storage and logs; state plainly that model inference may leave Canada | A customer requirement for Canadian inference |
| D9 | Network path for v1 | A blind relay we host (§3.2 A), both ends dial out, pods stay in private subnets; a public pod endpoint only in POC Slice 1. The pod never holds the network's epoch secret, by token or by delivery (C2) | Unwillingness to run a relay service (then v1 is public pod endpoints, with the inbound exposure and egress caveats in §3.2 B) |
| D10 | Price | Compute at actual infrastructure cost plus the tunnel margin rule, per size class per second; model at gateway prices; required per-task budget | Product pricing strategy |
| D11 | Default per-task limits | `--max-hours 2` hard stop; idle reap 15 min after the pod has **no turn in flight and no pending attention item** (independent of whether anyone is viewing, so a closed laptop does not cause a reap); budget required; `max_sessions: 1` | Measured usage in P2 |

---

## 12. Open questions (answerable by the POC or a spike, not by the operator)

1. Measured Fargate cold start for a ~1.5 GB lop image, with and without SOCI.
2. Can `lop exec` run usefully with a read-only root filesystem and only the workspace
   writable? (Config root, uv cache and scratch locations need checking.)
3. Slice 1 (direct path): the pod's relay with `network.listen_address = 0.0.0.0`
   behind a public IPv4. Does the duplicate-link dedupe (transport §6) behave when both
   sides dial? For the blind relay: what framing does the forwarder need so that the
   handshake runs end-to-end through it without the relay reading it?
4. Size of a typical session copy-set at drain, which sets the custodian upload time and
   `SYNC_DRAIN_DEADLINE_S`.
5. Whether `lop exec`'s headless gate plus `--tools` covers the toolset agents actually
   need without `--yolo` (`docs/EXEC.md` "Approvals and lifetime").
6. The `pool` handshake mode (C2): exact transcript binding for the attestation digest,
   the attestation-then-row ordering ahead of `verify_auth` step 6, the signature-only
   auth frame, the listener-side signature (signed `welcome` or bidirectional `auth`) so
   both ends authenticate, pinning `home_pubkey` from the attestation, the rotation
   carve-out, and the number-only epoch advance. This needs a design note amending
   `mesh-transport-identity.md` §6/§12.4 and `mesh-compute-pool.md` §3.1–3.2 before any
   P1 code. The verdict that it is implementable rests on the authorizer needing no
   secret: the pool capability set's `list` already admits
   `net_reconcile`, and the absence of `admin` already refuses `net_epoch`.

## 13. Doc-hygiene findings (stale comments; out of scope here, noted for a follow-up)

- `network/relay.py` module docstring still says `net_sync`, `net_broker` and
  `net_session_move` are "NOT IN THIS SLICE", but all three are registered and
  implemented (slice modules `sync`, `credentials`, `mobility`).
- `network/cli.py` `approvals` help comment says `run` "refuses truthfully while this
  build ships no install runner", but `onboard.step_runner` exists and is wired.

---

## Sources (accessed 2026-10-06)

- [C1] Cursor, Cloud Agents — https://cursor.com/docs/cloud-agent
- [C2] Cursor, Cloud Environment Setup — https://cursor.com/docs/cloud-agent/setup
- [C3] Cursor, Cloud agent capabilities — https://cursor.com/docs/cloud-agent/capabilities
- [C4] Cursor, Secrets & Network (Privacy Mode requirement, runtime/build secrets, signed
  commits, git egress proxy) — https://cursor.com/docs/cloud-agent/security-network.
  Cursor does not name its cloud. Its published egress and git-proxy IPs (e.g.
  54.184.235.255, 184.73.225.134) fall in AWS EC2 ranges for us-west-2/us-east-1
  (checked against https://ip-ranges.amazonaws.com/ip-ranges.json). This is an inference,
  not a Cursor statement.
- [A1] Anthropic, "Beyond permission prompts" (git proxy; credentials never inside the
  sandbox) — https://www.anthropic.com/engineering/claude-code-sandboxing
- [A2] Claude Code, Use Claude Code in the cloud (`--cloud`, `--teleport`, bundles,
  inactivity reclaim) — https://code.claude.com/docs/en/claude-code-on-the-web
- [O1] OpenAI Codex cloud, internet access (off by default during the agent phase;
  allowlist; prompt-injection example) — https://learn.chatgpt.com/docs/cloud/internet-access
- [G2] GitHub Copilot, customize the firewall ("By default, Copilot's access to the
  internet is limited by a firewall"; recommended allowlist on by default) —
  https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/customize-the-firewall
- [N2] AWS What's New, "Amazon EC2 supports nested virtualization on virtual Amazon EC2
  instances" (posted 2026-02-16) —
  https://aws.amazon.com/about-aws/whats-new/2026/02/amazon-ec2-nested-virtualization-on-virtual/
- [L5] AWS What's New, "AWS introduces Lambda MicroVMs for isolated execution of user and AI-generated code" (posted 2026-06-22) —
  https://aws.amazon.com/about-aws/whats-new/2026/06/aws-lambda-microvms/
- [G1] GitHub Copilot cloud agent (ephemeral Actions-powered environment; one repo, one
  branch, one PR; 59-minute cap) —
  https://docs.github.com/en/copilot/concepts/copilot-surfaces/copilot-on-github
- [D1] Devin environment (VM per session booted from a snapshot; blueprints) —
  https://docs.devin.ai/onboard-devin/environment
- [J1] Jules environment (short-lived VM per task; setup script; Run and Snapshot) —
  https://jules.google/docs/environment
- [E1] E2B pricing — https://e2b.dev/pricing (estimate)
- [F1] AWS, What is AWS Fargate (isolation boundary) —
  https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html
- [F2] AWS News Blog, Fargate + Seekable OCI —
  https://aws.amazon.com/blogs/aws/aws-fargate-enables-faster-container-startup-using-seekable-oci/
- [F3] AWS News Blog, Firecracker ("powering … AWS Lambda and AWS Fargate") —
  https://aws.amazon.com/blogs/aws/firecracker-lightweight-virtualization-for-serverless-computing/
- [F4] Firecracker (KVM; <125 ms boot; <5 MiB overhead) — https://firecracker-microvm.github.io/
- [F5] AWS, Fargate security considerations (no privileged containers / Docker-in-Docker) —
  https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-security-considerations.html
- [K1] Karpenter node provisioning latency (issue discussion; secondary) —
  https://github.com/aws/karpenter-provider-aws/issues/2906
- [K2] EKS Pod Identity considerations (Fargate pods not supported) —
  https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html
- [K3] EKS network policies (not applied to Fargate pods) —
  https://docs.aws.amazon.com/eks/latest/userguide/cni-network-policy.html
- [N1] EC2 nested virtualization (families; no extra cost) —
  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/amazon-ec2-nested-virtualization.html
- [L1] AWS Lambda MicroVMs developer guide — https://docs.aws.amazon.com/lambda/latest/dg/lambda-microvms-guide.html
- [L2] Lambda MicroVMs as a sandbox for Cursor Cloud Agents —
  https://docs.aws.amazon.com/lambda/latest/dg/microvms-integrations-cursor-self-hosted-machines.html
- [L3] Lambda MicroVMs, 5 additional regions (2026-08-19; 10 total) —
  https://aws.amazon.com/about-aws/whats-new/2026/08/lambda-microvms-5-additional-regions/
- [L4] InfoQ, AWS launches Lambda MicroVMs (launch regions; ARM64 up to 16 vCPU/32 GB;
  the cost remark is a quoted Reddit comment, secondary) — https://www.infoq.com/news/2026/06/aws-lambda-microvms/
- [B1] AWS Bedrock cross-region inference in Canada —
  https://aws.amazon.com/blogs/machine-learning/accelerate-generative-ai-innovation-in-canada-with-amazon-bedrock-cross-region-inference
  (plus the read-only `list-foundation-models` result above)
- [P] AWS Price List API (`AmazonECS`, `AmazonEKS`, `AmazonEC2`, `AWSNetworkFirewall`,
  region `ca-central-1`) read via `aws pricing get-products` and the public offer files
  (https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/…/current/ca-central-1/index.json).
  EBS gp3 ($0.088/GB-mo) and S3/ECR figures come from the scout's Price List read and are
  estimates where marked.
