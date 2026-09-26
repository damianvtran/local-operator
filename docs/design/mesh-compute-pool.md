# Metered on-demand compute on the mesh — the forward-compatibility contract

Status: **design, pre-implementation**. Nothing in this document is built in the
mesh pass it belongs to. Its whole job is to fix the *shapes* so that metered
on-demand capacity — **R20, R21 and R22, which this document owns** (spine's
table), plus the `pool` producer of the spine's **A8** placement — arrives later
as implementations of interfaces named here rather than as a refactor of what a
session, a member, or a record is.

Branch: `feat/mesh-network` @ `a7e6b9bd`, cut from `origin/main`. Every
`file:line` anchor below was read **at that revision in that worktree**;
`main` has moved, so a few numbers differ from the recon notes attached to this
work (measured: `SessionRecord` is `types.py:712` here against `:567` on
`~/local-operator` main). **Symbol names are the stable citation** — re-anchor
by symbol, never by the number alone.

Read `docs/design/mesh-network.md` first: it is authoritative for *what* the
mesh must do, and §9 of it is the parent of everything below. This document
never restates its requirements and never weakens them.

---

## 1. The boundary this document keeps

Four things are decided here and nothing else:

1. **Where placement lives**, so a scheduler can move a session without
   touching session semantics (A8).
2. **What an ephemeral, provider-owned member IS** in the membership model, and
   how its lifecycle differs from a user's own device (§3).
3. **The metering event stream** — its literal schema, its attribution, and why
   billing consumes it (R20, R21).
4. **Sync-before-spin-down as the mobility primitive** (R22).

### 1.1 Interfaces this document requires from its sibling designs

The sibling designs are being written in parallel and this document must not
fork their vocabulary. It requires exactly six names; if a sibling chooses a
different spelling, align the *name* and keep the *shape* — the shapes below are
the expensive part, names are cheap:

| Required from | Name | Shape required here |
|---|---|---|
| `mesh-transport-identity.md` | `MemberRecord` | §3.3 field list (a `kind`, a `lifecycle`, and ephemerality are fields, not assumptions) |
| `mesh-transport-identity.md` | member record namespace | a namespace that is **not** `run/mobile` (A2, `types.py:239`) |
| `mesh-transport-identity.md` | role/capability grants | per-capability grants, so a pool member's set can be smaller than a device's |
| `mesh-session-mobility.md` | the move primitive | the five steps of A4, with step 3 (copy) callable on its own |
| `mesh-credentials.md` | the broker | request-scoped tokens over the peer link; a pool member holds no credential |
| `mesh-incident-response.md` | the audit channel | A7's append-only JSONL, one record per semantic event, bounded rotation |

If `mesh-incident-response.md` names the audit record differently, this
document's `metering` kind is a value in **that** record's type field, not a
second log. A second log is the failure this boundary exists to prevent.

### 1.2 Reconciliation with the siblings, as landed (2026-09-19)

The five sibling designs landed while this one was being written. Every interface
this document *requires* is now owned by one of them; where my first draft
invented a name, the sibling's name wins and this document uses it.

| My subject | Owner | What changed here |
|---|---|---|
| The member record | `mesh-transport-identity.md` §4.2, §12.4 | uses its field names (`name`, `role`, `capabilities`, `kind`, `lifecycle`, `added_via`, `removed_at`, …) and states its own additions as an explicit **delta** (§3.3). Its pairing path already ships `--role drive --automated` and already refuses to assume a human at the far end, so **A8.1 is a naming and policy decision here, not a new mechanism** |
| Placement | `mesh-session-mobility.md` §5.1–5.2 | uses `SessionPlacement` / `mesh.json` / `placement{mode, network_id, home_device, policy, stamp_revision}` verbatim. This document's whole contribution is that `mode: "pool"` becomes **producible** and that the drain rewrites it (§4) |
| The sync primitive | `mesh-session-mobility.md` §7 | `net_sync` — the name `mesh-transport-identity.md` §12.4 already reserves with the `view` capability — implemented as `session.sync.plan` / `session.sync.fetch` with modes `exact` / `boundary` / **`flush`**. My draft's byte-offset cursor was **wrong** and is corrected in §6.1: `Transcript.compact_file` (`session/transcript.py:1807`) rewrites the transcript, so the cursor is `(history_generation, through_entry_id, prefix_bytes, prefix_digest)` |
| Metering | `mesh-incident-response.md` §4.2–4.3 | `meter_interval` / `meter_close` are **reserved there** and their literal fields are **delegated to §5.2 of this document**. They are therefore specified inside that envelope (`lop.mesh.audit.v1`, closed-whitelist `detail`, the 2048-byte cap), not as free-standing JSON |
| Credentials on a pod | `mesh-credentials.md` §6 | cited, not restated: a pool declares no credential, every grant to it is `grant_ttl_s`-bounded and in-memory, `identity` is omitted for a pool, and `expired` withdraws its holder entry |
| Pool lifecycle events | `mesh-incident-response.md` §Q7 | Q7 recommends they ride the one audit log; this document names them (§3.4) so the taxonomy and the producer cannot drift |

**Two things this document owns outright**, because no sibling does:

1. **The metering events' literal field lists** — incident-response reserves the two names and explicitly defers their fields to §5.2 here.
2. **`meter_push` / `meter_ack`** (§5.2.3) — the delivery path that carries a member's metering records to the durable device. The audit log is **per install** (IR §4.1) and a pod's log dies with the pod, so without this the metering stream would be evidence that never leaves the machine being billed. It is the one interface this document adds to the wire.

---

## 2. The user story, event by event

The story the operator stated, in the order it happens. "Covered today" names
the decision or module that already carries the step; where the answer is
"nothing", the smallest interface that closes it later is named on the same row.

| # | Event | Covered today by | The gap, and the smallest interface that closes it later |
|---|---|---|---|
| **E1** | The user's local device is out of capacity: sessions are queued behind a busy runtime, or the machine is saturated. The *user* says so. | Nothing. A8 (`mesh-network.md` §4) says placement is explicit from day one; no producer of placements exists. | The gap is a **decision surface**, deliberately not a scheduler: `/network pool request --size <class>` (TUI/CLI) writes user intent. Deferred: any automatic detection of "out of capacity". |
| **E2** | The user asks for extra compute: `lop network pool request --size m --hours 4 --json`. | Nothing. | A **request record**, not a provisioning API: `PoolRequest` (§4.1) is written to the install's own state and is what the control plane reads. The mesh never talks to a cloud provider (§6). |
| **E3** | Capacity is provisioned: a pod boots a clean `lop` from the released wheel, in a fresh config root, with no session store and no credential store. | The install path exists (`lop-update`-shaped install, a first-run config root). Provisioning itself is out of scope. | Nothing in the mesh needs to change: the pod is a **clean install**, which is the point of R21. The only requirement is that install can run with no TTY. |
| **E4** | The pod pairs into the user's network. No human is present at the pod's end. | R3/A3: pairing requires a **human on both devices** confirming the same SAS. A pod has no human. | **A8.1** (§3.2): a pre-authorised *pool grant* — the human confirms once, at mint time, on their own device, and the pod's admission is that token plus provider attestation of its device key. Without this the story cannot be told at all; with it, R3 is unchanged for devices. |
| **E5** | The pod appears as a member with a restricted role, a size class, and a lifetime. | Nothing to model it: the member record must carry `kind`, `lifecycle`, `ephemeral`, `size_class`, `expires_at` (§3.3). | The member record's field list. It must be **additive**: an older reader that ignores these keys must still see a usable member. |
| **E6** | A session is placed on the pod, or created there. | A8's `placement` — `SessionPlacement` in the session's `mesh.json`, carried on the record (mobility §5.1–5.2) — plus R8's session creation on a peer. | **Nothing new on the session path.** The pool is reached as a peer is, with `home_device` the pool member's device id and `mode: "pool"`; the `pool` producer is §4's delta. This is the load-bearing claim of R21, and §8.1 says what would break it. |
| **E7** | The session runs on the pod: the runtime is the owner (one detached process, `sessions/<id>/`), the user's TUI/desktop is a viewer over the relay link. | The whole detached-runtime split, unchanged (`session/runtime/process.py`; `session/protocol.py:132` `SessionProtocol`, `:157` `owns_runtime`, `:191` `runtime_locality`). | Credential brokering (A5) is what makes this safe on a provider-owned device: the pod asks the user's device for a scoped token and never holds one (§3.5). |
| **E8** | Compute is consumed and **measured**: per-session CPU time and the member's wall uptime, attributed to session, network, member. | Nothing measures CPU time today. Memory is measured (`mobile/resources.py:101` `ResourceUsage`, `:306` `session_resource_usage`) and the same libproc call already reads the fields needed (§5.1). | The `metering` audit event kind (§5.2) emitted by the member's relay on an interval — **batched**, never per frame (A7, R18). |
| **E9** | The session goes idle (turn finished, no further work). | The record's `busy` bit, the heartbeat, and the drain vocabulary already exist (`types.py` `SessionRecord.busy`; the drain phrase constants beside `SIGNAL_DRAIN_CAUSE`). | Nothing new: idleness is a fact the record already publishes. What is new is that the *reason* it matters here is billing, not display. |
| **E10** | The session syncs home: its durable state is copied to the user's device. | mobility §7 — `session.sync.plan`/`fetch` with modes `exact`/`boundary`/**`flush`**, over the reserved `net_sync` op; `fork.fork_session` (`fork.py:167`) is the file-set logic it generalises, with its allow-list (`COPIED_SIDECARS`, `fork.py:117`). | The pool's *use* of it: who flushes to whom, the tick that bounds an un-drained death, and the drain barrier (§6.2–6.3). No new copy semantics. |
| **E11** | The pod is drained and spins down; the session's placement is rewritten; the local device holds a finished, resumable session. | The drain is `lop network drain` → `session.sync` in `flush` mode (mobility §7.5); the `exclusive-move-v1` fence (`types.py:93`, `server.py:1391`/`:2725`/`:3631`) covers the *move* case and deliberately not this one (mobility §7.4). | The **barrier** (§6.2): a deadline-bounded wait for the last flush **and** the last meter push to be acked, plus the pool-side guard that a cold home copy is not engaged while its member is alive (§6.5 rule 2) — which is where a second writer could actually appear. |
| **E12** | Radient credits are deducted for the compute consumed, by time and by size. | Radient is already the billing identity: `lop tunnel billing --credential-id … --json` and the desktop's `/v1/desktop/radient` route; `mesh-credentials.md` §6.3 already puts `session_id` and the credential owner on the audit stream. | Nothing in the mesh prices anything. The gap is **ingestion**: a consumer reads `meter_interval`/`meter_close` (§5.2) — which reach the user's device by `meter_push` (§5.2.3) — and applies its own price table. That table is deliberately not here (§7). |

**The honest summary.** Of these twelve steps, **two** need interfaces no sibling
design created (E8/E11's metering events and their `meter_push` delivery; E4/E5's
seven additive keys on the member row), and the rest are satisfied by machinery
that exists and is cited above — including E10, which is already an instance of
the mobility primitive. That ratio is the evidence R21 asks for: this is an
extension, not a refactor.

---

## 3. The ephemeral member

### 3.1 What is new, in one paragraph

A **pool member** is a member whose device identity was minted by a machine
nobody is sitting at, whose admission was authorised in advance by a human on
another device, whose role grants are a strict subset of a user's device, and
whose lifetime is bounded by a policy rather than by the user's decision to
remove it. It is **not** a second member kind in the sense of a second wire
shape: it is the same member record with `kind: "pool"`, a `lifecycle`, and an
`ephemeral` flag. Everything else — authentication, epoch checks, capability
authorisation, the audit trail — is the identical code path, which is the point.

### 3.2 A8.1 — how a pod is admitted when R3 requires a human at both ends

*(Naming note, convergence round 1: **`A8.1` is this document's own sub-decision
under the spine's A8, not a spine decision id** — the spine's list ends at A8, so
the dotted name is not something to go looking for in `mesh-network.md`. The name
is kept as written because `mesh-prior-art.md` §2/§6 cites it under this spelling;
the note is here so a reader does not read it as a ninth spine decision.)*

**The tension is real and must not be papered over.** A3 makes a mismatched SAS
"a refused join, not a warning", and that is what buys resistance to a relay in
the middle. A pod has no screen and no human.

**Decision (A8.1).** Admission of a pool member is authorised **once, by the
human, on their own device, at grant-mint time**, and the pod's join is accepted
on the strength of three things that a device join does not need:

1. a **pool grant** — an invite token with `kind: "pool"`, a `role`, a
   `size_class`, an `expires_at`, `single_use: true`, and the `network_id` it is
   bound to (transport's invite shape, §4.2 there; the member row it produces is
   §3.3 here). Minting it is the human's act, and the CLI/TUI prints the
   grant's fingerprint for the audit log;
2. **provider attestation** — a signed statement from the provisioning control
   plane, bound to the pod's device public key, that this key was minted by the
   provider for this user's account and this grant. The user's relay verifies it
   against the control plane's public key, which is configured once, locally;
3. the **same epoch check every member passes**, so a revoked pod is refused
   exactly like a revoked device.

The SAS screen is therefore shown **to the human at mint time**, describing the
member about to be admitted ("a Radient-managed pool member, size `m`, expiring
in 4 h"), and the pod never displays one. What is *lost* relative to A3 is the
second human's confirmation of the transcript; what replaces it is the
provider's signature over the pod's key — a weaker guarantee, stated here rather
than implied, and the reason a pool member's grants never include anything
administrative (§3.5).

**What transport already built, which this decision reuses rather than
reinvents.** `mesh-transport-identity.md` §5.2 ships an invite flag for exactly
this case — `--automated`, which requires a human **on the inviter only** and
marks the admitted row `kind: "pool"` — and its §12.4 states the invariant this
section depends on: *"nothing in the pairing path assumes a human at the far
end."* `mesh-credentials.md` §6.1 goes further and skips the "what this device
owns" half of the pairing screen for a pool, because it owns nothing. So A8.1 is
not a new pairing path: it is **the naming and policy decision about what
`--automated` means for authority** — `ephemeral`, a bounded `expires_at`, a
stored capability set, provider attestation, and the grants of §3.5 — on top of a
mechanism transport has already made safe.

**Rejected alternatives.** (a) *Give the pod a SAS and have the user confirm it
out of band* — the pod has no way to display it and the user has no way to
compare it, so the confirmation would be a rubber stamp, i.e. worse than none.
(b) *Relax A3 for all joins* — that gives up the property A3 exists for.
(c) *Treat the pod as a non-member remote worker* — that removes the member
record's revocation and epoch machinery from exactly the device that is hardest
to trust, and would need a second authorisation path beside the one A3 built.

### 3.3 `MemberRecord` — the field list this design requires

**Owned by transport** (`mesh-transport-identity.md` §4.2) — cited, not restated:
`device_id`, `public_key`, `name`, `kind` (`device`|`pool`), `lifecycle`
(`active`|`provisioning`|`draining`|`expired`), `role`, `capabilities` (the
resolved set, stored rather than derived), `added_at`, `added_by`, `added_via`,
`endpoints`, `last_seen_at`, `last_seen_instance`, `duplicate_count`, `suspect`,
`previous_ids`, `removed_at`, `removed_by`. Transport §12.4 already states the
rule this document depends on: *"nothing in the pairing path assumes a human at
the far end"*.

**This document's delta — seven additive keys on that row**, needed by R20 and by
nothing else. They are additions to a stored row (not a derivation at read
time), for the same reason transport stores `capabilities` explicitly: a later
change to a policy must not silently reclassify an existing member.

```jsonc
{
  "device_id": "d_9f2c1a…",     // …the transport's row, unchanged…
  "kind": "pool",
  "lifecycle": "active",
  "added_via": "invite:pool",   // transport's field, one new value
  "ephemeral": true,            // NEW: no human at this end, bounded lifetime
  "size_class": "m",            // NEW: compute size class, opaque to the mesh
  "provider": "radient",        // NEW: the provisioning authority, "" for a device
  "provider_ref": "req_8c31",   // NEW: the provider's own handle for this instance
  "grant_id": "g_71f0",         // NEW: the pool grant that admitted it ("" for a device)
  "expires_at": 1789412400.0,   // NEW: a hard stop; null for a device
  "max_session_seconds": 14400, // NEW: the per-session ceiling the member enforces; null for a device
  "max_sessions": 4             // NEW: how many sessions it will hold at once (OQ5)
}
```

Three rules on the delta, all load-bearing:

* **`kind` is a field, not a naming convention.** Nothing may infer "is this a
  pod" from a name, an endpoint, or an id shape — the same mistake the desktop
  contract refuses to make about locality (spine §8), and the reason
  `mesh-credentials.md` §3.3 can decide `identity` exposure *by kind*.
* **`ephemeral` is separate from `kind`.** A provider-owned member the operator
  keeps is still `kind: "pool"`; a borrowed laptop added for an afternoon is a
  `device` that is ephemeral. Two facts, two fields.
* **Vocabulary convergence (round 1): the invented names are deleted, not
  aliased.** An earlier revision of this document named three capabilities
  `broker:request`, `broker:grant` and `member:admin`, and a fourth op
  `sync_session`. All four are **gone**, and nothing in the mesh may spell a
  capability with a colon: the broker is `broker_credential` and its op is
  `net_broker`, membership authority is `admin` (with `trust` for re-admission),
  the sync op is `net_sync`, and the session verbs are `net_session_create`
  (`prompt`), `net_session_engage` (`view`) and `net_session_stop` (`stop`).
  `mesh-transport-identity.md` §7.1 is the single list; this document cites it
  and never restates it, which is what makes the two impossible to drift.
* **The three capability/op names in transport §12.4 are authoritative:**
  `compute-pool-v1` (the capability a bootstrap list may carry),
  `broker_credential` (the broker op) and `net_sync` (the sync op). This document
  adds no capability name of its own — `meter_push` is an op, and an op needs no
  capability of its own because the `view` grant transport already requires for
  `net_sync` is the grant that reads and pushes a session's facts.

### 3.4 States and transitions

```
                    grant minted (human, on their device)
                                  │
                                  ▼
   ┌──────────────┐  pod boots, keys minted   ┌──────────────┐
   │ provisioning │ ────────────────────────► │   joining    │
   └──────────────┘                            └──────┬───────┘
          │  provider never delivered               │ epoch ok + grant ok
          │  the pod (timeout)                      │ + attestation verifies
          ▼                                         ▼
   ┌──────────────┐                            ┌──────────────┐
   │   expired    │ ◄──────── epoch rotated ───│    active    │
   └──────────────┘        / revoked           └──────┬───────┘
          ▲                                          │ drain requested
          │                                          │ (idle policy, expiry,
          │                                          │  user command, error)
          │                                          ▼
          │                                    ┌──────────────┐
          └──────────── power off ◄─────────── │   draining   │
                (after the sync barrier)       └──────────────┘
```

| State | Who can be in it | Entered by | Durable where | What the rest of the system does |
|---|---|---|---|---|
| `provisioning` | pool only | a `PoolRequest` is queued for the control plane | the *requester's* install (a member row exists only once the pod joins; until then this is the request's state) | lists as `provisioning` under `lop network pool ls`; no session may be placed on it |
| `joining` | pool only | the pod's relay has presented its grant and key | the pod, published in its own member record; peers see it after the first authenticated frame | not placeable; a placement already pointing at it queues (E6 refuses until `active`) |
| `active` | pool and device | admission complete | both sides | full grant-based participation |
| `draining` | pool and device (a device drains when the user disconnects it) | idle policy, expiry within `drain_lead`, user command, or an error the relay decides is fatal | the member record's `lifecycle`, plus the session's own drain vocabulary (§3.6) | no **new** turns are admitted; placements pointing here are rewritten (§6.4); the audit gets one `drain_started` event |
| `expired` | pool and device | the drain barrier completed, or the epoch was rotated under it, or `expires_at` passed | the member record and the audit | refused at the epoch/grant check; the user's install keeps the member row as a tombstone for one retention window so `lop network log` can explain the disappearance |

Transitions nobody may take:

* `active → active` on a *different* device key (a member may not rotate its
  identity; that is a new member, because revocation is keyed on the id);
* any state → `active` without the epoch check (fails closed — the spine's §5.7,
  *fails closed*);
* `expired → anything` (a member that expired is re-admitted by a **new** grant
  and a new member id, so the audit trail of the old device is never rewritten).

**The lifecycle is auditable, on the one log.** Each transition emits an event
into the taxonomy `mesh-incident-response.md` §4.3 owns (Q7 there recommends
exactly this rather than a second log), with these names and `detail` maps, and
this table is the delta that document should adopt:

| Event | When | `detail` |
|---|---|---|
| `pool_grant_minted` | the human authorises an automated member | `{grant_id, size_class, role, expires_in_s, provider}` |
| `pool_request_queued` | E2's `PoolRequest` is written | `{request_id, size_class, max_hours}` |
| `member_admitted` (transport's name) | the member reaches `active` | `{via: "invite:pool", grant_id, provider_ref}` |
| `member_draining` | the drain begins | `{trigger: idle\|expiry\|user\|error, sessions, grace_s}` |
| `drain_barrier_timeout` | the barrier's deadline expired before every ack landed | `{pending_flushes, pending_meter_pushes, waited_s}` |
| `pool_member_expired` | the member reached `expired` | `{reason: drained\|lost\|revoked, uptime_s, sessions_flushed}` |

Ephemerality is enforced on the member's own side as well as honoured on the
user's: a pod that loses its link for longer than `link_loss_grace` stops
admitting turns, finishes or cuts the in-flight one per §3.6, runs the final
sync if it can, and powers off. It must never run work it cannot bill and cannot
deliver.

### 3.5 Difference from a user's own device, stated as a table

| Fact | User's device | Pool member |
|---|---|---|
| Human present at admission | Yes, on both ends (A3, SAS) | No; the human acts once at grant mint (A8.1) |
| Admission evidence | SAS match on both devices | pool grant + provider attestation |
| Grants | whatever the operator gives it, up to `admin` (transport's vocabulary: `list`, `view`, `prompt`, `steer`, `stop`, `slash`, `delete`, `move`, `broker_credential`, `admin`, `trust`) | a subset, resolved at admission and stored on the row: `list`, `view`, `prompt`, `steer`, `stop`, `slash` for a member that runs sessions; **never** `admin`, `trust`, `broker_credential`, or `delete`/`move` without an explicit operator act (`mesh-session-mobility.md` §10 reaches the same verdict for `delete`) |
| Credentials | may own credentials; the broker runs on it | **declares none, ever**, and holds only `grant_ttl_s`-bounded in-memory grants; `identity` is never served to it (`mesh-credentials.md` §6.1–6.3) |
| Lifetime | until revoked | `expires_at` + idle policy; a drained member is expected to disappear |
| Its own store | the operator's; a forensics source | disposable; the pod's own session store dies with it |
| Audit | authors events a human can be shown | authors the same events, plus `metering`; its rows are labelled with the member id so a dispute has a subject |
| Sessions on it | the user's own | sessions whose durable copy of record is on the user's device (§6.4, §6.5) |

**Least authority, concretely.** A pool member's stored `capabilities` are
`list`, `view`, `prompt`, `steer`, `stop`, `slash` — the set a device that runs
sessions needs and no more. It may not list another member's sessions (`list`
answers for the sessions placed on it and for nothing else, §6.2), may not
archive, delete or move anything, may not mint invites, may not act as a broker for anyone, and may
not rotate the network. If the member id is compromised, the blast radius is "the
sessions that were placed there" — not the network.

### 3.6 What a session on the member sees when it drains

Not new vocabulary. The runtime already has a drain ladder and the words for it:
a signalled runtime commits to `leaving` and says which trigger
(`SIGNAL_DRAIN_CAUSE = "runtime-shutdown"`, `BUILD_DRAIN_OVERDUE_CAUSE =
"runtime-overdue"`, and the phrase helpers `leaving_phrase_for_frame` /
`drain_phrase_for_frame` beside them in `session/runtime/types.py`). The mesh
adds one more trigger to that ladder rather than a parallel one:

1. **`drain_started`** (member side, ~0 s): the relay puts the member in
   `draining`, tells each placed session's runtime to commit to leaving with
   reason `pre_spindown` (mobility §7.5's name for the trigger, reused rather
   than invented), and refuses new placements and new turns.
   A viewer sees the existing leaving notice, with the peer's words; it does not
   get a mesh-specific error.
2. **In-flight turn** (bounded): finished if it can be, cut at the turn
   boundary otherwise, with the cut recorded by the existing journal row —
   `session/runtime/registry.py:283` `TURN_JOURNAL_NAME` is the artifact a
   successor reads to learn a turn was in flight. The bound is
   `drain_grace` (recommend 120 s; calibrated from measurement in §9.4, never
   guessed at implementation time).
3. **Final sync** (§6.4): the barrier. Until it is acked, the member stays
   `draining` and does not power off.
4. **Acked → `expired`**: the member's relay publishes the tombstone if it still
   can, then exits; the pod's host reaps the instance.
5. **Home**: the synced session is a **cold** session on the user's device, with
   the same id, resumable by `/resume <id>`. If the user is looking at the
   session when the pod dies, the viewer gets the existing
   reconnecting/refused-after-drain sentence and a `/resume` route, not a
   deleted tab.

A viewer whose link to the pod dies *without* a drain (network outage, spot
reclaim) sees the same "unreachable" state the spine §7 defines for any peer:
the session is shown, marked unreachable, and **not** deleted. The difference
is that a pool member may never come back, which is why §6.5's cadence — not the
final sync — is what makes the loss bounded.

---

## 4. Placement (A8) — what the pool adds to a settled design

Placement is owned by `mesh-session-mobility.md` §5: `SessionPlacement`
(`local_operator/mesh/placement.py`) is
`{mode: "local"|"peer"|"pool", network_id, home_device, policy: "pinned"|"prefer-remote"|"cost-capped", stamp_revision}`,
carried additively on the session record (`SessionRecord.placement`, §5.1 there)
and in the session directory's `mesh.json` sidecar (§5.2 there). Nothing in this
section restates it. What R20 needs is a **delta of four facts**, and nothing
else:

1. **`mode: "pool"` becomes producible.** Today it is "reserved and unproduced"
   (mobility §5.2). It gains exactly two producers in this design: the user's
   placement request (E2, §4.1) and the drain's rewrite (E11, §6.2). A future
   scheduler is a third producer and needs **no schema change** — which is A8's
   whole claim.
2. **`home_device` is what the drain reads.** For a pool-placed session it names
   the device the final `flush` and the final `meter_push` go to (§6.2). It is
   never the member itself — a member is a device that may cease to exist, and a
   "home" that can die is not a home.
3. **No new audit event is needed.** A placement write emits the existing
   `session_placement_changed` (`{placement_before, placement_after, policy, reason}`,
   IR §4.3), and `reason: "drain"` is what makes a spin-down's rewrite auditable.
4. **`stamp_revision` is the idempotency key**, so the drain's rewrite is safe
   to retry without a placement-specific op.

**There is deliberately no standalone "set placement" op in this pass.** A
placement is written by whichever actor performs the operation that changes where
a session lives — creation on a peer (mobility §5.3), a move (§6.3 there), or the
drain. Adding an op that does nothing but write the field is the first step of a
scheduler, and it belongs to the scheduler's PR.

### 4.1 The request that produces `mode: "pool"`

`PoolRequest` is the mesh's own record of the user's intent — the artifact E1/E2
produce and the only thing a control plane needs to read:

```jsonc
// <config>/network/pool-requests/<request_id>.json   0600, staged write,
// bounded to the last MAX_POOL_REQUESTS (recommend 32) so a request log cannot grow
// without bound; the newest file is the one the control plane acts on.
{
  "version": 1,
  "request_id": "req_8c31",
  "network_id": "n_4a1c",
  "size_class": "m",
  "max_hours": 4,
  "requested_at": 1789399800.0,
  "requested_by": "d_6c1f…",
  "state": "queued",           // "queued" | "provisioning" | "fulfilled" | "cancelled" | "expired"
  "grant_id": "g_71f0"         // the pool grant minted for it, once it has one
}
```

`state: "provisioning"` names the window in which a member row may exist in
`lifecycle: "provisioning"` (transport's vocabulary) with no session placeable on
it. `lop network pool request|ls|cancel --json` reads and writes these; nothing
in the mesh acts on one, and the file is not a credential, not a schedule, and
not a price.

### 4.2 What the mesh does not do with a placement

No scheduling, no capacity detection, no negotiation, no price: a placement
change is *refused* rather than negotiated when the target is not `active`
(§3.4), and the refusals are the interface a scheduler will handle later
(`member_not_active`, `not_permitted`, `pool_cap_exceeded`, `session_busy`).
That boundary is what keeps the scheduler out of this schema (§7).

## 5. Metering as an event stream

### 5.1 What is measured, in one line

Four numbers, all of which live in §5.2.1 with their sources: **`cpu_ms`** per
session (CPU nanoseconds of the session's runtime process, read from the same
`libproc` rusage call that already reads its memory footprint), **`wall_ms`** per
member (its own record's timestamps plus the relay's link observation),
**`busy_ms`** per session (the record's `busy` bit transitions — the same bit the
sidebar renders), and **`size_class`** (an opaque string copied from the member
row). Nothing is inferred from anything else, and none of the four is priced by
the mesh.

### 5.2 The literal events

Both events live **inside the A7 audit envelope** of
`mesh-incident-response.md` §4.2 (`schema: "lop.mesh.audit.v1"`, `seq`, `ts`,
`actor`/`actor_kind`/`subject`, the closed-whitelist `detail` map, and the
**2048-byte cap** on `detail`). That document reserves both names and explicitly
defers their literal fields to this section, so this section is their sole owner
and must not be restated there.

```jsonc
// meter_interval — written by the member's relay every METER_INTERVAL_S (60 s)
// while the member is joining, active or draining.
{"schema":"lop.mesh.audit.v1","seq":1841,"ts":1789400040.0,"ts_iso":"2026-09-19T14:14:00.000Z",
 "event":"meter_interval","network_id":"n_4a1c","network_name":"devmesh","epoch":7,
 "actor":"d_9f2c…","actor_name":"radient-m-4h","actor_kind":"pool",
 "subject":"d_9f2c…","outcome":"ok","cause":"",
 "detail":{"meter_seq":42,"part":1,"parts":1,
           "interval_start":1789399980.0,"interval_end":1789400040.0,
           "size_class":"m","wall_ms":60000,"cpu_ms":18734,
           "sessions":[{"session_id":"b71e…","cpu_ms":18402,"turns":3,"busy_ms":21011},
                       {"session_id":"c902…","cpu_ms":332,"turns":0,"busy_ms":0}],
           "sig":"…","sig_key":"b64url(32)"},
 "prev_hash":"…","hash":"…"}
```

`detail` keys for `meter_interval`, in full and nothing beyond them: `meter_seq`,
`part`, `parts`, `interval_start`, `interval_end`, `size_class`, `wall_ms`,
`cpu_ms`, `sessions[]` (each `session_id`, `cpu_ms`, `turns`, `busy_ms`), `sig`,
`sig_key`.

```jsonc
// meter_close — exactly one per member lifetime, written before power-off.
{"schema":"lop.mesh.audit.v1","seq":1998,"ts":1789412380.0,"ts_iso":"…Z",
 "event":"meter_close","network_id":"n_4a1c","network_name":"devmesh","epoch":7,
 "actor":"d_9f2c…","actor_name":"radient-m-4h","actor_kind":"pool",
 "subject":"d_9f2c…","outcome":"ok","cause":"",
 "detail":{"meter_seq":118,"final":true,"reason":"drained",
           "meter_first_ts":1789399980.0,"meter_last_ts":1789412380.0,
           "size_class":"m","wall_ms":4170000,"cpu_ms":1204121,
           "sessions":{"b71e…":{"cpu_ms":1180021,"turns":41}},
           "part":1,"parts":1,"sig":"…","sig_key":"b64url(32)"},
 "prev_hash":"…","hash":"…"}
```

`detail` keys for `meter_close`: `meter_seq`, `final`, `reason`
(`drained`|`expired`|`revoked`|`lost`), `meter_first_ts`, `meter_last_ts`,
`size_class`, `wall_ms`, `cpu_ms`, `sessions{<id>: {cpu_ms, turns}}`, `part`,
`parts`, `sig`, `sig_key`.

Seven rules, each of which is why the shape is what it is:

1. **`meter_seq` is monotonic per member lifetime and never reused.** A gap is
   evidence of loss — the one thing a biller must be able to *see* rather than
   silently absorb. It is a millisecond-scale counter, so it never collides with
   the envelope's `seq` (a per-install audit sequence).
2. **Totals live on the close record**, so a billing run that has only the close
   record still has the whole truth; the intervals are what make the *running*
   number, the attribution and the disputes diagnosable.
3. **`network_id`, `epoch`, `actor`, `actor_name`, `actor_kind` and `subject`
   come from the envelope** and are never duplicated inside `detail`. `subject`
   is the member (`actor == subject` for a self-authored interval, which is the
   honest reading: the member measures itself).
4. **The record is signed by the member's device key** — `sig` over the canonical
   `detail` serialisation, `sig_key` the base64url public key — because
   possession of the row must be provable to a third party that did not observe
   the link. The envelope's hash chain proves the *local log* is intact; it does
   not prove who authored a record, which is what `sig` is for.
5. **The 2048-byte `detail` cap forces slicing, and slicing is specified.**
   `sessions[]` is capped at `METER_SESSIONS_PER_PART` (recommend 8, measured
   against the cap); a member running more emits further records with the same
   `interval_start`/`interval_end`, incrementing `part` and carrying **disjoint**
   `sessions[]` slices. **`wall_ms` and `cpu_ms` appear on `part: 1` only**, so
   summing the parts never double-counts instance time; a reader's total is
   `part 1`'s two totals plus the union of every part's `sessions[]`. When the
   cap is hit, the writer sets `"truncated": true` — the flag IR §4.2 already
   defines — rather than dropping a session silently.
6. **Attribution is the triple** from `network_id` (envelope), `subject`
   (member), `sessions[].session_id` — see §5.3.
7. **Never present:** transcript content, prompt text, token values, absolute
   paths, session titles. A metering record names sessions by id and by nothing
   else, which is the rule IR §4.2 states for every record in this log.

#### 5.2.1 Where the numbers come from

* **`cpu_ms`**, per session: accumulated **CPU nanoseconds** of the session's own
  runtime process. This is not a new probe — `mobile/resources.py` already opens
  `libproc` and reads `rusage_info_v2` per pid, with `ri_phys_footprint` at
  offset `16 + 7*8` (`:85`), and the two counters needed here sit at offsets 16
  and 24 of the same buffer (`ri_user_time`, `ri_system_time`). The reader today
  returns only memory (`ResourceUsage`, `:101`; `session_resource_usage`,
  `:306`), so the delta is two fields on an existing dataclass and read, at
  microsecond cost, with no subprocess and no new dependency. Linux's analogue is
  `utime + stime` from `/proc/<pid>/stat`, HZ-scaled, on the same batched probe.
* **`wall_ms`**, per member: from the member's own record timestamps
  (`started_at` / `heartbeat_at`) plus the relay's observation of the link, so
  the number exists even after a session's process has gone.
* **`busy_ms`**, per session: derived from the record's `busy` bit transitions —
  the same bit the sidebar renders — so the number a customer sees and the number
  support explains come from one source.
* **`size_class`**: copied from the member row (§3.3). The mesh never interprets
  it; it is an opaque label the price table knows.

#### 5.2.2 Compute time vs compute size class

Two prices are computed from one event, and the mesh keeps the inputs separate so
the table can change without a mesh change:

* **size class × `wall_ms`** prices *provisioned capacity* — "you held an `m` for
  69 minutes". It is a property of the member, not of the work, and it is what
  makes an idle pod cost something, which is the economic reason the on-demand
  story works at all.
* **`cpu_ms`** prices what was actually done and, more importantly, is the
  **anti-abuse** signal: a pod holding four cores for an hour and a pod running
  one 30-second turn must not look the same.

#### 5.2.3 Delivery: `meter_push` / `meter_ack` (this document's one wire addition)

The audit log is **per install** (IR §4.1) and a pod's log dies with the pod. A
metering stream that never leaves the member is therefore evidence about a machine
that no longer exists, which is exactly the failure §5.5 says billing must not
have. So the member **pushes**:

```jsonc
// member relay → each device holding one of its placed sessions (in practice
// placement.home_device, §4.2), over the peer link. Same additive-op rule as
// everything else (transport §12.3): an older peer answers `unknown op`.
{"op": "meter_push", "req": "mp_9",
 "records": [ /* the meter_interval / meter_close objects, verbatim, in meter_seq order */ ]}

{"op": "meter_ack", "req": "mp_9",
 "acked_through": 118,          // the highest meter_seq durably written by the receiver
 "log_seq": 1998}               // the receiver's own audit seq for the last one (for forensics)
```

Rules:

* **The receiver verifies `sig` against the member's public key** (from its own
  member row) before writing, and appends the record to **its own** audit log.
  The record of record is therefore the user's device, and the member's own copy
  is a redundant local one.
* **Unacked records are retained on the member** in a bounded spool and re-sent
  on the next tick — which is what makes "metering may not drop a cent"
  implementable, and what separates it from the analytics ledger's documented
  drop-on-full contract (`local_operator/analytics/__init__.py`).
* **A rejected record is audited, not silently discarded**: a signature failure
  is `meter_rejected` with `{meter_seq, cause}` on the *receiver*, and the close
  record the member eventually sends carries the same `meter_seq`, so a lost
  interval shows up as a documented gap rather than an absence.
* **The drain's barrier includes this ack** (§6.2): a pod that never lands its
  close record cannot have been "drained" — it was "lost", and it says so.
* `meter_push` carries no capability of its own: the `view` grant
  (`net_sync`'s, transport §12.4) is the grant that reads a session's facts, and
  a member pushes only about sessions it is running.

### 5.3 Attribution: the triple, and the case with no session

Every event carries `network_id` (envelope), `subject` (envelope — the member),
and — inside `sessions[]` — `session_id`. The triple is the billing key, and the
three axes are not interchangeable:

* `subject` is who computed (and therefore whose provider account and size class
  apply);
* `network_id` is who is accountable (a device can be in several networks and a
  pool member belongs to exactly one);
* `session_id` is what the user recognises, and what support answers questions
  about ("which chat cost me 4 hours?").

A member that is awake with no session on it (a just-provisioned pod, a drained
session's tail) still emits intervals with `sessions: []`. That is deliberate:
**idle instance time is billable time**, and a schema that can only express
"compute attributed to a session" would force that time into either a fake
session row or a second event type — both of which are refactors (§7.3).

### 5.4 Why billing consumes events rather than reading session state

Five reasons, in the order they would bite:

1. **The store it would read does not exist any more.** A drained pool member's
   session store dies with the instance (§6.5 makes the durable copy of record
   the user's, and it is a *session copy*, not a billing record). Billing cannot
   depend on a machine whose whole lifecycle is "spin down when idle".
2. **Session state is mutable and the bill is not.** Sessions are moved (R11),
   forked, archived, deleted (PR #1328's design), and re-titled. Any of those
   changes what a state read would say; events are immutable facts about
   intervals that already happened. A bill computed from a mutable projection is
   a bill that changes when the user reorganises their sidebar.
3. **Zero trust forbids it.** The pricing plane is not a member of the user's
   network (spine §5: multi-user networks are explicitly out of scope). Reading
   a user's session store to price their compute would be exactly the
   cross-boundary read the mesh's whole authorisation model refuses. Events flow
   because the *user's own device* chooses to send them.
4. **A state read is unbounded I/O; an event stream is not.** R18 requires the
   I/O argument to be measured, not asserted. Billing by state means walking
   transcripts (2.37 GB and ~2,000 sessions on the operator's own store, per
   `docs/design-session-spend-ledger.md` §1) on a schedule; metering by event is
   one log line per member per minute, bounded by construction.
5. **It cannot be attested or deduped.** A signed interval with a monotonic
   `meter_seq` can be verified and double-billed-safe (§5.2 rule 1). "The number I read out of your session
   directory" can be neither.

**The counterexample the repo already contains.** `local_operator/analytics/`
records token usage from every provider call and is explicitly
**best-effort**: "a full queue DROPS the sample rather than blocking a session"
(`analytics/__init__.py`). That is the right design for diagnostics and
**would be theft** for a bill. Metering therefore takes the opposite contract at
the one point that matters: a `meter_interval` that cannot be written is a
*refusal to continue working* (the member stops admitting turns and drains) or a
line in the audit, never a silently dropped sample. Stated plainly: analytics
may lose a sample, metering may not lose a cent.

### 5.5 Trust: what is attested, what is observed, and what a dispute is

* The **member attests** its own intervals, with `sig` over `detail` (§5.2 rule
  4). It has every incentive to inflate, so its number is never the only one —
  and the receiver verifies the signature against the member's public key before
  it writes the record into its *own* log (§5.2.3), so a forged interval never
  becomes a durable record on the device that pays.
* **The home device observes** what it can: link liveness, `wall_ms` from the
  member's own heartbeat record, and the gaps in `meter_seq`. A member that claims
  60 minutes of `wall_ms` while its link was down for 40 of them is visible
  without reading anything private.
* **A mismatch is an audit event** (`meter_dispute`, with the two numbers and
  the gap), never a silent correction. Support and the control plane decide; the
  mesh does not adjudicate its own bill.
* The residual risk is stated rather than solved: a compromised pod can inflate
  `cpu_ms` within a plausible band and no mesh-side check will catch it. The
  mitigation is the control plane's (provider-account quota, a per-member
  `expires_at`, and `pool_cap` on how many sessions a member may hold at once —
  §6.6). **A user's own device cannot abuse this way** (they pay for their own
  machine), which is why these checks are scoped to `kind: "pool"`.

### 5.6 Retention and I/O

`meter_interval` is one line per member per interval — for one pod that is 60
lines/hour, ~2.9 KB/hour at the shape above, i.e. **megabytes per member-year**,
which is what A7's bounded rotation is sized for. The stream rides the same log
as the audit (one file, one rotation policy) and takes the same age+size caps;
the *close* record is what a long-lived aggregate needs, and it is a single
line. No per-frame, no per-token, no per-provider-call writes: the rule A7
states ("never per frame") is met by construction, because nothing in the
metering path is on a hot path — the sampler runs on the relay's own slow tick.

---

## 6. Sync before spin-down (R22), as an instance of the mobility primitive

### 6.1 The primitive is mobility's; this section adds the pool's use of it

`session.sync.plan` / `session.sync.fetch` (`mesh-session-mobility.md` §7.1) is
the operation "bring a device's copy of a session to a frontier the source
chooses", with three modes: `exact` (a move), `boundary` (`--keep`) and **`flush`
(R22)** — no retirement, no ownership change, the destination keeps its copy. Its
wire name is `net_sync`, already reserved with the `view` capability
(`mesh-transport-identity.md` §12.4). So R22 is *an instance of the move
primitive*, and this document adds nothing to the transport: it fixes the pool's
**use** — who flushes to whom, when the pod may die, and what home is guaranteed
to hold (§6.2–6.4).

Two corrections this document makes to its own first draft, both because a
sibling read the code and I had not:

* **The cursor is not a byte offset.** `Transcript.compact_file`
  (`session/transcript.py:1807`) rewrites the transcript with `os.replace`, so
  offsets are not stable across a compaction. The cursor is
  `(history_generation, through_entry_id, prefix_bytes, prefix_digest)`
  (mobility §7.3). An implementation built on a byte offset corrupts a compacted
  session, and the corruption surfaces days later as a resume that reads someone
  else's row boundary.
* **A `flush` does not take the `exclusive-move-v1` fence.** It cannot: the
  source is still writing. Safety comes from two other rules (mobility §7.4): the
  source serves only through the last complete newline of its last durable append,
  and the copy is a **fork by construction**, so an off-by-one boundary is a
  divergence point rather than a corruption. The fence belongs to `exact`-mode
  moves, and my draft's "the pre-drain sync always takes the fence" was wrong for
  a `flush` (OQ6, revised).

### 6.2 Who flushes to whom, and the barrier

`lop network drain` (mobility §7.5) runs `flush` for every session the device
holds. For a pool member the sequence is ordered, and the order *is* the design:

1. **`draining` is announced** — the member row's `lifecycle`, audited as
   `member_draining` (§3.4) — and new turns are refused (§3.6).
2. **`flush` per session**, to every device that holds a copy; for a
   pool-placed session that is `placement.home_device` (§4). Each result is the
   existing `session_sync_completed` audit event with `reason: "pre_spindown"`
   (IR §4.3).
3. **A final `meter_push`** (§5.2.3) carrying the closing interval and the
   `meter_close` record for each network the member computed for. It must be
   **acked**.
4. **The barrier**: the member powers off only when *every* flush is acked *and*
   every meter push is acked, or when `SYNC_DRAIN_DEADLINE_S` (recommend 900 s)
   expires — in which case the member writes its close record locally with
   `reason: "lost"`, records `drain_barrier_timeout` with the pending counts, and
   the home device learns what it is missing from that event the next time
   anything reaches it.
5. **`expired`** (audited `pool_member_expired`), then the instance spins down.

The barrier is a **deadline, never an unbounded wait**: a home laptop that is
asleep must not keep a billable pod alive, and a pod that waits forever is a pod
billing for nothing. `drain_grace` (the in-flight turn's bound, §3.6) and
`SYNC_DRAIN_DEADLINE_S` are named here and must be calibrated from measurement
(§9.4), not from these guesses.

### 6.3 The pool's one cadence addition

Mobility §7.5's cadence is: on demand, after a turn settles (debounced ≥30 s), at
destination attach (>10 min), and the pre-power-off flush. A pool member adds two
triggers, because it can die without warning and a user cannot ask it to flush
afterwards:

| Trigger | When | Why the pool needs it |
|---|---|---|
| **tick** | every `POOL_SYNC_TICK_S` (recommend 60 s) while the member holds unsynced bytes | bounds an un-drained death (spot reclaim, outage, `kill -9`) to one tick; a policy knob, **no schema change** |
| **meter push** (§5.2.3) | with the tick, and at every turn end | the billing record is **member-scoped** — idle instance time has no session — so it cannot ride a session sync's schedule |

At one tick, a 60-second delta is a handful of kilobytes against an append-only
tail (the largest transcript in the operator's own store measured 216 KB total,
per `fork.fork_session`'s docstring) — closer to a no-op than to a cost, and far
cheaper than the analytics write path's per-call inserts.

### 6.4 What home is guaranteed to hold

After the pod dies, whatever killed it:

* **Held:** a session directory at home whose transcript is a copy at the flush
  boundary — every durable row through the frontier the source chose (mobility
  §7.4) — plus the sidecars mobility §7.2 copies (`title.json`,
  `attachment.json`, `origin.json`, `turn-journal.json`, `runtime-stop.json`,
  `inbox.jsonl`, the desktop marker, and every referenced attachment blob). **The
  copy set is §7.2's; this document removes nothing from it and adds nothing** —
  including the one thing R22 needs from it, which is already there: a
  `session_spend.v1` row is a transcript row, so a flush carries it, and its
  replacement-state contract (`session/spend.py`: the persisted row carries the
  running total, so a reader takes the newest and never sums the file) means the
  newest synced row *is* the correct running total at home with no accumulation.
* **Held, and the important one:** the session is **resumable at home by its own
  id**. The id rule is mobility §6.2's — a `flush` destination keeps the id,
  while a `--keep` copy at a new device mints a new one — and the pod's copy
  ceasing to exist changes nothing about what the id means. Worst case home holds
  a prefix of the history and `/resume <id>` opens with the turns that landed.
* **NOT held:** everything the pod held that was never flushed. A *drained* pod
  has no tail — the barrier is what `drained` means. A pod that *died* has one,
  bounded by the cadence above, and the loss is auditable
  (`meter_lost` / `drain_barrier_timeout`) rather than invisible.
* **NOT held, and stated so nobody assumes it:** the member's own local evidence
  that never crossed (its `network.log`, its un-sent spool entries). The
  guarantee is about the *session copy* and the *metering records*, and both are
  named above.

### 6.5 Adoption rules at home

1. **Same session id.** A `flush` keeps the identity; the destination's copy is
   the same session, and `/resume` finds it exactly where it was looked for
   before the move. A `--keep` copy at a *different* device is the case that
   mints a new id (mobility §6.2).
2. **No second writer while the member is alive.** The home copy must not be
   engaged while the member is `active`/`draining` — engaging it would create the
   two-writer situation the fence exists to prevent, and no fence is held in a
   `flush`. **This is a guard the pool case adds**: mobility §6.6 already refuses
   to `engage_runtime` for a session whose journal says it is *handing off*, and
   a pool-placed session needs the same refusal for a second reason — its owner
   is a remote member whose `lifecycle` has not reached `expired`. Recommended
   spelling: `launch.engage_runtime` also refuses when the session's `mesh.json`
   placememt names a live member whose `lifecycle` is not `expired` **and** this
   device's last sync cursor is not the member's last announced frontier (a cold,
   fully-synced copy *is* engageable, which is the whole point of R22).
3. **Collision is repaired, never merged.** If home holds a copy that cannot be
   continued, mobility §7.3's `replace` path is the recovery: the destination
   writes a staged file and `os.replace`s it. There is no row-wise merge, in this
   design or in that one.
4. **Placement is rewritten** to `local` (or to the peer that adopted it) as part
   of the drain, with the existing `session_placement_changed` audit event and
   `reason: "drain"` (§4).
5. **The tombstone stays for one retention window** so `lop network log` explains
   where the session went, then the member row is dropped (transport §4.2's
   tombstones are kept; a *pool* member's tombstone is dropped with the
   provider's instance record, which is what `previous_ids` is for).

### 6.6 Failure modes, and what each one costs

| Failure | Detected by | Cost | Recovery |
|---|---|---|---|
| Pod dies mid-sync | home: bytes land, the completion never arrives | nothing: the destination writes at row boundaries, so the partial tail is dropped (mobility §7.4 rule 1) | the next tick, or the next `flush` |
| Pod dies with no drain (spot reclaim, outage, `kill -9`) | home: link lost, heartbeat stops | up to one tick plus the in-flight turn | the session is still resumable; the tail is gone and the audit says what was missed |
| The transcript was compacted or written by a second process | generation/digest mismatch on the next plan (mobility §7.3) | one `replace` (whole transcript) | automatic; audited, because a second writer is a bug worth knowing about |
| Home offline at drain | the member's barrier deadline expires | the member stays `draining` (billing keeps running, deliberately) until `SYNC_DRAIN_DEADLINE_S`, then powers off as `lost` | `drain_barrier_timeout` names what is missing; the next reachable device learns the gaps |
| Meter push unacked | the member's spool still holds records at the deadline | the close record is written with `reason: "lost"`; the intervals are re-sent if any path opens before power-off | `meter_seq` gaps make the loss a *documented* gap |
| A member holds more sessions than `max_sessions` | the member's own accounting | `pool_cap_exceeded` on the placement | the user's request says so; the number is the member's policy, not the mesh's |
| A `flush` cannot start (the member's relay is gone) | the drain command | the drain reports `flush_unavailable` per session | the session is still on the member if the member is alive; if it is not, this is the "died with no drain" row |

## 7. Explicit deferrals — and the boundary that keeps each out

| Deferred | Why it is out | The boundary that keeps it out | Where it plugs in later |
|---|---|---|---|
| **Scheduler policy** (which session goes where, when) | A8: placement exists, nothing schedules automatically | the mesh has exactly one writer of placement, `placement_set`, and it *refuses* rather than negotiates; no component watches capacity in this pass | a new producer of `placement_set` calls; zero wire change |
| **Pricing / credit arithmetic** | the mesh must not be the price list | the metering event carries time and size class and **no money field at all** | the control plane's table, read against `meter_interval`/`meter_close` |
| **The provisioning API** (calling a cloud/Radient API to boot a pod) | it is a control-plane concern, not a mesh one — `mesh-transport-identity.md` §12.4 defers it in the same words | the mesh sees only the member a provider produced; no mesh module holds a provider credential | `PoolRequest` (§4.1) is read by the control plane; the pod arrives as a member (E4/E5) |
| **Kill-on-idle tuning** | it is a policy number, and it depends on the price model | the member record carries `expires_at`/`max_session_seconds`; the mesh only honours them | the control plane's policy engine |
| **Billing ingestion itself** | Radient is already the billing identity (mobile guide's `lop tunnel billing`, the desktop `radient` route) | the mesh writes an append-only stream; it never posts a charge | an ingest reader over the audit channel |
| **Detecting "out of capacity" automatically** | it is a product decision with an ugly failure mode (moving a user's work without asking) | E1 is a *user* action in this design | a later heuristic may pre-fill the request; it still goes through `placement_set` |
| **`meter_resend` / reconciliation protocol** | only needed once ingestion exists | `seq` gaps and the close record already make loss *visible* | a control-plane-side reader asks; the mesh needs one new op, and the schema does not move |

The boundary that matters most is the third row's: **nothing in the mesh holds a
capacity credential.** A mesh process that could call a provider's API would be
a mesh process worth stealing, on a device whose whole thesis is that peers
carry nothing worth stealing.

---

## 8. What would force a refactor, and the choice this design makes instead

Three schema shapes are expensive to change later. Each is one a reasonable
person would pick for speed today.

### 8.1 Session identity keyed by `(device_id, session_id)`

**The tempting shape.** "A session is `sessions/<id>/` inside a device's config
root, so its address is (device, id)". The natural next step is to make the
pair the key — it makes provenance free and collisions impossible.

**What breaks.** A move (R11) and a sync (§6) both move a session *between*
devices while it stays the same session. Under a composite key, every move is an
identity change: cursors, placements, audit rows, the user's `/resume <id>`, and
every reference in a peer's cache would have to be rewritten — and a
half-completed rewrite is two sessions where the user has one.

**The choice here.** A session id is globally unique, opaque, and **carried
unchanged across every device** — which is what the id already is
(`uuid4().hex[:12]`, `fork.py:155`, a single path component that every reader
treats as opaque). Location is **never** part of identity: it is a `placement`
and a `locality` on a projection. Consequence, stated as an invariant a test can
assert: *copying a session to another device never changes its id, and no
reader derives a session's device from its id.*

### 8.2 Placement as a field inside the session (or its transcript)

**The tempting shape.** Put `placement` in the runtime's session state, or in a
transcript row, so it travels with the conversation and one reader gets it for
free.

**What breaks.** (a) It makes the *conversation* an input to scheduling: a
transcript row written by a scheduler would appear in the model's history, and
the prompt cache's prefix breaks the moment a non-conversation fact becomes a
transcript entry. (b) It becomes fork-inherited, so a fork lands on a pod nobody
placed it on. (c) Billing and scheduling then *write* the session's durable
artifact, so a placement change is an event in the session's own record — and
the audit trail and the session history become two sources of one fact.

**The choice here** (mobility §5, which is authoritative; this paragraph is why
it is right): `mesh.json` in the session directory, absent-means-`local`, carried
on the record additively, and copied by **nothing** in `COPIED_SIDECARS`. A test
can assert both halves: a fork never inherits a placement, and a session with no
`mesh.json` behaves byte-identically to today. (Note for whoever writes that
test: `mesh.json` is a *new* file, so it must be added to `EXCLUDED_SIDECARS`
explicitly, or a fork will inherit its parent's placement by accident — the
allow-list is the protection, and this is the one place the pool can break it.)

### 8.3 Metering as state, or as a per-frame stream

**The tempting shape.** Either (i) record usage on the session (a field, a
counter, a cost column) and let billing read it, or (ii) stream a sample per
provider call / per turn, because that is the granularity the money is spent at.

**What breaks.** (i) is §5.5's five reasons, and the first one is fatal on its
own: the pod whose compute is being billed is the one machine guaranteed not to
exist afterwards. (ii) breaks A7/R18's I/O argument (it is a write per call,
i.e. the thing the audit design explicitly refuses) and it is
best-effort-by-birth: the analytics ledger shows why, since its documented
contract is to *drop* samples under pressure, which is correct for diagnostics
and wrong for money.

**The choice here.** Batched interval events with a monotonic `seq`, an explicit
attribution triple, a close record carrying totals, and a signature; plus the
explicit divergence from the analytics contract (metering may not drop). The
schema test that keeps this honest: *an event stream can be reconstructed into
totals from a single close record, and every event names session, member, and
network.*

### 8.4 The fourth, already decided elsewhere

Widening `run/mobile` to carry a member record is the same class of mistake, and
A2 already refuses it for the same reason ("a peer record carries fields a
session record must not carry" — `types.py:239`, the namespace comment). It is
restated here only so that a later implementer does not re-open it while adding
`size_class`/`lifecycle` and reaching for the nearest record file.

---

## 9. Test plan and the evidence this work must carry

Nothing here is built yet, so this section has two halves: what can be pinned
**when the interfaces land** (unit/schema), and what **must** be driven end to
end when the pool is real (the QA matrix a `qa-tester` runs).

### 9.1 Schema and unit gates (the PR that adds each interface)

```sh
# whole-tree gate, per ~/local-operator/AGENTS.md — no scoped substitute
.venv/bin/python -m pytest tests/unit -q -x
```

Named cases the implementing PR must add:

| Test | Asserts |
|---|---|
| `tests/unit/network/test_member_lifecycle.py` | every transition in §3.4 is legal, every illegal one refuses; `expired` is terminal; an unknown `lifecycle` value from a newer peer is *tolerated* (additive field rule) |
| `tests/unit/network/test_pool_grant.py` | a pool grant with a wrong `network_id`/expired/`single_use` already spent is refused; vestibule: no grant ⇒ no admission even with a valid attestation |
| `tests/unit/mesh/test_placement_pool_mode.py` (beside mobility's own `mesh/placement.py` tests) | `mode: "pool"` round-trips; absent ⇒ `local`; **a fork never inherits a placement** (`mesh.json` ∈ `EXCLUDED_SIDECARS`); a malformed file degrades to `local` with a warning, never a crash |
| `tests/unit/network/test_metering_events.py` | §5.2's two records round-trip through the IR §4.2 writer (closed whitelist, `detail` ≤2048 B); `meter_seq` monotonic; slicing is disjoint and `part 1` alone carries `wall_ms`/`cpu_ms`; the close record's totals equal the union of the intervals (property test); **no field can carry money, a token or transcript content** (a shape assertion, so §7's deferral stays true) |
| `tests/unit/network/test_meter_delivery.py` | `meter_push`/`meter_ack`; a bad `sig` is rejected and audited, never written; an unacked record is re-sent; the drain barrier refuses to complete with an unacked push |
| `tests/unit/mobile/test_resource_cpu.py` | the CPU fields read from the same `rusage_info_v2` buffer are nanosecond counters and monotonic per pid; `None` on a host without libproc, like the memory fields |
| mobility's `session.sync` tests (its §11.2) | the pool adds **no** cursor test of its own: `flush` is a mode of an operation mobility owns, and a second implementation of the boundary rules is exactly the drift this document avoids |

### 9.2 End-to-end, on the real topology

The spine's §10 matrix still governs (0 / 1 / 2 peers, real machines — the
remote device an EC2 instance provisioned with the operator's `minerva_nprod`
profile). For this document's scope add one row:

| Topology | What it proves |
|---|---|
| **1 peer + 1 pool member** | E1–E12 as written, with the pool member a real pod running a clean released install |

The sequence a QA agent drives, with the artifact that is the evidence:

1. `lop network init devmesh --json` on the user's device → network id.
2. `lop network pool request --size s --hours 1 --json` → the request record.
   (Deferred: the provisioning API; until it exists, the pod is booted by hand
   and the evidence says so.)
3. On the pod: `lop --version`, `lop network join <grant> --json` → the member
   record's `kind`/`lifecycle`. On the user's device: `lop network peers --json`
   shows it `active` with `size_class: "s"`.
4. `lop sessions --all-peers --json | jq` → the placed session's `locality` and
   `placement`.
5. Run a turn; `cat ~/.local-operator/sessions/<id>/transcript.jsonl | wc -l` on
   the pod and at home **before** the next tick, then after → the delta landed.
6. `lop sessions sync <id> --json` → `dry: true` when nothing changed (the cheap
   path), a byte delta when it did.
7. `lop network pool drain <member> --json` → the audit shows
   `drain_started`, then the final sync, then `meter_close`.
8. **The R22 assertion**, both ways: (a) with a drain — the pod's instance is
   gone (`aws ec2 describe-instances` / its state), and home's
   `transcript.jsonl` line count **equals** the pod's last count; (b) with a
   `kill -9` — home holds a **prefix**, the audit carries `meter_lost`, and
   `/resume <id>` opens the session with the turns that did land.
9. Metering: the audit file's `metering` lines, summed, match the pod's own CPU
   reading within tolerance; `seq` has no gaps; the close record's totals match
   the sum.
10. **The zero-peer regression**, which is the one that catches schema damage:
    an install with no network behaves exactly as before — `lop sessions` output
    identical, the TUI sidebar frame byte-identical to the base revision's
    (§9.3), and no new file written anywhere in the config root.

### 9.3 The visual half

Two surfaces change and both are covered in `mesh-ui.md`'s evidence plan (the
sidebar's peer sections and `/network`); the requirement this document adds is
narrow: **a change that alters nothing visible must show a byte-identical
frame.** The zero-peer capture is therefore the *before* frame, and the
one-peer capture is the *after*; the command pair, per
`~/local-operator/AGENTS.md` "Visual validation":

```sh
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py after.svg 100x30
env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/sidebar_shot.py before.svg 100x30
# before is captured from the base revision's checkout (the PR's merge-base), same script
```

Plus the geometry check the repo insists on (stills show the symptom, numbers
show the cause): `app.screen.virtual_size == app.screen.size` and
`show_vertical_scrollbar` false in both frames.

### 9.4 Measurements this design owes before it is called done

Named so they are not asserted instead of taken:

* **libproc CPU read cost**, next to the existing footprint read (expect: same
  order, single-digit microseconds, no subprocess).
* **Metering write volume**: bytes/hour and lines/hour for one member, measured,
  against A7's rotation caps.
* **Sync cost**: bytes transferred for a 60 s tick on a busy session and for a
  drain, measured on a real transcript.
* **Drain-to-power-off wall time**, to size `drain_grace` and
  `SYNC_DRAIN_DEADLINE_S` from data rather than from the guesses in §3.6/§6.6.

---

## 10. Open questions, each with my recommendation

1. **Meter interval length.** *Recommend* `METER_INTERVAL_S = 60`, plus a close
   record on any state change and a forced close if no interval has been written
   for 10 minutes (a member that stalls must not look free). Evidence that would
   settle it: the measured variance between interval-summed and close-record
   totals at 15/60/300 s on a real 4-hour run.
2. **Compute time: CPU or wall?** *Recommend* both, priced differently (§5.4):
   instance-seconds from `size_class × wall_ms`, and `cpu_ms` as the abuse
   signal and the support answer. A single number would force one of the two
   questions into a lie.
3. **Does the member keep a local durable transcript at all?** *Recommend yes.*
   The member's runtime is the session's owner (it is the only writer); the
   durability at home is a *copy*, not a replacement, and making the pod
   non-durable would mean a session's own runtime could not resume after an
   internal restart. It dies with the instance, which is why §6.5 exists.
4. **Who mints the pod's device identity?** *Recommend the pod*, at first boot,
   with the provider attesting the *result* (bound to that key) rather than
   minting a private key server-side. A provider that holds the key can
   impersonate the member forever, which is the property A3's self-certifying
   identity exists to avoid.
5. **Is `pool_cap` a mesh field or a policy?** *Recommend* the member record
   carries `max_sessions` (a number the member enforces), and the mesh only
   propagates refusals. Pricing-driven concurrency limits change too often to be
   wire.
6. **Does the pre-drain sync take the fence?** **Revised: no** — this draft said
   yes and mobility §7.4 settled it the other way, correctly: a `flush` copies
   from a live writer, and the fork boundary plus the last-complete-newline rule
   are what make that safe. *Recommend* keeping mobility's answer and adding only
   the pool-side guard of §6.5 rule 2 (refuse to engage a cold copy whose member
   is still alive), which is where a second writer could actually appear.
7. **Does the meter spool have a bound, and what is it?** *Recommend* a bounded
   ring of the last `METER_SPOOL_MAX` records (recommend 240 — four hours at one
   per minute) plus the close record, which is **never** dropped; beyond the
   bound the member stops admitting turns and drains, because a member that
   cannot account for its own compute must not keep computing. A larger bound
   buys nothing: a pod whose link is down for four hours is a pod whose sessions
   are not reachable either.
8. **Does the sync carry the spend ledger row?** *Recommend yes, for free*: it
   is a transcript line (`session_spend.v1`, `session/spend.py`), so a byte-range
   delta carries it with no special case, and its replacement-state contract
   means no accumulation is needed at home. This is a nice property of the
   transcript-as-record choice and worth an explicit test.

---

## 11. Risks to watch during rollout

* **A schema that only fits today's price model.** The mitigation is stated in
  §7 and testable (§9.1: no field can carry money) — but a reviewer should check
  it again at implementation time, because `size_class` is the field most likely
  to be quietly repurposed into a price.
* **The drain barrier as a hang.** A member that waits for a home device that is
  asleep is a member billing for nothing. `SYNC_DRAIN_DEADLINE_S` must be
  short enough to be honest, long enough to land a 216 KB-class transcript, and
  the timeout path must be *audited* (`reason: "lost"`), not silent.
* **Two writers**, introduced by accident: any future code that writes to a
  session placed elsewhere without taking the fence. The fence's existing
  implementation is the guard; a test that a sync refuses without it belongs in
  the PR.
* **Metering trust.** The residual hole (§5.6) is a compromised member inflating
  `cpu_ms` within a plausible band. Watch the *first* disputes on real pods:
  if provider-attested CPU (from the hypervisor, which the control plane has and
  the mesh does not) differs materially from member-attested CPU, the control
  plane should be the source of record and this document's `cpu_ms` should be
  demoted from a billing input to an abuse signal. That demotion is a
  control-plane change and needs no mesh change — which is the design working.
* **Scope creep into the mesh.** The provisioning API and the price table are
  the two things most likely to be pulled in "because they're small". Both would
  put a provider credential inside the mesh (§7's boundary row 3).

---

## 9. Convergence round 1 — what changed here

1. **The capability/op vocabulary is the transport's, and the invented names are
   deleted.** `broker:request`, `broker:grant` and `member:admin` (§3.3, §3.5 in
   an earlier revision) and the `sync_session` op are gone; §3.3 now records the
   deletion and §3.5's least-authority list is expressed in transport names
   (`list`, `view`, `prompt`, `steer`, `stop`, `slash`). Nothing here invents a
   capability; the document adds field names, not authority names.
2. **`A8.1` is disambiguated where it is defined** (§3.2): it is this document's
   own sub-decision under the spine's A8, not a ninth spine decision id. The
   spelling is unchanged, because `mesh-prior-art.md` §2/§6 cites it by name; the
   note is what stops a reader looking for it in the spine.
3. **The `mesh.json` fork requirement is stated where it can be enforced.**
   This document already warned that `mesh.json` must join `EXCLUDED_SIDECARS`
   (§8.1's note and the test row in §9). It is now an explicit *required* change
   in the document that owns the sidecar (`mesh-session-mobility.md` §1.2),
   because a fork that inherits its parent's placement is a live session claiming
   another device's ownership — a real defect, not a tidy-up.
4. **The claim about this document's own forward compatibility is unchanged**, and
   that is deliberate: the pool's whole contribution is that `kind: "pool"`,
   `placement.mode: "pool"` and the metering stream arrive as *implementations* of
   shapes the spine and the transport already fixed. Nothing in round 1 weakened
   that, and the metering events' field lists stay owned by §5.2 with
   `mesh-incident-response.md` §4.3 reserving the names.
