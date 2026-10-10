# Mesh rolling updates — primary-driven, idle-gated, one node at a time

Status: **design, pre-implementation**; trigger half: `mesh-update-propagation.md`
(@ `ff97ee6996`). Verified against `origin/main` @ `1f0a1b909` (v0.67.16), 2026-10-05.

This document hangs off `mesh-network.md` (the spine). It consumes, by name:

- `mesh-remote-onboarding.md` §3.3 step 5–8 and §3.4 — the **onboarding lane's**
  install and upgrade mechanics (`lop-update <tag>` / `uv tool install
  local-operator==<tag>`, the relay roll, the receipts shape). That lane owns
  *how a build lands on a device*; this lane owns *what makes an update happen,
  in what order, gated on what, and how it is seen and resumed*.
- `mesh-network.md` §5 (security model) and §7 (audit) for the authority
  reasoning and the record discipline.
- `readiness.py`'s build capability row and the `peers` surface — the two
  places that already say "this peer is behind", whose remedy this design makes
  executable.
- The operator's own local discipline, `~/tools/lop-fleet-update/docs/README.md`
  — snapshot → wait for no session busy → install → wait for runtimes to leave
  → re-engage; never force. Section 4 ports it onto the member.

One sentence of intent, from the operator (2026-10-05): *"if the primary host
updates, it will also trigger an update across the mesh in the same safe,
rolling way which waits for runtimes to complete/idle and then updates them."*

---

## 0. The answer in one page

**The problem.** An onboarded device has no standing path to receive an update.
The only existing update-of-a-peer runs *inside a one-shot onboarding approval*
(`onboard.py`'s `install` step), and that record is terminal on success:
`connected` is in `TERMINAL_STATES` (`local_operator/network/approvals.py:116-119`)
and the run gate refuses it ("only an approved (or retry-eligible failed) record
can run", `approvals.py:1446-1454`, `cli.py:7401-7426`). So today, refreshing an
already-onboarded device needs **a fresh card or a hand update** — while the
`peers` surface already tells the operator, truthfully, "ask Local Operator to
update it there" (`readiness.py:984-987`). This design makes that sentence
executable.

**The decisions, in order of consequence.**

1. **Authority (§2).** A new **`update`** capability — grantable per member,
   **self-decided** by the receiving device (the `approve`/`unattended`
   precedent, `types.py:460-475`), default **absent**, captured at onboarding on
   the same card that already collects the node-side grants, retro-grantable
   with `lop network member grant`. The bound on trust, stated once: *a granted
   peer may cause THIS device to install a build this device could have obtained
   itself — a version published on the device's own channel, strictly not older
   than what is installed — and nothing else.* The peer never delivers code,
   never names a ref, never runs a command.
2. **Trigger and scope (§3).** The device that just completed a local update —
   `lop update` from PyPI, `lop update --from-snapshot <ref>` (what the
   `lop-update` script execs), or the TUI's `/update` — the **origin** — rolls
   every member that granted it, network by network, under a per-network policy
   `network.rollout_on_update` (default `auto`, override `--no-roll`). Unreachable members are marked and caught on
   the origin's next contact; they never block the rest.
3. **The rolling discipline (§4).** One member at a time; on the member, a
   five-step pass ported from the local discipline — snapshot → drain (fresh
   idle probes, bounded) → install (generation layout when the member supports
   it; exact version pin; atomic flip) → roll services and ask idle runtimes to
   retire → re-engage and verify. A busy member is **waited on, then deferred
   and retried — never forced**; there is no force arm in the standing path.
   Busyness is a **self-report from the member's own registry**; the origin
   never infers it from silence.
4. **Mixed-version tolerance (§5).** During any rollout the mesh is
   heterogeneous *by construction*. Skew degrades quietly — both-sides
   capability strings, per-op `unknown_op` refusals, additive payloads, and "an
   absent answer is never a pass". First landed instance: the read-receipt
   routing fix (#1994, `a72c1f492`), with its skew-wording follow-up in flight.
5. **Ordering, records, resumability (§6).** Origin first (by construction),
   then members in network order from a membership snapshot. One rollout record
   on the origin (`<config>/network/rollouts/<id>.json`), one lock on the
   member, audit rows on both ends; a failed or deferred member never stalls the
   pass; every state re-enters from a re-run or the next contact.
6. **Visibility (§7).** The `peers` surface and the Mesh tab gain one rollout
   segment per member, built on the build-suffix the row already carries
   (`readiness.py:969-988`, `cli.py:5803-5812`); `lop network update status
   --json` is the machine shape; the update's own summary is where the operator
   learns the roll is happening and how it went.
7. **Delivery (§8).** Five slices (authority + single-peer trigger; member
   executor; rolling orchestration; visibility; drill), with the install
   mechanics landing in the **onboarding lane** and trigger/rolling/visibility
   in this lane — plus the acceptance drill: a rolling update across
   `damian-mesh` with a deliberately busy peer, never forced.

The one-flow picture:

```
lop update (origin)            member (any earlier release in the contract)
  ├─ local update settles       (nothing until asked)
  ├─ rollout record opens
  ├─ for member in order:  ───► net_update {target}          ── probe+lock
  │    fresh probes … busy?  ◄── {state:"busy", reason}         (no touch)
  │    … later ──────────────► net_update {target}          ── drain ok
  │                           ◄── {state:"done", version,         install →
  │                                 sessions:{moved, kept}}        flip →
  │    verify build row == target                                  services roll
  └─ next member                (busy sessions retire at their own
                                 next idle boundary; turns are never cut)
```

---

## 1. Ground truth this builds on (verified 2026-10-05)

Read this section as the contract this design must not contradict. Every line
was checked in the tree; anything unbuilt is marked so.

**1.1 The one-shot onboarding approval, and the friction.**

- The onboarding runner's install step is exactly the update mechanism that
  exists today: it prefers `lop-update <tag>` when the node has it and falls
  back to `uv tool install --force --refresh local-operator==<tag>`
  (`onboard.py:1392-1414`), bounded by `STEP_TIMEOUTS["install"] = 900 s`
  (`onboard.py:99-108`), and its receipt records `method`, `from`, `to`
  (`onboard.py:1431-1436`). §3.4 of the onboarding design names the same path
  for a stale node — `lop-update` + restart + re-verify, "the same receipts
  shape" — so an onboarded node *has* been moved between builds inside a run.
- The record that carries it is one-shot. States:
  `requested → approved → connecting → connected`, with `failed` retry-eligible
  (`approvals.py:105-142`); `connected` is terminal and "never re-openable,
  never re-runnable" (`approvals.py:116-119`); the CLI refuses with
  `approval_not_runnable` (`approvals.py:1446-1454`, `cli.py:7421-7426`).
- Consequence (§0): the standing path must be built, not found. The natural
  handles — receiver-side grants, an approval store whose receipts we can
  model on, a machine-readable update receipt (`update_report`,
  `update.py:6781-6810`) — all exist.

**1.2 The local update machinery (the executor half is mostly built).**

- `lop update` / `/update` run `update.perform_upgrade` (`update.py:5310`),
  which on uv-tool installs lands each build in **its own generation** and
  flips `~/.local/share/lop/current` atomically (`update.py:1240-1278`,
  `install_into_generation` at `update.py:2717`; a failure at any pre-flip step
  removes its own tree; `_PARTIAL_TTL_S = 3600 s` covers `kill -9` debris,
  `update.py:1304-1309`). A running process keeps its own tree; "a mixed-
  generation fleet is an accepted steady state" (`update.py:1276-1278`).
- After the install, a daemon-refresh stage rolls the supervised services —
  including the **network relay** (`update.py:6097-6148`), whose step asks the
  running process the second staleness question and kickstarts a relay left a
  generation behind (`relay.py:10846-10903`; landed as #1972, `b969590a4`).
- Runtimes are **never stopped** by an update; they retire themselves at an
  idle boundary (`process._should_refresh` → `_refresh_for`,
  `process.py:391-413`, `:1883-1960`), staggered fleet-wide via the registry's
  `updating` marker (`process.py:1833-1880`), with an update window that spools
  admissions rather than refusing them (`process.py:1905-1929`). The ask face
  exists as a top-level verb: `lop refresh --all` (`cli.py:1482-1516`; a
  runtime answers with its own sentence, `buildwatch.py:139-141`, and a busy
  one answers `kept: <reason>` — `types.py:794-801`, `server.py:6295`).
- The window may end the opposite way: an aborted handover keeps the build the
  runtime loaded and drains the spooled messages back (fail-open). That is the
  house direction and section 4 keeps it.

**1.3 The local fleet discipline (the part being ported).**

`~/tools/lop-fleet-update/docs/README.md` is the measured discipline this
design ports onto a member: snapshot the live fleet → wait until **no session
is `busy`** (bounded; the caller is excluded by ancestor walk so the drain can
come out clean) → install → wait (bounded, 60 s) for old runtimes to notice the
settled marker and leave → **re-engage every session from the snapshot** with a
one-line "you were restarted; re-check what actually completed" nudge → verify
the installed build and report how many came back. `--force` exists there as a
**human** escalation and "is still never implied"; a reached deadline exits
non-zero. Its README also names the durable fixes it wants in the product — a
drain-aware update path, generation-directory installs, a turn-boundary check —
of which the generation layout and the concurrent daemon-refresh stage have
since landed (#1838, `fff390360`; #1972). The remaining half — **re-engaging
unwatched sessions** — is the step this design must carry, because "a runtime
with no viewer attached is never looked at by a TUI" and nothing else performs
it.

**1.4 The mesh surfaces that already name the problem.**

- The **build stamp rides the handshake** (`handshake.py:190-217`,
  `PeerHello.peer_build` at `:538/:626`), and `readiness.py` composes the build
  verdict from it. The row is **ADMISSION-class**: "a peer on another build is
  not admitted to run this device's work" (`readiness.py:67-77`, `:744-745`).
  The suffix is the operator's sentence: `build 0.67.11 — behind this device
  (0.67.13); ask Local Operator to update it there` (`readiness.py:969-988`);
  the `peers` row appends it verbatim (`cli.py:5803-5812`).
- Readiness already models an old peer: `peer_too_old` rows, `not_asked` rows,
  and the axiom "an absent answer is never a pass" (`readiness.py:10-16`);
  unknown stamps render the old line **byte for byte** (`cli.py:5803-5807`).
- **Grant vocabulary.** `GRANTABLE_CAPABILITIES` (`types.py:444-457`) is what
  `member grant` may add; `admin` is absent on purpose; `approve` and
  `unattended` are the **self-decided onboarding scopes** the deciding device
  records in a PEER's row in its own record, no admin row required
  (`types.py:460-475`; the write rule at `relay.py:1627-1700`). Roles do not
  carry them; every consumer resolves them from the device's own member row.
- **Wire discipline.** Link features are capability strings, both-sides,
  unknown ignored (`wire.py:110-144`, `:715-722`); the link version refuses
  loudly but moves rarely (`wire.py:691-705`); a session-level addition "degrades
  per op via the existing unknown-op rule" rather than bumping the link
  (`mesh-transport-identity.md` §6.4). New peer ops must be registered in two
  **closed tables** — `OP_CAPABILITY` (`types.py:498-620`) and a slice's
  `SLICE_PEER_OPS` membership (`relay.py:334-345`) — with totality tests that
  fail by name when a capability row is missing.
- **The relayed-terminal carrier must not be used for this.** A relayed
  connection may not type `/update` into an owner's terminal: the verb is
  deliberately outside `RELAYED_TERMINAL_SLASH` and the routed dispatcher has
  no branch for it — "those come back as its own honest sentence"
  (`types.py:994-1037`, esp. `:1017`). The mesh update path is therefore an
  **op**, not a slash command, and it executes on the member's own relay.
- `MOVE_WAIT_POLL_S = 5 s` / `MOVE_MAX_WAIT_S = 1800 s` and "each retry is a
  fresh idle probe" (`mobility.py:63-94`) are the established shape for
  waiting on a busy owner over the mesh; long handlers ride the **slow-op
  pool** (`relay.py:310-323`, registration at `mobility.py:5106-5113`).

**1.5 What is NOT true today (so nothing below is inferred from it).**

- No peer op can trigger an update; there is no `net_update`.
- No capability covers it; `update` is not a word in any table.
- No rollout record, driver, or resumable state exists.
- `net_readiness` asks for the peer's facts and does **not** report session
  busyness; there is no remote "is this peer idle" fact today, and this design
  does not invent one (§4).

---

## 2. The authority model (the crux)

**The question.** An update executes code on a peer — the maximal act in the
mesh vocabulary's universe ("a session is arbitrary code execution by design",
`mesh-network.md` §5). What is the smallest standing authority that lets a
mesh do this safely, when the only existing consent is a one-shot card?

**Options.**

| Option | Shape | Why not (or why) |
|---|---|---|
| (a) A per-update card | Every update files an approval record; the operator decides per device per release | The act is low-variance and high-frequency — same release, same channel, one line of untrusted *input* (none: there is no content to review). Cards would recreate the exact friction the request exists to remove; approval fatigue is a worse security posture than one explicit standing consent with a stated bound. Also inherits the very one-shot machinery that produced the friction. **Rejected as the default path**; remains the path for members without a grant, unchanged. |
| (b) Reuse `admin` | `net_update` gated on the requester's `admin` row | `admin` is a network role granted by an admin invite + human SAS; it cannot be granted per-member (`types.py:425-431`), it is network-wide while the decision is per-device, and it would let a network administrator nudge an install that this device's own operator never consented to. The mesh's own precedent separates "a change to the network" (admin) from "a trust decision about THIS device's own files and sessions" (self-decided scopes). **Rejected.** |
| (c) New `update` capability, self-decided | Added to `CAPABILITIES`, `GRANTABLE_CAPABILITIES`, `SELF_DECIDED_SCOPES`; granted per (requester, receiver) direction; enforced per op on the receiving device | Fits every constraint above: per-member, receiver-side, explicit, revocable, and consistent with `approve`/`unattended`. **Recommended — this document's decision.** |

**Decision: (c).** Concretely, in the tables this repo already has:

- `CAPABILITY_WORDS["update"] = "install a newer build here when this peer asks,
  and only from this device's own update channel"` (a `words` string that names
  the bound, not just the act — `types.py:480-493`).
- `GRANTABLE_CAPABILITIES` gains `"update"` (grantable per member via
  `lop network member grant`; **no role carries it**, exactly like `approve`
  and `unattended` — adding it to a role would silently widen every existing
  member).
- `SELF_DECIDED_SCOPES` gains `"update"`: the decision governs **this device's
  own install**, the grant is deliberately receiver-side, and no wire op may
  write another device's copy (`types.py:460-475`, `relay.py:1665-1672`). The
  deciding device's own operator makes it — on the onboarding card's grants
  step (which already runs `member grant <net> <mac-id> approve unattended`,
  `mesh-remote-onboarding.md` §3.3 step 9) and, for existing members, with the
  same verb by hand.
- `OP_CAPABILITY["net_update"] = "update"`; `"net_update"` joins the slice-op
  table (`relay.py:334-345`); `"peer_update"` joins `LOCAL_OPS` (the `peer_*`
  boundary rule: a viewer asking its **own** relay to ask a peer is a local
  act).

**The bound on trust, stated once and repeated in the receipt's own words.**
The capability authorises *a nudge to fetch and install*, not *a delivery*:

1. The trigger carries a **target identity** — the origin's own build stamp
   `{version, source_ref}` (`update.py:1053-1064`) — and nothing else. No code,
   no wheel, no ref to check out, no command line.
2. The member resolves the **version** through **its own installer against its
   own channel** — for the standing path, the published release on PyPI, exact
   pin (`local-operator==<version>`), with `--refresh` so a cached index cannot
   hide it (`onboard.py:1392-1396` is the same spelling and its comment carries
   the measured reason).
3. **Monotonicity.** A target not strictly newer than what is installed is
   refused: `already_on_target` is a no-op receipt, `ahead_of_target` is a skip.
   There are no downgrades in the standing path.
4. **The member resolves `target.version`; `source_ref` is identity, never
   fetched.** A non-empty `source_ref` rides the record as the origin's build
   identity — the operator's own release flow produces exactly that shape
   (`lop-update` from `main`: version published, ref set) — and it is never
   fetched, checked out, or installed from. Refusal keys on the version being
   unresolvable on the member's own channel (`target_not_published`), or on an
   editable/dev checkout that has no channel at all (`editable_install`;
   "dev-tree skew is out of scope by design", `update.py:347`) — refused by
   name, not attempted.

So the worst a compromised member can do with the grant is make a device run a
*newer published release* earlier than the operator would have, bounded by
one-pending-target (one update in flight per member) and the monotonicity
check. It cannot ship code, cannot pin a git ref, cannot downgrade, and cannot
make the device run anything the release process did not publish. That is the
whole reason a standing grant is defensible here where a standing
"sessions-allowed" grant would not be.

**Refusals, by name** (the family register — cause, remedy, one string):

| Code | Sentence skeleton |
|---|---|
| `update_not_granted` | "`<device>` has not granted `update` to this device; its operator can grant it in the Mesh tab on `<device>`, or run `lop network member grant <net> <this-device> update` there." |
| `unknown_target` / `target_not_published` | "the target `<version>` is not resolvable on this device's channel; nothing was installed." |
| `ahead_of_target` | "this device runs `<v2>` and the target is `<v1>` — an update never moves backwards." |
| `editable_install` | "this device runs from a development tree; updates are out of scope there (`lop update` by hand is the path)." |
| `update_in_progress` | "an update is already running here (`<run-id or pid>`); retry when it settles." |
| `busy` | "`<n>` session(s) are busy; the update waits — nothing has been touched." (A *state*, not a failure; §4.) |

**The trigger's authority is settled (recorded 2026-10-09, `mesh-update-propagation.md`
§2).** The operator's standing instruction — *"if the primary host updates, it will also
trigger an update across the mesh in the same safe, rolling way which waits for runtimes to
complete/idle and then updates them"* (2026-10-05) — IS the authority for a primary's
decision to propagate its own completed update. His decisions on this note's open questions
are recorded (PR thread, 2026-10-05: roll after the update settles, with the `--no-roll`
escape; published releases from each device's own channel; re-engage displaced sessions);
§9 items 1–3 are answered by that record; items 4–6 remain the implementation's to settle.

**What is deliberately NOT decided here: transitive or network-wide grants.**
The grant is per direction and per member. There is no "follow whoever
updates" mode; an origin rolls exactly the members that granted *it*.

---

## 3. Trigger and scope

**Decision: the trigger is the origin's own completed update, by default, under
a per-network policy.**

- **Origin.** The device whose local update just settled — any of the
  spellings that run this product's own updater: `lop update` (PyPI), the
  `--from-snapshot <ref>` form the `lop-update` script execs
  (`~/.local/bin/lop-update`), and the TUI's `/update`. "Primary" in the
  operator's words is this: the device you actually update; nothing in the
  schema changes, and any member that its peers have granted may be an origin.
- **Policy.** `network.rollout_on_update`: `auto` | `ask` | `off`, defaulted in
  the config file's existing `network:` section (beside `keepalive_s`,
  `link_idle_s`). `auto` = roll after the local update settles (the operator's
  request, and the default); `ask` = the same, one confirmation; `off` = the
  manual verb only. `lop update --no-roll` overrides for one run. Rationale for
  the default: the operator's act — running the update — is already the
  explicit decision about *this build*; the members' standing consents are the
  gate that matters, and a member without the grant is never touched.
- **Scope.** For each active network the origin belongs to: the active members,
  minus self, **minus members whose own row refuses `net_update`** — the grant
  is receiver-side and cannot be read, so the origin learns it from the member's
  `not_authorised` answer (recorded `skipped` (`no_grant`), with the grants
  remedy above; `mesh-update-propagation.md` §3), **minus members that do not
  advertise the update feature string** (§5; recorded
  `skipped` (`predates_rolling_updates`)), **minus `kind: "pool"` members**
  (ephemeral pods are replaced, not updated — `mesh-compute-pool.md` §3.5;
  recorded `skipped` (`unsupported_kind`)). What
  remains is the roll set, ordered by the network's member order from a
  **snapshot** taken when the record opens (a member admitted mid-rollout rides
  the next rollout).
- **Target.** The origin's build stamp at trigger time. Members on the release
  channel install *that version* exactly — not "latest" — because the record,
  the readiness comparison, and the drill's acceptance surface all compare
  versions exactly (`mesh-remote-onboarding.md` §3.4, "the tag is pinned in the
  record"). A newer release published mid-rollout is the *next* rollout's
  target; this one converges the fleet onto one build.
- **Offline / unreachable.** A member that cannot be dialled within the existing
  connect budget is recorded `unreachable` and the pass continues. Catch-up is
  **the origin's job, lazily**: the rollout record stays active, and the
  origin's relay re-probes `pending`/`deferred`/`unreachable` members on (a)
  the next link-open from that member, (b) a bounded cadence while the record's
  window is open (default: every 5 minutes), and (c) `lop network update
  --resume <id>`. "Caught on next contact" is therefore literal: the next
  successful contact with a pending member triggers its update if the record is
  still active.

**The one new wire op** (`net_update`, slow-pool), shaped as
*apply-or-answer*:

```
net_update  { "target": {"version": "0.67.16", "source_ref": ""},
              "rollout": "ro_…" }
        →  { "state": "busy"|"done"|"already_on_target"|"ahead_of_target"
                       |"refused"|"failed",
             "code": "", "reason": "", "method": "", "version": "",
             "sessions": {"moved": 0, "kept": 0}, "updated_at": 0.0 }
```

One op, idempotent; each call is a **fresh probe**, and a call that observes
idle may perform the install inside its own bounded deadline (a slow op;
owner-side bound covers one install — the same 900 s the onboarding runner
gives that step — and the caller waits owner + slack). Draining is *not* inside
the call: a busy member answers `busy` in milliseconds with its own reason, and
the caller re-asks (fresh probe per retry; `MOVE_WAIT_POLL_S` shape, §4). A
second distinct target while one is in flight is refused `update_in_progress`;
a re-trigger of the same target is naturally idempotent.

`code` carries the member-side class when `state` is `refused`
(`target_not_published` / `editable_install` / `update_in_progress`); `method`
discloses the install shape on `done` (the onboarding runner's own values,
`onboard.py:1431-1436`); `reason` is the member's own sentence for
`busy`/`failed`; `sessions` counts what the pass moved and what it kept. The
member's checks run in one order — target validation first (the no-action and
refusal outcomes need no idle), then the fresh busy probe — so every reply
state has exactly one producer.

**The one vocabulary map** (S1 freezes it, one test per row). Three vocabularies
meet in this design — the §2 refusal codes, the reply above, and the record's
member states (§6.1) — and this table is their single reconciliation:

| Producer | Reply (`state`, `code`) | Record (`state`, `code`) |
|---|---|---|
| exclusion before any call: no grant / no `mesh-update-v1` / pool member | — | `skipped` (`no_grant` / `predates_rolling_updates` / `unsupported_kind`) |
| dial fails | — | `unreachable` |
| fresh probe: sessions busy, inside the wait budget | `busy` | `draining` |
| the slow call itself, installing | — | `applying` |
| pass completed | `done` | `done` |
| no action needed | `already_on_target` / `ahead_of_target` | `already_on_target` / `ahead_of_target` |
| member-side terminal refusal | `refused` (`target_not_published` / `editable_install`) | `refused` (same code) |
| retry-class: busy past the budget / no answer / an update already in flight | `busy` / — / `refused` (`update_in_progress`) | `deferred` (`busy` / `no_answer` / `update_in_progress`) |
| install or relay-roll failure | `failed` | `failed` |

`draining` and `applying` exist only on the record side (the origin's view of
its own wait and its own in-flight call); `rolling` is deliberately absent —
the services roll and the re-engage sit inside the one bounded call, so the
observable boundary is the reply set above.

---

## 4. The rolling discipline, and idle-gating ON the member

**Decision: one member at a time; the wait and the gate live on the member; the
origin polls; nothing is ever forced.**

**4.1 One at a time, and what serialisation buys.** The origin triggers one
member, drives it to a terminal state (`done`, or a recorded deferral), and only
then moves to the next. Serialisation is not what makes each swap safe — that
is the member's own drain and the generation layout — it buys: a bounded burst
of installer/relay churn on the fleet, legible attribution when something goes
wrong (one member in flight, one record entry), and the operator's stated
"rolling" intent. The origin itself is "rolled" before any member by
construction: its update *is* the trigger's precondition.

**4.2 The member pass** — five steps, ported one-for-one from
`~/tools/lop-fleet-update/docs/README.md`, with the product's own machinery
substituted where it exists:

1. **Snapshot.** The member's relay reads its own session registry (`registry.
   scan`, the same read `process._another_move_in_flight` uses,
   `process.py:1864-1873`) and records the live set. No viewer, no TUI, no
   operator needed on the member.
2. **Drain.** Wait until **no session is `busy`** — the registry walk plus the
   busy bit (`mesh-update-propagation.md` §4), re-checked under the update lock
   before the install. The wait budget is the record's window: each ask is a
   fresh probe, a member that answers `busy` is parked and re-asked on the
   origin's cadence (record `deferred`, code `busy`, detail `2 sessions busy
   since 10:02`) while the pass continues. The drain never
   stops, signals, or signs anything on the member — the fleet tool's own
   words, kept as the bound on this step.
3. **Install.** Under the member's update lock, re-probe idle (a turn may have
   started between calls; a fresh `busy` wins and nothing is touched), then:
   - resolve the exact version through the member's own channel with the
     onboarding lane's spellings (`onboard.py:1392-1414`): the generation-layout
     updater first — the lane's `lop update --to <version>` (§8.1) on a
     repo-less member, or `lop-update <tag>` where both a repo checkout and the
     script exist (it execs `lop update --from-snapshot <tag>` and exits 1
     without a repo, `~/.local/bin/lop-update`); the classic uv-tool path is the
     disclosed last resort — an in-place install, safe here because the drain
     succeeded moments before, and *only* then;
   - verify the built tree's own metadata agrees with the announced version
     before it becomes visible (`install_into_generation` step 2 — the guard
     against the stale-index incident, `update.py:2748-2752`);
   - flip the pointer atomically. A failure at any pre-flip step removes its
     own tree; nothing observable moved.
4. **Roll the services, ask the runtimes.** The existing daemon-refresh stage
   brings the supervised services — including the **relay itself** — onto the
   new build (`update.py:6097-6148`, `relay.py:10846-10903`); the operator sees
   a named "expect a brief mesh blip" line, which is the moment they learn why
   a peer's link dropped. Then ask idle runtimes to retire onto the new build
   (`lop refresh --all` semantics; `refresh_if_idle` re-checks the runtime's
   own `may_refresh` predicate and a busy runtime answers `kept: <reason>` —
   `types.py:794-801`), bounded at ~60 s. **Busy runtimes are left where they
   are** and retire at their own next idle boundary; the receipt says how many,
   honestly.
5. **Re-engage and verify.** Re-engage the snapshot's sessions — this is the
   half the fleet README says "nothing else performs for unwatched sessions" —
   with a truthful one-line nudge (a session whose runtime was only idle-retired
   gets "this device moved to `<version>`; re-check"; nothing claims a death
   that did not happen). Then verify `installed_version() == target` and answer
   `done` with the receipt.

**4.3 Busy, detected where it is knowable: on the member.** The origin cannot
see, and must not infer, whether a peer is busy — the existing `net_readiness`
call-takes-all facts deliberately do not include session state
(`readiness.py:633-656`), and a silent peer is not an idle one. So the rule is
structural: **the member's own registry is the sole authority on the member's
busyness**, and every answer is a fresh self-report. The origin's "waiting"
display is the member's words, not the origin's guess. (The same discipline as
mobility's "each retry is a fresh idle probe"; `mobility.py:87-94`.)

**4.4 Failure and timeout discipline.**

- **A reached deadline waits and retries; it never forces.** The default
  per-member behavior on a missed drain budget is *defer*, and the deferral
  re-enters from the catch-up hooks (§3). The standing path has **no force
  parameter at all** — `--force` remains what it is in the fleet tool: a named,
  human escalation, hand-run, never implied by anything an agent or a peer can
  send. If a future review wants a forced arm, it must be a new, named,
  human-only decision recorded here.
- **Install failure** answers `failed` with the installer's own tail (the
  `_install_failure_detail` shape, `onboard.py:966-1006`) and is
  **retry-eligible**; the generation layout guarantees the member is still on
  its old, working build.
- **Relay-restart failure** answers `failed` with the existing remedy sentence
  (`lop network install`). Sessions are unaffected.
- **One at a time on the member too**: a single update lock (pid-carrying,
  file-locked like the approvals store, `approvals.py:218-222`), stale-supersede
  after the `STALE_RUN_AFTER_S`-shaped bound (`approvals.py:148-162`) so a
  killed relay cannot wedge the member forever.
- **The origin's wait is bounded and visible**: `lop network update --wait`
  polls at 15 s; every poll's answer is recorded on the member's entry.

**4.5 The member executor lives in the relay** — the always-on process — and
spawns the installer as a child (the updater is an external `uv`/installer
invocation; `update.py:5300-5307`). It survives its own restart by construction:
the generation layout means the running relay keeps its tree, and the pointer
flip is what the restart resolves to (why the daemon units name the stable
shim: `update.py:2215-2268`; a relay left a generation behind names the same
fact, `relay.py:10854-10860`). If the relay dies mid-install, no pointer
moved; the stale lock is superseded on the next trigger.

**4.6 Where the drain gate is *not* needed, stated so nobody adds it.** The
origin's *own* update already ran the local discipline (runtimes retire at idle,
never stopped). Members do not wait for each other; a member converges at its
own pace and the record tracks it. And nothing waits for a member's **busy
sessions to finish before updating the install** once the drain has cleared —
the generation layout makes post-swap races survivable, and the update window
spools admissions (`process.py:1905-1929`), so a turn that begins mid-swap
completes on the old tree and the successor runs the new one.

---

## 5. Mixed-version tolerance

**Principle (decided):** during any rollout the mesh is heterogeneous *by
construction* — that is what a rolling update is — and skew **degrades
quietly**: every surface keeps working, the one capability that cannot cross a
version boundary is refused **by the existing per-op rules with a truthful
sentence**, and no surface treats an unknown as a failure or an absence as a
pass.

The mechanisms already in the repo, which this design reuses rather than
extends:

1. **Both-sides feature strings.** A new feature is used only when both sides
   advertise it; absent means "old peer" and unknown is ignored; the link
   version does not move (`wire.py:110-144`). The update feature ships as
   `mesh-update-v1`, and the origin **checks it before its first request**, so
   an old member never has to compose a refusal it did not ask to make (the
   `PEER_READINESS_V1`/`MCP_DEFS_V1` precedent, `wire.py:123-135`).
2. **Per-op refusal.** New peer ops ride the existing `unknown_op` rules
   (`mesh-transport-identity.md` §6.4); session-level additions degrade per op
   rather than at the link.
3. **Additive payloads.** New keys are ignored by builds that do not know them;
   a change to an existing field's *meaning* is a new feature string, not a
   reinterpretation (the `offer_digest` rule, `mesh-credentials.md:369-372`).

**The first instance of the principle, cited as precedent:** the read-receipt
routing fix (#1994, `a72c1f492`, merged 2026-10-05) and its in-flight
skew-wording follow-up (`fix/receipt-skew-wording`; edits in progress in its
worktree at this revision). A desktop read-receipt for a session that lives on
a peer **routes to the owner**; where the owner's build predates the receipt op
(`net_session_receipt`), the owner's raw authoriser refusal ("is not an
operation this build dispatches") is **not** echoed — the route classifies the
skew and composes a sentence carrying the three truths: where the mark lives,
why it did not clear (an older build), and that it clears when that device
updates. The mixed-build pair keeps working; the one op that cannot cross is
refused by name and rendered as a next step, never as an unknown failure.

**Surfaces that must comply, here** (the compliance list a reviewer can check
off):

| Surface | Must hold | How |
|---|---|---|
| `net_update` op | An old member answers `unknown_op`; the origin never asks one | origin checks `mesh-update-v1` first; result recorded `skipped` (`predates_rolling_updates`), remedy = the existing "update it there" path |
| Capability rows | A member that predates `update` never resolves it against a requester; an unknown name in a row is inert | capability rows are read per use from the device's own record; unknown names cannot grant anything |
| Receipts | Old origins (none can exist for `net_update`) and newer members | additive keys; unknown ignored; versions compared, never assumed |
| Rollout record | Schema additive for readers | unknown fields ride; states are closed but new states must be a decision here |
| Readiness / peers | `unknown` stamp renders the old line byte for byte; `peer_too_old`/`not_asked` never marked FAIL | existing pins are the A/B rule (`cli.py:5803-5807`, `readiness.py:10-16`); the rollout segment is *absent*, not "failed", for such peers |
| Offload gating | A `behind` build remains ADMISSION-class and refuses offloads with the existing sentence | unchanged; the rollout is the *mechanism that clears the row*, not a relaxation of it |
| The update executor itself | Never assumes the member's build beyond the contract | the member's executor is the member's own build; the origin only reads the op's reply |
| The member's own viewers | Runtime/viewer skew stays handled by buildwatch + the existing skew notices | unchanged (`fix/build-skew-completion`, #678; `process._should_refresh`) |

**The two asymmetries, made explicit.** (i) A member **ahead** of the target is
skipped (`ahead_of_target`) — a rollout never moves anything backwards, and a
device updated by hand past the fleet is left alone until the origin catches
up. (ii) A member on a **source/editable build** cannot take the standing path
at all (`editable_install` / `target_not_published`): dev trees are out of
scope by design (`update.py:347`), and the sentence says so rather than
pretending.

---

## 6. Ordering, safety, and resumability

**6.1 The record.** One JSON record on the origin,
`<config>/network/rollouts/<id>.json` (`ro_<crockford>`), written under the
approvals store's discipline: atomic write, per-record lock, reader-never-
creates (`approvals.py:193-227` are the pattern to copy, not the store). Shape:

```
{ "id": "ro_…", "state": "active"|"done"|"abandoned"|"superseded",
  "created_at": …, "window_s": 86400,
  "network": "…", "origin": "d_…",
  "target": {"version": "0.67.16", "source_ref": ""},
  "members": [
    {"device": "d_…", "name": "cloud-node-1",
     "state": "pending|draining|applying|done|already_on_target|ahead_of_target|deferred|failed|unreachable|refused|skipped",
     "code": "",
     "detail": "2 sessions busy since 10:02", "version": "",
     "receipts": [{"at": …, "state": …, "detail": "…"}]}]}
```

Member states are the map's record column (§3); `code` distinguishes
causes where the state alone cannot (`skipped`: `no_grant` /
`predates_rolling_updates` / `unsupported_kind`; `refused`:
`target_not_published` / `editable_install`; `deferred`: `busy` / `no_answer` /
`update_in_progress`).

The member keeps **no mirrored record** — deliberately. Its ground truth is the
install itself (dist-info + `.lop-source`), its lock, and its audit rows; the
origin's record is the orchestration view. Two stores that must agree would be
a new failure mode for no gain.

**6.2 Ordering.** Origin → members in the snapshot's order, one at a time
(§4.1). A member that is `done` or `already_on_target` is skipped on resume
(the re-probe is cheap and idempotent); the first non-terminal member in order
is where a resume continues.

**6.3 Resumability, case by case.**

| Interruption | What survives | What continues |
|---|---|---|
| Origin process dies mid-pass | the record (last observations) | `lop network update --resume <id>`, or the next trigger re-opens it; catch-up hooks re-probe pending members |
| Origin's relay restarts mid-pass | the record | the driver re-probes on link-open/cadence while the record's window is open |
| Member's relay restarts mid-install | nothing observable moved (pre-flip failure removes the tree) | next trigger re-runs the install; the stale lock is superseded |
| Network partition / member offline | record marks `unreachable` | caught on next contact (§3) |
| A newer update lands on the origin mid-pass | new record supersedes | new record re-snapshots; the old one keeps its history, marked `superseded` |
| Record's window (24 h) lapses | record archives with a sentence | "this rollout is no longer active — re-run `lop network update` to roll what remains" |

**6.4 What is audited (both ends, append-only JSONL, existing rotation).**

- Origin: `update_rollout_started`, `update_member_triggered`,
  `update_member_state`, `update_rollout_done` (one row per semantic change,
  never per poll — the audit module's cost rule, `audit.py:44/:515-534`).
- Member: `update_requested`, `update_refused` (the §3 map's code),
  `update_started`, `update_completed` / `update_failed`, alongside the
  existing `update_report` machine line the desktop already parses
  (`update.py:6781-6810`). One rollout is a handful of rows per device — the
  retention budget is untouched.

**6.5 Safety notes, so the shape is not re-litigated.**

- The rollout **never cuts a turn or edits an existing transcript**; it moves
  builds and services, and its one session-visible write is the step-5
  re-engage nudge (§4.2). The one loss mode is a turn that begins in the
  seconds around a swap on a member *without* the generation layout, which the
  drain is exactly sized to prevent; the receipt's `method` field says which
  install shape ran.
- The origin **must itself have settled** before rolling: its own runtimes may
  still be retiring (that is fine — the origin's sessions are not the members'
  concern), but its install and services must be on the target, because the
  record's target is its own stamp.
- The pass **stops on nothing**: `failed`/`deferred`/`unreachable` members are
  recorded and the next member proceeds. A member that fails verification
  twice is still retried — bounded by the record's window, not by a counter —
  and the record's window is the one bound that ends automatic retries.

---

## 7. Visibility

**7.1 The peers surface (build on it, do not replace it).** The row already
carries `build <v> — behind this device (<own>); ask Local Operator to update
it there` (`readiness.py:969-988`). With a rollout known to this origin, the
segment extends after the build clause, from the record's last observation:

```
cloud-node-1   reachable  build 0.67.13 — behind this device (0.67.16); update
                          pending (waiting: 2 sessions busy since 10:02)
cloud-node-1   reachable  build 0.67.16  update done (10:04)
cloud-node-1   reachable  build 0.67.13  update deferred — will retry when idle
```

New `--json` fields on the row: `rollout: {state, target, detail, updated_at}`
(absent when nothing is known — the byte-for-byte old-line rule holds).
`lop network ready`'s rows and verdicts are **unchanged**; its `remedies` list
gains one additive line where a grant exists ("or run `lop network update
<peer>` from this device"), leaving the human sentence byte-identical.

**7.2 The Mesh tab.** One rollout entry per member row (state word + reason,
in the family vocabulary), plus one aggregate line for the pass ("rolling to
0.67.16 — 2/4 done, 1 waiting, 1 unreachable"). The state vocabulary is the
record's; the pixels are the design round's, with before/after frames against
the real app.

**7.3 The update's own summary is the operator's first sight of it** — the
relay-roll comment's rule ("a peer's link drops for a moment — the update's own
summary is where they learn why", `relay.py:10882-10885`). After `lop update`
settles:

```
updated this device to 0.67.16
rolling to 3 peers: cloud-node-1, dev-vm-2, gpu-pod-3
  cloud-node-1  updated (was 0.67.13)
  dev-vm-2      waiting — 2 sessions busy since 10:02 (will retry)
  gpu-pod-3     unreachable — will retry on next contact
2/3 on 0.67.16; `lop network update --resume` to retry the rest
```

**7.4 The manual verbs.** `lop network update <peer> | --all | --status |
--resume [<id>]` (a bare `--resume` addresses the most recent active record —
which is why §7.3's summary prints it bare), `--wait/--json`, documented under
the `lop network` group;
`lop network update` rolls this device's grant-holding members, `lop update`
remains "this device only". The two must not be confused in help text or
sentences.

---

## 8. Delivery plan and the acceptance drill

**8.1 Lanes.**

- **Onboarding lane** owns install mechanics: the exact-version pin routed
  through the generation installer for the no-repo case (`lop update --to
  <version>` is the smallest addition; today `perform_upgrade` takes a target
  but the CLI only feeds it the fresh-check result, `update.py:5310`,
  `cli.py:1692-1770`), plus any `lop-update`-spelling changes. This design
  depends on it; the classic uv-tool fallback ships until it lands.
- **This lane** owns: the capability + op tables, the trigger, the rolling
  driver and record, the member pass, visibility, the drill.

**8.2 Slices** (each a PR with the standard rounds: coder + reviewer +
qa-tester; UI slices add designer; the flow's first real use adds ux-reviewer).

1. **S1 — Authority and a single-peer update.** `update` in the capability
   tables + words + totality tests; `net_update` peer op (slow pool) +
   `peer_update` local op + `lop network update <peer>`; the §2 refusals and
   the §3 vocabulary map, with tests per row. Acceptance: two-device rig;
   positive cell moves an idle member one version; refusal cells per the map
   (no grant, old peer, busy, downgrade, editable, update already in flight).
2. **S2 — The member pass.** Snapshot/drain/install/roll/re-engage, the update
   lock, receipts. Acceptance: unit cells + a two-relay rig with a synthetic
   busy session — the pass must **wait**, then update after the turn completes,
   and the transcript must be intact (never-force proof). Install failure
   carries the tail; the classic-path disclosure renders.
3. **S3 — Rolling orchestration.** The record, serial pass, deferral, the
   catch-up hooks (link-open + cadence + `--resume`), the policy key, the
   `lop update` hook and summary. Acceptance: three-member rig including one
   offline member (deferred, then rolled on "next contact") and one busy member
   (deferred, then rolled); resume across an origin restart; supersede on a
   second update.
4. **S4 — Visibility.** Peers segment + `--json`, Mesh tab rows, `ready` remedy
   addition. Acceptance: rendered before/after frames (real app, per the visual
   validation rules) + the JSON pins.
5. **S5 — The drill** (below), run against the real topology.

**8.3 The acceptance drill — rolling update across `damian-mesh`, never
forced.** A runbook in the `mesh-onboarding-drill.md` shape (commands, expected
receipts, what to keep), executed on the real topology (the Mac origin +
`cloud-node-1`), with the one addition this feature exists for:

- **Preconditions.** A release published; `damian-mesh` has both devices on the
  previous build; the node holds the Mac's `update` grant; synthetic sessions
  only; isolated config roots for every local harness; TUI boots unset every
  `CMUX_*`; evidence into one matrix file.
- **Step A — the busy peer.** Start a **deliberately long turn** on a synthetic
  session on the node (a controlled sleep-and-report prompt), then run
  `lop update` on the Mac. Expect: the rollout opens, reaches the node, answers
  `busy` with the node's own reason, and **nothing on the node is touched** —
  prove it with the node's session still running and its build stamp unchanged.
  The record shows `deferred` (`busy`); the pass has continued (nothing blocked).
- **Step B — the completion.** Let the turn finish (or pick the moment), then
  either wait for the cadence hook or run `lop network update --resume`. Expect
  the node to update **after** the idle probe: install receipt with `method`,
  relay roll line ("expect a brief mesh blip"), runtimes retired-or-kept with
  counts, `done`; the synthetic session's transcript **intact and complete**,
  and its successor on the new build. This is the never-forced proof: the turn
  was waited on, not cut.
- **Step C — offline catch-up.** Take the node's link down (or stop its relay),
  re-run a rollout; expect `unreachable` recorded and the pass to finish.
  Restore contact; expect the catch-up hook to roll it; record the audit rows
  on both ends.
- **Step D — resume.** Kill the origin's update mid-pass (between members);
  re-run `--resume`; expect continuation at the first non-terminal member and
  no double-update of a `done` member.
- **Evidence kept:** the rollout record, both ends' audit rows, `lop network
  peers` before/after, the node's `lop --version` and `readlink
  ~/.local/share/lop/current`, the update reports, and the busy session's
  transcript (the never-force artifact).
- **Fallback topology** if the remote leg cannot run: two local config roots +
  a second relay on loopback (the mesh suite's own pattern), stated as such —
  never implied coverage.

**8.4 Out of scope, named so it is not implied.** Downgrades; a source-ref
channel for members; Windows members (no generation layout,
`update.py:2344-2352`); pool members (replaced, not updated); forced updates;
updating via the relayed terminal carrier (deliberately excluded,
`types.py:1017`).

---

## 9. Open questions, each with my recommendation

1. **The auto-trigger default — SETTLED: `auto`.** Confirmed by the operator
   (2026-10-05, recorded in §2): roll after the origin's update settles,
   `--no-roll` to escape, `off` for anyone who wants it; `ask` is deferred to
   `mesh-update-propagation.md` §5. Watch for a drill where an auto-roll on a
   large fleet surprises its operator more than it helps.
2. **The standing channel: published releases only?** *Recommend yes* — the
   bound in §2 depends on it. If members must also follow source builds, that
   is a second, separately-consented channel with a per-device ref allowlist
   (the member checks the ref against its own policy before fetching); it is a
   new decision, not a parameter.
3. **Re-engaging unwatched sessions (step 5).** *Recommend porting the fleet
   tool's behaviour in full* (re-engage the snapshot; report how many came
   back), because nothing else performs it and "an unwatched runtime is never
   looked at". The alternative — leave them dormant until next use — saves
   processes and loses the verification half; if fleet memory pressure argues
   against it later, make it a per-member policy, not a silent change.
4. **Per-member deferral budget.** *Recommend 15 min per member, `--wait` up to
   30 min*, then defer-and-retry (§4.2). Settled by drill data on how long
   real turns run on the family's members.
5. **Order source.** *Recommend the network's member order* from the record's
   snapshot; a per-network priority list is a later, additive field.
6. **Does the origin's driver belong in the relay or the CLI? — SETTLED: the
   relay alone.** The catch-up hooks need the process that is present at "next
   contact", and a foreground driver dies with its terminal; the CLI reads the
   record (`mesh-update-propagation.md` §4); the drill in §8.3 exercises both.

---

## 10. Risks to watch during rollout

- **The auto-roll surprises an operator.** Mitigated by the summary sentence
  and `--no-roll`; watch for it in the drill and in the first real release
  after this ships.
- **PyPI (or the member's channel) unavailable mid-roll.** Members refuse with
  the installer's own tail; the pass continues; the record's window keeps the
  retry alive. Do not add a retry storm: the cadence hook is the bound.
- **Long-busy members.** The deferral budget is the knob; a member that is
  *permanently* busy is a member whose operator should decide, and the peers
  surface will say exactly that in the member's own words.
- **The in-place (non-generation) install window** on members that predate the
  layout. The drain is the protection; the receipt discloses the method; the
  upgrade to a generation-capable build is itself the first update that lands.
- **Audit/record growth.** One record per rollout, bounded window, tombstoned
  like the approvals store; rows per device are handful-scale. Measure at the
  drill.
- **Vocabulary drift between `lop update` and `lop network update`.** One
  sentence each in the help text; one refusal register; the drill reads them
  side by side.

---

## Relationship to the sibling designs (for the reader who starts here)

This design adds no `R`-numbered requirements to the spine; it implements
against `mesh-network.md` R18 (auditable operations), the spine's §5 security
posture, and the onboarding lane's §3.4 upgrade path. When it lands, the spine's
detail-design table gains its row (a one-line follow-up edit, deliberately not
made here to keep this change single-file). The requirement it effectively
discharges is the one the operator stated in words at the top: *updates should
propagate, safely and in a rolling way, and wait for work to finish.*
