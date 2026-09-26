# Design: mesh incident response — disconnect, panic, epochs, audit, forensics

Status: **proposal for implementation.** Author: architect.
Base: `origin/main` @ `a7e6b9bd`, on branch `feat/mesh-network`.
Parent: `docs/design/mesh-network.md` (all `R<n>` / `A<n>` references are to it).

**Closes R17, R18 and decisions A6, A7.** Depends on
`mesh-transport-identity.md` for the device keypair, the network secret, the
epoch handshake, the peer link and the relay's local op vocabulary. It touches
`mesh-session-mobility.md` only to state what an incident is **not** allowed to
do to a session. All file:line references are against the base tree above.

| Requirement | Where this document closes it |
|---|---|
| R17 — a one-command stop; a way to signal every peer | §1.3, §1.4, §1.5, §3 |
| R18 — forensics with sane, cheap retention | §4, §5 |
| A6 — incidents are a state change, not a cleanup task | §1, §2 |
| A7 — append-only JSONL, bounded batched writes | §4.6, §4.7 |
| R5's mechanism (the requirement is owned by `mesh-transport-identity.md`) — including a defect in it, and the bounded lifetime that completes it | §2.3, §3 |
| R9 (quitting is never fatal) — the negative constraint | §1.6 |

**Interface alignment, checked against `mesh-transport-identity.md` as it
stands.** This document does not invent verbs, records or audit kinds the
transport already owns; it adopts them. Specifically:

- **Verbs.** Local: `net_disconnect`, `net_panic`, `net_trust`, `net_log`,
  `net_doctor`, `net_status` (transport §2.5). Peer: `net_leave`, `net_panic`,
  `net_epoch`, `net_reconcile`, `net_bye` (transport §6.4). This document owns
  their *semantics* and their *state machine*, not their names.
- **States.** `trust: "active" | "untrusted"` is **network-wide** and owned by
  the transport's network record (§4.2, §8.5); a member's own identity is a
  row with `lifecycle` and a tombstone on removal (§4.2). This document adds one
  local membership state (`disconnected`) and one observation (`stale`), and
  nothing else — a second trust concept beside `trust` would be a defect.
- **Audit kinds.** The transport's §7.6 names the event kinds for its scope
  (`member_admitted`, `epoch_rotated`, `panic_raised`, `panic_received`,
  `trust_changed`, …). Those names are **adopted verbatim** and the taxonomy here
  is their superset, because it is one log and it must have one naming
  convention. §4.3 marks which kinds came from the transport.
- **Audit record fields.** The transport's `AuditEvent` payload (`ts`,
  `network_id`, `epoch`, `actor`, `subject`, `event`, `outcome`, `session_id`,
  `detail`) is the **required subset**; §4.2 adds `schema`, `seq`, `ts_iso`,
  `network_name`, `actor_kind`, `cause`, `prev_hash`, `hash`.
- **Ids.** `d_<hex32>` device, `n_<hex24>` network, `i_<hex16>` instance, epoch
  integers, `rotations: {epoch: device_id}`, `sequence` (transport §3.1, §4.2).

---

## 0. The answer in one page

**An incident verb changes state and severs links. It never reaches into a peer,
never stops a session, and never deletes anything.**

- **`net_disconnect`** — this device leaves. It sends `net_leave` (self-signed)
  to every reachable peer, closes the links, marks the network `disconnected`
  locally, **and deletes the local secret** (transport §8.2). The survivors
  tombstone the member and their lowest-id active admin initiates the rotation.
- **`net_panic`** — this device distrusts *the network*. It mints a fresh secret,
  bumps the epoch, latches a durable marker so a relay restart cannot silently
  un-panic, and broadcasts `net_panic` to every reachable peer. Every receiver
  sets `trust: "untrusted"`, drops every link, and refuses everything for that
  network — including a connection that arrives afterwards. Recovery is
  `lop network trust <network> --active` plus a re-handshake, on each device.
- **`lop network member rm <network> <device>`** — the eviction. Tombstone plus
  rotation, and the excluded device does not learn the new secret (§2.3). This
  is the verb for "one device is the problem".

The mechanism that makes a **missed** broadcast safe is the epoch: a member that
never learned of a rotation is refused at its next handshake, and whether it is
then re-keyed automatically or must be re-admitted depends on one fact — whether
it is still a member at the current epoch (transport §8.3, §8.4). That single
fact is what lets a network partition self-heal while a removal sticks.

**One defect found in the transport design, carried here as §2.3.4 and §8 Q1:**
the `net_epoch` frame carries the new secret in the clear *and is delivered to
the device it is removing*, and transport §8.3's statement that "a removed device
does not learn the new secret" is enforced by the receiver's good behaviour
rather than by the sender. It is a two-line fix and it is the difference between
a rotation and a ritual.

The audit log is one append-only JSONL file per install, one record per
**semantic** event, hash-chained, rotated by size and age, at **≤ 1 `write(2)`
per second in steady state** no matter how many frames cross the link, with
`fsync` only on the events that must survive a power cut (§4.7). §4.8 names the
two commands that prove it rather than assert it.

---

## 1. The incident shapes as a state machine

### 1.1 Vocabulary — and which half the transport already owns

| Concept | Owner | Values |
|---|---|---|
| Network-wide trust | **transport** (network record `trust`, §4.2; §8.5) | `active` ⇄ `untrusted` |
| Member identity | **transport** (member rows + tombstones, §4.2) | `lifecycle: active \| provisioning \| draining \| expired`; `removed_at`/`removed_by` set on removal |
| Pairing progress | **transport** (§5.4) | `pending` → admitted |
| Epoch | **transport** (§4.3, §8.4) | integer; `rotations: {epoch: device_id}`; two secrets retained (`current`, `previous`) |
| Link state | **transport** (in memory, §6.5) | `up`, `idle`, or a refusal code (`not_authorised`, `not_a_member`, `unknown_op`, `protocol_mismatch`, …) |
| **This device's own membership** | **this document** | `active` \| `disconnected` — a local fact, not a peer-visible one |
| **Epoch freshness of a peer, as observed** | **this document** | `stale` — an observation in `network/peers/<device_id>.json`, never a durable membership state |

Two things follow, and both are deliberate:

- **`disconnected` is not a trust value.** It says "this device deliberately
  left"; `trust` says "this network is not trusted". Conflating them would make
  `disconnect` look like a panic to every surface that reads `trust`.
- **`stale` never enters the network record.** It is derived from a handshake or
  a refusal and would be clobbered by the next `net_epoch`. Keeping it in the
  per-peer observation file means the durable membership document has one writer
  in one situation, which is what makes a forensic read of it trustworthy.

### 1.2 The state machine

```
                                   ┌──────── net_invite + join + SAS ────────┐
                                   │                                          │
   (transport)  pending ──admit──► active ◄──────────────────────────────────┐ │
                                     │                                       │ │
                                     ├── net_disconnect ──► disconnected ────┼─┘
                                     │   (local secret deleted; survivors    │
                                     │    tombstone + rotate)                │
                                     │                                       │
                                     ├── net_panic ──────► panicked ─────────┤
                                     │   trust=untrusted, epoch++, latch     │
                                     │                                       │
                                     └── net_panic received ► untrusted ─────┘
                                         trust=untrusted, links dropped      net_trust --active
                                                                             + re-handshake
   any ── net_rm ──► forgotten (no record, no secret)
```

States a reader will ask about, and where they live:

- `panicked` and `untrusted` are **the same transport state** (`trust:
  "untrusted"`) with a different *local* history: `panicked` means we raised it,
  `untrusted` means we received it. The distinction is worth keeping only in the
  incident record (`raised_by`, §5.1) and in the copy, never as a second durable
  field.
- `disconnected → active` requires an **invite**: the transport deletes the local
  secret on disconnect (its §8.2), so there is nothing to reconnect *with*. This
  is a change from this document's first draft, which proposed a `reconnect` verb
  on the theory that a leaving device keeps its secret. The transport's choice is
  the better one — a device that has left should not keep the key it left with —
  and the cost is stated plainly: **returning after `disconnect` is a fresh
  pairing.** An operator who only wanted a temporary silence does not run a verb
  at all; peer links reconnect on their own (transport §6.5).
- `member rm` on ourselves (`lop network rm <network>` locally, or an admin
  removing us elsewhere) leaves a tombstone. A tombstoned id can never be
  re-admitted (transport §4.2); coming back is a new identity via `net_identity_rotate`.

### 1.3 `net_disconnect` — this device leaves

| Aspect | Effect |
|---|---|
| Broadcast | `net_leave` (self-signed) to every reachable peer, one attempt each, 2 s deadline (§2.1) |
| Local state | `disconnected`; **the local secret is deleted** (transport §8.2); the audit trail is kept |
| Local records | Member rows are kept for the audit trail and for `lop network show`; they are not authority for anything while disconnected |
| Relay | Keeps running; it may serve other networks. `lop network stop` is the separate verb that stops the relay, and the docs must say so — "stop" reading as "revoke" would be dangerous (transport §8.2) |
| In-flight sessions | Untouched (§1.6) |
| Placements | Untouched; remote sessions become unreachable for display only (§1.6) |
| Survivors | Tombstone the member; the **lowest-id active admin** among them initiates the rotation, because the leaving device still holds the old secret (transport §8.2) |
| If nobody is reachable | We forget the network locally; the peers see us go unreachable and any of them can run `member rm`, which needs no cooperation (R5) |
| Audit (us) | `disconnect_initiated` with `{epoch, reachable_peers, sessions_became_unreachable}` |
| Audit (peers) | `member_left` with `{subject_device}`; then, from the rotating admin, `epoch_rotated` |

The leaving device does not rotate anything itself. That is the transport's
rule and it is right: a device cannot usefully mint a secret it is leaving
behind, so the survivors do it.

### 1.4 Where "deactivate this device" went

R17 says "deactivate / disconnect this device". This document's first draft
proposed a third verb (`deactivate`) alongside `disconnect` and `panic`. **It is
dropped.** R17's three useful answers already exist and do not overlap:

| The operator means | The verb |
|---|---|
| "I want out of this network now" | `lop network disconnect` |
| "This device must never be trusted again, and I don't know who else is reachable" | `lop network member rm <network> <self-or-device>` (from any surviving device) — and `disconnect` on the device itself, which is the same act from the other side |
| "I don't trust anyone" | `lop network panic` |

A fourth verb would be a second way to do one thing, with an extra state to
reason about in the one code path where ambiguity is most expensive. The report
should record this as a deliberate reduction, not an omission.

### 1.5 `net_panic` — the network is not trusted

| Aspect | Effect |
|---|---|
| Precondition | Zero arguments resolve only when this device is in exactly one network; otherwise it refuses and lists them. TTY confirmation, `--yes` required when stdin is not a TTY — the `lop stop --all` shape (`cli.py:783-793`) |
| Local state, **first** | `trust = "untrusted"`, `untrusted_reason = "panic raised by <self> at <ts>"`, **and the latch written and fsynced** before any network I/O (§5.2). A slow or hostile peer must not be able to delay the local device going untrusted |
| Secret | A fresh random secret is minted and the epoch bumped (transport §4.3: "rotation mints a fresh random `secret`; it does not ratchet"); `previous` is kept per the transport's two-epoch rule |
| Broadcast | `net_panic` to every reachable peer, one attempt each, 2 s deadline (§2.1), `sequence` incremented |
| Links | All dropped for that network; the relay refuses every inbound frame for it from now on, **including a fresh connection** (transport §6.2 step 3 checks `trust`) |
| Non-admin member raising it | The transport deliberately gates `net_panic` as `list`, so **any** member may raise it; a non-admin sender's frame carries no `secret` and rotates nothing, and receivers still go untrusted (transport §8.2). This document **adopts** that call: a false alarm costs an operator one `net_trust`, a suppressed alarm costs the network, and the direction of the failure is the right one. The DoS it permits is stated as an accepted risk in transport §10.2 |
| Local sessions | **Untouched and unaffected** |
| Remote sessions | Unreachable; not deleted, not stopped, placements not rewritten |
| Audit (us) | `panic_raised` **before** the fan-out, then one `panic_delivered` / `panic_undelivered` per peer, then `panic_broadcast_result` |
| Audit (peers) | `panic_received` with `{from_device, epoch_before, epoch_after, reason, rotation}`. `epoch_before`/`epoch_after` are the RECEIVER's own epochs, and `rotation` is which rule the frame's rotation half met (`applied`, or the reason it was not adopted — a valid alarm from a threadbare frame still goes untrusted) |

**What panic deliberately is not.** It is not a remote kill switch, and the copy
must say so in as many words, because that is what a person will assume:

> `Panicked network 'home-net'. Links are closed and this device will refuse peer traffic until you re-admit it with 'lop network trust home-net --active'. Sessions running on other devices are NOT stopped — they keep running and will be unreachable to you.`

A verb that both severs trust *and* reaches into a peer to kill processes has to
be authenticated by exactly the channel the operator has just [redacted]
untrustworthy, so it cannot be both safe and useful. Stopping a remote session is
`lop stop --peer <device> …`, issued *before* the panic if that is what the
operator wants.

**What panic also is not: an eviction.** Because the `net_panic` frame carries
the new secret (transport §8.2), every member that receives it — including a
member the operator suspects — ends up holding the new key. Panic buys a
**global stop-the-world pause**, not exclusion. To actually exclude a device,
run `member rm` (or let its `disconnect` propagate), which is the path that
withholds the new secret (§2.3.4). State this in the guide, in the panic output,
and here: an operator who panics and expects the suspect device to be locked out
has been misled by an intuition, not by this design.

### 1.6 What an incident does to the things people ask about

**1.6.1 Links.** All dropped for the affected network. Other networks on the same
relay are untouched: an incident is **per network**, and a device in two networks
must be able to panic one without losing the other. The transport's frames
already carry `network_id`, so this is a filter, not a new concept.

**1.6.2 Local records.** `network/<network_id>/network.json` (the transport's
record, §4.2) holds `trust`; this document adds no field to it. Written with the
repository's staged-write idiom (`registry._staged_write`,
`local_operator/session/runtime/registry.py:135`) at `0600` inside the `0700`
`network/` directory, so a crash mid-write cannot leave a half-written record.

**1.6.3 In-flight sessions — stated as a prohibition.** No incident verb may
signal, abort, dispose, unpublish or otherwise disturb a session runtime, local
or remote. Concretely, `local_operator/network/incident.py` must not import
`local_operator/session/runtime/` or any stop/lease machinery, and §6.7 asserts
it with an AST check. This is R9 ("quitting is never fatal to a remote session")
extended to the incident path, with one addition worth saying out loud because
it is where designs like this go wrong: **a peer's panic must not stop our
sessions either.** The result on that device is that device's; the link going
away is a fact about the link.

**1.6.4 Placements.** Unchanged. A placement records *where a session runs and
why* (A8) and that did not change when a link dropped. Rewriting a placement on a
link loss would record a lie and would fight `mesh-session-mobility.md` the
moment the link returned. What changes is the **display** — unreachable, with a
reason and a remedy.

**1.6.5 Credentials.** No broker grant is revoked by a panic, because a held
bearer cannot be un-issued (`mesh-credentials.md` §3.7). The panic does close
the link, so no *new* grant can be requested from either side. Stated explicitly
because the opposite is the intuitive assumption and it is wrong.

---

## 2. Broadcast mechanics

### 2.1 Fan-out and the deadline

One concurrent task per reachable peer, each with

```python
INCIDENT_ACK_DEADLINE_S = 2.0
```

reusing the repository's existing number for the same job — the secret broker's
`NOTIFY_ACK_TIMEOUT_S = 2.0` (`local_operator/secrets/broker.py:73`), chosen
there for exactly this shape: long enough for a healthy peer, short enough that
a dead one cannot hold up a security-relevant state change. The transport
classes `net_panic` and `net_epoch` as **RELIABLE** ops (its §6.6), so the
deadline is the *ack* deadline, and the retry policy below is this document's.

- **The local state change is never conditional on the broadcast.** `panic`
  latches first (§1.5); `disconnect` closes first. A design that waited for
  acknowledgements before distrusting would let a compromised peer veto its own
  eviction.
- **One attempt per peer, no retry loop.** For `panic` this is a hard rule:
  retrying against a possibly-compromised peer is the traffic the operator just
  asked to stop. The transport's **offline queue** is the one exception and it is
  the good kind — `net_epoch` is queued for every active member that is offline
  (its §8.1), so a rotation reaches a sleeping member when it wakes. That is
  transport state, not an incident retry loop, and this document states the
  requirement: **`net_panic` is deliberately not queued**; a panicked network
  recovers by `net_trust` plus a re-handshake, never by replaying old broadcasts.
  §8 Q2 records the tradeoff.
- `lop network status --json` reports the last broadcast's tally, so the operator
  sees `{sent: 3, acked: 2, unacked: ["d_4b2a…"]}` immediately rather than
  discovering it during recovery.

### 2.2 Partial reachability

| Case | What happens | Why it is safe |
|---|---|---|
| Peer acked | It is `untrusted` (or rekeyed) already | — |
| Peer unreachable | It keeps the old epoch and believes we are still a member | Refused at its next handshake; it cannot produce a frame the survivors accept (§2.4) |
| Reachable but the ack timed out | Same, recorded as `unacked` — a distinct outcome from `failed` | Same |
| Relay dies mid-broadcast | The latch was written and fsynced **before** the fan-out, so a restart resumes `untrusted` | The durable marker is the authority, not the in-memory loop |
| Peer reconnects during the window | The transport's keepalive/reconnect (§6.5) plus the epoch check cover it | §2.4 |

The three outcomes are counted distinctly — `acked`, `unacked` (deadline) and
`failed` (error) — because they mean different things to an operator reading
forensics: `unacked` is "still out there on the old epoch", `failed` is "could
not even try".

### 2.3 The epoch contract — what the transport provides, and one defect

#### 2.3.1 What exists

The transport's epoch design (its §4.3, §8.1, §8.4) is:

- A **fresh random 32-byte secret**, never a ratchet, with `current` and
  `previous` retained and a third dropped on rotation.
- `epoch_key(secret, network_id, epoch) = HKDF-SHA256(ikm=secret,
  salt=sha256(network_id), info=b"lop-mesh-epoch-v1\x00" || str(epoch).encode(),
  length=32)` — the actual link key.
- `net_epoch` carries `{epoch, sequence, rotation_id, secret, members_digest,
  members, removed, reason}`, is broadcast to every active member, and **queued**
  for every active member that is offline.
- A receiver applies it only if it came from an active member, the epoch is
  strictly greater, `rotation_id` matches the sender, the member list is
  internally consistent, and `members_digest` verifies.
- A **reconcile phase**: a `hello`/`auth` whose MAC matches
  `epoch_key(previous)` is admitted only if the device is an active member at the
  current epoch, and then exactly two ops are dispatchable — `ping` and
  `net_reconcile`, which answers with the current epoch, secret, member list,
  digest and `rotations`. Rate-limited at `RECONCILE_MAX_PER_HOUR = 3` per device
  per network, audited, and a fourth request is refused
  `reconcile_rate_limited`.
- Concurrent rotations converge on `rotations: {epoch: device_id}`.

This is a complete and careful design, and it answers the "offline member" half
of R5 without a per-device visit. This document adds nothing to it except the
defect below.

#### 2.3.2 What this document requires on top of it

Nothing, structurally — three requirements, stated so the transport's
implementation keeps them:

1. **`net_epoch` must not be applied by a non-member.** The transport's receiver
   check includes `self_device_id` present and active in the incoming member
   list, which is the right check. §2.3.4 is about the *sender*.
2. **The epoch must be applied atomically with the secret.** `record + secret
   atomic write` (transport §8.1 step 4) — a torn write that installs the epoch
   without the key, or the key without the member list, produces a device that
   cannot authenticate and cannot tell why.
3. **The latch is fsynced before the broadcast.** §1.5, because the alternative
   is a restart that silently un-panics.

#### 2.3.3 Auto-rekey on handshake, and why it is safe

After a rotation, a still-member device that missed the broadcast presents
`previous`. The transport distinguishes the two cases with one fact — presence
in the member list at the current epoch:

- **still a member** ⇒ reconcile completes and the device re-handshakes at the
  new epoch. A partition self-heals on the next connection; no operator action
  for a device that was merely asleep.
- **not a member** ⇒ refused; the device writes `handshake_refused:
  not_a_member`, marks its own copy `stale: refused_by_peers`, and the remedy is
  an invite. A removal sticks.

The member list is the authorisation authority, so this adds no trust
assumption: the table that decides whether a frame may dispatch decides whether a
handshake hands out the rekey.

#### 2.3.4 **Finding: the rotation hands the new secret to the device it removes**

Transport §8.1 step 3 broadcasts `net_epoch` — which contains `"secret":
"b64url(32)"` — "to every active member", and the frame's own `removed` array
names the device being removed. Transport §8.3 then states:

> It keeps no secret: the rotation delivered in that same frame is applied only
> if the device is still a member, so a removed device does **not** learn the new
> secret.

That inference does not hold. The removed device was an *active member at the
moment of the broadcast* — that is why it received the frame — and it is the
peer we have just decided not to trust. Nothing stops it from reading `secret`
out of a frame it holds and recomputing `epoch_key`. The exclusion is enforced by
the receiver's cooperation, which is the one thing a removed device has no
reason to provide. The consequence is concrete: after `member rm`, a hostile
former member **can decrypt** subsequent traffic it can intercept, so R5's
eviction is authorisation-only, and `mesh-network.md:239-240`'s "the network
secret is what makes R5 possible — revocation rotates it" is not yet true.

**Recommendation (two lines, either one sufficient).**

1. **Withhold it at the sender.** A `net_epoch` frame addressed to a device named
   in `removed` must carry `"secret": null` — or, equivalently, the removed
   device is *not* a recipient of the rotation at all: it receives the tombstone
   as its final frame with no key material, and the rotation goes only to the
   survivors. This is the minimal change and it closes the hole exactly.
2. **Or seal per recipient.** Wrap the secret to each survivor's device public
   key and send one wrapped copy per member, so the exclusion is cryptographic
   by construction rather than by omission. More code, and it also covers the
   offline-queue case (a queued frame sits on disk holding a plaintext network
   secret, which is itself worth avoiding).

**I recommend (1) now and (2) if the queue makes (1) awkward**, because (1) is
the change that makes §8.3's sentence true as written and costs nothing.

**Resolved in convergence round 1, in the sender.** This section is the canonical
statement of the defect and it stays as written; what changed is the transport
document, which now withholds `secret` from any `net_epoch` frame whose recipient
its own `removed` array names, and refuses to queue such a frame holding key
material (`mesh-transport-identity.md` §8.1 step 3, invariants
`epoch_secret_withheld_from_removed` and
`epoch_outbox_holds_no_secret_for_a_removed_member`, each with a named test in
its §13.1). Option (1) below is the one that landed, this document's
recommendation (§8 Q1); option (2), per-recipient sealing, remains the stronger
answer and is not needed while the queue is clean. Read the paragraphs below as
the *pre-fix* consequence analysis: they are kept because the argument for the
fix is only intelligible next to what the fix is for.

Until it lands, the honest consequences must be documented and this document does
document them: `member rm` **evicts from acting, not from reading**; the
credentials of a removed device are still safe, because the broker's holder-set
check is independent of the transport secret (`mesh-credentials.md` §3.6c step
1); and an operator who needs confidentiality against a removed device must
treat the network as burned and re-establish it (which is what `panic` plus
re-admission produces, if the panic frame's own secret is also withheld — see
§8 Q1's second half).

### 2.4 How a missed broadcast is safe — the property, precisely

> **If a peer did not learn of a rotation, it cannot produce a frame any current
> member will accept: its frames fail the epoch check, and its handshake is
> either completed (if it is still a member) or refused (if it is not).**

Three consequences worth naming, because each is a failure someone will report:

- **A stale peer can always try.** The refusal is at authorisation, not at the
  socket, so an attempt shows up as `handshake_refused` — the evidence an
  operator needs after an incident to see who was still around with old material.
- **The stale peer learns.** Its own log gets `handshake_refused` with its cause,
  and its `peers/<device_id>.json` gets `stale`, so its CLI renders the remedy
  rather than a generic connection error.
- **Nothing is deleted by a refusal.** Sessions on either side are untouched;
  only reachability changed.

### 2.5 Who may broadcast what

Adopted from the transport's capability model (§7.1, §7.2, §7.3) rather than
re-declared here:

| Frame | Gate | Source |
|---|---|---|
| `net_leave` (self) | `list` — anyone may always leave | transport §6.4 |
| `net_epoch` | `admin` | transport §6.4 |
| `net_panic` | `list`, deliberately: any member may raise it, and a non-admin's frame carries no secret and rotates nothing | transport §7.1, §8.2 |
| `net_trust` | `trust`, which resolves to `admin` | transport §7.1, §6.4 |
| `net_reconcile` | `list`, reconcile phase only, rate-limited | transport §6.4, §8.4 |

**Threat note.** A compromised `admin` peer can panic the network, and a
compromised `list` peer can raise a false panic. Both are accepted by the
transport's design with the reasoning that a false alarm costs an operator
action while a suppression costs the network; the mitigation is that the operator
sees **who** raised it (`panic_received`'s `from_device`, and the incident
record's `raised_by`) and can re-admit with one command. What this document adds
is that the raised-by fact must survive to the surface — `lop network log` and
the incident record both carry it, and the TUI copy names the device.

---

## 3. Recovery

### 3.1 Re-admitting after a panic

`lop network trust <network> --active` on **each** device (`net_trust` locally,
`net_trust` over the link when the operator drives it from one device). This:

1. refuses without a TTY confirmation (`--yes` when stdin is not a TTY);
2. removes `panic.latch` on this device, with an `fsync` of the file and its
   directory;
3. sets `trust = "active"` and clears `untrusted_reason`;
4. writes `trust_changed` with `{from: "untrusted", to: "active", by:
   <device>}`;
5. does **not** rotate, does not re-key, and does not re-admit anyone else. It
   re-opens this device; the next handshake establishes the link.

Two honest consequences, both in the guide:

- **Every device needs it.** A single `net_trust` on one device does not
  un-untrust the network for its peers; the operator clears it per device, or
  drives it remotely from one device if the peers are reachable. The reason is
  that `trust` is a local judgement — a "trust this network again" push from a
  device the operator just stopped trusting is exactly the frame they must not
  accept.
- **The network is coherent again immediately**, because the panic's `net_epoch`
  already gave every receiver the new epoch, the new secret and the member list.
  That is the payoff of the transport's choice to distribute the secret in the
  panic frame, and it is why this document does not re-pair devices after a
  panic.

### 3.2 Re-admitting a device that was removed, or that left

`net_invite` + `join` + SAS, exactly the pairing path (transport §5). There is no
un-remove: a tombstoned `device_id` cannot be re-admitted (transport §4.2), so
the returning device either rotates its identity key
(`net_identity_rotate`, transport §3.3) or arrives as a genuinely new device.

**Open question carried to the transport, with my recommendation (§8 Q3):**
after a compromise, an operator wants to say *which* device comes back, so
`net_invite` should gain an optional **device binding** —
`net_invite {network, role, ttl_s, device_id?}` — refused when the token is
redeemed by a different device id (`pairing_refused` with
`cause: "wrong_device"`). Today the transport's invite is device-agnostic and its
`invites` row records `redeemed_by` only after the fact, so an open invite on a
freshly-rekeyed network is a bearer token for the network's key material. The
change is small (one optional field plus one comparison) and it belongs in the
transport document; it is recorded here because the incident path is where the
need shows up.

### 3.3 A device that was offline through the rotation

The likely false alarm, and it needs no command at all: reopen and reconcile
(§2.3.3). The two sub-cases print different lines and both are true:

- still a member: `Network 'home-net' was re-keyed while this device was away; the current epoch is 8.`
- removed: `This device is no longer a member of 'home-net'. To return, ask for an invite: on a member device run 'lop network invite --role <role>'.` — the exact command the operator needs, produced by the device that was removed, which is the cheapest possible remedy path.

### 3.4 The audit trail an operator reads afterwards

Four artifacts, in the order a person actually looks at them:

1. **`lop network status --json`** — state, epoch, links, latch presence. What is
   true now.
2. **`lop network doctor [--peer <device>]`** — reachability, handshake, epoch
   skew, clock skew. Why it is true.
3. **The incident record** (`<config>/network/incidents/<incident_id>.json`,
   §5.1) — the one page: who did what, which peers got it, which sessions became
   unreachable. What just happened.
4. **`lop network log --since 2h --json`** then **`--verify`** — in what order,
   and whether anyone edited it.

A cross-device read is deliberately not built in this pass: there is no central
log, and a peer's log is its own. `lop network log --export <file>` per device,
concatenated by the operator, is the supported cross-device story and the guide
says so, so nobody waits for a feature that is not coming.

---

## 4. The audit log

### 4.1 Location, files, modes

The transport fixes the path and the shape (its §4.1 and §7.6); this document
fixes rotation, retention and the writer.

| Path | Mode | Contents |
|---|---|---|
| `<config>/network/audit.jsonl` | `0600` (dir `0700`) | The live log, one JSON object per line |
| `<config>/network/audit.jsonl.1.gz` … `.5.gz` | `0600` | Rotated generations, gzipped |
| `<config>/network/audit.state.json` | `0600` | `{schema, genesis_hash, last_hash, last_seq, records_total, bytes_total, last_rotated_at}` |
| `paths.log_dir()/network.log` | default | The **debug** log (`paths.py:121`, `paths.py:165`). Per-frame by nature. Not forensics; may be truncated freely; never confused with the above |

One audit log per install (A7); every record carries `network_id`, so a
multi-network device has one ordered history. Modes follow the repository's
existing convention for private stores (`secrets/keys.py:37-38`: `DIR_MODE =
0o700`, `FILE_MODE = 0o600`).

**Name collision, called out deliberately:** `local_operator/incidents.py`
already exists and means *session turn* incidents — cut-off reasons, credential
messages, MCP recovery (`incidents.py:1-20`). The mesh module is therefore
`local_operator/network/incident.py` (singular, inside the `network` package) and
its docstring must open by saying which `incidents` it is not. Two modules with
one name would be a reader trap, and the mesh one is the one that should yield.

### 4.2 Record schema

One JSON object per line, newline-terminated, UTF-8, no embedded newlines —
`json.dumps(..., separators=(",", ":"), ensure_ascii=False)` with a
control-character guard on every string field, for the reason
`credentials._reject_control_chars` gives on the flat store
(`local_operator/credentials.py:34-43`): a delimiter-scanning reader must not
have to define what happens when a payload contains the delimiter.

```json
{"schema":"lop.mesh.audit.v1","seq":1841,"ts":1789012345.678,"ts_iso":"2026-09-19T14:32:25.678Z","event":"credential_grant","network_id":"n_5f3c1a2b4d5e6f708192a3b4","network_name":"home-net","epoch":7,"actor":"d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b","actor_name":"damian-mbp","actor_kind":"human","subject":"d_4b2a91c4e0b87f3a","session_id":"2026-09-19T18-04-11_ab12","outcome":"ok","cause":"","detail":{"key":"openai","credential_kind":"oauth","refreshed":true,"grant_id":"g_2f9c1188a0","token_ttl_s":3600,"latency_ms":214},"prev_hash":"9f2c1e...","hash":"41ab98..."}
```

| Field | Type | Required | Rule |
|---|---|---|---|
| `ts` | float | yes | Epoch seconds, `time.time()`. A transport-named field. |
| `network_id` | string | yes* | *Absent only for install-scoped events (`audit_rotated`, `audit_verify_failed`, `audit_pruned`). |
| `epoch` | int | no | Omitted where the event has no epoch meaning. A transport-named field. |
| `actor` | string | yes | The acting device id (`"self"` for a local human action on this device). A transport-named field, as a **string**, not an object. |
| `subject` | string | no | The subject device id, where the event has one. A transport-named field. |
| `event` | string | yes | From the closed taxonomy in §4.3. |
| `outcome` | string | yes | `ok` \| `refused` \| `failed` \| `partial`. |
| `session_id` | string | no | Present where the event is about a session or was triggered by one. |
| `detail` | object | no | Keys are a **closed whitelist per event type**, enforced by the writer; unknown keys are dropped by a `TypeError`-free filter (never raised, because a lost audit record is worse than a dropped key). The whole map is capped at **2048 bytes**; a longer map is truncated to fit and gains `"truncated": true`. |
| `schema` | string | yes | Always `lop.mesh.audit.v1`. A reader refuses an unknown major. |
| `seq` | int | yes | Monotonic per install from 1, never reused. A gap is a detected deletion. |
| `ts_iso` | string | yes | UTC ISO-8601, millisecond precision, derived from `ts`; for humans and `jq`, never parsed back. |
| `network_name` | string | no | Convenience; never used in a decision. |
| `actor_name` | string | no | The device's human name; convenience only. |
| `actor_kind` | string | yes | `human` \| `agent` \| `relay` \| `unknown`. **A hint, not an attestation** — from `LOP_ACTOR` when a harness sets it, else `unknown`. Never presented as proof of who typed the command. |
| `cause` | string | yes (`""` when `ok`) | Closed machine enum: `owner_offline`, `not_a_holder`, `revoked`, `epoch_stale`, `not_a_member`, `not_authorised`, `sas_mismatch`, `wrong_device`, `tick_expired`, `capability_denied`, `auth_failed`, `replay`, `untrusted`, `reconcile_rate_limited`, `protocol_mismatch`, `duplicate_identity`, `policy`, `handshake_cap`, `timeout`, `internal`, `declined`, `unanswered`, `viewer_left`, `peer_closed`, `owner_gone`, `peer_unreachable`. Prose lives in the renderer. A value outside this enum is rendered as `internal`, which is why an out-of-enum cause reports a deliberate close as an internal fault — the four `viewer_left`/`peer_closed`/`owner_gone`/`peer_unreachable` members exist so a forwarded stream's close does not (`local_operator/network/relay.py`'s `STREAM_CLOSE_MACHINE_CAUSES` is the only place that translation happens, and it keeps the finer word — `viewer-gone`, `owner-socket-gone`, … — in `detail.cause`). |
| `prev_hash` | string | yes | Lowercase hex sha256 of the previous record's `hash`; the genesis record uses the chain anchor. |
| `hash` | string | yes | Lowercase hex sha256 over the canonical serialisation (§4.4). |

**Never present in any field:** a token, a token prefix, a token hash, a refresh
token, a secret value, the network secret, any wrapped key material, a link key,
a `control_key`, transcript content, prompt text, or any absolute path
containing a username.

### 4.3 Event taxonomy — the transport's names, adopted, plus this document's

One log, one naming convention: the transport's flat snake_case kinds are kept
**verbatim** (marked ▣) and this document's additions follow the same shape. The
taxonomy is a `frozenset` of strings plus a `dict[event, frozenset[detail_key]]`
in `local_operator/network/audit.py`, with a unit test asserting every emitted
event is in it — the drift guard `tests/unit/tui/test_noop_consumers.py` already
uses for its own producer/consumer seam.

**Transport scope (▣ = named by `mesh-transport-identity.md` §7.6).** These are
listed so the taxonomy is complete in one place; the transport owns their
emission points.

| Event | Emitted when | `detail` keys |
|---|---|---|
| ▣ `pairing_refused` | a join/SAS step is refused | `{cause, subject}` |
| ▣ `invite_minted` | `net_invite` | `{role, expires_at, bound_device?}` |
| ▣ `member_admitted` | a member row is written | `{role, member_kind, epoch}` |
| ▣ `member_removed` | `member rm` | `{initiated_by, rekeyed, epoch_after}` |
| ▣ `member_left` | a peer's `net_leave` or a clean departure is seen | `{}` |
| ▣ `device_rotated` | `net_identity_rotate` | `{old_device, new_device}` |
| ▣ `epoch_rotated` | a rotation is created or adopted | `{epoch_before, epoch_after, rotation_id, removed}` |
| ▣ `epoch_conflict` | two rotations at the same epoch | `{epoch, rotation_id, winner}` |
| ▣ `handshake_refused` | a handshake fails any step | `{cause, their_epoch?, their_device?}` |
| ▣ `authorisation_refused` | a dispatch is denied by capability | `{op, capability}` |
| ▣ `link_opened` | mutual auth + authorisation succeeded | `{role, epoch, phase}` |
| ▣ `link_closed` | any close | `{cause}` — `peer-closed \| we-closed \| timeout \| error \| epoch_stale` |
| ▣ `link_idle` | no report within the keepalive budget | `{last_seen_at, missed_beats}` |
| ▣ `duplicate_identity` | two live processes claim one device id | `{instance_id}` |
| ▣ `self_link` | a device dialled itself | `{}` |
| ▣ `panic_raised` | **before** the fan-out (§1.5) | `{epoch_before, epoch_after, reachable_peers, incident_id}` |
| ▣ `panic_received` | a `net_panic` arrived | `{from_device, epoch_before, epoch_after, reason}` |
| ▣ `trust_changed` | `net_trust`, i.e. re-admission | `{from, to, reason}` |

**This document's additions.**

| Event | Emitted when | `detail` keys |
|---|---|---|
| `disconnect_initiated` | `net_disconnect` | `{epoch, reachable_peers, sessions_became_unreachable}` |
| `panic_delivered` | one peer acked a panic | `{}` (`subject` names the peer) |
| `panic_undelivered` | one peer's deadline/attempt failed | `{outcome}` — `unacked \| failed` |
| `panic_broadcast_result` | the fan-out finished | `{sent, acked, unacked, failed, duration_ms}` |
| `member_role_changed` | a role is edited | `{role_before, role_after}` |
| `credential_placement_[redacted]` | an owner declares ownership or changes holders | `{key, kind, scope?, replicate}` |
| `credential_grant` | a grant was served | `{key, credential_kind, refreshed, grant_id, token_ttl_s, latency_ms, act, sub}` |
| `credential_grant_refused` | a grant was refused | `{key, cause}` |
| `credential_refresh` | the **owner** POSTed to the IdP | `{key, credential_kind, duration_ms, ok, act}` |
| `credential_refresh_failed` | that POST failed | `{key, credential_kind, cause}` |
| `credential_report` | a borrower reported a borrowed bearer failed | `{key, error_class, http_status?}` |
| `credential_repair_requested` | a borrower asked the owner's human for an interactive login | `{key}` |
| `credential_replicate_opt_in` / `_opt_out` | a static key was opted into/out of replication | `{key}` |
| `session_placement_changed` | a placement is set or changed | `{placement_before, placement_after, policy, reason}` |
| `session_handoff_started` | a move begins | `{session_id, from, to, mode}` |
| `session_handoff_completed` | a move finishes | `{from, to, bytes_copied, duration_ms}` |
| `session_handoff_refused` | a move is refused | `{cause}` — `fenced \| in_flight_turn \| unreachable \| occupied` |
| `session_remote_op_refused` | a remote op was denied by capability | `{op, capability}` |
| `session_stream_opened` | a viewer's forwarded pipe is opened, on BOTH relays (the opener and the owner) | `{stream, peer, role}` — `role` is `viewer` or `owner`, so a leak reads as an imbalance in the two counts |
| `session_stream_closed` | that pipe ends, on BOTH relays | `{stream, peer, role, cause}` — the machine `cause` is one of the four added above; `cause` here is the finer word the call site passed (`viewer-gone`, `viewer-requested`, `peer-requested`, `peer-closed`, `owner-gone`, `owner-socket-gone`, `no-peer-link`, `peer-stopped-answering`) |
| `session_sync_completed` | the R22 cadence / pre-spin-down sync ran | `{to_device, bytes, reason}` — `cadence \| pre_spindown \| move` |
| `meter_interval` / `meter_close` | **reserved**; owned by `mesh-compute-pool.md` §5.2, which fixes their literal fields | see that document — not duplicated here, so the two cannot drift |
| `pool_grant_minted`, `pool_request_queued`, `member_draining`, `drain_barrier_timeout`, `pool_member_expired` | **reserved**; owned by `mesh-compute-pool.md` §3.4, which fixes their literal fields | see that document — not duplicated here, so the two cannot drift |
| `epoch_rotated {reason: "max_age"}` | a rotation forced by the bounded secret lifetime, not by a removal | `mesh-transport-identity.md` §8.4 owns the mechanism; the `reason` value is the only addition to this document's field list |
| `audit_rotated` | a generation was closed | `{generation, bytes, records}` |
| `audit_verify_failed` | `--verify` found a break | `{at_seq, expected_hash, found_hash}` |
| `audit_pruned` | a generation was deleted by the age or count cap | `{generation, age_days}` |

**`act` and `sub` are the delegation markers, and they exist so a forensic
reader can tell brokered use from direct use** (OAuth 2.0 Token Exchange,
RFC 8693 — its §1.1 is titled *Delegation vs. Impersonation Semantics*, and our
broker performs a *delegation*; `mesh-credentials.md` §1.4 adopts the
vocabulary). `act` is the device whose relay performed the exchange — the
broker, the party that actually held the credential and talked to the IdP — and
`sub` is the device the grant was issued to. Without them a member-removal
review cannot answer "was this credential spent by the device that owns it, or
by a peer that asked it to", and both are device ids, so neither carries
credential material.

`session_*` events are listed here because they ride this log; their emission
points belong to `mesh-session-mobility.md`. The metering events **and the pool
lifecycle events** are **reserved now and produced later**: they are in the
taxonomy so the billing consumer and the member-lifecycle reader need no second
log, and their field lists are deliberately **not** restated here —
`mesh-compute-pool.md` §5.2 owns the metering schema and its §3.4 owns the
lifecycle one, because one schema with one owner is the point. Their names are
adopted from those documents verbatim rather than invented, so the taxonomy and
the producers cannot drift apart. (`mesh-compute-pool.md` §3.4's table is the
delta this document adopts; it names `member_admitted` too, which is already
above under the transport's scope and is deliberately not redefined.)

### 4.4 The hash chain

Each record hashes its predecessor, reusing the primitive and the reasoning
already committed for the secret store (`local_operator/secrets/audit.py:1-20`:
*"tamper-evident, not tamper-proof"*; `GENESIS` at `:30`; `canonical_row` at
`:48`, which length-prefixes each field so a `session_id` containing the
separator cannot make two rows serialise identically):

```
GENESIS = sha256(b"local-operator/mesh/audit/v1").digest()
hash_n  = sha256(GENESIS_or_hash_{n-1} || canonical_row(record_n))
```

- **The chain continues across rotation.** `audit.state.json` carries
  `last_hash` and `last_seq`, so generation N+1 starts from generation N's tail
  and verification resumes without reading every old generation.
- `--verify` walks the live file plus every retained generation oldest-first and
  reports the first break as `(seq, expected, found)`.
- **A deleted generation is detected.** `seq` is monotonic and
  `records_total` is a running count, so a gap is visible even though the missing
  bytes are gone. That is the honest limit — detection, not recovery — and it is
  why `--export` exists (§4.5).
- Cost: one sha256 over ~450 bytes per record, ~1 µs — negligible beside the
  `write(2)` it rides on.

**Recommendation: keep the chain** (§8 Q4). An incident log an attacker can
silently edit is not forensics, and the primitive is ~20 lines because
`secrets/audit.py` exists to copy from. The transport's §7.6 does not include it;
this document adds it and the transport's field list remains the required subset.

### 4.5 Rotation and retention

| Setting | Default | Where |
|---|---|---|
| `audit_max_bytes` | `8388608` (8 MiB **per generation, after compression**) | `config.yml: network.audit.max_bytes` |
| `audit_generations` | `5` | `network.audit.generations` |
| `audit_max_age_days` | `90` | `network.audit.max_age_days` |
| `audit_buffer_bytes` | `16384` | constant, with a comment (not configurable) |
| `audit_tick_s` | `1.0` | constant |
| export target | `--export <file>` | never pruned |

**Justification, with the arithmetic.**

- *Size.* 8 MiB of *compressed* JSONL at ~450 B/record and roughly 3:1 gzip is
  on the order of 55,000 records per generation. The steady-state ceiling is
  `5 × 8 = 40 MiB` compressed plus the live file's own 8 MiB, which meets A7's
  "megabytes, not gigabytes" **by construction**, not by tuning. The number that
  matters is real records/day, and §4.8's probe is its authority.
- *Generations = 5.* Enough to reach back past a week on a busy install, few
  enough that the ceiling is a constant a person can hold in their head.
- *Age = 90 days.* An incident is investigated within days; a lost or stolen
  device needs "what did this device do for the last month" with margin; and at
  90 days the *age* cap is what binds on a quiet install while the *size* cap
  binds on a busy one, so the two caps cover the two real populations. That is
  why both exist rather than one.
- *Export before prune.* `lop network log --export <file> [--since <t>]` writes a
  copy that rotation never touches, so retention is a **default, not a
  guarantee**, and the guide says so.
- *Prune order.* Oldest generation first, by age then by count, each deletion
  logged as `audit_pruned` in the surviving log — so the log records its own
  truncation, which is the one thing an attacker with disk access would most
  like to hide.

### 4.6 What is deliberately NOT logged

Each is a decision with a reason:

| Not logged | Why |
|---|---|
| Any per-frame event (a request, a projection push, a `ping`, a heartbeat) | It would make disk cost proportional to *traffic*, which is the A7 prohibition. A busy session moves thousands of frames a minute and a healthy link produces zero interesting facts about each one. Frame-rate evidence belongs in `network.log` and is discarded freely. |
| Token deltas, token prefixes, token hashes, or another device's `credential_id` | §4.2's never-list. The fact worth recording is that a grant *was served* (`credential_grant` with `grant_id`, `key`, `session_id`), never the bytes. |
| Provider usage and price detail | It already lives in `analytics.db` and in the session's `session_spend.v1` row (`session/spend.py:47-56`). Copying it here would create a second spend authority, which the spend design explicitly refuses ("One arithmetic site"). |
| Transcript content, prompt text, tool output | The transcript is the record. The audit log names `session_id` and never a word of the conversation. |
| `lop secret` retrieval detail | `secrets/audit.py`'s own chain already records retrievals with columns deliberately limited to "the record's id, the event, and the peer identity" so "a leaked audit log tells an attacker what happened, not what the secrets are". Duplicating that here widens the exposure without adding a fact. |
| The network secret, link keys, wrapped key material, `control_key` | §4.2. Key material never enters a log, in any encoding. Note that this is also an argument for §2.3.4's fix: the current `net_epoch` frame puts a plaintext network secret into an **offline queue on disk**. |
| Every heartbeat's liveness | `link_idle` is a *transition*. A per-beat record at a ~5 s interval is ~17,000 records/day for zero information. |
| Absolute paths containing a username | The log is operator-readable, may be exported, and may be shared during an incident review. The config dir is recorded as a role, never an absolute path. |

### 4.7 The I/O argument, and the writer that delivers it

**One writer per install.** The relay process owns it. No other process opens
`audit.jsonl` for append — that is what makes batching safe with no locking, and
it is a stated invariant with a test (§6.6).

```python
class AuditWriter:
    """Append-only, batched, fsync-bounded. The only writer of audit.jsonl."""

    BUFFER_BYTES = 16_384
    TICK_S = 1.0
    #: Events whose loss to a power cut would be unrecoverable.
    DURABLE_EVENTS = frozenset({
        "panic_raised", "panic_received", "disconnect_initiated", "trust_changed",
        "member_admitted", "member_removed", "member_left", "epoch_rotated",
        "handshake_refused", "audit_verify_failed",
    })

    def append(self, event: str, *, durable: bool | None = None, **fields) -> None: ...
    def flush(self) -> None: ...   # write the buffer; no fsync
    def sync(self) -> None: ...    # flush + os.fsync(fd); dir fsync on rotation
    def _tick(self) -> None: ...   # 1 Hz: flush if dirty; size check; age prune
```

- `os.open(path, O_APPEND | O_CREAT | O_WRONLY, 0o600)` once at start, held open.
  `O_APPEND` because a single writer with append semantics cannot corrupt an
  existing line on a crash mid-write: the file keeps every complete line, and a
  partial final line is detected and dropped by the reader (a `json.loads`
  failure on the **last** line is tolerated; on any other line it is a chain
  break).
- A 16 KiB in-memory buffer, flushed when it fills, **or** on the 1 Hz tick,
  **or** immediately for a `DURABLE_EVENTS` record.
- **Steady-state cost is `≤ 1 write(2)/second` regardless of frame rate.** This
  is the load-bearing sentence of A7 and it follows structurally: the tick flushes
  at most once per second, and the only other triggers are the buffer filling and
  a durable event. A link moving 10,000 frames a second adds **zero** syscalls
  here, because nothing in a frame path calls this writer — §4.6's first row,
  asserted in §6.6. The instrument is `write_calls`, and it counts the FLUSH CALLS
  that actually opened the file and wrote (one append-open plus one `.write()`
  carrying the batch) rather than `write(2)` syscalls, which a batch past the text
  layer's 8 KiB buffer can split; the rate bound is a bound on the calls, and it is
  the number §4.8 measures. *(Agent review round 3, NIT 1 — the wording here used to
  claim a syscall count no probe on this host without root can take.)*
- **A stream's lifecycle rows publish at the state change, not on the tick.**
  `session_stream_opened` / `session_stream_closed` (the writer's
  `STREAM_LIFECYCLE_EVENTS`) are exactly the rows an operator reads the tail for
  after a session looks stuck, and leaving them to the heartbeat left the owner's own
  close row invisible for a measured **14.74 / 14.72 / 14.73 s** — one full 15 s
  interval, because the close lands just after a flush and an otherwise idle relay
  has nothing else to drain. They now flush on the transition (measured 0.0006 /
  0.0015 / 0.0004 s) and still cost no `fsync`.
- **The lag itself is readable, and here is where.** The relay's `status` answer
  carries `audit_recorded_through` and `audit_published_through` beside the
  pre-existing `audit_degraded` / `audit_degraded_reason` / `audit_path`, and the
  three surfaces a person actually looks at print them in the reader's own words
  (`relay.audit_status_words` — one renderer, so the CLI, the TUI's `/network` panel
  and the agent's digest cannot drift): `13 recorded, published through 13` when the
  two agree, `13 recorded, published through 12 (1 not yet written)` while a row sits
  behind the tick, `DEGRADED — [Errno 28] No space left on device` after a failed
  write (the state word first, then the writer's own reason — the same string it
  already printed to stderr), and **a sentence rather than a silence** when the relay is
  running but not answering — the state this whole distinction exists for, since a
  reader who has just found a row missing from `audit.jsonl` is by construction in it
  (design round 3, D40).
- **What those two numbers do NOT say.** `published_through` is a **high-water
  mark**, not presence: everything at or below it went through this writer's own
  flush, but a row that rotation moved into a `.gz` generation — or that retention
  has since pruned (§4.5) — keeps its number below the mark, so `file`
  (`AuditLog.publication_of`) must not be read as "a record with this number is on
  disk". And with a second process appending to the same `audit.jsonl`, the two
  counters name a *different* row: the writer seeds its sequence from the file's tail
  once, at open, and would render numbers of its own from the same seed. That is
  pre-existing, disclosed in `publication_of`, and left alone here — §4.1's one writer
  per install is the contract that keeps them sound, and it is why the relay owns the
  file.
- **`fsync` is deliberately not per record.** It is called for `DURABLE_EVENTS`
  and on rotation. The reasoning, stated so it is not "fixed" later: the log
  exists to be *readable after* an incident, and losing the last <1 s of ordinary
  events costs an operator nothing they cannot reconstruct; losing the
  `panic_raised` record of an act that just severed the network would be exactly
  the unrecoverable evidence the repository already learned to keep when it added
  `runtime-stop.json` — whose docstring (`registry.py:194-204`) records that the
  one fact nobody could recover after the 2026-09-13 kill wave was who took a
  runtime down and whether it was asked for.
- Cost model, explicitly an estimate and not the claim:
  `bytes/day ≈ events/day × ~450 B`. At 5,000 events/day that is ~2.25 MB/day raw,
  ~0.75 MB/day compressed, so an 8 MiB generation closes every ~10 days and the
  age cap never binds; at 500 events/day the size cap never binds and the age cap
  does. §4.8 measures it.

### 4.8 The measurement a QA agent runs

R18 requires the I/O argument to be **measured**. Three commands, in increasing
cost:

**1. The discrimination test — the one that actually proves A7.** Many frames,
few semantic events, and the log barely moves:

```sh
cd ~/local-operator
env -i HOME="$(mktemp -d)" LOCAL_OPERATOR_CONFIG_DIR="$(mktemp -d)/.local-operator" \
  PATH="$PATH" TERM=xterm-256color .venv/bin/python -m pytest \
  tests/unit/network/test_audit_io.py -q
```

`test_audit_io.py::test_a_ten_thousand_frame_session_writes_fewer_than_fifty_records`
drives 10,000 frames over a live link with 12 deliberately triggered semantic
events and asserts `records_after - records_before <= 30`. That ratio (≈833
frames per record) is A7's structural claim and it fails loudly if anyone ever
adds an `audit.append` inside a frame path — measured with the fix neutralised
for one run, the same cell reported **10,013 records for 10,000 frames**.

The same file holds the BOUNDED-IDLE cell that Q-R1-3's class needs
(`test_an_idle_network_writes_no_rows_for_a_member_that_cannot_hold_the_op`: an
injected hour of the definitions cadence writes **zero** rows on the peer, against
the 20-in-300-seconds it wrote when a `drive` member kept asking for an op it can
never hold) and the cell that pins the probe's own instrument
(`test_the_writer_counts_the_calls_the_probe_reports`), because a bound read off a
counter that never increments passes forever.

**2. The probe, which is the authority on the numbers.**
`scripts/mesh_audit_probe.py`, following the repository's precedent of naming a
script as the authority rather than quoting a constant
(`scripts/spend_ledger_probe.py` is the model):

```sh
.venv/bin/python scripts/mesh_audit_probe.py --frames 10000 --duration 60 --json
```

It stands up two relays on loopback in an isolated root, completes a REAL member
handshake between them, drives the frames down that link, triggers a deliberate few
semantic events, and prints:

```json
{"frames": 10000, "records": 13, "frames_per_record": 769.2, "bytes_per_record": 416.5,
 "bytes_written": 5414, "write_calls": 12, "write_calls_per_minute": 12.0,
 "fsyncs": 12, "durable_events": 12, "rotations": 0,
 "events_by_type": {"authorisation_refused": 12, "link_opened": 1}}
```

**What the probe deliberately does NOT drive: a session turn and a credential
grant.** Both need a provider login, which a probe on a laptop or in a CI container
does not have, and a fabricated one would put an invented number behind a bound.
That half of the matrix is measured by the QA runs that have a provider; this
probe's run is the frame-rate half (A7) and the writer's syscall behaviour.

Bounds asserted **by the probe itself**, so a regression fails CI rather than
needing a human to read a number:

- `frames_per_record >= 100` — the structural property;
- `bytes_per_record <= 700` — the record has not grown an unbounded field;
- `write_calls_per_minute <= 90` — the tick plus buffer fills plus durable
  events; anything above ~90 means something is flushing per event. A run shorter
than a minute is judged instead by the same property in its structural form
(`write_calls <= durable_events + one-per-second of run time`), because the rate
over five seconds is dominated by the durable events a correct writer DOES write
through — and the report names which form was applied;
- `fsyncs <= durable_events + rotations` — nothing syncs per record.

**3. The live rate check, on the real topology.** With a two-peer session
streaming a long turn:

```sh
A="$ISO/A/.local-operator/network/audit.jsonl"
stat -f '%z %m' "$A"; sleep 60; stat -f '%z %m' "$A"
lop network log --since <t> --json | wc -l
```

Assert that `Δbytes` over 60 s is consistent with `Δrecords × ~450 B` — i.e. **all
growth is attributable to records**, with nothing unaccounted for. This is the
check that catches a call site the unit test does not know about.

---

## 5. Forensics beyond the log

### 5.1 The incident record — the one page

`<config>/network/incidents/<incident_id>.json`, mode `0600`, written by the same
staged-write idiom as everything else durable here (`registry._staged_write`,
`session/runtime/registry.py:135`), **before** the fan-out starts so that a crash
mid-broadcast still leaves the operator the intent:

```json
{
  "schema": 1,
  "incident_id": "inc_20260919T143225_5f3c",
  "kind": "panic",
  "at": 1789012345.678,
  "at_iso": "2026-09-19T14:32:25.678Z",
  "network_id": "n_5f3c1a2b4d5e6f708192a3b4",
  "network_name": "home-net",
  "raised_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "raised_by_name": "damian-mbp",
  "raised_by_kind": "human",
  "raised_locally": true,
  "epoch_before": 7,
  "epoch_after": 8,
  "secret_replaced": true,
  "latch": "network/panic.latch",
  "reachable_peers": ["d_4b2a91c4e0b87f3a", "d_9a02bb31c4e5f6a7"],
  "broadcast": [
    {"device": "d_4b2a91c4e0b87f3a", "name": "gpu-pod-3",  "outcome": "acked",   "acked_at": 1789012345.812},
    {"device": "d_9a02bb31c4e5f6a7", "name": "damian-imac", "outcome": "unacked", "deadline_s": 2.0},
    {"device": "d_5c11de88a7b6c5d4", "name": "old-laptop",  "outcome": "failed",  "error": "connection-refused"}
  ],
  "local_placements": [
    {"session_id": "2026-09-19T18-04-11_ab12", "placement": "peer:d_4b2a91c4e0b87f3a", "became_unreachable": true},
    {"session_id": "2026-09-18T09-22-04_cd34", "placement": "local",                   "became_unreachable": false}
  ],
  "sessions_stopped": [],
  "notes": "sessions on other devices keep running; re-admit with 'lop network trust home-net --active'"
}
```

`sessions_stopped` is always empty and exists so a reader **sees** the fact rather
than inferring it from §1.6.3. That is deliberate: the most likely misreading of a
panicked network is "it stopped everything", and this is the artifact a person
reads when they are least inclined to read prose.

### 5.2 The panic latch

`<config>/network/panic.latch`, mode `0600`:

```json
{"schema": 1, "network_id": "n_5f3c1a2b4d5e6f708192a3b4",
 "at": 1789012345.678, "incident_id": "inc_20260919T143225_5f3c",
 "cause": "operator_panic", "epoch_before": 7, "epoch_after": 8}
```

- Presence at relay start ⇒ the network loads as `trust: "untrusted"`. Without
  this file a relay restart would silently un-panic a network, which is the single
  worst bug this document could ship.
- Written and `fsync`ed, with its directory `fsync`ed, **before** the local state
  change is announced and before the broadcast (§1.5).
- Removed only by `net_trust`, which is itself audited (`trust_changed`).
- It is also a forensic artifact in its own right: it holds the reason and the
  incident id, so the file alone reconstructs what happened after the log has
  rotated.

Why a file and not a field: `trust` lives in the network record, which is
rewritten by routine traffic (member lists, sequences). A latch is the record of
an *act* and must be independent of any later rewrite — the same reasoning that
puts `runtime-stop.json` in the conversation directory rather than the run
directory, because "it must outlive the record it describes"
(`session/runtime/registry.py:194-204`).

### 5.3 Artifacts that already exist, and what they do and do not say

| Artifact | After an incident it tells you | What it does NOT tell you |
|---|---|---|
| `runtime-stop.json` per conversation (`registry.py:204`; writer at `:212`; payload from `control._stop_marker_payload`, `control.py:812`) | Whether a *local* runtime was stopped deliberately, and by which rung | Anything about a remote session — and its **absence is the signal**: a link loss does not stop a runtime (§1.6.3), so no marker appears, which is correct and confusing in equal measure |
| `run/peers/<pid>.json` (the relay record, transport §2.6) | The relay's liveness, its networks, `epoch`, `links`, `members`, and `trust` / `untrusted_reason` | Per-peer detail |
| `.execution-lease` per session (`session_lease.py:180-206`) | Whether a transcript still has a live writer, and who | Nothing mesh-related; a link loss does not touch a lease |
| `analytics.db` / the `session_spend.v1` row | What a session spent | Who it spent the *credential* of — which is why `mesh-credentials.md` §6.3 adds the owner field |

### 5.4 What this document adds

1. **Nothing new on the relay record.** `trust` and `untrusted_reason` already
   carry the incident state (transport §2.6, §4.2), so `lop network status` and
   the UI read the truth from the published record with no new field. An earlier
   draft of this document proposed a parallel `state` field; it is dropped,
   because two state fields on one record is how they come to disagree.
2. **`<config>/network/peers/<device_id>.json`**, mode `0600`, rewritten **on
   change only** — never per heartbeat, which would be the per-frame I/O A7
   forbids wearing a different hat:

   ```json
   {"schema": 1, "device_id": "d_4b2a91c4e0b87f3a", "name": "gpu-pod-3",
    "last_seen_at": 1789012330.1, "last_seen_instance": "i_2f9c1d4e8a7b6c5d",
    "last_epoch": 7, "our_epoch_at_last_seen": 7,
    "last_refusal": {"at": 1789012000.0, "cause": "epoch_stale", "their_epoch": 6},
    "last_acked_panic": {"incident_id": "inc_20260919T143225_5f3c", "at": 1789012345.812}}
   ```

   This is the file that answers "was `old-laptop` still around with old key
   material?" without a log query, and `last_acked_panic` is what lets `status`
   say "this peer has already been told" instead of leaving the operator guessing
   during recovery.
3. **No new stop rung.** `incidents.py` enumerates the rungs it renders
   (`_SIGNAL_REASON_LABELS`, `_BUILD_REASON_LABELS`,
   `session/runtime/types.py:601-602`) and a `network-offline` rung was considered
   and **rejected**: a rung labels a *cut-off turn*, and no turn is cut off by a
   link loss. Adding one would make a live remote session render as stopped — the
   most expensive kind of wrong. The incident record carries the fact instead,
   which is where a link-level fact belongs.

---

## 6. Test plan — the exact commands a QA agent runs

Isolation per AGENTS.md §Environment and §Isolating a run: `env -i` (strips
`CMUX_*`, which can rename the operator's real cmux workspaces, and `LOP_*`,
which changes what a child runtime believes it is), a fresh `HOME` per cell, and
never the operator's live config.

### 6.1 The two-device harness (one host, two config dirs)

Membership and incidents are keyed on **device id**, not host identity, so two
config dirs on one host model two devices exactly. This is the primary harness:
hermetic, cheap, and safe at the fleet's current load. §6.2 confirms it.

```sh
cd ~/local-operator-worktrees/mesh-network
ISO=${TMPDIR:-/tmp}/mesh-inc-$$; mkdir -p "$ISO/A" "$ISO/B"
runA() { env -i HOME="$ISO/A" LOCAL_OPERATOR_CONFIG_DIR="$ISO/A/.local-operator" \
         PATH="$PATH" TERM=xterm-256color "$@"; }
runB() { env -i HOME="$ISO/B" LOCAL_OPERATOR_CONFIG_DIR="$ISO/B/.local-operator" \
         PATH="$PATH" TERM=xterm-256color "$@"; }
LOP() { .venv/bin/python -m local_operator.cli "$@"; }

runA LOP network init home-net --json
INV=$(runA LOP network invite --role admin --json | jq -r .path)   # transport §5.1: the token is never printed
runB LOP network join "$INV" --confirm-sas XXXX-XXXX --json
runA LOP network peers --json        # assert: 1 reachable peer, epoch 1, trust active
```

### 6.2 Two hosts (confirmation)

The same matrix against the EC2 peer provisioned with the `minerva_nprod`
profile, per `mesh-network.md` §10's two-peer row. This run exists for what one
host cannot show: a real link drop (kill the listener, not the peer), clock skew
between hosts (assert `lop network doctor` reports it), and a broadcast that
partially lands.

### 6.3 Inducing and asserting a **disconnect**

```sh
runA LOP network disconnect
```

| Side | Assertion | How |
|---|---|---|
| A | state is `disconnected` | `runA LOP network ls --json \| jq '.[0].disconnected'` → `true` (the transport's record keeps `trust: "active"`; `disconnected` is this device's own flag) |
| A | the local secret is gone | the transport's `<network_id>.secrets.json` no longer holds this network's `current`; `runA LOP network doctor` reports `secret: not-held` |
| A | its own sessions kept working | create a local session before disconnecting, prompt it after, assert a reply |
| A | the audit has exactly one incident record and no delivery records | `runA LOP network log --json \| jq -c 'select(.event=="disconnect_initiated")'` → one line; **no** `panic_delivered` |
| B | saw a leave, not an incident | `runB LOP network log --json \| jq -c 'select(.event=="member_left" or .event=="epoch_rotated")'` → `member_left` for A, then `epoch_rotated` from the lowest-id active admin |
| B | A's sessions are unreachable, not deleted | `runB LOP sessions --all-peers --json \| jq '.[] \| select(.peer=="d_A")'` → present, `reachable: false` |
| B | B's own work is unaffected | prompt a local B session, assert a reply |
| A | returning needs an invite | `runA LOP network reconnect` is **not a command**; `runA LOP network join` with a fresh invite is the path, and the test asserts the CLI offers no reconnect verb (a negative test, because the temptation to add one is real) |

### 6.4 Inducing and asserting a **panic**

```sh
runA LOP network panic --yes        # --yes because stdin is not a TTY
```

| Side | Assertion | How |
|---|---|---|
| A | `trust == "untrusted"` and the epoch moved by exactly 1 | `runA LOP network status --json` |
| A | the latch exists, with the reason and the incident id | `jq . "$ISO/A/.local-operator/network/panic.latch"` |
| A | **the latch is fsynced before the broadcast** | fault-injection unit test: kill the process after the latch write and assert the reload is `untrusted`; and assert the ordering by patching the broadcast to raise and checking the latch already exists |
| A | the incident record exists, names B, and says nothing was stopped | `jq '{kind, broadcast, sessions_stopped, local_placements, raised_by}' "$ISO/A/.local-operator/network/incidents/"*.json` |
| A | a restart does not un-panic | `runA LOP network serve restart && runA LOP network status --json \| jq .trust` → `"untrusted"` |
| A | nothing was stopped | `runA LOP sessions --json` → same set as before, all live |
| B | `trust == "untrusted"`, all links dropped | `runB LOP network ls --json`, `runB LOP network peers --json` → empty |
| B | B's own sessions are unaffected | prompt a local B session, assert a reply — §1.6.3's prohibition, exercised |
| B | the refusal is total, including a *new* connection from A | restart A's relay, then `runB LOP network doctor --peer d_A` → refused; `runB LOP network log --json \| jq -c 'select(.event=="authorisation_refused" or .event=="handshake_refused")'` → a line with `cause:"untrusted"` |
| Both | each side logs its own half | A: `panic_raised`, `panic_delivered`, `panic_broadcast_result`. B: `panic_received` with `from_device` = A |
| A | the copy says sessions were not stopped | assert the panic output contains `are NOT stopped` (§1.5) |
| A | the copy names who can raise it | assert the output does not claim the panic was authenticated as an admin action |

**Recovery.**

```sh
runA LOP network trust home-net --active --yes
runB LOP network trust home-net --active --yes
runA LOP network peers --json                  # both trust active at the post-panic epoch
runA LOP exec --peer d_B --model <m> "hi"      # a remote session works again
```

Assert also that `trust --active` on **one** device does not un-untrust the other
(§3.1: the local-judgement rule), and that a mid-turn remote session on B was
never interrupted across the whole panic window — the R9 assertion, taken by
prompting B's session before the panic and reading its complete reply after.

### 6.5 Partial reachability and the epoch

Three devices (three config dirs), C stopped before the panic:

- `runA LOP network log --json | jq -c 'select(.event=="panic_undelivered")'` → one line, `detail.outcome == "unacked"`, subject `d_C`.
- `runA LOP network status --json` → the last broadcast tally `{sent: 2, acked: 1, unacked: 1}`.
- Start C and let it dial A: assert **on A** `handshake_refused` (`cause: epoch_stale`) and then, because C is still a member, the reconcile path completing (`link_opened` with `phase: "reconcile"`, then a fresh `link_opened` at the new epoch); and **on C** `peers/<d_A>.json.last_refusal` matching, plus its CLI printing the reconcile line of §3.3.
- Removed variant (`member rm C` on A before it returns): assert C is refused, its own copy is marked `stale: refused_by_peers`, and its CLI prints the §3.3 remedy line verbatim.
- Reconcile flood: four `net_reconcile` attempts from C in an hour ⇒ the fourth is refused with `cause: "reconcile_rate_limited"` and audited.

### 6.6 The I/O measurement, and the writer's invariants

```sh
env -i HOME="$(mktemp -d)" LOCAL_OPERATOR_CONFIG_DIR="$(mktemp -d)/.local-operator" \
  PATH="$PATH" TERM=xterm-256color .venv/bin/python -m pytest tests/unit/network/test_audit_io.py -q
.venv/bin/python scripts/mesh_audit_probe.py --frames 10000 --duration 60 --json
```

with §4.8's four bounds asserted by the probe, plus:

- `test_a_partial_final_line_is_dropped_not_fatal` — append a truncated line, then
  require the reader to serve every earlier record. **SHIPPED**, in
  ``tests/unit/network/test_audit_io.py``; the `--verify` half of this bullet is NOT,
  because there is no `lop network log --verify` in this build (see the note below).
- `test_only_the_relay_opens_the_audit_log_for_append` — an AST/`strace`-free
  check over `local_operator/network/**` that no module other than the relay
  calls `AuditWriter.append`, plus a runtime assertion that a second `AuditWriter`
  on the same path refuses to construct. **NOT SHIPPED, AND THE FIRST CLAUSE IS NOT
  TRUE OF THE SHIPPED DESIGN:** the CLI appends its own local acts (a panic raised
  with no relay running writes `panic_raised` itself), so "only the relay opens it"
  is not the property this build has. What IS true is what the writer enforces: one
  `AuditLog` per process, every append through it, `O_APPEND` on the shared file, and
  a per-process write lock. There is no cross-process writer test.
- the tamper check below, which requires a `--verify` that does not exist in this
  build (`lop network log` has no such flag) — **NOT SHIPPED**:

```sh
runA LOP network log --verify --json      # ok, N records, no break
# edit one byte mid-file
runA LOP network log --verify --json      # not ok: (seq, expected, found)
```

**THE CLASS THIS SECTION KEEPS GETTING WRONG, stated once so it is not repeated:**
the names above were written as a SPEC (§6 opens by listing the tests that must
exist), and a spec read later as a description of the tree is how §4.8 came to cite
`tests/unit/network/test_audit_io.py` and `scripts/mesh_audit_probe.py` for a
document nobody could run them against (perf-lane P-2). Both of those now exist and
are exercised; the two bullets marked NOT SHIPPED above are real gaps, recorded as
such here and on the PR rather than implied to be covered.

### 6.7 The prohibition, as an executable assertion

`tests/unit/network/test_incident_never_touches_sessions.py`:

- `test_incident_module_does_not_import_session_machinery` — parse
  `local_operator/network/incident.py` with `ast` and assert no import of
  `local_operator.session.runtime`, any stop/lease module, or `os.kill`.
- `test_no_audit_record_contains_a_token_shaped_string` — a property check over
  the `audit.jsonl` produced by the whole §6.3–§6.6 matrix, reusing the
  providers' patterns from `local_operator/redaction_shapes.py`.
- `test_every_emitted_event_is_in_the_taxonomy` — §4.3's drift guard.

---

## 7. File-by-file change list (for the coder)

| File | Change |
|---|---|
| `local_operator/network/incident.py` | the incident verbs and the state machine: `disconnect()`, `panic()`, `trust()`; the latch; the incident record; the fan-out. Docstring must open by disambiguating from `local_operator/incidents.py` (§4.1) |
| `local_operator/network/audit.py` | taxonomy constants + the per-event `detail` whitelist, `AuditRecord`, `canonical_row`, `AuditWriter`, rotation, `iter_records`, `verify` |
| `local_operator/network/peer_state.py` | `network/peers/<device_id>.json`; change-only writes |
| `local_operator/network/arguments.py`, `local_operator/network/cli.py` | the `lop network` group, self-contained and wired from `cli.py` the way `tunnels/arguments.py:9` and `secrets/cli.py:49` are: `disconnect`, `panic`, `trust --active`, `log [--follow\|--verify\|--export\|--since]`, `status`, `doctor`, `member rm`, plus `--json` and `--yes` on every one |
| `local_operator/network/membership.py` | **transport's file**; this document requires only that the `net_epoch` sender honours §2.3.4's fix and that `net_panic` is **not** queued for offline members |
| `local_operator/network/link.py` | **transport's file**; requires the `net_panic` / `net_leave` / `net_trust` send paths to return per-peer ack/outcome so the incident record can be precise |
| `local_operator/slash_commands.py` | `/network disconnect`, `/network panic`, `/network log`, `/network trust` entries, each with its `desktop_destination` **deliberately left unset** (the delta `mesh-ui.md` §2.7 records against this table: the mesh lifecycle has no `/v1/desktop` adapter in this pass, and an offered-but-dead palette row is the failure `/mobile`'s entry already documents. The read-only verbs get destinations in the PR that brings the adapter, in the same change) |
| `scripts/mesh_audit_probe.py` | §4.8's instrument |
| `tests/unit/network/test_incident_state_machine.py` | every transition in §1.2, including refusals |
| `tests/unit/network/test_audit_records.py` | §4.2 schema, §4.3 taxonomy drift guard, §4.4 chain, `detail` whitelist |
| `tests/unit/network/test_audit_io.py` | §4.8's bounds, the 10,000-frame discrimination test, the writer's single-owner invariant |
| `tests/unit/network/test_broadcast_partial.py` | §2.2's five rows and §6.5's epoch cases |
| `tests/unit/network/test_incident_never_touches_sessions.py` | §6.7 |
| `tests/e2e/` | the two-host confirmation run (§6.2) |

Nothing here writes to or reads `credentials.env`, `secrets/`, `auth.db`, the
keychain, or any session's transcript.

---

## 8. Open questions, each with my recommendation

**Q1 — Where does the rotation's secret inclusion get fixed, and does panic get
the same treatment?** *Recommendation:* fix it in the **sender**
(`mesh-transport-identity.md` §8.1 step 3), so a `net_epoch` frame addressed to a
device named in `removed` carries no `secret` — or, equivalently, the removed
device is not a rotation recipient at all. **[ADOPTED — convergence round 1.]**
The transport now does exactly this, at the sender, and goes one step further
than this question asked: the **offline queue** is held to the same rule, so a
queued rotation for a removed member is written without key material or dropped
(the second half of this question's own second sentence). It also states the rule
as two named invariants with named tests rather than as prose, so a regression
fails a test rather than a review.

The rest of the question, kept as the pre-fix record: apply the same rule to a
queued `net_epoch` (a plaintext network secret sitting in an offline queue on
disk is a second instance of the same defect) — which is precisely the half the
transport now implements explicitly. For `net_panic`, the trade is different and
I recommend **leaving it as designed**: panic wants every survivor coherent so
recovery is one command per device, and the operator who needs confidentiality
against a specific device uses `member rm`. *Evidence that would settle the
panic half:* a stated operator requirement that a panic must also be
confidentiality-preserving against a member that received it; absent that, the
coherence argument wins.

**Q2 — Should `net_panic` be queued for offline members like `net_epoch`?**
*Recommendation:* **no.** A queued panic replayed at an unknown later time is a
stale judgement about a network state that may have changed, and it would make
`net_trust --active` non-monotonic (a device re-admits, then a replay re-untrusts
it). The epoch check already makes an unqueued panic safe, and §3.3's reconcile
path covers the sleeping member. *Evidence that would change it:* a real incident
where a sleeping member stayed trusted and reachable after a panic *and* the
epoch check failed to refuse it — which would be a bug in §2.4, not in this
answer.

**Q3 — Should `net_invite` be device-bound?** *Recommendation:* yes, add the
optional `device_id` field (`net_invite {network, role, ttl_s, device_id?}`),
refused when redeemed by another id as `pairing_refused` with `cause:
"wrong_device"`. **[ADOPTED — convergence round 1.]** Implemented in
`mesh-transport-identity.md` §5.1 (`--device` / `bound_device` in the invite
envelope and in the record's `invites[]`) with the redemption check in its §5.2
and the state machine in its §5.4; the wrong-device case **consumes** the invite,
which this section did not specify and the transport now does. After a compromise an unbound invite is a bearer token for the
network's key material, and the operator's whole intent is "this device comes
back". The change is one optional field plus one comparison and belongs in the
transport document; it is raised here because the incident path is where the need
appears. *Evidence:* the transport's §5 design, which currently records
`redeemed_by` only after the fact.

**Q4 — Keep the hash chain, or ship plain JSONL?** *Recommendation:* keep it
(§4.4). It is ~20 lines because `secrets/audit.py` exists to copy, it costs ~1 µs
per record, and the alternative is a forensic log an attacker can edit. The
transport's §7.6 does not include it, which is why this document adds it
explicitly. *Evidence that would change it:* a `--verify` pass over a full 40 MiB
retained set that is not comfortably sub-second.

**Q5 — Is 40 MiB compressed / 90 days the right retention?** *Recommendation:*
ship it, then revisit once `mesh_audit_probe.py` has a week of real numbers.
*Evidence that settles a change:* the probe's `records_per_day` on the operator's
own machine; if it exceeds ~200,000/day, a frame path has crept in and the fix is
that call site, not the retention.

**Q6 — Should the audit log be per-network instead of per-install?**
*Recommendation:* **per install**, as A7 says. One ordered history is what an
incident review needs when a device is in two networks and the two events
interleave; `network_id` on every record gives the per-network view
(`net_log {network?}` already filters), so a per-network file would buy nothing
and cost a second writer path. *Evidence:* the transport's local `net_log` op
already takes an optional network filter (its §2.5), which confirms the filter is
the right seam.

**Q7 — Where do `pool` lifecycle events go?** *Recommendation:* this log, as
`meter_interval` / `meter_close` (`mesh-compute-pool.md` §5.2) plus the existing
`member_*` events — never a second log. A
second log would need its own chain, rotation and retention, and would split the
one ordered history an incident review depends on. *Evidence:* the final shape of
the member lifecycle in `mesh-transport-identity.md` §12.4. **[ADOPTED —
convergence round 1.]** §4.3 now reserves `mesh-compute-pool.md` §3.4's six
lifecycle names beside the metering names, so the taxonomy is complete in one
place without either document owning the other's field lists.

---

## 9. Convergence round 1 — what changed here

This document raised the security defect and the invite question, and both were
settled in the document that owns the mechanism, so most of round 1 changed this
document by *closing* rather than by editing. The edits:

1. **§2.3.4 keeps its text and gains a resolution paragraph.** It is the canonical
   statement of the defect — the sentence in `mesh-network.md` §4 and the
   paragraphs here are the argument, and the enforcement is
   `mesh-transport-identity.md` §8.1 step 3 / §8.3. §8 Q1 is marked adopted, and
   carries the one part this document did not specify: the **offline queue** rule.
2. **§3.2/§8 Q3, the device-bound invite, is marked adopted** in the transport,
   including the decision this section left open — a wrong-device redemption
   **consumes** the invite.
3. **§4.3 gains the `act`/`sub` delegation markers** on the credential audit
   events, so a forensic read can distinguish brokered use from direct use
   (RFC 8693 delegation; `mesh-credentials.md` §1.4). This document owns the
   audit schema, so the field lands here even though the vocabulary is that
   document's.
4. **§4.3 reserves the pool lifecycle events** (`pool_grant_minted`,
   `pool_request_queued`, `member_draining`, `drain_barrier_timeout`,
   `pool_member_expired`) owned by `mesh-compute-pool.md` §3.4, exactly as it
   already did for the metering pair, and adds `epoch_rotated
   {reason: "max_age"}` for the transport's new bounded secret lifetime.
5. **§7's file list records the delta `mesh-ui.md` §2.7 asked for**: the four
   `/network` slash entries ship with `desktop_destination` **unset**, and this
   document had said the opposite.

**One mechanism, one owner, and it is the transport's:** the bounded secret
lifetime (`network.epoch_max_age_s`) is specified in
`mesh-transport-identity.md` §8.4 because it is a change to the secret's schema
and to the handshake's phase decision. What this document contributes is the
audit consequence — a rotation with `reason: "max_age"` is distinguishable from
one forced by a removal, which is the difference between "the network re-keyed
itself on schedule" and "someone was evicted" when an operator reads the log
months later.
