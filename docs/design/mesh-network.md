# The lop mesh network — requirements and architecture spine

Status: **design, pre-implementation**. This document is the single authoritative
statement of *what* the mesh must do and *why* it is shaped the way it is. The
detailed designs hang off it by section reference:

| Detail design | Covers | Requirements owned |
|---|---|---|
| `docs/design/mesh-transport-identity.md` | Wire protocol, device identity, pairing, membership, revocation | R1, R2, R3, R4, R5 |
| `docs/design/mesh-session-mobility.md` | Remote sessions, move/fork, lease handoff, delete/archive on a peer | R7, R8, R9, R10, R11, R12 |
| `docs/design/mesh-credentials.md` | Credential ownership, brokered use, refresh-on-behalf, MCP grants | R13, R14, R15, R16 |
| `docs/design/mesh-incident-response.md` | Disconnect, broadcast kill, epochs, audit log, retention | R17, R18 |
| `docs/design/mesh-compute-pool.md` | Forward-compatibility: metered on-demand capacity (Radient credits) | R20, R21, R22 |
| `docs/design/mesh-ui.md` | TUI + desktop sidebar surfaces, the local/remote annotation, space budget | R6, R19 |
| `docs/design/mesh-prior-art.md` | Supporting research note: verified licences and prior art for the decisions above; explicitly not binding | — (carries no requirement) |
| `local_operator/guides/network/GUIDE.md` | The agent-facing playbook (how an agent sets a network up from a verbal request) | — (the R19 deliverable) |

**Ownership is total and disjoint**, so coverage is checkable rather than
asserted: every `R1`–`R22` appears in exactly one row, the owning document is
where that requirement's detail *and its evidence* live, and a document that
merely implements against a requirement says so instead of claiming it. Round 1
moved three of them to make that true: **R6** to `mesh-ui.md` (it is a display
requirement, and §8 already pointed there for its detail), **R17** to
`mesh-incident-response.md` (the transport owns the revocation *mechanism* it is
built on, not the requirement), and **R19** to `mesh-ui.md` (the guide and the
tool; `mesh-transport-identity.md` §12.5 owns R19's transport half). To verify
mechanically: collect the `R<n>` markers from each document's header table and
assert the union is exactly R1–R22 with no duplicates.

A requirement is `R<n>`; a design decision is `A<n>`. Every PR that implements
part of this names the requirements it closes.

---

## 1. The product in one paragraph

A `lop` install today owns its sessions on exactly one machine: a session is a
detached runtime process that publishes a pid-keyed record and listens on a
loopback control socket, and every front end — the TUI, `lop serve`, the phone
daemon, `lop send` — is a viewer that dials that socket. Everything is keyed by
pid and by same-uid, so the session plane stops at the machine boundary.

The mesh extends that plane across machines. Each install runs a **relay** that
joins a **network**: a named set of paired devices, each holding a private key
and a shared network secret. Any relay on the network can list every session on
every peer, serve a TUI or the desktop UI for any of them, and pilot a session
that physically runs on another device — start it there, prompt it, steer it,
stop it, run slash commands in it, archive or delete it there, or move it
between devices. Local and remote sessions are the same object with a different
`locality`; that is the whole point of the existing detached-runtime split.

The same machinery, extended later and not here, is also the substrate for
**metered on-demand capacity**: a pod that boots a clean `lop`, pairs itself as
a member of a network, runs payloads, and spins down when idle, billing Radient
credits by compute time and size. §9 states what must be true *now* so that is
an extension rather than a refactor.

---

## 2. Requirements

### 2.1 Networking and pairing

- **R1 — a relay per install.** A supervised per-install relay joins the
  network, accepts peer connections, and exposes the local session plane to
  authorised peers. It must not become a second owner of session state; the
  session runtime stays the owner (§3, A1).
- **R2 — networks are named, plural, and manageable.** A device can be a member
  of more than one network. A network has a stable id, a human name, a member
  set, and an epoch. Create / list / rename / modify / delete are first-class
  operations (§6).
- **R3 — pairing is explicit and key-based.** Joining a network requires the
  network's private key material to be transferred out of band, and requires a
  human on *both* devices to confirm the same short authentication string
  before the member is admitted. No device joins silently (§5, A3).
- **R4 — zero trust between peers.** Peers are not trusted because they are on
  the local network, in the same account, or on the same host. Every connection
  is mutually authenticated, every request is authorised against the specific
  network membership and the specific capability it needs, and the transport
  must assume an adversary on the path (§5).
- **R5 — membership is revocable without touching the other devices.** Removing
  a member, or a device deactivating itself, must take effect for the remaining
  members without a per-device visit (§7).

### 2.2 Sessions across the network

- **R6 — one session list.** Every front end shows all reachable sessions with
  a local/remote annotation; the desktop sidebar shows the same, inside its
  existing space budget (§8).
- **R7 — a remote session is a first-class session.** Prompt, steer, stop,
  slash commands, model/effort changes, approvals, asks, subagents, todos, and
  the transcript stream all work against a remote session exactly as against a
  local one. The user's words: "all supported commands should be allowed with
  the peers as if it were a local session".
- **R8 — create remote.** `/new remote <peer>` (TUI) and the CLI equivalent
  create the session *on* the peer. Peer names and ids are completed and
  suggested for autofill. `lop exec` gains the same placement option.
- **R9 — quitting is never fatal to a remote session.** Closing the local TUI,
  or losing the link, leaves the remote runtime running. This is the existing
  detached-runtime contract extended across the link; it must not regress the
  local case.
- **R10 — lifecycle operations apply where the session lives.** Delete and
  archive issued against a remote session act on the peer (the WIP on session
  archive/delete, PR #1328, is the local design this must mirror; see §4 and
  `mesh-session-mobility.md` for the interface it must expose).
- **R11 — mobility.** A session can be *moved* between devices. Moving off a
  peer copies the full session to the local device and, by default, deletes it
  on the peer; a fork-and-copy mode leaves the original in place. Moving a
  local session *to* a peer is the more common direction — it is how load is
  shed — and follows the same rules with the ends swapped.
- **R12 — mobility is safe under concurrency.** A move must not produce two
  live writers of one transcript, and must not lose a turn that was in flight
  when it was requested. The existing single-writer lease and the
  `exclusive-move-v1` fence are the primitives to build on (§4, A4).

### 2.3 Credentials and configuration

- **R13 — credentials follow the session, not the device.** A session created
  on a peer must have working model access, MCP servers, and account logins
  without the operator re-authenticating there.
- **R14 — credentials have an owner device, and are not spent off it.** The
  network records which device owns each credential. A peer that is not an
  approved holder must not use (and above all must not *refresh*) a credential:
  a rotating or single-use refresh token spent from a second device is exactly
  the failure that logs every device out (measured once already — see A5). 
- **R15 — refresh on behalf of a peer.** When a session on peer B needs a fresh
  access token that device A owns, B asks A; A performs the refresh and returns
  the usable token. B never contacts the provider's token endpoint itself.
- **R16 — no regression on provider or MCP logins.** Pairing, moving sessions,
  or running remote must not disturb the operator's existing logins, and must
  not leave a credential store in a state a later local run trips over.

### 2.4 Operations, incident response, forensics

- **R17 — a one-command stop.** There is a straightforward way to deactivate /
  disconnect this device from a network, and to signal every other peer to
  disconnect and stop trusting the network, for use in an incident (§7).
- **R18 — forensics with sane, cheap retention.** Membership changes, pairing,
  session handoffs, credential brokering, authorisation refusals, and incident
  actions are logged append-only with enough context to reconstruct who did
  what to which session when — with bounded retention and I/O that does not
  cost a disk write per frame (§7).

### 2.5 Agent-facing

- **R19 — agents can set a network up from a verbal request.** There is a
  `guides/network/GUIDE.md` playbook of the same shape as the mobile and tunnel
  guides, and an agent tool, so "pair this machine with my laptop and move the
  heavy session over" is something an agent can execute by driving the
  authenticated CLI.

### 2.6 Forward compatibility (designed now, built later)

- **R20 — metered on-demand capacity.** The schema must accommodate a member
  that is provider-owned and ephemeral: provisioned on demand for a user, paired
  as a clean `lop` instance, running payloads, and spinning down on idle, with
  Radient credits deducted by compute time and compute size.
- **R21 — no major refactor later.** Time tracking, capacity allocation,
  on-demand delegation, ephemeral pairing, and session sync-before-spin-down
  must arrive as new implementations of interfaces this design already names —
  not as a change to the shape of a session, a member, or a record.
- **R22 — the spin-down must not cost the result.** Sessions sync across the
  mesh at a sane cadence and on the final message before idle, so that when an
  on-demand device powers off, the local device still holds the finished
  transcript. This is the same primitive a move uses.

---

## 3. The architecture in one page

```
   device A (operator's laptop)                    device B (peer / pod)
 ┌───────────────────────────────┐              ┌───────────────────────────────┐
 │ TUI / desktop / phone         │              │ TUI / desktop / phone         │
 │      │  (viewer)               │              │      │                        │
 │      ▼                         │              │      ▼                        │
 │ session runtime  ← control     │   peer      │ session runtime  ← control     │
 │  (owner of transcript)  socket │   link      │  (owner of transcript)  socket │
 │      ▲                         │◄────────────►│      ▲                        │
 │      │                         │   relay     │      │                        │
 │  relay (this install)  ────────┼──────────────┼───► relay (this install)      │
 │   · membership + identity      │   mesh       │   · membership + identity      │
 │   · remote session projection  │              │   · credential authority       │
 │   · credential broker          │              │   · audit log                  │
 └───────────────────────────────┘              └───────────────────────────────┘
```

Layers, bottom to top:

1. **Session runtime** (exists) — one detached process per session, owns the
   transcript and the lease, listens on a loopback JSON-lines control socket
   authenticated by a per-record key. Unchanged by this work.
2. **Relay** (new) — per install. Owns device identity, network membership, the
   peer listener, the peer link sessions, the remote-session projection, the
   credential broker, and the audit log. It is *not* a session owner.
3. **Projection** (new) — a remote session, as seen locally: a
   `SessionHandle`-shaped adapter that turns local viewer traffic into peer
   requests, and peer events back into the local event vocabulary. This is what
   makes "a remote session is a first-class session" true without a second
   front-end code path.
4. **Front ends** (exist) — TUI, `lop serve` / desktop, phone daemon. They gain
   a locality dimension on their session rows, not a new rendering path.

---

## 4. Decisions (ADRs)

### A1 — Reuse the session control protocol; do not invent a session protocol

The peer wire protocol is the existing control-socket vocabulary
(`ControlOp` / `EventOp`, JSON-lines, `hello` → `welcome` → events) carried over
the peer link. Rationale: it is already the whole command surface a viewer
needs, it is versioned and gate-checked, and the repo already reserved the
exact field for this — `ClientLocality = Literal["local", "remote"]` in the
client auth frame, added with a docstring naming the relay case: a relay
forwarding a remote device's commands "must declare `remote`". Re-deciding
locality at each call site is precisely what that field exists to avoid.

Consequences: remote ops are additive entries in the existing dispatch, and any
new capability automatically has one place to declare its remote behaviour.

### A2 — One relay per install, in a new record namespace

The relay publishes under a **new** namespace (`run/peers`, following the
existing one-namespace-per-record-kind rule) and never widens `run/mobile`.
Rationale: the record namespace is a wire constant with a documented rule
against widening it, and a peer record carries fields (`network_id`,
`device_id`, `epoch`, `endpoint`) that a session record must not carry.

Consequences: `lop sessions` keeps meaning "sessions", `lop network status`
means "the mesh", and a federated catalogue scans two namespaces instead of one
overloaded one.

### A3 — Identity: per-device keypair, per-network shared secret, human-confirmed

- Every install generates a long-lived **device identity keypair** at first use.
  A device id is the fingerprint of that public key — self-certifying, so a
  member list can be verified without a CA.
- A **network** is created with a random **network secret** (the "private key to
  be shared" each member holds) and an epoch counter.
- **Joining** transfers the network secret out of band via a single-use,
  expiring invite token bound to the inviting device — and, optionally and
  preferably, to the *joining* device id, so an invite that escapes its intended
  device is refused and burned rather than being a bearer token for the key
  material (§6's `--device`) — and both devices display a short authentication
  string derived from the handshake transcript; the human
  confirms it matches on both ends before the member row is written. A
  mismatched SAS is a refused join, not a warning — this is what makes the
  pairing resistant to a relay in the middle.
- **Connections** are mutually authenticated at the transport layer *and*
  authorised at the application layer: transport proves possession of the
  device key, the application proves current membership at the current epoch.
  Both must hold for a request to dispatch.

This satisfies R3 and R4, and the network secret is what makes R5 (revocation
without a device visit) possible — revocation rotates it.

### A4 — Session mobility is fork + retire, fenced by the existing lease

`fork_session` already copies a session's file set; that is the transportable
half of a move. The half that does not exist is retiring the source safely, and
the repo already has the fence for it: the `exclusive-move-v1` capability and
`RuntimeServer._exclusive_move_fence` are the "retire this owner so a successor
takes over" protocol. A move is therefore:

1. resolve the session and its owner (local runtime or peer relay);
2. quiesce: refuse new turns, drain the in-flight turn (bounded, with a
   deterministic timeout);
3. copy the full session (transcript, origin, attachments, and the metadata
   that makes it resumable) to the destination, which adopts it as a cold
   session;
4. hand over the lease under the exclusive-move fence;
5. retire the source — delete it, or leave it as a fork, per the requested mode.

Consequences: the default (`move`) deletes on the source; `--keep` leaves a
fork. Both directions are the same operation with the ends swapped, so
"move this heavy session to the peer" and "bring it home" are one code path.
This satisfies R11 and R12.

### A5 — Credentials are brokered, never mirrored

The measured failure this decision exists to prevent: two processes racing a
rotating OAuth refresh token invalidated each other's new token (PR-24). The
repo's fix was a local SQLite lease table — which cannot span hosts. Copying
`auth.db` to a peer reproduces the bug exactly, so **token material does not
replicate**:

- Each credential row carries an **owner device** and, for agentic credentials
  that may be brokered, a **holder set** (which peers may use it, and whether
  they may use it for a specific session or for everything).
- A peer that needs a token asks the broker on the owner device. The owner
  performs the refresh — under the same local lease it already uses — and
  returns only what the caller needs, scoped and short-lived.
- A peer may hold a *copy* of a long-lived, non-rotating credential (a plain
  API key) when the operator explicitly opts that credential into replication;
  rotating credentials are never eligible.
- Grant state that is inherently interactive (an MCP OAuth browser callback)
  is brokered as a request to a human, not replayed on the peer.

This satisfies R13–R16 and is the seed for R20 (§9): a pod holds no credential
of its own; it borrows the operator's, metered.

### A6 — Incidents are a state change, not a cleanup task

`lop network disconnect` and `lop network panic` are the two shapes:

- **disconnect** — this device leaves the network. The local relay closes links,
  marks the network inactive, and keeps the audit trail.
- **panic** — this device broadcasts a *revoke* to every reachable peer, bumps
  the network epoch with a rotated secret, and refuses any further traffic until
  the operator re-admits devices. Peers that receive it drop the link, mark the
  network untrusted, and stop accepting work from it even if a connection
  arrives afterwards.

Losing a device is the same mechanism from the other side: any member may
initiate a rotation, and members that never learn of it are refused at the
epoch check. This satisfies R5 and R17.

### A7 — Audit is append-only JSONL with bounded, batched writes

One audit log per install, one record per *semantic* event (membership change,
pairing, handshake refusal, session handoff, credential brokering, incident
action, placement decision) — never per frame, never per token delta. Writes are
line-buffered; rotation is by size and by age with a fixed number of retained
generations, so retention is bounded by construction and disk I/O is bounded by
event rate rather than by traffic. Key material is never written. This satisfies
R18.

### A8 — Placement is an explicit property from day one

A session carries a **placement**: where it runs and why (`local`, `peer:<id>`,
`pool:<id>`), plus the policy that governs it (`pinned`, `prefer-remote`,
`cost-capped`). Nothing in this pass schedules automatically, but the field, the
record that carries it, and the operation that changes it exist now, so that a
future scheduler is a new producer of placements and a metered pod is a new kind
of member — not a change to what a session is. This is what makes R21 true.

---

## 5. Security model

**Assets.** Transcript contents and tool outputs (which routinely contain
credentials and customer data), provider and MCP tokens, the session control
keys, and the ability to run commands on a peer (a session is arbitrary code
execution by design — the agent has a shell).

**Adversary.** An attacker on the path between two peers; a hostile device that
was once a member; a compromised peer link; a curious process on the same
machine; and the operator's own mistakes (pairing with the wrong device).

**Properties to hold.**

1. **Confidentiality and integrity in transit.** No plaintext over a public
   boundary, ever. Traffic is encrypted and authenticated; a modified frame is
   rejected, not repaired.
2. **Peer authentication.** A connection is accepted only from a device holding
   a member key at the current epoch, proven by possession — not by address,
   hostname, or account.
3. **Least authority.** A peer may do exactly what its membership grants. The
   grant is per capability (list, view, prompt, steer, stop, slash, delete,
   move, broker-credential) and per network, defaulting to the smallest useful
   set. Read-only membership is a first-class level, because "let my laptop see
   the fleet" is a common and much safer ask than "let it drive everything".
4. **No session key on the wire.** The loopback control key stays local; the
   peer link authenticates the *relay*, and the relay is the only thing that
   ever dials a local control socket. A peer never receives a control key.
5. **Replay resistance.** Frames are bound to the link's fresh session and
   rejected if replayed across links.
6. **Revocation is real, in both directions, and no secret is unbounded.**
   Removing a member rotates the secret and bumps the epoch, and the rotated
   secret is **withheld from the removed device at the sender**: a frame (or a
   queued copy of one) addressed to a device that frame's own `removed` array
   names carries no key material, so the eviction costs the removed device the
   ability to *read* what follows as well as to act, and its next request fails
   authorisation even though it still holds the old secret. And no secret
   outlives `network.epoch_max_age_s` (default 30 days): a still-member device
   that never comes back ages out and must reconcile-and-rotate, which is the
   half of R5 that rotation-at-removal-time alone does not cover
   (`mesh-transport-identity.md` §8.1, §8.3, §8.4; prior art in
   `mesh-prior-art.md` §2).
7. **Fails closed.** Any error in authentication, authorisation, or epoch
   verification refuses the request. There is no degraded-trust mode, and no
   "localhost so it must be fine" shortcut for a peer link.
8. **Locality is declared, not assumed.** Every command carries `locality`, and
   the receiving side may use it in authorisation decisions; a command that
   claims `local` while arriving over a peer link is a protocol error, not a
   softer path.

**Not in scope for this pass** (stated so it is not implied): multi-user rather
than multi-device networks; per-user identity within a network; protection
against a malicious *member* on capabilities it legitimately holds; and any
form of key escrow.

---

## 6. CLI and slash-command surface

`lop network` (new group; args parser lives with the feature, wired from
`cli.py` like `secrets` and `tunnels`):

| Command | Effect |
|---|---|
| `lop network init <name>` | Create a network on this device; print its id and the invite instructions |
| `lop network invite [--expires 10m] [--role read\|drive\|admin] [--device <device>]` | Mint a single-use invite token for this network. `--device` binds redemption to one device id, so a token that reaches any other device is refused and burned (`mesh-transport-identity.md` §5.1); it is the form to use after any compromise |
| `lop network join <token>` | Join a network from an invite; shows the SAS for confirmation |
| `lop network ls` | Networks this device is in, with role, epoch, member count, reachability |
| `lop network show <network>` | Members, roles, endpoints, capability grants, audit tail |
| `lop network rename <network> <name>` / `rm <network>` | Modify / forget a network locally |
| `lop network member rm <network> <device>` | Revoke a member (rotates the secret, bumps the epoch) |
| `lop network peers` | Reachable peers right now, latency, session counts |
| `lop network serve` / `start` / `stop` / `restart` | The relay (supervised like `lop mobile serve`) |
| `lop network uninstall [--purge]` | Remove the LaunchAgent and the plist, reporting each step — exactly `lop mobile uninstall`'s shape (`mobile/install.py:332`). `--purge` additionally deletes the **network records, invites and the outbox for the networks being uninstalled** — the only way to make a device forget a network it can no longer reach. It does **not** delete the **device identity keypair**: that key is unrecoverable and is what every *other* network addresses this device by, so removing it needs its own flag, `--purge-identity`, which requires an interactive TTY confirmation naming every network still known to that identity; without a TTY it is refused outright with a message naming the flag that does work. This row is not optional: a plist that can be installed but never removed is an incomplete lifecycle, and installation and removal are one surface. The scope split is the same safety rule `lop network trust` applies to its own irreversible act — one flag, one blast radius, and the wider one needs a human |
| `lop network status` | Install state, health probe, links, log paths |
| `lop network disconnect [<network>]` | Leave; stop trusting; close links |
| `lop network panic [<network>]` | Broadcast revoke + epoch rotation + refuse further traffic |
| `lop network log [--follow] [--network x] [--since]` | The audit log |
| `lop network doctor` | Diagnose a link: DNS/reachability, handshake, epoch skew, clock skew |

Every action takes `--json`, following the mobile group's contract, because the
agent path drives the CLI and parses it.

Sessions gain:

| Command | Effect |
|---|---|
| `lop sessions --peer <peer>` / `--all-peers` | Federated session list with a locality column |
| `lop exec --peer <peer> "…"` | Run the one-shot on the peer |
| `lop send --peer <peer> <session> …` | Peer message to a remote session |
| `lop sessions move <session> --to <peer>` / `--to local` | Move (default: delete on source) |
| `lop sessions move <session> --to <peer> --keep` | Fork and copy, leave the source |

TUI slash commands:

| Command | Effect |
|---|---|
| `/network [ls\|status\|peers\|log]` | The network view; `/network` alone lists |
| `/network new <name>`, `/network invite`, `/network join <token>` | Lifecycle |
| `/network disconnect`, `/network panic` | Incident controls |
| `/new remote <peer> [prompt]` | Create the session on a peer; `<peer>` completes |
| `/move <session> --to <peer\|local> [--keep]` | Mobility |
| `/peers` | Alias for `/network peers` |

Slash commands are routed to the session's *owner*: a command typed at a remote
session executes on the peer runtime, reusing the existing
`route_shared_slash` → `run_slash_authoritative(locality="remote")` path. The
existing rule holds unchanged — process- and terminal-local commands (`/quit`,
`/resume`, pickers) never leave the viewer; that list is the design's one
explicit exception table and it is enumerated in `mesh-session-mobility.md`.

---

## 7. Incident response and forensics

**Disconnect (this device leaves).** Close links, mark the network inactive
locally, keep the audit trail, and surface it in `lop network ls` as
`disconnected`. Sessions that were remote and are no longer reachable are shown
as *unreachable*, not deleted — the operator must not lose a session because a
link dropped.

**Panic (the network is compromised).** Broadcast a revoke to every reachable
peer, rotate the network secret to a new epoch, and refuse all further peer
traffic until the operator explicitly re-admits devices. Peers mark the network
untrusted on receipt. This must be reachable from the TUI in one command and
from the CLI without arguments resolving to the obvious single network.

**Audit records** carry: timestamp, actor device (self), subject device, event
type, network id, epoch, session id when applicable, outcome, and a bounded
detail map. Never key material, never transcript content, never token values.

**Retention default.** Size-capped rotation with a small number of generations
(tuned so the steady-state footprint is measured in megabytes, not gigabytes),
plus an age cap; the exact numbers and the rotation implementation are pinned in
`mesh-incident-response.md` with a measured I/O check — R18 requires the I/O
argument to be measured, not asserted.

---

## 8. Surfaces

**TUI.** The session sidebar already groups sessions; it gains a peer group
heading and a per-row locality mark, reusing the existing status-glyph slot so
the row does not grow. `/new remote <peer>` autocompletes from the cached peer
list, offline-safe (a stale list is usable, an unreachable peer is refused with
its reason).

**Desktop (`local-operator-ui` + `/v1/desktop`).** Measured budget from recon:
sidebar default 280px, clamped 240–360; rows are 32px (`h-8`); each row has a
leading `ChatSessionStatus` glyph, a truncating title, and **exactly one**
trailing statement already spoken for by binding/not-sent/search-qualifier.
Therefore:

- the remoteness annotation must be **budget-neutral**: it joins the leading
  status glyph slot or the tooltip, and must not consume the trailing slot;
- a peer is a **group heading** (the heading primitive already renders
  label + count), not a new column;
- the session catalogue row gains an explicit `locality` field rather than the
  UI inferring it from an id shape.

The design detail, including the empty/loading/error/one-peer/many-peer states
and the screenshots to capture, is in `mesh-ui.md`.

---

## 9. Forward compatibility: metered on-demand capacity

Not built in this pass; **specified**, because R21 fails if the schema gets it
wrong now.

The future shape: a user on network N runs out of local capacity; a `pool`
member (provider-provisioned, ephemeral, running a clean `lop`) is created on
demand, pairs into N as a member with a restricted role, receives a session
placement, runs it, and spins down when idle. Credits are deducted by compute
time and size.

What must exist now for that to be an extension:

1. **Member kind is a field, not an assumption.** A member record already
   distinguishes `device` from `pool`, carries a lifecycle (`active`,
   `provisioning`, `draining`, `expired`), and can be ephemeral by policy.
   Nothing in the pairing path assumes a human is present at the other end.
2. **Placement is explicit** (A8) and independently mutable, so a scheduler can
   move a session without touching session semantics.
3. **Sync is a primitive, not a side effect.** R22's cadence sync and the
   pre-spin-down sync are the same operation a move uses (copy the session's
   durable state), so "the pod is about to die" is just a scheduled move with
   `--keep`.
4. **Metering is an event stream, not a field.** The relay already has to emit
   audit events; a `metering` event kind (compute seconds, size class, session
   attribution) rides the same channel, so billing consumes events and never
   reads session state.
5. **Credential brokering is already mediated** (A5), which is what makes it
   safe to hand work to a provider-owned device: the pod never holds a
   credential outright.
6. **Capacity is not a member.** The thing that *creates* pools is a separate
   control plane talking to a provider; the mesh only sees the member it
   produced. Keeping that boundary now is what avoids a refactor when the
   provider becomes Radient rather than a script.

Explicitly deferred, and named so nobody assumes otherwise: scheduling
decisions, price/credit computation, provisioning API, kill-on-idle policy
tuning, and the billing integration itself.

---

## 10. Testing and evidence plan

Requirements are verified end to end, not by unit tests alone. The matrix is
three (optionally four) real topologies:

| Topology | Why |
|---|---|
| **0 peers** | Regression: an install with no network must behave exactly as today (R16, R9, and the whole local path) |
| **1 peer** | The core: pair, list federated, create remote, prompt/steer/stop/slash, quit locally, move both directions, brokered credential |
| **2 peers** | Generalisation: three-way listing, a session moved A→B→C, simultaneous ops, peer-to-peer (not just peer-to-centre) reachability, incident broadcast to more than one listener |
| **3 peers** | Only if 2 shows something that might not generalise |

The peers are real: an EC2 instance provisioned with the operator's
`minerva_nprod` AWS profile for the remote device, piloted from this machine
over the mesh. Evidence captured per requirement: the command, its actual
output, and the observable effect (transcript written on the peer, process
present on the peer, audit record, sidebar frame). Visual claims carry rendered
frames, before and after, per the repo's visual-validation rules.

---

## 11. Convergence round 1 — the spine's changes

The six detail documents were written in parallel and reconciled in a single
round. Five things changed *here*, and they are the spine's own business rather
than any detail document's:

1. **Requirement ownership is now a column** in the table at the top of this
   document, one document per requirement id, with the three moves that were
   needed to make it disjoint (R6, R17, R19). Before this, R6 was claimed by two
   documents and R17 by two, which made "which document is authoritative for
   this requirement" a question with two answers.
2. **`lop network uninstall [--purge]` is in the §6 CLI table.** It carries no
   requirement id: it is a lifecycle gap the design's own supervision creates
   (the relay is a supervised LaunchAgent, so the command that removes it belongs
   on the same surface as the command that installs it), with
   `lop mobile uninstall`'s semantics. Its purge scope was then narrowed on the
   manager's decision (§12): `--purge` covers the networks being uninstalled, and
   the device identity keypair needs `--purge-identity` plus a TTY confirmation
   that names every network still using that identity.
3. **`lop network invite` gains `--device`** in the §6 table and in A3, matching
   the binding that `mesh-transport-identity.md` §5.1 now specifies.
4. **§5.6's revocation property is stated in both directions**, including the
   sender-side withholding of the rotated secret and the bounded secret lifetime.
   The finding and its argument are `mesh-incident-response.md` §2.3.4 / §8 Q1;
   the enforcement is the transport document's.
5. **The detail-document table names every document by its real filename**, and
   now also lists `docs/design/mesh-prior-art.md` as a supporting, non-binding
   research note so a reader does not mistake it for a seventh design document
   with requirements of its own.

Two mechanisms were named in this round that a reader should not have to
rediscover: the **bounded secret lifetime** (`network.epoch_max_age_s`, the
half of R5 that covers the member that never returns) and a **blind relay as the
only sanctioned relayed path** (`mesh-transport-identity.md` §10.4: metadata-only
disclosure, per-link reservation and caps). Both are recorded in the documents
that own them and summarised here because they change what an operator can
assume.

---

## 12. Decisions taken by the manager

Three items were escalated out of convergence round 1 because they are policy
rather than design: each is a choice about blast radius, tolerable loss, or who
owns a requirement. They are recorded here so a reader does not have to
reconstruct a decision from the detail documents, and each is *also* written
where it is enforced, so a reader who starts in a detail document is not sent
here for the rule itself.

| Decision | Reason | Detail lives in |
|---|---|---|
| `lop network uninstall --purge` is scoped to the **network records, invites and outbox** of the networks being uninstalled. Deleting the **device identity keypair** needs its own flag, `--purge-identity`, **plus an interactive TTY confirmation naming every network still known to that identity**; without a TTY it is refused outright, with a message naming the flag that does work | The keypair is unrecoverable and is what every *other* network addresses this device by, so one flag covering both would let a single-network action destroy identity that networks the operator was not thinking about still depend on. One flag, one blast radius; the wider one needs a human — the rule `lop network trust` already applies to its own irreversible act | §6 (the row); `mesh-transport-identity.md` §2.2, §16 Q1, invariant `purge_identity_needs_a_named_tty_confirmation` |
| `network.epoch_max_age_s` ships at **30 days**, configurable | It is a **policy statement** about the operator's device-loss window, not a derived number: above any plausible offline window (a sleeping laptop is never forced to re-pair), below the 180-day prior-art ceiling for node credentials. When it fires, the network reconciles and the lowest-`device_id` active admin rotates — a session in flight is not disrupted, because a session is a separate runtime that does not hold the link, and a viewer only sees its existing reconnect path | `mesh-transport-identity.md` §4.3, §8.4, §11; prior art in `mesh-prior-art.md` §2 |
| **R6 is owned by `mesh-ui.md`**; `mesh-session-mobility.md` implements the federation the surface renders | R6 is a statement about what a front end shows — one list, a local/remote annotation, inside the desktop's budget — and §8 already pointed at `mesh-ui.md` for its detail. Mobility supplies the data path, not the surface, and now says so as "implements on the way" rather than claiming the requirement | §2.1 (R6), §8; `mesh-ui.md` §1.3, §2; `mesh-session-mobility.md` header and §3 |

**One consequence worth stating:** decisions taken here bind the *shape* of what
gets built, so a PR that implements any of the three implements it as written —
scoped purge, a 30-day age with an operator-visible remedy, and R6 satisfied on
the surfaces — rather than re-deciding it in code.
