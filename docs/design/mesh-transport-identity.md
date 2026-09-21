# Mesh transport, identity, pairing and membership

Status: **design, pre-implementation.** This is the foundational detail design
under `mesh-network.md` (the spine). It **owns R1–R5** and implements the
revocation mechanism that **R17** is built on — R17 itself is owned by
`mesh-incident-response.md`, which owns the incident state machine and its audit
trail. It fixes the decisions the rest of the mesh is built on:
**A1** (reuse the control vocabulary), **A2** (one relay per install, in its own
record namespace), **A3** (per-device key, per-network secret, human-confirmed
join).

| Requirement | Closed here |
|---|---|
| R1 relay per install, never a session owner | §2, §2.2 |
| R2 networks are named, plural, manageable | §4, §11 |
| R3 pairing is explicit, key-based, human-confirmed on both ends | §5 |
| R4 zero trust between peers | §6, §7, §8 |
| R5 membership revocable without a device visit | §8 |
| R17's revocation mechanism: deactivate/disconnect, tell every peer, refuse afterwards (the requirement is owned by `mesh-incident-response.md`) | §8.2, §8.5 |
| A1 reuse `ControlOp`/`EventOp`; do not invent a session protocol | §6.4, §7.4 |
| A2 one relay per install, new record namespace | §2.6, §9.1 |
| A3 device keypair + network secret + SAS | §3, §4, §5 |

Touched but owned elsewhere: R6/R8 (listing and `/new remote`, see
`mesh-ui.md` and `mesh-session-mobility.md`), R10–R12 (mobility),
R13–R16 (credentials), R18 (audit retention numbers), R19 (the guide),
R20–R22 (forward compatibility — this design reserves the seams, §12.4).

Everything below cites the code as it exists on `origin/main` (`a7e6b9bd`).
Where I say "today", I mean that commit.

**Notation used in every JSON example and frame below.** `"b64url(N)"` means an
`N`-byte value in unpadded base64url, written with the real value's length so a
coder knows what to generate; `"hex64"` is a 64-character lowercase hex SHA-256
digest; `"n_5f3c…"` / `"d_6c1f…"` / `"i_2f9c…"` elide the middle of a real-length
id (never a shorter id — the lengths are fixed by §3.1 and §4.1); `…` inside an
array means further entries of the same shape; all timestamps are Unix seconds as
floats. A field shown as `"b64url(N)"` is schema, not a literal.

---

## 0. The answer up front

1. **A relay is one supervised, foreground, per-install process** that owns
   device identity, network records, peer links, the audit log, and the
   credential-broker faces. It is **not a session owner**: it writes no
   transcript, holds no lease, and runs no turn. It reaches the session plane
   only by dialling a local runtime's loopback control socket, exactly as every
   other viewer does (§2).
2. **Transport is TCP plus a purpose-built, signature-authenticated handshake**
   built from primitives this repo already depends on (`cryptography>=42`,
   `pyproject.toml:76`): Ed25519 identity, ephemeral X25519 ECDH, HKDF-SHA256,
   AES-256-GCM records. No TLS (Python's stdlib `ssl` has no certificate
   verification callback, so mTLS would need a per-network CA whose private key
   every member holds), no new dependency, no new primitive (§6.1).
3. **Identity is an Ed25519 keypair per install** at
   `<config>/network/identity/device.json`, mode 0600 in a 0700 directory, with
   `device_id` *derived* from the public key. Not the keychain, not the secret
   store: the relay is launchd-supervised and must boot unattended (§3).
4. **A network is a JSON record plus a secret file** under
   `<config>/network/networks/`: id, name, epoch, members (each with
   `kind`/`lifecycle`/`role`/`capabilities`/`endpoints`), invites, trust state.
   The secret is *never* in the record file, so no surface has to redact it (§4).
5. **Pairing is five steps with a human at both keyboards**: invite minted
   (single-use, TTL'd, bound to the inviting device) → joiner redeems over a
   `join`-mode handshake proving possession of the invite → both sides show a
   6-digit SAS over an out-of-band code entry → inviter admits → link upgrades
   in place to a member link. A mismatch burns the invite and refuses the join
   (§5).
6. **The wire carries the existing vocabulary.** `hello`/`challenge`/`auth`/
   `welcome` is a *transport* handshake; after it, frames are the control-socket
   vocabulary (`ControlOp`/`EventOp` from `local_operator/mobile/types.py:279`
   and `:346`) inside AEAD records, with a small `net_*` family for
   network-scoped operations. `locality: "remote"` in the auth frame is the
   existing seam for this (`session/runtime/types.py:172`) (§6.4).
7. **One authorisation chokepoint**, `Authorizer.check`
   (`local_operator/network/authorize.py`), called from exactly one place in the
   relay's dispatch, with an `OP_CAPABILITY` table that a totality test pins.
   A missing entry is a refusal *and* a named test failure (§7).
8. **Revocation rotates the secret and bumps the epoch**, so a removed device
   is refused by every remaining member without a visit. The offline-member
   problem is solved by a `reconcile` phase that admits a still-a-member device
   holding only the previous epoch's secret, and refuses a removed one at the
   membership check (§8).
9. **No inbound reachability is assumed, ever.** A link exists when one side can
   dial the other's advertised endpoint; a dial-only install is a supported
   configuration. No UPnP, no hole punching, no STUN. The sanctioned WAN path is
   the Radient tunnel that already exists (§10.4).
10. **The relay publishes under a new `run/peers` namespace** (A2) and the
    federated listing is a `net_catalog` op fanned out to peers, cached with a
    TTL, and rendered as a `locality` field on the row — never inferred from an
    id shape (§9).

---

## 1. The problem as I found it

What exists today, verified:

- **The session plane stops at the machine boundary by construction.** A session
  is one detached process (`python -m local_operator.session.runtime.process`)
  publishing a pid-keyed record and listening on a loopback TCP JSON-lines
  control socket; the socket's listener binds `127.0.0.1` only
  (`session/runtime/viewer_server.py:236`) and the repo states loopback-only as
  the design's security invariant in three places (`session/runtime/server.py:1680`,
  `mobile/service.py:64`, `session/protocol.py:121-128`).
  `protocol.py:121-128` is explicit that the locality vocabulary has *no*
  cross-host member, and says why: "Every listener and every dialer in this tree
  binds or dials `127.0.0.1` only". Any mesh has to widen that, deliberately and
  in one place.
- **Authorization is a file mode and a same-account boundary.** A session's
  `control_key` lives in a 0600 record under a 0700 directory, and the control
  socket compares the first frame's `key` with `hmac.compare_digest`
  (`session/runtime/server.py:1885-1912`). The module docstring of
  `mobile/peer_client.py` states the consequence plainly: "Loopback + the key is
  the entire authorization story; there is no cross-account path to guard."
  Across hosts that story does not survive — a peer is not a same-account
  process — so the mesh needs a real authentication protocol, not a longer key.
- **Same-host peer plumbing already exists and is *not* reusable as-is.**
  `mobile/peer_send.py:429` (`deliver_peer_message`), `mobile/peer_client.py:104`
  (`send_peer_message`), the `peer_message` op (`mobile/types.py:320`) and the
  `secrets/peer.py` ancestry authorizer are all same-uid/same-host mechanisms:
  `secrets/peer.py:1-80` authenticates a caller by its *kernel-reported
  ancestry* (`LOCAL_PEERSOCK`, audit token, pidversion). That is a beautiful
  boundary on one machine and nothing at all across two.
- **Records are namespaced per record kind, and the rule is written down.**
  `session/runtime/types.py:239-280` defines `run/mobile` (sessions),
  `run/serve` (a `lop serve` daemon), `run/host` (boot records), each with a
  comment explaining why it is not one of the others — chiefly that every reader
  of `run/mobile` treats each file as a *session*, and `SessionRecord.kind` is a
  `Literal` those readers pass through unvalidated, so a foreign record surfaces
  as a phantom session rather than an error. `session/runtime/registry.py`
  already parameterises the whole read/write path (`run_dir` L100,
  `record_path` L109, `publish` L125, `_staged_write` L135, `scan` L464,
  `classify` L395), so a fourth namespace is a constant plus four thin wrappers
  (`server/registry.py:440-471` is the worked example for `run/serve`).
- **The vocabulary the mesh must carry already exists.** `ControlOp`
  (`mobile/types.py:279`) is the whole command surface a viewer needs — prompt,
  steer, abort, cancel, set_model, set_effort, slash, approval/ask answers,
  snapshot, stop, retire, variables — and `EventOp` (`:346`) is the response
  side (`welcome`, `projection`, `ack`, `error`, `event`, `frontend_*`). The
  envelope is literal and tiny: a request is `{"op": …, "req": <int>, …fields}`
  (`mobile/attach_client.py:1327`) and a reply is `{"op": "ack", "req": …,
  "detail": …}` or `{"op": "error", "req": …, "message": …}`. `dispatch` lives
  at `session/runtime/server.py:3295` with the op table at `:3300-3800`, and its
  unknown-op behaviour is the compatibility rule the mesh inherits: an old peer
  answers `error: unknown op` and the caller degrades.
- **Locality is already a declared field, reserved for exactly this.** The auth
  frame carries `locality: "local"|"remote"` (`session/runtime/server.py:2700`
  parses it; `:4359-4426` dispatches with it) and `ClientLocality`
  (`session/runtime/types.py:164-172`) documents the relay case:
  "A relay that forwards a remote device's commands is the case that must
  declare `remote`". The spine's A1 is that field cashed in.
- **Everything that makes remote sessions *hard* is already solved locally.**
  The detached-runtime split means a session survives its viewer; the
  `exclusive-move-v1` capability (`types.py:93`) and
  `RuntimeServer._exclusive_move_fence` (`server.py:1207`, `:2823-2864`) give
  the single-writer handover; `route_shared_slash` (`protocol.py:1120` →
  `attached.py:7399` → `serving.py:3624`, with `locality` at `:3630` and the
  viewer-routed exceptions at `:3834-3839`) already routes slash commands to a
  session's owner. None of that needs reinventing; it needs a *transport*.
- **Supervision has a house pattern.** `mobile/service.py` is foreground by
  design ("supervision belongs to launchd … never to a self-daemonizing
  double-fork"), binds loopback only, and is installed by
  `mobile/install.py` (`LABEL = "com.local-operator.mobile"` L35, `plist_path`
  L110, `render_plist` L118, `refresh_plist_if_stale` L157). `launchd.py:1-90`
  documents the two traps the mesh must inherit rather than rediscover: the
  **addressability guard** (a test that patches `HOME` must not restart the
  operator's real daemon) and the **bootout-then-bootstrap race** (measured:
  8 of 8 back-to-back bootstraps failed with EIO; ~500 ms is enough) — hence
  `launchd.reload_job` is the only copy.
- **Key-at-rest precedent.** The secret store keeps `master.key` mode 0600
  inside a 0700 directory (`secrets/keys.py:6,38,72`) and deliberately does not
  use the keychain; the only keychain use in the whole tree is the mobile portal
  password (`mobile/auth.py:13,36-37`, shelling out to `security`). Provider
  OAuth grants are `auth.db` rows with a *local* lease table
  (`providers/auth_store.py:116-147`) — 30 s, in SQLite, which cannot span hosts,
  which is exactly why credentials are brokered and not copied (A5).

**The gap, stated once.** There is no way for two installs to authenticate each
other, no shared vocabulary for what one may ask of the other, no place to keep
membership, and no process whose job is to hold those three things. That is what
this document specifies.

---

## 2. The relay

### 2.1 What it is, and what it is not

The relay is one process per install. It **owns**:

- the device identity key (§3);
- the network records and their secrets (§4);
- the peer listener and every peer link (§6);
- invite state and the pairing ceremonies (§5);
- the authorisation decision for every inbound frame (§7);
- the audit log's writer (§7.6);
- the credential-broker faces (A5; `mesh-credentials.md` owns their content);
- a loopback control surface the CLI, the TUI, the desktop daemon, and the
  agent tool dial (§2.5).

It **does not own**:

- any transcript. It never opens `sessions/<session_id>/transcript.jsonl` for
  writing, never holds a session lease (`session_lease.py`), never runs a turn,
  never constructs a `Session`;
- a session's control key. A peer never receives one, and the relay is the only
  party that ever dials a local control socket (spine §5.4);
- session liveness. Its knowledge of the local session plane is a **read-through
  cache** over `registry.scan()` plus dials, with a short TTL. A relay restart
  therefore loses nothing stateful, and killing the relay cannot kill a session.

Why this matters, concretely: if the relay were a session owner, (a) two writers
could exist for one transcript, which is the invariant the lease and the
`exclusive-move-v1` fence exist to protect; (b) `lop network stop` would be a
way to lose work, coupling an install-wide service to per-conversation state;
and (c) a relay crash would orphan every session it owned, on every network.

**Enforcement, not intention.** Two structural tests:

- `tests/unit/network/test_relay_is_not_an_owner.py` asserts that no module
  under `local_operator/network/` imports `session.runtime.serving`,
  `session.session`, or `session.session_factory`, and that nothing under it
  references `sessions/` as a write target.
- `tests/unit/network/test_one_chokepoint.py` asserts that the only call sites of
  the relay's control-socket dial helper are inside the authoriser-gated
  dispatch path (§7.2). This is the same shape as
  `tests/unit/session/test_viewer_protocol.py`, which pins a structural property
  rather than a behaviour.

### 2.2 Supervision — the `lop mobile serve` shape, inherited

`network/service.py` is the `mobile/service.py` analogue and copies its
discipline:

- **Foreground.** It serves until SIGTERM/SIGINT; it never double-forks, never
  daemonises itself, never writes a pidfile of its own. Supervision is launchd's
  job or a developer's terminal.
- **Its stderr is its log.** `StandardOutPath`/`StandardErrorPath` both point at
  `paths.log_dir()/network.log` (the same `log_dir()` the mobile daemon uses,
  `paths.py:121`), and it calls `configure_console_logging()` +
  `quiet_wire_clients()` explicitly for the reason `mobile/service.py:22-41`
  records: without the pin a dependency's `basicConfig` floods the file with one
  record per request.
- **The plist is rendered by one pure function** and refreshed the same way.
  `LABEL = "com.local-operator.network"`, `render_plist(cfg)`,
  `refresh_plist_if_stale()`, `install()`, `uninstall()`, `service_action()`,
  `status()`, `health()` — the exact surface of `mobile/install.py:110-398`.
  `KeepAlive: {"SuccessfulExit": false}` (crash restarts, a deliberate refusal
  does not flap), `RunAtLoad: true`, `ProcessType: "Interactive"` (a relay holds
  long-lived sockets and timers and must not be App-Napped — same reason as the
  mobile daemon), and `procname.launchd_job("local_operator.network.service",
  "--config", …)` so Activity Monitor shows a named role rather than `python3`.
- **Every refresh goes through `launchd.reload_job`**, never `kickstart -k`
  (`launchd.py:47-60`: measured, `kickstart -k` after a plist rewrite keeps
  running the previous argv), and every path is guarded by
  `launchd.is_own_plist` so a sandboxed run cannot restart the operator's relay.
- **No launchd on a platform without it.** `is_supported()` (as in
  `mobile/install.py:239`) degrades to "run it yourself, here is the command";
  nothing else changes.

`lop network start` = install-if-needed + bootstrap (idempotent by *content*,
via `launchd.rewrite_if_stale`), `lop network stop` = bootout, `lop network
restart` = `reload_job`, `lop network serve` = foreground, `lop network status`
= the health probe. `init` and `join` call `start` unless `--no-start`, because
"streamlined install/deploy/pair" means pairing must not require knowing that a
daemon exists.

**Flags this design adds beyond the spine's §6 table**, listed so the coder does
not have to infer them: `invite --hosts`, `join --host`, `join --verify`,
`join --sas-stdin` (test-only, refused outside `LOP_NETWORK_TEST_MODE=1`, §13.2),
`join --emit-sas` (the R20 automated-join shape), `serve --no-launchd`
(foreground without touching a plist), `uninstall [--purge] [--purge-identity]`
(§16 Q1), and
`--json` on every action as the spine requires. Nothing here changes an existing
command's surface: `sessions` and `exec` only gain `--peer`/`--all-peers`.

### 2.3 Module layout

```
local_operator/network/
  __init__.py         # vocabulary re-exports (mirrors session/runtime/__init__.py:1-38)
  types.py            # PeerRecord, NetworkRecord, MemberRecord, InviteRecord,
                      # MESH_PROTOCOL_VERSION, NET_OPS, link/wire dataclasses
  identity.py         # device key: load/mint/rotate, device_id derivation
  store.py            # network records + secrets + invites on disk (atomic, 0600)
  registry.py         # run/peers publish/heartbeat/scan/unpublish (thin wrappers)
  crypto.py           # transcript hash, key schedule, AEAD record codec
  handshake.py        # hello/challenge/auth/welcome state machine, both roles
  link.py             # PeerLink: reader/writer, keepalive, backpressure, reconnect
  server.py           # RelayServer: listener, link registry, _dispatch
  authorize.py        # Authorizer + OP_CAPABILITY + INNER_OP_CAPABILITY  <- the choke point
  membership.py       # add/remove/rotate/epoch/reconcile (the ONLY writer of records)
  pairing.py          # invite mint/redeem, SAS, admission
  catalog.py          # local scan + peer net_catalog fan-out + TTL cache
  audit.py            # append-only JSONL writer (retention: mesh-incident-response.md)
  control.py          # loopback control socket: server + client (CLI/TUI/desktop/agent)
  service.py          # `lop network serve` foreground runner + install/status/plist
  cli.py              # add_parser/main for the `lop network` group
```
Reserved for the sibling designs: `projection.py` (doc 2),
`credentials.py` (doc 3), `metering.py` (doc 5). **Do not create them here.**

Import discipline follows the repo's CLI-startup rule (`registry.py` and
`types.py` are stdlib-only, cited in `control.py:1-60`): `network/types.py`,
`network/registry.py` and the argument registration in `network/cli.py` import
nothing heavy, because `lop network status` and `--help` run on the CLI startup
path. `cryptography` is imported inside the functions that handshake. Argument
registration is wired from `cli.py` exactly like `tunnels.arguments.add_parser`
(`cli.py:459`) and `secrets.cli.add_parser` (`cli.py:466`), with
`parents=[parent_parser]` so position-independent globals keep working
(`cli.py:96-116`).

### 2.4 Startup, steady state, shutdown

Startup, in order (each step is idempotent and each failure is fatal and loud):

1. Load config; if no network record exists, log and stay idle (the relay is
   harmless when unused, which matters because `-n peers` is the regression
   topology).
2. Load or mint the device identity (§3). A read-only or unwritable
   `<config>/network/` is fatal with a sentence naming the path and mode.
3. Load every network record and secret. An unparsable record is quarantined
   (`<id>.json.corrupt`) and reported, never silently deleted — a membership
   list is the one file a mistake there must not quietly reinterpret.
4. Bind the peer listener (`network.listen_address`:`network.port`) and the
   loopback control socket. Publish the peers record (§2.6) and start the
   heartbeat loop (`HEARTBEAT_INTERVAL_S`, shared with the session runtime).
5. Start a link manager per member with a known endpoint: dial with backoff,
   accept inbound, dedupe (below).
6. Start the audit writer, then the catalog cache warmer.
7. SIGTERM: stop accepting, send `net_bye` on every link, close them, unpublish
   the record, flush the audit log. **Sessions are untouched** and keep running;
   a reconnecting peer sees them again on the next handshake.

Steady state: one reader task and one writer task per link; one dispatch of each
inbound frame; a 30 s keepalive; a 60 s catalog refresh; heartbeats to the peers
record.

### 2.5 The relay's local control surface

The CLI, TUI, desktop daemon and agent tool do **not** open peer links
themselves. They dial the relay's loopback control socket, which reuses the
session runtime's own conventions verbatim:

- bind `127.0.0.1`, ephemeral port, recorded in the peers record;
- first frame `{"key": "<control_key>", "client": "cli"}` compared with
  `hmac.compare_digest` against the record's `control_key`; anything else closes
  without a reply (`server.py:1885-1912` is the model, including the "an open
  port that answers wrong keys with errors is an oracle" reasoning);
- JSON-lines frames in the `{"op": …, "req": …}` / `{"op":"ack"|"error"}` shape;
- records are 0600 inside 0700, so the key's protection is the account.

Ops on this socket (the local op vocabulary — distinct from the peer ops of §6.4
and from the network-scoped surface):

```
net_status            -> install state, relay pid, networks, links, log paths
net_ls                -> networks this device is in (role, epoch, members, trust)
net_show {network}    -> members, roles, capabilities, endpoints, audit tail
net_init {name}       -> create a network
net_rename / net_rm   -> local modify / forget
net_invite {network, role, ttl_s, hosts?, device_id?}
                        -> writes the token to a 0600 file and returns
                        {invite_id, expires_at, path}  (never the token: §5.1;
                        device_id binds redemption to one device: §5.1)
net_join {path|token, host?} -> join (interactive SAS confirmation)
net_member_rm {network, device_id}
net_peer_ls           -> reachable peers, latency, session counts, reachability
net_disconnect {network}
net_panic {network}
net_trust {network, trust}   -> re-admit after a panic
net_log {network?, follow?, since?}
net_doctor {peer?}    -> DNS/reachability, handshake, epoch skew, clock skew
stream_open {peer, session_id} -> open a forwarded session stream (doc 2)
stream_send {stream, frame}    -> forward one ControlOp frame on that stream
stream_close {stream}
```
The last three are the seam `mesh-session-mobility.md` implements against, and a
stream's responses are `EventOp` frames so a viewer can reuse the existing attach
client (§12.2).

**`stream_open` puts that connection into a pass-through mode, and the mode is
required rather than convenient** (`mesh-session-mobility.md` R-IF-1). After a
stream is opened, the connection stops being a generic control socket: frames
written on it are validated only as frames of the opened stream and forwarded
verbatim, and the peer's reply frames are written back unmodified, so a client
can subclass `AttachClient` (`mobile/attach_client.py`) and see exactly what it
would see over a local attach — including the projection, the raw event stream,
history paging and the approval cards, with no per-op relay code. The mode is per
connection, ends with `stream_close` or the socket, and reaches exactly one
session on one device: the session-scope rule of §7.2 still holds, and a stream
frame naming another session is refused, never re-pointed. A frame written before
a **successful** `stream_open` is answered `unknown_op`/`protocol_error` and is
never forwarded; `stream_send` stays for the frame-at-a-time client and for the
tests. This is an interface requirement on `network/control.py`, not a new op:
it changes what the connection accepts after `stream_open` returns, not the
vocabulary. They are named `stream_*` rather than `net_*` on purpose: this is
a **local** vocabulary (viewer → relay, authorised by the control key of §2.5),
while `net_forward` (§6.4) is the **link** carrier (relay → peer, authorised by
the capability model of §7). Two surfaces, two names, so a reader can tell from
the frame alone which boundary it crossed.

**The listener is the one non-loopback listener in this tree.** That is a
deliberate, single exception to an invariant the repo states loudly and in three
places — `session/runtime/server.py:1680` ("loopback only is the security
invariant of the whole design"), `mobile/service.py:64` ("THE security invariant:
loopback only, always"), and the comment at `protocol.py:121-128`, which
enumerates every listener and dialer in the tree —
and it is why: (a) the peer listener authenticates completely before any op
dispatches (§7), (b) it never proxies a raw local control socket and never
transmits a control key, (c) it is bound by an explicit config key with a
dial-only mode, and (d) its auth failure path is silence plus a local audit
record — no reply frame, no error oracle. Every other listener in the mesh
(control socket, session control sockets, `run/serve`) keeps binding loopback.
A test asserts it: `tests/unit/network/test_listeners.py` walks
`local_operator/network/**` for `start_server`/`listen` calls and fails for any
address other than the configured one.

### 2.6 The relay's record: `run/peers` (A2)

New dirname constant beside the existing three, with the same kind of comment
that `SERVE_RUN_DIRNAME` (types.py:260) and `HOST_RUN_DIRNAME` (:280) carry —
why it is a fourth namespace and not a field on the session record:

```python
#: Directory (under the config root) holding one record per live mesh RELAY.
#:
#: A FOURTH namespace for the reason ``run/serve`` is a second one: every reader
#: of ``run/mobile`` treats each file there as a SESSION. A relay record carries
#: ``network_id``, ``device_id``, ``epoch`` and endpoints — facts a session
#: record must not carry and a session reader would misread (a relay's pid is
#: not a session's, and its control socket does not speak a session's frames).
#: It is also not a ``run/serve`` record: that answers "which install is serving
#: HTTP", this answers "which install is on the mesh, as whom, and on which
#: networks". One install can have either, both or neither.
PEERS_RUN_DIRNAME = "run/peers"
```

`local_operator/network/registry.py` mirrors `server/registry.py:440-471`
exactly: `record_path(pid, root)`, `publisher(record, root)`,
`publish(record, root)`, `unpublish(pid, root)`, `scan(root)` — each a one-line
delegation to the shared `session.runtime.registry` functions with
`PEERS_RUN_DIRNAME` and `PeerRecord.from_json`. That buys the atomic staged
write (`_staged_write`, 0600-and-rename), the heartbeat, the stale-record
reaping and `classify`'s live/wedged/stale verdict for free.

`network/types.py`:

```python
MESH_PROTOCOL_VERSION = 1          # the LINK protocol (§6.4) — not PROTOCOL_VERSION

@dataclass
class PeerRecord:
    pid: int                       # the relay process, and the filename
    kind: str = "relay"            # a constant, present so a reader that
                                   # globbed the wrong directory sees it
    protocol: int = MESH_PROTOCOL_VERSION
    session_protocol: int = PROTOCOL_VERSION   # what the local runtimes speak
    device_id: str = ""
    device_name: str = ""          # operator-set, cosmetic, never authority
    instance_id: str = ""          # per-process (§3.4)
    control_port: int = 0
    control_key: str = ""          # 0600 record = the loopback boundary
    listen: dict[str, Any] = field(default_factory=dict)   # {address, port, advertised: [..]}
    networks: list[dict[str, Any]] = field(default_factory=list)
    links: int = 0
    capabilities: list[str] = field(default_factory=list)  # link feature strings
    version: str = ""
    source_ref: str = ""
    install_root: str = ""
    started_at: float = field(default_factory=time.time)
    heartbeat_at: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]: ...          # asdict, like every record here
    @staticmethod
    def from_json(data: dict[str, Any]) -> "PeerRecord": ...   # drop unknown keys
```

Literal, at `<config>/run/peers/48213.json`, `0600`:

```json
{
  "pid": 48213,
  "kind": "relay",
  "protocol": 1,
  "session_protocol": 5,
  "device_id": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "device_name": "damian-mbp",
  "instance_id": "i_2f9c1d4e8a7b6c5d",
  "control_port": 51284,
  "control_key": "8Qk2…",
  "listen": {
    "address": "0.0.0.0",
    "port": 4097,
    "advertised": ["192.168.1.24:4097"]
  },
  "networks": [
    {"network_id": "n_5f3c1a2b4d5e6f708192a3b4", "name": "home-net",
     "epoch": 7, "role": "admin", "trust": "active", "members": 2, "links": 1}
  ],
  "links": 1,
  "capabilities": ["mesh-net-v1", "credential-broker-v1"],
  "version": "0.47.1",
  "source_ref": "a7e6b9bd",
  "install_root": "/Users/damian/.local/share/uv/tools/local-operator",
  "started_at": 1758300000.0,
  "heartbeat_at": 1758300030.0
}
```

**Two absences, stated as properties**: the peer record never contains the
device private key and never contains a network secret. `lop sessions` keeps
meaning "sessions" (it scans `run/mobile` only); a federated catalogue is the
relay's `net_ls`/`net_peer_ls` and §9's `net_catalog`.

Written by `RecordPublisher` (`registry.py:683`-equivalent) so a reader sees the
old file or the new one, never a torn one; a `kill -9`'d relay leaves exactly one
stale file that the next scan reaps (`_reap_dead_record`, `registry.py:629`).

---

## 3. Device identity

### 3.1 Algorithm and encoding

- **Long-term device key: Ed25519** (32-byte seed, 32-byte public key). One per
  install. It signs handshake transcripts (§6.2) and membership statements
  (§8); it never encrypts.
- **Per-link ephemeral: X25519**, generated fresh for each handshake and
  discarded at link close. Forward secrecy comes from here, which is why the
  long-term key can be Ed25519-only.
- **`device_id` is derived, never assigned**:

```
device_id = "d_" + sha256(b"lop-device-id-v1\x00" || ed25519_public_key_bytes).hexdigest()[:32]
```

  Self-certifying, per A3: a member list can be checked without a CA. The
  truncation is 128 bits, and the collision argument is explicit: an accidental
  collision is bounded by 2⁻⁶⁴, and an *adversarial* one buys only a name clash,
  because authorisation compares the full public key, and `membership.add`
  refuses a `device_id` that already exists with a different public key
  (`device_id_conflict`, audited). An id is a name, never authority.
- **`instance_id`**: 12 random bytes, base32, minted per relay *process* start
  (§3.4).
- **Names** (`device_name`, a network's `name`) are cosmetic and operator-set.
  Nothing authenticates on them; `lop network show` prints the id beside the
  name wherever a decision could be made.
- **Wire encoding for binary material**: unpadded base64url (`b64url`) for
  public keys, secrets, nonces, tags; unpadded Crockford base32 for human-typed
  things (invite ids when printed, the fingerprint). Never hex on the wire where
  size matters; never base64 where a human transcribes.

### 3.2 On disk, and why there

```
<config>/network/                     0700
<config>/network/identity/            0700
<config>/network/identity/device.json 0600
```

```json
{
  "schema": 1,
  "algorithm": "ed25519",
  "generation": 1,
  "device_id": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "private_key": "b64url(32)",
  "public_key": "b64url(32)",
  "created_at": 1758300000.0,
  "name": "damian-mbp",
  "rotated_from": null
}
```

Written staged and `os.replace`d (the `_staged_write` shape), so a crash never
leaves a half-written key. `<config>` is `paths.config_dir()`, which honours
`LOCAL_OPERATOR_CONFIG_DIR` (`paths.py:56-67`) — so a test or a sandbox gets a
whole identity of its own for free, which is also how the QA harness runs two
"devices" on one host (§13).

**Why not the keychain.** Three reasons, in order of weight:

1. **The relay must boot unattended.** The operator's requirement is that an
   install which has logged in keeps working across sessions and across a
   restart with no re-authentication. A launchd job that runs at load, or on a
   machine where nobody is logged into the GUI, cannot rely on an unlocked login
   keychain: `security find-generic-password` either fails or raises a UI prompt
   into a session that may not exist. This repo already measured that class of
   pain — `mobile/auth.py:36-37` is the *only* keychain caller in the tree, and
   `LOP_MOBILE_PASSWORD` exists precisely as the escape hatch for containers and
   foreground runs.
2. **The keychain answers a different question.** It protects a secret from
   other *user accounts* and from filesystem reads; here the threat model
   includes a hostile process *in* the same account, and the keychain does not
   help with that (it is unlocked for the session).
3. **Portability.** Linux and CI installs (the on-demand pod of R20) have no
   macOS keychain at all. A feature whose identity only exists on macOS would
   have to be redesigned for the pod.

**What this honestly is.** A 0600 file is the same boundary as `control_key` and
`master.key`: a process running as this account can read it. Per the repo's own
rule for the secret store ("Do not describe this store as a vault, in code
comments, docs, or to the user", AGENTS.md), this document does not claim
otherwise: the device key protects the *network* from other accounts and from
hosts that are not members; it does not protect the device from itself.

Rejected for the same slot: `lop secret` (`secrets/`) — its key can be
passphrase-wrapped (`keys.py:82`, `master.key.wrapped`), which would put a
prompt on the relay's boot path; and an SSH-agent-style delegation — a new
dependency and a new failure mode for no gain here.

### 3.3 Generation, rotation, loss, copy

- **Generation.** `generation` increments on rotation; `rotated_from` names the
  previous `device_id`. Rotation mints a new keypair and therefore a **new
  `device_id`** — an id is a key fingerprint, so it cannot survive a key change.
- **Continuity.** `lop network identity rotate` writes the new key and, for each
  network, produces a **rotation statement** signed by the *old* key:

```json
{"kind": "lop-device-rotate", "network_id": "n_…", "old_device_id": "d_…",
 "new_device_id": "d_…", "new_public_key": "b64url", "rotated_at": 1758300000.0}
```

  ...plus `sig_old` (Ed25519 by the old key over the canonical JSON with
  `sig_old` removed). `membership.apply_device_rotation` verifies `sig_old`,
  proves `old_device_id` is an active member (and that `old_device_id` is in
  fact `device_id(new_public_key_old)` — a member row is not a licence to
  re-identify as anybody), then rewrites the row in place: same capabilities,
  same `added_at`, `device_id` replaced. Both id forms are kept for a bounded
  window in a `previous_ids` list so an in-flight link at the old id is not cut
  mid-turn; links at the old id are refused after the window.
- **Offline rotation.** If no peer is reachable, the statement is queued in the
  per-peer outbox and delivered on reconnect. If it is never delivered and the
  old key is gone, the device is simply unknown to its peers and must re-pair —
  which is correct, since nothing can prove continuity without the old key.
- **Key loss.** Membership is lost; **data is not**. Transcripts are not
  encrypted with the device key, so everything on that disk still opens locally.
  Recovery path: the operator removes the stale member row on any remaining
  device (`lop network member rm`), then re-pairs the device with a fresh
  invite. `lop network doctor` names the condition (`identity_missing`) the first
  time the CLI runs.
- **Key copied to another machine.** The copy *is* the device: same id, same
  signature. This is a real limitation and is stated as one — at the crypto
  layer a copied key is indistinguishable. What the design does instead is make
  it *visible* and make it *remediable*:
  - **Duplicate-use detection (§3.4)**: the relay keeps one active link per
    `device_id`; a second link from the same id with a different `instance_id`
    evicts the first and writes `duplicate_identity` to the audit log. Three
    distinct instance ids within an hour flag the member `suspect: true`, which
    `lop network ls`/`show`, the TUI sidebar and the desktop `peers` payload all
    surface.
  - **Remedy**: `lop network member rm`, which takes effect without visiting the
    other device (§8). Rotation of the device key is the *wrong* remedy here —
    the attacker holds it.
  - Stated honestly: detection is behavioural and the flag is advisory. It denies
    nothing; the operator's removal is what denies.

### 3.4 `instance_id` and the duplicate-identity fence

- `instance_id` is minted at relay startup (12 random bytes) and carried in
  `hello`/`challenge` (§6.2), so it is bound into the transcript and cannot be
  swapped by an intermediary.
- The relay keeps `active[device_id] -> (link_id, instance_id, opened_at)`.
  A new link for a live `device_id`:
  - same `instance_id` → impossible from a correct peer (one process, one id):
    treat as a copy or a fork, evict the older, audit `duplicate_identity`;
  - different `instance_id` → a restart (normal) or a copy. Distinguish by a
    **grace window**: an eviction within `LINK_RESTART_GRACE_S = 5 s` of the
    previous link's last frame is treated as a restart (audit `link_replaced`,
    no flag); outside it, audit `duplicate_identity` and increment the member's
    `duplicate_count`.
- The eviction policy is "newest wins" in both cases, because refusing the new
  link would let a stale copy pin a device's slot and deny service.

---

## 4. Networks on disk

### 4.1 Layout

```
<config>/network/networks/<network_id>.json         0600   the record (§4.2)
<config>/network/networks/<network_id>.secrets.json 0600   the epoch secrets (§4.3)
<config>/network/outbox/<invite_id>.invite          0600   a minted token (§5.1)
<config>/network/catalog.json                       0600   the listing cache (§9.4)
<config>/network/audit.jsonl                        0600   the audit log (§7.6)
```

Key material is deliberately in a **separate file** from the record: the record
is what `lop network show --json` dumps, what the desktop `/v1/desktop/peers`
route returns, and what a future syncer would copy. Keeping secrets out of it
means there is no redaction step to forget — the same inversion the secret store
enforces ("No surface returns a value to the model", AGENTS.md).

`network_id` is minted randomly at creation (`"n_" + secrets.token_hex(12)`), is
not secret, is stable for the network's life, and is a safe filename.

### 4.2 The network record (literal)

```json
{
  "schema": 1,
  "network_id": "n_5f3c1a2b4d5e6f708192a3b4",
  "name": "home-net",
  "epoch": 7,
  "epoch_minted_at": 1758300000.0,
  "created_at": 1758300000.0,
  "created_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "sequence": 12,
  "trust": "active",
  "untrusted_reason": "",
  "self_device_id": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "self_role": "admin",
  "self_capabilities": ["list", "view", "prompt", "steer", "stop", "slash",
                        "delete", "move", "broker_credential", "admin", "trust"],
  "listen": {"address": "0.0.0.0", "port": 4097, "advertised": ["192.168.1.24:4097"]},
  "rotations": {"7": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b"},
  "members": [
    {
      "device_id": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
      "public_key": "b64url(32)",
      "name": "damian-mbp",
      "kind": "device",
      "lifecycle": "active",
      "role": "admin",
      "capabilities": ["list", "view", "prompt", "steer", "stop", "slash",
                        "delete", "move", "broker_credential", "admin", "trust"],
      "added_at": 1758300000.0,
      "added_by": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
      "added_via": "self",
      "endpoints": ["192.168.1.24:4097"],
      "last_seen_at": 1758301234.5,
      "last_seen_instance": "i_9a3f2c1d4e8a7b6c",
      "duplicate_count": 0,
      "suspect": false,
      "previous_ids": [],
      "removed_at": null,
      "removed_by": null
    }
  ],
  "pending": [
    {"invite_id": "b7k2m9qd3xzc", "device_id": "d_…", "public_key": "b64url(32)",
     "name": "damian-ec2", "role": "drive", "started_at": 1758300500.0}
  ],
  "invites": [
    {"invite_id": "b7k2m9qd3xzc", "minted_at": 1758300400.0, "ttl_s": 600,
     "role": "drive", "capabilities": ["list", "view", "prompt", "steer", "stop", "slash"],
     "hosts": ["192.168.1.24:4097"], "bound_device": null, "state": "consumed",
     "redeemed_by": "d_4b2a…", "redeemed_at": 1758300501.0, "outcome": "admitted"}
  ]
}
```

Field notes that matter for implementation:

- **`members` is an array, and a removed member is a tombstone**, not a deleted
  row: `removed_at`/`removed_by` set, `lifecycle: "expired"`, and the id can
  never be re-added (a later admission with that `device_id` is refused unless
  it comes with a rotation statement from the *removed* key — which cannot exist,
  because removal is what we do when we no longer trust that key). Tombstones
  make revocation auditable and make a re-pair a genuinely new identity.
- **`kind` (`device`|`pool`) and `lifecycle`
  (`active`|`provisioning`|`draining`|`expired`)** are the R20/R21 seams:
  nothing in the pairing path may assume a human is at the other end (§12.4).
- **`capabilities` is explicit on every member row**, even though it is derived
  from `role` at admission. Storing the resolved set means a later change to
  `ROLE_CAPABILITIES` cannot silently widen an existing member's authority — the
  thing a derived-at-read-time design gets wrong.
- **`rotations`** maps epoch → the `device_id` that initiated it, which is what
  makes concurrent rotations converge (§8.4).
- **`sequence`** increments on every write and is carried in epoch broadcasts, so
  a receiver can tell "I already have this" from "this is newer"; it is *not* a
  Lamport clock for the member list (see §8.4's deterministic rule).
- **`bound_device`** is `null` for a device-agnostic invite, or the one
  `device_id` that may redeem it (§5.1). It is written at mint and never changed;
  `redeemed_by` records what actually happened, after the fact, which is exactly
  the pair of facts a forensic read needs after a compromise.
- **`trust`** is `active` | `untrusted`; `untrusted` refuses all peer traffic
  for that network until the operator re-admits (§8.5).
- **`epoch_minted_at`** is the minting device's clock when the current epoch was
  minted, and it is the age input for §8.4's bounded secret lifetime. It is
  duplicated into the secrets file beside the key, so an install that has the key
  also has its age, and it rides `net_epoch` / `net_reconcile` so a receiver
  learns it with the secret.

### 4.3 The secret, and what an epoch is

```
<network_id>.secrets.json
{
  "schema": 1,
  "network_id": "n_…",
  "current":  {"epoch": 7, "secret": "b64url(32)", "minted_at": 1758300000.0},
  "previous": {"epoch": 6, "secret": "b64url(32)", "minted_at": 1758213600.0}
}
```

- The **network secret** is 32 bytes from `secrets.token_bytes` at `init`, per
  A3. It is what an invite transfers out of band, and what a join proves
  possession of. It is never in the record file, never in the peers record,
  never logged, never returned by a surface that an agent reads (§5.1).
- **Two epochs are retained, never more.** `current` is used to authenticate
  member links; `previous` is accepted *only* for a `reconcile`-phase handshake
  (§8.3). A third is dropped on rotation. This bounds what a stolen old secret
  is worth to one rotation generation.
- **Epoch keys are derived, not stored**: `epoch_key(secret, network_id, epoch) =
  HKDF-SHA256(ikm=secret, salt=sha256(network_id), info=b"lop-mesh-epoch-v1\x00"
  || str(epoch).encode(), length=32)`. Deriving means a rotation only ever has to
  distribute one 32-byte value, and an epoch number that disagrees with the
  secret is visible as a MAC failure rather than as a silent identity confusion.
- Rotation mints a fresh random `secret` (it does not ratchet). Deliberate: a
  ratchet buys forward secrecy against an attacker who already holds a device
  key, and that attacker can read the disk anyway; a fresh random secret is
  simpler to reason about, is what an operator expects from "rotate", and cannot
  be predicted from a leaked older one.
- **A secret has a maximum age, and that age is part of the schema, not a policy
  bolted on later.** `minted_at` is stored beside the key and the epoch must be
  re-minted once it is older than `network.epoch_max_age_s` (default 30 days,
  §11), even when every member is alive and reachable — the mechanism and its
  consequences are §8.4. It is recorded here because putting the age *next to the
  key* is what makes it enforceable by every reader of the key, and adding a
  field to a 0600 secrets file after installs exist is the kind of migration this
  document exists to avoid. Prior art treats this as table stakes rather than a
  nicety: Nebula's 1-year certificates, Tailscale's 180-day node keys, and
  OpenZiti's 24-hour enrolment token are all bounded lifetimes, and every one of
  those systems pairs them with exactly our handshake-step-6 membership
  re-evaluation (`mesh-prior-art.md` §2).

---

## 5. Pairing

### 5.1 Invite mint (on the inviter, offline)

`lop network invite --role drive --expires 10m [--hosts h1,h2] [--network n]
[--device <device_id>]`

The envelope is a single line, pasteable, and self-contained:

```
lop1.<b64url(payload)>.<b64url(tag)>
```

`payload` is canonical JSON (sorted keys, no whitespace, UTF-8):

```json
{
  "v": 1,
  "kind": "lop-invite",
  "network_id": "n_5f3c1a2b4d5e6f708192a3b4",
  "network_name": "home-net",
  "epoch": 7,
  "secret": "b64url(32)",
  "inviter_device_id": "d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
  "inviter_name": "damian-mbp",
  "invite_id": "b7k2m9qd3xzc",
  "bound_device": null,
  "issued_at": 1758300400.0,
  "ttl_s": 600,
  "role": "drive",
  "capabilities": ["list", "view", "prompt", "steer", "stop", "slash"],
  "hosts": ["192.168.1.24:4097"]
}
```

and `tag = HMAC-SHA256(invite_key, b"lop-invite-v1\x00" || payload_bytes)` where

```
invite_key = HKDF-SHA256(ikm=secret, salt=network_id.encode(),
                         info=b"lop-mesh-invite-v1\x00" + invite_id.encode(), 32)
```

Note the `ttl_s` **duration**, not an absolute `expires_at`. That is deliberate
and it removes cross-host clock skew from pairing entirely: the *inviter*
enforces freshness against its own clock (it minted the token, so there is no
skew to speak of), and the joiner enforces nothing. `lop network doctor` reports
clock skew as a diagnostic, never as an authority. `expires_at` is still shown to
the human as a local convenience, computed locally.

**Device binding (`--device <device_id>`, optional, and recommended after any
compromise).** The envelope is device-agnostic by default: whoever holds the
token may redeem it from any device. That makes an open invite a **bearer token
for the network's key material** — the token carries `secret`, and the record
learns who redeemed it (`redeemed_by`) only *after* the fact, which is too late
to be a control. With `--device d_…` the mint writes `bound_device`, and §5.2's
validation refuses any other device with `pairing_refused` /
`cause: "wrong_device"`. The operator's intent after a compromise is always
"*this* device comes back", so the bound form is what the incident path should
mint (`mesh-incident-response.md` §3.2/§8 Q3 raised it; this is where it is
enforced). A bound invite redeemed by the wrong device is **consumed**, not left
standing: a token that reached a second device is the leak the binding exists to
contain, and the cost of burning it is one more `lop network invite`.

**Single-use** is enforced by `invite_id` in the record's `invites[]` with
`state ∈ {minted, redeemed, consumed}`: `redeemed` is written the instant a valid
redemption arrives (before any human sees anything), `consumed` when the pairing
ends in *either* outcome (admitted, aborted, SAS mismatch, timeout). A second
redemption against `redeemed` → `invite_in_use`; against `consumed` →
`invite_already_used`. Entries older than 24 h are pruned. The state is on disk,
so a relay restart mid-pairing cannot be used to replay an invite.

**Token handling (this is a credential transfer, and it is treated like one).**

- `lop network invite` writes the token to
  `<config>/network/outbox/<invite_id>.invite` (0600) and prints **the path**, the
  network name, the role, the expiry and the hosts. It does *not* print the
  token: a token in stdout is a token in the agent's transcript, and the
  transcript is replayed to the provider on every later turn (the redaction
  rule, AGENTS.md).
- `--print` prints it to stdout and is **refused when stdout is not a TTY**, and
  refused together with `--json`. `--json` returns
  `{"invite_id", "path", "expires_at", "role", "hosts", "expires_in_s"}` — never
  the token.
- `lop network join` accepts a token inline, `@<path>`, or no argument (it reads
  the newest file in the outbox directory).

**No flag anywhere accepts the SAS.** §5.3 is why.

### 5.2 The exchange

Roles: **A** = inviter/joiner's peer (listener), **B** = joiner (dialer).
`T` is the transcript defined in §6.2. All frames are JSON-lines, one per line,
≤ 16 KiB, each with a `network.handshake_timeout_s` (10 s) deadline. Before the
`auth` frame validates, a failure closes the socket **without a reply** and
writes a local audit record.

**Step 0 — dial.** B connects to `hosts[0]` (or `--host`), falling through the
list on failure with the per-attempt reason reported (`connect_timeout`,
`connection_refused`, `unreachable`). If every host fails, B reports each and
stops; it does not trust anything it received from the token beyond the host
list.

**Step 1 — hello (B → A).** `mode: "join"`, and `join` carries B's assertion of
its own device key:

```json
{"net":"hello","v":1,"protocol":5,"mode":"join",
 "network_id":"n_5f3c1a2b4d5e6f708192a3b4","epoch":7,
 "device_id":"d_4b2a1c3d5e6f70819a2b3c4d5e6f7081",
 "instance_id":"i_2f9c1d4e8a7b6c5d",
 "eph":"b64url(32 X25519 pk)","nonce":"b64url(32)",
 "join":{"invite_id":"b7k2m9qd3xzc",
         "joiner_public_key":"b64url(32 Ed25519 pk)",
         "joiner_name":"damian-ec2"},
 "caps":["mesh-net-v1"],"build":{"version":"0.47.1","source_ref":"a7e6b9bd"}}
```

**Step 2 — challenge (A → B).**

```json
{"net":"challenge","v":1,"epoch":7,
 "device_id":"d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
 "instance_id":"i_9a3f2c1d4e8a7b6c",
 "eph":"b64url(32 X25519 pk)","nonce":"b64url(32)","salt":"b64url(16)",
 "caps":["mesh-net-v1"],"build":{"version":"0.47.1","source_ref":"a7e6b9bd"}}
```

A has already validated, before sending this frame: the network exists; `epoch`
equals the network's current epoch (else `invite_epoch_stale` — a rotation during
the invite's life invalidates it, and re-minting is one command); the invite
record exists, is `minted`, is fresh by `ttl_s`, and its `inviter_device_id` is
A (a token stolen from A cannot be redeemed against a different, more permissive
member); when the record's `bound_device` is set, the `device_id` this same
`hello` asserts equals it — otherwise `pairing_refused` with
`cause: "wrong_device"` and the invite is consumed (never the `tag` alone, which
proves possession of the token, not identity of the holder); and the `tag`
verifies. On any failure: close, audit `pairing_refused` with the reason, no
reply.

**Step 3 — auth (B → A).** The transcript, signed by the joiner's asserted key
and MAC'd with the *invite* key (not the epoch key — B's only credential at this
point is the token):

```json
{"net":"auth","v":1,"mode":"join",
 "device_id":"d_4b2a1c3d5e6f70819a2b3c4d5e6f7081","epoch":7,
 "sig":"b64url(64)","mac":"b64url(32)"}
```

with

```
sig = Ed25519(joiner_sk, b"lop-mesh-auth-v1\x00" || sha256(T))
mac = HMAC-SHA256(invite_key, b"lop-mesh-mac-v1\x00" || sha256(T))
```

T binds the role byte, the full `hello`, the full `challenge` and the `auth`
frame minus `sig`/`mac` (§6.2), so the signature commits to both ephemerals, both
ids, both instance ids, the invite id, the joiner's public key, and the
negotiated capabilities. Nothing in it can be altered by an intermediary.

**Step 4 — welcome (A → B).**

```json
{"net":"welcome","v":1,"phase":"pair",
 "device_id":"d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b","instance_id":"i_9a3f2c1d4e8a7b6c",
 "epoch":7,
 "inviter":{"device_id":"d_6c1f…","name":"damian-mbp"},
 "network":{"network_id":"n_…","name":"home-net"},
 "grant":{"role":"drive","capabilities":["list","view","prompt","steer","stop","slash"]}}
```

**`welcome` is the last plaintext frame.** Every frame from here on — including
the pairing frames below — rides inside an AEAD record (§6.3), because the
transcript is complete once `auth` is verified and both sides can derive the
link keys.

**The SAS is derived by each side, never transmitted.** It is deliberately
*absent* from `welcome`: a peer that sent its value would let an on-path
attacker echo it, and the human's comparison would then be a round trip rather
than a check. Each side computes its own from its own transcript and displays
*that*, which is what makes the transcribed code meaningful.

**Step 5 — the human step, in both directions.**

- **B displays** `damian-mbp (home-net) offers role drive — code 481 926` and
  prompts `type the code shown there:`. B's human types the six digits **from
  B's own screen**. This is a *transcription*, not a yes/no, and both halves
  matter: a yes/no lets two people each press `y` without comparing anything, and
  a value received from the peer would let a relay in the middle echo the
  inviter's own digits back at it.
- B sends `{"op":"net_pair_ready","req":1,"sas":"481926"}` — the *typed* value.
- **A displays** `d_4b2a… ("damian-ec2", new device) transcribed 481 926 and
  asks for role drive. Does YOUR screen show 481 926? [y/N]` — note "yours".
  A's human compares the transcribed value against **A's own derived code**. The
  two agree only if both derivations agreed, which is the whole check.
- **Mismatch, either side:** B typed something that disagrees with A's value →
  A refuses; A's human answers `n` → A refuses. Either way A sends
  `{"op":"net_pair_abort","reason":"sas_mismatch"}` and closes, and B prints
  `the codes did not match — the other device did not admit this machine. Do not
  retry: ask for a new invite.` The invite is marked `consumed` in both cases, so
  the attacker's next attempt needs a fresh invite, i.e. another human action.
- **B declines locally** (its human rejects): B sends
  `{"op":"net_pair_abort","reason":"declined_local"}` and closes; A's prompt is
  cancelled with `the other device declined`.
- **Timeout:** A's prompt is bounded by `ttl_s` from the invite and by
  `PAIR_CONFIRM_TIMEOUT_S = 180`; when it fires A sends
  `{"op":"net_pair_abort","reason":"timeout"}` and consumes the invite.

**Step 6 — admission (A → B), and the upgrade.**

```json
{"op":"net_pair_result","req":2,"admit":true,
 "network":{"network_id":"n_…","name":"home-net","epoch":7,"sequence":13,
            "trust":"active"},
 "member":{"device_id":"d_4b2a…","name":"damian-ec2","kind":"device",
           "lifecycle":"active","role":"drive","capabilities":[…],
           "added_at":1758300501.0,"added_by":"d_6c1f…"},
 "members_digest":"hex64(sha256(canonical(members)))",
 "secret":"b64url(32 current)","rotations":{"7":"d_6c1f…"}}
```

A writes the member row **before** sending this frame (a member A has admitted is
durable even if the link dies in the next millisecond), sets the invite to
`consumed`, and audits `member_admitted`. The link's `phase` flips to `member`
in place — **no re-handshake**:

- the SAS already proved the channel end to end with both humans at the
  keyboard *at that instant*, which is strictly stronger evidence than a fresh
  handshake against the same key material;
- the secrets and epoch did not change, so a re-handshake would prove nothing
  new;
- tearing down and redialling would create a window in which B holds the secret
  but is not yet a member, which is a strictly worse state to be in.

B persists the record and secret atomically, then answers
`{"op":"ack","req":2,"detail":"joined"}`. B's `lop network join` prints the
network, its epoch, its member count, and the operator's next command.

### 5.3 The SAS, and what it is worth

```
Z   = X25519(my_eph_sk, peer_eph_pk)
th  = sha256(T)
okm = HKDF-SHA256(ikm=Z, salt=sha256(b"lop-mesh-salt-v1\x00" || th), info=b"lop-mesh-sas-v1", length=8)
sas = f"{int.from_bytes(okm) % 1_000_000:06d}"        # displayed as "481 926"
```

Both sides compute it from the shared secret and the transcript; neither sends
its own value as the *source* of truth (B transcribes, A compares).

**Against an attacker without the invite token.** It cannot compute `mac`, so
its `auth` frame is refused before a SAS exists. The SAS is not what stops it.

**The SAS is derived on both sides and never crosses the wire, in either
direction.** It is absent from every frame (the frame list of §5.2 is the
authority, and a test in §13.1 asserts it). That is what makes the human step a
*check* rather than a round trip: a peer that transmitted its value would let an
attacker on the path echo the inviter's own digits back at it.

**Against an attacker holding a leaked invite token** (the token is a bearer
credential, so this is the case that matters). It can complete two handshakes,
one with A and one with B, and to survive it must make A's SAS equal B's. Both
are `HKDF(Z, salt = sha256(T))`, and the attacker knows both Zs — but A's
transcript is fixed by A's ephemeral and the attacker's ephemeral toward A
*before* the comparison, so the only lever it has is its ephemeral toward B (or
its asserted device key), which it must grind against a 20-bit target. Roughly
2²⁰ hash evaluations, feasible in about a second on this hardware. So the honest
statement is:

- the six-digit SAS is a **detector worth ~20 bits per human interaction**, in
  the same class as Bluetooth's numeric comparison. Its strength comes from
  everything around the digits: a successful grind still needs *two* humans to
  confirm, a failed comparison burns the invite, and a fresh invite is a fresh
  human action on the inviter. It is not a proof;
- because it is not a proof, the CLI **also prints the full transcript
  fingerprint** — the leading 20 bytes of `sha256(T)` as 32 Crockford base32
  characters in 8 groups of 4 (`K7QM-3XPD-…-9T2B`, 160 bits) — in the same panel
  as the digits, and `lop network join --verify` makes the *fingerprint* the
  value that must be compared (the digits are then printed for logging only).
  The fingerprint is worth 2¹⁶⁰ and costs one paste. Recommended for any pairing that
  is not on a private LAN, and mandatory when `--host` points at a public
  address.

**No flag accepts the SAS or the fingerprint.** They are entered at a prompt. An
agent cannot complete a pairing, which is the required property (R3: "requires a
human on *both* devices"); the agent-facing tool stops at "run this and have the
user type the code" (§12.5).

### 5.4 Pairing state machine

```
mint ──► minted ──redeem──► redeemed ──ready──► awaiting_human ──admit──► admitted
            │                   │                    │  │                  │
            │                   │                    │  └──decline/timeout──┤
            └──ttl elapsed──────┴────────────────────┴──────────────────────┴──► consumed
                                                                                (terminal)
```
Every terminal state consumes the invite. `consumed` is written *before* the
frame that announces it, on both sides, so a crash between the two cannot leave
a replayable token. A `bound_device` mismatch (§5.1) is also terminal: the invite
is consumed and audited `pairing_refused {cause: "wrong_device"}`, so the state
machine has no path that leaves a bound token redeemable by a second device.

---

## 6. The wire protocol

### 6.1 Transport: TCP + a purpose-built authenticated handshake

**Decision.** Plain TCP (with `TCP_NODELAY`), a JSON-lines transport handshake
(§6.2), then length-prefixed AEAD records (§6.3). Primitives: Ed25519, X25519,
HKDF-SHA256, AES-256-GCM — all from `cryptography`, already a base dependency
(`pyproject.toml:76`, used by `secrets/crypto.py:27-35` and `secrets/keys.py`).

Two sentences of justification: Python's stdlib `ssl` offers no certificate
verification callback, so mutual TLS with per-device certificates is not
expressible without either a per-network CA (whose private key every member would
hold, making it incapable of distinguishing devices) or `CERT_NONE` (encryption
without authentication); and we must implement framing, multiplexing, keepalive,
reconnect and backpressure ourselves in any case, since the payload is a
deadline-bound, coalescable event stream rather than a byte stream.

**Rules that make hand-rolling acceptable.** We implement **no cryptographic
primitive**. The construction is the standard signature-authenticated ephemeral
Diffie-Hellman (TLS 1.3's `CertificateVerify` shape) with domain-separated HKDF
and a counter-based nonce, and §6.2/§6.3 pin every byte so an implementer cannot
improvise. Every failure is fatal to the link (§6.6): there is no
resynchronisation, no "recover and continue", and no partial-trust phase.

Rejected alternatives are listed in §14.

### 6.2 Handshake

Frames, role letters and the transcript are exactly as §5.2 uses them; member
mode differs only in which key verifies the MAC:

**Member mode.**

1. `C→S` `hello` — `mode: "member"`, no `join` block.
2. `S→C` `challenge`.
3. `C→S` `auth` — `mode: "member"`; `sig` verified against the **member row's
   stored public key** for the claimed `device_id`; `mac` verified against
   `epoch_key(current)` first, then `epoch_key(previous)`.
4. `S→C` `welcome`:

```json
{"net":"welcome","v":1,"phase":"member",
 "device_id":"d_6c1f…","instance_id":"i_9a3f…","epoch":7,
 "capabilities":["list","view","prompt","steer","stop","slash"],
 "members_digest":"hex64","session_protocol":5,
 "nets":[{"network_id":"n_…","name":"home-net","epoch":7,"sequence":13,"trust":"active"}]}
```

...or `"phase": "reconcile"` when the MAC matched only the previous epoch, in
which case only `net_reconcile` is dispatchable (§8.3).

**The transcript.**

```
T = b"lop-mesh-v1\x00"
  || lp(role)          # one byte, b"D" (dialer) or b"L" (listener)
  || lp(hello_jcs)     # full hello frame
  || lp(challenge_jcs) # full challenge frame
  || lp(auth_core_jcs) # the auth frame with "sig" and "mac" removed
lp(x)  = len(x).to_bytes(4, "big") + x
jcs(o) = json.dumps(o, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
sig    = Ed25519(sk, b"lop-mesh-auth-v1\x00" || sha256(T))
mac    = HMAC(epoch_key_or_invite_key, b"lop-mesh-mac-v1\x00" || sha256(T))
```

The role byte and the direction-separated key schedule (§6.3) together make
**reflection** impossible: an attacker that echoes A's `hello` back to A produces
a transcript whose role byte and nonces are A's own, and the schedule's
`k_l2d`/`k_d2l` split means neither side can decrypt its own output.

**Verification order on the listener** (each failure closes silently, audits
`handshake_refused` with a reason code, and never distinguishes the reason on the
wire):

1. `net` frame shape and `v` known; `MAX_HANDSHAKE_LINE` respected;
2. `network_id` is one this install is in;
3. `trust == "active"` (an untrusted network refuses every link, §8.5);
4. `epoch ∈ {current, current-1}`;
5. `device_id ∉ {self}` (`self_link` is refused and audited — an accidental
   self-connection is otherwise a confusing silent no-op);
6. member row exists and `removed_at is None`, **checked against the current
   epoch's member list** (this is the line that makes revocation real);
7. `sig` verifies against the stored public key;
8. `mac` verifies against `current`, else `previous` (→ reconcile phase);
9. dedupe, duplicate-identity fence (§3.4).

### 6.3 Records after the handshake

```
Z    = X25519(my_eph_sk, peer_eph_pk)
th   = sha256(T)
salt = sha256(b"lop-mesh-salt-v1\x00" || th || Z)
okm  = HKDF-SHA256(ikm=Z, salt=salt, info=b"lop-mesh-link-v1", length=72)
k_d2l, k_l2d = okm[0:32], okm[32:64]      # per-direction keys
iv_d,  iv_l  = okm[64:68], okm[68:72]     # per-direction 4-byte nonce prefixes
```

Record on the wire, both directions:

```
uint32_be(len(ct)) || ct(plaintext||GCM tag)
```
- `plaintext` = one UTF-8 JSON object, no trailing newline (the record's length is
  the frame boundary — this is why post-handshake frames are not JSON-lines).
- `nonce` is **derived, never transmitted**: `nonce = iv_x || uint64_be(seq)`,
  `seq` starting at 0 per direction and incrementing per record.
- `AAD = b"lop-mesh-rec-v1\x00" || link_id(16 random bytes, minted at handshake,
  never on the wire) || direction_byte || uint64_be(seq)`.
- `MAX_RECORD_BYTES = 8 MiB`. A length prefix larger than that closes the link
  *before* allocating.
- Any `InvalidTag`, any AAD mismatch, any impossible `seq` (a gap, or past
  `2**40`) is fatal: close and reconnect. No resync, ever.

Rejected here: random per-record nonces (a counter is strictly stronger for a
single link and needs no entropy at frame rate) and long-term-static Diffie-Hellman
without an ephemeral (no forward secrecy — see §14).

### 6.4 Frames, the op vocabulary, and version negotiation

After `welcome`, the envelope is the control socket's, unchanged:

```
request : {"op": "<name>", "req": <int>, "locality": "remote", ...fields}
reply   : {"op": "ack",   "req": <int>, "detail": <str|object>}
error   : {"op": "error", "req": <int>, "message": "<sentence>"}
event   : {"op": "<EventOp>", ...fields}
```

**A1 is honoured by construction**: session-plane work uses `ControlOp` values
and `EventOp` values verbatim (`local_operator/mobile/types.py:279`, `:346`).
Where `locality` lives is worth being exact about, because there are two places
and they answer two different questions:

- **Per link**: the auth frame carries `locality: "remote"` — the field
  `ClientLocality` (`session/runtime/types.py:164-172`) was added for, with the
  docstring naming exactly this case. Both the peer link's `auth` and the relay's
  own dial to a *local* runtime declare it, so the runtime's own routing
  (`session/runtime/serving.py:4087-4129`, `run_slash_authoritative`) sees the
  truth about who is asking. (Measured on this branch; the recon notes' `:3624`
  is ~450 lines stale — `mesh-session-mobility.md` §0 records the same drift.)
- **Per frame**: each forwarded frame also carries `locality: "remote"`, because
  spine §5.8 requires every command to declare it and §7.2 refuses any frame
  claiming `local` over a peer link. The field is additive on frames the
  receiver already parses leniently (`mobile/types.py`'s
  `validate_control_frame` checks the fields it knows and ignores the rest), so
  this costs no version bump — and a frame that omits it over a link is treated
  as `remote` by position, never as `local` by default. The relay's *session-side* dial
to a local runtime then declares `locality: "remote"` in *that* auth frame too,
because `session/runtime/types.py:164-172` says the relay forwarding a remote
device's commands is the case that must: the runtime's own authorisation
decisions (slash routing, `session/runtime/serving.py:4087-4129`) then see the
truth.

New ops introduced **by this document** (the transport and network scope; the
session-plane handlers belong to doc 2 and are listed for completeness):

| op | direction | capability | purpose |
|---|---|---|---|
| `net_reconcile` | either | `list` | learn the current epoch+secret+member list when authenticated at the previous epoch (phase `reconcile` only) |
| `net_catalog` | either | `list` | this peer's session rows (§9.2) |
| `net_member_list` | either | `list` | member rows of one network (no key material) |
| `net_epoch` | either | `admin` | a rotation: new epoch, new secret, member list |
| `net_leave` | either | `list` | a peer announcing its own departure (self-signed) |
| `net_panic` | either | `list` | incident broadcast; receivers go untrusted (§8.5) |
| `net_trust` | either | `trust` | re-admit a network after a panic |
| `net_identity_rotate` | either | `admin` | a device rotation statement (§3.3) |
| `net_forward` | both | the inner op's | carry one `ControlOp` frame for a session on this device |
| `net_sync` | both | `view` | **reserved** for R22's cadence / pre-spin-down sync (§12.4) |
| `net_broker` | both | `broker_credential` | credential brokering (doc 3) |
| `net_session_lifecycle` | both | `delete` | archive/delete/restore on the peer (doc 2; PR #1328's local design) |
| `net_session_move` | both | `move` | move/fork across the link (doc 2) |
| `net_session_create` | both | `prompt` | create a session **on** the peer (R8): mint the id, claim the directory, stamp `mesh.json`, engage, admit an optional first prompt (doc 2) |
| `net_session_engage` | both | `view` | make an owner exist on the peer (warm a cold session so it can be viewed or acted on); carries no prompt (doc 2) |
| `net_session_stop` | both | `stop` | run the **peer's own** kill-switch ladder for one of its sessions (doc 2) |
| `net_bye` | both | — | graceful link teardown before close |
| `ping` | both | `list` | **reused** `ControlOp`; the keepalive |

`net_forward` is a carrier, not an authorisation bypass: the receiver maps the
**inner** frame's `op` through `INNER_OP_CAPABILITY` and requires *that*
capability (§7.2). A frame whose inner op has no entry is refused
(`unknown_op`) — never forwarded on the grounds that the outer op is known.

**Version negotiation, and the two versions.**

- `MESH_PROTOCOL_VERSION = 1` (the link). Carried in `hello`/`challenge`/`welcome`
  as `v`. A peer whose `v` this build does not know is refused at handshake with
  `error: protocol_mismatch {"mine":1,"theirs":2}` **after** authentication, and
  a local audit record naming both numbers. Fail closed, name the skew: the same
  discipline the session record already uses to report a build skew rather than
  fail silently.
- `PROTOCOL_VERSION = 5` (`session/runtime/types.py:66`, the session control
  protocol) is carried as `session_protocol` in `hello`/`welcome` and in the peer
  record, and is **passed through untouched**: the relay never interprets session
  frames, so a mixed-version fleet degrades per op via the existing unknown-op
  rule (`server.py` answers `error: unknown op`, and `mobile/attach_client.py`'s
  callers already tolerate it) rather than at the link. Bumping the *link*
  version for a session-level addition would be the mistake `types.py:108`
  records: "Negotiated by capability string rather than a `PROTOCOL_VERSION`
  bump".
- **New link features are capability strings**, advertised as `caps` in
  `hello`/`welcome`: `mesh-net-v1`, `session-mobility-v1`,
  `credential-broker-v1`, `compute-pool-v1`. Unknown caps are ignored, absent
  caps mean "old peer", and a feature is used only when both sides advertise it.
  This mirrors `SessionRecord.capabilities` (exclusive-move is negotiated exactly
  this way, `types.py:93`).

### 6.5 Keepalive, reconnection, dedupe

- **Keepalive**: `ping` every `network.keepalive_s` (default 30), answered
  `ack {"detail":"pong"}`. No frame received for `network.link_idle_s` (default
  120) → close, audit `link_idle`.
- **Reconnect**: exponential backoff per peer from 1 s to `network.reconnect_max_s`
  (60 s), ±25 % jitter, reset on a successful handshake. A link is only ever
  established by a fresh handshake — there is no resume, and no session-ticket
  mechanism.
- **Dial precedence and dedupe**: both sides may dial (either may be the one with
  a reachable path). If two links between the same device pair exist, the relay
  keeps the one whose dialer has the lexicographically smaller `device_id` and
  closes the other with `net_bye {"reason":"dedupe"}`; the loser does not
  immediately redial, because the winner is already up. This is deterministic and
  needs no election round trip.
- **On link loss**: forwarded session streams are marked `unreachable` at the
  viewer, never silently closed (§10.5); reliable network ops (membership,
  rotation) sit in the durable per-peer outbox and are retried on reconnect.

### 6.6 Backpressure

Two classes, because dropping the wrong one loses work and blocking on the wrong
one stalls a chat.

- **DROPPABLE** — `projection` and other session-state pushes for a viewer. At
  most **one pending per (link, stream)**: a newer push replaces the older. This
  is not a new idea to invent — the existing event vocabulary is already
  "projection: full projection repaint (the only push form — no deltas)"
  (`mobile/types.py:346`), which is exactly a coalescable frame. A peer that is
  2 s behind gets the newest state, not 400 stale repaints.
- **RELIABLE** — acks, `net_reconcile` results, `net_epoch`, `net_member_*`,
  `net_panic`, and forwarded ops carrying a `command_id` (which the runtime
  already dedupes, `mobile/types.py:279`). The producer waits up to
  `network.op_wait_s` (default 10) and then **fails the op with a sentence**
  rather than dropping it. Membership ops that cannot be delivered go to the
  durable outbox; session-plane ops are **not** auto-retried — a `prompt` without
  a `command_id` retried blindly is a double prompt, so retry belongs to the
  viewer, which knows whether it already got an answer.
- Queue budget per link: `network.queue_frames` (default 256) or
  `network.queue_bytes` (default 8 MiB), whichever bites first; DROPPABLE frames
  are coalesced down first, then RELIABLE producers are blocked.
- Inbound: a peer that sends more than `network.max_inflight` (default 64)
  unanswered RELIABLE ops, or an oversized record, gets one
  `error: net_backpressure` and then a close. Unbounded inbound queues are how a
  peer turns a relay into swap.

---

## 7. Authorisation

### 7.1 The capability model

| capability | grants | default roles |
|---|---|---|
| `list` | see this peer's session catalogue and member list | **read**, drive, admin |
| `view` | read a session's transcript, subscribe to its events | **read**, drive, admin |
| `prompt` | start/continue a turn, answer an approval or an ask | drive, admin |
| `steer` | inject a mid-turn steering message, recall one | drive, admin |
| `stop` | `abort`, `cancel`, `stop` (the kill switch), `retire_if_pristine` | drive, admin |
| `slash` | run a shared slash command in the session | drive, admin |
| `delete` | `net_session_lifecycle` (archive, restore, delete) | admin |
| `move` | `net_session_move` (move, fork-and-copy) | admin |
| `broker_credential` | ask this device's credential broker for a token | admin, and only by explicit opt-in per credential (doc 3) |
| `admin` | membership: `net_epoch`, `net_identity_rotate` | admin |
| `trust` | `net_trust` (re-admit a network), answer a panic | admin |

`ROLE_CAPABILITIES`:

```python
ROLE_CAPABILITIES = {
    "read":  frozenset({"list", "view"}),
    "drive": frozenset({"list", "view", "prompt", "steer", "stop", "slash"}),
    "admin": frozenset({"list", "view", "prompt", "steer", "stop", "slash",
                        "delete", "move", "broker_credential", "admin", "trust"}),
}
```

The three session-scope ops mobility adds (`net_session_create`,
`net_session_engage`, `net_session_stop`, §6.4) introduce **no new capability
names**: they map onto `prompt`, `view` and `stop` above, which is why the set
stays eleven names long and a reviewer can still hold it in one hand.

**Read-only is first-class**, per the spine's §5.3: "let my laptop see the
fleet" is the common, much safer ask. `net_panic` is deliberately `list`-gated —
any member that *detects* a compromise must be able to raise the alarm (the
denial-of-service that enables is bounded: it forces a re-admit, it does not leak
or destroy anything, and it is audited with the sender's id, §8.5).

### 7.2 The chokepoint

One function, one call site.

```python
# local_operator/network/authorize.py

@dataclass(frozen=True)
class LinkContext:
    link_id: str
    device_id: str
    instance_id: str
    network_id: str
    epoch: int                                  # the epoch this link authed at
    capabilities: frozenset[str]                # resolved from the member row
    phase: Literal["member", "reconcile"]
    peer_addr: str

@dataclass(frozen=True)
class Granted:
    action: str                                 # the op's name, for the audit record
    session_id: str | None = None

class Refusal(Exception):
    code: str                                   # not_authorised | not_a_member |
                                                # epoch_stale | phase_forbidden |
                                                # unknown_op | protocol_error | self_link
    sentence: str                               # what the peer is told

class Authorizer:
    def __init__(self, networks: NetworkState, audit: AuditLog) -> None: ...
    def check(self, link: LinkContext, frame: dict[str, Any]) -> Granted:
        """Raises ``Refusal``, or returns the capability that admitted the frame.
        ``effective_op`` runs FIRST, so the only name this function's tables are
        ever consulted with is a real op — never a carrier."""
    def effective_op(self, link: LinkContext, frame: dict[str, Any]) -> str:
        """``net_forward`` resolves to its INNER frame's op; anything else to
        itself. Exposed separately so a test can assert the resolution rule
        without dispatching."""
    def dial_local(self, link: LinkContext, granted: Granted) -> "ControlDial":
        """The ONE way this package opens a local control socket. Called only
        from the dispatcher, after ``check`` has returned (#§13.1's structural
        test asserts the single call site)."""
```

`RelayServer._dispatch(link, frame)` (`local_operator/network/server.py`) is the
**only** caller, and it calls `check` **before** looking at any field of the
frame other than `op` and `req`. Enforcement details:

- **Three vocabularies, and only two of them reach this table.** `ControlOp`
  (the session plane, forwarded), `NET_OPS` (the peer scope, defined in
  `network/types.py`) and the loopback control ops of §2.5. The last group never
  reaches a peer link at all: it is authorised by the local control key, exactly
  as the session runtime's own control socket is (`server.py:1885-1912`). So the
  totality test's input space is `ControlOp ∪ NET_OPS`, and a name from §2.5
  appearing in this table is a *bug*, not a gap — the test asserts both
  directions.
- **Pairing frames are phase-`pair` only** and are deliberately *not* in this
  table. A pair-phase link has no member row yet, so there is nothing to
  authorise against; its authorisation **is** the invite validation of §5.2 step
  2 plus the two human confirmations of step 5. `NET_PAIR_OPS =
  ("net_pair_ready", "net_pair_abort", "net_pair_result")` lives beside `NET_OPS`
  so the dispatch switch can refuse them outside phase `pair`, and a test asserts
  that no pair op is dispatchable on a member-phase link or vice versa.

- **Fails closed.** Unknown op name anywhere in the chain → `Refusal("unknown_op")`
  — this is why the totality test below exists rather than a permissive default.
- **Phase.** `link.phase == "reconcile"` permits *only* `net_reconcile` and
  `ping`. Anything else is `phase_forbidden`. This is what keeps a
  previous-epoch credential from being a general-purpose one.
- **Epoch.** `link.epoch` must be `current` (or `current-1` *with* phase
  `reconcile`). A link whose epoch fell behind because a rotation landed mid-link
  is torn down and a fresh handshake is required (the relay sends
  `net_bye {"reason":"epoch_stale"}` and closes; the peer re-dials and reconciles).
- **Session scope.** For any op carrying a `session_id`, the id must appear in
  this relay's `catalog` **and** that session's owner must be *this* device. A
  peer cannot ask a relay to act on a session that lives on a third device; with
  forwarding unimplemented (§14) such a request is `not_authorised`, not a
  silent hop.
- **Locality.** A frame that declares `locality: "local"` is refused as a
  protocol error — the spine's §5.8 "a command that claims `local` while arriving
  over a peer link is a protocol error, not a softer path".
- **Audit on refusal.** Every `Refusal` writes an audit record with the actor,
  subject, op, network, epoch and outcome; the peer is told only
  `{"op":"error","req":n,"message":"<sentence>"}` — never which of membership,
  epoch or capability failed.

**Totality test** (`tests/unit/network/test_op_capability_table.py`): iterate
`typing.get_args(ControlOp)` from `local_operator/mobile/types.py` and
`NET_OPS` from `local_operator/network/types.py`, and assert every name appears
in `OP_CAPABILITY` or `INNER_OP_CAPABILITY`. It fails **by name**, like
`test_every_default_matches_its_consumer` does for settings
(`tests/unit/test_settings_io.py`). Adding an op without deciding its
authorisation is therefore impossible, which is the property a chokepoint is for.

### 7.3 One role dimension, and what a role is not

Roles are `read`, `drive`, `admin`. They are resolved to capabilities **at
admission** and stored per member (§4.2), so changing `ROLE_CAPABILITIES` in a
future release does not retroactively widen anyone. `lop network show` prints
the resolved capability list, not the role, whenever a grant is being reviewed —
the role is a convenience for `invite --role`, never the authority.

### 7.4 Locality, on both sides

- Peer link → relay: `locality: "remote"` in every forwarded frame.
- Relay → local runtime: `locality: "remote"` in the control auth frame
  (`session/runtime/server.py:2700` parses the field, `:4359-4426` carries it
  into the dispatch), so the runtime's own routing sees the truth and
  `run_slash_authoritative(locality="remote")`
  (`session/runtime/serving.py:4087-4129`) behaves correctly.
- A *local* viewer's traffic never claims `remote`; a *remote* claim over
  loopback that did not come from the relay is indistinguishable from the relay's
  own dial — which is correct and is why the local control socket is keyed
  (§2.5).

### 7.5 What a member's authority does *not* include

Stated so it is not implied: a member with `prompt` can run arbitrary code on
this device, because a session's agent has a shell. The capability model bounds
*which sessions* and *which conversations*, not the effect of a prompt. This is
the spine's stated out-of-scope item ("protection against a malicious *member* on
capabilities it legitimately holds") and it is the reason `drive` is not the
default role for an invite.

### 7.6 Audit writer interface

`network/audit.py` exposes `AuditLog.record(event: AuditEvent) -> None`, one
JSONL line per semantic event (never per frame, never per token delta), written
line-buffered with size+age rotation. `AuditEvent` carries: `ts`, `network_id`,
`epoch`, `actor` (device id), `subject`, `event`, `outcome`, `session_id`,
`detail` (a bounded map). **Never** key material, transcript content, or token
values. Event kinds for this document's scope: `pairing_refused`,
`invite_minted`, `member_admitted`, `member_removed`, `member_left`,
`device_rotated`, `epoch_rotated`, `epoch_conflict`, `handshake_refused`,
`authorisation_refused`, `link_opened`, `link_closed`, `link_idle`,
`duplicate_identity`, `self_link`, `panic_raised`, `panic_received`,
`trust_changed`. The retention numbers, the measured I/O budget and the rotation
implementation are `mesh-incident-response.md`'s (R18 requires them measured).

---

## 8. Revocation and membership

### 8.1 Removing a member

`lop network member rm <network> <device_id>` (admin only locally; over the link
it is `net_epoch`, §8.4):

1. Write the tombstone (`removed_at`, `removed_by`), `lifecycle: "expired"`,
   drop any pending links, and refuse a re-admission at that id forever (§4.2).
2. Mint a fresh secret and bump the epoch: `epoch: N → N+1`, record
   `rotations[N+1] = self.device_id`, keep `previous = N`'s secret.
3. Broadcast `net_epoch` to every active member and queue it for every active
   member that is offline — **with the secret withheld from any recipient this
   frame's own `removed` array names**. The defect this closes, and the argument
   for closing it this way, are `mesh-incident-response.md` §2.3.4 and its §8
   Q1; what follows is the mechanism:

```json
{"op":"net_epoch","req":41,"epoch":8,"sequence":14,
 "rotation_id":"d_6c1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b",
 "secret":"b64url(32)",
 "members_digest":"hex64","members":[…rows…],
 "removed":["d_4b2a…"],"reason":"member_removed"}
```

   **The secret is per recipient, and withholding it is a sender-side
   obligation.** The frame above is what a surviving member receives. A device
   named in that same frame's `removed` array receives the identical frame with
   `"secret": null` — it is handed the member list and its own tombstone, and no
   key material. It is the *sender* that enforces this, because the removed
   device is precisely the peer we have just decided not to trust, and a property
   that depends on the cooperation of the untrusted party is not a property:

   > **Invariant `epoch_secret_withheld_from_removed`.** For every frame a relay
   > emits whose recipient device id appears in that frame's `removed` array,
   > the frame's `secret` is `null`. Equivalently: a removed device is never a
   > rotation *recipient*, only a tombstone recipient.
   >
   > **Invariant `epoch_outbox_holds_no_secret_for_a_removed_member`.** No file
   > the relay writes for a removed member — the durable per-peer outbox entry of
   > §6.6 above all — contains the rotated secret or anything derived from it.
   > The queue persists what the sender would have sent, so a queued rotation
   > addressed to a removed member is written with `"secret": null` — **written,
   > not dropped**, because the frame is also how a removed device that is
   > offline learns it was removed (§8.3's tombstone), and the null form costs
   > nothing while dropping the entry costs that signal. This is the half of the
   > defect that
   > touches the disk rather than the wire: a plaintext network secret parked in
   > an offline queue waiting for the device the operator just evicted is a
   > secret handed back on reconnection, exactly as if the frame were sent.
   >
   > Named tests, both in §13.1:
   > `test_net_epoch_withholds_the_secret_from_a_removed_recipient` and
   > `test_offline_outbox_never_holds_a_secret_for_a_removed_member`.

   **What a sender must never do, stated so it is not re-litigated:** it must not
   send the secret to a removed recipient and rely on the receiver's epoch check
   to discard it (§8.3 keeps that check for the *other* half of the problem — the
   tombstoned row makes the secret useless, not unknowable), and it must not
   queue a rotation for a member it has just removed.

   **The panic frame is deliberately different, and stays as designed**
   (`mesh-incident-response.md` §8 Q1's second half, and the credentials
   architect's recommendation that this document keeps): `net_panic` carries its
   secret to *every* recipient, including one whose distrust it is announcing.
   Panic wants every survivor coherent on one epoch so recovery is one command
   per device, and the operator who needs confidentiality against one specific
   device has `member rm` — which, after this fix, is the command that actually
   delivers it. The asymmetry is intentional and §8.3 repeats it where a reader
   meets the panic frame.

4. Re-handshake every live link at the new epoch (close + redial, or an
   in-place rekey — v1 closes and redials, because "one way to change keys" is
   worth more than the round trip).

A receiver applies `net_epoch` only if: it came from an active member; the epoch
is strictly greater than its own; `rotation_id` matches the sender's device id;
the `members` list is internally consistent (`self_device_id` present and active,
no duplicate ids, every `capabilities` set a subset of a known role's ∪ explicit
grants); and `members_digest` verifies. It then writes the record + secret
atomically and re-handshakes.

### 8.2 Self-deactivation, disconnect and panic (R17)

Three distinct actions, each with its own blast radius:

- **`lop network disconnect [<network>]`** — this device leaves. It sends
  `net_leave` (self-signed) to every reachable peer, closes the links, marks the
  network `disconnected` locally, **deletes the local secret** and keeps the
  audit trail. Peers that received the `net_leave` tombstone the member (§8.3);
  the *lowest-id active admin* among them initiates the rotation (§8.4), because
  the leaving device still holds the old secret — leaving is not a reason to
  trust it less, but it is a reason to stop it being able to read new traffic.
  A device that disconnects while nobody is reachable simply forgets the network;
  the peers see it go unreachable and the operator removes it from any device
  (R5), which is the point of the design.
- **`lop network stop`** — the relay stops. Network membership is untouched;
  sessions are untouched; peers see the link drop and mark this device
  `unreachable`. Not an incident action, and the docs say so, because "stop"
  reading as "revoke" would be dangerous.
- **`lop network panic [<network>]`** — the incident action. Admin: mint a fresh
  secret, bump the epoch, and broadcast

```json
{"op":"net_panic","req":9,"epoch":9,"sequence":15,
 "rotation_id":"d_6c1f…","secret":"b64url(32)","reason":"operator_panic",
 "members":[…],"members_digest":"hex64"}
```

  Every receiver sets `trust: "untrusted"` (with `untrusted_reason: "panic
  received from <id>"`), closes every link for that network, and **refuses all
  further peer traffic for it, including a connection that arrives afterwards** —
  the spine's requirement, and the reason `trust` is checked at handshake step 3
  as well as at dispatch. Recovery is explicit and local:
  `lop network trust <net> --active` on each device, then re-handshake.

  **A panic is never queued for an offline member** (`mesh-incident-response.md`
  §8 Q2): a queued panic replayed at an unknown later time is a stale judgement
  about a network state that may have changed, and it would make
  `net_trust --active` non-monotonic. That is also why the plaintext secret in a
  panic frame is not the on-disk problem the `net_epoch` queue was: the frame
  exists only in flight, and a sleeping member is covered by the epoch check and
  the reconcile path instead.

  Non-admin panic: `net_panic` is `list`-gated, so any member may raise it; a
  non-admin sender's frame carries no `secret` and rotates nothing (receivers
  still go untrusted, and audit the sender). The DoS this allows is stated
  plainly in §10.2: a rogue member can force every peer to stop trusting the
  network until an operator re-admits. That is the correct direction for the
  failure mode to point (a false alarm costs an operator action; a suppressed
  alarm costs the network).

### 8.3 What a removed device experiences

- **Online at removal time:** it receives the tombstone in the `net_epoch`
  members list — and **no key material**. The sender withholds `secret` from the
  frame it addresses to a device its own `removed` array names (§8.1 step 3,
  invariant `epoch_secret_withheld_from_removed`), so a removed device does
  **not** learn the new secret, and nothing about that depends on its good
  behaviour. Its next frame is refused (`not_a_member`) and the link is closed.
  `lop network ls` on that device shows the network as `revoked` (learned from
  the last delivered state) with the sentence `this device is no longer a member
  of home-net (removed by damian-mbp at 20:41)`.

  What it does still hold is epoch N's secret, which it already had. That is
  worth nothing to it: every frame it produces fails the epoch check (§8.4), and
  the reconcile phase answers a tombstoned row with a refusal before it would
  hand over the rekey. So `member rm` **evicts from acting *and* from reading**,
  R5's eviction is not authorisation-only, and the spine's "the network secret is
  what makes R5 possible — revocation rotates it" (§4/A3) is true as written.

  **The panic frame is the deliberate exception.** `net_panic` still carries its
  secret to every recipient, including a member about to be distrusted
  (`mesh-incident-response.md` §8 Q1 keeps that trade for survivor coherence).
  A reader who finds the two frames behaving differently has found the design,
  not an oversight: `member rm` is the command that buys confidentiality from one
  device, and `panic` is the command that buys coherence across all of them.
- **Offline at removal time:** it learns nothing until it dials. It presents the
  old epoch, and is refused at handshake step 6 — *membership is evaluated
  against the current epoch's member list*, so its `previous`-epoch secret is
  worthless. The wire is silent (no reply), and it writes `handshake_refused:
  not_a_member` locally. Its own copy of the network is then marked `stale:
  refused_by_peers` so `lop network ls` says something true rather than showing a
  healthy network it cannot reach. The operator's remedy is to remove the local
  record: `lop network rm <net>` — purely local.
- **The removed device's local sessions are unaffected**, and its own transcripts
  remain readable. Revocation is about the network, never the disk.

### 8.4 The offline-member problem, concurrent rotation, and the bounded secret

**The second half of R5: a member that never comes back.** Membership re-evaluated
at every contact covers the device the operator *removed*; it does not cover the
device that was a member in good standing, still is a member, holds a **current**
secret, and never dials again. It keeps working authority forever, and if a
backup of its disk leaks a year later, the leak is a working credential. Prior art
is unanimous that this is answered by a *second* mechanism, not by better
re-evaluation: every credential carries a bounded lifetime — Nebula's 1-year
certificates, Tailscale's 180-day node keys, OpenZiti's 24-hour enrolment token
(`mesh-prior-art.md` §2) — so a device that never contacts again ages out on the
network's side without anyone visiting it.

> **Invariant `epoch_max_age_forces_rotation`.** An epoch older than
> `network.epoch_max_age_s` (default `2592000`, 30 days) may not be used to
> establish a member-phase link by anybody, including a fully current member.
> Past the age the only thing a peer can do with its secret is prove possession
> of it in a **reconcile**-phase handshake, whose answer tells it to rotate.
>
> **STATUS: THE INVARIANT IS NOT ENFORCED AND THE BOUND IS DEFERRED — this note
> is the record of that, so nothing below reads as shipped.** Nothing reads
> `network.epoch_max_age_s`, `SecretState` carries no epoch mint time, and no
> phase decision consults an age (round-1 review, MAJOR 2). It is deferred rather
> than half-built because the REMEDY half is missing and enforcing the age alone
> would be a regression, not a partial win: the invariant ends with "the answer
> tells it to rotate", and that answer has no client anywhere in this build —
> nothing sends `net_reconcile`, nothing applies the epoch and secret a
> `net_reconcile` answer carries, and its `close_after` has no reader either
> (`_op_reconcile` is the answering half only). A device that missed a rotation
> already lands in `reconcile` phase today and stays there. Enforcing the age
> would extend that trap to every link on an aged epoch, while §8.4 promises the
> opposite: "nothing stops, and no operator action is needed while an admin is
> reachable". So the two halves land together or not at all, in this order:
> **(1)** the reconcile client — send `net_reconcile`, apply the epoch/secret it
> answers, close and re-handshake at the new epoch, with the phase's own tests;
> **(2)** then the age bound here — the mint time beside the secret, the key, the
> phase decision, the admin-initiated rotation (`reason: "max_age"`), and the
> `ls`/`doctor` reporting below. Until then, the honest operator-facing statement
> is that a member which never returns keeps working authority: the bound is a
> design decision with a named invariant and no code.

Concretely, and this is the part that must be implemented rather than described:

- **Age is judged on the minting device's clock**, exactly as the invite's `ttl_s`
  is judged on the inviter's (§5.1: "the *inviter* enforces freshness against its
  own clock"). A member stores `minted_at` with the secret it accepted, recomputes
  the age locally, and reports the difference in `lop network doctor` — skew is
  diagnosed, never arbitrated (§10.3).
- **A live member past the age is admitted in `reconcile` phase, not `member`
  phase.** Its `hello`/`auth` still authenticates (it holds the current secret,
  so the MAC verifies), and it is dispatched `ping` and `net_reconcile` only —
  no session op may cross that link until a new epoch exists. This reuses the
  phase machinery that already exists for the still-member-with-`previous` case
  rather than adding a second gate.
- **The rotation is initiated, not negotiated.** The `net_reconcile` answer
  carries `minted_at`, so both ends know the epoch is aged. The
  **lowest-`device_id` active admin** among the parties that know mints
  `epoch+1` with a fresh secret (the same deterministic rule as a concurrent
  rotation, so two admins discovering it at once still converge) and broadcasts
  `net_epoch` per §8.1 — which is why the withholding rule there matters here
  too: the aged epoch's rotation must not re-expose a tombstoned device.
- **The aged case gets no extra allowance.** `RECONCILE_MAX_PER_HOUR = 3` below
  still applies — an aged epoch is not a licence to bypass the bound, because "I
  keep presenting an old epoch" is the same traffic shape whatever its reason,
  and a device that loops on it must not be able to make a relay do work. Three
  answers an hour is enough because the aged case needs **one successful answer
  per device**: the answer tells the device to expect a rotation, and the admin's
  `net_epoch` broadcast then rotates everyone at once. A network whose only admin
  is unreachable past the age is a network the operator re-admits (or rotates by
  hand from a device that is reachable) — stated here so it is not discovered in
  an incident.
- **What the operator sees.** `lop network ls` shows the epoch and its age; past
  the age it prints the remedy (`lop network member rm`-free: the network
  re-keys itself on the next contact with an admin). `lop network doctor` reports
  `epoch_age_days` and `epoch_max_age_days`.
- **What happens when it fires, in the order an operator experiences it.** Nothing
  stops, and no operator action is needed while an admin is reachable. An aged
  epoch cannot open a *member-phase* link, so the next contact between any two
  peers lands in `reconcile` phase; the `net_reconcile` answer carries `minted_at`
  and tells the asking device a rotation is due; the **lowest-`device_id` active
  admin** mints `epoch+1` with a fresh secret and broadcasts `net_epoch` per §8.1;
  every peer adopts it and re-handshakes. **A session in flight is not disrupted**:
  a session is a separate runtime process that does not hold the link, its turn
  runs to completion on the device it runs on, and the only visible effect at a
  viewer is the existing reconnect path of §10.5 — the projection re-opens its
  stream (`stream_open`) and re-lists, which it already does for any link drop.
  Nothing is stopped, deleted or re-prompted, and no transcript is touched.
- **The number is a policy, and it is recorded as one.** 30 days is chosen to sit
  above any plausible offline window (a sleeping laptop must never be forced to
  re-pair) and below the 180-day node-key ceiling the prior art sets; it is
  configurable because an operator's tolerance for "a lost laptop is still a
  member" is a property of their risk, not of the protocol. Lowering it shortens
  how long a *vanished* member stays able to authenticate, which is the mechanism's
  entire purpose and therefore the entire meaning of the number — so it is a
  decision the operator makes deliberately, not a tuning knob.
- **Why this is cheap now and expensive later.** It adds one field to the record,
  one field beside the secret, one config key, one phase decision and one audit
  detail (`epoch_rotated` gains `reason: "max_age"`). Added after installs exist,
  it is a secrets-file migration on every device plus a rule every reader of the
  key has to learn — and, worse, a network whose members have all been holding an
  unbounded secret for a year has no evidence of who still has it.

**Offline-member problem.** A member that is genuinely still a member but missed
a rotation holds only `previous`'s secret. If the handshake simply refused
everything below `current`, that member would be locked out permanently — the
thing R5's "without visiting each device" must not create. The answer is the
**reconcile phase**:

- A `hello`/`auth` whose `mac` matches `epoch_key(previous)` is admitted with
  `"phase": "reconcile"` **only if the device is an active member at the current
  epoch**;
- in that phase exactly two ops are dispatchable: `ping` and `net_reconcile`;
- `net_reconcile` answers with the current epoch, the current secret, the full
  member list, `members_digest` and `rotations`; the relay then closes the link
  and expects a fresh handshake at the new epoch;
- grants are rate-limited: at most `RECONCILE_MAX_PER_HOUR = 3` per device per
  network, audited (`reconcile_granted` / `reconcile_refused`), and a fourth
  request is refused with `reconcile_rate_limited` — because "I keep presenting
  an old epoch" is also what a replayed credential looks like.

Two properties to test explicitly: a **removed** device presenting `previous`
gets nothing (its row is a tombstone, checked first), and a **still-member**
device presenting `previous` gets exactly the epoch state and nothing else (a
`prompt` in reconcile phase is `phase_forbidden`).

**Concurrent rotation.** Two admins can rotate at the same moment, producing two
`net_epoch` frames with the same target epoch and different secrets. The
deterministic rule:

1. `epoch` is monotonic. A received `net_epoch` with `epoch <= current` is
   ignored (`ack {"detail":"already_at_epoch"}`), which absorbs duplicates and
   the ordinary race where both peers announce the same rotation.
2. A received `net_epoch` with `epoch == current` and a *different*
   `rotation_id` is refused (`epoch_conflict`, audited) — and the receiver
   sends its own state back, so the sender learns it lost the race.
3. Rotation ids are compared by the `min` rule only for *tie-breaking within the
   same epoch number when both are received before either is applied*: the
   device with the lexicographically smallest `device_id` wins, so all members
   converge on one secret. The loser then rotates again at `epoch+1`, which is
   convergent and cheap (`lop network doctor` reports `rotation_lost`).
4. In practice, concurrent rotations are made unlikely by a **30-second
   per-network rotation lock** (`rotations[N+1]` exists ⇒ a second local rotation
   is refused with `rotation_in_progress`).

### 8.5 Membership state machine

```
(member, active) ──leave──► tombstone(active=false, expired)
      │  │
      │  └──remove (by an admin) ──► tombstone + epoch++ + rotate
      │
      └──rotate-device-key──► same row, new device_id, previous_ids+=[old]
```
`trust` is orthogonal and network-wide: `active ⇄ untrusted`, with `untrusted`
refusing handshake step 3 and dispatch.

---

## 9. The record namespace and the federated listing

### 9.1 `run/peers`, and the two namespaces a catalogue reads

`lop sessions` continues to mean sessions (`run/mobile` only, no behaviour
change; that is the `0 peers` regression in the spine's §10). The mesh's own
listing reads **two** things: the local `run/mobile` scan (as today) and the peer
records of `run/peers` (§2.6) to know which relays exist, then asks each relay
for its sessions over the link (§9.2). A relay that is not running, or is
unreachable, yields a `degraded` entry, not a missing row.

### 9.2 The peer-level catalogue op

```json
{"op":"net_catalog","req":7,"limit":200}
```
answers, as an event so it can stream:
```json
{"op":"net_catalog","req":7,"complete":true,
 "device":{"device_id":"d_6c1f…","name":"damian-ec2"},
 "generated_at":1758301234.5,
 "sessions":[
   {"session_id":"2026-09-19T18-04-11_ab12","conversation_name":"mesh design",
    "cwd":"/home/ubuntu/work","model_label":"deepseek-flash","busy":true,
    "pending":null,"detached":true,"started":true,"pid":91234,
    "capabilities":["exclusive-move-v1"],
    "state":"live","age_s":0.4}
 ]}
```
Field-by-field this is the session record's own read model
(`SessionRecord`, `types.py:566`) minus anything a peer need not know, plus
`state` from `classify` (`registry.py:395`). The relay builds it from
`registry.scan()` — one scan per catalogue request, with a 2 s cache so a
sidebar polling every second does not turn into a scan storm.

### 9.3 The local aggregation API

`lop sessions --all-peers --json`:

```json
{
  "sessions": [
    {"session_id":"2026-09-19T18-04-11_ab12","conversation_name":"mesh design",
     "locality":"local","peer":null,"busy":true,"state":"live","model_label":"deepseek-flash"},
    {"session_id":"2026-09-19T17-52-03_cc91","conversation_name":"rust port",
     "locality":"remote","peer":{"device_id":"d_6c1f…","name":"damian-ec2",
     "network_id":"n_…","reachable":true,"age_s":1.2},
     "busy":false,"state":"live","model_label":"qwen3-coder"}
  ],
  "peers": [
    {"device_id":"d_6c1f…","name":"damian-ec2","networks":["n_…"],
     "reachable":true,"latency_ms":38,"session_count":3},
    {"device_id":"d_4b2a…","name":"pod-17","networks":["n_…"],
     "reachable":false,"reason":"connect_timeout","last_seen_at":1758300000.0}
  ],
  "degraded": ["peer:d_4b2a…:connect_timeout"]
}
```

- `locality` is `"local" | "remote"` and is **always present**, so no consumer
  infers it from an id shape (the spine's §8 requirement, and the reason the
  desktop row contract gains a field rather than a heuristic).
- `peers` is a first-class part of the payload, so "unreachable" is renderable
  instead of being a silently absent row (the spine's §7 rule: never lose a
  session because a link dropped).
- `degraded` mirrors the desktop list route's own marker
  (`server/routes/desktop_sessions.py:1101` region), so one word means the same
  thing on both surfaces.
- `--peer <id>` narrows to one peer; the flag is additive on `lop sessions`, and
  the same two flags are added to `lop exec` (`lop exec --peer <id> "…"`).

### 9.4 Caching, so an offline relay is still usable

`network/catalog.json` (0600) holds the last successful catalogue per peer with
`fetched_at`, TTL 24 h. It backs three things that must work when the relay is
down: `/new remote <peer>` autocompletion (offline-safe: a stale list is usable,
an unreachable peer is refused *with its reason*), `lop sessions` showing
`unreachable` rows rather than nothing, and the desktop sidebar's peer groups.
Rows from the cache carry `reachable: false` and `age_s`, and the UI never
renders a cached row as live.

### 9.5 Desktop contract

- `GET /v1/capabilities` gains `features.peers: 1` in the map documented at
  `docs/DESKTOP_API.md:10-20`. A client without the key shows an update action
  rather than calling the route.
- The session row (`src/shared/desktop-session-contract.ts:9-33`,
  `canonical-sessions-store.ts:29-54`) gains `locality` and `peer` as additive
  fields; `empty`/absent means a pre-mesh backend, and the UI renders exactly what
  it renders today.
- A new `GET /v1/desktop/peers` returns the `peers` block above, so the sidebar
  can draw one group heading per peer and a distinct unreachable treatment. The
  space budget rules (leading glyph slot or tooltip, never the trailing
  statement, 32 px rows, group headings) are `mesh-ui.md`'s; this document's
  obligation is only that the data exists and carries no presentation decisions.
- The desktop app still has **one** `backendUrl` (`desktop-transport.ts:47-51`);
  the route is served by `lop serve`, which reaches the relay over the loopback
  control socket of §2.5. No second transport enters the renderer.

---

## 10. Failure modes and threat notes

### 10.1 MITM at pairing

- Without the invite token, the attacker cannot produce a valid `mac` and is
  refused before a SAS exists (§5.3).
- With a leaked token, the attacker can run two handshakes and must make both
  SAS values agree: ~2²⁰ grinding work per human interaction, and the invite is
  burned on failure, so each further attempt needs a fresh invite (a fresh human
  action on the inviter). The 160-bit fingerprint (§5.3) is the strong check and
  `--verify` makes it mandatory.
- The attacker cannot silently *swap keys* later: member rows are written only by
  admission and by rotation statements, and the member's public key is what
  verifies every subsequent handshake.
- After admission, a MITM on the path holds no key. Confidentiality and integrity
  come from the AEAD records; a modified frame fails the tag and closes the link.

### 10.2 Hostile former member

- **Holds old secrets** (it had `previous`): refused at the membership check,
  which is evaluated against the current epoch's member list before any key is
  even tried (§6.2 step 6).
- **Holds no new secret**: cannot read new traffic even if it can sniff frames.
- **Can flood**: rate limits (handshake attempts per source, `max_inflight`,
  reconcile grants per device per hour) plus the audit trail; a flood is visible
  in `lop network log` and remediable by removing the member, which needs no
  cooperation.
- **Can raise a false panic**: accepted by design (§7.1), bounded by the fact that
  the operator sees who raised it and can re-admit with
  `lop network trust <net> --active`. Stated as an accepted risk, not hidden.
- **Was a legitimate member with `drive`**: it could run commands while a member.
  Revocation does not unwind what it already did — say so plainly; the forensic
  trail is `mesh-incident-response.md`'s subject.

### 10.3 Clock skew

The design has **no cross-host time dependency**, and that is a property worth
stating because the obvious implementation has several:

- invites carry a `ttl_s` duration, and the only clock consulted is the minting
  device's own (§5.1);
- epochs are integers compared by value, never by timestamp;
- the SAS contains no timestamp, so a device with a badly wrong clock still pairs
  correctly;
- audit timestamps are local and are never compared across devices;
- `lop network doctor` reports clock skew between peers as a diagnostic (it can,
  because the handshake exchanges no time and the doctor's probe can carry a
  local timestamp in a `ping` detail), and nothing is refused because of it.

### 10.4 NAT, firewalls, and no inbound reachability

**Assumption, stated as an assumption:** *a device may have no inbound
reachability at all.* Nothing in this design requires a device to accept an
inbound connection; a link is established when either side can dial the other's
advertised endpoint, and the link is then bidirectional. Concretely:

- `network.listen_address` has three meaningful values: `127.0.0.1` (**dial-only**
  — this install never accepts), `0.0.0.0`, or a specific interface address.
- The peer record and each member row publish `endpoints`, from
  `network.advertise_hosts` plus detected local addresses (excluding loopback
  unless dial-only). A member with no usable endpoint is shown as
  `reachable: false, reason: "no_endpoint"` and remains a member.
- **Two devices, both behind NAT, with no third party, cannot link.** This is
  stated rather than papered over. The remedies, in the order I would recommend:
  1. **The Radient tunnel** that already exists (`lop tunnel`, gateway on
     `DEFAULT_GATEWAY_PORT = 4100`, `tunnels/config.py:16`) — expose the peer
     port through it and put that hostname in `network.advertise_hosts`. This is
     the sanctioned WAN path: the tunnel gives reachability, and the peer link's
     own authentication gives access control, so a public endpoint grants an
     attacker nothing but a handshake attempt.
  2. One member with a public address (an EC2 peer, the spine's §10 topology).
  3. A future hub/forwarding capability (§14) — **not built here**.
- **Never** UPnP/NAT-PMP (silently changes the operator's router configuration),
  never hole punching or STUN (needs rendezvous infrastructure and gives
  unpredictable reachability), never a third-party relay (that is the opposite of
  zero trust between peers).
- **The one sanctioned relayed path is a blind relay, and it is the shape a
  future hub must be built to**: a member that forwards frames it cannot read,
  exactly as Tailscale's DERP servers do ("it's impossible for a DERP server to
  decrypt your traffic" — `mesh-prior-art.md` §3). Two rules come with that
  label, and they are the reason the shape is worth naming now even though §14
  does not build it: (1) **a blind relay learns metadata only** — who talks to
  whom, when, and how much, and never a transcript, a token or a session id
  meaningfully (the ids it forwards are opaque to it); that is a real disclosure
  and the threat notes must say so rather than implying a relay is free; (2) **a
  blind relay is capped and reserved per link, not an unbounded byte pipe** —
  libp2p's Circuit Relay v2 was rewritten for exactly this reason
  ("a continuous over-subscription … an expensive proposition",
  `mesh-prior-art.md` §3), so a hub capability ships with a per-link reservation,
  a duration cap and a byte cap, and a peer that exceeds them is refused rather
  than absorbed. The tunnel remedy above (item 1) is the same shape with the
  gateway already built; a peer that *is* the relay is the version that needs
  these two rules written down first.
- `lop network doctor --peer <id>` reports: DNS resolution of each endpoint,
  outbound TCP connect with latency, handshake outcome and refusal reason, epoch
  skew, and the local listener's bind state. It never claims reachability it has
  not just proven — the same rule the repo applies to instruments generally
  ("a dead instrument returns a reading, not an error", AGENTS.md).

### 10.5 One side's relay is down

- **Local work continues.** Sessions are separate processes; the relay's absence
  is invisible to them.
- **Local listing** keeps working: local rows render normally, remote rows come
  from the catalog cache and are marked `reachable: false` with `age_s` and a
  reason. Remote sessions are **never deleted** from the list because a link
  dropped (spine §7).
- **Forwarded ops to that peer** fail with a sentence (`cannot reach damian-ec2:
  connect_timeout`) and are not auto-retried (§6.6). A viewer that reconnects can
  resume because its `AttachedSession`-shaped projection re-opens a stream
  (`stream_open`) and re-lists.
- **Membership traffic** is durable: `net_epoch`/`net_panic` sit in the per-peer
  outbox and are delivered on reconnect. A peer that has been unreachable across
  two rotations presents an epoch we no longer accept (`current-1` at best):
  it is refused at the epoch check and re-dials with `previous`, which reconciles
  (§8.4). Three or more missed rotations means it is outside the retained window
  and must re-pair — a bounded, documented cost, and the reason the audit log
  records every rotation with time and reason.
- **The peer that went down and came back with a new `instance_id`** is the
  normal restart path: `link_replaced`, no alarm (§3.4).

### 10.6 Disk, and the audit log

- Every record write is staged + `os.replace`d; there is no fsync on the hot
  path, and the guarantee is stated at that strength — the same call
  `registry._staged_write` already documents (durability of *process*, not of
  *host*).
- A failed **audit** write does not stop the relay: it logs to stderr and sets a
  `degraded` flag that `lop network status` reports. A failed **membership**
  write **does** fail the operation — never tell a peer it was admitted when the
  row did not land.
- Audit I/O is bounded by event rate, not traffic (A7), and the rotation numbers
  are the incident design's to pin with a measurement.

### 10.7 Not defended against (stated, so it is not implied)

A same-account process that reads `device.json` and impersonates this device;
a malicious *member* exercising capabilities it legitimately holds; a
compromised operator machine holding `admin`; multi-user networks and
per-user identity inside a network; key escrow; and traffic analysis (frame
sizes and timing are visible to anyone on the path, though not to a plain
observer of an encrypted tunnel).

---

## 11. Configuration keys

New section `Network`, `Scope.NEW_LAUNCH` ("read once at process start; needs a
relaunch", `settings_io.py:150-151`) — uniform within the section, with the
section description naming the command that applies it: `lop network restart`.
Every key needs its `Setting`, its module-level default beside the reader, and
its `_consumer_defaults()` entry (`AGENTS.md`, "Adding a configuration key").

| key | path | default | consumer |
|---|---|---|---|
| `network.autostart` | `("network","autostart")` | `True` | `network/service.py:ensure_running` |
| `network.listen_address` | `("network","listen_address")` | `"0.0.0.0"` | `network/server.py:RelayServer.start` |
| `network.port` | `("network","port")` | `4097` | ditto (4098 mobile, 4099 browser bridge, 4100 tunnel gateway are taken) |
| `network.advertise_hosts` | `("network","advertise_hosts")` | `[]` | `network/server.py:advertised_endpoints` |
| `network.handshake_timeout_s` | `("network","handshake_timeout_s")` | `10.0` | `network/handshake.py` |
| `network.keepalive_s` | `("network","keepalive_s")` | `30.0` | `network/link.py` |
| `network.link_idle_s` | `("network","link_idle_s")` | `120.0` | `network/link.py` |
| `network.reconnect_max_s` | `("network","reconnect_max_s")` | `60.0` | `network/link.py` |
| `network.op_wait_s` | `("network","op_wait_s")` | `10.0` | `network/link.py:DROPPABLE/RELIABLE queues` |
| `network.queue_frames` | `("network","queue_frames")` | `256` | `network/link.py` |
| `network.queue_bytes` | `("network","queue_bytes")` | `8388608` | `network/link.py` |
| `network.max_inflight` | `("network","max_inflight")` | `64` | `network/server.py` |
| `network.max_links` | `("network","max_links")` | `32` | `network/server.py` |
| `network.epoch_max_age_s` | `("network","epoch_max_age_s")` | `2592000` (30 days — a policy statement about the operator's device-loss window, not a derived number; §8.4) | `network/membership.py:epoch_is_aged` (§8.4) |

Audit retention keys are `mesh-incident-response.md`'s. Nothing here is a secret,
so every key is `--json`-safe and `/settings`-visible.

---

## 12. Compatibility, migration, rollout

### 12.1 `0 peers` is the regression baseline

With no network record, the relay idles, publishes nothing (or publishes a record
with `networks: []` — recommend publishing always, so `lop network status` has an
answer), and no code path in the session plane changes. Requirement R16 ("no
regression on provider or MCP logins") and R9 ("quitting is never fatal") are
guaranteed here by construction: nothing in the local path is touched. The
spine's topology 0 is the gate.

### 12.2 One widening, named

`RuntimeLocality = Literal["this-process", "this-machine", "unknown"]`
(`session/protocol.py:128`) carries a comment saying there is deliberately no
`"another-machine"` member because "a cross-host runtime cannot occur". That
comment becomes **false** in this effort, and the change is therefore:

- add `"another-machine"` to the union;
- rewrite the comment to state the condition under which it *can* occur (a
  peer-backed session), rather than deleting it — the reason it was omitted was
  that the axis was dead, and the axis is alive now;
- handle it at the one consumer that switches on locality:
  `tui/app.py:13630` (`if session is None or session.runtime_locality ==
  "this-process"`) — a negative test, so it is safe by default, but the *label*
  it draws must gain a case, and the desktop equivalent is §9.5's `locality`; and
- add a test in `tests/unit/session/test_viewer_protocol.py`'s family that a new
  locality member is handled by every `==` comparison over it, so the next
  addition cannot be silently mistyped.

I recommend against a peer-qualified locality (`"peer:d_…"`) on this type: it
would put a device id in a field whose consumers compare it against literals, and
the device id already has a home (`peer` in the row). Keeping them separate is
what makes the type safe to extend.

### 12.3 New ops are additive, and that is enforced by the existing rule

Every `net_*` op is a new string in the dispatch table
(`server.py:3300-3800` for the session side; the relay's own table here), so an
old peer answers `error: unknown op` exactly as `mobile/attach_client.py`'s
callers already expect. Additive ops never move `PROTOCOL_VERSION` (the pattern
`mobile/types.py:279` records for `peer_message`, `stop`, `retire_if_pristine`,
`variables`), and never move `MESH_PROTOCOL_VERSION` either — that number moves
only for a change to the handshake, the transcript or the record framing.

### 12.4 Forward compatibility (R20, R21, R22)

What exists **now** so metered on-demand capacity is an extension:

- `MemberRecord.kind` (`device` | `pool`) and `lifecycle`
  (`active` | `provisioning` | `draining` | `expired`) exist and nothing in the
  pairing path assumes a human at the far end (§4.2). A pod is paired with
  `--role drive --automated` (§5.2), which requires a human on the inviter only
  and marks `kind: "pool"`.
- `net_sync` is a **reserved op name** with a reserved capability (`view`), so
  R22's cadence sync and its pre-spin-down sync are an implementation of a named
  op rather than a change to the envelope (§6.4).
- The credential broker is a capability (`broker_credential`) and an op
  (`net_broker`) from day one, which is what makes a pod that holds no credential
  safe (A5, spine §9.5).
- Nothing in the record schema ties a member to a device-shaped fact: no serial
  number, no "human present" flag, no assumptions about `endpoints` being
  stable. A pod that re-registers with a new endpoint is the ordinary path.
- Metering is an audit-event kind (`metering`, spine §9.4), so a future biller
  reads events and never session state. This document only reserves the kind.

Explicitly **not** built here: forwarding/hub, scheduling, credit computation,
provisioning, and session sync.

### 12.5 The agent-facing surface (R19, transport half)

- `local_operator/guides/network/GUIDE.md` (frontmatter `name: network`,
  discovered by `guides/discovery.py:23`), the same shape as
  `guides/mobile/GUIDE.md`.
- A `network` entry in `TOOL_BUILDERS` / `DEFAULT_TOOL_NAMES`
  (`tools/registry.py:32-97`), unconditional (unlike `secret`, which is
  `createIf`): an agent must be able to *create* the first network, so the tool
  cannot depend on one existing.
- The tool's actions mirror §2.5's ops (`status`, `init`, `invite`, `join`, `ls`,
  `show`, `peers`, `member_rm`, `disconnect`, `panic`, `log`, `doctor`) and shell
  the CLI with `--json`.
- **The invariant that matters:** the invite token never enters a tool result.
  `invite` returns a path; `join` requires an interactive prompt for the SAS, so
  the tool must tell the user to run it (or run it under a TTY it does not own)
  and can never complete a pairing on its own. That is a *feature* of the
  security model, and the guide must say so.

---

## 13. Test plan

Two layers. Unit invariants are cheap and total; the topologies are real
processes on real sockets. Nothing here is a substitute for the whole-tree gate
(`AGENTS.md`: flake8, `black --check`, `isort --check-only`, `pyright` via
`make type-check` with its bound, and the full unit suite exactly as CI runs it,
which is 40-55 min on this host under fleet load — check load first and never
leave a worktree or process behind).

### 13.1 Unit invariants (`tests/unit/network/`, isolated config dir)

Every invocation: `env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator"
PATH="$PATH" TERM=xterm-256color .venv/bin/python -m pytest tests/unit/network -q`
(TUI-booting tests additionally need `env -u NO_COLOR TERM=xterm-256color`).

Identity and keys:
- `test_device_id_is_derived_not_assigned` — fixed vector; editing the public key
  changes the id.
- `test_identity_file_is_0600_in_a_0700_dir`, and
  `test_identity_write_is_atomic` (kill between write and rename leaves the old
  key).
- `test_identity_never_reaches_a_log_or_a_frame` — run a full pairing under
  `caplog` at DEBUG and assert no serialized key material appears in any log
  record or in any frame captured by the fake peer. This is the redaction
  invariant stated as a test, not a review note.
- `test_device_id_conflict_is_refused` — a second public key with an existing id
  is rejected and audited.

Handshake and framing:
- `test_transcript_binds_every_field` — mutate one byte of each of `hello`,
  `challenge`, `auth-core`, the role byte, and each capability; every mutation
  must fail verification.
- `test_reflection_fails` — feed a peer its own `hello`; the role byte and the
  direction keys make it undecryptable/unsigned.
- `test_wrong_epoch_is_refused_without_a_reply` — assert the socket closes with
  zero bytes written after `hello`.
- `test_previous_epoch_yields_reconcile_phase`, and
  `test_reconcile_phase_refuses_a_prompt`.
- `test_nonce_is_never_reused` — 10 000 records, assert monotonic distinct
  counters and that a replayed record fails.
- `test_record_over_max_bytes_closes_without_allocating` (assert peak RSS stays
  bounded — the length prefix must be checked before allocation).
- `test_aad_binds_direction_and_sequence` — swap two records' order (they arrive
  out of order only via a harness that bypasses TCP).
- `test_link_restart_within_grace_is_not_duplicate_identity` and
  `test_duplicate_identity_evicts_and_audits`.

Pairing:
- `test_invite_is_single_use` — second redemption → `invite_already_used`;
  `redeemed` in flight → `invite_in_use`.
- `test_invite_ttl_uses_mint_clock_only` — the joiner's clock is shifted by ±1 h
  and pairing still succeeds; the minting relay enforces the TTL.
- `test_invite_epoch_stale_is_refused` — rotate between mint and redeem.
- `test_invite_cannot_be_redeemed_against_a_non_inviter`.
- `test_bound_invite_is_refused_for_another_device` (§5.1) — mint with
  `--device`, redeem from a second device id: `pairing_refused
  {cause: "wrong_device"}`, no member row written, and the invite `consumed`.
- `test_unbound_invite_still_redeems_from_any_device` — the regression half: the
  default (no `bound_device`) must keep behaving exactly as documented today, so
  the binding cannot silently become mandatory.
- `test_sas_matches_on_both_sides` — deterministic vectors; the joined member can
  authenticate afterwards.
- `test_sas_never_appears_in_any_frame` — blanket assertion over every frame both
  roles emit in a full pairing: the derived SAS string, and its digits with
  spaces removed, must not appear in any frame's JSON. This is the invariant that
  makes the human comparison a check rather than a round trip, so it is asserted
  rather than trusted.
- `test_sas_mismatch_burns_the_invite_and_admits_nothing` — assert no member row,
  `invites[]` state `consumed`, and two `pairing_refused` audit records.
- `test_mitm_cannot_produce_matching_sas` — the scripted two-role MITM of §13.2
  as a unit test with fixed keys: assert the two derived SAS values differ, and
  that a bounded grind (a loop over the attacker's ephemeral choices) does not
  find a match without having *chosen* the ephemeral after both peer ephemerals
  were fixed — i.e. assert the property that makes the residual bound 2²⁰ rather
  than 1.
- `test_admission_writes_the_row_before_the_frame` — kill the link between the
  write and the send; the member must persist and the invite must be consumed.

Authorisation:
- `test_op_capability_table_is_total` (§7.2) — fails by name.
- `test_read_role_cannot_prompt_steer_stop_slash` — four refusals, four
  `authorisation_refused` audit records, and *no* control-socket dial was made
  (assert on the dial helper, which is the point of the chokepoint).
- `test_net_forward_resolves_to_the_inner_op_capability` — a `net_forward`
  carrying `prompt` from a `read` member is refused.
- `test_frame_claiming_local_locality_is_a_protocol_error`.
- `test_session_scope_requires_local_ownership` — a peer asking for a session
  that lives on a third device is refused, not forwarded.
- `test_only_chokepoint_reaches_a_control_socket` (§2.1's structural test).

Membership and revocation:
- `test_remove_rotates_secret_and_bumps_epoch`, `test_removed_device_is_refused_at_both_epochs`,
  `test_removed_device_cannot_reconcile`, `test_still_member_reconciles_within_rate_limit`,
  `test_reconcile_rate_limit`.
- `test_epoch_broadcast_is_idempotent` (`epoch <= current` → `already_at_epoch`).
- `test_concurrent_rotation_converges` — two rotators, injected in both orders;
  assert both converge on the `min`-`device_id` secret and that each audits
  `epoch_conflict` or `rotation_lost` at most once.
- `test_leave_tombstones_and_the_lowest_id_admin_rotates`.
- `test_net_epoch_withholds_the_secret_from_a_removed_recipient` (§8.1) — one
  remover, one removed device and one surviving device, all on one host: assert
  the removed device's received frame has `secret: null`, the survivor's has the
  secret and derives the epoch key, and the removed device derives nothing.
- `test_offline_outbox_never_holds_a_secret_for_a_removed_member` (§8.1) — remove
  a member while it is down, then walk every file under
  `<config>/network/outbox/`: none may contain the rotated secret or a value that
  derives the epoch key, and the entry for the removed member, if it exists at
  all, has `secret: null`.
- `test_epoch_older_than_max_age_reconciles_only_and_rotates` (§8.4) — with
  `network.epoch_max_age_s` set to 1 s and a live two-device pair, assert the
  next handshake lands in `reconcile` phase, that a `prompt` over it is
  `phase_forbidden`, and that the lowest-id admin's rotation lands as
  `epoch_rotated {reason: "max_age"}` with both sides on the new secret.
- `test_aged_epoch_is_never_used_for_a_member_phase_link` (§8.4) — the negative
  half of the same rule, asserted on the dialer as well as the listener.
- `test_panic_marks_untrusted_and_refuses_later_connections` — a *fresh*
  connection after the panic is refused at handshake step 3.
- `test_device_rotation_statement_requires_the_old_key`.

Records and listing:
- `test_peer_record_is_additive` — unknown keys dropped, absent optional keys
  default (the `ServeRecord.from_json` contract).
- `test_stale_peer_record_is_reaped` — `kill -9` the relay, assert `scan`
  classifies `stale` and the file is replaced with its evidence sidecar.
- `test_peer_record_contains_no_key_material` — assert the JSON has no field
  whose value appears in `device.json` or the secrets file.
- `test_catalog_marks_unreachable_peers` — one live fake peer, one dead endpoint:
  the live rows are `locality: remote` with `reachable: true`, the dead peer
  appears in `peers` with a reason, and `degraded` names it.
- `test_catalog_never_omits_a_session_because_a_link_dropped`.

Backpressure:
- `test_droppable_frames_coalesce_to_one_pending_projection`;
- `test_reliable_op_blocks_then_fails_with_a_sentence`;
- `test_membership_op_survives_a_link_outage_in_the_outbox`;
- `test_session_op_without_command_id_is_not_retried`.

Structural / import:
- `test_relay_is_not_an_owner` (§2.1);
- `test_network_package_imports_are_light` (no `cryptography`, no `fastapi` at
  `import local_operator.network.types`) — the CLI startup rule;
- `test_listeners_except_the_peer_listener_bind_loopback` (§2.5).

### 13.2 Two real "devices" on one host (the fast topology)

Two config dirs, two relays, two identities, real TCP — the whole protocol,
hermetically, in CI:

```sh
ISO=$(mktemp -d)
A="$ISO/a"; B="$ISO/b"; mkdir -p "$A" "$B"
CLI=.venv/bin/python
# relay A
env -i HOME="$A" LOCAL_OPERATOR_CONFIG_DIR="$A/.local-operator" PATH="$PATH" \
  "$CLI" -m local_operator.cli network serve --port 4097 --no-launchd &
# relay B
env -i HOME="$B" LOCAL_OPERATOR_CONFIG_DIR="$B/.local-operator" PATH="$PATH" \
  "$CLI" -m local_operator.cli network serve --port 4197 --no-launchd &
# A: create + invite (the token goes to a file, never to stdout)
env -i HOME="$A" LOCAL_OPERATOR_CONFIG_DIR="$A/.local-operator" PATH="$PATH" \
  "$CLI" -m local_operator.cli network init home-net --json
env -i HOME="$A" … network invite --role drive --expires 5m --hosts 127.0.0.1:4097 --json
# B: join, SAS supplied by the harness at the prompt (the ONLY way in)
env -i HOME="$B" … network join @"$A/.local-operator/network/outbox/<id>.invite" \
  --host 127.0.0.1:4097 --sas-stdin   # harness feeds the digits A printed
env -i HOME="$A" … network show home-net --json     # 2 members, epoch 1
```
`--sas-stdin` exists **for tests only** and is documented as such: it reads the
digits from stdin, which a harness can drive and a human would type. It does not
let an agent (or a test) *skip* the comparison — A still requires a confirmation,
also fed from the harness. A test asserts the flag is refused without
`LOP_NETWORK_TEST_MODE=1`, so it cannot be reached in production by accident.

Deliberate breakage, scripted (`tests/e2e/mesh/fake_peer.py` and
`tests/e2e/mesh/mitm.py`, stdlib-only, no browser, no headless anything):

```sh
.venv/bin/python tests/e2e/mesh/mitm.py --listen 127.0.0.1:4317 \
  --to 127.0.0.1:4097 --invite @<token-file> --join-cmd '<B command>' \
  --assert-sas-differ
```
The harness runs both handshakes, prints both SAS values, asserts they differ,
asserts the inviter records `pairing_refused` with `sas_mismatch`, and asserts B
admits nothing. Three more scripted breaks: a replayed invite
(`invite_already_used`), a tampered `prompt` frame (`InvalidTag` → link closed,
`link_closed` audited), and a `read`-role peer issuing a `prompt`
(`authorisation_refused`, no session contact).

Revoking:

```sh
# A revokes B
env -i HOME="$A" … network member rm home-net d_<B>  --json
env -i HOME="$A" … network show home-net --json      # tombstone, epoch 2
# THE ROTATED SECRET NEVER REACHED B (invariant, §8.1): A's relay log for the
# outbound net_epoch addressed to B must show secret withheld, and B's own
# secrets file must still hold epoch 1's secret and no epoch-2 entry at all.
env -i HOME="$A" … network log --since -5m --json | grep -E 'epoch_rotated|member_removed'
test "$(jq -r .current.epoch "$B/.local-operator/network/networks/n_<id>.secrets.json")" = 1
# B proves it is refused, on a FRESH connection
env -i HOME="$B" … network peers                     # "refused: not_a_member"
env -i HOME="$B" … sessions --all-peers --json       # A unreachable, B's own rows fine
env -i HOME="$B" … network log --since -5m --json | grep -E 'handshake_refused'
```

The two assertions in the middle are the end-to-end form of §13.1's
`test_net_epoch_withholds_the_secret_from_a_removed_recipient`: the unit test
proves the frame, and this proves the file on the evicted device's disk.

### 13.3 The real topology (the evidence R1–R5 need)

Per the spine's §10: two real machines — this laptop and an EC2 instance
provisioned with the operator's `minerva_nprod` AWS profile (the
`minerva-platform-deployments` skill owns the mechanics) — with the peer port
open on the instance's security group and the laptop **dial-only**
(`network.listen_address: 127.0.0.1`), which is the point: the design must work
when exactly one side is reachable, and the pairing must be done over
`--verify` (fingerprint) because the path is public.

Evidence to capture per requirement, on the PR:

| requirement | evidence |
|---|---|
| R1 | `lop network status --json` on both; `ps` showing one relay per install; a session created while the relay is stopped and still listed after it starts |
| R2 | two networks on one device (`home-net`, `lab`), different member sets, `lop network ls --json` |
| R3 | the two terminal transcripts (A's prompt with the typed code, B's prompt), the outbox file's mode (`ls -l`), a replay attempt's refusal, and one deliberately wrong code |
| R4 | a handshake refusal with no reply frame (packet capture length), a `read`-role refusal, an `epoch_stale` teardown |
| R5 | the revocation sequence of §13.2 executed across the link, with the audit tail on the *remaining* device and the refusal on the removed one |
| R17 (revocation half) | `lop network disconnect` (peer tombstone + rotation), `lop network panic` (every peer untrusted, later connections refused), and the audit tail on both |
| regression | the `0 peers` topology: `lop sessions`, TUI boot, `lop exec`, `lop send` unchanged |

Also required by the operator's standing rules: this is a user-visible change
(new CLI group, new TUI slash commands, new desktop fields), so it needs a
**design round** (`D`-findings) with rendered TUI frames if `mesh-ui.md`'s
surfaces are part of the shipped PR, and a **UX round** for the pairing flow,
which is a new interaction. Both are the manager's call per PR; this document's
contribution is that the pairing flow has a scriptable, deterministic harness
(§13.2) so a reviewer or a designer can drive the real flow without hand-keying
a token.

---

## 14. Rejected alternatives

**Plain TLS with `CERT_NONE`.** Encrypts without authenticating: any peer
could open a link, and the app layer would have to re-do the authentication the
TLS layer claimed to provide. Rejected as strictly worse than doing the
authenticated handshake once.

**mTLS with a per-network CA** (the obvious "don't roll your own" answer). Two
fatal problems, and one that is merely expensive: (1) Python's `ssl` exposes no
verification callback, so the only way to express "a member of this network" is a
CA whose *private key* every member would hold, which cannot distinguish devices
and whose revocation is no better than the epoch check we need anyway; (2) it
gives no forward secrecy, because the CA and the device keys are static; (3) it
adds a cert lifecycle, two file formats and an `ssl` context per link, for a
KEX+AEAD we get in ~150 audited lines of composition over vetted primitives.
Rejected on all three, not merely on taste.

**`noiseprotocol` (PyPI) or `pynacl`.** A new dependency for a security-critical
path: `noiseprotocol` is pure-Python and unmaintained since 2021 (an unmaintained
implementation is a worse bet than a small composition over `cryptography`,
which the repo already ships and pins); `pynacl` is well maintained but adds a
compiled libsodium wheel to a base install (and gives `crypto_box`/`crypto_kx`,
which would still need a transcript-binding layer written by us). Rejected:
neither removes the part we have to write, and both add supply-chain surface.

**QUIC / `aioquic`.** Streams and multiplexing we do not need (we have small
messages and one logical stream per purpose), a heavy new dependency, and a
transport whose benefits (0-RTT, connection migration) are at odds with the
"no resume, re-handshake on every link" discipline that makes epoch revocation
reliable.

**Reusing the loopback control key as the peer credential.** The whole point of
the existing design is that the control key never leaves the machine
(spine §5.4). A shared key would also be unrevocable per device.

**Static-static Diffie-Hellman between device keys (no ephemeral).** Simpler, but
one leaked device key decrypts every recorded session forever. The ephemeral
X25519 costs one extra key exchange per link and buys per-link forward secrecy,
which is worth it for a channel that carries transcripts containing credentials
and customer data.

**Sharing the network secret with a new peer instead of an invite token.** A
naked secret paste is not single-use, not expiring, not bound to an inviter, and
not tied to a human confirmation. Rejected: R3 asks for all four.

**Copying `auth.db` (or any credential store) to a peer.** Reproduces the
measured PR-24 failure by construction (a rotating refresh token spent from two
devices). A5's brokering is the answer, and this document's contribution is that
the capability (`broker_credential`) and the op (`net_broker`) exist at the
transport layer.

**Hub/forwarding (A→C via B) in v1.** It is the difference between a full mesh
and a star, and it needs a byte-pipe op, hop-scoped routing headers, and a
decision about whether a forwarding member can read what it forwards. Deferred
deliberately: the 1-peer and 2-peer topologies of the spine's §10 are direct, the
`net_forward` envelope already reserves `hops`, and adding it later is a new op
plus a capability (`relay`), not a schema change. My recommendation is to build
it only when a real deployment needs a path through a third member.

**mDNS/Bonjour discovery of peers.** Attractive on a LAN and a privacy leak on a
coffee-shop network (it advertises the install and its device id to everyone on
the link); also a second discovery mechanism beside the endpoints the operator
already has. Rejected for v1; a future `--discover-mdns` opt-in key would be
additive.

**Absolute `expires_at` on invites instead of a `ttl_s` duration.** Rejected
because it is the one place a clock comparison across hosts would have entered
the design (§10.3).

**`lop network uninstall`** is not in the spine's §6 table, and the installer
writes a LaunchAgent that must be removable. See §16, Q1.

---

**STUN/TURN/ICE and hole punching** (RFC 5389 / RFC 8656). Rejected, unchanged
from §10.4: they need rendezvous infrastructure, UDP reachability the codebase has
never required, and a credential-issuing server, which is a control plane by
another name. `coturn` is permissively licensed and that is not the objection —
the objection is that the first three things each of them adds are exactly the
three assumptions this design refuses to make. Where a relayed path *is* wanted,
§10.4 names it: a blind, capped, reservation-based forwarder and nothing else.

**mDNS/Bonjour discovery.** See above: a privacy leak on an untrusted LAN, which
is the LAN the mesh is most likely to be used on.

**A CRDT as the session model** (Automerge, Yjs — both MIT). Rejected: a CRDT
exists to merge *concurrent* writers, and the single lease plus
`exclusive-move-v1` exist to make a second writer impossible by construction;
`Transcript.compact_file` also rewrites the transcript in place, which a CRDT
cannot represent. The one idea worth borrowing — the state vector — is already
ours in single-writer form (`mesh-prior-art.md` §5, `mesh-session-mobility.md`
§7.3).

---

## 15. File-by-file change list (for the coder)

New package:

| file | contents |
|---|---|
| `local_operator/network/__init__.py` | vocabulary re-exports, mirroring `session/runtime/__init__.py:1-38` |
| `local_operator/network/types.py` | `MESH_PROTOCOL_VERSION`, `NET_OPS`, `PeerRecord`, `NetworkRecord`, `MemberRecord`, `InviteRecord`, `LinkContext`, `Granted`, `Refusal`, `ROLE_CAPABILITIES`, `CAPABILITIES`; stdlib-only |
| `local_operator/network/identity.py` | `load_or_mint()`, `device_id_for(public_key)`, `rotate()`, `rotation_statement()` |
| `local_operator/network/store.py` | `list_networks()`, `load(id)`, `save(record)`, `load_secrets(id)`, `save_secrets()`; atomic, 0600/0700 |
| `local_operator/network/registry.py` | `record_path/publisher/publish/unpublish/scan` via `session.runtime.registry` + `PEERS_RUN_DIRNAME` |
| `local_operator/network/crypto.py` | `link_id()`, `transcript_hash()`, `link_keys()`, `sas()`, `fingerprint()`, `aead_seal/aead_open()` |
| `local_operator/network/handshake.py` | `DialerHandshake`, `ListenerHandshake`, verify order of §6.2 |
| `local_operator/network/link.py` | `PeerLink` (reader/writer, keepalive, two-class queues, backoff, dedupe) |
| `local_operator/network/server.py` | `RelayServer` (listener, link registry, `_dispatch` → `Authorizer.check`) |
| `local_operator/network/authorize.py` | `Authorizer`, `OP_CAPABILITY`, `INNER_OP_CAPABILITY` |
| `local_operator/network/membership.py` | `admit()`, `remove()`, `leave()`, `rotate_epoch()`, `apply_epoch()`, `reconcile()`, `apply_device_rotation()`, `panic()` |
| `local_operator/network/pairing.py` | `mint_invite()`, `redeem()`, `confirm()`, `abort()`; SAS prompt/code-entry |
| `local_operator/network/catalog.py` | `local_rows()`, `fetch_peer()`, `aggregate()`, cache read/write |
| `local_operator/network/audit.py` | `AuditLog.record()`, rotation |
| `local_operator/network/control.py` | loopback server + client (the §2.5 op set) |
| `local_operator/network/service.py` | `amain()`, `render_plist()`, `plist_path()`, `install()`, `uninstall()`, `service_action()`, `status()`, `health()`, `refresh_plist_if_stale()`, `is_supported()` |
| `local_operator/network/cli.py` | `add_parser()`, `main()`; stdlib-only registration |

Edits:

| file | change |
|---|---|
| `local_operator/session/runtime/types.py` | add `PEERS_RUN_DIRNAME = "run/peers"` (L~280) with the namespace comment of §2.6 |
| `local_operator/session/protocol.py` | add `"another-machine"` to `RuntimeLocality` (L128) and rewrite the comment |
| `local_operator/tui/app.py` | handle `"another-machine"` at L13630's label path |
| `local_operator/cli.py` | wire `network.add_parser` beside the tunnel/secrets groups (~L459/466); add `--peer`/`--all-peers` to `sessions` and `exec`; dispatch ladder in `main()` |
| `local_operator/settings_io.py` | the `Network` section + 13 settings (§11), Scope `NEW_LAUNCH` |
| `local_operator/tools/registry.py` | `network` entry in `TOOL_BUILDERS` (L32) and `DEFAULT_TOOL_NAMES` (L70) |
| `local_operator/guides/network/GUIDE.md` | new guide (§12.5) |
| `local_operator/slash_commands.py` | `/network …`, `/peers` (registry at L67-813; `SlashCommand` fields at `tui/autocomplete.py:173`) |
| `local_operator/server/routes/desktop_sessions.py` | `peers=` query, `locality`/`peer` row fields, `degraded` reasons |
| `local_operator/server/routes/*` (new `desktop_peers.py`) | `GET /v1/desktop/peers` |
| `local_operator/server/desktop.py` (capabilities) | `features.peers: 1` |
| `docs/DESKTOP_API.md` | document `features.peers`, the row fields, the new route |
| tests | `tests/unit/network/**` (§13.1), `tests/e2e/mesh/{fake_peer,mitm}.py` (§13.2) |

**Suggested PR split** (each independently reviewable, each closing named
requirements): (1) identity + store + registry + records — R2, A2, A3's key half;
(2) crypto + handshake + link + server + authoriser, with the fake-peer tests —
R1, R4, A1; (3) pairing (mint/redeem/SAS/admission) + CLI group + launchd service
— R3, R1's supervision; (4) membership: epoch, rotation, reconcile, panic,
disconnect + audit — R5, and the mechanism R17 is built on (R17 itself is owned
by `mesh-incident-response.md`); (5) catalogue + listing +
desktop contract — R6's data half; (6) guide + tool — R19.

---

## 16. Open questions, each with my recommendation

1. **`lop network uninstall` was missing from the spine's CLI table.**
   **Resolved in convergence round 1** — the row is in the spine's §6 table now,
   with this document's semantics: `uninstall` is `lop mobile uninstall`'s shape
   (`mobile/install.py:332` — bootout the LaunchAgent, remove the plist, report
   each step). The installer still writes a LaunchAgent, so the command that
   removes it is part of the surface, not a convenience.

   **Purge scope, decided in round 1 (manager's decision; spine §12).** `--purge`
   deletes the **network records, invites and outbox entries for the networks
   being uninstalled** — the analogue of mobile's Keychain password, and the only
   command that can make a device forget a network it can no longer reach.
   Deleting the **device identity keypair** is a separate act behind its own
   flag, **`--purge-identity`**, and it is gated:

   > **Safety rule `purge_identity_needs_a_named_tty_confirmation`.** Deleting
   > this device's identity keypair requires `--purge-identity` **and** an
   > interactive TTY confirmation that names every network still known to that
   > identity. A run without a TTY refuses `--purge-identity` outright and says
   > which flag does work.

   The reason is that the keypair is not a per-network resource: `device_id` is
   the fingerprint every *other* network addresses this device by (§3.1), so a
   single flag that removed the keypair alongside one network's records would let
   a one-network action silently destroy identity that the networks the operator
   was not thinking about still depend on — and unlike a record, the private key
   is unrecoverable. Naming the networks in the prompt is what makes the blast
   radius visible at the moment of the act. This is the rule `net_trust` already
   applies to its own irreversible operation (§3.1: re-admission refuses without
   a TTY confirmation), applied to the wider of the two blast radii rather than
   to the narrower one.
2. **Does `lop network invite` need to print the token at all?** The token is a
   credential and a transcript is replayed to the provider every turn.
   **Recommend: file-first** (§5.1) with `--print` refused on a non-TTY and
   refused with `--json`. If the operator wants to paste the token into a chat
   client, they read the file or run with a TTY.
3. **Should a non-admin member be allowed to rotate the epoch?** R17 wants "any
   member can raise the alarm"; a non-admin rotation would force every peer to a
   new secret the raiser chose. **Recommend: no rotation without `admin`**; a
   non-admin `net_panic` marks every receiver untrusted without carrying a
   secret (§8.2), which is the safe half of the same signal.
4. **Should the peer listener be able to bind a Unix socket for same-host
   peers?** A second install on the same host (the QA topology of §13.2) could
   use a UDS with the same handshake, which would (a) make the harness trivial
   and (b) let a same-host pair run with no bind at all. **Recommend: yes, later
   and additively** — a `unix:/path` endpoint in `listen.advertised`, refused by
   default empty allow-list. It changes no frame.
5. **`RuntimeLocality` member name.** I recommend `"another-machine"` because
   that is the name the existing comment reserved, and a peer-qualified value
   would put authority-shaped data (a device id) in a field consumers compare
   against literals. **Recommend: `"another-machine"`**, with the device id
   carried in the row's `peer` field (§9.3).
6. **Expiry for tombstones.** Keep the tombstone forever (a grew-with-time file)
   or prune after N days? **Recommend: prune the tombstone's *row* after 90 days
   but keep the audit record**; and keep a compact `removed_ids` set forever in
   the record, which is what actually prevents a re-admission at a burned id.
7. **What does `lop network join` print on success?** **Recommend:** network
   name, id, epoch, member count, and the two commands that matter next
   (`lop network peers`, and how to move a session to the new device), so the
   "streamlined pair" requirement ends with a next step rather than a receipt.
8. **Do we need a per-network `name` uniqueness rule?** Two networks may share a
   display name. **Recommend: no uniqueness, always resolve by id, and make
   `lop network <name>` disambiguate by prompting when ambiguous** — a rename
   must never need to rewrite member records.
9. **Should the relay publish a record at all when it has no networks?**
   **Recommend: yes** (with `networks: []`), so `lop network status` has an
   answer and the UI can distinguish "relay running, no networks" from "relay
   not installed".
---

## 17. Convergence round 1 — what changed, and why

Round 1 of the cross-document convergence pass changed this document in six
places. Each is a correction with a named reason; nothing here is a preference.

1. **The rotation no longer hands the new secret to the device it removes**
   (§8.1 step 3, §8.3). The finding and its argument live in
   `mesh-incident-response.md` §2.3.4 and §8 Q1, which is the canonical statement
   of the defect; this document now enforces it at the **sender**, where it
   cannot be argued away, as two named invariants
   (`epoch_secret_withheld_from_removed`,
   `epoch_outbox_holds_no_secret_for_a_removed_member`) with three named tests in
   §13.1. The offline queue is fixed in the same breath, because a queued frame is
   a secret written to disk rather than a secret sent. **The panic frame keeps its
   design** — secret to every recipient — per that document's Q1; §8.3 says so
   where a reader meets it.
2. **`net_invite` gains an optional device binding** (`--device` / `bound_device`,
   §5.1, §5.2, §4.2, §5.4), so an invite that reaches a second device is refused
   `pairing_refused {cause: "wrong_device"}` and consumed, instead of being a
   bearer token for the network's key material. Raised by
   `mesh-incident-response.md` §3.2 / §8 Q3.
3. **A bounded secret lifetime** (§4.2 `epoch_minted_at`, §4.3, §8.4, §11's
   `network.epoch_max_age_s` = 30 days). §8.4 covers the member you removed; it
   did not cover the still-member that never comes back, which prior art
   (Nebula, Tailscale, OpenZiti — `mesh-prior-art.md` §2) answers with a bounded
   credential lifetime rather than with better re-evaluation. This is the one
   change in the round that is a *schema shape* rather than a rule: the age sits
   beside the key, which is what makes it enforceable by every reader of the key.
4. **One capability vocabulary, and three new ops that add no capability.** The
   three session ops mobility needs (`net_session_create` → `prompt`,
   `net_session_engage` → `view`, `net_session_stop` → `stop`) are rows in §6.4,
   and §7.1 states that they introduce no new names.
   `mesh-compute-pool.md` §3.3's invented `broker:request` / `broker:grant` /
   `member:admin` are deleted there; the names in §7.1 are the only ones.
5. **`lop network uninstall [--purge]` is in the spine's CLI table** (§16 Q1),
   because this design writes a supervised LaunchAgent and a plist that can be
   installed but not removed is an incomplete lifecycle. Its purge scope was
   narrowed on the manager's decision: `--purge` covers the networks being
   uninstalled (records, invites, outbox), and the device identity keypair needs
   `--purge-identity` plus a TTY confirmation naming every network still known to
   that identity.
6. **`stream_open`'s pass-through mode is specified rather than implied** (§2.5),
   which is what lets a remote viewer reuse `AttachClient` instead of growing a
   second per-op relay path (`mesh-session-mobility.md` R-IF-1).

**Anchors re-measured in the same pass** (the documents cite line numbers, and a
stale one is a defect a reviewer pays for): `serving.py:3624-3630` →
`session/runtime/serving.py:4087-4129`; `server.py:1918-1923` →
`session/runtime/server.py:2700` (parse) and `:4359-4426` (dispatch), with the
`ClientLocality` docstring at `session/runtime/types.py:164-172`; and `types.py:172`
disambiguated to `session/runtime/types.py:172` wherever it was bare.
`mesh-session-mobility.md` §0 records the same `serving.py` drift independently,
so the two documents now agree on the number.

---

## 18. Open questions (continued)

10. **Is the 6-digit SAS enough, given §5.3's 2²⁰ grinding bound?**
    **Recommend: keep 6 digits as the default (it is the industry standard and
    the invite-key binding removes the no-token attacker entirely), always print
    the 160-bit fingerprint in the same panel, and make `--verify` mandatory when
    an endpoint is not a private address.** If the operator would rather not rely
    on humans reading a fingerprint, the alternative is `--verify` mandatory
    *always*, at the cost of one paste per pairing.
