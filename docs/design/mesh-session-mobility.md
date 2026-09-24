# Mesh session mobility — remote sessions, move/fork, lease handoff, delete/archive on a peer

Status: **design, pre-implementation**. Detail design hanging off
[`mesh-network.md`](mesh-network.md) §3–§4 per that document's table.

| | |
|---|---|
| Owns | **R7, R8, R9, R10, R11, R12**, decision **A4** |
| Implements on the way (not the owner) | **R6** — the federation and the projection that make one session list possible; the requirement itself is owned by `mesh-ui.md`, per the spine's table · **R18** (the session-plane audit events it names) · **R21/R22** (§7's sync primitive, owned by `mesh-compute-pool.md`) |
| Owned elsewhere, referenced here | The relay, the link, device identity, pairing, revocation, **the capability model**, the federated catalogue and its cache, and the relay's local control surface → `mesh-transport-identity.md` · credential broker mechanics → `mesh-credentials.md` · audit retention numbers → `mesh-incident-response.md` · rendered surfaces → `mesh-ui.md` · the agent playbook → `guides/network/GUIDE.md` |
| Naming authority | `local_operator/network/**` (that document's §2.3 lays out the package and **reserves `projection.py` for this one**) · `net_*` peer ops (its §6.4) · capabilities (its §7.1) · the `peer`/`locality` row fields and `features.peers` (its §9.5) |

Every line anchor below is a real anchor in the branch this document was written
on (`feat/mesh-network`, cut from `origin/main` `a7e6b9bd`). Where the spine's
recon notes and this document disagree about a line number, **this document is
the one measured against the tree** — the notes for the `route_shared_slash` →
`serving.py` chain in particular are ~450 lines stale, and the real
`ServingSessionHandle.run_slash_authoritative` is `session/runtime/serving.py:4087`,
not `:3624`.

**Relationship to the sibling designs.** This document was written against the
transport design as it stands, and it *consumes* rather than redefines it: the
relay and its loopback control surface, the `net_*` peer op vocabulary, the
capability model, `net_catalog` plus `network/catalog.json`'s TTL cache, and the
`locality`/`peer` catalogue row fields are all that document's, and are referenced
here by its names. Two interface requests are made *to* it, each stated where it
arises and collected in §13: §2.2's three new peer op names (with their capability
rows), and §3.2's pass-through stream mode for the viewer's forwarded stream. Where this document and the transport document
disagree about a repo line number, this one is measured against the tree
(`feat/mesh-network` @ `a7e6b9bd`).

---

## 0. The one-paragraph summary

A session is owned by exactly one runtime on exactly one device, and that fact is
written down durably on the owning device. A remote session is the *same object*
as a local one — the same `AttachedSession` facade, the same control protocol, the
same command surface — with one injected collaborator (`SessionOwner`) that
answers three questions the local case answers from `run/mobile`: where the owner
is, how to start one, and how to dial it. Everything a viewer does therefore works
remotely by construction rather than by enumeration, and the exceptions are a
short, named list with a reason each (§4). Mobility is *fork + retire* fenced by
the existing `exclusive-move-v1` protocol, with a two-phase handoff whose single
decider is the current owner, so no crash can produce two live writers or lose the
only copy. Sync is the same copy primitive without the retirement, which is what
makes R22 an instance of R11 rather than a second mechanism. Nothing here invents
a wire: the view of a remote session is the transport's forwarded stream
(`stream_open` / `net_forward`) carrying the control vocabulary unchanged.

---

## 1. The ownership model

### 1.1 The invariant

> **INV-1.** For every session id `S` there is at most one device `D` whose session
> store holds `S` as a session, and on that device at most one runtime process holds
> `S`'s transcript lease. That device is `S`'s **owner** and the only device that may
> turn a turn, append a row, retire the runtime, delete the session, or archive it.

Two invariants the existing code already enforces and this design does not weaken:

* **Local single writer.** One runtime per session, arbitrated by the transcript
  lease (`local_operator/session/runtime/launch.py:1-40`, "the invariant: at most
  one runtime per session, ever"), carried by `sessions/<id>/.session.pid`
  (`session/retention.py:220` `claim_session`, `:254` `release_session`) and read by
  `resume.live_runtime_pid` (`resume.py:1147`).
* **One writer of the control socket per session kind.** One `daemon` connection
  plus up to `ATTACH_MAX_CLIENTS = 4` attach clients
  (`session/runtime/types.py:179`), with the exclusive-move fence refusing new
  attach registrations while a move is committed
  (`session/runtime/server.py:2725`).

INV-1 extends the first of those across devices. It is *not* enforceable by the
local lease: the marker is a local file, a pid is meaningless on another host, and
two hosts cannot arbitrate a file they do not share. So the durable carrier is the
ownership stamp plus the handoff journal (§6.3), and the *authority* is the owner's
own answer, re-checked on every request (§2.2).

### 1.2 The three durable carriers

| Carrier | Where | Written by | Read by |
|---|---|---|---|
| **Ownership stamp** `sessions/<id>/mesh.json` | the owner's session directory | the owner, atomically, before the directory is published | the owner's resolver; every peer's catalogue builder |
| **Handoff journal** `<config>/network/pending-move.json` | the owner's mesh state | the owner, at `prepared` / `handing-off` / `committed` (§6.3) | the owner's `engage_runtime` guard, its reconcile pass, the requesting device's retry |
| **Tombstone index** `<config>/network/tombstones.json` | every device that has handed a session away | the owner at the commit of a `move` | the owner's resolver, listing and resume paths |

All three live under `<config>/network/` — the package's own root
(`network/catalog.json` sets the precedent, `mesh-transport-identity.md` §4.1/§9.4)
— except the stamp. `mesh.json` is the only one that lives inside a session directory, and it is
**additive and optional**: its absence is the statement "no mesh has ever governed
this session", which is exactly today's behaviour for every session on every
install (R16's zero-peer regression, spine §10 topology 0).

```json
{
  "version": 1,
  "session_id": "9f3ac1e0b7d2",
  "network_id": "n_7Yb3kQ",
  "home_device": "dev_a1b2c3d4e5f6",
  "placement": { "mode": "local", "policy": "pinned" },
  "origin": { "kind": "user", "source_device": "" },
  "created_at": 1758230400.512,
  "stamp_revision": 3
}
```

Field notes, all of which the resolver depends on:

* `home_device` — the owning device id (a device-key fingerprint, spine A3). This
  is the *only* field that decides INV-1's routing question.
* `placement` — spine A8's field, carried here so it survives the runtime exiting
  (an idle remote session has no record and the catalogue must still place it).
  `mode` is `"local"` | `"peer"` | `"pool"`; `pool` is reserved and unproduced in
  this pass; `policy` is `"pinned"` | `"prefer-remote"` | `"cost-capped"`.
* `origin` — how this session got here: `{"kind": "moved", "source_device": "<id>",
  "source_session_id": "<id>"}` for a `move` (id preserved), `{"kind": "fork",
  "source_device": "<id>", "source_session_id": "<id>"}` for a `--keep` copy or a
  local `fork_session` promoted across the mesh. The source id is what lets a row
  say "copy of <name> from <device>".
* `stamp_revision` — bumped on every rewrite; a peer's cached catalogue row carries
  the revision it saw, so a stale row is detectable rather than silently trusted.

**Why a sidecar and not a field in the discovery record alone.** The record
(`SessionRecord`, `session/runtime/types.py:712`) exists only while a runtime does;
an idle session on a peer must still list as remote, so placement cannot live only
on the record. The record *also* carries placement (additively, §5.1) because a
live view must not pay a second read.

**Required change: `mesh.json` joins the fork deny-list.** `fork.fork_session`
copies an explicit allow-list of sidecars (`COPIED_SIDECARS`, `fork.py:117`) and
refuses an explicit deny-list (`EXCLUDED_SIDECARS`, `fork.py:137`). `mesh.json`
is a **new** file in the session directory, so the same change that introduces it
adds it to `EXCLUDED_SIDECARS`; a fork is a new session on the machine that made
it, and a copied ownership stamp would make the fork claim its parent's
`home_device` and `placement` — a live session advertising itself as owned by
another device, which is exactly the row INV-1 (§1.1) then routes to. This is the
one place the requirement is enforceable, which is why it is written here rather
than only in `mesh-compute-pool.md` §8.1, which records the same requirement from
the pool side. Test: `test_fork_never_inherits_a_mesh_stamp` (§11.2), beside
`mesh-compute-pool.md` §9's `test_placement_pool_mode.py` row.

**Why not inside `origin.json`.** `origin.json` (`resume.py:68`) is a
migration/discrimination axis read by `USER_ORIGINS` (`resume.py:102`) and the
origin-verdict cache, with retired values and a scan sentinel around it
(`ORIGIN_SCAN_SENTINEL_NAME`, `resume.py:272`). Mesh ownership is a different
question with a different lifecycle; overloading one file makes every writer
arbitrate over both (the same argument `session/archived.py` on
`feat/session-archive-delete` makes for keeping the archive index out of
`sidebar_pins`).

### 1.3 Ownership and the existing `locality` vocabulary

`RuntimeLocality = Literal["this-process", "this-machine", "unknown"]`
(`session/protocol.py:128`) has, today, **no cross-host member**, and its docstring
gives the reason: *"a cross-host runtime cannot occur, so naming it would re-create
the dead axis `is_remote` named"*. A peer-backed session makes that reason false, and
the type change is owned by `mesh-transport-identity.md` §12.2: **add
`"another-machine"`**, rewrite the comment to state the condition under which it can
occur, and add a guard test that every `==` comparison over the union handles it.

I originally preferred not to widen the type and to answer `"unknown"` from a remote
facade, on the grounds that the arm is already the conservative answer. The transport
document's reading is better, and I adopt it: the union is three-valued precisely so
the answer is honest rather than merely safe, and here the answer is *known* — the
runtime is on another machine, which is strictly more information than "cannot be
proven". The trade-offs that come with it, and my part of the work:

* `SessionProtocol.owns_runtime` (`session/protocol.py:157`) is **False** for a
  remote session — correct and load-bearing: the loop and the transcript are the
  owner's, so this process must not end the session in-process, must not let an
  in-process abort stand in for the owner's, and must not auto-name.
* `outcome_is_synchronous` (`session/protocol.py:174`) is **False** — prompts return
  on the owner's durable-admission ACK, exactly as for a local attach
  (`attached.py:6911`).
* `runtime_locality` (`session/protocol.py:191`) becomes **`"another-machine"`**
  for a remote placement. `AttachedSession.runtime_locality` (`attached.py:6922`)
  currently returns `"this-machine"` unconditionally, on the strength of the
  loopback-only invariant — which is exactly the assumption the mesh breaks, so the
  property must become placement-dependent (`LocalOwner` → `"this-machine"` as
  today, `RemoteOwner` → `"another-machine"`). Its docstring's *"this never returns
  `unknown`"* sentence stays true. The one consumer that must gain a case is
  `App._session_runs_elsewhere` (`tui/app.py:13810-13874`): its `"this-process"`
  early-out (`:13825`) is a negative test so it is safe by default, and its
  registry-scan fallback (`:13865-13870`) already returns "elsewhere" for a session
  with no local record — but the **label** it draws must name the device (§9.2), and
  the guard test lives in `tests/unit/session/test_viewer_protocol.py`'s family
  (which already fails on an unvalidated session flag of that shape,
  `session/protocol.py:152-154`).
* `AttachedSession` must **never** be `surface="terminal"`-with-armed-takeover for a
  remote placement. `_can_go_cold` is `surface == "desktop"` today
  (`attached.py:953`), and the takeover factory is the CLI's `session_factory`
  closure (`cli.py:8464`), i.e. a **local in-process session** built from a local
  transcript. On a remote placement, owner loss must go cold and report the
  device — never take the lease — so the mesh passes `_no_takeover`-shaped factory
  (`server/utils/desktop_sessions.py:161`) and forces `_can_go_cold = True`.

This is the first of the design's explicit "refusals that are really protections":
a remote viewer that could take over would manufacture a second writer for one
transcript (INV-1's counterexample) the first time a link blipped.

---

## 2. Resolving the owner

### 2.1 Locally

```python
# local_operator/network/projection.py  (new; the module the transport design reserves)
#   SessionOwner / LocalOwner live in local_operator/session/owner.py instead, so the
#   facade's seam imports without the network package (see §3.1).
class OwnerLocation(NamedTuple):
    kind: Literal["local", "remote", "unreachable", "unknown"]   # the ROW's
    # classification, not `RuntimeLocality`: "unknown" here means "the peer has
    # not answered yet", and the `"another-machine"` widening of §1.3 does not
    # touch this union (nor does it replace this value)
    session_id: str
    device_id: str            # "" when local
    device_name: str
    record: SessionRecord | None   # local: the run/mobile record; remote: the peer's facts, re-addressed at our relay
    reason: str = ""          # one sentence, user-facing, for unreachable/unknown

def resolve_owner(
    session_id: str,
    *,
    config_dir: Path,
    catalog: PeerCatalog | None = None,   # network/catalog.py's cache+fan-out; None ⇒ no network
) -> OwnerLocation: ...
```

Resolution order, first match wins, and **it is the same order the transport's
aggregation API already uses** (its §9.1/§9.3):

1. **Local store is authoritative for "is it here".** `sessions/<session_id>/` exists
   and its `mesh.json` is absent **or** names `home_device == self` → `local`, with
   `record` from `find_runtime_record` (`mobile/attach_client.py:686`) — i.e. today's
   path, untouched.
2. **Tombstone.** `network/tombstones.json` holds the id → `remote` pointing at the
   recorded device if that device is a current member, else `unknown` with the
   sentence *"This conversation was moved to <device>, which is no longer in the
   network."*
3. **The peer catalogue.** The row arrives from `network/catalog.py` — the local
   `run/mobile` scan plus `net_catalog` fan-out with the 24 h TTL cache
   (`mesh-transport-identity.md` §9.2/§9.4) — and its `peer` names the device. A row
   served from cache is `unreachable` with `age_s`; the UI never renders a cached
   row as live (that document's rule, and the reason `lop sessions` can show
   `unreachable` rows at all).
4. **Nothing** → `unknown` with *"No device in this network holds that
   conversation."* (Not `local`: a mesh viewer must not engage a successor for an id
   nobody owns — that is how a moved-away session would be resurrected locally.)

There is deliberately **no targeted `locate` op** (`net_session_locate`) (a targeted "who owns this
id?"). It would be a second, redundant answer to a question the catalogue already
answers, and its only unique case — a session the peer does not list because it was
moved away — is answered by the owner's tombstone through `net_session_move`'s
`status` phase (§6.3). The catalogue fan-out is also what makes the `lop sessions`
regression (its §9.1: "`run/mobile` only, no behaviour change") hold: resolution for
a device with no network never leaves step 1.

A `remote` answer is **always re-verified on the owner** before it acts: the owner
answers authoritatively for its own store, and a request naming a session it does
not own is refused with `session.ownership_refused` (audited, R18) and a sentence
that names the device that does, when the owner itself can tell (§2.2). This is what
makes a stale cache a retry and never a wrong execution.

### 2.2 Over the mesh: the op surface this design requires

Four names already exist in the transport design and are used here unchanged
(`mesh-transport-identity.md` §6.4): **`net_forward`** (the carrier that moves one
`ControlOp` frame across the link, authorised by the inner op's capability),
**`net_catalog`** (a peer's session rows), **`net_session_move`** (`move`), and
**`net_session_lifecycle`** (`delete`), plus the **`net_sync`** name that document
*reserves* for R22's cadence and pre-spin-down sync (`view`) — §7 fills it in.

This design adds **three** names — `net_session_create`, `net_session_engage` and
`net_session_stop` — which must appear in `NET_OPS` (`network/types.py`) and in
`OP_CAPABILITY` (`network/authorize.py`) so the transport's totality test
(`tests/unit/network/test_op_capability_table.py`) stays satisfied, and it fills in
the phases of the two op names that document reserves (`net_session_move`,
`net_sync`):

| op | capability | purpose | frame (literal) |
|---|---|---|---|
| `net_session_create` | `prompt` | create a session **on** the peer (R8): mint id, claim the directory, stamp `mesh.json`, engage, admit an optional first prompt | `{"op":"net_session_create","req":11,"cwd":"/home/ubuntu/work","model":{"provider":"anthropic","model_id":"claude-sonnet-4-5"},"name":"","prompt":"port the parser","images":[],"origin":"user","locality":"remote"}` → `{"op":"ack","req":11,"detail":{"session_id":"9f3ac1e0b7d2","admitted":true,"duplicate":false,"record":{…}}}` |
| `net_session_engage` | `view` | make an owner exist on the peer (warm) so a cold session can be viewed or acted on; **carries no prompt** | `{"op":"net_session_engage","req":12,"session_id":"9f3ac1e0b7d2","cwd":"","warm":{"initial_model":null,"model_selection_override":false},"locality":"remote"}` → `{"op":"ack","req":12,"detail":{"engaged":true,"detail":"runtime joining"}}` |
| `net_session_stop` | `stop` | run the **peer's own** kill-switch implementation (`lop stop <remote>`, `/stop <target>`) | `{"op":"net_session_stop","req":13,"session_id":"9f3ac1e0b7d2","mode":"graceful","locality":"remote"}` → `{"op":"ack","req":13,"detail":{"rung":"stop-op","outcome":"stopped","pid":91234,"detail":"…"}}` |
| `net_session_move` phases *(name is the transport's)* | `move` | `phase: "status"｜"prepare"｜"ready"｜"commit"｜"done"｜"fork"｜"copy"`, `mode: "move"｜"copy"` (§6.3, §6.2) | see §6.3's literals |

`net_session_lifecycle` (`delete`) carries `action: "archive"｜"restore"｜"delete"`
(§8); `net_sync` (`view`) carries `phase: "plan"｜"fetch"｜"flush"` (§7).

Two properties of that vocabulary this design depends on, both already stated there:

* **A peer acts only on sessions it owns.** Its §7.2 "session scope" rule — the id
  must appear in *this* relay's catalogue and the owner must be *this* device, else
  `not_authorised` — is INV-1 enforced at the chokepoint, so an ownership question
  answered from a stale cache costs a refusal and never a wrong execution (§2.1).
* **`locality` is never taken from the frame that arrived.** The peer-side relay
  dials the owner's control socket and declares `locality: "remote"` in *that* auth
  frame (its §7.4), and it refuses any forwarded frame claiming `"local"` as a
  protocol error (spine §5.8). The property is enforced at the one process that can
  enforce it.

---

## 3. The remote session projection

### 3.1 Shape: one facade, one injected collaborator

The projection is **not** a new facade class and **not** a per-session proxy
process. It is the existing `AttachedSession` (`session/attached.py:886`) with the
three owner-facing seams routed through a collaborator. The reason is measurable:
`AttachedSession` is ~8,300 lines whose display, gate, transcript-fold and history
halves are already transport-agnostic — everything reaches the owner through
`self._client`, an `AttachClient` (`mobile/attach_client.py:815`) speaking the
control vocabulary. A second facade would be a second front-end path, which spine
§3 layer 3/4 explicitly forbids.

```python
# local_operator/session/owner.py  (new, import-light: no engine, no asyncio import at module level)
class SessionOwner(Protocol):
    """Where a session's runtime lives and how to reach it. Three questions."""

    placement: SessionPlacement                  # session/placement.py (§5.2)

    def locate(self) -> tuple[SessionRecord | None, int | None]:
        """The owner's record and pid, or (None, pid) / (None, None).
        Same contract as ``mobile.attach_client.find_runtime_record``, whose
        docstring is the specification (attach_client.py:686)."""

    async def engage(self, *, cwd: str, warm: WarmErrand) -> None:
        """Make an owner exist, or join one that is starting."""

    def make_client(self, **callbacks: Any) -> AttachClient:
        """The frame-level client for this owner, already configured."""

class LocalOwner(SessionOwner):   # the default; exactly today's code paths
    ...
class RemoteOwner(SessionOwner):  # network/projection.py
    ...
```

| Seam in `AttachedSession` | Line today | With `SessionOwner` |
|---|---|---|
| The bind's discovery read | `attached.py:3388` `find_runtime_record` | `self._owner.locate()` |
| The bind's engage | `attached.py:3299` `engage_runtime(...)` | `await self._owner.engage(...)` |
| The dial's client construction | `attached.py:3921` `AttachClient(...)` | `self._owner.make_client(...)` |

Those three, plus the takeover factory (§1.3) and the `cold()` seed (§3.4). **No
other member of `AttachedSession` learns about the mesh** — that is the falsifiable
statement a reviewer can check, and it is why the same facade serves the TUI, the
phone daemon, `lop attach`, the desktop bridge
(`server/utils/desktop_sessions.py:1049` `acquire`) and `lop --resume` unchanged.

`AttachedSession.__init__` gains one keyword-only parameter, defaulted:

```python
def __init__(self, *, config_dir, session_id, takeover_factory, surface="terminal",
             owner: SessionOwner | None = None, seed: SessionSeed | None = None) -> None:
    ...
    self._owner: SessionOwner = owner if owner is not None else LocalOwner(config_dir)
```

Callers that must supply the remote form are exactly the ones that already know the
placement: the TUI's session factory when a catalogue row is selected, the CLI's
resume/viewer factory, `lop exec`, and the desktop bridge's construction site
(`server/utils/desktop_sessions.py:1080`). A caller that supplies nothing gets
today's behaviour, byte for byte (R16 topology 0).

### 3.2 Transport: viewer → local relay → peer link → peer relay → owner runtime

The transport design already names this seam, and this section implements the far
side of it: its §2.5 local control ops **`stream_open {peer, session_id}`**,
**`stream_send {stream, frame}`** and **`stream_close {stream}`** are for exactly
this viewer, and its §12.2 adds that *"a stream's responses are `EventOp` frames so
a viewer can reuse the existing attach client"*. On the peer side, the relay's dial
to the owner's control socket goes through **`Authorizer.dial_local`**, its §7.2
*"ONE way this package opens a local control socket … called only from the
dispatcher, after `check` has returned"*.

```
   THIS DEVICE                                        PEER DEVICE
 ┌────────────────────────────────┐              ┌────────────────────────────────┐
 │ TUI / desktop / phone          │              │ relay (peer side)              │
 │   │ RemoteOwner.make_client()  │              │   │ Authorizer.dial_local()    │
 │   ▼                            │              │   ▼ 127.0.0.1:<control_port>   │
 │ RemoteSessionClient            │              │   auth {key: <owner's         │
 │   dials 127.0.0.1:<relay port> │              │         control_key>,         │
 │   auth {key:<relay control_key>│  stream_*    │         client:"attach",      │
 │         client:"cli"}          │──────────────┼──►      locality:"remote"}    │
 │   then stream_open{peer,       │  peer link   │ session runtime                │
 │        session_id}; afterwards │  (net_forward└────────────────────────────────┘
 │   the connection carries session│   carrier)
 │   frames, in BOTH directions   │
 └────────────────────────────────┘
```

**Interface requirement (R-IF-1) on `network/control.py`.** After `stream_open`
succeeds, that control connection becomes a **pass-through** for the named stream:
every subsequent line the viewer writes is a session frame (`ControlOp` shape) and
every line the relay writes back is a session frame (`EventOp` shape), with no
wrapper. This design needs it because `RemoteSessionClient` is an `AttachClient`
subclass whose reader (`_pump`, `attach_client.py:1049`) already parses exactly that
shape and correlates `req` ids; wrapping would force a second parser into the
transport half of the client and put a frame-shape translation on the hot path for
no benefit. `stream_send {stream, frame}` remains the **multiplexed** form for a
front end that wants several sessions on one socket; this design does not use it,
because one connection per viewer is required anyway (§3.5) and because the relay's
loopback control socket is keyed and local (its §2.5), so N connections cost
nothing a multiplexer would save. The requirement is collected in §13 as Q-IF.

**The client.** `network/projection.py`:

```python
class RemoteSessionClient(AttachClient):
    """An AttachClient whose endpoint is the LOCAL relay, not the runtime."""

    def __init__(self, *, relay: RelayEndpoint, remote: RemoteSessionFacts, **auth: Any) -> None: ...

    async def connect(self, record: SessionRecord, session_id: str) -> None:
        # 1. open_connection("127.0.0.1", relay.control_port)   — the RELAY's port
        # 2. {"key": relay.control_key, "client": "cli"}          — the local control key
        # 3. {"op":"stream_open","req":1,"peer": f"{remote.device_id}", "session_id": session_id}
        #    → ack; the connection is now a pass-through session stream (R-IF-1)
        # 4. write the SAME auth frame AttachClient.connect writes today
        #    (attach_client.py:988-1015): {"key": <owner-side key, supplied by the
        #    peer relay — see below>, "client":"attach", "locality":"local", ...}
        #    The relay replaces "key" with the owner's control_key on the peer side
        #    and sets locality="remote" in ITS dial; our own frame never carries the
        #    owner's key.
        # 5. first frame back must be welcome|projection with
        #    projection.session_id == session_id — the identity check is unchanged
        #    (attach_client.py:1038-1042)
```

Two boundary facts the implementation must respect, both of which the diagram
encodes: the viewer sends its auth fields (events, frontend_state, display_window,
`slash_consumers`, surface) **per connection**, because those are the per-connection
facts §3.5 depends on; and the peer relay is what supplies the `key` field the
owner-side dial needs, so `RemoteSessionClient` must send a **placeholder** (the
local relay strips and replaces it) or omit it — the sequencing is stated exactly
once, in `network/control.py`, so the client is not the place that knows.

`RemoteSessionFacts` is the peer-side session's fields the client needs before the
welcome: from `net_catalog`'s row — `pid`, `conversation_name`, `cwd`,
`model_label`, `capabilities`, `state`, `age_s`, `pid` — plus the `session_protocol`
value the link's `hello`/`welcome` carries and passes through untouched
(`mesh-transport-identity.md` §6.4). `control_port`/`control_key` are the *relay's*,
because that is what this client dials; the owner's real control key never leaves
the owning device (spine §5.4 property 4, and the transport's §2.5 bullet "never
transmits a control key").

The subclass overrides `connect` and the small set of methods that speak about the
*endpoint* rather than the session (`supports_exclusive_move` etc. read the owner's
advertised capabilities, which arrive in `RemoteSessionFacts`, so they read the
peer's answer rather than a local record). Every other method — `prompt`, `steer`,
`abort`, `slash_result`, `approval_answer`, `ask_answer`, `history_page`,
`set_model`, `set_effort`, `complete_aside`, `credential`, `variables`,
`job_trajectory`, `fork_snapshot` — is inherited untouched, because it is a frame.

Rejected alternative: **a control-socket proxy inside the relay process**, one per
remote session, so that local viewers need no new client class at all. It fails on a
repo fact rather than a preference: a discovery record is keyed by pid —
`registry.record_path(pid)` (`registry.py:109`), `RecordPublisher` (`registry.py:683`)
— so one relay process cannot publish one `run/mobile` record per remote session,
and without such a record `find_runtime_record` (`attach_client.py:686`) cannot find
the proxy, so every viewer would need a resolver change anyway — i.e. the same seam,
plus a stray control socket per remote session. Rejected alternative: **one shared
upstream connection for all local viewers of a session**; it fails on §3.5.

### 3.3 What is forwarded, what is refused, and why

**Forwarded (the default: everything the runtime dispatches).** Every op in
`RuntimeServer._dispatch` (`session/runtime/server.py:4120`) and
`_dispatch_payload` (`:4355`) is forwarded verbatim, because the relay forwards
*frames* rather than implementing a parallel API. The op list is enumerated and
classified in §4.

**Refused at the relay, before the frame ever leaves this device:**

| Op / behaviour | Where | Why |
|---|---|---|
| `retire_if_pristine` sent because the local viewer is quitting | `server.py:3538`, sent from `attached.py:8158` on `retire_if_unused` | It answers "I engaged this runtime and am leaving unused" — a claim only a viewer on the runtime's own machine can make. From a remote viewer it would let a laptop quitting its TUI stop a session on a peer. The relay answers `"kept: a remote viewer cannot offer back a runtime it does not hold"`. (Note the *safe* half is kept: the peer runtime's own residency drain still decides when an idle runtime exits.) |
| A forwarded frame whose auth claims `locality: "local"` | peer relay, at dial | spine §5.8: locality is a claim the receiving side checks, and only the dialling relay knows the truth. |
| `new_conversation` / `resume_session` from an attach-class connection | `server.py:3452` (existing) | Followers must not rebind the owner's conversation. A remote viewer is an attach client, so it inherits the refusal — with re-worded copy naming the device (see §4's exception table). |
| `stream_open` for a session the answering relay does not own | relay | INV-1, and the transport's §7.2 session-scope rule: the peer answers `not_authorised`, never a silent hop to a third device |

**Refused by the runtime, based on `locality`, with the caller's capability set as
the second input** — the three existing `locality == "remote"` gates, and how each is
settled.
Each of these is currently a flat refusal (`mcp_credentials` at `server.py:4426`,
`credential {action: "store"}` at `server.py:4476`, `/mcp login` driven by
`browser_is_reachable=locality != "remote"` at `serving.py:5060` →
`REMOTE_GRANT_NOTICE`, `mcp/grants.py:65`). The flat refusal is right for the case it
was written for — a phone browser, an untrusted UI, whose comment says so at
`server.py:4477-4486` — and wrong for a mesh member that already holds `prompt`.

The design's rule, stated once:

> **A peer may do what its membership grants (spine §5.3), and the resolved
> capability set is the ONLY input that decides these three.** The capabilities are
> the transport's own (`mesh-transport-identity.md` §7.1:
> `list, view, prompt, steer, stop, slash, delete, move, broker_credential, admin,
> trust`), resolved per member at admission and carried on the `LinkContext`
> (`network/authorize.py`) that its chokepoint already builds. `credential:store`,
> `mcp_credentials` and fork-of-a-remote-session are permitted when the caller holds
> **`prompt`**, and refused with the existing sentence otherwise.

Two lines of the transport's table already cover two of the three, so the work here
is one row plus one field:

* its `INNER_OP_CAPABILITY` maps every forwarded `ControlOp` to a capability, and
  `credential`, `mcp_credentials` and `fork_snapshot` should map to **`prompt`**
  (they are payload ops carrying session-scoped writes, not session reads);
* the runtime cannot see a `LinkContext`, so the peer relay's own dial to the owner
  carries the resolved set additively in the auth frame —
  `{"key": …, "client": "attach", "locality": "remote", "capabilities": ["view","prompt","slash"]}` —
  and the runtime's three gates read that field. **Absent means local**, exactly as
  every other additive auth field does (`slash_consumers`, `server.py:2715`: absent
  = a client built before the field), so nothing that reaches the socket today
  changes behaviour.

*Why `prompt` is the right threshold.* Spine §5 states that a session **is**
arbitrary code execution — the agent has a shell. A member that may prompt can
therefore already read everything the session can reach, run a command, and print
it into the transcript. Refusing such a member a credential-store write is not a
boundary; it is a speed bump that teaches members to paste secrets into chat, which
is the leak the gate was protecting against (`server.py:4481-4486`). The boundary
that matters is `broker-credential` (a member without it gets no token) and
`prompt` (a member without it cannot act at all).

`/mcp login` is the one that cannot be routed at all — the repo says so and says
what to do instead: *"A relay that wants to support grants must carry the
authorization URL to the device rather than route the verb."*
(`mcp/grants.py:23-26`), and the refusal stays `REMOTE_GRANT_NOTICE`
(`mcp/grants.py:65`, used at `:257`).

**The mechanism is `mesh-credentials.md`'s (§4.7), and this document follows it
rather than designing a second one**: for v1 a remote `/mcp login` is refused with
that notice, and the mesh adds the **repair** path — the viewer's attempt raises a
durable notice on the owner (*"gpu-pod-3 needs you to re-run '/mcp login datadog'
here"*), the owner's human runs the verb on the machine that has the browser, and
the brokered token is available from then on. Frames: `net_broker` with its `kind`
discriminator (`grant` | `report` | `placement` | `repair`), `broker_credential`-gated.

Recorded honestly, because it is a *gap* against R7/R13's intent rather than a
solution to it: the user who typed `/mcp login` from a laptop still has to walk to
the other machine. The complete answer is the one `grants.py` describes — the
viewer's relay binds the loopback callback (`mcp/grants.py:61-63`), the browser opens
on the machine the person is at, and the `{code, state}` is forwarded to the owner,
which performs the exchange against **its** `auth.db` (so A5 and R16 hold). I do not
fold it in here because it depends on a fixed local port being bindable by a
supervised daemon on a device we do not control, and because the *area* is the
credentials design's; it is stated as the follow-up in §13 (Q-CRED) with the
evidence that would settle it.

### 3.4 Cold start, history, and the one local read that must not lie

`AttachedSession.cold()` (`attached.py:1487`) synthesises canonical state without an
owner and reads the local transcript when one exists
(`attached.py:1527` — `if (config_dir / "sessions" / session_id / "transcript.jsonl").exists()`,
with `want_checkpoint=True`). For a remote session the directory does not exist
locally, so the read is already skipped by construction — the facade paints an
empty-but-honest cold state and is corrected by the first sync. Two refinements the
implementation must make explicit rather than inherit by luck:

* **The pre-bind paint must not claim "no history".** Pass `seed: SessionSeed`
  (name, `model_label`, `cwd`, `mtime`, `history_message_count` if the catalogue row
  knows it) so the row and title are right before the bind lands, and so the band
  reads "connecting to <device>…" instead of "0 messages". A remote viewer's blank
  first frame is a *visible* defect, not a cosmetic one, because the sidebar paints
  before the bind.
* **History comes from the wire, and only the wire.** The modern path already does
  this: when the auth frame negotiated `display_window`
  (`attach_client.py:1002-1009`) and the owner advertises
  `display-history-window-v1`, `_load_frontend_history` (`attached.py:4168`) hydrates
  from the sync's window and pages older rows through `history_page`
  (`attached.py:4260` → `client.history_page`, `attach_client.py:1520`). The local
  transcript replay (`_load_history`, `attached.py:4841`, → `_read_transcript`,
  `:4870`) is the **legacy fallback** (`attached.py:4179-4186`). A remote placement
  must therefore (a) always request `display_window` and (b) **refuse the fallback**:
  if the owner cannot serve a window, answer `ConnectionError("this peer's runtime is
  too old to serve history over the mesh; update it")` rather than replaying a
  local transcript that is not the session.

**Finding the owner over the mesh is not a local scan.** `_bind_under_lock`
(`attached.py:3266`) re-reads `find_runtime_record` on every attempt
(`:3388`) and branches on `(record is None and owner is not None)` — "an owner still
holds the lease but published no live record" (`:3397`). `RemoteOwner.locate()`
reproduces both states from the relay: `(None, None)` when the peer reports no
runtime and no claim, and `(facts, pid)` when the peer reports a claimed-but-not-yet
-published runtime, so the retry loop's semantics are preserved without the loop
knowing.

### 3.5 The receipt-consumer contract (the one place aggregation would be wrong)

`SLASH_ACTION_RECEIPTS = ("team_attached", "agent_attached", "goal_set")`
(`session/runtime/types.py:199`) exist because a receipt may carry an ACTION the
invoking terminal is expected to submit as a user turn, and the ownership rule is
`runtime_must_complete(receipt_type, consumers)` (`types.py:202`): *the owner
completes it only if the client did NOT declare that type*. Both ends apply the same
predicate — the runtime on the connection's declared `slash_consumers`
(`server.py:2715`, passed into the handle at `server.py:4415-4418`), and the
viewer on its own declaration (`tui/app.py:38088`, desktop at
`server/routes/desktop_sessions.py:1844` `desktop_viewer_must_submit` (its rule read at
`:1895`), reading
`outcome.data.get("request")` at `:1984`). A drift between the two is a **double
turn** (`types.py:203-214` documents exactly that drift).

This is why the relay forwards **one upstream connection per viewer connection**
rather than multiplexing: `slash_consumers` is a per-connection fact, as are
`events`, `frontend_state`, `display_window`, `surface` and the watch leases
(`server.py:2704-2762`). A relay that aggregated them would have to decide
consumption on behalf of viewers that disagree, and the only safe aggregation —
declare nothing, let the owner complete — is precisely the double-turn the predicate
exists to prevent, because the desktop route reads its OWN declaration.

Consequences to state in the code, because each is a real limit:

* The peer runtime's `ATTACH_MAX_CLIENTS = 4` bounds the number of *remote viewers
  across the whole other device*, since each one is a connection there. Exceeding it
  evicts the least-recently-seen (LRU, `server.py:2747-2750`) — the evicted viewer
  gets an ordinary owner-loss, which for a viewing surface is the existing recovery
  path. Documented, not fixed: raising the cap is a separate decision.
* `_other_observers` (`server.py:3828`) therefore counts remote viewers, so a `move`
  requested while a remote viewer is attached is refused with the existing sentence
  — which is correct and desirable, and is why the mesh does not need a new
  "quiesce the viewers" step.
* The relay is one client among those, kind `attach`, `surface` forwarded from the
  viewer's own auth frame (`server.py:2757`). The relay never uses kind `daemon`:
  there is exactly one daemon slot per runtime and it belongs to the local phone
  daemon (`server.py:2734-2740`); a remote relay claiming it would evict the phone
  from the peer's own runtime.

### 3.6 Event flow back

Frames flow back along the same pipe with no translation, because the peer runtime
is the same build speaking the same vocabulary:

| Frame | Producer | Consumer | Note |
|---|---|---|---|
| `welcome` / `projection` | `RuntimeServer._push_to` | `AttachClient._on_projection` | identity check at `attach_client.py:1041` is the attach's proof |
| `event` (raw `AgentEvent`) | runtime's handle relay | `AttachedSession._on_wire_event` | the transcript stream, R7 |
| `frontend_sync` / `frontend_update` | `subscribe_frontend` | `attached.py:_on_frontend_sync/_on_frontend_update` | gates, todos, subagents, usage/context, jobs — all of it rides this |
| `ack` / `error` | dispatch | `_pending` correlation by `req` | unchanged |
| `retiring` | `announce_retiring` | `_on_retiring_frame` | the move's and the build-refresh's handover signal |

The **relay adds no frame type of its own to this pipe.** Its own state (link
health, per-session reachability, the handoff journal) travels on relay-level
messages outside the session pipe, so a session's stream stays exactly the
vocabulary a local viewer sees.

Catalogue and presence facts do *not* come from a standing session subscription.
Row facts (name, mtime, preview, status, attention/unseen, `archived`, `placement`)
come from `net_catalog` through the transport's cache — its §9.2's 2 s per-relay
cache over a `registry.scan()`, refreshed peer-to-peer on its §2.4 cadence (60 s) —
which is the same layering the local sidebar already has: it reads the catalogue and
the attention store, never a session's socket (`session/catalog.py:891`
`load_catalog`). A standing per-session presence connection would burn a viewer slot
(§3.5) and would put the relay in competition with the peer's own phone daemon for
the daemon slot; it is rejected.

**The one consequence to watch**, and the reason it is not an interface request:
60 s is the freshness of a *remote* row's attention/status mark, where a local row
refreshes on the sidebar's own poll. That is acceptable for a group heading and a
spinner, and it is the transport's cadence to tune (its §11's `network.*` keys). If
completion marks on remote rows prove too stale in use, the answer is a pushed
`net_catalog` delta — a *new* op, so it is a deliberate follow-up rather than
something this design presupposes (§13 Q-CAT).

---

### 3.7 Viewer attach discipline — width, one read-only observer, and the exit status

Adopted from abduco (ISC — `mesh-prior-art.md` §4), because the mesh turns three
of its policies from cosmetic into load-bearing. None of them is a protocol
change; all three are rules the projection must state, because the local case
never has two viewers with different sizes:

1. **The width belongs to the most recently connected, non-read-only client.** A
   phone attaching to a session whose runtime lives on a desktop must not reflow
   the desktop's session. The rule to implement is abduco's — "resize request are
   only processed if they are initiated by the most recently connected, non read
   only client" — and a read-only observer's resize is recorded and reported (in
   the peer's `status`) and never applied. The tempting alternative, "the
   narrowest live viewer wins", is the one that makes a 40-column phone reflow the
   desktop's 200-column session, and it must not be reached by accident.
2. **Read-only attachment is a capability, not a UI state.** `view` without
   `prompt` already means "observer" in the transport's §7.1; a viewer holding
   only `list`/`view` therefore connects in a mode that renders the transcript and
   **never sends an input frame**, so "let me watch" cannot become "let me type"
   through a front-end bug. abduco is explicit that its own read-only flag is not
   a security feature; ours is, because it is the same capability the authoriser
   already enforces — the attach mode is the *presentation* of a grant, never a
   substitute for it.
3. **The exit status survives a detach and a move.** abduco reports a detached
   command's exit status on reattach with the status preserved; the mesh
   equivalent is that a session's terminal status is a durable fact of the record
   and transcript, carried across §6.3's handoff journal, so a moved session does
   not come home looking "still running" when it exited on the peer and a viewer
   that detaches and reattaches does not lose the code.

**Consequence for §6.5's failure rows:** the third rule is the one that can be
lost silently in a move, so the handoff's `committed` step must carry the status
rather than recomputing it at the destination.

---

## 4. The command matrix

Default is **works remotely**, routed to the owner. Exceptions are enumerated with a
reason, and every "routed" entry names the code path that carries it. The
process/terminal-local list is the repo's own: a slash command is frontend-local iff
its name is in `_FRONTEND_LOCAL_SLASHES` (`session/frontend_state.py:828-936`), and
everything else is advertised `CommandScope.AUTHORITATIVE_SESSION`
(`frontend_state.py:955`, built by `_slash_capabilities`, `:5680`). The TUI's
decision site is `tui/app.py:29179-29247`; the routed leg is
`ControlConnection.route_shared_slash` (`session/protocol.py:1136`) →
`AttachedSession.route_shared_slash` (`attached.py:7426`) → `slash_result` op
(`server.py:4392`) → `ServingSessionHandle.run_slash_authoritative`
(`serving.py:4087`) → `_slash_result` (`serving.py:4292`).

### 4.1 Capabilities the spine lists (R7)

| Capability | Verdict | Code path |
|---|---|---|
| **prompt** | works | `prompt` op (`server.py:4145`) → `AttachClient.prompt` (`attach_client.py:1384` / `send_command:1398`); durable admission + `command_id` idempotency unchanged |
| **steer** | works | `steer` op (`server.py:4151`), `recall_steer` (`:4236`, the Esc-recall parity) |
| **stop / abort** | works | `abort` (`:4156`), `cancel` graceful-vs-immediate (`:4158`); bare `/stop` **on the remote session** routes the `stop` op (`:4316`) → the owner's own ladder. `lop stop <id>` where `<id>` is remote routes via `net_session_stop` (§4.3) because the ladder's pid-identity proofs are local (`session/runtime/control.py:1-30`, `_identity_by_record:474`, `_same_uid:643`) |
| **slash** | works | §4.2 — routed by capability scope, with the frontend-local exception list |
| **model / effort change** | works | `set_model` (`:4179`, incl. the optional `set_model_effort` 3-arg probe) and `set_effort` (`:4198`). *Exception:* bare `/model` opens the **viewer's** picker (`tui/app.py:29246`, deliberate — the widget is the invoking terminal's), `/model default` and `/model saved` stay local because they write this machine's `config.yml` (`_FRONTEND_LOCAL_SLASHES`); the chosen value then routes |
| **approvals** | works | `approval_answer` (`:4230`) + the projection's `pending`/`PendingRequest` (`mobile/types.py:541`) render the card **on the remote viewer**; the answer routes back. Requires `prompt` grant — an approver is exactly as powerful as a prompter |
| **asks** | works | `ask_answer` (`:4245`) with `question_index` (the U8 guard is preserved across the hop because the index is a frame field) |
| **subagents** | works | roster + jobs ride `frontend_sync`/`frontend_update`; `cancel_subagents` (`_PAYLOAD_OPS`, `server.py:709`), `job_trajectory` (payload op), `watch_job`/`unwatch_job` (`server.py:3456`, connection-local, forwarded per viewer) |
| **todos** | works | `TodoItem`/`TodoPhase` in the projection (`mobile/types.py:443`, `:461`); `TODO_STORE` is owner-side state |
| **transcript stream** | works | `event` frames (`mobile/types.py:346`) + `history_page` (`:1520`) for paging |
| **resume / attach** | works with a re-pointed meaning | `/resume <id>` is frontend-local (it opens a viewer here); for a remote id it attaches through the relay (§3.2). The `resume_session` op — "rebind the runtime to another transcript" — is **refused** to an attach-class connection by `server.py:3452` today; see §4.4 |
| **archive** | works | §8; `net_session_lifecycle {action: "archive"}` → the owner's `session/archived.py` store (`feat/session-archive-delete`) |
| **delete** | works | §8; `net_session_lifecycle {action: "delete"}` → the owner's `cleanup.delete_session` (`feat/session-archive-delete`), guards and sentence included |
| **rename** | works | `/rename` is `authoritative_session` → `_slash_result` on the owner (`serving.py:4292`); the durable name lives in the owner's `title.json` (`resume.py:249`) |
| **fork** | works, owner-side | `/fork` is frontend-local in the TUI (`_FRONTEND_LOCAL_SLASHES`, reasoned at `frontend_state.py:860-864`: the fork window opens here). Over the mesh the *fork itself* must be created where the transcript is — the owner — via `net_session_move {phase: "fork"}` → `fork_session` (`fork.py:167`) → the fork's id is returned and the viewer opens a window **attached to that id, annotated remote**. See §4.4 for the `fork_snapshot` locality gate this changes |
| **wake** | works | `wake` is `authoritative_session` (the desktop route proves the pattern: `server/routes/desktop_wakes.py:659` calls `route_shared_slash("wake", …)`); over the mesh it runs on the owner's wake index, and a wake that fires there is the owner's business — the viewer sees the turn in the transcript stream. `lop send --peer P --wake` routes to the owner's `receive_peer_message` (`server.py:4266`) |
| **goal / loop** | works | `goal_set` is an action receipt (`types.py:199`); `/goal` and `/loop` are handled by `serving.py:4292` on the owner. *Exception:* `/loop` and `/sidebar` are pulled back to the invoking terminal (`tui/app.py:29187-29191` — the terminal owns scheduling and sidebar focus) while **each iteration still submits through the owner**; that behaviour is unchanged remotely and must be preserved |
| **usage / context / info** | split, as today | `/usage`, `/analytics` are frontend-local (they read local ledgers/credentials, `frontend_state.py:875-883`) — a remote viewer's usage panel reports *its own* ledger, which is the honest reading of "what has this terminal spent". `/context`, `/session`, `/info` are advertised `authoritative_session` where they describe the session, and `/info` is deliberately frontend-local (`frontend_state.py:884-891`: it answers "which code am I running") — so over the mesh `/info` answers about the **viewer's** install and the session's own facts come from the projection. This is the pre-existing split, not a mesh decision, and it is the one place a user may be surprised: the design keeps the `/info` copy exactly as it is and lets `mesh-ui.md` decide whether to add a "device" line |

### 4.2 The frontend-local exception table (slash)

Carried over verbatim from `_FRONTEND_LOCAL_SLASHES` (`frontend_state.py:828-936`):
`help`, `exit`, `clear`, `loop`, `sidebar`, `copy`, `links`, `new`, `reload`,
`update`, `resume`, `move`, `fork`, `theme`, `settings`, `provider`, `search`,
`accounts`, `failovers`, `usage`, `analytics`, `session`, `info`, `skills`, `login`,
`logout`, `mobile`, `notifications`, `credential`, `btw`, `stop`.

Over the mesh, four of these acquire a *mesh* meaning and two are extended. The
implementation change is small and localised in the TUI's dispatch
(`tui/app.py:29179-29247`), which already has the pull-back machinery
(`remote_capability = None`):

| Command | Remote meaning |
|---|---|
| `/stop` | Bare `/stop` stops **the session you are looking at** — the remote runtime — via the `stop` op (`server.py:4316`), as it does for a local attach today; the `/stop <target>` and `/stop all` forms enumerate the **viewer's** registry (`frontend_state.py:927-935`), and gain `--all-peers` (§4.3). |
| `/move` | A cwd change for a local session; **`/move <session> --to <peer\|local>` is mobility** (spine §6). Both spellings coexist: `/move <path>` keeps its cwd meaning, `/move <id> --to …` is the mesh form, and the picker's cwd mode is unchanged (`_FRONTEND_LOCAL_SLASHES`, `frontend_state.py:853-859`). |
| `/fork` | Forks **on the owner** (§4.1) and opens the window here, attached to the fork's id. |
| `/new` | **Local by default.** `/new` opens a new conversation on the machine you are sitting at; `/new remote <peer> [prompt]` creates it on the peer (R8). Rationale: `/new` is the "give me a fresh chat" gesture and the least-surprising target is the machine in front of the user; the peer form is explicit because it is a different act with a different lifecycle. |
| `/credential` | Unchanged split: the masked paste is hosted here, the store write routes (`frontend_state.py:910-924`), and the runtime's `locality` gate becomes grant-aware (§3.3). |
| `/mcp` | Unchanged split (bare = local listing, grants route); `/mcp login` becomes the relay-mediated grant of §3.3 step 2–4. |

### 4.3 Ops outside the session pipe (relay-routed)

`lop stop <id>` and `/stop <target>` with a remote target, `lop send --peer`, and the
CLI's `sessions` verbs (§9.3) do not become session ops: they go to the **relay**,
which asks the owner to run its own local implementation.

* **stop** → `net_session_stop` → the owner runs `control.stop_session`/`stop_all`
  (`session/runtime/control.py:1367` `_stop_targets`) with its own records, its own
  pid-identity proofs (`_identity_by_record:474`, `_pid_holds_port:546`,
  `_same_uid:643`) and its own escalation ladder. The viewer renders the peer's
  `StopOutcome` vocabulary (`control.py:237`) verbatim. This is the "one
  implementation, three front ends" rule (`control.py:1-30`) applied across a
  fourth front end, rather than a second kill implementation on the viewing device.
* **send** → **a forwarded `peer_message` frame** (`net_forward` carrying
  `ControlOp.peer_message`, capability `prompt`) → the owner's `mobile/peer_send.
  deliver_peer_message` (`peer_send.py:705`) with its own registry resolution,
  unengaged-composer gate (`server.py:4290`) and inbox spool
  (`session/runtime/inbox.py:1-40`) — so `lop send` to a *cold* remote session
  spools into the **owner's** inbox, which the owner's next runtime drains before it
  starts listening (`inbox.py`, "the ordering guarantee"). The first local hop keeps
  its existing target syntax (`cli.py:617-657`) and gains only `--peer`.

### 4.4 The two refusals that must be re-worded, not removed

1. **`new_conversation` / `resume_session` from an attach client** —
   `server.py:3452`: *"attached front ends cannot rebind the session; detach and
   /resume instead"*. Over the mesh a remote viewer that `/stop`s then `/resume`s
   needs the sentence to name the device, because "detach and /resume" is ambiguous
   with several devices: the relayed form reads *"a viewer on another device cannot
   rebind this session — run `/resume <id>` on <device name>, or reconnect here"*.
   The refusal itself stays (it protects the peer's own screen).
2. **`fork_snapshot`'s `locality` gate** — `server.py:4378`:
   *"fork requires a terminal on the session's machine"*, because *"a foreign viewer
   must not receive a local id it cannot reach"* (`server.py:4375-4376`). Over the
   mesh the second half is no longer true: a fork created on the owner is reachable
   from the viewer **through the peer's own catalogue**, so the id is not
   unreachable — it is remote, which is a first-class state now. The gate therefore
   becomes: *refused when the caller cannot resolve the fork's owner* (i.e. no mesh
   path and not local) and **permitted when the caller is a relayed member with the
   `move` or `prompt` grant**, returning `{fork_id, device_id}` instead of a bare
   id. This is the single place the mesh relaxes an existing refusal, and the review
   question to ask is whether the returned id is resolvable at the caller — which is
   exactly what the new gate asserts.

---

## 5. Placement

### 5.1 On the record (additive, no `PROTOCOL_VERSION` bump)

`SessionRecord` (`session/runtime/types.py:712`) gains one block, following the
existing additive contract stated at `types.py:739-746` (*"Purely ADDITIVE, and
PROTOCOL_VERSION deliberately does NOT move for them … an older reader drops unknown
keys in `from_json`"*):

```json
{
  "pid": 48213, "kind": "daemon", "session_id": "9f3ac1e0b7d2",
  "conversation_name": "mesh design", "cwd": "/Users/damian/oss/x",
  "model_label": "anthropic/claude-sonnet-4-5", "control_port": 51823,
  "control_key": "…", "protocol": 5, "capabilities": ["exclusive-move-v1", "…"],
  "placement": {
    "mode": "peer",
    "network_id": "n_7Yb3kQ",
    "home_device": "dev_a1b2c3d4e5f6",
    "policy": "pinned",
    "stamp_revision": 3
  }
}
```

* `placement: null` (or absent) is the local case and the older-build case; it is
  **always present** on a record written by a builder that knows about the mesh, with
  `mode: "local"` for a local session, so a reader can distinguish "local" from
  "this runtime is too old to know".
* `kind` is unchanged: a remote session is served by a runtime on its owner with its
  own honest kind. The *viewer-side* notion of remoteness is the catalogue row's
  `locality` (§9.2) — never a new `kind`, because `SessionRecord.kind` is a
  `Literal` those readers pass through unvalidated (`types.py:241-255`).
* A device id appearing on a record is not a security boundary — it is an assertion
  by the writer, verified by the link that reads it (§2.2).

### 5.2 On the sidecar and in the catalogue

`SessionPlacement` (`local_operator/session/placement.py`) is the one dataclass, serialised in both
places:

```python
@dataclass(frozen=True)
class SessionPlacement:
    mode: Literal["local", "peer", "pool"]      # pool reserved, never produced here
    network_id: str = ""
    home_device: str = ""
    policy: Literal["pinned", "prefer-remote", "cost-capped"] = "pinned"
    stamp_revision: int = 0
```

`mesh.json` carries it plus the two facts the record has no room for (the creation
origin and the session's own network). The catalogue row carries the *viewer-facing*
projection of it (§9.2).

### 5.3 How `/new remote` and `lop exec --peer` set it

**`/new remote <peer> [prompt]`** — the peer mints the id. That is the rule: *the
device that will own the session mints its session id*, so an id never exists in two
places.

1. TUI resolves `<peer>` from the cached peer list (offline-safe: a stale list is
   usable, an unreachable peer is refused **with its reason** — spine §8). Completion
   is driven from that cache, so it works with the relay down.
2. `net_session_create` to that peer:
   `{cwd, model, effort, name, agent, team, goal, prompt, images, origin: "user"}`.
   `cwd` defaults to the peer's own home (the peer decides; a viewer-supplied path is
   validated on the peer exactly as `session_factory.create_session`'s `cwd` is,
   `session_factory.py:4395-4420`).
3. The peer: mints an id (`fork.new_session_id`, `fork.py:155` — the same 12-hex
   shape every other id has), claims the directory (`retention.claim_session`,
   `retention.py:220`, *claim before mkdir* exactly as `fork_session` does at
   `fork.py:198-216`), writes `mesh.json` with `home_device = <peer>`, engages a
   runtime (`launch.engage_runtime`), and admits the first prompt if one was given —
   i.e. the ordinary local `/new` path, executed on the peer.
4. The peer answers `{session_id, admitted, record}`; the viewer opens a window
   attached to that id through the relay (§3.2) — or, with no prompt, shows the
   composition surface in the normal way.

**`lop exec --peer <peer> "…"`** (spine §6) is the same relay op with a headless
viewer: `exec` gains `--peer` beside its existing startup arguments
(`cli.py:1192-1233`), the one-shot composes on the peer, and the local process waits
for the terminal outcome by reading the forwarded stream — which it can do with no
new code because `exec` is already a client of the same vocabulary
(`session/runtime/exec_control.py:1-40`: *"the whole control vocabulary … is the one
the phone daemon and `lop attach` already speak"*). `--json` NDJSON stays the local
process's stdout: `exec` is the machine-facing surface and its stream must not
change shape because the work moved. `--peer` with `--background` is refused in v1
(a detached worker whose owner is another device needs the job ledger to be
device-aware; deferred, §13 Q7).

**Discovery.** A viewer learns the placement from (a) the catalogue it just listed
(§9.2), (b) `/resume`'s row, which carries it, and (c) `lop sessions --json`
(§9.3). It never infers remoteness from an id shape, a cwd, or a failed local scan —
which is why `locality` is a field rather than a derivation (spine §8).

---

## 6. Mobility

### 6.1 What exists, and what this adds

`fork_session` (`fork.py:167`) is the transportable half: it copies an allow-listed
file set (`COPIED_SIDECARS`, `fork.py:111` — transcript, `attachment.json`,
`title.json`) and deliberately not `.session.pid`, the parent's roster, or the scan
sentinels (`EXCLUDED_SIDECARS`, `fork.py:137`). The half that did not exist is
*safely retiring a source*, and the repo has the fence for it:
`EXCLUSIVE_MOVE_CAPABILITY = "exclusive-move-v1"`
(`session/runtime/types.py:93`), reserved and honoured at
`server.py:3618-3660`, re-checked at the latch at `server.py:4034`, driven today by
the desktop's cwd move (`server/utils/desktop_sessions.py:789-834`).

A `move` therefore reuses three existing primitives and invents one:

| Step | Primitive | Anchor |
|---|---|---|
| quiesce + retire the source runtime | `retire_now {exclusive: true}` → `_retire_for("moved", exclusive_owner=conn)` | `server.py:3588-3660`, `:3999` |
| nothing new can be admitted during the window | the `begin_retire` latch (check-and-commit in one synchronous step) | `server.py:4049-4075` |
| single-writer lease | `claim_session` / `.session.pid` / `live_runtime_pid` | `retention.py:220`, `resume.py:1147` |
| **the cross-device handoff journal (new)** | `network/pending-move.json` | §6.3 |

### 6.2 Two modes, and the id rule

> **The device that will OWN the session mints its id.**

* **`move` (default)** keeps the session id and transfers ownership: the source
  retires, the destination adopts, the source's directory is deleted and tombstoned.
  Exactly one directory for the id exists at the end, on the destination.
* **`--keep`** does **not** transfer anything: the destination mints a **new** id,
  pulls a full copy, and adopts it as a fork (`origin: fork` in its `mesh.json`,
  plus `fork-boundary.json` from `fork.write_boot_prompt`/`FORK_BOUNDARY_NAME`,
  `fork.py:86`, so the model is told not to continue the original). The source is
  untouched and keeps running; the two copies diverge from that point.

Rejected alternative: `--keep` giving the destination the *same* id (the "fork both
ways, one of them renames" reading). Two devices holding one id forever is a permanent
ambiguity for every routing question in §2, and the id is the address. The cost of the
rule chosen is that a kept copy has a different name until it is renamed, which is
what `origin.source_session_id` and the row's "copy of X from Y" label are for.

### 6.3 The move protocol, ordered

`O` = owner (source), `D` = destination (also the requestor; a third-device request is
refused in v1 with *"run this from <D>: a move is issued by the device that will hold
the session"*, §13 Q3). `mode: "move"`.

**Shorthand used below.** Every step of a move is carried by **one** peer op —
`net_session_move`, capability `move` — discriminated by `phase`; the steps below
write `move.<phase>` for `net_session_move {phase: …}` so the sequencing reads as a
sequence rather than as fifteen repetitions of one op name. Phases:
`status`, `prepare`, `ready`, `commit`, `done`, plus `fork` (an in-place fork on the
owner, §4.1) and `copy` (the `--keep` copy, §6.2). The copy itself rides `net_sync`
(§7), not a move phase, because R22 needs that operation to exist on its own.

```
{"op":"net_session_move","req":31,"phase":"prepare","session_id":"9f3ac1e0b7d2",
 "to_device":"d_4b2a91c4e0b87f3a","mode":"move","locality":"remote"}
→ {"op":"ack","req":31,"detail":{"result":"prepared","lease_epoch":"e_7f21c9a4",
   "manifest":{"generation":41,"items":[{"name":"transcript.jsonl","bytes":221184,
   "mtime_ns":1758231100000000000,"digest":"sha256:…","mode":"replace"}],
   "prefix_digest":"sha256:…","frontier_entry_id":"e_91c2"}}}
→ {"op":"ack","req":31,"detail":{"result":"refused","detail":"…This session is open in another terminal…"}}
```

**Phase 0 — resolve.** `D → O: move.status {session_id}`. `O` answers
`{owner: true, busy, pending, record}` or `{owner: false, device}`. `owner: false` →
`D` purges its cached row and refuses with the device that does own it; a `status`
against an id `O` has tombstoned answers with the tombstone (§6.5's recovery table
depends on exactly this).

**Phase 1 — prepare (owner-side quiesce; no mutation yet).**

1. `D → O: move.prepare {session_id, to_device: D, mode, requester: D}`.
2. `O` validates, in this order, refusing before any mutation: `D` is a current
   member at the current epoch with the `move` grant; the session's `mesh.json`
   `home_device` is `O`; there is no `pending-move.json` entry for the id (a stale
   one triggers the reconcile of §6.5 first, then the request is retried).
3. `O`'s relay retires the runtime: it dials the local runtime and sends
   `retire_now {exclusive: true}` on **its own attach connection** (so
   `_other_observers` measures everyone else, `server.py:3828`). Outcomes:
   * `kept: <idle reason>` → `O → D {result: "refused", detail: <sentence>}`.
     **Nothing has been mutated anywhere.** `D` renders the sentence; `--wait N`
     re-polls (§6.4).
   * `kept: This session is open in another terminal or attached client…`
     (`server.py:3636-3639`) → same refusal, same sentence; a viewer on either device
     can be the cause, which is correct.
   * `retiring` → the latch has committed (`server.py:4075`): no further turn can be
     admitted on `O`, and the runtime disposes.
4. `O` waits (bounded) for the runtime process to exit and its record to disappear
   (`registry.unpublish`, `registry.py:178`) — otherwise a local viewer on `O` could
   dial a dead record mid-handoff.
5. `O` writes the handoff journal **atomically** (same-directory temp +
   `os.replace`, `fsync`; the discipline `_stage_and_replace` already uses,
   `server/utils/desktop_sessions.py:388`):

   ```json
   { "version": 1, "session_id": "9f3ac1e0b7d2", "to_device": "dev_D",
     "mode": "move", "phase": "prepared", "lease_epoch": "e_7f21c9a4",
     "requester": "dev_D", "at": 1758231122.117 }
   ```
6. `O → D {result: "prepared", lease_epoch, manifest}` — `manifest` is the sync plan
   of §7 (item list + digests + `history_generation` + prefix digest), computed for
   `D`'s `have` (empty on a first move).

**Phase 2 — copy (destination pulls into staging).**

7. `D → O: sync.plan {session_id, lease_epoch, have}` (the manifest in step 6
   is the same answer; the explicit call makes retries idempotent).
8. `D → O: sync.fetch {plan_id, items}` — chunked, bounded by the control
   socket's 1 MiB line limit (`_MAX_LINE_BYTES = 1 << 20`, `server.py:182`;
   `_READ_LIMIT_BYTES`, `attach_client.py:103`), attachments streamed by digest.
   `D` writes into `<config>/network/staging/<session_id>/` — **outside `sessions/`**,
   so no scanner, picker, catalogue, search index or retention pass can see it.
9. `D` verifies: every item's digest, that `transcript.jsonl` parses, and that every
   digest referenced by the transcript exists in `D`'s attachment store (fetching the
   ones it lacks — the store is content-addressed and shared, `session/attachments.py:1-30`).
10. `D` writes the session's `mesh.json` (with `home_device: D`,
    `origin: {kind: "moved", source_device: O, source_session_id: <id>}`) **into the
    staging directory**, fsyncs, then writes `ready.json`
    `{lease_epoch, manifest_digest, promoted: false}` beside it.
11. `D → O: move.ready {lease_epoch, content_digest, plan_id}`, where
    `content_digest` is **a digest `D` derives from its own staged bytes** with
    `sync.copy_content_digest` — the copy set's names, the scratchpad tree's files
    and the referenced blobs with their sidecars, in a fixed order, minus the two
    names an adopting device writes itself (`origin.json`, `fork-boundary.json`).

**Phase 3 — commit (the owner decides, and only the owner).**

12. `O` re-derives **its own** `copy_content_digest` from its session directory and
    requires an exact match, and re-derives `plan_id` against the journal's. Mismatch
    — including a zeroed, absent or partial digest — → refuse
    (`{result: "refused", detail: "…the copy did not verify…"; "nothing was moved"}`)
    and leave phase `prepared`. A refusal here is a **rollback**, not a failure: `O`
    still holds an intact directory and no writer. (Review round 1, M-2: the field
    used to be the owner's own manifest digest echoed back and only checked for being
    NON-EMPTY, so a `sha256:000…` digest and a copy truncated to 100 bytes of 2,580
    both committed and deleted the source.)
    Before the journal advances, `O` also **re-reads the transcript lease and the
    runtime record** — the window between `prepare` and here is where a process that
    opens the session takes ownership (§6.6) — and **re-runs the copy-set
    completeness check**, then refuses (failing closed, naming the holder or the
    entry) if either fails. Review round 1, B-M1/B-M2.
13. `O` atomically advances the journal to `"phase": "handing-off"`. **From this
    instant a rollback is impossible by rule** (§6.5), and `O`'s own `engage_runtime`
    refuses the id (§6.6).
14. `O` writes the tombstone (`network/tombstones.json`:
    `{"<id>": {"to_device": D, "lease_epoch": …, "at": …}}`), deletes the session
    directory through the one `rmtree` in the codebase
    (`cleanup.remove_session_dir`, `feat/session-archive-delete`), then clears the
    journal entry. Each step is idempotent and the order is recoverable (§6.5).
15. `O → D: move.committed {lease_epoch, manifest_digest}`.

**Phase 4 — promote (destination).**

16. `D` promotes with **one atomic rename**:
    `os.replace(network/staging/<id>, sessions/<id>)` on the same filesystem — the
    session, its stamp and its fork/origin metadata become visible in a single step,
    so no window exists in which a local scanner sees a half-promoted session.
17. `D` claims the lease (`claim_session`, `retention.py:220`) and records the row.
    It does **not** auto-engage a runtime: the session is now an ordinary local
    session and the first use engages it (`attached.py:3299`), which is the
    residency model the repo already has. `--open` engages and attaches immediately.
18. `D → O: move.done {}` (best-effort; `O` has already committed).
19. Audit (R18): `session.handoff.prepare`, `.ready`, `.handing_off`, `.committed`,
    `.rolled_back`, `.done` — one record per *semantic* event, never per chunk
    (`mesh-incident-response.md` owns retention; this document owns the vocabulary).

### 6.4 Busy moves are refused, not drained

Spine A4 says "drain the in-flight turn (bounded, with a deterministic timeout)".
The design **refuses instead**, deliberately:

* `_retire_for` is already a refusal protocol — `may_refresh` is the idle predicate
  and every non-idle answer returns `kept: <reason>` (`server.py:4020-4029`), and its
  own docstring records the asymmetry this decision rests on: *"a wrong 'retire'
  costs a cold start nobody asked for, a wrong 'keep' costs one more check"*.
* R12's clause is "must not lose a turn that was in flight". A refusal **cannot**
  lose a turn. A preemptive drain can cut one, and a drain would need a new bound, a
  new cause token in the incident taxonomy (`session/runtime/types.py:561-579`
  already has two causes that mean something else) and a new reconciliation for the
  case where the drained turn was mid-`git push` — the exact argument
  `mobile/types.py:283-294` makes for `cancel` vs `abort`.
* The user's cost is one sentence and a retry.

`--wait N` (default `0`) polls: `D` re-sends `move.prepare` every 5 s up to
`N` seconds, or until the owner answers `prepared`. The wait is unbounded in kind,
never in effect: each retry is a fresh idle probe, so a session that stays busy for
`N` seconds produces a `refused` with the owner's own reason, and the operator
decides again. `lop sessions move <id> --wait` with no value means *"wait up to the
session's own idle horizon"* — capped at 30 min by the CLI.

### 6.5 Failure at every step, and the recovery

**When recovery RUNS, and why that had to be said.** The table below was correct and
unreachable in the product: nothing called `reconcile` except a test, so a `prepared`
entry left by a relay that died blocked the owner's own conversation until somebody
happened to run another move of the same id, and a destination stopped just after its
`os.replace` could not be opened at all (review round 1, M-3). It now runs on three
automatic points — a relay STARTING on a root (`mobility.recover_on_start`, the
relay's `on_start` slice hook: `sweep_staging`, then `reconcile`), the first ENGAGE of
the id (`launch.recover_stale_handoff`, and again from `session_factory._prepare`),
and the first MOVE attempt (`session_move`, which already did) — each scoped by the
INSTANCE RULE below.

**The instance rule is the difference between recovery and sabotage.** An entry is
not evidence of a crash: `prepared` is the normal state of a move whose copy is being
made right now. So recovery SKIPS an entry written by the relay currently running on
that root, and also an entry that names no writer at all while a relay is running
(the fail-closed direction: the writer cannot be established, and rolling a live
handoff back is the failure this rule exists to prevent). Only an entry whose writer
is provably not this root's relay is applied; with no live relay there can be no live
move, because every phase of one is driven from a relay.

The safety argument in one sentence: **`O` deletes only after `D` has a durable,
verified, complete copy (`ready`), and after `ready` `O` never rolls back** — so at
every instant at least one device holds a complete copy, and the promote decision is
a single monotone state the owner answers.

| Failure | State on disk | Recovery |
|---|---|---|
| `D` crashes before step 11 (`ready`) | `O`: journal `prepared`, directory intact. `D`: inert `network/staging/<id>/` | `O`'s reconcile (below) **rolls back**: clear the journal entry; the session stays `O`'s and is simply cold. `D`'s staging is swept by the `network/staging` GC (age-capped, bounded — one directory per interrupted move, nothing else) |
| `O` crashes after step 11, before step 13 | `O`: journal `prepared`, directory intact | Same as above — `prepared` with no `ready` recorded means rollback. `D` holds its staging and does not promote (it never saw `committed`), and reports *"waiting for <device> to confirm the handoff"* on retry |
| `O` crashes after step 13 (`handing-off`), before 14 | `O`: journal `handing-off`, directory still present, no writer | Reconcile **completes** the handoff: the phase says `handing-off`, so `O` finishes steps 14–15 (tombstone → delete → clear) and answers any `D` retry with `committed`. If `O` is unreachable, `D` keeps its staging and reports waiting; nothing is lost |
| `O` crashes after the tombstone, before the delete | `O`: tombstone + directory | Reconcile deletes the directory (the tombstone already governs the id: it is not offered, not resumable, refused with *"moved to <device>"*) |
| `O` crashes after the delete, before `committed` reaches `D` | `O`: tombstone, no directory | `D`'s retry gets no session but **does** get the tombstone with the matching `lease_epoch` → that *is* `committed`, so `D` promotes. This is the case the epoch exists for |
| `D` crashes after `committed`, before promoting | `D`: `ready.json` staging, `promoted: false`; `O`: tombstone | `D`'s boot reconcile finds a staging marker, asks `O` (`move.status` → no session + tombstone for the id naming `D` with epoch E) → promotes. If `O` is unreachable, `D` retries on the next catalogue refresh and the staging is **exempt from the GC** while it holds a `ready` marker naming an unreachable-but-member device (a bounded grace, then the user is asked — see §13 Q6) |
| `D` crashes *during* the promote rename | POSIX rename is atomic: either the old or the new name exists | Reconcile idempotently re-attempts; `os.replace` of an already-promoted staging is a no-op because the staging path no longer exists |
| The link dies mid-copy | as "before `ready`" | `D` retries `sync.fetch` from its manifest (`D` knows exactly what it has); the copy is resumable by construction |
| `D` is not a member any more (revoked mid-move) | `O`: journal `prepared` or `handing-off` | At `prepared` → rollback. At `handing-off` the epoch rotation has already refused `D`'s traffic, and the rotation is itself the incident response (`mesh-incident-response.md`); `O`'s reconcile then treats the handoff as **aborted-to-owner** and keeps its directory, because a revoked device must not be the only holder of a transcript. Recorded as `session.handoff.rolled_back` with the revoke as cause |
| **`O`'s own local viewer resumes the session during the window** (the "source resumes" case) | `O`: journal `prepared`/`handing-off` | Impossible by construction, not by luck — §6.6 |

**Reconcile** runs (a) at relay start, (b) after any handoff outcome the relay
cannot classify, and (c) on demand (`lop network doctor`). It reads
`network/pending-move.json` and applies exactly the table above, then clears the entry.

**RECONCILE IS INSTANCE-SCOPED, AND THAT IS NOT AN OPTIMISATION.** Every entry
records the `instance_id` of the relay that wrote it, and a relay reconciles only the
entries a DIFFERENT instance left behind. Without that rule the table above is
applied to a handoff that is still running: `prepared` is the normal state of a move
whose copy is being pulled right now, so a relay that treated it as a leftover would
roll back its own live handoff — and did, on every `ready`, because a targeted
reconcile runs before each peer frame. An entry from another instance is a different
animal: that relay is gone, so the table applies to it exactly as written here.
It is the analogue of `_settle_unconfirmed_move` (`server/utils/desktop_sessions.py:525`),
with a journal instead of a marker file, because the counterpart device is not
reachable by a `stat`.

**Rollback is not "undo":** `prepared → rollback` requires no write to the session
directory at all (nothing was moved), and `handing-off` never rolls back. That
asymmetry is what makes the protocol safe under the two-generals problem: the
decider is one party (the owner), the decision is monotone, and the other party only
ever needs to ask.

### 6.6 The two guards that make "two live writers" impossible

1. **`engage_runtime` refuses a handing-off session, and the open path refuses
   before it takes the lease.** `O`'s journal is consulted **inside
   `launch.engage_runtime`** (`launch.py`, the one entry point every engage path uses
   — *"every path that has something for a session to do … calls this"*), which
   raises `RuntimeStartupError("this conversation is being handed to <device>")`; a
   destination's entry names the device it is being received FROM (`from_name`, not
   the `to_device` that is this device). One guard, all callers: a local TUI on `O`,
   the phone daemon's `messaging`, a wake, `lop exec`, and the relay's own
   `net_session_engage`. Without it, an idle `O` with a cold viewer would spawn a
   successor during the window and `D` would promote a second owner.
   **AND the same refusal runs in `session_factory._prepare`, immediately before
   `acquire_session_lease`** — review round 1, B-M1: every way the product opens a
   session (`lop -r`, `lop exec`, the TUI's in-process open, a booting runtime) takes
   the transcript lease at that boundary and none of them consults the engage guard
   on the way in, so an opener arriving between `prepare` and `commit` used to take
   the claim and have its directory deleted under it. Refusing *before* the acquire
   is also what keeps a refusal from leaving a lease claim behind for a runtime that
   never started (a claim naming a process that holds no transcript is what `O`'s
   retire reads as "open in another process" — one refused resume was enough to
   block the move that refused it).
   **Both of those consult the recovery first** (§6.5's triggers), so an entry left
   by a relay that is GONE is settled instead of bricking the conversation.
2. **`D` cannot promote without an epoch.** Promotion requires a `ready.json` whose
   `lease_epoch` the owner has acknowledged (`handing-off`/tombstone), so a
   destination that was never told to go ahead cannot claim the id by simply having
   the bytes. A half-copied directory cannot be "accidentally promoted" by a
   `resume` either: staging is outside `sessions/`.

### 6.7 Direction symmetry

`move` and `--keep` are the same operations with the ends swapped, as A4 requires,
because nothing in the protocol is direction-specific: `net_session_move` names a
`to_device`, and the requestor is the destination. "Shed load to the peer" and
"bring it home" are the same code path with a different `to_device`; the only
direction-dependent code is presentation (which device is called "here" in the
confirmation sentence).

### 6.8 Surfaces

| Surface | Form | Notes |
|---|---|---|
| CLI | `lop sessions move <id> --to <peer\|local> [--keep] [--wait N] [--open] [--json]` (spine §6) | `<peer>` resolves from the peer list; `--json` emits the phase transcript (`prepared`/`handing-off`/`committed`/`done`) so the agent path can drive it (R19) |
| TUI | `/move <id> --to <peer\|local> [--keep]` | frontend-local: the terminal owns the act; the effect is the relay protocol |
| Desktop | `POST /v1/desktop/sessions/{id}/transfer` `{peer, keep, wait_s}` → phase transcript | its own route, not a slash, because it needs progress (a move is seconds long) and cannot park the messages endpoint |

The move's confirmation copy must name **both devices**: *"Move 'mesh design' from
laptop to build-box? The copy on laptop will be deleted."* / with `--keep`:
*"Copy 'mesh design' onto build-box and leave the original running on laptop?"*
(`mesh-ui.md` owns rendering).

---

## 7. Incremental sync — the primitive behind R11 and R22

### 7.1 One operation, three uses

`net_sync {phase: "plan"}` / `{phase: "fetch"}` is the transport; the *operation* is
"bring a device's copy of a session to a frontier the source chooses". Three callers:

| Caller | Mode | Retirement? | Ownership change? |
|---|---|---|---|
| `move` (§6) | `exact` | yes (source retired before the copy) | yes |
| `--keep` / opportunistic freshness | `boundary` | no | no (new id at the destination) |
| **R22 pre-spin-down** | `flush` | no (the pod is about to die) | no (the *local* device keeps its copy) |

Because all three are the same plan/fetch pair, R22 is an *instance* of the move
primitive rather than a second mechanism — which is exactly what R21 requires
(spine §9 point 3).

### 7.2 What is copied

**Session files** (the same set `fork_session` inherits, plus the durable state a
resume needs):

| File | Why |
|---|---|
| `transcript.jsonl` | the conversation |
| `title.json` (`resume.py:249`) | the name, instantly, without a window scan |
| `attachment.json` (`resume.py:293`) | persona/team/goal — and the provider cache prefix (`fork.py:99-110`: *"Persona continuity and cache continuity are the same requirement here"*) |
| `origin.json` (`resume.py:68`) | the origin axis travels or the fork is mis-classified |
| `turn-journal.json` (`registry.turn_journal_path`, `registry.py:286`) | so a moved session's first boot can classify the open turn honestly |
| `runtime-stop.json` (`registry.stop_marker_path`, `registry.py:207`) | the deliberate-stop record: a moved session must not re-announce a stop that already happened, nor lose one |
| `inbox.jsonl` (`session/runtime/inbox.py:1-40`) | unread spooled mail is session state and must not be lost by a move (and `inbox.jsonl` is deliberately absent from `retention._SIDECAR_NAMES`) |
| `fork-boundary.json` (`fork.py:86`) | required in `--keep` mode; carried when the source is itself a fork |
| the desktop marker (`retention.DESKTOP_MARKER_NAME`) | the cwd and the draft's model choice (`server/utils/desktop_sessions.py:335` `write_desktop_marker`) |
| the `mesh_credential_binding.v1` row | It lives **in the transcript**, so it travels with the copy and needs no separate treatment — but the assertion is explicit and belongs to `mesh-credentials.md` §5.5: after a move (either mode) the destination's binding is byte-identical, its resolve picks the same `owner_device`, and the source's `AuthStore` sees no refresh as a result of the move |
| `created_at.json` (`session/creation.py:16`) | the session's birth time. **Not derivable on the destination**: `session_created_at` falls back to the directory's `st_birthtime`, so without it a moved conversation reads as newly created *and the real date is gone with the source*. Measured on the operator's store: 10,840 of 10,841 session directories hold one |
| `scratchpad/`, as a **tree** | the files the session's own agent wrote for it (`scratchpad.py`: `scratchpad://notes.md`, downloads, scripts, logs). One plan item per regular file, verified byte-for-byte like the transcript, so a tree needs no second code path — and it is carried because the commit DELETES the source. Measured: 3,017 of 10,841 directories hold one (median 0 bytes, p99 221 MB), and it was in neither list before review round 1 (B-M2) |
| **referenced attachment blobs** | `attachments/<digest>.bin` are content-addressed and shared per install (`session/attachments.py:1-30`) — a transcript that references a digest the destination lacks renders a broken image, so the plan includes every digest the transcript references that the destination does not already hold |
| **each blob's `.json` sidecar** | `attachments/<digest>.json` carries the mime type the reference resolves through (`session/attachments.py:16`). A blob that arrives without it is a download of the wrong type, and it was missing from every earlier copy and every replica recovery (review round 1, M-1) |

**Never copied, each with its reason** (`sync.EXCLUDED_ENTRIES` is the
authoritative list; `sync.NEVER_COPIED` is the tuple the code branches on):
`.session.pid` (the liveness marker — `fork.py:127` states the failure: the
destination would report the *source's* pid as the owner), `mesh.json` (the
ownership stamp, rewritten by whoever adopts the copy), `sync.json` (a replica
cursor: about the copy, not the session), the `archived-sessions.json` **index** (an
install-level fact, not session state, and `--keep` of an archived session must not
make the copy archived — the same reason `session/archived.py` refuses a
per-session sidecar), `subagent-roster.v1.json` (parent-owned jobs,
`fork.py:128-132`), the scan sentinels (`fork.py:133-136`), the two halves of the
transcript lease (`.execution-lease` / `.execution-lease.recovery`: a claim copied
with a session would name the SOURCE's pid as the copy's writer, and the destination
could not open the conversation it just adopted), `.wake-write.lock` (a per-device
lock), `.browser-resource.json` (a browser-bridge ownership record minted by THIS
device's bridge generation), `origin-verdicts.json` (a recomputable cache),
`ready.json` (the move's own boot marker; the promote deletes it, and recovery
removes a stray one), and any machine-local cache or claim DB.

**The lists are checked against the product, and a gap FAILS CLOSED.** Two tests and
one runtime guard, because a file list cannot notice a file type nobody told it
about: `tests/unit/network/test_sync_copy_set.py` imports each entry name from the
module that creates it (and separately enumerates the names measured in a real
store), `test_mobility_integrity.py::test_a_move_carries_the_scratchpad_and_the_birth_time`
copies a directory holding every one of them, and
`sync.assert_complete` refuses a DELETING move whose source directory holds anything
neither list accounts for — at `prepare` (before the retire and before a byte is
copied) and again at the commit (an entry can appear in between). ``--keep`` copies
what it knows and leaves the source alone, so it has nothing to lose by skipping an
entry it cannot carry.

### 7.3 The cursor, and how a destination merges or replaces

A transcript is **not** a pure append-only file: `Transcript.compact_file`
(`session/transcript.py:1807`) folds the prune journal and rewrites it with
`os.replace`, and it does so *"semantically invisible: entry ids, order and types are
unchanged"*. So byte offsets are not a valid cursor, but **entry ids are stable and a
generation counter already exists**: `transcript._history_generation`, exposed as
`history_generation` on the display window (`session/history_window.py:82`, `:520`)
and validated on every sync by `AttachedSession._validate_display_window`
(`attached.py:4239-4246`). The cursor is:

```
cursor = (history_generation, through_entry_id, prefix_bytes, prefix_digest)
```

The planner (owner side) answers, for a destination's `have` cursor:

* **same generation, `through_entry_id` present in the source's id list, and the
  source's bytes up to the end of that row hash to the same `prefix_digest`** →
  `append`: send only the rows after the cursor, plus the new generation counter.
* **generation changed, or the id is gone, or the prefix digest differs** →
  `replace`: send the whole transcript; the destination writes a staged file and
  `os.replace`s it.

The destination never rewrites rows it already has, and never merges a rewritten
transcript row-wise. `history_generation` is the owner's own counter, so the
destination does not have to guess; the prefix digest makes "append" a verified
claim rather than a hopeful one. (Implementation note for the coder: confirm at
`history_window.py:355` and the `compact_file` call sites that every in-place
rewrite bumps the counter; if one does not, it must, and that is a one-line change
with a test that a compaction between two syncs forces `replace`.)

`title.json` / `attachment.json` / `origin.json` / the markers are small: sent when
their **mtime+size differ** from the destination's manifest entry (the plan carries
`{name, size, mtime_ns, digest}` per file). `inbox.jsonl` is append-only and is
merged by row id with the same generation rule applied to its own row count.

### 7.4 What "consistent" means, per mode

* **`move`** copies *after* the source runtime retired (§6.3 step 3), so the
  transcript is quiescent and the copy is exact. This is why `move` is the strict
  mode.
* **`--keep` / `flush`** copy from a runtime that is still writing. Two rules make
  that safe:
  1. The source serves the transcript up to the **last complete newline of the last
     durable append**; a partially written tail row is dropped by the reader. The
     same tolerance `fork_session` applies to a mid-write parent
     (`fork.py:236-246`: malformed lines are passed through, and the *fork boundary*
     is what makes the copy's divergence explicit).
  2. The copy is a **fork by construction** (`--keep` mode stamps
     `origin: fork` + `fork-boundary.json`), so an off-by-one row boundary is a
     divergence point rather than a corruption. `fork.py:19-30` states the same
     property for the mid-batch case: *"WHEN the copy is taken is load-bearing"* —
     and a cross-device copy cannot use `Session.request_fork`'s in-process boundary,
     so it uses the boundary *marker* instead.

**The `--keep` window, stated explicitly** (review round 1 asked for this to be
documented or removed). There is no grace period and no fence in the `--keep` branch,
and nothing that could be removed to make one: the copy is served from a LIVE runtime,
so a turn that lands while the copy is in flight may be absent from the copy while
present on the source. **The window is therefore the copy's own duration** — measured
on the operator's workstation at ~1 s for a session of a few hundred KB over loopback,
and bounded by the transfer rather than by a timer. Two mechanisms make the result an
honest snapshot rather than a splice: the plan guard (`_require_current_plan`) refuses
a plan the source has moved past, with `sync_from` re-planning up to
`SYNC_REPLAN_ATTEMPTS` times — so a source writing continuously produces a refusal,
never a splice — and the copy is a FORK by construction (`origin: fork` +
`fork-boundary.json`), so an off-by-one row boundary is a divergence point.

### 7.5 Cadence (R22)

| Trigger | Who | Cost |
|---|---|---|
| on demand (`lop sessions sync <id>`, `/sync`) | any member | one plan (2 stats + 1 hash) + the delta |
| after a turn settles, **debounced** (≥ 30 s apart, ≥ 1 new turn, per destination) | the owner relay pushes `sync.available {session_id, generation, frontier}`; the destination decides whether to pull | one push per turn (bounded by event rate, not by frames) |
| **the final flush before idle/power-off** | the device that is going away: its relay runs `flush` for every session it holds before `lop network drain` / its shutdown handler completes | one plan + the delta per session |
| at destination attach (`--keep` copy being viewed) | the destination, if its last sync is older than 10 min | one plan |

**The push is a WAKE-UP, and the holder owns the pull** (review round 1, M-4). The
owner's watcher sends `sync.available`; the holder's `ReplicaRefresher` — one per
relay, started lazily by the first push, one tick per `network.sync.tick_s` — pulls
every marked replica from its recorded owner over the link the push came in on. That
split is what keeps the link's reader thread out of a request it is serving, and it
is why a burst of pushes for one session costs ONE copy rather than one per frame: a
push that arrives while a pull for that id is running sets a flag, so the later change
is not lost and the next tick picks it up. A replica is therefore at most one tick
behind the owner's last change. Two things had to be fixed for this to happen at all:
the `available` handler's ack was the ENTIRE effect of a push (nothing anywhere pulled
on one), and the holder's chokepoint refused the push itself with *"session … does not
live on this device"* — a holder by definition does not own the id, so the carve-out
admits exactly one frame, `net_sync {phase: "available"}`, and only from the device
that replica was synced from.

The flush is *not* a new mechanism: `lop network drain` (the pod's own `sync.flush`
loop, then the pool spins the instance down) is `net_sync {phase: "flush"}`
against every member that holds a copy — i.e. **a move with `--keep` and no
retirement, run on a timer that ends with the device dying.** Cranking the cadence
down does not change any schema, which is R21's actual requirement.

I/O budget, because R18 asks for the argument rather than the assertion: a flush
reads one appended region per session (the transcript's tail since the last cursor)
and writes nothing on the source; the plan's hash is cached per generation, so a
session synced ten times without a compaction computes it once per generation, not
per sync.

---

**Why this is content-addressed, and what the frontier is** (prior art: restic,
BSD-2-Clause — `mesh-prior-art.md` §5; the negotiate-then-pack shape is git's).
Two facts already in this section are the whole design, and they are worth naming
because a reviewer should be able to check them against a rule rather than
against taste:

- **The blobs are immutable and named by their digest.** `attachments/<digest>.bin`
  is already content-addressed today, and restic's discipline — "all files … are
  only written once and never modified afterwards", the storage id being the
  SHA-256 of the content — is the argument for keeping it that way: an immutable
  blob can be verified by any device, resumed after a dropped link, and copied by
  a peer that cannot read anything else. It also gives the GC story a home
  (`prune` in restic's terms is this document's reconcile, and both only ever
  delete what no live transcript references).
- **The manifest is a flat digest frontier, sent before any bytes.** §7.2's
  `{name, size, mtime_ns, digest}` list is the frontier: the destination answers
  with the digests it lacks, and only those cross the link — one round trip of
  hashes plus the misses, which is git's negotiate-then-pack and restic's "what is
  missing" in one sentence. The `mtime_ns` is a fast-path hint, never an authority:
  a matching mtime with a mismatching digest is a conflict the digest resolves, and
  the transcript's own ordering is settled by §7.3's cursor, never by a timestamp.

**Rejected, and named so it is not re-proposed:** a CRDT merge of session state
(Automerge, Yjs — both MIT, both good at what they do). A CRDT exists to merge
*concurrent* writers, and this design forbids a second writer by construction (the
one lease, `exclusive-move-v1`, §6.6); the transcript is also compacted in place by
`Transcript.compact_file` (`session/transcript.py:1807`), a rewrite a CRDT cannot
represent. The one idea worth borrowing from that family is the state vector, and
§7.3 already is one in single-writer form.

---

## 8. Delete and archive on a peer

The local design is the WIP PR #1328 (`docs/design/session-archive.md` on
`feat/session-archive-delete`), which this section must mirror rather than
re-invent: archive is a durable JSON array at the config root
(`local_operator/session/archived.py`, `archived-sessions.json`), hidden from
default lists and search but resumable by explicit id and reversible; delete is
permanent, refused while the session is in use, and lives behind
`cleanup.delete_session` → `cleanup.remove_session_dir` (the only `rmtree` of a
session directory, walked by `tests/unit/session/test_no_session_deletion.py`).

### 8.1 Routing

```
TUI /archive /unarchive /delete      →  resolve_owner(id)
desktop POST .../{id}/archive        →    ├─ local  →  existing implementation, untouched
                     DELETE .../{id}  →    └─ remote →  relay op → owner runs ITS implementation
```

* `net_session_lifecycle {action: "archive", session_id, archived}` → the owner's
  `archived.set_archived`-equivalent (desired state, idempotent — the wire shape
  PR #1328 pinned: `{"archived": bool}`, `extra="forbid"`, `StrictBool`).
* `net_session_lifecycle {action: "delete", session_id, confirmed: true}` → the owner's
  `cleanup.delete_session`. `confirmed` is required on the peer hop exactly as it is
  on the HTTP route (`{"confirmed": true}`; a delete dispatched without it must not
  be a delete).

**Nothing about the local implementations changes except where they get called
from.** That is the point of routing rather than replicating: the guards, the
confirmation semantics, the retention interaction and the wake-index pruning all stay
in one place, on the device that owns the disk.

### 8.2 The in-use refusal crossing the link

The owner answers with the guard's identity **and its sentence**:

```json
{"refused": true, "code": "session_delete_refused",
 "guard": "has an armed wake",
 "message": "That conversation has a wake armed for it. Cancel the wake before deleting it."}
```

* The sentence is the one `cleanup._GUARD_REFUSALS` maps the guard to (there is one
  sentence per guard because the remedies differ: a running session is stopped, an
  armed wake is cancelled, unread mail is read), with
  `_GUARD_REFUSAL_FALLBACK` for a reason it does not carry — *"Whether that
  conversation is in use could not be checked, so nothing was deleted"* — and the
  guard set is closed on the safe side (`_guard` fails closed, `cleanup._guard`).
* The local surface renders `message` **verbatim** and never re-derives a sentence
  from `code`: the peer is the only party that can see which guard fired, and a
  second sentence table here is how the two devices would come to disagree about the
  remedy.
* The four guard names that can cross (`claimed by a live process`, `leased by a
  live process`, `has an armed wake`, `has unread spooled mail`) plus the two
  fallbacks are the whole vocabulary; `guard failed: <Exc>` also crosses, and is what
  the phrase *"could not be checked"* is for.
* HTTP: the desktop route keeps answering `409` with
  `{"code": "session_delete_refused", "message": …}` (PR #1328's contract) whether
  the refusal came from this device or a peer — the client cannot tell, and must not
  have to.
* The confirmation copy gains the device: *"Delete 'mesh design' **on build-box**?
  This removes one conversation and cannot be undone."* The blast radius is another
  machine, so the machine is named before the act.

### 8.3 The local archive index and a remote session

**A remote session's archived state is never written to the local
`archived-sessions.json`.** Two independent reasons:

1. The index is **pruned at read** against the local session store (its module
   docstring, and `resume._scan_sessions`'s `archived=archived_ids(config_dir)`
   path): an id with no local directory is dropped, so a written remote id would
   evaporate on the next read anyway.
2. Two writers over one file for facts that belong to different devices is the
   arbitration `archived.py` explicitly refuses to take on (*"last writer wins … the
   granularity is the whole index"*) — it would let an archive performed on one
   device silently erase another's.

So a remote row's `archived` comes from the **peer's catalogue answer** (§9.2),
merged into the local list the same way `pinned` already is: an absent key is not a
claim, a present key settles it. Consequences, stated so nobody treats them as bugs:

* `include_archived=true` on the local list is forwarded to the peers, so the reveal
  toggle shows remote archived rows too; with the flag off they are absent, exactly
  as they are locally.
* When a peer is **unreachable**, its archived rows are simply not in the list — there
  is no local state to leak and no phantom row. The last-known cached row (if any) is
  marked `unreachable` and hidden from the archived reveal until the peer answers
  (§13 Q4).
* Deleting a remote session never touches the local archive index or the local pin
  store: both prune at read, so the id simply stops being resolvable locally — the
  property PR #1328 asserts by leaving stale records on disk after a delete.

### 8.4 What a remote delete must NOT do

* **Not delete the local directory of a moved-away session** — there is none; the
  moved id's directory was removed at handoff (§6.3 step 14) and the tombstone is
  what remains. If a device somehow has both a directory and a tombstone (the crash
  window of §6.5), the **tombstone wins**: the id is not offered, not resumable, and
  `/delete` on it is answered with the tombstone sentence rather than by deleting the
  leftovers (reconcile owns those).
* **Not delete a fork's source**: `cleanup.delete_session` refuses an id that is not
  `is_user_session` (`resume.USER_ORIGINS`), and a `--keep` copy is a user session on
  its own device — deleting it leaves the original alone, which is the whole point of
  the mode.

---

## 9. The UI-facing contract

### 9.1 The catalogues this must not re-shape

* Desktop row model: `SessionRow` (`server/models/desktop_sessions.py:14`) with
  `pinned` REQUIRED-with-both-values because the renderer's merge is "an absent key is
  not a claim" and a missing `pinned` would leave a stale optimistic `true` forever
  (`:28-44`). The same discipline applies to every field added below.
* Desktop list envelope: `SessionList` (`:47`) with `truncated`, `limit`,
  `degraded` — `degraded` names the decoration sources that failed, which is the
  existing vocabulary for "a row I could not fully describe".
* Desktop client row: `SessionCatalogueRow`
  (`local-operator-ui/src/shared/desktop-session-contract.ts:9-33`), whose comment
  says the backend owns status precedence, partitioning and order — so peer grouping
  is a *backend* arrangement, not a client inference.
* TUI row: `CatalogEntry` (`session/catalog.py:38`) built by `load_catalog`
  (`catalog.py:891`) and ranked by `rank`/`rank_entries`; sections come from
  `SessionSidebar._section_of` (`tui/widgets/session_sidebar.py:438`) with names
  `_SECTION_NAMES` (`:197`).
* CLI: `session_rows` (`info/collect.py:595`) whose key order is a **published
  contract** pinned by `tests/unit/info/test_sessions_extraction.py`, extended only
  by appending (`completion_kind`, `completion_reason` set the precedent in-file).

### 9.2 The fields added

The transport design already pins **two** of them and this document does not restate
them (its §9.2/§9.3/§9.5): `locality` (`"local" | "remote"`, always present) and
`peer` (`{device_id, name, network_id, reachable, age_s}`), on the `net_catalog` row,
on `lop sessions --all-peers --json`, and additively on the desktop
`SessionCatalogueRow` (`local-operator-ui/src/shared/desktop-session-contract.ts:9-33`),
gated behind `features.peers: 1`.

This document adds **three** row keys, each optional and each with a defined meaning
when absent ("absent is not a claim", the rule `SessionRow.pinned` documents at
`server/models/desktop_sessions.py:28-44`):

| Field | Type | Meaning | Absent means |
|---|---|---|---|
| `placement` | `{mode, network_id, home_device, policy}` | spine A8's field, from the session's `mesh.json`: where it runs (`local`/`peer`/`pool`) and the policy that governs it | a pre-mesh session, i.e. `{mode: "local", policy: "pinned"}` |
| `origin` | `{kind, source_device, source_session_id} \| null` | how it got here: `"moved"` (id preserved, ownership transferred) or `"fork"` (a `--keep` copy or a local fork promoted across the mesh) | no provenance claim; the row shows no "copy of…" subtitle |
| `last_synced_at` | `float \| null` | for a `--keep` copy on a device that is not the owner: when this device last pulled (§7) — R22's visible half | never synced, or a session this device owns |

`locality` and `peer` are **required** on a remote row and `peer` is `null` on a local
one (the transport's shape); the three above are additive and optional on both. The
client groups by `peer.name` and treats an absent `peer` as the local group, so an
absent key can never file a row under the wrong device.

TypeScript (additive to the same interface, alongside the transport's `locality`/
`peer`):

```ts
placement?: { mode: "local" | "peer" | "pool"; network_id: string;
              home_device: string; policy: "pinned" | "prefer-remote" | "cost-capped" };
origin?: { kind: "moved" | "fork"; source_device: string; source_session_id: string } | null;
last_synced_at?: number | null;
```

Budget, per the spine's measured constraint (desktop sidebar 280 px default,
240–360 clamped, `h-8` rows, one trailing statement already spent at
`chat-sidebar.tsx:505-528`): the locality annotation joins the **leading
status-glyph slot** (`ChatSessionStatus`) and the tooltip, and a peer is a **group
heading** (the heading primitive already renders label + count) — no new column, no
trailing slot consumed, and `placement`/`origin` are tooltip-and-subtitle material
only. The desktop transport's single `backendUrl` per app
(`local-operator-ui/src/main/desktop-transport.ts:44-51`) is why the fan-out lives in
the local `lop serve` and the desktop never dials a peer: **one backend, N peers, one
contract.**

### 9.3 Routes and capabilities

`features.peers: 1` and the peer list route are the transport's (§9.5); this document
adds only what mobility needs.

| Surface | Change | Owner |
|---|---|---|
| `GET /v1/capabilities` `features.peers: 1` | additive key | transport §9.5 |
| `GET /v1/desktop/sessions`, `…/search` | `include_peers: bool = False`; default false ⇒ a client that does not ask sees today's answer | transport §9.5 |
| `GET /v1/desktop/peers` | peer list: id, name, networks, reachability, latency, session count | transport §9.5 |
| `GET /v1/capabilities` `features.session_transfer: 1` | the key for the transfer route below. `peers` advertises the *catalogue*; it does not advertise the ability to move a session, and a renderer given only `peers` would mount a control that 404s. `mesh-ui.md` §2.6 states the key and the mounting rule; it is listed here so this route table and that document's `features` map are one list rather than two | `mesh-ui.md` §2.6 (key); **this document** (the route) |
| `POST /v1/desktop/sessions/{id}/transfer` | `{to: <device_id\|"local">, keep: bool, wait_s: int}` → the phase transcript of §6.3 | **this document** |
| `POST /v1/desktop/sessions/{id}/sync` | run `net_sync` now and return the manifest delta; the desktop's "sync this copy" action | **this document** |
| `POST .../{id}/archive`, `DELETE .../{id}` | shapes unchanged (`{archived: bool}`, `{"confirmed": true}`); now routed to the owner, refusal envelope included (§8.2) | routing here; shapes PR #1328 |
| `POST .../{id}/fork` (exists) | the fork is created on the owner; the receipt gains the fork's `session_id` **and** its `peer`, so the desktop opens a window attached to the remote fork (§4.1) | **this document** |

`lop sessions` (CLI) — the transport defines the listing flags and the payload; this
document adds the verbs and the row keys:

| Form | Effect |
|---|---|
| `lop sessions [--json]`, `--peer`, `--all-peers`, `--limit` | the transport's §9.3 forms, unchanged |
| `lop sessions move <id> --to <peer\|local> [--keep] [--wait N] [--open] [--json]` | §6.8. `--json` emits the phase transcript (`prepared` / `handing_off` / `committed` / `done`) so the agent path (R19) can drive and verify it |
| `lop sessions sync <id> [--json]` | §7: force one `net_sync` and report the delta |
| row keys added by this document | `placement`, `origin`, `last_synced_at` — **appended** after the transport's keys, preserving the published key order (`info/collect.py:646-666` states the rule) and with `tests/unit/info/test_sessions_extraction.py` updated in the same change |

TUI: `/peers` (alias for `/network peers`, spine §6), `/new remote <peer>`,
`/move <id> --to …`, and the sidebar's peer group. Slash registry entries
(`local_operator/slash_commands.py:67-813`) are added for `move` (already exists at
`:204` with `desktop_destination="session.move"`), `peers`, and the `/new` argument
form; each entry's `desktop_destination` is *"the ONE host-specific fact per entry"*
(`slash_commands.py:7-11`), so a new entry must name its desktop destination or
deliberately leave it unset with a comment (the `/resume`-style precedent at
`:127`).

---

## 10. Security properties and threats

Properties this design must hold, each with the mechanism that holds it:

1. **INV-1 cannot be violated from either side** — the owner's exclusive fence +
   `engage_runtime`'s handoff guard (§6.6) + the destination's epoch-gated promotion
   (§6.3 step 16). Test: §11's `move-under-load` and `source-resumes` cells.
2. **No session control key on the wire** (spine §5.4) — the key is read by the
   peer-side relay from the local record (`SessionRecord.control_key`, `0600` under
   `0700`) and used only for its own loopback dial. A peer never receives one. Test:
   assert no frame on the peer link contains the owner's `control_key` string.
3. **`locality` is [redacted]** (spine §5.8) — the peer-side relay stamps
   `"remote"`; a forwarded frame claiming `"local"` is refused as a protocol error.
4. **Least authority** (spine §5.3) — the auth frame's `grant` set gates the three
   `locality`-sensitive ops (§3.3) and nothing else; every relay op is authorised
   against membership at the current epoch by the transport layer.
5. **Fails closed** (spine §5.7) — a refusal, an unreadable guard, an unverifiable
   digest, an unreachable peer and an unknown phase all *refuse to act* and say why;
   there is no degraded-trust path and no "the link is up so it must be fine".
6. **The delete/archive guards travel with the act** — the peer enforces its own
   guards and answers with the sentence; the local side cannot bypass them by
   calling a lower-level op, because there is no lower-level op (§8.1).
7. **Revocation reaches a handoff in flight** (§6.5's revoke row) — a revoked
   destination is never the only holder of a transcript.

Threats explicitly in the design's view, with the answer:

| Threat | Answer |
|---|---|
| A hostile peer asks to move a session onto itself and then deletes it | Only the owner deletes, and only after a verified complete copy exists on a *current member*; the handoff is audited (R18), and `lop network log` shows it |
| A peer replays an old `move.ready` after a rollback | `lease_epoch` is minted per attempt and checked at every step; a stale epoch is refused |
| A compromised viewer on the peer link writes "local" frames to escalate | §10.3; the runtime's own locality-gated refusals and the relay's grant check are the second line |
| A pod (future, R20) is compromised and calls `net_session_lifecycle {action: "delete"}` on everything | Requires the `delete` grant; a pool member's grant is `list`/`view`/`prompt` by default, and `delete` is never granted to an ephemeral member without an explicit operator act (spine §9.1: member kind and lifecycle are fields, so the transport can refuse by kind) |
| An in-flight conversation is lost by a move | Impossible by construction: a busy session is never moved (§6.4) |
| Two devices each believe they hold the id after a crash | The tombstone + epoch rule (§6.5); the promote requires the owner's acknowledgement, and the owner's answer is monotone |

---

## 11. Test plan

Written for the QA agent as literal commands. **Isolation first**: per
`AGENTS.md:511-600` (`LOCAL_OPERATOR_CONFIG_DIR` alone is not enough — the *cache*
root comes from `HOME`, and two `LOP_*`/`CMUX_*` prefixes are read by the child
product):

```sh
# Device A and device B on one host — two installs, two config roots, two HOMEs.
# Every cell below runs with BOTH prefixes stripped and the ones the cell means set.
DEV_A="env -u CMUX_WORKSPACE_ID -u CMUX_PANEL_ID -u CMUX_SESSION_ID \
       -u LOP_RUNTIME_ADOPT_SESSION -u LOP_RUNTIME_DEFER_MATERIALISE \
       -u LOP_MOBILE_CHILD_PROVIDER -u LOP_MOBILE_CHILD_MODEL \
       HOME=/tmp/mesh-a LOCAL_OPERATOR_CONFIG_DIR=/tmp/mesh-a/.local-operator \
       PATH=$PWD/.venv/bin:$PATH"
DEV_B="…HOME=/tmp/mesh-b LOCAL_OPERATOR_CONFIG_DIR=/tmp/mesh-b/.local-operator…"
```

### 11.1 Gates (whole tree, CI's shape; CI remains authoritative)

```sh
.venv/bin/python -m flake8 .
uvx --from black==26.1.0 black --check .
uvx isort==5.13.2 . --check-only
make type-check                      # bounded + process-group-reaped; BOUND_TIMEOUT=<s> to raise
.venv/bin/python -m pytest tests/ -n auto      # 40-55 min under fleet load; check `uptime` first
make check-changed                   # the CI-scoped subset, not a substitute
```

TUI cells need `env -u NO_COLOR TERM=xterm-256color`. Visual claims need rendered
frames (before/after) per `AGENTS.md:1552-1993`; a green test is not visual
evidence.

### 11.2 Targeted unit tests (new)

```sh
.venv/bin/python -m pytest tests/unit/mesh -n0
.venv/bin/python -m pytest \
  tests/unit/session/test_owner_resolver.py \
  tests/unit/session/test_remote_owner.py \
  tests/unit/session/test_handoff_protocol.py \
  tests/unit/session/test_sync_cursor.py \
  tests/unit/session/runtime/test_capability_surface.py \
  tests/unit/tui/test_noop_consumers.py -n0
```

`test_capability_surface.py` and `test_noop_consumers.py` are the existing guards
that fail if a newly advertised slash command is not dispatched by the runtime, or if
a `request`-carrying receipt is missing from `SLASH_ACTION_RECEIPTS`
(`session/runtime/types.py:195-196`) — both must stay red-green honest through this
work.

### 11.3 The topology matrix

Each row: **surface, exact command, what to read, PASS/FAIL**. `A` = local device,
`B` = peer (an EC2 box provisioned with the operator's `minerva_nprod` profile for
the real cross-machine run, per spine §10; two isolated installs on one host for the
fast loop).

| # | Cell | Command | Read |
|---|---|---|---|
| 0 | **zero peers (regression)** | `$DEV_A lop sessions --json`; `$DEV_A lop --resume <id>`; `$DEV_A lop exec 'hi'` | byte-identical to `origin/main` output for the same store; no `mesh.json` written; `placement`/`locality` absent-or-local |
| 1 | pair | `$DEV_A lop network init lab` → `$DEV_A lop network invite` → `$DEV_B lop network join <token>` (both confirm the SAS) | `lop network peers` on both sides; audit lines |
| 2 | **create remote** (R8) | `$DEV_A lop exec --peer B 'say hi'`; and in the TUI `/new remote B hello` | the session directory exists under `/tmp/mesh-b/.local-operator/sessions/<id>/` and NOT under A's; `$DEV_A lop sessions --all-peers --json` shows it with `locality: "remote"`, `owner_device_name` set |
| 3 | prompt | type in A's window attached to the remote session | a new user row in B's `transcript.jsonl`; A's window paints it |
| 4 | steer | send mid-turn from A | the steer lands on B (row + `steer` echo); no second turn |
| 5 | stop | `/stop` in A's window; then `lop stop <id>` from A | B's runtime exits (its record disappears on B); A shows the stopped state; **B's transcript is intact** |
| 6 | **quit locally (R9)** | note B's runtime pid; `q` A's TUI; from C/B prompt again | B's runtime **pid unchanged** (or a fresh runtime engaged on B by the next turn, never a local one), and A's exit produced no local session for that id |
| 7 | **link loss (R9)** | `$DEV_B lop network stop` (relay down) while A has the session open | A's row flips `reachable: false` with a reason; A does **not** engage a local successor (assert no new record in A's `run/mobile`, no directory under A's `sessions/`); A's next prompt refuses with a sentence |
| 8 | archive on a peer (R10) | `/archive` in A's window on a remote session; then `$DEV_A lop sessions --json` and `--all-peers --json` | the id is in B's `archived-sessions.json`, absent from both default lists, present with `include_archived`; A's own `archived-sessions.json` does **not** contain the id |
| 9 | delete on a peer — refused | leave a wake armed on B, then `/delete yes` from A | B's guard fires; A prints **B's** sentence verbatim (`"…has a wake armed for it. Cancel the wake before deleting it."` for a wake, etc.); B's directory is intact; HTTP path returns `409 {"code": "session_delete_refused", "message": …}` |
| 10 | delete on a peer — performed | `/delete yes` with no guard armed | B's directory is gone (and only that one — a sibling subagent run survives); A's list loses the row; audit has the delete on B |
| 11 | **move peer → local** (R11) | `$DEV_A lop sessions move <id> --to local` | phase transcript `prepared → handing-off → committed → done`; A has the directory + `mesh.json {home_device: A, origin.kind: "moved"}`; B has a tombstone and **no directory**; a fresh turn on A works; `$DEV_B lop --resume <id>` refuses with the moved-to sentence |
| 12 | **move local → peer** | mirror of 11 with `--to B` | as above, ends swapped |
| 13 | **move --keep** | `move <id> --to B --keep` | B has a **new id**, `origin.kind: "fork"`, `fork-boundary.json` present, `.session.pid` absent, no roster; A's original is untouched and still running; the two transcripts diverge after the copy |
| 14 | **move under load** (R12) | start a long turn on A, immediately `move <id> --to B` | `refused` with the owner's idle sentence; **no mutation** (A's directory, A's record, B's store all unchanged); with `--wait 60` the move lands after the turn settles, and the settled turn's rows are all present on B |
| 15 | **source resumes / crash windows** (§6.5) | (a) `kill -9` B's relay after `ready`; (b) `kill -9` B's relay between `handing-off` and the delete; (c) `kill -9` A's relay after `committed`, before promoting | each: restart the relay, run `lop network doctor`, and assert the session ends up on exactly one device with the correct transcript; (c) must promote at A without a retry from B |
| 16 | ownership refusal | move the session to B, then ask A to archive/delete/prompt it | `session.ownership_refused` in the audit; the sentence names B; no action taken |
| 17 | sync / R22 flush | with `--keep` on B, run `lop sessions sync <id>` twice; then `lop network drain` on B | second sync transfers only the delta (assert the byte count drops to the append region); after the drain, A's copy holds the final assistant message without any further link |
| 18 | catalogue delta | prompt on the remote session | A's sidebar row updates without a full reload; `sessions.delta` on the peer link is one event per change, not per frame (R18) |
| 19 | process/terminal-local exceptions | `/quit`, `/resume`, `/clear`, `/copy`, `/theme` from a remote session | handled locally; `/quit` does not touch B; `/resume` on B's runtime is refused with the re-worded sentence (§4.4) |
| 20 | receipts | `/team <name> <request>` and `/goal x` from a remote viewer | exactly ONE turn on B (assert the transcript has one `request` row, not two) — the §3.5 double-turn guard |
| 21 | `/mcp login` remote | `/mcp login <server>` from A's window | the browser opens **on A**; the token lands in **B's** `auth.db`; `REMOTE_GRANT_NOTICE` is NOT shown |
| 22 | **stream pass-through (R-IF-1)** | open a stream with `stream_open`, then write session frames directly | the owner's runtime sees one attach connection per viewer connection (`lop sessions` on the peer shows the count); `stream_send` is not needed by this client; a frame written before `stream_open` is answered `unknown_op`/`protocol_error`, never forwarded |
| 23 | credential binding survives a move | with a brokered credential bound to the session, move it both ways, then compare | `mesh-credentials.md` §5.5's three assertions, run as its own cell because a move is not a credential event |
| 24 | **three peers** (topology 2) | pair A, B, C; move a session A→B→C; list from all three; run §11.3 #6 from C while A is the viewer | listing agrees on one owner at each step; the move A→B→C produces no duplicate id on any device; simultaneous ops from two viewers serialise at the owner |

### 11.4 Evidence to attach to the PR

Per requirement: the command, its actual output, and the observable side effect —
`lop sessions --json` (before/after) from both devices, `ps -p <peer runtime pid>`
proving R9, the peer's `transcript.jsonl` tail, the audit lines for each handoff,
the refusal sentences verbatim, and rendered frames (before/after) for the TUI
sidebar peer group and the desktop sidebar's group heading + locality mark.

---

## 12. Rejected alternatives

| Rejected | Why |
|---|---|
| A second facade class (`RemoteSession`) implementing `ViewerSessionProtocol` | It would duplicate ~4,000 lines of display/gate/history fold that are already transport-agnostic, and would fork the desktop bridge's `AttachedSession` dependency (`server/utils/desktop_sessions.py:1049`). Spine §3 forbids the second front-end path; the injected-collaborator seam is 3 methods |
| A proxy runtime process per remote session publishing its own `run/mobile` record | Records are pid-keyed (`registry.record_path`, `registry.py:109`; `RecordPublisher`, `:683`), so one process cannot publish N records; one process per remote session is a process per session on the *viewing* device, which is the resource the mesh exists to avoid |
| One shared upstream control connection for all local viewers of a remote session | `slash_consumers` (and `events`, `frontend_state`, `display_window`, `surface`, the watch leases) are per-connection facts (`server.py:2704-2762`); aggregating them either drops a receipt's request or double-submits it (`types.py:202-216`). Also burns the single `daemon` slot, evicting the peer's own phone daemon (`server.py:2734`) |
| A standing presence subscription per remote session | Costs a viewer slot on the owner forever, and duplicates the catalogue's job; row facts belong in the catalogue, as they do locally |
| New record namespace for projected remote sessions (`run/remote`) plus a widened scan | Adds a second namespace for something which is not a local record at all (there is no local pid, port or key to publish); the catalogue is the right carrier for a remote session's facts, and the record namespace rule (A2) is about record *kinds* |
| Answering `"unknown"` from a remote facade instead of widening `RuntimeLocality` | My first preference, and the transport design's §12.2 is better: the union is three-valued so the answer is honest, and here it is *known* rather than unprovable. §1.3 |
| A targeted `net_session_locate` op | A second, redundant answer to a question `net_catalog` already answers; its only unique case (the id was moved away) is a tombstone the move op's `status` phase reports (§2.1) |
| The relay-mediated OAuth callback for `/mcp login` (browser on the viewer, `{code, state}` forwarded to the owner) | The area is `mesh-credentials.md`'s and its §4.7 resolves it as `REMOTE_GRANT_NOTICE` + the repair path; I adopt that for v1 and record the gap in §13 Q-CRED rather than ship two mechanisms |
| Multiplexing or aggregating `slash_consumers` | §3.5 — double-submit |
| Reusing the `daemon` client kind for the relay | One daemon slot per runtime; the phone owns it (`server.py:2734-2740`) |
| Draining an in-flight turn for a move | A refusal cannot lose a turn, a drain can; the idle predicate is already a refusal protocol (`server.py:3999-4029`). §6.4 |
| `--keep` giving the destination the same session id | Two devices, one address, forever (§6.2) |
| A third device issuing a move between two others | Adds a party that cannot verify either end's disk; refused in v1 with a sentence (§13 Q3) |
| Copying the archive index to peers, or writing remote ids into it | Pruned at read against the local store, and two writers over one whole-file index (§8.3) |
| Copying credentials with the session (`auth.db` to the destination) | The measured token-invalidation failure A5 exists to prevent; credentials are brokered (`mesh-credentials.md`) |
| Byte-offset sync cursors | `Transcript.compact_file` rewrites the file with `os.replace` (`session/transcript.py:1807`); the cursor is `(history_generation, entry_id, prefix_digest)` (§7.3) |
| A new `PROTOCOL_VERSION` for these fields | The record's additive contract is explicit (`types.py:739-746`): a field nobody must read does not spend the one number that carries frame compatibility |

---

## 13. Open questions, each with my recommendation

1. **Does `/new` on an already-remote session create locally or remotely?**
   *Recommendation:* locally (stated in §4.2). Settling evidence: user testing of the
   "fresh chat while looking at a peer's session" gesture; if users report reaching
   for remote, `/new remote` becomes the default for a remote context and the current
   behaviour moves behind `/new local`.
2. **Should `include_peers` default to true?**
   *Recommendation:* false, so a client that does not know the parameter keeps
   byte-identical behaviour and the desktop opts in after negotiating `peers: 1`.
   Settling evidence: how many desktop builds are in the field that would silently
   gain rows they cannot annotate; if the answer is "none", default true in the
   release that adds the UI.
3. **A third device requesting a move between two others.**
   *Recommendation:* refused in v1 with *"run this from <destination>"*. Settling
   evidence: whether `/move` from a phone becomes a real request; if it does, the
   phone is a *viewer* of A's relay, so it is still one of the two ends.
4. **How long does a last-known cached row for an unreachable peer survive?**
   *Not this document's decision:* the transport owns the cache and sets 24 h
   (`network/catalog.json`, its §9.4), and its rule — rows from cache carry
   `reachable: false` + `age_s`, and the UI never renders a cached row as live — is
   the one this design consumes. The only mobility-specific consequence: an
   unreachable row's session cannot be moved, archived, deleted or synced, and each
   of those must refuse with the peer's own reason rather than a generic timeout.
5. **May a remote viewer answer approval gates with only the `prompt` grant?**
   *Recommendation:* yes, and audit it with the device (`session.gate.answered
   {device, session, gate}`). An approver is exactly as powerful as a prompter, so a
   separate grant would be a distinction without a difference. Settling evidence: an
   operator who wants "may read and prompt, may not approve" — that needs a real
   product decision, not a schema change.
6. **What happens to a `ready` staging directory whose owner is gone for longer
   than the grace period?**
   *Recommendation:* never auto-promote and never auto-delete: the staging is
   exempted from the GC and the row surfaces an actionable state (*"a move from
   <device> is waiting to be confirmed"*) with a `/network doctor` remedy. Settling
   evidence: whether an operator finds a stuck row confusing enough to want an
   explicit "take it" button — which needs the owner to be unrecoverable *and* the
   operator's judgement, so it must be a deliberate act.
7. **A synced replica of a session whose owner is gone comes back as a NEW
   session, never under the original id.** *Implemented deviation from the first
   draft of §6.5, and the reason is INV-1 at rest rather than a permission question:*
   a peer that spun down can come back, and a same-id restore would then have two
   devices believing they hold the id — the state the tombstone, the epoch and the
   promote acknowledgement exist to make impossible, but which nothing can resolve
   once both copies are established and neither device is reachable from the other.
   So the restore is a fork: a fresh id, `origin.kind: "fork"` naming the source
   device and session, a `fork-boundary.json` and a transcript that ends where the
   replica's last sync ended. The cost is that the recovered conversation is a new
   row and the model is told it is a fork, which is the honest description of what
   it is: a copy that may be missing the owner's last turns.
8. **`lop exec --peer --background`.**
   *Recommendation:* refused in v1 (the background-job ledger and its log file are
   local to the spawning device; making them device-aware is a job-ledger change, not
   a mesh change). Settling evidence: demand from the agent path (R19) — a scheduled
   wake that wants its one-shot to run on a peer.
9. **Where does `mesh.json`'s `stamp_revision` get bumped?**
   *Recommendation:* on any rewrite of the stamp (placement change, policy change);
   NOT on every turn. Settling evidence: whether cache invalidation needs a finer
   grain (it does not, if the catalogue revision also moves on row-relevant changes).
10. **Should `stream_open` put the connection into a pass-through mode (R-IF-1)?**
   *Recommendation:* yes — after `stream_open`, session frames flow in both directions
   with no wrapper, and `stream_send {stream, frame}` stays the multiplexed form for a
   client that wants several sessions on one socket (§3.2). Settling evidence: whether
   the desktop or the phone ever wants N sessions on one connection (they do not
   today: the desktop has one `backendUrl` and one window per session, and the phone
   daemon already multiplexes at its own layer). If the transport prefers wrappers
   everywhere, the cost is a second frame-shaped parser in `RemoteSessionClient`'s
   transport half — measured, not assumed, before that trade is taken.
11. **`/mcp login` on a remote session: accept the repair path for v1?** (Q-CRED)
   *Recommendation:* yes, for v1 — it is the credentials design's area
   (`mesh-credentials.md` §4.7), it does not depend on a fixed local port being
   bindable on a device we do not control, and it keeps one refusal sentence
   (`REMOTE_GRANT_NOTICE`). The follow-up that closes it fully is the relay-mediated
   callback `mcp/grants.py:23-26` itself describes. Settling evidence: how often a
   member device is one the operator is physically at when the login is needed — if
   the common case is a *pod* (no human, no browser), the repair path is the only
   coherent answer and the callback is not worth building.
12. **Is a pushed catalogue delta needed?** (Q-CAT)
    *Recommendation:* not in v1 — the transport's 2 s cache over a 60 s peer
    refresh is enough for a group heading, a count and a status glyph, and a pushed
    delta is a new op with its own capability and its own event-rate budget (§3.6).
    Settling evidence: measured staleness of a *completion* mark on a remote row in
    real use; if a peer finishing a turn is noticed late enough to matter, the delta
    is the fix and this document's row fields are already its payload.
13. **Does an idle remote session keep its runtime warm?**
   *Recommendation:* no — the residency drain is the owner's business and unchanged
   (`process.py:645` `_should_exit`); a remote viewer's arrival engages on demand
   (`net_session_engage`). Settling evidence: measured attach latency over the link if
   the engage turns out to dominate; a "warm on catalogue open" heuristic is then a
   one-line policy at the viewer.

---

## 14. Compatibility, rollout, and risks to watch

**Compatibility.**

* No `PROTOCOL_VERSION` bump and no `run/mobile` record-shape change that an older
  reader must understand: `placement` is additive with an explicit `null`/absent
  meaning, and an older reader ignores it (`types.py:739-746`, `from_json` drops
  unknown keys, `:879-883`).
* A session directory without `mesh.json` is local — so an install with no network
  behaves exactly as today (R16), and a build that predates this work treats a
  moved-in session correctly, because the stamp's absence is exactly what it assumes.
* The dangerous direction is a *stale* build on the handing-off device: it would see
  the directory deleted (after a move) and simply not list the session — correct. It
  does not read the journal, so it could resurrect a `handing-off` session by
  engaging one — but `engage_runtime`'s guard is in the *new* build only. Rollout
  therefore requires: **on any device that hands a session off, the binary must be
  the mesh build** (the move is issued by that device, so this is checkable at the
  request: `O`'s relay refuses `move.prepare` unless its own build advertises the
  handoff capability).
* `lop` runtime is a separate non-editable install updated by `lop-update`; nothing
  here changes that. Branch switches do not affect the running runtime.

**Rollout order, and why.** (1) transport/identity (the link) → (2) this design's
§2–§4 (locate, project, the command matrix) → (3) §8 routing → (4) §7 sync → (5) §6
mobility. Mobility is last because it is the only irreversible operation, and it
depends on the sync primitive (4) which depends on the catalogue (2).

**Risks to watch during rollout**, in the order I would expect them:

* **The relay as a residency term.** The peer runtime counts attach clients
  (`server.py:3828`, residency term 3 in `process.py`), so a relay that leaks a
  connection keeps a runtime alive on the peer. Watch: `lop sessions` on the peer for
  idle runtimes with a single stale attach; the local relay must close upstream
  connections the moment their viewer connection dies (the same discipline the
  attach client's `on_disconnected` uses, `attach_client.py:815-822`).
* **`ATTACH_MAX_CLIENTS` eviction under a multi-viewer desktop.** One runtime, four
  slots, and a desktop that opens several windows against one remote session will
  evict the oldest. The eviction is graceful (owner-loss recovery), but the user
  experience is a re-connect. Watch the eviction counter in the peer's logs; the
  remedy is pinned in §13's open question about the cap (raise it, or accept).
* **Sync cost on large transcripts.** The reference session's transcript measured
  216 KB (typical) to ~103 MB (top end, `attached.py:1530`). A flush on the top-end
  case is a 103 MB read at the owner and a 103 MB write at the destination; the
  cadence's debounce is what keeps that bounded. Watch `lop network log` for flush
  frequency and the audit's byte counts.
* **The catalogue poll becoming a heartbeat storm** with many peers. Watch the
  `sessions.delta` rate; the design's answer is push-on-change with a minimum
  interval, which the transport owns.
* **A tombstone that outlives its device's membership.** After a revocation, a
  tombstone naming a removed device is an `unknown` resolution with a sentence that
  names a device no longer in the network — correct but confusing; `mesh-ui.md`
  should render it as *"moved to a device that has left the network"*.

---

## 15. Traceability

| Requirement / decision | Where it is closed |
|---|---|
| **R6** one session list | §9.2 (row fields), §9.3 (routes/CLI), §3.6 (catalogue + deltas) |
| **R7** remote is first-class | §3.1 (one facade), §3.2 (transport), §4 (matrix, default = works) |
| **R8** create remote | §5.3 (`/new remote`, `lop exec --peer`), §9.3 (`/peers` completion) |
| **R9** quitting is never fatal | §1.3 (no takeover, no local engage), §3.3 (`retire_if_pristine` refused), §11.3 #6/#7 |
| **R10** lifecycle where it lives | §8 (routing, guards, sentence, index) |
| **R11** mobility | §6 (both modes, both directions, §6.7 symmetry) |
| **R12** safe under concurrency | §6.4 (refuse-not-drain), §6.5 (crash table), §6.6 (two guards) |
| **A4** fork + retire fenced by the existing lease | §6.1 (primitive table), §6.3 (the ordered protocol) |
| **R18** audit + bounded I/O | §6.3 step 19 (event vocabulary), §7.5 (I/O argument), §9.3 (`degraded`/delta rates) |
| **R20/R21/R22** forward compatibility | §7 (sync is the primitive; R22 = `flush`), §5.2 (`pool` reserved, never produced), §7.5 (cadence) |
| **R13–R16** credentials | §3.3 (capabilities, and the `/mcp login` gap recorded as §13 Q-CRED), A5 referenced, mechanics in `mesh-credentials.md`; §7.2 + §11.3 #23 carry the binding across a move |

---

## 16. Anchor index (everything this document relies on)

Session abstraction: `session/protocol.py:128` (`RuntimeLocality`), `:132`
(`SessionProtocol`), `:157` (`owns_runtime`), `:174` (`outcome_is_synchronous`),
`:191` (`runtime_locality`), `:556` (`ViewerSessionProtocol`), `:1083`
(`admit_prompt`), `:1099` (`answer_gate`), `:1120` (`request_stop`), `:1132`
(`set_working_directory`), `:1136` (`route_shared_slash`), `:1165` (`move_will_wait`).

Facade: `session/attached.py:886` (`AttachedSession`), `:898` (ctor), `:953`
(`_can_go_cold`), `:1487` (`cold`), `:1527` (local transcript read), `:3266`
(`_bind_under_lock`), `:3299` (`engage_runtime` call), `:3388`
(`find_runtime_record` call), `:3849` (`_dial`), `:3921` (`AttachClient`
construction), `:4168` (`_load_frontend_history`), `:4179` (legacy fallback),
`:4239` (`_validate_display_window`), `:4260` (`_fetch_history_page`), `:4841`
(`_load_history`), `:4870` (`_read_transcript`), `:6151`
(`supports_exclusive_move`), `:6901` (`owns_runtime`), `:6911`
(`outcome_is_synchronous`), `:6922` (`runtime_locality`), `:7426`
(`route_shared_slash`), `:8130` (`retire_if_unused`), `:8167` (`dispose`).

Runtime server: `session/runtime/server.py:182` (`_MAX_LINE_BYTES`), `:709`
(`_PAYLOAD_OPS`), `:753` (`_SYNC_PRIORITY_OPS`), `:773` (`_SYNC_LOCAL_OPS`), `:1010`
(`SessionHandle`), `:1093` (`ProjectionSink`), `:1144` (`RuntimeServer`), `:2662`
(`_on_connection`), `:2686` (key compare), `:2700` (locality), `:2715`
(`slash_consumers`), `:2725` (fence vs attach), `:2734` (one daemon), `:2747`
(attach cap), `:3452` (rebind refusal), `:3456` (watch ops), `:3538`
(`retire_if_pristine`), `:3588-3660` (`retire_now` + fence), `:3828`
(`_other_observers`), `:3999` (`_retire_for`), `:4034` (latch re-check), `:4120`
(`_dispatch`), `:4145-4316` (the receipt ops), `:4355` (`_dispatch_payload`), `:4374`
(`fork_snapshot` + locality gate), `:4392` (`slash_result`), `:4423`
(`mcp_credentials`), `:4448` (`credential`), `:4496` (`variables`).

Runtime types/registry/launch: `session/runtime/types.py:93`
(`EXCLUSIVE_MOVE_CAPABILITY`), `:172` (`ClientLocality`), `:179`
(`ATTACH_MAX_CLIENTS`), `:199` (`SLASH_ACTION_RECEIPTS`), `:202`
(`runtime_must_complete`), `:239/260/280` (namespaces), `:287` (`session_dir`),
`:561-579` (cut-off causes), `:605/643` (drain phrases), `:712` (`SessionRecord`),
`:739-746` (additive contract), `:853/856` (`to_json`/`from_json`);
`session/runtime/registry.py:88` (`REAPED_DIRNAME`), `:109` (`record_path`), `:125`
(`publish`), `:178` (`unpublish`), `:207` (stop marker), `:286` (turn journal),
`:395` (`classify`), `:464` (`scan`), `:683` (`RecordPublisher`);
`session/runtime/launch.py:1-40` (the one engage entry point), `:125`
(`PromptErrand`), `:186` (`WarmErrand`), `:289` (`_spawn_runtime`), `:455`
(`RuntimeStartupError`); `session/runtime/serving.py:424` (`ServingSessionHandle`),
`:4087` (`run_slash_authoritative`), `:4133` (`_complete_unconsumed_action`), `:4292`
(`_slash_result`), `:5060` (`browser_is_reachable`); `session/runtime/process.py:645`
(`_should_exit`), `:2322` (spawn env), `:2658` (`main`); `session/runtime/control.py:1-30`
(the ladder), `:237` (`StopOutcome`), `:474/546/643` (pid identity proofs), `:1367`
(`_stop_targets`); `session/runtime/inbox.py:1-40` (spool + ordering).

Front end / wire: `session/frontend_state.py:828` (`_FRONTEND_LOCAL_SLASHES`), `:955`
(`CommandScope`), `:972` (`SlashCapability`), `:982` (`SlashResult`), `:5680`
(`_slash_capabilities`), `:2047` (`slash_capabilities` field); `mobile/types.py:279`
(`ControlOp`), `:346` (`EventOp`), `:443/461` (todos), `:541` (`PendingRequest`),
`:638` (`SessionProjection`), `:781` (`PROJECTION_TRANSCRIPT_LIMIT`);
`mobile/attach_client.py:103` (`_READ_LIMIT_BYTES`), `:686`
(`find_runtime_record`), `:737` (`dialable_owner_record`), `:815` (`AttachClient`),
`:824` (ctor/auth fields), `:929` (`supports_exclusive_move`), `:956` (`connect`),
`:988-1015` (auth frame), `:1049` (`_pump`), `:1384/1398` (`prompt`), `:1520`
(`history_page`), `:1539` (`retire_now`); `tui/app.py:13810`
(`_session_runs_elsewhere`), `:18839` (`retire_if_unused` on quit), `:29179-29247`
(the routing decision), `:38088` (receipt consumption).

Store / catalogue: `session/retention.py:220/254` (lease); `resume.py:56/68/73/85/91/102`
(origin vocabulary), `:249/293` (sidecars), `:1147` (`live_runtime_pid`), `:1288`
(`recent_sessions`), `:1495` (`_scan_sessions`); `session/transcript.py:1807`
(`compact_file`); `session/history_window.py:82/275/355/520` (`history_generation`);
`fork.py:86/111/137/145/155/167/198` (boundary, allow/deny lists, ids, fork, claim);
`session/attachments.py:55/72` (the content-addressed store); `session/catalog.py:38/82/410/438/891`
(entry, rank, marks, sections, `load_catalog`); `session/creation.py`
(`ensure_session_created_at`); `session/session_search.py` (search index);
`session/cleanup.py` (remove/delete — on `feat/session-archive-delete`);
`session/archived.py` (the archive index — same branch).

Desktop/CLI: `server/models/desktop_sessions.py:14/47/95` (row, list, search row);
`server/routes/desktop_sessions.py:1304` (`list_sessions`), `:1886-1895`
(`desktop_viewer_must_submit`, defined `:1844`), `:1961` (route_shared_slash),
`:1984` (`request`);
`server/routes/desktop_wakes.py:659`; `server/routes/desktop_lifecycle.py:241/252/395`;
`server/routes/capabilities.py:29` (`features`); `server/utils/desktop_sessions.py:161`
(`_no_takeover`), `:388` (`_stage_and_replace`), `:525` (`_settle_unconfirmed_move`),
`:636/652` (`move_session`/`_move_session`), `:789-834` (preconditions + exclusive
set), `:1049` (`acquire`); `info/collect.py:311` (`collect_sessions`), `:595`
(`session_rows`); `cli.py:136` (parent parser), `:259` (subparsers), `:617`
(`send`), `:659` (`sessions`), `:1164` (`exec`), `:3522` (`sessions_command`), `:7558`
(`main`); `mobile/peer_send.py:705`; `session/errors.py:374` (`SessionStoreUnavailable`);
`slash_commands.py:7-11/67-813` and `tui/autocomplete.py:179`
(`desktop_destination`); `tests/unit/info/test_sessions_extraction.py` (the pinned
`--json` key order); `tests/unit/session/runtime/test_capability_surface.py` and
`tests/unit/tui/test_noop_consumers.py` (the two guards this work must keep honest).

UI repo (`local-operator-ui`): `src/shared/desktop-session-contract.ts:9-33`
(`SessionCatalogueRow`), `src/shared/desktop-contract.ts:1697+` (route table),
`src/renderer/src/features/chat/components/chat-sidebar.tsx:505-528`
(`rowTrailingStatement` + the one-trailing-statement rule),
`src/renderer/src/shared/components/common/chat-layout.tsx:24-32` (280 / 240–360),
`src/main/desktop-transport.ts:44-51` (one `backendUrl` per app).

Sibling designs, consumed by name: `mesh-transport-identity.md` — §2.3 (module
layout, `projection.py` reserved for this document), §2.5 (`stream_open` /
`stream_send` / `stream_close`, and the relay's keyed loopback control socket), §2.6
(`run/peers`), §4.2 (`MemberRecord.kind` / `lifecycle`), §6.4 (`net_forward`,
`net_catalog`, `net_session_move`, `net_session_lifecycle`, `net_sync`,
`MESH_PROTOCOL_VERSION`, the `caps` strings), §7.1 (the capability model and
`ROLE_CAPABILITIES`), §7.2 (`Authorizer.check` / `dial_local`, the session-scope and
locality rules, the totality test), §7.4 (locality on both sides), §9.2/§9.3/§9.4
(the catalogue row, the aggregation payload, the 24 h cache), §9.5 (the desktop
contract), §12.2 (the `RuntimeLocality` widening);
`mesh-credentials.md` — §3.1/§3.2 (`net_broker` and its `kind` discriminator,
`net_credential_grant`), §4.7 (`/mcp login` and the repair path), §5.5 (a moved
session's binding), §5.6; `mesh-incident-response.md` (audit retention, the
`session.handoff.*` event kinds' bounds); `mesh-ui.md` (rendered surfaces).

Process: `~/local-operator/AGENTS.md` §"Isolating a run" (`:511-600`), §"Visual
validation" (`:1552-1993`), §"Who may merge" (`:1344`), §"Releasing" (`:683-963`);
`Makefile:91-165` (`test`, `lint`, `format`, `check-changed`, `type-check`).

---

## 15. Convergence round 1 — what changed here

1. **`features.session_transfer` is reconciled into §9.3** beside `features.peers`.
   Route table and capability map are now one list rather than two: this document
   owns the transfer *route*, `mesh-ui.md` §2.6 owns the *key* and states why a
   renderer must not draw a control it has no adapter for.
2. **`mesh.json` is an explicit required addition to `fork`'s deny-list** (§1.2),
   with a named test. Without it a fork inherits its parent's `home_device` and
   `placement` — a live session advertising another device's ownership, which
   §1.1's INV-1 then routes to.
3. **Three new session ops are now rows in the transport's §6.4 table**
   (`net_session_create` → `prompt`, `net_session_engage` → `view`,
   `net_session_stop` → `stop`), and the transport states that they add no new
   capability names. Nothing in this document's command matrix changed; what
   changed is that the ops it depends on are declared where the vocabulary lives.
4. **`stream_open`'s pass-through mode is specified in the transport's §2.5** as an
   interface requirement, which is what makes R-IF-1 here implementable without a
   second per-op relay path: after `stream_open`, a client can subclass
   `AttachClient` and receive the session's frames unmodified.
5. **The viewer attach discipline is written down** (§3.7, abduco's policy):
   width owned by the most recent non-read-only client, read-only observers that
   never send input, and an exit status that survives a detach and a move. The
   mesh is where the width question stops being cosmetic.
6. **The sync primitive's content-addressing is stated as a rule** (§7), with the
   CRDT rejection recorded — the same rejection the transport records, so the two
   documents cannot drift on why the transcript is not a CRDT.

**One thing this document deliberately does not restate:** the transport's
bounded secret lifetime (`network.epoch_max_age_s`) and its withholding rule. Both
are the transport's, both are cited from §6.5's revoke row and §7.4, and neither
changes what a handoff does.
