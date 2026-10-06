# Mesh cold reads: a peer's stored transcript is served, not hidden

**Slice: D5-core** (`remote-offload-firstrun-defects`). Lane: session
`60ba22ae6cf2` ("Core: cold peer session reads serve an empty page and never
engage"), routed from the remote-transcript lane `2197cee0a558`. This note is
the design/decision artifact — written before any code, so the reasoning can be
attacked first.

## The defect, verified against `origin/main` (`041190aaa`)

The operator, watching a remote session: *"at times I can't see the full
conversation — it says it's the end of the conversation when it's not."* The
core half, traced in the code:

1. **A cold read of a peer session serves an empty page.** For an id another
   device holds, the desktop bridge routes its history page through
   `DesktopSessionBridge._remote_history`
   (`local_operator/server/utils/desktop_sessions.py`), which reads "off the
   wire" via `AttachedSession.history()`. When the facade is not hydrated — no
   runtime is attached on the peer — `_remote_rows` swallows the `RuntimeError`
   and the method answers
   `{"entries": [], "has_more": false, "cursor_missing": false}`: the same
   envelope a conversation with no rows produces. The snapshot beside it merges
   the cold triple from `_cold_fields`, which for this state reads
   `cold: true, cold_reason: "no-runtime"` (`AttachedSession.cold_reason`'s
   documented default token).
2. **The open path never engages the peer — and must not.** The read makes ONE
   bounded attach attempt (`READ_ATTACH_BUDGET_S = 2.0` →
   `attach_existing`), and an attach dials an owner that already exists; a peer
   with no runtime keeps none. `_remote_history`'s own docstring states the
   policy by name: "the desktop's own watch lease is what warms one". The only
   automatic warm is a live VISIBLE watch lease: `POST /watch` sets
   `sub.visible` and arms `_lease_warm_loop`; a bare SSE subscription is built
   with `DesktopSubscription.visible = False` and arms nothing (by design).
3. **Rows exist on the owning device while the reader shows nothing — and for
   one class of session the reader can never show anything.** A deliberately
   STOPPED session is skipped by the visible-lease warm ("a stopped session
   stays stopped until a user action re-opens it", `refresh_watch`) and the
   owner's own engage refuses to resurrect one. Its transcript sits on the
   peer's disk; today no read path reaches it.

The asymmetry is the defect stated plainly: **a cold LOCAL session's read
serves its journal from disk** (`history()` → `load_transcript_page` on
`<root>/sessions/<id>`), while **a cold PEER session's read serves nothing**.
Same route, same contract, one population served and the other blank.

### Characterisation asked with the decision

- **While a peer is genuinely unreachable** (today's relay outage): a session
  whose row is marked unreachable is refused before anything is built —
  `PeerSessionUnreachable` → 409 `session_is_remote` carrying
  `remote_open.unreachable_peer_sentence`, the same sentence the TUI uses.
  A row still cached as reachable when the peer dies mid-window reaches the
  cold empty page above instead, and is indistinguishable from "no messages
  yet".
- **Wire-level distinguishability**: the `/history` envelope
  (`entries` / `has_more` / `cursor_missing`) carries nothing that separates
  "empty because cold" from "empty because empty"; only the snapshot's cold
  triple says *cold*, and it says it for both. After this change the
  distinction becomes real: a stored page that is SERVED and empty means the
  owner has no rows; a page that cannot be served at all says so (below).

## The decision

**A cold read of a peer session serves the stored transcript from the owning
device — a bounded page read over the existing transport — and reading still
starts nothing anywhere.**

Why this, and not merely "make the empty page honest":

1. The data exists and is the user's. A better word on an empty screen still
   does not show the conversation.
2. Stopped sessions can never be warmed (both halves refuse by design), so for
   them no amount of waiting produces rows. Only a stored read can.
3. It restores local/peer symmetry (above) rather than inventing a second
   behaviour for one population.
4. It is cheaper than every alternative: a page read on the owner is bounded —
   ~1.7 ms per page on the operator's 261 MB journal, as measured for the local
   reader this reuses — against a warm's spawn (an ~82 MB idle runtime on the
   peer, plus the spawn itself).
5. It makes "empty" mean something: the boundary then comes from the owner's
   own journal (`has_more` over real rows, real timestamps), which is D5's
   second judgement criterion — "a boundary derived from the peer's actual
   [quantity]" — fixed at the source.

It adds **no new authority**: a member that may open a session (`view`
capability) can already read the transcript through the runtime whenever one is
warm; this serves the same bytes from the same device under the same
capability, no runtime required.

## The mechanism

**A new relay-routed op `net_session_history`** (working name; capability
`view` — the transport's own rule for an opening read: "the act of opening is a
read"; session-scoped like every `net_session_*` op):

- The OWNER's relay answers it from the owner's own store, through the same
  semantics the local reader uses: a bounded page (`before_id`, `limit`
  1..500), `{entries, has_more, cursor_missing}`, entries in the journal's
  serialized row shape (`id` / `ts` / `type` / `payload`) with the same
  server-side visibility filter the local `/history` applies
  (`visible_transcript_rows`). Real timestamps — the serve-time `ts`
  compromise the wire path documents does not apply to this source.
- **Ownership is resolved the way every owner-side session op resolves it**
  (mirror `_engage_locally`'s `(self.root / "sessions" / session_id).is_dir()`
  check plus the transport's session-scope rule), so "this device does not own
  that id" is a refusal while "owned, no rows yet" is an empty page — the two
  are distinguishable at the wire.
- The READER (`_remote_history`) falls back to it exactly when the wire window
  cannot answer (cold / not hydrated). The snapshot serves the same method, so
  it inherits. The live path is unchanged.
- **Merge safety, stated because it is the race this change must not
  create:** wire rows and journal rows share one id space — "the message `id`
  is already the entry id" (`transcript.encode_message_payload`) — so rows
  served from the stored page and rows arriving later on a warm's window
  dedupe by id under the renderer's existing merge. A test must pin this
  across a cold-read-then-warm sequence.
- **When the stored page cannot be served** (relay or peer unreachable at read
  time), the answer must not read as "no messages". Recommended: mark the
  empty answer `cursor_missing: true` — the contract's existing word for "this
  page cannot be trusted as complete", which the UI half's rule ("never claim
  exhaustion over a page whose hydration is unproven") consumes. The
  alternative is the open path's own refusal shape (`session_is_remote` + the
  sentence); design review picks one, and the choice is recorded here.
- **Attachments degrade, named rather than discovered:** journal rows
  reference attachment digests that live in the OWNER's store, and a cold page
  will not inline bytes the way the live wire does. v1 renders the existing
  missing-media placeholder for such rows; a follow-up can serve attachment
  bytes over the relay if the operator misses images in old remote history.
  Inlining per page is rejected here: it re-introduces a size cap the page
  contract does not have today.

### Rejected alternatives

- **Engage a runtime on open** (warm-on-read): spawns work for a glance
  ("opening a terminal is not work"), changes live-continuation semantics, and
  cannot help stopped sessions at all.
- **Replicate the transcript to the reader** (a local mirror / `net_sync`):
  writes state on every reading device, needs its own freshness and cleanup
  story, and slides toward the one local read §3.4 forbids. A possible later
  optimisation if page latency ever matters; not v1.
- **Read the viewer's local disk**: absent, or a different conversation wearing
  the same id (`mesh-session-mobility.md` §3.4).
- **Make empty honest, only**: does not show the conversation. Kept as the
  failure-state discipline, not the fix.

### Documentation the change must carry

- `docs/design/mesh-session-mobility.md` §3.4: amend "history comes from the
  wire, and only the wire" — the prohibition's intent is the VIEWER's local
  disk; the owner's store served by the owner's relay is the wire's own source
  and becomes the cold fallback. §2.2's op table gains the new name.
- `docs/DESKTOP_API.md`: the cold-read contract — an empty history page on a
  peer now means "the owner has no rows" when it was servable, and how an
  unservable page is marked.
- `NET_OPS` / `OP_CAPABILITY` (`network/types.py`) + the op-capability
  totality test.

## Scope

- In: the core read (op + relay handler + reader fallback + tests + docs).
- Out (other lane): live-row ordering and the unproven-hydration copy (UI lane
  `2197cee0a558`, `ui-remote-turn-order`); TUI parity is a follow-up unless the
  seam makes it a one-site change.

## Evidence plan

- Unit: op served from the owner's store (pagination, empty-vs-unowned,
  capability row, visibility filter); `_remote_history` cold fallback under a
  fake peer (asserts the peer saw NO engage); the failure signal; the
  cold-read-then-warm id merge.
- QA: the two-daemon fixture (`tests/unit/network/test_session_plane.py`
  patterns) driven end to end — peer stopped, peer cold, peer live, peer
  unreachable — with the wire audit showing no spawn on a read.
- Docs + a real-device re-check when the pairing is up again (the lane's
  post-outage instrumented run slot).
