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

## Next slice (queued — NOT in this PR; scope stays as above)

The same producer's other half, found by the UI lane (`2197cee0a558`) and
routed here as a follow-up (Aida, 2026-10-06):

**A wire-served row is stamped with the reader's clock.** `_remote_history`
serves wire rows as `{"ts": <serve time>, ...}` because the display-window
messages carry no entry time of their own (the stated limit in the method's
docstring). Joining a session mid-turn makes the ordering consequence visible:
the wire's user row carries a serve-stamp while the snapshot seed's tool start
carries its true `started_at_epoch`, so the transcript renders tool-above-user
— sticky while mounted, healing on remount, on base AND on the UI lane's own
fix — because the two stamps are not comparable facts, and no held-echo rule
can reach that. The design note for that slice (before its code) must settle:

1. **A row served without a true entry time SAYS SO on the wire** — never a
   silently substituted reader-clock stamp. An absent (or explicitly null)
   stamp the reader can see is strictly better than an invented one it cannot.
2. **The consumer contract is written down**: what a reader does with a served
   row that carries no true stamp — its ordering, its merge with true-stamp
   rows, its display — so downstream ordering rules can be written against a
   contract rather than a guess.

Known constraints from this slice's work: the window DTO is `extra="forbid"`,
so a new field cannot ride un-negotiated past older viewers' validation
(capability-key discipline, as every other additive wire field); and the
stored-page source this PR adds already carries TRUE entry timestamps — the
two sources will share whichever shape that slice settles.



## Complete peer-source contract (advertised; no consumer until the UI slice)

`peer-journal-source-v1` IS advertised by this build — on every relay link
(`network/wire.py`), on the runtime record (`session/runtime/serving.py` +
`runtime/server.py`) and as the `peer_journal_source` feature key
(`server/features.py`). ADVERTISING IS NOT ACTIVATION: every read still passes
the four-part gate — owner advertisement, viewer declaration, authenticated
session source, per-response acknowledgment — and absent any half the answer
stays the legacy bytes. A remote `journal_order` alone remains
byte-compatible unsupported, and local ordering and local mobile partial
negotiation are unchanged. The recovery envelope's source facts are
receiver-side implemented and inert until a UI slice consumes them. Every new
field is omitted for legacy viewers on live and replay paths.

The authority bootstrap uses the existing session GET but applies only source
metadata via `applyJournalOrderMetadata`: install the equality-only order domain
under session/view/read witnesses, never incidentally adopt page bodies, causal
call owners, tail/cursor coverage or frontend state. Explicit verified page/body
acceptance is a DIFFERENT typed path. Lookup coordinates do not reattest a loaded
old body or release a provisional/local user. Fresh body plus coordinates must
be admitted atomically; stale-domain loaded bodies remain payload-owed.
`JournalOrderRead` remains `{revision, demand}`; generation is separate. A raced
same-domain wake keeps its new reference-identity debt without blanket rejection
of still-valid old facts. Per-ID body-owed revisions fence body acceptance.

Proposed purposes on the existing view-authorized stored-history RPC:

| Purpose | Authority and bounds |
| --- | --- |
| page | Verified bounded page; explicit page acceptance only |
| metadata | Requested raw-ID facts only; no body/cursor/frontend effects |
| payload | Expected accepted generation REQUIRED; fresh projected bodies and same-read coordinates together |
| availability | One bounded deferred index refresh/join signal; no rows or generation authority |

Requested IDs plus explicit dependency IDs are bounded at 512. Missing means
proven absent, unavailable means present but unsafe/unprojectable, remaining
means budget-deferred; these are not interchangeable. Exact encoded response
limits include every metadata map. Raw-ID lookup demand unions canonical,
quarantined and loaded-body-owed IDs BEFORE the no-debt/no-generation guard.
No alias, clock, wire sequence or receiver disk supplies source authority.

Proof prerequisites the advertisement was gated on (held before this slice shipped):

* Extend the SAME index with validated raw-ID byte ranges, retaining no bodies;
  FD-before/after, path identity and index source must agree for bounded reads.
* Prove actual settled projection equivalence including prune/injection/media
  dependencies. If closure cannot be bounded, report unavailable, not missing;
  a raw row converter is not a universal settled replay projector.
* Availability must defer at the dispatcher, not return None from an inline
  handler (which yields an empty ACK), and must not occupy the shared four slow
  workers or stall reader/prompt/live traffic. Recheck link/grant/session before
  one final answer. Cancellation cleanup must state actual protocol bounds.

The index witness inherits the existing inode/size/mtime assumption, not
cryptographic detection of externally preserved-stat rewrites. No new cache,
service, domain, payload queue, runtime lease, permission or view-grant expansion
is authorized. Complete-core review/QA precedes separate UI adoption. The UI
continuity amendment must separate metadata facts from rank/body/causal-owner
admission and retain later source demand while admitting valid earlier facts.


### Bounded structural source-body subset (unpublished implementation)

The same transcript-index cache schema now proposes version 7: raw ranges and
body-free structural facts share the existing full/suffix scanner. Facts are
extracted from already parsed JSON before hidden-row and assistant-call
classification loses identity. Inverse prune/call/compaction/duplicate views are
built once with the resident index or cache load, not by a payload request.
No notices, arguments, preserved text or message bodies enter the new table.
Suffix facts follow the existing window-row cut; last-ID ranges alone never
certify duplicate identities. The reader's segment row count is not a whole
journal count after incremental re-derivation.

The separate internal source-body getter discovers a complete ordinary
non-ask call unit and its last effective prune through the captured cut, reads
only those ranges on one source-coherent FD, and reuses the actual per-row
replay operation. It does not build an incomplete selected Transcript or infer
context/preserve graphs. Preserved candidates, ambiguous/ask/divert/hidden or
cross-cut units, opaque causal scope, unknown global scope, supplements and
unbounded legacy media expansion are terminal unavailable. Inline media and
artifact references retain their real typed body. A wholly historical ordinary
unit may have an audit-equivalent body; units crossing the governing cut cannot
be partially certified. A raw prune is named as a dependency, never leaked as
an unsolicited conversation row.

Each original raw ID has exactly one of present/missing/unavailable/remaining.
Missing requires proven source absence; unproven identity scope is envelope
unavailable. Intrinsic units over 512 unique IDs, 512 KiB raw bytes or an empty
encoded frame are unavailable. Aggregate budgets defer complete units as
remaining, with a smaller requested-ID retry batch that leaves room for explicit
dependencies. Huge units must not starve later small units. Encoded metadata is
never silently trimmed, and expanded media is never dropped to manufacture fit.
All body reads inherit the existing inode/size/mtime limitation, including
stat-preserving external rewrites. Publication still requires current view,
client, expected-domain and body-owed revision fences after awaits.

This implements no public DTO, complete capability, availability lane or UI
fallback/admission policy. Initial unprovable tails must decline optimization
before new-domain adoption. Later unavailable units retain known/provisional
records and require explicit source-unconfirmed reconcile; unavailable is not
missing, reset, or a reason to poll an old peer. Metadata bootstrap remains
lookup facts/domain only, never rank/causal-owner admission on an old body.
