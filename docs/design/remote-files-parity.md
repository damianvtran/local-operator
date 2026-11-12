# Remote files parity — design + slice plan

Base: core `origin/main` @ `89678ffc7`, UI `origin/main` @ `83a18c632`. Every `file:line`
below was read on those commits (core paths are relative to the core repo; UI paths start
with `ui:` and are relative to local-operator-ui). Author: architect (lopdev), static design
plus one loopback spike (§0). It builds on `docs/design/mesh-file-transfer.md` (S1 shipped
in v0.102.0) and replaces that note's S4 paragraph.

## Operator request (verbatim)

> "I noticed I can't see images and artifacts on remote nodes, can you have a session work on
> this remotely and test it, make sure that it works. We should be able to preview assets
> (sounds, music, images, video, etc.) from the local device and also see those show up in
> the canvas and ideally the library as well, these should be indexed too in order to be
> performant and not slow down the library view. It might make sense to cache or precache
> assets on active sessions so that clicking into them is fast and we can see the previews,
> but make sure it doesn't blow up RAM."
>
> "Right now it just says remote files can't be opened on this device but we need to fix that
> and make it so that remote files show up just like local files maybe just with a small
> non-intrusive indication somewhere that they're remote but they should show up under the
> same classified sticky headers, sections, etc. and we should be able to download them to
> local, and do other file actions."

Non-goals (from the manager's contract): no new auth model, no widening beyond S1's
confinement, no push (S2), no agent-tool pull (S3), no mobile daemon change, no TUI viewer
fetch-on-miss.

## 0. What the code does today

**Transcript bytes.**
- `DesktopSessions.attachment()` (`local_operator/server/utils/desktop_sessions.py:8296-8371`)
  reads only `<root>/attachments`. For a peer's session it raises `PeerAttachmentUnavailable`
  on a miss (`:8360-8362`).
- The waveform route does the same (`:8405-8411`), and so does the supplement-document route
  (`:8458-8464`).
- All three map to 409 `attachment_on_peer` (`server/routes/desktop_sessions.py:2229-2236`).
- The CHILD attachment route never reaches a peer branch: `_contained_child_dir` needs the
  parent directory locally, so a remote subagent's image is a 404
  (`desktop_sessions.py:8611-8632`).
- The success path buffers the whole file several times over. `AttachmentStore.get` reads the
  file and base64-encodes it (`session/attachments.py:426-457`). The route decodes it again
  (`desktop_sessions.py:8363-8364`) and returns `Response(content=data)`
  (`routes/desktop_sessions.py:4144`). There is no `Range` support. The Library content route
  already streams and answers `Range` with a 206 (`server/routes/desktop_library.py:358-468`).
- G1 (the replica store is never read) and G2 (nothing fetches from the owner) are unchanged
  since `mesh-file-transfer.md` §6. Replica blobs live under `replica_dir`
  (`network/sync.py:675`, `blobs = destination` at `:3579`).

**Files panel.**
- `GET …/files` answers a peer with `state: "unsupported"` and no rows
  (`server/routes/desktop_session_files.py:224`). `…/user-attachments` does the same at
  `:327`, and `…/materialise` answers 404 at `:279-284`.
- The UI shows that state as "Remote files cannot be opened on this device."
  (`ui:src/renderer/src/features/chat/canonical/use-session-files.ts:121-128,170-178`).

**S1 byte path.**
- `net_file_get` (view role) serves two things: a `scratchpad://` path, confined by the pad's
  own parser plus the credential denylist, or an attachment digest that the session's
  transcript references (`network/files.py:180-273`).
- The local op `session_file_pull` stages the file, verifies it and adopts it into
  `<root>/attachments` (`files.py:767-891`, `adopt_file` at `session/attachments.py:273-325`).
- An old peer is recognised by `wire.MESH_FILE_V1` (`files.py:800-808`, `wire.py:239`).
- The desktop daemon reaches this device's relay through
  `server/utils/desktop_mesh.py:_call` (`:132-153`, `control_request` at
  `network/relay.py:18148`).

**Spike (loopback, `_pair_settled`, both relays in one process, this host, n=3 per arm):**
- A 32 MiB attachment pulls in **2.26 / 2.30 / 2.38 s** when the transcript is tiny.
- With a 64 MiB transcript it takes **5.78 / 5.85 / 6.00 s**. Peak RSS rises 196 → 262 MiB.
- The cause is that `_resolve_source` runs on EVERY phase (`files.py:185-188`) and calls
  `sync.referenced_attachments_in`, which `read_bytes()` the whole transcript
  (`sync.py:1106-1126`; measured at 58 ms per phase at 64 MiB). A 32 MiB pull has 65 phases.
- At the 261 MB journal that `relay.py:4052` cites, that is about 240 ms and 261 MB of
  transient allocation per 512 KiB chunk on the OWNER.
- Precache would multiply this, so the fix is commit C1 below.
- Script: `scratchpad://rfp-spike/test_rfp_spike.py`, the architect session's own pad.

**Measured distributions (this lopdev host, 2026-11-12; the operator's Mac is unmeasured — see
§8):**

| Population | n | median | p90 | max | total |
|---|---|---|---|---|---|
| `~/.local-operator/attachments/*.bin` (all image) | 317 | 99.6 KiB | 187.5 KiB | 0.40 MiB | 31.7 MiB |
| pad images | 12 118 | 42.0 KiB | 59.5 KiB | 1.06 MiB | 536 MiB |
| pad docs (pdf/html/md/csv/json/txt/office) | 20 980 | 2.5 KiB | 14 KiB | 3.8 MiB | 264 MiB |
| whole pads | 182 | 40.9 KiB | — | 269 MiB | 2.1 GiB |
| distinct referenced digests per session (sessions with any) | 48 | 6 | — | 28 | — |
| transcripts | 239 | 0.27 MiB | — | 21.2 MiB | — |
| sessions touched ≤1 h / ≤2 h / ≤6 h / ≤24 h | 239 | 8 / 11 / 27 / 117 | | | |

**UI constraints that bound the design:**
- Every byte op goes through `desktopMedia`, which buffers the whole body over IPC and caps it
  at `MAX_DOWNLOAD_BYTES` = 32 MiB (`ui:src/main/desktop-media.ts:63,1249-1262`). Its timeout
  is 120 s (`:980-981`).
- On `/v1/desktop/`, an undeclared 503 is read as the PLANE being closed
  (`ui:src/shared/desktop-contract.ts:6148-6154`). A retryable "owner unreachable" answer must
  therefore NOT be a 503.
- A refusal's `code` reaches the renderer only if it is a plane code (`desktop-media.ts:1186-1196`).
  `status` always reaches it. So retryable and permanent answers have to differ by **status**.

## 1. Transcript bytes: fetch-on-miss (S4)

**Change sites:**
- `desktop_sessions.py:8350-8364` (`attachment`), `:8399-8411` (`attachment_waveform`),
  `:8452-8466` (`supplement_document`) and `:8611-8632` (`child_attachment`) gain one shared
  peer branch, `_peer_bytes(session_id, digest, *, member_id)`.
- `member_id` is the child id for the child routes; the owner's `_session_scope` admits any
  session it owns, children included (`network/relay.py:4632-4670`, `authorizer.py:322-359`).
- The child route's peer branch runs only when `files_target`/`remote_row_for` says the
  PARENT is a peer's (`desktop_sessions.py:8945-8980`). The local containment proof is
  unchanged for local parents.

**Lookup order** (the first hit wins; every step is a stat, never a read):
1. `<root>/attachments/<d>.bin` — today's behaviour; a moved session still serves.
2. The replica store `replica_dir(root, sid)/attachments/<d>.bin` (closes G1).
3. The remote cache `<root>/cache/remote-files/<d>.bin` (§4).
4. A pull from the owner: `session_file_pull {peer, session_id: member_id, attachment: d,
   land: "cache"}` over `desktop_mesh._call`. The bytes then land in step 3's directory.

**Serving.**
- The peer branch returns a PATH, not bytes. The routes then serve with
  `desktop_library._stream`/`_parse_range` (`desktop_library.py:358-397`), lifted into
  `server/utils/byte_range.py` so that both routes import ONE implementation.
- The response carries 200 with `Accept-Ranges: bytes`, or 206 for a `Range` request.
- The local branch keeps `AttachmentStore.get` (a non-goal; unchanged bytes for local
  sessions).
- The file is OPENED before the route answers, and the stream reads from that fd. Eviction may
  then unlink it concurrently (POSIX keeps the inode alive), so serving needs no pin.

**Single-flight and timeout.**
- In-process: `dict[(owner_device, member_id, d)] -> asyncio.Task` in the new
  `local_operator/remote_files/fetch.py`.
- Across processes: the relay's existing `_part_lock(root, sha)` (`files.py:506-512`).
- Each route waits `REMOTE_FETCH_WAIT_S = 20.0` under `asyncio.shield`. The basis is the
  loopback max for a 32 MiB file (6.0 s, §0) × 3, which stays under the UI's 120 s media
  timeout. When the wait expires, the task KEEPS RUNNING, bounded by
  `FILE_PULL_CLIENT_TIMEOUT_S` (`files.py:83`).
- Concurrency: `asyncio.Semaphore(2)` per owner device. This mirrors S1's "≤2 inbound
  transfers per link" (`mesh-file-transfer.md` §1).

**Statuses for the peer branch** (new i18n keys in `i18n/keys/wire_errors_desktop`):

| Condition (relay refusal code) | Status | code | UI behaviour |
|---|---|---|---|
| bytes served | 200/206 | — | paint |
| `peer_unreachable`, `relay_unavailable`, a link that cannot be dialled | **504** | `attachment_owner_unreachable` | retryable |
| still transferring at the wait bound | **504** + `Retry-After: 2` | `attachment_fetch_pending` | retryable |
| `predates_file_transfer` | 409 | `attachment_peer_outdated` (sentence names `lop network update <device>`) | permanent; no fallback |
| any other owner refusal (`not_found`, `not_authorised`, not referenced, `corrupt_source`, `integrity_failed`) | 409 | `attachment_on_peer` (today's code, owner sentence) | permanent |

**UI change (the "UI must not cache the 409 forever" check):**
- Failures are NOT cached module-wide: only a success is `publish`ed
  (`ui:…/canonical/use-attachment-url.ts:137-148`), and `inflight` is cleared in `finally`
  (`:153-155`).
- The per-mount `failed` flag, however, sticks until the row remounts
  (`:298-301,343-349,367-371`), and nothing ever retries.
- Fix: make `fetchAttachment` (`:107-159`) return `{url} | {status}`. On **504**,
  `useAttachmentSource` re-arms the effect with backoff (2, 4, 8, 16 s; at most 5 attempts),
  then shows a `retryable` state with a click-to-retry. A 409 or 404 stays `missing`.
- Apply the same change to `fetchAttachmentBlob` (`:180-206`, a 504 gets its own toast) and to
  `use-waveform.ts` (op at `:68`).

**Invariants and checks:**
- **I1-1:** no byte is served from a peer-owned session unless the owner served it under S1
  confinement.
  Check: the owner revokes view → 409 `attachment_on_peer`, with nothing in the cache or
  staging.
  Check: an unreferenced digest → 409, and the cache is empty.
- **I1-2:** retryable is distinguishable from permanent by status alone.
  Check: unit tests of the route mapping, plus the UI test "504 retries, 409 does not".
- **I1-3:** RAM stays flat while serving.
  Check: a 64 MiB cached blob served through the route with `tracemalloc` peak < 4 MiB.
  RED on base: base buffers ≥ 3×.

## 2. Remote Files panel listing

**Options:**
- **(A) A new view-role peer op that proxies the owner's index.** The owner runs the SAME
  `session_files.service.list_files` (`session_files/service.py:760-830`) over its own
  journal. Cursor and revision are owner-minted (`encode_cursor`, `:616`) and pass through
  opaquely, so pagination, `building` polling and `cursor_reset` keep the local semantics.
  `exists` and the stats are the owner's truth.
- **(B) Derive the index on the viewer from a replica or history.** This needs the whole
  journal, family included, on the viewer (paged at 500 rows per `net_session_history`,
  `relay.py:4007-4009`), a second scan of it, and stats that cannot be known (the paths are on
  the owner's disk). Cost is O(journal) per viewer, and the answer is wrong for `exists`, size
  and mtime.

**Decision: (A).**

**Wire.**
- New peer op `net_session_files` in `NET_OPS` (`network/types.py:346-414`), with
  `OP_CAPABILITY` `"view"` (beside `net_file_get`, `:668`).
- New local op `peer_session_files` in `LOCAL_OPS` (`types.py:443-537`).
- Both are registered by `network/files.py:install` (`:909-915`), so `SLICE_MODULES` is
  untouched.
- New link capability `mesh-file-index-v1` in `wire.LINK_CAPABILITIES` (`wire.py:149-191`).
- Phases:
  - `list {session_id, index: "files"|"user_attachments", query…}` returns the listing dict
    plus per-row remote fields (below).
  - `row {session_id, file_id}` returns one row, or a `not_found` refusal.
- The frame names `session_id` at top level, so `_session_scope` decides ownership at the
  chokepoint.
- The handler refuses a nested `frame`, exactly as `files.py:471-484` does.

**Owner execution.**
- Family discovery currently lives in the FastAPI module (`desktop_session_files.py:96-138`
  uses `contained_descendants`, `desktop_sessions.py:7221-7280`). Importing
  `server.utils.desktop_sessions` into the relay was measured at **+27 MiB RSS, +0.8 s**
  (56 → 83 MiB) and would also pull in `fastapi`.
- So **move** `_persisted_children`, `_contained_child_dir`, `_prove_child`,
  `contained_descendants`, `FAMILY_MAX_*` and `SubagentChildUnavailable`
  (`desktop_sessions.py:7027-7280`) into a new `session_files/family.py`. `desktop_sessions`
  re-imports them under the same names.
- Lift the route's `_family` (`desktop_session_files.py:96-138`) into the same module as
  `family_for(config_dir, session_id, directory, scope, budget_s)`.
- The relay handler runs `list_files(…, display=True, budget_s=DEFAULT_BUDGET_S)`.
  `display=True` is the projection that the module docstring says mesh surfaces must use
  (`desktop_session_files.py:28-30`): `~/` or cwd-relative paths, never the owner's absolute
  layout.
- `building` is finished by a thread single-flight (`service.complete_in_background`,
  `service.py:399`) keyed like `_IN_FLIGHT`.
- Ledger residency is bounded at 24 snapshots per process (`ledger.py:81,186-193`). Writes
  are atomic across processes (`ledger.py:36-38`).
- Reply size: measure the encoded detail. If it exceeds
  `relay.SESSION_HISTORY_REPLY_BUDGET_BYTES` (`relay.py:4028`), drop tail rows and re-mint
  `next_cursor` at the trimmed offset. An oversize record kills the whole link
  (`relay.py:4011-4024`).

**Row shape = local row** (`server/models/desktop_session_files.py`, `extra="allow"`) **plus**:
- `remote: true` and `origin_device: {id, name}`. These are added by the VIEWER from the
  catalogue row (`SessionRow.owner_device`/`owner_device_name`, `resume.py:2728-2733`), never
  trusted from the owner.
- `remote_read: {kind: "attachment"|"path"} | null` (owner-computed).
- `remote_unreadable: <reason> | null` (owner-computed).
- The listing gains `unsupported_reason: "peer_update_required" | "capability_absent" | null`,
  with `device_name`.

The classification fields are unchanged (`kind`, `origin`, `session.role`, `counts`, sorts),
so rows group under the same sticky headers and sections, with the same filters.

**Per-row readability under S1 confinement** (computed by the owner for PAGE ROWS only, and
re-checked by `net_file_get` on every phase, so the listing is advisory and never authority):

| Row kind | Remotely readable? | Rule the owner applies | `remote_unreadable` |
|---|---|---|---|
| artifact (`ref.type == "artifact"`: generated image/video/audio, screenshots) | yes | digest ∈ `referenced_attachments_in(member_dir)` (memoised, C1) | `not_referenced` (compaction dropped it) |
| user attachment (`…/user-attachments` index) | yes | same digest rule on the parent | `not_referenced` |
| path inside the member's scratchpad | yes | `realpath` inside `scratchpad_root(member_dir)` and not `is_sensitive_for_supplements` → `remote_read.kind="path"` | `sensitive` |
| path in the session cwd/workspace/home that the transcript mentions or wrote | **no** | outside S1 (`files.py:98` comment, `mesh-file-transfer.md` §3) | `outside_session_files` |
| path that no longer exists | no | `exists: false` | `missing` |
| file over `FILE_MAX_BYTES` (256 MiB, `files.py:73`) | no | size from the stat | `too_large` |

**Authority.** No row becomes readable that `net_sync` (view) does not already hand a view
member: pad tree plus referenced blobs (`mesh-file-transfer.md` §2). Workspace files stay
unreadable.

**Operator decision (optional, not taken):** making workspace files readable would let a
view member read arbitrary files in the owner's project directory. That is a new authority,
because `net_sync` does not ship the cwd. The non-widened default gives `outside_session_files`
rows a disabled open with the explanation "Lives in <device>'s workspace — ask the session to
copy it to its scratchpad", using the existing `send(peer=…)` path.

**Viewer route change.**
- `desktop_session_files.py:223-224`: for `kind_of_target == "peer"` with a capable link,
  call `peer_session_files` and attach `remote`/`origin_device`.
- For a link without `mesh-file-index-v1`, answer `state: "unsupported"`,
  `unsupported_reason: "peer_update_required"`.
- Apply the same change to `…/user-attachments` (`:326-327`).

**Invariants and checks:**
- **I2-1:** a peer page and a local page of the same journal are equal modulo the
  remote-only fields and the display form of paths.
  Check: list through the loopback, then list locally on the owner with `display=True`,
  strip the remote fields, and assert equality.
- **I2-2:** cursor passthrough. Check: page 2 via the cursor, then an owner-side append; the
  next page answers `cursor_reset: true`.
- **I2-3:** for every row with `remote_read`, `net_file_get` succeeds, and for every row with
  `remote_unreadable`, it refuses.

## 3. Remote read and download

**New viewer routes** (peer sessions only; a local session answers 404
`session_files_file_not_found`, because local rows already have a path or `materialise`):
- `GET /v1/desktop/sessions/{sid}/files/{file_id}/content`, with `Range`. The viewer asks the
  owner `net_session_files {phase: "row", file_id}`, then fetches through §1's machinery,
  keyed by the stat sha256, and streams. Statuses follow §1's table. A row with
  `remote_unreadable` answers 409 `session_files_remote_unreadable`, with the reason in
  `params.reason`.
- `POST /v1/desktop/sessions/{sid}/files/{file_id}/download` follows the same fetch path, then
  lands the file on THIS device:
  - **Where:** `~/Downloads` when it is a directory, else `<config>/downloads/`. No new
    config key.
  - **Name:** `service.sanitise_name` (`session_files/service.py:897`). A name that
    `_bare_name_denied` refuses (`:935`, e.g. `.env`, `id_rsa`) gets 409
    `session_files_sensitive_name`.
  - **Collision:** if an existing file at the name has the same sha256, reuse it
    (`created: false`, mirroring `materialise_bytes`, `:1035`). Otherwise use `name (1).ext`
    through `name (99).ext`, then refuse.
  - **Write:** an `O_EXCL` temp in the target directory, a streamed copy from the open
    cache fd (1 MiB blocks), `fsync`, a full sha256 compared against the owner's stat sha256,
    then `os.link(temp, final)` (a no-replace publish) and an unlink of the temp. Nothing is
    visible on a mismatch.
  - **Receipt:** `{path, name, bytes, sha256, created, origin_device}`.
  - Idempotent. While the fetch is pending it answers 504 `…fetch_pending`, and a re-POST
    joins the single-flight.

**UI file actions, enumerated, with their remote disposition.** The menus are
`ui:src/renderer/src/shared/components/common/file-actions-menu.tsx:113-140` for path rows
and `ui:…/features/chat/utils/image-actions.ts:224-251` for artifact rows, wired at
`ui:…/components/canvas/file-row.tsx:495-547`.

| Action | Local today | Remote row (readable) | Remote row (unreadable) |
|---|---|---|---|
| Preview / Show in canvas | path bridge or digest | digest → `sessions.attachment` (fetch-on-miss); pad path → new media op `sessions.files.content` | disabled + reason tooltip |
| Open file (`window.api.openFile`) | yes | after **Download to this device**, on the landed path | disabled + reason |
| Open folder (`showItemInFolder`) | yes | after download, on the landed path | disabled |
| Copy file path | yes | before download: "Copy path on <device>" (owner display path, text only); after: landed path | "Copy path on <device>" |
| Save as… (≤32 MiB bytes via `desktopMedia`) | artifacts | yes (same bytes route); over 32 MiB → hidden, Download offered | hidden |
| Copy image | images | yes | hidden |
| **Download to this device** (new) | — (local has `materialise`) | yes → receipt; row then gains the local path actions | hidden |

**UI invariant I3-1:** a remote row's `ref.path` is NEVER passed to `window.api.openFile`,
`showItemInFolder` or the static route. It is the owner's display path and may name an
unrelated file on this disk.
Check: a renderer unit test over the action table with `remote: true`.

**Core invariant I3-2:** download never overwrites, and lands only verified bytes.
Check: a pre-existing different file → the download lands as ` (1)`. A tampered cache blob →
refused, nothing landed.

## 4. Cache and precache

**Decision: a separate bounded cache, `<config>/cache/remote-files/`**, laid out in the
AttachmentStore format (`<d>.bin` + `<d>.json`, plus `<d>.peaks.json` written by
`waveform_payload`). It is not the shared `<config>/attachments`:
- That store is unbounded and AUTHORITATIVE. Local transcripts resolve their images from it
  (`attachments.py:174-186`), so any eviction there could delete a local session's only copy.
- Everything in the remote cache can be re-fetched by construction.
- To land there, `pull_file` gains `land: "store"|"cache"` (default `"store"` keeps S1's CLI
  receipt). The frame field `land` is passed from `local_pull_handler` (`files.py:894-906`).
  `AttachmentStore(cache_root).adopt_file` works unchanged: staging and cache share the config
  filesystem.
- **Daemon/relay generation skew:** an older relay ignores `land` and adopts into
  `<config>/attachments`. Lookup step 1 still finds the bytes, so the skew degrades to "not
  evictable", not to a failure.

**Budget.**
- Config key `desktop.remote_files_cache_bytes`: `path=("desktop", "remote_files_cache_bytes")`,
  `Kind.INT`, section `desktop` (Scope.LIVE, `settings_io.py:920-925`). It is read per eviction
  pass.
- Default constant `REMOTE_FILES_CACHE_BYTES = 1 GiB` in `remote_files/cache.py`, mapped in
  `_consumer_defaults()` (`tests/unit/test_settings_io.py`), per AGENTS.md "Adding a
  configuration key".
- Basis: a typical active set of 16 sessions × median 6 referenced digests × p90 187.5 KiB is
  about 18 MiB, so 1 GiB is about 55× that. It is also 3.8× the largest measured pad (269 MiB)
  and 32 files at the per-file preview ceiling.
- `0` turns precache off and evicts everything after serving, so the cache acts as staging
  only.

**Eviction.**
- LRU on `.bin` mtime, bumped with `os.utime` on every serve hit. atime is unreliable under
  `relatime`/`noatime`.
- A pass runs after each landing and at daemon start, under `flock(<cache>/.lock)`. It deletes
  the oldest `.bin`/`.json`/`.peaks.json` triples until total ≤ budget.
- In-flight pins are unnecessary: staging parts live in `network/staging/files`
  (`files.py:90,515`), and serving reads from an fd it already opened (§1).
- **Invariant I4-1:** after any landing, total cache bytes ≤ budget + the size of the one file
  just landed.
  Check: budget = 3 files, land 5, assert the two oldest are gone, the newest is present, and
  the total is ≤ budget.

**Precache.**
- **Active** remote session: a catalogue row (`remote_session_rows`,
  `desktop_mesh.py:530`) that is `reachable`, not `archived`, with `mtime` within the last
  **2 h**. That is 11 of 239 sessions on this host (§0). Add the last **8** remote sessions this
  daemon opened (an LRU, cleared on restart). At most **16** sessions per pass, which exceeds
  the observed 11.
- **Trigger:** the desktop catalogue poll (`peer_catalogue`/`remote_session_rows`), at most
  once per 60 s per session, and only when the row's `mtime` moved. Opening a remote session
  triggers it immediately.
- **What:** each active session's `…/user-attachments` digests, plus files-index rows of kind
  image, audio or video with `remote_read != null`. One `list` request each, newest first.
- **Per-file ceiling:** 32 MiB, equal to the UI's `MAX_DOWNLOAD_BYTES`
  (`desktop-media.ts:63`). Anything larger cannot be previewed through the media relay, so
  precaching it buys nothing. Such files are fetched only on explicit Download.
- **Concurrency:** the same per-owner `Semaphore(2)` as fetch-on-miss, so precache can never
  starve an on-screen request. On-screen requests take priority because precache runs at most
  1 in flight per owner (a nested `Semaphore(1)`).
- **Stops** at budget: a pass never evicts what it landed in the same pass.

**RAM: paths that must stream, with the check for each.**
- Owner `_resolve_source` → the referenced-digest set must be **memoised** per
  `(transcript path, size, mtime_ns, inode)`, in an LRU of 8. The scan itself streams
  4 MiB windows with a 64-byte overlap instead of `read_bytes()` (`sync.py:1120`). This is C1.
  Check: the spike at 64 MiB returns to ≤ 2.4 s per 32 MiB pull, and the scan count is 1 per
  pull.
- Requester `_transfer` (`files.py:734`) already writes 512 KiB chunks to the part file.
  Unchanged.
- `adopt_file` hashes in 1 MiB blocks (`attachments.py:296-301`). Unchanged.
- Route responses stream from the fd (§1). Download copies in 1 MiB blocks (§3).
- Nothing on the precache path ever holds the bytes of a file in memory. Check: a precache of
  10 × 16 MiB files with RSS delta < 64 MiB in the daemon process.
- **Renderer, a non-goal but named:** `desktopMedia` still buffers ≤32 MiB per open
  (`desktop-media.ts:1249`) — the same as local today.

## 5. Canvas and Library

**Canvas.** `ui:…/components/canvas/index.tsx:470-480` mounts `useSessionFiles` and
`useUserAttachments`. Once §2 answers rows for peers, the Canvas Files and Attachments
sections fill exactly as for local sessions. Text and document previews resolve by digest
(`artifact-text-preview.tsx`, `sessions.attachment`) or by the new `sessions.files.content`
op. No separate canvas index exists, so there is nothing else to change.

**Library (how local assets get in today).**
- The background indexer walks `<config>/sessions` (`library/indexer.py:1-30`). Per session it
  turns the `session_files` ledger into occurrence rows (`_occurrence_rows`, `:247-296`). It
  writes them with `store.replace_session_occurrences` (`library/store.py:1546`) and
  `recompute_assets`, and checkpoints in `idx_sessions` (`store.py:374-389`).
- Thumbnails are 480 px WebP produced off the request path (`store.py:1992`). Their liveness
  is keyed by the source's stat (`thumb_is_live`, `:1969-1990`; `thumb_source_stamp`,
  `:2132`).

**Remote rows enter the SAME tables**, through a new `library/remote.py` feeder called from
the indexer worker:
- **Schema:** `SCHEMA_VERSION` 2 → 3 (`store.py:60`). A version move drops and rebuilds the
  derived tables only (`store.py:954-966`); metadata is untouched.
  - `idx_occurrences` adds `origin_device TEXT NOT NULL DEFAULT ''` and
    `remote_file_id TEXT NOT NULL DEFAULT ''`.
  - `idx_sessions` adds `origin_device` and uses `revision` as the owner's files revision.
  - `idx_assets` adds `remote INTEGER NOT NULL DEFAULT 0`, which is 1 iff every occurrence is
    remote, plus an index on it.
  - Cost: one rebuild. The indexer docstring measured 26.5 s cold for the old in-request
    build; the rebuild is backfilled newest-first, so the grid fills from the top.
- **Identity:**
  - Digest rows keep `a:<digest>`. The same bytes on both devices are ONE asset, which is
    correct for content-addressed media.
  - Remote path rows get a new `identity.remote_key(device_id, session_id, display_path)` =
    `r:<device>:<sid>:<path>`, with `path` stored NULL.
  - **Invariant I5-1:** a remote asset never resolves to a viewer-local path, even when the
    owner's path string exists on this disk.
    Check: the owner row `~/x.png` while the viewer also has `~/x.png` with other bytes; the
    library content of the remote asset ≠ the local file, and no `os.stat` of that path
    happens (use a spy).
- **Feed:**
  - Metadata only. One `peer_session_files list` per remote session per owner-revision
    change; the full list is paged at `limit=200`.
  - Backfill covers all reachable remote sessions newest-first, with 1 request in flight per
    owner. It resumes from `idx_sessions` (key = remote session id, signature = owner
    revision).
  - When the viewer's own Files panel lists a remote session, the page rows are upserted for
    free.
  - Sessions that leave the catalogue for two consecutive successful catalogue reads (moved,
    deleted) are dropped. An UNREACHABLE peer's rows are kept.
  - A session that becomes local is dropped from the remote feed, and the local scanner owns
    it.
- **Thumbnails:**
  - Generated by the existing thread only when bytes are local (`attachments/`, a replica, or
    the remote cache), which happens via precache or an open.
  - `thumbnail_source` (`store.py:1936-1959`) looks in the remote cache for digest assets.
  - For digest assets, the liveness stamp becomes `a:<digest>` instead of the path stat, so
    evicting the cached bytes never invalidates a thumbnail. The digest IS the content.
- **No per-view mesh fetch:**
  - `list_assets`, `get_asset`, `thumbnail` and `waveform` never call the relay. A remote
    asset with no thumbnail answers today's designed 404 (`desktop_library.py:471-509`), and
    the card shows its glyph.
  - Only the explicit `…/content` (open, lightbox, download) of an asset with a remote
    occurrence goes through §1's fetch-on-miss, using `remote_file_id` or the digest.
  - **Invariant I5-2.** Check: monkeypatch `desktop_mesh._call` to raise; list (page of 200),
    detail and thumbnail of remote assets all answer 200/404 with zero calls.
- **Row:** `server/models/desktop_library.py` `LibraryRow` gains `remote: {device_id,
  device_name} | null` (additive). Library actions in
  `ui:…/library/components/library-asset-menu.tsx:120-216`:
  - Open source conversation: works (it opens the remote session).
  - Ask, Favorite, Edit metadata, Edit tags, Copy asset id: work (local metadata keyed by
    `asset_id`).
  - Copy path, Reveal in Finder: absent (no `path`, by construction).
  - Download: `library.content?download=1`, ≤32 MiB through the media relay; larger assets go
    to the core download route through the source occurrence.

## 6. Capability and version skew

- **Desktop flag:** add `"remote_files": 1` to the static features table beside
  `session_files` (`server/features.py:1141`). It means: peer fetch-on-miss, peer files and
  user-attachments listings, and the content and download routes.
- **UI gating:** `use-session-files.ts:62-71,121-128` keeps today's "unsupported" message only
  when `remote && !remote_files`.
- **Older OWNER peer:** detected from link capabilities BEFORE any request, the same pattern
  as `files.py:800-808`.
  - No `mesh-file-v1`: bytes answer 409 `attachment_peer_outdated` with "update <device>".
  - No `mesh-file-index-v1`: the listing answers `unsupported_reason: "peer_update_required"`
    plus `device_name`, and the UI says "Update <device> to see its files here".
  - The library feeder records `last_error="peer_update_required"` for that device, and
    `index_state` lists the outdated devices.
  - Never a silent fallback to a client-side scan: `use-session-files.ts:46-49` already
    forbids that.
- **Older VIEWER daemon with a newer UI:** the flag is absent, so the UI behaves as today.

## 7. Slice plan

### Core PR — `feat/remote-files-parity` (very significant, 3–4 h; backend + design round for the wire copy)

The commits are ordered so each one is reviewable alone. **Split point:** if the lane reaches
its budget after C4, C5–C6 go to a follow-up PR in this session. C1–C4 alone satisfy the
transcript and Files-panel acceptance.

1. **C1 `perf(network): memoise and stream the referenced-digest scan`.** Change sites:
   `sync.py:1106-1126` and `files.py:185-213`. Test: the spike shape, as a counting test (scan
   invocations per pull == 1). RED on base: 65.
2. **C2 `feat(desktop): fetch peer attachments on miss`.** Change sites:
   - `remote_files/{cache,fetch}.py`;
   - `files.py` `land`;
   - `desktop_sessions.py` (the four peer branches in §1);
   - `routes/desktop_sessions.py` status mapping (`:2229`) and streaming responses
     (`:4021-4050,4083-4155,4245,4281`);
   - `server/utils/byte_range.py` (lifted from `desktop_library.py:358-397`);
   - the config key plus `settings_io`;
   - `features.py`.

   Acceptance on the real loopback: build `DesktopSessions(server_b.root)` over a started
   pair (pattern `tests/unit/network/test_remote_viewer.py:307-360`, fixture `mesh` from
   `test_mesh_file_pull.py:60-66`):
   - (a) A G2 PNG, WAV and MP4 → 200, byte-identical, landed in `cache/remote-files` and not
     in `attachments/`.
   - (b) G1: a replica-held blob → 200 with no relay call.
   - (c) `Range: bytes=0-99` → 206 with 100 bytes.
   - (d) Owner relay stopped → 504 `attachment_owner_unreachable`; a later restart → 200.
   - (e) Throttled `_transfer` → 504 `…fetch_pending`, and a later GET → 200 from the same
     single transfer (count the `stat` phases == 1).
   - (f) View revoked → 409 `attachment_on_peer`, cache empty.
   - (g) Unreferenced digest → 409.
   - (h) Peer without `mesh-file-v1` (monkeypatch the caps) → 409 `attachment_peer_outdated`.
   - (i) A child route for a remote subagent's image → 200.
   - (j) Waveform for a remote WAV → payload.
   - (k) The I1-3 tracemalloc check.
   - (l) The I4-1 eviction check.
   - RED on base: (a) and (b) are 409; (i) is 404.
3. **C3 `refactor(session-files): move family discovery out of the desktop module`.** A pure
   move plus re-exports. Tests: `contained_descendants is family.contained_descendants`, and
   an import guard: `import local_operator.network.files` leaves `fastapi` out of
   `sys.modules` (run in a subprocess).
4. **C4 `feat(network): net_session_files peer index`** plus the viewer route changes
   (`desktop_session_files.py:223-224,326-327`), the content and download routes,
   `wire.LINK_CAPABILITIES`, `types.py` rows, and the totality test
   (`authorizer.py:470-495`).

   Acceptance:
   - I2-1, I2-2, I2-3.
   - A per-row `remote_unreadable` for a cwd file, a missing file and a `.env` pad file.
   - The I3-2 download checks, plus a `~/Downloads`-absent fallback.
   - An owner without `mesh-file-index-v1` → `peer_update_required`.
   - A >8 MiB-reply fixture (rows padded via long names) → trimmed page, correct next cursor,
     link still up.
   - RED on base: the listing is `unsupported`.
5. **C5 `feat(desktop): precache active remote sessions`.** Acceptance:
   - The active-set selection is unit-tested against synthetic catalogue rows (2 h, the
     opened LRU, the cap of 16).
   - A loopback precache lands only image/audio/video ≤ 32 MiB and respects the budget and
     priority (an on-screen request completes while the precache queue is full).
   - The RSS check from §4.
6. **C6 `feat(library): index remote sessions' files`.** Change sites:
   - `store.py` schema v3;
   - `identity.remote_key`;
   - `library/remote.py`;
   - `indexer.py` worker hook;
   - `service.resolve_content`/`thumbnail_source` remote branches;
   - the `LibraryRow.remote` field.

   Acceptance: I5-1, I5-2, a remote image thumbnail generated after precache and surviving
   cache eviction, dedup of a digest present locally and remotely to one asset with two
   occurrences, and removal on move.

- **Docs:** `docs/DESKTOP_API.md` "Session files" plus the new routes and statuses. i18n keys
  for every new sentence (AGENTS.md §"Internationalization is mandatory").
- **Test lease:** run only the new and changed test files plus
  `tests/unit/network/test_mesh_file_pull.py`, `tests/unit/server/test_desktop_remote_open.py`
  and `tests/unit/test_settings_io.py`. Wrap with `run_bounded.py`, ≤ 10 min each. CI runs
  the suite.

### UI PR — `feat/remote-files-parity` (feature, ≤3 h; coder/reviewer/QA + designer + ux-reviewer)

Change sites:
- `use-attachment-url.ts` (§1 retry), `use-waveform.ts`.
- `use-session-files.ts` (gate on `remote_files`; `unsupported_reason`).
- `session-files.ts` row type (carry `remote`, `origin_device`, `remote_read`,
  `remote_unreadable`).
- `file-row.tsx` and both menus (the §3 table, a remote indicator per design).
- `src/main/desktop-media.ts` (new ops `sessions.files.content`, plus the GET allowlist at
  `:864-880`).
- desktop request op `sessions.files.download` (long-budget write).
- Library card and menu (`library-asset-menu.tsx`, `library-model.ts`).
- i18n keys.

Acceptance:
- The node tests:
  - 504 → retry ×5 then `retryable`, 409 → `missing`, success after one 504 paints;
  - the remote action table, including I3-1;
  - capability-absent → today's message;
  - `peer_update_required` → the named sentence.
- Xvfb captures (RIG.md, one heavy rig at a time):
  - (1) A transcript with 1 and 5 remote images: before = broken tiles, after = images.
  - (2) Files panel of a remote session under the same sticky headers with the remote
    indicator, light and dark.
  - (3) The row menu for readable and unreadable rows.
  - (4) The Download receipt toast, then the Open folder action.
  - (5) The Library grid with remote cards.
  - (6) The 504 retryable tile state.

**Fixture changes the rig needs** (`loading-capture.mjs` `startFixture`, or `qa/fixture.mjs`):
- `/v1/capabilities` gains `features.remote_files: 1` and `session_files: 1`.
- The catalogue has one remote row (`locality: "remote"`, `owner_device_name:
  "lop-remote-node-2"`).
- `/v1/desktop/sessions/<rid>/history` rows reference 5 image digests, 1 WAV and 1 MP4.
- `/attachments/<d>` serves real bytes from fixture files, with a per-digest toggle that
  answers 504 N times and then 200.
- `/attachments/<d>/waveform`.
- `/files` returns rows of every kind in the §2 table, with `remote`, `remote_read` and
  `remote_unreadable`.
- `/user-attachments`, `/files/<id>/content`, and `POST /files/<id>/download`, which returns a
  path inside the rig tmp.
- `/v1/desktop/library/assets` returns mixed local and remote rows, plus `/thumbnail`.
- Anything unlisted still 404s by name.

Real-path QA (core): the two-relay loopback above, plus one cross-host pull to
`lop-remote-node-2` of a ≥ 32 MiB file. Record the wall time; this measures the WAN number §8
asks for.

## 8. Risks to watch during rollout

- **WAN throughput is unmeasured.** The 20 s wait was set from loopback, and slow links will
  show `fetch_pending` more often. That is safe, because the transfer continues, but QA should
  record the cross-host 32 MiB time. If max × 3 > 20 s, the constant moves on that evidence.
- **The operator's Mac distribution is unmeasured.** It has music and video; this host has
  images only. The 1 GiB default and the 32 MiB ceiling rest on host data plus the UI's own
  limit. To settle it, run §0's two Python snippets on the Mac (n, median, max per mime
  major), then revisit the default.
- **Owner load from Library backfill:** each cold remote session costs one ledger scan on the
  owner (budgeted at 1.5 s per request, resumable). Watch `net_session_files` latency and
  owner CPU during the first backfill. There is only 1 request in flight per owner.
- **The schema v3 rebuild** happens once per viewer on upgrade, and the grid refills
  newest-first.
- **Renderer memory:** media still arrives as a ≤32 MiB blob per open (an existing local
  behaviour). A Range-capable custom protocol would be a separate change.
- **Relay/daemon generation skew:** handled by lookup order (§4). Watch for remote blobs
  accumulating in `attachments/` on skewed installs.
- **Compaction:** a file the owner's transcript no longer references becomes
  `not_referenced`, and is honestly unreadable even though it was listed before.
