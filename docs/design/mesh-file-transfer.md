# Mesh file/artifact transfer — design + slice plan

Base: `origin/main` @ `10baa6e6d367`. All citations are lines on that commit.

Author: architect (lopdev), static design. Committed with S1 (`feat/mesh-file-pull`) so
S2–S4 cite one text. **Citation audit (S1 author, session `99ffc68d3d05`):** every
`file:line` below was re-read on `10baa6e6d367`. Corrections from the circulated draft:
the desktop image route's host is `local_operator/server/utils/desktop_sessions.py:8296-8371`
(`attachment()` → `PeerAttachmentUnavailable`), not a `server/routes/...:8355` site;
`_session_scope` is `authorizer.py:322-359` (was 325-362); `NET_OPS` is `types.py:346-411`
(was 347-409); `LOCAL_OPS` is `types.py:440-531` (was 440-497); `OP_CAPABILITY` is
`types.py:632-722` (was 632-680); `SessionsParams` is `builtin.py:17011` (was 17016); the
peek handler is `builtin.py:19858-19918` (was -19916). Everything else matched.

## Problem

A session on a peer has no byte path back to the head node. `sessions op='peek' peer=…`
returns clipped text rows (`tools/builtin.py:19858-19918`). The only base64 route goes
through the model, which corrupts bytes. Pieces already in the code:

- **Blob store.** `<config>/attachments/<d>.bin` + `.json` {digest, mime_type, bytes}.
  `d` = the first 32 hex chars of sha256 (`session/attachments.py:239-270`). `get_bytes`
  re-verifies on read (`:404-430`).
- **Chunked, verified copy over the link.** `net_sync` phases plan/fetch/verify. 512 KiB
  raw chunks (`SYNC_CHUNK_BYTES`, `network/sync.py:91`), each with `chunk_digest`
  (`sync.py:2068-2077`), under the 8 MiB record ceiling (`MAX_RECORD_BYTES`,
  `network/wire.py:66`). Resume uses a prefix `verify` (`sync.py:3177-3240`). Writes are
  contained by `_write_scope`/`_open_stage`/`_replace_at` (`sync.py:2925-3149`). It is
  plan-bound, though: one fetch needs a whole-session plan, and the plan records the asker
  as a replica holder (`sync.py:4316-4355`). That is wrong for a single file, so **reuse
  the primitives and framing, not the op.**
- **Authority.** Capability is checked against the link's member row
  (`network/authorizer.py:214-226`). The session must be owned by the serving device
  (`_session_scope`, `authorizer.py:322-359`). Op→capability is a total table
  (`network/types.py:632-722`; totality `authorizer.py:470-496`). Slices register through
  `register_ops` (`network/relay.py:8300`) and `SLICE_MODULES` (`relay.py:551`).
- **Local verb pattern.** CLI → `control_request` → local op → `server._ensure_link(owner)`
  → `LinkTransport.ask` (`sync.py:4418-4539`, `network/mobility.py:1348-1364`).

## Decisions

1. **Wire.** Two new peer ops in the `NET_OPS` tuple (`types.py:346-411`), each registered
   as slow with `SYNC_OP_DEADLINE_S` (`sync.py:84`):
   - `net_file_get`: `stat` → {name, bytes, sha256 (all 64 hex), attachment (32 hex),
     mime, mtime_ns}, then `fetch` {offset, limit≤512 KiB, expect_bytes, expect_mtime_ns}.
     The reply has `serve_fetch`'s chunk shape.
   - `net_file_put`: `begin` {session_id, name, bytes, sha256, mime} → {transfer_id,
     have_bytes}; `chunk` {transfer_id, offset, data, chunk_digest}; `commit`.

   **Digest first.** The full sha256 is known before any byte moves.

   **Resumable.** The receiver stages `<root>/network/staging/files/<sha256>.part`. This
   sits outside `sessions/`, for the reason given at `sync.py:665`. A re-run with the same
   sha256 resumes from the part's size. `begin` returns `have_bytes` for push.

   **End to end.** Nothing becomes visible (store, `--out`, pad) until the whole-file
   sha256 matches. On a mismatch the part is deleted and the client retries once from
   zero, then refuses.

   **Caps.** 512 KiB per chunk, about 683 KiB as base64. 256 MiB per transfer, which
   equals `SCRATCHPAD_TOTAL_BUDGET_BYTES` (`scratchpad.py:605`). At most 2 inbound puts
   per link, and at most 512 MiB of inbound staging. Parts older than 24 h are swept when
   the relay starts.

   **No new ports.** Peer traffic rides the existing PeerLink. CLI↔relay traffic uses the
   control socket and carries paths, never bytes. The relay reads and writes only inside
   `server.root`. Push: after the sender's checks, the CLI clones the source into
   `network/staging/files-out/`. Pull: the CLI copies the verified file from the store to
   `--out`.

2. **Authority.** The requesting DEVICE's member row on the serving device decides.
   Sessions have no per-session ACL.
   - `net_file_get` needs **view**. This widens nothing: `net_sync` is already `view`
     (`types.py:657`) and already serves the whole `scratchpad/` tree (`sync.py:263`) plus
     the referenced blobs to any view member.
   - `net_file_put` needs **prompt** (drive). Drive members can already put content into
     a session, for example prompts with images (`relay.py:11463`).
   - `_session_scope` is used unchanged, so both ops are refused for a session the peer
     does not own.

3. **Pull confinement.** There are exactly two allowed sources:
   - **`path`.** Parsed by `parse_scratchpad_url(url, scratchpad_root(session_dir))`
     (`scratchpad.py:247-323`). That already refuses absolute paths, `..`, dotfiles,
     nested schemes and symlink escapes (`is_relative_to(base)`). Then
     `is_sensitive_for_supplements(resolved, session_scratchpad=pad)`
     (`supplements/denylist.py:300`) checks the credential vocabulary (`.env`, `id_rsa`,
     `*token*`, …) on both the written and the resolved form. This carve-out lifts only
     the config-dir root rule, and only for this session's own pad. Then the file is
     opened `O_NOFOLLOW` and `fstat` must say S_ISREG (`_open_for_read`/`_opened_regular`,
     `sync.py:3067,3017`). Directories, fifos, sockets and devices are refused, and so is
     a size over the cap.
   - **`attachment=<32hex>`.** Served only if that session's transcript references it
     (`referenced_attachments_in`, regex at `sync.py:613`, function `sync.py:1106`). This
     is the sound route's "referenced, not merely present" rule (`mobile/daemon.py`
     `api_session_sound`, `:7945`). It stops a view member probing the shared store.

   Nothing else is reachable: not the cwd, the config dir, other sessions' pads, or
   credentials.

4. **Landing.**
   - **Pull, on the origin.** The verified part is renamed into the origin's store by a
     new `AttachmentStore.adopt_file(staged, mime)`. It writes content and then the
     sidecar, in the `put_bytes` sidecar shape, by rename instead of `write_bytes`,
     because 256 MiB will not fit in memory. Then:
     - CLI `--out PATH` uses O_EXCL and refuses an existing file unless `--force`.
     - The agent tool also lands the file at `scratchpad://pulled/<peer>/<basename>`
       through `check_scratchpad_write` (`scratchpad.py:969`). If that is refused (a
       write-refused suffix, or the budget), it reports the store path instead.
   - **Push, on the peer.** The file goes into the peer store and also into
     `<session>/scratchpad/received/<basename>`, written only through `_write_scope` and
     `_open_stage`:
     - The sender supplies a basename only. `/`, `\`, `..`, a leading `.` and NUL are
       refused at `begin`.
     - `check_scratchpad_write` (suffix and budget) runs at `begin`, before any byte moves.
     - A name collision does not overwrite; the file becomes `name (1).ext`.
     - The receipt returns `scratchpad://received/<name>` plus the digest, and the sender
       puts that reference into its next `send(peer=…)`.
     - Auto-attaching to a prompt (`files` refs, `session/runtime/server.py:8482-8494`,
       gated by `document-attachments-v1`) is deferred. It is not needed for byte fidelity.

5. **Surfaces.** Minimal set:
   - **CLI:** `lop network files pull <peer> <session> <scratchpad://path|digest>
     [--out F] [--force] [--json]` (S1 as built: positional, per the manager's S1 brief;
     the draft's `--peer/--session` flags were not adopted) and, in S2,
     `lop network files push <peer> <session> <file> [--name N] [--json]`. It is a new
     subparser beside `approvals` (`network/cli.py:337`). `network sessions` is a flag
     family (`cli.py:450-560`), and two path-taking verbs do not fit it.
   - **Agent:** `sessions op='pull'|'push'` with one new field, `path`. It reuses the
     tool's existing `peer`/`session`/`target` addressing and its `_mesh_op` → CLI path
     (`builtin.py:19858-19880`, argv `:20256`), so one writer stays in charge. Push runs
     the normal read guardrail plus `is_sensitive_for_supplements` with the caller's own
     pad on the local path. An image, video or audio pull also returns an
     `AttachmentContent` block (the `cache_media` shape, `attachments.py:434`), so it
     renders in the head session.
   - **Receipt** (both surfaces): `{ok, direction, peer, session_id, name, bytes, sha256,
     attachment, mime, verified: true, landed: {store, out|scratchpad}, resumed_bytes}`.
     Refusals carry `{ok: false, code, message}`. An old peer is recognised before the
     first request by the `mesh-file-v1` link capability (S1 as built, the family's
     pattern: `meshrestart.py`, `readiness.py`) and maps to "update <device> to use file
     transfer".

6. **Viewer gap (3).** Two precise gaps:
   - **G1: replica blobs exist but are never read.** `sync_from` lands replica blobs in
     `<root>/network/replicas/<sid>/attachments/`: the default is `blobs = destination`
     (`sync.py:3525-3527`), and both callers pass no `attachments_root` (`sync.py:4217`,
     `:4508`). Only a move points it at the shared store (`mobility.py:1794`). The desktop
     byte route reads only `<root>/attachments`
     (`server/utils/desktop_sessions.py:8360`). On a miss it raises
     `PeerAttachmentUnavailable` (`:8362`) → 409 (`server/routes/desktop_sessions.py:2221`).
     So a replica-held image still fails, despite the docstring's claim at
     `server/utils/desktop_sessions.py:8334`.
   - **G2: nothing fetches a digest from the owner.** Cold history does not inline blobs
     (`relay.py:10023-10026`; `docs/design/mesh-cold-read-stored-history.md:125` names
     this follow-up). Only the live stream carries bytes, and only during a move does
     anything fetch a referenced digest the receiving device lacks (mobility §9 step 9).

   **Fix.** Edit the peer branch at `server/utils/desktop_sessions.py:8355-8362`, and its
   waveform/sibling branches at `:8410` and `:8464`:
   - first try the replica store for that session id;
   - then try the local op `session_file_pull {session_id, attachment}` with a 20 s
     timeout. It lands the blob in `<root>/attachments`, and the route serves it;
   - on refusal or unreachable, keep today's 409 sentence.

7. **Out of scope.**
   - local-operator-ui: the route's success shape is unchanged, so a 200 renders where a
     409 drew the placeholder. Whether the UI caches the 409 and never re-requests is the
     one fact to check in the S4 design round. If it does, a one-line retry fix in the UI
     goes in a follow-up PR.
   - The mobile daemon's `_attachment_bytes`: it is local-session-only
     (`mobile/daemon.py:3790-3810`).
   - The TUI remote viewer's cold replay (`store_for_transcript_dir`,
     `attachments.py:145`): it benefits once a blob is in the shared store, but has no
     fetch-on-miss of its own.
   - Auto-attach on push, and directory or glob transfers.

## Slices (order: S1 → S2 ∥ S4 → S3)

Every slice tests on the real two-relay loopback (`_pair`,
`tests/unit/network/test_relay_e2e.py:1158`).

**S1 — `net_file_get` + `lop network files pull` (feature, ≤3 h; backend:
coder/reviewer/QA).** Changes:
- new `local_operator/network/files.py`, with `install` registering `net_file_get` (slow)
  and local `session_file_pull`;
- add the module to `SLICE_MODULES` (`relay.py:551`);
- `types.py`: `NET_OPS` `:346-411`, `OP_CAPABILITY` `"net_file_get": "view"` `:657`,
  `LOCAL_OPS` `:440-531`;
- `AttachmentStore.adopt_file` (`attachments.py` ~`:270`);
- CLI subparser + `_cmd_files` (`cli.py` ~`:337`);
- the needed sync primitives (`_open_for_read`, `_opened_regular`, `_write_scope`,
  `_stream_digest`) made public under stable names, not copied.

Acceptance:
- (a) a 3 MB PNG and a log are pulled byte-identical, the receipt sha256 equals
  `shasum -a 256` on both ends, and the store sidecar mime is correct;
- (b) a read-role member succeeds;
- (c) negative controls: a member without view gets `not_authorised`; a session not owned
  by the peer is refused; in both cases no staging file is created;
- (d) traversal refusals, each naming its rule, with nothing staged: `../x`, `/etc/hosts`,
  `scratchpad://.env`, a pad symlink → `<config>/secrets`, `id_rsa`, a fifo, a directory,
  and a file over the cap;
- (e) corruption control: the owner flips one byte in one served chunk, the transfer is
  refused, and the store, `--out` and the part are all absent;
- (f) resume: cut the link after N chunks, re-run, see `resumed_bytes > 0` and the
  transfer still verifies;
- (g) `attachment:<d>` not referenced by the session is refused;
- (h) a peer without the op gives the named update refusal.

RED on base: `lop network files` exits 2. This is a missing capability, not a regression.

**S2 — `net_file_put` + `lop network files push` (feature, ≤3 h; backend).**
Changes: `files.py` put phases with the staging/inflight caps and the start-up sweep;
`OP_CAPABILITY` `"net_file_put": "prompt"`; CLI push; landing in the peer store +
`scratchpad/received/`.

Acceptance:
- a drive member's push lands byte-identical in both the store and the pad;
- a read member is refused at `begin`, with nothing staged;
- names `../../x`, `a/b`, `.bashrc` and one containing NUL are refused;
- a pre-planted `scratchpad/received` symlink pointing outside is refused by containment;
- a collision gives ` (1)` and no overwrite;
- an over-budget push and a refused suffix (`.zip`) are refused at `begin`;
- corruption control: a tampered chunk makes `commit` refuse, with no store or pad file;
- resume via `have_bytes`;
- a local source path in `~/.ssh` is refused by the sender's sensitivity check.

**S4 — viewer fetch-on-miss (fix, <2 h; backend + design round, since it is visible).**
Changes: `server/utils/desktop_sessions.py:8355-8362`, `:8410`, `:8464` → replica store,
then `session_file_pull`.

Acceptance:
- RED on base:
  - a remote session whose replica holds the blob → 409 (G1);
  - a remote session with no replica → 409 (G2);
- after the fix, both return 200 with identical bytes, and in G2 the blob is now in
  `<root>/attachments`;
- negative: when the peer revokes view, the 409 sentence is unchanged;
- negative: an unreferenced digest is refused by the owner;
- design round D: before/after desktop screenshots of a remote session's image row
  (placeholder → image), plus the 409 state.

**S3 — agent tool `sessions op='pull'|'push'` (feature, ≤2 h; backend + design round for
the rendered attachment).** Depends on S1+S2.
Changes:
- `SessionsParams.op` Literal + `path` field (`builtin.py:17011`);
- validation beside `_sessions_peek_validation_error` (`:17517`);
- handlers beside `_sessions_peek_mesh` (`:19858`);
- help/tool_docs entry.

Acceptance:
- on the real loopback, a tool call yields the receipt, and the result text holds no
  base64 (assert there is no run of ≥200 base64-alphabet characters);
- the pulled PNG renders as an attachment row: a TUI SVG via `save_screenshot`, and the
  desktop;
- refusals come back verbatim;
- `path` is refused on the other ops.

## Operator decisions

None needed. No new authority is introduced. I decided that `pull` = view (read-role
devices can pull pad files), because `net_sync` already gives read-role devices the same
bytes. If the operator wants pad pulls to be drive-only, that is a one-row narrowing in
`OP_CAPABILITY`.

## Rollout risks to watch

- Relay reader occupancy and link fairness. Each chunk is one ≤683 KiB record,
  interleaved with other ops. Watch `net_session_history` latency during a 256 MiB pull.
- Staging disk use: the inflight and staging caps, and the sweep.
- `*token*`-style denylist false positives, for example `token-usage.log`. The refusal
  names the rule, so an adjustment is a denylist PR, not a bypass.
- Staleness is detected by mtime+size per chunk. A same-size, same-mtime rewrite is caught
  by the final sha256.
- The store key is 128 bits. The receipt carries the full sha256, and verification uses
  the full value.
- Version skew: the requester must map an old peer's refusal to an update hint, never a
  silent fallback.
