# Library — the global asset catalog

Status: implementation contract, 2026-10-18.

The Library is the user's long-term, cross-conversation catalog of assets. It is
deliberately not the Canvas Files panel: that panel answers "what did **this**
conversation produce?"; the Library answers "where is that thing I made or was
shown, months ago?" across every conversation, subagent and device-local store.

It is one feature with five surfaces:

- the desktop UI's left-sidebar **Library** destination (`local-operator-ui`);
- a full-page **TUI** mode opened with `/library`;
- the mobile relay's phone web surface (`#/library`);
- the Expo mobile app (`app/(app)/library.tsx`);
- a second always-on agent, the **Librarian**, whose pinned session keeps the
  catalog organized.

## 1. User outcomes

1. Every generated image, video and audio asset is findable in one place.
2. Every created or highlighted document — markdown, PDF, slideshow, HTML,
   spreadsheet, code, plain text — is findable there too.
3. Assets merely *mentioned* in a conversation are visible separately from
   assets a turn *created*.
4. The user can search, filter by kind/category and tag, and sort the catalog.
5. Selecting an asset previews it in the right renderer; PPTX is a
   first-class, multi-slide preview rather than an OS hand-off.
6. A composer at the bottom of the preview starts or resumes a chat about the
   selected asset and keeps that association.
7. Assets carry editable name, description and tags, so the catalog is
   searchable and organizable beyond filenames.
8. The Librarian keeps the catalog organized by default: a daily, bounded run
   that names, describes and tags new/changed assets, learns the user's tag
   vocabulary and records its conventions.

## 2. Vocabulary and classification

### 2.1 Asset identity

An **asset** is one durable artifact, deduplicated across conversations.

- Path asset key: `p:<realpath>` (realpath is resolved when the file exists;
  otherwise the absolute path is used). The absolute path never appears in a
  phone/mesh projection.
- Artifact asset key: `a:<digest>` for content-addressed media in
  `<config>/attachments/<digest>.bin`.
- Asset id: `la_` + first 16 hex characters of `sha256(key)`. Opaque, stable,
  URL-safe, and never a path.
- An asset seen in several sessions is **one row** with several occurrences.
  Same digest in two sessions dedupes. The same file both written and mentioned
  merges into one asset; the strongest origin group wins and the mention remains
  as an occurrence.

### 2.2 Origin and origin group

The scanner's raw origins stay authoritative:
`generated | written | screenshot | download | edited | command | highlight |
mentioned` (strongest first; `mentioned` is new and last).

The Library projects them into three user-facing groups:

- `generated` — a tool generated the bytes (image/audio/video, digest-only).
- `created` — a turn wrote, edited, downloaded, screenshotted or produced a
  path through a command.
- `mentioned` — the asset was only referenced by the conversation
  (`highlight` or `mentioned`), never produced by it.

A file that is both created and highlighted is **created**, with
`highlighted: true`; highlight is a flag, not a competing category.

### 2.3 Kind and category

`kind` is the preview classifier (one of `image`, `audio`, `video`, `pdf`,
`pptx`, `docx`, `markdown`, `html`, `spreadsheet`, `text`, `code`, `other`).
`category` is the sidebar bucket:

`image`, `sound`, `document`, `presentation`, `video`, `spreadsheet`, `code`,
`other`.

Both are derived from the extension/MIME in one pure module
(`local_operator/library/classify.py`), shared by the index, routes and tests.
No surface re-derives them.

### 2.4 Tags

Library tags are a **separate vocabulary from conversation tags**. Conversation
tags organize conversations; library tags organize assets. They share the same
grammar and the same UI primitives, but not the same registry or limits.

- Grammar: `[a-z0-9][a-z0-9_-]{0,23}` (the projects/session tag grammar).
- A tag has a stable 10-hex id, a display name (lowercase), a category
  (`topic | project | format | status | source | custom`) and a creator
  (`user | librarian | system`).
- An asset may carry at most 12 tags; one may be `primary`.
- Renaming a tag preserves assignments. Merging tags is explicit.
- The Librarian may create tags, but must first reuse an existing tag whose
  meaning matches; near-duplicates are a defect, not variety.

## 3. Storage

Authoritative metadata and the derived catalog are separate, as elsewhere in
the codebase.

### 3.1 Authoritative metadata — `<config>/library/library.sqlite3`

SQLite, WAL, `busy_timeout=5000`; every write is transactional. Tables:

```sql
assets(asset_id TEXT PRIMARY KEY, key TEXT UNIQUE NOT NULL,
       name TEXT, description TEXT, name_source TEXT, description_source TEXT,
       favorite INTEGER NOT NULL DEFAULT 0, locked INTEGER NOT NULL DEFAULT 0,
       archived INTEGER NOT NULL DEFAULT 0,
       organized_at REAL, organized_by TEXT, confidence REAL,
       created_at REAL NOT NULL, updated_at REAL NOT NULL)
tags(id TEXT PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE NOT NULL,
     category TEXT NOT NULL DEFAULT 'custom', created_by TEXT NOT NULL,
     created_at REAL NOT NULL, updated_at REAL NOT NULL)
asset_tags(asset_id TEXT NOT NULL, tag_id TEXT NOT NULL,
           source TEXT NOT NULL, confidence REAL, created_at REAL NOT NULL,
           PRIMARY KEY(asset_id, tag_id))
chats(asset_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
      created_at REAL NOT NULL, last_used_at REAL NOT NULL)
chat_history(asset_id TEXT NOT NULL, session_id TEXT NOT NULL,
             created_at REAL NOT NULL, last_used_at REAL NOT NULL,
             PRIMARY KEY(asset_id, session_id))
conventions(id TEXT PRIMARY KEY, kind TEXT NOT NULL, body TEXT NOT NULL,
            created_by TEXT, updated_at REAL NOT NULL)
edits(id INTEGER PRIMARY KEY AUTOINCREMENT, asset_id TEXT NOT NULL,
      actor TEXT NOT NULL, fields TEXT NOT NULL, before TEXT, after TEXT,
      at REAL NOT NULL)
state(key TEXT PRIMARY KEY, value TEXT NOT NULL)
```

A schema version lives in `state`; readers that do not understand a future
version answer an empty catalog rather than guessing. Metadata rows are never
deleted when an asset disappears from the index: the row is retained so a
returning file keeps its name, description and tags.

### 3.2 Derived catalog — `<config>/cache/library/index.json`

Derived, discardable, rebuilt from journals. It stores **per session**:

```json
{"version": 1, "built_at": 0.0, "sessions": {
  "<session_id>": {"sig": {"size": 0, "mtime": 0.0, "inode": 0},
                   "revision": 0, "title": "...", "cwd": "...",
                   "entries": [ /* FileRecord-shaped rows + mentions */ ]}}}
```

The aggregate (asset rows, occurrence merge) is built in memory from this file
and cached in-process. A missing/torn/wrong-version file costs a rebuild, never
an error.

Refresh is incremental and resumable:

1. enumerate user-visible sessions (`resume._scan_sessions`,
   `resume.is_user_session`) plus contained subagent journals attributed to
   their parent;
2. for each session, reuse the cached entry list when its signature matches;
3. otherwise refresh that session through the existing
   `session_files.ledger.refresh` path and persist its new signature;
4. stop at a time budget, persist what was scanned and answer
   `state: "building"` with `sessions_scanned/total_sessions`; a single-flight
   background task finishes the rest and the client polls.

`revision` is a monotone integer over the aggregate (sum of per-session
revisions plus index generation). A cursor minted at an older revision restarts
at the top with `cursor_reset: true`, exactly like the session-files list.

### 3.3 Mentions

`session_files.scan` grows a conservative mention pass over the journal, so all
surfaces get the same answer:

- `<operator-references>` tokens (`typed="@…"`) and `file://` URLs;
- path-valued keys in tool arguments (`path`, `file`, `files`, `output_path`,
  `output_file`, `directory`, `dir`, `cwd`);
- absolute or `~/` prose paths with a known extension in **user** messages;
- user-attached media digests (recorded as mentioned artifacts).

A mention must resolve and stat successfully, pass the sensitive-path denylist,
and be a regular file. Relative prose is not admitted. `SCAN_SCHEMA` is bumped;
existing caches are discarded and rebuilt.

## 4. HTTP contract

All desktop routes are bearer-gated (`require_desktop`) and follow the existing
`CRUDResponse` envelope. The relay exposes the same semantics under `/api/library`
with `display=True` (no absolute paths).

```
GET    /v1/desktop/library/assets
       ?q=&category=&kind=&origin_group=&origin=&tag=&tag_mode=all|any
       &highlighted=0|1&organized=0|1&favorite=0|1&session=
       &sort=recent|name|kind|size|created|modified|organized
       &order=asc|desc&limit=1..200&cursor=
GET    /v1/desktop/library/assets/{asset_id}
PATCH  /v1/desktop/library/assets/{asset_id}
       {name?, description?, favorite?, locked?, archived?}
PUT    /v1/desktop/library/assets/{asset_id}/tags
       {tags:[tag_id...], primary?}
GET    /v1/desktop/library/assets/{asset_id}/content
       Range supported; inline disposition; nosniff; no absolute path input
GET    /v1/desktop/library/assets/{asset_id}/thumbnail?w=480
PUT    /v1/desktop/library/assets/{asset_id}/chat  {session_id}
DELETE /v1/desktop/library/assets/{asset_id}/chat
GET    /v1/desktop/library/tags
POST   /v1/desktop/library/tags  {name, category?}
PATCH  /v1/desktop/library/tags/{tag_id}  {name?, category?}
DELETE /v1/desktop/library/tags/{tag_id}
POST   /v1/desktop/library/tags/{tag_id}/merge  {into}
GET    /v1/desktop/librarian
POST   /v1/desktop/librarian  {op: open|pause|resume|run|status}
```

`GET /library/assets` answers:

```json
{"state": "ready|building", "revision": 12, "total": 42,
 "counts": {"category": {...}, "origin_group": {...}, "kind": {...}},
 "tags": [{"id": "...", "name": "...", "category": "topic", "count": 3}],
 "rows": [ ... ], "next_cursor": null, "cursor_reset": false,
 "scan": {"sessions_total": 142, "sessions_scanned": 142},
 "librarian": {"enabled": true, "session_id": "...", "paused": false,
               "last_run_at": 0.0, "last_summary": "..."}}
```

A row:

```json
{"id": "la_0123456789abcdef",
 "ref": {"type": "path|artifact", "path": "/abs/only/on/desktop",
         "display_path": "~/x.png", "digest": null,
         "content_type": "image/png"},
 "name": "...", "derived_name": "...", "description": "...",
 "kind": "image", "category": "image",
 "origin": "generated", "origin_group": "generated", "origins": ["generated"],
 "tags": [{"id": "...", "name": "...", "category": "topic"}],
 "primary_tag": "...", "size_bytes": 0, "width": 0, "height": 0,
 "duration_s": 0.0, "created_at": 0.0, "modified_at": 0.0, "last_seen_at": 0.0,
 "exists": true, "highlighted": false, "highlight_why": null,
 "occurrences": [{"session_id": "...", "session_title": "...",
                  "origin": "generated", "tool": "generate_image",
                  "at": 0.0, "intent": "..."}],
 "session_count": 1,
 "chat": {"session_id": "...", "title": "...", "updated_at": 0.0},
 "preview": {"state": "ready|unsupported", "mime_type": "...", "pages": null},
 "organized": {"at": 0.0, "by": "librarian|user", "confidence": 0.8},
 "locked": false, "favorite": false}
```

`q` searches the effective name, description, tag names, derived name, display
path and source conversation titles, case-insensitively. Tag filters are AND by
default (`tag_mode=any` for OR). Sort defaults: `recent` desc; name asc; kind
asc; size/created/modified/organized desc.

`content` resolves the asset id to its indexed ref server-side. It never accepts
a caller path. Path assets are re-checked against the denylist and must still be
one of the indexed paths. The digest route reads the attachment store only.
Range requests answer 206; `Content-Disposition: inline`; `X-Content-Type-Options:
nosniff`; `Cache-Control: private, max-age=60` for paths and
`private, max-age=31536000, immutable` for digests.

`thumbnail` is best-effort and image-only (Pillow; max width 480; WebP/JPEG
cache under `<config>/cache/library/thumbs/`). SVG/GIF are served original;
video/audio/document rows carry no thumbnail and surfaces draw a kind icon.

### 4.1 Chat association

`PUT .../chat` records `{session_id}` after the caller has created a real
session (the desktop/mobile/TUI create routes stay the one way sessions are
made). The route verifies the session directory exists and that the id has the
session-id shape. `GET asset` returns the active chat; if the session directory
has been deleted, the row answers `chat: null` and the stale link is cleared on
the next read. `DELETE` clears the link.

The first message is the **caller's** composition: the caller creates the
session, sends the prompt (with image/audio bytes where the asset is media, or
a materialised path/digest reference where it is a document), then records the
link. This keeps one session-creation path per surface and avoids a second
runner in the library route. The Library composer then continues the same
session through the ordinary prompt route.

Digest-only artifacts that a chat must read as a file are materialised through
the existing per-session materialise route first; the composer passes the
source session id. This is a documented cost, not a silent assumption.

## 5. The Librarian

The Librarian is Aida's twin, not a new scheduler.

- Packaged seed `agent_seeds/librarian.md`, manifest entry with
  `class:proactive`; no `origin.json` and attachment `agent="librarian"`, so the
  session is user-visible on every surface.
- Modules `local_operator/librarian/{state,bootstrap,proactive}.py` mirror
  Aida's lifecycle. State lives under `<config>/librarian/`.
- Config: `librarian.enabled` (default true), `librarian.name` (default
  "Librarian"), `librarian.cadence.at` (default `03:30`),
  `librarian.cadence.paused`, `librarian.cadence.max_extra_per_day` (1),
  `librarian.cadence.min_gap_minutes` (720). `LOCAL_OPERATOR_NO_LIBRARIAN=1`
  is the hard off switch.
- One owned wake row `librarian-cadence` plus bounded `librarian-extra-N`
  rows; the engine is the only re-armer/dropper. Pause/disable semantics match
  Aida's: config key + `held_at` + row removal; delivery-time guard; load-time
  filter.
- Boot hooks: TUI boot and server lifespan call `ensure_session`, which creates
  the session if missing, titles it, pins it, writes state, arms the cadence on
  every call, and reaps orphaned rows.
- The daily run reviews only unorganized or changed assets, bounded per run.
  It applies best-effort metadata, records confidence and provenance, never
  overwrites a user-edited field, skips `locked` assets, and records taxonomy
  conventions. A quiet day is a quiet message.
- `/v1/desktop/librarian` exposes status/open/pause/resume/run; the TUI adds
  `/librarian`; the desktop Library header shows status and "Organize now".
  The phone app needs no new tab: the pinned session is the surface.

### 5.1 Library memory and preference learning

- `conventions` rows hold the Librarian's taxonomy: preferred tag categories,
  naming rules, recurring themes, and aliases to avoid.
- `edits` is the learning signal: every user change is recorded with actor,
  before/after and timestamp. The Librarian reads it before inventing a tag.
- The `library` agent tool exposes `stats`, `list`, `get`, `update`, `tags`
  and `conventions`. It is available to every agent; the Librarian seed tells
  it how to use the write ops. The Librarian is the only agent expected to run
  unattended.
- The tool records `source: librarian` and confidence; user edits set
  `source: user`, and a user field is never overwritten by an agent unless
  `locked` is false and the user explicitly asks.

### 5.2 Highlights integration

- Highlighted files already reach the catalog through the scan's `highlight`
  origin. The Library index overlays the newest supplement version exactly as
  the session-files service does, so retracted callouts disappear.
- The stable files-decision prompt gains a constant paragraph: selected files
  enter the user's long-term Library; prefer durable deliverables over
  incidental scratch files. This text is constant (cache-stable), not a
  per-turn taxonomy dump.
- The base system prompt gains a short Library section: the Library is global
  and long-term; the Canvas Files panel is session-local; deliverables and
  highlights enter the Library; use the `library` tool for durable names,
  descriptions and tags. This rides the stable head block.
- The Librarian picks up highlighted assets as `unorganized` and tags them in
  its daily pass. No highlight job blocks on library writes.

## 6. Surface contracts

### 6.1 Desktop UI (`local-operator-ui`)

- New `/library` route and a left-sidebar destination (Library icon), gated on
  capability `features.library`.
- Sidebar information architecture: the primary column is Aida, Library,
  Projects, Agents, Schedules, Browser. Agent Hub and Mesh move into one
  collapsed overflow group at the bottom, opened by an ellipsis / `More` row
  that reveals the rows inline (ChatGPT-style), with `aria-expanded`, keyboard
  order and a persisted expanded state. Low-frequency destinations added later
  join that group rather than lengthening the primary column. New chat and
  Search stay at the top.
- Page layout: filter rail (kinds, origin groups, tags, organized state) +
  search/sort header + asset grid/list + preview dock. Narrow widths collapse
  the rail into a sheet and the preview into the selected route.
- Cards: thumbnail/kind icon, effective name, kind/size, source conversation,
  tags, highlight mark, origin badge. Every card/row carries a per-item
  ellipsis menu: open the source conversation (one entry per occurrence,
  deep-linked to the message sequence when available), ask about the asset,
  copy path/id, reveal or save a copy, favorite, edit tags/metadata, and filter
  the catalog to that conversation. Digest-only artifacts offer materialise/
  save-a-copy.
- Preview: reuse the existing canvas viewers for image/audio/video/PDF/HTML/
  markdown/spreadsheet/code; add a PPTX viewer using `pptx-browser` (MIT, zero
  dependencies) with slide navigation, thumbnail rail, fit/zoom, keyboard
  arrows and honest progress/error states. The occurrence/session link is the
  provenance surface: the preview header names the source conversation and
  links back to it.
- The category model stays extensible for a future `music` kind (Suno/Radient
  generation is a follow-up; no empty Music filter is shipped until it has
  assets).
- Metadata: inline name edit, description editor, tag chips, lock, favorite.
  User edits are optimistic but re-read after a refusal.
- Composer: borderless, no divider above it, max width tied to the preview
  column, 1-3 rows. If the asset has a chat, it posts to that session; otherwise
  it creates a session, sends the first prompt with the asset reference, and
  records the link. A compact "Chat about this asset" thread chip opens the
  session.
- Librarian: a small status row in the Library header (last run, counts,
  Organize now, Open). No separate nav tab. The pinned session appears in the
  chat sidebar.
- States: loading skeleton grid, empty (no assets / no matches), building
  progress, error with retry, unsupported preview, missing file.
- Keyboard: `/` search, `j/k` move selection, `Enter` preview, `Esc` clear,
  `g` grid/list toggle. Focus visible; all controls reachable.

### 6.2 TUI (`local-operator`)

- `/library` opens a full-page mode (same pattern as `/settings`), leaving the
  dock and composer in place but read-only.
- Two panes: list + detail/preview; the tag/filter rail collapses below 90
  columns. Search `/`, sort `s`, filters `f`, `j/k` move, `space` inline image
  preview, `Enter` open externally, `y` copy path, `e` edit metadata, `a` ask
  about the asset (leave the mode and seed the composer), `L` open the
  Librarian.
- Detail shows name, description, tags, kind, size, source conversations,
  highlight mark, organized state, and preview availability. PPTX shows slide
  metadata and an open-externally action; no terminal renderer is claimed.
- Footer line: Librarian last run and counts.
- `/librarian [status|pause|resume|run]` mirrors `/aida`.

### 6.3 Mobile relay web (`local-operator/mobile/web`)

- Hash route `#/library` and `#/library/a/:assetId`; footer entry beside Past,
  gated on the relay's `library` capability.
- 2-column tile grid at phone width; search (server-backed, debounced), kind
  chips, origin tabs (All / Created / Mentioned), tag filter in a Sheet, sort
  in a Sheet, cursor paging with an explicit "Load more".
- Detail: full-screen sheet with preview, metadata editor, tags, and a
  bottom composer. The composer creates/resumes a session and navigates to it;
  the Library detail keeps the chat link.
- Previews: image, audio, video, markdown, HTML, text; PPTX via a client-side
  PPTX viewer if the bundle budget allows, otherwise an honest "Open on
  desktop" row (decision recorded in the PR). PDF gets a genuine open/download
  action unless a checked-in viewer is added.
- All asset paths are display-form; no absolute path reaches the phone.

### 6.4 Expo mobile app (`local-operator-mobile`)

- `app/(app)/library.tsx` pushed from the conversations pane footer; no tabs
  navigator and no change to the composer-home ADR.
- List: search, kind segmented control, origin chips, tag filter Sheet, grid
  or list, cursor paging.
- Detail: native image/audio/video previews; markdown/HTML through the existing
  renderers; PDF/PPTX through a WebView of the relay-rendered preview page
  (or an honest external-open fallback where a native renderer is unavailable).
- Metadata editor: name, description, tags in a Sheet, following the milestone
  editor's busy-latch and relay-sentence refusal rules.
- Composer: starts/resumes a chat and navigates to the session; the link is
  recorded server-side.
- The pinned Librarian session needs no mobile code beyond the pin; it appears
  in the PINNED group.

## 7. Preview support matrix

| Kind | Desktop UI | TUI | Phone web | Phone app |
|---|---|---|---|---|
| image | yes | inline where supported | yes | native |
| audio | yes | metadata + open | yes | native |
| video | yes | metadata + open | yes (Range) | native |
| markdown | yes | rendered text | yes | native |
| html | sandboxed frame | metadata + open | sandboxed frame | WebView |
| pdf | yes (Chromium) | metadata + open | open/download | WebView |
| pptx | `pptx-browser` | slide metadata + open | `pptx-browser` | WebView preview page |
| docx | unsupported/open | metadata + open | open/download | WebView preview page |
| spreadsheet/code/text | yes | text | text/code | text |

An unsupported cell is a designed state, not a blank pane: it names the file,
the reason, and the action that does work.

## 8. Security and performance

- Every content byte is resolved from an asset id; no caller path is accepted.
- Path assets are re-checked against `session_files.service.make_deny` and the
  indexed set on every read. Phone/mesh projections use `display=True`.
- Metadata writes are local, reversible and non-destructive; no delete/rename of
  user files is in the tool or routes.
- The daily Librarian run is bounded (assets per run, tokens, wall clock) and
  never edits files on disk.
- Cold index cost is paid once, incrementally, off the request path. List calls
  answer within the same ~1.5 s budget as session files and report `building`.
- Grid images use thumbnails, lazy loading and a page cursor; never load every
  asset's bytes to paint the grid.

## 9. Testing and evidence

- Backend: unit tests for classification, identity, index merge, mentions,
  metadata/tags, content/Range, chat link, tool ops, Librarian bootstrap/
  cadence and highlights integration. A real FastAPI app test with a synthetic
  store exercises list → detail → content → tag write → chat link.
- Desktop UI: unit tests for the pure filters/sort and composer association;
  Storybook stories for loading/empty/error/populated and the preview states;
  rendered-frame evidence in the PR for grid, preview, PPTX and composer.
- TUI: render tests and exported frames for list, filter, preview and metadata
  edit.
- Phone web: vitest for the screen states and a real relay pass with a
  synthetic library; phone-width screenshots.
- Phone app: vitest for screen logic and the mock relay; a real relay pass where
  the environment allows it; screenshots/recordings for the detail and composer.
- Every PR carries the independent agent-review round, QA matrix, design round
  for visible surfaces and UX round for changed flows.

## 10. Non-goals for this release

- Copying assets into a Library-owned blob store. Assets stay where they are;
  the catalog references them.
- Rasterising office documents on the server. PPTX rendering is client-side.
- Cross-device transfer of asset bytes. A mesh peer's asset can be named and
  linked, not previewed in this release.
- Automatic deletion, moving or renaming of user files.
- A second full-text search service. SQLite/in-memory search over metadata is
  sufficient at the catalog sizes measured here; FTS5 is a later upgrade.

## 11. Resolved decisions

- **Subagent assets are included**, attributed to their parent conversation
  when containment is proven; otherwise they remain their own occurrence and
  are not shown as a separate source conversation.
- **Mentioned** is conservative: structured evidence (`@`, `file://`, tool path
  args) plus absolute/`~` prose paths with known extensions in user messages.
  Relative prose is not admitted.
- **Library tags are a separate vocabulary** from conversation tags; the same
  grammar and UI primitives, different registry.
- **The Librarian applies metadata by default** (best effort), but never
  overwrites a user-edited field or a locked asset; low-confidence suggestions
  are recorded as unorganized rather than guessed.
- **The Library preview is not the Canvas.** The Canvas remains the session
  workspace; the Library reuses its viewers but has its own state and routes.
