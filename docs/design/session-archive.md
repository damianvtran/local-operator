# Session archive and delete

Status: implemented (this PR). Author: coder subagent. Date: 2026-09-18.
Base: `origin/main` @ `2484cfa4`.

Two sibling features over one durable fact. **Archive** hides a conversation from
every list and from search while leaving it resumable by explicit id; **delete**
removes one conversation permanently, behind a typed confirmation and a set of
hard guards.

The pin feature already merged is the pattern for the storage, the capability
keys and the wire shape, and this record cites it rather than re-deriving it: a
new durable flag on a session that was not previously addressable cross-surface.

---

## 1. Semantics

| | visible in default lists | findable by search | resumable by id | recoverable |
|---|---|---|---|---|
| ordinary | yes | yes | yes | n/a |
| **archived** | no | no | **yes** | yes — un-archive |
| **deleted** | no | no | no | **no** |

* **Archive is a statement about OFFERING, never about EXISTENCE.** The rule is
  the one `resume._scan_sessions`'s docstring already states for the subagent
  axis: a listing narrows what is OFFERED, never what exists. `lop resume <id>`,
  the picker's reveal toggle and the desktop snapshot route all reach an archived
  conversation; nothing about the transcript changes.
* **Delete removes exactly one directory.** Subagent runs a conversation
  launched live as siblings under `sessions/` and are NOT removed. That is stated
  in the confirmation a user reads BEFORE the act, because a receipt is the wrong
  place to learn the blast radius of something irreversible.
* **A delete is refused for a session that is in use**, with a sentence naming
  the guard (a live claim or lease, an armed wake, unread spooled mail, or a
  guard that could not be read), because each has a different remedy.

## 2. Storage

A new config-root file, `archived-sessions.json`: a **bare JSON array** of
session ids, written through a same-directory temporary plus `os.replace`,
capped, and **pruned at read** against the session store. It lives in
`local_operator/session/archived.py` beside `cleanup.py` and is deliberately free
of Textual imports so the server and the TUI share one implementation.

### 2.1 What was rejected, and why

* **A per-session sidecar** (`sessions/<id>/archived.json`). It would ride along
  when a session is forked or copied — `fork._EXCLUDED_SIDECARS` is an allow-list,
  so a marker nothing remembers to exclude makes every fork of an archived
  conversation archived too, silently inheriting a state nobody asked for. It
  would also have to be added to `retention._SIDECAR_NAMES` (whose spellings a
  test pins) purely to keep it out of every byte and activity count, and it would
  cost one read per ROW on the picker's synchronous UI-thread path, where the
  index is read once per scan.
* **A field inside the sidebar pins file.** `sidebar_pins`'s own docstring
  refuses to become an object, and the two are different populations with
  different lifecycles — an unpin is a keystroke, an archive is a deliberate act.
  Two facts in one file means every writer arbitrates over both.
* **A database.** The other durable per-session stores here are small JSON
  indexes replaced atomically; a new SQLite dependency for one array of ids
  would need a schema, a migration and a connection per frontend to buy nothing.

### 2.2 Multi-process

Last writer wins, and the granularity is the whole index rather than one id —
the concession `sidebar_pins` already records from its own first cut, made for
the same reason (no cross-process lock for a small index; `os.replace` in the
same directory so a reader never sees a torn file). It is stated here because
this store has more than one writer (the TUI, the desktop route), so the window
a collision can land in is wider than the pins store's was when it made the same
trade. A no-op (re-archiving an archived session) writes nothing, which is what
keeps a client retry from reordering the user's list.

### 2.3 The pin and archive stores are independent, and deletion needs neither

Both prune at read against the session store, so a deleted session reads back as
neither pinned nor archived and `cleanup` knows nothing about either file. The
tests assert this by leaving the records on disk after a delete: a write-time
prune is what would couple deletion to two other modules.

## 3. Filtering

**One predicate, in `resume._scan_sessions`.** Every listing surface reaches its
rows through that function — the `/resume` picker, the TUI sidebar catalogue, the
desktop catalogue and the search digests — so one filter there is what makes them
agree about which conversations exist to be offered. A second filter at a second
call site is how the sidebar and the phone came to disagree about subagent
visibility before the scan owned that question too.

* The parameter is explicit and defaults to `False`: `include_archived`.
* Each row carries `archived`, always present with both values, on the resume row
  and on both wire shapes. The renderer's merge treats an absent key as no claim;
  the same rule `pinned` already follows.
* The index is read **once per scan** and only when it can matter: a store with
  nothing archived pays one failed `open` of a file that is not there and no stat
  or scandir at all.
* **Pinned-and-archived is settled rather than left to fall out.** An archived
  session that is pinned is not offered, so it cannot appear in a sidebar's
  pinned SECTION, and the catalogue's off-page pinned resolution answers `None`
  for the id — no phantom row with a section header and nothing under it.
* **The search index needs no change**, and that is asserted rather than assumed:
  `build_index` is built from exactly the ids its caller listed, so an archived
  id is never handed to it and cannot be reached by a body match. Stale cache
  entries are inert because `_load` is version-gated.
* **The one caller that must override the default is the retention guard.**
  `cleanup._picker_rows` ranks the listing to decide what to KEEP; with the
  default filter an archived conversation would drop out of the recent-N guard
  the moment it was archived — and the sessions a user archives are the older
  ones, so it would drop straight into the ranked set every limit draws from and
  a routine sweep would delete the archive. It passes `include_archived=True`.

## 4. Delete

One entry point besides the automatic policy: `cleanup.delete_session`, which
goes through `cleanup.remove_session_dir` — the only `rmtree` of a session
directory in this codebase, and the thing `test_no_session_deletion.py` walks the
package to enforce.

**What it is allowed that the policy is not.** `RECENT_KEEP` does not apply (it
bounds the AUTOMATIC sweep; a person who names a conversation and confirms a
typed `yes` has answered the question that constant is a proxy for), and neither
does the `session.cleanup.enabled` switch (the incident that switch exists for
was a reaper, not a user).

**What it may not do.** A session in use is refused with a sentence naming the
guard, and a session the user did not open — a delegated subagent run — is
answered as unknown rather than deleted. Deleting is irreversible and the user
cannot see the row they are naming, so an id that is not `is_user_session` is not
a conversation they can mean. The REVERSIBLE verbs keep the looser id-shape
admission the pin route uses, and that asymmetry is deliberate.

**Consistency.** The wake index entry is pruned by the deletion path itself (the
same call the automatic path makes). The pin store, the archive store and the
search cache need nothing: the first two prune at read, and the cache is keyed by
ids a caller lists.

**The LIVE row is the second place the predicate is asked.** A session directory
the scan cannot rank (no transcript, no inbox — a conversation whose owner is
running but which has not been written to yet) is re-added to the catalogue from
the RUNTIME REGISTRY by `catalog.decorate_rows(include_live=True)`. That walk
knows nothing about archives, so it asks the same index, skips a live archived id
unless the caller asked for archived rows, and stamps `archived` from the store
rather than from the dataclass default — the omission that made the sidebar and
`GET /v1/desktop/sessions` disagree with the store about a session a user had
just archived from inside it (QA round 1, Q1).

**The refusal is a sentence, not a status.** The desktop route answers 409 with
`{"code": "session_delete_refused", "message": "<the sentence>"}` — the code is
the machine contract, the sentence names the remedy, and a client that rendered
only the code would have to invent four remedies itself.

## 5. Surfaces

**Where the three commands run, and why that is the frontend's own process.**
`/archive`, `/unarchive` and `/delete` are `_FRONTEND_LOCAL_SLASHES`
(`session/frontend_state.py`), answered by the terminal the user is typing in
rather than routed to the session's runtime owner — `/stop`'s classification, for
`/notifications`' machine-boundary reason plus one of its own:

* The store they write is `config_dir()/archived-sessions.json` and the session
they remove is a directory in `config_dir()/sessions/`. Both are THIS machine's,
and so is the sidebar and picker that render the result. Routed, a follower
attached to a runtime on another host would archive a conversation on the
runtime's machine while the receipt promised the sidebar on screen.
* **The conversation a user wants to delete is the one they are in, and by
  construction it has a live owner.** An attached viewer's owner holds both
  `.session.pid` and `.execution-lease`, so the owner-side delete is refused by
the same guard it applies to every other session. The reachable flow is
`/stop` — which releases both markers — then `/delete`, and that only works if
`/delete` is answered here: a stopped facade's pre-route answer ("this session
was stopped; /resume …") intercepts every *routed* command, so the remedy the
refusal sentence names could not be carried out in the terminal that printed it
(review round 1, MAJOR-1). The refusal sentence is unchanged and still accurate;
what changed is that carrying it out now works.

* **TUI.** `/archive` (acts on the current session, receipt names the way back),
  `/unarchive` (offered ONLY while the current session is archived, filtered out
  of the suggestion list; typed when it does not apply it answers with a sentence
  rather than silence), `/delete` (typed confirmation: bare `/delete` is a
  rehearsal that reports exactly what the real one would remove, `/delete yes`
  performs it; the picker paints the row as dangerous and one Enter only fills
  the word). A successful delete of the current session lands the app on a fresh
  conversation, because the one it was standing on no longer exists — reachable
  from an attached viewer as `/stop` then `/delete yes`, and from any viewer whose
  session is not held by a live owner directly. A viewer attached to an owner
  running OLDER code advertises these three as `authoritative_session` from a
  capability snapshot taken when the socket opened; the frontend's own
  classification wins over that advertisement, so the local answer runs either
  way.
* **`/resume` picker.** Archived rows are excluded from the list and from search;
  an `Archived (N)` toggle at the top of the list pane reveals them, is
  **clickable** as well as reachable by `ctrl+a`, and is drawn **only when the
  store actually holds an archived conversation**. Revealed rows carry an
  `[archived]` mark, and the column's cells are reserved for the whole result set
  so the mark does not rag the name column. The reveal filters on the row's NAME
  and ID only, never its body — the picker's filter was never a body search for
  any session (the body digest index is built for the search surfaces), so the
  toggle reveals a POPULATION rather than changing how filtering works. The
  desktop search, which a user reaches when searching textually, DOES match body
  text through `include_archived` (QA round 1, Q2).
* **Desktop.** `GET /v1/desktop/sessions?include_archived=`,
  `GET /v1/desktop/sessions/search?include_archived=`, `POST .../{id}/archive`
  (desired state, receipt-free, idempotent) and `DELETE .../{id}` with a required
  `{"confirmed": true}` body. Capability keys `session_archive: 1` and
  `session_delete: 1` — separate keys, never a `session_catalogue` bump: an
  additive row field is not a shape change, and bumping would hide a working
  catalogue from a client that predates this feature.

## 6. What this does NOT do

* **No CLI subcommands.** No `lop sessions archive`/`delete`; the surfaces are
  the TUI and the desktop app. The CLI's `sessions cleanup` still exists and is
  the automatic policy's manual door.
* **No bulk archive.** One conversation per act, because the archive is durable
  state a user has to be able to reason about; a bulk verb needs a selection
  model that does not exist yet.
* **No auto-archive.** Nothing archives on age, inactivity or size. Retention is
  unchanged by this feature, including the fact that an ARCHIVED session is still
  eligible for the automatic sweep — archive hides a conversation from the lists,
  it does not exempt it from the limits a user configured. Exempting them would
  make every configured limit unsatisfiable by archiving, which is the one way to
  fill a disk with a gesture that reads as tidying.
* **No per-row archive in the picker.** The picker reveals archived rows and
  opens them; archiving happens from the session itself (or the desktop app). A
  key that archived an arbitrary row under the cursor is a destructive control in
  a list where every other gesture means "open this".
* **No archive of a session with no transcript.** `/archive` acts on the current
  session and says so when there is nothing saved yet: an id no resume path would
  accept is not a conversation the store can hide.
* **No undo for delete.** It is a removal; the guards and the confirmation are the
  whole of the protection, and nothing is kept on the side.
* **The archive is capped at 200 conversations, and reaching the cap puts the
  OLDEST one back in every list.** `ARCHIVED_LIMIT` bounds the file by dropping
  its oldest entry, so archiving a 201st conversation makes the first one
  reappear in the picker, the sidebar, the desktop catalogue and search. The
  receipt says so at the moment it happens, naming the conversation that came
  back (`session.archived.eviction_clause`), because silence there reads as the
  archive forgetting — and the desktop route cannot say it on the wire: the
  response shape is frozen and carries state, not a report (review round 1,
  MINOR-2).
