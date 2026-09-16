# Design brief: a centralised, cached session-load path

STATUS: **implemented** on `perf/session-load-central-cache`, with one section
that did not survive contact with the tree and one correction to its own
measurements. Read the notes below before using any number in the body.

- **(A), the backward page reader, was already shipped** when this brief was
  written: PR #1109 (`work/backward-page-read`, `d437dfdc6` + `d3d88dec1`)
  merged it on 2026-09-14, with the shared backward iterator
  (`_iter_complete_lines_backward`), the reader's new docstring and the
  forward-oracle differential test (`tests/unit/session/test_transcript.py`)
  all in place. The brief is scoped to `efd48c10a` because the reconnaissance
  ran against a checkout 20+ releases behind `main`, so §1, §3(A) and §7 quote
  the PRE-#1109 forward scan — 1.5-3.9 s page reads and "3-7 whole-file scans
  per switch". **Those are not the before-state of anything shipping**, and (A)
  is not re-claimed here.
- **(B) the process-wide page cache and (C) the one-row metadata read** were
  implemented from this brief, as was a fourth change the brief does not contain:
  **(E) the pool lock's reach**, which the live measurement added — a 25 ms
  request for a 642-byte session answered in 829 ms while the 261 MB session's
  open was in flight, because `DesktopSessions.session` held the pool-wide lock
  across `locate()` and `acquire()`.
- **One number in §2.4 did not survive implementation**: `PAGE_CACHE_BYTES` was
  sized from serialized page sizes, while the same section specifies an
  accounting instrument that charges ~2x more. At 4 MiB accounted the cache
  admitted nothing at all for an ordinary conversation (measured; the shipped
  constant and the reason are in `session/page_cache.py`).
- **(D) lazy `Transcript._entries` remains deferred**, for exactly the reason
  §3(D) gives, and is now the largest cost left on the runtime's cold engage.

**Where the evidence lives**: `docs/evidence/session-load-central-cache/README.md`
(the before/after table with host load averages, the isolated live
reproduction, and the unit-level results). The corrected before-tree
measurements — page reads already ~1-2 ms on `main`, `Transcript(...)`
construction and its metadata read at 0.2-3.0 s — are in that file, not in this
one.


Superseded in part by the STATUS banner above: (A) had already shipped, (B)/(C)
and the pool-lock change shipped from this document, and (D) stays deferred.
Scope: `local-operator` @ `efd48c10a`.

Read with the diagnosis that motivated it (three read-only reconnaissance passes,
summarised in the task that produced this file); this document does not re-derive
those measurements, it builds on them. Every factual claim below is cited to
`file:line` at `efd48c10a` unless marked **unverified**.

Sections: §1 what the load path does today · §2 the design · §3 four decisions ·
§4 invalidation matrix · §5 blast radius and census · §6 what I would not do ·
§7 benchmark plan · §8 risk register · §9 what I ran.

---

## 1. What the load path does today, and what the seam is

The operator's measured cost (taken as given) is that a desktop switch pays a
`through_id`-bounded page plus an unbounded tail page: **~3.9 s of JSON decode on
the 261 MB conversation**, 1.05-1.13 s on the 87-96 MB ones. Four independent code
paths read the same journal with the same forward whole-file scan, and two more
parse the whole journal to answer a one-row metadata question:

| # | surface | today | cacheable? |
|---|---|---|---|
| 1 | `DesktopSessionBridge.snapshot()` → `history(through_id=…)` (`server/utils/desktop_sessions.py:510-535`, `:537-558`) | `asyncio.to_thread(read_transcript_page, …)` | yes |
| 2 | the panel's SSE open frame → `events()` does `await self.snapshot()` **unconditionally** (`:680-690`) | the same read again, when the frontend cursor has not moved between the two requests — which is the normal case for an open, and a different key (still ~1 ms after (A)) when a turn committed in between | yes — same key |
| 3 | the renderer's `reconcileTail` walk (`local-operator-ui/src/renderer/src/shared/hooks/use-canonical-session.ts:830-940`) | `sessions.history` with `before_id`, 1-5 more scans (bounded by `RECONCILE_WALK_MAX_ROWS = 500` at `:374`) | partly (repeats) |
| 4 | TUI `SubagentView` paging (`tui/widgets/subagent_view.py:2334-2341`) | one whole-file scan **per page**, plus a speculative re-look probe | yes |
| 5 | desktop child panel at 1 Hz (`DesktopSessions.child_transcript`, `desktop_sessions.py:1026-1089`) | one whole-file scan **per second per open child** | yes |
| 6 | `Transcript(path).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)` in `DesktopSessions.session().locate()` (`desktop_sessions.py:1207-1222`) and `Transcript(parent_dir).latest_custom_entry(SUBAGENT_ROSTER_CUSTOM_TYPE)` in `_persisted_children` (`:755-807`, call at `:802`) | whole-journal parse for one row | yes (different fix) |
| 7 | the same shape in the TUI: `todo_panel._read_restored_todos` (`tui/widgets/todo_panel.py:272`), `runtime/serving._read_child_todo_snapshot` (`session/runtime/serving.py:196-230`, call at `:202-205`), `mobile/tui_handle.py:1178-1179`, `cli.py:2851-2852` | whole-journal parse for one row | yes |
| 8 | the runtime's own `Transcript.__init__` on cold engage (`session/runtime/serving.py:202`, `mobile/daemon.py:561`) | whole-journal parse — the recorded **1.37 s / 231 MB, 0.34 s / 75 MB** baseline (`docs/design/full-history-audit-window.md:185-191`) | no — see §3(D) |

Rows 1-5 (plus the read behind row 4) are one defect: *the same journal is
forward-parsed from byte zero to answer a question about its tail*. Rows 6-7 are
a second, smaller one: *the whole journal is parsed to answer a one-row question*.

**What "centralise" means, concretely.** Today there are two independent scan
implementations used as load paths — `read_transcript_page` (forward,
`session/transcript.py:361-412`) and `read_replay_suffix` (backward,
`:441-579`) — plus ad-hoc whole-file parses. After this change there are two, and
their division is by *what they answer*, not by *who calls them*:

- `iter_rows_backward(path)` — the backward chunk machinery, factored out of
  `read_replay_suffix` and shared by every reader that walks toward the file
  start. (One implementation; three stop conditions.)
- `read_transcript_page(directory, *, before_id, through_id, limit)` — unchanged
  signature, unchanged contract, backward internals. Every page read in the
  codebase routes here: rows 1-5 above.
- `read_latest_custom_entry(directory, custom_type)` — the one-row metadata
  read. Rows 6-7.
- `load_transcript_page(directory, *, before_id, through_id, limit)` — the async
  façade that owns the process-wide cache and single-flight (§2.4). Every
  *caller* of a page read routes here.
- `read_replay_suffix` — unchanged in behaviour, now built on the shared
  iterator. The attach path (`session/attached.py:3749-3768`) keeps using it.

Nothing else in the load path is re-plumbed. In particular the runtime-side
display window keeps replaying from resident `_entries`
(`session/history_window.py:371,405,439`), and the TUI's switch path keeps
mounting without a disk read (`tui/app.py:11047-11057`). Those are already right.

---

## 2. The design

### 2.1 `session/transcript.py` — the shared backward iterator

```python
#: Backward read granularity, unchanged from read_replay_suffix (see :438).
_SUFFIX_CHUNK_BYTES = 1 << 20

def iter_rows_backward(
    path: Path, *, chunk_bytes: int = _SUFFIX_CHUNK_BYTES
) -> Iterator[TranscriptEntry]:
    """Yield parsed journal rows newest -> oldest, one complete row at a time."""
```

This is a pure factoring of lines `499-573` of `read_replay_suffix`: the same
`pending` buffer discipline (read one chunk, split exactly once when a row
boundary lands inside it, carry the partial head, so a row larger than a chunk —
the 256 KiB bookkeeping fixtures, or the 937 KiB row measured on
`29435655756c` — costs one join and not a re-split per chunk), the same
`errors="replace"` decode, the same per-row `TranscriptEntry.from_json` skip of
malformed rows. What changes is *who owns the stop test*: as a generator it reads
the next chunk only when a caller asks for the next row, so a reader that stops
early stops paying. `read_replay_suffix` becomes a `for entry in
iter_rows_backward(path): … break` loop whose body and stop conditions
(`:538-572`) are copied verbatim.

Two consequences to record rather than discover:

- **`bytes_read` gets tighter, and it is not a contract.** `ReplaySuffix.bytes_read`
  is documented "for the caller's own evidence; not a contract" (`:430-431`). A
  generator can stop mid-chunk, so the reported figure can now be smaller than
  today's. Nothing reads it but benches and tests; `tests/unit/session/
  test_transcript.py:689` asserts only `bytes_read < file size`, which stays true.
- **The iterator takes the path, not a handle.** `read_replay_suffix` records
  `end_of_file` from the handle it opened (`:499-503`) so a concurrent append
  cannot inflate the reading. The iterator keeps that inside itself and the
  caller never sees it; if the caller wants a fingerprint it takes its own
  `stat` (which the page cache does, §2.4).

### 2.2 `session/transcript.py` — the page reader, backward

`read_transcript_page` keeps its signature, its docstring's contract, and its
three special returns; only the scan direction changes.

```python
def read_transcript_page(
    directory: str | Path, *,
    before_id: str | None = None, through_id: str | None = None, limit: int = 100,
) -> TranscriptPage: ...
```

The equivalence obligations, stated exactly (each one is a line of the forward
reader at `:388-412`, and each is exercised by the differential test in §7):

- **`before_id` is EXCLUSIVE and scopes the walk.** The forward reader appends
  rows until it meets `before_id`, then breaks *before* appending (`:397-399`),
  so the page is the last `limit` rows strictly older than the cursor. The
  backward reader must therefore *skip rows newer than the cursor*, skip the
  cursor row itself, and only then start collecting. Getting this wrong is not
  theoretical: my first prototype broke out of the loop at the cursor instead of
  switching to collect, and every `before_id` page came back empty — a silent
  wrong answer that only a differential test sees.
- **`through_id` is INCLUSIVE and out-of-scope rows are skipped.** Forward
  appends then breaks (`:400-403`), so the page ends at the cursor row.
- **`has_more` counts a bounded retention, not a lookahead.** Forward retains at
  most `limit + 1` rows in a `deque` and reports `len(retained) > limit`
  (`:388,412`). A backward reader needs exactly the same rule: collect until it
  has `limit + 1` rows *in scope* or the file start, then `has_more = len(rows) >
  limit`. It must **not** read a `limit + 2`-th row to decide.
- **Missing cursors keep their two distinct shapes.** `through_id` not found →
  `TranscriptPage((), False, True)` (`:404-407`); `before_id` not found → the
  genuine tail page, `reconciled=True` (`:408-410`). Both are returned without a
  second scan.
- **Same precondition errors**, from one shared validator used by both the sync
  reader and the async façade: both cursors set → `ValueError`; `limit < 1` →
  `ValueError`; file absent → `FileNotFoundError` (`:381-387`). Callers depend on
  all three (`desktop_sessions.py:548-553` catches the third;
  `tests/unit/server/test_desktop_sessions.py:325` pins the first;
  `tests/unit/session/test_transcript.py:371` pins the third).

Two divergences from the forward reader that I recommend adopting *deliberately*
and documenting in the docstring, because they are the corrupt-file path and
neither has a caller that can observe the difference on a well-formed journal:

1. **Invalid UTF-8.** Forward is `path.open("r", encoding="utf-8")` → a bad byte
   raises `UnicodeDecodeError` out through `bridge.history` (which catches only
   `FileNotFoundError`) as a 500. Backward decodes with `errors="replace"` and
   then the row fails `json.loads` and is skipped per-row — which is exactly what
   `read_replay_suffix` already does (`:533`), and what the docstring's "malformed
   rows are skipped independently" (`:373-374`) says the reader means.
2. **Duplicate ids.** With an id appearing twice, forward stops at the *oldest*
   occurrence and backward at the *newest*. Ids are message uuids and I measured
   **0 duplicates across 21,454 rows in 3 real journals** (§9), so this is
   unreachable in practice; the backward rule ("the newest row with that id") is
   the one that matches the sentence the cursor actually means, and it should be
   pinned by a test rather than left to be rediscovered.

**The no-compaction fallback, and its bound.** The page reader has no dependency
on compaction at all: a tail page stops after `limit + 1` in-scope rows however
the journal is shaped. The only unbounded case is a cursor at or near byte zero,
where the walk visits every row — bounded by the file, identical to today's cost
for that same page, and it is by construction the *last* page of a walk. The
`read_replay_suffix` fallback is unchanged and stays as its docstring describes
(`:570-572`): a journal with no compaction reads to the start, because pretending
a prefix is the whole history is the one thing that reader must not do.

### 2.3 `session/transcript.py` — the one-row metadata read

```python
def read_latest_custom_entry(directory: str | Path, custom_type: str) -> TranscriptEntry | None:
    """Newest custom row of ``custom_type``, without parsing the journal above it."""

def read_latest_custom(directory: str | Path, custom_type: str) -> dict[str, Any] | None:
    """``Transcript.latest_custom``'s details mapping, single-scan. See :1098-1102."""
```

One implementation, `iter_rows_backward` underneath; the two wrappers differ only
in what they project, mirroring the `latest_custom`/`latest_custom_entry` split
that already exists (`:1098-1115`).

**Stop conditions**, all of them: (a) `directory / TRANSCRIPT_FILENAME` absent →
`None`, matching `Transcript(absent)` reading as empty history — the same
decision, and the same reasoning, as `read_replay_suffix:482-490`; (b) the first
row walking backward with `type == ENTRY_CUSTOM` and
`str(payload.get("custom_type", "")) == custom_type` → that row (the exact
expression `Transcript._index_entry:1060-1061` uses, so a row with no
`custom_type` key is treated identically by both); (c) the file start → `None`.
Malformed rows are skipped per row. `read_latest_custom` reproduces
`dict(entry.payload.get("details", {}))` verbatim, including its behaviour on a
non-mapping `details`, so a caller cannot tell which implementation answered.

No `max_bytes` knob. It would be a second bound beside the honest one (read to
the start when the row genuinely is not there), it would need a policy for what
"not found within budget" *means* to a caller that today gets a real answer, and
today's caller already pays the whole parse — so unbounded-backward is never
worse than what ships. If a caller ever needs a hard ceiling, that is a
follow-up with a named consumer, not a defaulted parameter.

**Why this is not a second scan implementation beside `CustomSnapshotCache`.**
`mobile/durable.py:505-560` also finds newest-wins custom rows; it does so in one
forward pass for *all* of `_TRACKED_CUSTOM_TYPES = {"subagent_roster",
"todo_snapshot"}` (`:82`). Replacing it with N backward searches would be a
regression, so it stays — but it is the reader that must agree with this one, and
both must be "newest wins, malformed rows skipped". Recorded here so the next
reader does not discover two answers to one question; see §6.

### 2.4 `session/page_cache.py` — the new module

New module, not an addition to `transcript.py`, for one reason: `transcript.py`
is a leaf the stdlib-only paths and the TUI both import, and it holds no
process-wide state today. A cache with a single-flight map is state; it belongs
in a module whose name says so. Dependency direction is one-way —
`page_cache → transcript` — so nothing can cycle.

```python
#: LRU bounds. Entries bound the bookkeeping; bytes bound the real cost, which is
#: why both exist: a 100-row page measured 167 KiB on one real conversation and
#: 2.1 MiB on another, with a single row at 937 KiB (§9).
PAGE_CACHE_ENTRIES = 16
PAGE_CACHE_BYTES = 4 * 1024 * 1024
_PAGE_CACHE_ALLOWANCE = 4096

@dataclass(frozen=True)
class PageKey:
    directory: str          # str(Path), the session directory
    inode: int              # st_ino: compact_file's os.replace is a new inode
    size: int               # st_size: an append or a rollback truncate moves it
    before_id: str | None
    through_id: str | None
    limit: int

class TranscriptPageCache:
    def get(self, key: PageKey) -> TranscriptPage | None: ...
    def put(self, key: PageKey, page: TranscriptPage) -> None: ...
    def invalidate(self, directory: str | Path) -> None: ...

async def load_transcript_page(
    directory: str | Path, *,
    before_id: str | None = None, through_id: str | None = None, limit: int = 100,
) -> TranscriptPage: ...
```

**The key is file identity, not a generation counter.** `DurableFoldCache`
(`mobile/durable.py:103-117,185-221`) is the closest precedent and the one to
copy: `_FileFingerprint(inode, size, mtime)` with mtime carried for diagnostics
only, because "a same-inode same-size file cannot have changed regardless of what
its mtime claims". Size is load-bearing here for a second reason the fold cache
does not have: `preserve_mtime` appends deliberately rewind the transcript's
mtime (`transcript.py:935-966`), so mtime alone would serve a stale tail page
after a bookkeeping batch. `size` moves on every append regardless.

**Invalidation by construction, not by notification.** There is no writer
callback, no generation subscription, no `invalidate` on the append path:

- append (size↑, same inode) → a different key; the tail page's old entry is
  simply never asked for again and ages out of the LRU.
- rollback `os.truncate(self.path, previous_size)` (`transcript.py:1010`, same
  inode, shrink) → a different key. If the truncate lands back on a size that is
  still cached, that is the *correct* answer: the rollback restores the exact
  pre-append bytes, so the cached page for that size describes the file that is
  now there.
- `compact_file` → `_replace_file` writes a temp file and `os.replace`s it
  (`transcript.py:1401-1410`) → **new inode** → different key.
- a *different process* appending or compacting → the same three rules, because
  the key is read from the filesystem, not from this process's memory.

That is the whole invalidation design, and it is the reason the key carries
inode+size rather than a per-`Transcript` counter: the desktop server, the TUI
and the mobile daemon are separate processes that can all touch one journal.

**Publication is guarded by a re-stat.** A read that spans a compaction or an
append must not become the cached answer, so the façade takes the stat *before*
the read, reads, stats again, and publishes only when `(st_ino, st_size)` is
unchanged. A torn read is still returned to the caller — that is today's
behaviour and not this change's to fix — but it is never cached.

**Single-flight, and why it is loop-owned.** Two `snapshot()` calls in one switch
do the identical read (rows 1 and 2 in §1). They run on one asyncio loop; the
reads run in `asyncio.to_thread` workers and the GIL serialises their parses
against the loop, so the duplicate is not merely wasted work, it is loop
contention on the sessions that are already slow. The façade therefore keeps a
map `PageKey -> (loop, Task)`, and:

- the *first* caller creates the task and awaits it;
- every *other* caller `await asyncio.shield(task)` — shield because a follower's
  cancellation must not cancel the read the other waiter is depending on;
- the entry is removed by a task done-callback, so leader cancellation cannot
  orphan an entry, and a cancelled leader still publishes for whoever is left;
- an entry whose `loop` is not `asyncio.get_running_loop()` is treated as absent.
  This is not defensive padding: `tests/unit/tui/*` create and tear down loops in
  one process, and awaiting a future bound to a dead loop raises `RuntimeError`.
  Tagging the loop costs three lines and removes a class of intermittent test
  failure.

**No lock beyond that.** The cache dict is mutated only from the loop: `get` in
the façade's own frame, `put` in the leader's `_read_and_publish` after its
`await asyncio.to_thread` returns. Nothing touches it from a worker, so there is
no `threading.Lock` here, and in particular no second lock beside the one the
question already has — `DesktopSessionBridge.warm`'s docblock is the in-tree
statement of that rule (`desktop_sessions.py:597-604`). **The one thing the coder
must not do is call `cache.get`/`put` from inside a `to_thread` closure**; that
would silently introduce the cross-thread mutation this shape exists to avoid.

**Byte accounting needs a fix, not a copy.** `history_window._retained_size`
(`:239-284`) is the repo's instrument for exactly this, and it does **not** work
on `TranscriptPage`: it descends into `BaseModel` (via `__dict__`), `dict` and
`list`/`tuple`/`set`/`frozenset`, but `TranscriptPage` and `TranscriptEntry` are
plain `@dataclass`es (`transcript.py:193-194,346-347`), so the walk stops at the
page object and never sees the rows — the figure would be a few dozen bytes for a
2 MiB page, i.e. an unbounded cache. (`history_window` is unaffected: its pages
hold pydantic `Message`/`CustomMessage`, `harness/types.py:272,436`.) So:

- move the walker to `page_cache.retained_bytes(value, limit)` and add a
  `dataclasses.is_dataclass(item) and not isinstance(item, type)` branch that
  descends through `vars(item).values()`;
- have `history_window` import it from there (`:223` is its only call site);
- add a test that pushes a known page through the instrument and asserts the
  figure exceeds its serialized size — a dead instrument returns a reading, not
  an error (AGENTS.md §Timing), and this one is one missing branch away from
  reporting `2 MiB` as `48 bytes`.

**Failure modes are ordinary.** A page too large to admit is a miss, not an
error. A cache that is cold, wrong-loop, or evicted degrades to exactly today's
behaviour. Nothing on the read path raises because of the cache.

### 2.5 Caller migration

| caller | today | after |
|---|---|---|
| `DesktopSessionBridge.history` (`desktop_sessions.py:537-558`) | `await asyncio.to_thread(read_transcript_page, …)` | `page = await load_transcript_page(…)`; the `FileNotFoundError` arm (`:548-553`) is unchanged |
| `DesktopSessions.child_transcript` (`:1026-1089`) | one `to_thread(read)` closure doing containment + probe + page | containment + `is_dir()` probe stay in one `to_thread`; the page moves to `await load_transcript_page(…)`; the `FileNotFoundError` arm (`:1078-1081`) is unchanged. Keep the `TRANSCRIPT_FILENAME.exists()` pre-check to hold the behavioural delta at zero |
| TUI `SubagentView` load (`subagent_view.py:2326-2341`) | `await asyncio.to_thread(read_transcript_page, directory, before_id=cursor, limit=HISTORY_PAGE_ROWS)` | `await load_transcript_page(directory, before_id=cursor, limit=HISTORY_PAGE_ROWS)` |
| `locate()` (`desktop_sessions.py:1216-1222`) | `Transcript(path).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)` | `read_latest_custom(path, FRONTEND_CHECKPOINT_CUSTOM_TYPE)` — and the `import Transcript` goes away |
| `_persisted_children` (`:797-807`) | `Transcript(parent_dir).latest_custom_entry(SUBAGENT_ROSTER_CUSTOM_TYPE)` | `read_latest_custom_entry(parent_dir, SUBAGENT_ROSTER_CUSTOM_TYPE)`; the `fork_instant` comparison on `entry.ts` is unchanged |
| `todo_panel._read_restored_todos` (`:257-275`) | `Transcript(dir, defer_materialise=True).latest_custom("todo_snapshot")` | `read_latest_custom(dir, TODO_SNAPSHOT_CUSTOM_TYPE)`; **keeps** `defer_materialise`'s guarantee by construction — the new reader never mkdirs |
| `serving._read_child_todo_snapshot` (`:196-230`) | same, plus `transcript.path.is_file()` | `read_latest_custom(dir, …)`; the `is_file()` probe becomes the reader's own absent-journal answer |
| `mobile/tui_handle.py:1178-1179`, `cli.py:2851-2852` | `Transcript(dir).latest_custom*` | the same two new readers |
| `read_replay_suffix` / attach (`attached.py:3749-3768`) | own backward loop | same function, loop body now driven by `iter_rows_backward` |
| runtime display window, TUI switch, `Transcript.__init__` | — | **unchanged** |

The `defer_materialise` point is worth stating because it is a review finding
waiting to happen: `Transcript.__init__` mkdirs its directory
(`transcript.py:608-612`), and `todo_panel`'s docstring (`:257-275`) records a
round-1 defect where opening a child's page created a phantom session directory.
`iter_rows_backward` opens the journal read-only and never touches the
directory; the replacement is strictly safer, and its test asserts no directory
appears.

---

## 3. The four candidate changes

### (A) Backward page reader — **GO**

It is the load-bearing change: it removes the whole-file parse from every page
read in the codebase, including the two that a desktop open pays twice.

Evidence it works: I wrote a throwaway prototype of the reader — with the iterator
inlined, since the factoring of §2.1 is the coder's first step — and diffed it to
`read_transcript_page` on two real journals — **1,632
comparisons, 0 mismatches**, covering tail / `through_id` / `before_id` / both
missing-cursor shapes, at limits 1, 2, 100, 1000, plus 200 random cursors per
session (§9). On the 261 MB conversation the tail page went from **1188 ms to
1.1 ms** with byte-identical rows.

The factoring is the easy half; the *semantics* are where a coder will slip, so
§2.2 states them one by one and §7 turns each into a differential assertion. The
no-compaction fallback is unchanged and bounded (§2.2).

Risk that it lands wrong: a cursor handled with the wrong inclusivity, which is
silent (an off-by-one page, not a crash). Caught by the differential test only —
which is why that test is required, not optional.

### (B) Process-wide decoded-page cache with single-flight — **GO**

Second-order after (A) — it is worth ~10 ms per *open*, not 3.9 s — but it is
what makes the rest of the cost disappear, and it is the part that makes
*switching back* to a conversation instant instead of merely fast:

1. the duplicate `snapshot()` read (rows 1+2) becomes one read;
2. the reconcile walk's pages, which the renderer re-requests and merges by id
   ("Merged even when it is the page we already have",
   `use-canonical-session.ts:911-925`), and the `SubagentView`'s speculative
   re-look probe, become free on the repeat;
3. the 1 Hz child read (row 5) becomes free while the child is idle — it re-stats
   and finds the same `(inode, size)`;
4. a warm switch to a conversation already visited in this process touches no
   disk at all.

Where it lives: `session/page_cache.py`, process-wide module state, one LRU of
16 entries / 4 MiB accounted bytes. `DesktopSessions` is one instance per app
(`routes/desktop_sessions.py:225-230` caches it on `app.state`) in a single
uvicorn process (`cli.py:4030-4038` runs one), so a module-level cache is
coherent; the TUI is a different process and gets its own, which is correct —
they hold different journals' worth of state and neither can see the other's.

Why not per-`Transcript`: the TUI's page reads hold no `Transcript` at all
(`subagent_view.py:2334`), and the desktop bridge's history read is deliberately
independent of the runtime. The cache must sit at the *page read*, which is
exactly where the seam is.

Byte-bounding is mandatory, not a nicety: a 100-row page measured 2.1 MiB and a
500-row page 7.1 MiB on real sessions, with a single row at 937 KiB (§9). And the
instrument that bounds it needs the dataclass fix in §2.4 or it reports a 2 MiB
page as a few dozen bytes.

### (C) Bounded backward search for one custom row — **GO**

The smallest of the three and the one with the widest caller list (7 call sites
across the server, the TUI, mobile and the CLI). The stop condition already
exists in this codebase: `read_replay_suffix`'s `checkpoint_type` parameter
(`:470-474,555-561`) is the same scan, the same "first hit walking backward wins",
and the same reason for taking the type as a parameter rather than importing the
constant (`:472-474`: `frontend_state` reaches the TUI, and `transcript.py` must
stay a leaf). The new function is that condition, alone, without the compaction
and cursor conditions beside it — so the chunk machinery is shared and only the
stop test differs, which is the difference the brief asks to be explicit about.

Boundedness, honestly: the common case is a handful of rows (a checkpoint or a
todo snapshot is re-appended on every change, so the newest is near the tail).
The worst case is a row that was written once near the byte zero of a 261 MB
journal — a legacy `subagent_roster` entry is exactly that shape — and there the
walk reaches the file start, which costs what today's whole-file parse costs.
Never worse, and the answer is correct rather than a bounded "unknown".

### (D) Lazy `Transcript._entries` — **DEFER**, with the reason recorded

This is the largest remaining item (**1.37 s on the 231 MB journal, 0.34 s on the
75 MB one**, `docs/design/full-history-audit-window.md:185-191`) and the widest
blast radius in this document. I recommend **not** doing it in this change, and
the honest reason is not "it is big" but *where the hazard sits*.

What `__init__` derives (`transcript.py:643-652` → `_index_entry:1054-1063`), and
every reader of each:

| derived at construction | read through | readers |
|---|---|---|
| `_entries` | `entries`, `build_llm_history`, `pending_prunes`, `usages_since_compaction`, `search_spend_rows`, `reclaimable_bytes`, `compact_file`, `fork_snapshot` (`:1094-1390`) | + `history_window.py:371,405,439` |
| `_entry_ids` | `has_entry` (`:1065`) | `session.py:2879,2938,6326,6465,8220,8289,8440,9139,11045,11072` |
| `_latest_by_type` | `latest_entry` (`:1069`) | `session.py:1940,3522,9547,10198`, `session_factory.py:686,750` |
| `_latest_custom_entries` | `latest_custom_entry`/`latest_custom` (`:1098-1115`) | `session.py` ×10, `attention.py:476-477`, `frontend_state.py:3571`, `serving.py:205` |
| `_latest_user` | `latest_user_entry` (`:1073`) | `session_factory.py:685,1479` |
| `_admitted_command_ids` | `has_admitted_command` (`:1076`) | `session.py:3635-3637`, `serving.py:2157` |

For the pure readers, "load before returning" is semantically transparent. For
the **mutators it is not**, and one of them is data loss:

- `_write_entries`' rebuild branch writes `(*self._entries, *entries)`
  (`:1023-1025`). If `_entries` is an unmaterialised empty list, that branch
  **replaces the journal with the new batch alone** — a 261 MB conversation
  destroyed by a single append that raced a `rebuild`.
- `_commit` appends to `_entries` and publishes derived indexes afterwards
  (`:848-850`); an unmaterialised list makes `has_entry` answer `False` for a
  durable row, which is how the same message gets admitted twice
  (`session.py:2879`).
- so every mutating path must force a load *first* — and `_ensure_loaded` costs
  1.37 s, while `_commit`'s write runs in a worker thread under `self._lock` and
  the sync readers are called straight from the loop (`session.py:2879`,
  `attention.py:476`). A `threading.Lock`-guarded load called from a loop thread
  is precisely the freeze this repo has already fought once (#401, AGENTS.md
  §Environment), and #401's own reproduction is the `/resume` path.

This is the point that makes laziness a *contract* change rather than an
optimisation: the load must become `await`-only, and ~30 call sites plus every
`hasattr(transcript, …)` duck-type check (`session_factory.py:684-686,749-750`)
have to be answered for.

**After (A)+(B)+(C), though, that 1.37 s is almost alone on the load path.** The
audit of §1 leaves exactly two whole-journal parses that a user gesture reaches:
the runtime's own cold engage (row 8, where the journal is genuinely needed —
`build_llm_history` replays from `_entries`, and the append path needs the
rollback boundary), and `harness/comms.py:1187`'s hub peek (which renders *all*
rows to number the steps, then pages 50 at a time — a candidate for the same
bounded-tail treatment). Everything else in rows 1-7 is a page read or a one-row
read, and (A)+(B)+(C) remove them.

**Follow-up shape, recorded now so the next session does not re-derive it:** don't
make `_entries` lazy; make the *runtime's boot* not need the whole parse, by
windowed replay through the machinery that already exists in this module
(`context_cut_index:1621`, `first_message_index:1730`, `audit_slice:1744`,
`replay_entries:1830` — landed by `docs/design/full-history-audit-window.md`).
Prerequisites, in order: (1) a windowed `build_llm_history` with a proven
equivalence test against the whole-file replay; (2) an explicit
`await transcript.materialise()` boundary the append path calls before it
computes a rollback size; (3) the duck-type `hasattr` call sites converted so
they cannot be answered by a partially-loaded object. Until (1) lands there is no
safe place to put the laziness, which is why this is a deferral and not a
"later, smaller version".

---

## 4. Invalidation matrix

Per cache, with the stale-data consequence stated. Only the first is new.

| cache | key | invalidated by | failure mode if it ever serves stale data |
|---|---|---|---|
| **`TranscriptPageCache`** (new, process-wide in `session/page_cache.py`) | `(str(session dir), st_ino, st_size, before_id, through_id, limit)` | append (size↑) · rollback `os.truncate` (size↓, `transcript.py:1010`) · `compact_file` → `os.replace` (new inode, `:1401-1410`) · LRU eviction by entries or accounted bytes · `invalidate(dir)` · a different process's appends and compactions, read from the filesystem rather than remembered | a **wrong page of the same conversation** is served — no error, no crash. Reachable only if `(inode, size)` names a file with different content: inode recycling onto an identical byte length *and* the same cursor ids, or the writer's own rollback racing a second writer — a hazard `transcript.py:958-970` already documents as out of scope for the append path. The residual is accepted on the `DurableFoldCache` precedent (`mobile/durable.py:107-113`) and is strictly narrower than it, because the tail page's key moves on every append |
| **single-flight inflight map** (same module) | `PageKey` + the loop identity | the task settling (done-callback) or raising; a foreign loop is treated as absent | a caller waits on a task that never settles → the request hangs. Bounded by the reader's own I/O (there is no timeout today either) and by the fact that the leader's exception is propagated to every waiter |
| `_DisplayWindowCache` (unchanged) | generation-flavoured key incl. `_history_generation` (`history_window.py:204-236`) | `_index_entry` bumping the generation on compaction/prune, `compact_file` clearing it (`transcript.py:1054-1058,1377-1378`) | not this change's, and unchanged |
| `_audit_hoisted_cache` (unchanged) | `_history_generation` (`transcript.py:619-624`, `history_window.py:387-408`) | same generation bumps | unchanged |
| `DurableFoldCache` (unchanged) | `_FileFingerprint` per directory (`mobile/durable.py:103-117,185-221`) | new inode, shrink, or a failed incremental tail | unchanged; the precedent this design copies |
| `catalog._ROW_CACHE` (unchanged) | `session_id -> ((mtime, size), row)` (`catalog.py:452,473-542`) | rebuilt every poll | unchanged |

---

## 5. Blast radius and census

**Production callers whose behaviour changes** (all of §2.5): `transcript.py`
(`read_transcript_page` internals, `read_replay_suffix` internals, two new
functions), `server/utils/desktop_sessions.py` (5 sites), `tui/widgets/
subagent_view.py` (1 site + the module-level import at `:83`),
`tui/widgets/todo_panel.py`, `session/runtime/serving.py`, `mobile/tui_handle.py`,
`cli.py`, `session/history_window.py` (the `_retained_size` import), plus the new
`session/page_cache.py`.

**Unaffected on purpose:** `session/runtime/serving.py:202` and
`mobile/daemon.py:561` still build an eager `Transcript`; `session/attached.py`
keeps its `full_required` (`:3670`) and cut-off-journal (`:4728`) whole-file
paths; `harness/comms.py:1187` keeps its peek parse; the display window, the TUI
switch path and the reconcile walk do not change.

**Tests that pin the behaviour being replaced** — these must be green *unmodified*
except where noted:

- `tests/unit/session/test_transcript.py:329,352,368` — the three page-reader
  contracts (backward pages reach start with stable ids and skip malformed rows;
  a cursor removed by replacement reconciles; an absent journal raises without
  ever creating the file). These are the regression guard for (A) and they should
  not need a single edit.
- `tests/unit/session/test_transcript.py:678,707,721` — the suffix reader's
  equivalence shapes, which is what proves the shared-iterator refactor is
  behaviour-preserving. Note `:689` asserts `bytes_read < file size`; the tighter
  figure stays true.
- `tests/unit/server/test_desktop_sessions.py:316-325` (both-cursors
  `ValueError`, streaming page, missing cursor, `before_id` page) and
  `:1574-1593` (child page rows verbatim).
- `tests/e2e/test_desktop_sessions.py:872,938` — the assembled `/resume` and
  child-page path through the real app.
- `tests/unit/session/test_transcript.py:385` (`test_latest_custom_backward_scan`)
  — pins the semantics (C) must reproduce.

**Tests that will need an edit, and why that is expected.** `subagent_view` will
call `load_transcript_page`, so the tests that monkeypatch the view module's
`read_transcript_page` symbol must point at the new name —
`tests/unit/tui/test_subagent_view.py:1494-1502,1670-1690,1787,1992-2002,
2478-2481,4624-4631,4718-4751,4851` and `tests/unit/tui/test_page_back_latch.py:
66-72,130-136` — **8 patch sites plus a dozen references** in the first file, 2
plus 2 in the second (counted with `grep -c 'setattr(.*read_transcript_page\|read_transcript_page = '`). They assert the *view's* degradation behaviour
(slow read, failing read, page-back latch, "exactly one read") through a
module-level seam that the test file itself documents as deliberately module
state (`test_subagent_view.py:1687-1690`); the change is mechanical and the
assertions keep their intent. `tests/unit/tui/test_resume_render.py:531` mentions
the function in a docstring only.

**Docs and contracts to update:** `transcript.py`'s own docblocks for
`read_transcript_page` (`:368-377` — "this costs a sequential disk pass for older
pages" is the sentence that becomes false), `read_replay_suffix` (its
`bytes_read` note, `:430-431`) and the new functions; the final version of this
file in `docs/design/` (the coder commits it, per the task). No wire format
changes: `bridge.history` and `child_transcript` keep emitting
`{entries, has_more, cursor_missing}` (`desktop_sessions.py:554-558,1082-1087`),
and no token or cursor shape moves. `docs/design/full-history-audit-window.md`
mentions `read_transcript_page` but is a historical record with a STATUS banner
convention (`history-fold-convergence.md:3-15`) — leave it alone.

**Release class: `patch`.** Performance/reliability, no new user-facing
capability — AGENTS.md §Versioning's default. The PR body should carry
`Release: patch — session switching stops paying a whole-journal JSON parse`.

---

## 6. What I would NOT do, and why

- **A byte-offset index sidecar** (an offsets file beside the journal). Offsets
  are already rejected as cursors in this codebase for a stated reason —
  compaction replaces the file atomically so "offsets become lies while IDs
  remain meaningful" (`transcript.py:346-353`) — and an index file would add a
  second source of truth to keep honest across three processes, plus the
  crash-recovery story `search_index.py:297-324` already shows is real. Backward
  chunk reads cost ~1 ms and need no such contract.
- **Renderer virtualisation, or touching the reconcile walk.** The walk is
  bounded at 500 rows already (`use-canonical-session.ts:374,852`) and lives in a
  different repository behind the desktop wire contract. Making pages cheap is
  the server's job; changing what the walk asks for is a separate design with its
  own review.
- **Changing any wire shape** — `TranscriptPage`, the `/history` envelope, the
  signed display-window tokens, or the SSE frame order. None of the three
  candidates needs it, and each would need its own caller census across two
  repos.
- **Making `read_replay_suffix` delegate to `read_latest_custom_entry`.** It needs
  the suffix rows *and* the checkpoint from one pass (`attached.py:3754-3768`);
  calling the per-type reader as well would be a second read of the same bytes.
  Sharing `iter_rows_backward` is the correct shared surface; sharing the stop
  condition is not.
- **Replacing `CustomSnapshotCache._scan`** (`mobile/durable.py:505-560`). It
  answers for *all* tracked custom types in one forward pass; N backward searches
  would be worse. Leave it, and keep the "newest wins, malformed rows skipped"
  agreement explicit in both docstrings.
- **A `max_bytes` ceiling on the metadata reader, an on-disk page cache, or a new
  dependency.** Each adds a policy question (what does a bounded "not found"
  mean?) or a lifecycle (eviction, versioning, crash recovery) for a path whose
  worst case already equals today's cost.
- **Making `_entries` lazy** — §3(D), deferred with its reason and its follow-up
  shape recorded. Do not do "a small version of it" in this change; the small
  version is `_write_entries:1023-1025` truncating a journal.
- **The `/resume` picker's synchronous open** (`tui/app.py:12469-12512`: an
  uncapped `recent_session_rows(config_dir(), limit=None)` plus a full-store
  `build_index` over 4,552 ids). Real, on the TUI's slow path, and out of scope:
  it is a directory *scan* plus a digest index, not a session-page read, and
  `build_index` is already incremental on `[size, mtime, title_mtime]`
  (`search_index.py:297-304`). Record it as its own follow-up rather than
  smuggling it into this seam.
- **A second lock, a generation subscription, or an invalidation callback on the
  append path.** The fingerprint is the invalidation; anything notifying the
  cache would be a mechanism beside the one that already decides
  (`desktop_sessions.py:597-604`).

---

## 7. Benchmark plan

**The table a reviewer reads.** One row per (session, operation), with a
structural column that is identical on an idle laptop and a wedged CI runner
(AGENTS.md §Timing: "prefer a structural invariant to a numeric one"; the counts
are the fact, the milliseconds are weather). My own probe's numbers are filled in
where measured; the coder fills the rest.

| session | size | operation | before | after (cold) | after (warm) | rows decoded |
|---|---|---|---|---|---|---|
| `bda7b76d34e0` | 261 MB | tail page, limit 100 | 1188 ms *(probe)* / 1940 ms *(recorded median)* | 1.1 ms *(probe)* | ~0 | ≤ 101 |
| `bda7b76d34e0` | 261 MB | `through_id` page, limit 100 | 1954 ms *(recorded)* | — | — | ≤ 101 + rows after the cursor |
| `bda7b76d34e0` | 261 MB | open = snapshot + SSE snapshot | ~3.9 s *(recorded)* | 1.1 ms | ~0 | ≤ 202 |
| `9f8e5b652ac7` | 96 MB | tail page, limit 100 | 698 ms *(recorded)* | — | — | ≤ 101 |
| `29435655756c` | 32 MB | tail page, limit 500 | — | — | — | ≤ 501 |
| `bda7b76d34e0` | 261 MB | `locate()` metadata read | 1370 ms *(recorded)* | — | — | rows until the newest checkpoint |
| `03f18d75b736` | 9 MB | reconcile walk, 5 pages | — | — | — | ≤ 505 |

**Commands.** A new read-only harness, because the existing switch harness
deliberately uses no live sessions (`scripts/bench_session_switch.py:20-24`) and
this matrix must run against the operator's real store, as the diagnosis did:

```sh
# baseline on this tree, then the same harness against the pre-change tree
.venv/bin/python scripts/bench_session_page.py --output /tmp/page-bench/baseline.json
.venv/bin/python scripts/bench_session_page.py --source-root /tmp/lo-bench-base \
    --output /tmp/page-bench/ab.json
```

Follow `bench_session_switch.py`'s conventions exactly: two worker
subprocesses, one per tree, alternating group by group with the order flipped
each round so before/after share the same load weather; `probe_isolation`
before the package import; `CMUX_*` scrubbed; provenance in the JSON;
`--store` defaulting to `~/.local-operator/sessions` and refusing to write
anywhere inside a session directory. The A/B works because the *called symbol*
(`read_transcript_page`) exists in both trees — one is forward, one is backward.
The cache has no "before" tree, so its row is measured within the after tree
only (cold vs warm), which is the honest comparison.

Sibling harnesses, for the claims they already own and are **not** re-run here:
`scripts/bench_resume_picker.py` (picker open),
`scripts/display_cache_performance.py` (runtime display pages),
`local-operator-ui/scripts/session-switch-latency.mjs` (click-to-usable frame).

**Unit tests, each with the failure it is proven able to catch.** The house rule
is that a test which cannot go red is worse than no test (AGENTS.md §Timing), so
each of these names the mutation that must break it:

1. `test_backward_page_decodes_at_most_limit_plus_one_rows` — a counting spy on
   `TranscriptEntry.from_json` (a structural spy, the
   `test_store_maintenance_callbacks_run_off_the_event_loop_thread` shape) over a
   journal with thousands of rows **before** the page window; assert the decode
   count is `≤ limit + 1` for the tail and `≤ limit + 1 + rows-after-cursor` for
   `through_id`. **Must fail** on the forward implementation (count becomes the
   whole file). This is the assertion that survives a loaded box, and it is the
   one CI should gate on.
2. `test_backward_page_is_row_for_row_equivalent_to_the_documented_contract` — a
   differential test against a small literal forward reference kept in the test
   file (not against the implementation it is replacing, which is gone), over
   the cross product: `{tail, through_id, before_id} × {missing cursor} ×
   {limit 1, 2, 100} × {empty file, no trailing newline, a row larger than a
   chunk, a malformed line, two ids that repeat}`. **Must fail** when
   `before_id`'s exclusivity or `through_id`'s inclusivity is inverted, and when
   `has_more` reads a `limit + 2`-th row.
3. `test_page_cache_is_keyed_on_file_identity` — miss, hit, then append → miss;
   `compact_file` → miss; rollback truncate → miss (or the correct re-hit at the
   restored size). **Must fail** if the key drops `size` or `inode`.
4. `test_two_concurrent_loads_issue_one_read` — two `load_transcript_page` calls
   for one key with a counting spy; assert exactly one read and identical return
   values. Wait on the completion events the code publishes, never on a clock;
   the follower's await is the event. **Must fail** if single-flight is removed.
5. `test_retained_bytes_sees_dataclass_payloads` — the instrument check: a known
   page whose serialized size is measured, asserted to be *below* the accounted
   figure. **Must fail** if the dataclass branch is dropped (it currently would,
   which is the point of writing it).
6. `test_read_latest_custom_matches_the_resident_transcript` — differential
   against `Transcript(dir).latest_custom(type)` on a journal carrying several
   copies of that type plus other custom types interleaved; plus absent journal →
   `None` and no directory created. **Must fail** if the scan takes the oldest
   match or stops on a different type.
7. Unchanged and unedited: the existing tests listed in §5.

House rules that apply to all of them: no `sleep`-then-assert; assert on rows
decoded and on thread identity rather than on elapsed milliseconds; if a numeric
bound is ever needed, measure CPU with `time.thread_time()`, never wall time.

---

## 8. Risk register

| # | risk | evidence that would catch it |
|---|---|---|
| 1 | **A cursor's inclusivity is inverted** → a page off by one row, silently, on one of the two cursors | test 2's differential matrix; it is the only instrument that sees it (I hit exactly this class in my own prototype: every `before_id` page came back empty) |
| 2 | **A row spanning a chunk boundary or an unterminated final line is dropped** → missing rows deep in a long conversation | test 2's "row larger than a chunk" and "no trailing newline" shapes; `read_replay_suffix`'s existing chunk-boundary fixtures (`test_transcript.py:642-676`) |
| 3 | **The cache serves a page from a replaced or appended journal** → a wrong page of the right conversation | inode+size in the key; the post-read re-stat before publish; test 3; the residual (inode recycling, concurrent second writer) is named in §4 and accepted on the `DurableFoldCache` precedent |
| 4 | **A missing `dataclass` branch leaves the byte bound inert** → an unbounded process cache on a long-lived server | test 5, which is written to fail against today's walker |
| 5 | **The façade is awaited on a different loop than the leader's** → `RuntimeError` in tests, intermittently | the loop tag + a test that runs two sequential `asyncio.run()` loads |
| 6 | **The `subagent_view` seam move breaks the monkeypatch sites** → a burst of TUI test failures that look like a regression in the view | §5's list; do it as one mechanical commit and re-run `tests/unit/tui` with `env -u NO_COLOR TERM=xterm-256color` |
| 7 | **The shared-iterator refactor changes `read_replay_suffix`** — a subtly different suffix on a compacted journal, i.e. a resumed context that differs from the live one | the six existing suffix tests, unmodified and green; the no-compaction case must still read to the start (`test_transcript.py:690-691`) |
| 8 | **A caller starts mutating the cache from a worker thread** (the natural mistake when a `to_thread` closure is "just wrapped around it") → a rare, unreproducible wrong page | the loop-owned rule in §2.4; the single-flight and identity tests; and a review question on any new `to_thread` in the diff |
| 9 | **The perf win is measured on a loaded box and cannot be reproduced** → the merged change is judged on a number that was noise | the structural column (rows decoded) is the gate; wall figures are reported with load average and are never asserted on |
| 10 | **The desktop server holds a different `DesktopSessions` per request** and a per-instance cache would silently never hit | verified against the code: `host(request)` caches the pool on `app.state` (`routes/desktop_sessions.py:225-230`); the cache is module-level regardless, which makes the question moot |

---

## 9. What I ran, and what remains open

**Read-only, against the operator's live store**, with
`~/local-operator/.venv/bin/python`, at load average 188 on 14 cores:

- `read_transcript_page` vs a throwaway prototype of the backward reader
  (`/tmp/backward_page_probe.py`, scratch, not committed — the iterator is inlined
  in it because it does not exist in the tree yet), on sessions `03f18d75b736`
  (9 MB / 1,333 rows) and `29435655756c` (32 MB / 5,544 rows): **1,632
  comparisons, 0 mismatches**, across tail / `through_id` / `before_id` /
  `through_id`-missing / `before_id`-missing, limits 1, 2, 100, 1000, and 200
  random cursors per session.
- On `bda7b76d34e0` (261 MB), single-shot tail page limit 100: **forward
  1188 ms, backward 1.1 ms, identical row ids**. (The recorded baseline's median
  of three is 1940 ms; I report a single shot and do not treat the difference as
  a claim.)
- Page sizes on four real conversations, limit 100: 167 KiB, 312 KiB, 2.0 MiB,
  2.1 MiB; limit 500: 1.4-7.1 MiB; largest single row 937 KiB. This is what sizes
  `PAGE_CACHE_BYTES`.
- Id uniqueness: **0 duplicate ids across 21,454 rows** in three journals — the
  evidence behind §2.2's duplicate-id note.

**Open, and deliberately not settled here:**

- **Whether (B)'s cache earns its keep after (A)** is a hit-rate question the
  benchmark answers, not this document: the duplicate read is ~1 ms on the
  261 MB conversation once (A) lands, so the case rests on repeat opens and on
  the 1 Hz child read. If the measured hit rate on a normal session-switching
  session is near zero, drop (B) from the change and keep (A)+(C) — the seam
  (`load_transcript_page`) can stay as a thin pass-through with the cache added
  later without touching a caller.
- **`harness/comms.py:1187`'s peek parse** (whole journal to number steps) is the
  same defect shape one layer away; I have not designed it. It is a candidate for
  a bounded tail read, and it is out of this change.
- **The runtime's own cold engage** is the remaining 1.37 s. §3(D) records why it
  is deferred and what would settle it: a windowed `build_llm_history` with a
  proven equivalence test.
