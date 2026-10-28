"""Immutable, witnessed projections of the canonical transcript index.

JSON remains the commit point and the derivation authority. SQLite is only an
additive read model: no WAL, repair, lazy event-loop queries or partial indexes
masquerading as TranscriptIndex.

TWO FILES PER PUBLICATION, and the split is the point:

``<token>.sqlite``
    immutable; a pure function of one derived index. It is built OUTSIDE the
    writer's publish lock because its build is the expensive part and a scan
    must not serialize every naming patch behind it.

``<token>.naming.json``
    tiny and rewritten atomically on every publish. ``naming`` is mutable
    metadata (a ``patch_naming`` can land at any time), so keeping it out of the
    companion is what lets a naming-only write reuse the companion instead of
    rewriting its multi-megabyte body — the growth this cache previously caused.

Readers materialize bounded data in the caller's worker hop and reject any
publication whose canonical identity, journal signature or selection
completeness does not hold. Unsupported caches keep the existing fallback, and
nothing here may turn a previously-ready answer into ``building``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_operator.session import transcript_index as ti

#: Bumped whenever the companion's tables/semantics change. An older companion
#: is then *unsupported* (fallback), never misread.
SCHEMA = 2
PROJECTION = 1

#: How long a relay link thread may wait for one bounded projection. The page
#: read's own bound is 60 s and a MISSING optimization must never spend it: the
#: original rederive path is admitted immediately after this expires.
READ_MODEL_WAIT_S = 2.0

#: Publications kept per session: the referenced one plus this many predecessors.
#: Open readers hold FDs, so reclaiming a file a request already opened is safe;
#: a reader that has only a PATH re-reads the canonical head and retries.
_RETAIN_PREVIOUS = 2

#: Orphan session directories are only reclaimed when nothing in them is newer
#: than this, so a session's first publication is never swept mid-write.
_ORPHAN_GRACE_S = 3600.0

logger = logging.getLogger(__name__)
# Per-process bounded background work only; no full index retained after completion.
_MIGRATIONS: dict[tuple[str, str], asyncio.Task[None]] = {}
#: Canonical identity whose companion publication already failed. Remembering it
#: is what stops an exact HIT from re-attempting a doomed build on every open.
_FAILED: dict[tuple[str, str], tuple[int, int, int]] = {}
#: (canonical identity, token, naming identity, checkpoints, naming) per session.
_MANIFEST_MEMO: dict[tuple[str, str], tuple[Any, ...]] = {}
_MEMO_LOCK = threading.Lock()
_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_SID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


@dataclass(frozen=True)
class Result:
    state: str
    value: Any = None
    reason: str = ""


@dataclass(frozen=True)
class Publication:
    root: Path
    sid: str
    header: dict[str, Any]
    identity: tuple[int, int, int]
    built_at: float
    path: Path

    @property
    def naming_path(self) -> Path:
        return self.path.with_suffix(".naming.json")

    @property
    def signature(self) -> tuple[int, int, float]:
        sig = self.header["sig"]
        return sig["inode"], sig["size"], sig["mtime"]


@dataclass(frozen=True)
class FrameSummary:
    """Only tail-hunt capabilities, never a complete transcript index."""

    scan: ti.ScanState
    runs: list[ti.RunRecord]
    source: tuple[int, int, float]


@dataclass(frozen=True)
class FrameSelection:
    """Bounded source records for the existing pure frame/order derivation."""

    checkpoints: list[ti.Checkpoint]
    runs: list[ti.RunRecord]
    messages: list[ti.MessageDoc]
    journal_rows: dict[str, int]
    journal_generation: str


@dataclass(frozen=True)
class FrameFacts:
    runs: list[dict[str, Any]]
    rows: dict[str, int]
    ordinals: dict[str, int]
    openers: dict[str, tuple[str, int]]
    missing_ids: list[str]
    generation: str
    source: tuple[int, int, float]
    summary: FrameSummary


def _identity(st: os.stat_result) -> tuple[int, int, int]:
    return st.st_ino, st.st_size, st.st_mtime_ns


def _directory(root: str | Path) -> Path:
    return ti._index_dir(root) / "read-models"


def _path(root: Path, sid: str, token: str) -> Path:
    if not _SID.fullmatch(sid) or not _TOKEN.fullmatch(token):
        raise ValueError("unsafe read-model identity")
    directory = _directory(root)
    path = directory / sid / f"{token}.sqlite"
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("read-model path escapes cache directory")
    return path


def header(root: str | Path, sid: str) -> Result:
    """Read only the ordered head, binding built_at and token to the same FD."""
    if not _SID.fullmatch(sid):
        return Result("unsupported")
    try:
        with ti.index_path(root, sid).open("rb") as handle:
            st = os.fstat(handle.fileno())
            text = handle.read(ti._RESUME_HEAD_BYTES).decode("utf-8", errors="ignore")
        decoder = json.JSONDecoder()
        cursor = 1
        values = {}
        if not text.startswith("{"):
            return Result("unsupported")
        for expected in (*ti._HEAD_KEYS, "read_model"):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            key, cursor = decoder.raw_decode(text, cursor)
            if key != expected:
                return Result("unsupported")
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if text[cursor] != ":":
                return Result("corrupt")
            cursor += 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            values[key], cursor = decoder.raw_decode(text, cursor)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if text[cursor] != ",":
                return Result("corrupt")
            cursor += 1
        marker = values["read_model"]
        if (
            values["version"] != ti.TRANSCRIPT_INDEX_VERSION
            or not isinstance(marker, dict)
            or marker.get("schema") != SCHEMA
            or not isinstance(marker.get("publication_id"), str)
        ):
            return Result("unsupported")
        publication = Publication(
            Path(root),
            sid,
            values,
            _identity(st),
            st.st_mtime,
            _path(Path(root), sid, marker["publication_id"]),
        )
        # These fields attest the actual journal scan, never a fresh guessed stat.
        sig = publication.signature
        if (
            type(sig[0]) is not int
            or type(sig[1]) is not int
            or not isinstance(sig[2], (int, float))
        ):
            return Result("corrupt")
        return Result("ready", publication)
    except FileNotFoundError:
        return Result("missing")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return Result("unsupported")


def _current(pub: Publication) -> bool:
    try:
        return _identity(
            ti.index_path(pub.root, pub.sid).stat()
        ) == pub.identity and ti._sig_matches(
            pub.header["sig"], ti._journal_path(pub.root, pub.sid).stat()
        )
    except OSError:
        return False


def failed(root: str | Path, sid: str, identity: tuple[int, int, int] | None) -> bool:
    """Whether this exact canonical document already failed to publish (R3)."""
    if identity is None:
        return False
    return _FAILED.get((str(root), sid)) == identity


def remember_failure(root: str | Path, sid: str, identity: tuple[int, int, int] | None) -> None:
    """Remember ONE failed canonical identity per session, bounded across sessions.

    A torn journal tail or an unwritable cache is not transient within a
    publication: retrying it on every open rewrites a multi-MB document each
    time, which also moves the cache mtime and disables the resident fast path.
    The next successful derivation carries a new identity and is tried normally.
    """
    if identity is None:
        return
    key = (str(root), sid)
    if key not in _FAILED and len(_FAILED) >= 32:
        _FAILED.pop(next(iter(_FAILED)))
    _FAILED[key] = identity


def reuse(root: str | Path, sid: str, index: ti.TranscriptIndex) -> dict[str, Any] | None:
    """The companion a non-derivation write may keep sharing.

    Exact HITs and naming-only patches re-publish the SAME derived facts; minting
    another multi-megabyte companion for them is the disk growth this cache used
    to cause. The token must still name a readable companion.
    """
    token = index.read_model_token
    if not token:
        return None
    try:
        # ONE stat, deliberately: this runs on the naming path, where opening
        # SQLite would put disk I/O back on the event loop. A companion that is
        # present but unusable is caught by the READER (``corrupt``/``unsupported``
        # and the existing fallback), which is the only place that must be exact.
        path = _path(Path(root), sid, token)
        if path.stat().st_size <= 0:
            return None
    except (OSError, ValueError):
        return None
    return {"schema": SCHEMA, "publication_id": token}


def publish(root: str | Path, sid: str, index: ti.TranscriptIndex) -> dict[str, Any]:
    """Build one immutable companion from the final authoritative object.

    Called BEFORE the canonical JSON is committed, so a failed replace leaves a
    harmless unreferenced companion and readers keep the previous publication.
    A failed build raises; the caller records the failure and publishes JSON
    only, which is exactly the pre-read-model behaviour.
    """
    token = uuid.uuid4().hex
    path = _path(Path(root), sid, token)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    meta = {
        "schema": SCHEMA,
        "projection": PROJECTION,
        "publication_id": token,
        "version": ti.TRANSCRIPT_INDEX_VERSION,
        "sig": index.sig,
        "scan": index.scan.to_payload(),
        "journal_generation": index.journal_generation,
        "checkpoint_count": len(index.checkpoints),
        "run_count": len(index.runs),
        "last_run": len(index.runs) - 1,
        "coverage_complete": index.coverage.get("complete") is True,
    }
    if not meta["coverage_complete"] or not index.journal_generation:
        raise ValueError("incomplete index projection")
    try:
        with closing(sqlite3.connect(temp)) as db:
            db.executescript("""
                PRAGMA journal_mode=DELETE;
                CREATE TABLE meta(payload TEXT NOT NULL);
                CREATE TABLE rows(id TEXT PRIMARY KEY, seq INTEGER NOT NULL) WITHOUT ROWID;
                CREATE TABLE messages(seq INTEGER PRIMARY KEY, id TEXT,
                                      role TEXT, injected INTEGER);
                CREATE TABLE runs(position INTEGER PRIMARY KEY, first_seq INTEGER, last_seq INTEGER,
                                  payload TEXT NOT NULL);
                CREATE INDEX runs_first ON runs(first_seq);
                CREATE TABLE checkpoints(position INTEGER PRIMARY KEY, id TEXT, seq INTEGER,
                                         payload TEXT NOT NULL);
                CREATE INDEX checkpoints_id ON checkpoints(id, position);
            """)
            db.execute("INSERT INTO meta VALUES (?)", (json.dumps(meta),))
            db.executemany("INSERT INTO rows VALUES (?,?)", index.journal_rows.items())
            # Only rows the opener proof can ever consult: a genuine user or
            # assistant doc that is not injected (``_journal_order_doc``'s rule).
            # Tool/marker bodies and injected docs are dead weight here.
            db.executemany(
                "INSERT INTO messages VALUES (?,?,?,?)",
                (
                    (m.seq, m.id, m.role, int(m.injected))
                    for m in index.messages
                    if m.role in ("user", "assistant")
                ),
            )
            db.executemany(
                "INSERT INTO runs VALUES (?,?,?,?)",
                (
                    (n, r.first_seq, r.last_seq, json.dumps(r.to_payload()))
                    for n, r in enumerate(index.runs)
                ),
            )
            db.executemany(
                "INSERT INTO checkpoints VALUES (?,?,?,?)",
                (
                    (n, c.id, c.seq, json.dumps(c.to_payload()))
                    for n, c in enumerate(index.checkpoints)
                ),
            )
            db.commit()
            # VACUUM compacts the b-tree after the bulk inserts; without it the
            # file keeps the free pages a row-at-a-time insert left behind.
            db.execute("VACUUM")
        with temp.open("rb") as handle:
            os.fsync(handle.fileno())
        temp.replace(path)
        # The JSON commit may survive a crash only if its referenced immutable
        # filename is already durable; fsync the directory as well as DB bytes.
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        # Only the unique scratch file owned by this writer, never a publication.
        temp.unlink(missing_ok=True)
    return {"schema": SCHEMA, "publication_id": token}


def write_naming(
    root: str | Path, sid: str, marker: dict[str, Any] | None, naming: dict[str, Any]
) -> bool:
    """Publish the tiny naming sidecar for an existing companion.

    Naming is the one mutable section: a patch must reach readers without
    rewriting the companion it belongs to. Atomic replace means a reader sees
    either the previous or the new complete sidecar, never a torn one.
    """
    if not isinstance(marker, dict) or not isinstance(naming, dict):
        return False
    token = marker.get("publication_id")
    if not isinstance(token, str):
        return False
    path = _path(Path(root), sid, token).with_suffix(".naming.json")
    temp = path.with_suffix(".naming.tmp")
    payload = {
        "schema": SCHEMA,
        "publication_id": token,
        "prompt_version": naming.get("prompt_version"),
        "items": naming.get("items") if isinstance(naming.get("items"), dict) else {},
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return True


def reset_for_tests() -> None:
    """Drop in-process memo/failure/migration state (tests only, never production)."""
    with _MEMO_LOCK:
        _MANIFEST_MEMO.clear()
    _FAILED.clear()
    for task in list(_MIGRATIONS.values()):
        task.cancel()
    _MIGRATIONS.clear()


def _naming_identity(pub: Publication) -> tuple[int, int, int] | None:
    try:
        return _identity(pub.naming_path.stat())
    except OSError:
        return None


def _naming(pub: Publication) -> dict[str, Any]:
    """The sidecar as a naming section; a missing/torn one is not usable."""
    raw = json.loads(pub.naming_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != SCHEMA
        or raw.get("publication_id") != pub.header["read_model"]["publication_id"]
        or not isinstance(raw.get("items"), dict)
    ):
        raise ValueError("unusable naming sidecar")
    return {"prompt_version": raw.get("prompt_version"), "items": raw["items"]}


def _meta(db: sqlite3.Connection, pub: Publication) -> dict[str, Any]:
    rows = db.execute("SELECT payload FROM meta LIMIT 2").fetchall()
    if len(rows) != 1:
        raise ValueError("missing/duplicate metadata")
    meta = json.loads(rows[0][0])
    for key in (*ti._HEAD_KEYS,):
        if meta[key] != pub.header[key]:
            raise ValueError("canonical metadata mismatch")
    if (
        meta["schema"] != SCHEMA
        or meta["projection"] != PROJECTION
        or meta["publication_id"] != pub.header["read_model"]["publication_id"]
        or meta["coverage_complete"] is not True
    ):
        raise ValueError("unsupported projection")
    return meta


def _read(
    root: str | Path,
    sid: str,
    select: Callable[[sqlite3.Connection, Publication, dict[str, Any]], Any],
    *,
    with_publication: bool = False,
) -> Result:
    """One read-only handle per attempt; bounded retry on publication motion.

    ``with_publication`` returns ``(publication, payload)`` for the publication the
    read ACTUALLY resolved. A caller must never pair facts from one publication
    with a ``built_at`` read from another: a flip between two header reads is
    exactly the incoherence the retry exists to prevent.
    """
    for _ in range(2):
        probe = header(root, sid)
        if probe.state != "ready":
            return probe
        pub = probe.value
        if not _current(pub):
            continue
        try:
            with closing(
                sqlite3.connect(pub.path.as_uri() + "?mode=ro&immutable=1", uri=True)
            ) as db:
                db.execute("PRAGMA trusted_schema=OFF")
                db.execute("PRAGMA query_only=ON")
                value = select(db, pub, _meta(db, pub))
            if _current(pub):
                return Result("ready", (pub, value) if with_publication else value)
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, IndexError) as exc:
            if not _current(pub):
                continue
            return Result("corrupt", reason=str(exc))
    return Result("stale")


def _run(payload: str) -> ti.RunRecord:
    run = ti.RunRecord.from_payload(json.loads(payload))
    if run is None:
        raise ValueError("invalid run projection")
    return run


def _summary(db: sqlite3.Connection, pub: Publication, meta: dict[str, Any]) -> FrameSummary:
    scan = ti.ScanState.from_payload(meta["scan"])
    if scan is None:
        raise ValueError("invalid scan metadata")
    last = db.execute("SELECT position,payload FROM runs ORDER BY position DESC LIMIT 1").fetchone()
    if (last is None) != (meta["run_count"] == 0):
        raise ValueError("missing tail summary")
    if last is not None and last[0] != meta["last_run"]:
        raise ValueError("incomplete tail summary")
    return FrameSummary(scan, [_run(last[1])] if last else [], pub.signature)


def summary(root: str | Path, sid: str, source: tuple[int, int, float] | None) -> Result:
    """The tail summary, only for a page read from THIS publication's source."""

    def select(db: sqlite3.Connection, pub: Publication, meta: dict[str, Any]) -> FrameSummary:
        if source is None or source != pub.signature:
            raise ValueError("page/source mismatch")
        return _summary(db, pub, meta)

    return _read(root, sid, select)


def frame_facts(
    root: str | Path,
    sid: str,
    *,
    collected_ids: list[str],
    reaches_eof: bool,
    order_ids: list[str],
    source: tuple[int, int, float] | None,
) -> Result:
    """Materialize bounded records in this worker; reuse authoritative helpers.

    SQL selects the containing run, its predecessor and the next boundary, not
    new run semantics. No connection/accessor escapes to the rendering loop.
    Missing page identities reject the join; absent requested IDs remain explicit.
    """
    from local_operator.session import open_frame as frame

    def select(db: sqlite3.Connection, pub: Publication, meta: dict[str, Any]) -> FrameFacts:
        if source is None or source != pub.signature:
            raise ValueError("page/source mismatch")
        ids = list(dict.fromkeys(collected_ids + order_ids))
        rows: dict[str, int] = {}
        checkpoints: list[ti.Checkpoint] = []
        # Chunk by input identities, not history. Checkpoint duplicate IDs follow
        # the canonical dict's LAST-position policy without truncating manifests.
        for start in range(0, len(ids), 128):
            chunk = ids[start : start + 128]
            slots = ",".join("?" for _ in chunk)
            rows.update(db.execute(f"SELECT id,seq FROM rows WHERE id IN ({slots})", chunk))
            for (payload,) in db.execute(
                f"SELECT payload FROM checkpoints c WHERE id IN ({slots}) AND position="
                "(SELECT MAX(position) FROM checkpoints WHERE id=c.id)",
                chunk,
            ):
                checkpoint = ti.Checkpoint.from_payload(json.loads(payload))
                if checkpoint is None:
                    raise ValueError("invalid checkpoint projection")
                checkpoints.append(checkpoint)
        if any(id_ not in rows for id_ in collected_ids):
            raise ValueError("page identity missing from projection")
        tail = _summary(db, pub, meta)
        records: dict[int, ti.RunRecord] = {}

        def take(query: str, args: tuple[Any, ...]) -> list[int]:
            positions = []
            for position, payload in db.execute(query, args):
                records[position] = _run(payload)
                positions.append(position)
            return positions

        known = {c.id: c.seq for c in checkpoints}
        seqs = [known[id_] for id_ in collected_ids if id_ in known]
        if seqs:
            before = db.execute(
                "SELECT position FROM runs WHERE first_seq<=? "
                "ORDER BY first_seq DESC,position DESC LIMIT 1",
                (min(seqs),),
            ).fetchone()
            beginning = max(0, before[0] - 1) if before else 0
            positions = take(
                "SELECT position,payload FROM runs WHERE position>=? ORDER BY position LIMIT ?",
                (beginning, frame.OPEN_FRAME_MAX_ROWS),
            )
            expected = min(frame.OPEN_FRAME_MAX_ROWS, meta["run_count"] - beginning)
            if positions != list(range(beginning, beginning + expected)):
                raise ValueError("incomplete run selection")
            if len(positions) == frame.OPEN_FRAME_MAX_ROWS:
                # A pathological one-row-run page can reach the helper's cap.
                # Materialize the one boundary row needed to distinguish a real
                # cap from missing rows; never truncate the tail just to match.
                take(
                    "SELECT position,payload FROM runs WHERE position>=? ORDER BY position LIMIT 1",
                    (beginning + len(positions),),
                )
            # The bounds rule also needs the run AFTER the newest checkpoint,
            # even if a display cap stopped the selected publication earlier.
            take(
                "SELECT position,payload FROM runs WHERE first_seq>? "
                "ORDER BY first_seq,position LIMIT 1",
                (max(seqs),),
            )
        messages: dict[int, ti.MessageDoc] = {}
        for id_, seq in list(rows.items()):
            take(
                "SELECT position,payload FROM runs WHERE first_seq<=? "
                "ORDER BY first_seq DESC,position DESC LIMIT 1",
                (seq,),
            )
        # Real opener proof uses role/non-injection, exact raw ordinal and run
        # membership through journal_order_opener, not a last-user heuristic.
        for run in list(records.values()):
            if run.opening_user_id:
                row = db.execute(
                    "SELECT seq FROM rows WHERE id=?", (run.opening_user_id,)
                ).fetchone()
                if row:
                    rows[run.opening_user_id] = row[0]
        for seq in set(rows.values()):
            row = db.execute("SELECT id,role,injected FROM messages WHERE seq=?", (seq,)).fetchone()
            if row:
                messages[seq] = ti.MessageDoc(row[0], 0.0, row[1], "", bool(row[2]), seq)
        selection = FrameSelection(
            checkpoints,
            [records[n] for n in sorted(records)],
            [messages[n] for n in sorted(messages)],
            rows,
            meta["journal_generation"],
        )
        bounds = frame.page_seq_bounds(
            selection, [{"id": id_} for id_ in collected_ids], reaches_eof=reaches_eof
        )
        runs = (
            frame.publish_runs(selection, first_seq=bounds[0], last_seq=bounds[1]) if bounds else []
        )
        ordinals = {id_: rows[id_] for id_ in order_ids if id_ in rows}
        # Served-vs-collected is decided by the CALLER's cap, so materialize the
        # opener of every candidate and let ``apply_order`` narrow to what it
        # actually serves. Emitting them all here is what put coordinates for
        # rows the page dropped into the wire payload.
        openers: dict[str, tuple[str, int]] = {}
        for id_ in ids:
            if id_ in rows:
                opener = ti.journal_order_opener(selection, id_, rows[id_])
                if opener:
                    openers[id_] = opener
        return FrameFacts(
            runs,
            # EVERY identity this request proved, not just the page's: an id the
            # caller asked about but the cap dropped still has to reach the wire
            # in ``journal_ordinals``, which is what the resident join does.
            {id_: rows[id_] for id_ in ids if id_ in rows},
            ordinals,
            openers,
            [id_ for id_ in order_ids if id_ not in rows],
            meta["journal_generation"],
            pub.signature,
            tail,
        )

    return _read(root, sid, select)


def schedule_migration(root: str | Path, sid: str, index: ti.TranscriptIndex) -> None:
    """Reuse an already parsed exact HIT after paint; never hydrate it again.

    Sync workers return the parsed object to start_refresh's loop, which calls
    this hook. Direct loop-side refresh/probe callers can schedule here too.
    A missing loop is not permission to start an unmanaged thread/service.
    """
    if index.publication_identity is None or index.read_model_token:
        return
    if failed(root, sid, index.publication_identity):
        # A companion that already failed for this exact document is not retried:
        # each attempt would rewrite the whole canonical JSON for nothing.
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    key = ti._key(root, sid)
    started = time.monotonic()
    if key in _MIGRATIONS or len(_MIGRATIONS) >= 16:
        return

    def materialize() -> None:
        # Publication is optional. Never let a missing/unwritable companion turn
        # an otherwise correct parsed HIT into an unbounded background retry.
        if time.monotonic() - started >= ti._PAINT_HOLD_CEILING_S:
            return
        # A bounded metadata probe avoids republishing already-supported hits.
        # No second full-index decode, and writer-side compare-and-publish rejects
        # both a replaced journal and a naming publication that landed meanwhile.
        supported = _read(root, sid, lambda db, pub, meta: None)
        if supported.state == "ready":
            return
        ti.write_index(root, sid, index, expected_publication=index.publication_identity)

    async def run() -> None:
        try:
            held = ti._PAINT_RELEASED.get(key)
            if held is not None and held[0] is loop and not held[1].is_set():
                try:
                    await asyncio.wait_for(held[1].wait(), ti._PAINT_HOLD_CEILING_S)
                except asyncio.TimeoutError:
                    pass
            # Cancellation cannot abandon a worker that still owns an index/file
            # handle: drain its bounded publication before completing this task.
            worker = asyncio.create_task(asyncio.to_thread(materialize))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await worker
                raise
        except Exception:
            logger.debug("read-model migration unavailable for %s", sid, exc_info=True)
        finally:
            _MIGRATIONS.pop(key, None)

    task = loop.create_task(run(), name=f"index-read-model-{sid}")
    _MIGRATIONS[key] = task

    def finished(done: asyncio.Task[None]) -> None:
        # A task cancelled before its first instruction never enters finally.
        if _MIGRATIONS.get(key) is done:
            _MIGRATIONS.pop(key, None)

    task.add_done_callback(finished)


def collect(root: str | Path, sid: str, marker: dict[str, Any] | None) -> None:
    """Bounded retention for this publisher: the referenced token plus a window.

    The companion files are a cache this writer made, never session evidence, and
    a failure to remove is treated as RETAINED rather than as an error: the only
    cost is disk, and the canonical head never references a token it does not
    name. Open readers hold descriptors, so reclaiming a path a request already
    opened cannot pull data out from under it; a reader that only kept a PATH
    re-reads the head and retries (see :func:`_read`).

    Session directories whose canonical index is gone (a deleted session) are
    swept by :func:`_sweep_orphans` from the same call.
    """
    directory = _directory(Path(root))
    current = marker.get("publication_id") if isinstance(marker, dict) else None
    session = directory / sid
    try:
        files = sorted(
            (path for path in session.iterdir() if path.suffix in {".sqlite", ".json"}),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
    except OSError:
        return
    keep = {current} if isinstance(current, str) else set()
    for path in files:
        if len(keep) >= _RETAIN_PREVIOUS + 1:
            break
        keep.add(path.stem.split(".")[0])
    for path in files:
        if path.stem.split(".")[0] in keep:
            continue
        try:
            path.unlink()
        except OSError:
            pass
    _sweep_orphans(root, directory, sid)


def _sweep_orphans(root: str | Path, directory: Path, sid: str, limit: int = 64) -> None:
    """Reclaim read models of deleted sessions, bounded and never mid-write."""
    cutoff = time.time() - _ORPHAN_GRACE_S
    try:
        entries = list(os.scandir(directory))[:limit]
    except OSError:
        return
    for entry in entries:
        if entry.name == sid or not entry.is_dir():
            continue
        try:
            if ti.index_path(root, entry.name).exists():
                continue
            if any(child.stat().st_mtime > cutoff for child in Path(entry.path).iterdir()):
                continue
            for child in Path(entry.path).iterdir():
                child.unlink(missing_ok=True)
            Path(entry.path).rmdir()
        except OSError:
            continue


def page_source(pages: list[Any]) -> tuple[int, int, float] | None:
    """Require every page's original read witness, not a current-path guess."""
    sources = [page.source_signature for page in pages]
    return sources[0] if sources and all(s == sources[0] for s in sources) else None


def apply_order(answer: dict[str, Any], facts: FrameFacts) -> bool:
    """Attach only the already-materialized source facts (no loop-side I/O).

    ``journal_ordinals`` describes what THIS answer serves plus the ids the
    caller asked about — never a row a cap dropped — which is the same rule
    ``apply_journal_order`` applies against a resident index.
    """
    if any(row["id"] not in facts.rows for row in answer["entries"]):
        return False
    answer["entries"] = [{**row, "journal_seq": facts.rows[row["id"]]} for row in answer["entries"]]
    answer["journal_order_state"] = "ready"
    answer["journal_generation"] = facts.generation
    # ``journal_seq`` rides each row above; ``journal_ordinals`` carries the ids
    # the caller ASKED about plus the genuine OPENERS of what this answer serves —
    # exactly what ``apply_journal_order`` emits against a resident index. A row
    # the cap dropped contributes neither, which is the pre-fix defect (a
    # ``limit=5`` page answered an opener belonging to a row it never served).
    served = list(dict.fromkeys([row["id"] for row in answer["entries"]] + list(facts.ordinals)))
    ordinals = {id_: facts.rows[id_] for id_ in facts.ordinals if id_ in facts.rows}
    for id_ in served:
        opener = facts.openers.get(id_)
        if opener is not None:
            ordinals[opener[0]] = opener[1]
    answer["journal_ordinals"] = ordinals
    answer["journal_missing_ids"] = facts.missing_ids
    return True


def checkpoint_manifest(root: str | Path, sid: str) -> Result:
    """Complete O(output) rail, never a capped frame slice labelled complete.

    The manifest is re-derived from cached checkpoint/naming objects when the
    canonical document and its naming sidecar are byte-for-byte the same pair,
    because two stats are what separates a resident-fast-path poll (6-38 ms) from
    re-opening SQLite and rebuilding the projection on every poll (79-104 ms).
    Naming STATE is still evaluated per call, so a cooldown expiry is visible
    without a new publication.
    """
    key = (str(root), sid)
    probe = header(root, sid)
    if probe.state != "ready":
        return probe
    with _MEMO_LOCK:
        memo = _MANIFEST_MEMO.get(key)
    if memo is not None:
        identity, naming_identity, built_at, checkpoints, naming = memo
        if identity == probe.value.identity and _naming_identity(probe.value) == naming_identity:
            pub = Publication(
                probe.value.root,
                sid,
                probe.value.header,
                identity,
                built_at,
                probe.value.path,
            )
            return Result("ready", _manifest(sid, pub, checkpoints, naming))

    def select(db: sqlite3.Connection, current: Publication, meta: dict[str, Any]) -> Any:
        naming = _naming(current)
        checkpoints = []
        for expected, (position, payload) in enumerate(
            db.execute("SELECT position,payload FROM checkpoints ORDER BY position")
        ):
            checkpoint = ti.Checkpoint.from_payload(json.loads(payload))
            if position != expected or checkpoint is None:
                raise ValueError("incomplete checkpoint projection")
            checkpoints.append(checkpoint)
        if len(checkpoints) != meta["checkpoint_count"]:
            raise ValueError("incomplete manifest")
        return naming, checkpoints

    result = _read(root, sid, select, with_publication=True)
    if result.state != "ready":
        return result
    resolved, (naming, checkpoints) = result.value
    with _MEMO_LOCK:
        if len(_MANIFEST_MEMO) >= 8 and key not in _MANIFEST_MEMO:
            _MANIFEST_MEMO.pop(next(iter(_MANIFEST_MEMO)))
        _MANIFEST_MEMO[key] = (
            resolved.identity,
            _naming_identity(resolved),
            resolved.built_at,
            checkpoints,
            naming,
        )
    return Result("ready", _manifest(sid, resolved, checkpoints, naming))


def _manifest(
    sid: str,
    pub: Publication,
    checkpoints: list[ti.Checkpoint],
    naming: dict[str, Any],
) -> dict[str, Any]:
    answer = ti._manifest_state(sid, "ready", None, pub.built_at)
    answer["checkpoints"] = [
        ti.checkpoint_entry(checkpoint, item)
        for checkpoint, item in ti.resolve_naming_items(checkpoints, naming)
    ]
    return answer
