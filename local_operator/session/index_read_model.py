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
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import zlib
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_operator.session import transcript_index as ti

#: Bumped whenever the companion's tables/semantics change. An older companion
#: is then *unsupported* (fallback), never misread.
SCHEMA = 3
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
_FAILED: dict[tuple[str, str], tuple[int, int, float]] = {}
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
    token: str = ""

    @property
    def token_id(self) -> str:
        return self.token or self.path.stem.split(".")[0]

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


def _token_for(values: dict[str, Any]) -> str:
    """The content address of one canonical publication.

    DERIVED, NOT STORED. The companion is named after the derivation it describes
    (index version + journal signature + scan + generation), so publishing it
    needs NO canonical rewrite and a journal that moves mid-build simply produces
    a token the next head no longer names. That is the whole compare-and-publish:
    the writer re-reads the head before renaming, and readers only look up the
    token their own head derives.
    """
    sig = values["sig"]
    material = json.dumps(
        [
            SCHEMA,
            values["version"],
            sig.get("inode"),
            sig.get("size"),
            sig.get("mtime"),
            values["scan"],
            values["journal_generation"],
        ],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def token_for(index: ti.TranscriptIndex) -> str:
    """The derivation token an index would publish under."""
    return _token_for(
        {
            "version": ti.TRANSCRIPT_INDEX_VERSION,
            "sig": index.sig,
            "scan": index.scan.to_payload(),
            "journal_generation": index.journal_generation,
        }
    )


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
        for expected in ti._HEAD_KEYS:
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
            if expected != ti._HEAD_KEYS[-1]:
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if text[cursor] != ",":
                    return Result("corrupt")
                cursor += 1
        if values["version"] != ti.TRANSCRIPT_INDEX_VERSION:
            return Result("unsupported")
        try:
            token = _token_for(values)
            path = _path(Path(root), sid, token)
        except (TypeError, ValueError, KeyError):
            return Result("corrupt")
        if not path.is_file():
            # No companion for THIS derivation: the caller takes base's path.
            return Result("missing")
        publication = Publication(Path(root), sid, values, _identity(st), st.st_mtime, path, token)
        # THE JOURNAL ITSELF IS WITNESSED HERE (round-3 N1). The cached document
        # and its companion describe the scan they were built from; an append
        # moves neither of those two files, so a manifest served on their word
        # alone answered `ready N` forever while the journal had grown. One stat
        # of the journal is the difference between "this projection is the
        # current derivation" and "this projection is old but internally
        # consistent", and it is the same signature the derivation token uses.
        sig = publication.signature
        if (
            type(sig[0]) is not int
            or type(sig[1]) is not int
            or not isinstance(sig[2], (int, float))
        ):
            return Result("corrupt")
        live = _journal_signature(Path(root), sid)
        if live is not None and live != sig:
            # A moved journal makes this a PREVIOUS derivation: the caller takes
            # base's refresh path (which is what makes the append visible at all).
            return Result("stale", publication)
        return Result("ready", publication)
    except FileNotFoundError:
        return Result("missing")
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return Result("unsupported")


def _journal_signature(root: Path, sid: str) -> tuple[int, int, float] | None:
    """The live journal's signature, or ``None`` when there is no journal."""
    st = ti._stat_or_none(ti._journal_path(root, sid))
    if st is None:
        return None
    return (st.st_ino, st.st_size, st.st_mtime)


def _current(pub: Publication) -> bool:
    try:
        return _identity(
            ti.index_path(pub.root, pub.sid).stat()
        ) == pub.identity and ti._sig_matches(
            pub.header["sig"], ti._journal_path(pub.root, pub.sid).stat()
        )
    except OSError:
        return False


def failed(root: str | Path, sid: str, source: tuple[int, int, float] | None) -> bool:
    """Whether THIS source already failed to publish a companion.

    Keyed on the JOURNAL's signature, not on the cached document's identity: the
    failure path rewrites the document, so a document identity is stale by the
    next open and the old key retried forever (round-2 R3). The negative entry is
    honoured until the source itself moves.
    """
    if source is None:
        return False
    return _FAILED.get((str(root), sid)) == source


def remember_failure(root: str | Path, sid: str, source: tuple[int, int, float] | None) -> None:
    """Remember ONE failed source per session, bounded across sessions."""
    if source is None:
        return
    key = (str(root), sid)
    if key not in _FAILED and len(_FAILED) >= 32:
        _FAILED.pop(next(iter(_FAILED)))
    _FAILED[key] = source


def source_of(index: ti.TranscriptIndex) -> tuple[int, int, float]:
    """The journal witness an index was derived from."""
    return (
        int(index.sig.get("inode", 0)),
        int(index.sig.get("size", 0)),
        float(index.sig.get("mtime", 0.0)),
    )


def publish(root: str | Path, sid: str, index: ti.TranscriptIndex) -> str | None:
    """Build one immutable companion for this derivation, if it is still current.

    CALLED FROM A BACKGROUND PASS, never from the canonical write: the build is
    the expensive part (~1.3 s measured on the append path) and putting it in the
    writer delayed every refresh's publish. Everything here is compare-and-
    publish against the head on disk — a journal that moved mid-build, or a
    document replaced under us, leaves the finished file unreferenced and the
    collector reclaims it.

    Returns the token published, or ``None`` when the derivation is no longer the
    current one.
    """
    values = {
        "version": ti.TRANSCRIPT_INDEX_VERSION,
        "sig": index.sig,
        "scan": index.scan.to_payload(),
        "journal_generation": index.journal_generation,
    }
    if index.coverage.get("complete") is not True or not index.journal_generation:
        raise ValueError("incomplete index projection")
    token = _token_for(values)
    path = _path(Path(root), sid, token)
    if path.is_file():
        return token  # already published for this exact derivation
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    meta = {
        "schema": SCHEMA,
        "projection": PROJECTION,
        "publication_id": token,
        **values,
        "checkpoint_count": len(index.checkpoints),
        "run_count": len(index.runs),
        "last_run": len(index.runs) - 1,
        "coverage_complete": True,
    }
    try:
        with closing(sqlite3.connect(temp)) as db:
            db.executescript("""
                -- 1024-byte pages: this file is thousands of TINY rows and the
                -- default 4 KiB page left most of every page as slack.
                PRAGMA page_size=1024;
                PRAGMA journal_mode=DELETE;
                CREATE TABLE meta(payload TEXT NOT NULL);
                -- ONE table where the projection used to keep two: the id->ordinal
                -- map and the role/injected witness answer the same question about
                -- the same row, and shared every id.
                CREATE TABLE nodes(id TEXT PRIMARY KEY, seq INTEGER NOT NULL,
                                   role TEXT, injected INTEGER);
                CREATE INDEX nodes_seq ON nodes(seq);
                CREATE TABLE runs(position INTEGER PRIMARY KEY, first_seq INTEGER, last_seq INTEGER,
                                  payload BLOB NOT NULL);
                CREATE INDEX runs_first ON runs(first_seq);
                CREATE TABLE checkpoints(position INTEGER PRIMARY KEY, id TEXT, seq INTEGER,
                                         payload BLOB NOT NULL);
                CREATE INDEX checkpoints_id ON checkpoints(id, position);
            """)
            db.execute("INSERT INTO meta VALUES (?)", (json.dumps(meta),))
            # Only the rows the opener proof can consult carry a role: a genuine
            # user or assistant doc, never a tool body or an injected doc.
            roles = {
                m.id: (m.role, int(m.injected))
                for m in index.messages
                if m.role in ("user", "assistant")
            }
            db.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?)",
                (
                    (row_id, seq, *(roles.get(row_id) or (None, None)))
                    for row_id, seq in index.journal_rows.items()
                ),
            )
            db.executemany(
                "INSERT INTO runs VALUES (?,?,?,?)",
                (
                    (n, r.first_seq, r.last_seq, _packed(r.to_payload()))
                    for n, r in enumerate(index.runs)
                ),
            )
            db.executemany(
                "INSERT INTO checkpoints VALUES (?,?,?,?)",
                (
                    (n, c.id, c.seq, _packed(c.to_payload()))
                    for n, c in enumerate(index.checkpoints)
                ),
            )
            db.commit()
            db.execute("VACUUM")
        # fsync THE FILE WE WROTE, through a descriptor opened for writing: a
        # read-only handle's fsync is EINVAL/EBADF on Windows, and this ran there
        # inside a bare ``except OSError`` once already.
        fd = os.open(temp, os.O_RDWR)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if _head_token(root, sid) != token:
            return None  # the derivation moved while this was building
        temp.replace(path)
        _fsync_directory(path.parent)
    finally:
        temp.unlink(missing_ok=True)
    return token


def _fsync_directory(directory: Path) -> None:
    """Durability hint for the rename itself; unsupported on Windows."""
    if os.name != "posix":
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystems that refuse directory sync
        pass
    finally:
        os.close(fd)


def _head_token(root: str | Path, sid: str) -> str | None:
    """The token the canonical head currently derives, without opening the DB."""
    head = ti._read_resume_head(root, sid)
    if head is None:
        return None
    sig, scan, generation, _size = head
    return _token_for(
        {
            "version": ti.TRANSCRIPT_INDEX_VERSION,
            "sig": sig,
            "scan": scan.to_payload(),
            "journal_generation": generation,
        }
    )


def _packed(payload: dict[str, Any]) -> bytes:
    """One record payload, zlib-compressed.

    The companion held 69% of the index's bytes before this: records were stored
    as plain JSON text, and these payloads are highly repetitive
    (``"settled":true,"complete":true,...``). Compression is applied per record
    so a bounded read still decodes only the rows it selected.
    """
    return zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), 6)


def _unpacked(blob: Any) -> dict[str, Any]:
    raw = zlib.decompress(bytes(blob)).decode("utf-8") if isinstance(blob, bytes) else str(blob)
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("invalid packed record")
    return loaded


def write_naming(root: str | Path, sid: str, token: str | None, naming: dict[str, Any]) -> bool:
    """Publish the tiny naming sidecar for an existing (or imminent) companion.

    Naming is the one mutable section: a patch must reach readers without
    rewriting the companion, and a naming-only write keeps the same token because
    the token describes the DERIVATION, not the names.
    """
    if not isinstance(token, str) or not isinstance(naming, dict):
        return False
    path = _path(Path(root), sid, token).with_suffix(".naming.json")
    temp = path.with_suffix(f".{os.getpid()}.naming.tmp")
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


def _naming(pub: Publication) -> dict[str, Any]:
    """The sidecar as a naming section; a missing/torn one is not usable."""
    raw = json.loads(pub.naming_path.read_text(encoding="utf-8"))
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != SCHEMA
        or raw.get("publication_id") != pub.token_id
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
        or meta["publication_id"] != pub.token_id
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


def _run(payload: Any) -> ti.RunRecord:
    run = ti.RunRecord.from_payload(_unpacked(payload))
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
            rows.update(db.execute(f"SELECT id,seq FROM nodes WHERE id IN ({slots})", chunk))
            for (payload,) in db.execute(
                # LAST position wins for a duplicate id, matching the canonical
                # dict's own construction (``{c.id: c.seq}`` over the list).
                f"SELECT payload FROM checkpoints c WHERE id IN ({slots}) AND position="
                "(SELECT MAX(position) FROM checkpoints WHERE id=c.id)",
                chunk,
            ):
                checkpoint = ti.Checkpoint.from_payload(_unpacked(payload))
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
                    "SELECT seq FROM nodes WHERE id=?", (run.opening_user_id,)
                ).fetchone()
                if row:
                    rows[run.opening_user_id] = row[0]
        for seq in set(rows.values()):
            row = db.execute("SELECT id,role,injected FROM nodes WHERE seq=?", (seq,)).fetchone()
            if row and row[1] is not None:
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


def schedule_materialize(root: str | Path, sid: str, index: ti.TranscriptIndex) -> None:
    """Publish a companion for an ALREADY-DERIVED index, after the paint.

    THE COMPANION IS NEVER ON THE REFRESH CRITICAL PATH (round-2 scope): the
    canonical document is written exactly as base writes it, and this hook runs
    afterwards with the completed derivation already in hand — so it neither
    hydrates a second time nor makes an append wait ~1.3 s. Exact HITs reach it
    the same way, which is what removes the old "migration rewrites the document
    on every open" loop entirely: nothing here touches the canonical file.

    A failed build is remembered per SOURCE signature (``failed``), so a torn-tail
    journal does not retry on every open; the next derivation moves the signature
    and is tried normally.
    """
    if index.publication_identity is None and not index.journal_generation:
        return
    source = source_of(index)
    if failed(root, sid, source):
        return
    # ONE head read, not two (round-3 N2): the token this pass would publish is
    # derived from the index it was handed, and the head only has to say whether
    # that exact derivation is already on disk.
    if _companion_exists(root, sid, index):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # A worker thread cannot own a task; the loop-side callers schedule it.
        return
    key = ti._key(root, sid)
    if key in _MIGRATIONS or len(_MIGRATIONS) >= 16:
        return
    started = time.monotonic()

    def build() -> None:
        if time.monotonic() - started >= ti._PAINT_HOLD_CEILING_S:
            return
        try:
            publish(root, sid, index)
        except (OSError, ValueError, sqlite3.Error):
            remember_failure(root, sid, source)
            logger.debug("read-model publication unavailable for %s", sid, exc_info=True)

    async def run() -> None:
        try:
            held = ti._PAINT_RELEASED.get(key)
            if held is not None and held[0] is loop and not held[1].is_set():
                try:
                    await asyncio.wait_for(held[1].wait(), ti._PAINT_HOLD_CEILING_S)
                except asyncio.TimeoutError:
                    pass
            worker = asyncio.create_task(asyncio.to_thread(build))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await worker
                raise
        except Exception:  # noqa: BLE001 - an optimization never fails a read
            logger.debug("read-model materialisation unavailable for %s", sid, exc_info=True)
        finally:
            _MIGRATIONS.pop(key, None)

    task = loop.create_task(run(), name=f"index-read-model-{sid}")
    _MIGRATIONS[key] = task

    def finished(done: asyncio.Task[None]) -> None:
        if _MIGRATIONS.get(key) is done:
            _MIGRATIONS.pop(key, None)

    task.add_done_callback(finished)


def _companion_exists(root: str | Path, sid: str, index: ti.TranscriptIndex) -> bool:
    """Whether a companion already exists for THIS index's derivation.

    Deliberately does not consult the canonical head: the caller holds the
    completed derivation, so the only question left is whether its token is
    published — and the head would be a second read of the same document the
    caller just wrote.
    """
    try:
        return _path(Path(root), sid, token_for(index)).is_file()
    except ValueError:
        return False


def collect(root: str | Path, sid: str, token: str | None) -> None:
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
    current = token if isinstance(token, str) else None
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
            # The directory itself is LEFT IN PLACE. Removing it would make this
            # cache module a directory remover, and the store's guard
            # (tests/unit/session/test_no_session_deletion.py) treats that shape as
            # a session-deletion surface wherever it appears — an empty directory of
            # a few hundred bytes is not worth an allow-list exception on that
            # guard, and the next publication for that session reuses the path.
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
        identity, naming_identity, built_at, journal, checkpoints, naming = memo
        if (
            identity == probe.value.identity
            and _naming_identity(probe.value) == naming_identity
            and journal == _journal_signature(probe.value.root, sid)
        ):
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
            checkpoint = ti.Checkpoint.from_payload(_unpacked(payload))
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
            _journal_signature(resolved.root, sid),
            checkpoints,
            naming,
        )
    return Result("ready", _manifest(sid, resolved, checkpoints, naming))


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
