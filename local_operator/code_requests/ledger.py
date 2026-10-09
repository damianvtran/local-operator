"""Store and read a session's code-request facts: event rows, index, and cache.

THREE FILES, THREE JOBS, one writer each.

1. **The event row** — ``code_request_event.v1``, appended to the session's own
   transcript by the live hook (``hook.py``). It is the only AUTHORITATIVE record:
   it survives the compaction that blanks the tool result it came from, and it rides
   along when the transcript is resumed or forked. ``append_event`` is the one
   writer, and it mirrors ``mesh_credential_binding.v1``'s treatment: append-only and
   deliberately NOT in ``_COLLAPSIBLE_CUSTOM_TYPES``, because collapsing keeps one
   newest row per type and every event is a distinct fact.

2. **The derived index** — ``<config>/code_requests/<session_id>.json``. It is what a
   reader uses when it has no session open: one small file naming a session's rows, so
   a machine-wide view never opens thousands of session directories. It follows
   ``monitors/store.py`` exactly — derived, never authoritative, stdlib-only, written
   atomically (pid-suffixed temp + ``os.replace``), and rewritten the moment the
   scanner produces a new answer.

3. **The scan cache** — ``<config>/cache/code_requests/<session_id>.json``. The
   incremental scan's resume point and its result, keyed on the journal's
   ``{size, mtime, inode, last_id}`` signature — ``session/transcript_index.py``'s
   invalidation ladder, and for the same reason: ``compact_file`` REPLACES the
   transcript (tmp + ``os.replace``), so a rewrite can keep the size while no byte at
   the recorded tail matches, and an append never changes the inode.

WHY BOTH AN INDEX AND A CACHE. The index is the ANSWER (rows, for a cold reader); the
cache is the COST CONTROL (a signature and a resume offset). They are separate files
because a reader that only wants "does this session have code requests?" must not pay
for a cache read, and a scan that is discarded for being stale must not throw away the
last good answer.

CONSTRAINTS.

* **stdlib only.** The index readers include cold-scan paths in the cleanup guards'
  neighbourhood, so this module imports nothing but the standard library plus its own
  package (``refs``/``scan``/``detect`` are themselves stdlib-only).
* **Never raises into a turn.** ``record_event`` is called from a post-tool hook: a
  full disk or a read-only session directory must cost the row, not the turn. Every
  write path swallows ``OSError`` into a warning.
* **Bounded reads.** The scanner streams the journal in chunks and stops at
  :data:`MAX_SCAN_BYTES` past the resume point without a signature match, because a
  cold full scan of the operator's heaviest journal (124 MB) is ~1 s and a rig that
  ran it per poll would be a different problem. The signature check is what makes that
  bound unreachable in normal use.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from local_operator.code_requests.refs import EMPTY_CONTEXT, HostContext, Ref
from local_operator.code_requests.scan import (
    EVENT_CUSTOM_TYPE,
    MAX_EVIDENCE,
    RELATION_ORDER,
    SOURCE_TOOL,
    TOOL_MENTION_CAP,
    ScanResult,
    row_sort_key,
    scan_rows,
)

logger = logging.getLogger(__name__)

#: Subdirectory of the config dir holding one ``<session_id>.json`` per session with
#: rows. Flat, like ``monitors/``: a cold reader must not walk the session store.
INDEX_DIRNAME = "code_requests"

#: Subdirectory of the config CACHE dir holding the incremental scan state.
CACHE_DIRNAME = "code_requests"

#: Bumped on an incompatible change to either document's shape. A reader that does
#: not understand a schema skips that file rather than guessing, and the next scan
#: rewrites it — the ``monitors/store.py`` healing rule.
INDEX_SCHEMA = 1
CACHE_SCHEMA = 1

#: How many collapsed refs a scan keeps, and how many evidence entries a row keeps.
#: The collapsed list is capped so a scan over a 132 MB journal cannot hold thousands
#: of refs in memory; exceeding the cap sets ``tool_only_truncated`` rather than
#: letting a reader mistake the stored length for the exact count. Evidence is capped
#: at the newest entries, because the newest event is the one a reader checks against.

#: How far back the tail probe will look for the last complete row, and how much of
#: a JOURNAL an incremental scan will read before giving up and rescanning whole.
#: 8 MB is far more than a turn's appended rows and far less than a 124 MB journal.
_TAIL_SCAN_LIMIT = 8 << 20

#: The signature's fields. ``inode`` is part of it because ``compact_file`` REPLACES
#: the journal (tmp + ``os.replace``) and an append never changes the inode.
_SIG_KEYS = ("size", "mtime", "inode")

#: The transcript filename inside a session directory. Spelled locally rather than
#: imported from ``session.runtime.engagement``: that import pulls the runtime into
#: every cold reader, and the name is a fact of the store, not of the runtime.
TRANSCRIPT_FILENAME = "transcript.jsonl"

#: How much of a journal a single scan will read past its resume point before it gives
#: up and starts over. 64 MB covers every whole-journal read measured here (the
#: heaviest real journal is 124 MB, read end to end in 1.0 s), so this is a guard
#: against a pathological file, not a routine limit.
MAX_SCAN_BYTES = 64 << 20


#: The journal file this module scans, for one session directory.
def transcript_path(session_dir: Path) -> Path:
    return Path(session_dir) / TRANSCRIPT_FILENAME


def index_dir(config_dir: str | Path) -> Path:
    return Path(config_dir) / INDEX_DIRNAME


def index_path(config_dir: str | Path, session_id: str) -> Path:
    return index_dir(config_dir) / f"{session_id}.json"


def cache_dir(config_dir: str | Path) -> Path:
    return Path(config_dir) / "cache" / CACHE_DIRNAME


def cache_path(config_dir: str | Path, session_id: str) -> Path:
    return cache_dir(config_dir) / f"{session_id}.json"


# ---------------------------------------------------------------------------
# The event row (the one authoritative record)
# ---------------------------------------------------------------------------


def build_event(
    *,
    kind: str,
    ref: Ref,
    evidence: Mapping[str, Any],
    act: str | None = None,
    via: Mapping[str, Any] | None = None,
    at: float | None = None,
) -> dict[str, Any]:
    """The ``code_request_event.v1`` details payload for one fact.

    ``evidence`` is not decoration: every row's ``opened`` must be auditable back to
    the exact rule and command that produced it (design risk 1 — a false "opened" is
    the failure mode that would make the whole surface untrustworthy). It is written
    once, here, so the hook and the backfill cannot disagree about what an event says.
    """
    details: dict[str, Any] = {
        "v": 1,
        "kind": kind,
        "ref": ref.to_payload(),
        "evidence": dict(evidence),
        "at": round(at if at is not None else time.time(), 3),
    }
    if act:
        details["act"] = act
    if via:
        details["via"] = dict(via)
    return details


async def append_event(transcript: Any, details: Mapping[str, Any]) -> bool:
    """Append one event row through the session's transcript. True when it landed.

    ``transcript`` is duck-typed (anything with the ``append_custom`` coroutine), so
    this module never imports the session package: the same reason
    ``monitors/store.py`` is stdlib-only. A failure is logged and reported as False;
    the caller is a hook and must not fail a turn over bookkeeping.
    """
    append = getattr(transcript, "append_custom", None)
    if append is None:
        return False
    try:
        await append(EVENT_CUSTOM_TYPE, dict(details))
        return True
    except Exception:  # noqa: BLE001 - bookkeeping never breaks a turn
        logger.warning("could not append a code-request event", exc_info=True)
        return False


def event_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """The event payloads in a stream of transcript rows, in order."""
    for entry in rows:
        if not isinstance(entry, Mapping) or entry.get("type") != "custom":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("custom_type") == EVENT_CUSTOM_TYPE:
            yield payload


# ---------------------------------------------------------------------------
# Reading and writing the two derived files
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("code-request store: unreadable file %s; treating as absent", path)
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: Mapping[str, Any]) -> bool:
    """Atomic write (pid-suffixed temp + replace). False when it could not land.

    The PID in the temp name is ``session/transcript_index.py``'s measured remedy:
    two processes building one session wrote the same fixed temp path and produced
    torn documents.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        logger.warning("code-request store: could not write %s", path, exc_info=True)
        return False


def read_index(config_dir: str | Path, session_id: str) -> dict[str, Any] | None:
    """One session's derived index, or ``None`` when absent/unreadable/stale-schema."""
    data = _read_json(index_path(config_dir, session_id))
    if data is None or data.get("schema") != INDEX_SCHEMA:
        return None
    return data


def write_index(config_dir: str | Path, session_id: str, result: ScanResult) -> bool:
    """Persist one scan's rows. An empty result REMOVES the file.

    Empty means "this session has no code requests", and a file saying so costs a
    reader the same stat as an absent one while adding a second thing to keep in
    step. The monitor index makes the same call for an emptied list.
    """
    path = index_path(config_dir, session_id)
    if not result.rows and not result.tool_output_only and not result.hints:
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False
    rows = [row.to_payload() for row in result.rows]
    payload: dict[str, Any] = {
        "schema": INDEX_SCHEMA,
        "session_id": session_id,
        "updated_at": round(time.time(), 3),
        "rows": rows,
        # The COLLAPSED refs are deliberately absent. They are the expansion a reader
        # asks for by name ("show the tool-output mentions"), and this file exists to be
        # read cold and cheap for every session on the machine; the count rides along so
        # a list renders "67 more seen in tool output" without opening the cache.
        "tool_output_only": result.tool_output_only,
        "hints": list(result.hints),
        "events": result.events,
    }
    if result.tool_only_truncated:
        payload["tool_only_truncated"] = True
    return _write_json(path, payload)


def read_cache(config_dir: str | Path, session_id: str) -> dict[str, Any] | None:
    data = _read_json(cache_path(config_dir, session_id))
    if data is None or data.get("schema") != CACHE_SCHEMA:
        return None
    return data


def write_cache(
    config_dir: str | Path, session_id: str, sig: Mapping[str, Any], result: ScanResult
) -> bool:
    return _write_json(
        cache_path(config_dir, session_id),
        {
            "schema": CACHE_SCHEMA,
            "session_id": session_id,
            "updated_at": round(time.time(), 3),
            "sig": dict(sig),
            "scan": result.to_payload(),
        },
    )


class RowReader:
    """Streams a journal's rows from ``start`` and records where it stopped.

    ``last_end`` is the byte offset just past the last COMPLETE row, and ``last_id``
    that row's id: together they are the resume point a later scan verifies before it
    trusts the file as a prefix of what is there now (``_verify_tail``). A torn final
    line (a row being written while we read) sets :attr:`torn` and is NOT counted, so
    the recorded offset always ends on a row boundary.
    """

    def __init__(self, path: Path, start: int = 0) -> None:
        self.path = path
        self.start = start
        self.last_end = start
        self.last_id = ""
        self.first_id = ""
        self.rows = 0
        self.torn = False

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.start)
            offset = self.start
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith("\n"):
                    # The writer is mid-append: stop before this fragment, so the
                    # next scan re-reads the completed row rather than losing it.
                    self.torn = True
                    break
                offset = position + len(line.encode("utf-8", errors="replace"))
                stripped = line.strip()
                if not stripped:
                    self.last_end = offset
                    continue
                try:
                    row = json.loads(stripped)
                except ValueError:
                    # A malformed line is skipped (the transcript reader's own rule)
                    # but still counts as covered: re-reading it every scan would
                    # make a corrupt byte a permanent cost.
                    self.last_end = offset
                    continue
                if not isinstance(row, dict):
                    self.last_end = offset
                    continue
                self.last_end = offset
                self.last_id = str(row.get("id", ""))
                if not self.first_id:
                    self.first_id = self.last_id
                self.rows += 1
                yield row


def iter_rows(path: Path, start: int = 0) -> Iterator[Mapping[str, Any]]:
    """Stream a transcript's rows from ``start``, dropping malformed lines individually.

    Line-by-line rather than a whole-file ``read_text``: the heaviest real journal is
    124 MB, and a cold scan must not hold it twice.
    """
    yield from RowReader(path, start)


def _verify_tail(path: Path, end: int, last_id: str) -> bool:
    """Whether the row ending at ``end`` is still the one the cache recorded.

    The append-only cheap check, ``transcript_index._verify_tail``'s rule: the
    recorded coverage end must still end a complete row, and that row must still parse
    to ``last_id``. Anything else — a compaction rewrite, a truncation, a torn tail —
    means the file is not a prefix of what is there now, and the caller rescans whole.
    """
    if end <= 0 or not last_id:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                return False
            position = end - 1
            start: int | None = None
            scanned = 0
            while position > 0 and scanned <= _TAIL_SCAN_LIMIT:
                step = min(position, 1 << 20)
                position -= step
                scanned += step
                handle.seek(position)
                found = handle.read(step).rfind(b"\n")
                if found >= 0:
                    start = position + found + 1
                    break
            if start is None:
                start = 0
            handle.seek(start)
            raw = handle.read(end - 1 - start)
    except OSError:
        return False
    try:
        entry = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return False
    return isinstance(entry, dict) and str(entry.get("id", "")) == last_id


def _sig_of(stat: os.stat_result, last_id: str, offset: int) -> dict[str, Any]:
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "inode": stat.st_ino,
        "last_id": last_id,
        "offset": offset,
    }
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "inode": stat.st_ino,
        "last_id": last_id,
    }


def sig_matches(sig: Mapping[str, Any] | None, stat: os.stat_result) -> bool:
    """Whether a recorded signature is exactly current. The inode is load-bearing."""
    if not isinstance(sig, Mapping):
        return False
    return (
        sig.get("size") == stat.st_size
        and sig.get("mtime") == stat.st_mtime
        and sig.get("inode") == stat.st_ino
    )


def _last_row_probe(path: Path, *, limit: int = _TAIL_SCAN_LIMIT) -> tuple[str, int]:
    """``(last complete row's id, byte offset past it)``, read from the file's tail.

    Bounded, and that bound is what keeps a signature check cheap on a 124 MB
    journal: only the last ``limit`` bytes are read. A segment whose final line is
    torn (the writer mid-append) yields the LAST COMPLETE row before it, so the
    recorded resume point always lands on a row boundary — the next scan picks the
    rest up. ``("", 0)`` on any failure, which makes the caller rescan whole rather
    than trust a resume point it could not verify.
    """
    try:
        size = path.stat().st_size
        start = max(0, size - limit)
        with path.open("rb") as handle:
            handle.seek(start)
            blob = handle.read()
    except OSError:
        return "", 0
    # Walk the segment's lines, remembering the last one that is complete and parses.
    end = start
    last_id = ""
    for chunk in blob.split(b"\n")[:-1]:
        end += len(chunk) + 1
        try:
            row = json.loads(chunk.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if isinstance(row, dict):
            last_id = str(row.get("id", ""))
    if start > 0 and last_id:
        # The segment began mid-line; its first partial line was skipped above, so
        # ``end`` is still a true row boundary. Only an EMPTY result is unusable.
        return last_id, end
    return (last_id, end) if start == 0 else ("", 0)


def scan_file(
    path: Path,
    *,
    context: HostContext = EMPTY_CONTEXT,
    mcp_servers: Iterable[Any] = (),
    forked_at: float | None = None,
    parent_id: str | None = None,
) -> ScanResult:
    """A whole-journal scan of one transcript file. Blocking; run it off the loop.

    This is the cold path. :func:`refresh` is the routine one, and it is incremental
    on the recorded signature.
    """
    return scan_rows(
        iter_rows(path),
        context,
        mcp_servers=mcp_servers,
        forked_at=forked_at,
        parent_id=parent_id,
    )


@dataclass
class RefreshOutcome:
    """What one refresh did, so a caller can report or test the work honestly."""

    result: ScanResult
    rescanned: bool
    signature: dict[str, Any] = field(default_factory=dict)


def refresh(
    config_dir: str | Path,
    session_id: str,
    session_dir: Path,
    *,
    context: HostContext = EMPTY_CONTEXT,
    mcp_servers: Iterable[Any] = (),
    forked_at: float | None = None,
    parent_id: str | None = None,
    force: bool = False,
) -> RefreshOutcome | None:
    """Bring one session's index up to date. Blocking. ``None`` when there is no journal.

    The invalidation ladder, ``session/transcript_index.py``'s and for its reasons:

    1. an exact ``(size, mtime, inode)`` match serves the cached result — the routine
       case, one stat and one small JSON read;
    2. a GROWN file on the SAME inode whose recorded tail still verifies
       (``_verify_tail``) is scanned INCREMENTALLY: only the appended region is read,
       and its rows are merged into the cached ones (see :func:`_merge`), which is
       what makes a poll over a 124 MB journal cost the last turn rather than the
       journal;
    3. anything else — a compact rewrite (new inode), a truncation, a tail that no
       longer parses — rescans whole. A compaction is exactly the case the whole scan
       exists for: it can move rows, and only a full pass sees the new arrangement.

    The appended region is capped at :data:`MAX_SCAN_BYTES`; past that the whole scan
    is cheaper than the merge bookkeeping it would need.
    """
    path = transcript_path(session_dir)
    try:
        stat = path.stat()
    except OSError:
        return None
    cached = read_cache(config_dir, session_id)
    if not force and cached is not None and sig_matches(cached.get("sig"), stat):
        payload = cached.get("scan")
        if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
            return RefreshOutcome(_result_from_payload(payload), False, dict(cached["sig"]))
    previous = cached if (cached is not None and not force) else None
    if previous is not None and _can_increment(previous, stat):
        outcome = _refresh_incremental(
            config_dir,
            session_id,
            path,
            previous,
            context,
            mcp_servers,
            forked_at,
            parent_id,
        )
        if outcome is not None:
            return outcome
    result = scan_file(
        path,
        context=context,
        mcp_servers=mcp_servers,
        forked_at=forked_at,
        parent_id=parent_id,
    )
    return _publish(config_dir, session_id, path, result)


def _can_increment(previous: Mapping[str, Any], stat: os.stat_result) -> bool:
    """Whether the cached scan can be extended instead of rebuilt."""
    sig = previous.get("sig")
    if not isinstance(sig, Mapping):
        return False
    if sig.get("inode") != stat.st_ino or stat.st_size <= int(sig.get("offset") or 0):
        return False
    if stat.st_size - int(sig.get("offset") or 0) > MAX_SCAN_BYTES:
        return False
    scan = previous.get("scan")
    return isinstance(scan, Mapping) and isinstance(scan.get("rows"), list)


def _refresh_incremental(
    config_dir: str | Path,
    session_id: str,
    path: Path,
    previous: Mapping[str, Any],
    context: HostContext,
    mcp_servers: Iterable[Any],
    forked_at: float | None,
    parent_id: str | None,
) -> RefreshOutcome | None:
    """Scan only the appended region and merge it into the cached rows.

    Returns ``None`` when the resume point cannot be trusted, which sends the caller
    to the whole-journal path. The verification is the load-bearing part: without it a
    compaction that kept the file's size would have this function append new bytes
    onto rows derived from the OLD arrangement.
    """
    sig = previous.get("sig")
    assert isinstance(sig, Mapping)  # _can_increment established this
    offset = int(sig.get("offset") or 0)
    if not _verify_tail(path, offset, str(sig.get("last_id") or "")):
        return None
    prior = _result_from_payload(previous.get("scan") or {})
    reader = RowReader(path, offset)
    try:
        tail = scan_rows(
            reader,
            context,
            mcp_servers=mcp_servers,
            forked_at=forked_at,
            parent_id=parent_id,
        )
    except OSError:
        return None
    merged = _merge(prior, tail)
    # The READER's own boundary, not a fresh probe: it stopped exactly where the bytes
    # ended, and on a torn tail it stopped BEFORE the incomplete row, so this offset is
    # a row boundary whose successor is re-read next time. Recording the stat's tail
    # instead would claim bytes this scan did not cover.
    if reader.last_id:
        return _publish(
            config_dir,
            session_id,
            path,
            merged,
            override=(reader.last_id, reader.last_end),
        )
    last_id, end = _last_row_probe(path)
    return _publish(
        config_dir, session_id, path, merged, override=(last_id, end) if last_id else None
    )


def _publish(
    config_dir: str | Path,
    session_id: str,
    path: Path,
    result: ScanResult,
    *,
    override: tuple[str, int] | None = None,
) -> RefreshOutcome | None:
    """Write the cache and the index for one finished scan, and return its outcome."""
    try:
        stat = path.stat()
    except OSError:
        return None
    last_id, end = override if override is not None else _last_row_probe(path)
    signature = _sig_of(stat, last_id, end)
    write_cache(config_dir, session_id, signature, result)
    write_index(config_dir, session_id, result)
    return RefreshOutcome(result, True, signature)


def _publish_sig(
    config_dir: str | Path,
    session_id: str,
    result: ScanResult,
    stat: os.stat_result,
    last_id: str,
    end: int,
) -> dict[str, Any]:
    """The torn-read arm of :func:`_refresh_incremental`: publish, return the signature."""
    signature = _sig_of(stat, last_id, end)
    write_cache(config_dir, session_id, signature, result)
    write_index(config_dir, session_id, result)
    return signature


def _merge(prior: ScanResult, tail: ScanResult) -> ScanResult:
    """Merge an incremental tail into the cached result, per row, by key.

    THE RULES, and why each is a union rather than a replacement:

    * ``mentions`` are SUMMED per source with first/last timestamps min/maxed, because
      a count is a fact about the whole session and the tail only saw its part;
    * ``relations`` and ``acts`` union, so a row first mentioned in the head and opened
      in the tail reads as opened;
    * ``evidence`` concatenates, capped at the LAST :data:`MAX_EVIDENCE` entries — the
      newest events are what a reader checks against, and a row acted on nightly for a
      month must not grow without limit;
    * ``unknown_reason``/``via``/``inherited_from`` keep the first non-empty value;
      they describe how the row was FIRST seen, and a later mention cannot revise that.

    Collapsing is then RE-DERIVED over the whole merged set rather than counted on each
    side and added up. That is the choice worth stating: a row's bucket is a function of
    its own accumulated state (no relations, and no source other than tool output), so
    recomputing it is exact by construction, where arithmetic across two scans would
    have to keep the previous list untruncated to stay honest. A row the tail saw only
    in tool output therefore lands collapsed, and a head-collapsed row the tail mentions
    in the user's own text moves OUT of the collapsed bucket and becomes visible — the
    behaviour a reader expects from a session that finally talks about a PR it had only
    ever listed.
    """
    result = ScanResult()
    result.hints = list(prior.hints) + list(tail.hints)
    result.events = prior.events + tail.events
    merged: dict[str, Any] = {}
    for row in [*prior.rows, *prior.tool_only_rows, *tail.rows, *tail.tool_only_rows]:
        target = merged.get(row.ref.key)
        if target is None:
            merged[row.ref.key] = row
            continue
        for source, mention in row.mentions.items():
            existing = target.mentions.get(source)
            if existing is None:
                target.mentions[source] = mention
                continue
            existing.count += mention.count
            if mention.first_at and (not existing.first_at or mention.first_at < existing.first_at):
                existing.first_at = mention.first_at
            if mention.last_at > existing.last_at:
                existing.last_at = mention.last_at
        target.relations |= row.relations
        for act in row.acts:
            if act not in target.acts:
                target.acts.append(act)
        target.evidence = (target.evidence + row.evidence)[-MAX_EVIDENCE:]
        target.unknown_reason = target.unknown_reason or row.unknown_reason
        target.via = target.via or row.via
        target.inherited_from = target.inherited_from or row.inherited_from
        if row.first_at and (not target.first_at or row.first_at < target.first_at):
            target.first_at = row.first_at
        target.last_at = max(target.last_at, row.last_at)
        target.relation = next(
            (item for item in RELATION_ORDER if item in target.relations), target.relation
        )
    for row in merged.values():
        if row.relations or set(row.mentions) - {SOURCE_TOOL}:
            result.rows.append(row)
            continue
        result.tool_only_rows.append(row)
    result.tool_output_only = len(result.tool_only_rows)
    if result.tool_output_only > TOOL_MENTION_CAP:
        # The flag, not a silently wrong count: a reader is told the list is partial
        # rather than being handed a number it cannot reconcile with what it can expand.
        result.tool_only_truncated = True
        result.tool_only_rows = result.tool_only_rows[:TOOL_MENTION_CAP]
    result.rows.sort(key=row_sort_key)
    result.tool_only_rows.sort(key=row_sort_key)
    return result


def _result_from_payload(payload: Mapping[str, Any]) -> ScanResult:
    """Rebuild a :class:`ScanResult` from a cache document, tolerating junk rows."""
    from local_operator.code_requests.scan import Mention, Row

    def row_from(raw: Any) -> "Row | None":
        if not isinstance(raw, Mapping):
            return None
        ref = Ref.from_payload(raw.get("ref"))
        if ref is None:
            return None
        row = Row(ref=ref)
        row.relation = str(raw.get("relation") or row.relation)
        relations = raw.get("relations")
        row.relations = {str(item) for item in relations} if isinstance(relations, list) else set()
        acts = raw.get("acts")
        row.acts = [str(item) for item in acts] if isinstance(acts, list) else []
        mentions = raw.get("mentions")
        if isinstance(mentions, list):
            for item in mentions:
                if not isinstance(item, Mapping):
                    continue
                source = str(item.get("source") or "")
                if not source:
                    continue
                mention = Mention(source=source)
                mention.count = int(item.get("count") or 0)
                mention.first_at = float(item.get("first_at") or 0.0)
                mention.last_at = float(item.get("last_at") or 0.0)
                row.mentions[source] = mention
        row.first_at = float(raw.get("first_at") or 0.0)
        row.last_at = float(raw.get("last_at") or 0.0)
        evidence = raw.get("evidence")
        row.evidence = [dict(item) for item in evidence] if isinstance(evidence, list) else []
        via = raw.get("via")
        row.via = dict(via) if isinstance(via, Mapping) else None
        inherited = raw.get("inherited_from")
        row.inherited_from = str(inherited) if isinstance(inherited, str) else None
        reason = raw.get("unknown_reason")
        row.unknown_reason = str(reason) if isinstance(reason, str) else None
        return row

    result = ScanResult()
    for raw in payload.get("rows") or ():
        row = row_from(raw)
        if row is not None:
            result.rows.append(row)
    for raw in payload.get("tool_only_rows") or ():
        row = row_from(raw)
        if row is not None:
            result.tool_only_rows.append(row)
    result.tool_output_only = int(payload.get("tool_output_only") or 0)
    result.tool_only_truncated = bool(payload.get("tool_only_truncated"))
    hints = payload.get("hints")
    result.hints = [dict(item) for item in hints] if isinstance(hints, list) else []
    result.events = int(payload.get("events") or 0)
    return result


async def refresh_async(*args: Any, **kwargs: Any) -> RefreshOutcome | None:
    """``refresh`` on a worker thread, for callers on an event loop.

    One hop, so a route handler never blocks the loop on a 124 MB scan; the daemon's
    own poller is per-session and this is called at most once per changed journal.
    """
    return await asyncio.to_thread(refresh, *args, **kwargs)


def index_sessions(
    config_dir: str | Path,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Every readable index entry keyed by session id, plus whether listing failed.

    ``(entries, read_error)`` — the ``monitors/store.py`` distinction: "no session has
    code requests" and "we cannot tell" are different answers to a listing route.
    """
    directory = index_dir(config_dir)
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return {}, False
    except OSError:
        logger.warning("code-request index: cannot list %s", directory)
        return {}, True
    entries: dict[str, dict[str, Any]] = {}
    for name in names:
        if not name.endswith(".json") or name.startswith("."):
            continue
        session_id = name[: -len(".json")]
        entry = read_index(config_dir, session_id)
        if entry is None:
            continue
        entries[session_id] = entry
    return entries, False


def drop_session(config_dir: str | Path, session_id: str) -> None:
    """Remove both derived files. Best-effort; used when a session is deleted."""
    for path in (index_path(config_dir, session_id), cache_path(config_dir, session_id)):
        try:
            path.unlink()
        except OSError:
            continue


__all__ = [
    "CACHE_DIRNAME",
    "CACHE_SCHEMA",
    "INDEX_DIRNAME",
    "INDEX_SCHEMA",
    "MAX_SCAN_BYTES",
    "RefreshOutcome",
    "TRANSCRIPT_FILENAME",
    "append_event",
    "build_event",
    "cache_path",
    "drop_session",
    "event_rows",
    "index_path",
    "index_sessions",
    "iter_rows",
    "read_cache",
    "read_index",
    "refresh",
    "refresh_async",
    "scan_file",
    "sig_matches",
    "transcript_path",
    "write_cache",
    "write_index",
]
