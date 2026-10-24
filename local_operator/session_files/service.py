"""Read side of the per-session file index: merge, filter, sort, page, ``stat``, materialise.

``scan.py`` says what each journal produced and ``ledger.py`` keeps that cheap; this module
answers the question a surface actually asks - *show me this conversation's files, sorted and
paged* - and does the one write the feature owns (copying a digest-only artifact to a real
file on request).

NO HTTP COUPLING, AND WHY. The desktop route is the first caller, not the only one: the mobile
daemon and the TUI are expected to reuse :func:`list_files` rather than grow a second index.
Two seams make that a flag instead of a rewrite:

* ``display=True`` projects every path to the form that may leave this machine - cwd-relative,
  else ``~/``-relative, never absolute (memo S-R14, the Highlights rule). The desktop route
  sends absolute paths (the bearer-gated renderer already holds them); a mesh or phone surface
  MUST pass ``display=True``. Nothing else on a row carries a path (artifact rows are digests;
  ``intent`` is the call's sanitised one-line ``i``).
* The family (self + contained subagents) is an INPUT (:class:`Family`), so a caller with its
  own proof of containment (the desktop route uses the roster proof in
  ``server/utils/desktop_sessions``) supplies it; this module never decides who is a child.

THE ROW IS A MERGE OF THREE THINGS, each with a different freshness:

1. the SCAN's record (what the transcript says happened - authoritative, cached, revisioned);
2. Highlights callouts, overlaid at read time from the newest ``supplement_v1`` version per
   anchor (so a regenerated callout can retract an older one - which a flag baked into the
   record at scan time could not);
3. a ``stat`` of the file *now* (size, mtime, birth, existence) - page rows only, plus every
   candidate row when the caller sorts by a stat-derived column, under a time budget. The
   revision does NOT move when a file changes on disk; a client that wants fresh stats
   re-requests the page.

SORTS. ``recent`` (the default) orders by when the SESSION last touched the file - an event
time recorded in the journal, so it is stable and free. ``modified``/``created``/``size`` order
by the values the rows DISPLAY (stat, falling back to the recorded event time/size for a file
that is gone), so the list can never sort differently from what the user reads; that costs one
``stat`` per candidate row, which is why it is budgeted (:data:`SORT_STAT_BUDGET_S`) and
reported (``stat_complete``) rather than assumed. ``name`` and ``kind`` need no I/O.

CURSORS are opaque, bound to the query that minted them, and to the revision they were minted
at. A cursor from a DIFFERENT query is a client bug (``InvalidCursor``, 422); a cursor from an
older revision is the normal race (a turn settled between pages) and restarts at the top with
``cursor_reset: true`` so the client re-renders instead of showing a page that skipped rows.
"""

from __future__ import annotations

import base64
import binascii
import errno
import hashlib
import json
import logging
import mimetypes
import os
import re
import stat as stat_module
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Sequence

from local_operator.session_files import ledger
from local_operator.session_files.scan import ORIGINS, FileRecord, resolve_path
from local_operator.supplements.candidates import kind_of

logger = logging.getLogger(__name__)

SCOPES: Final = ("family", "self")
SORTS: Final = ("recent", "name", "kind", "modified", "created", "size")
ORDERS: Final = ("asc", "desc")

#: The order a sort uses when the caller does not say. Newest/biggest first for the quantities
#: people scan for "what just happened"; A-Z for the labels.
DEFAULT_ORDER: Final = {
    "recent": "desc",
    "name": "asc",
    "kind": "asc",
    "modified": "desc",
    "created": "desc",
    "size": "desc",
}

DEFAULT_LIMIT: Final = 100
MAX_LIMIT: Final = 200
MAX_QUERY_CHARS: Final = 256

#: How long one GET may spend advancing cold scans before it answers ``building`` with what it
#: has (the design's ~1.5 s, ``find``'s precedent). The scan CONTINUES in the background.
DEFAULT_BUDGET_S: Final = 1.5

#: Time a stat-derived sort may spend ``stat``-ing candidate rows. Local APFS does ~100k stats/s,
#: so this covers the 50k-record cap with room; the budget exists for a network mount, where
#: one stalled stat must not stall the page. Rows not reached sort by their recorded values.
SORT_STAT_BUDGET_S: Final = 0.75

#: Seconds a ``stat`` answer is reused. Paging a stat-sorted list re-asks for the same files on
#: every page; without this each page would re-stat the whole candidate set.
STAT_TTL_S: Final = 5.0
_STAT_CACHE_CAP: Final = 100_000

_ID_RE: Final = re.compile(r"^[a-f0-9]{16}$")
_TOKEN_RE: Final = re.compile(r"^[a-z0-9_]{1,24}$")
_MAX_TOKENS: Final = 32


class InvalidCursor(ValueError):
    """The cursor is malformed or belongs to a different query (a client bug, HTTP 422)."""


class InvalidFilter(ValueError):
    """A filter value is outside its vocabulary (HTTP 422)."""


class MaterialiseError(Exception):
    """A materialise request that cannot be honoured; ``code`` is the wire machine token."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class ChildRef:
    """One PROVEN subagent session. Proof of containment is the caller's job (see module doc)."""

    child_id: str
    label: str | None
    depth: int
    session_dir: Path


@dataclass(frozen=True)
class Family:
    """A conversation plus the children it is allowed to list."""

    session_id: str
    session_dir: Path
    children: tuple[ChildRef, ...] = ()
    #: Children dropped by a cap or refused by the proof; reported as ``children_skipped``.
    skipped: int = 0


@dataclass(frozen=True)
class Query:
    scope: str = "family"
    kinds: frozenset[str] = frozenset()
    origins: frozenset[str] = frozenset()
    highlighted: bool | None = None
    q: str = ""
    sort: str = "recent"
    order: str | None = None
    limit: int = DEFAULT_LIMIT
    cursor: str | None = None

    @property
    def effective_order(self) -> str:
        return self.order or DEFAULT_ORDER[self.sort]


def parse_csv(raw: str | None, *, vocabulary: Iterable[str] | None = None) -> frozenset[str]:
    """A comma list as a set. Tokens are lower-snake words; ``vocabulary`` closes the set."""
    if not raw:
        return frozenset()
    tokens = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if len(tokens) > _MAX_TOKENS or any(not _TOKEN_RE.match(token) for token in tokens):
        raise InvalidFilter(raw[:80])
    if vocabulary is not None:
        allowed = set(vocabulary)
        unknown = [token for token in tokens if token not in allowed]
        if unknown:
            raise InvalidFilter(",".join(unknown)[:80])
    return frozenset(tokens)


# ---------------------------------------------------------------------------
# Paths, cwd, denylist
# ---------------------------------------------------------------------------


def session_cwd(config_dir: str | Path, session_id: str, session_dir: Path) -> str:
    """Where a session was working, best effort; ``""`` when nothing records it.

    ``desktop.json`` is the durable record the desktop door writes at create; the code-request
    ledger's runtime-record / wake-index fallback covers sessions started from a terminal.
    Subagent sessions usually have neither - the caller then inherits the parent's cwd.
    """
    try:
        with (session_dir / "desktop.json").open("r", encoding="utf-8", errors="replace") as handle:
            marker = json.load(handle)
        cwd = marker.get("cwd") if isinstance(marker, dict) else None
        if isinstance(cwd, str) and os.path.isabs(cwd):
            return cwd
    except (OSError, ValueError):
        pass
    try:
        from local_operator.code_requests import ledger as code_ledger

        return code_ledger.session_cwd(config_dir, session_id)
    except Exception:  # noqa: BLE001 - an unknown cwd only costs relative-path resolution
        return ""


def display_path(absolute: str, cwd: str, home: str) -> str:
    """The form of ``absolute`` that may leave this machine: cwd-relative, else ``~/``, else name.

    Never absolute. Differs from ``candidates._display_path`` in the last arm: that helper
    assumes the path is under cwd or home and would emit ``~/../..`` for anything else.
    """
    for root, prefix in ((cwd, ""), (home, "~/")):
        if root and (absolute == root or absolute.startswith(root.rstrip(os.sep) + os.sep)):
            return prefix + os.path.relpath(absolute, root)
    return os.path.basename(absolute) or absolute


def make_deny(config_dir: str | Path) -> Callable[[str], bool]:
    """The sensitive-path predicate, tuned for an INVENTORY rather than a preview.

    The Highlights denylist (``supplements.denylist``) denies the whole config directory and
    the scratchpad - correct for "offer this file to a vendor", wrong here: a session's own
    ``scratchpad://`` notes (and the artifacts this service materialises) live under
    ``<config>/sessions/<id>/scratchpad`` and are exactly what the user wants listed. So a path
    inside a session scratchpad is re-judged WITHOUT the config-dir prefix (credential-shaped
    names such as ``.env`` or ``id_rsa`` stay denied); every other config-dir path (the secret
    store, ``master.key``, ``audit.log``) stays denied by the original rule.
    """
    from local_operator.supplements import denylist

    sessions_root = os.path.join(os.path.realpath(str(config_dir)), "sessions") + os.sep
    sessions_raw = os.path.join(os.path.abspath(str(config_dir)), "sessions") + os.sep

    def _scratch_relative(path: str) -> str | None:
        for root in (sessions_root, sessions_raw):
            if path.startswith(root):
                parts = path[len(root) :].split(os.sep)
                if len(parts) >= 3 and parts[1] == "scratchpad":
                    return os.sep + os.path.join("scratchpad", *parts[2:])
        return None

    def deny(path: str) -> bool:
        # Exempt the scratchpad independently in BOTH spellings. Exempting only the
        # written name would launder a harmless report.txt link to config/master.key.
        for spelling in (path, os.path.realpath(path)):
            rule = denylist.is_sensitive(spelling)
            if not rule:
                continue
            relative = _scratch_relative(spelling)
            if rule in (denylist.RULE_CONFIG_DIR, denylist.RULE_SCRATCHPAD) and relative:
                if not denylist.is_sensitive(relative):
                    continue
            return True
        return False

    return deny


# ---------------------------------------------------------------------------
# stat
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Facts:
    exists: bool
    size: int | None
    mtime: float | None
    created: float | None


_MISSING: Final = Facts(False, None, None, None)
_STAT_CACHE: dict[str, tuple[float, Facts]] = {}
_STAT_GUARD = threading.Lock()


def stat_path(path: str, now: float | None = None) -> Facts:
    """``stat`` one path through a short-lived cache. Never raises."""
    clock = time.monotonic() if now is None else now
    held = _STAT_CACHE.get(path)
    if held is not None and clock - held[0] < STAT_TTL_S:
        return held[1]
    try:
        info = os.stat(path)
    except (OSError, ValueError):
        facts = _MISSING
    else:
        birth = getattr(info, "st_birthtime", None)
        facts = Facts(
            True,
            info.st_size if stat_module.S_ISREG(info.st_mode) else None,
            info.st_mtime,
            birth if isinstance(birth, (int, float)) else None,
        )
    with _STAT_GUARD:
        if len(_STAT_CACHE) >= _STAT_CACHE_CAP:
            _STAT_CACHE.clear()
        _STAT_CACHE[path] = (clock, facts)
    return facts


def forget_stats() -> None:
    """Drop the stat cache. Tests only."""
    with _STAT_GUARD:
        _STAT_CACHE.clear()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    """One row before projection: the winning record plus everything merged into it."""

    record: FileRecord
    session_id: str
    child: ChildRef | None
    cwd: str
    touches: int
    last_ts: float
    last_seq: int
    first_ts: float
    first_seq: int
    highlighted: bool = False
    highlight_why: str | None = None
    #: Filled during sorting/paging.
    facts: Facts | None = None

    @property
    def row_id(self) -> str:
        return _row_id(self.session_id, self.record.key)


def _row_id(session_id: str, key: str) -> str:
    """Stable across re-scans: derived from WHO produced the file and WHAT it is."""
    return hashlib.sha1(f"{session_id}\0{key}".encode("utf-8")).hexdigest()[:16]


@dataclass
class Merged:
    entries: list[Entry]
    revision: int
    complete: bool
    rows_scanned: int
    covered_bytes: int
    total_bytes: int
    truncated: bool
    snapshots: list[tuple[str, ChildRef | None, ledger.Snapshot]] = field(default_factory=list)


def _members(family: Family, scope: str) -> list[tuple[str, ChildRef | None, Path]]:
    members: list[tuple[str, ChildRef | None, Path]] = [
        (family.session_id, None, family.session_dir)
    ]
    if scope == "family":
        members.extend((child.child_id, child, child.session_dir) for child in family.children)
    return members


def advance(
    config_dir: str | Path,
    family: Family,
    scope: str,
    *,
    budget_s: float | None,
    home: str,
    deny: Callable[[str], bool] | None,
) -> list[tuple[str, ChildRef | None, str, ledger.Snapshot | None]]:
    """Advance every member's scan, sharing ONE time budget. Blocking.

    Returns ``(session_id, child, cwd, snapshot)`` per member; ``snapshot`` is ``None`` for a
    member with no journal. A member reached after the budget is spent still gets a zero-budget
    call: an already-complete cache answers it for free and an incomplete one makes a little
    progress, so a family of many children converges across polls instead of starving the
    tail.
    """
    deadline = None if budget_s is None else time.monotonic() + budget_s
    parent_cwd = session_cwd(config_dir, family.session_id, family.session_dir)
    out: list[tuple[str, ChildRef | None, str, ledger.Snapshot | None]] = []
    for session_id, child, directory in _members(family, scope):
        cwd = parent_cwd
        if child is not None:
            cwd = session_cwd(config_dir, session_id, directory) or parent_cwd
        remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        snapshot = ledger.refresh(
            config_dir,
            session_id,
            directory,
            cwd=cwd,
            home=home,
            deny=deny,
            budget_s=remaining,
        )
        out.append((session_id, child, cwd, snapshot))
    return out


def complete_in_background(
    config_dir: str | Path,
    family: Family,
    scope: str,
    *,
    home: str,
    deny: Callable[[str], bool] | None,
    slice_s: float = 1.0,
) -> None:
    """Drive every member's scan to completion in short slices. Blocking; run in a thread.

    SLICED, not one long ``refresh``: each slice publishes its partial scan, so a client polling
    a ``building`` answer sees rows appear, and the per-session lock is released between slices
    so a foreground GET never queues behind a 300 MB scan.
    """
    for _ in range(100_000):
        started = time.monotonic()
        states = advance(config_dir, family, scope, budget_s=slice_s, home=home, deny=deny)
        if all(snapshot is None or snapshot.complete for *_, snapshot in states):
            return
        if time.monotonic() - started < 0.01:
            # A foreground slice owns the nonblocking ledger lock; don't busy-spin.
            time.sleep(0.01)


_FAMILY_REVISION_LOCK = threading.Lock()
_FAMILY_REVISIONS: OrderedDict[Any, Any] = OrderedDict()


def merge(
    states: Sequence[tuple[str, ChildRef | None, str, ledger.Snapshot | None]],
    *,
    home: str,
    deny: Callable[[str], bool] | None = None,
    revision_scope: str = "family",
) -> Merged:
    """Fold the members' scans into ONE entry per file, overlay Highlights, report progress."""
    entries: dict[str, Entry] = {}
    revision = 0
    complete = True
    rows = covered = total = 0
    truncated = False
    snapshots: list[tuple[str, ChildRef | None, ledger.Snapshot]] = []
    for session_id, child, cwd, snapshot in states:
        if snapshot is None:
            continue
        snapshots.append((session_id, child, snapshot))
        revision = max(revision, snapshot.revision)
        complete = complete and snapshot.complete
        rows += snapshot.scan.rows
        covered += snapshot.covered_bytes
        total += snapshot.total_bytes
        truncated = truncated or snapshot.scan.truncated
        for record in snapshot.scan.records.values():
            held = entries.get(record.key)
            if held is None:
                entries[record.key] = Entry(
                    record=record,
                    session_id=session_id,
                    child=child,
                    cwd=cwd,
                    touches=record.touches,
                    last_ts=record.last_ts,
                    last_seq=record.last_seq,
                    first_ts=record.first_ts,
                    first_seq=record.first_seq,
                )
                continue
            # The same file reached by two members (a child wrote it, the parent edited it):
            # ONE row, owned by whoever touched it first - deterministic, so the id is stable
            # once both scans are complete - with the counters and recency summed/maxed.
            held.touches += record.touches
            if (record.last_ts, record.last_seq) > (held.last_ts, held.last_seq):
                held.last_ts, held.last_seq = record.last_ts, record.last_seq
            if (record.first_ts, 0 if child is None else 1, session_id) < (
                held.first_ts,
                0 if held.child is None else 1,
                held.session_id,
            ):
                held.record, held.session_id, held.child, held.cwd = record, session_id, child, cwd
                held.first_ts, held.first_seq = record.first_ts, record.first_seq
    _overlay_highlights(entries, snapshots, states, home, deny)
    # A family generation includes every member, not just the maximum publication time.
    # The process-local clock is monotone for membership removal as well as append. Restart
    # or bounded-cache eviction can conservatively reset cursors, never reuse a stale offset.
    identity = tuple(sorted((sid, snap.revision if snap else 0) for sid, _, _, snap in states))
    key = (revision_scope, tuple((sid, cwd) for sid, _, cwd, _ in states[:1]))
    with _FAMILY_REVISION_LOCK:
        old_identity, old_revision = _FAMILY_REVISIONS.get(key, ((), 0))
        revision = (
            old_revision
            if identity == old_identity
            else max(int(time.time() * 1000), old_revision + 1)
        )
        _FAMILY_REVISIONS[key] = (identity, revision)
        _FAMILY_REVISIONS.move_to_end(key)
        while len(_FAMILY_REVISIONS) > 256:
            _FAMILY_REVISIONS.popitem(last=False)
    return Merged(
        entries=list(entries.values()),
        revision=revision,
        complete=complete,
        rows_scanned=rows,
        covered_bytes=covered,
        total_bytes=total,
        truncated=truncated,
        snapshots=snapshots,
    )


def _overlay_highlights(
    entries: dict[str, Entry],
    snapshots: Sequence[tuple[str, ChildRef | None, ledger.Snapshot]],
    states: Sequence[tuple[str, ChildRef | None, str, ledger.Snapshot | None]],
    home: str,
    deny: Callable[[str], bool] | None,
) -> None:
    """Flag (or add) a row per Highlights callout - the NEWEST version of each anchor only.

    A callout was already denylisted when it was written, but the denylist is a policy that can
    tighten, and this is the one place a path enters the inventory without passing the scanner,
    so it is re-judged here rather than trusted.
    """
    cwd_of = {session_id: cwd for session_id, _, cwd, _ in states}
    for session_id, child, snapshot in snapshots:
        cwd = cwd_of.get(session_id, "")
        for anchor in snapshot.scan.anchors.values():
            callouts: list[dict[str, Any]] = list(anchor.files)
            callouts.extend(
                {"path": p, "name": None, "kind": None, "size": None, "why": None}
                for p in anchor.more
            )
            for callout in callouts:
                path = resolve_path(callout["path"], cwd=cwd, home=home)
                if not path:
                    continue
                key = f"p:{path}"
                entry = entries.get(key)
                if entry is None and deny is not None and deny(path):
                    continue
                if entry is None:
                    name = callout.get("name") or os.path.basename(path)
                    record = FileRecord(
                        key=key,
                        path=path,
                        digest=None,
                        name=name,
                        kind=callout.get("kind") or kind_of(name),
                        content_type=None,
                        size=callout.get("size"),
                        width=None,
                        height=None,
                        duration_s=None,
                        origin="highlight",
                        tool="supplements",
                        call_id=None,
                        intent=None,
                        first_seq=anchor.seq,
                        last_seq=anchor.seq,
                        first_ts=anchor.ts,
                        last_ts=anchor.ts,
                        touches=1,
                    )
                    entry = entries[key] = Entry(
                        record=record,
                        session_id=session_id,
                        child=child,
                        cwd=cwd,
                        touches=1,
                        last_ts=anchor.ts,
                        last_seq=anchor.seq,
                        first_ts=anchor.ts,
                        first_seq=anchor.seq,
                    )
                entry.highlighted = True
                entry.highlight_why = entry.highlight_why or callout.get("why")


# ---------------------------------------------------------------------------
# Filter, sort, page
# ---------------------------------------------------------------------------


def _matches(entry: Entry, query: Query, needle: str, home: str, *, with_kind: bool) -> bool:
    record = entry.record
    if with_kind and query.kinds and record.kind not in query.kinds:
        return False
    if query.origins and record.origin not in query.origins:
        return False
    if query.highlighted is not None and entry.highlighted != query.highlighted:
        return False
    if needle:
        if needle in record.name.casefold():
            return True
        if record.path and needle in display_path(record.path, entry.cwd, home).casefold():
            return True
        return False
    return True


def _fingerprint(query: Query) -> str:
    """What a cursor is bound to: everything that changes WHICH rows exist or their order."""
    body = json.dumps(
        [
            query.scope,
            sorted(query.kinds),
            sorted(query.origins),
            query.highlighted,
            query.q,
            query.sort,
            query.effective_order,
        ],
        separators=(",", ":"),
    )
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:12]


def encode_cursor(revision: int, query: Query, offset: int) -> str:
    raw = json.dumps(
        {"v": 1, "r": revision, "f": _fingerprint(query), "o": offset}, separators=(",", ":")
    )
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(cursor: str, query: Query) -> tuple[int, int]:
    """``(revision, offset)``; :class:`InvalidCursor` for anything not minted for ``query``."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        doc = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        revision, offset, fingerprint = doc["r"], doc["o"], doc["f"]
        if (
            doc.get("v") != 1
            or not isinstance(revision, int)
            or not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
        ):
            raise ValueError("shape")
    except (ValueError, KeyError, TypeError, binascii.Error, UnicodeError) as error:
        raise InvalidCursor("malformed cursor") from error
    if fingerprint != _fingerprint(query):
        raise InvalidCursor("cursor belongs to a different query")
    return revision, offset


def _effective(entry: Entry) -> tuple[int | None, float, float]:
    """``(size, modified, created)`` as the row will DISPLAY them."""
    record, facts = entry.record, entry.facts
    if record.path is None:
        return record.size, entry.last_ts, entry.first_ts
    if facts is not None and facts.exists:
        return (
            facts.size if facts.size is not None else record.size,
            facts.mtime if facts.mtime is not None else entry.last_ts,
            facts.created if facts.created is not None else entry.first_ts,
        )
    return record.size, entry.last_ts, entry.first_ts


def _sort(entries: list[Entry], query: Query) -> bool:
    """Sort in place. Returns ``stat_complete`` (True when no stat was needed or all landed)."""
    reverse = query.effective_order == "desc"
    stat_complete = True
    if query.sort in ("modified", "created", "size"):
        deadline = time.monotonic() + SORT_STAT_BUDGET_S
        now = time.monotonic()
        for entry in entries:
            if entry.record.path is None or entry.facts is not None:
                continue
            if time.monotonic() > deadline:
                stat_complete = False
                break
            entry.facts = stat_path(entry.record.path, now)
    # Unknown values sort LAST in both directions: a missing size is not "smallest".
    if query.sort == "name":
        entries.sort(key=lambda e: (e.record.name.casefold(), e.record.key), reverse=reverse)
    elif query.sort == "kind":
        entries.sort(
            key=lambda e: (e.record.kind, e.record.name.casefold(), e.record.key), reverse=reverse
        )
    elif query.sort == "recent":
        entries.sort(key=lambda e: (e.last_ts, e.last_seq, e.record.key), reverse=reverse)
    else:
        index = {"size": 0, "modified": 1, "created": 2}[query.sort]
        known = [e for e in entries if _effective(e)[index] is not None]
        unknown = [e for e in entries if _effective(e)[index] is None]
        known.sort(key=lambda e: (_effective(e)[index], e.record.key), reverse=reverse)
        unknown.sort(key=lambda e: e.record.key)
        entries[:] = known + unknown
    return stat_complete


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------


def _artifact_exists(store_root: Path | None, digest: str) -> bool | None:
    if store_root is None:
        return None
    try:
        return (store_root / f"{digest}.bin").is_file()
    except OSError:
        return None


def project(
    entry: Entry,
    *,
    home: str,
    display: bool,
    store_root: Path | None,
) -> dict[str, Any]:
    """The wire row for one entry. ``stat`` happens here (page rows only)."""
    record = entry.record
    if entry.facts is None and record.path is not None:
        entry.facts = stat_path(record.path)
    size, modified, created = _effective(entry)
    if record.path is not None:
        exists: bool | None = bool(entry.facts and entry.facts.exists)
        ref: dict[str, Any] = {
            "type": "path",
            "path": display_path(record.path, entry.cwd, home) if display else record.path,
        }
    else:
        assert record.digest is not None
        exists = _artifact_exists(store_root, record.digest)
        ref = {"type": "artifact", "digest": record.digest, "content_type": record.content_type}
    child = entry.child
    return {
        "id": entry.row_id,
        "ref": ref,
        "name": record.name,
        "kind": record.kind,
        "size_bytes": size,
        "created_at": created,
        "modified_at": modified,
        "width": record.width,
        "height": record.height,
        "duration_s": record.duration_s,
        "origin": record.origin,
        "tool": record.tool,
        "tool_call_id": record.call_id,
        "intent": record.intent,
        "first_seq": entry.first_seq,
        "last_seq": entry.last_seq,
        "first_ts": entry.first_ts,
        "last_ts": entry.last_ts,
        "touches": entry.touches,
        "highlighted": entry.highlighted,
        "highlight_why": entry.highlight_why,
        "session": {
            "id": entry.session_id,
            "role": "subagent" if child is not None else "self",
            "child_id": child.child_id if child is not None else None,
            "label": child.label if child is not None else None,
        },
        "exists": exists,
    }


def list_files(
    config_dir: str | Path,
    family: Family,
    query: Query,
    *,
    display: bool = False,
    budget_s: float | None = DEFAULT_BUDGET_S,
    home: str | None = None,
    expected_parent_revision: int | None = None,
) -> dict[str, Any]:
    """The whole answer for one request. Blocking: call from a worker thread.

    ``budget_s=None`` scans to completion (a CLI or a test); the route passes
    :data:`DEFAULT_BUDGET_S` and finishes in the background when the state is ``building``.
    """
    home = home if home is not None else str(Path.home())
    deny = make_deny(config_dir)
    states = advance(config_dir, family, query.scope, budget_s=budget_s, home=home, deny=deny)
    merged = merge(states, home=home, deny=deny, revision_scope=query.scope)
    # Discovery and listing are two bounded passes. If this pass advanced the parent,
    # its roster may name children absent from the earlier Family input. Report honest
    # building (and poll) rather than certify a complete list for outdated membership.
    parent = states[0][3] if states else None
    if expected_parent_revision is not None and (
        (parent.revision if parent else 0) != expected_parent_revision
    ):
        merged.complete = False
    store_root = Path(config_dir) / "attachments"

    cursor_reset = False
    offset = 0
    if query.cursor:
        minted_at, offset = decode_cursor(query.cursor, query)
        if minted_at != merged.revision:
            # The list moved under the client (or this process restarted and re-published):
            # restart rather than serve a page that may skip or repeat rows, and SAY so.
            offset, cursor_reset = 0, True

    needle = query.q.casefold()
    in_scope = [e for e in merged.entries if _matches(e, query, needle, home, with_kind=False)]
    counts: dict[str, int] = {}
    for entry in in_scope:
        counts[entry.record.kind] = counts.get(entry.record.kind, 0) + 1
    selected = [e for e in in_scope if not query.kinds or e.record.kind in query.kinds]
    stat_complete = _sort(selected, query)

    page = selected[offset : offset + query.limit]
    end = offset + len(page)
    next_cursor = encode_cursor(merged.revision, query, end) if end < len(selected) else None
    return {
        "state": "ready" if merged.complete else "building",
        "revision": merged.revision,
        "scope": query.scope,
        "sort": query.sort,
        "order": query.effective_order,
        "total": len(selected),
        "counts": counts,
        "rows": [project(e, home=home, display=display, store_root=store_root) for e in page],
        "next_cursor": next_cursor,
        "cursor_reset": cursor_reset,
        "stat_complete": stat_complete,
        "scan": {
            "rows": merged.rows_scanned,
            "complete": merged.complete,
            "children": len(family.children) if query.scope == "family" else 0,
            "children_skipped": family.skipped if query.scope == "family" else 0,
            "truncated": merged.truncated,
            "covered_bytes": merged.covered_bytes,
            "total_bytes": merged.total_bytes,
        },
    }


def empty_listing(query: Query, *, state: str = "ready") -> dict[str, Any]:
    """The answer for a session with nothing to list (draft) or a peer (``unsupported``)."""
    return {
        "state": state,
        "revision": 0,
        "scope": query.scope,
        "sort": query.sort,
        "order": query.effective_order,
        "total": 0,
        "counts": {},
        "rows": [],
        "next_cursor": None,
        "cursor_reset": False,
        "stat_complete": True,
        "scan": {
            "rows": 0,
            "complete": True,
            "children": 0,
            "children_skipped": 0,
            "truncated": False,
            "covered_bytes": 0,
            "total_bytes": 0,
        },
    }


# ---------------------------------------------------------------------------
# Materialise
# ---------------------------------------------------------------------------

#: A name this long is a prompt, not a label. Longer is cut BEFORE the extension is fixed.
_NAME_MAX_CHARS: Final = 80
_SAFE_CHARS: Final = re.compile(r"[^A-Za-z0-9._ -]+")


def _safe_stem(text: str) -> str:
    """Replace anything but unicode alphanumerics and ``._ -`` with ``_``.

    A function rather than the ASCII character class used for extensions: ``str.isalnum``
    is unicode-aware, so a request-derived name keeps its letters (D1) while separators,
    NUL and punctuation still collapse.
    """
    return "".join(ch if (ch.isalnum() or ch in "._ -") else "_" for ch in text)


#: ``mimetypes`` maps ``image/jpeg`` to ``.jpe`` on some platforms; prefer the spellings people
#: expect, and only fall back to the registry for the rest.
_PREFERRED_EXT: Final = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    # Ogg is the one type where ``mimetypes`` tie-breaks to ``.oga``; the namer
    # pins ``.ogg``, so the destination must agree or the Library label and the
    # materialised file diverge for the same asset.
    "audio/ogg": ".ogg",
}


def sanitise_name(name: str | None, digest: str, content_type: str | None) -> str:
    """A single safe path component for a materialised artifact.

    The name originates in a provider-influenced transcript row, so it is treated as hostile:
    only its last path component survives (``../../x.png`` -> ``x.png``), characters outside
    unicode alphanumerics and ``._ -`` become ``_``, leading dots are stripped (no hidden
    files, no ``..``), and the EXTENSION is re-derived from the MIME type (ASCII-only) so
    ``photo.png`` holding JPEG bytes is saved as ``photo.jpg``. An empty result falls back to
    ``artifact-<digest[:8]>``.

    Unicode is kept in the STEM on purpose: the Library shows the request-derived label, so
    a CJK or accented name must round-trip to a readable file rather than ``artifact-<digest>``
    (design review D1). The extension stays ASCII because it is a MIME fact, not prose.
    """
    raw = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem, ext = os.path.splitext(raw)
    stem = _safe_stem(stem).strip(" .-_")[:_NAME_MAX_CHARS] or f"artifact-{digest[:8]}"
    mime = (content_type or "").split(";")[0].strip().lower()
    chosen = _PREFERRED_EXT.get(mime) or mimetypes.guess_extension(mime) or ""
    if not chosen:
        cleaned = _SAFE_CHARS.sub("", ext)[:6]
        chosen = cleaned if cleaned.startswith(".") and len(cleaned) > 1 else ".bin"
    return f"{stem}{chosen}"


def find_artifact(merged: Merged, file_id: str) -> Entry | None:
    return next((e for e in merged.entries if e.row_id == file_id), None)


def materialise(
    config_dir: str | Path,
    family: Family,
    file_id: str,
    *,
    home: str | None = None,
) -> dict[str, Any]:
    """Copy a digest-only artifact to ``<session scratchpad>/generated/<name>``. Blocking.

    WHY A COPY, NOT A LINK: the attachment store is content-addressed and shared; a hard link
    would let an editor mutate stored bytes in place and corrupt every other session that
    references the digest. WHY LAZY: the dedup the store exists for is not given up for files
    nobody opens.

    CONFINEMENT: row ids are session-scoped, names are sanitized, and destination directory
    components are pinned with no-follow handles (see ``_copy_artifact``). Creation, reuse and
    cleanup are fd-relative: a concurrent parent-symlink swap cannot redirect a write/read.
    The attachment store re-verifies bytes before copying. Unsupported platforms fail closed.

    IDEMPOTENT: an existing file with identical bytes is reused (``created: false``); a
    different file at the name (the user's) is never overwritten - the copy lands at
    ``<stem>-<digest[:8]><ext>``, then ``-2``, ``-3``…
    """
    from local_operator.scratchpad import scratchpad_root
    from local_operator.session.attachments import store_for_transcript_dir

    if not _ID_RE.match(file_id or ""):
        raise MaterialiseError("file_not_found")
    home = home if home is not None else str(Path.home())
    deny = make_deny(config_dir)
    # Unbudgeted: a row id only exists because a listing already scanned this family, so the cache
    # is warm; scanning to completion beats answering "not found" about a row the user can see.
    states = advance(config_dir, family, "family", budget_s=None, home=home, deny=deny)
    entry = find_artifact(merge(states, home=home, deny=deny), file_id)
    if entry is None:
        raise MaterialiseError("file_not_found")
    record = entry.record
    if record.digest is None:
        raise MaterialiseError("not_an_artifact")

    store = store_for_transcript_dir(family.session_dir)
    blob = store.get_bytes(record.digest)
    if blob is None:
        raise MaterialiseError("artifact_missing")
    data, stored_mime = blob

    scratch = scratchpad_root(family.session_dir)
    if scratch is None:
        raise MaterialiseError("materialise_failed", "session has no scratchpad")
    name = sanitise_name(record.name, record.digest, record.content_type or stored_mime)
    return _copy_artifact(family.session_dir, name, record.digest, data)


_COPY_LOCK = threading.Lock()


def _copy_artifact(session_dir: Path, name: str, digest: str, data: bytes) -> dict[str, Any]:
    """Pin every destination ancestor; all leaf operations use the pinned directory fd.

    ``resolve``/``open('xb')`` is not confinement: a renamed parent can become a link between
    checking and creating. Open each component with O_DIRECTORY|O_NOFOLLOW, then create,
    inspect existing files, and clean partial copies with dir_fd operations. Unsupported
    platforms fail closed rather than silently falling back to the vulnerable path.
    The small copy lock also prevents another POST mistaking our in-progress copy for a clash.
    """
    if not all(fn in os.supports_dir_fd for fn in (os.open, os.mkdir, os.unlink, os.stat)) or not (
        hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
    ):
        raise MaterialiseError("unsafe_destination", "safe directory operations unavailable")
    fds: list[int] = []
    created_name: str | None = None
    with _COPY_LOCK:
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            fds.append(os.open(os.sep, flags))
            # Resolve system aliases such as macOS /var once, then pin EACH component of
            # that absolute spelling. The scratchpad/generated names are never resolved.
            session_real = session_dir.resolve()
            for part in session_real.parts[1:]:
                fds.append(os.open(part, flags, dir_fd=fds[-1]))
            for part in ("scratchpad", "generated"):
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fds[-1])
                except FileExistsError:
                    pass
                fds.append(os.open(part, flags, dir_fd=fds[-1]))
            directory = session_real / "scratchpad" / "generated"
            dirfd = fds[-1]

            def still_named() -> bool:
                # A pinned directory can be renamed safely (no external write), but returning
                # its former pathname as "saved" would lie. Refuse if any pinned name moved.
                paths = (session_real, session_real / "scratchpad", directory)
                for path, fd in zip(paths, fds[-3:]):
                    actual, pinned = os.stat(path, follow_symlinks=False), os.fstat(fd)
                    if (actual.st_dev, actual.st_ino) != (pinned.st_dev, pinned.st_ino):
                        return False
                return True

            stem, ext = os.path.splitext(name)
            candidates = [name, f"{stem}-{digest[:8]}{ext}"] + [
                f"{stem}-{digest[:8]}-{n}{ext}" for n in range(2, 6)
            ]
            want = hashlib.sha256(data).digest()
            for candidate in candidates:
                if not still_named():
                    raise MaterialiseError("unsafe_destination", "destination directory moved")
                try:
                    fd = os.open(
                        candidate,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=dirfd,
                    )
                except FileExistsError:
                    try:
                        existing = os.open(
                            candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dirfd
                        )
                        with os.fdopen(existing, "rb") as handle:
                            if not stat_module.S_ISREG(os.fstat(handle.fileno()).st_mode):
                                continue
                            if hashlib.sha256(handle.read()).digest() != want:
                                continue
                        if not still_named():
                            raise MaterialiseError(
                                "unsafe_destination", "destination directory moved"
                            )
                        return _receipt(
                            directory / candidate, name=candidate, size=len(data), created=False
                        )
                    except OSError:
                        continue
                else:
                    created_name = candidate
                    with os.fdopen(fd, "wb") as handle:
                        handle.write(data)
                    if not still_named():
                        raise MaterialiseError("unsafe_destination", "destination directory moved")
                    created_name = None
                    return _receipt(
                        directory / candidate, name=candidate, size=len(data), created=True
                    )
            raise MaterialiseError("materialise_failed", "no free file name")
        except OSError as error:
            # ELOOP/ENOTDIR are expected refusals of a hostile symlink, not server faults.
            code = (
                "unsafe_destination"
                if error.errno in (errno.ELOOP, errno.ENOTDIR)
                else "materialise_failed"
            )
            raise MaterialiseError(code, str(error)) from error
        finally:
            if created_name is not None:
                try:
                    os.unlink(created_name, dir_fd=fds[-1])
                except OSError:
                    pass
            for fd in reversed(fds):
                os.close(fd)


def _receipt(target: Path, *, name: str, size: int, created: bool) -> dict[str, Any]:
    return {"path": str(target), "name": name, "size_bytes": size, "created": created}


__all__ = [
    "ChildRef",
    "DEFAULT_BUDGET_S",
    "DEFAULT_LIMIT",
    "DEFAULT_ORDER",
    "Family",
    "InvalidCursor",
    "InvalidFilter",
    "MAX_LIMIT",
    "MAX_QUERY_CHARS",
    "MaterialiseError",
    "ORDERS",
    "ORIGINS",
    "Query",
    "SCOPES",
    "SORTS",
    "complete_in_background",
    "display_path",
    "empty_listing",
    "list_files",
    "make_deny",
    "materialise",
    "parse_csv",
    "sanitise_name",
    "session_cwd",
]
