"""Who made this, and who asked for it: provenance for the dangerous-delete guard.

WHY. The built-in guard (``classify.py``) ranks a direct child of ``$HOME`` and
any git repository root HIGH, and a HIGH call goes to a forked judge that only
sees the operator's words. That refused agents cleaning up after THEMSELVES:
a QA lane could not remove the twenty ``~/qa-r5-dg-*`` rig directories it had
just made, a merge step could not drop its own ``.worktrees/<name>``, and disk
cleanup lanes stalled on their own old scratch. The operator's direction: let
those cleanups through, keep dangerous deletes blocked unless a TRUSTED source
explicitly asks for them. Three facts make that decidable without lowering the
bar for anything else:

1. **Creation ledger.** A session that created a directory may delete it. The
   loop records what a bash call created (``plan_creations`` before it runs,
   ``commit_creations`` after) and ``owned_record`` answers "did this session,
   or one of its subagents, create this exact inode?". A record is keyed by
   realpath AND ``st_dev``/``st_ino`` AND a creation stamp (see
   ``creation_stamp``), so a directory deleted and re-made by someone else —
   which on Linux tmpfs/ext4 routinely gets the SAME inode back — or swapped
   for a symlink, no longer matches.
2. **Clean linked worktree.** ``git worktree remove <path>`` without
   ``--force`` refuses tracked/untracked changes and the primary checkout by
   itself, but takes ignored files and a detached HEAD's commits with it.
   ``worktree_state`` proves those absent too (only regenerable ignored
   output, HEAD on a branch or in some ref) before the guard ranks it LOW.
3. **Trusted words.** Only the operator's direct messages (in a session the
   operator opened — never a delegating agent's brief), and their answers
   to ``ask`` as recorded in the ask store, are consent. Team Bridge
   deliveries, peer-agent messages, hub notes and tool output are DATA: a
   compromised or confused agent can write anything into them. A peer message
   may still POINT at an ask (``a-xxxx``); the pointer is checked against the
   operator's durable ask store (``<state>/sessions/<sid>/asks.jsonl``), never
   believed on its own.

CONSTRAINTS. Nothing here may raise CRITICAL to anything lower; the ledger only
ever turns HIGH into LOW. Every probe fails toward the existing ranking: a
missing ledger, a sqlite error, a git timeout, an unreadable ask log all mean
"no provenance", never "allowed".

LEDGER LOCATION. ``<state>/guardrails/created.sqlite`` — inside the state root
but OUTSIDE ``sessions/<id>/``, so the session's own bash cannot rewrite it at
LOW: ``classify`` ranks overwriting or deleting it CRITICAL. A shell append or
an interpreter one-liner is not modelled by the resolver (the same residual as
``config.yml``); the inode check bounds what a forged row can claim.

SUBAGENTS. A child's creation is written under the child's id AND every
ancestor id it knows at write time (``ToolContext.ancestor_session_ids``), so
a parent cleaning up after its subagents needs no lineage lookup at read time.
"""

from __future__ import annotations

import fnmatch
import glob as _glob
import itertools
import json
import logging
import math
import os
import re
import sqlite3
import stat
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

LEDGER_DIRNAME = "guardrails"
LEDGER_FILENAME = "created.sqlite"
#: sqlite busy wait. Rows are tiny and written once per creating bash call;
#: a held lock past this is a stuck writer, and the guard degrades to "no
#: provenance" (the old ranking) rather than stalling a turn.
_DB_TIMEOUT_S = 2.0
#: Bound on every git probe. Measured on this machine's worktrees (n=12,
#: ``git status --porcelain --untracked-files=all`` on local-operator linked
#: worktrees, 2026-11-07): median 0.015 s, max 0.86 s. 10 s is far above that and still
#: short enough that a hung filesystem cannot hold the turn.
_GIT_TIMEOUT_S = 10.0
#: Bound on the ONE ``git status --porcelain --ignored`` that ``worktree_state``
#: runs. The 10 s probe bound above was measured on an idle machine; under the
#: disk pressure of ~25 concurrent sessions (2026-11-09, ~4 GiB free) the same
#: status calls on the operator's real linked worktrees measured n=6, median
#: ~21 s, max 34.3 s (``lo-i18n-es`` 20.4/28.9 s, ``.worktrees/durable-write``
#: 34.3/23.5 s, local-operator-ui ``i18n-restored-content`` 5.0/3.3 s). Every
#: one timed out, ``worktree_state`` said ``unknown``, and a clean
#: ``git worktree remove`` fell through to the repository-root refusal. 60 s
#: sits above that max; the ``git worktree remove`` it gates walks the same
#: tree, so the guard costs at most what the delete itself costs.
_WORKTREE_STATUS_TIMEOUT_S = 60.0
#: How far back the operator brief reads. A transcript can hold thousands of
#: rows; the judge needs the recent operator words, not the session's history.
_SCAN_LIMIT = 400
_MAX_DIRECT = 3
_MAX_ASKS = 4

#: Ask-answer surfaces where a PERSON pressed the button. ``chat`` is excluded
#: on purpose: it is written by the AGENT's ``ask_withdraw answered_in_chat``,
#: i.e. the agent asserting what the operator said. ``relay``/``remote`` are
#: another session's hop, not a human surface.
TRUSTED_ASK_SURFACES = frozenset({"terminal", "tui", "desktop", "mirror"})

#: Words that make an ask answer a refusal ("Leave them", "No, keep it").
_DECLINE_RE = re.compile(
    r"(?i)\b(no|nope|keep|cancel|don'?t|do not|stop|leave|not now|not yet|hold off"
    r"|skip|later|wait|no thanks|abort|nah|negative|never ?mind|reject|deny|decline)\b"
)
_ASK_ID_RE = re.compile(r"(?<![\w-])(a-[0-9a-f]{4,12})(?![\w-])")
#: Where a free-text answer's first clause ends ("Go ahead and copy | but …").
_CLAUSE_BREAK_RE = re.compile(r"(?i)\s*(?:\bbut\b|\bhowever\b|;|\.(?:\s|$)|\s[—–-]\s)\s*")
#: Words AFTER the first clause that narrow or delay the approval.
_CARVE_OUT_RE = re.compile(
    r"(?i)\b(except|excluding|keep|leave|skip|only|cancel|stop|wait|hold off|not yet|not now"
    r"|later|instead)\b"
    r"|\b(?:don'?t|do not|never|no)\s+"
    r"(?:delete|remove|copy|overwrite|touch|move|rm|run|do|change)\b"
    r"|\bnot\s+(?:the|that|those|this|these|any|all)\b"
)
#: A typed answer that is a BARE yes, matched against the WHOLE answer
#: (review R1-B2, PR #3710): "Go ahead and copy", "yes please", "ok, delete
#: them". Anything more — a second clause, "but", "not", "just", "only",
#: "minus", "stays" — is not settled here; the judge reads its text. A prefix
#: match let "Yes for es; no for fr" approve all three files.
_BARE_AFFIRM_RE = re.compile(
    r"(?i)\W*(?:yes|yep|yeah|ok|okay|sure|go ahead|go for it|approved?|proceed|do it|please do"
    r"|confirm(?:ed)?|sounds good|lgtm)"
    r"(?:[\s,]+(?:and\s+)?(?:copy|delete|remove|move|proceed|do it|go ahead)"
    r"(?:\s+(?:it|them|those|these|all|all of them|everything|the files))?)?"
    r"(?:[\s,]+please)?\W*"
)
#: A PICKED option label that says yes to the question (review R1-M1): only
#: such a pick settles without the judge; "Abort", "Installer only", "Nah"
#: never do, whatever their description names.
#: A picked label that says yes to the WHOLE question, nothing narrower
#: (review R2-M1): "Yes", "Delete", "Yes, delete them", "Remove all 57",
#: "Go ahead", "Delete both". "Delete es only" or "Delete the es receipt"
#: does not match, so it approves only the paths its own label names.
_WHOLE_SET_OPTION_RE = re.compile(
    r"(?i)\W*(?:yes|ok|okay|sure|approved?|go ahead|proceed|confirm|do it"
    r"|delete|remove|copy|move|overwrite|replace|clean ?up)"
    r"(?:[\s,]+(?:and\s+)?(?:delete|remove|copy|move|overwrite|replace|proceed|go ahead|do it))?"
    r"(?:\s+(?:all(?:\s+(?:\d+|of\s+them|them))?|them(?:\s+all)?|everything|both|it|these|those))?"
    r"\W*"
)
_AFFIRM_OPTION_RE = re.compile(
    r"(?i)^\W*(?:yes|ok|okay|sure|approve[ds]?|go ahead|proceed|confirm|do it"
    r"|delete|remove|copy|move|overwrite|replace|clean ?up)\b"
)
#: Pointer-less ask lookup in the lineage (``lineage_asks``): recent only.
_LINEAGE_ASK_WINDOW_MS = 24 * 3600 * 1000
_LINEAGE_ASKS_PER_SESSION = 8


# -- ledger -------------------------------------------------------------------


#: The harness-written lineage registry (review R2-MAJOR-1/2): one JSON line
#: per session the harness opened on an agent's behalf — ``{"child", "origin",
#: "parent", "opener", "at"}`` — appended by ``resume.mark_session_origin`` at
#: the spawn sites, never by a tool. It sits beside the ledger under
#: ``<state>/guardrails/``, which the shell and the file tools cannot write
#: (CRITICAL), so "who is whose child" no longer rests on ``origin.json``,
#: which the session's own shell could once reach with ``ln``/``rsync``.
LINEAGE_FILENAME = "lineage.jsonl"
#: Origins that are NOT agent-opened: an operator's fork of their own
#: conversation is theirs (``resume.USER_ORIGINS`` lists it as a user origin).
_HUMAN_ORIGINS = frozenset({"fork"})


def lineage_path(state_root: str | os.PathLike[str] | None = None) -> Path:
    from local_operator.paths import config_dir

    root = Path(state_root) if state_root is not None else config_dir()
    return root / LEDGER_DIRNAME / LINEAGE_FILENAME


def record_lineage(
    child: str,
    *,
    origin: str,
    parent: str | None = None,
    opener: str | None = None,
    state_root: str | os.PathLike[str] | None = None,
) -> None:
    """Append one registry row. Best-effort: a spawn never fails for it — the
    cost of a missing row is that the child's creations go unproved (the old
    ranking), never that a child is adopted it should not be."""
    if not child or origin in _HUMAN_ORIGINS:
        return
    row = {"child": child, "origin": origin, "parent": parent, "opener": opener, "at": time.time()}
    try:
        path = lineage_path(state_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, (json.dumps(row) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        logger.debug("guardrails: lineage row not recorded", exc_info=True)


def ensure_lineage_registry(state_root: str | os.PathLike[str] | None = None) -> float | None:
    """Create the registry if missing; return its birth time (the legacy cutoff).

    Created at the first checkpoint that needs lineage, not only at the first
    spawn, so the migration window for ``origin.json``-only children closes
    as early as possible on an upgraded install.
    """
    path = lineage_path(state_root)
    try:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            os.close(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600))
        st = os.lstat(path)
    except OSError:
        return None
    if not os.path.isfile(path) or os.path.islink(path):
        return None
    return float(getattr(st, "st_birthtime", st.st_ctime))


#: The module whose arrival on this machine made the plant-blocking guard
#: live: a generation carrying it ranks every shell write into ``sessions/``
#: CRITICAL. Globbed relative to a generation root (``update`` layout:
#: ``<gen>/tools/<tool>/lib/python3.X/site-packages/...``).
_GUARD_MODULE_GLOB = "tools/*/lib/python*/site-packages/local_operator/guardrails/provenance.py"
#: ``generations_dir`` → cutoff. Generations are written once and never
#: touched, so one scan per process is enough.
_GUARD_LIVE_CACHE: dict[str, float | None] = {}


def guard_live_at() -> float | None:
    """Birth time of the EARLIEST installed generation carrying the guard.

    ``None`` when no generation carries it (an editable/dev run, or a layout
    from before the generation installer); callers then fall back to the
    registry file's birth.
    """
    from local_operator.update import generations_dir

    root = generations_dir()
    key = str(root)
    if key in _GUARD_LIVE_CACHE:
        return _GUARD_LIVE_CACHE[key]
    births: list[float] = []
    try:
        entries = [e for e in os.scandir(root) if e.is_dir(follow_symlinks=False)]
    except OSError:
        entries = []
    for entry in entries:
        try:
            if not any(Path(entry.path).glob(_GUARD_MODULE_GLOB)):
                continue
            st = entry.stat(follow_symlinks=False)
        except OSError:
            continue
        births.append(float(getattr(st, "st_birthtime", st.st_ctime)))
    value = min(births) if births else None
    _GUARD_LIVE_CACHE[key] = value
    return value


def legacy_lineage_cutoff(state_root: str | os.PathLike[str] | None = None) -> float | None:
    """Children born before this may be adopted from ``origin.json`` alone.

    WHY THE INSTALL, NOT THE REGISTRY FILE. The window exists for children
    spawned by a runtime too old to write registry rows. It closes when the
    plant-blocking guard went live HERE — from then on a shell write into
    ``sessions/`` (a forged ``origin.json``) is CRITICAL, so a later legacy
    child cannot have been planted from the shell. The registry file's birth
    was a proxy for that, and a bad one: a dev or QA run against real state
    created ``lineage.jsonl`` early (Nov 7, before any guarded runtime was
    installed), which shrank the window and orphaned a real ``task`` child
    spawned by a pre-registry runtime, so its parent could not delete the
    clone the child made. Generation dirs live in the install layout, which
    is CRITICAL to shell writes, so their birth cannot be moved by an agent's
    shell. An interpreter computing paths itself remains the documented
    ceiling (see ``evidence`` module doc).

    ``None`` when the registry cannot be created/read: no legacy adoption.
    """
    registry_born = ensure_lineage_registry(state_root)
    if registry_born is None:
        return None
    live = guard_live_at()
    return live if live is not None else registry_born


def read_lineage(state_root: str | os.PathLike[str] | None = None) -> dict[str, dict[str, Any]]:
    """``child → row``; the FIRST row for a child wins (a later row cannot re-parent it)."""
    out: dict[str, dict[str, Any]] = {}
    try:
        fd = os.open(lineage_path(state_root), os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                child = row.get("child") if isinstance(row, dict) else None
                if isinstance(child, str) and child and child not in out:
                    out[child] = row
    except OSError:
        pass
    return out


def ledger_path(state_root: str | os.PathLike[str] | None = None) -> Path:
    from local_operator.paths import config_dir

    root = Path(state_root) if state_root is not None else config_dir()
    return root / LEDGER_DIRNAME / LEDGER_FILENAME


def creation_stamp(st: os.stat_result) -> int:
    """An inode GENERATION stamp: ``dev+ino`` alone is reused after a delete.

    * Birth time where the platform has it (macOS/BSD ``st_birthtime``): set
      once at creation, never by later writes, so it is matched exactly.
    * Otherwise (Linux; Python does not expose statx btime) ``st_ctime_ns``,
      matched exactly too. A directory's ctime moves whenever an entry is
      added, removed or renamed, or on chmod — so on Linux a ledger row stops
      matching once the directory changes after the recording call. That is
      deliberate: it fails CLOSED to the normal ranking, where the judge's
      history evidence still covers a session's own directory. A ctime is
      never accepted as "newer is fine": a recreation is exactly "newer".
    """
    birth = getattr(st, "st_birthtime", None)
    # 0 means "unknown" on some filesystems (FAT/network mounts on macOS):
    # every directory there would share one stamp, so use ctime instead.
    if birth:
        return int(round(float(birth) * 1_000_000_000))
    return int(st.st_ctime_ns)


@dataclass(frozen=True)
class CreationRecord:
    path: str
    dev: int
    ino: int
    created_at: float
    creator: str
    kind: str
    command: str
    #: ``creation_stamp`` at record time; ``None`` on a pre-stamp row, which
    #: therefore never matches.
    stamp: int | None = None

    def describe(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))
        return f"created by this session ({self.kind}, {when})"

    def ledger_fact(self) -> str:
        """The full row for a judge brief: who, what, when and the creating command.

        WHY (live case 2026-11-11, session 316765a40496): the judge read only
        transcript evidence, which compaction had dropped for day-old
        creations, and concluded "no evidence shows this session created it"
        while this row sat in the ledger. The command is agent-authored, so the
        brief fences it as data like every other target fact.
        """
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))
        # Fenced the way ``judge.render_brief`` fences every agent-authored
        # field (security review N2, PR #3800): a newline could start a forged
        # fact line, and a crafted command could otherwise read as part of this
        # fact (``…; uncommitted changes: none``). One line, own closing tag
        # neutralised, so it stays data inside ``<command>``.
        # Every closing-tag shape (``</COMMAND >``, ``< /command>``, any tag
        # name; NIT-1, PR #3802), not just the exact lowercase ``</command>``:
        # a model reads those as a close too.
        command = _CLOSE_TAG_RE.sub("<\\/", " ".join(self.command[:200].split()))
        return (
            f"ledger: created by {self.creator} (this session or its subagent), "
            f"kind {self.kind}, at {when}, via <command>{command}</command>"
        )


#: Any closing-tag opener, however cased or spaced: ``</``, ``< /``.
_CLOSE_TAG_RE = re.compile(r"<\s*/")


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS created ("
    " owner TEXT NOT NULL, path TEXT NOT NULL, dev INTEGER NOT NULL, ino INTEGER NOT NULL,"
    " created_at REAL NOT NULL, creator TEXT NOT NULL, kind TEXT NOT NULL,"
    " command TEXT NOT NULL, stamp INTEGER, PRIMARY KEY (owner, path))"
)


#: ``ForeignCreations`` looks rows up by path (``_foreign_query``); the
#: primary key leads with ``owner`` and cannot serve that. ``IF NOT EXISTS``
#: adds it to ledgers written before it existed, on their next open.
_PATH_INDEX = "CREATE INDEX IF NOT EXISTS created_path ON created (path)"


def _connect(path: Path, *, create: bool) -> sqlite3.Connection | None:
    if not create and not path.exists():
        return None
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_DB_TIMEOUT_S)
    try:
        conn.execute(_SCHEMA)
        conn.execute(_PATH_INDEX)
        # A ledger written before the stamp column existed: add it (old rows
        # keep NULL and so never match — fail closed).
        columns = {row[1] for row in conn.execute("PRAGMA table_info(created)")}
        if "stamp" not in columns:
            try:
                conn.execute("ALTER TABLE created ADD COLUMN stamp INTEGER")
            except sqlite3.OperationalError:
                # Two processes migrating at once: the loser sees "duplicate
                # column". Re-check rather than fail the whole record/lookup.
                columns = {row[1] for row in conn.execute("PRAGMA table_info(created)")}
                if "stamp" not in columns:
                    raise
    except BaseException:
        conn.close()  # never leak a half-set-up connection
        raise
    return conn


def write_records(
    entries: Iterable[tuple[str, str]],
    *,
    owners: Sequence[str],
    creator: str,
    command: str,
    state_root: str | os.PathLike[str] | None = None,
) -> list[str]:
    """Record ``(realpath, kind)`` entries for every owner; returns what was written."""
    owners = [o for o in dict.fromkeys(owners) if o]
    rows: list[tuple[Any, ...]] = []
    written: list[str] = []
    now = time.time()
    for path, kind in entries:
        try:
            # A link row identifies the link ENTRY, so it is lstat'ed: its
            # pointee may be an old directory nobody here made.
            st = os.lstat(path) if kind == "symlink" else os.stat(path)
        except OSError:
            continue
        written.append(path)
        for owner in owners:
            rows.append(
                (
                    owner,
                    path,
                    st.st_dev,
                    st.st_ino,
                    now,
                    creator,
                    kind,
                    command[:500],
                    creation_stamp(st),
                )
            )
    if not rows:
        return []
    conn = _connect(ledger_path(state_root), create=True)
    assert conn is not None
    try:
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO created"
                " (owner, path, dev, ino, created_at, creator, kind, command, stamp)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
    finally:
        conn.close()
    return written


def owned_record(
    path: str, session_id: str, *, state_root: str | os.PathLike[str] | None = None
) -> CreationRecord | None:
    """The ledger row covering ``path`` for ``session_id``, or ``None``.

    Covers = the recorded directory itself or anything strictly inside it,
    compared on REALPATH so a symlink inside an owned directory that points
    elsewhere is not covered. The recorded directory must still be the same
    inode generation (dev, ino and ``creation_stamp``): a path deleted and
    re-made by someone else is no longer this session's. Any error means "not owned".
    """
    if not session_id or not path:
        return None
    try:
        real = os.path.realpath(path)
        conn = _connect(ledger_path(state_root), create=False)
        if conn is None:
            return None
        try:
            rows = conn.execute(
                "SELECT path, dev, ino, created_at, creator, kind, command, stamp FROM created"
                " WHERE owner = ?",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        logger.debug("guardrail ledger unreadable", exc_info=True)
        return None
    best: CreationRecord | None = None
    for row in rows:
        rec = CreationRecord(*row)
        if rec.kind == "symlink":
            # A link row vouches for removing that link entry only
            # (``owned_link``), never for its pointee or anything under it.
            continue
        if not (real == rec.path or real.startswith(rec.path.rstrip("/") + "/")):
            continue
        try:
            st = os.stat(rec.path)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) != (rec.dev, rec.ino):
            continue
        if rec.stamp is None or creation_stamp(st) != rec.stamp:
            continue  # same inode number, different generation (or unstamped)
        if best is None or len(rec.path) > len(best.path):
            best = rec
    if best is not None and _foreign_inside(real, session_id, state_root=state_root):
        return None
    return best


def _foreign_query(paths: Iterable[str]) -> tuple[str, list[str]]:
    """SQL selecting only the ledger rows that can relate to ``paths``.

    m3 (review of PR #3835): reading the whole table (``ORDER BY created_at``,
    no index) cost 230-890 ms per classify at 20k rows, on every LOW delete.
    A row relates to a target only when its path is the target or one of its
    ancestors (an exact ``IN`` list) or lies under it (a half-open range: every
    string starting ``t/`` sorts in ``[t/, t0)`` because ``'0'`` follows
    ``'/'`` in the BINARY collation SQLite compares ``path`` with). Both shapes
    are served by the ``created_path`` index, so the read stays millisecond-
    scale however large the never-pruned ledger grows.
    """
    ancestors: dict[str, None] = {}
    ranges: list[str] = []
    for real in paths:
        node = real
        while True:
            ancestors[node] = None
            parent = os.path.dirname(node)
            if parent == node:
                break
            node = parent
        prefix = real.rstrip("/") + "/"
        ranges += [prefix, prefix[:-1] + "0"]
    clauses = []
    if ancestors:
        clauses.append(f"path IN ({','.join('?' * len(ancestors))})")
    clauses += ["(path >= ? AND path < ?)"] * (len(ranges) // 2)
    sql = (
        "SELECT owner, path, dev, ino, created_at, creator, kind, command, stamp"
        f" FROM created WHERE kind != 'symlink' AND ({' OR '.join(clauses) or '0'})"
    )
    return sql, [*ancestors, *ranges]


@dataclass(frozen=True)
class ForeignHit:
    """A live directory another session created, related to a delete target."""

    #: The session named to the judge: the row's ``creator`` (the session that
    #: ran the creating command), never an ancestor copy of the row.
    owner: str
    record: CreationRecord
    #: ``"at"`` (the target is that directory), ``"inside"`` (the target is
    #: within it) or ``"holds"`` (the target contains it).
    relation: str


class ForeignCreations:
    """Another session's live ledger rows, for one classify call.

    WHY (operator incident, osworld): ``rm -rf ~/worktrees/osworld/osworld-21/runs``
    deleted another session's sealed benchmark evidence. A deep non-git
    directory under HOME is an "ordinary path" (LOW), so nothing asked. The
    creation ledger already knew a different session made that tree; this
    surfaces that fact so ``classify._rank`` sends the delete to the judge
    (HIGH) with the owning session named.

    Constraints:

    * LIVE IDENTITY. A row counts only while its directory is that same inode
      generation (dev, ino and ``creation_stamp``, as ``owned_record``
      checks). Rows are grouped by path AND identity: a dead row at a reused
      path (``runs``, ``/tmp/qa``) neither excuses the live directory as this
      session's nor gets named as its owner (review M1).
    * NEAREST OWNER. ``mkdir -p a/b`` records only the topmost new component,
      so the first session into a shared parent "owns" it. For a target at or
      inside recorded directories, only the DEEPEST live one decides: this
      session's ``~/shared/mine`` inside another's ``~/shared`` stays this
      session's (review M2). A target that HOLDS a foreign live directory is
      HIGH whoever owns the target, since deleting it destroys that work.
    * A row any owner of which is this session, or a session in its subagent
      tree (``same_task_tree``), is not foreign: the ledger copies a child's
      creation to its ancestors, and a manager may remove its child's work and
      vice versa. Lineage reads run only after the cheap ``stat`` identity
      check passes, memoised per owner.
    * One connection per classify call, opened on first use; the caller
      closes it (``close``). Only rows related to the target are read
      (``_foreign_query``), so no row cap is needed.
    * Unreadable ledger (locked, corrupt, missing): ``match`` returns ``None``
      and the caller adds nothing. Failing "HIGH" there would turn every
      delete on the machine into a judge call while one writer holds the DB.
    * Linux: ``creation_stamp`` falls back to ``st_ctime_ns``, which moves
      whenever an entry is added to the directory, so a foreign row there
      stops matching once its directory changes after the recording call —
      this rule fails OPEN on Linux (ranking returns to what it was before
      it existed). The guarantee is macOS/BSD-only, where birth time is used.
    * Symlink rows are skipped: they vouch for a link entry, never a tree.
    """

    def __init__(
        self, session_id: str, *, state_root: str | os.PathLike[str] | None = None
    ) -> None:
        self.session_id = session_id
        self.state_root = state_root
        self._conn: sqlite3.Connection | None = None
        self._tried = False
        self._tree: dict[str, bool] = {}
        self._identity: dict[str, tuple[int, int, int] | None] = {}

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> sqlite3.Connection | None:
        """The cached connection; ``None`` with no ledger or an unreadable one."""
        if not self._tried:
            self._tried = True
            try:
                self._conn = _connect(ledger_path(self.state_root), create=False)
            except (OSError, sqlite3.Error):
                logger.debug("guardrail ledger unreadable", exc_info=True)
        return self._conn

    def _live(self, path: str) -> tuple[int, int, int] | None:
        if path not in self._identity:
            try:
                st = os.stat(path)
            except OSError:
                self._identity[path] = None
            else:
                self._identity[path] = (st.st_dev, st.st_ino, creation_stamp(st))
        return self._identity[path]

    def _related(self, owner: str) -> bool:
        if owner == self.session_id:
            return True
        if owner not in self._tree:
            self._tree[owner] = bool(
                same_task_tree(self.session_id, owner, state_root=self.state_root)
            )
        return self._tree[owner]

    def match(self, paths: Iterable[str]) -> ForeignHit | None:
        """The foreign live directory at, under or above any of ``paths``.

        ``None`` when nothing foreign relates — or when the ledger is unreadable.
        """
        reals = sorted({os.path.realpath(p).rstrip("/") or "/" for p in paths if p})
        conn = self._connection()
        if not reals or conn is None:
            return None
        sql, params = _foreign_query(reals)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            logger.debug("guardrail ledger unreadable", exc_info=True)
            return None
        # path -> (owners, row) over rows of the LIVE generation only.
        live: dict[str, tuple[set[str], CreationRecord]] = {}
        for owner, *fields in rows:
            rec = CreationRecord(*fields)
            if rec.stamp is None or self._live(rec.path) != (rec.dev, rec.ino, rec.stamp):
                continue
            live.setdefault(rec.path, (set(), rec))[0].add(owner)

        def foreign(path: str) -> CreationRecord | None:
            owners, rec = live[path]
            return None if any(self._related(o) for o in owners) else rec

        for real in reals:
            covering = [p for p in live if p == real or real.startswith(p.rstrip("/") + "/")]
            if covering:
                nearest = max(covering, key=len)
                rec = foreign(nearest)
                if rec is not None:
                    relation = "at" if nearest == real else "inside"
                    return ForeignHit(rec.creator, rec, relation)
            prefix = real.rstrip("/") + "/"
            for p in sorted(p for p in live if p.startswith(prefix)):
                rec = foreign(p)
                if rec is not None:
                    return ForeignHit(rec.creator, rec, "holds")
        return None


def link_path(path: str) -> str:
    """The canonical path of the link ENTRY at ``path``: its parent resolved, its
    own last component kept, so the link itself is never followed."""
    parent, name = os.path.split(os.path.abspath(path))
    return os.path.join(os.path.realpath(parent), name)


def _link_identity(path: str) -> tuple[int, int, int] | None:
    try:
        st = os.lstat(path)
    except OSError:
        return None
    return (st.st_dev, st.st_ino, creation_stamp(st))


def owned_link(
    path: str, session_id: str, *, state_root: str | os.PathLike[str] | None = None
) -> CreationRecord | None:
    """The ledger row for the symlink ENTRY at ``path`` this session made, or ``None``.

    WHY A SEPARATE LOOKUP (q1029 incident). ``rm LINK`` unlinks the link and
    leaves its pointee alone, so ownership of the link is a question about
    the link entry: matched EXACTLY on its canonical path (``link_path``) and
    its lstat identity (dev, ino, ``creation_stamp``), never by realpath
    and never as a prefix — a path under a link lives in the pointee.

    WHY IT MUST STILL BE A SYMLINK. A link the session made and someone then
    replaced with a real directory (same name, ``rm link && mkdir link``)
    holds that someone's data; the identity check already rejects a new
    inode, and the ``islink`` check keeps the rule explicit even where an
    inode number is reused. Any error means "not owned".
    """
    if not session_id or not path or not os.path.islink(path):
        return None
    canonical = link_path(path)
    identity = _link_identity(canonical)
    if identity is None:
        return None
    try:
        conn = _connect(ledger_path(state_root), create=False)
        if conn is None:
            return None
        try:
            rows = conn.execute(
                "SELECT path, dev, ino, created_at, creator, kind, command, stamp FROM created"
                " WHERE owner = ? AND path = ? AND kind = 'symlink'",
                (session_id, canonical),
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        logger.debug("guardrail ledger unreadable", exc_info=True)
        return None
    for row in rows:
        rec = CreationRecord(*row)
        if rec.stamp is not None and (rec.dev, rec.ino, rec.stamp) == identity:
            return rec
    return None


def _holds_git(directory: str) -> bool:
    """A ``.git`` anywhere in ``directory`` (bounded; over the bound → True)."""
    return _git_inside(directory) is not None


class GitInside(NamedTuple):
    """What :func:`_git_inside` found: ``path`` holds a ``.git`` when
    ``exhausted`` is False; when True the bounded walk ran out under ``path``
    and a nested repository could NOT be ruled out (R2-MAJOR-1, PR #3802: the
    two were one bare path, so a 20k-file ignored ``.expo/cache`` reached the
    judge as "nested repository … commits may exist nowhere else")."""

    path: str
    exhausted: bool


def _git_inside(directory: str) -> GitInside | None:
    """Where a ``.git`` sits in ``directory`` (its parent's path), or that the
    bounded walk ran out first (unknown: callers count it as holding one, so
    it fails closed), or ``None``. The path is for the judge (MINOR-3, PR
    #3802): a tree held back for a nested repository must say WHICH one —
    and must not claim one it never saw."""
    if not os.path.isdir(directory) or os.path.islink(directory):
        return None
    seen = 0
    stack: list[tuple[str, int]] = [(directory, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        seen += len(entries)
        if seen > _WALK_ENTRIES:
            return GitInside(directory, exhausted=True)
        for entry in entries:
            if entry.name == ".git":
                return GitInside(current, exhausted=False)
            if depth < _WALK_DEPTH and entry.is_dir(follow_symlinks=False):
                stack.append((entry.path, depth + 1))
    return None


#: Bounds on the work-tree walks (``_foreign_inside``, ``_holds_git``). Hitting either fails
#: closed (not owned → the normal ranking/judge), never open.
_WALK_DEPTH = 4
_WALK_ENTRIES = 20_000


def _foreign_inside(
    real: str, session_id: str, *, state_root: str | os.PathLike[str] | None = None
) -> bool:
    """Does ``real`` hold something this session's ledger does not vouch for?

    WHY (QA Q7). The ledger records the TOPMOST new directory, so the first
    ``git worktree add .worktrees/a`` in a fresh repo records ``.worktrees``
    itself — and "strictly inside a recorded path" then covered every later
    sibling another session added there. A ledger-owned ancestor therefore
    does not make a target LOW when (a) another session's ledger row is at or
    under the target, or (b) the target is, or contains, a git work tree or
    repository (a ``.git`` file or directory) whose directory this session's
    ledger does not record itself. Either way the target goes to the normal
    ranking and the judge.
    """
    prefix = real.rstrip("/") + "/"
    try:
        conn = _connect(ledger_path(state_root), create=False)
        if conn is None:
            return False
        try:
            rows = conn.execute("SELECT path, owner FROM created").fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return True
    own = {p for p, o in rows if o == session_id}
    if any((p == real or p.startswith(prefix)) and p not in own for p, _ in rows):
        return True
    seen = 0
    stack: list[tuple[str, int]] = [(real, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        seen += len(entries)
        if seen > _WALK_ENTRIES:
            return True
        for entry in entries:
            if entry.name == ".git":
                if os.path.realpath(directory) not in own:
                    return True
                continue
            if depth < _WALK_DEPTH and entry.is_dir(follow_symlinks=False):
                stack.append((entry.path, depth + 1))
    return False


@dataclass
class PendingCreations:
    """What a bash call may create, snapshotted BEFORE it runs."""

    command: str
    #: ``(kind, candidate path, topmost component that does not exist yet)``.
    exact: list[tuple[str, str, str]] = field(default_factory=list)
    #: ``(parent, pattern, matches that already existed)`` for ``mktemp -d``.
    patterned: list[tuple[str, str, frozenset[str]]] = field(default_factory=list)
    #: ``(link path, lstat identity before the call or None)`` for ``ln -s``.
    #: ``ln -sf`` REPLACES an existing link, so "did not exist yet" (the
    #: ``exact`` rule) would miss it; a changed identity is the creation.
    links: list[tuple[str, tuple[int, int, int] | None]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.exact or self.patterned or self.links)


def _topmost_new(path: str) -> str | None:
    if os.path.lexists(path):
        return None
    top = path
    parent = os.path.dirname(top)
    while parent and parent != top and not os.path.lexists(parent):
        top, parent = parent, os.path.dirname(parent)
    return top


def plan_creations(
    command: str, *, cwd: str | None, env: Mapping[str, str]
) -> PendingCreations | None:
    """Snapshot the creation candidates of ``command``; ``None`` when there are none."""
    from local_operator.guardrails.resolve import resolve_command

    home = env.get("HOME") or os.path.expanduser("~")
    cwd_n = os.path.normpath(os.path.abspath(os.path.expanduser(cwd))) if cwd else None
    resolution = resolve_command(command, cwd=cwd_n, env=env, home=home)
    pending = PendingCreations(command=command)
    for created in resolution.created:
        if created.kind == "symlink" and created.path is not None:
            if os.path.isdir(created.path) and not os.path.islink(created.path):
                continue  # ln would put the link INSIDE this directory
            pending.links.append((created.path, _link_identity(created.path)))
        elif created.path is not None:
            top = _topmost_new(created.path)
            if top is not None:
                pending.exact.append((created.kind, created.path, top))
        elif created.parent and created.pattern:
            before = frozenset(
                _glob.glob(os.path.join(_glob.escape(created.parent), created.pattern))
            )
            pending.patterned.append((created.parent, created.pattern, before))
    return pending or None


def _recordable(
    real: str, *, env: Mapping[str, str], scratchpad: str | None, session_id: str
) -> bool:
    """Never record a protected root, an ancestor of one, or someone else's state."""
    from local_operator.guardrails.classify import (
        _is_ancestor_or_self,
        _protected,
        _under,
    )

    prot = _protected(env, scratchpad, session_id)
    if real == "/":
        return False
    roots = [
        *prot.homes,
        *prot.state_roots,
        *prot.install_roots,
        *prot.config_files,
        *prot.secrets_dirs,
        *prot.ledger_files,
    ]
    if any(_is_ancestor_or_self(real, root) for root in roots):
        return False
    for state in prot.state_roots:
        if _under(real, state) and not any(_under(real, own) for own in prot.own_session):
            return False
    return True


def commit_creations(
    pending: PendingCreations,
    *,
    session_id: str,
    ancestors: Sequence[str] = (),
    env: Mapping[str, str],
    scratchpad: str | None,
) -> list[str]:
    """Record what ``pending`` really created. Best-effort; never raises."""
    try:
        entries: list[tuple[str, str]] = []
        for kind, path, top in pending.exact:
            if not os.path.isdir(path) or not os.path.isdir(top):
                continue
            entries.append((os.path.realpath(top), kind))
            if kind in ("worktree", "clone") and os.path.realpath(path) != os.path.realpath(top):
                # The checkout itself too: a ledger-owned CONTAINER does not
                # vouch for a work tree inside it (``_foreign_inside``), so
                # the session's own worktree needs its own row (QA Q7).
                entries.append((os.path.realpath(path), kind))
        for parent, pattern, before in pending.patterned:
            now = _glob.glob(os.path.join(_glob.escape(parent), pattern))
            for match in now:
                if match not in before and os.path.isdir(match):
                    entries.append((os.path.realpath(match), "mktemp"))
        for path, before in pending.links:
            # Recorded only if the call left a NEW link entry at that exact
            # path: an ``ln -sf`` that dereferenced an existing link to a
            # directory made its link inside the pointee instead.
            if os.path.islink(path) and _link_identity(path) != before:
                entries.append((link_path(path), "symlink"))
        entries = [
            (p, k)
            for p, k in dict(entries).items()
            if _recordable(p, env=env, scratchpad=scratchpad, session_id=session_id)
        ]
        if not entries or not session_id:
            return []
        return write_records(
            entries,
            owners=[session_id, *ancestors],
            creator=session_id,
            command=pending.command,
        )
    except Exception:  # noqa: BLE001 — provenance is an optimisation, never a failure
        logger.warning("guardrail creation ledger write failed", exc_info=True)
        return []


# -- git worktrees -------------------------------------------------------------


def _git(args: Sequence[str], cwd: str, *, timeout: float = _GIT_TIMEOUT_S) -> str | None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Read-only probes: no prompts, no index refresh writes, no credential helper.
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


#: Ignored entries a clean-worktree removal may take with it: build and
#: dependency output any checkout regenerates. Matched on the entry's basename.
_REGENERABLE = frozenset(
    {
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        ".next",
        "target",
        ".turbo",
        "coverage",
        ".tox",
        # Added 2026-11-09 from the operator's real clean worktrees, each of
        # which the old list called dirty: setuptools metadata from an editable
        # install, generic tool caches (``.cache`` — vite/eslint/parcel), and
        # Finder's per-directory ``.DS_Store``.
        ".eggs",
        ".cache",
        ".DS_Store",
        # Added 2026-11-11 from the operator's finished local-operator-mobile
        # worktrees (session 316765a40496): Expo's per-project state and a
        # pnpm content-addressed store, both rebuilt by ``expo start`` /
        # ``pnpm install``, made merged, change-free trees read ``dirty``.
        ".expo",
        ".pnpm-store",
    }
)
#: Basename patterns with the same standing as :data:`_REGENERABLE`:
#: bytecode, editable-install metadata (``<dist>.egg-info``), macOS AppleDouble
#: sidecars (``._<name>``, metadata about a sibling), and the throwaway esbuild
#: bundles local-operator-ui tests write beside themselves and unlink
#: (``scripts/_*.bundle.mjs``; a killed run leaves them behind — that repo's
#: ``.gitignore`` documents the convention).
_REGENERABLE_GLOBS = ("*.pyc", "*.egg-info", "._*", "_*.bundle.mjs", "_*.bundle.cjs")


def _regenerable(name: str) -> bool:
    return name in _REGENERABLE or any(fnmatch.fnmatchcase(name, g) for g in _REGENERABLE_GLOBS)


#: Bound on ONE process-cwd inventory (``lsof -a -u <uid> -d cwd -Fn -w``).
#: Measured on this machine under ~25 concurrent sessions: QA round 1 of
#: PR #3700, n=28, 1.6–5.54 s with 11/28 censored at the old 5 s bound; the
#: manager's lane measured 3–10 s; a 2026-11-09 re-run here, n=7, 1.30–3.95 s.
#: 20 s sits above the observed max (10 s). A timeout still returns ``None``,
#: which ``worktree_state`` reads as ``busy`` — fail closed.
_CWD_INVENTORY_TIMEOUT_S = 20.0
#: How long one successful inventory is reused within this process. WHY: a
#: ``git worktree remove`` call is checked, previewed and often retried within
#: seconds, and each inventory costs seconds under load (QA Q-1: every removal
#: walked it, a preview doubled it). INVARIANT and its price: a process that
#: ``cd``s into the tree less than 30 s after an inventory is not seen; that is
#: the same window a process starting right after the probe always had, and
#: the owning-session check (``worktree_activity``) reads the session
#: registry fresh on every call. Failures are never cached.
_CWD_INVENTORY_TTL_S = 30.0
_cwd_inventory_cache: tuple[float, tuple[str, ...]] | None = None


def _cwd_inventory() -> tuple[str, ...] | None:
    """Realpaths of every cwd this uid's processes hold, or ``None`` if unknown.

    lsof's cwd inventory is read-only and avoids probing process command lines
    (which may carry secrets). Linux exposes the same data under /proc. A
    disappearing process no longer owns a cwd; other inspection errors give
    ``None``. A success is cached for :data:`_CWD_INVENTORY_TTL_S`.
    """
    global _cwd_inventory_cache
    now = time.monotonic()
    cached = _cwd_inventory_cache
    if cached is not None and now - cached[0] < _CWD_INVENTORY_TTL_S:
        return cached[1]
    paths: list[str] = []
    if os.path.isdir("/proc/self"):
        try:
            for entry in os.scandir("/proc"):
                if not entry.name.isdecimal():
                    continue
                try:
                    if entry.stat().st_uid != os.getuid():
                        continue
                    paths.append(os.readlink(os.path.join(entry.path, "cwd")))
                except FileNotFoundError:
                    continue
        except OSError:
            return None
    else:
        try:
            proc = subprocess.run(
                ["lsof", "-a", "-u", str(os.getuid()), "-d", "cwd", "-Fn", "-w"],
                capture_output=True,
                text=True,
                timeout=_CWD_INVENTORY_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        paths = [line[1:] for line in proc.stdout.splitlines() if line.startswith("n/")]
        if not paths:
            return None  # this process alone holds a cwd: an empty list is a failed read
    inventory = tuple(os.path.realpath(p) for p in paths)
    _cwd_inventory_cache = (now, inventory)
    return inventory


def _live_cwd_inside(real: str) -> bool | None:
    """Whether a live process uses this tree; unknown never grants an exemption.

    Reads :func:`_cwd_inventory` (bounded, cached per process). ``None`` —
    timeout or an unreadable inventory — fails closed in ``worktree_state``.
    This protects ordinary clean removals as well as abandoned adds.
    """
    inventory = _cwd_inventory()
    if inventory is None:
        return None
    return any(cwd == real or cwd.startswith(real + os.sep) for cwd in inventory)


def _only_regenerable(directory: str) -> bool:
    """Every entry under ``directory`` is regenerable (bounded; over → False).

    WHY. ``git status --ignored`` collapses a directory whose content is ALL
    ignored into the directory itself: ``scripts/`` holding only throwaway
    ``_*.bundle.mjs`` files reports as ``!! scripts/``, and that basename says
    nothing about its content. So the walk judges content, not names: a FILE
    passes by :func:`_regenerable`, but a nested DIRECTORY named like build
    output (``work/build/``, ``data/.cache/``) is walked like any other —
    under an arbitrarily named ignored parent its name proves nothing (review
    m1, PR #3700: ``work/build/thesis.md`` read as clean). Only a directory
    matching a pattern in :data:`_REGENERABLE_GLOBS` (``*.egg-info``) is
    accepted whole; the ignored ROOT itself is accepted by name in
    ``worktree_state``. Not a directory, a symlink, an unreadable tree or more
    than :data:`_WALK_ENTRIES` entries all count as NOT regenerable — fail
    closed.
    """
    if not os.path.isdir(directory) or os.path.islink(directory):
        return False
    seen = 0
    stack = [directory]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            return False
        seen += len(entries)
        if seen > _WALK_ENTRIES:
            return False
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if any(fnmatch.fnmatchcase(entry.name, g) for g in _REGENERABLE_GLOBS):
                    continue
                stack.append(entry.path)
            elif not _regenerable(entry.name):
                return False
    return True


def worktree_state(path: str, *, own: bool = False) -> str:
    """The removal-relevant state of a worktree path.

    ``own``: the caller proved (creation ledger) that this session or its
    subagent created the tree. Then IGNORED entries never make it dirty —
    ``git worktree remove`` deletes them itself, and its creator made them
    (operator escalation 2026-11-11: refusing on ignored ``.expo/`` drove
    agents to ``--force`` and then ``rm -rf``). Tracked and untracked changes,
    a lock, a live cwd, an unreferenced detached HEAD and a nested repository
    (``.git`` inside an ignored entry) still count. CONSEQUENCE, by design:
    an ignored secret or config file (``.env``, ``.envrc``, local settings) in
    your OWN tree does not make it dirty, so removing the tree deletes it with
    no judge call — exactly as ``git worktree remove`` itself does.

    One of ``clean``, ``dirty``, ``locked``, ``stale-initializing`` (locked
    "initializing" for over an hour, otherwise clean), ``busy`` (a live process
    cwd is inside it, or the inventory could not be read), ``not-linked`` or
    ``unknown``.

    Linked = its ``.git`` is a FILE, its git-dir differs from the common dir
    and sits in ``<common>/worktrees/<name>``, and that registration's
    ``gitdir`` file points back at this path — exactly the records
    ``git worktree list`` reads, read here without spawning git. Clean = ALL
    of: not locked, no tracked or untracked change, no ignored entry that is
    not regenerable (:func:`_regenerable`), and HEAD on a branch or contained
    in some ref. Anything less is ``dirty`` and takes the ordinary ranking —
    where an OWN worktree can still pass through the ledger or history
    evidence.

    WHY no "pushed or merged" test for a branch HEAD. ``git worktree remove``
    deletes the working tree and ``<common>/worktrees/<name>``; it never
    deletes the branch. The branch ref lives in the common repository, so
    every commit on it is still reachable after the removal — whether pushed,
    squash-merged (not an ancestor of any remote ref) or purely local. Asking
    for remote reachability refused clean worktrees of squash-merged branches
    and protected nothing. Only a DETACHED HEAD loses commits (no ref names
    them once the worktree's own HEAD file is gone), so only that case checks
    ``--contains``. Deleting the branch afterwards is a separate command
    (``git branch -D``) the guard does not exempt.

    COST. Under heavy disk pressure each git probe here measured up to 34 s
    (see :data:`_WORKTREE_STATUS_TIMEOUT_S`), so this makes the fewest probes
    that answer the question: one ``rev-parse``, file reads for registration
    and lock, ONE ``git status --porcelain --ignored --untracked-files=normal``
    for tracked, untracked
    (``??``) and ignored (``!!``) entries alike, and ``symbolic-ref``. A
    timeout is still ``unknown`` — never ``clean``.
    """
    try:
        real = os.path.realpath(path)
        if not os.path.isfile(os.path.join(real, ".git")):
            return "not-linked"
    except OSError:
        return "unknown"
    dirs = _git(["rev-parse", "--absolute-git-dir", "--git-common-dir"], real)
    if dirs is None:
        return "unknown"
    lines = [ln for ln in dirs.splitlines() if ln.strip()]
    if len(lines) != 2:
        return "unknown"
    git_dir = os.path.realpath(lines[0])
    common = os.path.realpath(os.path.join(real, lines[1]))
    if git_dir == common or os.path.dirname(git_dir) != os.path.join(common, "worktrees"):
        return "not-linked"
    try:
        with open(os.path.join(git_dir, "gitdir"), encoding="utf-8") as fh:
            back = fh.read().strip()
    except OSError:
        return "not-linked"
    # Relative when ``worktree.useRelativePaths`` is set (git >= 2.48):
    # relative to the registration directory itself.
    back = os.path.join(git_dir, back)
    if os.path.realpath(os.path.dirname(back)) != real:
        # Registered for another path (a moved or copied tree): not this one.
        return "not-linked"
    stale_initializing = False
    lock = os.path.join(git_dir, "locked")
    try:
        with open(lock, encoding="utf-8") as fh:
            reason = fh.read().strip()
        # One hour is the operator's explicit stale-add policy, not a measured
        # probe timeout. Never unlock or mutate this record from the guard.
        age = time.time() - os.stat(lock).st_mtime
        stale_initializing = reason == "initializing" and age > 3600
        if not stale_initializing:
            return "locked"
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError):
        return "unknown"
    if _live_cwd_inside(real) is not False:
        return "busy"  # unknown process inventory also fails closed
    # ``git worktree remove`` deletes IGNORED content too (review R1-MAJOR-4 /
    # QA Q1: an ignored ``.env`` and an ignored nested repo with the only copy
    # of its history went with it). Only regenerable entries may remain. A
    # nested repository inside an ignored directory surfaces as that
    # directory: under a name not on the list it is dirty outright, and under
    # an allowlisted name (``build/keep/.git``) the bounded walk below finds
    # it (review R2-MINOR-1).
    # ``--untracked-files=normal`` is explicit (review M1, PR #3700): with
    # ``status.showUntrackedFiles=no`` in repo or user config a bare status
    # hides ``??`` entries, the tree reads clean, and ``git worktree remove``
    # (which reads the same config) deletes them. ``normal`` overrides the
    # config and keeps the collapsed ``!! dir/`` shape the walk expects.
    ignored = _git(
        ["status", "--porcelain", "--ignored", "--untracked-files=normal"],
        real,
        timeout=_WORKTREE_STATUS_TIMEOUT_S,
    )
    if ignored is None:
        return "unknown"
    if any(ln and not ln.startswith("!! ") for ln in ignored.splitlines()):
        return "dirty"  # a tracked change or an untracked (``??``) entry
    for line in ignored.splitlines():
        if not line.startswith("!! "):
            continue
        entry = line[3:].strip().strip('"').rstrip("/")
        name = os.path.basename(entry)
        if not own and not _regenerable(name) and not _only_regenerable(os.path.join(real, entry)):
            return "dirty"
        # Kept for OWN trees too (security review M2, PR #3800): a nested
        # repository's commits may exist nowhere else. ``owned_record``'s
        # ``_foreign_inside`` also refuses such a tree, but this check must
        # not depend on that caller having run first.
        if _holds_git(os.path.join(real, entry)):
            return "dirty"
    # A detached HEAD whose commit no ref contains is lost with the tree.
    if _git(["symbolic-ref", "-q", "HEAD"], real) is None:
        refs = _git(
            [
                "for-each-ref",
                "--count=1",
                "--contains",
                "HEAD",
                "refs/heads",
                "refs/remotes",
                "refs/tags",
            ],
            real,
        )
        if not (refs or "").strip():
            return "dirty"
    return "stale-initializing" if stale_initializing else "clean"


#: Bounds on :func:`linked_worktrees_reached`. ``_REACH_MAX_LEVELS``:
#: ancestor levels climbed to find an enclosing repository or to match a
#: registered root (deeper than any real path). ``_REACH_MAX_ENTRIES``:
#: DIRECTORIES the whole call may inspect (each a ``_common_dir`` probe) —
#: one budget per call, not per target (MINOR-4, PR #3802: a per-match
#: 2000-child probe took a 200×200 ``big/*/*`` from 0.011 s to 9.5 s). The
#: same value as ``_WALK_ENTRIES`` and ``classify._GLOB_WALK_CAP``, the
#: guard's existing per-call walk bound. Past it the scan cannot rule a
#: linked worktree out, so it answers ``None`` (unknown) and the caller ranks
#: the target HIGH — fail closed, where the old per-target child cap stopped
#: and reported only what it had found.
#:
#: Only directories are charged (R2-MINOR-1 / QA Q2, PR #3802): a plain file
#: can neither be nor hold a linked worktree (that takes a ``.git`` file in
#: a directory, or a ``worktrees/`` registry), and charging files turned
#: ``rm -rf logs/`` of 25k log files from LOW into a judge call.
#: ``_REACH_MAX_LISTED`` still bounds the raw listing: 50× the directory
#: budget, so only a pathological single directory (a million entries) fails
#: closed on listing cost alone.
_REACH_MAX_LEVELS = 64
_REACH_MAX_ENTRIES = 20_000
_REACH_MAX_LISTED = 1_000_000


def _common_dir(root: str) -> str | None:
    """The git common directory of a work tree rooted at ``root``, read from files.

    No git process: this runs for EVERY delete target, so it must cost a few
    stats (one when there is no ``.git``). A ``.git`` directory is its own
    common dir; a ``.git`` FILE (``gitdir: X``) names a linked worktree's
    private dir, whose ``commondir`` file points at the shared one (a
    submodule's has none).
    """
    dotgit = os.path.join(root, ".git")
    try:
        mode = os.stat(dotgit).st_mode
    except OSError:
        return None
    try:
        if stat.S_ISDIR(mode):
            return os.path.realpath(dotgit)
        if not stat.S_ISREG(mode):
            return None
        with open(dotgit, encoding="utf-8") as fh:
            first = fh.readline().strip()
        if not first.startswith("gitdir:"):
            return None
        git_dir = os.path.join(root, first[len("gitdir:") :].strip())
        try:
            with open(os.path.join(git_dir, "commondir"), encoding="utf-8") as fh:
                return os.path.realpath(os.path.join(git_dir, fh.read().strip()))
        except FileNotFoundError:
            return os.path.realpath(git_dir)
    except (OSError, UnicodeError):
        return None


def _registered_worktrees(common: str) -> list[str]:
    """Roots of the linked worktrees ``<common>/worktrees/*`` registers and that exist."""
    registry = os.path.join(common, "worktrees")
    roots: list[str] = []
    try:
        entries = list(os.scandir(registry))
    except OSError:
        return roots
    for entry in entries:
        try:
            with open(os.path.join(entry.path, "gitdir"), encoding="utf-8") as fh:
                back = fh.read().strip()
        except (OSError, UnicodeError):
            continue
        # Relative when ``worktree.useRelativePaths`` is set (git >= 2.48).
        root = os.path.realpath(os.path.dirname(os.path.join(entry.path, back)))
        if os.path.isfile(os.path.join(root, ".git")):
            roots.append(root)  # a pruned/missing tree has nothing left to delete
    return roots


def linked_worktrees_reached(paths: Iterable[str], *, resolved: bool = False) -> list[str] | None:
    """Linked worktree roots a recursive delete of ``paths`` would destroy.

    ``resolved``: the caller vouches each path is absolute with no symlink
    on the way (a glob match ``classify._expand_glob`` saw no link in), so
    the per-path ``realpath`` is skipped.

    ``None`` = unknown: the scan ran past ``_REACH_MAX_ENTRIES``, so the
    caller must not treat the delete as reaching nothing.

    WHY (security review of PR #3800, pre-existing gap): ``rm -rf
    <repo>/.worktrees/*`` from an unrelated session ranked LOW as an
    "ordinary path", because only the target ITSELF was tested for a
    ``.git``; a glob match or an ancestor (``rm -rf <repo>/.worktrees``)
    deleted every linked worktree under it unjudged. A tree is reached when
    its root IS a deleted directory or lies under one.

    "Is" means filesystem identity, ``(st_dev, st_ino)``, never a string
    prefix (MAJOR-1, PR #3802): ``realpath`` keeps the case a command typed,
    so on case-insensitive APFS ``rm -rf <repo>/.WORKTREES`` deleted the
    trees while ``…/.WORKTREES/`` was no prefix of the registry's
    ``…/.worktrees/x``. Each registered root and its ancestors are stat'ed
    and looked up among the deleted directories' identities; that also holds
    through firmlinks and bind mounts.

    HOW, without spawning git: find the repositories whose registries could
    list such a tree — the one enclosing each path (climbing ancestors) and
    any repository that is a direct child of it — and read their
    ``worktrees/*/gitdir`` records. The cost is bounded per CALL, not per
    path (MINOR-4): sibling glob matches share their ancestors, so a
    directory already climbed or probed ends the climb (its chain is done),
    and child probes draw on one entry budget. CEILING: a worktree
    registered by a repository that neither encloses the target nor sits
    directly in it (``rm -rf ~/a`` where ``~/a/b/repo`` registers
    ``~/a/b/wt``) is not found here; ``~/a`` itself still takes its ordinary
    ranking.

    A symlink named without a trailing slash is skipped: ``rm -rf link``
    removes the link, not what it points at. Targets arrive normalised, so a
    ``link/`` delete (which DOES reach the pointee) would be skipped too; the
    caller resolves it first (``classify._reached_worktrees``, MAJOR-1 of the
    #3802 restack).
    """
    deleted: set[tuple[int, int]] = set()
    commons: set[str] = set()
    # Directories whose ``.git`` was probed; a climb reaching one stops, as
    # everything above it was climbed when it was first seen.
    probed: set[str] = set()
    budget = _REACH_MAX_ENTRIES
    listed = 0
    for raw in paths:
        try:
            if not resolved and os.path.islink(raw) and not raw.endswith("/"):
                continue
            st = os.stat(raw)
        except OSError:
            continue
        if not stat.S_ISDIR(st.st_mode):
            continue
        deleted.add((st.st_dev, st.st_ino))
        real = os.path.normpath(raw) if resolved else os.path.realpath(raw)
        # The target's own listing answers "is it a work tree?" (a ``.git``
        # entry) and yields its child directories: one ``scandir``, no extra
        # stat — ``is_dir(follow_symlinks=False)`` reads d_type. Files are
        # dropped here and never charged to ``budget`` (R2-MINOR-1).
        listing: list[os.DirEntry[str]] | None = []
        try:
            with os.scandir(real) as it:
                for entry in it:
                    listed += 1
                    if listed > _REACH_MAX_LISTED:
                        return None
                    if entry.name == ".git" or entry.is_dir(follow_symlinks=False):
                        listing.append(entry)
        except OSError:
            listing = None
        if listing is not None:
            budget -= len(listing)
            if budget < 0:
                return None
        here: str | None = real
        if real not in probed:
            probed.add(real)
            if listing is None or any(e.name == ".git" for e in listing):
                common = _common_dir(real)
                if common is not None:
                    commons.add(common)
                    here = None
            if here is not None:
                here = os.path.dirname(real)
                here = None if here == real else here
        else:
            here = None
        for _ in range(_REACH_MAX_LEVELS):
            if here is None or here in probed:
                break
            probed.add(here)
            common = _common_dir(here)
            if common is not None:
                commons.add(common)
                break
            parent = os.path.dirname(here)
            if parent == here:
                break
            here = parent
        for entry in listing or ():
            if entry.path in probed or not entry.is_dir(follow_symlinks=False):
                continue
            probed.add(entry.path)
            common = _common_dir(entry.path)
            if common is not None:
                commons.add(common)
    if not deleted or not commons:
        return []
    # Per directory: does it, or an ancestor, belong to ``deleted``? Shared
    # across roots, which share their ancestors.
    hit: dict[str, bool] = {}

    def _reached(root: str) -> bool:
        chain: list[str] = []
        here, found = root, False
        for _ in range(_REACH_MAX_LEVELS):
            if here in hit:
                found = hit[here]
                break
            chain.append(here)
            try:
                st = os.stat(here)
            except OSError:
                st = None
            if st is not None and (st.st_dev, st.st_ino) in deleted:
                found = True
                break
            parent = os.path.dirname(here)
            if parent == here:
                break
            here = parent
        for c in chain:
            hit[c] = found
        return found

    reached = {
        root for common in commons for root in _registered_worktrees(common) if _reached(root)
    }
    return sorted(reached)


def recorded_creators(path: str, *, state_root: str | os.PathLike[str] | None = None) -> list[str]:
    """Creators of the ledger rows recording exactly ``path`` (same inode generation).

    For the judge brief of a FOREIGN worktree: whose work it is, which this
    session's ``owned_record`` cannot say. Display only, never ownership.
    """
    try:
        real = os.path.realpath(path)
        conn = _connect(ledger_path(state_root), create=False)
        if conn is None:
            return []
        try:
            rows = conn.execute(
                "SELECT dev, ino, creator, stamp FROM created WHERE path = ?", (real,)
            ).fetchall()
        finally:
            conn.close()
        st = os.stat(real)
    except (OSError, sqlite3.Error):
        return []
    return sorted(
        {
            creator
            for dev, ino, creator, stamp in rows
            if (dev, ino) == (st.st_dev, st.st_ino) and stamp == creation_stamp(st)
        }
    )


#: How many dirty entries a judge brief or refusal reason shows per worktree.
_MAX_DIRTY_SHOWN = 8


def worktree_changes(path: str) -> list[str] | None:
    """The tracked and untracked (non-ignored) ``git status`` entries of a tree.

    The exact lines a judge needs to tell "unfinished work" from "finished"
    (operator escalation 2026-11-11: a brief saying only ``dirty`` gave it
    nothing to weigh). Ignored entries are left out on purpose: git's own
    ``worktree remove`` deletes those. ``None`` = git could not answer.
    """
    out = _git(
        ["status", "--porcelain", "--untracked-files=normal"],
        os.path.realpath(path),
        timeout=_WORKTREE_STATUS_TIMEOUT_S,
    )
    if out is None:
        return None
    lines = [ln.rstrip() for ln in out.splitlines() if ln.strip()]
    if len(lines) > _MAX_DIRTY_SHOWN:
        lines = [*lines[:_MAX_DIRTY_SHOWN], f"… and {len(lines) - _MAX_DIRTY_SHOWN} more"]
    return lines


def describe_changes(path: str) -> str:
    """``worktree_changes`` as one fact phrase."""
    changes = worktree_changes(path)
    if changes is None:
        return "uncommitted changes: unknown (git status failed)"
    if not changes:
        hidden = _hidden_work(path)
        return "uncommitted changes: none" + "".join(f"; {h}" for h in hidden)
    return "uncommitted changes: " + ", ".join(changes)


def _hidden_work(path: str) -> list[str]:
    """What makes a tree with no ``git status`` entry still ``dirty``.

    WHY (MINOR-3 / QA Q1, PR #3802): ``worktree_state`` also calls a tree
    dirty for a nested repository inside an IGNORED entry (its commits may
    exist nowhere else) — ignored entries are not in ``worktree_changes`` —
    so the brief read "dirty; uncommitted changes: none" and gave the judge
    nothing to weigh. Called only for a dirty tree, so the extra status is
    paid only where the judge is asked anyway.
    """
    real = os.path.realpath(path)
    # ``-z``: paths verbatim, NUL-terminated. Without it git C-quotes a
    # non-ASCII name (``"\303\251…"``), which never resolved, so that entry
    # silently dropped out of the brief (R2-NIT-1, PR #3802).
    out = _git(
        ["status", "--porcelain", "-z", "--ignored", "--untracked-files=normal"],
        real,
        timeout=_WORKTREE_STATUS_TIMEOUT_S,
    )
    facts: list[str] = []
    records = iter((out or "").split("\0"))
    for record in records:
        if len(record) > 3 and record[2] == " " and ("R" in record[:2] or "C" in record[:2]):
            next(records, None)  # a rename/copy's source path is its own record
        if not record.startswith("!! "):
            continue
        entry = record[3:].rstrip("/")
        inside = _git_inside(os.path.join(real, entry))
        if inside is None:
            continue
        if inside.exhausted:
            # Unknown, not found: the tree stays dirty (fail closed), but the
            # judge must not be told a repository exists (R2-MAJOR-1).
            facts.append(f"ignored entry too large to rule out a nested repository: {inside.path}")
        else:
            facts.append(
                f"nested repository (ignored, its commits may exist nowhere else): {inside.path}"
            )
        if len(facts) >= _MAX_DIRTY_SHOWN:
            break
    return facts


#: Bound on each ``gh pr list`` that :func:`worktree_activity` runs (one, or
#: two for a branch with a full history page — see ``_FORGE_HISTORY_LIMIT``): the
#: manager's allocation for this lookup (~8 s). Measured 2026-11-09 on this
#: machine, n=5 against damianvtran/local-operator: 1.29–1.81 s. A timeout is
#: "PR state unknown", which never ranks LOW. The ``glab mr list`` lookup shares
#: it: measured 2026-11-09, n=3 against minervaai/go-protos, 1.12–1.32 s.
_GH_TIMEOUT_S = 8.0
_GITHUB_URL_RE = re.compile(
    r"^(?:https?://(?:[^@/]+@)?github\.com/|(?:ssh://)?git@github\.com[:/])"
    r"([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
#: A GitLab remote: gitlab.com or a self-hosted host named ``gitlab…``. The
#: project path may nest groups (``group/sub/project``). WHY GitLab at all:
#: the disk lane's positive controls are Minerva (GitLab) worktrees, and with
#: GitHub alone every one of them read "PR state unknown" and went to the judge.
_GITLAB_URL_RE = re.compile(
    r"^(?:https?://(?:[^@/]+@)?|(?:ssh://)?git@)(?P<host>[^/:@]*gitlab[^/:]*)[:/]"
    r"(?P<path>[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+)/?$"
)


class ForgeChange(NamedTuple):
    """One PR (GitHub) or MR (GitLab) headed by a branch, as the guard reads it."""

    number: int
    #: ``open`` / ``merged`` / ``closed`` (GitLab may also say ``locked``).
    state: str
    #: ``YYYY-MM-DD`` of the merge/close (or last update while open); may be "".
    date: str


#: Rows the one history lookup asks for. A branch with more changes than this
#: (``release-next`` had 10+ on damianvtran/local-operator, 2026-11-11) may hide
#: an older OPEN one past the page, so a full page with no open row falls back
#: to the open-only listing the guard used before — ``pr_state`` stays exact.
_FORGE_HISTORY_LIMIT = 10


@dataclass(frozen=True)
class WorktreeActivity:
    """Is anyone still working in a linked worktree? Facts for the guard.

    WHY (operator incident, 2026-11-09). A disk-pressure lane removed
    ``local-operator-ui/.worktrees/profile-proposals``: clean, no live cwd,
    head already an ancestor of main — every ``worktree_state`` test passed —
    yet its branch carried an OPEN PR under QA, and the removal deleted 34
    tracked files from under that PR and broke Storybook. "Clean" says nothing
    is LOST; it does not say nothing is IN USE. These are the two in-use
    signals the guard can read cheaply: an open PR for the branch, and a
    running session that owns the tree.
    """

    branch: str | None
    #: ``open`` / ``none`` / ``unknown`` — the ONLY PR fact ranking reads.
    pr_state: str
    pr_number: int | None = None
    #: How the forge names it: ``PR #`` (GitHub) or ``MR !`` (GitLab).
    pr_label: str = "PR #"
    #: Why the PR state is unknown (no GitHub/GitLab remote, lookup failed, timeout).
    pr_detail: str = ""
    #: Running sessions other than the caller that own the tree.
    owners_running: tuple[str, ...] = ()
    #: The newest PR/MR ever headed by the branch, open or not. DISPLAY ONLY:
    #: ranking rests on ``pr_state`` alone. WHY shown at all (operator request,
    #: 2026-11-11): "no open PR" read the same for merged, closed-unmerged and
    #: never-proposed branches, so an ``ask`` to delete a worktree carried git
    #: ancestry but not the forge history the operator decides on. WHY not
    #: ranked: a merged PR proves the branch landed once, not that nobody is
    #: still working in the tree (head-branch names are reused — hence the
    #: date), so it must never lower a rank the open/owner signals set.
    newest: ForgeChange | None = None
    #: False when the lookup could only see OPEN changes, so the facts never
    #: claim merged/closed history nobody checked.
    history_checked: bool = True

    def facts(self) -> str:
        branch = self.branch or "(detached HEAD)"
        noun = "PR" if self.pr_label == "PR #" else "MR"
        if self.pr_state == "open":
            pr = f"open {self.pr_ref}"
        elif self.pr_state == "none":
            pr = f"no open {noun}"
            if not self.history_checked:
                pr += f"; merged/closed {noun}s not checked"
            elif self.newest is None:
                pr += f"; no {noun} found for this branch"
            else:
                n = self.newest
                state = "closed unmerged" if n.state == "closed" else n.state
                when = f" {n.date}" if n.date else ""
                pr += f"; newest {self.pr_label}{n.number} {state}{when}"
        else:
            pr = f"PR state unknown ({self.pr_detail})"
        owner = (
            "owning session running: " + ", ".join(self.owners_running)
            if self.owners_running
            else "no other running session owns it"
        )
        return f"branch {branch}; {pr}; {owner}"

    @property
    def pr_ref(self) -> str:
        return f"{self.pr_label}{self.pr_number}"


def _remote_urls(real: str) -> list[str]:
    """The worktree's remote URLs, ``origin`` first."""
    out = _git(["config", "--get-regexp", r"^remote\..*\.url$"], real)
    urls: list[tuple[bool, str]] = []
    for line in (out or "").splitlines():
        key, _, url = line.partition(" ")
        urls.append((key != "remote.origin.url", url.strip()))
    return [url for _, url in sorted(urls)]


def _github_repo(real: str) -> str | None:
    """``owner/name`` of the worktree's GitHub remote (``origin`` first)."""
    for url in _remote_urls(real):
        m = _GITHUB_URL_RE.match(url)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    return None


def _gitlab_repo(real: str) -> str | None:
    """The worktree's GitLab project as ``glab --repo`` takes it, or ``None``.

    ``group/sub/project`` on gitlab.com; ``https://<host>/<path>`` on a
    self-hosted host, so glab resolves the right instance.
    """
    for url in _remote_urls(real):
        m = _GITLAB_URL_RE.match(url)
        if m:
            host, path = m.group("host"), m.group("path").removesuffix(".git")
            return path if host == "gitlab.com" else f"https://{host}/{path}"
    return None


def _gh_open_prs(repo: str, branch: str) -> list[int] | None:
    """Open PR numbers whose head is ``branch`` on ``repo``; ``None`` if unknown.

    The open-only fallback behind :func:`_gh_prs` (tests replace it).
    ``GH_PROMPT_DISABLED`` keeps gh from ever prompting inside a guard check.
    """
    env = dict(os.environ, GH_PROMPT_DISABLED="1")
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--head",
                branch,
                "--state",
                "open",
                "--json",
                "number",
            ],
            capture_output=True,
            text=True,
            timeout=_GH_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        rows = json.loads(proc.stdout or "[]")
        return [int(r["number"]) for r in rows]
    except (ValueError, TypeError, KeyError):
        return None


def _glab_open_mrs(repo: str, branch: str) -> list[int] | None:
    """Open MR iids whose source is ``branch`` on ``repo``; ``None`` if unknown.

    The GitLab twin of :func:`_gh_open_prs` (tests replace it). ``glab mr
    list`` lists OPEN MRs by default; glab 1.91 has no ``--state`` flag, and an
    unknown flag would exit non-zero and read as "unknown" on every call.
    ``GLAB_NO_PROMPT`` keeps glab from prompting inside a guard check.
    """
    env = dict(os.environ, GLAB_NO_PROMPT="1")
    try:
        proc = subprocess.run(
            ["glab", "mr", "list", "--repo", repo, "--source-branch", branch, "-F", "json"],
            capture_output=True,
            text=True,
            timeout=_GH_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        rows = json.loads(proc.stdout or "[]")
        return [int(r["iid"]) for r in rows if r.get("state", "opened") == "opened"]
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _day(*stamps: object) -> str:
    """The first ISO timestamp's ``YYYY-MM-DD``, or "" when none is a string."""
    for stamp in stamps:
        if isinstance(stamp, str) and len(stamp) >= 10:
            return stamp[:10]
    return ""


def _gh_prs(repo: str, branch: str) -> list[ForgeChange] | None:
    """Every PR (any state, newest page) headed by ``branch``; ``None`` if unknown.

    ONE bounded call replaces the open-only listing: same ``_GH_TIMEOUT_S``
    bound, measured 2026-11-11 at 0.47–0.55 s (n=3, ``--head release-next
    --state all --limit 10`` on damianvtran/local-operator). Tests replace it.
    """
    env = dict(os.environ, GH_PROMPT_DISABLED="1")
    argv = [
        "gh",
        "pr",
        "list",
        "--repo",
        repo,
        "--head",
        branch,
        "--state",
        "all",
        "--json",
        "number,state,mergedAt,closedAt,updatedAt",
        "--limit",
        str(_FORGE_HISTORY_LIMIT),
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_GH_TIMEOUT_S, env=env, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        rows = json.loads(proc.stdout or "[]")
        out: list[ForgeChange] = []
        for r in rows:
            state = str(r["state"]).lower()  # OPEN / MERGED / CLOSED
            date = (
                _day(r.get("mergedAt"))
                if state == "merged"
                else _day(r.get("closedAt")) if state == "closed" else _day(r.get("updatedAt"))
            )
            out.append(ForgeChange(int(r["number"]), state, date))
        return out
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _glab_mrs(repo: str, branch: str) -> list[ForgeChange] | None:
    """Every MR (any state, newest page) sourced from ``branch``; ``None`` if unknown.

    glab 1.91's ``mr list`` takes ``--all`` (any state) and ``--per-page``
    (checked from ``--help``), so GitLab gets the same history as GitHub.
    GitLab's open state is ``opened``; it is normalised to ``open`` here.
    """
    env = dict(os.environ, GLAB_NO_PROMPT="1")
    argv = [
        "glab",
        "mr",
        "list",
        "--repo",
        repo,
        "--source-branch",
        branch,
        "--all",
        "--per-page",
        str(_FORGE_HISTORY_LIMIT),
        "-F",
        "json",
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_GH_TIMEOUT_S, env=env, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        rows = json.loads(proc.stdout or "[]")
        out: list[ForgeChange] = []
        for r in rows:
            state = str(r["state"])
            state = "open" if state == "opened" else state
            date = (
                _day(r.get("merged_at"))
                if state == "merged"
                else _day(r.get("closed_at")) if state == "closed" else _day(r.get("updated_at"))
            )
            out.append(ForgeChange(int(r["iid"]), state, date))
        return out
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _running_owners(real: str, session_id: str, *, strict: bool = False) -> tuple[str, ...]:
    """Running sessions (not the caller) that own the worktree at ``real``.

    Owner = the session the creation ledger attributes the ``git worktree
    add`` to (``kind='worktree'`` row for this exact path), or any live/wedged
    session whose recorded cwd is inside the tree. The CALLER is excluded: a
    session removing its own worktree is the owner acting, not a conflict.
    Read fresh on every call (no cache). Default: unreadable sources name no
    owner. ``strict`` requires current-generation attribution and a complete
    registry, raising on missing/unreadable facts rather than claiming idle.
    """
    creators: set[str] = set()
    try:
        conn = _connect(ledger_path(), create=False)
        if conn is not None:
            try:
                st = os.stat(real) if strict else None
                generation = (st.st_dev, st.st_ino, creation_stamp(st)) if st else ()
                creators = {
                    str(r[0])
                    for r in conn.execute(
                        "SELECT DISTINCT creator FROM created WHERE path=? AND kind='worktree'"
                        + (" AND dev=? AND ino=? AND stamp=?" if strict else ""),
                        (real, *generation),
                    )
                }
            finally:
                conn.close()
    except (sqlite3.Error, OSError):
        if strict:
            raise
        logger.debug("guardrail ledger unreadable", exc_info=True)
    # No attributed creator is unknown ownership, not proof of an idle tree.
    if strict and not creators:
        raise OSError("worktree owner attribution unavailable")
    running: list[str] = []
    try:
        from local_operator.session.runtime import registry

        # reap=False: a guard check must never move another session's record.
        records = (
            registry.scan(None, reap=False, strict=True)
            if strict
            else registry.scan(None, reap=False)
        )
        for rec, state in records:
            sid = str(getattr(rec, "session_id", "") or "")
            if strict and (not sid or not os.path.isabs(str(getattr(rec, "cwd", "") or ""))):
                raise OSError("session owner record incomplete")
            if state not in ("live", "wedged") or not sid or sid == session_id:
                continue
            cwd = os.path.realpath(str(getattr(rec, "cwd", "") or "/nonexistent-cwd"))
            if sid in creators or cwd == real or cwd.startswith(real + os.sep):
                running.append(sid)
    except Exception:  # noqa: BLE001 — an unreadable registry names no owner
        if strict:
            raise
        logger.debug("session registry unreadable", exc_info=True)
    return tuple(dict.fromkeys(running))


def worktree_activity(path: str, session_id: str = "") -> WorktreeActivity:
    """Open-PR and owning-session facts for a linked worktree (see the class)."""
    real = os.path.realpath(path)
    owners = _running_owners(real, session_id)
    branch = _git(["symbolic-ref", "--quiet", "--short", "HEAD"], real)
    branch = (branch or "").strip() or None
    if branch is None:
        # Detached: no branch, so no PR can be headed by it.
        return WorktreeActivity(None, "none", owners_running=owners)
    repo = _github_repo(real)
    tool, label, lookup, open_lookup = "gh", "PR #", _gh_prs, _gh_open_prs
    if repo is None:
        repo = _gitlab_repo(real)
        tool, label, lookup, open_lookup = "glab", "MR !", _glab_mrs, _glab_open_mrs
    if repo is None:
        return WorktreeActivity(
            branch, "unknown", pr_detail="no GitHub or GitLab remote", owners_running=owners
        )
    unknown = WorktreeActivity(
        branch,
        "unknown",
        pr_label=label,
        pr_detail=f"{tool} lookup on {repo} failed or timed out",
        owners_running=owners,
    )
    changes = lookup(repo, branch)
    if changes is None:
        return unknown
    opened = [c.number for c in changes if c.state == "open"]
    newest = max(changes, key=lambda c: c.number, default=None)
    if not opened and len(changes) >= _FORGE_HISTORY_LIMIT:
        # A full page may hide an older open change (see _FORGE_HISTORY_LIMIT):
        # only the open-only listing can prove "none". Reused branches only.
        fallback = open_lookup(repo, branch)
        if fallback is None:
            return unknown
        opened = fallback
    if opened:
        return WorktreeActivity(
            branch,
            "open",
            pr_number=min(opened),
            pr_label=label,
            owners_running=owners,
            newest=newest,
        )
    return WorktreeActivity(branch, "none", pr_label=label, owners_running=owners, newest=newest)


def _ignored_unregenerable(real: str) -> bool:
    """An ignored entry outside :data:`_REGENERABLE`, or a nested repository
    in one (``None`` from git counts: fail closed)."""
    ignored = _git(["status", "--porcelain", "--ignored"], real)
    if ignored is None:
        return True
    for line in ignored.splitlines():
        if not line.startswith("!! "):
            continue
        entry = line[3:].strip().strip('"').rstrip("/")
        name = os.path.basename(entry)
        if not _regenerable(name):
            return True
        if _holds_git(os.path.join(real, entry)):
            return True
    return False


def _source_checkout() -> str | None:
    """The checkout this code runs from, when it runs from one (a dev run)."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    return here if os.path.exists(os.path.join(here, ".git")) else None


def _is_repo_at(where: str) -> bool:
    """Is ``where`` ITSELF a git repository (work tree, bare, or linked)?

    ``git -C`` discovers upward, so a plain directory inside some other
    repository would pass ``rev-parse`` alone: the git dir found must be
    ``where`` (bare), ``where/.git``, or belong to a work tree rooted there.
    """
    if not os.path.isdir(where):
        return False
    real = os.path.realpath(where)
    git_dir = _git(["rev-parse", "--absolute-git-dir"], real)
    if git_dir is None:
        return False
    found = os.path.realpath(git_dir.strip())
    if found in (real, os.path.join(real, ".git")):
        return True
    top = _git(["rev-parse", "--show-toplevel"], real)
    return top is not None and os.path.realpath(top.strip()) == real


def _tag_commits(repo: str) -> dict[str, str] | None:
    """``tag name -> peeled commit`` for every tag in ``repo`` (one call)."""
    out = _git(
        ["for-each-ref", "--format=%(refname:strip=2) %(*objectname) %(objectname)", "refs/tags"],
        repo,
    )
    if out is None:
        return None
    tags: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            # An annotated tag peels to ``%(*objectname)``; a lightweight one
            # leaves it empty, so its own object is the commit.
            tags[parts[0]] = parts[1] if len(parts) == 3 else parts[-1]
    return tags


def _local_tag_commits(real: str) -> set[str] | None:
    """Commits of the clone's tags that did NOT come from its origin.

    WHY. A local clone copies every tag of its source (861 on this machine's
    field clone), many on commits no branch holds, so counting all tags made
    every real clone dirty. Only a tag the origin lacks can be the last name
    for a commit:
      * origin is a local path or ``file://`` that exists: a tag is the
        origin's when its commit there is the same (both sides listed with
        one ``for-each-ref``, compared here);
      * otherwise (a network origin; never contacted): ``git clone`` writes
        tags to ``packed-refs``, while ``git tag`` — and a later ``fetch`` —
        write LOOSE refs under ``.git/refs/tags/``. Loose ones are treated as
        possibly local; that is safe because their commits are then judged by
        ``--not --remotes``, so a fetched tag on a pushed commit stays clean.
    Accepted limitation: a ``git pack-refs`` after tagging makes a local tag
    look cloned under a network origin.
    """
    tags = _tag_commits(real)
    if tags is None:
        return None
    if not tags:
        return set()
    url = (_git(["config", "--get", "remote.origin.url"], real) or "").strip()
    local = url[len("file://") :] if url.startswith("file://") else url
    if local.startswith(("/", "~", ".")):
        origin = os.path.join(real, os.path.expanduser(local))
        theirs = _tag_commits(origin) if os.path.isdir(origin) else None
        if theirs is None:
            return None
        return {sha for name, sha in tags.items() if theirs.get(name) != sha}
    loose_root = os.path.join(real, ".git", "refs", "tags")
    loose: set[str] = set()
    for dirpath, _dirs, files in os.walk(loose_root):
        for name in files:
            loose.add(os.path.relpath(os.path.join(dirpath, name), loose_root))
    return {sha for name, sha in tags.items() if name in loose}


def clone_state(path: str) -> str:
    """``clean`` when deleting the standalone clone at ``path`` loses nothing.

    WHY. The operator's delete spec lists "a clone" among what its creator may
    remove, but a primary checkout (a ``.git`` DIRECTORY) may hold the only
    copy of work. History may vouch for one (``evidence.covers``) only when
    this says ``clean``: ALL of
      * not a repository with linked worktrees (the operator's checkouts,
        ``~/local-operator`` with its ``.worktrees/``, look like that) and
        not the checkout this code itself runs from;
      * no tracked, untracked or non-regenerable ignored change — or, for a
        ``--no-checkout`` clone (no index, nothing but ``.git``), no work
        tree at all, whose "staged deletions" are only files never written;
      * every local branch and HEAD reachable from a remote-tracking ref
        (``refs/remotes``), and so is every tag the origin lacks
        (``_local_tag_commits``), so no commit exists only here;
      * no stash, and no nested repository in the work tree.
    Anything else is ``dirty: <why>`` (or ``unknown``), and the clone stays
    unowned: the ordinary ranking, with this state in the judge's facts.
    Read-only git, same environment as :func:`worktree_state`.
    """
    try:
        real = os.path.realpath(path)
        if not os.path.isdir(os.path.join(real, ".git")):
            return "not-a-clone"
        if real == _source_checkout():
            return "dirty: the running harness's own checkout"
    except OSError:
        return "unknown"
    listing = _git(["worktree", "list", "--porcelain"], real)
    if listing is None:
        return "unknown"
    if sum(ln.startswith("worktree ") for ln in listing.splitlines()) > 1:
        return "dirty: has linked worktrees"
    try:
        never_checked_out = not os.path.exists(os.path.join(real, ".git", "index")) and (
            os.listdir(real) == [".git"]
        )
    except OSError:
        return "unknown"
    # A local remote that is gone means the remote-tracking refs below name
    # commits nothing else holds any more (review M2).
    remotes = _git(["config", "--get-regexp", r"^remote\..*\.url$"], real) or ""
    for line in remotes.splitlines():
        url = line.split(" ", 1)[1].strip() if " " in line else ""
        local = url[len("file://") :] if url.startswith("file://") else url
        if local.startswith(("/", "~", ".")):
            where = os.path.join(real, os.path.expanduser(local))
            # A repository, not merely a path: ``mv src src-old && mkdir src``
            # leaves a directory that holds none of the commits the
            # remote-tracking refs name (review round 4, MAJOR-1).
            if not _is_repo_at(where):
                return "dirty: origin gone"
    if not never_checked_out:
        # assume-unchanged (lowercase tag) and skip-worktree (``S``) entries
        # hide edits from ``git status`` (review M1).
        flags = _git(["ls-files", "-v"], real)
        if flags is None:
            return "unknown"
        # A sparse checkout marks every path outside its cone ``S``: there is
        # no work-tree copy to hide, so only assume-unchanged counts there
        # (review round 4, MINOR-1).
        sparse = (_git(["config", "--bool", "core.sparseCheckout"], real) or "").strip()
        hidden_tags = "" if sparse == "true" else "S"
        if any(
            (hidden_tags and ln[:1] == hidden_tags) or ln[:1].islower() for ln in flags.splitlines()
        ):
            return "dirty: files hidden from git status"
        status = _git(["status", "--porcelain", "--untracked-files=all"], real)
        if status is None:
            return "unknown"
        if status.strip():
            return "dirty: uncommitted or untracked changes"
        if _ignored_unregenerable(real):
            return "dirty: ignored files that are not regenerable"
        try:
            nested = any(
                _holds_git(e.path)
                for e in os.scandir(real)
                if e.name != ".git" and e.is_dir(follow_symlinks=False)
            )
        except OSError:
            return "unknown"
        if nested:
            return "dirty: a nested repository"
    if _git(["rev-parse", "-q", "--verify", "refs/stash"], real) is not None:
        return "dirty: a stash"
    # One walk, not a ``--contains`` per ref: commits reachable from HEAD or a
    # local branch but from no remote-tracking ref exist only here.
    unpushed = _git(["rev-list", "--count", "HEAD", "--branches", "--not", "--remotes"], real)
    if unpushed is None or not unpushed.strip().isdigit():
        return "unknown"
    if int(unpushed) == 0:
        local = _local_tag_commits(real)
        if local is None:
            return "unknown"
        if local:
            # A tag the origin lacks may be the only name for its commit
            # (review M1). Copied tags are not: a local clone copies hundreds.
            # (A tag on a tree or blob fails or counts here too: fail closed,
            # whatever the message says.)
            unpushed = _git(["rev-list", "--count", *sorted(local), "--not", "--remotes"], real)
            if unpushed is None or not unpushed.strip().isdigit():
                return "unknown"
    if int(unpushed) > 0:
        return "dirty: a commit no remote-tracking ref contains"
    return "clean"


def is_primary_checkout(path: str) -> bool:
    try:
        return os.path.isdir(os.path.join(path, ".git"))
    except OSError:
        return False


# -- trusted operator words ------------------------------------------------------


@dataclass(frozen=True)
class OperatorWords:
    """One trusted row: ``label`` says why it is trusted."""

    label: str
    text: str
    #: For an ask answer: the question/option text and the chosen answers, so
    #: path naming is checked against what the OPERATOR saw.
    ask_id: str = ""
    session_id: str = ""
    #: The part of an ask answer that approves with no judge reading it
    #: (``_approved_text``): the yes/no question the operator said yes to, or
    #: the picked options. ``DeleteCase.evidence_basis`` settles on it.
    approved: str = ""


@dataclass
class OperatorBrief:
    words: list[OperatorWords] = field(default_factory=list)
    #: kind -> count of untrusted rows deliberately left out (never their text).
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def latest_direct(self) -> str:
        for w in self.words:
            if w.label == "operator, direct":
                return w.text
        return ""


def _custom_kind(custom_type: str) -> str | None:
    from local_operator.harness.message_types import (
        HUB_MESSAGE_TYPE,
        PEER_MESSAGE_MESSAGE_TYPE,
        TEAM_BRIDGE_MESSAGE_TYPE,
    )

    return {
        PEER_MESSAGE_MESSAGE_TYPE: "peer-agent message",
        TEAM_BRIDGE_MESSAGE_TYPE: "Team Bridge delivery",
        HUB_MESSAGE_TYPE: "hub message",
    }.get(custom_type)


def verified_ask(
    session_id: str,
    ask_id: str,
    *,
    state_root: str | os.PathLike[str] | None = None,
) -> OperatorWords | None:
    """The ask ``ask_id`` of ``session_id`` as consent, if the STORE proves it.

    All must hold: the ask exists in that session's durable log, it was
    ``answered``/``late``, the answer came from a human surface
    (:data:`TRUSTED_ASK_SURFACES`), and it is not a Team Bridge question (a
    ``question_id`` means the sender's picks, not the operator's). Whether the
    answer APPROVES the delete is a judgement left to the judge, which sees the
    question, the options and the chosen answer verbatim.
    """
    if not session_id or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
        return None
    try:
        from local_operator.asks import store
        from local_operator.paths import config_dir

        root = Path(state_root) if state_root is not None else config_dir()
        directory = store.session_dir(root, session_id)
        if not directory.is_dir():
            return None
        records = store.fold(store.read_events(directory), store.now_ms())
    except Exception:  # noqa: BLE001 — an unreadable store is no consent
        logger.debug("guardrail: ask store unreadable", exc_info=True)
        return None
    record = next((r for r in records if str(r.get("ask_id")) == ask_id), None)
    return _ask_words(record, session_id) if record is not None else None


def _answer_declines(answer: str, labels: Sequence[str]) -> bool:
    """Does one chosen answer refuse (or carve out) what the ask proposed?

    An option the operator PICKED is judged by its label ("Keep them", "No,
    re-audit instead"). A free-text answer is read clause-wise (incident
    2026-11-09, ask a-daca): "Go ahead and copy but also send this to Aida to
    tune the guardrails so they don't block this" approves — its ``don't``
    is about the guard, not the copy — while "Yes, but keep the third one"
    and "Don't delete them" still refuse. The first clause may not decline at
    all, and no clause may carve anything out (``except``, ``keep``, ``don't
    delete``…). When in doubt it declines: the judge never sees a veto.
    """
    if answer in labels:
        return bool(_DECLINE_RE.search(answer))
    # The first clause may not decline; no clause may carve anything out
    # ("Go ahead, except the last one").
    head = _CLAUSE_BREAK_RE.split(answer, maxsplit=1)[0]
    return bool(_DECLINE_RE.search(head)) or bool(_CARVE_OUT_RE.search(answer))


def _ask_words(record: Mapping[str, Any], session_id: str) -> OperatorWords | None:
    """One folded ask record as consent (see :func:`verified_ask` for the bar)."""
    ask_id = str(record.get("ask_id") or "")
    if str(record.get("status") or "") not in ("answered", "late"):
        return None
    surface = str((record.get("answered_by") or {}).get("surface") or "")
    if surface not in TRUSTED_ASK_SURFACES:
        return None
    questions = [q for q in record.get("questions") or [] if isinstance(q, Mapping)]
    if any(str(q.get("question_id") or "") for q in questions):
        return None
    answers = record.get("answers") or {}
    parts: list[str] = []
    approved: list[str] = []
    for q in questions:
        if q.get("secret"):
            continue
        chosen = [str(v) for v in (answers.get(str(q.get("id") or "")) or [])]
        opts = [o for o in (q.get("options") or []) if isinstance(o, Mapping)]
        labels = [str(o.get("label") or "") for o in opts]
        if not chosen or any(_answer_declines(v, labels) for v in chosen):
            # Deterministic veto (review R1-MINOR-3): a declining answer is
            # never consent, whatever a judge that read the history concludes.
            continue
        options = "; ".join(labels)
        parts.append(
            f"Q: {q.get('question') or ''}"
            + (f" [options: {options}]" if options else "")
            + f" -> operator answered: {', '.join(chosen)}"
        )
        approved.extend(_approved_text(q, opts, labels, chosen))
    if not parts:
        return None
    return OperatorWords(
        label="operator, answer to ask",
        text="\n".join(parts),
        ask_id=ask_id,
        session_id=session_id,
        approved="\n".join(approved),
    )


def _approved_text(
    q: Mapping[str, Any],
    opts: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    chosen: list[str],
) -> list[str]:
    """What this answer approves WITHOUT a judge reading it (``OperatorWords.approved``).

    Settled deterministically only when the answer can mean one thing:
    * the WHOLE question, when it is a yes/no ask (every option but the one
      picked declines) and each picked label is a bare affirmative over the
      whole set — "Yes", "Delete", a-a382's "Remove all 57", "Go ahead" — or
      the operator typed a bare affirmative ("Go ahead and copy");
    * otherwise only the paths a picked label NAMES ITSELF, and only when
      each picked label says yes and nothing in it declines.
    An option's description approves nothing (review R2-M1): "Delete es only"
    described as "…; fr remains" must not settle fr, and a narrowing label
    ("Delete es only", "Delete the es receipt") must not approve the
    question's other paths. Anything else — a qualified typed answer, a pick
    like "Installer only" / "Abort" — still reaches the judge as the
    operator's words (``OperatorWords.text``), unless ``_answer_declines``
    vetoed it.
    """
    out: list[str] = []
    picked = [o for o in opts if str(o.get("label") or "") in chosen]
    typed = [v for v in chosen if v not in labels]
    # A pick approves only when its label says yes and nothing in label or
    # description declines (review R1-M1: "Abort" / "Leave … in place").
    affirmed = [
        o
        for o in picked
        if _AFFIRM_OPTION_RE.match(str(o.get("label") or ""))
        and not _DECLINE_RE.search(f"{o.get('label') or ''} {o.get('description') or ''}")
    ]
    if len(affirmed) != len(picked):
        return []
    if typed and not all(_BARE_AFFIRM_RE.fullmatch(v) for v in typed):
        # A qualified typed answer is the judge's to read (R1-B2); the
        # question text is never approved by it, nor are picks beside it.
        return []
    # Only the label: it is what the operator read and chose. The label's own
    # paths are approved; a description is context, never consent (R2-M1).
    out.extend(str(o.get("label") or "") for o in affirmed)
    others = [lbl for lbl in labels if lbl not in chosen]
    yes_no = bool(picked) and bool(others) and all(_DECLINE_RE.search(lbl) for lbl in others)
    whole = all(_WHOLE_SET_OPTION_RE.fullmatch(str(o.get("label") or "")) for o in affirmed)
    if (yes_no and whole) or (typed and not picked):
        out.append(str(q.get("question") or ""))
    return out


def lineage_asks(
    session_id: str,
    *,
    known: Iterable[tuple[str, str]] = (),
    state_root: str | os.PathLike[str] | None = None,
    now_ms: int | None = None,
) -> list[OperatorWords]:
    """Recent answered asks in ``session_id``'s lineage, no pointer needed.

    WHY. A task tree is one owner: the operator answers an ask in the parent
    and a subagent runs the approved command (incident 2026-11-09, a-daca was
    answered in 9ad475b6f5d9 while the copy ran lower in its tree). A pointer
    (``a-xxxx`` in the brief) was the only way to reach that ask; a brief that
    forgot it left the guard blind to consent it could verify. Bounded: only
    the lineage (ancestors, never siblings or strangers), only asks answered
    in the last :data:`_LINEAGE_ASK_WINDOW_MS`, at most
    :data:`_LINEAGE_ASKS_PER_SESSION` per session, every one through the same
    store checks as a pointer (:func:`_ask_words`). ``known`` = ``(ask_id,
    session_id)`` pairs already in the brief.
    """
    try:
        from local_operator.asks import store
        from local_operator.paths import config_dir

        root = Path(state_root) if state_root is not None else config_dir()
        lineage, _ = session_lineage(session_id, state_root=root)
        now = now_ms if now_ms is not None else store.now_ms()
    except Exception:  # noqa: BLE001 — no lineage, no extra consent
        logger.debug("guardrail: lineage asks unreadable", exc_info=True)
        return []
    seen = set(known)
    out: list[OperatorWords] = []
    for sid in lineage:
        try:
            directory = store.session_dir(root, sid)
            if not directory.is_dir():
                continue
            records = store.fold(store.read_events(directory), now)
        except Exception:  # noqa: BLE001 — an unreadable store is no consent
            continue
        recent = [
            r
            for r in records
            if now - int(r.get("answered_at") or 0) <= _LINEAGE_ASK_WINDOW_MS
            and (str(r.get("ask_id") or ""), sid) not in seen
        ]
        recent.sort(key=lambda r: int(r.get("answered_at") or 0), reverse=True)
        for record in recent[:_LINEAGE_ASKS_PER_SESSION]:
            words = _ask_words(record, sid)
            if words is None:
                continue
            label = words.label + (f" (session {sid})" if sid != session_id else "")
            out.append(replace(words, label=label))
    return out


def _text_of(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    details = getattr(message, "details", None) or {}
    return str(details.get("body") or details.get("text") or "")


def read_origin(root: Path, session_id: str) -> dict[str, Any] | None:
    """``origin.json`` of a session; ``{}`` when it has none.

    Opened with ``O_NOFOLLOW`` and refused when the session directory itself
    is a symlink: a dangling ``ln -sf`` raised ``FileNotFoundError`` and so
    read as "operator-opened" (review R2-MAJOR-2). A symlink is unreadable
    here — neither human nor a lineage pointer.
    """
    directory = root / "sessions" / session_id
    if os.path.islink(directory):
        return None
    try:
        fd = os.open(directory / "origin.json", os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return {} if os.path.isdir(directory) else None
    except OSError:
        return None
    try:
        with os.fdopen(fd, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def session_lineage(
    session_id: str,
    ancestors: Sequence[str] = (),
    *,
    state_root: str | os.PathLike[str] | None = None,
) -> tuple[list[str], bool]:
    """``(lineage, operator_opened)`` for ``session_id``.

    Lineage = the session, its harness-known ancestors, and every parent or
    opener their ``origin.json`` names (``parent`` for a subagent,
    ``opened_by.session`` for an agent-opened workstream), followed upward.
    Those are the only ask stores a pointer may resolve in (review R1-MINOR-1:
    a bare hex string must not aim the lookup at any session on disk).
    The harness lineage registry (``read_lineage``) is read first; any shell
    path into ``sessions/`` ranks CRITICAL, and ``origin.json`` is read
    without following symlinks.

    ``operator_opened`` (review R2-MAJOR-2) is True only when the harness
    knows of NO parent or opener: no in-process ``ancestors``, no registry
    row, and an ``origin.json`` that is genuinely absent from a real session
    directory — how every session a person opened (terminal, desktop, phone)
    is stored. A missing ``origin.json`` alone is never enough.
    """
    from local_operator.paths import config_dir

    root = Path(state_root) if state_root is not None else config_dir()
    valid = re.compile(r"[A-Za-z0-9_-]{1,64}")
    registry = read_lineage(root)
    lineage: list[str] = []
    queue = [session_id, *ancestors]
    operator_opened = False
    while queue and len(lineage) < 32:
        sid = queue.pop(0)
        if not sid or sid in lineage or not valid.fullmatch(sid):
            continue
        lineage.append(sid)
        row = registry.get(sid)
        origin = read_origin(root, sid)
        if sid == session_id:
            operator_opened = not ancestors and row is None and origin == {}
        pointers: list[object] = []
        if row is not None:
            pointers += [row.get("parent"), row.get("opener")]
        if origin:
            opened_by = origin.get("opened_by")
            pointers += [
                origin.get("parent"),
                opened_by.get("session") if isinstance(opened_by, Mapping) else None,
            ]
        queue.extend(p for p in pointers if isinstance(p, str))
    return lineage, operator_opened


def task_tree_root(session_id: str, *, state_root: str | os.PathLike[str] | None = None) -> str:
    """The top of ``session_id``'s SUBAGENT tree: ``parent`` links only.

    WHY parent-only (case 7, 2026-11-09): sibling lanes under one manager
    (``a13e8dbc573f``'s auditor and settings lane) are one owner, but an
    agent-opened workstream (``opened_by``) is a separate task, so every
    workstream an assistant ever opened never becomes one tree. The registry
    row is read first (``read_lineage``), then ``origin.json``.
    """
    from local_operator.paths import config_dir

    root = Path(state_root) if state_root is not None else config_dir()
    registry = read_lineage(root)
    valid = re.compile(r"[A-Za-z0-9_-]{1,64}")
    seen: list[str] = []
    sid = session_id
    while sid and valid.fullmatch(sid) and sid not in seen and len(seen) < 32:
        seen.append(sid)
        row = registry.get(sid) or {}
        parent = row.get("parent") or (read_origin(root, sid) or {}).get("parent")
        if not isinstance(parent, str) or not parent:
            break
        sid = parent
    return seen[-1] if seen else session_id


def same_task_tree(a: str, b: str, *, state_root: str | os.PathLike[str] | None = None) -> str:
    """How ``b`` relates to ``a`` in one subagent tree, or ``""`` if it does not.

    ``"ancestor"``, ``"descendant"`` or ``"sibling"`` (same root, neither above
    the other). Shown to the judge verbatim as the lineage fact.
    """
    if not a or not b or a == b:
        return ""
    root_a = task_tree_root(a, state_root=state_root)
    if root_a != task_tree_root(b, state_root=state_root):
        return ""
    lineage_a, _ = session_lineage(a, state_root=state_root)
    if b in lineage_a:
        return "ancestor"
    lineage_b, _ = session_lineage(b, state_root=state_root)
    return "descendant" if a in lineage_b else "sibling"


def tracked_script(path: str | None, *, state_roots: Iterable[str] = ()) -> bool:
    """Is ``path`` a committed, unmodified file of a repository outside every session dir?

    Review R2-m1: "in the index" was not enough. A tracked script with
    uncommitted edits, or one staged in a repo the agent just ``git init``-ed
    in a scratchpad, is code the agent wrote, not the project's reviewed
    tool, so it keeps the HIGH. The bar is therefore: tracked
    (``ls-files --error-unmatch``), identical to ``HEAD`` (``diff --quiet``,
    which also fails when there is no commit) and in a repository whose
    top level is not under any state root's ``sessions/`` directory.
    """
    if not path or os.path.islink(path) or not os.path.isfile(path):
        return False
    directory, name = os.path.split(path)
    if _git(["ls-files", "--error-unmatch", "--", name], directory) is None:
        return False
    if _git(["diff", "--quiet", "HEAD", "--", name], directory) is None:
        return False
    top = _git(["rev-parse", "--show-toplevel"], directory)
    if not top:
        return False
    top_real = os.path.realpath(top.strip())
    for state in state_roots:
        for spelling in {state, os.path.realpath(state)}:
            sessions = os.path.join(spelling, "sessions") + os.sep
            if (top_real + os.sep).startswith(sessions) or (top.strip() + os.sep).startswith(
                sessions
            ):
                return False
    return True


def operator_brief(
    messages: Any,
    *,
    session_id: str = "",
    ancestors: Sequence[str] = (),
    cited: Sequence[str] = (),
    state_root: str | os.PathLike[str] | None = None,
) -> OperatorBrief:
    """The trusted operator words in the recent transcript, newest first.

    Trusted: a plain user row with no harness stamp — ONLY in a session the
    operator opened (``session_lineage``): in a subagent or an agent-opened
    workstream that row is the delegating agent's brief, which may relay a
    Team Bridge or peer request (review R1-MAJOR-3 / QA Q3), so it is counted
    as excluded and never quoted. And an ask answered by a human surface,
    checked in the STORE of a session in this session's lineage. Everything
    else is excluded BY STAMP: harness injections by ``harness_injected``;
    peer, hub and Team Bridge rows by their ``custom_type``. Excluded rows are
    counted by kind so the judge knows something was left out, but their text
    never reaches it.

    Ask ids (``a-xxxx``) cited anywhere — a peer body, an operator row, the
    call's intent or command — are POINTERS only, resolved against the
    lineage's ask stores; the store decides.
    """
    from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
    from local_operator.harness.rows import is_harness_chrome, is_harness_injection

    brief = OperatorBrief()
    lineage, operator_opened = session_lineage(session_id, ancestors, state_root=state_root)
    direct = 0
    pointers: list[tuple[str, str]] = []  # (ask_id, relayed_by)
    try:
        rows = list(messages or ())[-_SCAN_LIMIT:]
    except Exception:  # noqa: BLE001
        rows = []
    for message in reversed(rows):
        try:
            custom_type = getattr(message, "custom_type", None)
            if custom_type is not None:
                if custom_type == ASK_RESPONSE_MESSAGE_TYPE:
                    ask_id = str((getattr(message, "details", None) or {}).get("ask_id") or "")
                    if ask_id:
                        pointers.append((ask_id, ""))
                    continue
                kind = _custom_kind(str(custom_type))
                if kind is None:
                    continue
                brief.excluded[kind] = brief.excluded.get(kind, 0) + 1
                details = getattr(message, "details", None) or {}
                sender = details.get("sender") if isinstance(details, Mapping) else None
                sender = sender if isinstance(sender, Mapping) else {}
                who = str(sender.get("conversation_name") or sender.get("session_id") or kind)
                for ask_id in _ASK_ID_RE.findall(_text_of(message)):
                    pointers.append((ask_id, who))
                continue
            if getattr(message, "role", None) != "user":
                continue
            if is_harness_injection(message):
                continue
            text = getattr(message, "text", "")
            if not isinstance(text, str) or not text.strip() or is_harness_chrome(text):
                continue
            pointers.extend((ask_id, "") for ask_id in _ASK_ID_RE.findall(text))
            if not operator_opened:
                kind = "delegating-agent message"
                brief.excluded[kind] = brief.excluded.get(kind, 0) + 1
                continue
            if direct < _MAX_DIRECT:
                brief.words.append(OperatorWords(label="operator, direct", text=text))
                direct += 1
        except Exception:  # noqa: BLE001 — one odd row never breaks the brief
            continue
    for text in cited:
        pointers.extend((ask_id, "") for ask_id in _ASK_ID_RE.findall(text or ""))
    asks = 0
    seen: set[tuple[str, str]] = set()
    for ask_id, relayed_by in pointers:
        for sid in lineage:
            if asks >= _MAX_ASKS or (ask_id, sid) in seen:
                continue
            seen.add((ask_id, sid))
            found = verified_ask(sid, ask_id, state_root=state_root)
            if found is None:
                continue
            label = found.label + (f" (relayed by {relayed_by})" if relayed_by else "")
            if sid != session_id:
                label += f" (session {sid})"
            brief.words.append(replace(found, label=label, session_id=sid))
            asks += 1
    return brief


# -- naming --------------------------------------------------------------------

_GLOBISH = re.compile(r"[*?\[]")
_PATHISH = re.compile(r"[`'\"]?((?:~|/)[^\s`'\",;)]*)")


def _expand_word(word: str, home: str, cwd: str | None) -> str | None:
    word = word.rstrip(".:")
    if word.startswith("~"):
        if word != "~" and not word.startswith("~/"):
            return None
        word = home + word[1:]
    elif not word.startswith("/"):
        # Only absolute and ``~`` spellings name a path exactly (R1-MINOR-2).
        del cwd
        return None
    return os.path.normpath(word)


#: One trailing character prose may put after a path. Closing quotes,
#: backticks, commas, semicolons and ``)`` already end a word in ``_PATHISH``.
_PROSE_END = frozenset("?!:]}")


#: A path word holding brace lists, as asks write a set of siblings:
#: ``~/x/merge-{es,fr,zh-cn}-909a.json``, ``~/r/.worktrees/{a, b (QA
#: scratch), c}`` or ``~/x/{a,b}/{c,d}``. A list may hold spaces and
#: parenthetical notes. The lookbehind anchors the word at its start, so no
#: tail of a word (``/{c,d}``) is ever read as a path of its own.
_BRACE_PATH_RE = re.compile(
    r"(?<![^\s`'\"(\[])((?:~|/)(?:[^\s`'\",;(){}]|\{[^{}]{1,4000}\})*\{[^{}]{1,4000}\}"
    r"(?:[^\s`'\",;(){}]|\{[^{}]{1,4000}\})*)"
)
_BRACE_LIST_RE = re.compile(r"\{([^{}]*)\}")
_MAX_PROSE_BRACE_ITEMS = 256


def _expand_prose_braces(text: str) -> str:
    """Spell out brace lists in path words, so each leaf is named literally.

    WHY (incident 2026-11-09, ask a-daca): the operator approved "…/onboarding-
    value-merge-{es,fr,hi,ru,ur,vi,zh-cn}-909a2c50a0bd.json"; ``_PATHISH``
    stopped at the first comma, so no receipt was "named". Each item must be
    one path piece (no whitespace once its ``(note)`` is dropped, no wildcard
    added) or the whole word is left as written — never guess a name.

    Review R1-M2 (PR #3710): a word yields ONLY its fully expanded leaves
    (``~/x/{a,b}/{c,d}`` → four paths, every list expanded), never a
    half-expanded word whose first piece is a PARENT of what the operator
    named. ``.``/``..``/empty items are refused: ``~/r/{.,build}`` would
    name ``~/r`` itself, approving a recursive delete of the whole tree.
    """

    def spell(match: re.Match[str]) -> str:
        word = match.group(1)
        pieces = _BRACE_LIST_RE.split(word)  # literal, items, literal, items, …
        lists: list[list[str]] = []
        for body in pieces[1::2]:
            items = [re.sub(r"\([^()]*\)", "", item).strip() for item in body.split(",")]
            if len(items) < 2 or any(
                not item
                or re.search(r"[\s*?\[{}]", item)
                or any(part in ("", ".", "..") for part in item.split("/"))
                for item in items
            ):
                return word  # not a sibling list: left exactly as written
            lists.append(items)
        if math.prod(len(items) for items in lists) > _MAX_PROSE_BRACE_ITEMS:
            return word
        leaves = []
        for combo in itertools.product(*lists):
            leaf = pieces[0]
            for item, literal in zip(combo, pieces[2::2], strict=True):
                leaf += item + literal
            leaves.append(leaf)
        return " ".join(leaves)

    return _BRACE_PATH_RE.sub(spell, text)


def _path_readings(text: str) -> list[tuple[str, bool]]:
    """``(word, may_be_a_glob)`` readings of every path-looking word in ``text``.

    Two per word. As written (``rstrip("/.:")``), where a trailing ``?`` is a
    glob, so ``~/qa-r5-dg-?`` still covers its matches. As prose, at most ONE
    trailing sentence mark is dropped ("Delete ~/x/or.env?") and the rest is
    compared LITERALLY, never as a glob, so a repeated run (``or.env??``)
    never names ``or.env`` (review round 4, MINOR-2).
    """
    out: list[tuple[str, bool]] = []
    for match in _PATHISH.finditer(_expand_prose_braces(text)):
        raw = match.group(1)
        out.append((raw.rstrip("/.:"), True))
        stripped = raw[:-1].rstrip("/.:") if raw[-1:] in _PROSE_END else ""
        if len(stripped) > 1:
            out.append((stripped, False))
    return out


def names_path(
    text: str,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> bool:
    """Does ``text`` (operator words) name ``path``, literally or by a safe glob?

    Literal: the exact path (realpath or as spelled) or its ``~`` form —
    never a ``parent/basename`` suffix, which names every same-named path in
    every repository (review R1-MINOR-2), and never a cwd-relative word. A glob
    covers ``path`` only when (a) it matches the realpath with ``fnmatch``,
    component by component, (b) ``path`` sits under the glob's literal parent
    directory, (c) that parent is not ``$HOME``'s ancestor and not a
    protected root other than ``$HOME`` itself, (d) the wildcard sits only in
    the last component and leaves at least three literal characters there
    (``~/qa-r5-dg-*`` names the operator's rigs; ``~/*`` names nothing in
    particular), and (e) ``path`` is not a primary checkout. CRITICAL targets
    never reach this function: the caller ranks them first.
    """
    if not text or not path:
        return False
    real = os.path.realpath(path)
    norm = os.path.normpath(path)
    candidates = {real, norm}
    for cand in list(candidates):
        if cand.startswith(home.rstrip("/") + "/"):
            candidates.add("~" + cand[len(home.rstrip("/")) :])
    protected = {os.path.normpath(p) for p in protected}
    for word, as_glob in _path_readings(text):
        if not word:
            continue
        if not as_glob or not _GLOBISH.search(word):
            if word in candidates or word.rstrip("/") in candidates:
                return True
            expanded = _expand_word(word, home, cwd)
            if expanded is not None and expanded in (real, norm):
                return True
            continue
        pattern = _expand_word(word, home, cwd)
        if pattern is None:
            continue
        last = pattern.rsplit("/", 1)[-1]
        if _GLOBISH.search(pattern.rsplit("/", 1)[0]) or len(_GLOBISH.sub("", last)) < 3:
            # ``~/*``, ``~/??*`` or a wildcard above the last component names a
            # class of directories, not the operator's chosen ones: never consent.
            continue
        literal = pattern[: _GLOBISH.search(pattern).start()]  # type: ignore[union-attr]
        parent = literal[: literal.rfind("/")] if "/" in literal else ""
        parent = os.path.normpath(parent or "/")
        if parent == "/" or home.startswith(parent.rstrip("/") + "/"):
            continue
        if parent in protected and parent != os.path.normpath(home):
            continue
        if is_primary_checkout(real):
            # A glob never vouches for a repository's primary checkout; only
            # its literal name does (the branch above).
            continue
        for spelled in (real, norm):
            if not spelled.startswith(parent.rstrip("/") + "/"):
                continue
            pat = [c for c in pattern.split("/") if c]
            got = [c for c in spelled.split("/") if c]
            if len(pat) == len(got) and all(
                fnmatch.fnmatchcase(g, p) for g, p in zip(got, pat, strict=True)
            ):
                return True
    return False
