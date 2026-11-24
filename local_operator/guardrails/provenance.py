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
import functools
import glob as _glob
import itertools
import json
import logging
import math
import os
import re
import shutil
import signal
import sqlite3
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

from local_operator.guardrails import spawned

logger = logging.getLogger(__name__)

LEDGER_DIRNAME = "guardrails"
LEDGER_FILENAME = "created.sqlite"
#: sqlite busy wait. Rows are tiny and written once per creating bash call;
#: a held lock past this is a stuck writer, and the guard degrades to "no
#: provenance" (the old ranking) rather than stalling a turn.
_DB_TIMEOUT_S = 2.0
#: Busy wait for the foreign-ledger lookup (``ForeignCreations``), which runs
#: on every LOW delete classify. Readers only wait on an EXCLUSIVE writer
#: (rollback journal: RESERVED/PENDING never block a reader's shared lock
#: until commit); a commit of these tiny rows takes milliseconds, so a lock
#: still held after this is not worth stalling the turn for — the lookup
#: gives up and the rank stays what it was before the rule existed (m5,
#: review round 2 of PR #3835: the 2 s wait measured 3.86 s per classify).
#: This is the busy wait handed to SQLite, not the lookup's wall time: SQLite's
#: own lock-reporting latency dominates under an EXCLUSIVE lock (QA measured
#: the lookup at 0.74-0.79 s; a bare stdlib ``connect(timeout=0.25)`` took
#: ~1.3 s to report "locked" on a loaded box). Still bounded, and well under
#: the 2 s path.
_DB_READ_TIMEOUT_S = 0.25
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
    #: How the asking session relates to ``creator`` (set by ``owned_record``,
    #: never stored): ``""`` its own or a descendant's creation, ``"ancestor"``
    #: a creation of a session that launched it, ``"kin"`` a sibling's or
    #: cousin's in the same subagent tree (``git worktree remove`` only).
    relation: str = ""

    def describe(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))
        if self.relation == "ancestor":
            # "ancestor", not "parent" (QA Q2, PR #3964): the relation covers
            # every session above this one, and ``owned_record`` does not
            # carry the depth, so a grandchild must not be told "parent".
            return f"created by ancestor session {self.creator} of this one ({self.kind}, {when})"
        if self.relation == "kin":
            return (
                f"created by sibling session {self.creator} in this subagent tree "
                f"({self.kind}, {when})"
            )
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
            f"ledger: created by {self.creator} ({_RELATION_WORDS[self.relation]}), "
            f"kind {self.kind}, at {when}, via <command>{command}</command>"
        )


#: Any closing-tag opener, however cased or spaced: ``</``, ``< /``.
_CLOSE_TAG_RE = re.compile(r"<\s*/")


#: ``CreationRecord.ledger_fact``'s wording for each ``relation``.
_RELATION_WORDS = {
    "": "this session or its subagent",
    "ancestor": "an ancestor session of this one",
    "kin": "a sibling session in this subagent tree",
}


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS created ("
    " owner TEXT NOT NULL, path TEXT NOT NULL, dev INTEGER NOT NULL, ino INTEGER NOT NULL,"
    " created_at REAL NOT NULL, creator TEXT NOT NULL, kind TEXT NOT NULL,"
    " command TEXT NOT NULL, stamp INTEGER, PRIMARY KEY (owner, path))"
)


#: ``ForeignCreations`` looks rows up by path (``_foreign_query``); the
#: primary key leads with ``owner`` and cannot serve that. ``IF NOT EXISTS``
#: adds it to ledgers written before it existed, on their next WRITE open
#: only (m5): creating it is a write, and a read-only ledger (or one a writer
#: holds) must keep serving lookups — unindexed until the next recording
#: call, which SQLite answers with a table scan, never an error.
_PATH_INDEX = "CREATE INDEX IF NOT EXISTS created_path ON created (path)"


def _connect(
    path: Path, *, create: bool, timeout: float = _DB_TIMEOUT_S
) -> sqlite3.Connection | None:
    if not create and not path.exists():
        return None
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        conn.execute(_SCHEMA)
        if create:
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
    path: str,
    session_id: str,
    *,
    state_root: str | os.PathLike[str] | None = None,
    kin: bool = False,
    family: bool = True,
) -> CreationRecord | None:
    """The ledger row covering ``path`` for ``session_id``'s family, or ``None``.

    Covers = the recorded directory itself or anything strictly inside it,
    compared on REALPATH so a symlink inside an owned directory that points
    elsewhere is not covered. The recorded directory must still be the same
    inode generation (dev, ino and ``creation_stamp``): a path deleted and
    re-made by someone else is no longer this session's. Any error means "not owned".

    FAMILY (operator request 2026-11-14, field case: cleanup subagent
    ``bbb5e6c44eb8`` of ``2183114d32b1`` refused on its own workstream's
    worktrees). One subagent tree is one owner's work, so a row counts when:

    - ``owner`` is this session — its own creation, or a DESCENDANT's (a
      child writes its rows under every ancestor id, ``commit_creations``);
      a descendant proved by lineage (``same_task_tree``) counts too, for a
      row written before the child knew its ancestors;
    - ``creator`` is an ANCESTOR (``subagent_ancestors``: harness-written
      ``parent`` links only) — the parent that delegated the cleanup;
    - only with ``kin=True``: ``creator`` is a sibling or cousin under the
      same tree root. The caller grants that for ``git worktree remove``
      without ``--force`` ONLY (git refuses a dirty tree and keeps the
      branch, so that removal loses nothing); see ``classify._rank``.

    Lineage comes from the runtime records alone (``lineage.jsonl`` and
    ``origin.json``, both CRITICAL to write from a session), never from
    message text or the environment. An unrelated session stays foreign.
    ``relation`` on the returned record says which rule matched.

    ``family=False`` is the pre-family rule, unchanged: only rows whose
    ``owner`` is ``session_id`` (its own creations and those its descendants
    filed under it). ``check.escalate_asked_targets`` needs exactly that
    (QA Q1, PR #3964): an ancestor's row must not lift the refusal of an
    unanswered ask naming the path, which binds the whole family (#3966).
    """
    if not session_id or not path:
        return None
    try:
        real = os.path.realpath(path)
        prefixes = _covering_paths(real)
        # Family lookup is path-bounded, unlike the caller-only ledger scan on
        # main. Include each covering directory's .git companion so a root
        # ctime change keeps #3987's generation proof available on Linux.
        prefixes += [os.path.join(prefix, ".git") for prefix in prefixes]
        conn = _connect(ledger_path(state_root), create=False)
        if conn is None:
            return None
        try:
            rows = conn.execute(
                "SELECT path, dev, ino, created_at, creator, kind, command, stamp, owner"
                " FROM created WHERE path IN (" + ",".join("?" * len(prefixes)) + ")",
                prefixes,
            ).fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        logger.debug("guardrail ledger unreadable", exc_info=True)
        return None
    relation_of = (
        _family(session_id, state_root=state_root, kin=kin)
        if family
        else (lambda owner, _creator: "" if owner == session_id else None)
    )
    best: CreationRecord | None = None
    rank = {"": 0, "ancestor": 1, "kin": 2}
    # A family row may grant ownership, but only its own owner's .git witness
    # may vouch for its generation. A relative's companion must never revive
    # another owner's stale directory row.
    gitfiles_by_owner: dict[str, dict[str, CreationRecord]] = {}
    for row in rows:
        if relation_of(row[8], row[4]) is not None and row[5] == WORKTREE_GITFILE_KIND:
            gitfiles_by_owner.setdefault(row[8], {})[row[0]] = CreationRecord(*row[:8])
    for row in rows:
        relation = relation_of(row[8], row[4])
        if relation is None or row[5] == WORKTREE_GITFILE_KIND:
            continue
        rec = CreationRecord(*row[:8], relation=relation)
        gitfiles = gitfiles_by_owner.get(row[8], {})
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
        if (rec.stamp is None or creation_stamp(st) != rec.stamp) and not (
            _gitfile_generation_holds(rec, gitfiles)
        ):
            continue  # same inode number, different generation (or unstamped)
        # The deepest row wins; at one depth the closest relation does.
        if (
            best is None
            or len(rec.path) > len(best.path)
            or (len(rec.path) == len(best.path) and rank[relation] < rank[best.relation])
        ):
            best = rec
    if best is not None and _foreign_inside(
        real, session_id, state_root=state_root, relation_of=relation_of if family else None
    ):
        return None
    return best


def _covering_paths(real: str) -> list[str]:
    """``real`` and every ancestor directory below ``/``: the rows that can cover it."""
    out = [real]
    parent = os.path.dirname(real)
    while parent and parent != "/" and parent != out[-1]:
        out.append(parent)
        parent = os.path.dirname(parent)
    return out


def _family(session_id: str, *, state_root: str | os.PathLike[str] | None, kin: bool) -> Any:
    """``relation_of(owner, creator)`` for ``owned_record``: ``""``, ``"ancestor"``,
    ``"kin"`` or ``None`` (foreign). Lineage is read once, lazily, per call."""
    ancestors: list[str] | None = None
    cache: dict[str, str] = {}

    def relation_of(owner: str, creator: str) -> str | None:
        nonlocal ancestors
        if owner == session_id or creator == session_id:
            return ""
        if ancestors is None:
            ancestors = subagent_ancestors(session_id, state_root=state_root)
        if creator in ancestors:
            return "ancestor"
        if creator not in cache:
            cache[creator] = same_task_tree(session_id, creator, state_root=state_root)
        tie = cache[creator]
        if tie == "descendant":
            return ""
        if kin and tie == "sibling":
            return "kin"
        return None

    return relation_of


@functools.lru_cache(maxsize=1)
def _path_max() -> int:
    r"""Longest path a ledger row can hold (``_foreign_query``'s M3 bound).

    ``os.pathconf`` is POSIX-only and simply absent on Windows, where this
    module still has to import (CI ``xplat-probe-windows`` caught an
    ``AttributeError: module 'os' has no attribute 'pathconf'`` at import
    time), so it is looked up, never called unguarded. Without it each
    platform keeps its own ceiling: an NTFS extended-length (``\\?\``) path
    runs to 32767 characters and a realpath'd ledger row can legitimately
    reach that, so a smaller Windows bound would silently drop ancestors;
    1024 is the conservative floor everywhere else.

    Cached and lazy on purpose: importing the module must not probe the
    filesystem or be able to fail, and the answer cannot change in a process.
    """
    pathconf = getattr(os, "pathconf", None)
    if pathconf is None:
        return 32767 if os.name == "nt" else 1024
    try:
        value = pathconf("/", "PC_PATH_MAX")
    except (OSError, ValueError):
        return 1024
    return value if value > 0 else 1024


#: Most ancestors bound per target; see ``_foreign_query``.
_MAX_ANCESTORS = 512


def _bindable(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


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

    BOUNDS (M3, review round 2). Keeping every prefix of ``real`` costs
    Σ len(prefix) = O(L²): a single 20k-component token in a command OOMs the
    classifier. No stored row can be longer than ``PATH_MAX`` (rows are
    realpaths of directories that existed), so only ancestors within it are
    bound, and the prefixes are sliced from separator offsets rather than
    built by repeated ``dirname``. Under a huge ``PATH_MAX`` the shallowest
    ``_MAX_ANCESTORS`` still cap the list; deeper ancestors (and a target too
    long to be a row) add no foreign info — the rank stays what it was
    without this rule. Paths SQLite cannot bind (a lone surrogate from a
    surrogateescape'd name, m6) are dropped the same way; their encodable
    ancestors still match.
    """
    ancestors: dict[str, None] = {}
    ranges: list[str] = []
    path_max = _path_max()
    for real in paths:
        ends = [1] if real.startswith("/") else []  # "/" itself
        ends += [i for i, ch in enumerate(real) if ch == "/" and 0 < i <= path_max]
        if len(real) <= path_max:
            ends.append(len(real))
        kept = 0
        for end in dict.fromkeys(ends):
            if kept >= _MAX_ANCESTORS:
                break
            if _bindable(node := real[:end]):
                ancestors[node] = None
                kept += 1
        prefix = real.rstrip("/") + "/"
        # Rows under the target: only when the target itself was bound (not
        # past PATH_MAX or the cap, and encodable), so beyond the bound means
        # no foreign info for the whole subtree, consistently.
        if real in ancestors:
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
      it existed). The guarantee is macOS/BSD-only, where birth time is used —
      except a ``worktree`` row, kept live by its ``.git`` companion row
      (``_gitfile_holds``) as ``owned_record`` does.
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
        #: Per-classification state of the finished-foreign-worktree route
        #: (item A of the cleanup follow-on). It lives on this object because
        #: this is the one thing ``classify`` already creates once per call
        #: and threads through every ``_rank``: ``finished`` memoises
        #: :func:`finished_foreign_worktree` (no duplicate git/forge lookups for
        #: a tree reached twice), ``cleared`` holds roots it proved safe, whose
        #: own ledger row must not re-raise the objection the predicate just
        #: settled, and ``notices`` are the warnings the LOW result carries.
        #: Never reused across calls: an owner can start between two calls.
        self.finished: dict[str, FinishedForeign] = {}
        self.cleared: set[str] = set()
        self.notices: list[str] = []

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self) -> sqlite3.Connection | None:
        """The cached connection; ``None`` with no ledger or an unreadable one."""
        if not self._tried:
            self._tried = True
            try:
                self._conn = _connect(
                    ledger_path(self.state_root), create=False, timeout=_DB_READ_TIMEOUT_S
                )
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

    def _gitfile_holds(self, conn: sqlite3.Connection, rec: CreationRecord, owner: str) -> bool:
        """A ``worktree`` row whose dir stamp moved is still live via its ``.git`` row.

        WHY (rebase of PR #3987 onto #3835): ``owned_record`` already accepts
        the ``WORKTREE_GITFILE_KIND`` companion as the generation witness on
        Linux, where the directory's ctime stamp moves on every root-level
        change. Without the same rule here the nearest-owner walk dropped a
        session's OWN worktree row after its first checkout, so a broader
        foreign ancestor row became "nearest" and the session's own tree read
        HIGH; and another session's worktree row failed open.

        The companion must belong to the SAME ``owner`` as ``rec`` (R-C1, as
        ``owned_record`` scopes it with ``WHERE owner = ?``). Ledger rows are
        never pruned, and ext4/xfs reuse a freed inode number, so a dead
        worktree row of session A can share dev+ino with session B's new
        worktree at the same path. Accepting B's ``.git`` row as A's witness
        revived A's dead row, put A among the owners of B's live tree, and
        dropped ``rm -rf <B's wt>/sub`` from HIGH to an ordinary path.
        Owner-scoped, a companion only vouches for its own owner's row, so
        it can keep a genuinely live row live but never adds a dead owner
        to a live path. Companions are written under the same owners as
        their worktree row (``commit_creations``), so no real hold is lost.
        The companion is read by exact path because ``_foreign_query`` binds
        only ancestors of and rows under the target, never ``<wt>/.git``
        when the target is inside ``<wt>``.
        """
        if rec.kind != "worktree":
            return False
        gitfile = os.path.join(rec.path, ".git")
        try:
            companions = conn.execute(
                "SELECT path, dev, ino, created_at, creator, kind, command, stamp"
                " FROM created WHERE path = ? AND kind = ? AND owner = ?",
                (gitfile, WORKTREE_GITFILE_KIND, owner),
            ).fetchall()
        except (sqlite3.Error, ValueError):
            return False
        if not companions:
            # No companion (a row written before it existed): the ``.git``
            # file's age is the witness (``_gitfile_predates_row``).
            return _gitfile_generation_holds(rec, {})
        return any(
            _gitfile_generation_holds(rec, {gitfile: CreationRecord(*row)}) for row in companions
        )

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
        live = self._live_rows(reals)
        if live is None:
            return None

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

    def foreign_under(self, root: str, *, strict: bool = False) -> ForeignHit | None:
        """A foreign live row strictly UNDER ``root`` whose creator did not create ``root``.

        WHY (review R2 on PR #3992). A finished foreign worktree is cleared
        by :func:`finished_foreign_worktree`, which proves the TREE's work is
        merged and clean; it says nothing about a directory a THIRD session
        recorded inside it (an empty ``mkdir`` passes ``clean``). Such a row
        keeps the route from applying. Rows by the tree's own creator are
        that creator's finished work, already covered by the proof, so they
        do not object. Unreadable ledger: ``None`` (as :meth:`match`).
        """
        real = os.path.realpath(root).rstrip("/") or "/"
        live = self._live_rows([real])
        if live is None:
            if strict:
                raise OSError("worktree descendant attribution unavailable")
            return None
        creators = live[real][0] if real in live else set()
        prefix = real + "/"
        for p in sorted(p for p in live if p.startswith(prefix)):
            owners, rec = live[p]
            if owners & creators or any(self._related(o) for o in owners):
                continue
            return ForeignHit(rec.creator, rec, "holds")
        return None

    def _live_rows(self, reals: list[str]) -> dict[str, tuple[set[str], CreationRecord]] | None:
        """path -> (owners, row) for the LIVE generation of rows related to ``reals``.

        Rows of a root in :attr:`cleared` are kept out; ``None`` when the
        ledger is unreadable.
        """
        conn = self._connection()
        if not reals or conn is None:
            return None
        try:
            sql, params = _foreign_query(reals)
            rows = conn.execute(sql, params).fetchall()
        # ValueError covers UnicodeEncodeError: a lone surrogate SQLite cannot
        # bind (m6). ``_foreign_query`` already drops such parameters; this
        # keeps any other unbindable input at "no foreign info" (the rank
        # main gives), never a guardrail-error hard block.
        except (sqlite3.Error, ValueError):
            logger.debug("guardrail ledger unreadable", exc_info=True)
            return None
        # path -> (owners, row) over rows of the LIVE generation only.
        live: dict[str, tuple[set[str], CreationRecord]] = {}
        for owner, *fields in rows:
            rec = CreationRecord(*fields)
            identity = self._live(rec.path)
            # Keep upstream's owner-scoped gitfile witness in the shared loader:
            # root ctime changes must not erase a live foreign ownership hold.
            if identity != (rec.dev, rec.ino, rec.stamp) and not (
                identity is not None
                and identity[:2] == (rec.dev, rec.ino)
                and self._gitfile_holds(conn, rec, owner)
            ):
                continue
            if rec.path in self.cleared:
                # Exactly a worktree root ``finished_foreign_worktree`` proved
                # safe in this classification: its creator's row is the very
                # objection that predicate settles. Rows ABOVE it are still
                # matched here; a third session's row INSIDE it already barred
                # the route (``foreign_under``, checked before clearing), so
                # other foreign data stays protected.
                continue
            live.setdefault(rec.path, (set(), rec))[0].add(owner)
        return live


#: Ledger kind of the companion row ``commit_creations`` writes for a
#: recorded worktree's ``.git`` FILE. WHY: on Linux ``creation_stamp`` is the
#: directory's ``st_ctime_ns`` (no birth time), which moves whenever an entry
#: is added, removed or renamed in the worktree ROOT — and git rewrites root
#: files by unlink+create on checkout/read-tree -u/merge. So on a Linux host
#: (every headless mesh node, e.g. lop-remote-node-2, Amazon Linux) the
#: worktree row stopped matching after the session's first root-level change
#: and its own tree read as foreign (operator directive 2026-11-14, field case
#: attr3880-ci). The ``.git`` file git writes once at ``worktree add`` is not
#: rewritten by any of those (only ``worktree repair``/``move``, which then
#: fail closed), so its inode generation is a stable second witness that the
#: directory is the one this session created. Never a covering row itself.
WORKTREE_GITFILE_KIND = "worktree-gitfile"


def _gitfile_generation_holds(rec: CreationRecord, gitfiles: Mapping[str, CreationRecord]) -> bool:
    """A ``worktree`` row whose dir stamp moved still matches via its ``.git`` file.

    Only ``kind == "worktree"`` (the directory's dev+ino already matched in
    the caller). With a companion row it must match dev, ino AND
    ``creation_stamp``; a companion that does not match is final. Without one
    — the row predates the companion, or predates the ``stamp`` column
    itself — :func:`_gitfile_predates_row` decides.
    """
    if rec.kind != "worktree":
        return False
    gitfile = os.path.join(rec.path, ".git")
    companion = gitfiles.get(gitfile)
    try:
        st = os.lstat(gitfile)
    except OSError:
        return False
    if not stat.S_ISREG(st.st_mode):
        return False
    if companion is not None:
        return companion.stamp is not None and (
            st.st_dev,
            st.st_ino,
            creation_stamp(st),
        ) == (companion.dev, companion.ino, companion.stamp)
    return _gitfile_predates_row(rec, gitfile, st)


#: How much of a ``.git`` file is read to confirm it is a gitdir pointer.
_GITFILE_READ_MAX = 4096


def _gitfile_predates_row(rec: CreationRecord, gitfile: str, st: os.stat_result) -> bool:
    """The worktree's ``.git`` file already existed when the row was written.

    WHY (operator directive 2026-11-14, node-2 field case 1a, session
    99ffc68d3d05, ``.worktrees/github-default-repos``). The row was NOT lost:
    the node's ledger still held it (same owner, path, dev and ino), but on
    Linux its ``stamp`` is the directory's ctime, which had moved from
    1794449817716387514 to 1794683305395946140 with the session's own
    checkouts — and the row was written by a runtime older than the
    ``WORKTREE_GITFILE_KIND`` companion, so nothing could re-match it after a
    restart or model switch. Every later delete read the session's own clean
    tree as unrecorded. The same holds for rows written before the ``stamp``
    column existed (field case 3, ``radient-site/.worktrees/copy-overhaul``).

    The witness: ``creation_stamp`` exists to tell a directory from a LATER
    one that reused its inode number. Any later ``git worktree add`` at that
    path writes a NEW ``.git`` file, whose ctime is then later than the row.
    A ``.git`` file whose ctime is at or before the row's ``created_at``
    therefore existed when the session recorded it, so the directory holding
    it is the recorded generation. ctime cannot be set by an unprivileged
    process (only a clock change moves it), and a rename, hard link, chmod or
    rewrite of the file bumps it — each then fails closed to "not recorded".
    The row is written AFTER the creating command returns, so a genuine
    worktree's ``.git`` (made during that command) always predates it
    (field: 1794449816.24 s vs row 1794449818.40 s).

    CONSTRAINTS: same filesystem as the recorded directory; a regular file
    whose content is a ``gitdir:`` pointer (a linked worktree, never a clone,
    whose ``.git`` directory's ctime moves on every commit). Residual risk:
    a system clock stepped backwards after a recreation could make a newer
    ``.git`` look older; that needs root, which this guard does not defend
    against.
    """
    if st.st_dev != rec.dev or st.st_ctime_ns > int(rec.created_at * 1_000_000_000):
        return False
    try:
        with open(gitfile, "rb") as fh:
            head = fh.read(_GITFILE_READ_MAX)
    except OSError:
        return False
    return head.startswith(b"gitdir:")


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
    real: str,
    session_id: str,
    *,
    state_root: str | os.PathLike[str] | None = None,
    relation_of: Any = None,
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
            rows = conn.execute("SELECT path, owner, creator FROM created").fetchall()
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return True
    # "Own" = rows ``owned_record`` would accept for this session: its own,
    # and with ``relation_of`` its family's (see ``owned_record``). Lineage
    # is asked only about rows at or under the target.
    own = {p for p, o, _c in rows if o == session_id}
    if relation_of is not None:
        own |= {
            p
            for p, o, c in rows
            if p not in own
            and (p == real or p.startswith(prefix))
            and relation_of(o, c) is not None
        }
    if any((p == real or p.startswith(prefix)) and p not in own for p, _o, _c in rows):
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
            if kind == "worktree" and os.path.isfile(os.path.join(path, ".git")):
                # A second generation stamp for Linux hosts (headless mesh
                # nodes such as lop-remote-node-2 run Linux): see
                # ``WORKTREE_GITFILE_KIND``.
                entries.append(
                    (os.path.join(os.path.realpath(path), ".git"), WORKTREE_GITFILE_KIND)
                )
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


def _git_proc(
    args: Sequence[str], cwd: str, *, timeout: float = _GIT_TIMEOUT_S
) -> subprocess.CompletedProcess[str] | None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Read-only probes: no prompts, no index refresh writes, no credential helper.
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        return subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _git(args: Sequence[str], cwd: str, *, timeout: float = _GIT_TIMEOUT_S) -> str | None:
    proc = _git_proc(args, cwd, timeout=timeout)
    return proc.stdout if proc is not None and proc.returncode == 0 else None


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
        # Added 2026-11-14 (field case 3, radient-site/.worktrees/copy-overhaul,
        # PR #32 merged): Vite's server-side build output, listed in every
        # create-vite ``.gitignore`` beside ``dist``. Its ignored ``dist-ssr/``
        # alone made that merged, change-free tree read ``dirty``.
        "dist-ssr",
        # Added 2026-11-23 (PR #4493, CI lane's own native-scope worktree):
        # the Windows counterparts of ``.DS_Store``, folder metadata Explorer
        # rewrites on its own (thumbnail cache, per-folder view settings).
        "Thumbs.db",
        "desktop.ini",
        # Added 2026-11-23 (PR #4493, lane QA worktrees qa2069-head/base in
        # local-operator-ui): electron-vite's build output directory, the
        # same standing as ``build``/``dist`` (only ever matched when Git
        # itself reports the entry IGNORED).
        "out",
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


#: OS folder-metadata FILE names on :data:`_REGENERABLE` (plus ``._*``). Under
#: these names only a regular file is the OS's (#4503): a directory or special
#: file named ``.DS_Store`` is someone's data and stays held.
_OS_METADATA_FILES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def _own_tree_sheds(physical: str) -> bool:
    """May removal of a ledger-proven OWN tree take this ignored entry? (#4493)

    One rule for the plain and the ``--force`` routes. An ignored symlink:
    yes, Git and ``rm`` remove the link, never its target. An OS metadata
    name: only as a regular file. Any other name on :data:`_REGENERABLE`:
    yes, unless it holds a nested repository. Everything else: no. Unreadable
    is no (fail closed)."""
    name = os.path.basename(physical)
    try:
        st = os.lstat(physical)
    except OSError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    if name in _OS_METADATA_FILES or name.startswith("._"):
        return stat.S_ISREG(st.st_mode)
    return _regenerable(name) and not _holds_git(physical)


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


class CwdInventory(NamedTuple):
    """One process-cwd inventory: the cwds it could read and the pids it could not.

    ``hidden`` WHY (2026-11-14, found by QA on PR #3975): on Linux the kernel
    refuses ``readlink /proc/<pid>/cwd`` (EACCES) for a same-uid process that
    is non-dumpable — ``(sd-pam)``, the systemd user-session helper, is one on
    every host with a systemd user session, and ``ssh-agent`` makes itself
    one. That single refusal used to void the whole inventory (``None``), so
    every caller read "unknown" on such hosts. A hidden pid is now reported
    instead, and still counts as "might be inside" for non-own removal and
    interrupted-add cleanup (:func:`_live_cwd_inside`). Ledger-proven own
    cleanup uses positive fd/maps evidence instead (:func:`cwd_holders`).
    """

    cwds: tuple[str, ...]
    hidden: tuple[int, ...] = ()


#: One raw inventory row: ``(pid, realpath of its cwd)``, or ``(pid, None)``
#: for a same-uid pid whose cwd the kernel hides (see :class:`CwdInventory`).
#: Pids ride along so :func:`cwd_holders` can attribute each holder (PR
#: #3987); hidden pids ride in the same rows so one cache serves both views.
CwdEntry = tuple[int, str | None]
_cwd_inventory_cache: tuple[float, tuple[CwdEntry, ...]] | None = None
#: Why the last inventory failed, for the judge (operator directive
#: 2026-11-14: never a bare "busy"). Empty after a success.
_cwd_inventory_error = ""
#: Where :func:`_cwd_entries` reads Linux's process table. A module constant
#: only so tests can point it at a fake tree (EACCES on one pid's ``cwd``)
#: on hosts without ``/proc``; never configured at runtime.
_PROC_ROOT = "/proc"


def _cwd_entries(*, fresh: bool = False) -> tuple[CwdEntry, ...] | None:
    """``(pid, realpath of cwd)`` for every process of this uid, or ``None``.

    lsof's cwd inventory is read-only and avoids probing process command lines
    (which may carry secrets). Linux exposes the same data under /proc. A
    disappearing process no longer owns a cwd; a same-uid process whose cwd the
    kernel will not reveal (EACCES) is kept as ``(pid, None)`` rather than
    voiding the inventory (PR #3980). Any other inspection error (an
    unreadable ``/proc``, lsof missing, failing or timing out) gives ``None``
    and leaves the reason in :data:`_cwd_inventory_error` — fail closed, but
    named. A success is cached for :data:`_CWD_INVENTORY_TTL_S` unless
    ``fresh``.

    The inventory only sees processes exposed by the current procfs view.
    The liveness re-check for an observed holder (:func:`_still_exists`) does
    not treat a missing procfs entry as proof of exit.

    A per-pid ``EACCES`` on this uid's ``cwd`` link does NOT fail the scan
    (operator directive 2026-11-14, node-2 field cases 1c and 2): see
    :func:`_unreadable_cwd_evidence`. Such a pid contributes the paths its
    readable open files and mappings name (so ``_inside`` still finds it when
    those point into a tree), and is listed in :data:`_cwd_unreadable` for
    the facts. Any other per-pid error, and an unreadable ``/proc``, still
    fail closed.
    """
    global _cwd_inventory_cache, _cwd_inventory_error, _cwd_unreadable
    now = time.monotonic()
    cached = _cwd_inventory_cache
    if not fresh and cached is not None and now - cached[0] < _CWD_INVENTORY_TTL_S:
        return cached[1]
    entries: list[CwdEntry] = []
    unreadable: dict[int, tuple[str, ...]] = {}
    if os.path.isdir(os.path.join(_PROC_ROOT, "self")):
        try:
            for entry in os.scandir(_PROC_ROOT):
                if not entry.name.isdecimal():
                    continue
                try:
                    if entry.stat().st_uid != os.getuid():
                        continue
                    entries.append((int(entry.name), os.readlink(os.path.join(entry.path, "cwd"))))
                except (FileNotFoundError, ProcessLookupError):
                    continue  # exited mid-scan: it holds no cwd
                except PermissionError:
                    evidence = _unreadable_cwd_evidence(entry.path)
                    unreadable[int(entry.name)] = evidence
                    entries.append((int(entry.name), None))
                    entries.extend((int(entry.name), p) for p in evidence)
                except OSError as exc:
                    _cwd_inventory_error = (
                        f"cwd of pid {entry.name} unreadable ({exc.strerror or exc})"
                    )
                    return None
        except OSError as exc:
            _cwd_inventory_error = f"/proc unreadable ({exc.strerror or exc})"
            return None
    else:
        try:
            proc = subprocess.run(
                ["lsof", "-a", "-u", str(os.getuid()), "-d", "cwd", "-Fpn", "-w"],
                capture_output=True,
                text=True,
                timeout=_CWD_INVENTORY_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _cwd_inventory_error = f"lsof timed out after {_CWD_INVENTORY_TIMEOUT_S:g} s"
            return None
        except (OSError, subprocess.SubprocessError) as exc:
            _cwd_inventory_error = f"lsof unavailable ({exc})"
            return None
        if proc.returncode != 0:
            _cwd_inventory_error = f"lsof exited {proc.returncode}"
            return None
        pid = 0
        for line in proc.stdout.splitlines():
            if line.startswith("p") and line[1:].isdecimal():
                pid = int(line[1:])
            elif line.startswith("n/"):
                entries.append((pid, line[1:]))
        if not entries:
            # This process alone holds a cwd: an empty list is a failed read.
            _cwd_inventory_error = "lsof listed no process cwd"
            return None
    inventory = tuple((pid, None if cwd is None else os.path.realpath(cwd)) for pid, cwd in entries)
    _cwd_inventory_cache = (now, inventory)
    _cwd_inventory_error = ""
    _cwd_unreadable = {
        pid: tuple(os.path.realpath(p) for p in paths) for pid, paths in unreadable.items()
    }
    return inventory


#: ``pid -> realpaths its readable fds/maps name`` for this uid's processes
#: whose ``cwd`` link read EACCES in the last successful scan (empty tuple:
#: nothing about it was readable). Facts only: see
#: :func:`_unreadable_cwd_evidence` and :func:`cwd_scan_note`.
_cwd_unreadable: dict[int, tuple[str, ...]] = {}
#: Paths read per unreadable pid, across fds and mappings: a process with
#: thousands of fds must not stall the guard. Paths, not holders: the first
#: hit inside a tree is all ``_inside`` needs, and an Electron renderer's
#: maps list ~300 libraries.
_UNREADABLE_EVIDENCE_MAX = 2048


def _unreadable_cwd_evidence(proc_dir: str) -> tuple[str, ...]:
    """Absolute paths a cwd-unreadable process holds open or mapped, if readable.

    WHY (operator directive 2026-11-14; node-2 field data, and GitHub's Linux
    runners: "cwd of pid 1460 unreadable (Permission denied)"). Reading
    ``/proc/<pid>/cwd`` needs ``ptrace_may_access(PTRACE_MODE_READ_FSCREDS)``.
    For a process of OUR uid that fails only when the target is not an
    ordinary peer: it is non-dumpable (``prctl(PR_SET_DUMPABLE, 0)`` —
    Chromium/Electron zygotes and renderers, which on node-2 were ten
    "Local Operator" processes), set-uid/set-gid or file-capability
    privileged (``(sd-pam)``, gnome-keyring, sudo helpers), or its gid sets
    differ. Yama's ``ptrace_scope`` does not apply (it gates ATTACH, not
    READ). Failing the WHOLE scan on one such pid made every own tree on
    a Linux node read ``busy=unknown``, which disabled the own-tree LOW path
    and sent clean or scratch removals of the session's own trees to a judge
    that refused them.

    The pid therefore counts as a holder only on POSITIVE evidence: an open
    file (``fd/*``) or mapping (``maps``) whose path lies inside the tree.
    Without readable evidence it is excluded and named in the facts
    (:func:`cwd_scan_note`).

    RESIDUAL RISK, accepted: ``fd/`` and ``maps`` sit behind the SAME ptrace
    check, so for most such pids they are unreadable too and the pid is
    simply invisible. A privileged or non-dumpable process whose cwd is
    inside an agent worktree, with nothing open there, would not hold it.
    Such processes are system daemons, PAM helpers and browser sandboxes,
    not agent work; and the removal costs that process its cwd, never a file
    it holds open (those would be evidence). Agent shells, eval kernels and
    git are dumpable peers and are always seen.
    """
    found: list[str] = []
    try:
        fds = os.scandir(os.path.join(proc_dir, "fd"))
    except OSError:
        fds = None
    if fds is not None:
        with fds:
            for fd in fds:
                if len(found) >= _UNREADABLE_EVIDENCE_MAX:
                    break
                try:
                    target = os.readlink(fd.path)
                except OSError:
                    continue
                if target.startswith("/"):
                    found.append(target.removesuffix(" (deleted)"))
    try:
        with open(os.path.join(proc_dir, "maps"), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if len(found) >= _UNREADABLE_EVIDENCE_MAX:
                    break
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[5].startswith("/"):
                    found.append(parts[5].rstrip("\n").removesuffix(" (deleted)"))
    except OSError:
        pass
    return tuple(dict.fromkeys(found))


def cwd_scan_note(path: str) -> str:
    """The facts line for cwd-unreadable processes NOT counted against ``path``.

    Empty when the last scan read every cwd. Pids whose readable evidence
    points inside ``path`` are holders (named by :func:`busy_phrase`), so
    they are left out here.
    """
    real = os.path.realpath(path)
    ignored = sorted(
        pid for pid, paths in _cwd_unreadable.items() if not any(_inside(p, real) for p in paths)
    )
    if not ignored:
        return ""
    shown = ", ".join(str(p) for p in ignored[:6]) + (" …" if len(ignored) > 6 else "")
    noun = "process" if len(ignored) == 1 else "processes"
    return (
        f"{len(ignored)} same-user {noun} had an unreadable cwd (pid {shown}); "
        "ignored: no evidence they hold the tree"
    )


def _cwd_inventory() -> CwdInventory | None:
    """Readable cwd/evidence paths and hidden pids; non-own callers stay conservative."""
    entries = _cwd_entries()
    if entries is None:
        return None
    return CwdInventory(
        tuple(cwd for _pid, cwd in entries if cwd is not None),
        tuple(pid for pid, cwd in entries if cwd is None),
    )


def _inside(cwd: str | None, real: str) -> bool:
    """A readable cwd at or under ``real``; a hidden cwd (``None``) is never a definite yes."""
    return cwd is not None and (cwd == real or cwd.startswith(real + os.sep))


def _live_cwd_inside(real: str, *, hidden_blocks: bool = False) -> bool | None:
    """Whether a live process uses this tree; unknown never grants an exemption.

    Reads :func:`_cwd_entries` (bounded, cached per process). ``None`` —
    timeout or an unreadable inventory — fails closed in ``worktree_state``.
    This protects ordinary clean removals as well as abandoned adds.

    A HIT from the cache is re-read fresh before it is believed (operator
    directive 2026-11-14): a holder that exited inside the 30 s TTL — an eval
    kernel restarted, a ``git`` that finished — must not keep the tree "busy".
    A cached miss is still reused (the TTL's documented price).

    A hidden pid (cwd unreadable, EACCES) counts only on POSITIVE evidence:
    its readable fds/maps are already merged into the entries by
    :func:`_cwd_entries`, so one naming a path inside answers ``True`` above.
    Without such evidence it no longer answers ``None`` (field report
    2026-11-17, node-2 session d5188ee174d4, guard 0.106.6): ``(sd-pam)`` —
    same uid, non-dumpable, up for days — made EVERY tree on the host read
    ``busy=unknown``, so clean, merged worktrees removed with a plain ``git
    worktree remove`` (the guide's finished-work rule, whoever created them)
    were refused. #4024 already applied this rule to own trees; the residual
    risk is the one documented on :func:`_unreadable_cwd_evidence`. The
    pid's name is never consulted: a same-uid process picks its own comm.

    ``hidden_blocks`` keeps the pre-#4024 refusal for a caller that DELETES
    with no judge in the loop (``interrupted_worktree_add``'s settle): there
    a hidden pid still answers ``None``. Every caller treats ``None`` like
    ``True`` (``busy`` / never delete); an unreadable ``/proc`` or a failed
    scan is ``None`` for every caller.
    """
    cached = _cwd_inventory_cache
    from_cache = cached is not None and time.monotonic() - cached[0] < _CWD_INVENTORY_TTL_S
    entries = _cwd_entries()
    if entries is None:
        return None
    if from_cache and any(_inside(cwd, real) for _pid, cwd in entries):
        entries = _cwd_entries(fresh=True)
        if entries is None:
            return None
    if any(_inside(cwd, real) for _pid, cwd in entries):
        return True
    if hidden_blocks and any(cwd is None for _pid, cwd in entries):
        return None
    return False


#: Ancestry walk bound for :func:`cwd_holders` (far deeper than any real tree).
_PARENT_WALK_MAX = 64
#: Shown command length: a basename, never arguments (they may carry secrets).
_HOLDER_COMMAND_MAX = 40


@dataclass(frozen=True)
class CwdHolder:
    """One live process whose cwd is inside a worktree, and whose it is.

    ``family``: the process descends (by parent pid) from the deleting
    session's own harness process or a registry record of that session or of
    an ancestor/descendant in its subagent tree — except a bash child or eval
    kernel of this harness, which is attributed to the session that spawned
    it (:mod:`.spawned`; in-process siblings share the pid). ``owner`` is the judge-facing
    attribution ("this session", "session <id> (another session)", "no
    session record").
    """

    pid: int
    command: str
    owner: str
    family: bool

    def describe(self) -> str:
        return f"pid {self.pid} {self.command or '?'} ({self.owner})"


def _process_table() -> dict[int, tuple[int, str]] | None:
    """``pid -> (parent pid, command basename)`` for every visible process.

    Linux: ``/proc/<pid>/stat`` (parent) and ``/proc/<pid>/comm`` (the kernel's
    16-byte name — never ``cmdline``, which may carry secrets). Elsewhere:
    ``ps -axo pid=,ppid=,comm=`` (executable path, no arguments). ``None`` when
    neither is readable: ownership is then unknown and the tree stays busy.
    """
    table: dict[int, tuple[int, str]] = {}
    if os.path.isdir("/proc/self"):
        try:
            names = [e.name for e in os.scandir("/proc") if e.name.isdecimal()]
        except OSError:
            return None
        for name in names:
            try:
                with open(f"/proc/{name}/stat", encoding="utf-8", errors="replace") as fh:
                    raw = fh.read()
            except OSError:
                continue  # exited, or another uid's under hidepid: not a holder here
            # ``comm`` sits in parentheses and may itself hold ")": split on the LAST.
            head, _, tail = raw.rpartition(")")
            fields = tail.split()
            if len(fields) < 2 or not fields[1].isdecimal():
                continue
            comm = head.partition("(")[2]
            table[int(name)] = (int(fields[1]), comm)
        return table or None
    try:
        proc = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,comm="],
            capture_output=True,
            text=True,
            timeout=_CWD_INVENTORY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdecimal() and parts[1].isdecimal():
            comm = os.path.basename(parts[2].strip()) if len(parts) == 3 else ""
            table[int(parts[0])] = (int(parts[1]), comm)
    return table or None


def _session_pids(session_id: str) -> tuple[dict[int, str], dict[int, str]]:
    """``(family, others)``: harness pid -> label, from the session registry.

    Family = the caller's own record, records of its ancestors/descendants in
    one subagent tree (``same_task_tree``), and THIS process — the guard runs
    inside the deleting session's harness — unless a record says this pid is
    another session's. Siblings in separate processes count as others. THIS
    process may also host sibling subagents (in-process ``Session``s share the
    harness pid; review m1, PR #3987), so its family label covers only what no
    session spawned: :func:`cwd_holders` attributes the harness's bash
    children and eval kernels by the session that spawned them
    (:mod:`.spawned`), not by this pid. Records, never text: a process's
    environment or command line is not consulted.
    """
    family: dict[int, str] = {}
    others: dict[int, str] = {}
    try:
        from local_operator.session.runtime import registry

        for rec, state in registry.scan(None, reap=False):
            sid = str(getattr(rec, "session_id", "") or "")
            pid = getattr(rec, "pid", None)
            if not sid or not isinstance(pid, int) or state == "stale":
                continue
            if sid == session_id:
                family[pid] = "this session"
                continue
            relation = same_task_tree(session_id, sid) if session_id else ""
            if relation in ("ancestor", "descendant"):
                family[pid] = f"this session's {relation} {sid}"
            else:
                others[pid] = f"session {sid}, another session"
    except Exception:  # noqa: BLE001 — an unreadable registry attributes nothing
        logger.debug("session registry unreadable", exc_info=True)
    me = os.getpid()
    if me not in others:
        family.setdefault(me, "this session")
    return family, others


def cwd_holders(
    real: str, session_id: str, *, hidden_blocks: bool = False
) -> tuple[CwdHolder, ...] | None:
    """Every live process with its cwd inside ``real``, attributed; ``None`` if unknown.

    WHY (operator directive 2026-11-14, field case ``attr3880-ci`` on
    lop-remote-node-2): a session's own eval kernel or bash child with its cwd
    in its own tree made ``git worktree remove --force`` read "busy", and the
    judge refused the creator's cleanup. Attribution is by parent-pid ancestry
    to the runtime's own records (:func:`_session_pids`). A process whose
    ancestry reaches no family pid — another session's, or an orphan
    reparented to init — is foreign. ``None`` (inventory or process table
    unreadable) means ownership is unknown: the caller keeps the tree busy.
    A cwd-unreadable pid is a holder only when its readable fd/maps evidence
    lies inside the tree (merged into the entries by :func:`_cwd_entries`);
    otherwise it is left to :func:`cwd_scan_note` for every tree, own or
    not (field report 2026-11-17, see :func:`_live_cwd_inside`). Until then
    only ledger-proven own trees had that rule, so ``(sd-pam)`` alone voided
    the scan for the finished-work path and its phrase read ``busy=unknown``.

    ``hidden_blocks`` restores the pre-relaxation rule (review M1, PR #4161)
    for a caller whose LOW answer is an ``rm -rf`` (or a forced removal) with
    no judge in the loop, e.g. :func:`finished_rig`: such a delete has none of
    ``git worktree remove``'s own refusal of untracked content, so a hidden
    pid with no evidence answers ``None`` (unknown, with the pids recorded in
    the inventory error), and next to a definite holder it is listed as one
    that "may hold the tree". Only the plain ``git worktree remove`` route
    may leave it ``False``.
    """
    global _cwd_inventory_error
    entries = _cwd_entries(fresh=True)
    if entries is None:
        return None
    hidden = [pid for pid, cwd in entries if cwd is None] if hidden_blocks else []
    inside = [(pid, cwd) for pid, cwd in entries if _inside(cwd, real)]
    # Definite holders must be attributed before unrelated hidden pids are
    # reported (#3987 m1); otherwise the reason conceals the actual blocker.
    if hidden and not inside:
        _cwd_inventory_error = "cwd of pid {} unreadable (Permission denied)".format(
            ", ".join(map(str, hidden[:5])) + (", ..." if len(hidden) > 5 else "")
        )
        return None
    if not inside:
        return ()
    table = _process_table()
    if table is None:
        return None
    family, others = _session_pids(session_id)
    # Read AFTER the process table (QA Q3): ``spawning`` is entered before
    # the fork and left after ``record``, so a child the table shows is
    # either recorded already or still counted here.
    in_flight = spawned.pending() > 0
    me = os.getpid()
    holders: list[CwdHolder] = []
    for pid, _cwd in inside:
        owner, is_family = "no session record; unknown process", False
        node, below, seen = pid, 0, 0
        while node > 1 and seen < _PARENT_WALK_MAX:
            spawner = spawned.session_of(below) if node == me and below else None
            if spawner is not None:
                # A bash child or eval kernel of THIS harness: in-process
                # subagents share its pid, so the spawning session decides
                # (review m1, PR #3987) — a sibling's shell is foreign.
                owner, is_family = _spawner_owner(session_id, spawner)
                break
            if node == me and below and in_flight:
                # An unrecorded direct child while an agent spawn is between
                # fork and ``record`` (QA Q3, PR #3987): it may be a
                # sibling's brand-new shell, so it is not this session's.
                owner = "harness child spawned while an agent spawn was unrecorded"
                break
            if node in family:
                owner, is_family = family[node], True
                break
            if node in others:
                owner = others[node]
                break
            below, node = node, table.get(node, (0, ""))[0]
            seen += 1
        command = table.get(pid, (0, ""))[1][:_HOLDER_COMMAND_MAX]
        holders.append(CwdHolder(pid, command, owner, is_family))
    if hidden:
        definite_pids = {holder.pid for holder in holders}
        holders.extend(
            CwdHolder(pid, "?", "cwd unreadable; may hold the tree", False)
            for pid in hidden
            if pid not in definite_pids
        )
    # A non-family holder that has EXITED by now held nothing worth refusing
    # for: re-check each one's existence before it can make the tree busy.
    return tuple(h for h in holders if h.family or _still_exists(h.pid))


def _still_exists(pid: int) -> bool:
    """Whether ``pid`` names a process right now (``False`` only for ESRCH).

    WHY (main Full Suite runs 43718276774 job 132827789967 and 43737923202
    job 132897283649, ``test_a_field_shape_with_own_process_inside_is_low[rm
    -rf …]``, Linux 3.13 shards): ``_reached_worktrees`` runs an own tree's
    ``worktree_state`` and ``worktree_activity`` concurrently, and activity
    spawns ``git -C <tree> …`` (and the forge CLI) whose cwd is INSIDE the
    tree for a few milliseconds. :func:`cwd_holders` reads the cwd inventory
    first and the process table second; a git child caught by the first and
    gone by the second has no table row, so its ancestry walk stops at pid 0
    and it reads "no session record; unknown process", family False — the
    creator's clean own tree read busy and ``rm -rf`` of it went HIGH. The
    displayed holder list (a second scan) then showed only the family
    ``sleep``, because the git child had already exited. Reproduced on
    Darwin with a lingering ``git symbolic-ref`` shim: 10/10 decision-time
    holder sets carried two such pids, both dead when re-checked.

    Constraint, fail closed: only a process that provably no longer exists
    (``ProcessLookupError``) is dropped. One that still exists keeps its
    unknown/foreign attribution — an orphan or another session's child stays
    busy and named. ``EPERM`` (another uid) means it exists. A pid recycled
    between the scan and this check reads as existing and keeps the tree
    busy: the safe direction, so no start-time comparison is needed.

    A visible procfs entry is positive evidence. A missing entry can reflect
    restricted visibility rather than exit, so confirm it with ``kill(pid, 0)``
    just as on systems without procfs.
    """
    if os.path.isdir(os.path.join(_PROC_ROOT, "self")) and os.path.lexists(
        os.path.join(_PROC_ROOT, str(pid))
    ):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True  # EPERM and anything else: it may exist, so it may hold the tree
    return True


def _spawner_owner(session_id: str, spawner: str) -> tuple[str, bool]:
    """``(owner label, family?)`` for a harness child spawned by session ``spawner``."""
    if session_id and spawner == session_id:
        return "this session", True
    relation = same_task_tree(session_id, spawner) if session_id else ""
    if relation in ("ancestor", "descendant"):
        return f"this session's {relation} {spawner}", True
    return f"session {spawner}, another session in this harness process", False


def busy_phrase(path: str, session_id: str = "", *, hidden_blocks: bool = False) -> str:
    """The judge-facing account of WHY a tree reads busy: holders, or why unknown.

    Never a bare "busy" (operator directive 2026-11-14): the field refusal
    named "busy" with no process, so nobody could tell an agent's own shell
    from another session's work.
    """
    real = os.path.realpath(path)
    # The same rule as the ``worktree_state`` call that read ``busy``, or a
    # hidden pid that made it busy would be dropped from its own explanation.
    holders = cwd_holders(real, session_id, hidden_blocks=hidden_blocks)
    if holders is None:
        reason = _cwd_inventory_error or "process table unreadable"
        return (
            f"busy=unknown: process inventory unreadable ({reason}); "
            "a live process inside cannot be ruled out"
        )
    if not holders:
        return "busy: the holder exited before it could be named"
    return "busy: cwd held by " + ", ".join(h.describe() for h in holders)


def state_phrase(
    path: str, state: str, session_id: str = "", *, hidden_blocks: bool = False
) -> str:
    """``state``, with the holders spelled out when it is ``busy``."""
    return busy_phrase(path, session_id, hidden_blocks=hidden_blocks) if state == "busy" else state


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


def worktree_state(
    path: str,
    *,
    own: bool = False,
    session_id: str = "",
    facts: dict[str, bool] | None = None,
    hidden_blocks: bool = False,
    interrupted_cleanup: bool = False,
) -> str:
    """The removal-relevant state of a worktree path.

    ``interrupted_cleanup`` is the stricter payload inventory for an exact
    removal with separate commit-preservation proof. It accepts deletions of
    tracked files and independently verified disposable links, not ignored file data.

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
    cwd is inside it, or the inventory could not be read — with ``own`` and
    ``session_id``, holders from the deleting session's own family do not
    count; :func:`busy_phrase` names the holders), ``not-linked`` or
    ``unknown``. ``hidden_blocks`` (review M1, PR #4161): a NOT-own tree also
    reads ``busy`` past a cwd-unreadable pid with no evidence, for callers
    whose LOW answer is an ``rm -rf`` or ``--force`` with no judge
    (:func:`finished_foreign_worktree`, a reached tree); only the plain ``git
    worktree remove`` route leaves it off. An own tree keeps #4024's rule.

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
    if interrupted_cleanup:
        top = _git(["rev-parse", "--show-toplevel"], real)
        if not top or top.rstrip("\n") != real:
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
    except (OSError, UnicodeError):
        return "unknown" if interrupted_cleanup else "not-linked"
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
    live: bool | None
    if own and session_id:
        # OWN tree: ALWAYS a fresh inventory (QA Q1, PR #3987). This answer
        # is what lets an own tree rank LOW (clean, or detached scratch), and
        # the 30 s cache's "nobody inside" is no proof: a foreign process that
        # entered after the cached scan was missed, the own dirty detached
        # tree ranked LOW with no judge call, and ``git worktree remove
        # --force`` deleted it under that process. Holders from this
        # session's own family (operator directive 2026-11-14, field case
        # ``attr3880-ci``: the creator's eval kernel, bash child or subagent)
        # do not make it busy; any foreign holder, or an inventory or process
        # table that cannot be read (``None``), keeps it busy — fail closed.
        holders = (
            cwd_holders(real, session_id, hidden_blocks=hidden_blocks)
            if interrupted_cleanup
            else cwd_holders(real, session_id)
        )
        live = None if holders is None else any(not h.family for h in holders)
    else:
        if interrupted_cleanup:
            global _cwd_inventory_cache
            _cwd_inventory_cache = None
        live = _live_cwd_inside(real, hidden_blocks=hidden_blocks)
    if live is not False:
        return "busy"  # unknown process inventory also fails closed
    if interrupted_cleanup:
        if stale_initializing:
            return "locked"
        return _interrupted_cleanup_state(real, facts, own_regenerable=own and bool(session_id))
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
    lines = ignored.splitlines()
    if facts is not None and any(ln.startswith(("D  ", " D ")) for ln in lines):
        facts["tracked_deletions"] = True
    if any(ln and not ln.startswith("!! ") for ln in lines):
        return "dirty"  # a tracked change or an untracked (``??``) entry
    if facts is not None:
        # For ``finished_foreign_worktree``, which may not rest on basenames
        # (``.venv/private-backup.json`` passes ``_regenerable``): ANY ignored
        # entry, read from this same status call rather than a second one.
        facts["ignored"] = any(ln.startswith("!! ") for ln in lines)
    for line in lines:
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


@dataclass(frozen=True)
class CleanupIdentity:
    path: str
    registration: tuple[tuple[str, int, int, int], ...]
    ref: str | None
    head: str
    origin: str


def cleanup_identity(path: str) -> CleanupIdentity | None:
    """Pin both sides of the linked registration across the proof's I/O."""
    dotgit = os.path.join(path, ".git")
    dirs = _git(
        ["rev-parse", "--absolute-git-dir", "--path-format=absolute", "--git-common-dir"], path
    )
    lines = (dirs or "").splitlines()
    if len(lines) != 2 or os.path.realpath(path) != path:
        return None
    git_dir, common = lines
    if os.path.dirname(git_dir) != os.path.join(common, "worktrees"):
        return None
    back = os.path.join(git_dir, "gitdir")
    try:
        items = []
        for name, is_dir in (
            (path, True),
            (dotgit, False),
            (common, True),
            (git_dir, True),
            (back, False),
        ):
            st = os.lstat(name)
            if not (stat.S_ISDIR(st.st_mode) if is_dir else stat.S_ISREG(st.st_mode)):
                return None
            items.append((name, st.st_dev, st.st_ino, creation_stamp(st)))
        with open(back, encoding="utf-8") as fh:
            if os.path.realpath(os.path.join(git_dir, fh.read().rstrip("\n"))) != dotgit:
                return None
    except (OSError, UnicodeError):
        return None
    symbolic = _git_proc(["symbolic-ref", "--quiet", "HEAD"], path)
    if symbolic is None or symbolic.returncode not in (0, 1):
        return None
    ref = symbolic.stdout.rstrip("\n") if symbolic.returncode == 0 else None
    if ref is not None and not ref.startswith("refs/heads/"):
        return None
    head = (_git(["rev-parse", "--verify", "HEAD"], path) or "").strip()
    origin = (_git(["remote", "get-url", "origin"], path) or "").rstrip("\n")
    return CleanupIdentity(path, tuple(items), ref, head, origin) if head and origin else None


def cleanup_activity_known(identity: CleanupIdentity, activity: WorktreeActivity) -> bool:
    if identity.ref is None:
        return activity.branch is None and activity.pr_state == "none"
    branch = identity.ref.removeprefix("refs/heads/")
    if activity.branch != branch:
        return False
    if activity.pr_state == "none":
        return True
    # A bare repository on an absolute local path has no forge on this
    # origin. Network remotes and unresolved paths do not get this exception.
    return (
        activity.pr_detail == "no GitHub or GitLab remote"
        and os.path.isabs(identity.origin)
        and (_git(["rev-parse", "--is-bare-repository"], identity.origin) or "").strip() == "true"
    )


def _shared_dependencies_link(real: str, entry: str) -> bool:
    if entry != "node_modules" or os.path.basename(os.path.dirname(real)) != ".worktrees":
        return False
    root = os.path.dirname(os.path.dirname(real))
    common = (_git(["rev-parse", "--path-format=absolute", "--git-common-dir"], real) or "").rstrip(
        "\n"
    )
    if common != os.path.join(root, ".git") or os.path.realpath(root) != root:
        return False
    try:
        if os.readlink(os.path.join(real, entry)) != "../../node_modules":
            return False
        if not stat.S_ISDIR(os.lstat(os.path.join(root, entry)).st_mode):
            return False
    except OSError:
        return False
    # The link is recreated by the repository's documented sharing procedure,
    # only for the same lockfile. A manifest beside a directory proves nothing
    # about that directory's actual contents; no ignored file data is removed.
    for lock in ("pnpm-lock.yaml", "package-lock.json", "yarn.lock", "bun.lock"):
        blob = (_git_quiet(["rev-parse", "--verify", f"HEAD:{lock}"], real) or "").strip()
        if not blob:
            continue
        try:
            if not all(stat.S_ISREG(os.lstat(os.path.join(p, lock)).st_mode) for p in (root, real)):
                continue
        except OSError:
            continue
        if all(
            (_git_quiet(["hash-object", "--no-filters", "--", lock], p) or "").strip() == blob
            for p in (root, real)
        ):
            return True
    return False


def _finder_metadata_file(real: str, entry: str) -> bool:
    """An ignored ``.DS_Store`` that is a plain regular file inside the tree.

    Why (operator request 2026-11-23, guard 0.112.5 field case): removing an
    agent's own clean, merged worktree was held for "ignored entries the guard
    cannot prove regenerable" when the only ignored entries were Finder
    ``.DS_Store`` files. Finder rewrites them whenever a folder is viewed, so
    that hold can never be satisfied on an indexed tree, and they carry only
    view state (icon positions, sort order): losing them loses no user data.

    Constraints: exact basename only, and only a regular file by ``lstat`` — a
    directory, symlink or special file under that name is not Finder's and
    stays held. This is deliberately NOT the broad regenerable basename set
    (build/dist/.venv stay held, PR #4399), and AppleDouble ``._*`` files are
    excluded because they can carry resource-fork data. Fails closed.
    """
    if os.path.basename(entry) != ".DS_Store":
        return False
    physical = os.path.join(real, entry)
    try:
        if not stat.S_ISREG(os.lstat(physical).st_mode):
            return False
        # A symlinked parent would put the file outside the tree being removed.
        parent = os.path.realpath(os.path.dirname(physical))
    except OSError:
        return False
    return parent == real or parent.startswith(real + os.sep)


def _strict_cleanup_status(real: str) -> str | None:
    return _git_quiet(
        [
            "-c",
            "core.ignorestat=false",
            "-c",
            "core.trustctime=true",
            "-c",
            "core.checkStat=default",
            "status",
            "--porcelain=v1",
            "-z",
            "--no-renames",
            "--untracked-files=normal",
            "--ignored=matching",
            "--ignore-submodules=none",
        ],
        real,
    )


def _cleanup_status_path(row: str) -> str | None:
    # NUL porcelain preserves literal path names (not C quoting).
    entry = row[3:].rstrip("/")
    return entry if entry and not os.path.isabs(entry) and ".." not in entry.split("/") else None


@dataclass(frozen=True)
class IgnoredInventory:
    """What one strict status read says about a plain cleanup's ignored payload.

    Why counts and not just a verdict (operator follow-up 2026-11-21, PR #4359
    field case): the hold refusal used to give commit/stash/push guidance even
    for a clean tree whose only blocker was 66 ignored entries, which read like
    the earlier open-PR/unpushed-work refusal. The counts let the refusal name
    the condition that actually failed. Both come from the SAME status read
    the verdict already needed — no extra filesystem walk, no file names.
    """

    #: Ignored entries Git would delete that are not a supported witness (the
    #: shared ``node_modules`` link or a regular-file Finder ``.DS_Store``).
    unverified_ignored: int
    #: Tracked or untracked status rows (the work plain Git itself refuses).
    changes: int


def plain_cleanup_ignored_inventory(
    real: str, *, own_regenerable: bool = False
) -> IgnoredInventory | None:
    """Count unverified ignored entries and other changes; ``None`` = unreadable.

    ``own_regenerable`` (PR #4493): the caller proved, through the creation
    ledger, that the removing session or its family (descendants, or an ancestor
    that delegated the cleanup; never a sibling) made
    this tree. Then an ignored entry whose basename is on :data:`_REGENERABLE`
    (``.venv``, ``__pycache__``, ``.pytest_cache``, ``node_modules`` …) and that
    holds no nested repository is build output the creator produced, and Git
    may take it. WHY: the operator's escalation of 2026-11-11 (finished own
    trees refused for ignored ``.expo``/``.pnpm-store``) was settled by the
    "OWN FIRST" rule in ``classify``; this inventory, added in #4240, then
    held every ignored entry "regardless of creator" and silently reversed
    it (disk-reclamation field reports 2026-11-23: three owners refused on
    their own merged trees holding only ``.venv``/``__pycache__``). A name is
    NOT a witness for anyone else's tree (#4240's point stands there), and
    any ignored entry off the list (``.env``, ``.terraform``) stays held even
    in an own tree: it may be the only copy of hand-made local data.
    """
    status = _strict_cleanup_status(real)
    if status is None:
        return None
    unverified = changes = 0
    # A tracked edit cannot mask an independent ignored-data hold.
    for row in status.split("\0"):
        if not row:
            continue
        if not row.startswith("!! "):
            changes += 1
            continue
        entry = _cleanup_status_path(row)
        if entry is None:
            return None
        # Any tree (#4503): the shared ``node_modules`` link and a regular-file
        # Finder ``.DS_Store`` are verified witnesses for plain removal.
        if _shared_dependencies_link(real, entry) or _finder_metadata_file(real, entry):
            continue
        if own_regenerable and _own_tree_sheds(os.path.join(real, entry)):
            continue
        unverified += 1
    return IgnoredInventory(unverified, changes)


def plain_cleanup_ignored_preserved(real: str) -> bool | None:
    """Plain Git refuses ordinary changes but deletes ignored payload too."""
    inventory = plain_cleanup_ignored_inventory(real)
    if inventory is None:
        return None
    return inventory.unverified_ignored == 0


def _interrupted_cleanup_state(
    real: str, facts: dict[str, bool] | None, *, own_regenerable: bool = False
) -> str:
    """The payload state for the exact-removal (``--force``) route.

    ``own_regenerable`` (PR #4493, QA Q-R4-1): the caller proved through the
    creation ledger that this session or its delegating family made the tree
    (``classify`` drops kin/sibling attribution before calling). Then an
    ignored symlink (Git removes the link, never its target) or an ignored
    entry on :data:`_REGENERABLE` with no nested repository is shed here
    exactly as in :func:`plain_cleanup_ignored_inventory`; without that the
    ``--force`` spelling held an own tree whose only ignored file was
    ``.DS_Store`` while plain ``git worktree remove`` allowed it. Any other
    ignored entry (``.env``) still makes it dirty."""
    # An index entry marked assume-unchanged or skip-worktree can conceal a
    # modified file from status. Gitlinks can conceal an independent repo.
    index = _git_quiet(["ls-files", "-v", "--stage", "-z"], real)
    if index is None:
        return "unknown"
    status = _strict_cleanup_status(real)
    if status is None:
        return "unknown"
    deleted = False
    rows = status.split("\0")
    if facts is not None:
        facts["ignored"] = any(row.startswith("!! ") for row in rows)
    if any(
        not row.startswith("H ") or row.startswith("H 160000 ") for row in index.split("\0") if row
    ):
        return "dirty"
    for row in rows:
        if not row:
            continue
        if row[:3] not in (" D ", "D  ", "!! "):
            return "dirty"
        entry = _cleanup_status_path(row)
        if entry is None:
            return "unknown"
        physical = os.path.join(real, entry)
        if row[:3] != "!! ":
            # A staged deletion may leave a replacement ignored at the same path.
            if os.path.lexists(physical):
                return "dirty"
            deleted = True
            continue
        # A name or package manifest cannot account for the bytes under an
        # ignored directory. The only supported witness is a shared dependency
        # link: Git removes the reproducible link entry, not its referent.
        # For a tree this session did not make, Finder ``.DS_Store`` is NOT a
        # witness on this ``--force`` route (#4503 R1/Q1 kept ``--force`` as on
        # base for foreign trees). A ledger-proven OWN tree sheds the same
        # regenerable names and links as plain removal (#4493 QA Q-R4-1).
        if _shared_dependencies_link(real, entry):
            continue
        if own_regenerable and _own_tree_sheds(physical):
            continue
        return "dirty"
    if _nested_repository(real, worktrees=True, linked_root=True) is not None:
        return "dirty"
    return "deletions-only" if deleted else "clean"


def detached_head(path: str) -> tuple[str, str | None] | None:
    """``(short HEAD, ref containing it)`` for a DETACHED tree, else ``None``.

    ``None`` = HEAD is on a branch, or git could not answer (fail closed: the
    caller then keeps its ordinary ranking). The ref is ``None`` when no
    branch, remote-tracking ref or tag contains HEAD — commits made in the
    detached tree that only its own HEAD file names, which removal loses.
    """
    real = os.path.realpath(path)
    if _git(["symbolic-ref", "-q", "HEAD"], real) is not None:
        return None  # on a branch
    head = _git(["rev-parse", "--short", "HEAD"], real)
    if not (head or "").strip():
        return None
    refs = _git(
        [
            "for-each-ref",
            "--count=1",
            "--format=%(refname:short)",
            "--contains",
            "HEAD",
            "refs/heads",
            "refs/remotes",
            "refs/tags",
        ],
        real,
    )
    if refs is None:
        return None  # unknown reachability is not "reachable"
    return (head or "").strip(), (refs.strip() or None)


def own_detached_scratch(path: str) -> str | None:
    """A fact phrase when an OWN dirty linked tree is disposable scratch, else ``None``.

    WHY (operator directive 2026-11-14, relayed by Aida: "the safety guards
    should never refuse being able to clean up uncommitted files that look
    like scratch, safe to delete, etc. … this sort of ask should never come
    up again"). Field case, mesh node lop-remote-node-2, session
    99ffc68d3d05: it ran ``git worktree add --detach .worktrees/attr3880-ci``,
    then ``git merge-tree`` + ``git read-tree`` left an uncommitted index and
    working-tree changes; ``git worktree remove --force`` of that path went
    to the judge as an own DIRTY tree and the judge refused ("busy detached
    worktree with substantial uncommitted changes" — judge wording, no
    deterministic branch writes it).

    What makes such a tree safe for its creator to drop: there is NO branch,
    so nobody's branch work can be lost, and HEAD itself is contained in a
    ref, so no commit is lost either. What remains — the uncommitted index
    and files — is what the creating session itself put there.

    CONSTRAINTS, each a fail-closed ``None``:
    - only called for a tree the creation ledger already proves own
      (``classify._owned_by_session`` / ``owned_record``) and whose state is
      ``dirty`` — locked/busy/unknown keep their ranking;
    - HEAD on a branch → ``None`` (branch work is the judge's to weigh);
    - HEAD reachable from no ref → ``None`` (dangling commits made in the
      tree are lost with it; treated like unpushed work);
    - a nested repository in any untracked or ignored entry → ``None`` (its
      commits may exist nowhere else; ``worktree_state`` holds the same line).
    """
    real = os.path.realpath(path)
    detached = detached_head(real)
    if detached is None or detached[1] is None:
        return None
    # ``-z`` (review m3, PR #3987): without it git C-quotes special names
    # (``?? "caf\303\251/"``), the quoted spelling is no directory, and the
    # nested-repo check below was silently skipped for exactly those names.
    # With ``-z`` each record is ``XY <path>`` NUL-terminated, verbatim; a
    # rename's extra source record follows its ``R``/``C`` entry and is
    # skipped (only ``??``/``!!`` records are read, and they never have one).
    status = _git(
        ["status", "--porcelain", "-z", "--ignored", "--untracked-files=normal"],
        real,
        timeout=_WORKTREE_STATUS_TIMEOUT_S,
    )
    if status is None:
        return None
    records = status.split("\0")
    skip_next = False
    for record in records:
        if skip_next:
            skip_next = False
            continue
        if record[:1] in ("R", "C") or record[1:2] in ("R", "C"):
            skip_next = True  # the rename/copy source path follows as its own record
            continue
        if not (record.startswith("?? ") or record.startswith("!! ")):
            continue
        entry = os.path.join(real, record[3:].rstrip("/"))
        if os.path.isdir(entry) and _holds_git(entry):
            return None
    head, ref = detached
    return (
        f"own-created detached worktree (no branch; HEAD {head} is kept by {ref}): "
        "removal loses only this session's uncommitted scratch"
    )


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
#: The forge auth the deleting session's bash child would carry, as a zero-arg
#: provider of ``{name: value}`` (set per guard call by ``check_tool_call``).
#:
#: WHY (field report 2026-11-17, lop-remote-node-2 session d5188ee174d4, guard
#: 0.106.6): every ``gh pr list`` the guard ran read "PR state unknown (gh
#: lookup on damianvtran/local-operator failed or timed out)" while the same
#: query in the agent's own shell answered at once (#3963 MERGED). node-2 has
#: no ``gh`` login of its own: it BORROWS the operator's github credential over
#: the mesh, and ``execute_bash`` injects that (``GH_TOKEN``/``GITHUB_TOKEN``
#: via ``network.credentials.github.lent_github_env``) plus the session
#: credential store's ``credential_env()`` into the child only. The guard's
#: ``gh`` ran on the harness's bare ``os.environ``, unauthenticated, exited
#: non-zero, and every clean merged worktree lost its LOW path.
#:
#: Constraints: the provider is LAZY (the borrow can cost one bounded broker
#: round trip, so only a forge lookup pays it, never every classified call);
#: its values reach the ``gh``/``glab`` child env only — never facts, logs or
#: the judge; a provider that raises or yields nothing leaves the bare
#: environment, so a real failure still reads "unknown" (fail closed).
FORGE_AUTH: ContextVar[Callable[[], Mapping[str, str]] | None] = ContextVar(
    "guardrail_forge_auth", default=None
)


def bind_forge_auth(fn: Callable[[], Any]) -> Callable[[], Any]:
    """``fn`` carrying this thread's :data:`FORGE_AUTH` into a worker thread.

    ``ThreadPoolExecutor`` does not copy context variables, and the reach
    lookups (``classify._bounded_lookups``) run ``worktree_activity`` on a
    pool. Only this one variable is carried: copying the whole context would
    also share ``classify``'s per-call glob memo across threads.
    """
    provider = FORGE_AUTH.get()
    if provider is None:
        return fn

    def bound() -> Any:
        token = FORGE_AUTH.set(provider)
        try:
            return fn()
        finally:
            FORGE_AUTH.reset(token)

    return bound


#: The only session-injected names a forge child gets, per forge (review m1,
#: PR #4161). Tokens only: a stored ``GH_HOST``/``GH_CONFIG_DIR``/
#: ``GITLAB_HOST`` would re-point the guard's lookup at another host or
#: config, and ``GH_ENTERPRISE_TOKEN`` is for hosts the pinned ``--repo``
#: never names. The host itself is pinned in argv (:func:`_gh_repo_arg`,
#: :func:`_glab_repo_arg`), so a token goes only to the host it was minted for.
_FORGE_TOKEN_NAMES = {
    "gh": ("GH_TOKEN", "GITHUB_TOKEN"),
    "glab": ("GITLAB_TOKEN", "GLAB_TOKEN"),
}


def _gh_repo_arg(repo: str) -> str:
    """``repo`` as ``github.com/<owner>/<name>``: ``gh`` then ignores ``GH_HOST``.

    :func:`_github_repo` only ever reads github.com remotes, so the host is
    known; pinning it keeps an operator's enterprise ``GH_HOST`` from routing
    the guard's lookup (review m1, PR #4161). An already-qualified value passes.
    """
    return repo if repo.count("/") >= 2 else f"github.com/{repo}"


def _glab_repo_arg(repo: str) -> str:
    """``repo`` as a full ``https://<host>/<path>`` URL, so ``GITLAB_HOST`` cannot move it.

    :func:`_gitlab_repo` returns a bare path for gitlab.com and a URL for a
    self-hosted host; the bare one is pinned to gitlab.com here (review m1).
    """
    return repo if "://" in repo else f"https://gitlab.com/{repo}"


def _forge_cli_env(cli: str = "gh", **overrides: str) -> dict[str, str]:
    """The environment for one ``gh``/``glab`` child: ours, plus the session's forge token.

    Only :data:`_FORGE_TOKEN_NAMES` for ``cli`` are taken from the provider,
    whatever else it returns.
    """
    env = dict(os.environ, **overrides)
    provider = FORGE_AUTH.get()
    if provider is None:
        return env
    try:
        auth = provider()
    except Exception:  # noqa: BLE001 — no auth is the old behaviour: "unknown", never a pass
        logger.debug("guard forge auth unavailable")
        return env
    allowed = _FORGE_TOKEN_NAMES.get(cli, ())
    env.update({str(k): str(v) for k, v in (auth or {}).items() if str(k) in allowed})
    return env


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
    #: The change's head commit as the forge records it (GitHub ``headRefOid``,
    #: GitLab ``sha``); "" when not read. Only :func:`finished_foreign_worktree`
    #: reads it, to accept a squash merge whose MERGED head is this tree's HEAD.
    head: str = ""
    #: The branch the change targets (GitHub ``baseRefName``, GitLab
    #: ``target_branch``); "" when not read. A merged change counts as landed
    #: only when this is origin's default branch (review m2, PR #4161): a
    #: stacked PR merged into another feature branch never reached it.
    base: str = ""


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
    #: Set when the forge could not be asked and git proved the branch landed
    #: instead (:func:`git_merged`): the fact phrase. ``pr_state`` is then
    #: ``none`` and ``pr_detail`` says why the forge was not read.
    git_merged: str = ""
    #: Why git could NOT prove the branch landed, when the forge was not read
    #: either (shown after the unknown state, never ranked).
    git_unproven: str = ""
    #: ``(head commit, base branch)`` of this branch's MERGED changes on the
    #: page read. Never ranked on its own: only
    #: :func:`finished_foreign_worktree` uses it, and only as the merge proof
    #: for a tree whose HEAD equals a head merged into origin's default branch.
    merged_heads: tuple[tuple[str, str], ...] = ()

    def facts(self) -> str:
        branch = self.branch or "(detached HEAD)"
        noun = "PR" if self.pr_label == "PR #" else "MR"
        if self.pr_state == "open":
            pr = f"open {self.pr_ref}"
        elif self.pr_state == "none" and self.git_merged:
            pr = f"{noun} state from git ({self.pr_detail}): {self.git_merged}"
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
            if self.git_unproven:
                pr += f"; git did not prove it merged: {self.git_unproven}"
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
    env = _forge_cli_env("gh", GH_PROMPT_DISABLED="1")
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                _gh_repo_arg(repo),
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
    env = _forge_cli_env("glab", GLAB_NO_PROMPT="1")
    try:
        proc = subprocess.run(
            [
                "glab",
                "mr",
                "list",
                "--repo",
                _glab_repo_arg(repo),
                "--source-branch",
                branch,
                "-F",
                "json",
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
    env = _forge_cli_env("gh", GH_PROMPT_DISABLED="1")
    argv = [
        "gh",
        "pr",
        "list",
        "--repo",
        _gh_repo_arg(repo),
        "--head",
        branch,
        "--state",
        "all",
        "--json",
        "number,state,mergedAt,closedAt,updatedAt,headRefOid,baseRefName",
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
            out.append(
                ForgeChange(
                    int(r["number"]),
                    state,
                    date,
                    str(r.get("headRefOid") or ""),
                    str(r.get("baseRefName") or ""),
                )
            )
        return out
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _glab_mrs(repo: str, branch: str) -> list[ForgeChange] | None:
    """Every MR (any state, newest page) sourced from ``branch``; ``None`` if unknown.

    glab 1.91's ``mr list`` takes ``--all`` (any state) and ``--per-page``
    (checked from ``--help``), so GitLab gets the same history as GitHub.
    GitLab's open state is ``opened``; it is normalised to ``open`` here.
    """
    env = _forge_cli_env("glab", GLAB_NO_PROMPT="1")
    argv = [
        "glab",
        "mr",
        "list",
        "--repo",
        _glab_repo_arg(repo),
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
            out.append(
                ForgeChange(
                    int(r["iid"]),
                    state,
                    date,
                    str(r.get("sha") or ""),
                    str(r.get("target_branch") or ""),
                )
            )
        return out
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


class UnattributedWorktree(OSError):
    """No current-generation ``git worktree add`` row for this path: the
    ledger was read and holds none, or no ledger file exists yet (nothing was
    ever recorded). Either way nobody is on record as its owner. Distinct
    from an unreadable ledger (plain ``OSError``/``sqlite3.Error``), which
    says nothing (QA Q-R2-1, PR #4493)."""


def worktree_creators(real: str, *, strict: bool = True) -> frozenset[str]:
    """Current generation attribution for strict cleanup (missing is unknown)."""
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
        raise UnattributedWorktree("worktree owner attribution unavailable")
    return frozenset(creators)


def _running_owners(
    real: str, session_id: str, *, strict: bool = False, allow_unattributed: bool = False
) -> tuple[str, ...]:
    """Running sessions (not the caller) that own the worktree at ``real``.

    Owner = the session the creation ledger attributes the ``git worktree
    add`` to (``kind='worktree'`` row for this exact path), or any live/wedged
    session whose recorded cwd is inside the tree. The CALLER is excluded: a
    session removing its own worktree is the owner acting, not a conflict.
    Read fresh on every call (no cache). Default: unreadable sources name no
    owner. ``strict`` requires current-generation attribution and a complete
    registry, raising on missing/unreadable facts rather than claiming idle.

    ``allow_unattributed`` (strict only; the operator's exact-path approval
    waiver, PR #4493 Q-R2-1): a tree made in a terminal, before the ledger,
    or whose Linux ctime stamp drifted has NO recorded creator, so the only
    owner left to find is a live session whose cwd is inside it, and the
    strict registry scan below still finds that (and still raises when the
    registry is incomplete). An unreadable ledger still raises.
    """
    try:
        creators = worktree_creators(real, strict=strict)
    except UnattributedWorktree:
        if not (strict and allow_unattributed):
            raise
        creators = frozenset()
    # A RUNNING ANCESTOR that created the tree is the delegator, not a
    # conflicting owner (field case 2026-11-14): a parent is live for as long
    # as its cleanup subagent runs, so counting it as "owning session
    # running" refused every delegated cleanup of the parent's own worktree.
    # Parent links only (``subagent_ancestors``). A family member whose CWD is
    # inside the tree still counts below, and a running sibling or cousin
    # creator always counts: only the delegation chain above is waived.
    if session_id and not strict and creators:
        creators -= set(subagent_ancestors(session_id))
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
        return _git_fallback(real, branch, unknown)
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
    return WorktreeActivity(
        branch,
        "none",
        pr_label=label,
        owners_running=owners,
        newest=newest,
        merged_heads=tuple(
            (c.head, c.base) for c in changes if c.state == "merged" and c.head and c.base
        ),
    )


@dataclass(frozen=True)
class FinishedForeign:
    """Whether a FOREIGN linked worktree is finished work that is safe to remove.

    ``verdict``: ``safe`` / ``not-safe`` / ``unknown``; only ``safe`` earns the
    route. ``reasons``: value-free codes (the positive facts when safe, the
    first failed fact otherwise), so a caller can journal or show them without
    leaking branch names or paths. ``state`` and ``activity`` are the
    ``worktree_state``/``worktree_activity`` readings it used, kept so the
    judge brief of an unsafe tree reuses them instead of looking twice.
    """

    verdict: str
    reasons: tuple[str, ...]
    state: str = "unknown"
    activity: WorktreeActivity | None = None

    @property
    def safe(self) -> bool:
        return self.verdict == "safe"

    def warning(self, root: str) -> str:
        """The short steering note a LOW from this route carries (``GuardOutcome.warnings``)."""
        # Facts follow the proof actually used (#3992 author's constraint 3):
        # a squash merge reads "content already on origin", an unread forge
        # never claims "no open PR".
        merged = (
            "HEAD's content already in origin's default branch (merged by content)"
            if "merged-by-content" in self.reasons
            else (
                "HEAD is the head of a MERGED PR/MR (squash merge)"
                if "merged-by-pr" in self.reasons
                else "merged into origin's default branch"
            )
        )
        pr = (
            "PR state unread (forge unavailable)"
            if "forge-unread" in self.reasons
            else "no open PR"
        )
        return (
            f"Guardrail {_DANGEROUS_DELETE}: finished worktree {root} removed without review — "
            f"registered linked tree, HEAD on its branch and {merged}, "
            f"{pr}, no running owner, clean with no ignored files. "
            # Q4 on PR #3992: the ref lives in the primary checkout, which an
            # enclosing delete in the same call may remove too, so the note
            # states the condition instead of promising the ref survives.
            "Its branch ref stays in the main checkout unless that is deleted too."
        )


#: The rule name, spelled here to avoid importing ``check`` (it imports us).
_DANGEROUS_DELETE = "dangerous-delete"
#: The positive facts a ``safe`` :class:`FinishedForeign` lists.
_FINISHED_FACTS = (
    "registered-linked",
    "branch-retained",
    "merged-on-origin",
    "no-open-pr",
    "owner-idle",
    "clean-no-ignored",
)


def finished_foreign_worktree(
    path: str,
    session_id: str,
    *,
    activity: WorktreeActivity | None = None,
    interrupted_cleanup: bool = False,
    identity: CleanupIdentity | None = None,
) -> FinishedForeign:
    """Prove a worktree another (finished) session made is safe to remove, or say why not.

    WHY (operator request, cleanup follow-on): "the guardrails are not allowing
    agents to clean up after themselves on obviously safe directories and
    worktree folders". A tree is often made by a subagent or an earlier
    session and finished later; every such foreign tree went to the judge.
    Operator decision (ask a-891c, "Yes, rank it LOW"): a clean, merged-on-
    origin, idle worktree with no open PR is LOW even when another, finished
    session created it. Every fact below must be POSITIVELY established; a
    failed or unreadable check is ``unknown``, never safe:

    1. registered linked tree, clean, no lock, no live cwd (``worktree_state``)
       and NO ignored entry at all \u2014 a basename (``.venv``, ``build``) is not
       proof the content is regenerable, so such trees stay with the judge;
    2. HEAD on a named branch whose ref equals HEAD (detached \u2192 not-safe);
    3. HEAD is an ancestor of the SHA origin ADVERTISES for its default branch
       (``ls-remote --symref``: never a hardcoded ``main``, never a fetch),
       and the local ``refs/remotes/origin/<default>`` equals it, since the
       ancestry walk needs that object locally. A squash/rebase merge has no
       ancestry: it passes when HEAD's content is already in that tip
       (:func:`content_merged`), or when the forge lists a MERGED PR/MR whose
       head commit IS HEAD (``merged-by-pr``). A merged badge on any other
       head proves nothing about later commits and stays with the judge;
    4. no open PR/MR (``worktree_activity`` says ``none``; ``unknown`` fails);
    5. no running owner, VERIFIED: ``_running_owners(strict=True)`` raises on
       an unreadable ledger/registry or a tree with no ledger creator row
       (pre-ledger trees therefore stay with the judge);
    6. HEAD unchanged at the end (a commit during the checks fails closed).

    COST: reuses ``_git`` (``_GIT_TIMEOUT_S``) and ``worktree_state``'s status
    bound, plus one forge lookup; callers run it inside their own deadline
    (``classify._bounded_lookups``) and memoise it per classification. No
    timeout of its own. Pass ``activity`` when the caller already has it.
    """

    def no(code: str, state: str = "unknown") -> FinishedForeign:
        return FinishedForeign("not-safe", (code,), state, activity)

    def unknown(code: str, state: str = "unknown") -> FinishedForeign:
        return FinishedForeign("unknown", (code,), state, activity)

    real = os.path.realpath(path)
    facts: dict[str, bool] = {}
    # Strict on hidden pids (review M1, PR #4161): a ``safe`` answer here
    # ranks ``--force`` and ``rm -rf`` LOW, and neither refuses untracked
    # content the way a plain ``git worktree remove`` does.
    states = ("clean", "deletions-only") if interrupted_cleanup else ("clean",)
    state = worktree_state(
        real, facts=facts, hidden_blocks=True, interrupted_cleanup=interrupted_cleanup
    )
    if state in ("not-linked", "dirty", "locked", "stale-initializing", "busy"):
        return no(f"state-{state}", state)
    if state not in states:
        return unknown("state-unknown", state)
    if not interrupted_cleanup and facts.get("ignored", True):
        return no("ignored-content", state)
    ref = (_git(["symbolic-ref", "--quiet", "HEAD"], real) or "").strip()
    if not ref.startswith("refs/heads/"):
        return no("detached", state)
    heads = (_git(["rev-parse", "HEAD", ref], real) or "").split()
    if len(heads) != 2:
        return unknown("branch-unreadable", state)
    if heads[0] != heads[1]:
        return no("branch-moved", state)
    head = heads[0]
    if interrupted_cleanup and (
        identity is None
        or identity.head != head
        or identity.ref != ref
        or cleanup_identity(real) != identity
    ):
        return unknown("registration-or-ref-moved", state)
    try:
        owners = _running_owners(real, session_id, strict=True)
    except Exception:  # noqa: BLE001 — incomplete owner facts are unknown, not idle
        return unknown("owner-unknown", state)
    if owners:
        return no("owner-running", state)
    # Prompts and credential helpers off: the guard never asks for, or reads,
    # a credential. ``BatchMode`` stops ssh from prompting for a passphrase.
    advert = _git(
        [
            "-c",
            "credential.helper=",
            "-c",
            "core.sshCommand=ssh -o BatchMode=yes",
            "ls-remote",
            "--symref",
            "origin",
            "HEAD",
        ],
        real,
    )
    if advert is None:
        return unknown("origin-unreachable", state)
    default, advertised = None, None
    for line in advert.splitlines():
        left, _, right = line.partition("\t")
        if left.startswith("ref: refs/heads/") and right == "HEAD":
            default = left[len("ref: refs/heads/") :]
        elif right == "HEAD" and re.fullmatch(r"[0-9a-f]{40,64}", left):
            advertised = left
    if not default or not advertised:
        return unknown("origin-default-unknown", state)
    local = _git(["rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{default}"], real)
    if (local or "").strip() != advertised:
        # Stale or missing local copy of what origin advertises: the caller
        # may fetch and retry; the guard never fetches.
        return unknown("origin-ref-stale", state)
    proc = _git_proc(
        ["--no-replace-objects", "merge-base", "--is-ancestor", head, advertised], real
    )
    if proc is None or proc.returncode not in (0, 1):
        return unknown("ancestry-unknown", state)
    merged_fact = "merged-on-origin"
    if proc.returncode == 1:
        # Squash/rebase merge (node-2 census 2026-11-15: approval-provenance,
        # github-default-repos #3842, copy-wipe-dispatch-identity #3948,
        # mobility-link-drop #3855 — all squash-merged, so no ancestry). HEAD's
        # CONTENT must already be in the advertised tip, which the local ref
        # was just shown to equal; a local commit after the squash fails it.
        if not interrupted_cleanup and content_merged(real, head, advertised):
            merged_fact = "merged-by-content"
        else:
            # The forge's word, for the squash merge content cannot prove
            # (field report 2026-11-17, node-2: main later edited a path the
            # branch changed — routine in a busy repo — so the tree-equality
            # proof fails though PR #3963 MERGED). Accepted ONLY when a
            # MERGED change's recorded head IS this tree's HEAD: nothing was
            # committed after what landed. A merged badge on an older head
            # proves nothing about later commits, so that stays not-merged.
            # The change must also have merged INTO ``default`` (review m2, PR
            # #4161): a stacked PR merged into a feature branch has the same
            # head but its content never reached origin's default branch.
            # The open-PR, owner, clean and race checks below all still run.
            if activity is None:
                activity = worktree_activity(real, session_id)
            if activity.git_merged or (head, default) not in activity.merged_heads:
                return no("not-merged", state)
            merged_fact = "merged-by-pr"
    if activity is None:
        activity = worktree_activity(real, session_id)
    if activity.pr_state == "open":
        return no("open-pr", state)
    if activity.pr_state != "none" or activity.branch != ref[len("refs/heads/") :]:
        return unknown("pr-unknown", state)
    pr_fact = "no-open-pr"
    if activity.git_merged:
        # The forge lookup FAILED and ``_git_fallback`` inferred ``none`` from
        # git. That is still NOT "no open PR" (#3992: an unread forge stays
        # unknown), so the fact is named for what was proven instead: HEAD's
        # content is already on origin's ADVERTISED default tip (checked
        # above, ancestry or tree-equality), so nothing on this tree is
        # unlanded — a PR still open on these commits would propose nothing
        # main lacks. WHY accept it (operator, ask a-798d, 2026-11-15:
        # "these requests should not require human approval"): on node-2 gh
        # fails or times out, and every finished foreign tree went to the
        # judge. A KNOWN open PR still refuses above (CRITICAL in ``_rank``).
        pr_fact = "forge-unread"
    # Race re-check (review R4 on PR #3992): the forge lookup above can take
    # up to the ``gh`` timeout, so the clean / no-ignored / not-busy facts read
    # at the start are re-read with the same helper just before answering. A
    # file written, or a shell that cd'd in, during that window fails it. The
    # process-cwd inventory is cached for ``_CWD_INVENTORY_TTL_S``, longer than
    # that window, so the cache is dropped first or "busy" would be the stale
    # reading. Check-then-act stays inherent; this narrows it to the end.
    global _cwd_inventory_cache
    _cwd_inventory_cache = None
    again: dict[str, bool] = {}
    if worktree_state(
        real, facts=again, hidden_blocks=True, interrupted_cleanup=interrupted_cleanup
    ) not in states or (not interrupted_cleanup and again.get("ignored", True)):
        return unknown("state-moved", state)
    if (_git(["rev-parse", "--verify", "HEAD"], real) or "").strip() != head:
        return unknown("head-moved", state)
    facts_out = tuple(
        merged_fact if f == "merged-on-origin" else pr_fact if f == "no-open-pr" else f
        for f in _FINISHED_FACTS
    )
    return FinishedForeign("safe", facts_out, state, activity)


def pushed_worktree_head(real: str, head: str, ref: str | None) -> bool:
    """Prove an owned HEAD survives on origin, never just a stale local ref."""
    refs = ["HEAD"]
    if ref:
        refs.append(ref)
    advert = _git_net(
        ["-c", "core.sshCommand=ssh -o BatchMode=yes", "ls-remote", "origin", *refs],
        real,
        timeout=_LS_REMOTE_TIMEOUT_S,
    )
    if advert is None:
        return False
    for line in advert.splitlines():
        tip, _, name = line.partition("\t")
        if name in refs and re.fullmatch(r"[0-9a-f]{40,64}", tip):
            proc = _git_proc(
                ["--no-replace-objects", "merge-base", "--is-ancestor", head, tip], real
            )
            if tip == head or (proc is not None and proc.returncode == 0):
                return True
    return False


def _running_session_ids() -> set[str]:
    """Session ids the registry reads as ``live`` or ``wedged``; RAISES when unreadable.

    ``strict`` scan, no reaping (a guard check never moves another session's
    record): an incomplete registry is "unknown", never "everyone ended".
    """
    from local_operator.session.runtime import registry

    running: set[str] = set()
    for rec, state in registry.scan(None, reap=False, strict=True):
        sid = str(getattr(rec, "session_id", "") or "")
        if not sid:
            raise OSError("session record without an id")
        if state in ("live", "wedged"):
            running.add(sid)
    return running


def _creators_inside(real: str) -> set[str] | None:
    """Creators of every ledger row strictly INSIDE ``real``; ``None`` if unreadable.

    Range scan on the indexed ``path`` column (``real + "/"`` up to
    ``real + "0"``, the next byte after ``/``), so no LIKE escaping.
    """
    try:
        conn = _connect(ledger_path(), create=False)
        if conn is None:
            return None
        try:
            return {
                str(r[0])
                for r in conn.execute(
                    "SELECT DISTINCT creator FROM created WHERE path > ? AND path < ?",
                    (real + os.sep, real + chr(ord(os.sep) + 1)),
                )
            }
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None


def _entry_dev(entry: os.DirEntry[str]) -> int:
    """The device an entry lives on (a seam: tests stand in a mount point)."""
    return entry.stat(follow_symlinks=False).st_dev


def _nested_repository(
    real: str, *, worktrees: bool = False, linked_root: bool = False
) -> str | None:
    """Why ``real`` may hold a repository of its own (not a linked worktree), or ``None``.

    A ``.git`` DIRECTORY anywhere in the bounded walk is a clone or ``git
    init`` whose commits may exist nowhere else; a ``.git`` FILE is a linked
    worktree's pointer, which :func:`classify._reached_worktrees` already held
    to its own bar. ``None`` only when the walk saw EVERYTHING: an exhausted
    walk cannot rule one out, so it counts.

    No depth cap here (review M2, PR #4098): the shared ``_WALK_DEPTH`` (4)
    stopped this walk early and answered "no repository" with directories
    left unscanned, while scratch-HOME rigs nest clones at
    ``home/src/github.com/<org>/<repo>``. A depth cap that reported "too
    deep" instead would send most real rigs (``home/.local-operator/
    sessions/<id>/…``) back to the judge, so the walk is bounded by
    ``_WALK_ENTRIES`` alone — exhausting it still counts — and stays finite
    without one: symlinks are never followed, and (review m2) a directory on
    another device than the rig is refused, never entered (a mounted volume
    is not the rig's to delete, and BSD ``rm -rf`` without ``-x`` would
    descend into it).

    ``worktrees=True`` also counts a ``.git`` FILE (a linked worktree inside):
    for callers with no ``_reached_worktrees`` pass of their own, such as the
    typed whole-set settlement (``check._typed_set_recoverable``, review B2).
    """
    try:
        root_dev = os.stat(real).st_dev
    except OSError:
        return f"unreadable directory {real}"
    seen = 0
    stack: list[str] = [real]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            return f"unreadable directory {current}"
        seen += len(entries)
        if seen > _WALK_ENTRIES:
            return "too many entries to rule out a nested repository"
        if linked_root and current != real and {"HEAD", "objects"} <= {e.name for e in entries}:
            return f"possible bare repository at {current}"
        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
                dev = _entry_dev(entry) if is_dir else root_dev
            except OSError:
                return f"unreadable entry {entry.path}"
            if (
                linked_root
                and current == real
                and entry.name == ".git"
                and entry.is_file(follow_symlinks=False)
            ):
                continue
            if entry.name == ".git" and is_dir:
                return f"nested repository at {current}"
            if worktrees and entry.name == ".git":
                return f"linked worktree at {current}"
            if not is_dir:
                continue
            if dev != root_dev:
                return f"mount point at {entry.path}"
            stack.append(entry.path)
    return None


def finished_rig(path: str, session_id: str) -> tuple[str | None, str]:
    """``(fact, "")`` for a ``$HOME`` rig another, ended session left; else ``(None, why)``.

    WHY (node-2 census, 2026-11-15, v0.105.3; operator, ask a-798d: "these
    requests should not require human approval and shouldn't be blocked by
    guardrails in the future"): 17 ``~/qa-*`` rigs were refused with "No
    operator ask answer names <path>" although the sessions that built them
    had ended. A top-level ``$HOME`` directory is HIGH because it may be the
    operator's own data; a directory the creation ledger records as made by
    an agent session that has ENDED is that session's leftover, not data.

    Every fact is positively established, else ``None`` (the judge decides):
    - a ledger row records exactly this directory (same inode generation),
      by a session other than the caller (the caller's own rig already has
      the ledger LOW). NO row \u2014 a pre-ledger or hand-made directory \u2014 never
      qualifies: only the judge can tell a rig from the operator's data;
    - every creator of that row AND of every row inside it is NOT running per
      a strict registry read (unreadable registry/ledger: unknown);
    - no live process outside the caller's family has its cwd inside
      (:func:`cwd_holders`; an unreadable inventory is busy);
    - it is not a repository itself and holds no ``.git`` directory
      (:func:`_nested_repository`) \u2014 a clone's unpushed commits may exist
      nowhere else. Linked worktrees inside are ranked by the caller first
      (``_reached_worktrees``), each held to the finished-worktree bar.
    Location (direct child of ``$HOME``, never the state root, secrets or
    install layout) is the caller's check: those are CRITICAL before this.
    """
    real = os.path.realpath(path)
    if not os.path.isdir(real) or os.path.islink(path):
        return None, "not a directory"
    creators = set(recorded_creators(real))
    if not creators:
        return None, "no creation ledger row for this directory"
    if session_id and session_id in creators:
        return None, "created by this session"
    inside = _creators_inside(real)
    if inside is None:
        return None, "creation ledger unreadable"
    try:
        running = _running_session_ids()
    except Exception:  # noqa: BLE001 \u2014 an unreadable registry is unknown, not ended
        return None, "session registry unreadable"
    live = sorted((creators | (inside - {session_id})) & running)
    if live:
        return None, "creating session still running: " + ", ".join(live)
    if os.path.lexists(os.path.join(real, ".git")) or is_primary_checkout(real):
        return None, "a git repository root"
    nested = _nested_repository(real)
    if nested:
        return None, nested
    # ``rm -rf`` of a rig ranks LOW with no judge, and it deletes whatever a
    # hidden process is doing in there: a hidden pid stays blocking (review
    # M1, PR #4161; the relaxed rule is for plain ``git worktree remove``).
    holders = cwd_holders(real, session_id, hidden_blocks=True)
    if holders is None:
        return None, "live-process scan unavailable"
    foreign_holders = [h.describe() for h in holders if not h.family]
    if foreign_holders:
        return None, "in use by " + ", ".join(foreign_holders)
    return (
        "finished work: rig recorded by ended session "
        + ", ".join(sorted(creators))
        + "; no running creator, no live holder, no nested repository"
    ), ""


def _git_fallback(real: str, branch: str, unknown: WorktreeActivity) -> WorktreeActivity:
    """``unknown``, or ``none`` when git proves the branch landed (gap 1b).

    WHY (operator directive 2026-11-14, node-2 field case 1): where ``gh`` has
    no auth or no network, a branch whose PR merged long ago read "PR state
    unknown", which holds every tree that is not provably the caller's own.
    A branch whose every commit is already in the remote's default branch has
    nothing a PR could still be proposing from it, so no PR can be open on
    THESE commits. ``pr_state`` stays ``unknown`` whenever git cannot prove
    that (:func:`git_merged`).
    """
    merged, why = git_merged(real, branch)
    if not merged:
        return replace(unknown, git_unproven=why)
    return replace(
        unknown,
        pr_state="none",
        pr_detail=unknown.pr_detail,
        history_checked=False,
        git_merged=why,
    )


#: ``git ls-remote`` bound. Measured 2026-11-14 against
#: github.com/damianvtran/local-operator from this Mac: 0.59-1.32 s (n=5);
#: the ``gh`` bound (8 s) already sits above that, so the same value.
_LS_REMOTE_TIMEOUT_S = 8.0
#: ``git fetch`` of ONE branch when its tip is not local yet. Measured
#: 2026-11-14: 5.21 s (n=1, ``main`` several merges behind); a fetch that does
#: not finish leaves the already-fetched refs as the only evidence.
_FETCH_TIMEOUT_S = 20.0


def _askpass_false() -> str:
    """An absolute ``false`` for ``GIT_ASKPASS``, resolved where it actually lives.

    WHY (QA round 2, PR #4098): ``/bin/false`` does not exist on macOS (it is
    ``/usr/bin/false``), so git failed to spawn the askpass at all. That still
    failed closed, but by accident. Looked up only in the system directories,
    never ``$PATH``, so nothing on the caller's path can stand in for it; the
    last resort is an empty string, which git treats as "no askpass" (and
    ``GIT_TERMINAL_PROMPT=0`` plus no tty still refuses any prompt).
    """
    return shutil.which("false", path="/usr/bin:/bin") or ""


def _git_net(args: Sequence[str], cwd: str, *, timeout: float) -> str | None:
    """A git call that may reach the network: no prompt, no tty, group-reaped.

    ``start_new_session`` drops the controlling terminal, so neither git nor
    ssh can prompt for a password or passphrase inside a guard check, and a
    timeout kills the WHOLE group (``git-remote-https``/``ssh`` children
    included), not just the git parent. Read-only on the working tree: callers
    pass only ``ls-remote`` and a ``fetch`` that writes objects only (no
    remote-tracking ref, no ``FETCH_HEAD``: :func:`git_merged`).
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Review m4 (PR #4098): no credential helper, no prompt, no askpass, so
    # an https origin never reaches the keychain (the system gitconfig sets
    # ``credential.helper=osxkeychain`` here) — the same posture as the
    # ``finished_foreign_worktree`` ls-remote. Unauthenticated reads only.
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_ASKPASS=_askpass_false())
    env.pop("SSH_ASKPASS", None)
    try:
        proc = subprocess.Popen(
            ["git", "-c", "credential.helper=", "-c", "core.hooksPath=/dev/null", "-C", cwd, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            start_new_session=True,
        )
    except OSError:
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.communicate()
        return None
    return out if proc.returncode == 0 else None


def _has_commit(real: str, sha: str) -> bool:
    return _git(["cat-file", "-e", f"{sha}^{{commit}}"], real) is not None


def _is_ancestor(real: str, sha: str, of: str) -> bool:
    return _git(["merge-base", "--is-ancestor", sha, of], real) is not None


def _local_ref(real: str, ref: str) -> str | None:
    out = _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], real)
    return (out or "").strip() or None


def git_merged(path: str, branch: str) -> tuple[bool, str]:
    """``(True, fact)`` when ``branch``'s tip is in ``origin``'s default branch.

    Order: one bounded ``git ls-remote --symref origin HEAD refs/heads/<b>``
    names the default branch and both remote tips; if the default tip is not
    local yet, one bounded ``git fetch`` of that branch brings it. When the
    remote cannot be read, the ALREADY-FETCHED refs decide
    (``refs/remotes/origin/HEAD``, else ``origin/main``/``origin/master``, and
    ``refs/remotes/origin/<b>``) and the fact says so.

    Merged only when BOTH hold — otherwise ``(False, why)``:
    - the local branch tip is an ancestor of the default branch's tip (so no
      unmerged or unpushed commit; a squash-merged branch is NOT proven and
      stays unknown);
    - the remote branch, if it exists, is the same commit or also an ancestor
      of the default tip. WHY (operator incident 2026-11-09): a tree whose
      local HEAD was an ancestor of main still had an OPEN PR, because the
      PR's newer commits were on the remote branch. A remote tip this clone
      does not have is not proof either way, so it is not merged.
    Stale refs can only under-claim: an existing local ref must itself prove
    the ancestry, and every check is a git object query, never a text claim.
    """
    real = os.path.realpath(path)
    tip = _local_ref(real, f"refs/heads/{branch}")
    if tip is None:
        return False, "branch tip unreadable"
    default = default_tip = remote_tip = None
    listing = _git_net(
        ["ls-remote", "--symref", "origin", "HEAD", f"refs/heads/{branch}"],
        real,
        timeout=_LS_REMOTE_TIMEOUT_S,
    )
    source = "read from origin"
    # The local copy of origin's default branch as it stood BEFORE any fetch
    # below: the squash proof only counts against a ref that already equals
    # what origin advertises (#3992 author's constraint, 2026-11-15).
    pre_fetch_local: str | None = None
    if listing is not None:
        for line in listing.splitlines():
            left, _, name = line.partition("\t")
            if left.startswith("ref: refs/heads/") and name == "HEAD":
                default = left.removeprefix("ref: refs/heads/")
            elif name == "HEAD":
                default_tip = left.strip()
            elif name == f"refs/heads/{branch}":
                remote_tip = left.strip()
        if default:
            pre_fetch_local = _local_ref(real, f"refs/remotes/origin/{default}")
        if default and default_tip and not _has_commit(real, default_tip):
            # Objects only (QA Q1, PR #4098): ``--refmap=`` stops the
            # opportunistic update of ``refs/remotes/origin/<default>`` and
            # ``--no-write-fetch-head`` leaves FETCH_HEAD, so no ref moves.
            # The ancestry proof below reads the advertised SHA itself; the
            # squash proof needs the local ref to have equalled it BEFORE this
            # call, and since this fetch never moves that ref, a second call
            # gives the same answer as the first (preview == check).
            _git_net(
                [
                    "fetch",
                    "--no-tags",
                    "--quiet",
                    "--refmap=",
                    "--no-write-fetch-head",
                    "origin",
                    f"refs/heads/{default}",
                ],
                real,
                timeout=_FETCH_TIMEOUT_S,
            )
    if listing is None or not default or not default_tip:
        source = "origin unreachable; already-fetched refs"
        remote_tip = _local_ref(real, f"refs/remotes/origin/{branch}")
        head = _git(["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"], real)
        candidates = [(head or "").strip().removeprefix("origin/"), "main", "master"]
        default = default_tip = None
        for name in (c for c in candidates if c):
            default_tip = _local_ref(real, f"refs/remotes/origin/{name}")
            if default_tip:
                default = name
                break
        if not default or not default_tip:
            return False, "no default branch ref for origin"
    if not _has_commit(real, default_tip):
        return False, f"origin/{default} tip {default_tip[:9]} not fetched"
    # Squash/rebase proof (:func:`content_merged`) is held to a stricter bar
    # than ancestry: only against the tip origin ADVERTISED just now, and only
    # when the local ``refs/remotes/origin/<default>`` already equalled it
    # before any fetch. An unreachable origin (already-fetched refs) or a
    # stale/mismatched local ref leaves a squash-merged branch unproven, never
    # merged: a tree-equality against an old snapshot of main says nothing
    # about whether the content is on origin NOW.
    squash_ok = listing is not None and pre_fetch_local == default_tip
    squashed = False
    if not _is_ancestor(real, tip, default_tip):
        if not (squash_ok and content_merged(real, tip, default_tip)):
            return False, f"branch tip {tip[:9]} is not in origin/{default} ({source})"
        squashed = True
    if remote_tip and remote_tip != tip:
        if not _has_commit(real, remote_tip) or not (
            _is_ancestor(real, remote_tip, default_tip)
            or (squash_ok and content_merged(real, remote_tip, default_tip))
        ):
            return False, (
                f"origin/{branch} at {remote_tip[:9]} has commits not proven in "
                f"origin/{default} ({source})"
            )
    if squashed:
        return True, (
            f"merged by content: branch tip {tip[:9]}'s content is already in "
            f"origin/{default} (squash/rebase merge; merging it changes nothing; {source})"
        )
    return True, f"branch tip {tip[:9]} is in origin/{default} ({source})"


def content_merged(real: str, tip: str, default_tip: str) -> bool:
    """Is every change on ``tip`` already in ``default_tip``'s tree? (squash/rebase merges)

    WHY (node-2 field census, 2026-11-15, v0.105.3): the operator squash-merges
    PRs, so a merged branch's tip is NEVER an ancestor of ``origin/main``.
    approval-provenance was refused from its own session with "tip 78c35f517
    not in origin/main", and 5 more worktrees read "git did not prove it
    merged" when the ``gh`` lookup timed out on the node — every one of them
    finished work.

    Proof, by object ids only (review m1, PR #4098): every path the branch
    changed since its merge-base with ``<default>``
    (``diff-tree <merge-base> <tip>``) must be IDENTICAL — same blob and mode,
    or absent on both — in ``<default>`` and ``<tip>`` (``diff-tree <tip>
    <default>`` lists none of them). The first version ran ``git merge-tree
    --write-tree``, which executes any repo-configured merge driver
    (``merge.<name>.driver`` mapped by an attribute): it ran a planted
    command, and a common ``merge=ours``-style driver made an unmerged
    CHANGELOG edit read "merged by content". ``diff-tree`` plumbing with
    ``--name-only`` runs no driver, textconv or external diff.

    Callers pass the tree's CURRENT HEAD as ``tip``: a local commit made
    after the squashed one changes a path main does not have, so it is
    unproven (#3992 author's constraint 2). A later revert on main, or main
    editing a branch-changed path again afterwards, leaves that path
    different: not proven (stricter than a merge would be, on purpose — it
    goes to the judge or the forge lookup). Any error is also "not proven":
    this only ever ADDS a proof, so failing closed leaves the old ranking.
    Local objects only, never a fetch: ``default_tip`` must already be local
    (:func:`git_merged` checks ``_has_commit`` first).
    """
    base = _git_quiet(["merge-base", default_tip, tip], real)
    if base is None or not base.strip():
        return False
    changed = _git_quiet(
        [
            "diff-tree",
            "-r",
            "-z",
            "--no-renames",
            "--no-ext-diff",
            "--name-only",
            base.strip(),
            tip,
        ],
        real,
    )
    differs = _git_quiet(
        ["diff-tree", "-r", "-z", "--no-renames", "--no-ext-diff", "--name-only", tip, default_tip],
        real,
    )
    if changed is None or differs is None:
        return False
    return not ({p for p in changed.split("\0") if p} & {p for p in differs.split("\0") if p})


def _git_quiet(args: Sequence[str], real: str) -> str | None:
    """:func:`_git` with repo-configured programs switched off (review m1, PR #4098).

    ``core.fsmonitor`` and hooks are the configured executables a read-only
    probe could otherwise start; the object-store reads that use this need
    neither.
    """
    return _git(["-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", *args], real)


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
    #: Labels of the options the operator did NOT pick, in the questions this
    #: answer kept. A judge's ``ask=`` citation never verifies a path only
    #: these reach (review R2-m6, :func:`ask_citation_covers`).
    unpicked: str = ""
    #: The picked labels plus any typed answer: a path these name is the
    #: operator's pick even when an unpicked option names it too.
    picked: str = ""
    #: The part of ``approved`` that only a TYPED answer to a multi-target
    #: question approved (:func:`_typed_set_question`), one question per
    #: line. ``DeleteCase.evidence_basis`` settles a path reached ONLY
    #: through these on the answer alone when nothing unrecoverable can be
    #: lost (review B2 defence in depth, PR #4098); otherwise the judge reads
    #: the operator's words.
    typed_set: str = ""

    @property
    def firm(self) -> str:
        """``approved`` minus the typed whole-set questions: picks and exact-path answers."""
        if not self.typed_set:
            return self.approved
        typed = set(self.typed_set.split("\n"))
        return "\n".join(line for line in self.approved.split("\n") if line not in typed)


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
    typed_set: list[str] = []
    picked: list[str] = []
    unpicked: list[str] = []
    for q in questions:
        if q.get("secret"):
            continue
        chosen = _kept_answer(q, answers)
        if chosen is None:
            # Deterministic veto (review R1-MINOR-3): a declining answer is
            # never consent, whatever a judge that read the history concludes.
            continue
        opts = [o for o in (q.get("options") or []) if isinstance(o, Mapping)]
        labels = [str(o.get("label") or "") for o in opts]
        # The consent rendering, which the refusal side also reads for every
        # question this answer kept (``unanswered_asks``, review R2-M3).
        parts.append(f"{_ask_question_text(q)} -> operator answered: {', '.join(chosen)}")
        got = _approved_text(q, opts, labels, chosen)
        approved.extend(got)
        typed_set.extend(_typed_set_question(q, labels, chosen, got))
        picked.extend(chosen)
        unpicked.extend(lbl for lbl in labels if lbl not in chosen)
    if not parts:
        return None
    return OperatorWords(
        label="operator, answer to ask",
        text="\n".join(parts),
        ask_id=ask_id,
        session_id=session_id,
        approved="\n".join(approved),
        unpicked="\n".join(unpicked),
        picked="\n".join(picked),
        typed_set="\n".join(typed_set),
    )


def _kept_answer(q: Mapping[str, Any], answers: Mapping[str, Any]) -> list[str] | None:
    """The operator's answer to ``q`` when it can be consent, else None.

    None for a secret question, no answer, or any declining answer. The ONE
    test both :func:`_ask_words` (consent) and :func:`unanswered_asks`
    (refusal) apply per question, so the two never disagree on which
    questions a trusted answer kept.
    """
    if q.get("secret"):
        return None
    chosen = [str(v) for v in (answers.get(str(q.get("id") or "")) or [])]
    labels = [str(o.get("label") or "") for o in q.get("options") or [] if isinstance(o, Mapping)]
    if not chosen or any(_answer_declines(v, labels) for v in chosen):
        return None
    return chosen


def _ask_question_text(q: Mapping[str, Any], *, descriptions: bool = False) -> str:
    """One question as text: the question, its option labels, optionally descriptions.

    The ONE rendering consent (:func:`_ask_words`) and refusal
    (:func:`unanswered_asks`) read. Consent reads question + labels: an
    option's description names nothing (live ask a-1d54, review R2-M1). The
    refusal side adds descriptions only for questions with no trusted,
    non-declining answer (review m4/QA Q2 vs R2-M3). No closing bracket after
    the last option: ``_PATHISH`` would keep it on a trailing glob
    (``~/osworld/runs*]``), which then matches nothing.

    The suggested terminal command (``AskQuestion.command``, PR #3926; stored
    on the question only, never per option) rides with ``descriptions``: it
    is the agent's proposal, like a description, not words the operator
    approved. So it names a path on the REFUSAL side (``rm -rf ~/bench/rigs``
    suggested, then the ask timed out: refuse, not LOW), but a trusted yes
    never lifts a path named only there — consent reads question + labels
    (review R2-M3), and a kept question's refusal text matches it, so such a
    path keeps its base rank, exactly as a description-only path does.
    """
    options = "; ".join(
        (
            f"{o.get('label') or ''} ({o.get('description')})"
            if descriptions and o.get("description")
            else str(o.get("label") or "")
        )
        for o in q.get("options") or []
        if isinstance(o, Mapping)
    )
    command = q.get("command")
    suggested = (
        f" | suggested command: {command}" if descriptions and isinstance(command, str) else ""
    )
    # Before the options: the text keeps ending on the last option (see above).
    return f"Q: {q.get('question') or ''}{suggested} | options: {options}"


def ask_citation_covers(
    words: OperatorWords,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> bool:
    """May a judge's ``ask=<id>`` citation of ``words`` verify deleting ``path``?

    The answer's text must name the path (question or labels, never a
    description), AND the path must not be one that only an option the
    operator did NOT pick reaches. WHY (review R2-m6, PR #3966): with
    "Delete ~/A" picked and "Delete ~/B" not, a judge citing the ask for
    ``~/B`` passed the verifier, which read every label. The question still
    counts ("Delete ~/A and ~/B?" answered "Both"), unless an unpicked option
    claims the path and no pick or typed answer names it.
    """
    protected = tuple(protected)
    if not names_path(words.text, path, home=home, cwd=cwd, protected=protected):
        return False
    if not words.unpicked:
        return True
    reach = ask_reach(words.unpicked, path, home=home, cwd=cwd, protected=protected)
    if reach is None or reach[0] not in ("exact", "under"):
        return True
    return names_path(words.picked, path, home=home, cwd=cwd, protected=protected)


def ask_answer_covers(
    words: OperatorWords,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> bool:
    """Would an ``ask=`` citation of ``words`` verify ``path`` or a dir above it?

    The rank side of :func:`ask_citation_covers`: a path inside a tree the
    operator's answer covers (``~/m/.worktrees/x/node_modules`` under an
    answered "Delete all five" naming ``~/m/.worktrees/{x,…}``) is not lifted
    out of its base rank by that ask. Only ancestors below ``$HOME`` count,
    so an answer that names ``~`` covers nothing beneath it.
    """
    protected = tuple(protected)
    home_real = os.path.realpath(home)
    for spelled in dict.fromkeys([os.path.normpath(path), os.path.realpath(path)]):
        current = spelled
        while True:
            if ask_citation_covers(words, current, home=home, cwd=cwd, protected=protected):
                return True
            parent = os.path.dirname(current)
            if parent == current or parent in (home, home_real) or len(parent) <= len(home_real):
                break
            current = parent
    return False


#: Trailing words a typed whole-set first clause may carry that only restate
#: the approval: "Remove all as recommended", "Remove all I give approval".
_WHOLE_SET_TAIL_RE = re.compile(
    r"(?i)(?:\s+(?:as\s+(?:recommended|suggested|proposed|planned)"
    r"|i\s+give\s+(?:my\s+)?approval|i\s+approve|approved|please|now))*"
)
#: The ONLY first clauses :func:`_typed_whole_set` accepts: a removal verb over
#: the whole set ("Remove all", "Delete them", "Remove all 34"). A bare "yes"/
#: "ok" is NOT one: "ok, just es" and "Yes. Also check the logs" put the
#: qualification after it (test_classification_accuracy), and a bare yes is
#: settled only by ``_BARE_AFFIRM_RE`` over the WHOLE answer.
_TYPED_WHOLE_SET_RE = re.compile(
    r"(?i)\W*(?:delete|remove|clean\s?up|wipe|discard)\s+"
    r"(?:all(?:\s+(?:\d+|of\s+them|them|of\s+these|of\s+those))?|them(?:\s+all)?"
    r"|everything|both|these|those|it)"
)
#: Narrowing words anywhere in the answer that ``_CARVE_OUT_RE`` does not
#: already list ("ok, just es", "yes, minus fr", "zh-cn stays").
_NARROWING_RE = re.compile(
    r"(?i)\b(?:just|minus|stays?|remains?|apart\s+from|other\s+than|save\s+for|besides"
    r"|aside\s+from|without)\b"
)
#: How the whole-set head may end before a hand-off clause: ", and", ",",
#: ";", ". " or " and " (also "and also"). A bare " - " or "but" is not one.
_HANDOFF_BREAK_RE = re.compile(r"(?i)\s*(?:[,;.]\s*(?:and\s+)?|\s+and\s+)(?:also\s+)?")
#: The ONLY clauses allowed after a whole-set head (review B1, PR #4098): an
#: instruction to tell or involve SOMEONE ELSE, which is the shape of both
#: real answers ("check in with Aida …", "send to Aida …"). Anything else
#: after the head — "es should be kept", "I changed my mind", "then restore
#: es" — is not on this list, so the judge reads it.
_HANDOFF_RE = re.compile(
    r"(?i)(?:check\s+in\s+with|send\s+(?:(?:it|this|that|them)\s+)?to|tell|ping|message"
    r"|notify|loop\s+in|follow\s+up\s+with|report\s+(?:(?:it|this|back)\s+)?to"
    r"|let\s+\w+\s+know|cc)\s+\S"
)
#: Vocabulary that, ANYWHERE in a hand-off clause and in ANY form, sends the
#: answer to the judge (review B2, PR #4098). WHY so broad: the B1 version
#: listed a few exact qualifier words, so "tell Aida the dirty one shouldn't
#: be removed", "… must not be touched", "… is still needed", "… put qa-push
#: back", "… sauf qa-push", "… a mistake to delete", "… is precious" and
#: "… I am reconsidering" all read as a benign hand-off and settled a dirty,
#: unmerged tree with no judge. A hand-off is benign only when it says
#: NOTHING about deleting, keeping or retaining anything; these are matched
#: as word STEMS (``remov\w*`` = remove/removed/removal/removing) so passive
#: and noun forms cannot slip through. Over-matching is the safe direction:
#: it costs one judge call that sees the operator's words verbatim.
#: Deliberately NOT here: "blocked", "require … approval", "guardrails",
#: "guidance", "tuning" — both of Damian's real answers (ask a-798d and its
#: sibling lane) use them, and they are about the guard process, not about
#: which target goes.
_HANDOFF_VETO_RE = re.compile(
    r"(?i)\b(?:"
    # deletion, in any form
    r"remov\w*|delet\w*|wip(?:e|ed|es|ing)\b|eras\w*|discard\w*|purg\w*|destroy\w*"
    r"|trash\w*|nuk\w*|rm|prun\w*|drop\w*|touch\w*|clean\w*|los(?:e|es|ing|t)\b"
    r"|get\s+rid|throw\w*"
    # keeping / retention / recovery
    r"|keep\w*|kept|hold\w*|held|sav(?:e|ed|es|ing)\b|retain\w*|preserv\w*|protect\w*"
    r"|spar(?:e|ed|ing)\b|need\w*|necessar\w*|back|restor\w*|recover\w*|undo\w*|undid"
    r"|revert\w*|mistake\w*|wrong|precious|important|valuable|matters?|reconsider\w*"
    r"|rethink\w*|still|stay\w*|remain\w*|leav(?:e|es|ing)\b|left|untouched|alone"
    r"|ignor\w*|skip\w*|spare"
    # stopping / retraction / ordering
    r"|wait\w*|stop\w*|cancel\w*|paus\w*|halt\w*|abort\w*|retract\w*|instead|actually"
    r"|scratch|mind|first|before|after|until|till|once|then|later|yet"
    # exclusion / condition
    r"|except\w*|exclud\w*|sauf|but|only|unless|if|apart|besides|without|other\s+than"
    # refusal
    r"|no|nope|never|nothing|none|neither|nor"
    r")\b"
)
#: A reference to a TARGET inside a hand-off clause (review B2): a path, a
#: name-like token (``qa-push``, ``es_2``, ``v1.2``), a state adjective that
#: picks trees out ("the dirty one"), or a noun for the things being removed
#: ("her worktree", "that one", "this one"). The question's own basenames
#: are caught over the whole answer by :func:`_names_question_path`; this
#: catches the names and descriptions the question never spelled. A hand-off
#: that has to point at a target is not "tell someone else" any more.
_HANDOFF_TARGET_RE = re.compile(
    r"(?i)(?:[~/]|\b\w+[-_./]\w|\b\w*\d\w*\b"
    r"|\b(?:one|ones|worktrees?|trees?|rigs?|branch\w*|folders?|director\w*|dirs?"
    r"|repos?|repositor\w*|checkouts?|clones?|copy|copies|files?|work|changes|commits?"
    r"|data|stuff|dirty|clean|merged|unmerged|uncommitted|wip|detached)\b)"
)
#: A negation inside a hand-off clause counts only when it can bend the
#: delete: "don't remove", "not es", "not the dirty one", a bare "not". The
#: real answer's "should not require human approval" / "shouldn't be blocked"
#: / "doesn't get blocked" is a statement about the guard. Review B2: the B1
#: version accepted any "<modal> not be …", so "shouldn't be removed" was
#: harmless; now only the guard-process verbs follow the negation (an
#: optional "be"/"get" then blocked/required/asked/prompted/gated), and
#: the vocabulary veto above still runs over the whole clause.
_NEGATION_RE = re.compile(r"(?i)\b(?:not\b|\w*n['\u2019]t\b)")
_HARMLESS_NEGATION_RE = re.compile(
    r"(?i)\b(?:should|would|could|does|did|is|are|was|were|will|must|can)"
    r"(?:\s+not|n['\u2019]t)\s+"
    r"(?:(?:be|get)\s+(?:blocked|required|asked|prompted|gated|flagged)"
    r"|(?:require|block|ask|prompt|gate)\b)"
)


def _names_question_path(answer: str, question: str) -> bool:
    """Does the answer spell any path, or any path basename, the question lists?

    Any path-ish word in the answer counts (the operator pointed at
    something specific); so does a question path's basename as a whole word
    ("Remove all but es" for ``…/.worktrees/es``).
    """
    if _PATHISH.search(answer):
        return True
    for m in _PATHISH.finditer(question):
        base = os.path.basename(m.group(1).rstrip("/.:?!"))
        if base and re.search(rf"(?i)(?<![\w.-]){re.escape(base)}(?![\w-])", answer):
            return True
    return False


def _typed_whole_set(answer: str, question: str = "") -> bool:
    """Does a typed answer approve the WHOLE question? An allow-shape, not a deny-list.

    WHY (node-2, 2026-11-15, ask a-798d): Damian typed "Remove all I give
    approval, and check in with Aida to make sure that we're pushing along the
    tuning to guidance, these requests should not require human approval and
    shouldn't be blocked by guardrails in the future" (and, on another lane,
    "Remove all as recommended, and send to Aida ..."). Only a BARE yes
    (``_BARE_AFFIRM_RE``) used to settle a typed answer, so both approved
    nothing deterministically and the judge refused.

    WHY an allow-shape (review B1, PR #4098): the first version vetoed known
    carve-out words and ignored the rest, so "Remove all, save es", "Remove
    all, es should be kept" and "Remove all, I changed my mind" approved the
    whole question — and that settled a dirty, unmerged tree with no judge.
    Now the answer must be EXACTLY:

    1. a removal verb over the whole set (``_TYPED_WHOLE_SET_RE``) plus only
       approval-restating words (``_WHOLE_SET_TAIL_RE``), then either
    2. nothing, or a clause break (``_HANDOFF_BREAK_RE``) and a hand-off to
       someone else (``_HANDOFF_RE``: "check in with", "send to", "tell" …),
       whose text carries no deletion/keep/retention/exclusion vocabulary in
       any form (``_HANDOFF_VETO_RE``, review B2), no reference to a target
       (``_HANDOFF_TARGET_RE``: a path, a name-like token, "the dirty one",
       "her worktree") and no negation except a statement about the guard
       process (``_HARMLESS_NEGATION_RE``: "should not require approval",
       "shouldn't be blocked");
    3. and nowhere names a path, or a question path's basename.

    Anything else returns False: the answer still reaches the judge as the
    operator's words (``OperatorWords.text``), so a missed approval costs one
    judge call, never a wrong delete.
    """
    if _DECLINE_RE.search(answer) or _CARVE_OUT_RE.search(answer):
        return False
    if _NARROWING_RE.search(answer) or _names_question_path(answer, question):
        return False
    m = _TYPED_WHOLE_SET_RE.match(answer)
    if m is None or not m.group(0).strip():
        return False
    # The restating words start where the whole-set words end (the head's
    # trailing ``\W*`` would swallow the clause break the hand-off needs).
    rest = answer[len(m.group(0).rstrip()) :]
    tail = _WHOLE_SET_TAIL_RE.match(rest)
    rest = rest[tail.end() if tail else 0 :]
    if re.fullmatch(r"[\s.!]*", rest):
        return True
    brk = _HANDOFF_BREAK_RE.match(rest)
    if brk is None:
        return False
    clause = rest[brk.end() :]
    if not _HANDOFF_RE.match(clause):
        return False
    # The WHOLE clause is vetted, recipient included: the hand-off verb is
    # only the shape, and whatever follows it ("tell Aida the dirty one is
    # precious") is where an exclusion would hide (review B2).
    if _HANDOFF_VETO_RE.search(clause) or _HANDOFF_TARGET_RE.search(clause):
        return False
    # Every negation must sit inside a harmless modal statement.
    harmless = [h.span() for h in _HARMLESS_NEGATION_RE.finditer(clause)]
    return all(
        any(a <= n.start() and n.end() <= b for a, b in harmless)
        for n in _NEGATION_RE.finditer(clause)
    )


def _typed_set_question(
    q: Mapping[str, Any], labels: Sequence[str], chosen: Sequence[str], approved: Sequence[str]
) -> list[str]:
    """``[question]`` when only a TYPED answer approved this question, else ``[]``.

    WHY (review B2 defence in depth, manager decision on PR #4098): a typed
    answer is free text, and however carefully :func:`_typed_whole_set`
    parses it, a phrasing nobody listed can still hide "but not that one".
    So a question approved by typing ("Remove all as recommended, and send
    to Aida …", even a bare "yes") is remembered separately, and
    ``DeleteCase.evidence_basis`` lets it settle a target alone only when
    removing that target cannot lose unrecoverable work. A PICKED option
    ("Remove all", "Delete ~/x") is what the operator chose from what he
    saw, so it is not in here and keeps settling as before (that includes
    the exact-path asks a-822f/a-72ff, which settled a dirty tree).
    """
    question = str(q.get("question") or "")
    typed = [v for v in chosen if v not in labels]
    picked = [v for v in chosen if v in labels]
    if typed and not picked and question and question in approved:
        return [question]
    return []


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
    if typed and not all(
        _BARE_AFFIRM_RE.fullmatch(v) or _typed_whole_set(v, str(q.get("question") or ""))
        for v in typed
    ):
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


@dataclass(frozen=True)
class UnansweredAsk:
    """An ask in the family whose end state is NOT consent for what it names.

    Despite the name (kept for its callers) this is every ask that names
    paths and is not a trusted operator answer approving them: never answered
    (open, timed out, expired, withdrawn, dismissed), declined, or answered in
    a way :func:`_ask_words` refuses (the agent's own ``answered_in_chat``, a
    relay or bridge row, a declining pick such as "Keep everything").
    """

    ask_id: str
    session_id: str
    #: The fold's status (``timed_out``, ``answered``, ``declined``, ...).
    status: str
    #: Question text and option labels, plus descriptions for every question
    #: the operator's trusted answer did not keep: wider than consent's naming
    #: bar there on purpose, since that text is only ever matched to REFUSE.
    #: A kept question reads exactly as consent reads it (review R2-M3).
    text: str
    #: Why it is not consent, for the judge fact and the refusal text.
    label: str = ""
    #: What a trusted operator answer FIRMLY approved without a judge
    #: (``OperatorWords.firm``: picks and exact-path answers, never a typed
    #: whole-set approval, which only ``check._ask_approval`` may read); a
    #: path it reaches is consented, not blocked (:func:`ask_consents`).
    #: Empty for every non-answer.
    approved: str = ""
    #: ``created_at`` (ms) of the ask; :data:`ASK_ESCALATION_MAX_AGE_MS` bounds it.
    created_at: int = 0
    #: True when the operator never answered (open/timed out/expired/...),
    #: False when an answer exists but does not approve the path.
    never_answered: bool = True
    #: The operator's trusted, non-declining answer (:func:`_ask_words`), or
    #: None. A path that answer's text reaches but did not approve
    #: deterministically goes to the judge, never to the refusal (review
    #: R2-M3): only an ask with no trusted yes may refuse outright.
    answer: OperatorWords | None = None
    #: Answered, non-declining, but on an untrusted surface (``remote``,
    #: ``chat``, ``unverified``): not consent, not a refusal (:func:`_untrusted_yes`).
    untrusted_yes: bool = False
    #: Effective answer time is the revision time, not the first answer time.
    answered_at: int = 0
    #: An expired folded record can still contain a late answer or a decline.
    has_answer: bool = False
    expires_at: int = 0


#: Statuses that mean "the operator never answered".
_UNANSWERED_LABEL = {
    "timed_out": "TIMED OUT with no answer",
    "expired": "expired with no answer",
    "withdrawn": "withdrawn by the agent, never answered",
    "dismissed": "dismissed by the user without an answer",
    "open": "still open, not yet answered",
}

#: An ask stops lifting paths out of LOW once it is this old (review R1-M2,
#: PR #3966). WHY 7 days: the longest ask timeout is 24 h
#: (``asks.policy.MAX_TIMEOUT_S``) and a cleanup that follows an ask runs
#: within hours of its notice (incident a-20fc: one hour); a week covers a
#: session resumed after a weekend with margin, while an ask about some path
#: last Tuesday no longer stands between a session family and ordinary work
#: under that path forever. Only escalation is bounded: the not-consent FACT
#: still reaches the judge for any target that ranks HIGH on its own.
ASK_ESCALATION_MAX_AGE_MS = 7 * 24 * 3600 * 1000


def unanswered_label(status: str) -> str:
    return _UNANSWERED_LABEL.get(status, "never answered")


#: The not-consent label for an answer stamped ``unverified`` (incident
#: 2026-11-16, ask a-48e3, session 439818272d84). WHY its own wording: the
#: operator DID answer, on the desktop card, but the runtime had been started
#: by the wake supervisor, so the desktop backend held no operator capability
#: for it (that in-memory table in ``harness/approval.py`` is held only by the
#: process that started the runtime) and the runtime stamped the
#: local, proof-less connection ``unverified`` (``asks.surfaces``). The old
#: generic label made the agent tell the operator he had "no answer approving
#: it" after he had answered. The fail-closed itself is deliberate and stays:
#: the model's own ``bash`` can dial the socket with the 0600 control key and
#: is indistinguishable from a proof-less window, and an unforgeable desktop
#: credential (Electron fuse hardening + code-signature check) is an operator
#: decision not yet taken. :func:`lapsed_refusal_reason` keys its recovery
#: text on this exact string.
UNPROVEN_SURFACE_LABEL = (
    "answered from a surface that could not prove it is the operator's "
    "(it did not start this session's runtime)"
)


def _not_consent_label(record: Mapping[str, Any], status: str) -> str:
    """Why a folded ask that HAS an end state still is not consent."""
    if status in _UNANSWERED_LABEL:
        return _UNANSWERED_LABEL[status]
    if status == "declined":
        return "declined"
    surface = str((record.get("answered_by") or {}).get("surface") or "")
    if status in ("answered", "late") and surface == "unverified":
        return UNPROVEN_SURFACE_LABEL
    if status in ("answered", "late") and surface not in TRUSTED_ASK_SURFACES:
        # ``answered_in_chat`` is written by the AGENT (surface ``chat``); a
        # relay row is another session's carriage. Neither is the operator.
        return f"answered on the '{surface or 'unknown'}' surface, not by the operator"
    return "answered without approving it"


def unanswered_asks(
    session_id: str,
    *,
    state_root: str | os.PathLike[str] | None = None,
    now_ms: int | None = None,
) -> list[UnansweredAsk]:
    """Asks in ``session_id``'s family that the store does NOT prove as consent.

    WHY (incident 2026-11-14, session 1faad822298d, ask a-20fc): an ask naming
    rig and benchmark paths timed out unanswered, the timeout notice said
    "proceed … use your recommended option(s)", and the session deleted them.
    :func:`verified_ask` already refused that ask as consent, but nothing said
    so. This lists those asks so the checkpoint can (a) tell the judge each one
    is NOT consent and (b) refuse deterministically when such an ask is the
    only thing pointing at a path (``check.DeleteCase.lapsed_only``).

    Review R1-M1 (PR #3966): listing only never-answered STATUSES let the
    agent's own ``ask_withdraw answered_in_chat`` after the deadline (folded
    ``late``, surface ``chat``), a relay answer, a bare ``declined`` and a
    trusted "Keep everything" pick drop the ask from this list, so the path
    fell back to LOW. Now: blocks = asks naming the path − asks granting
    consent for it. Every ask is listed unless it has no visible question;
    a trusted answer carries its ``approved`` text, and :func:`ask_consents`
    lifts only the paths that text reaches. Never a source of consent.
    """
    try:
        from local_operator.asks import store
        from local_operator.paths import config_dir

        root = Path(state_root) if state_root is not None else config_dir()
        lineage = session_family(session_id, state_root=root)
        now = now_ms if now_ms is not None else store.now_ms()
    except Exception:  # noqa: BLE001 — unreadable: nothing extra to report
        logger.debug("guardrail: unanswered asks unreadable", exc_info=True)
        return []
    out: list[UnansweredAsk] = []
    for sid in lineage or [session_id]:
        try:
            directory = store.session_dir(root, sid)
            if not directory.is_dir():
                continue
            records = store.fold(store.read_events(directory), now)
        except Exception:  # noqa: BLE001
            continue
        for record in records:
            status = str(record.get("status") or "")
            questions = [q for q in record.get("questions") or [] if isinstance(q, Mapping)]
            words = _ask_words(record, sid)
            answers = record.get("answers") or {}
            # One parser for consent and refusal (review R2-M3): a question
            # the operator's trusted answer kept is read exactly as consent
            # reads it (question + labels), so a path named only in its
            # description is neither approved nor refused (main's rank, the
            # same as consent sees it). Every other question keeps its
            # descriptions (review m4/QA Q2): with no trusted yes, a path
            # named only there must still refuse.
            parts = [
                _ask_question_text(
                    q, descriptions=words is None or _kept_answer(q, answers) is None
                )
                for q in questions
                if not q.get("secret")
            ]
            if not parts:
                continue
            out.append(
                UnansweredAsk(
                    ask_id=str(record.get("ask_id") or ""),
                    session_id=sid,
                    status=status,
                    text="\n".join(parts),
                    label=(
                        "answered by the operator with no picked label approving this path"
                        if words is not None
                        else _not_consent_label(record, status)
                    ),
                    # FIRM only (review round 3 M2, PR #4098): a typed
                    # whole-set approval is consent only where
                    # ``check._ask_approval`` proves the target recoverable,
                    # so ``ask_consents`` must never read it.
                    approved=words.firm if words is not None else "",
                    answer=words,
                    created_at=int(record.get("created_at") or 0),
                    never_answered=status in _UNANSWERED_LABEL,
                    untrusted_yes=words is None and _untrusted_yes(record, status, questions),
                    answered_at=int(record.get("revised_at") or record.get("answered_at") or 0),
                    has_answer=bool(record.get("answered_at") or record.get("answers")),
                    expires_at=int(record.get("expires_at") or 0),
                )
            )
    return out


def _untrusted_yes(record: Mapping[str, Any], status: str, questions: Sequence[Any]) -> bool:
    """An answer that exists, declines nothing, but came from an untrusted surface.

    For :func:`check.escalate_asked_targets` only: such an answer is never
    consent (field case a-2432: answered from the desktop's remote view of
    node-2, stamped ``remote``), but it is not a refusal either, so it must
    not turn a target the guard PROVED finished into a refusal. Any declining
    answer, any non-answer, and a question-id ask are False.
    """
    if status not in ("answered", "late") or not questions:
        return False
    surface = str((record.get("answered_by") or {}).get("surface") or "")
    if surface in TRUSTED_ASK_SURFACES:
        return False
    answers = record.get("answers") or {}
    return all(
        isinstance(q, Mapping) and not q.get("secret") and _kept_answer(q, answers) is not None
        for q in questions
    )


def ask_consents(
    ask: UnansweredAsk,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> bool:
    """Did the operator's trusted answer to ``ask`` approve ``path`` (equal or under)?

    The approved text is the label(s) the operator picked (or the question, for
    a bare yes to a yes/no ask). A path inside an approved tree is approved
    (``rm -rf ~/osworld/runs-3/raw`` under an approved ``~/osworld/runs*``); a
    path ABOVE it is not — approving ``~/x/build`` never approves ``~/x``.
    """
    if not ask.approved:
        return False
    kind = ask_reach(ask.approved, path, home=home, cwd=cwd, protected=protected)
    return kind is not None and kind[0] in ("exact", "under")


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


def session_family(
    session_id: str, *, state_root: str | os.PathLike[str] | None = None
) -> list[str]:
    """``session_id``, its ancestors (:func:`session_lineage`) and its descendants.

    WHY descendants too (incident 2026-11-14 follow-up): an ask a subagent
    raised and nobody answered still means "the operator has not said yes" to
    the manager that spawned it, and the reverse. Descendants come from the
    harness lineage registry (``read_lineage``: ``parent``/``opener`` rows),
    followed down to a fixpoint; ``origin.json`` files are not scanned (that
    would read every session on disk per call). Bounded at 64 sessions.
    """
    from local_operator.paths import config_dir

    root = Path(state_root) if state_root is not None else config_dir()
    lineage, _ = session_lineage(session_id, state_root=root)
    family = list(lineage or [session_id])
    registry = read_lineage(root)
    below = {session_id}
    grew = True
    while grew and len(family) < 64:
        grew = False
        for child, row in registry.items():
            if child in below or len(family) >= 64:
                continue
            if row.get("parent") in below or row.get("opener") in below:
                below.add(child)
                if child not in family:
                    family.append(child)
                grew = True
    return family


def task_tree_root(session_id: str, *, state_root: str | os.PathLike[str] | None = None) -> str:
    """The top of ``session_id``'s SUBAGENT tree: ``parent`` links only.

    WHY parent-only (case 7, 2026-11-09): sibling lanes under one manager
    (``a13e8dbc573f``'s auditor and settings lane) are one owner, but an
    agent-opened workstream (``opened_by``) is a separate task, so every
    workstream an assistant ever opened never becomes one tree. The registry
    row is read first (``read_lineage``), then ``origin.json``.
    """
    chain = subagent_ancestors(session_id, state_root=state_root)
    return chain[-1] if chain else session_id


def subagent_ancestors(
    session_id: str, *, state_root: str | os.PathLike[str] | None = None
) -> list[str]:
    """``session_id``'s launchers, nearest first: ``parent`` links only, self excluded.

    WHY (field report 2026-11-12, child fbd3094083f4 of a7450d93932d): one
    subagent lineage is one owner's working state, so ``classify._tree_read``
    lets an unresolvable read of an ancestor's or a descendant's pad rank LOW
    (a descendant is a session whose chain contains the caller). That credit
    must come from authoritative metadata, never text:
    the harness-written registry row first (``read_lineage``, first row wins),
    then ``origin.json`` — both guard evidence a session cannot forge
    (``_touch_reason`` ranks a write to either CRITICAL). ``opener`` links are
    NOT followed: an agent-opened workstream is a separate task. A sibling is
    in neither session's chain, so it inherits nothing.
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
    return seen[1:]


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


def ask_reaches(
    text: str,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> bool:
    """Does an ask's ``text`` name ``path``, a path above it, or one inside it?

    A boolean view of :func:`ask_reach`; see there for the rules.
    """
    return ask_reach(text, path, home=home, cwd=cwd, protected=protected) is not None


def ask_reach(
    text: str,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> tuple[str, str] | None:
    """HOW an ask's ``text`` reaches ``path``: ``(kind, named path)`` or ``None``.

    WHY (incident 2026-11-14, ask a-20fc, rows 302/320/365): the session asked
    which rig, build and benchmark trees to delete, got no answer, and removed
    the deep ones anyway — ``~/hyperplane-cpp/build`` and the benchmark raw
    data ranked LOW, so no check ever ran. Asking about a path is the agent's
    own admission that removing it needs the operator; silence cannot turn it
    back into LOW. So this match is deliberately WIDER than consent's
    :func:`names_path`. It lifts or keeps escalation and refusals; the callers
    that subtract consent with it (:func:`ask_consents` on the operator's
    approved labels, :func:`ask_citation_covers` on unpicked options) can at
    most return a target to its base rank, never below it (over-matching on
    the refusal side fails closed):

    - ``("exact", path)``: :func:`names_path` names ``path`` itself, literally
      or by its glob (``~/osworld/runs*`` names ``~/osworld/runs-3``);
    - ``("under", ancestor)``: it names an ancestor of ``path`` below
      ``$HOME`` (same glob and brace-list rules). The caller needs the
      ancestor: under a BROAD named dir (``~/workspace``) the target goes to
      the judge instead of a deterministic refusal (review R1-M2);
    - ``("above", named)``: ``path`` is an ancestor of a named path, or of a
      glob's literal parent directory (``rm -rf ~/osworld`` removes every ``runs*``).

    ``$HOME``, its ancestors and the protected roots are never treated as
    "named" for the under-direction: an ask that merely mentions ``~`` or the
    state root must not make every deep delete in the session refusable.

    Case (review R1-m1): macOS volumes are case-insensitive by default, so
    ``rm -rf ~/Workspace/Harness-Cost-Bench`` deletes the tree an ask spelled
    ``~/workspace/harness-cost-bench``. On Darwin every spelling is compared
    casefolded. On a case-SENSITIVE Darwin volume that can over-match two
    distinct paths; that may lift only the escalation it added, never take a
    target below its base rank.
    """
    if not text or not path:
        return None
    protected = tuple(protected)
    spellings = list(dict.fromkeys([os.path.realpath(path), os.path.normpath(path)]))
    if sys.platform == "darwin":
        text = text.casefold()
        home = home.casefold()
        cwd = cwd.casefold() if cwd else cwd
        protected = tuple(p.casefold() for p in protected)
        spellings = list(dict.fromkeys(s.casefold() for s in spellings))
    for spelled in spellings:
        if names_path(text, spelled, home=home, cwd=cwd, protected=protected):
            return ("exact", spelled)
    home_n = os.path.normpath(home)
    stops = {os.path.normpath(p) for p in protected} | {home_n, "/"}
    for spelled in spellings:
        anc = os.path.dirname(spelled)
        while anc and anc not in stops and not home_n.startswith(anc.rstrip("/") + "/"):
            if names_path(text, anc, home=home, cwd=cwd, protected=protected):
                return ("under", anc)
            parent = os.path.dirname(anc)
            if parent == anc:
                break
            anc = parent
    for word, _ in _path_readings(text):
        expanded = _expand_word(word, home, cwd) if word else None
        if not expanded:
            continue
        wild = _GLOBISH.search(expanded)
        named = expanded if wild is None else expanded[: wild.start()].rsplit("/", 1)[0]
        named = os.path.normpath(named or "/")
        if any(named == s or named.startswith(s.rstrip("/") + "/") for s in spellings):
            return ("above", named)
    return None


#: A named dir with more entries than this is "broad" (review R1-M2): an ask
#: mentioning it ("~/workspace or ~/Documents?") is too wide a net to refuse
#: everything under it without a judge. 1000 entries is far above a project
#: tree's top level and the incident's rig/bench dirs (tens of entries) yet
#: below a home-wide workspace or Downloads folder; the count stops at
#: ``_BROAD_ASK_DIR_ENTRIES + 1`` so the check stays bounded.
_BROAD_ASK_DIR_ENTRIES = 1000


def broad_ask_dir(named: str, *, home: str) -> bool:
    """Is ``named`` too broad for an ask about it to refuse paths UNDER it outright?

    Broad: a direct child of ``$HOME`` (``~/workspace``, ``~/Documents``) or
    any directory holding more than :data:`_BROAD_ASK_DIR_ENTRIES` entries.
    Paths under a broad named dir still rank HIGH and reach the judge with the
    not-consent fact; only the exact named path (or its glob matches) and
    paths above it keep the deterministic refusal.
    """
    norm = os.path.normpath(named)
    homes = {os.path.normpath(home), os.path.realpath(home)}
    if sys.platform == "darwin":
        norm = norm.casefold()
        homes = {h.casefold() for h in homes}
    if os.path.dirname(norm) in homes:
        return True
    try:
        with os.scandir(named) as it:
            for n, _ in enumerate(it, 1):
                if n > _BROAD_ASK_DIR_ENTRIES:
                    return True
    except OSError:
        return False
    return False


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
