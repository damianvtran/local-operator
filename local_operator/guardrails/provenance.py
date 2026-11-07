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
   realpath AND ``st_dev``/``st_ino``, so a directory deleted and re-made by
   someone else, or swapped for a symlink, no longer matches.
2. **Clean linked worktree.** ``git worktree remove <path>`` without
   ``--force`` deletes nothing git cannot restore: git itself refuses a dirty
   tree or the primary checkout. ``clean_linked_worktree`` proves that ahead
   of time so the guard can rank it LOW.
3. **Trusted words.** Only the operator's direct messages, and their answers
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
import logging
import os
import re
import sqlite3
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

_ASK_ID_RE = re.compile(r"(?<![\w-])(a-[0-9a-f]{4,12})(?![\w-])")
_SESSION_ID_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{12})(?![0-9a-f])")


# -- ledger -------------------------------------------------------------------


def ledger_path(state_root: str | os.PathLike[str] | None = None) -> Path:
    from local_operator.paths import config_dir

    root = Path(state_root) if state_root is not None else config_dir()
    return root / LEDGER_DIRNAME / LEDGER_FILENAME


@dataclass(frozen=True)
class CreationRecord:
    path: str
    dev: int
    ino: int
    created_at: float
    creator: str
    kind: str
    command: str

    def describe(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.created_at))
        return f"created by this session ({self.kind}, {when})"


_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS created ("
    " owner TEXT NOT NULL, path TEXT NOT NULL, dev INTEGER NOT NULL, ino INTEGER NOT NULL,"
    " created_at REAL NOT NULL, creator TEXT NOT NULL, kind TEXT NOT NULL,"
    " command TEXT NOT NULL, PRIMARY KEY (owner, path))"
)


def _connect(path: Path, *, create: bool) -> sqlite3.Connection | None:
    if not create and not path.exists():
        return None
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_DB_TIMEOUT_S)
    conn.execute(_SCHEMA)
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
            st = os.stat(path)
        except OSError:
            continue
        written.append(path)
        for owner in owners:
            rows.append((owner, path, st.st_dev, st.st_ino, now, creator, kind, command[:500]))
    if not rows:
        return []
    conn = _connect(ledger_path(state_root), create=True)
    assert conn is not None
    try:
        with conn:
            conn.executemany("INSERT OR REPLACE INTO created VALUES (?,?,?,?,?,?,?,?)", rows)
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
    inode: a path deleted and re-made by someone else is no longer this
    session's. Any error means "not owned".
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
                "SELECT path, dev, ino, created_at, creator, kind, command FROM created"
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
        if not (real == rec.path or real.startswith(rec.path.rstrip("/") + "/")):
            continue
        try:
            st = os.stat(rec.path)
        except OSError:
            continue
        if (st.st_dev, st.st_ino) != (rec.dev, rec.ino):
            continue
        if best is None or len(rec.path) > len(best.path):
            best = rec
    return best


@dataclass
class PendingCreations:
    """What a bash call may create, snapshotted BEFORE it runs."""

    command: str
    #: ``(kind, candidate path, topmost component that does not exist yet)``.
    exact: list[tuple[str, str, str]] = field(default_factory=list)
    #: ``(parent, pattern, matches that already existed)`` for ``mktemp -d``.
    patterned: list[tuple[str, str, frozenset[str]]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.exact or self.patterned)


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
        if created.path is not None:
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
        for parent, pattern, before in pending.patterned:
            now = _glob.glob(os.path.join(_glob.escape(parent), pattern))
            for match in now:
                if match not in before and os.path.isdir(match):
                    entries.append((os.path.realpath(match), "mktemp"))
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


def _git(args: Sequence[str], cwd: str) -> str | None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    # Read-only probes: no prompts, no index refresh writes, no credential helper.
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0")
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def worktree_state(path: str) -> str:
    """``clean`` / ``dirty`` / ``not-linked`` / ``unknown`` for a worktree path.

    Linked = its ``.git`` is a FILE, its git-dir differs from the common dir,
    and the repository's ``git worktree list`` names it (not as the main
    checkout). Clean = ``git status --porcelain --untracked-files=all`` prints
    nothing; ignored files are not reported, matching what ``git worktree
    remove`` itself refuses on.
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
    if git_dir == common:
        return "not-linked"
    listing = _git(["worktree", "list", "--porcelain"], real)
    if listing is None:
        return "unknown"
    trees = [
        os.path.realpath(ln[len("worktree ") :])
        for ln in listing.splitlines()
        if ln.startswith("worktree ")
    ]
    if real not in trees[1:]:
        return "not-linked"
    status = _git(["status", "--porcelain", "--untracked-files=all"], real)
    if status is None:
        return "unknown"
    return "clean" if not status.strip() else "dirty"


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
    if record is None or str(record.get("status") or "") not in ("answered", "late"):
        return None
    surface = str((record.get("answered_by") or {}).get("surface") or "")
    if surface not in TRUSTED_ASK_SURFACES:
        return None
    questions = [q for q in record.get("questions") or [] if isinstance(q, Mapping)]
    if any(str(q.get("question_id") or "") for q in questions):
        return None
    answers = record.get("answers") or {}
    parts: list[str] = []
    for q in questions:
        if q.get("secret"):
            continue
        chosen = [str(v) for v in (answers.get(str(q.get("id") or "")) or [])]
        if not chosen:
            continue
        options = "; ".join(
            str(o.get("label") or "") for o in (q.get("options") or []) if isinstance(o, Mapping)
        )
        parts.append(
            f"Q: {q.get('question') or ''}"
            + (f" [options: {options}]" if options else "")
            + f" -> operator answered: {', '.join(chosen)}"
        )
    if not parts:
        return None
    return OperatorWords(
        label="operator, answer to ask",
        text="\n".join(parts),
        ask_id=ask_id,
        session_id=session_id,
    )


def _text_of(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    details = getattr(message, "details", None) or {}
    return str(details.get("body") or details.get("text") or "")


def operator_brief(
    messages: Any,
    *,
    session_id: str = "",
    ancestors: Sequence[str] = (),
    cited: Sequence[str] = (),
    state_root: str | os.PathLike[str] | None = None,
) -> OperatorBrief:
    """The trusted operator words in the recent transcript, newest first.

    Trusted: a plain user row with no harness stamp (the operator typed it),
    and an ask answered by a human surface (checked in the store, not the
    transcript row). Everything else is excluded BY STAMP: harness injections
    by ``harness_injected``; peer, hub and Team Bridge rows by their
    ``custom_type``. Excluded peer/bridge/hub rows are counted by kind so the
    judge knows something was left out, but their text never reaches it.

    ``cited`` is extra text that may name an ask id (the call's intent): like a
    peer message, it is a POINTER only — the store decides.
    """
    from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
    from local_operator.harness.rows import is_harness_chrome, is_harness_injection

    brief = OperatorBrief()
    direct = 0
    pointers: list[tuple[str, str, str]] = []  # (ask_id, session, relayed_by)
    seen_asks: set[tuple[str, str]] = set()
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
                        pointers.append((ask_id, session_id, ""))
                    continue
                kind = _custom_kind(str(custom_type))
                if kind is None:
                    continue
                brief.excluded[kind] = brief.excluded.get(kind, 0) + 1
                body = _text_of(message)
                details = getattr(message, "details", None) or {}
                sender = details.get("sender") if isinstance(details, Mapping) else None
                sender = sender if isinstance(sender, Mapping) else {}
                sender_sid = str(sender.get("session_id") or "")
                who = str(sender.get("conversation_name") or sender_sid or kind)
                sessions = [s for s in [sender_sid, *_SESSION_ID_RE.findall(body)] if s]
                for ask_id in _ASK_ID_RE.findall(body):
                    for sid in dict.fromkeys(sessions):
                        pointers.append((ask_id, sid, who))
                continue
            if getattr(message, "role", None) != "user":
                continue
            if is_harness_injection(message):
                continue
            text = getattr(message, "text", "")
            if not isinstance(text, str) or not text.strip() or is_harness_chrome(text):
                continue
            if direct < _MAX_DIRECT:
                brief.words.append(OperatorWords(label="operator, direct", text=text))
                direct += 1
                for ask_id in _ASK_ID_RE.findall(text):
                    for sid in dict.fromkeys([session_id, *_SESSION_ID_RE.findall(text)]):
                        pointers.append((ask_id, sid, ""))
        except Exception:  # noqa: BLE001 — one odd row never breaks the brief
            continue
    for text in cited:
        for ask_id in _ASK_ID_RE.findall(text or ""):
            for sid in dict.fromkeys([session_id, *ancestors, *_SESSION_ID_RE.findall(text)]):
                pointers.append((ask_id, sid, ""))
    asks = 0
    for ask_id, sid, relayed_by in pointers:
        if asks >= _MAX_ASKS or (ask_id, sid) in seen_asks:
            continue
        seen_asks.add((ask_id, sid))
        found = verified_ask(sid, ask_id, state_root=state_root)
        if found is None:
            continue
        label = found.label + (f" (relayed by {relayed_by})" if relayed_by else "")
        if sid != session_id and not relayed_by:
            label += f" (session {sid})"
        brief.words.append(
            OperatorWords(label=label, text=found.text, ask_id=ask_id, session_id=sid)
        )
        asks += 1
    return brief


# -- naming --------------------------------------------------------------------

_GLOBISH = re.compile(r"[*?\[]")
_PATHISH = re.compile(r"[`'\"]?((?:~|/|\.{1,2}/)[^\s`'\",;)]*|[\w.-]+/[^\s`'\",;)]+)")


def _expand_word(word: str, home: str, cwd: str | None) -> str | None:
    word = word.rstrip(".:")
    if word.startswith("~"):
        if word != "~" and not word.startswith("~/"):
            return None
        word = home + word[1:]
    elif not word.startswith("/"):
        if cwd is None:
            return None
        word = os.path.join(cwd, word)
    return os.path.normpath(word)


def names_path(
    text: str,
    path: str,
    *,
    home: str,
    cwd: str | None = None,
    protected: Iterable[str] = (),
) -> bool:
    """Does ``text`` (operator words) name ``path``, literally or by a safe glob?

    Literal: the exact path, its ``~`` form, or ``parent/basename``. A glob
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
    for cand in list(candidates):
        candidates.add("/".join(cand.rstrip("/").split("/")[-2:]))
    protected = {os.path.normpath(p) for p in protected}
    for match in _PATHISH.finditer(text):
        word = match.group(1).rstrip("/.:")
        if not word:
            continue
        if not _GLOBISH.search(word):
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
