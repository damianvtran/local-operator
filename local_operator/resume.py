"""Which previous session ``--resume`` reopens.

A module of its own, and a deliberately tiny one: it imports nothing but
``pathlib``. The CLI has to resolve ``--resume`` before it starts anything (a
typo must be one line on stderr, not a full-screen app that launches, paints and
tears down to report it), and the CLI's startup path is guarded by tests that
FAIL if importing it drags in the engine, the providers, or even ``asyncio``.
Putting this policy in ``session_factory`` — the obvious home — is what broke
that guard: the sentinel alone pulled ``local_operator.harness`` and asyncio onto
every ``local-operator --help``.

Resuming is a filesystem question ("which transcript directory"), so nothing
here needs the engine. ``session_factory`` imports these same functions for the
transcript-directory decision, so the rule has one definition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

#: ``--resume`` with no id. A sentinel rather than a second boolean flag so the
#: whole "which session" decision stays ONE value threaded through one parameter.
RESUME_LATEST = "@latest"

#: The file whose presence makes a directory a resumable session. Also what the
#: recency ordering is read from: a directory's own mtime moves for reasons that
#: are not turns (retention sweeps touch it), so it is not the clock to use.
TRANSCRIPT_NAME = "transcript.jsonl"

#: How much of the opening message a session name may keep. Long enough to tell
#: two days' work apart, short enough that a column of them still scans.
NAME_MAX_CHARS = 64

#: Bytes of a transcript the name scan will read before giving up. A name is a
#: convenience; a pathological first line (a pasted file, a base64 image) must
#: not turn opening the picker into reading megabytes off disk for every row.
NAME_SCAN_BYTES = 64_000


class ResumeNotFound(Exception):
    """``--resume`` named a session that is not on disk (or none exist)."""


def resume_dir(config_dir: Path, requested: str) -> Path:
    """The session directory ``--resume`` names, or raise :class:`ResumeNotFound`.

    Resuming is deliberately CONFINED to ``sessions/``: an agent directory is
    that agent's own long-lived history, reached with ``--agent``/``--train``, and
    letting an id select one would silently append a throwaway session's turns
    onto it.

    Existence is checked HERE rather than left to the transcript reader, because
    a typo'd id would otherwise create an empty directory and start a session
    that looks resumed and has no history — the one failure a resume must never
    have.
    """
    sessions = config_dir / "sessions"
    if requested == RESUME_LATEST:
        candidates = [path for path in sessions.glob("*") if (path / TRANSCRIPT_NAME).is_file()]
        if not candidates:
            raise ResumeNotFound("no previous session to resume")

        def newest(path: Path) -> float:
            try:
                return (path / TRANSCRIPT_NAME).stat().st_mtime
            except OSError:
                # A directory that vanished mid-scan (retention sweeps run
                # concurrently) or one whose transcript is unreadable sorts
                # oldest rather than taking down the resolver on the way to the
                # TUI.
                return 0.0

        return max(candidates, key=newest)

    # A session id must be ONE path component and nothing else. Enumerating the
    # ways to escape (`/`, `\`, `..`, and on Windows the drive-relative `C:x`
    # form) is a list that is never finished; asking the path library whether the
    # string survives as its own basename is the same question asked once. The
    # empty/dot cases are named because `Path("").name` is `""`, which would pass
    # a bare equality check.
    if requested in ("", ".", "..") or Path(requested).name != requested:
        raise ResumeNotFound(f"not a session id: {requested!r}")
    candidate = sessions / requested
    try:
        present = (candidate / TRANSCRIPT_NAME).is_file()
    except OSError:
        # Same race the `@latest` scan guards, on the path a user reaches by
        # typing an id: a retention sweep unlinking the directory, or a
        # permission/ENAMETOOLONG error from the stat. "That session is not
        # there" is the honest answer, and it is what the caller already knows
        # how to report — a bare OSError here is a traceback on the way to the
        # TUI instead.
        present = False
    if not present:
        raise ResumeNotFound(f"no session {requested!r} to resume")
    return candidate


def resolve_resume_id(config_dir: Path, requested: str) -> str:
    """Validate ``--resume`` up front and return the CONCRETE session id.

    Returning the resolved id (never the ``@latest`` sentinel) means the session
    factory sees a real directory name, and the resume command the app prints on
    exit names the same id the user could pass back in.
    """
    return resume_dir(config_dir, requested).name


def recent_sessions(config_dir: Path, limit: int = 10) -> list[tuple[str, float]]:
    """``(id, mtime)`` for resumable sessions, newest first.

    The mtime is RETURNED rather than used and dropped: the sort already reads
    it, and it is the one fact that makes a list of 12-hex ids pickable instead
    of a wall of hashes.

    Best-effort: a directory that vanishes mid-scan (retention sweeps run
    concurrently) is skipped rather than raising out of an error path whose whole
    job is to be helpful.
    """
    rows: list[tuple[str, float]] = []
    for path in (config_dir / "sessions").glob("*"):
        try:
            rows.append((path.name, (path / TRANSCRIPT_NAME).stat().st_mtime))
        except OSError:
            continue
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:limit]


def format_age(seconds: float) -> str:
    """A coarse "how long ago" for the recovery list: ``2h ago``, ``3d ago``.

    Coarse on purpose — the list exists to let someone recognise WHICH session,
    and a timestamp to the second is harder to scan than a rough age.
    """
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


class SessionRow(NamedTuple):
    """One pickable conversation: what it was about, when, and its id.

    The id alone is what the recovery list used to offer, and a column of
    12-hex strings is not something anyone recognises their own work in. The
    name is the part a human picks by; the id is what the machine resumes.
    """

    id: str
    mtime: float
    name: str


def session_name(session_dir: Path, *, max_chars: int = NAME_MAX_CHARS) -> str:
    """A conversation's display name: its opening user message, on one line.

    Read from the TRANSCRIPT rather than from a stored title because there is
    no stored title: ``ConversationName`` lives in memory for the life of a
    session and is never journalled, so the only per-session name on disk is
    what the user actually typed first. That turns out to be the better name
    anyway — it is the thing the user remembers about the session.

    Deliberately tolerant. This runs over every session directory to paint a
    picker, so a transcript that is truncated, half-written by a session still
    running, or corrupt yields ``""`` and a nameless row rather than taking
    the picker down. The scan also stops at the first user message and at
    :data:`NAME_SCAN_BYTES`, so it costs one short read per session instead of
    a full parse of a file that can be hundreds of kilobytes.
    """
    transcript = session_dir / TRANSCRIPT_NAME
    scanned = 0
    try:
        with transcript.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                scanned += len(line)
                if scanned > NAME_SCAN_BYTES:
                    return ""
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    # A partial final line is normal for a session that is
                    # still running: the writer appends, we may read mid-write.
                    continue
                if not isinstance(entry, dict) or entry.get("type") != "message":
                    continue
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    continue
                # ``role`` is matched EXACTLY: a tool result is also a
                # four-character role and carries the tool's output, which
                # would name the conversation after a directory listing.
                if payload.get("role") != "user":
                    continue
                text = _first_text(payload.get("content"))
                if text:
                    return _condense(text, max_chars)
    except OSError:
        return ""
    return ""


def _first_text(content: object) -> str:
    """The first text part of a persisted message's content list."""
    if not isinstance(content, list):
        return ""
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _condense(text: str, max_chars: int) -> str:
    """One line, no runs of whitespace, ellipsised at ``max_chars``.

    A prompt is usually several lines and often starts with a pasted block;
    the picker has one row per session, so the name has to survive being cut.
    """
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    # Cut on a word boundary when one is close to the limit, so the name ends
    # on a word rather than mid-token.
    cut = flat[: max_chars - 1]
    spaced = cut.rsplit(" ", 1)[0]
    if len(spaced) >= max_chars - 12:
        cut = spaced
    return cut.rstrip(" ,.;:") + "…"


def recent_session_rows(config_dir: Path, limit: int = 10) -> list[SessionRow]:
    """:class:`SessionRow` per resumable session, newest first.

    Layered over :func:`recent_sessions` rather than replacing it: the CLI's
    recovery listing wants only ``(id, mtime)`` and must stay as cheap as it
    is, while the picker pays one extra short read per row for the name.
    """
    return [
        SessionRow(session_id, mtime, session_name(config_dir / "sessions" / session_id))
        for session_id, mtime in recent_sessions(config_dir, limit)
    ]
