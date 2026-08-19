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
import os
import re
from pathlib import Path
from typing import NamedTuple

#: ``--resume`` with no id. A sentinel rather than a second boolean flag so the
#: whole "which session" decision stays ONE value threaded through one parameter.
RESUME_LATEST = "@latest"

#: The file whose presence makes a directory a resumable session. Also what the
#: recency ordering is read from: a directory's own mtime moves for reasons that
#: are not turns (retention sweeps touch it), so it is not the clock to use.
TRANSCRIPT_NAME = "transcript.jsonl"

#: Marks a session directory as machine-started rather than user-started. A
#: subagent's child session is an ephemeral directory under ``sessions/`` with
#: exactly the shape of a real conversation, so nothing on disk told the two
#: apart and the ``/resume`` picker offered every delegated review, design and
#: scout run as if the user had opened it — on one machine 40 of 50 rows.
#:
#: A SIDECAR file rather than a field in the transcript: the picker's whole
#: cost model is one bounded read per row, and a marker inside the JSONL could
#: only be found by parsing it. ``Path.is_file()`` is one stat, and it answers
#: even for a child whose transcript has not been written yet.
ORIGIN_NAME = "origin.json"

#: ``origin`` value for a session a subagent runs. The file is JSON, and the
#: key is a string rather than a bare flag, so a future non-user origin (a
#: scheduled run, a server-side session) is a new value and not a second file.
ORIGIN_SUBAGENT = "subagent"

#: The two openings only the subagent runner can produce, used ONLY by the
#: one-time backfill for directories that predate the marker.
#:
#: ``[role: <name>]`` is built by ``AgentProfile.preamble`` and ``[scout mode:``
#: is a literal constant in the subagent module; both are stamped in FRONT of
#: the caller's prompt, so they can only appear at offset 0 of a child's first
#: user message. Anchored and exact for that reason: the cost of a false
#: positive is hiding one of the user's own conversations, which is the very
#: failure the absence-means-user default exists to avoid, so these match what
#: the machine writes and nothing that merely resembles it.
_ROLE_PREAMBLE = re.compile(r"\[role: [a-z0-9_-]+\]\n")
_SCOUT_PREAMBLE = "[scout mode:"

#: How much of the opening message a session name may keep. Long enough to tell
#: two days' work apart, short enough that a column of them still scans.
NAME_MAX_CHARS = 64

#: The custom-entry type a session journals its title under. Spelled here as
#: well as in ``session/naming.py`` because this module may not import the
#: engine (see the module docstring — the CLI's startup guard fails if it
#: does), and :func:`stored_session_title` scans the raw JSONL rather than
#: replaying it. ``test_the_journalled_title_type_matches_the_writer`` pins the
#: two spellings together, so a rename breaks a test instead of silently
#: returning every session to its opening message.
_TITLE_CUSTOM_TYPE = "conversation_name"

#: Bytes of the transcript the stored-title scan reads at EACH END. Both ends,
#: because the two facts about a title pull in opposite directions:
#:
#: * The title in force is the NEWEST one — a rename appends a fresh row rather
#:   than rewriting the old one — so a rename made an hour into a long session
#:   is only findable near the tail.
#: * The FIRST title is journalled when the session is auto-named, at turn 2,
#:   which is near the head and is pushed further from the tail by every turn
#:   that follows.
#:
#: A tail-only scan therefore missed the title on most real sessions: measured
#: on this store, 145 of 187 transcripts (78%) are larger than this window, so
#: a session named at turn 2 and then worked in for an hour silently reverted
#: to being labelled by its opening message — the exact failure this function
#: exists to fix, and the long sessions it hit hardest are the ones a user is
#: most likely to be hunting for a week later.
#:
#: Reading both ends rather than the whole file is what keeps the cost per
#: picker row bounded on a 6 MB transcript.
#:
#: **The known gap, stated honestly.** A rename made mid-conversation, then
#: buried under a further 128 KB and never renamed again, falls between the two
#: windows. The head still holds the ORIGINAL title, so that session is
#: labelled with the name it was renamed *away from* rather than falling back
#: to the opener. That is a stale name, and a stale name is stated with exactly
#: the confidence of a correct one \u2014 so it is a real cost, not a rounding
#: error, and this comment says so rather than claiming the scan can only ever
#: miss.
#:
#: It is accepted for now because the alternative is reading whole transcripts
#: on the picker's synchronous path (400 ms against 64 ms across a real store),
#: and because the case requires a mid-session rename specifically: an
#: auto-named session has its title at the head, and a session renamed near the
#: end has it in the tail. If it proves to bite, the fix is to journal the
#: title to a sidecar the way ``mark_session_origin`` does \u2014 one stat and one
#: small read, no size dependence at all \u2014 rather than widening this window.
TITLE_SCAN_BYTES = 131_072

#: Matches a journalled title row in raw JSONL, tolerant of whitespace after
#: the colons for the same reason :data:`_FRAGMENT_USER_RE` is: the session
#: writer emits compact JSON, but a transcript written by a fixture or a future
#: exporter is the same document. ``(?:[^"\\]|\\.)*`` steps over escaped quotes
#: so a title containing one is not cut at it.
_TITLE_ROW_RE = re.compile(
    r'"custom_type"\s*:\s*"' + _TITLE_CUSTOM_TYPE + r'".*?"text"\s*:\s*"((?:[^"\\]|\\.)*)"'
)

#: Characters of a transcript the name scan will read before giving up — not
#: bytes: the file is opened in text mode, so this bounds the decoded string,
#: which is what actually occupies memory here. A name is a convenience; a
#: pathological first line (a pasted file, a base64 image) must not turn
#: opening the picker into reading megabytes off disk for every row.
NAME_SCAN_CHARS = 64_000

#: How far into a half-read first line the scan will look for the marker that
#: says the fragment is a user message. The writer emits ``id``/``ts``/``type``
#: before the payload, so the role sits ~110 characters in; a few hundred is
#: slack for a longer id without letting the marker match something deep in a
#: pasted body.
_FRAGMENT_HEAD_CHARS = 400

#: The marker that says a fragment is a user message, and the first COMPLETE
#: JSON string value of a ``text`` key. Both tolerate whitespace around the
#: colon: the session writer emits compact JSON, but a transcript written by
#: anything else (a test fixture, a hand-edited file, a future exporter) is
#: still the same document, and a scan that only matched the compact spelling
#: silently returned no name for it.
#:
#: The closing quote is what makes the text value complete: a value still
#: running when the read window ended cannot match, so a name is never a word
#: cut in half. ``(?:[^"\\]|\\.)*`` steps over escaped quotes rather than
#: stopping at the first one.
_FRAGMENT_USER_RE = re.compile(r'"role"\s*:\s*"user"')
_TEXT_VALUE_RE = re.compile(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"')

#: The image-payload key, whose position relative to the text decides whether a
#: fragment is trustworthy. Same whitespace tolerance, same reason.
_DATA_KEY_RE = re.compile(r'"data"\s*:')


class ResumeNotFound(Exception):
    """``--resume`` named a session that is not on disk (or none exist)."""


def mark_session_origin(session_dir: Path, origin: str, **details: object) -> None:
    """Record that ``session_dir`` was started by ``origin``, not by the user.

    Written by whoever CREATES the directory, which is the only place that
    knows: by the time the picker reads it back, a child session and a user's
    conversation are the same shape on disk.

    Best-effort by contract. Marking is bookkeeping for a listing, and a child
    that cannot write its marker (read-only volume, a race with a retention
    sweep that just removed the directory) must still RUN — the cost of the
    failure is one extra row in a picker, and taking a delegated task down for
    it would be the more expensive bug.

    **The directory's mtime is preserved**, and that is load-bearing rather
    than tidy. Retention sorts and age-expires on the DIRECTORY's mtime
    (``session/retention.py``), and creating a file inside a directory moves
    it — so stamping a session silently reset its retention clock to now.
    Measured on twin stores: marking existing directories kept 3 delegated
    runs the picker will never show while deleting 3 of the user's own
    conversations, and resurrected 59 children that were already past the age
    ceiling. Writing a marker is bookkeeping ABOUT a session, never activity
    IN it, so it must not answer the question "when was this session last
    used". A directory this call creates has no prior mtime and is unaffected.
    """
    payload = {"origin": origin, **details}
    try:
        # Read before the write: this is the value the write is about to
        # destroy. ``None`` for a directory that does not exist yet, which is
        # the fresh-child path and needs no restore.
        try:
            previous = session_dir.stat().st_mtime
        except OSError:
            previous = None
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / ORIGIN_NAME).write_text(json.dumps(payload), encoding="utf-8")
        if previous is not None:
            os.utime(session_dir, (previous, previous))
    except (OSError, TypeError, ValueError):
        return


def session_origin(session_dir: Path) -> str:
    """``origin`` recorded for a session, or ``""`` when it is the user's own.

    Absence means USER, and that direction is deliberate. The alternative —
    marking user sessions and hiding everything unmarked — would have made
    every conversation that predates the marker disappear from the picker,
    which loses real work; an unmarked child merely shows one stale row that
    retention eventually evicts. A listing that shows too much is recoverable
    by typing a filter, one that hides your own session is not.

    Tolerant for the same reason :func:`session_name` is: this runs over every
    session directory to paint a picker, so a truncated or hand-edited marker
    yields ``""`` (treated as the user's) rather than taking the picker down.

    ``errors="replace"`` is load-bearing, not decoration. :func:`mark_session_origin`
    writes non-atomically, so a child killed mid-write (SIGKILL, sleep, a full
    volume) leaves the file cut INSIDE a multi-byte character — and a strict
    decode raises ``UnicodeDecodeError``, which is a ``ValueError`` and would
    sail past an ``except OSError``. One such sidecar took down the whole
    picker and every ``--resume`` with no id, for every session, until the
    user found and deleted the file by hand.
    """
    try:
        raw = (session_dir / ORIGIN_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    try:
        payload = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    origin = payload.get("origin")
    return origin if isinstance(origin, str) else ""


def backfill_session_origins(config_dir: Path, limit: int = 500) -> int:
    """Stamp pre-existing subagent directories once, and return how many.

    Without this the fix only applies to sessions created after the upgrade,
    so the person who reported a picker full of ``[role: reviewer]`` rows
    would upgrade, look, and see the same 40 rows — the change would be
    correct and appear to do nothing until natural churn cleared the store.

    Identification is by the openings only the subagent runner can produce
    (:data:`_ROLE_PREAMBLE`, :data:`_SCOUT_PREAMBLE`), matched at offset 0 of
    the first user message because both are stamped in FRONT of the caller's
    prompt. This deliberately under-claims: a delegated run launched with no
    role profile is indistinguishable from a user's own session and stays
    listed. That is the right direction — an unmarked child costs one stale
    row, while a false positive hides the user's real work, and the whole
    point of a one-time sweep is that a row it misses is one the user can
    still reach.

    Best-effort and bounded like every other function here: it runs at
    startup, so an unreadable directory is skipped rather than raised, and
    ``limit`` caps how many directories are STAMPED per run.

    The cap is on work done, never on how far the scan reaches. Capping the
    scan instead — slicing the directory list — sounds equivalent and is not:
    the list sorts by hex NAME, and the same prefix is recomputed on every
    startup, so any directory sorting past the cut was never visited on any
    run, ever. Measured: a 600-directory store with 50 children sorting after
    the cut stamped 0 on three consecutive startups. Deciding a session's
    origin by where its random name falls in an alphabet is not a policy
    anyone would choose deliberately.
    """
    stamped = 0
    sessions = config_dir / "sessions"
    try:
        directories = sorted(sessions.iterdir())
    except OSError:
        return 0
    for directory in directories:
        if stamped >= limit:
            break
        try:
            if not (directory / TRANSCRIPT_NAME).is_file():
                continue
            # Already answered: never re-stamp, so a marker a user removed by
            # hand to un-hide a session is not silently written back.
            if (directory / ORIGIN_NAME).exists():
                continue
        except OSError:
            continue
        opening = session_name(directory, max_chars=NAME_MAX_CHARS, condense=False)
        if not opening:
            continue
        if _ROLE_PREAMBLE.match(opening) or opening.startswith(_SCOUT_PREAMBLE):
            mark_session_origin(directory, ORIGIN_SUBAGENT, backfilled=True)
            stamped += 1
    return stamped


def is_user_session(session_dir: Path) -> bool:
    """True when a human started this session, so a picker may offer it.

    EVERY non-empty origin is hidden, not a listed set of them: a new value
    added later (a scheduled run, a server-side session) is therefore opt-OUT
    of the picker by default, and an author who wants a new origin to remain
    listable has to say so here. That default is the safe direction — a value
    is minted by whichever code path creates the directory, and the paths that
    do so are the machine's own.
    """
    return not session_origin(session_dir)


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
        # ``@latest`` means the latest conversation THE USER had. A subagent
        # writes its child transcript into the same directory, and a delegated
        # review finishing after the parent's last turn made it the newest
        # directory on disk — so a bare ``--resume`` reopened the reviewer
        # rather than the session that launched it.
        candidates = [
            path
            for path in sessions.glob("*")
            if (path / TRANSCRIPT_NAME).is_file() and is_user_session(path)
        ]
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
    """``(id, mtime)`` for the USER's resumable sessions, newest first.

    Subagent sessions are excluded (:func:`is_user_session`): they are the
    machine's own scratch conversations, and a listing offered to a human is
    about work the human did. They remain resumable by explicit id — nothing
    here removes a directory, and ``hub op='resume'`` continues a child by its
    own path — so this narrows what is OFFERED, never what exists.

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
            mtime = (path / TRANSCRIPT_NAME).stat().st_mtime
        except OSError:
            continue
        # After the stat, not before: the stat is what proves the directory is
        # a session at all, and an unreadable marker must not cost a row.
        if not is_user_session(path):
            continue
        rows.append((path.name, mtime))
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


def stored_session_title(session_dir: Path) -> str:
    """The title this session was last named, or ``""`` when it has none.

    The name a user searches by is the name they last SAW, and that is the
    stored title — auto-generated on the first substantive turn, or typed at
    ``/rename``. Before this existed the picker labelled every row with the
    session's opening message, so a conversation renamed to something
    memorable was still listed under whatever happened to be typed first, and
    a user who could not recall that opening line could not find the session
    at all. That is the reported failure this function closes.

    Scanned out of the raw JSONL rather than replayed through ``Transcript``
    on purpose. This module is import-guarded (see the module docstring): a
    picker row must not drag the engine, the providers or ``asyncio`` onto
    ``local-operator --help``. A regex over two bounded windows is the same
    question asked cheaply.

    BOTH ENDS are read — see :data:`TITLE_SCAN_BYTES` for why a tail-only scan
    missed the title on 78% of real sessions. The LAST match wins across the
    two windows, because each rename appends a full snapshot and the newest row
    is the title in force.

    Tolerant like everything else on this path — an unreadable or truncated
    transcript yields ``""`` and the caller falls back to the opening message
    rather than the picker failing.
    """
    transcript = session_dir / TRANSCRIPT_NAME
    try:
        with transcript.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size > TITLE_SCAN_BYTES * 2:
                # Large enough for two disjoint windows: read the head, then
                # seek to the last TITLE_SCAN_BYTES for the tail.
                handle.seek(0)
                head = handle.read(TITLE_SCAN_BYTES)
                handle.seek(size - TITLE_SCAN_BYTES)
                tail = handle.read()
            else:
                # Small enough that the two windows would overlap: read the
                # WHOLE file once and let it serve as both. Splitting it here
                # is what broke this the first time round -- the head was read,
                # the handle was left at EOF by the size probe, and the `else`
                # branch's read returned b"", so files between 1x and 2x the
                # window (30% of a real store) were searched head-only. That
                # silently reverted a late rename to the name it was renamed
                # AWAY from, which is worse than the missing name this function
                # exists to prevent.
                handle.seek(0)
                head = tail = handle.read()
    except OSError:
        return ""
    # The tail is searched FIRST and wins: a rename made late in a long session
    # is the newest title, and the head can only hold older ones.
    for window in (tail, head):
        matches = _TITLE_ROW_RE.findall(window.decode("utf-8", errors="replace"))
        if matches:
            break
    if not matches:
        return ""
    try:
        # Through the JSON decoder rather than a manual unescape, so a title
        # holding a quote, a backslash or a \uXXXX escape reads back as the
        # characters the user actually saw.
        title = json.loads(f'"{matches[-1]}"')
    except ValueError:
        return ""
    return " ".join(str(title).split())


def session_name(
    session_dir: Path, *, max_chars: int = NAME_MAX_CHARS, condense: bool = True
) -> str:
    """A conversation's display name: its stored title, else its opening message.

    The stored title comes first because it is the name the user last saw on
    the band and in the terminal tab, and therefore the one they will search
    for. The opening message is the FALLBACK, for the two cases that have no
    stored title: a transcript written before titles were journalled, and a
    session closed before its naming call landed. Both still deserve a
    recognisable row, and the opener is what the picker always used.

    Deliberately tolerant. This runs over every session directory to paint a
    picker, so a transcript that is truncated, half-written by a session still
    running, or corrupt yields ``""`` and a nameless row rather than taking
    the picker down. The scan also stops at the first user message and at
    :data:`NAME_SCAN_CHARS`, so it costs one short read per session instead of
    a full parse of a file that can be hundreds of kilobytes.
    """
    stored = stored_session_title(session_dir)
    if stored:
        return _condense(stored, max_chars) if condense else stored
    transcript = session_dir / TRANSCRIPT_NAME
    try:
        with transcript.open("r", encoding="utf-8", errors="replace") as handle:
            # ONE bounded read, not `for line in handle`. Iterating the file
            # materialises each line in full BEFORE any cap can be checked, so a
            # transcript whose first line is a pasted file or a base64 image —
            # exactly the case this cap exists for — allocated the whole line
            # anyway (measured: an 80 MB first line peaked at 168 MB before the
            # check that was supposed to prevent it). Reading a fixed window
            # first makes the bound real.
            head = handle.read(NAME_SCAN_CHARS)
    except OSError:
        return ""
    # A final line with no newline after it is HELD BACK from the strict parse
    # only when the window was actually filled — i.e. the read stopped because
    # of the cap, so that line is a half-READ one and parsing it as JSON would
    # be parsing a fragment. When the whole file fitted, the same shape is a
    # complete last line that simply has no trailing newline, and dropping it
    # lost the name of any session whose transcript is a single entry. Held
    # rather than discarded because the fragment still carries the opener's
    # text: see ``_text_from_fragment``.
    truncated = len(head) >= NAME_SCAN_CHARS
    lines = head.splitlines()
    fragment = ""
    if truncated and lines and not head.endswith("\n"):
        fragment = lines.pop()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            # A partial final line is normal for a session that is still
            # running: the writer appends, we may read mid-write.
            continue
        if not isinstance(entry, dict) or entry.get("type") != "message":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        # ``role`` is matched EXACTLY: a tool result is also a four-character
        # role and carries the tool's output, which would name the
        # conversation after a directory listing.
        if payload.get("role") != "user":
            continue
        text = _first_text(payload.get("content"))
        if text:
            # ``condense=False`` returns the opening text with its line
            # breaks intact, which the backfill needs: the role preamble it
            # matches is ``[role: <name>]\n``, and condensing flattens that
            # newline into a space before the pattern could ever see it.
            return _condense(text, max_chars) if condense else text
    # The window held no COMPLETE line, so the opener is a fragment. Dropping
    # it (which is all this used to do) left every session that begins with a
    # pasted screenshot permanently nameless: one base64 image puts the first
    # line past the cap, and the picker then showed `(unnamed session)` for the
    # rest of that conversation's life. Measured on two real sessions whose
    # first lines were 115,289 and 733,034 chars.
    return _condense(_text_from_fragment(fragment), max_chars) if fragment else ""


def _text_from_fragment(fragment: str) -> str:
    """The opening user message's text, recovered from a HALF-READ first line.

    Deliberately a scan and not a parse: the fragment is an incomplete JSON
    object, so there is nothing `json.loads` can do with it. What makes the scan
    safe is the ORDER the writer emits: a user message with attachments
    serializes its text block before the image data (``Message.user(text,
    images)`` keeps that order), so on a line whose tail is megabytes of base64
    the topic sits in the first few hundred characters — measured at offset 135,
    with the image ``data`` key at 443.

    Three guards, because a wrong name here is worse than none. The fragment
    must identify itself as a user message, the text value must be COMPLETE (a
    closing quote inside the window, never a mid-word cut), and it must appear
    before any ``data`` key so a session can never be named after base64.
    """
    if _FRAGMENT_USER_RE.search(fragment[:_FRAGMENT_HEAD_CHARS]) is None:
        return ""
    match = _TEXT_VALUE_RE.search(fragment)
    if match is None:
        return ""
    data = _DATA_KEY_RE.search(fragment)
    if data is not None and data.start() < match.start():
        return ""
    try:
        # Through the JSON decoder rather than a hand-rolled unescape: the
        # captured span is a JSON string body, and a title showing a literal
        # ``\u2014`` would be its own bug.
        return json.loads(f'"{match.group(1)}"')
    except ValueError:
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

    Synchronous, and called from the UI thread when the picker opens. That is
    deliberate: each read is bounded (:data:`NAME_SCAN_CHARS`) and stops at the
    first user turn, so the pathological case — a transcript whose first line
    is an 80 MB paste — measures 0.2 ms, and fifty of them are still under a
    frame. Moving this to a worker would trade that for a picker that opens
    empty and fills in, which is worse for a list the user is about to read.
    """
    return [
        SessionRow(session_id, mtime, session_name(config_dir / "sessions" / session_id))
        for session_id, mtime in recent_sessions(config_dir, limit)
    ]
