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
import sys
from pathlib import Path
from typing import Any, NamedTuple

#: ``--resume`` with no id. A sentinel rather than a second boolean flag so the
#: whole "which session" decision stays ONE value threaded through one parameter.
RESUME_LATEST = "@latest"

#: Rows the CLI's ``--resume <typo>`` recovery listing prints to stderr.
#:
#: Named rather than a bare ``10`` at the call site because it is a DELIBERATELY
#: short list, not an incidental one: the listing is an error message helping a
#: user who mistyped an id, where the newest few sessions are the help and the
#: whole store would bury it. The picker is the surface that shows everything
#: (:func:`recent_session_rows` returns the full store by default); this is the
#: one place a cap is the right answer, so it says so.
RESUME_RECOVERY_LISTING = 10

#: The file whose presence makes a directory a resumable session. Also what the
#: recency ordering is read from: a directory's own mtime moves for reasons that
#: are not turns (an origin stamp, a sibling file), so it is not the clock to use.
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

#: ``origin`` value for a session ``/fork`` branched off another. Unlike
#: :data:`ORIGIN_SUBAGENT` this marks the user's OWN work: the marker records
#: PROVENANCE (which conversation this branched from, in its ``parent`` field),
#: and provenance is not a reason to hide a row.
ORIGIN_FORK = "fork"

#: Origins that are still the user's own conversation, so :func:`is_user_session`
#: keeps listing them.
#:
#: An ALLOW-LIST rather than an ``or`` bolted onto the predicate, because the
#: default for a new origin must stay "not the user's" (see
#: :func:`is_user_session`): visibility is opt-IN, so an author minting a new
#: origin value has to come here and say so deliberately. That is exactly the
#: "has to say so here" the predicate's docstring already demanded — this makes
#: the place to say it a named constant instead of an edit to a boolean.
USER_ORIGINS: frozenset[str] = frozenset({ORIGIN_FORK})

#: Memoised ``origin.json`` verdicts for :func:`recent_sessions`, keyed on each
#: marker's own ``(mtime, size)``.
#:
#: Why this exists. The listing must READ AND PARSE every marker that exists —
#: existence alone must never be read as "subagent", because a truncated or
#: hand-edited sidecar deliberately parses to ``""`` so it reads as the USER's
#: session rather than vanishing from the picker (see :func:`session_origin`;
#: one such file took the picker down for every session on the machine). The
#: markers that exist are the SUBAGENT ones, and subagents outnumber user
#: sessions ~10.6:1, so that rule costs one file read per subagent directory:
#: measured at 1127 ms over a 31,700-directory store, of which 639 ms is the
#: reads and only 17 ms the parsing. Skipping the read for unmarked directories
#: therefore saves ~8% and cannot fix it; the parse is not the cost.
#:
#: Why memoising is SOUND rather than a guess: the marker is written once, at
#: directory creation (``harness.subagent``), and the only other writer is the
#: one-shot backfill below. The verdict is immutable once written, so a marker
#: whose ``(mtime, size)`` is unchanged cannot have changed its meaning.
#:
#: What may NOT go in here, and why each would be a bug:
#: * Only a verdict actually PARSED from a marker that was READ is stored. A
#:   stat failure or an unreadable file falls through to the real read every
#:   time (:func:`_session_origin_read` reports readability separately for
#:   exactly this): those describe the MOMENT — EMFILE under the descriptor
#:   pressure this scan itself creates, a network volume blip — while the key
#:   is the marker's immutable ``(mtime, size)``, so caching one would serve a
#:   transient outage as a permanent wrong verdict for the life of the file.
#: * A CORRUPT payload is cached, and that is deliberate rather than an
#:   oversight: a parse failure is a fact about the file's CONTENT, stable for
#:   as long as the bytes are, and re-deriving it every scan would return the
#:   same ``""``. Rewriting the marker changes its ``(mtime, size)`` and
#:   expires the entry, which is exactly when the verdict could differ.
#: * ABSENCE is never cached. Unmarked already means user and is the cheap path,
#:   and a directory the backfill stamps later must be re-read, not answered
#:   from a stale "no marker" fact.
ORIGIN_CACHE_NAME = "origin-verdicts.json"

#: Bumped when the cache's shape or key changes, so an older file is discarded
#: rather than misread. Independent of ``search_index.INDEX_VERSION`` — the two
#: caches version separately and neither number constrains the other; only the
#: mechanism is borrowed.
ORIGIN_CACHE_VERSION = 1

#: Journals the session's title (and every name it has ever borne) beside the
#: transcript, mirroring :data:`ORIGIN_NAME` exactly. A SIDECAR rather than a
#: field in the JSONL for the same reason the origin marker is one: the picker's
#: cost model is one bounded read per row, and the title in force can sit
#: anywhere in a multi-megabyte transcript (the auto-name lands at turn 2, a
#: mid-session ``/rename`` lands in the untouched middle — see
#: :data:`TITLE_SCAN_BYTES` for the window-scan gap this closes). ``Path.stat``
#: plus a sub-kilobyte read is O(1) in transcript size where the scan is not,
#: so :func:`stored_session_title` consults this first and only falls back to
#: the window scan for sessions written before the sidecar existed.
TITLE_SIDECAR_NAME = "title.json"

#: The title backfill's per-directory "nothing to journal" marker. The sweep
#: used to treat only an existing :data:`TITLE_SIDECAR_NAME` as answered, so a
#: session with no journalled title anywhere in its transcript — the majority
#: of a long-lived store, since every session that simply never got renamed
#: stays in that state forever — was FULLY RE-READ on every boot: measured 323
#: ms per boot on a 1,365-session store, 1,268 of which could never produce a
#: sidecar. The sentinel records that the scan RAN and found nothing, so the
#: second boot's answer costs one ``stat`` per directory. JSON rather than an
#: empty file so a future reader can carry a reason (``scanned_at``) without a
#: second format migration; ``write_session_title``'s mtime-preservation
#: contract applies to it too, for the reason its docstring gives.
TITLE_SCAN_SENTINEL_NAME = "title-scan.json"

#: The origin sweep's own "considered and not a subagent" marker, mirroring
#: :data:`TITLE_SCAN_SENTINEL_NAME`. It exists SEPARATELY from that sentinel
#: because the two sweeps answer different questions and traverse
#: independently: the origin sweep stops at ``limit`` STAMPS while the title
#: sweep is uncapped, so a title sentinel must never stand in for an origin
#: answer — a directory the origin sweep never reached would be suppressed
#: forever. Only the origin sweep writes this file, so its presence means
#: exactly "this pass read this opener and it was not a subagent's".
ORIGIN_SCAN_SENTINEL_NAME = "origin-scan.json"

#: The session's ATTACHED IDENTITY — the ``/team`` roster, the ``/agent``
#: profile, and the ``/goal`` — journalled beside the transcript, mirroring
#: :data:`TITLE_SIDECAR_NAME` exactly.
#:
#: Why this file has to exist at all: the team and agent briefs ride the
#: VOLATILE TAIL of the system prompt (see ``prompts_api.build_system_blocks``
#: and ``session/goal.py``), and the tail is rebuilt from a ``GoalState`` that
#: ``session_factory`` constructs EMPTY on every session. Nothing in the
#: transcript reproduces it, so a ``--resume`` genuinely dropped the persona
#: from the prompt — the manager a user attached with ``/team`` was not merely
#: missing from the status band, it was gone from the model's instructions and
#: the conversation carried on as an ordinary session. Only the FRONT END could
#: see the blank band, which is why this read as a display bug for so long.
#:
#: A SIDECAR rather than a transcript row, for the reasons the title sidecar
#: documents and one more that is specific to this state: the attachment is a
#: property OF the session, not an event IN the conversation, and the restore
#: runs during construction, before any replay, so a value it had to scan the
#: JSONL for would arrive too late to reach the first prompt's tail.
ATTACHMENT_SIDECAR_NAME = "attachment.json"

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
#: **The window gap this scan cannot close, and where it is closed instead.**
#: A rename made mid-conversation, then buried under a further 128 KB and never
#: renamed again, falls between the two windows: the head still holds the
#: ORIGINAL title, so a window-only scan labels that session with the name it
#: was renamed *away from*. Worse, a topic-pivot session auto-named late (its
#: only titles sitting in the untouched middle of a multi-megabyte transcript)
#: is invisible to both windows and reverts to its opening message — the
#: reported failure of a session that could not be found by its own subject.
#:
#: This scan does not try to fix that by widening: reading whole transcripts on
#: the picker's synchronous path was measured at 400 ms against 64 ms across a
#: real store. Instead the fix the previous comment here PRESCRIBED is now
#: implemented — the title is journalled to a sidecar (``title.json``, see
#: :func:`write_session_title`) the way :func:`mark_session_origin` journals
#: origin, one stat and one small read with no size dependence at all.
#: :func:`stored_session_title` consults that sidecar first, so this window
#: scan is now the FALLBACK for sessions written before the sidecar existed and
#: not yet reached by :func:`backfill_session_titles`, not the primary path.
#: Both ends are still read because that fallback still wants the newest title
#: it can reach on a pre-sidecar session.
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

#: BYTES of a transcript's tail read when previewing its last reply.
#:
#: Bytes, not characters, because this seeks from the END of the file, and a
#: byte offset is the only thing ``seek`` accepts. The window is decoded with
#: ``errors="replace"`` and its first (probably partial) line dropped, so
#: landing mid-codepoint is harmless.
#:
#: Sized for the same reason :data:`NAME_SCAN_CHARS` is: the preview is a
#: convenience shown on a list row, and one pathological entry — a pasted file,
#: a base64 image — must not turn painting that list into reading megabytes per
#: session. 64 KiB comfortably holds the last several entries of an ordinary
#: transcript while bounding the pathological one.
PREVIEW_SCAN_BYTES = 64_000

#: Characters of the previewed reply kept. A list row shows one line; anything
#: past this is cut by the surface anyway, and carrying more over the wire for
#: every row in the list is pure weight.
PREVIEW_MAX_CHARS = 200

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
    that cannot write its marker (read-only volume) must still RUN — the cost
    of the failure is one extra row in a picker, and taking a delegated task
    down for it would be the more expensive bug.

    **The directory's mtime is preserved.** Recency for ``--resume`` and the
    picker is read from the transcript's mtime, not the directory's, but
    other readers still look at the directory (``os.listdir`` + ``stat``
    listings, backup tools). Writing a marker is bookkeeping ABOUT a
    session, never activity IN it, so it must not answer the question "when
    was this session last used". A directory this call creates has no prior
    mtime and is unaffected.
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
    origin, _readable = _session_origin_read(session_dir)
    return origin


def _session_origin_read(session_dir: Path) -> tuple[str, bool]:
    """:func:`session_origin`'s verdict, plus whether the marker was READ at all.

    Exists because those two facts are different and only the traversal needs
    the second. ``session_origin`` deliberately collapses every failure into
    ``""`` — that tolerance is its whole point and its public contract, and a
    caller asking "is this the user's session" is right to be told "yes" when
    the claim cannot be trusted.

    A CACHE, though, must not memoise that answer. ``""`` from a parse failure
    is a fact about the file's CONTENT, so it is stable while the bytes are:
    re-deriving it on every scan would return the same verdict, and the marker
    changing is exactly when the memo's ``(mtime, size)`` key expires. ``""``
    from an ``OSError`` is a fact about the MOMENT — EMFILE under the descriptor
    pressure a 30,000-directory scan creates, a network volume blip, a
    permissions change mid-scan — and the file it describes is immutable by
    design, so memoising it pins a wrong verdict for the life of the marker
    rather than for the life of the outage. ``readable=False`` is how the
    traversal tells those apart and declines to cache the second.
    """
    try:
        raw = (session_dir / ORIGIN_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", False
    try:
        payload = json.loads(raw)
    except ValueError:
        return "", True
    if not isinstance(payload, dict):
        return "", True
    origin = payload.get("origin")
    return (origin if isinstance(origin, str) else ""), True


class SessionTitle(NamedTuple):
    """The title sidecar's contents: the in-force name and every past one.

    ``text`` is the name currently on the band and the terminal tab — the one
    the user last saw and will search by. ``names`` is every distinct title the
    session has ever carried, first-seen order, so a search matches a name the
    session was renamed *away from* as well as its current one. ``user_set``
    rides along for parity with the transcript entry: the picker does not need
    it, but a future reader deciding rename precedence might, and it costs
    nothing to keep.
    """

    text: str
    user_set: bool
    names: tuple[str, ...]


def _read_title_sidecar(session_dir: Path) -> SessionTitle | None:
    """Parse ``title.json``, or ``None`` when it is absent or unusable.

    Tolerant for the same reason :func:`session_origin` is, and with the same
    ``errors="replace"`` load-bearing detail: this runs over every session
    directory to paint a picker, and :func:`write_session_title` writes
    non-atomically, so a process killed mid-write can leave the file cut inside
    a multi-byte character. A strict decode would raise ``UnicodeDecodeError``
    (a ``ValueError``) and could sail past an ``except OSError`` and take the
    whole picker down — the exact failure a corrupt ``origin.json`` once caused.
    A missing or malformed sidecar yields ``None`` so the caller falls back to
    the window scan, never an exception.
    """
    try:
        raw = (session_dir / TITLE_SIDECAR_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    text = payload.get("text")
    if not isinstance(text, str):
        return None
    raw_names = payload.get("names")
    names = (
        tuple(name for name in raw_names if isinstance(name, str))
        if (isinstance(raw_names, list))
        else ()
    )
    return SessionTitle(
        text=" ".join(text.split()),
        user_set=bool(payload.get("user_set")),
        names=names,
    )


def read_title_names(session_dir: Path) -> list[str]:
    """Every name this session has borne, from the sidecar; ``[]`` when absent.

    The source the digest folds in (see ``search_index.build_index``) so a
    session is findable by any subject it was ever named for, not only its
    current title. Empty for a pre-sidecar session, which the backfill sweep
    (:func:`backfill_session_titles`) fills in once at startup.
    """
    sidecar = _read_title_sidecar(session_dir)
    return list(sidecar.names) if sidecar else []


def write_session_title(
    session_dir: Path, text: str, *, user_set: bool, past_names: list[str]
) -> None:
    """Journal the in-force title (and every name ever borne) to a sidecar.

    Why a sidecar: :func:`stored_session_title`'s window scan is blind to a
    title in the middle of a large transcript (see :data:`TITLE_SCAN_BYTES` —
    an auto-name at turn 2 pushed past the head window by an hour of work, or a
    mid-session ``/rename`` buried between the two windows). One stat and one
    small read here is O(1) in transcript size, closing that gap for good.

    Best-effort by contract, exactly like :func:`mark_session_origin`: a title
    is decoration, and a session that cannot write its sidecar (read-only
    volume, full disk) must still RUN. The cost of the failure is a stale
    picker label until the next rename or the backfill sweep, never a lost turn.

    ``names`` accumulates: ``text`` is appended to ``past_names`` (deduped,
    first-seen order preserved) so a search matches a name the session was
    renamed away from. The list is authoritative for names seen since the
    sidecar began; the one-time :func:`backfill_session_titles` sweep recovers
    the complete history for sessions that predate it.

    **The directory's mtime is preserved**, for the same reason
    :func:`mark_session_origin` preserves it: recency ranks by the transcript's
    mtime, but other readers (``os.listdir`` + ``stat`` listings, backups) look
    at the directory, and journalling a title is bookkeeping ABOUT a session,
    never activity IN it. The write is atomic (pid-named temp + ``replace``,
    like ``search_index._save``) because the picker may read this file while a
    concurrent session rewrites it.
    """
    normalized = " ".join(text.split())
    # Whitespace-normalise every name the same way ``text`` is, so the sidecar's
    # ``names`` and its ``text`` agree and the digest folds a name in exactly as
    # the reader will match it. Dedup runs AFTER normalisation so two names that
    # differ only in internal whitespace collapse to one. ``dict.fromkeys`` is
    # the stdlib ordered-set idiom, preserving first-seen order; empties (a name
    # that was all whitespace) are dropped. The in-force title is folded in as
    # the newest name so a caller that passes only the prior history still gets
    # a complete list.
    normalized_past = [n for n in (" ".join(p.split()) for p in past_names) if n]
    names = (
        list(dict.fromkeys([*normalized_past, normalized]))
        if normalized
        else list(dict.fromkeys(normalized_past))
    )
    payload = {"text": normalized, "user_set": user_set, "names": names}
    try:
        try:
            previous = session_dir.stat().st_mtime
        except OSError:
            previous = None
        session_dir.mkdir(parents=True, exist_ok=True)
        sidecar = session_dir / TITLE_SIDECAR_NAME
        # The temp carries the writer's PID so two sessions writing at once do
        # not ``replace`` a document the other is still filling — the same
        # torn-write hazard ``search_index._save`` documents.
        tmp = sidecar.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(sidecar)
        if previous is not None:
            os.utime(session_dir, (previous, previous))
    except (OSError, TypeError, ValueError):
        return


class SessionAttachment(NamedTuple):
    """What ``/team``, ``/agent`` and ``/goal`` had put on a session.

    NAMES, never brief BODIES, and that is the load-bearing decision rather
    than a size optimisation. A stored brief is a SNAPSHOT of a team's
    collaboration/project text or a profile's instructions at attach time; the
    operator edits those files between sessions (that is the whole point of a
    durable registry), so replaying a stored copy would resume the session onto
    instructions that no longer exist anywhere. Re-resolving the name through
    the live registry on restore means a resumed session runs the CURRENT
    definition, which is what the user means by "resume my lopdev manager".

    The tradeoff is accepted deliberately: a renamed or deleted team cannot be
    restored, where a stored brief could have been. That case is handled by
    saying so plainly (see the TUI's restore notice) rather than by silently
    reviving a definition the operator removed.
    """

    #: ``/team`` roster name; "" when no team was attached.
    team: str
    #: ``/agent`` profile DISPLAY name; "" when no profile was attached.
    agent: str
    #: The standing ``/goal`` text; "" when unset. Stored here rather than left
    #: to the transcript because it shares the volatile tail's fate exactly.
    goal: str


def read_session_attachment(session_dir: Path) -> SessionAttachment | None:
    """Parse ``attachment.json``, or ``None`` when absent or unusable.

    Tolerant on exactly the same terms as :func:`_read_title_sidecar`, and the
    ``errors="replace"`` is load-bearing for the same reason: a process killed
    mid-write can cut the file inside a multi-byte character, and a strict
    decode raises ``UnicodeDecodeError`` — a ``ValueError``, which would sail
    past an ``except OSError`` and take down not a picker row this time but the
    whole RESUME. A session must always reopen; losing an attachment is a
    notice, losing the conversation is not survivable.
    """
    try:
        raw = (session_dir / ATTACHMENT_SIDECAR_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None

    def _text(key: str) -> str:
        value = payload.get(key)
        return value.strip() if isinstance(value, str) else ""

    return SessionAttachment(team=_text("team"), agent=_text("agent"), goal=_text("goal"))


def write_session_attachment(session_dir: Path, *, team: str, agent: str, goal: str) -> None:
    """Journal the session's attached identity so a resume can rebuild it.

    Called on CHANGE (attach, detach, goal set/clear), never per turn: the
    attachment moves a handful of times in a session's life, and a per-turn
    write would be pure I/O for a value that did not move.

    Best-effort by contract, exactly like :func:`write_session_title` and
    :func:`mark_session_origin`. An attachment that cannot be journalled
    (read-only volume, full disk) must never fail the turn that changed it: the
    cost is one resume that opens unattached, which is the behaviour every
    session had before this file existed.

    The write is ATOMIC (pid-named temp + ``replace``) because two processes
    can hold the same session directory — a live owner and a ``/resume`` that
    is about to be refused both touch it — and a reader hitting a half-written
    document would parse as "no attachment" and silently drop the persona. The
    PID in the temp name keeps two concurrent writers from ``replace``-ing a
    document the other is still filling, the same hazard ``write_session_title``
    and ``search_index._save`` document.

    **The directory's mtime is preserved**, for the reason the other two
    sidecars preserve it: recency ranks by the transcript's mtime, and
    journalling an attachment is bookkeeping ABOUT a session, never activity IN
    it. Attaching a team must not reorder the ``/resume`` picker.
    """
    # ``strip()`` ONLY — never ``" ".join(x.split())``. These are LOOKUP KEYS,
    # not display titles, and every resolver they are matched against strips
    # without collapsing (``resolve_profile``, ``resolve_profile_or_specialist``,
    # ``TeamRegistry.get_team_by_name``, which casefolds and strips). Collapsing
    # internal whitespace here broke the round trip for any agent profile whose
    # registered name contains repeated spaces — free-form and not normalised by
    # ``AgentRegistry.create_agent`` — so a profile named ``"Deep  Auditor"``
    # attached live, was written as ``"Deep Auditor"``, then failed to resolve on
    # resume and told the user it had been renamed or deleted (R2). Normalising
    # is right for a title (the sidecar this shape was copied from) and wrong
    # for a key: what is stored has to be what the resolver will compare.
    payload = {
        "team": (team or "").strip(),
        "agent": (agent or "").strip(),
        "goal": (goal or "").strip(),
    }
    try:
        try:
            previous = session_dir.stat().st_mtime
        except OSError:
            previous = None
        session_dir.mkdir(parents=True, exist_ok=True)
        sidecar = session_dir / ATTACHMENT_SIDECAR_NAME
        tmp = sidecar.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(sidecar)
        if previous is not None:
            os.utime(session_dir, (previous, previous))
    except (OSError, TypeError, ValueError):
        return


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
            # This pass's OWN "considered and not a subagent" marker — not
            # the title sweep's sentinel. The two sweeps traverse
            # independently and THIS one stops at ``limit`` stamps, so a
            # title sentinel written for a directory this pass never reached
            # would suppress the origin question forever (the 501st
            # stampable subagent behind a >500 backlog, permanently
            # unmarked). A marker only this pass writes can only exist for
            # a directory this pass genuinely visited.
            if (directory / ORIGIN_SCAN_SENTINEL_NAME).exists():
                continue
        except OSError:
            continue
        opening = session_name(directory, max_chars=NAME_MAX_CHARS, condense=False)
        if not opening:
            continue
        if _ROLE_PREAMBLE.match(opening) or opening.startswith(_SCOUT_PREAMBLE):
            mark_session_origin(directory, ORIGIN_SUBAGENT, backfilled=True)
            stamped += 1
        else:
            # Not a subagent: record it the same way the marker records the
            # opposite answer, so the next boot's sweep costs one stat here
            # too. Best-effort and mtime-preserving for the same reasons the
            # title sentinel's writer gives; a failed write costs one redundant
            # opener read, never a lost session.
            _write_origin_scan_sentinel(directory)
    return stamped


def _scan_all_titles(transcript: Path) -> list[tuple[str, bool]]:
    """Every ``(text, user_set)`` title journalled in ``transcript``, in order.

    A FULL read, unlike :func:`stored_session_title`'s two windows — which is
    why it lives only on the backfill path and never on the picker's hot path.
    Finding *all* titles (not just the newest) requires it: they can sit
    anywhere, as the topic-pivot session that motivated this proved. Parsed
    line-by-line rather than by the windowed regex so ``user_set`` is read
    alongside each ``text`` and the ordering is exact.

    Tolerant like every reader here: an unreadable transcript or a half-written
    final line yields what it could parse, never an exception.
    """
    titles: list[tuple[str, bool]] = []
    try:
        with transcript.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line or _TITLE_CUSTOM_TYPE not in line:
                    # Cheap reject before the JSON parse: title rows are a tiny
                    # fraction of a transcript, and the substring check skips
                    # the decode for every message and tool line.
                    continue
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                payload = entry.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("custom_type") != _TITLE_CUSTOM_TYPE:
                    continue
                details = payload.get("details")
                if not isinstance(details, dict):
                    continue
                text = details.get("text")
                if isinstance(text, str) and text.strip():
                    titles.append((" ".join(text.split()), bool(details.get("user_set"))))
    except OSError:
        return titles
    return titles


def _write_origin_scan_sentinel(session_dir: Path) -> None:
    """Record that the origin sweep read this opener and found no subagent.

    Same best-effort, mtime-preserving, atomic-write contract as
    :func:`_write_title_scan_sentinel` — see its docstring for the reasoning;
    this is that function with a different file name, kept separate so each
    sweep owns its own answer.
    """
    try:
        try:
            previous = session_dir.stat().st_mtime
        except OSError:
            previous = None
        sentinel = session_dir / ORIGIN_SCAN_SENTINEL_NAME
        tmp = sentinel.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"scanned": True}), encoding="utf-8")
        tmp.replace(sentinel)
        if previous is not None:
            os.utime(session_dir, (previous, previous))
    except (OSError, TypeError, ValueError):
        return


def _write_title_scan_sentinel(session_dir: Path) -> None:
    """Record that the title backfill scanned this directory and found nothing.

    Mirrors :func:`write_session_title`'s best-effort contract, because it is
    the same trade: the sentinel is a boot-cost optimisation, and a session on
    a read-only volume must still RUN. The cost of a failed write is one
    redundant full scan on the next boot — the pre-fix behaviour — never a
    lost turn. Mtime is preserved and the write is atomic (pid-named temp +
    ``replace``) for the reasons the title sidecar's writer documents at
    length: journalling a scan is bookkeeping ABOUT a session, never activity
    IN it, and a concurrent reader must never see a torn file.
    """
    try:
        try:
            previous = session_dir.stat().st_mtime
        except OSError:
            previous = None
        sentinel = session_dir / TITLE_SCAN_SENTINEL_NAME
        tmp = sentinel.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"scanned": True}), encoding="utf-8")
        tmp.replace(sentinel)
        if previous is not None:
            os.utime(session_dir, (previous, previous))
    except (OSError, TypeError, ValueError):
        return


def backfill_session_titles(config_dir: Path, limit: int = 500) -> int:
    """Write the title sidecar for sessions that predate it, and return how many.

    Mirrors :func:`backfill_session_origins` exactly, and for the same reason:
    without it the sidecar fix only applies to sessions renamed AFTER the
    upgrade, so the person who reported being unable to find a topic-pivot
    session by its subject would upgrade, look, and see the same unfindable row
    — the change would be correct and appear to do nothing until each session's
    next rename. This one-time sweep makes every existing session findable by
    every name it has borne immediately after upgrade.

    A session with a title in the untouched middle of a large transcript is the
    case this exists for: :func:`stored_session_title`'s window scan misses it,
    so a full read (:func:`_scan_all_titles`) is the only way to recover its
    real title and its past names. That read is O(transcript size), which is why
    it is confined to this startup path — bounded by ``limit`` and run once per
    session ever, never per picker-open — exactly the trade
    :func:`backfill_session_origins` makes.

    "Once per session ever" now includes the no-title case. A directory with
    no journalled title gets a sentinel (:data:`TITLE_SCAN_SENTINEL_NAME`) so
    the next boot answers it with one ``stat`` instead of another full read;
    without it the sweep was perpetual on exactly the store it was meant to
    fix once — a session that never bore a title can never grow a sidecar, so
    it was re-scanned to the same "nothing" on every launch for the store's
    whole life (measured 323 ms per boot on a real 1,365-session store,
    1,268 of them permanently in that state). The sentinel is deliberately
    NOT a title: neither the picker nor :func:`stored_session_title` reads it,
    so their behaviour is byte-identical before and after.

    Best-effort and bounded like every other function here: an unreadable
    directory is skipped rather than raised, and ``limit`` caps how many
    sidecars are WRITTEN per run.

    The cap is on work done, never on how far the scan reaches, for the reason
    :func:`backfill_session_origins` documents at length: slicing the directory
    list instead would leave any session sorting past the cut unvisited on
    every run forever, because the list sorts by hex name and the same prefix
    is recomputed each startup.
    """
    written = 0
    sessions = config_dir / "sessions"
    try:
        directories = sorted(sessions.iterdir())
    except OSError:
        return 0
    for directory in directories:
        if written >= limit:
            break
        try:
            transcript = directory / TRANSCRIPT_NAME
            if not transcript.is_file():
                continue
            # Already answered: never re-stamp. The sidecar is event-sourced
            # from here on, so a rewrite would only risk clobbering a newer
            # sidecar with an older full scan on a session that has since been
            # renamed. The scan sentinel answers the same "considered" question
            # for the no-title case, which is what ends the perpetual rescan.
            if (directory / TITLE_SIDECAR_NAME).exists():
                continue
            if (directory / TITLE_SCAN_SENTINEL_NAME).exists():
                continue
        except OSError:
            continue
        titles = _scan_all_titles(transcript)
        if not titles:
            # No journalled title at all (a session that predates title
            # journalling, or one closed before its naming call landed). Leave
            # it to the window-scan fallback and the opening-message name; there
            # is nothing to journal — but RECORD that the scan ran, so the next
            # boot does not pay for the same answer again.
            _write_title_scan_sentinel(directory)
            continue
        past_names = [text for text, _ in titles]
        newest_text, newest_user_set = titles[-1]
        write_session_title(directory, newest_text, user_set=newest_user_set, past_names=past_names)
        written += 1
    return written


def is_user_session(session_dir: Path) -> bool:
    """True when a human started this session, so a picker may offer it.

    Every non-empty origin is hidden EXCEPT the ones named in
    :data:`USER_ORIGINS`: a new value added later (a scheduled run, a
    server-side session) is therefore opt-OUT of the picker by default, and an
    author who wants a new origin to remain listable has to say so there. That
    default is the safe direction — a value is minted by whichever code path
    creates the directory, and the paths that do so are the machine's own.

    ``fork`` is the first origin to take the opt-in, and it is worth stating why
    it differs in kind from ``subagent``: a subagent directory is a machine's
    delegated run that the user never opened, while a fork is a conversation the
    user deliberately branched. Both carry a marker; only one of them is
    somebody else's work. Four consumers read this predicate and a fork is
    wanted in all four — the ``/resume`` picker, ``resume_dir``'s ``@latest``
    scan, the multiplexer's crash-restore binding, and the mobile session list.
    """
    origin = session_origin(session_dir)
    return not origin or origin in USER_ORIGINS


class _reverse_name(str):
    """A ``max`` key for "ascending id wins on a tie": ``max`` over
    ``(activity, id)`` would pick the LARGEST id, and the picker's sort puts
    the smallest first."""

    def __lt__(self, other: object) -> bool:
        return str.__gt__(self, str(other))

    def __gt__(self, other: object) -> bool:
        return str.__lt__(self, str(other))


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

    WHAT COUNTS AS RESUMABLE IS WHAT THE PICKER LISTS: a directory with
    activity (``session_activity`` — a transcript OR an unread mail spool).
    One rule on both surfaces, or the picker offers a row this function then
    refuses (review round 3, R3-5: a peer's message spooled into an idle
    open-and-quit session gave it the top picker row and ``ResumeNotFound``).
    An inbox-only session IS worth reopening — a spooled message is a reason
    to come back, and the transcript store starts empty for it exactly as it
    does for a fresh session — so the rule was widened rather than the row
    hidden.
    """
    from local_operator.session.retention import session_activity

    sessions = config_dir / "sessions"
    if requested == RESUME_LATEST:
        # ``@latest`` means the latest conversation THE USER had. A subagent
        # writes its child transcript into the same directory, and a delegated
        # review finishing after the parent's last turn made it the newest
        # directory on disk — so a bare ``--resume`` reopened the reviewer
        # rather than the session that launched it.
        # Ranked by the picker's clock with the picker's tie-break, so
        # ``@latest`` is the picker's first row by construction (R3-5).
        candidates = [
            (activity, path)
            for path in sessions.glob("*")
            if (activity := session_activity(path)) is not None and is_user_session(path)
        ]
        if not candidates:
            raise ResumeNotFound("no previous session to resume")
        return max(candidates, key=lambda item: (item[0], _reverse_name(item[1].name)))[1]

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
        present = session_activity(candidate) is not None
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


def live_session_owner(config_dir: Path, session_id: str) -> int | None:
    """Pid of the process currently hosting ``session_id``, or ``None``.

    Two writers on one transcript is how a TUI ``/resume`` of a phone-started
    session painted the splash: the second process claimed the directory,
    replayed a mid-write journal, and left the first process as the only one
    still appending. The live process already publishes to the phone, so a
    second front end should attach to THAT process rather than open another
    writer.

    Consults the session directory's ``.session.pid`` liveness marker — the
    same file the retention sweep uses. Stdlib-only and import-light: this
    module must stay off the engine and the mobile package (see the module
    docstring). A live TUI or phone-started child always writes that marker
    when it claims the directory.
    """
    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        return None
    marker = config_dir / "sessions" / session_id / ".session.pid"
    try:
        raw = marker.read_text(encoding="utf-8").strip()
        pid = int(raw)
    except (OSError, ValueError):
        return None
    if pid <= 0:
        return None
    # Windows has no signal 0 — ``os.kill`` there TERMINATES the target
    # (see ``session.retention._process_alive``). A parseable marker is
    # treated as live rather than probed, so a ``/resume`` cannot kill
    # the phone-started child it is trying to share (F2).
    if sys.platform == "win32":
        return pid
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    except OSError:
        return pid
    return pid


def origin_cache_path(config_dir: Path) -> Path:
    """Where this store's ``origin.json`` verdict cache lives.

    Beside the search index, under ``cache/``: both are derived data a user may
    delete at any time to force a rebuild, and neither is a source of truth.
    """
    return config_dir / "cache" / ORIGIN_CACHE_NAME


def _load_origin_cache(path: Path) -> dict[str, Any]:
    """The cached verdicts, or an empty mapping when absent, stale or corrupt.

    Every failure yields an empty mapping rather than raising, mirroring
    ``search_index._load``: this is a cache whose worst cost must be a rebuild
    (today's full-read behaviour), never a wrong verdict and never the picker.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    try:
        loaded = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(loaded, dict) or loaded.get("version") != ORIGIN_CACHE_VERSION:
        return {}
    entries = loaded.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_origin_cache(path: Path, entries: dict[str, Any]) -> None:
    """Persist the verdicts, best-effort and atomically.

    Atomic with a PID-suffixed temp for the reason ``search_index._save``
    documents: several sessions open a picker at once, and a fixed temp name
    lets one process ``replace`` a document another is still filling. A torn
    document is discarded by the loader, so the bound is a needless rebuild.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps({"version": ORIGIN_CACHE_VERSION, "entries": entries}),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        return


def recent_sessions(config_dir: Path, limit: int | None = None) -> list[tuple[str, float]]:
    """``(id, mtime)`` for the USER's resumable sessions, newest first.

    ``limit=None`` means NO TRUNCATION and is the default, so a caller that says
    nothing gets the whole store. That direction is deliberate and was learned
    the expensive way: this defaulted to ``10`` while the picker called it with
    no argument, and the "uncapped" picker silently showed ten rows on a
    236-session store — a worse version of the bug the change was written to
    fix. A default that truncates makes forgetting to pass a limit look like
    working code, so the safe default is the complete answer and every caller
    that wants less has to say so at its own call site, where a reader can see
    it.

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

    The traversal is ``os.scandir``-based over the store, with one ``stat`` per
    directory for the transcript and one for the marker. The store is scanned
    once rather than each directory being scanned individually: the latter is
    what the origin design proposed, and it measures ~2x WORSE (1986 ms vs
    1127 ms over 31,700 dirs) because it stats every entry in every directory
    to learn two filenames. Do not "fix" it back.

    The cost that matters is the ORIGIN check, which runs once per directory
    and therefore scales with the SUBAGENT population — ~10.6x the user
    population on the reporting machine, and the part that actually grows.
    Because a marker that EXISTS must still be read and parsed (see
    :data:`ORIGIN_CACHE_NAME`), that is one file read per subagent directory:
    1127 ms over 31,700 dirs, of which the reads are 639 ms. The verdict cache
    is what removes it, taking the same scan to ~310 ms warm
    (``bench/resume-picker-after.json``; independently re-measured at 307 ms in
    agent review round 1). Quote the committed bench figure here rather than a
    remembered one — an optimistic number in a docstring is how the next
    person's regression looks like an improvement.

    ``limit`` truncates the RESULT, never the work: every directory is visited
    regardless, so a caller asking for all sessions costs the same as one
    asking for ten.
    """
    return [
        (name, mtime) for name, mtime, _origin in _recent_sessions_with_origin(config_dir, limit)
    ]


def _recent_sessions_with_origin(
    config_dir: Path, limit: int | None = None
) -> list[tuple[str, float, str]]:
    """:func:`recent_sessions`, plus the ``origin`` this scan already parsed.

    The scan reads and parses every marker that exists in order to decide
    visibility, then threw that verdict away — so a caller needing the origin
    (the picker, to mark a fork) re-opened the same file per row. Returning it
    costs nothing: the read has happened, and for the common unmarked session
    the value is ``""`` with no syscall added at all.

    Private because the public pair is what every other caller wants and the
    CLI's recovery listing pins its shape.
    """
    # Lazy and stdlib-only on the other side: ``retention`` imports nothing
    # heavier than ``logging``, and the CLI startup guard measures this
    # module's import, not this function's.
    from local_operator.session.retention import session_activity

    rows: list[tuple[str, float, str]] = []
    try:
        scan = os.scandir(config_dir / "sessions")
    except OSError:
        return []
    cache_path = origin_cache_path(config_dir)
    cached = _load_origin_cache(cache_path)
    fresh: dict[str, Any] = {}
    # Every name that carried a marker in THIS scan. The cache is rewritten to
    # exactly this set, which is what drops entries for disposed sessions and
    # keeps the file bounded by the live store rather than by every session
    # that has ever existed.
    seen: set[str] = set()
    with scan:
        for entry in scan:
            # ONE ranking clock, shared with the cleanup policy
            # (``session.retention.session_activity``): the picker's "most
            # recent" and the policy's "most recent" must be the same
            # directories, or the policy removes rows the picker shows
            # (QA round 1 Q2, UX round 2 U11). A directory with no activity
            # is not a resumable session and gets no row.
            activity = session_activity(Path(entry.path))
            if activity is None:
                continue
            mtime = activity
            # After the transcript stat, not before: the stat is what proves the
            # directory is a session at all, and an unreadable marker must not
            # cost a row.
            #
            # This stat does double duty — it answers "is there a marker" AND
            # produces the cache key — so the cache costs no extra syscall.
            marker = os.path.join(entry.path, ORIGIN_NAME)
            try:
                marker_stat: os.stat_result | None = os.stat(marker)
            except OSError:
                # No marker, or it cannot be stat'd. ABSENCE means the user's
                # own session and is deliberately NOT cached: it is already the
                # cheap path, and a directory the backfill stamps later must be
                # re-read rather than answered from a stale "unmarked" fact.
                marker_stat = None
            if marker_stat is not None:
                seen.add(entry.name)
                key = [marker_stat.st_mtime, marker_stat.st_size]
                previous = cached.get(entry.name)
                if (
                    isinstance(previous, dict)
                    and previous.get("key") == key
                    and isinstance(previous.get("origin"), str)
                ):
                    origin = previous["origin"]
                else:
                    # Existence gates the READ, never the verdict: the file is
                    # read and PARSED, because ``session_origin`` returns "" for
                    # a truncated or hand-edited sidecar so a CORRUPT marker
                    # reads as the user's own session rather than vanishing from
                    # the picker. Treating "file exists" as "subagent" would
                    # invert that fail-safe and hide real work.
                    origin, readable = _session_origin_read(Path(entry.path))
                    # Only a verdict PARSED off a marker that was actually read
                    # is memoised. A read failure yields the same "" as a
                    # corrupt payload — safe for the listing, which shows the
                    # session — but it describes the moment, not the file, and
                    # the key is the marker's immutable (mtime, size): caching
                    # it would serve one transient EMFILE or volume blip as a
                    # permanent wrong verdict for the life of that marker. So it
                    # falls through and is re-read on the next scan instead.
                    if readable:
                        fresh[entry.name] = {"key": key, "origin": origin}
                    else:
                        # Drop any entry inherited from ``cached``: this scan
                        # could not confirm it, and ``merged`` below is built
                        # from the names seen here.
                        seen.discard(entry.name)
                # The same verdict :func:`is_user_session` reaches, spelled out
                # here rather than delegated because this loop must not pay a
                # second stat per directory to re-read the marker it just read.
                # It is therefore the ONE place that has to be kept in step with
                # that predicate by hand — ``USER_ORIGINS`` is the shared fact
                # both consult, so a new user-visible origin is added there once
                # rather than in two places that can drift.
                if origin and origin not in USER_ORIGINS:
                    continue
            else:
                # No marker: the user's own session, and the cheap path this
                # scan is careful to keep free of reads.
                origin = ""
            rows.append((entry.name, mtime, origin))
    merged = {
        name: entry for name, entry in cached.items() if name in seen and isinstance(entry, dict)
    }
    merged.update(fresh)
    # Written only when it would actually change, so a steady store's picker
    # open stays read-only: an unconditional save would rewrite a multi-megabyte
    # file on every open to persist nothing.
    if merged != cached:
        _save_origin_cache(cache_path, merged)
    # Newest first; EQUAL stamps break on the id, ascending, so the order is
    # a property of the store rather than of ``scandir`` on this filesystem.
    # The cleanup policy sorts on the same key: with an unstable tie order
    # the policy's "first page" and the picker's disagreed on a store of
    # equal stamps, and each launch shaved one more session (QA round 2,
    # Q10).
    rows.sort(key=lambda row: (-row[1], row[0]))
    # Sliced only when a limit was actually asked for: ``rows[:None]`` would
    # also return everything, but spelling it out keeps "no limit" a decision
    # the code states rather than a property of slice syntax.
    return rows if limit is None else rows[:limit]


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
    #: True while this session is a FORK still wearing the title it inherited
    #: from its parent, so the picker can tell the branch from the trunk.
    #:
    #: Without it a fresh fork and its parent are byte-identical rows — same
    #: name, same "just now" — separable only by a 12-hex id, in exactly the
    #: window where a user is most likely to be looking for one of them. The
    #: state is not always brief either: a bare ``/fork`` keeps the borrowed
    #: title until the user sends it something, which may be never.
    #:
    #: Defaulted so every existing construction site keeps working; only the
    #: picker's row builder sets it.
    forked: bool = False

    # -- live state, supplied by the CALLER -------------------------------
    # This module stays stdlib-only and never scans the registry itself: it
    # sits on the CLI startup path, and `lop --resume` must not pay for a
    # record walk. The picker does one ``registry.scan()`` and one
    # ``wakes.store.read_index()`` when it opens and fills these in; every
    # other construction site keeps the defaults and renders exactly as before.

    #: ``"busy"`` (a turn is running), ``"idle"`` (resident, warm),
    #: ``"attached"`` (another terminal is watching), ``"wedged"`` (a live pid
    #: whose heartbeat went stale), or ``""`` for a cold session.
    live_state: str = ""
    #: ``"approval"`` / ``"ask"`` when the session is waiting for a PERSON.
    #: The needs-you marker, and the reason a row sorts first.
    pending: str | None = None
    #: How many wakes are scheduled, and whether they are dormant because the
    #: session was deliberately stopped.
    wakes: int = 0
    wakes_dormant: bool = False


#: The fork tag's text as a FILTER sees it. The mark itself is drawn per
#: surface (``session_picker.FORK_MARKER`` in the TUI, the phone's list
#: renderer on mobile); this is the one spelling every one of them searches by.
FORK_HAYSTACK = "[fork]"


def fork_haystack(row: SessionRow) -> str:
    """``row``'s searchable text, including the fork tag when it wears one.

    Every surface splices the tag in at RENDER time, so without this a user who
    reads ``[fork]`` on screen and types it into the filter gets zero rows back
    — a picker reporting "no matches" about a store full of visibly marked
    forks, which reads as a broken filter rather than as an unsupported query.
    A filter has to hold the invariant that what is displayed is matchable.

    Lives HERE, beside :class:`SessionRow`, rather than in the TUI picker that
    first needed it, because the phone's session search matches on the same
    rows and must not disagree about what a row's text is — and importing the
    picker into ``mobile.daemon`` to share one expression would pull Textual
    into the daemon's import graph for a string join.
    """
    return f"{FORK_HAYSTACK} {row.name}" if row.forked else row.name


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

    **The title sidecar is consulted FIRST** (``title.json``, written by
    :func:`write_session_title` on the same event that journals the title to
    the transcript). It is one stat and a sub-kilobyte read, O(1) in transcript
    size, and it is what closes the window-scan gap :data:`TITLE_SCAN_BYTES`
    describes: a title in the untouched middle of a multi-megabyte transcript
    is invisible to the two windows but sits in the sidecar. The scan below
    remains the fallback for sessions written before the sidecar existed and
    not yet reached by :func:`backfill_session_titles`.
    """
    sidecar = _read_title_sidecar(session_dir)
    if sidecar is not None and sidecar.text:
        return sidecar.text
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


def session_preview(session_dir: Path, *, max_chars: int = PREVIEW_MAX_CHARS) -> str:
    """The session's most recent ASSISTANT reply, condensed for a list row.

    The conversation-list counterpart to :func:`session_name`: the name says
    what a conversation is about, the preview says where it got to.

    Canonical sessions keep their conversation in ``transcript.jsonl`` and never
    write the legacy agent record's ``last_message`` field, so a list rendering
    that field showed "No messages yet" against conversations with a full
    transcript on disk — a false statement about the user's own data, sitting
    inches from the timestamp of the very message it denied (design D19).
    Reading the transcript makes the durable conversation the ONE authority for
    both facts.

    Bounded like the name scan and tolerant for the same reasons, but it reads
    the TAIL rather than the head: the newest entry is the last line. A
    transcript shorter than the window is read whole; a longer one is seeked to
    its final :data:`PREVIEW_SCAN_BYTES`, whose first line is dropped because a
    seek to a byte offset lands mid-line.

    An assistant entry with no text — a turn that only made tool calls — is
    skipped rather than previewed as an empty string, so the row shows the last
    thing the model actually SAID. Returns ``""`` when the transcript is
    missing, unreadable, or contains no assistant text, and the caller renders
    its own empty state.
    """
    transcript = session_dir / TRANSCRIPT_NAME
    try:
        size = transcript.stat().st_size
        with transcript.open("rb") as handle:
            if size > PREVIEW_SCAN_BYTES:
                handle.seek(size - PREVIEW_SCAN_BYTES)
                window = handle.read()
                # The seek landed at an arbitrary byte, so the first line is a
                # fragment. Unlike the name scan there is nothing to recover
                # from it: the newest entry is at the other end.
                _, _, window = window.partition(b"\n")
            else:
                window = handle.read()
    except OSError:
        return ""
    for line in reversed(window.decode("utf-8", "replace").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            # Normal for a live session: the writer appends and we may read
            # mid-write, so the final line can be half-written.
            continue
        if not isinstance(entry, dict) or entry.get("type") != "message":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict) or payload.get("role") != "assistant":
            continue
        text = _first_text(payload.get("content"))
        if text.strip():
            return _condense(text, max_chars)
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


def recent_session_rows(config_dir: Path, limit: int | None = None) -> list[SessionRow]:
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

    ``limit=None`` means NO TRUNCATION, matching :func:`recent_sessions` — see
    its docstring for why the untruncated answer is the DEFAULT rather than the
    opt-in. The ``/resume`` picker relies on that default; every caller that
    wants a short list passes its own number at its own call site (the CLI's
    recovery listing, the mobile daemon's history and search), so the cap is
    visible where it is chosen instead of hiding in this signature.

    Uncapping is affordable because the scan underneath is limit-independent
    (see :func:`recent_sessions`) and the only per-row cost added is
    :func:`session_name`, one bounded head read.

    **The fork mark costs nothing on an unmarked session.** The scan already
    parsed every ``origin.json`` that exists, so the verdict is threaded out of
    it (:func:`_recent_sessions_with_origin`) rather than re-read here; a
    session with no marker — the overwhelming majority — adds no syscall at
    all, and only a row already known to be a FORK pays the title probe. An
    earlier revision asked ``wears_inherited_title`` per row unconditionally,
    which attempted two reads per row on a store containing zero forks (the
    absence was discovered from the ``OSError``) and measured +52% on a
    3,000-session store, on this synchronous UI-thread path. That is the exact
    "unmarked is the cheap path" property :func:`recent_sessions` documents at
    length, and it must not be given back here.
    """
    rows: list[SessionRow] = []
    for session_id, mtime, origin in _recent_sessions_with_origin(config_dir, limit):
        session_dir = config_dir / "sessions" / session_id
        rows.append(
            SessionRow(
                session_id,
                mtime,
                session_name(session_dir),
                # Gated on the origin the scan ALREADY parsed. Non-forks — every
                # ordinary conversation — short-circuit here without touching
                # the disk again.
                forked=origin == ORIGIN_FORK and wears_inherited_title(session_dir),
            )
        )
    return rows


def wears_inherited_title(session_dir: Path) -> bool:
    """True while a FORK is still displaying the title it inherited.

    The marker is about the AMBIGUOUS STATE, not about ancestry: a fork that has
    named itself is a conversation in its own right and tagging it forever would
    be noise on every row it ever appears in. So this asks the same question
    ``Session._is_unnamed_fork`` asks at boot — is the newest journalled title
    older than the fork instant — and answers False as soon as the fork writes
    its own name.

    Read from the sidecar rather than the transcript so the picker keeps its
    one-bounded-read-per-row cost model; a fork always has the sidecar, because
    the clone copies it precisely so the row is never blank.
    """
    forked_at = _fork_instant(session_dir)
    if forked_at is None:
        return False
    sidecar = _read_title_sidecar(session_dir)
    if sidecar is None or not sidecar.text:
        # A fork of a NEVER-NAMED parent is still ambiguous, and this used to
        # return False on the reasoning that nothing was inherited. That was
        # wrong: ``session_name`` falls back to the transcript's opening
        # message, and the clone copies the transcript — so the fork displays
        # the identical opener beside its parent, which is exactly the
        # duplicate-row confusion the mark exists to resolve. It has no title
        # of its own yet by definition, so it is still borrowing.
        return True
    try:
        stamped = (session_dir / TITLE_SIDECAR_NAME).stat().st_mtime
    except OSError:
        return False
    # The sidecar is rewritten when this session names itself, so a stamp newer
    # than the fork means the title on show is its own.
    return stamped <= forked_at


def _fork_instant(session_dir: Path) -> float | None:
    """``forked_at`` from the origin marker, or None when this is not a fork.

    Duplicated in spirit with ``fork.fork_instant`` and deliberately NOT
    imported from it: this module is import-guarded (see the module docstring)
    and ``fork`` imports ``shutil``/``uuid`` plus the retention module, which
    the CLI's ``--resume`` path must not acquire. The payload is three keys and
    the reader is five lines; the import edge would cost more than the copy.
    """
    try:
        raw = (session_dir / ORIGIN_NAME).read_text(encoding="utf-8", errors="replace")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("origin") != ORIGIN_FORK:
        return None
    forked_at = payload.get("forked_at")
    return float(forked_at) if isinstance(forked_at, (int, float)) else None
