"""Branching one session's conversation into a new one on disk.

The ACTION half of forking. ``resume.py`` next door is deliberately import-free
policy about *which* directory a resume reopens; this module performs an
operation *on* directories, and the two are kept apart so neither grows the
other's concerns. Same import discipline as ``resume.py`` though, and for the
same measured reason: the CLI resolves ``--resume`` before it starts anything,
and ``tests/unit/test_import_graph.py`` FAILS if that path gains an edge into
the engine, the providers, or asyncio. Nothing here imports beyond the stdlib.

WHY THE CLONE IS A FILE COPY AND NOT A CONTEXT SNAPSHOT
-------------------------------------------------------
The transcript file is the artifact a resume replays
(``Transcript.build_llm_history``), while the live ``_context.messages`` list is
*not always legal on the wire*: for the whole duration of a tool batch it ends
in an assistant message whose ``tool_calls`` have no answers, which is why
``Session._wire_legal_snapshot`` exists at all. Copying the file inherits the
replay's compaction and prune handling for free and avoids reproducing that
repair. The corollary is that WHEN the copy is taken is load-bearing rather than
a formality — a clone taken mid-batch would carry a persisted assistant
``tool_use`` with no ``tool_result``, and the fork's first request would 400 in
a different window minutes later, about the worst diagnostic distance a bug can
have. The boundary that guarantees this is in ``Session`` (see ``request_fork``);
this module is only correct when called from one.

THE COPY IS AN EXPLICIT ALLOW-LIST, NEVER A DIRECTORY COPY
----------------------------------------------------------
:data:`COPIED_SIDECARS` names every file a fork inherits. A ``shutil.copytree``
would be one line shorter and would copy ``.session.pid`` — the liveness marker
— which makes ``live_session_owner`` report the fork as owned by the PARENT's
pid, so the fork's own boot takes the *attach* path instead of opening its
conversation. That failure is spectacular and silent, so the allow-list is
pinned by a set-equality test rather than by review attention.

ATTACHMENTS NEED NO WORK, WHICH IS NOT OBVIOUS
----------------------------------------------
Images live in the content-addressed store at ``<config>/attachments/<digest>.bin``,
shared by every session and never deleted per-session, so the copied
transcript's references resolve unchanged in the fork. The obvious worry — "do
the fork's images survive?" — has a non-obvious answer, hence this note.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from pathlib import Path

from local_operator.resume import (
    ATTACHMENT_SIDECAR_NAME,
    ORIGIN_FORK,
    TITLE_SIDECAR_NAME,
    TRANSCRIPT_NAME,
    mark_session_origin,
)

logger = logging.getLogger(__name__)

#: The boot-prompt sidecar: the message ``/fork <message>`` injects as the
#: forked session's first user turn.
#:
#: A sidecar and NOT a CLI flag, which is a safety decision rather than a
#: stylistic one. ``multiplexer.broadcast.resume_argv`` is documented as a
#: safety boundary — the argv it builds replays a transcript and then waits for
#: the user, carrying no prompt and nothing that continues an interrupted turn,
#: precisely so an unattended crash-restore of a dozen panes cannot resume a
#: dozen agents mid-tool. Every spawn backend builds its command from that one
#: function so no call site can opt out. A ``--prompt`` flag would make exactly
#: the argv shape that invariant exists to render unconstructable, and it would
#: then be one copy-paste from a crash-restore binding. Two lesser reasons that
#: point the same way: user text on an argv needs correct quoting on five spawn
#: backends instead of zero, and argv is world-readable in ``ps``.
BOOT_PROMPT_NAME = "boot-prompt.json"

#: Schema version for :data:`BOOT_PROMPT_NAME`. Present so a future field can be
#: added without a second format, matching the title/attachment sidecars.
BOOT_PROMPT_VERSION = 1

#: One-shot context-tail marker consumed by ``Session`` before the fork's first
#: request. It is separate from the optional boot prompt because a BARE fork
#: still needs the model-visible lineage boundary on the first instruction the
#: user eventually types, while remaining idle until then.
FORK_BOUNDARY_NAME = "fork-boundary.json"
FORK_BOUNDARY_VERSION = 1

#: The exact non-persistent instruction appended after inherited history. Keep
#: this stable: the inherited transcript remains byte-identical and cacheable;
#: only this new tail distinguishes the branch from the still-running parent.
FORK_BOUNDARY_INSTRUCTION = (
    "<fork-boundary>\n"
    "This session is a fork of an existing conversation. Work inherited in the "
    "transcript may still be continuing in the original session; do not continue "
    "or duplicate it here on your own. Wait for and follow the next divergent "
    "instruction given in this fork.\n"
    "</fork-boundary>"
)

#: Files a fork inherits from its parent, and nothing else. Ordered by what they
#: are FOR, because each entry earns its place differently:
#:
#: - ``transcript.jsonl`` is the conversation itself — the whole point.
#: - ``attachment.json`` carries the ``/team``, ``/agent`` and ``/goal`` the
#:   parent was wearing. Copying it is what makes a fork of a team session still
#:   a team session, and it is ALSO a cache-warmth requirement: the agent
#:   profile's instructions ride system block ``[0]``, so a fork that lost its
#:   profile would rebuild the cached prefix differently and miss the provider
#:   cache on its very first request. Persona continuity and cache continuity
#:   are the same requirement here.
#: - ``title.json`` carries the title in force plus every name the session has
#:   borne, so the fork's picker row is labelled instantly rather than by a
#:   transcript window scan, and the fork starts under the parent's name until
#:   the ordinary retitle path moves it.
COPIED_SIDECARS: tuple[str, ...] = (
    TRANSCRIPT_NAME,
    ATTACHMENT_SIDECAR_NAME,
    TITLE_SIDECAR_NAME,
)

#: Files that exist in a parent directory and must NOT reach the fork. Not read
#: by the copy (which is an allow-list and needs no deny-list to be correct) —
#: this is documentation with a test attached, so the REASON each file is
#: excluded survives the next person who reaches for ``copytree``.
#:
#: - ``.session.pid``: the liveness marker. See the module docstring.
#: - ``subagent-roster.v1.json``: rows for jobs owned by the PARENT's process
#:   and its ``JobManager``. A fork inheriting them would list children it
#:   cannot peek, steer, cancel or resume, because ``hub`` resolves jobs against
#:   its own comms registry. An empty roster is correct: the fork has launched
#:   nothing. The parent keeps its children, untouched.
#: - ``origin-scan.json`` / ``title-scan.json``: per-directory backfill
#:   sentinels meaning "a sweep has already considered this directory". Copying
#:   one into a directory no sweep has ever seen is a lie that suppresses a
#:   future backfill.
EXCLUDED_SIDECARS: tuple[str, ...] = (
    ".session.pid",
    "subagent-roster.v1.json",
    "origin-scan.json",
    "title-scan.json",
)


class ForkError(RuntimeError):
    """The clone itself failed, so no fork exists.

    Its own type because callers must tell this apart from a failed *spawn*:
    a spawn failure means "your fork is waiting for you and here is its id",
    while this means "there is no fork". Presenting them with the same weight
    is how a user goes looking for a session that was never created.
    """


def new_session_id() -> str:
    """Mint a session id, the same way ``session_factory`` does.

    Deliberately NOT prefixed with ``fork-``. ``resume_dir`` validates that an
    id is a single path component and every other reader treats the name as
    opaque, so a prefix would make forks visually distinct in exactly one place
    (a directory listing) while minting a second id shape that every future
    reader has to handle.
    """
    return uuid.uuid4().hex[:12]


def fork_session(config_dir: Path, parent_id: str, *, message: str = "") -> str:
    """Clone ``parent_id``'s conversation into a fresh session. Returns its id.

    Synchronous and blocking-by-design at the file level, so callers on an event
    loop should run it through ``asyncio.to_thread`` — not because it is slow
    (the largest transcript in a real store measured 216 KB, a sub-millisecond
    ``copyfile``) but because a TUI must never do disk I/O of unbounded size on
    the loop, which is how ``session_factory`` treats every store walk.

    **The parent directory is not touched at all** — no truncation, no marker,
    no mtime change. That invariant holds trivially because nothing below writes
    to the parent, and it is what makes a fork safe to take from a conversation
    the user is still working in.

    Raises :class:`ForkError` when the clone cannot be made (a read-only volume,
    ENOSPC, a parent that does not exist). Nothing is created in that case.
    """
    parent_dir = config_dir / "sessions" / parent_id
    parent_transcript = parent_dir / TRANSCRIPT_NAME
    if not parent_transcript.is_file():
        raise ForkError(f"session {parent_id} has no transcript to fork")

    new_id = new_session_id()
    fork_dir = config_dir / "sessions" / new_id

    # CLAIM BEFORE mkdir, and in that order — a hard requirement, not style.
    # ``claim_session`` creates the directory AND writes the liveness marker in
    # one step precisely so there is no instant at which an empty unclaimed
    # directory exists for a concurrent session's retention sweep to reap. A
    # fork that mkdir'd first would reopen exactly that window, and the failure
    # (a FileNotFoundError raised in a DIFFERENT process) is near-undiagnosable.
    # Imported function-locally to keep this module's import graph stdlib-only
    # for the CLI startup path the module docstring describes.
    from local_operator.session.retention import claim_session, release_session

    try:
        claim_session(fork_dir)
        fork_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ForkError(f"cannot create the fork's session directory: {exc}") from exc

    try:
        for name in COPIED_SIDECARS:
            source = parent_dir / name
            if not source.is_file():
                # Only the transcript is mandatory (checked above); the identity
                # sidecars are absent for any session that was never named and
                # never wore a persona, which is the ordinary case for a young
                # conversation rather than an error.
                continue
            # copyfile and NOT a read-parse-write: the fork's first request must
            # reproduce the parent's cached prefix byte-for-byte, and a
            # re-serialisation would reorder JSON keys — changing nothing
            # semantically while changing every byte the replay derives from.
            shutil.copyfile(source, fork_dir / name)
    except OSError as exc:
        raise ForkError(f"cannot copy the conversation into the fork: {exc}") from exc

    # Written for EVERY fork, including a bare one. The marker is consumed into
    # memory by the fork's first ``Session`` construction and never enters the
    # transcript, preserving both the parent's bytes and its prompt-cache prefix.
    _write_json_sidecar(
        fork_dir / FORK_BOUNDARY_NAME,
        {"version": FORK_BOUNDARY_VERSION, "created_at": time.time()},
    )
    if message.strip():
        write_boot_prompt(fork_dir, message)

    # RELEASE THE CLAIM. The claim above is held only for the instant between
    # creating the directory and filling it, which is the window a concurrent
    # retention sweep could otherwise reap; by here the transcript is on disk
    # and the directory is no longer empty, so the claim has done its job.
    #
    # Releasing is NOT tidiness — leaving it is a correctness bug, and a
    # spectacular one. ``claim_session`` stamps ``os.getpid()``, and this
    # function runs inside the PARENT's TUI process, so the marker left behind
    # names the parent as the fork's live owner. The fork's own boot then reads
    # it through ``live_session_owner`` and either refuses outright ("session
    # <id> is open in an older Local Operator process") or, when a discovery
    # record exists, attaches the new window as a FOLLOWER of its parent
    # instead of opening the branched conversation. That is exactly the failure
    # the sidecar allow-list exists to prevent, arriving through a different
    # door than ``copytree`` — the module docstring above describes the
    # consequence, and the claim reintroduced the cause. Guarded by
    # ``live_session_owner(config_dir, fork_id) is None`` in the tests, which is
    # the property that actually matters; asserting the file's absence alone
    # would not survive a future claim written by some other path.
    release_session(fork_dir)

    # Provenance last. ``mark_session_origin`` preserves the directory's mtime,
    # which is moot here (the directory was just created) but means the ordering
    # of this call carries no constraint.
    #
    # ``forked_at`` is a FLOAT epoch, deliberately, and its precision is
    # load-bearing rather than incidental: ``Session._is_unnamed_fork`` compares
    # it against the timestamp of the newest journalled title to tell the
    # PARENT's inherited name from one the fork has since chosen for itself.
    # Truncating to whole seconds would make a title the parent wrote earlier in
    # the same second look like it was written after the fork, and the branch
    # would silently wear its parent's name.
    mark_session_origin(fork_dir, ORIGIN_FORK, parent=parent_id, forked_at=time.time())
    return new_id


def _write_json_sidecar(path: Path, payload: dict[str, object]) -> None:
    """Best-effort atomic-enough JSON write for disposable fork boot state."""
    try:
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        logger.debug("fork: cannot write sidecar %s", path, exc_info=True)


def consume_fork_boundary(session_dir: Path) -> str:
    """Consume the fork's one-shot model boundary, returning its instruction.

    Unlink before returning for the same safer-than-repetition rule as the boot
    prompt: a crash after construction may lose the boundary, but a resume must
    never inject it repeatedly into an established fork.
    """
    sidecar = session_dir / FORK_BOUNDARY_NAME
    try:
        raw = sidecar.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    finally:
        try:
            sidecar.unlink()
        except OSError:
            pass
    try:
        payload = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(payload, dict) or payload.get("version") != FORK_BOUNDARY_VERSION:
        return ""
    return FORK_BOUNDARY_INSTRUCTION


def write_boot_prompt(session_dir: Path, text: str) -> None:
    """Park ``text`` as the session's first user turn, for its next boot.

    Best-effort by contract, like every other sidecar write in this store: the
    fork's whole conversation is already on disk by the time this runs, and
    failing the fork because its opening message could not be parked would trade
    a recoverable annoyance (retype one sentence) for the loss of the branch.
    """
    payload = {
        "version": BOOT_PROMPT_VERSION,
        "text": text,
        "created_at": time.time(),
    }
    _write_json_sidecar(session_dir / BOOT_PROMPT_NAME, payload)


def consume_boot_prompt(session_dir: Path) -> str:
    """Read and REMOVE the boot prompt, returning it (``""`` when there is none).

    **The delete is load-bearing, not tidiness.** The injected message must fire
    exactly once: a fork the user later ``/resume``s, or one brought back by a
    crash restore, must replay its transcript and idle like every other session
    rather than re-running the instruction that opened it. Consuming the sidecar
    on the FIRST boot is what makes that true by construction instead of by a
    flag someone has to remember to clear.

    Unlinked BEFORE the caller submits, so a crash between the two loses the
    message rather than repeating it — the safer direction, and the same
    restore-and-idle posture ``resume_argv`` enforces on the argv.

    Tolerant of a corrupt or truncated file, for the reason the title and
    attachment readers give: this runs on a BOOT path, and losing an injected
    message is a notice, while failing to start is not survivable.
    """
    sidecar = session_dir / BOOT_PROMPT_NAME
    try:
        raw = sidecar.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    finally:
        # Unlink whatever was there, readable or not: a file that cannot be
        # parsed must not be retried on every subsequent boot of this session.
        try:
            sidecar.unlink()
        except OSError:
            logger.debug("fork: cannot clear the boot prompt at %s", sidecar, exc_info=True)
    try:
        payload = json.loads(raw)
    except ValueError:
        return ""
    if not isinstance(payload, dict):
        return ""
    text = payload.get("text")
    return text if isinstance(text, str) else ""


def fork_instant(session_dir: Path) -> float | None:
    """When this session was forked, or ``None`` when it is not a fork.

    The float epoch :func:`fork_session` stamped. Read by
    ``Session._is_unnamed_fork`` to tell a title that came across in the clone
    from one this fork has since chosen for itself — which is why the stamp is
    written with sub-second precision.

    ``None`` for a fork whose marker predates the stamp or cannot be parsed,
    which the caller treats as "not a fork": that preserves the ordinary restore
    behaviour rather than suppressing a title on a guess.
    """
    payload = _origin_payload(session_dir)
    if payload is None:
        return None
    forked_at = payload.get("forked_at")
    return float(forked_at) if isinstance(forked_at, (int, float)) else None


def _origin_payload(session_dir: Path) -> dict[str, object] | None:
    """The origin marker's payload IF this session is a fork, else ``None``.

    Tolerant in the same way ``session_origin`` is: an unreadable or corrupt
    marker means "nothing known", never an exception on a construction path.
    """
    from local_operator.resume import ORIGIN_NAME

    try:
        raw = (session_dir / ORIGIN_NAME).read_text(encoding="utf-8", errors="replace")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("origin") != ORIGIN_FORK:
        return None
    return payload


def fork_parent(session_dir: Path) -> str:
    """The id this session was forked from, or ``""`` when it is not a fork.

    Reads the origin payload rather than a second marker file, which is what
    ``mark_session_origin``'s free-form ``**details`` is for. Tolerant in the
    same way ``session_origin`` is: an unreadable marker means "no known
    parent", never an exception on a startup path.
    """
    payload = _origin_payload(session_dir)
    if payload is None:
        return ""
    parent = payload.get("parent")
    return parent if isinstance(parent, str) else ""
