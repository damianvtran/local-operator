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

THE COPY IS AN EXPLICIT ALLOW-LIST, PLUS ONE NAMED DIRECTORY
------------------------------------------------------------
:data:`COPIED_SIDECARS` names every FILE in the session root a fork inherits. A
``shutil.copytree`` of the session directory would be one line shorter and would
copy ``.session.pid`` — the liveness marker — which makes ``live_runtime_pid``
report the fork as owned by the PARENT's pid, so the fork's own boot takes the
*attach* path instead of opening its conversation. That failure is spectacular
and silent, so the allow-list is pinned by a set-equality test rather than by
review attention.

The one directory copied is ``roles/`` — the conversation-scoped agent and team
definitions (``profile_scope``). It is copied by :func:`_copy_roles_tree`, by
name, into the FORK's own directory: a fork is a new conversation, so it must
be able to resolve the persona its inherited ``attachment.json`` names without
reaching back into the parent. Nothing else is copied wholesale, and no global
root is ever a source — the parent's ``roles/`` is the entire input.

ATTACHMENTS NEED NO WORK, WHICH IS NOT OBVIOUS
----------------------------------------------
Images live in the content-addressed store at ``<config>/attachments/<digest>.bin``,
shared by every session and never deleted per-session, so the copied
transcript's references resolve unchanged in the fork. The obvious worry — "do
the fork's images survive?" — has a non-obvious answer, hence this note.
"""

from __future__ import annotations

import errno
import json
import logging
import math
import os
import shutil
import stat
import time
import uuid
from pathlib import Path
from typing import Any

from local_operator.resume import (
    ATTACHMENT_SIDECAR_NAME,
    GOAL_SIDECAR_NAME,
    ORIGIN_FORK,
    TITLE_SIDECAR_NAME,
    TRANSCRIPT_NAME,
    mark_session_origin,
    read_goal_record,
    write_goal_record,
)

#: The ownership stamp's name, imported rather than spelled again: the file this
#: deny-list excludes and the file the resolver reads must be the same string, and
#: a second literal here is how they would come apart (§1.2). ``placement`` is
#: stdlib-only, so this keeps the module's no-beyond-stdlib import discipline.
from local_operator.session.placement import MESH_STAMP_NAME

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
#: - ``goal.json`` carries the judged-goal record. The child IS a continuation of
#:   the same work, so it inherits the objective, its status and its settled
#:   history — but NOT the live judge, which is rewritten to a fresh state at the
#:   fork boundary (see :func:`_rewrite_fork_goal_judge`).
COPIED_SIDECARS: tuple[str, ...] = (
    TRANSCRIPT_NAME,
    ATTACHMENT_SIDECAR_NAME,
    TITLE_SIDECAR_NAME,
    GOAL_SIDECAR_NAME,
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
#: - ``mesh.json``: the mesh ownership stamp (``session/placement.py``). A copy
#:   would make the FORK claim its parent's ``home_device`` and ``placement`` — a
#:   live session advertising itself as owned by another device, and a row the
#:   resolver (``network/projection.resolve_owner``) then routes to that device.
#:   INV-1 says one device owns a session; a fork with an inherited stamp is the
#:   counterexample that arrives without any move having happened
#:   (``mesh-session-mobility.md`` §1.2).
#: - ``auto-tag.json``: a desktop fork's pending auto-tag intent
#:   (``session/auto_tag.py``). It names the PARENT's own child and carries that
#:   child's assignment revision, so a fork of that fork would either auto-tag a
#:   conversation whose UI did not ask for it, or fail the revision guard against
#:   a row it does not own. Intent is per-fork and re-recorded only when a new
#:   fork request carries ``auto_tag``.
EXCLUDED_SIDECARS: tuple[str, ...] = (
    ".session.pid",
    "subagent-roster.v1.json",
    "origin-scan.json",
    "title-scan.json",
    MESH_STAMP_NAME,
    "auto-tag.json",
)

#: Ceiling on the conversation-scoped definitions a fork will copy.
#:
#: ``roles/`` is a definitions tree, not a data directory: a role is an
#: ``agent.yml`` of a few KiB plus a ``system_prompt.md``, and a team is a
#: ``team.yml`` plus two briefs that ``teams`` caps at a few KiB each. A
#: conversation carrying hundreds of definitions therefore stays well under a
#: megabyte, so 64 MiB is deliberately far past any tree the feature produces.
#: A tree that reaches it is something else that landed in ``roles/`` (a
#: dataset, a model blob, a mis-typed copy), and the copy would run on the
#: PARENT's process, between two turns of a conversation the user is still
#: using — the cost is paid by the session being forked, not by a background
#: job. Refusing with the path and the two ways forward is the cheaper, honest
#: failure. The bound is on FILE BYTES summed before the copy starts, so it
#: refuses before any disk is spent rather than half way through.
MAX_ROLES_COPY_BYTES = 64 * 1024 * 1024


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


def fork_session(
    config_dir: Path,
    parent_id: str,
    *,
    message: str = "",
    exclude_entry_ids: frozenset[str] = frozenset(),
) -> str:
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

    ``exclude_entry_ids`` drops exactly those journal rows from the clone: how a
    fork taken mid-turn leaves the unpaired live suffix behind, and how a fork
    aimed at a named cut point drops everything after it. Truncating the file
    instead would either rewrite the bytes the retained prefix replays from or
    take a torn row.

    The parameter is a filter and the caller owns its correctness: rows NOT in
    the set are copied verbatim. That is a rule about ROWS, not about kinds —
    the caller decides which bookkeeping rows a child must not carry (a
    compaction written after the cut that would take the child's replay back
    past its own cut is excluded like any conversation row), and every row it
    leaves in is preserved byte-for-byte, including the compaction, prune and
    custom rows whose metadata a replay needs.

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
        # A fork inherits journal bytes, never its parent's creation identity.
        from local_operator.session.creation import ensure_session_created_at

        ensure_session_created_at(fork_dir, time.time())
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
            if name == TRANSCRIPT_NAME and exclude_entry_ids:
                # Only rows the owner has already validated as omissible may be
                # omitted: the live suffix that is not yet durably committed, or
                # every conversation row after a named cut point. Keep every
                # OTHER retained byte and journal row, rather than rebuilding
                # history and losing compaction/prune metadata or attachment refs.
                with source.open("rb") as incoming, (fork_dir / name).open("wb") as outgoing:
                    for line in incoming:
                        try:
                            row = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            # Transcript replay skips malformed lines individually.
                            # Keep their bytes, too: trimming a known message id
                            # must not introduce a new parser or repair the source.
                            outgoing.write(line)
                            continue
                        if not isinstance(row, dict) or row.get("id") not in exclude_entry_ids:
                            outgoing.write(line)
            else:
                shutil.copyfile(source, fork_dir / name)
    except OSError as exc:
        raise ForkError(f"cannot copy the conversation into the fork: {exc}") from exc

    # While the claim is still held, and BEFORE the goal judge is rewritten or
    # the boot prompt is written: the roles tree is part of the fork's identity,
    # so a failure here discards the whole directory rather than leaving a fork
    # whose inherited ``attachment.json`` names a persona it cannot resolve.
    try:
        _copy_roles_tree(parent_dir, fork_dir)
    except BaseException:
        # BaseException, not Exception: a cancelled or interrupted copy leaves
        # the same half-fork an I/O error does, and the discard cannot raise
        # (``rmtree(ignore_errors=True)``, a claim release that swallows OSError),
        # so the original exception is the one the caller sees.
        _discard_failed_fork(fork_dir)
        raise

    # A fork's goal is the same work, so the record above is inherited — but its
    # LIVE judge is not, and this is the one place that can be said once for
    # every fork (bare, boot-prompted or spawned).
    _rewrite_fork_goal_judge(fork_dir)

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
    # it through ``live_runtime_pid`` and either refuses outright ("session
    # <id> is open in an older Local Operator process") or, when a discovery
    # record exists, attaches the new window as a FOLLOWER of its parent
    # instead of opening the branched conversation. That is exactly the failure
    # the sidecar allow-list exists to prevent, arriving through a different
    # door than ``copytree`` — the module docstring above describes the
    # consequence, and the claim reintroduced the cause. Guarded by
    # ``live_runtime_pid(config_dir, fork_id) is None`` in the tests, which is
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


def _discard_failed_fork(fork_dir: Path) -> None:
    """Leave no fork behind after a failed clone.

    The claim is released BEFORE the removal: the marker names the PARENT's pid
    (see the release note in :func:`fork_session`), so a removal that only
    partially succeeds must still leave a directory no scanner reads as a live
    session. Both steps are best-effort on purpose — this runs inside the
    ``except`` of the real failure, and a second failure must never replace the
    message the user needs.
    """
    from local_operator.session.retention import release_session

    release_session(fork_dir)
    shutil.rmtree(fork_dir, ignore_errors=True)


#: How much the bounded copy reads per chunk. Small enough that the running
#: total never drifts far past the bound before the next check, large enough
#: that a 64 MiB copy is not syscall-bound.
_COPY_CHUNK_BYTES = 256 * 1024


def _excess_words(excess: int) -> str:
    """A size a refusal can print without rounding itself back to the bound.

    ``:.1f`` printed "64 MiB + 1 byte" as "64.0 MiB … over the 64 MiB", which
    reads as "64 is over 64" (UX U1). The EXCESS is what actually crossed the
    limit, so printing it is both shorter than printing the total and never
    ambiguous at the boundary. Values are FLOORED, not rounded: 1048575 bytes
    is 1023.999 KiB, and rounding printed "1024 KiB" — a number the unit can no
    longer hold (UX U9).
    """
    if excess < 1024:
        return f"{excess} byte" if excess == 1 else f"{excess} bytes"
    if excess < 1024 * 1024:
        return f"{math.floor(excess / 1024 * 10) / 10:.1f} KiB"
    return f"{math.floor(excess / (1024 * 1024) * 10) / 10:.1f} MiB"


def _relative_path(path: str | Path, source: Path) -> str:
    """``path`` relative to the roles folder, or its bare name if outside it.

    A refusal must never print the ephemeral ``.roles-<hex>`` staging directory
    or a second absolute path (UX U6/D1): the relative form is what the user
    can find, and the folder is named once in words.
    """
    try:
        return Path(path).relative_to(source).as_posix()
    except ValueError:
        return Path(path).name


def _errno_of(detail: Any) -> int | None:
    """The errno in an OSError or in ``shutil.Error``'s ``"[Errno N] ..."``."""
    number = getattr(detail, "errno", None)
    if isinstance(number, int):
        return number
    text = str(detail)
    if text.startswith("[Errno "):
        head = text[len("[Errno ") :].split("]", 1)[0]
        try:
            return int(head)
        except ValueError:
            return None
    return None


def _error_reason(detail: Any) -> str:
    """The human half of an I/O failure: no path, no ``[Errno N]``, no repr."""
    text = str(detail)
    # ``copytree`` stores ``str(OSError)``: "[Errno 13] Permission denied: '/p'".
    text = text.split(": '", 1)[0]
    if text.startswith("[Errno "):
        _, _, text = text.partition("] ")
    return (text.strip() or "an I/O error").lower()


def _io_refusal(exc: BaseException, source: Path) -> str:
    """One actionable sentence for a read/copy failure, mirroring the others.

    ``shutil.Error`` carries a LIST of ``(src, dst, why)`` triples, full of
    absolute paths and the staging name; printing it leaked both (review Q2,
    UX U6). The refusal names the first failing source RELATIVE to the roles
    folder and its reason in words, and never mentions the destination.
    """
    failures: list[tuple[Any, ...]] = []
    if isinstance(exc, shutil.Error) and exc.args and isinstance(exc.args[0], list):
        failures = [item for item in exc.args[0] if isinstance(item, tuple) and len(item) >= 3]
    if failures:
        rel = _relative_path(str(failures[0][0]), source)
        detail = failures[0][2]
        more = f", and {len(failures) - 1} more" if len(failures) > 1 else ""
    else:
        filename = getattr(exc, "filename", None)
        rel = _relative_path(str(filename), source) if filename else ""
        detail = exc
        more = ""
    reason = _error_reason(detail) + more
    errno_number = _errno_of(detail)
    if errno_number in (errno.EACCES, errno.EPERM):
        action = f"make {rel or 'the roles folder'} readable"
    elif errno_number in (errno.ENOSPC, errno.EDQUOT):
        action = "free some disk space"
    else:
        action = f"fix {rel or 'the roles folder'}"
    return f"{action} and fork again ({reason}), or keep working in this conversation."


def _bound_refusal(rel: str, excess: int) -> str:
    """The one over-bound sentence, shared by the walk and the live copy."""
    bound = MAX_ROLES_COPY_BYTES // (1024 * 1024)
    return (
        f"move {rel} out of this conversation's roles folder and fork again "
        f"({_excess_words(excess)} over the {bound} MiB a fork copies), or keep "
        "working in this conversation."
    )


def _refuse_entry(entry: Path, source: Path) -> None:
    """Refuse a symlink or a non-regular entry, naming it RELATIVE to the tree.

    A link is refused because ``copytree(symlinks=False)`` copies the TARGET's
    bytes: whatever it points at may belong to neither this conversation nor
    the fork, so neither copying nor skipping it is honest — skipping would
    open the fork with some of this conversation's roles missing, and copying
    would carry a file the conversation never owned. A named pipe, socket or
    device has no bytes to copy at all; without this check the walk would
    measure it as 0 and the failure would surface mid-copy as an opaque
    ``shutil.Error`` (or a FIFO could hang it). Both messages name the file
    relative to the roles folder and say which folder that is (UX U7); the
    symlink one carries no second "keep working" sentence because the refusal
    already leaves the user exactly where they were (design D2).
    """
    rel = _relative_path(entry, source)
    if entry.is_symlink():
        raise ForkError(
            f"replace the link at {rel} in this conversation's roles folder with a real "
            "copy of what it points at, then fork again — a fork does not copy links, "
            "and skipping one would open it with some of this conversation's roles "
            "missing."
        )
    if entry.is_dir():
        return
    if not entry.is_file():
        raise ForkError(
            f"replace {rel} in this conversation's roles folder with a regular file and "
            "fork again — a fork copies files, and skipping this entry would open it "
            "with some of this conversation's roles missing. Or keep working in this "
            "conversation."
        )


def _roles_tree_bytes(source: Path) -> tuple[int, Path | None, int]:
    """Sum the FILE bytes under ``source``, refusing anything uncopyable.

    Returns ``(total, largest_path, largest_bytes)``. The largest file is what
    makes a refusal actionable; the caller names it relative to ``source``.

    Two passes over the tree are deliberate (this one, then the copy): the
    bound has to be decided BEFORE the first byte lands in the fork, or a tree
    past the bound would already have spent the disk and the copy time it was
    meant to prevent. This pass is only a SNAPSHOT, though: a file can grow
    between the walk and the copy, so :func:`_copy_roles_tree` counts every
    byte again while it streams. The live count is what holds the budget.

    ``os.walk(followlinks=False)`` does not descend INTO a symlinked directory,
    but it still lists the link in ``dirnames``, and ``Path.stat`` follows
    links, so every entry is checked before it is measured. See
    :func:`_refuse_entry` for why a link and a non-regular entry are refusals
    rather than skips.
    """
    total = 0
    largest: Path | None = None
    largest_bytes = 0
    for root_name, dirnames, filenames in os.walk(source, followlinks=False):
        root = Path(root_name)
        # A symlinked directory is listed here but never descended into; it is
        # still refused, because its CONTENTS are part of the tree this
        # conversation sees and a fork that skipped it would open with
        # definitions missing.
        for name in dirnames:
            _refuse_entry(root / name, source)
        for name in filenames:
            entry = root / name
            _refuse_entry(entry, source)
            size = entry.stat().st_size
            total += size
            if size > largest_bytes:
                largest, largest_bytes = entry, size
    return total, largest, largest_bytes


def _copy_roles_tree(parent_dir: Path, fork_dir: Path) -> None:
    """Copy ``<parent>/roles`` into the fork as ONE published directory.

    A fork is a new conversation, so it must NOT resolve its inherited persona
    out of the original conversation's definitions: an edit or delete in the
    fork would then change the conversation it was branched from, and vice
    versa. On-disk identity is what the scope readers walk (``<session>/roles``),
    so independence is achieved by the copy itself — separate directories,
    nothing shared, no hardlinks.

    Lazy creation is an S1 invariant: a session that never authored a scoped
    definition has no ``roles/``, and a fork must not manufacture an empty one
    (an empty tree would make every downstream "does this session have a
    scope?" question answer yes where the honest answer is no). Absent is
    therefore a no-op, not an error.

    Written to a temp sibling INSIDE the fork directory and published with one
    ``os.replace``, because the two failure modes that matter are both about
    what a reader can observe mid-copy: an interrupted copy would otherwise
    leave ``roles/`` holding a partial tree, and the inherited attachment would
    then restore a persona whose definitions are half present. The staging name
    is unique per attempt and lives on the fork's own filesystem, which is what
    makes the rename atomic.

    Only ``<parent>/roles`` is read. The device-level stash
    (``<config>/dropped-session-roles``) and every global root
    (``<config>/agents``, ``<config>/teams``) are out of reach by construction:
    the source path is derived from ``parent_dir`` and nowhere else.
    """
    # Imported here, not at module scope: this module's import graph is pinned
    # stdlib-only for the CLI's ``--resume`` path (see the module docstring), and
    # the dirname is shared with the scope readers rather than duplicated.
    from local_operator.profile_scope import SESSION_ROLES_DIRNAME

    source = parent_dir / SESSION_ROLES_DIRNAME
    if not os.path.lexists(source):
        return
    if source.is_symlink() or not source.is_dir():
        raise ForkError(
            "replace the roles folder with a folder of definitions and fork again — "
            f"{source} is not a folder, so this conversation's roles cannot be copied. "
            "Or keep working in this conversation."
        )

    try:
        total, largest, _largest_bytes = _roles_tree_bytes(source)
    except OSError as exc:
        raise ForkError(_io_refusal(exc, source)) from exc
    if total > MAX_ROLES_COPY_BYTES:
        rel = _relative_path(largest, source) if largest is not None else "the largest file"
        raise ForkError(_bound_refusal(rel, total - MAX_ROLES_COPY_BYTES))

    staging = fork_dir / f".roles-{uuid.uuid4().hex[:8]}"
    copied = 0

    def _bounded_copy(src: str, dst: str, *, follow_symlinks: bool = True) -> str:
        """``copy_function`` that counts the bytes it actually streams.

        ``copy2`` follows links and reads a file to its END, so neither a
        pre-copy ``lstat`` nor a single ``stat`` can bound what lands: review
        and QA both measured a link counted at its ~90-byte path length that
        copied a 2 MiB target, and review measured a file that grew between the
        stat and the read. So the count is the STREAM'S: every chunk is checked
        before it is written, and no byte past the budget ever reaches the
        staging tree (the directory is discarded on refusal, so nothing is
        published either). The ``stat`` below follows links, matching what
        ``copy2`` will read, and refuses a non-regular entry that replaced a
        file after the walk rather than blocking on it.
        """
        nonlocal copied
        rel = _relative_path(src, source)
        try:
            info = os.stat(src)
        except OSError as exc:
            raise ForkError(_io_refusal(exc, source)) from exc
        if not stat.S_ISREG(info.st_mode):
            raise ForkError(
                f"replace {rel} in this conversation's roles folder with a regular file "
                "and fork again — a fork copies files. Or keep working in this "
                "conversation."
            )
        if copied + info.st_size > MAX_ROLES_COPY_BYTES:
            raise ForkError(_bound_refusal(rel, copied + info.st_size - MAX_ROLES_COPY_BYTES))
        try:
            with open(src, "rb") as incoming, open(dst, "wb") as outgoing:
                while True:
                    chunk = incoming.read(_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    if copied + len(chunk) > MAX_ROLES_COPY_BYTES:
                        raise ForkError(
                            _bound_refusal(rel, copied + len(chunk) - MAX_ROLES_COPY_BYTES)
                        )
                    copied += len(chunk)
                    outgoing.write(chunk)
            shutil.copystat(src, dst, follow_symlinks=follow_symlinks)
        except OSError as exc:
            raise ForkError(_io_refusal(exc, source)) from exc
        return dst

    try:
        # ``symlinks=False`` keeps the second pass honest in the safe direction:
        # the walk refuses every link, and if one appeared between the two passes
        # the copy takes the TARGET's bytes rather than writing a pointer out of
        # the fork — and counts those bytes, so the bound still holds.
        shutil.copytree(source, staging, symlinks=False, copy_function=_bounded_copy)
        os.replace(staging, fork_dir / SESSION_ROLES_DIRNAME)
    except ForkError:
        # A bound crossed mid-copy (or a refusal from the copy function): the
        # message is already the user-facing one.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except OSError as exc:
        # ``shutil.Error`` subclasses OSError, so a per-file failure that
        # ``copytree`` collected arrives here too. The caller's contract is
        # ``ForkError`` (review MAJOR-1 / QA Q1), and the detail is unwrapped
        # into one sentence (review Q2 / UX U6).
        shutil.rmtree(staging, ignore_errors=True)
        raise ForkError(_io_refusal(exc, source)) from exc
    except BaseException:
        # A cancelled or interrupted copy: discard the staging directory, but
        # keep the original exception type and message.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _rewrite_fork_goal_judge(fork_dir: Path) -> None:
    """Reset an inherited goal record's LIVE judge, keeping the goal itself.

    ``goal.json`` is copied byte-for-byte with the rest of the sidecars, which is
    right for the objective, its status and its settled history — a fork of a
    judged goal IS a continuation of that work, and a child that opened with no
    goal would be a different conversation wearing the parent's transcript.

    What must NOT survive the copy is the judge's live state. Copied verbatim,
    an in-flight ``judging``/``continuing`` record is exactly the shape the
    re-arm rule fires on (a restart that finds one re-arms rather than leaving
    the goal inert), so the child would judge the parent's objective at its first
    turn end — spending tokens on work the parent is still doing, in a fork the
    user may never open.

    The token is RE-MINTED rather than merely reset, and that is the half a
    state reset alone would miss: the staleness guard drops a verdict whose token
    has moved, so a child that kept the parent's token would accept the PARENT's
    in-flight answer as its own.

    No history entry is written: a fork is not a settle, and recording one would
    put a row in every fork's history for work the user never completed.
    """
    payload = read_goal_record(fork_dir)
    if payload is None:
        return
    judge = payload.get("judge")
    payload["judge"] = {
        **(judge if isinstance(judge, dict) else {}),
        "state": "waiting",
        "run": 0,
        "failures": 0,
    }
    if payload.get("goal"):
        payload["token"] = uuid.uuid4().hex
    write_goal_record(fork_dir, payload)


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
