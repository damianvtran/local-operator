"""Session sync (R22): replicas of sessions this device does not own.

TWO SIDES, ONE PRIMITIVE. The owner answers ``net_sync {phase: plan|fetch}`` and
runs a watcher that tells holders when a new cut exists. Every other device is a
HOLDER: it pulls a plan, fetches the miss, and keeps the result in a REPLICA at
``<config>/network/replicas/<id>/`` — outside ``sessions/``, so no scanner,
picker, catalogue, retirement pass or resume can see it. That placement is the
INV-1 argument, not tidiness: a replica under ``sessions/`` would be a second
directory for an id the owner still holds, which is the state the mobility design
exists to make impossible.

THE COPY SET IS THE SPEC OF WHAT A SESSION DIRECTORY MAY HOLD (§7.2), and it is
an ALLOW-LIST of whole entry types rather than a list of files: every name the
product can write into a session directory is either carried (``COPY_SET_NAMES``,
``COPY_SET_TREES``) or excluded with a stated reason (``EXCLUDED_ENTRIES``). A
move DELETES the source directory, so a name nobody added to either list is data
the user loses permanently with nothing to recover it from — which is what
happened to ``scratchpad/`` and ``created_at.json`` (review round 1, B-M2). Two
structural guards keep that from recurring: ``assert_complete`` refuses a commit
whose source directory holds anything neither list accounts for, and
``tests/unit/network/test_sync_copy_set.py`` enumerates the entry types from the
modules that create them and fails when the two lists no longer cover them.

THE REPLICA IS NOT A SESSION. It is a byte copy of the copy set (§7.2) plus
``sync.json`` (who owns it, the cursor, when it last synced). Opening it is
RECOVERY, and recovery promotes it **as a fork with a new id** — never under the
original id, because a peer that spins down can come back, and a same-id
promotion is then two writers on one transcript (build plan §7, unsafe item 4).
That is a deliberate deviation from the design's expected-to-die pool member and
it is argued at :func:`promote_replica`.

WHAT "INCREMENTAL" RESTS ON (§7.3). A transcript is not append-only:
``Transcript.compact_file`` rewrites it with ``os.replace``. So a byte offset is
not a cursor by itself, but the destination's own bytes are a claim that can be
CHECKED, and that is what this module does. A holder's cursor records the length
and the sha256 of the prefix it holds; the owner re-hashes its own first
``prefix_bytes`` and offers an append only when the two agree. A compaction
changes those bytes, so the check fails and the holder is sent the whole file.
The design's ``history_generation`` is deliberately NOT the authority here: it is
``Transcript._history_generation``, in memory, and a COLD session has none — a
cursor that trusted it would read "same generation" for a session nobody has
opened since two compactions. The digest is computed from the file, so it is true
whatever has run.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

logger = logging.getLogger(__name__)

#: ``network.sync.debounce_s``: seconds of quiet after a transcript change before
#: the owner tells replica holders a new cut is available. Long enough that a turn
#: streaming tokens is one push rather than hundreds; short enough that a holder is
#: at most a turn behind. The final message before idle is not subject to it: the
#: runtime's record disappearing pushes immediately (build plan §1.1).
SYNC_DEBOUNCE_S = 30.0

#: ``network.sync.tick_s``: how often the owner's watcher stats the transcripts of
#: sessions that have replicas. A stat per replicated session per tick is the whole
#: cost, which is why this can be short without a filesystem watcher.
SYNC_TICK_S = 15.0

#: The owner-side deadline for one ``net_sync`` request (seconds). A ``fetch`` is
#: one ≤512 KiB chunk, so this bounds a slow disk, not a whole transcript.
SYNC_OP_DEADLINE_S = 60.0

#: One chunk of raw file bytes per ``fetch``. The frame carries them base64'd, so
#: the wire cost is 4/3 of this (~683 KiB) — under the control socket's 8 MiB
#: ``MAX_RECORD_BYTES`` and the relay's session-frame cap — and large enough that a
#: 103 MB transcript is ~200 round trips rather than thousands. Chunking exists so
#: a dropped link costs one chunk's progress, not a copy.
SYNC_CHUNK_BYTES = 512 * 1024

#: How many times a holder re-plans when the owner reports the copy moved under it
#: (a turn landed between the plan and the fetch). Bounded, because each retry
#: restarts the copy of a session that is being written to, and an unbounded loop
#: would livelock against a busy peer.
SYNC_REPLAN_ATTEMPTS = 3

#: The owner-side phases this op serves. ``verify`` is this build's addition to the
#: design's ``plan``/``fetch``/``available`` set, and it exists for RESUMPTION: a
#: destination that was interrupted mid-copy asks "are my first N bytes the same
#: as yours?" before appending, so a resumed copy is a verified append rather than
#: a hopeful splice (§7.4's resumable copy, made checkable).
SYNC_PHASES: tuple[str, ...] = ("plan", "fetch", "verify", "available")

#: Replica store, cursor file and the staging directory a move copies into.
NETWORK_DIRNAME = "network"
REPLICA_DIRNAME = "replicas"
REPLICA_CURSOR_NAME = "sync.json"
STAGING_DIRNAME = "staging"

#: ``attachments/<ref>.bin`` — the store-relative spelling blobs travel under, so
#: the two sides cannot disagree about where a blob lives.
ATTACHMENT_PREFIX = "attachments/"

TRANSCRIPT_NAME = "transcript.jsonl"

#: ``attachments`` — the store's own directory name (``session/attachments.py``'s
#: ``ATTACHMENTS_DIRNAME``), spelled here for the same import-weight reason the copy set
#: is. ``content_trees`` needs it: inside a REPLICA the blob store sits under the copy,
#: and a blob is a member of the copy set in its own right
#: (``referenced_attachments_in``), so walking it as a tree would give one file a second
#: name and a second digest.
ATTACHMENTS_DIRNAME = "attachments"

#: The session's birth-time sidecar's name (``session/creation.CREATED_AT_NAME``),
#: spelled here for the same import-weight reason the rest of the copy set is: this
#: module is imported at relay construction and ``session.creation`` is not lean.
#: ``test_sync_copy_set.py`` pins the literal against the module that owns it.
CREATED_AT_NAME = "created_at.json"

#: The copy set (§7.2). Spelled here rather than imported from the three modules
#: that own the names (``resume``, ``retention``, ``session/runtime/inbox``): this
#: module is imported at relay CONSTRUCTION, and those are session-engine modules.
#: ``tests/unit/network/test_sync_copy_set.py`` pins every literal against the
#: module that owns it, so the duplication cannot drift.
COPY_SET_NAMES: tuple[str, ...] = (
    TRANSCRIPT_NAME,
    "title.json",
    "attachment.json",
    "origin.json",
    "turn-journal.json",
    "runtime-stop.json",
    "inbox.jsonl",
    "fork-boundary.json",
    "desktop.json",
    # THE SESSION'S BIRTH TIME (``session/creation.CREATED_AT_NAME``). Without it
    # a moved conversation reads as newly created on the destination, because
    # ``session_created_at`` falls back to the directory's ``st_birthtime`` — and
    # the SOURCE directory is gone, so the real date is unrecoverable (review
    # round 1, B-M2). Copied rather than recomputed: a birth time is not derivable
    # from anything else on the destination.
    CREATED_AT_NAME,
    # THE JUDGED-GOAL RECORD (``resume.GOAL_SIDECAR_NAME``). ``origin/main`` added
    # it while this branch was open and it landed in NEITHER list, so a session with
    # a judged goal REFUSED to move, ``--keep`` dropped the objective silently and a
    # recovered replica came back with no goal at all — the exact failure B-M2's
    # guard exists for, arriving through the guard (review round 2, MAJOR 1). A
    # judged goal is work in progress that the user cannot get back, so it travels.
    "goal.json",
    # A FORK'S UNCONSUMED OPENING MESSAGE (``fork.BOOT_PROMPT_NAME``): ``/fork <msg>``
    # parks the message here and the fork's first boot consumes it. If the fork is
    # moved before it was ever opened, leaving this behind would silently discard the
    # instruction the user typed. (Consequence, stated so the next reader does not
    # have to derive it: a ``--keep`` copy of a never-opened fork also carries it, so
    # that copy's first boot injects the parent's opening message. ``fork_session``
    # deliberately does not — but ``--keep`` is a copy of a session, not a fork of a
    # running one, and losing the message outright is the worse of the two.)
    "boot-prompt.json",
)

#: Directories a session directory may hold whose FILES are the user's own
#: content and therefore travel with the session. ``scratchpad/`` is the one the
#: product creates (``scratchpad.py``): notes, downloaded files and scripts an
#: agent wrote for that conversation. 3,017 of the operator's 10,841 session
#: directories hold one, so leaving it out of the copy set while the commit
#: ``rmtree``d the source destroyed real work (review round 1, B-M2).
COPY_SET_TREES: tuple[str, ...] = ("scratchpad",)

#: Directories whose CONTENTS are re-creatable machinery rather than work: a python
#: bytecode cache or a test runner's cache an agent's own command left behind. They
#: are excluded rather than refused (their bytes are derivable, and a 300 MB
#: ``.pytest_cache`` in a move is pure cost), and rather than copied (a cache is not
#: content, and the destination's own tools rebuild theirs).
CACHE_TREE_NAMES: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
)

#: The suffix EVERY atomic writer in this codebase puts on its temporary (see
#: ``session/runtime/registry._staged_write`` and ``resume``'s sidecar writers): the
#: temp sits in the same directory as its target and is renamed over it, so a writer
#: killed mid-call leaves a ``<name>.<pid>.tmp`` sibling. Such a leftover is a crash
#: artifact, not content, and before this it made a session PERMANENTLY unmovable
#: (review round 2, MINOR 2: ``.subagent-roster.v1.json.<rand>.tmp`` and
#: ``title-scan.<pid>.tmp`` in the real store). Matched by suffix at the session ROOT
#: only: inside a copy-set tree a ``.tmp`` file is the user's own, and it travels.
TRANSIENT_SUFFIX = ".tmp"

#: The prefix a file inside a tree travels under, so both ends agree where it
#: lands (``scratchpad/notes.md``) exactly as ``ATTACHMENT_PREFIX`` does for a
#: blob. One flat namespace is deliberate: the wire carries ``name`` and nothing
#: else, so the name has to say which container it belongs to.
TREE_PREFIXES: dict[str, str] = {name: f"{name}/" for name in COPY_SET_TREES}

#: ``.json`` — the attachment store's sidecar suffix (``session/attachments.py``).
ATTACHMENT_SIDECAR_SUFFIX = ".json"

#: Files and directories the copy must NEVER carry, EACH WITH ITS REASON (design
#: §7.2). The copy is an allow-list and needs no deny-list to be correct — this is
#: the spec of what does not travel, with a test attached, so the next person who
#: reaches for ``copytree`` (or adds a name to one list and not the other) sees why.
#:
#: ``EXCLUDED_ENTRIES`` is keyed by name and is THE authoritative statement;
#: ``NEVER_COPIED`` is the tuple the code branches on and is checked against it by
#: ``tests/unit/network/test_sync_copy_set.py``, which also proves that every
#: entry the retention module calls bookkeeping is on one list or the other.
EXCLUDED_ENTRIES: dict[str, str] = {
    # The liveness marker: copied, the destination would report the SOURCE's pid
    # as the owner of its copy, which is how a viewer refuses to open a session
    # that is not running there.
    ".session.pid": "the source process's liveness marker",
    # Written by the destination itself, because ``home_device`` has to name the
    # NEW owner. Copying it would leave the moved session stamped as this
    # device's.
    "mesh.json": "the ownership stamp, rewritten by whoever adopts the copy",
    # Describes a replica, means nothing inside a session.
    REPLICA_CURSOR_NAME: "the replica cursor, which is about this device's copy",
    # The definitions index (``network.definitions.INDEX_NAME``): the agent and team
    # rows this device mirrors to a peer, written BESIDE the sessions tree
    # (``<config>/network/definitions.json``) rather than into a session. It is not an
    # entry of a session directory at all, so it belongs on this side of the copy
    # set's answer rather than in ``COPY_SET_NAMES``.
    "definitions.json": "the definitions index, which lives beside the sessions tree",
    # Lists jobs owned by the SOURCE's process; the destination's roster is its
    # own, and a copied one would name subagents that do not exist here.
    "subagent-roster.v1.json": "the source process's subagent roster",
    # Per-directory "already swept" claims: on the destination they would suppress
    # a backfill in a directory no sweep has ever seen.
    "origin-scan.json": "a 'this directory was scanned' sentinel",
    "title-scan.json": "a 'this directory was scanned' sentinel",
    # The two halves of the transcript lease. A claim copied with the session
    # would name the SOURCE's pid as this copy's writer, and the recovery lock
    # would look like a live take-over in progress: the destination could not
    # open the conversation it just adopted.
    ".execution-lease": "the source's transcript lease claim",
    ".execution-lease.recovery": "the source's stale-lease recovery lock",
    # A lock file for the wake write path. Copying a lock is copying a claim that
    # some process holds it; the destination's own writes would then look blocked.
    ".wake-write.lock": "a per-device lock file, meaningful only where it was taken",
    # The browser bridge's ownership record: it names a bridge GENERATION and an
    # allocation id minted by THIS device's bridge, so on the destination it is a
    # claim about browsers that do not exist there.
    ".browser-resource.json": "this device's browser-bridge ownership record",
    # A recomputable cache of origin verdicts (``resume.ORIGIN_CACHE_NAME``): pure
    # derived state, rebuilt on the destination on first read.
    "origin-verdicts.json": "a recomputable origin-verdict cache",
    # The move's own boot marker: it exists so a crash before the promote can be
    # settled, and ``mobility._promote`` deletes it on the way in (its docstring
    # says why an adopted session must not carry it). A crash between the rename
    # and that unlink can leave one behind, and ``mobility``'s recovery removes it
    # rather than treating it as content.
    "ready.json": "the move's own boot marker, not session content",
    # An UPDATE WINDOW's marker (``session/runtime/inbox.UPDATE_WINDOW_NAME``): a live
    # process's note to its successor that it is moving to a new build. On the
    # destination it would tell an unrelated runtime that an update it never ran was
    # applied. It is also transient by construction (the writer clears it), so
    # copying it would be a claim about the wrong device.
    "update-window.json": "an update window's own marker, meaningful only where it was opened",
    # An EXIT REQUEST (``session/runtime/registry.EXIT_REQUEST_NAME``) is a control
    # message ADDRESSED to a runtime, and the reader honours it only while it is fresh
    # (``process._EXIT_REQUEST_FRESH_S``). Carried to the destination it would be an
    # instruction from another device to stop a conversation that device does not own.
    "runtime-exit-request.json": "an exit request addressed to the runtime that wrote it",
}

#: The tuple the code branches on. Kept as a tuple (rather than reading
#: ``EXCLUDED_ENTRIES`` everywhere) because these are the names a fan-out of
#: ``in`` tests and one copy loop consult on the hot path; the pair is pinned by
#: the copy-set test so a name cannot live in one and not the other.
NEVER_COPIED: tuple[str, ...] = tuple(EXCLUDED_ENTRIES)

#: The names an ADOPTING device writes itself while it takes a copy, so they can
#: never be part of a digest compared byte-for-byte against the source's own
#: directory: ``origin.json`` records the lineage of THIS copy (its parent names
#: the source) and ``fork-boundary.json`` is the divergence marker the adopting
#: device adds. Every other name in the copy set is compared.
ADOPTED_LOCALLY: tuple[str, ...] = ("origin.json", "fork-boundary.json")


class SyncRefused(Exception):
    """A sync step was refused. Carries the family's ``code`` and its sentence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


#: A transcript row's reference to a content-addressed attachment. The store names
#: files by the first 32 hex chars of the decoded bytes' sha256
#: (``session/attachments.py``), so that is what a reference scan looks for.
_ATTACHMENT_REF = re.compile(rb'"attachment"\s*:\s*"([0-9a-f]{32})"')


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncSettings:
    debounce_s: float = SYNC_DEBOUNCE_S
    tick_s: float = SYNC_TICK_S

    @classmethod
    def from_config(cls, root: Path | None = None) -> SyncSettings:
        """Read ``network.sync.*`` through the package's ONE config reader."""
        from local_operator.network import store

        return cls(
            debounce_s=float(
                store.read_config(("network", "sync", "debounce_s"), SYNC_DEBOUNCE_S, root)
            ),
            tick_s=float(store.read_config(("network", "sync", "tick_s"), SYNC_TICK_S, root)),
        )


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def network_dir(root: Path) -> Path:
    return Path(root) / NETWORK_DIRNAME


def staging_dir(root: Path, session_id: str) -> Path:
    """Where a copy lands before it is promoted. OUTSIDE ``sessions/``, on purpose.

    Nothing scans this: a picker, a catalogue, a retention sweep or a resume would
    see a half-copied session the moment it were inside ``sessions/``, and the
    promote is one ``os.replace`` precisely so that window never exists.
    """
    return network_dir(root) / STAGING_DIRNAME / session_id


def replica_dir(root: Path, session_id: str) -> Path:
    return network_dir(root) / REPLICA_DIRNAME / session_id


def read_replica_cursor(root: Path, session_id: str) -> dict[str, Any]:
    """The holder's last synced position, or ``{}`` when it has never synced.

    A cursor that cannot be parsed reads as ``{}`` rather than refusing: every byte
    of the next pull is verified against the owner's digests, so the only cost of
    forgetting the position is a full copy.
    """
    try:
        raw = (replica_dir(root, session_id) / REPLICA_CURSOR_NAME).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_replica_cursor(root: Path, session_id: str, payload: dict[str, Any]) -> Path:
    directory = replica_dir(root, session_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / REPLICA_CURSOR_NAME
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    return target


# ---------------------------------------------------------------------------
# The owner's half: what to send, and the bytes
# ---------------------------------------------------------------------------


def session_dir(root: Path, session_id: str) -> Path:
    """One session's directory. THE path both ends build their digests from."""
    return Path(root) / "sessions" / session_id


def copy_set(root: Path, session_id: str) -> list[str]:
    """The copy set's names that exist for this session, in a fixed order."""
    directory = session_dir(root, session_id)
    return [name for name in COPY_SET_NAMES if (directory / name).is_file()]


#: ``b"\x00link\x00"`` / ``b"\x00file\x00"`` — what a member IS, fed into the content
#: digest beside its name, so a link and a one-byte file with the same bytes cannot
#: produce the same digest.
_LINK_MARK = b"\x00link\x00"
_FILE_MARK = b"\x00file\x00"


def content_trees(directory: Path) -> list[str]:
    """Every tree under ``directory`` whose contents travel, as flat names.

    ``COPY_SET_TREES`` plus any OTHER directory at the session root. The second half
    is review round 2's real-store finding: 17 of the operator's 11,115 sessions hold
    a directory an agent or the user created at the root (``notes``, ``drafts``,
    ``workspace``, ``qa-1443``, ``bench``, …), and refusing those made a move
    impossible on exactly the sessions people work in. A directory at the session root
    is CONTENT: an unknown FILE beside it is still refused, and the asymmetry is
    deliberate and narrow — a file's meaning is keyed by its NAME (``resume`` reads
    sidecars by name, so an unrecognised name may be read as something it is not, and
    guessing about that is how the two entries B-M2 exists for were destroyed), while a
    directory is a container whose contents can be copied byte for byte without
    interpreting them. ``CACHE_TREE_NAMES`` are the exceptions, and they are excluded
    by name with the reason attached.

    ``attachments`` is never walked here even when it sits inside ``directory`` (it does
    inside a REPLICA): a blob is a member of the copy set in its own right
    (``referenced_attachments_in``), with the store-relative name both ends agree on, so
    walking it a second time would be a second name for one file.
    """
    directory = Path(directory)
    found: list[str] = []
    for name in COPY_SET_TREES:
        path = directory / name
        if path.is_dir() and not path.is_symlink():
            found.append(name)
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return found
    for child in children:
        if child.name in found or child.name in CACHE_TREE_NAMES or child.name in NEVER_COPIED:
            continue
        if child.name == ATTACHMENTS_DIRNAME:
            continue
        if child.is_dir() and not child.is_symlink():
            found.append(child.name)
    return found


def _inside_session(directory: Path, resolved: Path) -> bool:
    """Is a symlink's resolved target inside the session directory ``directory``?"""
    try:
        base = Path(directory).resolve()
        target = Path(resolved).resolve()
    except OSError:
        return False
    return target == base or base in target.parents


def _portable_link_target(directory: Path, link: Path) -> str:
    """The target text to WRITE on the destination for the symlink ``link``, or ``""``.

    ``""`` means the link points OUTSIDE the session, and that is the one shape a
    deleting move refuses (see :func:`_tree_entries`). For a link that stays inside, the
    text is rewritten to a RELATIVE path from the link's own directory, so the link
    means the same thing after the session lands somewhere else — which an absolute path
    spelling the source's store root would not (the destination's config root is a
    different path, and on another device it is a different machine's home).

    A relative path is computed LEXICALLY (``os.path.relpath``), deliberately: the target
    may not exist (a dangling link inside the session is still carried, and stays exactly
    as dangling as it was), and a resolved path would collapse the very component the
    link exists to name.
    """
    try:
        target = os.readlink(link)
    except OSError:
        return ""
    resolved = Path(target) if os.path.isabs(target) else link.parent / target
    if not _inside_session(directory, resolved):
        return ""
    return os.path.relpath(str(resolved), str(link.parent))


def _tree_entries(
    directory: Path,
) -> tuple[list[str], list[tuple[str, str]], list[str], list[str]]:
    """ONE walk of every content tree. Four answers, because they get four treatments.

    * regular files, ``<tree>/<relpath>`` — they travel as bytes;
    * symlinks whose target stays INSIDE this session, as ``(name, target)`` — carried,
      with the target rewritten to a relative path (see :func:`_portable_link_target`);
    * symlinks whose target is OUTSIDE the session — a device-local path, which on the
      destination resolves to a DIFFERENT file or to nothing, so a deleting move refuses
      and names it (this is the shape the module's own docstring has always argued
      about, and it is now the only one);
    * entries that are neither file nor symlink — a fifo, a socket, a device node. They
      hold NO data: the payload lives in the process that created them, which the
      destination cannot have. So nothing is lost by leaving one behind, and a tmux
      socket left by a killed rig must not make a session permanently unmovable. They
      are reported in the plan rather than refused.

    136 of the operator's sessions were refused for a symlink in ``scratchpad/`` before
    this: 6,629 of those links point INSIDE their own session (pytest's ``*-current``
    links, a venv's ``python3``), so carrying those is what makes the feature usable on
    this machine, and 341 point outside (a scratch checkout, ``/var/folders``), so those
    still refuse with the link named.
    """
    directory = Path(directory)
    files: list[str] = []
    links: list[tuple[str, str]] = []
    irregular: list[str] = []
    unportable: list[str] = []
    for tree in content_trees(directory):
        base = directory / tree
        for parent, dirnames, filenames in os.walk(base, followlinks=False):
            keep: list[str] = []
            for name in sorted(dirnames):
                path = Path(parent) / name
                if not path.is_symlink():
                    keep.append(name)
                    continue
                flat = f"{tree}/{path.relative_to(base).as_posix()}"
                target = _portable_link_target(directory, path)
                if target:
                    links.append((flat, target))
                else:
                    irregular.append(flat)
            dirnames[:] = sorted(keep)
            for name in sorted(filenames):
                path = Path(parent) / name
                flat = f"{tree}/{path.relative_to(base).as_posix()}"
                if path.is_symlink():
                    target = _portable_link_target(directory, path)
                    if target:
                        links.append((flat, target))
                    else:
                        irregular.append(flat)
                elif path.is_file():
                    files.append(flat)
                else:
                    unportable.append(flat)
    return sorted(files), sorted(links), sorted(irregular), sorted(unportable)


def tree_entry_names(directory: Path) -> list[str]:
    """Every regular file under the content trees, as ``<tree>/<relpath>``.

    Sorted, and flat rather than nested, because this list is not only an iteration
    order: the spelling is part of the format rather than an implementation detail:
    ``_member_stamps`` hashes this list on the owner and again on the copy.
    """
    return _tree_entries(directory)[0]


def tree_link_entries(directory: Path) -> list[tuple[str, str]]:
    """Every carried symlink under the content trees, as ``(name, target)``.

    ``target`` is what to WRITE, not what the source holds: a relative path, so the
    link survives the session's location changing (see ``_portable_link_target``).
    """
    return _tree_entries(directory)[1]


def unportable_tree_entries(directory: Path) -> list[str]:
    """Fifos, sockets and device nodes inside a content tree. Reported, never carried.

    They are not data (see :func:`_tree_entries`), so a move neither copies them — it
    cannot — nor refuses for them.
    """
    return _tree_entries(directory)[3]


def irregular_tree_entries(directory: Path) -> list[str]:
    """Entries inside a content tree that a DELETING move must not walk past.

    ONE shape remains: a symlink whose target is not inside this session. A
    device-local target is not portable data — the same text on the destination names a
    different file, or nothing — and since a move DELETES the source, "we did not carry
    this and then deleted it" is data loss whatever the entry was. Refusing with the
    path named is the honest answer: nothing is lost, and the sentence says what to do.
    """
    return _tree_entries(directory)[2]


def unlisted_entries(directory: Path) -> list[str]:
    """Top-level names in ``directory`` that NO list accounts for.

    A name here is an entry type the copy set has never been taught — which is
    exactly the state ``scratchpad/`` and ``created_at.json`` were in while a move
    deleted them (review round 1, B-M2). Adding a genuinely untravelling name to
    ``EXCLUDED_ENTRIES`` with its reason is the escape hatch; the point is that the
    decision be made on purpose rather than by omission.

    Four shapes are NOT listed, and each is a fact about this directory rather than an
    unclassified entry type:

    * a DIRECTORY — carried by ``content_trees`` (or excluded there by name);
    * a ``*.tmp`` leftover — the corpse of an atomic write, not content;
    * a cache directory (``CACHE_TREE_NAMES``);
    * a symlink — carried by the tree walk if it is inside a tree (its target rewritten
      relative), and refused HERE if it sits at the root, whichever name it carries: the
      root is the one place no product writer puts one, and a root link is a pointer to
      something OUTSIDE the session that the destination would either lose or resolve
      elsewhere.

    THE SHAPE IS CHECKED BEFORE THE NAME LISTS, and that ordering is the fix for review
    round 3's MAJOR. It used to be the other way round, on the reasoning that a classified
    name needs no classification — and a session whose ``scratchpad`` was a SYMLINK then
    fell through both the copy set (``content_trees`` does not walk a link, correctly) and
    this function (the name was classified, so it was skipped). The plan reported
    ``trees: []``, ``trees_skipped: []``, the move committed and the source directory was
    removed WITH the link in it: the session's only pointer to the user's tree gone, and
    nothing in the copy or the plan naming it. Names in ``NEVER_COPIED`` keep their
    exclusion, because their reason is about whose state the file is rather than about its
    shape.
    """
    listed = set(COPY_SET_NAMES) | set(NEVER_COPIED) | set(COPY_SET_TREES)
    directory = Path(directory)
    try:
        children = sorted(directory.iterdir())
    except OSError:
        return []
    found: list[str] = []
    for child in children:
        name = child.name
        if child.is_symlink() and name in listed and name not in NEVER_COPIED:
            found.append(name)
            continue
        if name in listed or name in CACHE_TREE_NAMES:
            continue
        if name.endswith(TRANSIENT_SUFFIX) and child.is_file() and not child.is_symlink():
            continue
        if child.is_dir() and not child.is_symlink():
            # A DIRECTORY IS CONTENT, EXCEPT WHERE IT IS THE STORE'S OWN. ``attachments``
            # at a session root is not a blob store (the store lives at the config root)
            # and not a member of the copy set, so it is refused like any other name this
            # build cannot account for: an adopter reads blobs through the store, and a
            # copy that carried this directory would land content nothing looks at.
            if name == ATTACHMENTS_DIRNAME:
                found.append(name)
            continue
        found.append(name)
    return found


def excluded_cache_trees(directory: Path) -> list[str]:
    """Directories at the session root whose contents are caches (reported, not copied).

    Reported for the same reason ``transient_entries`` is: an entry a copy deliberately
    leaves behind should be visible in the plan, so "why did my ``.pytest_cache`` not come
    across?" has an answer in the document rather than in a comment.
    """
    try:
        children = sorted(Path(directory).iterdir())
    except OSError:
        return []
    return [
        child.name
        for child in children
        if child.name in CACHE_TREE_NAMES and child.is_dir() and not child.is_symlink()
    ]


def symlinked_tree_entries(directory: Path) -> list[str]:
    """``COPY_SET_TREES`` names that exist at the root as a SYMLINK (reported, refused).

    Reported for the same reason ``transient_entries`` is: an entry a copy will not carry
    must be visible in the plan, so "why is this session not moving?" has an answer in the
    document rather than in the sentence of whichever refusal a person happens to hit.
    ``content_trees`` does not walk a symlinked tree (see it), so without this list a plan
    showed ``trees: []`` and ``trees_skipped: []`` for a session whose ``scratchpad`` was
    a link — the exact blindness that let review round 3's MAJOR through.
    """
    try:
        children = sorted(Path(directory).iterdir())
    except OSError:
        return []
    return [child.name for child in children if child.name in COPY_SET_TREES and child.is_symlink()]


def transient_entries(directory: Path) -> list[str]:
    """The ``*.tmp`` leftovers at the session root, for the plan to report.

    Reported rather than silently ignored so a person reading a plan can see what a
    move left behind, and so the rule is visible at the moment someone wonders why a
    ``.tmp`` file did not travel. The rule itself is in ``unlisted_entries``.
    """
    try:
        children = sorted(Path(directory).iterdir())
    except OSError:
        return []
    return [
        child.name
        for child in children
        if child.name.endswith(TRANSIENT_SUFFIX) and child.is_file() and not child.is_symlink()
    ]


def assert_complete(directory: Path) -> None:
    """Refuse when ``directory`` holds anything the copy set does not account for.

    ONLY a deleting move calls this (``mobility._source_commit``), because only a
    deleting move turns "not copied" into "not copied and then deleted": ``--keep``
    copies what it knows and leaves the source where it is, so anything it skipped
    is still on disk for its owner.

    Fail-closed on purpose. The alternative — copy what we know and delete the
    rest — is how the two entry types this guard exists for were destroyed. The
    sentence names every offending entry, because the person reading it is the one who
    can move or remove it.
    """
    named = sorted(set(unlisted_entries(directory)) | set(irregular_tree_entries(directory)))
    if not named:
        return
    # A ROOT-LEVEL SYMLINK NEEDS ITS OWN SENTENCE. "This build does not carry it" is true
    # of an unknown name and misleading for a name the copy set DOES carry, so the shape is
    # named and the remedy is the one that works: the link is not the tree.
    links = symlinked_tree_entries(directory)
    shape = (
        f" {', '.join(links)} is a symlink rather than a directory: a copy carries a "
        "session's own trees, and a link points at one that a copy on another device would "
        "not have. Replace the link with the directory, or copy the tree in by hand, then "
        "try again."
        if links
        else ""
    )
    raise SyncRefused(
        "unlisted_content",
        f"{Path(directory).name} holds {', '.join(named)}, which this build's copy set "
        "does not carry, so it was not moved. A symlink out of a session, or a file "
        "named for something this build does not know, would be deleted with nothing to "
        "copy it from: move or remove it, or copy it across by hand, then try again." + shape,
    )


def _safe_region(payload: bytes) -> bytes:
    """``payload`` up to and including its last complete line.

    THE TORN TAIL. ``--keep`` copies from a runtime that is still writing, so the
    last line may be half-written. A reader tolerates that (``fork_session``
    passes malformed lines through), but a CURSOR cannot: the next sync has to know
    exactly which bytes it already holds, and "everything up to the last newline"
    is the only boundary both sides can agree on without parsing. The tail row is
    not lost — the next sync carries it, whole.

    A DELETING MOVE DOES NOT USE THIS. Nothing is writing that file (the runtime was
    retired first), the source directory is about to be deleted, and there is no "next
    sync" to carry the tail: cutting it deleted the user's bytes (review round 2, the
    round-1 torn-tail finding, which reproduced as source 203 bytes / destination 150 /
    source gone). ``whole_transcript=True`` is how a move says so.
    """
    cut = payload.rfind(b"\n")
    return payload if cut < 0 else payload[: cut + 1]


def _frontier(region: bytes) -> str:
    """The ``id`` of the last complete row in ``region``, or ``""``.

    Best effort and never a gate: it is a hint a surface can print ("synced
    through e1234"), and the design's cursor carries it for that reason. Every
    decision here rests on the byte digest, which cannot be misread.
    """
    for line in reversed(region.splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            return ""
        return str(row.get("id") or "") if isinstance(row, dict) else ""
    return ""


def referenced_attachments_in(directory: Path) -> list[str]:
    """Attachment digests the transcript IN ``directory`` references, deduplicated.

    Takes the directory rather than a root plus an id because BOTH ends need it:
    the owner reads it from the session directory and a destination from its
    staging or replica copy. ``_member_stamps`` hashes the blob set this returns,
    so the two ends only agree if they derive the set the same way.
    """
    try:
        payload = (Path(directory) / TRANSCRIPT_NAME).read_bytes()
    except OSError:
        return []
    return sorted({match.decode("ascii") for match in _ATTACHMENT_REF.findall(payload)})


def referenced_attachments(root: Path, session_id: str) -> list[str]:
    """Attachment digests this session's transcript references, deduplicated."""
    return referenced_attachments_in(session_dir(root, session_id))


@dataclass(frozen=True)
class _MemberStamp:
    """One member of the copy set, as one pass over it saw it.

    ``digest`` is over the bytes that are SERVED (the safe region for a transcript a
    live runtime owns, every byte for a move), ``stat_bytes``/``mtime_ns`` are the real
    file's, and they are what a fetch's cheap staleness check compares — one ``stat``
    instead of re-hashing the copy set.
    """

    digest: str
    served_bytes: int
    stat_bytes: int
    mtime_ns: int
    link: str = ""


def _stream_digest(path: Path) -> str:
    """sha256 of a whole file, streamed in 1 MiB blocks.

    STREAMED, not ``read_bytes``: the members this hashes are the user's own files, and
    the real store holds scratchpad files in the hundreds of MB (the largest scratchpad
    is 1.9 GB). Reading one whole to hash it is an allocation, and the copy used to make
    that allocation once per 512 KiB chunk.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _member_stamps(
    directory: Path,
    session_id: str,
    attachments_dir: Path,
    *,
    whole_transcript: bool = False,
) -> dict[str, _MemberStamp]:
    """ONE pass over the copy set: every member's digest, size and stat.

    THE PASS THAT REPLACED THE QUADRATIC COPY. It was ``_content_digest`` reading
    everything on every 512 KiB chunk (review round 2, MAJOR 2: an 8 MB scratchpad cost
    2.6 s, a 32 MB one 38.7 s, and the real store has 129 scratchpads over 32 MB) while
    the same bytes were read AGAIN to serve the chunk. Now the digests are taken once,
    here, and returned so the plan can hand each member's digest to the holder and the
    fetch path can validate the member it is serving with a single ``stat``.

    Members: every copy-set name present, every regular file and every CARRIED symlink
    under the content trees, and every attachment blob and sidecar the transcript
    references. Unreferenced blobs are deliberately out: they are store members this
    session does not use.

    A member that cannot be read is skipped, not refused: it can vanish between the walk
    and the read (a session being written to), and the copy that follows either carries
    what is there or fails its digests. Absent-on-one-side members make the two digests
    disagree, which is the fail-closed direction.
    """
    directory = Path(directory)
    stamps: dict[str, _MemberStamp] = {}
    for name in COPY_SET_NAMES:
        path = directory / name
        try:
            if name == TRANSCRIPT_NAME:
                raw = path.read_bytes()
                payload = raw if whole_transcript else _safe_region(raw)
                stat = path.stat()
                stamps[name] = _MemberStamp(
                    sha256_bytes(payload), len(payload), len(raw), stat.st_mtime_ns
                )
                continue
            stat = path.stat()
            stamps[name] = _MemberStamp(
                _stream_digest(path), stat.st_size, stat.st_size, stat.st_mtime_ns
            )
        except OSError:
            continue
    for name, target in tree_link_entries(directory):
        try:
            stat = (directory / name).lstat()
        except OSError:
            continue
        payload = target.encode("utf-8")
        stamps[name] = _MemberStamp(
            sha256_bytes(payload), len(payload), len(payload), stat.st_mtime_ns, link=target
        )
    for name in tree_entry_names(directory):
        path = directory / name
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            digest = _stream_digest(path)
        except OSError:
            continue
        stamps[name] = _MemberStamp(digest, stat.st_size, stat.st_size, stat.st_mtime_ns)
    store = Path(attachments_dir)
    for ref in referenced_attachments_in(directory):
        for suffix in (".bin", ATTACHMENT_SIDECAR_SUFFIX):
            wire = f"{ATTACHMENT_PREFIX}{ref}{suffix}"
            path = store / f"{ref}{suffix}"
            try:
                stat = path.stat()
                digest = _stream_digest(path)
            except OSError:
                continue
            stamps[wire] = _MemberStamp(digest, stat.st_size, stat.st_size, stat.st_mtime_ns)
    return stamps


#: The wire value a destination sends when it is taking a session OVER rather than
#: keeping a replica of it. It is a string on the plan request, so both sides have to
#: agree on it; spelling it once is what ``whole_transcript_for`` is for.
COPY_PURPOSE_MOVE = "move"


def whole_transcript_for(purpose: str) -> bool:
    """Does a copy made for ``purpose`` carry the transcript WHOLE?

    ONE SPELLING OF THE TAIL PROPERTY, which is spread over two modules and four call
    sites (review round 3, MINOR 2). ``_plan`` keys it on the wire string the destination
    sends, and the SOURCE's own prepare/commit comparison passes it directly from
    ``mobility`` — where three ``whole_transcript=True`` arguments sit in the file a
    maintainer reads first, none of which changes what the destination receives. Reverting
    those three leaves every torn-tail cell green; reverting this predicate fails them
    with ``digest_mismatch``. Deriving all four from here means "fixing the wrong one" is
    no longer possible: there is only one.

    A MOVE deletes its source, so there is no next sync to carry a torn tail and the
    transcript travels whole (review round 2). A replica of a live session keeps the safe
    region: the next sync carries the rest (see ``_safe_region``).
    """
    return purpose == COPY_PURPOSE_MOVE


#: The last few plans this process built, keyed ``(root, session_id, plan_id)``.
#:
#: WHY IT EXISTS. ``serve_fetch`` must answer "does this plan still describe the source?"
#: and a full re-hash per chunk is the quadratic cost this round removes. A destination
#: that sends ``expect_bytes``/``expect_mtime_ns`` (every destination built from here on)
#: is answered from ITS OWN numbers, so the cache is not on that path at all. It serves
#: the two cases where the frame carries none: an older peer, and the direct
#: ``serve_fetch(plan=…)`` calls the tests make. Missing an entry degrades to the
#: destination's own end-of-member digest check — one failed copy, never a bad one — which
#: is stated rather than implied.
#: The bound is small on purpose: a relay serves one move at a time, and an entry is a
#: handful of strings per member.
_PLAN_CACHE: dict[tuple[str, str, str], tuple[dict[str, _MemberStamp], bool]] = {}
_PLAN_CACHE_MAX = 8
_PLAN_CACHE_LOCK = threading.Lock()


def _remember_plan(
    root: Path,
    session_id: str,
    plan: str,
    stamps: dict[str, _MemberStamp],
    whole_transcript: bool,
) -> None:
    """Record the stamps behind one plan id, so a later fetch can validate cheaply.

    ``whole_transcript`` is recorded WITH them because the served bytes have to be the bytes
    the plan described: a move carries its transcript whole, a replica of a live session
    carries the safe region, and a fetch that served the other variant would hand the holder
    bytes whose digest is not the one it verified against.
    """
    key = (str(root), session_id, plan)
    with _PLAN_CACHE_LOCK:
        _PLAN_CACHE.pop(key, None)
        _PLAN_CACHE[key] = (stamps, whole_transcript)
        while len(_PLAN_CACHE) > _PLAN_CACHE_MAX:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))


def _cached_stamp(root: Path, session_id: str, plan: str, name: str) -> _MemberStamp | None:
    if not plan:
        return None
    with _PLAN_CACHE_LOCK:
        record = _PLAN_CACHE.get((str(root), session_id, plan))
    stamps = record[0] if record else None
    return stamps.get(name) if stamps else None


def member_stamps(
    root: Path,
    session_id: str,
    *,
    attachments_dir: Path | None = None,
    whole_transcript: bool = False,
) -> dict[str, _MemberStamp]:
    """``_member_stamps`` for a session, for a caller that needs the plan and the digest.

    The move's ``prepare`` keeps these across its retire (see ``stamps_valid``), so one
    pass gives it the plan id, the manifest and the content digest it records in the
    journal — where the previous shape read the whole copy set three times before the
    destination had asked for a byte.
    """
    store = Path(attachments_dir) if attachments_dir is not None else Path(root) / "attachments"
    return _member_stamps(
        session_dir(root, session_id), session_id, store, whole_transcript=whole_transcript
    )


def stamps_valid(
    directory: Path,
    session_id: str,
    attachments_dir: Path,
    stamps: dict[str, _MemberStamp],
) -> bool:
    """Do these stamps still describe what is on disk? One ``stat`` per member.

    THE CHEAP RE-CHECK, and the reason a move can hash BEFORE it stops the source's
    runtime. It compares the size and ``mtime_ns`` each member had when it was digested
    with the file's now. A member that vanished counts as changed. Two writes inside one
    mtime tick could hide a change (the filesystem's mtime granularity is the limit, and
    it is why the COMMIT still re-derives the digest from the bytes), so this answers
    "nothing visible moved", never "the bytes are correct" — a caller that needs the
    second answer re-digests, which is what ``_source_prepare`` does when this says no.
    """
    directory = Path(directory)
    store = Path(attachments_dir)
    for name, stamp in stamps.items():
        # A blob lives in the store under its store-relative name; every other member is
        # already relative to the session directory.
        path = (
            store / name[len(ATTACHMENT_PREFIX) :]
            if name.startswith(ATTACHMENT_PREFIX)
            else directory / name
        )
        try:
            if stamp.link:
                # A LINK's size is its target's text length, and ``lstat`` is the only
                # stat that describes the link rather than what it points at.
                if path.lstat().st_mtime_ns != stamp.mtime_ns:
                    return False
            else:
                stat = path.stat()
                if stat.st_size != stamp.stat_bytes or stat.st_mtime_ns != stamp.mtime_ns:
                    return False
        except OSError:
            # GONE IS CHANGED. A member that has been removed cannot be served from a
            # stale stamp, and skipping it silently would make the two ends' digests
            # disagree somewhere the caller cannot see.
            return False
    return True


def content_digest_from(
    stamps: dict[str, _MemberStamp], session_id: str, *, skip: tuple[str, ...] = ()
) -> str:
    """The content digest of a stamp set (see ``_content_digest_from``)."""
    return _content_digest_from(stamps, session_id, skip=skip)


def _content_digest_from(
    stamps: dict[str, _MemberStamp], session_id: str, *, skip: tuple[str, ...] = ()
) -> str:
    """The digest of what a directory holds, from stamps already taken.

    THE SAME FUNCTION RUNS ON BOTH SIDES OF A MOVE: the owner digests its own
    directory, the destination digests the copy it holds, and the owner COMMITS ONLY
    WHEN THE TWO ARE EQUAL — the check review round 1 found missing (M-2). Feeding each
    member's own digest rather than its bytes is what lets one pass serve both the plan
    and this value; it is not a weaker claim, because the member digest is a sha256 of
    exactly those bytes, and the SET of names is part of the digest in both directions
    (a member present on one side and absent on the other changes the value).

    Sorted by name, so the value cannot depend on walk order, and marked file-versus-link
    so a link to ``x`` cannot collide with a file containing ``x``.
    """
    digest = hashlib.sha256()
    digest.update(session_id.encode("utf-8"))
    for name in sorted(stamps):
        if name in skip:
            continue
        stamp = stamps[name]
        digest.update(name.encode("utf-8"))
        digest.update(_LINK_MARK if stamp.link else _FILE_MARK)
        digest.update(stamp.digest.encode("ascii"))
    return "sha256:" + digest.hexdigest()


def _content_digest(
    directory: Path,
    session_id: str,
    attachments_dir: Path,
    *,
    skip: tuple[str, ...] = (),
    whole_transcript: bool = False,
) -> str:
    """A digest of the CONTENT a session directory holds: one value, both ends.

    Covered: every copy-set name present, every regular file and carried symlink under
    the content trees, and every attachment blob AND its sidecar that the transcript
    references. ``skip`` exists for exactly one caller — the adopting device compares
    against the source with ``ADOPTED_LOCALLY`` skipped, because it writes its own
    lineage marker and fork boundary while it adopts.

    Cost is O(bytes of the copy), paid once per plan, once per ``prepare``, once at the
    commit and once on the destination before it adopts. It is NOT paid per chunk any
    more: a chunk is validated against the plan's recorded stamp for the one member it
    serves (review round 2, MAJOR 2).
    """
    return _content_digest_from(
        _member_stamps(directory, session_id, attachments_dir, whole_transcript=whole_transcript),
        session_id,
        skip=skip,
    )


def plan_id(
    root: Path,
    session_id: str,
    *,
    attachments_dir: Path | None = None,
    whole_transcript: bool = False,
) -> str:
    """A digest of the SOURCE's state, independent of what the holder has.

    The commit re-derives it: a plan that no longer describes the source must not be
    the basis for deleting the source's directory. ``whole_transcript`` is ``True`` for
    a move, whose transcript is carried every byte (see ``_safe_region``).
    """
    store = Path(attachments_dir) if attachments_dir is not None else Path(root) / "attachments"
    return _content_digest(
        session_dir(root, session_id),
        session_id,
        store,
        whole_transcript=whole_transcript,
    )


def copy_content_digest(directory: Path, session_id: str, attachments_dir: Path) -> str:
    """``_content_digest`` for a MOVE: whole transcript, names an adopter writes skipped.

    What a move compares across devices: the source's own directory against the copy the
    destination holds, once each. ``ADOPTED_LOCALLY`` is the two names the adopting
    device legitimately differs on — its own ``origin.json`` (whose parent names the
    source) and the ``fork-boundary.json`` divergence marker it adds. The transcript is
    taken whole because a move carries every byte of it (``_safe_region``'s cut is for a
    live source, and cutting it here is what deleted a torn tail in review round 2).
    """
    return _content_digest(
        directory,
        session_id,
        attachments_dir,
        skip=ADOPTED_LOCALLY,
        whole_transcript=True,
    )


def replica_content_digest(directory: Path, session_id: str, attachments_dir: Path) -> str:
    """``_content_digest`` for a device's own REPLICA, as its cursor records it.

    The replica's transcript is the source's safe region (a replica is pulled from a
    live session), so it is digested the same way the owner digested it: ``whole_transcript``
    stays off. This is the value ``_verify_replica_against_cursor`` re-derives before a
    recovery promotes the bytes, which is what stops a corrupted sidecar being promoted
    as a real session (review round 2, MINOR 3).
    """
    return _content_digest(directory, session_id, attachments_dir)


def _attested_digest(files: dict[str, str], links: tuple[str, ...], session_id: str) -> str:
    """The content digest of a member set a device can attest to by DIGEST alone.

    Built from the same :func:`_content_digest_from` a full pass uses, so a recovery can
    re-derive it from the replica on disk and compare: equal means every member is the one
    that was verified when it was synced, and unequal means something has since changed
    (a truncation, a rewrite, an edit). The digests here are the ones the owner published
    in the plan and this device checked member by member, so this is a cheaper spelling of
    the same claim rather than a second, weaker one.
    """
    stamps = {
        name: _MemberStamp(digest, 0, 0, 0, link="carried" if name in set(links) else "")
        for name, digest in files.items()
    }
    return _content_digest_from(stamps, session_id)


def build_manifest(
    root: Path,
    session_id: str,
    *,
    have: dict[str, Any] | None = None,
    attachments_dir: Path | None = None,
    whole_transcript: bool = False,
    stamps: dict[str, _MemberStamp] | None = None,
) -> dict[str, Any]:
    """The plan a holder pulls: what to send, with the hashes that verify it.

    ``have`` is the holder's own report (``{"cursor": …, "files": …, "attachments": …}``)
    and is used ONLY as a filter. The owner still digests everything it sends, so
    a holder that misreports what it holds gets a correct copy of the wrong size
    (a re-send), never a corrupt one.

    ``whole_transcript`` carries every byte of the transcript, which a DELETING move
    needs and nothing else does (see ``_safe_region``).

    One ``_member_stamps`` pass produces the plan id and every item's digest at once:
    the previous shape read the whole copy set twice for the same plan (once for the
    id, once for the items) and then re-read it on every chunk.
    """
    have = have or {}
    directory = session_dir(root, session_id)
    transcript = directory / TRANSCRIPT_NAME
    try:
        raw = transcript.read_bytes()
    except OSError as exc:
        # THREE FAILURES, AND EACH GETS ITS OWN FACT (QA delta, Q-D4; Aida round 2, finding 2).
        # This ``except`` catches every errno, not only a missing file, so the wait sentence
        # is gated on ``ENOENT`` — measured with the directory present in both cases: a
        # ``transcript.jsonl`` that is a directory (EISDIR) and one this process may not read
        # (EACCES) both got "try again in a few seconds", which for those never resolves and
        # threw away the one fact the old sentence did carry. The wait is named as a wait
        # only where waiting is what fixes it: the ordinary case is a conversation created
        # moments ago whose runtime has not written its first row yet — the transcript appears
        # about 10 s later and the identical move then succeeds (measured on two real devices;
        # the runtime writes that first row as it finishes starting up). Anything else carries
        # the reason it could not be read.
        if exc.errno != errno.ENOENT:
            raise SyncRefused(
                "no_session",
                f"{session_id} has a transcript on this device that could not be read "
                f"({exc}), so there is nothing to send",
            ) from exc
        if not directory.is_dir():
            raise SyncRefused(
                "no_session", f"this device does not hold {session_id}, so there is nothing to send"
            ) from exc
        raise SyncRefused(
            "no_session",
            f"{session_id} has no transcript on this device yet, so there is nothing to "
            f"send. A conversation created a moment ago writes its first row when its "
            f"runtime starts, so try again in a few seconds.",
        ) from exc
    safe = raw if whole_transcript else _safe_region(raw)
    total = len(safe)
    digest = sha256_bytes(safe)
    store = Path(attachments_dir) if attachments_dir is not None else Path(root) / "attachments"
    if stamps is None:
        stamps = _member_stamps(directory, session_id, store, whole_transcript=whole_transcript)
    content = _content_digest_from(stamps, session_id)
    _remember_plan(Path(root), session_id, content, stamps, whole_transcript)

    cursor_raw = have.get("cursor")
    cursor: dict[str, Any] = cursor_raw if isinstance(cursor_raw, dict) else {}
    prefix_bytes = int(cursor.get("prefix_bytes") or 0)
    prefix_digest = str(cursor.get("prefix_digest") or "")
    mode = "replace"
    if (
        prefix_bytes
        and prefix_bytes <= total
        and prefix_digest
        and sha256_bytes(safe[:prefix_bytes]) == prefix_digest
    ):
        # The holder's prefix is byte-identical to ours, so the rest is an append.
        # This is the ONLY test that admits an append: a compaction or a prune
        # changes these bytes and lands on `replace`.
        mode = "append"

    held_raw = have.get("files")
    held: dict[str, Any] = held_raw if isinstance(held_raw, dict) else {}
    items: list[dict[str, Any]] = []
    for name, stamp in sorted(stamps.items()):
        if name == TRANSCRIPT_NAME or name.startswith(ATTACHMENT_PREFIX):
            continue
        if str(held.get(name) or "") == stamp.digest:
            # The same bytes are already there. The design's mtime hint is
            # deliberately not consulted: two clocks disagree and a digest is a
            # fact.
            continue
        items.append(_item_document(name, stamp))

    blobs: list[dict[str, Any]] = []
    held_blobs_raw = have.get("attachments")
    held_blobs: dict[str, Any] = held_blobs_raw if isinstance(held_blobs_raw, dict) else {}
    for name, stamp in sorted(stamps.items()):
        if not name.startswith(ATTACHMENT_PREFIX):
            continue
        # THE HOLDER'S OWN KEY, which is what ``held_report`` writes and therefore what
        # this filter has to look up: a blob under its ``<ref>`` and its sidecar under
        # ``<ref>.json``. A mismatch here is not a correctness bug (the member is simply
        # re-sent), but it is the difference between a resume that costs a delta and one
        # that re-downloads every image in the conversation.
        if str(held_blobs.get(_held_blob_key(name)) or "") == stamp.digest:
            continue
        blobs.append(_item_document(name, stamp, ref=_blob_ref(name)))

    return {
        "session_id": session_id,
        "plan_id": content,
        "transcript": {
            "name": TRANSCRIPT_NAME,
            "mode": mode,
            "prefix_bytes": prefix_bytes if mode == "append" else 0,
            "total_bytes": total,
            "region_bytes": total - prefix_bytes if mode == "append" else total,
            "source_bytes": len(raw),
            # THE REAL FILE'S SIZE, which is not the served length when the safe region
            # cut a torn tail. It is what a fetch tells the owner to compare against, so
            # a transcript that grew or was rewritten under the copy is refused.
            "stat_bytes": len(raw),
            "digest": digest,
            "frontier": _frontier(safe),
        },
        "items": items,
        "attachments": blobs,
        "copy_set": copy_set(root, session_id),
        # The trees this plan carries, as flat names, and the entries it deliberately
        # does not: a symlink out of the session (refused by a deleting move) and a fifo
        # or socket (reported, never carried). A plan is the one document a reviewer, a
        # surface or a test can read to see what a copy will move.
        "trees": tree_entry_names(directory),
        "links": tree_link_entries(directory),
        # A symlinked content TREE is skipped by the walk rather than by `content_trees`,
        # so it is reported here beside the links the walk did find: both are entries this
        # copy will not carry, and a plan that stayed silent about the first was how the
        # round-3 MAJOR escaped notice.
        "trees_skipped": sorted(
            set(irregular_tree_entries(directory)) | set(symlinked_tree_entries(directory))
        ),
        "trees_unportable": unportable_tree_entries(directory),
        "transients": transient_entries(directory),
        "trees_excluded": excluded_cache_trees(directory),
    }


def _item_document(name: str, stamp: _MemberStamp, *, ref: str = "") -> dict[str, Any]:
    """One manifest item: what to send, how big, and the digest that verifies it.

    ``link`` distinguishes a carried symlink from a file (the holder writes a syMLINK,
    not the bytes), which is why the target text travels as the item's payload with its
    own length — the wire shape stays one kind of member, so a link needs no second code
    path in the fetch, the digest or the resumption check.
    """
    document: dict[str, Any] = {
        "name": name,
        "bytes": stamp.served_bytes,
        # The FILE's size, which differs from ``bytes`` only for a transcript whose torn
        # tail was cut: the owner's staleness check stats the file, so the number it must
        # compare is this one.
        "stat_bytes": stamp.stat_bytes,
        "mtime_ns": stamp.mtime_ns,
        "digest": stamp.digest,
    }
    if stamp.link:
        # A LINK TRAVELS AS ITS TARGET TEXT, in the plan rather than over the wire: it is
        # ≤ a few hundred bytes, and the holder can write it the moment it reads the plan.
        # The digest still gates it (``_write_link`` verifies it), so a target cannot be
        # swapped in flight.
        document["link"] = stamp.link
    if ref:
        document["ref"] = ref
    return document


def _blob_ref(name: str) -> str:
    """The digest a blob item's ``name`` refers to, for the sidecar AND its blob.

    ``attachments/<ref>.bin`` and ``attachments/<ref>.json`` are two members of ONE
    attachment; ``ref`` is that attachment's digest, so a surface counting attachments
    sees one thing rather than two files.
    """
    rest = name[len(ATTACHMENT_PREFIX) :]
    for suffix in (".bin", ATTACHMENT_SIDECAR_SUFFIX):
        if rest.endswith(suffix):
            return rest[: -len(suffix)]
    return rest


def _held_blob_key(name: str) -> str:
    """The key ``held_report`` uses for the blob member ``name``.

    The holder keys a blob by ``<ref>`` and its sidecar by ``<ref>.json`` (the two
    spellings that report has always used), so a plan's filter has to ask in the same
    vocabularies. Kept beside :func:`_blob_ref` because the pair is the one place those
    spellings are decided.
    """
    rest = name[len(ATTACHMENT_PREFIX) :]
    return rest if rest.endswith(ATTACHMENT_SIDECAR_SUFFIX) else _blob_ref(name)


def _blob_file_name(name: str) -> str | None:
    """The single-segment file name in a peer's ``attachments/<file>`` name, or ``None``.

    THE PEER-FACING GUARD FOR THE STORE, and it is a refusal rather than a lookup
    because ``name`` travels in a fetch/verify frame: ``attachments/../../etc/passwd``
    has to be rejected, not resolved. The store only ever writes one segment
    (``<32 hex>.bin`` / ``<32 hex>.json``), so anything with a separator, a ``..``
    or an empty remainder is not a name this protocol has.
    """
    rest = name[len(ATTACHMENT_PREFIX) :]
    if not rest or "/" in rest or "\\" in rest or rest in (".", ".."):
        return None
    return rest


def _attachment_path(store_dir: Path, name: str) -> Path | None:
    """The blob ``name`` refers to, INSIDE the blob directory ``store_dir``.

    The OWNER's convention: ``store_dir`` is the directory the blobs are in
    (``<root>/attachments``), so the wire prefix is stripped to find the file. The
    destination's convention is the other way round — see ``_destination_for``,
    which is handed the STORE ROOT and keeps the prefix.
    """
    file_name = _blob_file_name(name)
    return None if file_name is None else Path(store_dir) / file_name


def _tree_entry_path(base: Path, name: str, *, landed: bool = False) -> Path | None:
    """``name`` as a path inside a content tree, or ``None`` if it is not one.

    The same guard as :func:`_attachment_path`, for the same reason: an absolute
    name, a ``..`` segment, a backslash or an empty segment is refused rather than
    normalised, and the prefix has to be a tree of this directory.

    ``landed`` DECIDES WHICH QUESTION IS BEING ASKED, and the two are different:
    ``landed=False`` (the owner SERVING a plan) requires the tree to exist in
    ``content_trees(base)``, because a name for a tree this session does not have is a
    request about nothing; ``landed=True`` (a destination WRITING what a plan it accepted
    described) has to accept a tree that does not exist yet — the first member of a tree
    is what creates its directory. The path cannot escape either way: ``head`` and the
    remaining segments are checked to be single legal segments, so the result is always
    inside ``base``.
    """
    head = name.split("/", 1)[0]
    if not head or "\\" in head or head in (".", "..") or "/" in head:
        return None
    if not landed and head not in content_trees(base):
        return None
    rest = name[len(head) + 1 :]
    if not rest or "\\" in rest:
        return None
    parts = rest.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return Path(base) / head / Path(*parts)


def _item_path(root: Path, session_id: str, name: str, attachments_dir: Path | None) -> Path | None:
    """Where ``name`` lives on the owner, or ``None`` for a name outside the set.

    A REFUSAL, not a lookup: every branch is an allow-list check that cannot be
    walked out of, because ``name`` comes from a peer's frame.
    """
    if name.startswith(ATTACHMENT_PREFIX):
        store_dir = (
            Path(attachments_dir) if attachments_dir is not None else Path(root) / "attachments"
        )
        return _attachment_path(store_dir, name)
    tree_path = _tree_entry_path(session_dir(root, session_id), name)
    if tree_path is not None:
        return tree_path
    if name not in COPY_SET_NAMES or name in NEVER_COPIED:
        return None
    return session_dir(root, session_id) / name


def _stamp_refusal(
    root: Path, session_id: str, path: Path, name: str, frame: dict[str, Any]
) -> str:
    """``""`` when the member being served is still the one the plan described.

    THE CHEAP CHECK THAT REPLACED THE QUADRATIC ONE. It used to re-derive the whole
    copy set's digest on every 512 KiB chunk — 129 of the operator's scratchpads are
    over 32 MB and three are over 1 GB, so a move read the whole session once per chunk
    (review round 2, MAJOR 2). The holder knows the member's size and mtime from the
    plan, so one ``stat`` here answers "may I serve these bytes as the plan describes
    them": a member that was rewritten under the copy refuses with ``stale_plan``, which
    the holder answers by re-planning a full copy.

    WHAT THIS DOES NOT CATCH, stated rather than implied: a change that leaves both size
    and mtime identical (two writes inside one filesystem timestamp tick). That cannot
    corrupt a copy — the holder verifies the WHOLE member against the plan's digest when
    the member completes (``_fetch_item``), and the owner compares the two ends' content
    digests before it deletes anything — so it costs one failed copy, never a bad one.
    """
    plan = str(frame.get("plan_id") or "")
    expected_bytes = frame.get("expect_bytes")
    expected_mtime = frame.get("expect_mtime_ns")
    if expected_bytes is None or expected_mtime is None:
        stamp = _cached_stamp(root, session_id, plan, name)
        if stamp is None:
            # NOTHING TO COMPARE AGAINST: this process neither built the plan nor was
            # told its numbers (a peer from before this round, or a restart between the
            # plan and the fetch). The destination still verifies the whole member
            # against the plan's digest when the member completes, so the cost of
            # answering "" here is one failed copy, never a corrupt one.
            return ""
        expected_bytes, expected_mtime = stamp.stat_bytes, stamp.mtime_ns
    if path.is_symlink() and "/" not in name:
        # A ROOT-LEVEL LINK IS REFUSED AS ONE (review round 3, NIT 2). The stamp above was
        # taken through the link (a copy-set name is a FILE whose bytes travel), so the
        # lstat comparison below reported a size mismatch as "the conversation changed
        # while it was being copied" — a sentence that sends a person looking for a writer
        # that does not exist. This shape is refused the same way `unlisted_entries`
        # refuses it for a deleting move, so a `--keep` copy and a replica get the same
        # answer as the move does.
        return (
            f"{name} is a symlink rather than a file, and this build carries a session's "
            "own files, not a link: replace it with the file it points at, then try again"
        )
    try:
        stat = path.lstat() if path.is_symlink() else path.stat()
    except OSError:
        return f"{name} is no longer on this device"
    if int(expected_bytes) != stat.st_size:
        return (
            f"{name} changed while it was being copied ({stat.st_size} bytes where the plan "
            f"described {expected_bytes})"
        )
    if int(expected_mtime) != stat.st_mtime_ns:
        return f"{name} changed while it was being copied (rewritten on this device)"
    return ""


def _served_bytes(path: Path, name: str, *, whole_transcript: bool) -> bytes:
    payload = path.read_bytes()
    if name == TRANSCRIPT_NAME and not whole_transcript:
        return _safe_region(payload)
    return payload


def _cached_served_bytes(root: Path, session_id: str, plan: str, name: str) -> int | None:
    """The length of ``name``'s served bytes, from this process's plan record.

    ``None`` when the plan is not this process's (a restart between the plan and the
    fetch, or a peer from before this round): the caller then reads the member whole,
    which is correct and only slower.
    """
    stamp = _cached_stamp(Path(root), session_id, plan, name)
    return stamp.served_bytes if stamp is not None else None


def _served_span(
    path: Path,
    name: str,
    *,
    offset: int,
    limit: int,
    whole_transcript: bool,
    served_len: int | None = None,
) -> tuple[bytes, int, int]:
    """The requested span of one member, WITHOUT reading the rest of it.

    ``(span, file_bytes, eof)``. The whole-file read this replaces is the other half of
    MAJOR 2: every chunk of every file read the entire file again, so serving an 8 MB
    member in 512 KiB chunks read 128 MB. A regular file is read at its offset; a
    transcript a live runtime owns still goes through ``_safe_region`` (its served length
    is not the file's length), and a link is served from its target text.
    """
    if path.is_symlink():
        payload = os.readlink(path).encode("utf-8")
        return payload[offset : offset + limit], len(payload), offset + limit >= len(payload)
    if name != TRANSCRIPT_NAME or whole_transcript:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(offset)
            span = handle.read(limit)
        return span, size, offset + len(span) >= size
    if served_len is not None:
        # The safe region is a PREFIX of the file, so the region's bytes in
        # ``[offset, offset + limit)`` are the file's — read at the offset, not by
        # materialising the region.
        with path.open("rb") as handle:
            handle.seek(offset)
            span = handle.read(max(0, min(limit, served_len - offset)))
        return span, served_len, offset + len(span) >= served_len
    payload = _served_bytes(path, name, whole_transcript=whole_transcript)
    span = payload[offset : offset + limit]
    return span, len(payload), offset + len(span) >= len(payload)


def _plan_whole_transcript(root: Path, session_id: str, plan: str) -> bool:
    """Does the plan ``plan`` describe the transcript WHOLE, or as its safe region?

    THE PLAN DECIDES WHAT IS SERVED, never a property of the session at fetch time. The two
    variants produce different bytes for a torn tail, and the holder verifies what it
    receives against the digest ITS plan carried: serving the other variant hands it bytes
    that cannot verify. This matters in one real case — a replica pull that races a MOVE of
    the same session: its plan is the safe-region variant, so it must be served that variant
    even though the source's journal says a move is in flight. When this process no longer
    remembers the plan (a relay restart between the plan and the fetch) the session's own
    state is the only thing left to consult, which is the fallback here.
    """
    with _PLAN_CACHE_LOCK:
        record = _PLAN_CACHE.get((str(root), session_id, plan))
    if record is not None:
        return record[1]
    return _whole_transcript_for(root, session_id)


def _whole_transcript_for(root: Path, session_id: str) -> bool:
    """Is a DELETING move of ``session_id`` in flight on this device?

    THE OWNER'S OWN FACT, never the peer's claim. A move carries the transcript whole
    (``_safe_region``'s cut would delete a torn tail — review round 2), and the fact that
    a move is in flight is exactly the handoff journal entry ``_source_prepare`` writes
    before the destination can ask for a byte. ``--keep`` writes no entry and therefore
    serves the safe region, which is what a copy out of a live runtime must do.
    """
    from local_operator.session.placement import handoff_in_flight

    try:
        entry = handoff_in_flight(root, session_id)
    except Exception:  # noqa: BLE001 — an unreadable journal is the guard's refusal
        return False
    if not entry:
        return False
    return str(entry.get("mode") or "") == "move" and str(entry.get("role") or "source") == (
        "source"
    )


def serve_fetch(
    root: Path,
    session_id: str,
    *,
    plan: str,
    name: str,
    offset: int,
    limit: int = SYNC_CHUNK_BYTES,
    attachments_dir: Path | None = None,
    expect_bytes: int | None = None,
    expect_mtime_ns: int | None = None,
) -> dict[str, Any]:
    """One chunk of one file: the owner's ``net_sync {phase:"fetch"}`` answer.

    ``expect_bytes``/``expect_mtime_ns`` are the holder's copy of the plan's own stamp
    for this member; a mismatch is a ``stale_plan`` refusal (see ``_stamp_refusal``).
    ``file_digest`` is deliberately NOT returned any more: computing it meant hashing the
    whole file on every chunk, and the value the holder actually verifies is the plan's
    digest for that member, which it already holds (review round 2, MAJOR 2).
    """
    path = _item_path(root, session_id, name, attachments_dir)
    if path is None:
        raise SyncRefused("unknown_item", f"{name} is not part of a session copy")
    whole = _plan_whole_transcript(Path(root), session_id, plan)
    stale = _stamp_refusal(
        Path(root),
        session_id,
        path,
        name,
        {"plan_id": plan, "expect_bytes": expect_bytes, "expect_mtime_ns": expect_mtime_ns},
    )
    if stale:
        # THE REASON IS KEPT (review round 3, NIT 2). This raised with a fixed sentence,
        # "the conversation changed while it was being copied", which is right for a rewrite
        # and WRONG for the other shapes the same check answers — a member deleted mid-copy,
        # or a copy-set FILE that is a symlink rather than a file. Both sent a person looking
        # for a writer that does not exist; the sentence already computed names the member and
        # the shape, so it is used, with the one piece of advice the fixed sentence added.
        raise SyncRefused("stale_plan", f"{stale}; asking again picks up the new cut")
    try:
        span, file_bytes, eof = _served_span(
            path,
            name,
            offset=max(0, int(offset)),
            limit=max(0, int(limit)),
            whole_transcript=whole,
            # THE SERVED LENGTH, when this process still has the plan's stamp. For a
            # transcript a live runtime owns, the bytes that travel are the safe region —
            # not the file — so knowing its length is what lets the span be READ AT AN
            # OFFSET instead of reading the whole transcript per chunk (which is what the
            # replica path did, and is the same quadratic shape MAJOR 2 is about, one
            # member further along).
            served_len=_cached_served_bytes(root, session_id, plan, name),
        )
    except OSError as exc:
        raise SyncRefused("missing", f"{name} is not readable on this device") from exc
    start = max(0, int(offset))
    return {
        "name": name,
        "offset": start,
        "bytes": len(span),
        "data": base64.b64encode(span).decode("ascii"),
        "eof": eof,
        "chunk_digest": sha256_bytes(span),
        "file_bytes": file_bytes,
        "plan_id": plan,
    }


def serve_verify(
    root: Path,
    session_id: str,
    *,
    plan: str,
    name: str,
    prefix_bytes: int,
    prefix_digest: str,
    attachments_dir: Path | None = None,
    expect_bytes: int | None = None,
    expect_mtime_ns: int | None = None,
) -> dict[str, Any]:
    """Does the owner's first ``prefix_bytes`` of ``name`` hash to ``prefix_digest``?

    THE RESUMPTION CHECK. A destination interrupted mid-copy holds bytes it never
    saw verified as a whole file. Rather than trusting them (a splice) or throwing
    them away (a re-send of up to 100 MB), it asks this: a matching digest means
    those bytes are the source's own prefix, so the copy can continue from there
    with the same guarantee a fresh copy has.

    Called once per member, not once per chunk, so hashing the prefix here is not the
    per-chunk cost MAJOR 2 was about — and the member's stamp is still checked first.
    """
    path = _item_path(root, session_id, name, attachments_dir)
    if path is None:
        raise SyncRefused("unknown_item", f"{name} is not part of a session copy")
    stale = _stamp_refusal(
        Path(root),
        session_id,
        path,
        name,
        {"plan_id": plan, "expect_bytes": expect_bytes, "expect_mtime_ns": expect_mtime_ns},
    )
    if stale:
        # THE REASON IS KEPT (review round 3, NIT 2). This raised with a fixed sentence,
        # "the conversation changed while it was being copied", which is right for a rewrite
        # and WRONG for the other shapes the same check answers — a member deleted mid-copy,
        # or a copy-set FILE that is a symlink rather than a file. Both sent a person looking
        # for a writer that does not exist; the sentence already computed names the member and
        # the shape, so it is used, with the one piece of advice the fixed sentence added.
        raise SyncRefused("stale_plan", f"{stale}; asking again picks up the new cut")
    whole = _plan_whole_transcript(Path(root), session_id, plan)
    try:
        payload = _served_bytes(path, name, whole_transcript=whole)
    except OSError as exc:
        raise SyncRefused("missing", f"{name} is not readable on this device") from exc
    if prefix_bytes > len(payload):
        return {"matches": False, "reason": "shorter"}
    digest = hashlib.sha256(payload[:prefix_bytes]).hexdigest()
    return {"matches": "sha256:" + digest == prefix_digest}


# ---------------------------------------------------------------------------
# The holder's half: pull, verify, keep a replica
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PullResult:
    """What one pull did, in the shape both the replica and the move report."""

    session_id: str
    bytes: int
    cursor: dict[str, Any]
    transcript_mode: str
    items: int
    attachments: int
    resumed_bytes: int = 0
    #: ``name -> the digest of the bytes this device now holds`` for every member the
    #: pull accounted for, plus the names that are SYMLINKS. Together they are what a
    #: replica's cursor records, so a recovery can re-derive the same digest and refuse
    #: a copy something has since corrupted (review round 2, MINOR 3).
    files: dict[str, str] = field(default_factory=dict)
    links: tuple[str, ...] = ()


def pull(
    root: Path,
    session_id: str,
    *,
    ask: Callable[[dict[str, Any]], dict[str, Any]],
    dest_dir: Path,
    attachments_root: Path,
    have: dict[str, Any] | None = None,
    stop: Callable[[], bool] | None = None,
    purpose: str = "replica",
) -> PullResult:
    """Bring ``dest_dir`` up to the source's cut, verifying every byte.

    ``ask`` is the transport: it takes one ``net_sync`` frame and returns its
    ``detail``, raising :class:`SyncRefused` for a refusal. ONE callable rather
    than a link means this same code runs over a peer link in production, over a
    control socket in a test, and against a plain in-process callable against a
    second config root — which is what makes "every byte is verified" a claim that
    can be tested without two machines.
    """
    have = have or {}
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # THE PLAN IS ASKED FOR AGAIN WHEN IT CANNOT DESCRIBE THIS DESTINATION. An
    # "append" plan is a claim that this destination already holds the source's
    # first ``prefix_bytes``; if the file here is not that long, the claim is
    # about a copy that is no longer on disk, and continuing would splice. One
    # extra round trip re-plans as a full copy.
    plan: dict[str, Any] = {}
    for _ in range(2):
        plan = ask(
            {
                "op": "net_sync",
                "phase": "plan",
                "session_id": session_id,
                "have": dict(have or {}),
                # WHY the copy is being made, so the owner records the right fact
                # about this device (see ``sync_from``): a replica holder is
                # pushed to, a move's destination is about to own the id.
                "purpose": purpose,
            }
        )
        transcript = plan.get("transcript") or {}
        if str(transcript.get("mode") or "replace") != "append":
            break
        start = int(transcript.get("prefix_bytes") or 0)
        existing = (dest_dir / TRANSCRIPT_NAME).stat().st_size if transcript_exists(dest_dir) else 0
        if existing == start:
            break
        have = {key: value for key, value in (have or {}).items() if key != "cursor"}

    plan_value = str(plan.get("plan_id") or "")
    if not plan_value:
        raise SyncRefused("stale_plan", "the owner answered no plan for that conversation")

    transcript = plan.get("transcript") or {}
    mode = str(transcript.get("mode") or "replace")
    start = int(transcript.get("prefix_bytes") or 0)
    resize = int(transcript.get("region_bytes") or 0)
    total = int(transcript.get("total_bytes") or 0)
    digest = str(transcript.get("digest") or "")
    written = 0
    resumed = 0
    # ALWAYS. A source with a zero-byte transcript still needs that empty file
    # created and its digest (of no bytes) checked, or the destination's copy is
    # missing a member of the set it claims to hold.
    outcome = _fetch_item(
        ask,
        session_id=session_id,
        plan=plan_value,
        name=TRANSCRIPT_NAME,
        source_start=start,
        source_end=start + resize,
        dest_path=dest_dir / TRANSCRIPT_NAME,
        whole_digest=digest,
        final_bytes=total,
        stop=stop,
        expect_bytes=int(transcript.get("stat_bytes") or total),
        expect_mtime_ns=_optional_int(transcript.get("mtime_ns")),
    )
    written += outcome.written
    resumed += outcome.resumed

    items = 0
    blobs = 0
    links = 0
    attested: dict[str, str] = {}
    link_names: list[str] = []
    for item in list(plan.get("items") or []) + list(plan.get("attachments") or []):
        name = str(item.get("name") or "")
        if not name:
            continue
        # ``item_digest``, NOT ``digest``: the cursor below records the TRANSCRIPT's
        # digest, and shadowing it here wrote the last member's digest into every
        # replica cursor (found by the cursor test, which is what it is for).
        item_digest = str(item.get("digest") or "")
        if str(item.get("link") or ""):
            # A SYMLINK: its target travelled in the plan, so there is nothing to fetch —
            # and a link is written as a link, never as the bytes of whatever it points
            # at. Still staged and replaced, so a reader of this directory sees the old
            # link or the new one and never neither.
            _write_link(dest_dir, name, str(item["link"]), item_digest)
            attested[name] = item_digest
            link_names.append(name)
            links += 1
            continue
        target = _destination_for(dest_dir, Path(attachments_root), name)
        if target is None:
            raise SyncRefused("unknown_item", f"{name} is not part of a session copy")
        size = int(item.get("bytes") or 0)
        outcome = _fetch_item(
            ask,
            session_id=session_id,
            plan=plan_value,
            name=name,
            source_start=0,
            source_end=size,
            dest_path=target,
            whole_digest=item_digest,
            final_bytes=size,
            stop=stop,
            expect_bytes=_optional_int(item.get("stat_bytes")),
            expect_mtime_ns=_optional_int(item.get("mtime_ns")),
        )
        written += outcome.written
        resumed += outcome.resumed
        attested[name] = item_digest
        if name.startswith(ATTACHMENT_PREFIX):
            blobs += 1
        else:
            items += 1

    # EVERY MEMBER THE HOLDER NOW HOLDS, whether this pull fetched it or it was already
    # there (``have["files"]`` is the holder's own digest report, re-verified by the plan
    # on the owner's side). This is the set a replica's cursor records.
    held_raw = have.get("files")
    held: dict[str, Any] = held_raw if isinstance(held_raw, dict) else {}
    for name, value in held.items():
        attested.setdefault(str(name), str(value))
    held_blobs_raw = have.get("attachments")
    held_blobs: dict[str, Any] = held_blobs_raw if isinstance(held_blobs_raw, dict) else {}
    for name, value in held_blobs.items():
        # The caller keys blobs the way ``held_report`` does (``<ref>`` for a blob,
        # ``<ref>.json`` for its sidecar). A digest set is keyed by the WIRE name, so each
        # is rebuilt from that key rather than concatenated onto it — appending a suffix to
        # ``<ref>.json`` would invent a member no plan ever mentioned (``<ref>.json.bin``)
        # and a recovery would then find the replica's digest disagreeing with its own
        # cursor.
        file_name = name if name.endswith(ATTACHMENT_SIDECAR_SUFFIX) else f"{name}.bin"
        attested.setdefault(f"{ATTACHMENT_PREFIX}{file_name}", str(value))
    attested[TRANSCRIPT_NAME] = digest
    return PullResult(
        session_id=session_id,
        bytes=written,
        cursor={
            "prefix_bytes": total,
            "prefix_digest": digest,
            "frontier": str(transcript.get("frontier") or ""),
        },
        transcript_mode=mode,
        items=items,
        attachments=blobs,
        resumed_bytes=resumed,
        files=attested,
        links=tuple(link_names),
    )


def transcript_exists(directory: Path) -> bool:
    return (Path(directory) / TRANSCRIPT_NAME).is_file()


def _destination_for(dest_dir: Path, attachments_root: Path, name: str) -> Path | None:
    if name.startswith(ATTACHMENT_PREFIX):
        # THE PREFIX IS KEPT, and ``attachments_root`` is the STORE ROOT — the
        # directory that CONTAINS ``attachments/`` — rather than the blob directory
        # itself: a move points it at the config root, so a blob lands in the
        # install's shared content-addressed store (``<config>/attachments/<d>.bin``,
        # where the transcript's references resolve), while a replica points it at
        # itself, so the copy stays self-contained for a recovery or a delete.
        #
        # THAT IS WHY THIS LOOKUP IS THE PREVIOUS ONE'S BUG, NOT A TIDY-UP: a move
        # used to pass ``<config>/attachments`` here, the prefix was appended to it
        # again, and every moved image landed at
        # ``<config>/attachments/attachments/<d>.bin`` — a path nothing reads, so the
        # conversation rendered broken on the device it moved to (review round 1,
        # M-1). The store root goes in, the store-relative name comes out.
        file_name = _blob_file_name(name)
        if file_name is None:
            return None
        return Path(attachments_root) / ATTACHMENT_PREFIX / file_name
    tree_path = _tree_entry_path(Path(dest_dir), name, landed=True)
    if tree_path is not None:
        return tree_path
    if name not in COPY_SET_NAMES or name in NEVER_COPIED:
        return None
    return Path(dest_dir) / name


@dataclass(frozen=True)
class _FetchOutcome:
    written: int
    resumed: int


def _fetch_item(
    ask: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    session_id: str,
    plan: str,
    name: str,
    source_start: int,
    source_end: int,
    dest_path: Path,
    whole_digest: str,
    final_bytes: int,
    stop: Callable[[], bool] | None,
    expect_bytes: int | None = None,
    expect_mtime_ns: int | None = None,
) -> _FetchOutcome:
    """Fetch ``[source_start, source_end)`` into ``dest_path`` and verify the whole.

    THE DESTINATION'S OWN BYTES ARE THE ONLY STATE PERSISTED BETWEEN ATTEMPTS, so
    they are the thing that must be checked before they are appended to. A partial
    file is resumed only when the bytes already here are PROVEN to be the source's
    own prefix — either because the owner verified that exact length when it
    offered the append (``source_start``), or because it answers ``verify`` for the
    length this file happens to have (see ``serve_verify``). Anything else
    truncates and starts the file again: an unverified splice is a corrupt
    conversation that no later check would catch, and a re-send is only slow.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    existing = dest_path.stat().st_size if dest_path.is_file() else 0
    offset = source_start
    resumed = 0
    keep = 0
    if existing and source_start <= existing <= final_bytes:
        if existing == source_start:
            # EXACTLY THE OWNER-VERIFIED PREFIX. ``source_start`` is only non-zero
            # for an append, and an append is only offered after the owner hashed
            # its own first ``prefix_bytes`` against the cursor's digest — so these
            # bytes are already verified and asking again would be a second hash of
            # the same fact.
            keep = existing
        else:
            try:
                answer = ask(
                    {
                        "op": "net_sync",
                        "phase": "verify",
                        # EVERY frame names the session: the owner's handler reads it
                        # to find the files, and its plan guard is computed FROM it —
                        # a frame without one would be a request about nothing.
                        "session_id": session_id,
                        "plan_id": plan,
                        "name": name,
                        "prefix_bytes": existing,
                        "prefix_digest": _file_prefix_digest(dest_path, existing),
                        # The holder's copy of the plan's stamp for this member, so the
                        # owner answers "may I serve this?" with ONE stat instead of
                        # re-hashing the copy set (review round 2, MAJOR 2).
                        "expect_bytes": expect_bytes,
                        "expect_mtime_ns": expect_mtime_ns,
                    }
                )
            except SyncRefused:
                answer = {}
            if bool(answer.get("matches")):
                # The bytes here are the SOURCE's own first ``existing`` bytes, so
                # the copy continues from them. This is the resumption the design
                # promises ("the copy is resumable by construction"), made
                # checkable: without the check it would be a hopeful splice.
                keep = existing
                resumed = existing - source_start
    if source_start and not keep:
        # An append whose prefix is not on disk. Not a corruption to raise about:
        # the caller drops its cursor and asks for a plan that describes THIS
        # destination, which is a full copy.
        raise SyncRefused(
            "stale_plan",
            "this device no longer holds the part of the copy it had synced, so the rest of "
            "the conversation was asked for from the start",
        )
    if keep:
        offset = keep
    # STAGE, VERIFY, THEN REPLACE — never write into the live copy (review round 1,
    # M-5). A copy out of a runtime that is still writing used to be written in
    # place with ``"wb"``, so a process killed mid-append left a TRUNCATED file
    # where a complete one had been: the reviewer's probe cut the replica off
    # partway through a replace and the recovery promoted an 8,969-byte session
    # ending mid-row, while the cursor still reported 26,886 verified bytes. One
    # ``os.replace`` on the same filesystem means a reader of this directory sees
    # either the previous complete copy or the new complete copy, and an
    # interrupted attempt costs a re-send rather than the last good copy.
    staged = dest_path.with_name(f".{dest_path.name}.{os.getpid()}.fetch")
    staged.unlink(missing_ok=True)
    try:
        with staged.open("ab" if keep else "wb") as handle:
            if keep:
                _copy_prefix(dest_path, handle, keep)
            written = 0
            while offset < source_end:
                if stop is not None and stop():
                    raise SyncRefused("interrupted", "the copy was interrupted before it finished")
                chunk = ask(
                    {
                        "op": "net_sync",
                        "phase": "fetch",
                        "session_id": session_id,
                        "plan_id": plan,
                        "name": name,
                        "offset": offset,
                        "limit": min(SYNC_CHUNK_BYTES, source_end - offset),
                        "expect_bytes": expect_bytes,
                        "expect_mtime_ns": expect_mtime_ns,
                    }
                )
                payload = base64.b64decode(str(chunk.get("data") or ""))
                if sha256_bytes(payload) != str(chunk.get("chunk_digest") or ""):
                    raise SyncRefused(
                        "digest_mismatch",
                        f"the copy of {name} did not arrive intact, so it was lost",
                    )
                if not payload:
                    raise SyncRefused(
                        "digest_mismatch",
                        f"the copy of {name} stopped {source_end - offset} bytes short of "
                        "the source",
                    )
                handle.write(payload)
                offset += len(payload)
                written += len(payload)
        produced = staged.stat().st_size
        if produced != final_bytes:
            raise SyncRefused(
                "digest_mismatch",
                f"the copy of {name} is {produced} bytes where the source has "
                f"{final_bytes}, so it was not kept",
            )
        if whole_digest and sha256_bytes(staged.read_bytes()) != whole_digest:
            raise SyncRefused(
                "digest_mismatch", f"the copy of {name} did not verify, so it was not kept"
            )
        os.replace(staged, dest_path)
    except BaseException:
        # The staging file is this attempt's own garbage and NEVER the live copy:
        # removing it is what leaves the previous complete copy in place.
        staged.unlink(missing_ok=True)
        raise
    return _FetchOutcome(written=written, resumed=resumed)


def _write_link(dest_dir: Path, name: str, target: str, digest: str) -> None:
    """Write the carried symlink ``name`` under ``dest_dir``, pointing at ``target``.

    ``target`` is the RELATIVE text the owner computed (``_portable_link_target``), never
    the source's absolute spelling, which is what makes the link mean the same thing on
    this device. The digest is re-verified here rather than trusted: it arrives in the
    same plan as every file's, and a link is a member like any other — checking it costs
    one hash of a path string and closes the only hole a plan could open (a swapped
    target, which would point an agent at a file the owner never named).

    Staged and ``os.replace``d like every other member, so a reader of this directory sees
    the previous link or the new one and never a half-written one.
    """
    path = _tree_entry_path(Path(dest_dir), name, landed=True)
    if path is None:
        raise SyncRefused("unknown_item", f"{name} is not part of a session copy")
    if sha256_bytes(target.encode("utf-8")) != digest:
        raise SyncRefused(
            "digest_mismatch",
            f"the link {name} did not arrive as the owner described it, so it was not " "written",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.{os.getpid()}.link")
    staged.unlink(missing_ok=True)
    try:
        os.symlink(target, staged)
        os.replace(staged, path)
    except OSError as exc:
        staged.unlink(missing_ok=True)
        raise SyncRefused(
            "no_replica", f"the link {name} could not be written on this device ({exc})"
        ) from exc


def _copy_prefix(source: Path, handle: Any, count: int) -> None:
    """Copy ``source``'s first ``count`` bytes into the staging ``handle``.

    The bytes are the caller's already-verified prefix (``keep`` is only ever set
    to a length the owner proved or answered ``verify`` for), so this is a local
    copy of trusted bytes rather than a second check — the whole-file digest below
    still has to pass before the file is adopted.
    """
    remaining = count
    with source.open("rb") as origin:
        while remaining > 0:
            block = origin.read(min(1 << 20, remaining))
            if not block:
                break
            handle.write(block)
            remaining -= len(block)
    if remaining:
        raise SyncRefused(
            "stale_plan",
            "the part of the copy this device held is no longer on disk, so the rest of "
            "the conversation was asked for from the start",
        )


def _file_prefix_digest(path: Path, prefix_bytes: int) -> str:
    """sha256 of the first ``prefix_bytes`` of ``path``, streamed."""
    digest = hashlib.sha256()
    remaining = prefix_bytes
    with path.open("rb") as handle:
        while remaining > 0:
            block = handle.read(min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return "sha256:" + digest.hexdigest()


def held_report(dest_dir: Path, attachments_root: Path) -> dict[str, Any]:
    """The ``have`` document for a destination that already holds some of the set.

    Keys are the SAME names the manifest uses: copy-set names, flat tree names
    (``scratchpad/<rel>``) and the store-relative blob names including their
    ``.json`` sidecars. A name spelled differently here would be a file the
    destination re-downloads on every resume — slow, not wrong — so the two lists
    are derived from one place each (``COPY_SET_NAMES`` + ``tree_entry_names``).
    """
    files: dict[str, str] = {}
    for name in list(COPY_SET_NAMES) + tree_entry_names(dest_dir):
        if name == TRANSCRIPT_NAME:
            continue
        path = Path(dest_dir) / name
        try:
            files[name] = sha256_bytes(path.read_bytes())
        except OSError:
            continue
    # THE LINKS THIS DESTINATION ALREADY HOLDS, keyed and digested exactly as the plan
    # spells them, so a resume does not re-send a link it has (and so a link whose target
    # the SOURCE changed is not mistaken for one already carried).
    for name, target in tree_link_entries(Path(dest_dir)):
        files[name] = sha256_bytes(target.encode("utf-8"))
    blobs: dict[str, str] = {}
    store = Path(attachments_root) / ATTACHMENT_PREFIX
    for pattern, keyed_by_stem in (("*.bin", True), (f"*{ATTACHMENT_SIDECAR_SUFFIX}", False)):
        try:
            for candidate in sorted(store.glob(pattern)):
                key = candidate.stem if keyed_by_stem else candidate.name
                blobs[key] = sha256_bytes(candidate.read_bytes())
        except OSError:
            pass
    return {"files": files, "attachments": blobs}


def sync_from(
    root: Path,
    session_id: str,
    *,
    ask: Callable[[dict[str, Any]], dict[str, Any]],
    owner_device: str = "",
    into: Path | None = None,
    attachments_root: Path | None = None,
    write_cursor: bool = True,
    purpose: str = "replica",
) -> dict[str, Any]:
    """Pull ``session_id`` from its owner — into the replica store, or a staging dir.

    ONE implementation for both, because a move's copy and a replica's copy must
    not be able to disagree about what they copy or what they verify. ``into`` and
    ``attachments_root`` are the whole difference; a move points them at its
    staging area and the install's shared content-addressed store (a re-downloaded
    blob would be wasted bytes), a replica points them at itself (so one directory
    can be recovered or deleted as a unit).

    ``purpose`` travels to the owner with the plan request, because the two cases
    are not the same fact to record: a replica holder is somebody to PUSH to
    (``record_replica``), while a move's destination is a device that is about to
    OWN the conversation — registering it as a holder made the owner's watcher push
    a copy to a device that had already been handed the id, and made a REFUSED move
    mutate the source's stamp (review round 1, MINOR 1).
    """
    destination = Path(into) if into is not None else replica_dir(root, session_id)
    blobs = Path(attachments_root) if attachments_root is not None else destination
    destination.mkdir(parents=True, exist_ok=True)
    cursor = read_replica_cursor(root, session_id).get("cursor") if write_cursor else None
    have: dict[str, Any] = held_report(destination, blobs)
    if cursor:
        have["cursor"] = cursor
    last: SyncRefused | None = None
    for _attempt in range(SYNC_REPLAN_ATTEMPTS):
        try:
            result = pull(
                root,
                session_id,
                ask=ask,
                dest_dir=destination,
                attachments_root=blobs,
                have=have,
                purpose=purpose,
            )
        except SyncRefused as exc:
            if exc.code != "stale_plan":
                raise
            # The source moved under us (a turn landed, a compaction ran), or this
            # destination no longer holds the prefix it reported. Either way the
            # next attempt is a plan that describes what is really there: the
            # cursor is dropped, so the owner offers a full copy.
            last = exc
            have.pop("cursor", None)
            continue
        if write_cursor:
            write_replica_cursor(
                root,
                session_id,
                {
                    "owner_device": owner_device,
                    "cursor": result.cursor,
                    "bytes": result.bytes,
                    "resumed_bytes": result.resumed_bytes,
                    "items": result.items,
                    "attachments": result.attachments,
                    "transcript_mode": result.transcript_mode,
                    "last_synced_at": time.time(),
                    # WHAT THIS DEVICE CAN ATTEST TO, member by member, and the digest of
                    # that set. The cursor used to record the transcript alone, so a
                    # recovery promoted a replica with a CORRUPTED ``title.json`` (review
                    # round 2, MINOR 3): the promoted session came back wearing a title
                    # its owner never wrote, and nothing had verified it. Recovery
                    # re-derives this digest before it promotes a byte.
                    "files": result.files,
                    "links": list(result.links),
                    "content_digest": _attested_digest(result.files, result.links, session_id),
                },
            )
        return {
            "ok": True,
            "session_id": session_id,
            "owner_device": owner_device,
            "bytes": result.bytes,
            "resumed_bytes": result.resumed_bytes,
            "items": result.items,
            "attachments": result.attachments,
            "mode": result.transcript_mode,
            "cursor": result.cursor,
        }
    raise last or SyncRefused("stale_plan", "the conversation kept changing while it was copied")


# ---------------------------------------------------------------------------
# Recovery: a replica becomes a fork with a NEW id (never the original)
# ---------------------------------------------------------------------------


def _verify_replica_against_cursor(root: Path, session_id: str, source: Path) -> None:
    """Refuse to recover a replica that disagrees with the cursor's own record.

    THE CURSOR IS THE REPLICA'S OWN PROOF OF COMPLETENESS, and it is written by
    ``sync_from`` only after every byte verified, so it is the one authority on how
    long this copy should be. Recovery never consulted it: the reviewer's probe cut
    a replica off partway through a replace and the recovery promoted a
    session ending MID-ROW, 8,969 bytes of the 26,886 the owner had verified
    (review round 1, M-5). Since ``_fetch_item`` now replaces atomically the live
    case is a killed process or a disk-level truncation — both of which leave the
    bytes disagreeing with the cursor, which is exactly what this catches.

    A replica with NO cursor refuses too: nothing has verified those bytes, and
    presenting an unverified directory as a conversation is the failure this whole
    span exists to prevent.
    """
    record = read_replica_cursor(root, session_id) or {}
    # THE INNER CURSOR is the transcript's position; the RECORD BESIDE IT holds what the
    # rest of the copy set was verified to be (``files``, ``links`` and their digest).
    # Reading both from one place put the full-set check in a dict that never carried it,
    # which is how a corrupted ``title.json`` was promoted (review round 2, MINOR 3).
    cursor = record.get("cursor") or {}
    expected = int(cursor.get("prefix_bytes") or 0)
    digest = str(cursor.get("prefix_digest") or "")
    transcript = source / TRANSCRIPT_NAME
    try:
        size = transcript.stat().st_size
    except OSError as exc:
        raise SyncRefused(
            "no_replica", f"this device's copy of {session_id} has no transcript ({exc})"
        ) from exc
    if not expected or not digest or size != expected:
        raise SyncRefused(
            "incomplete_replica",
            f"this device's copy of {session_id} is incomplete ({size} bytes where the "
            f"synced copy records {expected}), so it was not recovered as a session; "
            "syncing it again replaces it with a complete copy",
        )
    if _file_prefix_digest(transcript, expected) != digest:
        raise SyncRefused(
            "incomplete_replica",
            f"this device's copy of {session_id} does not match the bytes that were "
            "verified when it was synced, so it was not recovered as a session; syncing "
            "it again replaces it with a complete copy",
        )
    # THE REST OF WHAT A RECOVERY PROMOTES (review round 2, MINOR 3). The sentences above
    # check the transcript, and a recovery promotes the WHOLE copy set plus its trees —
    # so a corrupted ``title.json`` used to be promoted as-is, because the cursor had no
    # digest for it. Two checks, in order of what they can say:
    #
    # * per member, when the cursor names that member, so a failure names the FILE. A
    #   member the cursor does not mention is one this build never verified (a cursor
    #   written by an older build, or a file added since) and is left to the digest below;
    # * over the whole attested set, which is the claim the cursor exists to make.
    #
    # WHAT IS TRUSTED, stated rather than implied: the CONTENT the last sync verified
    # (every member's digest, and now its trees and links). What is NOT checked here is
    # whether the OWNER still holds the same bytes — a replica is a snapshot, and the
    # promoted session is stamped as a fork for exactly that reason. Nothing about the
    # source's liveness or ownership is re-asked, and no member outside the copy set is
    # promoted at all: ``_promote_replica`` copies the copy-set names, the content trees
    # and the carried links, and nothing else.
    recorded_raw = record.get("files")
    recorded: dict[str, Any] = recorded_raw if isinstance(recorded_raw, dict) else {}
    # WHERE EACH MEMBER'S BYTES LAND: a blob lives in this device's own attachments
    # directory, never inside the session directory (``_member_stamps`` reads its store
    # through ``attachments_dir``), so resolving the name per member is what keeps a blob's
    # absence from being read as a missing session member.
    store = replica_dir(root, session_id) / ATTACHMENTS_DIRNAME
    for name, wanted in sorted(recorded.items()):
        if name == TRANSCRIPT_NAME:
            continue
        landing = (
            store / str(name)[len(ATTACHMENT_PREFIX) :]
            if str(name).startswith(ATTACHMENT_PREFIX)
            else source / str(name)
        )
        if not landing.is_symlink() and not landing.is_file():
            # A MEMBER THE CURSOR ATTESTED AND THAT IS NO LONGER ON DISK IS A MISMATCH,
            # NAMED (review round 3, MINOR 3). The whole-set digest below does refuse this
            # recovery — the promoted copy is missing a member the sync verified — but it
            # refuses it anonymously, and the sentence is what a person acts on: a
            # rewritten ``title.json`` named its file, a deleted one did not.
            raise SyncRefused(
                "incomplete_replica",
                f"this device's copy of {session_id} no longer matches the bytes that were "
                f"verified when it was synced ({name} is missing), so it was not recovered "
                "as a session; syncing it again replaces it with a complete copy",
            )
        got = (
            sha256_bytes(os.readlink(landing).encode("utf-8"))
            if landing.is_symlink()
            else _stream_digest(landing)
        )
        if got != str(wanted):
            raise SyncRefused(
                "incomplete_replica",
                f"this device's copy of {session_id} no longer matches the bytes that were "
                f"verified when it was synced ({name} has changed), so it was not recovered "
                "as a session; syncing it again replaces it with a complete copy",
            )
    attested = str(record.get("content_digest") or "")
    if attested:
        # RE-DERIVED FROM THE REPLICA ON DISK, over the same function and the same member
        # set the cursor's digest was built from, so this compares a claim about THIS
        # device's bytes with those bytes.
        current = _content_digest_from(
            _member_stamps(
                source,
                session_id,
                replica_dir(root, session_id) / ATTACHMENTS_DIRNAME,
            ),
            session_id,
        )
        if current != attested:
            raise SyncRefused(
                "incomplete_replica",
                f"this device's copy of {session_id} no longer matches the content verified "
                "when it was synced, so it was not recovered as a session; syncing it again "
                "replaces it with a complete copy",
            )


def promote_replica(root: Path, session_id: str, *, new_id: str = "") -> dict[str, Any]:
    """Turn a replica into a LOCAL session under a freshly minted id.

    WHY A NEW ID, the one place this build deliberately departs from the design
    (build plan §7, unsafe item 4). The design promotes a replica under the
    ORIGINAL id for a pool member that is expected to die; that reason is sound
    there and unsound here. A peer that spins down can come back — an EC2 instance
    restarts, a laptop reopens — and a device that comes back holds its directory
    and will spawn a runtime for the id. Two devices, one id, both live, both
    appending: exactly INV-1, and nothing downstream can tell which conversation
    is the real one. A new id costs a rename and one line of prose, which
    ``origin`` supplies ("copy of <id> from <device>"); a resurrected id costs a
    forked trajectory.
    """
    import shutil

    from local_operator.fork import (
        FORK_BOUNDARY_NAME,
        FORK_BOUNDARY_VERSION,
        new_session_id,
    )
    from local_operator.resume import ORIGIN_FORK, mark_session_origin
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )

    source = replica_dir(root, session_id)
    if not transcript_exists(source):
        raise SyncRefused(
            "no_replica", f"this device holds no synced copy of {session_id} to recover"
        )
    _verify_replica_against_cursor(root, session_id, source)
    new_id = new_id or new_session_id()
    if new_id == session_id:
        # THE ONE ID A RECOVERY MAY NEVER TAKE. The only caller mints a fresh id, but
        # a rule enforced in the caller is a rule the next caller inherits by
        # accident: a same-id promotion is two writers on one transcript the moment
        # the device that went away comes back (INV-1, and this function's own
        # docstring). Refused here rather than documented.
        raise SyncRefused(
            "in_progress",
            f"a copy of {session_id} cannot be recovered under the same id; the device "
            "that holds the original would become a second writer",
        )
    target = Path(root) / "sessions" / new_id
    if target.exists():
        raise SyncRefused("in_progress", f"{new_id} already exists on this device")
    # CLAIM BEFORE the first byte, exactly as ``fork_session`` does, so a
    # concurrent retention sweep can never see an empty unclaimed directory. The
    # claim is released below: it names THIS process, and leaving it would make the
    # promoted session read as open in the prompter's own process.
    from local_operator.session.retention import claim_session, release_session

    claim_session(target)
    copied: list[str] = []
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in COPY_SET_NAMES:
            if name in NEVER_COPIED:
                continue
            candidate = source / name
            if not candidate.is_file():
                continue
            shutil.copyfile(candidate, target / name)
            copied.append(name)
        # THE COPY SET'S TREES COME TOO, or the recovered conversation silently
        # loses the files its own agent wrote for it (``scratchpad/``), which is
        # the same loss this round fixes for a move (review round 1, B-M2).
        for name in tree_entry_names(source):
            candidate = source / name
            landing = target / name
            landing.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(candidate, landing)
            copied.append(name)
        # AND THE CARRIED LINKS, recreated as links rather than as copies of what they
        # point at (review round 2's real-store finding: 6,629 of the operator's symlinks
        # are inside their own session, and dropping them here would make a recovery lose
        # a different thing from the move that filled the replica).
        for name, link_target in tree_link_entries(source):
            landing = target / name
            landing.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.symlink(link_target, landing)
            except FileExistsError as exc:
                # The directory was claimed and created by this call, so a name already
                # present is either a member copied a moment ago (impossible: one name, one
                # kind) or somebody else writing into the target. Refusing beats removing a
                # file this function did not create.
                raise SyncRefused(
                    "in_progress", f"{new_id} already holds {name}; nothing was recovered"
                ) from exc
            copied.append(name)
        # The replica's blobs live inside it; the recovered session's transcripts
        # reference them by digest, so they have to reach the SHARED store or the
        # recovered conversation renders broken images. The ``.json`` sidecar
        # carries the mime type and was already copied below; a blob without it is
        # a download of the wrong type, which is why the two travel together.
        blob_source = source / "attachments"
        if blob_source.is_dir():
            store = Path(root) / "attachments"
            store.mkdir(parents=True, exist_ok=True)
            for blob in sorted(blob_source.glob("*.bin")):
                shutil.copyfile(blob, store / blob.name)
                sidecar = blob.with_name(f"{blob.stem}{ATTACHMENT_SIDECAR_SUFFIX}")
                if sidecar.is_file():
                    shutil.copyfile(sidecar, store / sidecar.name)
        # The lineage boundary rides EVERY recovered copy, exactly as it rides a
        # fork: the transcript is inherited, and the model must be told not to
        # continue work that may still be running on the device it came from.
        (target / FORK_BOUNDARY_NAME).write_text(
            json.dumps({"version": FORK_BOUNDARY_VERSION, "created_at": time.time()}),
            encoding="utf-8",
        )
    except OSError as exc:
        release_session(target)
        raise SyncRefused(
            "no_replica", f"the synced copy could not be recovered on this device ({exc})"
        ) from exc
    release_session(target)
    cursor = read_replica_cursor(root, session_id)
    owner = str(cursor.get("owner_device") or "")
    from local_operator.network.projection import self_device_id

    write_stamp(
        root,
        MeshStamp(
            session_id=new_id,
            network_id="",
            home_device=self_device_id(Path(root)),
            placement=SessionPlacement(mode="local", home_device=self_device_id(Path(root))),
            origin={
                "kind": REPLICA_ORIGIN_KIND,
                "source_device": owner,
                "source_session_id": session_id,
                "recovered_at": time.time(),
            },
        ),
    )
    try:
        mark_session_origin(target, ORIGIN_FORK, parent=session_id)
    except Exception:  # noqa: BLE001 — provenance decoration, never a gate
        logger.debug("sync: could not stamp the origin of %s", new_id, exc_info=True)
    return {
        "ok": True,
        "session_id": new_id,
        "source_session_id": session_id,
        "owner_device": owner,
        "files": copied,
    }


#: The stamp ``origin.kind`` a recovered replica carries. ``fork`` and not
#: ``moved``: the copy is a divergence point, because the owner may still be
#: alive with the same conversation — see :func:`promote_replica`.
REPLICA_ORIGIN_KIND = "fork"


def replica_summary(root: Path, session_id: str) -> dict[str, Any]:
    """What this device knows about its replica, for a listing or a sentence."""
    cursor = read_replica_cursor(root, session_id)
    if not cursor:
        return {}
    return {
        "session_id": session_id,
        "owner_device": str(cursor.get("owner_device") or ""),
        "last_synced_at": float(cursor.get("last_synced_at") or 0.0),
        "bytes": int(cursor.get("bytes") or 0),
        "cursor": cursor.get("cursor") or {},
    }


def replicas_held(root: Path) -> list[str]:
    """Every session id this device holds a replica of, sorted."""
    directory = network_dir(root) / REPLICA_DIRNAME
    try:
        return sorted(child.name for child in directory.iterdir() if child.is_dir())
    except OSError:
        return []


# ---------------------------------------------------------------------------
# The owner's watcher: tell holders a new cut exists (§7.5)
# ---------------------------------------------------------------------------


@dataclass
class _Watched:
    """One replicated session, as the watcher last saw it."""

    fingerprint: tuple[int, int]
    changed_at: float | None
    pushed: tuple[int, int] | None
    present: bool


class SyncWatcher(threading.Thread):
    """The owner's tick: a stat per replicated session, then a debounced push.

    NOT a filesystem watcher: one ``stat`` per replicated session per tick is the
    whole cost and needs no platform API. The two triggers are the design's
    (§1.1), and the second is what makes "the final message before idle" arrive
    with no runtime change at all:

    * a transcript that has changed and then been QUIET for ``debounce_s`` — one
      push per turn instead of one per token batch;
    * the runtime's registry record DISAPPEARING. That is the idle exit, and at
      that instant the last turn's bytes are already on disk.
    """

    def __init__(self, server: "RelayServer", settings: SyncSettings | None = None) -> None:
        super().__init__(name="mesh-sync-watch", daemon=True)
        self._server = server
        self._settings = settings or SyncSettings.from_config(server.root)
        self._seen: dict[str, _Watched] = {}
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:  # pragma: no cover — the tick is what tests drive
        while not self._stop.wait(self._settings.tick_s):
            try:
                self.tick(now=time.time())
            except Exception:  # noqa: BLE001 — a watcher must never kill the relay
                logger.debug("sync: watcher tick failed", exc_info=True)

    def tick(self, *, now: float) -> list[tuple[str, str]]:
        """One pass. Returns ``(session_id, reason)`` per push, so cadence is testable.

        Injected ``now`` rather than an internal clock read: the debounce window is
        this class's whole behaviour, and a test that has to sleep 30 real seconds
        to observe it is a test nobody runs.
        """
        from local_operator.session.placement import read_stamp
        from local_operator.session.runtime import registry

        live = {
            record.session_id
            for record, state in registry.scan(self._server.root)
            if state == "live"
        }
        pushed: list[tuple[str, str]] = []
        for session_id, holders in self._replicated().items():
            try:
                stat = (Path(self._server.root) / "sessions" / session_id / TRANSCRIPT_NAME).stat()
            except OSError:
                read_stamp(self._server.root, session_id)
                continue
            fingerprint = (int(stat.st_mtime_ns), int(stat.st_size))
            present = session_id in live
            state = self._seen.get(session_id)
            reason = ""
            if state is None:
                state = _Watched(
                    fingerprint=fingerprint, changed_at=None, pushed=fingerprint, present=present
                )
            else:
                if state.fingerprint != fingerprint:
                    state.fingerprint = fingerprint
                    state.changed_at = now
                if state.changed_at is not None and (
                    now - state.changed_at >= self._settings.debounce_s
                ):
                    reason = "quiet"
                if state.present and not present:
                    # THE IDLE EXIT, and it outranks the debounce: the device that
                    # was writing has stopped, so this is the last chance to say so
                    # before the holder's view goes stale for a whole tick.
                    reason = "idle-exit"
                state.present = present
            if reason:
                state.changed_at = None
                # THE PUSH HAPPENS ONLY WHEN THERE IS SOMETHING TO SAY. Without
                # this guard every tick notified every holder — a frame per 15 s per
                # session per holder, which is a keepalive nobody reads and a wakeup
                # on a phone that should be idle.
                pushed.extend(
                    (session_id, reason)
                    for holder in holders
                    if self._push(holder, session_id, reason)
                )
            self._seen[session_id] = state
        return pushed

    def _replicated(self) -> dict[str, list[str]]:
        """Session id -> replica holders, from the stamps on this device.

        The stamp is the only place a replica holder is recorded, and it is the
        OWNER's own file: a holder learns nothing from another holder, so one
        writer per stamp (this one) is what keeps it consistent.
        """
        from local_operator.session.placement import read_stamp

        found: dict[str, list[str]] = {}
        try:
            children = list((Path(self._server.root) / "sessions").iterdir())
        except OSError:
            return {}
        for child in children:
            if not child.is_dir():
                continue
            stamp = read_stamp(self._server.root, child.name)
            if stamp is None or not stamp.replicas:
                continue
            found[child.name] = list(stamp.replicas)
        return found

    def _push(self, holder: str, session_id: str, reason: str) -> bool:
        """One ``available`` notification. Best effort by contract.

        A holder that cannot be reached is simply not told: it pulls on its next
        attach or relay start, and a notification that retried would spend a link
        dial on a device that is already gone.
        """
        link = self._server._ensure_link(holder)  # noqa: SLF001 — the one dial seam
        if link is None:
            return False
        try:
            reply = link.request(
                {
                    "op": "net_sync",
                    "req": self._server._next_relay_req(),  # noqa: SLF001
                    "phase": "available",
                    "session_id": session_id,
                    "reason": reason,
                }
            )
        except Exception:  # noqa: BLE001 — a notification is never worth a crash
            logger.debug("sync: could not tell %s about %s", holder, session_id, exc_info=True)
            return False
        return bool(reply and reply.get("op") == "ack")


#: One watcher per relay process. Keyed by relay object id, because a process can
#: hold two relays (the test suite does) and one watcher cannot serve both roots.
_watcher_lock = threading.Lock()
_watchers: dict[int, SyncWatcher] = {}


class ReplicaRefresher(threading.Thread):
    """The HOLDER's tick: pull the replicas a push just said had changed.

    WHY THIS EXISTS (review round 1, M-4). A push was ACKNOWLEDGED and nothing
    followed it: no code in the tree called :func:`sync_from` except
    ``lop sessions sync`` and a move, so a replica never refreshed on its own and
    never came to hold the last assistant message. The owner's watcher already
    sends the design's two triggers; this is the half that acts on them.

    CADENCE, from the settings P0 put in the config: one pull per marked replica per
    ``network.sync.tick_s``. A burst of pushes for one session therefore costs one
    copy rather than one per frame, and a replica is at most one tick behind a
    change. A push that arrives while a pull for that session is running marks it
    again, so the later change is not lost; ``network.sync.debounce_s`` stays the
    OWNER's window (what stops a push per token batch), not the holder's floor.

    NEVER on the link's reader thread: ``sync_from`` asks the peer questions, and
    ``PeerLink.request`` refuses a request issued by the handler serving that very
    link (build plan §0 finding 4). The ack is returned first and the pull happens
    here, on this thread.
    """

    def __init__(self, server: "RelayServer", settings: SyncSettings | None = None) -> None:
        super().__init__(name="mesh-replica-refresh", daemon=True)
        self._server = server
        self._settings = settings or SyncSettings.from_config(server.root)
        self._lock = threading.Lock()
        self._marked: dict[str, str] = {}
        self._running: set[str] = set()
        self._again: set[str] = set()
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def mark(self, session_id: str, reason: str) -> None:
        """Note that this device's copy of ``session_id`` is behind."""
        with self._lock:
            if session_id in self._running:
                self._again.add(session_id)
            else:
                self._marked[session_id] = reason

    def run(self) -> None:  # pragma: no cover — the tick is what tests drive
        while not self._stop.wait(self._settings.tick_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 — a refresher must never kill the relay
                logger.debug("sync: replica refresh tick failed", exc_info=True)

    def tick(self) -> list[tuple[str, str]]:
        """One pass. Returns ``(session_id, outcome)`` per attempt, so it is testable.

        Driven with an injected pass rather than only by the thread, for the same
        reason :meth:`SyncWatcher.tick` is: the behaviour under test is WHAT HAPPENS
        ON A TICK, and a test that had to wait 15 real seconds to see one is a test
        nobody runs.
        """
        with self._lock:
            due = sorted(self._marked.items())
            self._marked.clear()
            self._running = {session_id for session_id, _reason in due}
        outcomes: list[tuple[str, str]] = []
        for session_id, reason in due:
            outcomes.append((session_id, self._refresh(session_id, reason)))
        with self._lock:
            self._running.clear()
            for session_id in sorted(self._again):
                self._marked.setdefault(session_id, "again")
            self._again.clear()
        return outcomes

    def _refresh(self, session_id: str, reason: str) -> str:
        """Pull one replica from its recorded owner over a real link."""
        from local_operator.network.mobility import LinkTransport, Moved

        owner = str(read_replica_cursor(self._server.root, session_id).get("owner_device") or "")
        if not owner:
            # No recorded owner: this device synced before the cursor carried one,
            # or the cursor was lost. Nothing is guessed here — a pull needs a peer
            # to ask, and ``lop sessions sync <id>`` resolves it from the catalogue.
            logger.debug("sync: %s has no recorded owner, so %s was not pulled", session_id, reason)
            return "no_owner"
        link = self._server._ensure_link(owner)  # noqa: SLF001 — the one dial seam
        if link is None:
            # A holder that cannot be reached is simply not refreshed: the next push
            # marks it again, and a retry would spend a link dial on a device that is
            # already gone.
            return "unreachable"
        transport = LinkTransport(self._server, link, session_id)
        try:
            sync_from(self._server.root, session_id, ask=transport.ask, owner_device=owner)
        except Moved as refusal:
            return f"refused:{refusal.code}"
        except SyncRefused as refusal:
            return f"refused:{refusal.code}"
        except Exception:  # noqa: BLE001 — a refresh is best effort by contract
            logger.debug("sync: could not refresh %s", session_id, exc_info=True)
            return "failed"
        return "pulled"


#: One refresher per relay process, keyed like the watcher and for the same reason.
_refresher_lock = threading.Lock()
_refreshers: dict[int, ReplicaRefresher] = {}


def ensure_refresher(server: "RelayServer") -> ReplicaRefresher:
    """Start this relay's replica refresher if it is not already running.

    LAZY, from the first push: before a peer has ever told this device that a
    replica changed there is nothing to pull, and starting a thread at construction
    would mean a thread per relay in a suite that builds hundreds it never starts.
    """
    with _refresher_lock:
        existing = _refreshers.get(id(server))
        if existing is not None and existing.is_alive():
            return existing
        refresher = ReplicaRefresher(server)
        _refreshers[id(server)] = refresher
        refresher.start()
        return refresher


def mark_replica_available(server: "RelayServer", session_id: str) -> None:
    """A peer says this device's copy is behind: make sure something pulls it."""
    if not session_id:
        return
    ensure_refresher(server).mark(session_id, "available")


def ensure_watcher(server: "RelayServer") -> SyncWatcher:
    """Start this relay's watcher if it is not already running.

    LAZY, from the first holder that pulls: a watcher exists to push cuts to
    holders, so before a holder has ever asked there is nothing to push and
    nowhere to push it. Starting it at construction would also mean a thread per
    relay in a test suite that builds hundreds of relays it never starts.
    """
    with _watcher_lock:
        existing = _watchers.get(id(server))
        if existing is not None and existing.is_alive():
            return existing
        watcher = SyncWatcher(server)
        _watchers[id(server)] = watcher
        watcher.start()
        return watcher


# ---------------------------------------------------------------------------
# The relay's handler
# ---------------------------------------------------------------------------


def make_handler(server: "RelayServer") -> Callable[[Any, dict[str, Any]], dict[str, Any]]:
    """The ``net_sync`` handler for one relay.

    It never issues a request over the link it is answering: the only outbound
    call this module makes is the watcher's ``available`` push, which runs on its
    own thread for a different link. That is what keeps this op out of the
    deadlock ``PeerLink.request`` refuses (build plan §0 finding 4).
    """

    def _handle(link: Any, frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame.get("session_id") or "")
        if not session_id:
            raise _refusal("bad_request", "a sync needs a conversation id")
        if phase == "plan":
            return _plan(server, link, frame)
        if phase == "fetch":
            return _fetch(server, frame)
        if phase == "verify":
            return _verify(server, frame)
        if phase == "available":
            # A NOTIFICATION, not a request: the holder decides whether to pull, and
            # the ack is the whole handler. IT IS NO LONGER THE WHOLE BEHAVIOUR: the
            # mark below is what makes the holder pull, on its own thread, because
            # asking the peer questions from here would be a request issued by the
            # handler serving that very link (``PeerLink.request`` refuses it).
            # Before this, an ack with nothing behind it was the entire effect of a
            # push, so a replica never refreshed and never held the last assistant
            # message (review round 1, M-4).
            mark_replica_available(server, session_id)
            return {"acknowledged": True, "session_id": session_id}
        raise _refusal("bad_request", f"unknown sync phase {phase!r}")

    return _handle


def _plan(server: "RelayServer", link: Any, frame: dict[str, Any]) -> dict[str, Any]:
    from local_operator.session.placement import record_replica

    session_id = str(frame["session_id"])
    have = frame.get("have") if isinstance(frame.get("have"), dict) else {}
    purpose = str(frame.get("purpose") or "replica")
    try:
        # A MOVE CARRIES THE TRANSCRIPT WHOLE. Its source is retired before the
        # destination can ask for a byte, so there is no half-written row to cut and no
        # next sync to carry the tail: ``_safe_region``'s boundary would DELETE it
        # (review round 2's torn tail: 203 bytes on the source, 150 on the destination,
        # source gone). A replica of a live session keeps the cut, which is what its
        # cursor is built on.
        plan = build_manifest(
            server.root,
            session_id,
            have=have,
            whole_transcript=whole_transcript_for(purpose),
        )
    except SyncRefused as exc:
        raise _refusal(exc.code, exc.message) from exc
    if str(frame.get("purpose") or "replica") == "move":
        # A MOVE'S DESTINATION IS NOT A REPLICA HOLDER (review round 1, MINOR 1).
        # Recording it here made a REFUSED move rewrite the source's ``mesh.json``,
        # put the destination in ``replicas`` so the owner's watcher then pushed a
        # copy of the session to it indefinitely, and told the mesh that a device
        # which is about to OWN the id also holds a replica of it. The move records
        # its destination where that belongs: the handoff journal and the tombstone.
        return plan
    # THIS REQUEST IS HOW THE OWNER LEARNS WHO HOLDS A REPLICA, and it is the only
    # way it can: the holder asks, so the stamp records it, so the watcher has
    # somewhere to push. Best effort — a session with no stamp is not governed by
    # a mesh, and bookkeeping must never refuse a copy.
    try:
        record_replica(server.root, session_id, str(getattr(link, "device_id", "") or ""))
        ensure_watcher(server)
    except Exception:  # noqa: BLE001 — see above
        logger.debug("sync: could not record the replica for %s", session_id, exc_info=True)
    return plan


def _fetch(server: "RelayServer", frame: dict[str, Any]) -> dict[str, Any]:
    try:
        return serve_fetch(
            server.root,
            str(frame["session_id"]),
            plan=str(frame.get("plan_id") or ""),
            name=str(frame.get("name") or ""),
            offset=int(frame.get("offset") or 0),
            limit=int(frame.get("limit") or SYNC_CHUNK_BYTES),
            expect_bytes=_optional_int(frame.get("expect_bytes")),
            expect_mtime_ns=_optional_int(frame.get("expect_mtime_ns")),
        )
    except SyncRefused as exc:
        raise _refusal(exc.code, exc.message) from exc
    except (TypeError, ValueError) as exc:
        raise _refusal("bad_request", f"a fetch needs an offset and a limit ({exc})") from exc


def _verify(server: "RelayServer", frame: dict[str, Any]) -> dict[str, Any]:
    try:
        return serve_verify(
            server.root,
            str(frame["session_id"]),
            plan=str(frame.get("plan_id") or ""),
            name=str(frame.get("name") or ""),
            prefix_bytes=int(frame.get("prefix_bytes") or 0),
            prefix_digest=str(frame.get("prefix_digest") or ""),
            expect_bytes=_optional_int(frame.get("expect_bytes")),
            expect_mtime_ns=_optional_int(frame.get("expect_mtime_ns")),
        )
    except SyncRefused as exc:
        raise _refusal(exc.code, exc.message) from exc
    except (TypeError, ValueError) as exc:
        raise _refusal("bad_request", f"a verify needs a prefix length ({exc})") from exc


def _optional_int(value: Any) -> int | None:
    """``int(value)`` when the frame carried one, else ``None``.

    ``None`` and ``0`` mean different things to the staleness check (0 is a real size), so
    a missing field must not collapse into a zero.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _refusal(code: str, message: str) -> Exception:
    from local_operator.network.types import MeshRefusal

    return MeshRefusal(code, message)


# ---------------------------------------------------------------------------
# The CLI's entry point
# ---------------------------------------------------------------------------


def request_sync(
    session_id: str,
    *,
    owner: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Pull the latest cut of ``session_id`` into this device's replica store.

    ``lop sessions sync``. It runs through this device's RELAY rather than opening
    a peer link here: the relay is the process that holds the links and the only
    one that speaks the mesh (``network/cli.py``'s rule), and a second
    implementation of "dial a peer and ask" in the CLI is exactly the drift that
    rule exists to prevent. Returns the family's ``{"ok", …}`` shape and never
    raises for a refusal.
    """
    from local_operator.network import relay, store

    resolved = Path(root) if root is not None else None
    record = store.find_own_relay(resolved)
    reply = (
        relay.control_request(
            record,
            "session_sync",
            timeout=SYNC_OP_DEADLINE_S * 4,
            session_id=session_id,
            owner=owner or "",
        )
        if record
        else None
    )
    if reply is None:
        return _relay_unavailable(session_id)
    detail = reply.get("detail")
    if reply.get("op") != "ack" or not isinstance(detail, dict):
        return {
            "ok": False,
            "code": str(reply.get("code") or "relay_refused"),
            "message": str(reply.get("message") or "this device's relay refused the sync"),
            "session_id": session_id,
        }
    return detail


def _relay_unavailable(session_id: str = "") -> dict[str, Any]:
    from local_operator.network.cli import (
        CODE_RELAY_UNAVAILABLE,
        _relay_unavailable_message,
    )

    return {
        "ok": False,
        "code": CODE_RELAY_UNAVAILABLE,
        "message": _relay_unavailable_message(),
        "session_id": session_id,
    }


def local_sync_handler(server: "RelayServer") -> Any:
    """The ``session_sync`` local op: pull a session this device does not own.

    The holder's whole job. The owner is resolved from the peer catalogue when the
    caller did not name one, so ``lop sessions sync <id>`` works from the id alone
    — which is the form a person has in front of them.
    """

    def _handle(frame: dict[str, Any]) -> dict[str, Any]:
        from local_operator.network.mobility import (
            LinkTransport,
            Moved,
            resolve_remote_owner,
        )

        session_id = str(frame.get("session_id") or "")
        if not session_id:
            return {"ok": False, "code": "bad_request", "message": "a sync needs a session id"}
        owner = str(frame.get("owner") or "")
        try:
            if not owner:
                owner, _name = resolve_remote_owner(server, session_id)
            link = server._ensure_link(owner)  # noqa: SLF001 — the one dial seam
            if link is None:
                label = server._member_name(owner) or owner  # noqa: SLF001 — the mesh's own name
                return {
                    "ok": False,
                    "code": "unreachable",
                    "message": f"{label} is not answering right now, so nothing was synced",
                    "session_id": session_id,
                }
            transport = LinkTransport(server, link, session_id)
            return sync_from(server.root, session_id, ask=transport.ask, owner_device=owner)
        except Moved as refusal:
            return {
                "ok": False,
                "code": refusal.code or "unreachable",
                "message": refusal.message,
                "session_id": session_id,
            }
        except SyncRefused as refusal:
            return {
                "ok": False,
                "code": refusal.code,
                "message": refusal.message,
                "session_id": session_id,
            }

    return _handle


def install(server: RelayServer) -> None:
    """Register this slice's peer op and local verb on ``server``.

    The watcher is NOT started here: ``install`` runs at relay CONSTRUCTION and the
    test suite builds hundreds of relays that never start, so a thread each would
    be paid for nothing and would stat a store no test asked about. It starts
    lazily from :func:`_plan`, which is the first moment a push has anywhere to go.

    ``net_sync`` is SLOW: one ``fetch`` chunk is bounded, but the op as a whole is
    reached by a peer whose deadline must outlast a slow disk rather than the 10 s
    every inline op gets.
    """
    server.register_ops(
        {"net_sync": make_handler(server)},
        local_handlers={"session_sync": local_sync_handler(server)},
        slow={"net_sync": SYNC_OP_DEADLINE_S},
    )
