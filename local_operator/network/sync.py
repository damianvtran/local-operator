"""Session sync (R22): replicas of sessions this device does not own.

TWO SIDES, ONE PRIMITIVE. The owner answers ``net_sync {phase: plan|fetch}`` and
runs a watcher that tells holders when a new cut exists. Every other device is a
HOLDER: it pulls a plan, fetches the miss, and keeps the result in a REPLICA at
``<config>/network/replicas/<id>/`` — outside ``sessions/``, so no scanner,
picker, catalogue, retirement pass or resume can see it. That placement is the
INV-1 argument, not tidiness: a replica under ``sessions/`` would be a second
directory for an id the owner still holds, which is the state the mobility design
exists to make impossible.

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
import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
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
)

#: Files the copy must NEVER carry (design §7.2). The copy is an allow-list and
#: needs no deny-list to be correct — this is documentation with a test attached,
#: so the next person who reaches for ``copytree`` sees why each one is absent.
#: ``.session.pid`` is the liveness marker: copied, the destination would report
#: the SOURCE's pid as the owner. ``mesh.json`` is written by the destination
#: (``home_device`` must name the new owner). ``sync.json`` describes a replica
#: and means nothing in a session. ``subagent-roster.v1.json`` lists jobs owned by
#: the source's process. The scan sentinels are per-directory "already swept"
#: claims that would suppress a backfill in a directory no sweep has seen.
NEVER_COPIED: tuple[str, ...] = (
    ".session.pid",
    "mesh.json",
    REPLICA_CURSOR_NAME,
    "subagent-roster.v1.json",
    "origin-scan.json",
    "title-scan.json",
)

#: A transcript row's reference to a content-addressed attachment. The store names
#: files by the first 32 hex chars of the decoded bytes' sha256
#: (``session/attachments.py``), so that is what a reference scan looks for.
_ATTACHMENT_REF = re.compile(rb'"attachment"\s*:\s*"([0-9a-f]{32})"')


class SyncRefused(Exception):
    """A sync step was refused. Carries the family's ``code`` and its sentence."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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


def copy_set(root: Path, session_id: str) -> list[str]:
    """The copy set's names that exist for this session, in a fixed order."""
    directory = Path(root) / "sessions" / session_id
    return [name for name in COPY_SET_NAMES if (directory / name).is_file()]


def _safe_region(payload: bytes) -> bytes:
    """``payload`` up to and including its last complete line.

    THE TORN TAIL. ``--keep`` copies from a runtime that is still writing, so the
    last line may be half-written. A reader tolerates that (``fork_session``
    passes malformed lines through), but a CURSOR cannot: the next sync has to know
    exactly which bytes it already holds, and "everything up to the last newline"
    is the only boundary both sides can agree on without parsing. The tail row is
    not lost — the next sync carries it, whole.
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


def referenced_attachments(root: Path, session_id: str) -> list[str]:
    """Attachment digests this session's transcript references, deduplicated."""
    try:
        payload = (Path(root) / "sessions" / session_id / TRANSCRIPT_NAME).read_bytes()
    except OSError:
        return []
    return sorted({match.decode("ascii") for match in _ATTACHMENT_REF.findall(payload)})


def build_manifest(
    root: Path,
    session_id: str,
    *,
    have: dict[str, Any] | None = None,
    attachments_dir: Path | None = None,
) -> dict[str, Any]:
    """The plan a holder pulls: what to send, with the hashes that verify it.

    ``have`` is the holder's own report (``{"cursor": …, "files": …, "attachments": …}``)
    and is used ONLY as a filter. The owner still digests everything it sends, so
    a holder that misreports what it holds gets a correct copy of the wrong size
    (a re-send), never a corrupt one.
    """
    have = have or {}
    session_dir = Path(root) / "sessions" / session_id
    transcript = session_dir / TRANSCRIPT_NAME
    try:
        raw = transcript.read_bytes()
    except OSError as exc:
        raise SyncRefused(
            "no_session", f"{session_id} has no transcript to sync on this device ({exc})"
        ) from exc
    safe = _safe_region(raw)
    total = len(safe)
    digest = sha256_bytes(safe)

    have = have or {}
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
    for name in COPY_SET_NAMES:
        if name == TRANSCRIPT_NAME:
            continue
        path = session_dir / name
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        item_digest = sha256_bytes(payload)
        if str(held.get(name) or "") == item_digest:
            # The same bytes are already there. The design's mtime hint is
            # deliberately not consulted: two clocks disagree and a digest is a
            # fact.
            continue
        items.append(
            {
                "name": name,
                "bytes": len(payload),
                "mtime_ns": int(path.stat().st_mtime_ns),
                "digest": item_digest,
            }
        )

    blobs: list[dict[str, Any]] = []
    store_dir = Path(attachments_dir) if attachments_dir is not None else Path(root) / "attachments"
    held_blobs_raw = have.get("attachments")
    held_blobs: dict[str, Any] = held_blobs_raw if isinstance(held_blobs_raw, dict) else {}
    for ref in referenced_attachments(root, session_id):
        blob = store_dir / f"{ref}.bin"
        try:
            payload = blob.read_bytes()
        except OSError:
            # A transcript can reference a blob this install no longer has (the
            # store is never pruned, but an install can be copied without it).
            # Skipping is honest: the copy is complete for everything the owner
            # holds, and the missing image renders here as broken too.
            continue
        blob_digest = sha256_bytes(payload)
        if str(held_blobs.get(ref) or "") == blob_digest:
            continue
        blobs.append(
            {
                "name": f"{ATTACHMENT_PREFIX}{ref}.bin",
                "ref": ref,
                "bytes": len(payload),
                "mtime_ns": int(blob.stat().st_mtime_ns),
                "digest": blob_digest,
            }
        )

    return {
        "session_id": session_id,
        "plan_id": plan_id(root, session_id),
        "transcript": {
            "name": TRANSCRIPT_NAME,
            "mode": mode,
            "prefix_bytes": prefix_bytes if mode == "append" else 0,
            "total_bytes": total,
            "region_bytes": total - prefix_bytes if mode == "append" else total,
            "source_bytes": len(raw),
            "digest": digest,
            "frontier": _frontier(safe),
        },
        "items": items,
        "attachments": blobs,
        "copy_set": copy_set(root, session_id),
    }


def _item_path(root: Path, session_id: str, name: str, attachments_dir: Path | None) -> Path | None:
    """Where ``name`` lives on the owner, or ``None`` for a name outside the set."""
    if name.startswith(ATTACHMENT_PREFIX):
        store_dir = (
            Path(attachments_dir) if attachments_dir is not None else Path(root) / "attachments"
        )
        return store_dir / name[len(ATTACHMENT_PREFIX) :]
    if name not in COPY_SET_NAMES or name in NEVER_COPIED:
        return None
    return Path(root) / "sessions" / session_id / name


def _served_bytes(path: Path, name: str) -> bytes:
    payload = path.read_bytes()
    return _safe_region(payload) if name == TRANSCRIPT_NAME else payload


def plan_id(root: Path, session_id: str, *, attachments_dir: Path | None = None) -> str:
    """A digest of the SOURCE's state, independent of what the holder has.

    ``fetch`` and ``verify`` re-derive it and refuse work whose plan no longer
    describes the source. Without it, a fetch issued against a stale plan would
    append bytes from a rewritten file onto a holder's cursor — a transcript that
    is internally inconsistent and that nothing downstream could detect.
    """
    digest = hashlib.sha256()
    digest.update(session_id.encode("utf-8"))
    for name in COPY_SET_NAMES:
        path = _item_path(root, session_id, name, attachments_dir)
        if path is None:
            continue
        try:
            payload = _served_bytes(path, name)
        except OSError:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(payload)
    for ref in referenced_attachments(root, session_id):
        path = _item_path(root, session_id, f"{ATTACHMENT_PREFIX}{ref}.bin", attachments_dir)
        if path is None:
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        digest.update(ref.encode("utf-8"))
        digest.update(payload)
    return "sha256:" + digest.hexdigest()


def serve_fetch(
    root: Path,
    session_id: str,
    *,
    plan: str,
    name: str,
    offset: int,
    limit: int = SYNC_CHUNK_BYTES,
    attachments_dir: Path | None = None,
) -> dict[str, Any]:
    """One chunk of one file: the owner's ``net_sync {phase:"fetch"}`` answer."""
    _require_current_plan(root, session_id, plan, attachments_dir=attachments_dir)
    path = _item_path(root, session_id, name, attachments_dir)
    if path is None:
        raise SyncRefused("unknown_item", f"{name} is not part of a session copy")
    try:
        payload = _served_bytes(path, name)
    except OSError as exc:
        raise SyncRefused("missing", f"{name} is not readable on this device") from exc
    start = max(0, int(offset))
    span = payload[start : start + max(0, int(limit))]
    return {
        "name": name,
        "offset": start,
        "bytes": len(span),
        "data": base64.b64encode(span).decode("ascii"),
        "eof": start + len(span) >= len(payload),
        "chunk_digest": sha256_bytes(span),
        "file_digest": sha256_bytes(payload),
        "file_bytes": len(payload),
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
) -> dict[str, Any]:
    """Does the owner's first ``prefix_bytes`` of ``name`` hash to ``prefix_digest``?

    THE RESUMPTION CHECK. A destination interrupted mid-copy holds bytes it never
    saw verified as a whole file. Rather than trusting them (a splice) or throwing
    them away (a re-send of up to 100 MB), it asks this: a matching digest means
    those bytes are the source's own prefix, so the copy can continue from there
    with the same guarantee a fresh copy has.
    """
    _require_current_plan(root, session_id, plan, attachments_dir=attachments_dir)
    path = _item_path(root, session_id, name, attachments_dir)
    if path is None:
        raise SyncRefused("unknown_item", f"{name} is not part of a session copy")
    try:
        payload = _served_bytes(path, name)
    except OSError as exc:
        raise SyncRefused("missing", f"{name} is not readable on this device") from exc
    if prefix_bytes > len(payload):
        return {"matches": False, "reason": "shorter"}
    return {"matches": sha256_bytes(payload[:prefix_bytes]) == prefix_digest}


def _require_current_plan(
    root: Path, session_id: str, plan: str, *, attachments_dir: Path | None
) -> None:
    current = plan_id(root, session_id, attachments_dir=attachments_dir)
    if not current or current != plan:
        raise SyncRefused(
            "stale_plan",
            "the conversation changed while it was being copied, so this part was not "
            "sent; asking again picks up the new cut",
        )


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


def pull(
    root: Path,
    session_id: str,
    *,
    ask: Callable[[dict[str, Any]], dict[str, Any]],
    dest_dir: Path,
    attachments_root: Path,
    have: dict[str, Any] | None = None,
    stop: Callable[[], bool] | None = None,
) -> PullResult:
    """Bring ``dest_dir`` up to the source's cut, verifying every byte.

    ``ask`` is the transport: it takes one ``net_sync`` frame and returns its
    ``detail``, raising :class:`SyncRefused` for a refusal. ONE callable rather
    than a link means this same code runs over a peer link in production, over a
    control socket in a test, and against a plain in-process callable against a
    second config root — which is what makes "every byte is verified" a claim that
    can be tested without two machines.
    """
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
    )
    written += outcome.written
    resumed += outcome.resumed

    items = 0
    blobs = 0
    for item in list(plan.get("items") or []) + list(plan.get("attachments") or []):
        name = str(item.get("name") or "")
        if not name:
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
            whole_digest=str(item.get("digest") or ""),
            final_bytes=size,
            stop=stop,
        )
        written += outcome.written
        resumed += outcome.resumed
        if name.startswith(ATTACHMENT_PREFIX):
            blobs += 1
        else:
            items += 1

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
    )


def transcript_exists(directory: Path) -> bool:
    return (Path(directory) / TRANSCRIPT_NAME).is_file()


def _destination_for(dest_dir: Path, attachments_root: Path, name: str) -> Path | None:
    if name.startswith(ATTACHMENT_PREFIX):
        # THE PREFIX IS KEPT, and ``attachments_root`` is the STORE ROOT rather than
        # the blob directory: a move points it at the config root, so a blob lands
        # in the install's shared content-addressed store (``<config>/attachments/``,
        # where the transcript's references resolve), while a replica points it at
        # itself, so the copy stays self-contained for a recovery or a delete.
        return Path(attachments_root) / name
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
    with dest_path.open("ab" if keep else "wb") as handle:
        if not keep:
            handle.truncate(0)
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
                }
            )
            payload = base64.b64decode(str(chunk.get("data") or ""))
            if sha256_bytes(payload) != str(chunk.get("chunk_digest") or ""):
                raise SyncRefused(
                    "digest_mismatch", f"the copy of {name} did not arrive intact, so it was lost"
                )
            if not payload:
                raise SyncRefused(
                    "digest_mismatch",
                    f"the copy of {name} stopped {source_end - offset} bytes short of the source",
                )
            handle.write(payload)
            offset += len(payload)
            written += len(payload)
    produced = dest_path.stat().st_size
    if produced != final_bytes:
        raise SyncRefused(
            "digest_mismatch",
            f"the copy of {name} is {produced} bytes where the source has {final_bytes}, so it "
            "was not kept",
        )
    if whole_digest and sha256_bytes(dest_path.read_bytes()) != whole_digest:
        raise SyncRefused(
            "digest_mismatch", f"the copy of {name} did not verify, so it was not kept"
        )
    return _FetchOutcome(written=written, resumed=resumed)


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
    """The ``have`` document for a destination that already holds some of the set."""
    files: dict[str, str] = {}
    for name in COPY_SET_NAMES:
        if name == TRANSCRIPT_NAME:
            continue
        path = Path(dest_dir) / name
        try:
            files[name] = sha256_bytes(path.read_bytes())
        except OSError:
            continue
    blobs: dict[str, str] = {}
    try:
        for candidate in sorted((Path(attachments_root) / ATTACHMENT_PREFIX).glob("*.bin")):
            blobs[candidate.stem] = sha256_bytes(candidate.read_bytes())
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
) -> dict[str, Any]:
    """Pull ``session_id`` from its owner — into the replica store, or a staging dir.

    ONE implementation for both, because a move's copy and a replica's copy must
    not be able to disagree about what they copy or what they verify. ``into`` and
    ``attachments_root`` are the whole difference; a move points them at its
    staging area and the install's shared content-addressed store (a re-downloaded
    blob would be wasted bytes), a replica points them at itself (so one directory
    can be recovered or deleted as a unit).
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
    new_id = new_id or new_session_id()
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
        # The replica's blobs live inside it; the recovered session's transcripts
        # reference them by digest, so they have to reach the SHARED store or the
        # recovered conversation renders broken images.
        blob_source = source / "attachments"
        if blob_source.is_dir():
            store = Path(root) / "attachments"
            store.mkdir(parents=True, exist_ok=True)
            for blob in sorted(blob_source.glob("*.bin")):
                shutil.copyfile(blob, store / blob.name)
                sidecar = blob.with_suffix(".json")
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
            # A NOTIFICATION, not a request: the holder decides whether to pull,
            # and the ack is the whole handler.
            return {"acknowledged": True, "session_id": session_id}
        raise _refusal("bad_request", f"unknown sync phase {phase!r}")

    return _handle


def _plan(server: "RelayServer", link: Any, frame: dict[str, Any]) -> dict[str, Any]:
    from local_operator.session.placement import record_replica

    session_id = str(frame["session_id"])
    have = frame.get("have") if isinstance(frame.get("have"), dict) else {}
    try:
        plan = build_manifest(server.root, session_id, have=have)
    except SyncRefused as exc:
        raise _refusal(exc.code, exc.message) from exc
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
        )
    except SyncRefused as exc:
        raise _refusal(exc.code, exc.message) from exc
    except (TypeError, ValueError) as exc:
        raise _refusal("bad_request", f"a verify needs a prefix length ({exc})") from exc


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
