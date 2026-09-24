"""Where a session's runtime lives, and the durable stamp that records it.

WHY THIS MODULE EXISTS. A session is owned by exactly one runtime on exactly
one device (``mesh-session-mobility.md`` INV-1, §1.1). Locally that fact is the
transcript lease, a file under ``sessions/<id>/``. Across devices a lease is
meaningless — the marker is a local file and a pid is meaningless on another
host — so the ownership fact needs a carrier that travels and that survives the
runtime exiting: an idle session on a peer must still list as remote.

THAT CARRIER IS THE STAMP, ``sessions/<id>/mesh.json`` (§1.2). It is
**additive and optional**: its absence is the statement "no mesh has ever
governed this session", which is exactly today's behaviour on every install.
That is what makes the zero-peer regression testable (R16, spine §10 topology
0) — nothing here runs, and no file here exists, until a session is created on
or moved to another device.

WHY A SIDECAR AND NOT A FIELD ON THE DISCOVERY RECORD ALONE (§1.2). The record
(``session/runtime/types.py`` ``SessionRecord``) exists only while a runtime
does; an idle session on a peer must still list as remote, so placement cannot
live only there. The record *also* carries placement, additively, because a
live view must not pay a second read — but the stamp is the durable one.

WHY NOT INSIDE ``origin.json``. ``origin.json`` is a migration/discrimination
axis read by ``resume.USER_ORIGINS`` with retired values and scan sentinels
around it. Mesh ownership is a different question with a different lifecycle;
overloading one file makes every writer arbitrate over both.

IMPORT-LIGHT BY CONTRACT. ``local_operator/cli.py`` and the session facade both
reach this module, and the relay reads the stamp for its session rows, so this
module stays stdlib-only (no pydantic, no asyncio, no yaml) exactly as
``session/runtime/types.py`` does.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

#: The sidecar's name inside a session directory. A WIRE-ADJACENT CONSTANT:
#: ``fork.EXCLUDED_SIDECARS`` names it, a peer's catalogue builder reads it, and
#: ``tests/unit/session/test_mesh_stamp.py`` pins the pair — a second spelling
#: anywhere is a stamp one reader cannot see.
MESH_STAMP_NAME = "mesh.json"

#: The stamp schema version, bumped only for a shape change an older reader
#: could misread. A reader that does not know a version treats the stamp as
#: absent (fail closed to "unplaced", never to a guessed owner).
MESH_STAMP_VERSION = 1

#: ``mode``: where the runtime lives. ``pool`` is reserved and never produced in
#: this pass (``mesh-compute-pool.md``). ``policy``: what governs placement
#: decisions for the session (spine A8).
PlacementMode = Literal["local", "peer", "pool"]
PlacementPolicy = Literal["pinned", "prefer-remote", "cost-capped"]

#: How the session got to the device holding it now — the row's "copy of X from
#: Y" provenance, and the reason ``source_session_id`` exists at all.
OriginKind = Literal["user", "moved", "fork"]


@dataclass(frozen=True)
class SessionPlacement:
    """Placement as both the stamp and the catalogue carry it (§5.2).

    ONE dataclass serialised in both places rather than two shapes that agree
    today: the catalogue row is the viewer-facing projection of this, and a
    second field list is how the two would come to disagree about which device
    owns a conversation.
    """

    mode: PlacementMode = "local"
    network_id: str = ""
    home_device: str = ""
    policy: PlacementPolicy = "pinned"
    stamp_revision: int = 0

    @property
    def is_local(self) -> bool:
        return self.mode == "local"

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "network_id": self.network_id,
            "home_device": self.home_device,
            "policy": self.policy,
            "stamp_revision": self.stamp_revision,
        }

    @staticmethod
    def from_json(data: Any) -> SessionPlacement:
        """Parse leniently, defaulting every field.

        Absent is not a claim: a row or stamp that omits a field gets the
        conservative default (``local``/``pinned``) rather than an error, which
        is the same discipline ``SessionRecord.from_json`` applies to keys an
        older build never wrote.
        """
        if not isinstance(data, dict):
            return SessionPlacement()
        mode = data.get("mode")
        policy = data.get("policy")
        return SessionPlacement(
            mode=mode if mode in ("local", "peer", "pool") else "local",
            network_id=str(data.get("network_id") or ""),
            home_device=str(data.get("home_device") or ""),
            policy=policy if policy in ("pinned", "prefer-remote", "cost-capped") else "pinned",
            stamp_revision=_as_int(data.get("stamp_revision")),
        )


@dataclass
class MeshStamp:
    """``sessions/<id>/mesh.json``: who owns this session and how it got here."""

    session_id: str
    network_id: str = ""
    #: The owning device id (a device-key fingerprint, spine A3). This is the
    #: ONLY field that decides INV-1's routing question (§1.2).
    home_device: str = ""
    placement: SessionPlacement = field(default_factory=SessionPlacement)
    origin: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    version: int = MESH_STAMP_VERSION
    #: Devices that hold a REPLICA of this session (§7.5, R22). Written by the
    #: OWNER and only the owner, because it is what the owner's sync watcher
    #: iterates: "who must I tell when a new cut exists". A replica is never a
    #: second owner — it lives outside ``sessions/`` precisely so no scanner can
    #: mistake it for one (INV-1) — so this is a notification list, not a
    #: placement claim, and it is the one field here that is about OTHER devices.
    replicas: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "network_id": self.network_id,
            "home_device": self.home_device,
            "placement": self.placement.to_json(),
            "origin": dict(self.origin),
            "created_at": self.created_at,
            "replicas": list(self.replicas),
        }

    @staticmethod
    def from_json(data: Any) -> MeshStamp | None:
        """Parse a stamp, or ``None`` when this build must not trust it.

        A version this build does not know is treated as NO stamp: the
        resolver's step 1 then answers from the local store alone (§2.1),
        which is the pre-mesh behaviour, and nothing is routed on a shape that
        was not understood.
        """
        if not isinstance(data, dict):
            return None
        if _as_int(data.get("version")) not in (0, MESH_STAMP_VERSION):
            return None
        placement = SessionPlacement.from_json(data.get("placement"))
        home = str(data.get("home_device") or "")
        if home and not placement.home_device:
            # The two carriers must not disagree about the owner, and the
            # top-level field is the authoritative one (§1.2): a stamp whose
            # nested placement lost its device still names its owner.
            placement = SessionPlacement(
                mode=placement.mode if placement.mode != "local" else "peer",
                network_id=placement.network_id or str(data.get("network_id") or ""),
                home_device=home,
                policy=placement.policy,
                stamp_revision=placement.stamp_revision,
            )
        origin = data.get("origin")
        # A stamp written by a build that predates replicas has no key at all, and
        # one written by a build that has it may hold a corrupt list. Both read as
        # "nobody holds a replica" — the conservative direction, because a wrong
        # entry here makes the owner push cuts to a device that never asked, while
        # a missing one costs a holder its push and nothing more (it still pulls
        # on demand and on attach).
        raw_replicas = data.get("replicas")
        replicas = (
            [str(item) for item in raw_replicas if isinstance(item, str)]
            if isinstance(raw_replicas, list)
            else []
        )
        return MeshStamp(
            session_id=str(data.get("session_id") or ""),
            network_id=str(data.get("network_id") or ""),
            home_device=home,
            placement=placement,
            origin=dict(origin) if isinstance(origin, dict) else {},
            created_at=float(data.get("created_at") or 0.0),
            version=_as_int(data.get("version")) or MESH_STAMP_VERSION,
            replicas=replicas,
        )


def stamp_path(config_dir: Path, session_id: str) -> Path:
    return Path(config_dir) / "sessions" / session_id / MESH_STAMP_NAME


def read_stamp(config_dir: Path, session_id: str) -> MeshStamp | None:
    """The stamp for one session, or ``None`` when there is none to trust.

    A malformed or unreadable stamp reads as absent rather than raising: every
    caller of this is a listing or a resolver, and a corrupt file must not be
    able to make a session unresolvable — it must merely make it *unplaced*.
    """
    if not session_id:
        return None
    try:
        raw = stamp_path(config_dir, session_id).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return MeshStamp.from_json(data)


def write_stamp(config_dir: Path, stamp: MeshStamp) -> Path:
    """Write the stamp atomically, 0600, and return its path.

    Atomic because a reader is another process on this machine (the relay's
    catalogue scan, a concurrent `lop sessions`), and a torn stamp would be a
    session that is neither local nor remote. 0600 because the stamp is
    session metadata: the account is the boundary, exactly as it is for the
    records beside it.
    """
    return write_stamp_into(Path(config_dir) / "sessions" / stamp.session_id, stamp)


def write_stamp_into(directory: Path, stamp: MeshStamp) -> Path:
    """Write a stamp into an EXPLICIT directory, as ``directory/mesh.json``.

    The move needs this, and needs it for a correctness reason rather than a
    convenience one: the destination writes the stamp INTO ITS STAGING DIRECTORY so
    the session, its ownership and its lineage become visible in the single
    ``os.replace`` that promotes it (§6.3 step 16). Writing it through
    :func:`write_stamp` would create ``sessions/<id>/`` as a side effect of
    ``mkdir(parents=True)`` — a session directory that exists before the copy is
    verified, which is the half-promoted state the atomic rename exists to make
    impossible, and which then makes the promote itself refuse because the target
    is in the way.

    ONE WRITER for the format either way: this is the only place the document is
    serialised, so a staging stamp and a normal one cannot drift.
    """
    path = Path(directory) / MESH_STAMP_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(stamp.to_json(), indent=2, sort_keys=True).encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(str(tmp), str(path))
    except OSError:
        # A half-written stamp is deliberately NOT renamed into place: the
        # reader treats a missing stamp as "unplaced", while a truncated one
        # would be a claim nobody made.
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise
    return path


def remove_stamp(config_dir: Path, session_id: str) -> None:
    """Drop a stamp. Best effort: a missing file is the desired state."""
    try:
        stamp_path(config_dir, session_id).unlink()
    except OSError:
        return


def local_placement() -> SessionPlacement:
    """The placement every pre-mesh session has, and the default for a local one."""
    return SessionPlacement(mode="local", policy="pinned")


def record_replica(config_dir: Path, session_id: str, device_id: str) -> bool:
    """Note that ``device_id`` holds a replica of ``session_id``. Owner-side.

    Returns whether anything changed, so the caller can skip a write on the
    steady-state path (a holder pulls repeatedly and the same device must not
    rewrite the stamp every time). A session with no stamp is left alone: a
    session no mesh has ever governed has no owner-side watcher to feed.
    """
    if not device_id:
        return False
    stamp = read_stamp(config_dir, session_id)
    if stamp is None or device_id in stamp.replicas:
        return False
    stamp.replicas = [*stamp.replicas, device_id]
    write_stamp(config_dir, stamp)
    return True


def drop_replica(config_dir: Path, session_id: str, device_id: str) -> bool:
    """Forget a replica holder. Best effort, and idempotent by construction."""
    stamp = read_stamp(config_dir, session_id)
    if stamp is None or device_id not in stamp.replicas:
        return False
    stamp.replicas = [item for item in stamp.replicas if item != device_id]
    write_stamp(config_dir, stamp)
    return True


# ---------------------------------------------------------------------------
# The handoff journal (``network/pending-move.json``)
# ---------------------------------------------------------------------------
#
# WHY IT LIVES HERE, and not in ``network/mobility.py`` where it is written.
# Two modules must agree about one file: the mover writes it, and
# ``session/runtime/launch.py`` READS it on every engage — that read is the
# guard that makes INV-1 impossible rather than unlikely (§6.6), so it must be
# stdlib-only (launch.py is on the boot path, and importing the network package
# there would be a cycle: ``network.relay`` imports ``launch``). Keeping the
# schema, the atomic writer and the guard query in ONE stdlib-only module is
# what stops the reader and the writer from drifting — a second spelling of a
# phase name here is a session that gets two runtimes.
#
# THE FAIL-CLOSED RULE. A journal that exists but cannot be parsed refuses
# engages. Disk corruption is the only way to get one (every write below is a
# same-directory temp + ``os.replace`` + ``fsync``), and the two candidate
# answers are not symmetric: refusing costs an operator one sentence and a file
# deletion, while proceeding can spawn a second runtime for a session that is
# mid-handoff, which is the one outcome this whole design exists to prevent.

#: The directory under the config root, matching the rest of the mesh's state
#: (``network/``), so one ``rm -rf`` of a device's mesh state takes this with it.
HANDOFF_DIRNAME = "network"

#: The journal's file name. WIRE-ADJACENT: ``network/mobility.py`` writes it and
#: the launch guard reads it, both through this module's helpers.
HANDOFF_JOURNAL_NAME = "pending-move.json"

#: The two phases a handoff occupies while it is in flight (§6.3): ``prepared``
#: before the copy verifies, ``handing-off`` after. Both mean "this id has a
#: handoff in progress", which is the whole of the guard's question; the
#: distinction is what the RECONCILE branches on, not the guard.
HANDOFF_PHASE_PREPARED = "prepared"
HANDOFF_PHASE_HANDING_OFF = "handing-off"
HANDOFF_PHASES: frozenset[str] = frozenset({HANDOFF_PHASE_PREPARED, HANDOFF_PHASE_HANDING_OFF})


class HandoffJournalUnreadable(Exception):
    """The journal exists and cannot be read. Refuse; see the rule above."""

    def __init__(self, path: Path, cause: str) -> None:
        super().__init__(f"{path}: {cause}")
        self.path = path
        self.cause = cause


def handoff_journal_path(config_dir: Path) -> Path:
    return Path(config_dir) / HANDOFF_DIRNAME / HANDOFF_JOURNAL_NAME


def read_handoff_journal(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Every in-flight handoff, keyed by session id. ``{}`` when there is none.

    Raises :class:`HandoffJournalUnreadable` rather than returning ``{}`` for a
    file it could not read: an absent file and an unreadable one are different
    answers, and only the first one means "no handoff is in flight".
    """
    path = handoff_journal_path(config_dir)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise HandoffJournalUnreadable(path, str(exc)) from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise HandoffJournalUnreadable(path, "not valid JSON") from exc
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, dict):
        raise HandoffJournalUnreadable(path, "not a journal document")
    return {str(key): dict(value) for key, value in entries.items() if isinstance(value, dict)}


def write_handoff_entry(config_dir: Path, session_id: str, entry: dict[str, Any]) -> None:
    """Add or replace one session's journal entry, atomically.

    Same-directory temp + ``os.replace`` + ``fsync``, which is the discipline
    ``_stage_and_replace`` already uses for the desktop marker: a torn journal is
    a session whose guard cannot be evaluated, and the guard's whole job is to be
    answerable after a crash.
    """
    path = handoff_journal_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = read_handoff_journal(config_dir)
    entries[session_id] = dict(entry)
    _write_journal(path, entries)


def clear_handoff_entry(config_dir: Path, session_id: str) -> bool:
    """Drop one entry, leaving the others alone. Returns whether it was there."""
    entries = read_handoff_journal(config_dir)
    if session_id not in entries:
        return False
    del entries[session_id]
    path = handoff_journal_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_journal(path, entries)
    return True


def _write_journal(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    payload = json.dumps({"version": 1, "entries": entries}, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    except OSError:
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise


def handoff_in_flight(config_dir: Path, session_id: str) -> dict[str, Any] | None:
    """This id's in-flight handoff entry, or ``None``. THE LAUNCH GUARD'S READ.

    Raises :class:`HandoffJournalUnreadable`; callers on a user-facing path turn
    that into a refusal sentence rather than an exception (see
    :func:`handoff_guard_refusal`).
    """
    if not session_id:
        return None
    entry = read_handoff_journal(config_dir).get(session_id)
    if entry is None:
        return None
    if str(entry.get("phase") or "") not in HANDOFF_PHASES:
        return None
    return entry


def handoff_guard_refusal(config_dir: Path, session_id: str) -> str:
    """The sentence that stops an engage during a handoff, or ``""`` to allow it.

    One function so the launch guard and any front end that wants to explain the
    empty editor in advance cannot word it differently. Empty string, never
    ``None``: the caller's test is truthiness, and a ``None`` that reads as "no
    refusal" is the shape that lets a guard be wired backwards.
    """
    try:
        entry = handoff_in_flight(config_dir, session_id)
    except HandoffJournalUnreadable as exc:
        return (
            "Local Operator could not tell whether this conversation is being moved, so "
            f"it was not started. Its handoff journal ({exc.path.name}) is unreadable; "
            "delete that file if no move is in progress, then try again."
        )
    if entry is None:
        return ""
    device = str(entry.get("to_name") or entry.get("to_device") or "another device")
    if str(entry.get("role") or "source") == "destination":
        return (
            f"This conversation is being received from {device}; it will be available "
            "when the move finishes."
        )
    return f"This conversation is being handed to {device}."


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
