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

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "network_id": self.network_id,
            "home_device": self.home_device,
            "placement": self.placement.to_json(),
            "origin": dict(self.origin),
            "created_at": self.created_at,
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
        return MeshStamp(
            session_id=str(data.get("session_id") or ""),
            network_id=str(data.get("network_id") or ""),
            home_device=home,
            placement=placement,
            origin=dict(origin) if isinstance(origin, dict) else {},
            created_at=float(data.get("created_at") or 0.0),
            version=_as_int(data.get("version")) or MESH_STAMP_VERSION,
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
    path = stamp_path(config_dir, stamp.session_id)
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


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
