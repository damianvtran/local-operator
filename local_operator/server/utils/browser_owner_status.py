"""Read-only owner evidence for the desktop browser host, not a close permit."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from local_operator.browser_bridge.resources import RESOURCE_NAME
from local_operator.session.retention import SESSIONS_DIRNAME
from local_operator.session_lease import (
    LEASE_NAME,
    MIRROR_NAME,
    _read_claim,
    lease_holder,
)

# The batch size is the desktop protocol bound; this path does not walk a roster.
OwnerID = Annotated[str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9_-]{1,64}$")]
# The current writers use token_hex(16) for leases and token_urlsafe(24) for
# unleased browser owners. Neither token is a public session identifier.
_GENERATION = re.compile(r"[A-Za-z0-9_-]{32}")
# Match the existing BrowserResource.inspect_ownership sidecar read bound.
_MAX_WITNESS_BYTES = 65536


class BrowserOwnerQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_ids: list[OwnerID] = Field(min_length=1, max_length=64)

    @field_validator("session_ids")
    @classmethod
    def distinct_ids(cls, ids: list[str]) -> list[str]:
        if len(set(ids)) != len(ids):
            raise ValueError("session_ids must be unique")
        return ids


class BrowserOwnerRow(BaseModel):
    session_id: str
    # Keep the protocol's terminal verdict reserved for execution-fenced proof.
    verdict: Literal["live", "ended", "unknown"] = "unknown"
    # Discovery/roster turn flags don't carry the browser generation; a fresh
    # lease proves a writer, not its turn state. In particular, never guess idle.
    turn: Literal["running", "delegating", "awaiting_user", "idle", "unknown"] = "unknown"
    retained: bool | None = None
    provenance: Literal["execution_lease", "session_claim", "browser_terminal", "none"] = "none"
    generation: str | None = None


class BrowserOwnerStatus(BaseModel):
    owners: list[BrowserOwnerRow]


def _generation(value: object) -> str | None:
    return value if isinstance(value, str) and _GENERATION.fullmatch(value) else None


def _stamp(path: Path, *, directory: bool = False) -> tuple[int, ...] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError("owner evidence is not a regular store entry")
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)


def _read_owner(sessions: Path, session_id: str) -> BrowserOwnerRow:
    unknown = BrowserOwnerRow(session_id=session_id)
    directory = sessions / session_id
    lease, mirror, resource = (
        directory / name for name in (LEASE_NAME, MIRROR_NAME, RESOURCE_NAME)
    )

    def snapshot() -> tuple[tuple[int, ...] | None, ...]:
        root = _stamp(sessions, directory=True)
        return (
            root[:2] if root is not None else None,
            _stamp(directory, directory=True),
            _stamp(lease),
            _stamp(mirror),
            _stamp(resource),
        )

    try:
        before = snapshot()
        if before[1] is None:
            return unknown
        if any(stamp and stamp[-1] > _MAX_WITNESS_BYTES for stamp in before[2:]):
            return unknown
        has_lease, has_mirror, has_resource = (stamp is not None for stamp in before[2:])
        receipt = {}
        if has_resource:
            with resource.open("rb") as stream:
                raw = stream.read(_MAX_WITNESS_BYTES + 1)
            if len(raw) > _MAX_WITNESS_BYTES:
                return unknown
            receipt = json.loads(raw)
        if not isinstance(receipt, dict) or (
            has_resource and receipt.get("session_id") != session_id
        ):
            return unknown
        scope = _generation(receipt.get("generation"))
        retention = receipt.get("retention", "")
        if has_resource and (
            scope is None
            or not all(
                isinstance(receipt.get(field, ""), str)
                for field in (
                    "state",
                    "retention",
                    "terminal",
                    "handed_to",
                    "ownership",
                    "host",
                    "surface_id",
                    "unresolved_surface_id",
                )
            )
            or not isinstance(receipt.get("resumed_scope", False), bool)
        ):
            return unknown
        retained = bool(retention or receipt.get("handed_to") or receipt.get("state") == "retained")

        owner_generation = scope
        row = unknown
        if has_lease or has_mirror:
            # A present but unreadable/dead claim is not proof that a run finished.
            # A child's in-process claim is only a positive liveness signal. Its
            # parent may stay alive long after the child's marker is released.
            expected_pid = None
            if has_lease:
                claim = _read_claim(lease)
                owner_generation = _generation(claim.generation)
                expected_pid = claim.pid
                if owner_generation is None or expected_pid is None or expected_pid <= 0:
                    return unknown
            pid, verdict = lease_holder(directory)
            if (
                verdict == "live"
                and pid is not None
                and (expected_pid is None or pid == expected_pid)
            ):
                if not has_lease and receipt.get("terminal"):
                    # Unleased children reuse a generation on resume. The current
                    # claim wins, but it cannot bind an earlier run's terminal data.
                    owner_generation = None
                row = BrowserOwnerRow(
                    session_id=session_id,
                    verdict="live",
                    provenance="execution_lease" if has_lease else "session_claim",
                    retained=retained if scope and scope == owner_generation else None,
                )
        elif retained:
            row = BrowserOwnerRow(session_id=session_id, retained=True)

        # A browser close ACK settles that scope, not the latest execution.
        # A child can resume on the old generation after a failed marker write,
        # before browser initialization touches the receipt. Once a lease is
        # removed, a markerless receipt cannot prove it came from a leased run
        # either. Leave these receipts unknown, including legacy orphaned owners.

        # Re-sample the witnesses as a set: an atomic replacement or a new claim
        # during this read invalidates the observation. This remains a sample;
        # the host must not carry it across a resume or treat it as authorization.
        if snapshot() != before:
            return unknown
        if row.verdict != "unknown" and owner_generation:
            row.generation = hashlib.sha256(owner_generation.encode("ascii")).hexdigest()
        return row
    except (OSError, ValueError, TypeError, AttributeError, RecursionError):
        # Corrupt/partially written evidence is a per-owner unknown, not a batch
        # failure (and no exception text from the local store goes over the wire).
        return unknown


def read_browser_owner_status(root: Path, session_ids: list[str]) -> BrowserOwnerStatus:
    sessions = root / SESSIONS_DIRNAME
    return BrowserOwnerStatus(
        owners=[_read_owner(sessions, session_id) for session_id in session_ids]
    )
