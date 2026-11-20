"""Explicit local workstream notification policy, separate from agent identity.

An attached specialist or proactive profile is not permission to interrupt the
operator. Only the supported operator-authorized grant operation writes this
record. It lives in a separate additive sidecar, never rewriting spawn
provenance. This is same-account routing policy, NOT a
cryptographic boundary against arbitrary code running as the operator.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from local_operator.resume import ORIGIN_AGENT_WORKSTREAM, ORIGIN_NAME

GRANT_NAME = "notification-authority.json"


def _origin(directory: Path) -> dict[str, Any]:
    data = json.loads((directory / ORIGIN_NAME).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid_origin")
    return data


def read_decision(directory: Path) -> dict[str, Any] | None:
    """Inspectable policy metadata; absence or malformed JSON grants nothing."""
    try:
        decision = json.loads((directory / GRANT_NAME).read_text(encoding="utf-8"))
        return decision if isinstance(decision, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def granted(directory: Path) -> bool:
    """Read fresh on every invocation; malformed or copied grants fail closed.

    Binding to the session id prevents a copied/forked origin sidecar from
    promoting another conversation. Remote-placement exclusions are enforced by
    Session independently; this grant can relax only local workstream policy.
    """
    try:
        if (directory / "mesh.json").exists():
            return False
        data = _origin(directory)
        grant = read_decision(directory)
        return (
            data.get("origin") == ORIGIN_AGENT_WORKSTREAM
            and isinstance(grant, dict)
            and type(grant.get("version")) is int
            and grant.get("version") == 1
            and grant.get("session_id") == directory.name
            and grant.get("granted") is True
            and grant.get("source") == "operator-cli"
            and isinstance(grant.get("reason"), str)
            and bool(grant["reason"].strip())
        )
    except (OSError, ValueError, TypeError):
        return False


def set_grant(directory: Path, *, allow: bool, reason: str) -> dict[str, Any]:
    """Persist an explicit operator decision, preserving all origin metadata.

    No profile/spawn option calls this writer. The CLI is an operator control
    operation, executable by an assistant only under explicit user authority.
    Reject remote placement altogether: local policy cannot attest ownership on
    another device. Revocation remains available even after placement changes.
    """
    reason = reason.strip()
    if not reason:
        raise ValueError("reason_required")
    data = _origin(directory)
    if data.get("origin") != ORIGIN_AGENT_WORKSTREAM:
        raise ValueError("local_workstream_required")
    if allow and (directory / "mesh.json").exists():
        raise ValueError("remote_placement_not_supported")
    grant = {
        "version": 1,
        "session_id": directory.name,
        "granted": allow,
        "source": "operator-cli",
        "reason": reason,
        "updated_at": int(time.time() * 1000),
    }
    # Same-directory replacement makes readers see one complete policy. Unique
    # staging names avoid collisions between simultaneous operator decisions;
    # the last completed decision wins, as with other session sidecars.
    fd, temporary = tempfile.mkstemp(prefix=".notification-authority-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(grant, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, directory / GRANT_NAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return grant
