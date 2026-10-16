"""Opaque definition revisions for local agent/profile and team edit CAS.

An editor reads an authored snapshot, the operator edits it, and the write goes
back minutes later — long enough for another window, the scheduled hub update, or
the desktop to have written the same row. A revision token lets the write say
which snapshot it was based on, so a stale save is refused (409) under the SAME
writer lock the save path already holds, instead of silently overwriting.

The digest binds the STABLE id and the AUTHORED editable fields only. Runtime and
provenance noise must not participate, or an unrelated chat message or a hub
re-pull would invalidate every open editor for no user-visible reason.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

#: Domain prefix keeps this digest from ever colliding with a link_id or a
#: manifest fingerprint built from the same fields elsewhere.
_PREFIX = "definition-revision:v1:"

#: Authored editable fields per kind. Deliberately excludes last_message,
#: created_date, execution/conversation history, provenance tags (seed_origin,
#: hub_*), hub policy (subscribe/auto_push), and the revision token itself.
AGENT_FIELDS = (
    "name",
    "label",
    "kind",
    "description",
    "instructions",
    "tools",
    "effort",
    "delegate",
    "action_class",
)
TEAM_FIELDS = (
    "name",
    "label",
    "description",
    "manager",
    "members",
    "aliases",
    "instructions",
    "project",
)


def _canonical(fields: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: fields.get(name) for name in names}


def definition_revision(kind: str, stable_id: str, fields: Mapping[str, Any]) -> str:
    """Opaque revision over the stable id plus this kind's authored fields.

    ``fields`` may carry extra keys (a whole row); only the authored set is
    read, so a caller never needs to pre-filter. ``instructions`` must already
    be the FULL uncropped authored body — a truncated body here would make a
    silently shortened save look unchanged.
    """
    if kind not in ("agent", "team", "skill"):
        raise ValueError(f"unknown definition kind {kind!r}")
    names = TEAM_FIELDS if kind == "team" else AGENT_FIELDS
    payload = {"kind": kind, "id": str(stable_id), "fields": _canonical(fields, names)}
    body = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return _PREFIX + hashlib.sha256(body.encode("utf-8")).hexdigest()


class DefinitionRevisionMismatch(Exception):
    """The supplied expected revision is stale for the current row."""

    code = "local-concurrent-edit"


def require_fresh_revision(
    kind: str, stable_id: str, current_fields: Mapping[str, Any], expected: str | None
) -> None:
    """Refuse a write whose expected definition revision is not the current one.

    Call UNDER the definition writer lock, against the row as re-read under that
    lock. ``None`` is the documented legacy unconditional write and is not a
    protected dirty-edit guarantee. A malformed token is refused by name rather
    than coerced.
    """
    if expected is None:
        return
    if not isinstance(expected, str) or not expected.startswith(_PREFIX):
        raise DefinitionRevisionMismatch("definition revision is malformed")
    if expected != definition_revision(kind, stable_id, current_fields):
        raise DefinitionRevisionMismatch("the definition changed since it was read")
