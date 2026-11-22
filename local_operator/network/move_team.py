"""Bind a moved session's named attachment to a definition, not to a registry key.

A source supplies only an expectation. The destination never installs a team
from the move payload or uses it to configure the runtime: only its own loaded
team can satisfy the comparison, and the runtime attaches that same object.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.i18n.keys import wire_teams_effective as keys
from local_operator.i18n.messages import render_resolved

if TYPE_CHECKING:
    from local_operator.teams import Team, TeamRegistry

KEY = "team_identity"
CODE = "team_definition_unavailable"
_UNREADABLE_STAMP = object()


class TeamIdentityRefused(ValueError):
    def __init__(self, name: str) -> None:
        # The expected definition was not found under this name. This is also
        # true of a conflict; the name alone is not a candidate for attachment.
        super().__init__(render_resolved(keys.team_missing(role=name)))


def _digest(team: Team) -> str:
    from local_operator.network.definitions import digest_of

    # Storage identity and local display/address aliases are not the team
    # definition. Compare the full hydrated briefs (not a transport-truncated
    # projection) and include the manager, roster and other definition fields.
    return digest_of(
        team.model_dump(mode="json", exclude={"id", "created_date", "label", "aliases"})
    )


def _resolve(registry: TeamRegistry, name: str) -> Team | None:
    from local_operator.types import TeamRegistryRecoveryError

    try:
        team = registry.get_team_by_name(name, require_loaded=True)
    except (OSError, TeamRegistryRecoveryError) as error:
        raise TeamIdentityRefused(name) from error
    return team if team is not None and team.name.casefold() == name.casefold() else None


def _local_binding(directory: Path) -> Any:
    from local_operator.session.placement import MESH_STAMP_NAME, MeshStamp

    # mesh.json is destination-authored, not in the network copy set. In
    # particular, an attachment sidecar supplied by a peer cannot set this key.
    try:
        stamp = MeshStamp.from_json(json.loads((directory / MESH_STAMP_NAME).read_text("utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError):
        # A legible legacy record may be unbound. A broken record is not
        # evidence that a previously bound attachment was ever unbound.
        return _UNREADABLE_STAMP
    return stamp.origin.get(KEY) if stamp is not None else _UNREADABLE_STAMP


def _verify(name: str, expected: Any, team: Team | None) -> dict[str, str]:
    digest = expected.get("digest") if isinstance(expected, dict) else None
    expected_name = expected.get("name") if isinstance(expected, dict) else None
    if (
        not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        or not isinstance(expected_name, str)
        or expected_name.casefold() != name.casefold()
        or team is None
        or team.name.casefold() != name.casefold()
        or _digest(team) != digest
    ):
        raise TeamIdentityRefused(name)
    return {"name": name, "digest": digest}


def source_offer(root: Path, directory: Path, *, supported: bool) -> dict[str, str] | None:
    from local_operator.resume import read_session_attachment
    from local_operator.teams import TeamRegistry

    stored = read_session_attachment(directory)
    if stored is None or not stored.team:
        return None
    name = stored.team
    # Old receivers restore by name. Refuse before handoff rather than offer
    # them a binding they cannot enforce.
    if not supported:
        raise TeamIdentityRefused(name)
    team = _resolve(TeamRegistry(root), name)
    if team is None:
        raise TeamIdentityRefused(name)
    expected = _binding_for(directory, name)
    if expected is not None:
        return _verify(name, expected, team)
    return {"name": name, "digest": _digest(team)}


def check_arrival(root: Path, directory: Path, expected: Any) -> dict[str, str] | None:
    from local_operator.resume import read_session_attachment
    from local_operator.teams import TeamRegistry

    stored = read_session_attachment(directory)
    if stored is None or not stored.team:
        if isinstance(expected, dict) and isinstance(name := expected.get("name"), str) and name:
            raise TeamIdentityRefused(name)
        return None
    # Use the actual attachment from the verified copy. The prepare reply
    # alone does not get to choose which name the destination resolves.
    return _verify(stored.team, expected, _resolve(TeamRegistry(root), stored.team))


def check_promotion(root: Path, directory: Path) -> None:
    check_arrival(root, directory, _local_binding(directory))


def _binding_for(directory: Path, name: str) -> Any:
    expected = _local_binding(directory)
    bound_name = expected.get("name") if isinstance(expected, dict) else None
    if isinstance(bound_name, str) and bound_name.casefold() != name.casefold():
        # The verified handoff checked the staged name before promotion. A
        # later, separate local attach can change the attachment to a new name.
        return None
    return expected


def restore_bound(directory: Path, name: str, registry: TeamRegistry | None) -> Team | None:
    """Return the exact verified object to attach, or None for a local attachment."""
    expected = _binding_for(directory, name)
    if expected is None:
        return None
    # The registry can change between the protocol check and construction.
    # Fail before attach_team; never verify one lookup then attach a second.
    team = _resolve(registry, name) if registry is not None else None
    if team is None:
        raise TeamIdentityRefused(name)
    _verify(name, expected, team)
    return team
