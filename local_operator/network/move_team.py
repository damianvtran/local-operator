"""Bind a moved session's named attachment to a definition, not to a registry key.

A source supplies only an expectation. The destination never installs a team
from the move payload or uses it to configure the runtime: only its own loaded
team can satisfy the comparison, and the runtime attaches that same object.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from local_operator.i18n.keys import wire_teams_effective as keys
from local_operator.i18n.messages import render_resolved

if TYPE_CHECKING:
    from local_operator.agent_profiles import AgentProfile
    from local_operator.resume import SessionAttachment
    from local_operator.teams import Team, TeamRegistry

KEY = "team_identity"
# A boolean capability only promised a team-content comparison. It did not
# promise a manager/authority ceiling, so never offer authority to that peer.
CAPABILITY = "authority-ceiling-v1"
CODE = "team_definition_unavailable"
LOCAL_BINDING_NAME = "local-team-binding.json"
_REAUTHORIZE = {"reauthorize": True}
_UNREADABLE_STAMP = object()


class TeamIdentityRefused(ValueError):
    def __init__(self, name: str, authority: str = "") -> None:
        # The expected definition was not found under this name. This is also
        # true of a conflict; the name alone is not a candidate for attachment.
        message = render_resolved(keys.team_missing(role=name))
        super().__init__(f"{name}: {authority}" if authority else message)


class UnboundAttachmentUnreadable(TeamIdentityRefused):
    """History can open, but no attachment can be restored from this evidence."""


def _digest(team: Team) -> str:
    from local_operator.network.definitions import digest_of

    # Identity is compared separately so a later hop can refresh the revision.
    # Local display/address aliases are not part of that revision. Compare the full hydrated briefs (not a transport-truncated
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

    # A local fork can inherit the comparison without inheriting ownership.
    # Explicit local attach may also supersede an inherited binding. Neither
    # journal is admitted from a peer's copy set.
    try:
        record = json.loads((directory / LOCAL_BINDING_NAME).read_text("utf-8"))
    except FileNotFoundError:
        pass
    except (OSError, ValueError):
        return _UNREADABLE_STAMP
    else:
        if isinstance(record, dict) and set(record) == {KEY} and isinstance(record[KEY], dict):
            return record[KEY]
        return _UNREADABLE_STAMP

    # mesh.json is destination-authored, not in the network copy set. In
    # particular, an attachment sidecar supplied by a peer cannot set this key.
    try:
        raw = json.loads((directory / MESH_STAMP_NAME).read_text("utf-8"))
        if isinstance(raw, dict) and "origin" in raw and not isinstance(raw["origin"], dict):
            return _UNREADABLE_STAMP
        stamp = MeshStamp.from_json(raw)
    except FileNotFoundError:
        return None
    except (OSError, TypeError, ValueError):
        # A legible legacy record may be unbound. A broken record is not
        # evidence that a previously bound attachment was ever unbound.
        return _UNREADABLE_STAMP
    if stamp is None:
        return _UNREADABLE_STAMP
    if KEY not in stamp.origin:
        return None
    value = stamp.origin[KEY]
    return value if isinstance(value, dict) else _UNREADABLE_STAMP


def _write_local_binding(directory: Path, binding: Any) -> None:
    # Use an atomic local-only write so an interrupted explicit choice never
    # leaves a half-record that can be mistaken for authorization.
    path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as temp:
            path = Path(temp.name)
            json.dump({KEY: binding}, temp)
        path.replace(directory / LOCAL_BINDING_NAME)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def inherit_fork_binding(parent: Path, child: Path) -> None:
    binding = _local_binding(parent)
    if binding is not None:
        _write_local_binding(child, _REAUTHORIZE if binding is _UNREADABLE_STAMP else binding)


def require_replica_reauthorization(directory: Path) -> None:
    # Replica bytes do not include the owner's trusted comparison. Record a
    # local denial before recovery releases its claim; only an explicit local
    # attach may clear it. Unreadable sidecars are held too; a definitively
    # absent or detached replica has no team authority to carry.
    from local_operator.resume import (
        SessionAttachmentUnreadable,
        read_session_attachment_evidence,
    )

    try:
        stored = read_session_attachment_evidence(directory)
    except SessionAttachmentUnreadable:
        _write_local_binding(directory, _REAUTHORIZE)
    else:
        if stored is not None and stored.team:
            _write_local_binding(directory, _REAUTHORIZE)


def needs_reauthorization(directory: Path) -> bool:
    return _local_binding(directory) == _REAUTHORIZE


def journal_local_authority(
    directory: Path, name: str, manager: str, *, pins: tuple[str, ...], readonly_shell: bool
) -> None:
    if name:
        _write_local_binding(
            directory,
            {
                "local_name": name,
                "manager": {"name": manager, "pins": list(pins), "readonly_shell": readonly_shell},
            },
        )


def _bound_for(binding: Any, name: str) -> Any:
    # This override is written only after an explicit local attach. A crash
    # before the normal attachment journal commits cannot authorize another name.
    if isinstance(binding, dict) and binding.get("local_name") == name:
        return None
    return binding


def _attachment(directory: Path, expected: Any) -> SessionAttachment | None:
    from local_operator.resume import (
        SessionAttachmentUnreadable,
        read_session_attachment_evidence,
    )

    name = str(expected.get("name") or "?") if isinstance(expected, dict) else "?"
    try:
        stored = read_session_attachment_evidence(directory)
    except SessionAttachmentUnreadable as error:
        if expected is None:
            raise UnboundAttachmentUnreadable(name, "attachment evidence is unreadable") from error
        raise TeamIdentityRefused(name, "attachment evidence is unreadable") from error
    if expected is _UNREADABLE_STAMP or (stored is None and expected is not None):
        raise TeamIdentityRefused(name, "attachment evidence is unavailable")
    return stored


def read_local_attachment(directory: Path) -> SessionAttachment | None:
    return _attachment(directory, _local_binding(directory))


def _verify_name(name: str, expected: Any, team: Team | None) -> Team:
    if team is None:
        raise TeamIdentityRefused(name, "team definition is unavailable")
    if (
        team.name.casefold() != name.casefold()
        or not isinstance(expected, dict)
        or not isinstance(expected_name := expected.get("name"), str)
        or expected_name.casefold() != name.casefold()
    ):
        raise TeamIdentityRefused(name)
    return team


def _verify(name: str, expected: Any, team: Team | None) -> None:
    team = _verify_name(name, expected, team)
    # The UUID is an attestation signal, not a cross-device requirement: two
    # devices can independently author the same definition. Do not trim the
    # revision to the mesh sync projection: per-slot execution overrides and
    # model suggestions are not yet transported. Such teams fail closed until
    # those settings can be mirrored without loss.
    if expected.get("digest") != _digest(team):
        raise TeamIdentityRefused(name)


def _ceiling(name: str, allowed: Any, actual: dict[str, Any]) -> None:
    """Refuse any widening; an admitted result is already the intersection."""
    if (
        not isinstance(allowed, dict)
        or not isinstance(allowed.get("name"), str)
        or not isinstance(allowed.get("pins"), list)
        or not all(isinstance(pin, str) for pin in allowed["pins"])
        or type(allowed.get("readonly_shell")) is not bool
    ):
        raise TeamIdentityRefused(name)
    if allowed["readonly_shell"] and not actual["readonly_shell"]:
        raise TeamIdentityRefused(name, "would grant an unrestricted shell")
    if allowed["name"] != actual["name"]:
        raise TeamIdentityRefused(name, f"would grant manager role {actual['name']!r}")
    added = sorted(set(actual["pins"]) - set(allowed["pins"]))
    if added:
        raise TeamIdentityRefused(name, f"would grant tool pins: {', '.join(added)}")


def _manager_authority(
    team: Team, registry: Any
) -> tuple[dict[str, Any], AgentProfile | None, str]:
    from local_operator.agent_profiles import resolve_profile_or_specialist_identity
    from local_operator.harness.subagent import is_readonly_shell_role
    from local_operator.network.definitions import digest_of

    try:
        kind, profile, prompt, _, author = resolve_profile_or_specialist_identity(
            team.manager, registry=registry, strict=True
        )
    except (OSError, ValueError, KeyError) as error:
        raise TeamIdentityRefused(team.name) from error
    if kind is None:
        raise TeamIdentityRefused(team.name, f"manager definition {team.manager!r} is unavailable")
    # A seed's identity is its packaged name; a specialist/role's identity
    # comes from the very lookup that produced its applied prompt or pins.
    # Hash the authored profile, not any machine-local delivery pointer.
    profile_data = asdict(profile) if profile is not None else {}
    profile_data.pop("label", None)
    profile_data.pop("agent_id", None)
    return (
        {
            "name": team.manager,
            "kind": kind,
            "id": author or "",
            "digest": digest_of({"profile": profile_data, "prompt": prompt}),
            "pins": sorted(set(profile.tools or ())) if profile is not None else [],
            "readonly_shell": is_readonly_shell_role(team.manager),
        },
        profile,
        prompt,
    )


def _manager_at_root(root: Path, team: Team) -> dict[str, Any]:
    from local_operator.agents import AgentRegistry

    return _manager_authority(team, AgentRegistry(root))[0]


def _mirrored_origin(root: Path, team: Team) -> str:
    from local_operator.network.definitions import (
        _recorded,
        _team_row_from_bundle,
        _team_row_from_team,
        digest_of,
        read_index,
    )

    # Carry an authenticated sync origin only for an unchanged mirror. The
    # attestation is supplemental; an independent identical row can still
    # pass the same revision and ceiling checks without it.
    row = _recorded(read_index(root), "teams", team.name)
    if (
        row is not None
        and row.get("id") == team.id
        and row.get("digest") == digest_of(_team_row_from_bundle(_team_row_from_team(team)))
    ):
        return str(row.get("origin") or "")
    return ""


def source_offer(root: Path, directory: Path, *, supported: bool) -> dict[str, Any] | None:
    from local_operator.teams import TeamRegistry

    prior = _local_binding(directory)
    stored = _attachment(directory, prior)
    if stored is None or not stored.team:
        return None
    name = stored.team
    # Old receivers restore by name. Refuse before handoff rather than offer
    # them a binding they cannot enforce.
    if not supported:
        raise TeamIdentityRefused(name)
    team = _resolve(TeamRegistry(root), name)
    if team is None:
        raise TeamIdentityRefused(name, "team definition is unavailable")
    manager = _manager_at_root(root, team)
    if isinstance(prior, dict) and prior.get("local_name") == name:
        # A running source may still hold an older or more restrictive profile
        # than its registry now advertises. Use the last runtime's actual ceiling.
        _ceiling(name, prior.get("manager"), manager)
    prior = _bound_for(prior, name)
    if prior is not None:
        # The last hop is a ceiling, not the next hop's revision. A new brief
        # can travel but a later owner cannot silently widen an inherited grant.
        _verify_name(name, prior, team)
        _ceiling(name, prior.get("manager"), manager)
    return {
        "name": name,
        "id": team.id,
        "digest": _digest(team),
        "manager": manager,
        "origin": _mirrored_origin(root, team),
    }


def check_arrival(root: Path, directory: Path, expected: Any) -> dict[str, Any] | None:
    from local_operator.teams import TeamRegistry

    stored = _attachment(directory, expected)
    if stored is None or not stored.team:
        if expected is not None:
            name = str(expected.get("name") or "?") if isinstance(expected, dict) else "?"
            raise TeamIdentityRefused(name)
        return None
    # The copied attachment supplies the name; an absent destination row is
    # never a candidate, even if a later definitions refresh would match it.
    name = stored.team
    team = _verify_name(name, expected, _resolve(TeamRegistry(root), name))
    manager = _manager_at_root(root, team)
    ceiling = expected.get("manager")
    _ceiling(name, ceiling, manager)
    _verify(name, expected, team)
    # Compare the actual revision independently of the registry's UUID. Only
    # the destination's own resolved objects may configure its runtime.
    if (ceiling.get("kind"), ceiling.get("digest")) != (manager["kind"], manager["digest"]):
        raise TeamIdentityRefused(name, f"manager definition {team.manager!r} is unavailable")
    resolved = {
        "local_team_id": team.id,
        "local_manager_id": manager["id"],
        "local_origin": _mirrored_origin(root, team),
    }
    if any(key in expected and expected[key] != value for key, value in resolved.items()):
        raise TeamIdentityRefused(name)
    return {**expected, **resolved}


def check_promotion(root: Path, directory: Path) -> None:
    check_arrival(root, directory, _local_binding(directory))


def restore_bound(directory: Path, name: str, registry: TeamRegistry | None) -> Team | None:
    """Return the exact verified object to attach, or None for a local attachment."""
    expected = _bound_for(_local_binding(directory), name)
    if expected is None:
        return None
    # The registry can change between the protocol check and construction.
    # Fail before attach_team; never verify one lookup then attach a second.
    team = _resolve(registry, name) if registry is not None else None
    if team is None:
        raise TeamIdentityRefused(name, "team definition is unavailable")
    _verify(name, expected, team)
    return team


def restore_manager(
    directory: Path, team: Team, registry: Any
) -> tuple[str, AgentProfile | None, str]:
    expected = _local_binding(directory)
    _verify(team.name, expected, team)
    authority, profile, prompt = _manager_authority(team, registry)
    ceiling = expected.get("manager") if isinstance(expected, dict) else None
    _ceiling(team.name, ceiling, authority)
    if (
        expected.get("local_team_id") != team.id
        or expected.get("local_manager_id") != authority["id"]
        or (ceiling.get("kind"), ceiling.get("digest")) != (authority["kind"], authority["digest"])
    ):
        raise TeamIdentityRefused(team.name, f"manager definition {team.manager!r} is unavailable")
    if authority["kind"] == "role" and profile is not None:
        from local_operator.agent_profiles import with_delivery_pointer

        profile = with_delivery_pointer(profile)
    return authority["kind"], profile, prompt
