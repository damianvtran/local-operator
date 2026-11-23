"""Bind a moved session's named attachment to a definition, not to a registry key.

A source supplies only an expectation. The destination never installs a team
from the move payload or uses it to configure the runtime: only its own loaded
team can satisfy the comparison, and the runtime attaches that same object.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
# Earlier readers do not enforce the delegation ceiling. Never hand them an
# attachment, even when the top-level team and manager would match.
CAPABILITY = "authority-ceiling-v2"
CODE = "team_definition_unavailable"
LOCAL_BINDING_NAME = "local-team-binding.json"
_REAUTHORIZE = {"reauthorize": True}
_UNREADABLE_STAMP = object()


class TeamIdentityRefused(ValueError):
    code = CODE

    def __init__(self, name: str, authority: str = "") -> None:
        # The expected definition was not found under this name. This is also
        # true of a conflict; the name alone is not a candidate for attachment.
        message = render_resolved(keys.team_missing(role=name))
        super().__init__(f"{name}: {authority}" if authority else message)


class _DefinitionNotLoaded(TeamIdentityRefused):
    def __init__(self, team: str, subject: str) -> None:
        super().__init__(team, f"{subject} is unavailable")
        self.team = team
        self.subject = subject


class TeamDefinitionsPending(TeamIdentityRefused):
    """Independent reconciliation can supply a missing destination definition."""

    code = "team_definitions_pending"

    def __init__(self, missing: _DefinitionNotLoaded) -> None:
        super().__init__(
            missing.team,
            # i18n: ignore technical diagnostic carried verbatim by the relay protocol
            f"{missing.subject} is not available yet; retry after definitions sync",
        )


class UnboundAttachmentUnreadable(TeamIdentityRefused):
    """History can open, but no attachment can be restored from this evidence."""


def _digest(team: Team, *, structure_only: bool = False) -> str:
    from local_operator.network.definitions import digest_of

    # Identity is compared separately so a later hop can refresh the revision.
    # Local display/address aliases are not part of that revision. Compare the full hydrated briefs (not a transport-truncated
    # projection) and include the manager, roster and other definition fields.
    exclude = {"id", "created_date", "label", "aliases"}
    if structure_only:
        # After admission, the same locally identified team may edit its briefs
        # without reauthorizing its roster, manager or execution overrides.
        exclude.update({"instructions", "project", "description"})
    return digest_of(team.model_dump(mode="json", exclude=exclude))


def _resolve(registry: TeamRegistry, name: str) -> Team | None:
    try:
        team = registry.get_team_by_name(name, require_loaded=True)
    except Exception as error:  # noqa: BLE001 — an unreadable registry cannot authorize a team
        raise TeamIdentityRefused(name, "team definition is unavailable") from error
    if team is not None and team.name.casefold() != name.casefold():
        raise TeamIdentityRefused(name, "a team alias does not establish identity")
    return team


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
        if (
            isinstance(record, dict)
            and set(record) == {KEY}
            and isinstance(record[KEY], dict)
        ):
            return record[KEY]
        return _UNREADABLE_STAMP

    # mesh.json is destination-authored, not in the network copy set. In
    # particular, an attachment sidecar supplied by a peer cannot set this key.
    try:
        raw = json.loads((directory / MESH_STAMP_NAME).read_text("utf-8"))
        if (
            isinstance(raw, dict)
            and "origin" in raw
            and not isinstance(raw["origin"], dict)
        ):
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
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, delete=False
        ) as temp:
            path = Path(temp.name)
            json.dump({KEY: binding}, temp)
        path.replace(directory / LOCAL_BINDING_NAME)
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def inherit_fork_binding(parent: Path, child: Path) -> None:
    binding = _local_binding(parent)
    if binding is None or binding is _UNREADABLE_STAMP:
        require_replica_reauthorization(child)
    else:
        _write_local_binding(child, binding)


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
    binding = _local_binding(directory)
    # Presence of attachment.json is not a commit marker. A legacy copy or
    # hard interruption between copying it and publishing the local binding
    # provides no authority, even on a session that appears locally owned.
    return binding is None or (
        isinstance(binding, dict) and "delegation" not in binding
    )


def was_authorized_locally(directory: Path) -> bool:
    binding = _local_binding(directory)
    return isinstance(binding, dict) and binding.get("explicit_local") is True


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
            raise UnboundAttachmentUnreadable(
                name, "attachment evidence is unreadable"
            ) from error
        raise TeamIdentityRefused(name, "attachment evidence is unreadable") from error
    if expected is _UNREADABLE_STAMP or (stored is None and expected is not None):
        raise TeamIdentityRefused(name, "attachment evidence is unavailable")
    return stored


def read_local_attachment(directory: Path) -> SessionAttachment | None:
    return _attachment(directory, _local_binding(directory))


def _verify_name(name: str, expected: Any, team: Team | None) -> Team:
    if not isinstance(expected, dict) or "delegation" not in expected:
        raise TeamIdentityRefused(
            name, "recorded authority is unavailable; re-attach via /team"
        )
    expected_name = expected.get("name")
    if not isinstance(expected_name, str) or expected_name.casefold() != name.casefold():
        raise TeamIdentityRefused(name)
    if team is None:
        raise _DefinitionNotLoaded(name, "team definition")
    if team.name.casefold() != name.casefold():
        raise TeamIdentityRefused(name)
    return team


def _ceiling(name: str, allowed: Any, actual: dict[str, Any]) -> None:
    """Refuse any widening; an admitted result is already the intersection."""
    if (
        not isinstance(allowed, dict)
        or not isinstance(allowed.get("name"), str)
        or not isinstance(allowed.get("pins"), list)
        or not all(isinstance(pin, str) for pin in allowed["pins"])
        or type(allowed.get("readonly_shell")) is not bool
        or type(allowed.get("unrestricted_tools")) is not bool
        or type(allowed.get("may_delegate")) is not bool
    ):
        raise TeamIdentityRefused(name)
    if allowed["readonly_shell"] and not actual["readonly_shell"]:
        raise TeamIdentityRefused(name, "would grant an unrestricted shell")
    if allowed["name"] != actual["name"]:
        raise TeamIdentityRefused(name, f"would grant manager role {actual['name']!r}")
    if actual["unrestricted_tools"] and not allowed["unrestricted_tools"]:
        raise TeamIdentityRefused(
            name, f"would grant unrestricted tools to {actual['name']!r}"
        )
    added = sorted(set(actual["pins"]) - set(allowed["pins"]))
    if added and not allowed["unrestricted_tools"]:
        raise TeamIdentityRefused(name, f"would grant tool pins: {', '.join(added)}")
    if actual["may_delegate"] and not allowed["may_delegate"]:
        raise TeamIdentityRefused(name, f"would grant delegation to {actual['name']!r}")
    if (allowed.get("kind"), allowed.get("digest")) != (
        actual["kind"],
        actual["digest"],
    ):
        raise TeamIdentityRefused(
            name, f"role definition {actual['name']!r} is unavailable"
        )


def _role_authority(
    team_name: str, role: str, registry: Any, *, delegated: bool = False
) -> tuple[dict[str, Any], AgentProfile | None, str]:
    from local_operator.agent_profiles import resolve_profile_or_specialist_identity
    from local_operator.harness.subagent import is_readonly_shell_role
    from local_operator.network.definitions import digest_of

    try:
        kind, profile, prompt, _, author = resolve_profile_or_specialist_identity(
            role, registry=registry, strict=True
        )
    except (OSError, ValueError, KeyError) as error:
        raise TeamIdentityRefused(team_name) from error
    if kind is None:
        subject = "delegated role" if delegated else "manager"
        raise _DefinitionNotLoaded(team_name, f"{subject} definition {role!r}")
    # A seed's identity is its packaged name; a specialist/role's identity
    # comes from the very lookup that produced its applied prompt or pins.
    # Hash the authored profile, not any machine-local delivery pointer.
    profile_data = asdict(profile) if profile is not None else {}
    profile_data.pop("label", None)
    profile_data.pop("agent_id", None)
    # Pins and delegation permissions are ordered, not identity fields: hashing
    # them into the revision would reject a provably narrower role after the
    # ceiling check had already accepted it.
    profile_data.pop("tools", None)
    profile_data.pop("may_delegate", None)
    return (
        {
            "name": role,
            "kind": kind,
            "id": author or "",
            "digest": digest_of({"profile": profile_data, "prompt": prompt}),
            "pins": sorted(set(profile.tools or ())) if profile is not None else [],
            "unrestricted_tools": delegated and (profile is None or not profile.tools),
            "may_delegate": profile.may_delegate
            if profile is not None
            else kind == "specialist",
            "readonly_shell": is_readonly_shell_role(
                role, profile.name if profile else None
            ),
        },
        profile,
        prompt,
    )


@dataclass(frozen=True)
class TeamAuthority:
    """Resolved local objects and the record that attests to those very objects."""

    team: Team
    teams: dict[str, Team]
    roles: dict[str, tuple[dict[str, Any], AgentProfile | None, str]]
    manager: tuple[dict[str, Any], AgentProfile | None, str]
    max_depth: int

    def record(self) -> dict[str, Any]:
        return {
            "name": self.team.name,
            "id": self.team.id,
            "digest": _digest(self.team),
            "manager": self.manager[0],
            "delegation": {
                "teams": {
                    name: {
                        "id": team.id,
                        "digest": _digest(team),
                        "structure": _digest(team, structure_only=True),
                    }
                    for name, team in self.teams.items()
                },
                "roles": {name: data[0] for name, data in self.roles.items()},
                "max_depth": self.max_depth,
            },
        }


def _snapshot(
    team: Team,
    teams: TeamRegistry | None,
    agents: Any,
    *,
    max_depth: int,
    readonly_shell: bool = False,
) -> TeamAuthority:
    from local_operator.teams import HUB_TEAM_MEMBERS_MAX_ITEMS, MAX_ORG_DEPTH

    # Store the transitive roster, not just names from the top team. A nested
    # manager gets a child role's allowlist (including the unrestricted case),
    # whereas the root manager's tools pin schemas in the existing session.
    manager = _role_authority(team.name, team.manager, agents)
    manager[0]["readonly_shell"] |= readonly_shell
    team = team.model_copy(deep=True)
    resolved = {team.name.casefold(): team}
    roles: dict[str, tuple[dict[str, Any], AgentProfile | None, str]] = {}
    heights: dict[str, int] = {}

    def check_size(size: int) -> None:
        # Reuse the hub's supported flat roster size as the budget for one
        # transported closure. Each row also bounds edges, including repeats.
        if size > HUB_TEAM_MEMBERS_MAX_ITEMS:
            raise TeamIdentityRefused(
                team.name,
                f"delegation exceeds the {HUB_TEAM_MEMBERS_MAX_ITEMS}-slot bound",
            )

    def record_role(role: str) -> None:
        if role not in roles:
            check_size(len(resolved) + len(roles))  # +new role, -root team
            roles[role] = _role_authority(team.name, role, agents, delegated=True)

    def visit(current: Team, ancestors: tuple[str, ...]) -> int:
        key = current.name.casefold()
        if key in ancestors:
            raise TeamIdentityRefused(
                team.name, f"delegation cycle: {' > '.join((*ancestors, key))}"
            )
        if len(ancestors) + heights.get(key, 1) > MAX_ORG_DEPTH:
            raise TeamIdentityRefused(
                team.name, f"delegated team {key!r} exceeds depth {MAX_ORG_DEPTH}"
            )
        if key in heights:
            return heights[key]
        check_size(len(current.members))
        if current is not team:
            if current.manager == "task":
                # The launch resolver reserves this name for an unprofiled
                # generic child, so a role row cannot certify a nested lead.
                raise TeamIdentityRefused(
                    team.name, f"delegated manager of {key!r} is unavailable"
                )
            record_role(current.manager)
        height = 1
        for slot in current.members:
            if slot.kind == "agent":
                if slot.role != "task":
                    record_role(slot.role)
                continue
            child_key = slot.role.casefold()
            if child_key not in resolved:
                check_size(len(resolved) + len(roles))  # +new team, -root team
                nested = _resolve(teams, slot.role) if teams is not None else None
                if nested is None:
                    raise _DefinitionNotLoaded(team.name, f"delegated team {slot.role!r}")
                resolved[child_key] = nested.model_copy(deep=True)
            height = max(height, 1 + visit(resolved[child_key], (*ancestors, key)))
        heights[key] = height
        return height

    visit(team, ())
    return TeamAuthority(team, resolved, roles, manager, max_depth)


def _compare(
    expected: Any, actual: dict[str, Any], *, local: bool, exact_revision: bool = False
) -> None:
    name = actual["name"]
    if not isinstance(expected, dict) or not isinstance(
        expected.get("delegation"), dict
    ):
        raise TeamIdentityRefused(
            name, "recorded authority is unavailable; re-attach via /team"
        )
    _ceiling(name, expected.get("manager"), actual["manager"])
    if expected.get("name") != name or (local and expected.get("id") != actual["id"]):
        raise TeamIdentityRefused(name)
    if local and expected["manager"].get("id") != actual["manager"]["id"]:
        raise TeamIdentityRefused(name)
    old, new = expected["delegation"], actual["delegation"]
    if exact_revision and expected.get("digest") != actual["digest"]:
        raise TeamIdentityRefused(name)
    revision = "digest" if exact_revision else "structure"
    for key in ("teams", "roles"):
        allowed = old.get(key)
        if not isinstance(allowed, dict) or set(allowed) != set(new[key]):
            raise TeamIdentityRefused(
                name, "recorded delegation does not match; re-attach via /team"
            )
        for slot, candidate in new[key].items():
            entry = allowed[slot]
            if not isinstance(entry, dict) or (
                local and entry.get("id") != candidate["id"]
            ):
                raise TeamIdentityRefused(name)
            if key == "roles":
                _ceiling(name, entry, candidate)
            elif entry.get(revision) != candidate[revision]:
                raise TeamIdentityRefused(
                    name, f"team definition {slot!r} does not match"
                )


def _recorded_limits(expected: Any, root: Path | None, role: str) -> tuple[int, bool]:
    from local_operator.harness.subagent import read_max_team_depth
    from local_operator.teams import MAX_ORG_DEPTH

    depth = read_max_team_depth(root)
    if not isinstance(expected, dict):
        return depth, False  # _compare refuses the absent ceiling before any grant.
    graph, manager = expected.get("delegation"), expected.get("manager")
    prior = graph.get("max_depth") if isinstance(graph, dict) else None
    if type(prior) is not int or not 1 <= prior <= MAX_ORG_DEPTH:
        raise TeamIdentityRefused(
            str(expected.get("name") or "?"), "recorded delegation is unavailable"
        )
    # A sticky source shell restriction does not depend on the user's last
    # attached role; reinstall it at the next owner. A lower depth cap can
    # take effect, but a registry/config refresh cannot increase it.
    confined = (
        isinstance(manager, dict)
        and manager.get("name") == role
        and manager.get("readonly_shell") is True
    )
    return min(depth, prior), confined


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
        and row.get("digest")
        == digest_of(_team_row_from_bundle(_team_row_from_team(team)))
    ):
        return str(row.get("origin") or "")
    return ""


def _snapshot_at_root(
    root: Path, team: Team, teams: TeamRegistry, expected: Any
) -> TeamAuthority:
    from local_operator.agents import AgentRegistry

    depth, readonly_shell = _recorded_limits(expected, root, team.manager)
    return _snapshot(
        team, teams, AgentRegistry(root), max_depth=depth, readonly_shell=readonly_shell
    )


def source_offer(
    root: Path, directory: Path, *, supported: bool
) -> dict[str, Any] | None:
    from local_operator.teams import TeamRegistry

    prior = _local_binding(directory)
    stored = _attachment(directory, prior)
    if stored is None or not stored.team:
        return None
    name = stored.team
    if not supported:
        raise TeamIdentityRefused(name)
    if prior is None:
        # Nothing on this device attests to the named team (a legacy sidecar, or
        # a team this device never held), so there is no authority to offer. The
        # name still travels in the sidecar and the destination reports it as
        # unresolved — refusing the MOVE here would let a missing or absent
        # definition block moving the conversation itself (R5-3).
        return None
    registry = TeamRegistry(root)
    team = _verify_name(name, prior, _resolve(registry, name))
    record = _snapshot_at_root(root, team, registry, prior).record()
    _compare(prior, record, local=True)
    # A local edit may change the revision offered on the next hop, but can
    # never replace the last owner’s authority with a wider current registry.
    return {**record, "origin": _mirrored_origin(root, team)}


def check_arrival(root: Path, directory: Path, expected: Any) -> dict[str, Any] | None:
    try:
        return _check_destination(root, directory, expected, promoting=False)
    except _DefinitionNotLoaded as missing:
        # Only arrival depends on the destination's background reconciliation.
        # Explicit read failures and rejected authority comparisons keep their
        # own refusal; waiting for another definitions tick is not their remedy.
        raise TeamDefinitionsPending(missing) from missing


def _check_destination(
    root: Path, directory: Path, expected: Any, *, promoting: bool
) -> dict[str, Any] | None:
    from local_operator.teams import TeamRegistry

    stored = _attachment(directory, expected)
    if stored is None or not stored.team:
        if expected is not None:
            name = (
                str(expected.get("name") or "?") if isinstance(expected, dict) else "?"
            )
            raise TeamIdentityRefused(name)
        return None
    if expected is None:
        # The source offered NO record (it could not attest the name). A name is
        # not authority: install nothing and record no team binding, so the
        # arrival carries the hint and the receipt says it is unresolved. Only an
        # attested record can be verified, and only that can ever attach.
        return None
    registry = TeamRegistry(root)
    team = _verify_name(stored.team, expected, _resolve(registry, stored.team))
    record = _snapshot_at_root(root, team, registry, expected).record()
    _compare(expected, record, local=promoting, exact_revision=True)
    # Save the destination's actual authority, not the source's (possibly
    # wider) offer. UUIDs may differ across devices, never within one owner.
    origin = _mirrored_origin(root, team)
    if promoting and expected.get("origin") != origin:
        raise TeamIdentityRefused(team.name)
    return {**record, "origin": origin}


def check_promotion(root: Path, directory: Path) -> None:
    _check_destination(root, directory, _local_binding(directory), promoting=True)


def restore_bound(directory: Path, name: str, registry: TeamRegistry | None) -> Team:
    """Resolve the named object; attach still requires its recorded ceiling."""
    expected = _local_binding(directory)
    team = _resolve(registry, name) if registry is not None else None
    return _verify_name(name, expected, team)


def is_pending_definition(error: BaseException) -> bool:
    """Whether a refusal ONLY says a definition has not reached this device yet.

    WHY (review r5 Q-R5-1): a definition sync is independent of a move, so its
    absence is a WAIT. The caller keeps the recorded team and retries on the next
    turn, instead of treating an in-flight row exactly like a rejected record —
    which lost the recorded team and, on the raise path, made the conversation
    unopenable. A RECORD that does not match stays a refusal and this returns
    False for it, so no authority follows from waiting.
    """
    return isinstance(error, _DefinitionNotLoaded)


def authorize_named_team(root: Path, directory: Path, name: str) -> str:
    from local_operator.agents import AgentRegistry
    from local_operator.teams import TeamRegistry

    registry = TeamRegistry(root)
    team = _resolve(registry, name)
    if team is None:
        raise TeamIdentityRefused(name, "team definition is unavailable")
    bind_runtime_authority(
        directory, team, registry, AgentRegistry(root), restoring=False
    )
    return team.name


def bind_runtime_authority(
    directory: Path | None,
    team: Team,
    teams: TeamRegistry | None,
    agents: Any,
    *,
    restoring: bool,
    readonly_shell: bool = False,
) -> TeamAuthority:
    from local_operator.harness.subagent import read_max_team_depth

    root = teams.config_dir if teams is not None else None
    if restoring:
        if directory is None:
            raise TeamIdentityRefused(team.name)
        expected = _local_binding(directory)
        depth, confined = _recorded_limits(expected, root, team.manager)
    else:
        # Only an explicit local attach starts a new authorization. Resume,
        # fork and transit must only consume ceilings already written.
        expected, depth, confined = None, read_max_team_depth(root), False
    snapshot = _snapshot(
        team, teams, agents, max_depth=depth, readonly_shell=readonly_shell or confined
    )
    record = snapshot.record()
    if restoring:
        _compare(expected, record, local=True)
    if directory is not None:
        # Tightening must survive reopen and the next hop. In particular, an
        # intermediate owner's reopened brief is not pinned to an old digest.
        _write_local_binding(
            directory,
            {
                **record,
                "explicit_local": not restoring
                or expected.get("explicit_local") is True,
            },
        )
    return snapshot
