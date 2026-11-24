"""Bind a moved session's named attachment to a definition, not to a registry key.

A source supplies only an expectation. The destination never installs a team
from the move payload or uses it to configure the runtime: only its own loaded
team can satisfy the comparison, and the runtime attaches that same object.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
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

logger = logging.getLogger(__name__)


#: Record ``kind`` of a roster member that names no registered profile,
#: specialist or packaged seed. See ``_role_authority``.
PLAIN_ROLE = "plain"
#: Record of a ``team:`` slot naming no team. Empty strings never equal a real
#: id or digest, so an absent record can never pass for a present one.
_ABSENT_TEAM: dict[str, str] = {"id": "", "digest": "", "structure": ""}


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
    # Local display/address aliases are not part of that revision. Compare the
    # full hydrated briefs (not a transport-truncated projection) and include
    # the manager, roster and other definition fields.
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
    #
    # FAILURE IS FAIL-CLOSED, SO IT PROPAGATES. Every caller treats an
    # exception here as "no authority recorded": a fork is discarded
    # (fork.py), a recovered replica is refused and quarantined (sync.py), an
    # attach raises before the team is exposed (session.py). ``Path.replace``
    # is atomic, so the previous binding (or its absence) survives intact and
    # restore keeps refusing anything it does not match.
    #
    # WINDOWS. ``Path.replace`` onto an existing binding raises PermissionError
    # while another process holds that file open without FILE_SHARE_DELETE
    # (CPython's own ``open`` does not grant it), e.g. a concurrent reader of
    # this session. That is the same fail-closed outcome, so it is NOT retried
    # or swallowed: there is no measured contention to size a retry against,
    # and a swallowed error would leave the caller believing authority was
    # recorded. A note names the consequence on the propagating error.
    #
    # The temp cleanup runs only on failure (after a successful replace the
    # temp no longer exists). On Windows its unlink can itself fail while a
    # scanner/indexer holds the file; raising THAT would replace the real
    # failure the caller must see, so it is logged by name and the original
    # error propagates. A leftover temp is inert: readers open only
    # ``LOCAL_BINDING_NAME``, and that journal is never admitted from a peer.
    path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as temp:
            path = Path(temp.name)
            json.dump({KEY: binding}, temp)
        path.replace(directory / LOCAL_BINDING_NAME)
    except BaseException as error:
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup:
                logger.warning(
                    "left local team binding temp %s in %s after a failed write: %s",
                    path.name,
                    directory,
                    cleanup,
                )
        if isinstance(error, OSError):
            # i18n: ignore technical diagnostic attached to the propagating error
            error.add_note(
                f"{LOCAL_BINDING_NAME} was not written in {directory}; "
                "no team authority was recorded (fail-closed)"
            )
        raise


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
    return binding is None or (isinstance(binding, dict) and "delegation" not in binding)


def resolve_legacy_local_team(
    directory: Path, name: str, registry: TeamRegistry | None
) -> Team | None:
    """Resolve a legacy team only with a device-local workstream creation witness."""
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.guardrails.provenance import read_lineage
    from local_operator.resume import ORIGIN_AGENT_SHELL, ORIGIN_AGENT_WORKSTREAM, ORIGIN_NAME
    from local_operator.session.placement import MESH_STAMP_NAME

    if registry is None or directory.parent.name != "sessions":
        return None
    # v0.112.10 began requiring a binding. Treat an old attachment on its own
    # device as a local attach, but do NOT infer origin from two missing files:
    # historical recovery could leave a partial copy before stamping it. The
    # protected lineage journal is outside the portable session tree, records
    # agent-shell sessions only at local creation, and is keyed by the new ID
    # (fork/recovery allocate their own IDs). Legacy sessions without this
    # positive witness still require explicit re-attachment.
    for marker in (LOCAL_BINDING_NAME, MESH_STAMP_NAME, FORK_BOUNDARY_NAME):
        try:
            (directory / marker).lstat()
        except FileNotFoundError:
            continue
        except OSError:
            return None
        return None
    try:
        origin = json.loads((directory / ORIGIN_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    kind = origin.get("origin") if isinstance(origin, dict) else None
    if kind not in (ORIGIN_AGENT_WORKSTREAM, ORIGIN_AGENT_SHELL):
        return None
    row = read_lineage(directory.parent.parent).get(directory.name)
    if row is None or row.get("origin") != kind:
        return None
    return _resolve(registry, name)


def has_local_authority(directory: Path, name: str) -> bool:
    """Whether a completed local binding exists FOR ``name`` — the warm gates.

    The complement of :func:`needs_reauthorization` restricted to a readable
    record, plus the same case-insensitive name match :func:`_verify_name`
    applies on the warm restore: a cold reader (``AttachedSession``) must not
    show a team in force where the runtime would refuse it, whether as
    name-only (no binding) or as a different team than the one bound (a
    checkpoint naming ``admins`` beside a ``lopdev`` binding). An unreadable
    journal is not authority either.
    """
    binding = _local_binding(directory)
    if not isinstance(binding, dict) or needs_reauthorization(directory):
        return False
    bound = binding.get("name")
    return isinstance(bound, str) and bound.casefold() == name.casefold()


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
            raise UnboundAttachmentUnreadable(name, "attachment evidence is unreadable") from error
        raise TeamIdentityRefused(name, "attachment evidence is unreadable") from error
    if expected is _UNREADABLE_STAMP or (stored is None and expected is not None):
        raise TeamIdentityRefused(name, "attachment evidence is unavailable")
    return stored


def read_local_attachment(directory: Path) -> SessionAttachment | None:
    return _attachment(directory, _local_binding(directory))


def _verify_name(name: str, expected: Any, team: Team | None) -> Team:
    if not isinstance(expected, dict) or "delegation" not in expected:
        raise TeamIdentityRefused(name, "recorded authority is unavailable; re-attach via /team")
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
    if actual["kind"] == PLAIN_ROLE and allowed.get("kind") != PLAIN_ROLE:
        # The record attests a real definition that this device does not hold
        # (yet): a definitions-sync WAIT, not a widening to a full child. This
        # must run before the tool checks below, which would otherwise read the
        # missing row as "would grant unrestricted tools" and refuse for good.
        raise _DefinitionNotLoaded(name, f"delegated role definition {actual['name']!r}")
    if allowed["readonly_shell"] and not actual["readonly_shell"]:
        raise TeamIdentityRefused(name, "would grant an unrestricted shell")
    if allowed["name"] != actual["name"]:
        raise TeamIdentityRefused(name, f"would grant manager role {actual['name']!r}")
    if actual["unrestricted_tools"] and not allowed["unrestricted_tools"]:
        raise TeamIdentityRefused(name, f"would grant unrestricted tools to {actual['name']!r}")
    added = sorted(set(actual["pins"]) - set(allowed["pins"]))
    if added and not allowed["unrestricted_tools"]:
        raise TeamIdentityRefused(name, f"would grant tool pins: {', '.join(added)}")
    if actual["may_delegate"] and not allowed["may_delegate"]:
        raise TeamIdentityRefused(name, f"would grant delegation to {actual['name']!r}")
    if (allowed.get("kind"), allowed.get("digest")) != (
        actual["kind"],
        actual["digest"],
    ):
        # The definition IS here; it is just not the one the record attested
        # (including one appearing where the record attested none). Saying
        # "unavailable" sent readers looking for a missing file.
        raise TeamIdentityRefused(
            name,
            # i18n: ignore technical diagnostic carried verbatim by the relay protocol
            f"role definition {actual['name']!r} differs from the recorded one; "
            "re-attach via /team",
        )


def _role_digest(profile: AgentProfile | None, prompt: str) -> str:
    """The revision a role's record attests. ONE rule, shared by the snapshot
    (``_role_authority``) and the restore-time effort substitution
    (``_with_recorded_efforts``), so the two can never hash differently."""
    from local_operator.network.definitions import digest_of

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
    return digest_of({"profile": profile_data, "prompt": prompt})


def _role_authority(
    team_name: str, role: str, registry: Any, *, delegated: bool = False
) -> tuple[dict[str, Any], AgentProfile | None, str]:
    from local_operator.agent_profiles import resolve_profile_or_specialist_identity
    from local_operator.harness.subagent import is_readonly_shell_role

    try:
        kind, profile, prompt, _, author = resolve_profile_or_specialist_identity(
            role, registry=registry, strict=True
        )
    except (OSError, ValueError, KeyError) as error:
        raise TeamIdentityRefused(team_name) from error
    if kind is None and delegated:
        # A roster may name a member that never had a profile: the teams guide
        # says such a team "still launches", hub sync reports it as a
        # ``missing-role`` warning, and ``_resolve_role`` launches it as a full
        # (unprofiled) child. Refusing it here refused real teams. Record it as
        # a PLAIN role instead, so the ceiling is exactly that full child, and
        # let ``_ceiling`` tell "the record attested a definition this device
        # lacks" (a wait) apart from "it never had one" (this record).
        return (
            {
                "name": role,
                "kind": PLAIN_ROLE,
                "id": "",
                "digest": "",
                "pins": [],
                "unrestricted_tools": True,
                # Without a profile the launch inherits delegation from the
                # parent's own tools (``may_delegate`` in run_subagent), so the
                # honest ceiling is "may".
                "may_delegate": True,
                "readonly_shell": is_readonly_shell_role(role, None),
            },
            None,
            "",
        )
    if kind is None:
        raise _DefinitionNotLoaded(team_name, f"manager definition {role!r}")
    return (
        {
            "name": role,
            "kind": kind,
            "id": author or "",
            "digest": _role_digest(profile, prompt),
            "pins": sorted(set(profile.tools or ())) if profile is not None else [],
            "unrestricted_tools": delegated and (profile is None or not profile.tools),
            "may_delegate": profile.may_delegate if profile is not None else kind == "specialist",
            "readonly_shell": is_readonly_shell_role(role, profile.name if profile else None),
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
    #: Casefolded ``team:`` slots that named no team when the roster was
    #: snapshotted. Recorded (not refused) because a roster may legitimately
    #: name a team that does not exist -- the org chart draws it as an
    #: unresolved ghost and a launch of it is a ``TeamLaunchError`` -- while a
    #: record that attested a REAL team keeps the definitions-sync wait.
    absent_teams: frozenset[str] = frozenset()
    #: Whether THIS device's operator chose the binding with a local ``/team``
    #: (directly, or by restoring a journal that one wrote). False for a mesh
    #: arrival/replica and for any snapshot not made by
    #: :func:`bind_runtime_authority`, so the default is the closed one: see
    #: :func:`live_member_roster` for what it gates.
    explicit_local: bool = False

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
                }
                | {name: dict(_ABSENT_TEAM) for name in self.absent_teams},
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
    absent: set[str] = set()
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
            check_size(len(resolved) + len(roles) + len(absent))  # +new role, -root team
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
                raise TeamIdentityRefused(team.name, f"delegated manager of {key!r} is unavailable")
            record_role(current.manager)
        height = 1
        for slot in current.members:
            if slot.kind == "agent":
                if slot.role != "task":
                    record_role(slot.role)
                continue
            child_key = slot.role.casefold()
            if child_key in absent:
                continue
            if child_key not in resolved:
                check_size(len(resolved) + len(roles) + len(absent))  # +new team, -root
                nested = _resolve(teams, slot.role) if teams is not None else None
                if nested is None:
                    # Nothing to descend into: no roles or depth to grant.
                    absent.add(child_key)
                    continue
                resolved[child_key] = nested.model_copy(deep=True)
            height = max(height, 1 + visit(resolved[child_key], (*ancestors, key)))
        heights[key] = height
        return height

    visit(team, ())
    return TeamAuthority(team, resolved, roles, manager, max_depth, frozenset(absent))


def _compare(
    expected: Any, actual: dict[str, Any], *, local: bool, exact_revision: bool = False
) -> None:
    name = actual["name"]
    if not isinstance(expected, dict) or not isinstance(expected.get("delegation"), dict):
        raise TeamIdentityRefused(name, "recorded authority is unavailable; re-attach via /team")
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
            if (
                key == "teams"
                and candidate == _ABSENT_TEAM
                and isinstance(entry, dict)
                and entry != _ABSENT_TEAM
            ):
                # Same split as a plain role: the record attested a real team
                # this device does not hold yet, so wait for definitions sync.
                raise _DefinitionNotLoaded(name, f"delegated team {slot!r}")
            if not isinstance(entry, dict) or (local and entry.get("id") != candidate["id"]):
                raise TeamIdentityRefused(name)
            if key == "roles":
                _ceiling(name, entry, candidate)
            elif entry.get(revision) != candidate[revision]:
                raise TeamIdentityRefused(name, f"team definition {slot!r} does not match")


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
        and row.get("digest") == digest_of(_team_row_from_bundle(_team_row_from_team(team)))
    ):
        return str(row.get("origin") or "")
    return ""


def _snapshot_at_root(root: Path, team: Team, teams: TeamRegistry, expected: Any) -> TeamAuthority:
    from local_operator.agents import AgentRegistry

    depth, readonly_shell = _recorded_limits(expected, root, team.manager)
    return _snapshot(
        team, teams, AgentRegistry(root), max_depth=depth, readonly_shell=readonly_shell
    )


def source_offer(root: Path, directory: Path, *, supported: bool) -> dict[str, Any] | None:
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
            name = str(expected.get("name") or "?") if isinstance(expected, dict) else "?"
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
    bind_runtime_authority(directory, team, registry, AgentRegistry(root), restoring=False)
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
    if restoring:
        snapshot = _with_recorded_efforts(_with_recorded_tiers(snapshot, expected), expected)
    record = snapshot.record()
    if restoring:
        _compare(expected, record, local=True)
    # ``isinstance``: a restore only reaches here after _compare accepted
    # ``expected`` as a record; anything else must never read as an explicit
    # local choice.
    explicit_local = not restoring or (
        isinstance(expected, dict) and expected.get("explicit_local") is True
    )
    snapshot = replace(snapshot, explicit_local=explicit_local)
    if directory is not None:
        # Tightening must survive reopen and the next hop. In particular, an
        # intermediate owner's reopened brief is not pinned to an old digest.
        _write_local_binding(
            directory,
            {
                **record,
                "explicit_local": explicit_local,
                MEMBER_TIERS_KEY: _member_tiers(snapshot),
                ROLE_EFFORTS_KEY: _role_efforts(snapshot),
            },
        )
    return snapshot


# ---------------------------------------------------------------------------
# Restore applies the same split: the RECORDED tiers are the snapshot's
# ---------------------------------------------------------------------------
#
# WHY. A warm bound session keeps its snapshot in memory, so a tier-only roster
# edit reaches its launches through ``live_member_roster`` (the split below).
# A STORED session has no snapshot in memory: ``restore_bound`` re-snapshots
# the live row, and the live row's edited ``tier``/``effort`` changed the
# ``structure`` digest the recorded authority is compared on, so the reopen
# refused ("team definition ... does not match") and the conversation could not
# send -- after an edit the warm path explicitly allows. Main had no snapshot
# to refuse against, so this refusal was a regression, not a stricter rule.
#
# HOW. The binding also journals each snapshotted row's member
# ``[tier, effort]`` pairs (``MEMBER_TIERS_KEY``). On restore those recorded
# values are put back onto the freshly read rows BEFORE the record is built,
# so:
#
# * the authority comparison (``_compare``) is untouched and still runs on the
#   full structure digest: a role, slot, kind, manager, saver or count edit
#   changes bytes that substitution never touches, so it still refuses with the
#   existing "re-attach via /team" text;
# * the recorded pairs are authenticated by that same digest -- a pair that is
#   not the one the binding attested cannot reproduce it, so a tampered or
#   stale journal refuses rather than granting a tier;
# * the restored snapshot carries the ATTESTED tiers, so the edited tier then
#   reaches a launch only through ``live_member_roster`` -- the provider
#   ceiling, the mesh-replica rule and the ``not applied`` receipt, exactly as
#   on a warm resume -- instead of silently becoming the new authority.
#
# A binding without the key (written before it existed, or a mesh arrival,
# which records the destination's own snapshot) gets no substitution: the live
# row is compared as-is, i.e. the pre-existing behaviour.

#: Binding key for the per-row ``[tier, effort]`` pairs the snapshot attested.
#: Local journal only: ``record()`` (what travels on a move) does not carry it.
MEMBER_TIERS_KEY = "member_tiers"


def _member_tiers(authority: TeamAuthority) -> dict[str, list[list[str | None]]]:
    return {
        key: [[member.tier, member.effort] for member in team.members]
        for key, team in authority.teams.items()
    }


def _with_recorded_tiers(authority: TeamAuthority, expected: Any) -> TeamAuthority:
    """``authority`` with each row's member tier/effort replaced by the recorded
    pairs. Only those two fields are ever written; a row whose recorded pairs are
    missing or malformed (wrong count, non-string values) is left as read, so
    the comparison decides it on the live bytes. See the section comment above.
    """
    recorded = expected.get(MEMBER_TIERS_KEY) if isinstance(expected, dict) else None
    if not isinstance(recorded, dict):
        return authority
    teams: dict[str, Team] = {}
    for key, team in authority.teams.items():
        pairs = recorded.get(key)
        if (
            not isinstance(pairs, list)
            or len(pairs) != len(team.members)
            or not all(
                isinstance(pair, list)
                and len(pair) == 2
                and all(value is None or isinstance(value, str) for value in pair)
                for pair in pairs
            )
        ):
            teams[key] = team
            continue
        members = [
            member.model_copy(update={"tier": tier, "effort": effort})
            for member, (tier, effort) in zip(team.members, pairs, strict=True)
        ]
        teams[key] = team.model_copy(update={"members": members})
    root = teams[authority.team.name.casefold()]
    return replace(authority, team=root, teams=teams)


# The SAME split for a role's own ``effort:`` tag (the third tier layer: launch
# tier -> roster slot -> role tag). The tag is part of the role's profile, so
# it is inside the role ``digest`` the record attests, and an edited tag made
# the reopen refuse ("role definition ... differs from the recorded one") just
# as an edited slot tier did. The fix mirrors ``member_tiers``: the binding
# journals each role's attested tag; restore puts it back on the live profile
# and re-derives that role's digest with ``_role_digest`` (the one rule
# ``_role_authority`` uses) before the unchanged ``_compare``/``_ceiling``.
#
# ONLY ``effort`` IS SUBSTITUTED, AND THAT IS THE BOUNDARY. ``effort`` is a
# live preference -- reasoning depth on an already-chosen model -- whose
# provider consequence ``live_role_tier`` then gates (same provider taken;
# crossing keeps the attested tag with a ``not applied`` receipt). A role's
# ``model:`` and ``hosting:`` ARE provider authority: an edit to either still
# changes the digest and still refuses. That refusal is correct behaviour, not
# a gap -- do not "complete" this by substituting them too. Pins, delegation
# and shell confinement are judged by ``_ceiling`` on the record, untouched.
#
# As with the slot pairs, a tag that is not the attested one cannot reproduce
# the attested digest, so a tampered or stale journal refuses; a binding
# without the key gets no substitution (the pre-existing behaviour).

#: Binding key for each role's attested ``effort:`` tag (roles and manager).
#: Local journal only, like :data:`MEMBER_TIERS_KEY`.
ROLE_EFFORTS_KEY = "role_efforts"


def _role_efforts(authority: TeamAuthority) -> dict[str, str | None]:
    entries = [
        *authority.roles.items(),
        (str(authority.manager[0].get("name", "")), authority.manager),
    ]
    return {name: data[1].effort for name, data in entries if data[1] is not None}


def _with_recorded_effort(
    entry: tuple[dict[str, Any], AgentProfile | None, str], recorded: dict[Any, Any], name: str
) -> tuple[dict[str, Any], AgentProfile | None, str]:
    meta, profile, prompt = entry
    if profile is None or name not in recorded:
        return entry
    effort = recorded[name]
    if effort is not None and not isinstance(effort, str):
        return entry  # malformed: compare the live bytes, which then refuse
    if effort == profile.effort:
        return entry
    attested = replace(profile, effort=effort)
    return {**meta, "digest": _role_digest(attested, prompt)}, attested, prompt


def _with_recorded_efforts(authority: TeamAuthority, expected: Any) -> TeamAuthority:
    """``authority`` with each role's (and the manager's) ``effort`` replaced by
    the recorded tag and its digest re-derived. See the section comment above."""
    recorded = expected.get(ROLE_EFFORTS_KEY) if isinstance(expected, dict) else None
    if not isinstance(recorded, dict):
        return authority
    roles = {
        name: _with_recorded_effort(entry, recorded, name)
        for name, entry in authority.roles.items()
    }
    manager = _with_recorded_effort(
        authority.manager, recorded, str(authority.manager[0].get("name", ""))
    )
    return replace(authority, roles=roles, manager=manager)


# ---------------------------------------------------------------------------
# The split: tool authority is snapshotted, the model tier is read live
# ---------------------------------------------------------------------------
#
# WHY. A bound session resolves every roster read against the snapshot taken at
# ``/team`` attach (``TeamAuthority.teams``), because that snapshot IS the
# authority: which slots exist, which roles they run as, and what tools those
# roles hold. Freezing the member ``tier``/``effort`` along with it broke the
# documented promise (teams guide; ``_reread_roster``) that "edit the override,
# then the next launch -- or resume -- uses it". So the two halves are split:
#
# * Tool authority (roster shape, manager, roles, pins, delegation, depth) stays
#   the snapshot. Nothing below can add, remove or re-role a slot.
# * ``tier``/``effort`` on an EXISTING slot are re-read from the registry.
#
# CONSTRAINTS that keep the live read from becoming a back door:
#
# 1. Whole-row gate. The live row is used only when it equals the snapshot in
#    EVERY field once member ``tier``/``effort`` are set aside (id, name,
#    manager, members and their order/kind/count/saver, briefs, labels...).
#    Any other difference means the row was edited in ways that need a fresh
#    ``/team``, so the snapshot is used whole -- a live read can never smuggle
#    in a different authority alongside a tier change.
# 2. Provider ceiling. A tier names a model, and a model names a PROVIDER (the
#    selector's ``provider`` token, which is also the account it routes
#    through), i.e. where the conversation's data goes. The binding authorized
#    the provider the snapshotted tier resolved to. A live tier is taken only
#    when it resolves to that same provider; otherwise the snapshotted tier (and
#    effort) are kept for that slot and the reason is returned. "Cannot tell"
#    (no tier on either side, the inherit sentinel, an unconfigured or malformed
#    tier, an unreadable config) is treated as crossing: fail closed. An
#    effort-only edit cannot change the provider and is always taken.
# 3. Mesh replicas. A binding that no local ``/team`` produced (a mesh arrival,
#    a synced replica: ``explicit_local`` False) keeps the snapshot's tiers
#    until the operator attaches locally: a synced tier edit is somebody else's
#    choice of provider, and therefore of where this device's data goes.
#
# Every gating decision is made against the snapshot; the live row only ever
# supplies the two fields, and only after those checks admit them.

#: The only member fields a live re-read may supply.
_LIVE_MEMBER_FIELDS = frozenset({"tier", "effort"})


def _authority_shape(team: Team) -> dict[str, Any]:
    """Everything in a team row except the fields a live read may supply."""
    data = team.model_dump(mode="json")
    for member in data.get("members") or ():
        for field in _LIVE_MEMBER_FIELDS:
            member.pop(field, None)
    return data


def _tier_provider(tier: str | None, targets: Any) -> str | None:
    """The provider token ``tier`` routes to, or ``None`` when it cannot be told."""
    from local_operator.harness.subagent import is_inherit_tier_sentinel

    if not tier:
        return None
    raw = targets.get(tier, (None, None))[0]
    if not isinstance(raw, str) or is_inherit_tier_sentinel(raw):
        return None
    provider, _, model_id = raw.strip().partition("/")
    return provider if provider and model_id else None


def live_member_roster(
    authority: TeamAuthority, snapshot: Team | None, live: Team | None
) -> tuple[Team | None, list[str]]:
    """The roster a bound launch reads member overrides from, plus why any
    live edit was held back. See the section comment above for the rules.

    Returns ``snapshot`` unchanged unless every rule admits a live field.
    """
    if snapshot is None or live is None:
        return snapshot, []
    if _authority_shape(live) != _authority_shape(snapshot):
        # Not a tier-only edit: the row now says something the binding never
        # authorized. Nothing is taken; the remedy is the usual re-attach.
        # Logged (not returned as a reason) when a tier/effort edit is among
        # what was held back, so an operator reading the log can tell "my tier
        # edit was ignored because the row changed in other ways too" from "no
        # edit was seen". Slots are paired by (role, kind), first match, only to
        # word the log line: the gate itself never pairs anything.
        before = {(m.role.casefold(), m.kind): (m.tier, m.effort) for m in snapshot.members[::-1]}
        held = sorted(
            m.role
            for m in live.members
            if (m.role.casefold(), m.kind) in before
            and before[(m.role.casefold(), m.kind)] != (m.tier, m.effort)
        )
        if held:
            logger.warning(
                # i18n: ignore technical diagnostic logged for the operator, English-only
                "team %r: tier/effort edit on slot(s) %s not applied; the registry row also "
                "changed beyond tier/effort since /team attached it, so the attached roster "
                "is used whole; re-attach via /team to take the edit",
                snapshot.name,
                ", ".join(repr(role) for role in held),
            )
        return snapshot, []
    edited = [
        (old, new)
        for old, new in zip(snapshot.members, live.members, strict=True)
        if (old.tier, old.effort) != (new.tier, new.effort)
    ]
    if not edited:
        return snapshot, []
    if not authority.explicit_local:
        return snapshot, [
            f"team {snapshot.name!r} slot {new.role!r}: kept the attached tier "
            f"{old.tier or 'none'!r}; a synced tier edit ({new.tier or 'none'!r}) on a "
            "mesh replica is a choice of provider and waits for a local /team"
            for old, new in edited
        ]
    from local_operator.harness.subagent import read_effort_tier_targets

    targets: Any = None
    reasons: list[str] = []
    accepted: dict[int, Any] = {}
    for index, (old, new) in enumerate(zip(snapshot.members, live.members, strict=True)):
        if (old.tier, old.effort) == (new.tier, new.effort):
            continue
        if old.tier == new.tier:
            accepted[index] = old.model_copy(update={"effort": new.effort})
            continue
        if targets is None:
            try:
                targets = read_effort_tier_targets()
            except Exception as error:  # noqa: BLE001 — unreadable config fails closed
                targets = {}
                logger.debug("tier map unreadable for a live tier check: %s", error)
        before = _tier_provider(old.tier, targets)
        after = _tier_provider(new.tier, targets)
        if before is not None and before == after:
            accepted[index] = old.model_copy(update={"tier": new.tier, "effort": new.effort})
            continue
        reasons.append(
            # i18n: ignore technical diagnostic logged for the operator, English-only
            f"team {snapshot.name!r} slot {new.role!r}: kept the attached tier "
            f"{old.tier or 'none'!r} (provider {before or 'unresolved'}); the edited tier "
            f"{new.tier or 'none'!r} routes to provider {after or 'unresolved'}, which "
            "this binding did not authorize; re-attach via /team to use it"
        )
    if not accepted:
        return snapshot, reasons
    members = [accepted.get(index, member) for index, member in enumerate(snapshot.members)]
    return snapshot.model_copy(update={"members": members}), reasons


def _snapshot_role_tier(authority: TeamAuthority, role: str) -> str | None:
    """The ``effort:`` tag ``role``'s profile carried when ``/team`` attached.

    ``None`` both for a role whose snapshotted profile pins no tier (or that had
    no profile: a plain role) and for a role outside the bound graph. The second
    is deliberately not told apart: a bound launch of a role outside the graph is
    refused before any model is resolved (``_require_roster_slot``), so only
    non-launch reads (naming, display) get here, and "no tier was attested" is
    the fail-closed answer for them too.
    """
    entries = [
        *authority.roles.items(),
        (str(authority.manager[0].get("name", "")), authority.manager),
    ]
    # Exact spelling first, then casefolded, the order the launch resolver uses.
    match = next((data for name, data in entries if name == role), None) or next(
        (data for name, data in entries if str(name).casefold() == role.casefold()), None
    )
    effort = getattr(match[1], "effort", None) if match is not None else None
    return str(effort) if effort else None


def live_role_tier(
    authority: TeamAuthority, role: str, live: str | None, targets: Any
) -> tuple[str | None, str]:
    """The role ``effort:`` tag a bound launch may use, plus why a live edit was
    held back (``""`` when nothing was).

    WHY. The role's own ``effort:`` tag is the third tier layer (launch tier ->
    roster slot -> role tag) and, like the slot ``tier``, it resolves to a model
    and the model to a PROVIDER. Reading it live from the agent registry let a
    role-pin edit move a bound child to a provider/account the binding never
    authorized, around the ceiling :func:`live_member_roster` enforces for the
    slot. So the snapshot's tag is the authority, and a live edit passes through
    the SAME three rules (section comment above):

    * mesh replica (``explicit_local`` False): the snapshotted tag stands;
    * provider ceiling: taken only when both tags resolve to the same provider
      in ``targets``; "cannot tell" -- no tag on either side, the inherit
      sentinel ("the launching session's model" is not a provider this check
      can compare), unconfigured or malformed -- counts as crossing;
    * the decision is made against the snapshot; ``live`` only supplies a value.

    ``targets`` is ``read_effort_tier_targets()``'s mapping; the caller reads it
    (and fails closed on an unreadable config by passing ``{}``).
    """
    snapshot = _snapshot_role_tier(authority, role)
    if (live or None) == snapshot:
        return snapshot, ""
    if not authority.explicit_local:
        return snapshot, (
            # i18n: ignore technical diagnostic surfaced on the launch receipt, English-only
            f"role {role!r}: kept the attached effort tag {snapshot or 'none'!r}; a synced "
            f"edit ({live or 'none'!r}) on a mesh replica is a choice of provider and waits "
            "for a local /team"
        )
    before = _tier_provider(snapshot, targets)
    after = _tier_provider(live, targets)
    if before is not None and before == after:
        return live, ""
    return snapshot, (
        # i18n: ignore technical diagnostic surfaced on the launch receipt, English-only
        f"role {role!r}: kept the attached effort tag {snapshot or 'none'!r} (provider "
        f"{before or 'unresolved'}); the edited tag {live or 'none'!r} routes to provider "
        f"{after or 'unresolved'}, which this binding did not authorize; re-attach via "
        "/team to use it"
    )
