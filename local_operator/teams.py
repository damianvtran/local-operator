"""Teams: a named roster of reusable agents under one manager.

WHY THIS EXISTS
---------------

Roles and specialist agents are reusable building blocks — a ``coder`` or a
"User Dashboard Agent" should be writable once and usable in many groupings.
A Team is the grouping: a manager, a roster of members (role or specialist,
with counts), plus TWO instruction layers that do not belong on any one
agent because they describe the GROUP rather than the person:

1. **Collaboration** (``instructions.md``) — how this team works together:
   review order, who blocks a release, how the manager delegates.
2. **Project** (``project.md``) — the product or domain this instance of the
   team is responsible for. The same Feature Release roster can staff two
   products by swapping only this file.

The three layers a member actually sees, outermost last:

- the agent's own ``system_prompt.md`` (base behaviour, reusable)
- the team's collaboration brief (how we work)
- the team's project brief (what we are responsible for)

A manager session also gets a roster so it can ``task(agent=...)`` the right
people without the operator restating the org chart every turn.

STORAGE
-------

``<config_dir>/teams/<id>/``:

- ``team.yml`` — id, name, description, manager, members
- ``instructions.md`` — collaboration brief
- ``project.md`` — project / product brief

Members are referenced by NAME (a role or a specialist agent), never by
registry id, so a team survives an agent being deleted and recreated and so
the same ``coder`` row can sit on many teams. Counts let a team ask for two
coders without inventing a second profile.

The registry is NEVER enumerated into the prompt. ``list`` is the explicit
action that reveals names; the ``teams`` guide is what the model reads to
learn the concept.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

#: Cap on each team instruction file. Same bound as a role body: these ride
#: in front of a manager session and every member launch, so an unbounded
#: paste is an unbounded per-turn bill.
MAX_TEAM_INSTRUCTIONS_CHARS = 8_000

#: A team name is also a slash-command argument, so it cannot contain spaces
#: or slashes — ``/team feature-release ship it`` has to parse unambiguously.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: How deep an org (a team whose members are themselves teams) may nest before
#: the resolver stops descending. Eight levels is far past any real human org
#: and keeps the resolver, the tidy-tree layout, and the render all bounded: a
#: reference deeper than this is truncated with a visible "depth limit" node
#: rather than followed. It also backstops the cycle guard — even a
#: mis-detected cycle cannot run away past this. Lives here, in the MODEL layer,
#: because depth is a property of the data, not of any one widget that draws it.
MAX_ORG_DEPTH = 8


class TeamMember(BaseModel):
    """One roster slot: a named agent/role, or a nested TEAM, possibly repeated."""

    role: str = Field(
        ...,
        description="Role/specialist name, or team name when kind='team'.",
    )
    count: int = Field(
        default=1,
        ge=1,
        le=16,
        description="How many of this role to run.",
    )
    # NEW discriminator. Absent ("agent") in every existing ``team.yml``, so old
    # files load unchanged — the field defaults to "agent" and validation is a
    # no-op on a roster that never mentions it, which is the whole point of a
    # default rather than a required key (a required field would silently drop
    # every stored team through ``_load``'s except-and-skip).
    #
    # "team" marks this slot as a reference to ANOTHER team by name, turning a
    # flat roster into an "org" (a team of teams). The referenced name still
    # lives in ``role`` rather than a new field so that ``member_names()``,
    # ``member_count()``, and every existing reader keep working without
    # knowing nesting exists — a team slot simply reads as its team name. One
    # roster, one authored order: a separate ``subteams`` list would split the
    # roster into two lists the author has to keep mentally merged and would
    # force every reader to concatenate them.
    kind: Literal["agent", "team"] = Field(
        default="agent",
        description="'agent' (a role/specialist) or 'team' (a nested sub-team).",
    )

    @field_validator("role")
    @classmethod
    def _role_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("member role is required")
        return name


class TeamEditFields(BaseModel):
    """Partial update. ``None`` means leave the stored value alone."""

    name: str | None = None
    description: str | None = None
    manager: str | None = None
    members: list[TeamMember] | None = None
    instructions: str | None = None
    project: str | None = None


class _UnloadedBrief(str):
    """Module-private sentinel: a brief whose file has not been read yet.

    A ``str`` SUBCLASS so it flows through pydantic validation and every
    existing ``str`` consumer unchanged, and private (leading underscore,
    no re-export) so it can never leak into a tool schema, ``model_dump``
    output, or public API. Its single job is to make "the registry handed
    out this team without reading its briefs" a distinguishable state from
    "the brief is intentionally empty": both surface as ``''`` on the
    public field, and only the first must be PRESERVED by ``save_team``
    rather than written back as an empty file (review round 1, R1-1).
    ``save_team`` and ``_load_briefs`` are the only readers; everyone else
    sees an ordinary empty string via :attr:`Team.instructions`. It deliberately
    inherits ``str.__repr__`` too: even debug/public model representations show
    ``''``, never a sentinel label.
    """


class Team(BaseModel):
    """A durable team: manager + members + layered instruction briefs.

    ``instructions`` and ``project`` read as plain strings everywhere, but
    their DEFAULT is the :class:`_UnloadedBrief` sentinel rather than ``""``:
    a team constructed from ``team.yml`` alone (``list_teams`` and every
    picker refresh) has not paid the brief I/O yet, and a later
    ``save_team`` of that object must re-read the files instead of
    overwriting them with the empty strings it happens to carry. Constructing
    a ``Team`` with explicit ``instructions=""`` — as ``create_team`` and a
    deliberate ``update_team(..., TeamEditFields(instructions=""))`` do —
    keeps the empty string and therefore still clears the file on purpose.
    """

    id: str
    name: str
    created_date: datetime
    description: str = ""
    manager: str = "manager"
    members: list[TeamMember] = Field(default_factory=list)
    # Sentinel default, not "": see the class docstring. ``exclude`` in
    # ``save_team``'s dump already keeps both fields out of ``team.yml``;
    # the sentinel itself is a private str subclass that serializes as ""
    # should anyone ever dump these fields, so it cannot leak a marker into
    # YAML, JSON, or tool output.
    instructions: str = _UnloadedBrief()
    project: str = _UnloadedBrief()

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        name = (value or "").strip()
        if not _NAME_RE.match(name):
            raise ValueError(
                "team name must be 1-64 characters of letters, digits, "
                "dot, underscore or hyphen, and cannot start with a hyphen"
            )
        return name

    @field_validator("manager")
    @classmethod
    def _manager(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("manager is required")
        return name

    def roster_lines(self) -> list[str]:
        """One scannable line per slot, manager first.

        A nested-team slot is badged ``(team)`` so a reader (and ``team show``)
        can tell an org apart from a flat roster — a member named ``pod`` and a
        sub-team named ``pod`` would otherwise render identically.
        """
        lines = [f"- manager: {self.manager} (you, when this team is invoked)"]
        for member in self.members:
            suffix = f" x{member.count}" if member.count > 1 else ""
            badge = " (team)" if member.kind == "team" else ""
            lines.append(f"- {member.role}{badge}{suffix}")
        return lines

    def member_count(self) -> int:
        """Total member copies on the roster (counts summed, manager excluded).

        Distinct from ``len(members)``: a ``reviewer x2`` slot is two members
        in one slot, and a summary that reports it as one understates the
        team the user assembled.
        """

        return sum(member.count for member in self.members)

    def member_names(self) -> list[str]:
        """Role names on the roster, manager included, first occurrence winning."""
        names: list[str] = []
        for name in (self.manager, *(member.role for member in self.members)):
            if name not in names:
                names.append(name)
        return names

    def manager_preamble(self) -> str:
        """Standing brief stamped into a manager session's instructions.

        Empty briefs cost nothing: a team that has only a roster still names
        the roster, and a team with nothing at all yields an empty string so
        it does not tax a session that has not been briefed yet.
        """
        parts: list[str] = [f"[team: {self.name}]"]
        if self.description.strip():
            parts.append(self.description.strip())
        parts.append("You are the manager of this team. You coordinate; you do not implement.")
        parts.append(
            "Delegate with task(agent='<role>') using the roster below. "
            "Each member already carries this team's collaboration and project "
            "briefs — give them the TASK, not a restatement of the team. "
            "Spin up the counts the roster names; do not invent extra copies."
        )
        parts.append("Roster:\n" + "\n".join(self.roster_lines()))
        collab = self.instructions.strip()
        if collab:
            parts.append("Collaboration:\n" + collab)
        project = self.project.strip()
        if project:
            parts.append("Project:\n" + project)
        return "\n\n".join(parts) + "\n"

    def member_preamble(self, role: str) -> str:
        """Brief stamped in front of a member's one-shot prompt.

        Shorter than the manager brief: a member does not need to be told how
        to delegate, and the role's own preamble already said how to do the
        job. This is the GROUP context the role file must not carry, because
        the same role sits on many teams.
        """
        parts: list[str] = [
            f"[team: {self.name}]",
            f"You are {role} on this team. The manager is {self.manager}.",
            "Teammates:\n" + "\n".join(self.roster_lines()),
        ]
        collab = self.instructions.strip()
        if collab:
            parts.append("Collaboration:\n" + collab)
        project = self.project.strip()
        if project:
            parts.append("Project:\n" + project)
        return "\n\n".join(parts) + "\n\n"


def validate_team_name(name: str) -> str:
    """Return a stripped, legal team name or raise ``ValueError``."""
    return Team.model_validate(
        {
            "id": "x",
            "name": name,
            "created_date": datetime.now(timezone.utc),
            "manager": "manager",
        }
    ).name


class TeamRegistry:
    """On-disk registry of teams under ``<config_dir>/teams``."""

    def __init__(self, config_dir: Path, refresh_interval: float = 5.0) -> None:
        self.config_dir = Path(config_dir)
        self.teams_dir = self.config_dir / "teams"
        # No mkdir here: every interactive session constructs a registry, and
        # an unused feature must not litter the config dir. ``save_team``
        # creates the tree on first write, and ``_load`` treats a missing
        # directory as "no teams".
        self._teams: dict[str, Team] = {}
        # Picker/listing callers need only ``team.yml`` metadata. Briefs can be
        # 8k each and sit on the session's five-second refresh path, so remember
        # which teams paid that I/O only when an attach/show/preamble lookup asks.
        self._briefs_loaded: set[str] = set()
        self._last_refresh_time = 0.0
        self._refresh_interval = refresh_interval
        self._load()

    def _load(self) -> None:
        loaded: dict[str, Team] = {}
        try:
            children = list(self.teams_dir.iterdir())
        except OSError:
            self._teams = {}
            self._briefs_loaded = set()
            self._last_refresh_time = time.time()
            return
        for child in children:
            if not child.is_dir():
                continue
            path = child / "team.yml"
            if not path.is_file():
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                if not isinstance(data, dict):
                    continue
                team = Team.model_validate(data)
                loaded[team.id] = team
            except Exception as exc:  # noqa: BLE001 — one bad file must not hide the rest
                logger.warning("invalid team metadata in %s: %s", child.name, exc)
        self._teams = loaded
        # A refresh replaces every model with metadata-only instances. Keeping
        # an old loaded marker would return blank briefs from the replacement.
        self._briefs_loaded = set()
        self._last_refresh_time = time.time()

    def _refresh_if_needed(self) -> None:
        if time.time() - self._last_refresh_time > self._refresh_interval:
            self._load()

    def list_teams(self) -> list[Team]:
        self._refresh_if_needed()
        return sorted(self._teams.values(), key=lambda team: team.name.lower())

    def _load_briefs(self, team: Team) -> Team:
        """Populate a metadata-only team the first time a full lookup needs it."""
        if team.id in self._briefs_loaded:
            return team
        team_dir = self.teams_dir / team.id
        team.instructions = _read_optional(team_dir / "instructions.md")
        team.project = _read_optional(team_dir / "project.md")
        self._briefs_loaded.add(team.id)
        return team

    def _persist_brief(self, team: Team, filename: str, value: str) -> str:
        """The brief text to write for ``filename``, hydrating an unloaded one.

        R1-1: ``list_teams`` hands out teams whose briefs were never read (they
        are two 8k files on a five-second refresh path), and a caller that
        mutates such an object's METADATA and saves it must not lose the briefs
        it never touched. The sentinel default marks "not loaded"; here it is
        the one place that turns it back into the on-disk text so the write
        preserves the file verbatim. An explicit empty string — the only other
        value ``Team`` construction can carry once ``create_team`` and
        ``update_team`` normalize to ``""`` — is an INTENTIONAL clear and is
        written as-is. Reads are best-effort like ``_load_briefs``: an
        unreadable file logs and saves empty rather than failing the whole
        metadata save, matching how ``_load`` treats bad YAML.
        """
        if not isinstance(value, _UnloadedBrief):
            return value
        return _read_optional(self.teams_dir / team.id / filename)

    def get_team(self, team_id: str) -> Team:
        self._refresh_if_needed()
        team = self._teams.get(team_id)
        if team is None:
            raise KeyError(f"Team with id {team_id} not found")
        return self._load_briefs(team)

    def get_team_by_name(self, name: str) -> Team | None:
        self._refresh_if_needed()
        key = (name or "").strip().casefold()
        if not key:
            return None
        for team in self._teams.values():
            if team.name.casefold() == key:
                return self._load_briefs(team)
        return None

    def create_team(self, fields: TeamEditFields) -> Team:
        name = validate_team_name(fields.name or "")
        if self.get_team_by_name(name) is not None:
            raise ValueError(f"Team with name {name} already exists")
        team = Team(
            id=str(uuid.uuid4()),
            name=name,
            created_date=datetime.now(timezone.utc),
            description=(fields.description or "").strip(),
            manager=(fields.manager or "manager").strip() or "manager",
            members=list(fields.members or []),
            instructions=_bounded(fields.instructions) if fields.instructions is not None else "",
            project=_bounded(fields.project) if fields.project is not None else "",
        )
        return self.save_team(team)

    def update_team(self, team_id: str, fields: TeamEditFields) -> Team:
        current = self.get_team(team_id)
        updates = fields.model_dump(exclude_unset=True)
        if "name" in updates and updates["name"] is not None:
            new_name = validate_team_name(updates["name"])
            occupant = self.get_team_by_name(new_name)
            if occupant is not None and occupant.id != team_id:
                raise ValueError(f"Team with name {new_name} already exists")
            current.name = new_name
        if "description" in updates and updates["description"] is not None:
            current.description = updates["description"].strip()
        if "manager" in updates and updates["manager"] is not None:
            manager = updates["manager"].strip()
            if not manager:
                raise ValueError("manager is required")
            current.manager = manager
        if "members" in fields.model_fields_set and fields.members is not None:
            # ``model_dump`` recursively turns Pydantic children into dicts.
            # Keep the validated TeamMember objects: roster rendering and
            # orchestration call ``member.role`` / ``member.count`` immediately
            # after an update, before a reload can rehydrate them from YAML.
            current.members = list(fields.members)
        if "instructions" in updates and updates["instructions"] is not None:
            # An explicit "" is a DELIBERATE clear (R1-1's other edge): the
            # sentinel only ever means "never read", so normalizing to ""
            # here keeps update_team's save writing exactly what was asked.
            current.instructions = _bounded(updates["instructions"])
        if "project" in updates and updates["project"] is not None:
            current.project = _bounded(updates["project"])
        return self.save_team(current)

    def save_team(self, team: Team) -> Team:
        team_dir = self.teams_dir / team.id
        team_dir.mkdir(parents=True, exist_ok=True)
        payload = team.model_dump(mode="json", exclude={"instructions", "project"})
        # R1-1: hydrate before writing AND before caching. A metadata-only team
        # (straight from ``list_teams``) reaches here carrying sentinels; writing
        # them as text would truncate the briefs to empty files, and caching the
        # object with sentinels still set would hand the NEXT ``get_team`` an
        # unloaded-looking team it considers loaded. Both fixes are this pair of
        # assignments — the sentinel never survives a save.
        team.instructions = self._persist_brief(team, "instructions.md", team.instructions)
        team.project = self._persist_brief(team, "project.md", team.project)
        try:
            with (team_dir / "team.yml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
            (team_dir / "instructions.md").write_text(team.instructions, encoding="utf-8")
            (team_dir / "project.md").write_text(team.project, encoding="utf-8")
        except Exception as exc:
            raise Exception(f"Failed to save team metadata: {exc}") from exc
        self._teams[team.id] = team
        # The caller supplied the briefs that were just persisted; rereading the
        # same files on the next get would add I/O without recovering any state.
        self._briefs_loaded.add(team.id)
        return team

    def delete_team(self, team_id: str) -> None:
        if team_id not in self._teams:
            # Refresh once in case another process wrote it.
            self._load()
        if team_id not in self._teams:
            raise KeyError(f"Team with id {team_id} not found")
        # Remove the on-disk copy FIRST: if the rmtree fails (permissions, an
        # open handle) the cache must still agree with disk, so the row stays
        # and the error propagates. Popping before the rmtree left a deleted
        # row in the cache only when the filesystem said no.
        team_dir = self.teams_dir / team_id
        if team_dir.exists():
            shutil.rmtree(team_dir)
        self._teams.pop(team_id)
        self._briefs_loaded.discard(team_id)


def parse_members(raw: Iterable[str] | None) -> list[TeamMember]:
    """Parse ``coder`` / ``coder:2`` / ``team:pod`` / ``team:pod:2`` tokens.

    Two members of the same role collapse into one slot with a summed count
    so a caller can pass ``--member coder --member coder`` or ``coder:2``.

    A leading ``team:`` prefix marks the slot as a nested TEAM (an org): the
    name after the prefix is a team name, and ``team:pod:2`` is two independent
    copies of the ``pod`` sub-org. This is the ONE place the tool authors
    nesting; a bare token (no ``team:`` prefix) stays ``kind='agent'`` so the
    existing ``coder`` / ``coder:2`` grammar is untouched. Agent and team slots
    live in one keyed namespace here (``(kind, role)``) so a member ``pod`` and
    a sub-team ``pod`` do not collapse into each other.
    """
    slots: dict[tuple[Literal["agent", "team"], str], int] = {}
    order: list[tuple[Literal["agent", "team"], str]] = []
    for token in raw or ():
        text = (token or "").strip()
        if not text:
            continue
        kind: Literal["agent", "team"] = "agent"
        # ``team:`` is a case-insensitive prefix on the WHOLE token, stripped
        # before the ``role:count`` split so ``team:pod:2`` still parses its
        # count. A bare ``pod:2`` is unaffected — no prefix, stays an agent.
        if text[:5].lower() == "team:":
            kind = "team"
            text = text[5:].strip()
            if not text:
                raise ValueError(f"invalid team member {token!r}: no team name")
        role, sep, count_text = text.partition(":")
        role = role.strip()
        if not role:
            raise ValueError(f"invalid member {token!r}")
        count = 1
        if sep:
            try:
                count = int(count_text.strip())
            except ValueError as exc:
                raise ValueError(f"invalid member count in {token!r}") from exc
        if count < 1:
            raise ValueError(f"member count must be >= 1 in {token!r}")
        key = (kind, role)
        if key not in slots:
            order.append(key)
            slots[key] = 0
        slots[key] += count
    return [TeamMember(role=role, count=slots[(kind, role)], kind=kind) for (kind, role) in order]


def _bounded(text: str) -> str:
    body = text or ""
    if len(body) > MAX_TEAM_INSTRUCTIONS_CHARS:
        raise ValueError(
            f"team instructions exceed {MAX_TEAM_INSTRUCTIONS_CHARS} characters; "
            "they ride in front of every run of this team, so they must stay short."
        )
    return body


def _read_optional(path: Path) -> str:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        logger.warning("could not read %s", path)
    return ""
