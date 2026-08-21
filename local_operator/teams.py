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
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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


class TeamMember(BaseModel):
    """One roster slot: a named agent/role, possibly more than one copy."""

    role: str = Field(..., description="Role or specialist agent name.")
    count: int = Field(1, ge=1, le=16, description="How many of this role to run.")

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


class Team(BaseModel):
    """A durable team: manager + members + layered instruction briefs."""

    id: str
    name: str
    created_date: datetime
    description: str = ""
    manager: str = "manager"
    members: list[TeamMember] = Field(default_factory=list)
    instructions: str = ""
    project: str = ""

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
        """One scannable line per slot, manager first."""
        lines = [f"- manager: {self.manager} (you, when this team is invoked)"]
        for member in self.members:
            suffix = f" x{member.count}" if member.count > 1 else ""
            lines.append(f"- {member.role}{suffix}")
        return lines

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
        self.teams_dir.mkdir(parents=True, exist_ok=True)
        self._teams: dict[str, Team] = {}
        self._last_refresh_time = 0.0
        self._refresh_interval = refresh_interval
        self._load()

    def _load(self) -> None:
        loaded: dict[str, Team] = {}
        try:
            children = list(self.teams_dir.iterdir())
        except OSError:
            self._teams = {}
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
                data["instructions"] = _read_optional(child / "instructions.md")
                data["project"] = _read_optional(child / "project.md")
                team = Team.model_validate(data)
                loaded[team.id] = team
            except Exception as exc:  # noqa: BLE001 — one bad file must not hide the rest
                logger.warning("invalid team metadata in %s: %s", child.name, exc)
        self._teams = loaded
        self._last_refresh_time = time.time()

    def _refresh_if_needed(self) -> None:
        if time.time() - self._last_refresh_time > self._refresh_interval:
            self._load()

    def list_teams(self) -> list[Team]:
        self._refresh_if_needed()
        return sorted(self._teams.values(), key=lambda team: team.name.lower())

    def get_team(self, team_id: str) -> Team:
        self._refresh_if_needed()
        team = self._teams.get(team_id)
        if team is None:
            raise KeyError(f"Team with id {team_id} not found")
        return team

    def get_team_by_name(self, name: str) -> Team | None:
        self._refresh_if_needed()
        key = (name or "").strip().casefold()
        if not key:
            return None
        for team in self._teams.values():
            if team.name.casefold() == key:
                return team
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
            instructions=_bounded(fields.instructions or ""),
            project=_bounded(fields.project or ""),
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
        if "members" in updates and updates["members"] is not None:
            current.members = list(updates["members"])
        if "instructions" in updates and updates["instructions"] is not None:
            current.instructions = _bounded(updates["instructions"])
        if "project" in updates and updates["project"] is not None:
            current.project = _bounded(updates["project"])
        return self.save_team(current)

    def save_team(self, team: Team) -> Team:
        team_dir = self.teams_dir / team.id
        team_dir.mkdir(parents=True, exist_ok=True)
        payload = team.model_dump(mode="json", exclude={"instructions", "project"})
        try:
            with (team_dir / "team.yml").open("w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, default_flow_style=False, sort_keys=False)
            (team_dir / "instructions.md").write_text(team.instructions, encoding="utf-8")
            (team_dir / "project.md").write_text(team.project, encoding="utf-8")
        except Exception as exc:
            raise Exception(f"Failed to save team metadata: {exc}") from exc
        self._teams[team.id] = team
        return team

    def delete_team(self, team_id: str) -> None:
        if team_id not in self._teams:
            # Refresh once in case another process wrote it.
            self._load()
        if team_id not in self._teams:
            raise KeyError(f"Team with id {team_id} not found")
        self._teams.pop(team_id)
        team_dir = self.teams_dir / team_id
        if team_dir.exists():
            import shutil

            shutil.rmtree(team_dir)


def parse_members(raw: Iterable[str] | None) -> list[TeamMember]:
    """Parse ``coder`` / ``coder:2`` tokens into roster slots.

    Two members of the same role collapse into one slot with a summed count
    so a caller can pass ``--member coder --member coder`` or ``coder:2``.
    """
    slots: dict[str, int] = {}
    order: list[str] = []
    for token in raw or ():
        text = (token or "").strip()
        if not text:
            continue
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
        if role not in slots:
            order.append(role)
            slots[role] = 0
        slots[role] += count
    return [TeamMember(role=role, count=slots[role]) for role in order]


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
