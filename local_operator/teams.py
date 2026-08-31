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

import errno
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Literal

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

#: Persistence is normally a few local-file writes. A bounded retry keeps a
#: wedged peer from parking a synchronous tool call forever while still making
#: ordinary concurrent registry mutations serialize rather than fail spuriously.
_TEAM_LOCK_TIMEOUT_S = 10.0
_TEAM_LOCK_RETRY_S = 0.01

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


class Team(BaseModel):
    """A durable team: manager + members + layered instruction briefs.

    ``instructions`` and ``project`` are plain strings with NO marker for
    whether they have been read: an unloaded brief and an intentionally
    empty one are indistinguishable ON THE MODEL, by design. The loaded/
    unloaded distinction is REGISTRY-owned (``TeamRegistry._briefs_loaded``),
    because it is a fact about which files THIS registry has read — not a
    property of the value, and not something that can survive a
    ``model_dump``/``model_validate`` round trip through a tool or transport
    (review round 2, R2-1: a str-subclass sentinel serialized as ``""`` and
    revalidated as a plain ``""``, so a transported metadata-only team read
    as a deliberate clear and ``save_team`` truncated the briefs). The
    consequence is a documented constraint on ``save_team``: it cannot trust
    a transported object's empty briefs and preserves the on-disk files for
    any team id this registry has not loaded. Explicit brief writes go
    through :meth:`TeamRegistry.update_team` (or ``create_team``), which
    hydrate first and apply the caller's fields — including ``""`` clears —
    onto a known-loaded object.
    """

    id: str
    name: str
    created_date: datetime
    description: str = ""
    manager: str = "manager"
    members: list[TeamMember] = Field(default_factory=list)
    # Both briefs default to "" and never carry a marker: see the class
    # docstring and ``save_team`` for why the loaded state lives in the
    # registry instead of the value. ``save_team``'s dump excludes both
    # fields from ``team.yml`` either way.
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


def _try_lock_exclusive(fd: int) -> bool:
    """Take one non-blocking exclusive lock attempt on ``fd``."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):
                return False
            raise

    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            return False
        raise


def _unlock(fd: int) -> None:
    """Release a lock acquired by :func:`_try_lock_exclusive`."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one complete text file without exposing a truncated target."""
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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

    def _hydrate_briefs_for_save(self, team: Team) -> Team:
        """Make ``team`` safe to persist: its briefs must be REAL values.

        R2-1: the loaded/unloaded distinction is registry-owned
        (``_briefs_loaded``), so a ``Team`` that reached this registry from
        ANYWHERE but this registry's own ``get_team*`` — a ``list_teams``
        metadata row, or a model dumped to JSON/Python and revalidated by a
        tool or transport — carries ``""`` briefs that mean "never read",
        not "deliberately emptied". Writing those verbatim truncated both
        brief files. This method is the single choke point that resolves the
        ambiguity the ONLY way that cannot lose data: it reads the brief
        files this registry has on disk and merges them into the object,
        then records the id as loaded so the write persists exactly what was
        on disk.

        A team whose briefs this registry HAS loaded (``get_team*`` result,
        ``create_team``/``update_team`` output, or a previous save) keeps its
        in-memory values untouched: the caller is editing briefs it actually
        read, so its strings are authoritative and are written as-is. An
        explicit clear is authored through :meth:`update_team`, which hydrates
        first and then applies ``instructions=""`` onto the loaded object —
        reaching this method already loaded, so the empty string persists.

        Reads are best-effort like ``_load_briefs``: an unreadable file logs
        and saves empty rather than failing the whole metadata save, matching
        how ``_load`` treats bad YAML.

        A team id this registry has NEVER SEEN (not in ``_teams``, no brief
        files on disk) is a CREATE: the caller-supplied briefs are the only
        briefs that will ever exist, so they are kept verbatim and the id is
        simply marked loaded. Without this guard the fresh, empty directory
        the save is about to make would be read back as two absent files and
        the caller's briefs would be replaced with ``""`` on the very first
        write — the exact truncation this method exists to prevent, reached
        from the other side.
        """
        canonical = self._teams.get(team.id)
        if team.id in self._briefs_loaded and team is canonical:
            # Loadedness authorizes only the exact canonical object this
            # registry hydrated. A model-dumped/revalidated COPY with the same
            # id is not evidence that its "" briefs are intentional; trusting
            # the id alone would reintroduce R2-1 whenever a registry hydrated
            # the original before receiving the transported copy.
            return team
        team_dir = self.teams_dir / team.id
        on_disk_instructions = _read_optional(team_dir / "instructions.md")
        on_disk_project = _read_optional(team_dir / "project.md")
        if (
            canonical is not None
            or (team_dir / "instructions.md").is_file()
            or (team_dir / "project.md").is_file()
        ):
            # An existing entry: the disk files are the authoritative briefs
            # for an object that never read them.
            team.instructions = on_disk_instructions
            team.project = on_disk_project
        self._briefs_loaded.add(team.id)
        return team

    def get_team(self, team_id: str) -> Team:
        self._refresh_if_needed()
        team = self._teams.get(team_id)
        if team is None:
            raise KeyError(f"Team with id {team_id} not found")
        return self._load_briefs(team)

    def _find_cached_team_by_name(self, name: str) -> Team | None:
        """Find metadata in the current snapshot without refresh or hydration.

        Mutation paths must validate collisions against the SAME snapshot as
        the object they mutate. Calling a public getter mid-mutation can refresh
        ``_teams`` and orphan an already-hydrated canonical object.
        """
        key = (name or "").strip().casefold()
        if not key:
            return None
        return next((team for team in self._teams.values() if team.name.casefold() == key), None)

    def get_team_by_name(self, name: str) -> Team | None:
        self._refresh_if_needed()
        team = self._find_cached_team_by_name(name)
        return self._load_briefs(team) if team is not None else None

    @contextmanager
    def _persistence_lock(self) -> Iterator[None]:
        """Serialize one registry mutation across processes.

        The sidecar lives at config level rather than inside ``teams/`` so it
        can never be mistaken for a team row. Construction remains read-only:
        the config directory and lock file appear only on the first mutation.

        Registry methods are synchronous local-filesystem calls already, so a
        short synchronous wait matches their API. As with the project's OAuth
        and process-ledger locks, each kernel attempt is NON-BLOCKING and the
        retry is bounded; a dead peer can never park the caller indefinitely.
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.config_dir / ".teams.lock", os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        deadline = time.monotonic() + _TEAM_LOCK_TIMEOUT_S
        try:
            while not acquired:
                acquired = _try_lock_exclusive(fd)
                if acquired:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for the teams registry lock")
                time.sleep(_TEAM_LOCK_RETRY_S)
            yield
        finally:
            if acquired:
                _unlock(fd)
            os.close(fd)

    def create_team(self, fields: TeamEditFields) -> Team:
        name = validate_team_name(fields.name or "")
        with self._persistence_lock():
            # The refresh is unconditional and occurs inside the same lock as
            # validation and publication. Interval-gated snapshots cannot prove
            # uniqueness when another process has written since our last read.
            self._load()
            team = Team(
                id=str(uuid.uuid4()),
                name=name,
                created_date=datetime.now(timezone.utc),
                description=(fields.description or "").strip(),
                manager=(fields.manager or "manager").strip() or "manager",
                members=list(fields.members or []),
                instructions=(
                    _bounded(fields.instructions) if fields.instructions is not None else ""
                ),
                project=_bounded(fields.project) if fields.project is not None else "",
            )
            return self._save_team_locked(team, briefs_authoritative=True)

    def update_team(self, team_id: str, fields: TeamEditFields) -> Team:
        with self._persistence_lock():
            # Refresh exactly once under the writer lock, then mutate and save
            # that canonical object without re-locking or replacing the cache.
            self._load()
            current = self._teams.get(team_id)
            if current is None:
                raise KeyError(f"Team with id {team_id} not found")
            self._load_briefs(current)

            updates = fields.model_dump(exclude_unset=True)
            if "name" in updates and updates["name"] is not None:
                new_name = validate_team_name(updates["name"])
                occupant = self._find_cached_team_by_name(new_name)
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
                # An explicit "" is a DELIBERATE clear. This path hydrated first,
                # so the empty string is authoritative rather than transported.
                current.instructions = _bounded(updates["instructions"])
            if "project" in updates and updates["project"] is not None:
                current.project = _bounded(updates["project"])
            return self._save_team_locked(current, briefs_authoritative=True)

    def save_team(self, team: Team) -> Team:
        """Write ``team`` to disk and adopt it as this registry's current row.

        CONSTRAINT (R2-1): the loaded/unloaded brief state is registry-owned,
        so this method cannot tell from the OBJECT whether its ``""`` briefs
        are "never read" or "deliberately emptied" — a model-dumped,
        revalidated team has lost the distinction by construction. It
        therefore resolves the ambiguity the safe way: for an id this
        registry has not loaded, the on-disk brief files are merged into the
        object before anything is written (``_hydrate_briefs_for_save``), so
        a metadata-only ``list_teams`` row — or a round-tripped copy of one —
        preserves both briefs while its metadata edits persist. A team this
        registry HAS loaded (a ``get_team*`` result the caller edited, or the
        output of ``create_team``/``update_team``) saves its briefs verbatim.
        To CLEAR a brief, use :meth:`update_team` with an explicit empty
        string: that path hydrates first, so the clear lands on a loaded
        object and persists.
        """
        # Loadedness authorizes only the canonical object this registry actually
        # hydrated. Preserve that fact across the mandatory disk refresh; a
        # transported copy with the same id remains untrusted by construction.
        briefs_authoritative = team.id in self._briefs_loaded and team is self._teams.get(team.id)
        with self._persistence_lock():
            self._load()
            return self._save_team_locked(team, briefs_authoritative=briefs_authoritative)

    def _save_team_locked(self, team: Team, *, briefs_authoritative: bool) -> Team:
        """Validate and publish ``team`` while ``_persistence_lock`` is held."""
        name_key = team.name.casefold()
        occupant = next(
            (
                stored
                for stored in self._teams.values()
                if stored.id != team.id and stored.name.casefold() == name_key
            ),
            None,
        )
        if occupant is not None:
            raise ValueError(f"Team with name {team.name} already exists")

        # Hydrate BEFORE creating the final directory. An untrusted transported
        # row carries empty brief strings that mean "not loaded", while create
        # and update explicitly mark their already-known values authoritative.
        if briefs_authoritative:
            self._briefs_loaded.add(team.id)
        else:
            self._hydrate_briefs_for_save(team)

        team_dir = self.teams_dir / team.id
        payload = team.model_dump(mode="json", exclude={"instructions", "project"})
        metadata = yaml.safe_dump(payload, default_flow_style=False, sort_keys=False)
        try:
            self.teams_dir.mkdir(parents=True, exist_ok=True)
            if team_dir.exists():
                # Each replacement is atomic for unlocked readers; the writer
                # lock keeps peer registries from interleaving the three-file set.
                _atomic_write_text(team_dir / "team.yml", metadata)
                _atomic_write_text(team_dir / "instructions.md", team.instructions)
                _atomic_write_text(team_dir / "project.md", team.project)
            else:
                # Publish a new id as one directory rename so a failed create
                # cannot leave a half-written team row visible on disk.
                staging = Path(tempfile.mkdtemp(prefix=f".{team.id}.", dir=self.teams_dir))
                try:
                    (staging / "team.yml").write_text(metadata, encoding="utf-8")
                    (staging / "instructions.md").write_text(team.instructions, encoding="utf-8")
                    (staging / "project.md").write_text(team.project, encoding="utf-8")
                    os.replace(staging, team_dir)
                except BaseException:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
        except Exception as exc:
            raise Exception(f"Failed to save team metadata: {exc}") from exc
        self._teams[team.id] = team
        self._briefs_loaded.add(team.id)
        return team

    def delete_team(self, team_id: str) -> None:
        with self._persistence_lock():
            # Delete participates in the same ordering as create/save so a
            # delete-recreate-stale-save sequence has one unambiguous winner.
            self._load()
            if team_id not in self._teams:
                raise KeyError(f"Team with id {team_id} not found")
            # Remove the on-disk copy FIRST: if rmtree fails, cache still agrees
            # with disk and the row remains visible after the exception.
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
