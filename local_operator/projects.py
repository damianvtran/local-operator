"""Projects: a tracked workstream row, one JSON file per project.

WHY THIS EXISTS
---------------
Work regularly spans several sessions — parallel features, a migration, a
release train — and nothing durable in the harness names that stream or says
where it stands. A *project* is that row. It is deliberately **not** an agent,
not a schedule and not a second team: it is the operator's own metadata about a
workstream, stored beside the teams they authored. (A team's own ``project``
brief — ``teams/<id>/project.md`` — is a different thing that happens to share
the word; see ``guide://projects``.)

STORAGE
-------
``<config_dir>/projects/<id>.json`` — one small document per project, plus a
store-wide lock file ``<config_dir>/projects/.lock``. Writes are
read-modify-write under a bounded ``flock``, published by a temp-write +
``fsync`` + ``os.replace``, so a reader never sees a torn row and a crashed
writer leaves at most a temp file. Reads are lock-free: ``os.replace`` is
atomic, so a reader sees complete bytes of one revision. This is the deliberate
simplification of :mod:`local_operator.teams`: a project has no briefs, so it
needs no per-row directory or swap machinery — and the ``schema`` int in the
row is the hook if that ever changes.

THE LINK IS STORED ONE WAY. A linked session id lives in the project row's
``sessions`` list; the reverse ("which projects is this session in") is derived
by scanning loaded rows (:meth:`ProjectRegistry.projects_for_session`). A
linked session that no longer exists is MARked, never auto-removed: silent
mutation on a read path is the class of bug this store refuses.

THE WRITE GUARD. Reads are lenient (``extra="ignore"``), so an older build can
read a newer row. A build REFUSES TO MUTATE a row whose ``schema`` exceeds
:data:`PROJECT_SCHEMA` — that is the rule which makes any future additive
format change bump safely while an older build is still in the field.

THE VIEW IS ONE COMPOSITION. :func:`build_project_view` is the single reading
of "what state is each linked session in" (runtime records, durable session
files, and an optional in-process overlay), used by the tool, the desktop
route and the mobile daemon alike — no second derivation per surface.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from local_operator.procstate import O_BINARY

logger = logging.getLogger(__name__)

#: The row format this build writes. A row whose ``schema`` is HIGHER than this
#: is readable but not mutable (see the module docstring): it was written by a
#: newer local-operator, and this build cannot promise to preserve fields it
#: does not know about.
PROJECT_SCHEMA = 1

#: A project name is also a slash-command argument and an ``@project:<name>``
#: token, so it cannot contain spaces — the exact rule team names follow
#: (``teams._NAME_RE``).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: IDs are filesystem row addresses. ``uuid4().hex`` at write time; the shape
#: check mirrors ``validate_team_id`` so a transported or hand-edited row can
#: never point a path outside the store.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Session ids are the UI contract's shape (``src/shared/desktop-contract.ts``),
#: which is also the transcript directory name the harness generates.
_SESSION_ID_RE = re.compile(r"^[a-f0-9]{12}$")

#: ``q4``, ``payments-v2`` — short, lowercase, no spaces: a tag is a filter key.
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")

DESCRIPTION_MAX = 240
PROGRESS_MAX = 1000
TAGS_MAX = 8
SESSIONS_MAX = 64
MILESTONES_MAX = 20
MILESTONE_NAME_MAX = 80
ESTIMATE_MAX = 1000.0

#: How long a progress snippet stays "fresh". ONE constant, read by the
#: view/route payloads and by the completion-time check, so the stale badge and
#: the nudge can never disagree about a record: a work turn yielding within this
#: window of the last report is not asked to re-report; anything older is.
PROJECT_PROGRESS_STALE_S: float = 1800.0

#: Registry mutation lock budget: a bounded waiter, so a dead peer can never
#: park a tool call forever (the ``teams`` constants' shape).
_LOCK_TIMEOUT_S = 10.0
_LOCK_RETRY_S = 0.01

#: Cap on rows loaded from one store. Past this, the scan stops and warns: a
#: store with thousands of rows is a runaway writer, and a read path must stay
#: bounded.
_MAX_ROWS = 500

ProjectStatus = Literal["active", "paused", "done", "archived"]
EstimateUnit = Literal["points", "days"]


class ProjectRegistryLockTimeout(TimeoutError):
    """The bounded wait for the cross-process projects registry lock expired."""


class ProjectNameConflictError(ValueError):
    """A create or rename landed on a name that is already taken (case-insensitive).

    A ValueError subclass so every existing ``except ValueError`` arm keeps
    catching it — the tool renders the same sentence it always did — while the
    desktop route can map THIS subclass to its 409 without matching on prose:
    a duplicate name is a state conflict the client resolves differently from a
    malformed value (422).
    """


class ProjectSchemaGuardError(RuntimeError):
    """A mutation was refused because the row was written by a newer build.

    Reads stay lenient on purpose; only writes are guarded, so a future format
    change can ship without an older build silently rewriting a row into a
    shape it does not understand (see the module docstring).
    """


def validate_project_id(project_id: str) -> str:
    """Return an ID that is exactly one safe filesystem path segment."""
    candidate = project_id or ""
    if not _ID_RE.fullmatch(candidate) or candidate in {".", ".."}:
        raise ValueError(
            "project id must be 1-128 characters of letters, digits, dot, "
            "underscore or hyphen, start with a letter or digit, and contain no "
            "path separators"
        )
    return candidate


def validate_project_name(name: str) -> str:
    """Return a stripped, legal project name or raise ``ValueError``."""
    candidate = (name or "").strip()
    if not _NAME_RE.match(candidate):
        raise ValueError(
            "project name must be 1-64 characters of letters, digits, dot, "
            "underscore or hyphen, and cannot start with a hyphen"
        )
    return candidate


def _iso_date_or_none(value: object, label: str) -> str | None:
    """Canonical ``YYYY-MM-DD``, or ``None`` for "not set".

    An empty string reads as unset: a row that echoed a cleared form can carry
    ``""`` meaning exactly what absent means, and a strict reader would drop a
    whole project row over it. Anything else must parse as an ISO date — the
    field is day-granular by design, and a value that is not a date is a bug
    worth refusing rather than storing.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD date") from exc
    return parsed.isoformat()


def _edit_date(value: str | None, label: str) -> str | None:
    """A date as the EDIT vocabulary spells it.

    Three states reach an update: absent (leave as it is), cleared (``""`` or
    an explicit ``null``), and a real date. The empty string must NOT collapse
    into ``None`` at this layer — ``None`` is what "the caller said nothing"
    looks like after ``model_dump`` — so this validator keeps ``""`` intact and
    only canonicalises real dates. The apply path maps both clear spellings to
    ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD date, or '' to clear it")
    text = value.strip()
    if not text:
        return ""
    return _iso_date_or_none(text, label)


def _validate_estimate(value: object) -> float | None:
    """One estimate bound, shared by the row and by every input model.

    ``0 < estimate <= ESTIMATE_MAX``, fractional allowed: the two vocabularies
    Linear itself offers (points, days) are both summable numbers, and a value
    that could be zero or negative could not be rendered on any view.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("estimate must be a number")
    number = float(value)
    if not (0.0 < number <= ESTIMATE_MAX):
        raise ValueError(f"estimate must be greater than 0 and at most {ESTIMATE_MAX:g}")
    return number


def _validate_milestone_name(value: str | None) -> str:
    """One milestone-name rule, shared by the row and by every input model."""
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("milestone name must not be empty")
    if len(candidate) > MILESTONE_NAME_MAX:
        raise ValueError(f"milestone name must be at most {MILESTONE_NAME_MAX} characters")
    return candidate


def _validate_tags(tags: list[str] | None) -> list[str]:
    values = list(tags or [])
    if len(values) > TAGS_MAX:
        raise ValueError(f"at most {TAGS_MAX} tags")
    for tag in values:
        if not isinstance(tag, str) or not _TAG_RE.match(tag):
            raise ValueError(
                f"tag {tag!r} must be 1-24 characters of lowercase letters, "
                "digits, underscore or hyphen, and cannot start with a hyphen"
            )
    return values


def _validate_sessions(sessions: list[str] | None) -> list[str]:
    values = list(sessions or [])
    if len(values) > SESSIONS_MAX:
        raise ValueError(f"at most {SESSIONS_MAX} linked sessions")
    for session_id in values:
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"session id {session_id!r} is not a 12-character hex id")
    return values


def validate_milestones(milestones: list[Any] | None) -> list["ProjectMilestone"]:
    """The ONE list validator, shared by the row and by every input model.

    One validator and two entry points is the design's rule (the ``todo``
    tool's ``init``/``add`` split is the precedent): the full-replace form and
    the surgical ``milestone`` op must refuse the same lists, or a bulk edit
    could smuggle what the surgical op rejects.
    """
    values: list[ProjectMilestone] = []
    for item in milestones or []:
        if isinstance(item, ProjectMilestone):
            values.append(item)
        else:
            values.append(ProjectMilestone.model_validate(item))
    if len(values) > MILESTONES_MAX:
        raise ValueError(f"at most {MILESTONES_MAX} milestones")
    seen: set[str] = set()
    for milestone in values:
        key = milestone.name.casefold()
        if key in seen:
            raise ValueError(f"duplicate milestone name {milestone.name!r}")
        seen.add(key)
    return values


class ProjectMilestone(BaseModel):
    """One milestone: a name, a target date, and the completion date.

    Milestone *status* (``completed`` / ``overdue`` / ``upcoming``) is
    deliberately NOT stored: it is derived at render from ``completed_at`` and
    ``target_date`` (:func:`milestone_status`), so a stored status can never
    drift from the dates that contradict it — the same "derived, not stored"
    rule the view composer applies to session liveness.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Short milestone name, unique per project.")
    target_date: str | None = Field(
        default=None, description="ISO YYYY-MM-DD the milestone is due, or null."
    )
    completed_at: str | None = Field(
        default=None, description="ISO YYYY-MM-DD the milestone was completed, or null."
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _validate_milestone_name(value)

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "milestone target_date")

    @field_validator("completed_at")
    @classmethod
    def _completed_at(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "milestone completed_at")


def milestone_status(
    milestone: ProjectMilestone, *, today: date | None = None
) -> Literal["completed", "overdue", "upcoming"]:
    """The milestone's status, DERIVED at render, never stored."""
    if milestone.completed_at:
        return "completed"
    moment = today or date.today()
    if milestone.target_date and date.fromisoformat(milestone.target_date) < moment:
        return "overdue"
    return "upcoming"


class Project(BaseModel):
    """One project row, as stored in ``<config_dir>/projects/<id>.json``.

    The field name in the file is ``schema`` (the alias on ``schema_version``):
    a field literally named ``schema`` shadows ``BaseModel``'s own attribute
    and pydantic warns at class definition, so the attribute is named and the
    wire spelling is carried by the alias. Dump with ``by_alias=True`` for disk
    or the wire.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_version: int = Field(
        default=PROJECT_SCHEMA,
        validation_alias="schema",
        serialization_alias="schema",
        description="Row format version; a higher value than this build knows refuses mutation.",
    )
    id: str = Field(description="uuid4 hex; also the row's file name.")
    name: str = Field(description="Unique (case-insensitive), 1-64 [A-Za-z0-9._-].")
    description: str = Field(default="", max_length=DESCRIPTION_MAX)
    status: ProjectStatus = "active"
    progress: str = Field(default="", max_length=PROGRESS_MAX)
    progress_updated_at: float | None = None
    #: Session id, ``"operator"`` (a surface with no session), or ``""``.
    progress_reported_by: str = ""
    tags: list[str] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    # -- v2 planning fields (slice 1 writes them; nothing has shipped without
    # them, so there is no migration and ``schema`` stays 1) ------------------
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = Field(default=None, description="0 < estimate <= 1000.")
    estimate_unit: EstimateUnit = "points"
    milestones: list[ProjectMilestone] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return validate_project_name(value)

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str]) -> list[str]:
        return _validate_tags(value)

    @field_validator("sessions")
    @classmethod
    def _sessions(cls, value: list[str]) -> list[str]:
        return _validate_sessions(value)

    @field_validator("progress_reported_by")
    @classmethod
    def _reported_by(cls, value: str) -> str:
        candidate = (value or "").strip()
        if candidate in {"", "operator"} or _SESSION_ID_RE.fullmatch(candidate):
            return candidate
        raise ValueError("progress_reported_by must be a session id, 'operator', or ''")

    @field_validator("start_date")
    @classmethod
    def _start_date(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "start_date")

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "target_date")

    @field_validator("completed_at")
    @classmethod
    def _completed_at(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "completed_at")

    @field_validator("estimate")
    @classmethod
    def _estimate(cls, value: object) -> float | None:
        return _validate_estimate(value)

    @field_validator("milestones")
    @classmethod
    def _milestones(cls, value: list[Any]) -> list[ProjectMilestone]:
        return validate_milestones(value)

    @model_validator(mode="after")
    def _range_order(self) -> "Project":
        """An inverted range renders as nonsense, so it is refused at write.

        Clearing one side expresses "TBD", so only the both-set case is
        checked. Milestone dates are independent of this range on purpose (a
        milestone may deliberately sit outside it).
        """
        if self.start_date and self.target_date and self.target_date < self.start_date:
            raise ValueError("target_date cannot be before start_date")
        return self


class ProjectEdit(BaseModel):
    """The fields one create/update call may set.

    ABSENT means "leave it as it is"; every mutation path reads
    ``model_fields_set`` rather than treating ``None`` as a value, so a caller
    can set one field without clobbering the others. Dates accept ``""``/``null``
    as an explicit clear (see :func:`_edit_date`); ``milestones`` replaces the
    whole list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    status: ProjectStatus | None = None
    progress: str | None = Field(default=None, max_length=PROGRESS_MAX)
    tags: list[str] | None = None
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: EstimateUnit | None = None
    milestones: list[ProjectMilestone] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        return None if value is None else validate_project_name(value)

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else _validate_tags(value)

    @field_validator("start_date")
    @classmethod
    def _start_date(cls, value: str | None) -> str | None:
        return _edit_date(value, "start_date")

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: str | None) -> str | None:
        return _edit_date(value, "target_date")

    @field_validator("completed_at")
    @classmethod
    def _completed_at(cls, value: str | None) -> str | None:
        return _edit_date(value, "completed_at")

    @field_validator("estimate")
    @classmethod
    def _estimate(cls, value: object) -> float | None:
        return _validate_estimate(value)

    @field_validator("milestones")
    @classmethod
    def _milestones(cls, value: list[Any] | None) -> list[ProjectMilestone] | None:
        return None if value is None else validate_milestones(value)


class MilestoneEdit(BaseModel):
    """The ``milestone`` op's arguments: add-or-update by name, or remove.

    ``target_date``'s ALIAS is the tri-state: absent (leave it), ``""``/null
    (clear it), or a date (set it) — the same vocabulary the full-replace list
    speaks, validated by the same function.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Milestone name; a name that does not exist is added.")
    target_date: str | None = None
    completed: bool | None = Field(
        default=None,
        description="True sets completed_at to today, False clears it, null leaves it.",
    )
    remove: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _validate_milestone_name(value)

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: str | None) -> str | None:
        return _edit_date(value, "milestone target_date")


class ProjectUpdate:
    """What one :meth:`ProjectRegistry.update_project` call did.

    ``changed`` is False when every supplied field already held the value the
    caller sent (the no-op the model is told about rather than a silent write);
    ``refreshed`` is True when an identical progress text on a STALE record
    re-stamped its freshness instead of being suppressed.
    """

    __slots__ = ("project", "changed", "refreshed")

    def __init__(self, project: Project, *, changed: bool, refreshed: bool) -> None:
        self.project = project
        self.changed = changed
        self.refreshed = refreshed


def progress_is_stale(project: Project, *, now: float | None = None) -> bool:
    """Whether the project's recorded progress needs refreshing.

    THE single staleness rule: no report yet is stale by construction (the
    first honest line is still owed), and a report older than
    :data:`PROJECT_PROGRESS_STALE_S` is stale. Computed here so the tool, the
    routes and the completion check cannot disagree about one record.
    """
    if not project.progress:
        return True
    if project.progress_updated_at is None:
        return True
    moment = time.time() if now is None else now
    return (moment - project.progress_updated_at) > PROJECT_PROGRESS_STALE_S


def _utc_now() -> float:
    return time.time()


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


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
            if exc.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):  # noqa: F821
                return False
            raise

    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):  # noqa: F821
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


def _fsync_dir(path: Path) -> None:
    """Flush a directory entry, suppressing only unsupported implementations."""
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            errno.EBADF,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported:
            raise
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one complete file without exposing a truncated target."""
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


class ProjectRegistry:
    """On-disk registry of projects under ``<config_dir>/projects``."""

    def __init__(self, config_dir: Path, refresh_interval: float = 5.0) -> None:
        self.config_dir = Path(config_dir)
        self.projects_dir = self.config_dir / "projects"
        self._projects: dict[str, Project] = {}
        self._last_refresh_time = 0.0
        self._refresh_interval = refresh_interval
        # The directory mtime the snapshot was taken at. A peer's publish is a
        # create/rename inside this directory, so a changed mtime is proof the
        # snapshot is behind even when the interval has not elapsed — the
        # refresh stays bounded (ONE stat per read) and a fresh row is visible
        # without waiting out the interval.
        self._dir_mtime_ns: int | None = None
        self._load()

    def _load(self) -> None:
        """Replace the snapshot with every valid row currently on disk."""
        loaded: dict[str, Project] = {}
        try:
            children = sorted(self.projects_dir.iterdir(), key=lambda path: path.name)
        except OSError:
            self._projects = {}
            self._dir_mtime_ns = self._dir_mtime()
            self._last_refresh_time = time.time()
            return
        rows = 0
        for child in children:
            # Dot-prefixed entries are NEVER rows: the lock file and every
            # in-flight temp write live there, and a half-written temp must
            # stay invisible.
            if child.name.startswith("."):
                continue
            # A crafted symlink must never turn a read or a delete into an
            # operation outside the registry root.
            if child.is_symlink() or not child.is_file():
                continue
            if not child.name.endswith(".json"):
                continue
            rows += 1
            if rows > _MAX_ROWS:
                logger.warning(
                    "projects store at %s has more than %d rows; the rest are not loaded",
                    self.projects_dir,
                    _MAX_ROWS,
                )
                break
            try:
                with child.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    continue
                project = Project.model_validate(data)
                if project.id != child.name[: -len(".json")]:
                    logger.warning(
                        "ignoring project row %s: row id %s does not match",
                        child.name,
                        project.id,
                    )
                    continue
                loaded[project.id] = project
            except FileNotFoundError:
                # The row was replaced between the scan and this open — the
                # documented atomic-publish gap, not a corrupt row.
                continue
            except Exception as exc:  # noqa: BLE001 — one bad file must not hide the rest
                logger.warning("invalid project row in %s: %s", child.name, exc)
        self._projects = loaded
        self._dir_mtime_ns = self._dir_mtime()
        self._last_refresh_time = time.time()

    def _dir_mtime(self) -> int | None:
        try:
            return self.projects_dir.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh_if_needed(self) -> None:
        """Reload when the interval elapsed OR a peer changed the directory."""
        stale = (time.time() - self._last_refresh_time) > self._refresh_interval
        if not stale and self._dir_mtime() != self._dir_mtime_ns:
            stale = True
        if stale:
            self._load()

    def list_projects(self) -> list[Project]:
        """Every project, metadata only, sorted by name (case-insensitive)."""
        self._refresh_if_needed()
        return sorted(self._projects.values(), key=lambda project: project.name.casefold())

    def get_project(self, project_id: str) -> Project:
        """One project by id. Raises ``KeyError`` when the id is unknown."""
        project_id = validate_project_id(project_id)
        self._refresh_if_needed()
        found = self._projects.get(project_id)
        if found is None:
            raise KeyError(f"Project with id {project_id} not found")
        return found

    def _find_cached_project_by_name(self, name: str) -> Project | None:
        wanted = (name or "").strip().casefold()
        if not wanted:
            return None
        for project in self._projects.values():
            if project.name.casefold() == wanted:
                return project
        return None

    def get_project_by_name(self, name: str) -> Project | None:
        """One project by name (case-insensitive), or ``None``."""
        self._refresh_if_needed()
        return self._find_cached_project_by_name(name)

    def projects_for_session(self, session_id: str) -> list[Project]:
        """Derived reverse lookup: the projects this session is linked to."""
        self._refresh_if_needed()
        return sorted(
            (p for p in self._projects.values() if session_id in p.sessions),
            key=lambda project: project.name.casefold(),
        )

    @contextmanager
    def _persistence_lock(self, *, wait: float | None = None) -> Iterator[None]:
        """Serialize one registry mutation across processes.

        The sidecar lives INSIDE ``projects/`` as ``.lock`` and is never
        mistaken for a row (dot-prefixed entries are skipped by ``_load``).
        Every kernel attempt is non-blocking and the retry loop is bounded; a
        timeout raises :class:`ProjectRegistryLockTimeout` so the tool boundary
        can present contention as a recoverable state rather than a traceback.
        Modifications keep the full budget — a write that gives up has failed
        to do the thing the user asked for.
        """
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.projects_dir / ".lock", os.O_CREAT | os.O_RDWR | O_BINARY, 0o600)
        acquired = False
        budget = _LOCK_TIMEOUT_S if wait is None else max(0.0, wait)
        deadline = time.monotonic() + budget
        try:
            while not acquired:
                acquired = _try_lock_exclusive(fd)
                if acquired:
                    break
                if time.monotonic() >= deadline:
                    raise ProjectRegistryLockTimeout(
                        "Timed out waiting for the projects registry lock; "
                        "retry after the other lop process finishes"
                    )
                time.sleep(_LOCK_RETRY_S)
            yield
        finally:
            if acquired:
                _unlock(fd)
            os.close(fd)

    def _save_project_locked(self, project: Project) -> Project:
        """Validate and publish ``project`` while ``_persistence_lock`` is held."""
        project_id = validate_project_id(project.id)
        # The write guard: a row from the future is readable but never
        # rewritten by this build (see the module docstring).
        if project.schema_version > PROJECT_SCHEMA:
            raise ProjectSchemaGuardError(
                f"project {project.name!r} was written by a newer local-operator "
                f"(schema {project.schema_version}); update this build to change it"
            )
        name_key = project.name.casefold()
        occupant = next(
            (
                stored
                for stored in self._projects.values()
                if stored.id != project.id and stored.name.casefold() == name_key
            ),
            None,
        )
        if occupant is not None:
            raise ProjectNameConflictError(f"project {project.name!r} already exists")
        target = self.projects_dir / f"{project_id}.json"
        if target.is_symlink():
            raise ValueError("refusing to save through a symlinked project row")
        payload = project.model_dump(mode="json", by_alias=True)
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        _atomic_write_text(target, text)
        try:
            _fsync_dir(self.projects_dir)
        except BaseException:
            # A NEW row is not acknowledged unless its directory entry is
            # durable, so remove the unacknowledged row and a retry cannot
            # discover phantom success. An UPDATE keeps the just-replaced
            # bytes: ``os.replace`` has already overwritten the old revision,
            # and unlinking here would turn "possibly not durable" into
            # "definitely gone".
            if not existed:
                target.unlink(missing_ok=True)
                _fsync_dir(self.projects_dir)
            raise
        self._projects[project_id] = project
        self._dir_mtime_ns = self._dir_mtime()
        return project

    def create_project(
        self,
        fields: ProjectEdit,
        *,
        sessions: Sequence[str] = (),
        progress_reported_by: str = "",
    ) -> Project:
        """Create one project from ``fields``; the name must be free.

        ``sessions`` is the auto-link the ``project`` tool passes (the calling
        session); the desktop create route passes none, so a UI-created project
        starts unlinked and is linked from the projects surface.
        """
        name = validate_project_name(fields.name or "")
        now = _utc_now()
        with self._persistence_lock():
            # Unconditional refresh INSIDE the lock: an interval-gated snapshot
            # cannot prove uniqueness when another process wrote since our last
            # read.
            self._load()
            if self._find_cached_project_by_name(name) is not None:
                raise ProjectNameConflictError(f"project {name!r} already exists")
            supplied = fields.model_fields_set
            status = fields.status or "active"
            if "completed_at" in supplied:
                completed_at = None if not fields.completed_at else fields.completed_at
            elif status == "done":
                # Setting status='done' with completed_at OMITTED stamps today
                # (the one convenience of the pair; the full rule is stated in
                # the tool description).
                completed_at = _today_iso()
            else:
                completed_at = None
            project = Project(
                schema_version=PROJECT_SCHEMA,
                id=uuid.uuid4().hex,
                name=name,
                description=(fields.description or "").strip(),
                status=status,
                progress=fields.progress or "",
                progress_updated_at=now if (fields.progress or "") else None,
                progress_reported_by=(progress_reported_by if (fields.progress or "") else ""),
                tags=list(fields.tags or []),
                sessions=list(sessions),
                created_at=now,
                updated_at=now,
                start_date=fields.start_date or None,
                target_date=fields.target_date or None,
                completed_at=completed_at,
                estimate=fields.estimate,
                estimate_unit=fields.estimate_unit or "points",
                milestones=list(fields.milestones or []),
            )
            return self._save_project_locked(project)

    def update_project(
        self, project_id: str, fields: ProjectEdit, *, reporter: str = ""
    ) -> ProjectUpdate:
        """Merge ``fields`` into one project under the store lock.

        Per-field last-writer-wins: only the fields present in
        ``fields.model_fields_set`` are touched, so two sessions updating
        different fields serialize without clobbering. The progress rule is the
        refresh amendment: identical text on a FRESH record writes nothing,
        identical text on a STALE record re-stamps freshness (which is what
        makes the completion check's "already current" exit real), and new text
        writes and stamps.
        """
        project_id = validate_project_id(project_id)
        now = _utc_now()
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            candidate = current.model_copy(deep=True)
            supplied = fields.model_fields_set
            changed = False
            refreshed = False

            if "name" in supplied and fields.name is not None:
                candidate.name = fields.name
            if "description" in supplied and fields.description is not None:
                candidate.description = fields.description.strip()
            if "tags" in supplied and fields.tags is not None:
                candidate.tags = list(fields.tags)
            if "start_date" in supplied:
                candidate.start_date = fields.start_date or None
            if "target_date" in supplied:
                candidate.target_date = fields.target_date or None
            if "estimate" in supplied and fields.estimate is not None:
                candidate.estimate = fields.estimate
            if "estimate_unit" in supplied and fields.estimate_unit is not None:
                candidate.estimate_unit = fields.estimate_unit
            if "milestones" in supplied and fields.milestones is not None:
                candidate.milestones = list(fields.milestones)

            # completed_at and status interact, so the rule lives here once:
            #   * completed_at provided ("" clears / a date sets) wins.
            #   * else a transition INTO "done" stamps today.
            #   * moving away from done leaves the date untouched.
            if "completed_at" in supplied:
                candidate.completed_at = fields.completed_at or None
            if "status" in supplied and fields.status is not None:
                was_done = current.status == "done"
                candidate.status = fields.status
                if fields.status == "done" and not was_done and "completed_at" not in supplied:
                    candidate.completed_at = _today_iso()

            if "progress" in supplied:
                new_text = fields.progress or ""
                if new_text != candidate.progress:
                    candidate.progress = new_text
                    if new_text:
                        candidate.progress_updated_at = now
                        candidate.progress_reported_by = reporter
                    else:
                        # An empty snippet IS "no progress recorded": clearing
                        # the text clears the freshness pair with it, so a
                        # reader never sees a timestamp over nothing.
                        candidate.progress_updated_at = None
                        candidate.progress_reported_by = ""
                    changed = True
                elif progress_is_stale(candidate, now=now):
                    # THE REFRESH AMENDMENT: identical text on a stale record.
                    candidate.progress_updated_at = now
                    candidate.progress_reported_by = reporter or candidate.progress_reported_by
                    changed = True
                    refreshed = True
                # else: identical text on a fresh record — a no-op, no write.

            # Re-validate the merged candidate through the model's own rules
            # before anything touches disk (dates order, caps, grammar).
            candidate = Project.model_validate(candidate.model_dump(mode="json", by_alias=True))
            if candidate == current:
                # Pydantic equality covers every field, so "nothing moved" is
                # provable rather than inferred from the changed flags.
                return ProjectUpdate(current, changed=False, refreshed=refreshed)
            if not changed:
                # A merged value differs from the stored one only through
                # equality above; reaching here with changed=False means the
                # diff came from normalisation, which still deserves a write.
                changed = True
            candidate.updated_at = now
            saved = self._save_project_locked(candidate)
            return ProjectUpdate(saved, changed=changed, refreshed=refreshed)

    def link_session(self, project_id: str, session_id: str) -> tuple[Project, bool]:
        """Link one session to one project. Returns ``(project, added)``."""
        project_id = validate_project_id(project_id)
        session_id = _validate_sessions([session_id])[0]
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            if session_id in current.sessions:
                return current, False
            if len(current.sessions) >= SESSIONS_MAX:
                raise ValueError(
                    f"project {current.name!r} already has {SESSIONS_MAX} linked "
                    "sessions; unlink one first"
                )
            candidate = current.model_copy(deep=True)
            candidate.sessions = [*candidate.sessions, session_id]
            candidate.updated_at = _utc_now()
            return self._save_project_locked(candidate), True

    def unlink_session(self, project_id: str, session_id: str) -> tuple[Project, bool]:
        """Unlink one session from one project. Returns ``(project, removed)``."""
        project_id = validate_project_id(project_id)
        session_id = _validate_sessions([session_id])[0]
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            if session_id not in current.sessions:
                return current, False
            candidate = current.model_copy(deep=True)
            candidate.sessions = [s for s in candidate.sessions if s != session_id]
            candidate.updated_at = _utc_now()
            return self._save_project_locked(candidate), True

    def set_milestone(self, project_id: str, fields: MilestoneEdit) -> tuple[Project, str]:
        """Add, update or remove one milestone by name. Returns ``(project, action)``.

        ``action`` is ``"added"``, ``"updated"`` or ``"removed"`` — the receipt
        the tool and the routes report. Renaming is remove + add, documented
        and deliberate: two names in one call is a rename vocabulary the surgical
        op does not need.
        """
        project_id = validate_project_id(project_id)
        supplied = fields.model_fields_set
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            candidate = current.model_copy(deep=True)
            wanted = fields.name.casefold()
            index = next(
                (i for i, m in enumerate(candidate.milestones) if m.name.casefold() == wanted),
                None,
            )
            if fields.remove:
                if index is None:
                    raise KeyError(f"no milestone named {fields.name!r}")
                del candidate.milestones[index]
                action = "removed"
            elif index is None:
                candidate.milestones.append(
                    ProjectMilestone(
                        name=fields.name,
                        target_date=(
                            None if fields.target_date in (None, "") else fields.target_date
                        ),
                        completed_at=_today_iso() if fields.completed else None,
                    )
                )
                action = "added"
            else:
                milestone = candidate.milestones[index]
                before = milestone.model_dump(mode="json")
                if "target_date" in supplied:
                    milestone.target_date = fields.target_date or None
                if fields.completed is not None:
                    milestone.completed_at = _today_iso() if fields.completed else None
                if milestone.model_dump(mode="json") == before:
                    # Nothing supplied (or nothing moved): report the truthful
                    # no-change instead of churning ``updated_at`` on the row.
                    return current, "unchanged"
                candidate.milestones[index] = ProjectMilestone.model_validate(
                    milestone.model_dump(mode="json")
                )
                action = "updated"
            # The shared validator runs on the whole list, so the surgical op
            # and the full-replace form cannot disagree about what is legal.
            candidate.milestones = validate_milestones(candidate.milestones)
            candidate.updated_at = _utc_now()
            return self._save_project_locked(candidate), action

    def delete_project(self, project_id: str) -> None:
        """Permanently remove one row. Session directories are never touched."""
        project_id = validate_project_id(project_id)
        with self._persistence_lock():
            # Delete participates in the same ordering as save, so a
            # delete-recreate-stale-save sequence has one unambiguous winner.
            self._load()
            if project_id not in self._projects:
                raise KeyError(f"Project with id {project_id} not found")
            current = self._projects.get(project_id)
            if current is not None and current.schema_version > PROJECT_SCHEMA:
                # The GUARD covers removal too, deliberately: a row written by a
                # newer build holds fields this one cannot render, so deleting it
                # would destroy data the deleter was never shown. The remedy is
                # the same sentence every mutation uses (update this build).
                raise ProjectSchemaGuardError(
                    f"project {current.name!r} was written by a newer local-operator "
                    f"(schema {current.schema_version}); update this build before deleting it"
                )
            target = self.projects_dir / f"{project_id}.json"
            if target.is_symlink():
                raise ValueError("refusing to delete a symlinked project row")
            # Remove the on-disk copy FIRST: if the unlink fails, the cache
            # still agrees with disk and the row remains visible.
            if target.exists():
                target.unlink()
                _fsync_dir(self.projects_dir)
            self._projects.pop(project_id, None)
            self._dir_mtime_ns = self._dir_mtime()


# ---------------------------------------------------------------------------
# The view composer
# ---------------------------------------------------------------------------


def _load_subagent_summary(session_dir: Path) -> dict[str, Any] | None:
    """``{"running", "settled", "names"}`` from the roster sidecar, or ``None``.

    A missing sidecar and an unreadable one both read ``None`` ("no roster"),
    never zeroes: a session that never launched a child has no sidecar, and a
    0/0 summary would claim an empty roster it cannot prove.
    """
    path = session_dir / "subagent-roster.v1.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return None
    running = 0
    settled = 0
    names: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status", ""))
        label = str(job.get("label", "") or job.get("id", ""))
        if status == "running":
            running += 1
            if len(names) < 5:
                names.append(label)
        else:
            settled += 1
    for job in jobs:
        if len(names) >= 5:
            break
        if isinstance(job, dict) and str(job.get("status", "")) != "running":
            names.append(str(job.get("label", "") or job.get("id", "")))
    return {"running": running, "settled": settled, "names": names[:5]}


def _load_todo_summary(session_dir: Path) -> dict[str, int] | None:
    """``{"open", "total"}`` from the newest ``todo_snapshot`` row, or ``None``.

    The bounded backward scan (``read_latest_custom``) answers a one-row
    question without parsing the journal: transcripts reach tens of MB, and a
    project view must never pay a full parse for a count. A session that never
    persisted a snapshot reads ``None`` ("unknown"), never 0.
    """
    from local_operator.session.transcript import read_latest_custom

    details = read_latest_custom(session_dir, "todo_snapshot")
    if not details:
        return None
    items = details.get("items")
    if not isinstance(items, list):
        return None
    total = 0
    open_count = 0
    for phase in items:
        if not isinstance(phase, dict):
            continue
        rows = phase.get("items")
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            total += 1
            if item.get("status") == "pending":
                open_count += 1
    return {"open": open_count, "total": total}


def scan_runtime_states(config_dir: Path) -> dict[str, dict[str, Any]]:
    """One runtime scan, keyed by session id, in the view payload's own shape.

    THE one audit of "is each session running, and where", read by
    :func:`build_project_view` and by callers that compose MANY projects (the
    desktop listing counts live sessions per row): one scan serves the whole
    call, so a listing of N projects does not walk the runtime registry N times.

    READER MODE (``reap=False``): a view must leave ``run/mobile``
    byte-identical — the record on disk is evidence a later "why did this die"
    question reads, and reaping is somebody else's job (every ``lop sessions``
    and every interactive boot sweeps). The run directory's EXISTENCE is checked
    first, so a machine that has never run a runtime does not get one created by
    a project view. Any failure degrades to an empty map: reading a project
    never fails on runtime state.
    """
    from local_operator.session.runtime import registry as runtime_registry
    from local_operator.session.runtime.types import RUN_DIRNAME

    root = Path(config_dir)
    by_session: dict[str, list[Any]] = {}
    if (root / RUN_DIRNAME).is_dir():
        try:
            for record, state in runtime_registry.scan(root, reap=False):
                by_session.setdefault(record.session_id, []).append((record, state))
        except Exception:  # noqa: BLE001 — a view never fails on runtime state
            logger.debug("could not scan runtime records", exc_info=True)

    states: dict[str, dict[str, Any]] = {}
    for session_id, records in by_session.items():
        picked = _pick_record(records)
        if picked is None:
            continue
        record, state = picked
        states[session_id] = {
            "state": state,
            "busy": bool(getattr(record, "busy", False)),
            "heartbeat_age_s": max(0.0, time.time() - record.heartbeat_at),
            "pid": record.pid,
        }
    return states


def _pick_record(records: list[Any]) -> Any | None:
    """The best record for one session id: live beats wedged beats stale, then newest."""
    rank = {"live": 2, "wedged": 1, "stale": 0}
    best: Any | None = None
    for record, state in records:
        if best is None:
            best = (record, state)
            continue
        best_record, best_state = best
        if (rank.get(state, 0), record.heartbeat_at) > (
            rank.get(best_state, 0),
            best_record.heartbeat_at,
        ):
            best = (record, state)
    return best


def build_project_view(
    project: Project,
    *,
    config_dir: Path,
    live: dict[str, dict[str, Any]] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose the ONE view of a project: its row plus one row per linked session.

    Three sources, in priority order, exactly as the design states:

    1. **Runtime records** — one ``registry.scan`` for the whole call, filtered
       per session id. A record classifies ``live`` / ``wedged`` / ``stale``;
       no record at all reads ``stopped``, which is the common, honest case.
    2. **Durable session directory** — exists? title, creation date, archived
       flag; the subagent roster sidecar and the newest ``todo_snapshot`` row.
    3. **In-process overlay** (``live``, TUI only) — narrow per-session overrides
       merged over the computed row for the calling process's own session, whose
       in-memory values are fresher than disk.

    ``None`` means unknown and is never rendered as 0: a session that never
    launched a subagent has ``subagents: null``, and a session that never
    persisted a todo snapshot has ``todos: null``.

    ``records`` is :func:`scan_runtime_states`'s answer for a caller composing
    several projects from ONE scan (the desktop listing); ``None`` scans here.
    """
    from local_operator.resume import read_title_state
    from local_operator.session.archived import archived_ids
    from local_operator.session.creation import session_created_at

    root = Path(config_dir)
    sessions_root = root / "sessions"
    states = scan_runtime_states(root) if records is None else records

    try:
        archived = archived_ids(root)
    except Exception:  # noqa: BLE001
        archived = frozenset()

    rows: list[dict[str, Any]] = []
    for session_id in project.sessions:
        session_dir = sessions_root / session_id
        exists = session_dir.is_dir()
        title: str | None = None
        created_at: float | None = None
        if exists:
            state = read_title_state(session_dir)
            title = state.text if state is not None else None
            created_at = session_created_at(session_dir) or None

        runtime = states.get(session_id) or {
            "state": "stopped",
            "busy": None,
            "heartbeat_age_s": None,
            "pid": None,
        }

        row: dict[str, Any] = {
            "session_id": session_id,
            "exists": exists,
            "title": title,
            "created_at": created_at,
            "archived": session_id in archived,
            "runtime": runtime,
            "subagents": _load_subagent_summary(session_dir) if exists else None,
            "todos": _load_todo_summary(session_dir) if exists else None,
        }
        if live and session_id in live:
            # The overlay is shallow and per field: the caller owns fresher
            # values for the fields it passes and nothing else.
            row.update(live[session_id])
        rows.append(row)

    return {
        "project": project.model_dump(mode="json", by_alias=True),
        "progress_stale": progress_is_stale(project),
        "sessions": rows,
    }


def readable_error(exc: Exception) -> str:
    """A one-line, human-readable rendering of a validation failure.

    ``ValidationError`` stringifies to a multi-line report with the input value
    repeated; both consumers (the ``project`` tool's result and the desktop
    route's 422 body) carry prose a model or a user reads, so the first error's
    message is quoted and the rest are dropped. Shared HERE rather than
    duplicated per surface: a caller must not be able to tell which one
    answered it.
    """
    if isinstance(exc, KeyError):
        # ``str(KeyError("x"))`` quotes its message; the sentence inside is the
        # one a caller wrote, so it is quoted out here.
        return str(exc.args[0]) if exc.args else str(exc)
    if isinstance(exc, ValidationError):
        problems = exc.errors()
        if problems:
            first = problems[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "invalid value"))
            return f"{location}: {message}" if location else message
    return str(exc)
