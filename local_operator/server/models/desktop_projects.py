"""Wire models for ``/v1/desktop/projects*`` — the desktop half of the primitive.

**What a project is.** A tracked workstream row in the operator's own config
root (``<config_dir>/projects/``), stored and validated by
:mod:`local_operator.projects`. It is not an agent, not a schedule and not a
team's ``project`` brief — this module only carries it to the wire.

**Derived status is computed once, HERE.** A milestone's status is never
stored: it is ``completed`` when ``completed_at`` is set, else ``overdue`` when
its ``target_date`` has passed, else ``upcoming``. The derivation is
:func:`local_operator.projects.milestone_status` — the same function the tool
and every other reader calls — so a chip on a card and a line in a tool result
cannot disagree about which milestone is late.

**``progress_stale`` is server-computed**, from the one 30-minute constant in
:mod:`local_operator.projects`, so the UI's stale badge and the completion check
can never disagree about one record.

**``extra="allow"`` on the view models**, matching the other desktop payloads: a
field added by a later build crosses additively and an older renderer ignores
it. The listed fields are the frozen contract the UI repo codes against.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.projects import Project, milestone_status, progress_is_stale


class ProjectMilestoneView(BaseModel):
    """One milestone, with its DERIVED status (see the module docstring)."""

    model_config = ConfigDict(extra="allow")

    name: str
    target_date: str | None = None
    completed_at: str | None = None
    status: Literal["completed", "overdue", "upcoming"]


class ProjectView(BaseModel):
    """The full record — what ``GET .../{key}`` and every write returns."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    status: str
    progress: str = ""
    progress_updated_at: float | None = None
    progress_reported_by: str = ""
    #: Computed from the single staleness constant, never stored (see above).
    progress_stale: bool
    tags: list[str] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: str = "points"
    milestones: list[ProjectMilestoneView] = Field(default_factory=list)


class ProjectSummary(BaseModel):
    """One row of the listing / board — the fields those views paint.

    ``live_sessions`` is a count from ONE machine-wide runtime scan per
    listing call (``projects.scan_runtime_states``), not a per-row dial: the
    board card's ``2 live`` chip and the list column read the same number the
    detail view's per-session dots derive from.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    status: str
    tags: list[str] = Field(default_factory=list)
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: str = "points"
    milestones_completed: int = 0
    milestones_total: int = 0
    sessions: int = 0
    live_sessions: int = 0
    progress_stale: bool = True
    progress_updated_at: float | None = None
    updated_at: float = 0.0


class LinkedSessionView(BaseModel):
    """One linked session row of the composed view.

    ``None`` means UNKNOWN and is never rendered as 0 — for ``subagents`` (no
    roster sidecar) and ``todos`` (no persisted snapshot). ``runtime.state`` is
    one of ``live`` / ``wedged`` / ``stale`` / ``stopped``; ``stopped`` means no
    runtime record at all, which is the common case for a session the operator
    finished with.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    exists: bool
    title: str | None = None
    created_at: float | None = None
    archived: bool = False
    runtime: dict[str, Any]
    subagents: dict[str, Any] | None = None
    todos: dict[str, Any] | None = None


class ProjectList(BaseModel):
    """``GET /v1/desktop/projects``."""

    projects: list[ProjectSummary] = Field(default_factory=list)


class ProjectDetail(BaseModel):
    """``GET /v1/desktop/projects/{key}`` — the row plus its linked sessions."""

    project: ProjectView
    links: list[LinkedSessionView] = Field(default_factory=list)


class ProjectDeleted(BaseModel):
    """``DELETE /v1/desktop/projects/{key}``."""

    deleted: bool = True


def project_view(project: Project) -> ProjectView:
    """The full wire view of one row, with derived milestone statuses."""

    return ProjectView(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status,
        progress=project.progress,
        progress_updated_at=project.progress_updated_at,
        progress_reported_by=project.progress_reported_by,
        progress_stale=progress_is_stale(project),
        tags=list(project.tags),
        sessions=list(project.sessions),
        created_at=project.created_at,
        updated_at=project.updated_at,
        start_date=project.start_date,
        target_date=project.target_date,
        completed_at=project.completed_at,
        estimate=project.estimate,
        estimate_unit=project.estimate_unit,
        milestones=[
            ProjectMilestoneView(
                name=item.name,
                target_date=item.target_date,
                completed_at=item.completed_at,
                status=milestone_status(item),
            )
            for item in project.milestones
        ],
    )


def project_summary(project: Project, *, live_sessions: int) -> ProjectSummary:
    """The compact wire row, with the counts the list/board render."""

    return ProjectSummary(
        id=project.id,
        name=project.name,
        description=project.description,
        status=project.status,
        tags=list(project.tags),
        start_date=project.start_date,
        target_date=project.target_date,
        completed_at=project.completed_at,
        estimate=project.estimate,
        estimate_unit=project.estimate_unit,
        milestones_completed=sum(1 for item in project.milestones if item.completed_at),
        milestones_total=len(project.milestones),
        sessions=len(project.sessions),
        live_sessions=live_sessions,
        progress_stale=progress_is_stale(project),
        progress_updated_at=project.progress_updated_at,
        updated_at=project.updated_at,
    )


def linked_session_view(row: dict[str, Any]) -> LinkedSessionView:
    """One ``build_project_view`` session row, as the wire model."""

    return LinkedSessionView.model_validate(row)
