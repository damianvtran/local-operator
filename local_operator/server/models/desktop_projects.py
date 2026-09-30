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

**``progress_stale`` is server-computed**, from the staleness window in
:mod:`local_operator.projects` (``projects.stale_after_hours``, four hours by
default, and only for the in-flight rows (``planning``/``active``/``qa``/
``validation``; settled records never read stale) — so the UI's stale badge and
the completion check can never disagree about one record. The refresher side of
the same pair (``progress_refreshed_at``/``progress_refreshed_by``) crosses for
the "refreshed … — no new content since …" annotation; ``progress_stale`` does
NOT clear on a refresh (the badge tells the truth about content age).

**``coordination_sessions``** is the "filed by" provenance list (schema 2):
``ProjectSummary`` carries its COUNT (mirroring how ``sessions`` is a count
there) and ``ProjectView`` the ids. It is never counted as a working link —
``sessions`` IS the work set — and a composition shows every coordination id
as a session row with ``role="coordination"`` and NO runtime facts at all.

**``extra="allow"`` on the view models**, matching the other desktop payloads: a
field added by a later build crosses additively and an older renderer ignores
it. The listed fields are the frozen contract the UI repo codes against.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.projects import (
    DESCRIPTION_MAX,
    PROGRESS_MAX,
    Project,
    ProjectStatus,
    milestone_status,
    progress_is_stale,
)


class ProjectMilestoneView(BaseModel):
    """One milestone, with its DERIVED status (see the module docstring)."""

    model_config = ConfigDict(extra="allow")

    name: str
    target_date: str | None = None
    completed_at: str | None = None
    status: Literal["completed", "overdue", "upcoming"]


class ProjectAttachmentView(BaseModel):
    """One stored attachment on a history entry.

    ``path`` is the stored copy's location — resolvable on the machine that
    serves the payload — and ``kind`` is the classification made when the
    file was copied (``image`` for screenshots and image formats, ``data``
    otherwise).
    """

    name: str = ""
    kind: str = "data"
    path: str = ""
    bytes: int = 0
    added_at: str = ""


class ProjectUpdateView(BaseModel):
    """One append-only history entry, newest last (the log's own order)."""

    at: str = ""
    text: str = ""
    by: str = ""
    attachments: list[ProjectAttachmentView] = Field(default_factory=list)


class ProjectView(BaseModel):
    """The full record — what ``GET .../{key}`` and every write returns."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str = ""
    owner: str | None = None
    team: str | None = None
    title: str | None = None
    status: str
    progress: str = ""
    progress_updated_at: float | None = None
    progress_reported_by: str = ""
    #: The refresh assertion pair (schema 2): when a writer last checked the
    #: line still describes reality, and who. Renders the "refreshed …"
    #: annotation; never a freshness claim (see the module docstring).
    progress_refreshed_at: float | None = None
    progress_refreshed_by: str = ""
    #: Computed from the configured staleness window, never stored (see above).
    progress_stale: bool
    tags: list[str] = Field(default_factory=list)
    sessions: list[str] = Field(default_factory=list)
    #: "Filed by" provenance ids (schema 2) — never work, never liveness.
    coordination_sessions: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: str = "points"
    milestones: list[ProjectMilestoneView] = Field(default_factory=list)
    #: The append-only history (newest last) — the full log as the row holds
    #: it, bounded at the store's cap. Detail-only: the LISTING (summary)
    #: must not carry it (one row's log is not a list-row's payload).
    updates: list[ProjectUpdateView] = Field(default_factory=list)


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
    owner: str | None = None
    team: str | None = None
    title: str | None = None
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
    #: Filing count only — never added to ``sessions``/``live_sessions``.
    coordination_sessions: int = 0
    progress_stale: bool = True
    progress_updated_at: float | None = None
    progress_refreshed_at: float | None = None
    progress_refreshed_by: str = ""
    updated_at: float = 0.0


class LinkedSessionView(BaseModel):
    """One linked session row of the composed view.

    ``None`` means UNKNOWN and is never rendered as 0 — for ``subagents`` (no
    roster sidecar) and ``todos`` (no persisted snapshot). ``runtime.state`` is
    one of ``live`` / ``wedged`` / ``stale`` / ``stopped``; ``stopped`` means no
    runtime record at all, which is the common case for a session the operator
    finished with.

    ``role`` separates the two lists (schema 2): ``work`` rows carry every
    liveness fact; ``coordination`` rows ("filed by") carry NONE — no
    ``runtime``, no ``subagents``, no ``todos`` — so a renderer that gates its
    liveness chrome on the presence of those fields cannot paint a filing as a
    worker. ``runtime=None`` is therefore the wired shape of a coordination
    row, not an error.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str
    role: Literal["work", "coordination"] = "work"
    exists: bool
    title: str | None = None
    created_at: float | None = None
    archived: bool = False
    runtime: dict[str, Any] | None = None
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


#: The fixed board order, and therefore both listings' sort: ``archived`` last,
#: a status the board only draws when non-empty. An unknown status (a row from a
#: newer build) sorts after them all rather than crashing the sort. ONE copy:
#: the desktop routes and the mobile daemon both import it from here, and the
#: phone's compile-time section order mirrors it (``STATUS_ORDER`` in
#: ``mobile/web/src/components/projects-sheet.tsx``, which cannot import
#: Python).
STATUS_RANK = {
    "planning": 0,
    "active": 1,
    "qa": 2,
    "validation": 3,
    "paused": 4,
    "done": 5,
    "archived": 6,
}


class _Request(BaseModel):
    """Base for the write bodies, mirroring the desktop routes' ``Input``.

    ``Input`` itself lives in ``routes/desktop_sessions.py`` and cannot be
    imported here — this module stays importable without the server's HTTP
    stack, and the mobile daemon reads it — so the one rule it carries,
    ``extra="forbid"``, is restated. Request models live in THIS module rather
    than beside their handlers so both surfaces can derive their accepted keys
    from one source: the mobile daemon validates its flat bodies against
    ``tuple(Model.model_fields)`` (round-1 review, [m]3).
    """

    model_config = ConfigDict(extra="forbid")


class ProjectCreate(_Request):
    """``POST`` body — the design's frozen create contract (§4.1).

    Dates, the estimate and milestones are set afterwards via ``PATCH`` and the
    milestone routes; the tool's ``create`` accepts them in one call because it
    is a different surface with a different budget.
    """

    name: str = Field(min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    status: ProjectStatus | None = None
    tags: list[str] | None = None


class ProjectPatch(_Request):
    """``PATCH`` body — every field optional; omitted fields are untouched.

    ``""`` CLEARS a date (or the progress snippet); omitting the key leaves it
    alone. That tri-state is why the route forwards only the keys the caller
    actually sent (``model_fields_set``) into :class:`ProjectEdit`.
    """

    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    owner: str | None = None
    team: str | None = None
    title: str | None = None
    status: ProjectStatus | None = None
    progress: str | None = Field(default=None, max_length=PROGRESS_MAX)
    tags: list[str] | None = None
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: str | None = None


class ProjectDelete(_Request):
    """``DELETE`` body — the project's NAME, typed, is the confirmation.

    A mismatch is a 422 rather than a silent success: the body is the whole
    request, and a client that sends the wrong name is not asking for this
    deletion (the same ladder ``ConfirmDeletion`` climbs for sessions).
    """

    confirm: str = Field(min_length=1, max_length=64)


class LinkMutation(_Request):
    session_id: str = Field(min_length=1, max_length=64)


class MilestoneMutation(_Request):
    name: str = Field(min_length=1, max_length=80)
    target_date: str | None = None
    completed: bool | None = None


def project_view(project: Project, *, window: float | None = None) -> ProjectView:
    """The full wire view of one row, with derived milestone statuses."""

    return ProjectView(
        id=project.id,
        name=project.name,
        description=project.description,
        owner=project.owner,
        team=project.team,
        title=project.title,
        status=project.status,
        progress=project.progress,
        progress_updated_at=project.progress_updated_at,
        progress_reported_by=project.progress_reported_by,
        progress_refreshed_at=project.progress_refreshed_at,
        progress_refreshed_by=project.progress_refreshed_by,
        progress_stale=progress_is_stale(project, window=window),
        tags=list(project.tags),
        sessions=list(project.sessions),
        coordination_sessions=list(project.coordination_sessions),
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
        updates=[
            ProjectUpdateView(
                at=entry.at,
                text=entry.text,
                by=entry.by,
                attachments=[
                    ProjectAttachmentView(
                        name=attachment.name,
                        kind=attachment.kind,
                        path=attachment.path,
                        bytes=attachment.bytes,
                        added_at=attachment.added_at,
                    )
                    for attachment in entry.attachments
                ],
            )
            for entry in project.updates
        ],
    )


def project_summary(
    project: Project, *, live_sessions: int, window: float | None = None
) -> ProjectSummary:
    """The compact wire row, with the counts the list/board render."""

    return ProjectSummary(
        id=project.id,
        name=project.name,
        description=project.description,
        owner=project.owner,
        team=project.team,
        title=project.title,
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
        coordination_sessions=len(project.coordination_sessions),
        progress_stale=progress_is_stale(project, window=window),
        progress_updated_at=project.progress_updated_at,
        progress_refreshed_at=project.progress_refreshed_at,
        progress_refreshed_by=project.progress_refreshed_by,
        updated_at=project.updated_at,
    )


def linked_session_view(row: dict[str, Any]) -> LinkedSessionView:
    """One ``build_project_view`` session row, as the wire model."""

    return LinkedSessionView.model_validate(row)
