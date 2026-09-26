"""``/v1/desktop/projects*`` — the desktop CRUD, links and milestones surface.

Seven + two routes over the project store (:mod:`local_operator.projects`):

- ``GET    /v1/desktop/projects``                        — the listing (summaries)
- ``POST   /v1/desktop/projects``                        — create
- ``GET    /v1/desktop/projects/{key}``                  — one project + its links
- ``PATCH  /v1/desktop/projects/{key}``                  — partial edit
- ``DELETE /v1/desktop/projects/{key}``                  — confirm-by-name delete
- ``POST   /v1/desktop/projects/{key}/links``            — link one session
- ``DELETE /v1/desktop/projects/{key}/links/{sid}``      — unlink one session
- ``POST   /v1/desktop/projects/{key}/milestones``       — add-or-update one
- ``DELETE /v1/desktop/projects/{key}/milestones/{name}``— remove one

**The store is the only writer.** Every route calls the same
:class:`~local_operator.projects.ProjectRegistry` methods the ``project`` tool
calls, so the two surfaces cannot validate differently, and the composed view
(``{project, links}``) comes from the same ``build_project_view`` the tool's
``show`` renders — no second derivation of "what are the linked sessions
doing".

**Key resolution.** ``{key}`` is an exact project id first, then a
case-insensitive name. Ids are ``uuid4().hex`` and names are grammar-limited,
so a collision would require a project literally named as another's hex id;
the id arm wins and is documented here rather than left to chance.

**No receipts journal.** Unlike the session routes, these mutations carry no
``request_id``: a retried ``POST`` cannot duplicate a project because the name
must be free (409), and ``PATCH``/``DELETE`` converge on the same state. The
create body is the design's frozen contract for the UI repo (§4.1).

**Status codes.** 404 unknown key; 409 a free-name conflict or a row written by
a newer build (the write guard); 422 a body or value the store refuses (the
same sentence the tool receives, via ``projects.readable_error``); 503 the
store lock timed out, which is retryable.

**This module talks to no session.** No route mutates a session directory, and
no route dials a runtime: the linked-session rows are read from files, exactly
as the tool's view is.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, ValidationError

from local_operator.projects import (
    DESCRIPTION_MAX,
    PROGRESS_MAX,
    MilestoneEdit,
    Project,
    ProjectEdit,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    ProjectStatus,
    build_project_view,
    readable_error,
    scan_runtime_states,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_projects import (
    ProjectDeleted,
    ProjectDetail,
    ProjectList,
    ProjectSummary,
    ProjectView,
    linked_session_view,
    project_summary,
    project_view,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import Input, errors, reply

router = APIRouter(tags=["Desktop projects"], dependencies=[Depends(require_desktop)])

#: The fixed board order, and therefore the listing's sort: ``archived`` last,
#: a status the board only draws when non-empty. An unknown status (a row from a
#: newer build) sorts after them all rather than crashing the sort.
_STATUS_RANK = {"active": 0, "paused": 1, "done": 2, "archived": 3}


class ProjectCreate(Input):
    """``POST`` body — the design's frozen create contract (§4.1).

    Dates, the estimate and milestones are set afterwards via ``PATCH`` and the
    milestone routes; the tool's ``create`` accepts them in one call because it
    is a different surface with a different budget.
    """

    name: str = Field(min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    status: ProjectStatus | None = None
    tags: list[str] | None = None


class ProjectPatch(Input):
    """``PATCH`` body — every field optional; omitted fields are untouched.

    ``""`` CLEARS a date (or the progress snippet); omitting the key leaves it
    alone. That tri-state is why the route forwards only the keys the caller
    actually sent (``model_fields_set``) into :class:`ProjectEdit`.
    """

    name: str | None = Field(default=None, min_length=1, max_length=64)
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    status: ProjectStatus | None = None
    progress: str | None = Field(default=None, max_length=PROGRESS_MAX)
    tags: list[str] | None = None
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: str | None = None


class ProjectDelete(Input):
    """``DELETE`` body — the project's NAME, typed, is the confirmation.

    A mismatch is a 422 rather than a silent success: the body is the whole
    request, and a client that sends the wrong name is not asking for this
    deletion (the same ladder ``ConfirmDeletion`` climbs for sessions).
    """

    confirm: str = Field(min_length=1, max_length=64)


class LinkMutation(Input):
    session_id: str = Field(min_length=1, max_length=64)


class MilestoneMutation(Input):
    name: str = Field(min_length=1, max_length=80)
    target_date: str | None = None
    completed: bool | None = None


def _registry(request: Request) -> ProjectRegistry:
    """A fresh adapter over ``<config_dir>/projects`` for one request.

    Fresh metadata per operation (the ``desktop_profiles`` convention): another
    process's write is visible without waiting for any polling cache. This is an
    adapter over the same directory the tool writes, not a second registry.
    """
    return ProjectRegistry(request.app.state.config_manager.config_dir)


def _find(registry: ProjectRegistry, key: str) -> Project | None:
    """Resolve a route key: exact id first, then a case-insensitive name."""
    try:
        return registry.get_project(key)
    except (KeyError, ValueError):
        # Not an existing row id — either the id is unknown or the key is not
        # id-shaped at all. Either way the name arm is the only one left.
        return registry.get_project_by_name(key)


def _not_found(registry: ProjectRegistry, key: str) -> HTTPException:
    """The 404, with up to two prefix-matches of what was typed as a remedy.

    "try /project list" tells a user nothing about a typo; naming the nearest
    names is the difference between a dead end and a fix. Only offered when
    something actually shares a leading fragment, because an unrelated list
    would be noise.
    """
    prefix = key.strip().casefold()[:4]
    near = (
        [
            project.name
            for project in registry.list_projects()
            if prefix and project.name.casefold().startswith(prefix)
        ][:2]
        if prefix
        else []
    )
    names = f" — closest: {', '.join(near)}" if near else ""
    return HTTPException(
        404,
        {
            "code": "project_not_found",
            "message": f"no project with id or name {key!r}{names}",
        },
    )


def _refusal(exc: Exception, key: str) -> HTTPException:
    """Map one store refusal to the response the client can act on."""
    if isinstance(exc, ProjectNameConflictError):
        # The name is taken (case-insensitive): a STATE conflict, not a
        # malformed value — the client's move is to pick another name or open
        # the existing project, which 409 (with a machine code) says and 422
        # does not.
        return HTTPException(409, {"code": "project_name_exists", "message": str(exc)})
    if isinstance(exc, ProjectSchemaGuardError):
        # The row EXISTS and is readable; this build is too old to rewrite it.
        # 409, not 404 (the operator can see it) and not 422 (nothing about the
        # request is malformed): the state of the target is what refuses.
        return HTTPException(409, {"code": "project_schema_newer", "message": str(exc)})
    if isinstance(exc, ProjectRegistryLockTimeout):
        # Contention is transient; the client retries.
        return HTTPException(503, {"code": "project_store_busy", "message": str(exc)})
    return HTTPException(422, {"code": "project_invalid", "message": readable_error(exc)})


def _live_counts(registry: ProjectRegistry, projects: list[Project]) -> dict[str, int]:
    """Live-session counts for a batch of projects, from ONE runtime scan."""
    if not projects:
        return {}
    states = scan_runtime_states(registry.config_dir)
    return {
        project.id: sum(1 for sid in project.sessions if states.get(sid, {}).get("state") == "live")
        for project in projects
    }


@router.get("/v1/desktop/projects", response_model=CRUDResponse)
async def projects(request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def read() -> ProjectList:
            registry = _registry(request)
            listed = registry.list_projects()
            live = _live_counts(registry, listed)
            rows = [
                project_summary(project, live_sessions=live.get(project.id, 0))
                for project in listed
            ]
            rows.sort(key=lambda row: (_STATUS_RANK.get(row.status, 99), -row.updated_at))
            return ProjectList(projects=rows)

        return reply(await asyncio.to_thread(read))


@router.post("/v1/desktop/projects", response_model=CRUDResponse)
async def create(body: ProjectCreate, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectSummary:
            registry = _registry(request)
            try:
                # Construction inside the try: the edit model runs the same
                # validators as the store, and its ValidationError must reach
                # this route's 422 rather than `errors()`' bare 409.
                fields = ProjectEdit(
                    name=body.name,
                    description=body.description,
                    status=body.status,
                    tags=body.tags,
                )
                created = registry.create_project(fields)
            except (
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                raise _refusal(exc, body.name) from exc
            # A route-created project starts unlinked by design (§4.1's body
            # carries no sessions), so the live count is 0 without a scan.
            return project_summary(created, live_sessions=0)

        return reply(await asyncio.to_thread(mutate))


@router.get("/v1/desktop/projects/{key}", response_model=CRUDResponse)
async def detail(key: str, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def read() -> ProjectDetail:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            view = build_project_view(found, config_dir=registry.config_dir)
            return ProjectDetail(
                project=project_view(found),
                links=[linked_session_view(row) for row in view["sessions"]],
            )

        return reply(await asyncio.to_thread(read))


@router.patch("/v1/desktop/projects/{key}", response_model=CRUDResponse)
async def patch(key: str, body: ProjectPatch, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectSummary:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            # Only the keys the caller actually sent: an omitted key means
            # "leave it alone", and the store's edit vocabulary reads the same
            # absence from `model_fields_set`.
            payload = {field: getattr(body, field) for field in body.model_fields_set}
            try:
                # The EDIT MODEL is built inside the try too: it runs the same
                # validators (dates, tag grammar, estimate bounds), and a
                # ValidationError escaping to `errors()` would be answered by
                # its ``(ReceiptConflict, ValueError)`` arm — a bare 409 string,
                # not this route's 422 sentence.
                fields = ProjectEdit(**payload)
                outcome = registry.update_project(found.id, fields, reporter="operator")
            except (
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                raise _refusal(exc, key) from exc
            updated = outcome.project
            live = _live_counts(registry, [updated])
            return project_summary(updated, live_sessions=live.get(updated.id, 0))

        return reply(await asyncio.to_thread(mutate))


@router.delete("/v1/desktop/projects/{key}", response_model=CRUDResponse)
async def remove(key: str, body: ProjectDelete, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectDeleted:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            if body.confirm.strip().casefold() != found.name.casefold():
                raise HTTPException(
                    422,
                    {
                        "code": "project_confirm_mismatch",
                        "message": (
                            f"confirm must repeat the project name {found.name!r} "
                            "exactly (the name, not the id)"
                        ),
                    },
                )
            try:
                registry.delete_project(found.id)
            except ProjectRegistryLockTimeout as exc:
                raise _refusal(exc, key) from exc
            return ProjectDeleted()

        return reply(await asyncio.to_thread(mutate))


@router.post("/v1/desktop/projects/{key}/links", response_model=CRUDResponse)
async def link(key: str, body: LinkMutation, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectSummary:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            try:
                project, _added = registry.link_session(found.id, body.session_id)
            except (ValueError, ValidationError, ProjectRegistryLockTimeout) as exc:
                raise _refusal(exc, key) from exc
            live = _live_counts(registry, [project])
            return project_summary(project, live_sessions=live.get(project.id, 0))

        return reply(await asyncio.to_thread(mutate))


@router.delete("/v1/desktop/projects/{key}/links/{session_id}", response_model=CRUDResponse)
async def unlink(key: str, session_id: str, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectSummary:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            try:
                project, _removed = registry.unlink_session(found.id, session_id)
            except (ValueError, ValidationError, ProjectRegistryLockTimeout) as exc:
                raise _refusal(exc, key) from exc
            live = _live_counts(registry, [project])
            return project_summary(project, live_sessions=live.get(project.id, 0))

        return reply(await asyncio.to_thread(mutate))


@router.post("/v1/desktop/projects/{key}/milestones", response_model=CRUDResponse)
async def milestone(key: str, body: MilestoneMutation, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectView:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            try:
                fields = MilestoneEdit(
                    **{field: getattr(body, field) for field in body.model_fields_set}
                )
                project, _action = registry.set_milestone(found.id, fields)
            except (
                KeyError,
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                raise _refusal(exc, key) from exc
            return project_view(project)

        return reply(await asyncio.to_thread(mutate))


@router.delete("/v1/desktop/projects/{key}/milestones/{name}", response_model=CRUDResponse)
async def milestone_remove(key: str, name: str, request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def mutate() -> ProjectView:
            registry = _registry(request)
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            try:
                project, _action = registry.set_milestone(
                    found.id, MilestoneEdit(name=name, remove=True)
                )
            except (
                KeyError,
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                raise _refusal(exc, key) from exc
            return project_view(project)

        return reply(await asyncio.to_thread(mutate))
