"""``/v1/desktop/projects*`` — the desktop CRUD, links and milestones surface.

Nine + two + one routes over the project store (:mod:`local_operator.projects`;
request-update is the one that also dials — see below):

- ``GET    /v1/desktop/projects``                        — the listing (summaries)
- ``GET    /v1/desktop/projects/search``                 — ranked soft matches (``q``, ``limit``)
- ``GET    /v1/desktop/projects/timeline``               — one milestones document
- ``POST   /v1/desktop/projects``                        — create
- ``GET    /v1/desktop/projects/{key}``                  — one project + its links
- ``PATCH  /v1/desktop/projects/{key}``                  — partial edit
- ``DELETE /v1/desktop/projects/{key}``                  — confirm-by-name delete
- ``POST   /v1/desktop/projects/{key}/links``            — link one session
- ``DELETE /v1/desktop/projects/{key}/links/{sid}``      — unlink one session
- ``POST   /v1/desktop/projects/{key}/milestones``       — add-or-update one
- ``DELETE /v1/desktop/projects/{key}/milestones/{name}``— remove one
- ``POST   /v1/desktop/projects/{key}/request-update``   — ask the linked sessions for an update

**The store is the only writer.** Every route calls the same
:class:`~local_operator.projects.ProjectRegistry` methods the ``project`` tool
calls, so the two surfaces cannot validate differently, and the composed view
(``{project, links}``) comes from the same ``build_project_view`` the tool's
``show`` renders — no second derivation of "what are the linked sessions
doing".

**Two derived reads.** ``search`` ranks over the rows with the ONE ranking
model (:mod:`local_operator.projects_search`) — soft match, per-field weights
(name/title > description > updates), a deterministic total order; ``timeline``
answers the Timeline view's whole document (every row's milestones, statuses
derived) so the client stops fanning out one ``projects.get`` per row. Both
static paths are declared BEFORE ``/v1/desktop/projects/{key}`` — FastAPI
matches in declaration order, and a ``{key}`` route declared first would
swallow them; the sessions search route carries the same load-bearing note.

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
same sentence the tool receives, via ``projects.readable_error``), with
``project_done_incomplete`` (plus the ``incomplete`` milestone names) for the
one refusal that has a deliberate way through — ``PATCH`` with ``status: done``
over open milestones, resent with ``force_done: true``; 503 the store lock
timed out, which is retryable.

**One route talks to other sessions.** ``POST .../{key}/request-update`` is the
single exception to the rule below: it hands each linked session one check-in
through the shared peer-send core (:mod:`local_operator.server.request_update`)
and answers only after every dial has settled, so the sends complete even if the
UI closes. It mutates no session directory — the receive side owns that — and
every other route here still talks to no session: their linked-session rows are
read from files, exactly as the tool's view is.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError

from local_operator.projects import (
    MilestoneEdit,
    Project,
    ProjectDoneGateError,
    ProjectEdit,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    build_project_view,
    closed_with_open_milestones,
    display_name,
    readable_error,
    scan_runtime_states,
    stale_after_s,
)
from local_operator.projects_search import search_projects
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_projects import (
    STATUS_RANK,
    LinkMutation,
    MilestoneMutation,
    ProjectCreate,
    ProjectDelete,
    ProjectDeleted,
    ProjectDetail,
    ProjectList,
    ProjectPatch,
    ProjectSearchHit,
    ProjectSearchResults,
    ProjectSummary,
    ProjectTimeline,
    ProjectView,
    linked_session_view,
    project_patched,
    project_summary,
    project_timeline_entry,
    project_view,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.request_update import request_updates
from local_operator.server.routes.desktop_sessions import errors, reply

router = APIRouter(tags=["Desktop projects"], dependencies=[Depends(require_desktop)])


def _registry(request: Request) -> ProjectRegistry:
    """A fresh adapter over ``<config_dir>/projects`` for one request.

    Fresh metadata per operation (the ``desktop_profiles`` convention): another
    process's write is visible without waiting for any polling cache. This is an
    adapter over the same directory the tool writes, not a second registry.
    """
    return ProjectRegistry(request.app.state.config_manager.config_dir)


def _window(registry: ProjectRegistry) -> float:
    """The configured staleness window for this request's config root.

    Resolved ONCE per handler and passed to every ``project_summary``/
    ``project_view`` call there, so every row of one payload is measured
    against the same boundary ("server-computed once").
    """
    return stale_after_s(registry.config_dir)


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
    if isinstance(exc, ProjectDoneGateError):
        # Still 422 (the request is well-formed; it asks for a state the plan's
        # milestones contradict), but with its OWN machine code: the remedy is
        # a deliberate choice — complete the milestones or resend with
        # `force_done: true` — which a client can only offer if it can tell this
        # refusal from a malformed value. `message` is the store's exact
        # sentence; `incomplete` names what the confirm dialog should list.
        # This mapping is shared with the create routes: they would emit the
        # same code if create ever carried milestones (it cannot today).
        return HTTPException(
            422,
            {
                "code": "project_done_incomplete",
                "message": readable_error(exc),
                "incomplete": list(exc.incomplete),
            },
        )
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


def _listing_order(rows: list[Project]) -> list[Project]:
    """The listing's own order — status rank, then newest first.

    ONE expression for the listing and its derived siblings: the search
    route's empty-query pass-through and the timeline document both answer in
    "the listing's own order, untouched", so they must not grow a second sort
    that could drift from it.
    """
    return sorted(
        rows, key=lambda project: (STATUS_RANK.get(project.status, 99), -project.updated_at)
    )


@router.get("/v1/desktop/projects", response_model=CRUDResponse)
async def projects(request: Request) -> CRUDResponse[Any]:
    async with errors(request):

        def read() -> ProjectList:
            registry = _registry(request)
            listed = _listing_order(registry.list_projects())
            live = _live_counts(registry, listed)
            rows = [
                project_summary(
                    project, live_sessions=live.get(project.id, 0), window=_window(registry)
                )
                for project in listed
            ]
            return ProjectList(projects=rows)

        return reply(await asyncio.to_thread(read))


# NOTE THE DECLARATION ORDER — load-bearing. The two static siblings are
# declared BEFORE ``/v1/desktop/projects/{key}`` below, because FastAPI
# matches routes in declaration order: a ``{key}`` route declared first would
# swallow ``search``/``timeline`` and answer with the snapshot of a project
# literally named that (a 404, in practice). The sessions search route carries
# the same load-bearing comment.
@router.get("/v1/desktop/projects/search", response_model=CRUDResponse)
async def search(
    request: Request,
    q: str = Query(default="", max_length=256),
    limit: int = Query(default=50, ge=1, le=200),
) -> CRUDResponse[Any]:
    """Ranked soft matches over the rows — see :mod:`local_operator.projects_search`.

    The tiers are the session-search contract (casefold + diacritic fold,
    prefix, bounded typo on 4+ char tokens, order-independent token-AND across
    fields) with the operator's field weights on top: title/name > description
    > tags > people > updates > progress. ``q`` is bounded at 256 characters
    (the sessions precedent) and ``limit`` at 200; the answer echoes the query
    so the client can apply only the answer whose echo equals the box.

    An EMPTY ``q`` is not a search: the answer is the LISTING's own order
    truncated to ``limit`` (the same truncation every answer carries), each row
    with score 0 and no matched fields — mirroring ``search_store``'s empty
    arm, so a caller can render this answer directly for an empty box without
    re-ranking anything.
    """
    async with errors(request):

        def read() -> ProjectSearchResults:
            registry = _registry(request)
            rows = registry.list_projects()
            if not q.strip():
                hits = [
                    ProjectSearchHit(
                        id=project.id, name=display_name(project), score=0.0, fields=[]
                    )
                    for project in _listing_order(rows)[:limit]
                ]
            else:
                hits = [
                    ProjectSearchHit(
                        id=match.id,
                        name=match.name,
                        score=match.score,
                        fields=list(match.fields),
                    )
                    for match in search_projects(rows, q, limit=limit)
                ]
            return ProjectSearchResults(projects=hits, query=q, count=len(hits))

        return reply(await asyncio.to_thread(read))


@router.get("/v1/desktop/projects/timeline", response_model=CRUDResponse)
async def timeline(request: Request) -> CRUDResponse[Any]:
    """The ONE timeline document: every row's milestones, statuses derived.

    Replaces the Timeline view's per-row fan-out (one ``projects.get`` per row,
    each of which runs a runtime scan plus per-linked-session reads) with a
    single read: this route touches no session file and dials no runtime — the
    milestones are already on the rows the listing parses. Status is derived
    server-side so the client stays a painter.
    """
    async with errors(request):

        def read() -> ProjectTimeline:
            registry = _registry(request)
            listed = _listing_order(registry.list_projects())
            return ProjectTimeline(projects=[project_timeline_entry(project) for project in listed])

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
            return project_summary(created, live_sessions=0, window=_window(registry))

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
                project=project_view(found, window=_window(registry)),
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
            # `force_done` is a flag about THIS call, not a field of the row:
            # it never reaches `ProjectEdit` (which would not know it) and
            # rides the registry keyword instead — the same door the tool uses.
            force_done = bool(payload.pop("force_done", False))
            try:
                # The EDIT MODEL is built inside the try too: it runs the same
                # validators (dates, tag grammar, estimate bounds), and a
                # ValidationError escaping to `errors()` would be answered by
                # its ``(ReceiptConflict, ValueError)`` arm — a bare 409 string,
                # not this route's 422 sentence.
                fields = ProjectEdit(**payload)
                outcome = registry.update_project(
                    found.id, fields, reporter="operator", force_done=force_done
                )
            except (
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                raise _refusal(exc, key) from exc
            updated = outcome.project
            live = _live_counts(registry, [updated])
            # The store persists nothing about a forced close, so this answer
            # is the client's only chance to say "closed with N milestones
            # open" (see `ProjectPatched`).
            return project_patched(
                updated,
                live_sessions=live.get(updated.id, 0),
                window=_window(registry),
                forced_done=closed_with_open_milestones(
                    updated, force_done=force_done, was_done=found.status == "done"
                ),
            )

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
            except (ProjectSchemaGuardError, ProjectRegistryLockTimeout) as exc:
                # The write guard subclasses RuntimeError, and ``errors()``'s
                # generic RuntimeError arm answers 503 runtime_unreachable —
                # a reconnect remedy that can never work while the true
                # remedy ("update this build") is lost. Caught HERE so the
                # refusal matrix holds on every mutating route (QA round 1,
                # Q1: delete/link/unlink were the three that missed it).
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
            except (
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                # ``ProjectSchemaGuardError`` last-but-named for the reason the
                # delete route states: it subclasses RuntimeError, so an
                # uncaught guard reaches ``errors()``'s 503 arm.
                raise _refusal(exc, key) from exc
            live = _live_counts(registry, [project])
            return project_summary(
                project, live_sessions=live.get(project.id, 0), window=_window(registry)
            )

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
            except (
                ValueError,
                ValidationError,
                ProjectSchemaGuardError,
                ProjectRegistryLockTimeout,
            ) as exc:
                raise _refusal(exc, key) from exc
            live = _live_counts(registry, [project])
            return project_summary(
                project, live_sessions=live.get(project.id, 0), window=_window(registry)
            )

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
            return project_view(project, window=_window(registry))

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
            return project_view(project, window=_window(registry))

        return reply(await asyncio.to_thread(mutate))


@router.post("/v1/desktop/projects/{key}/request-update", response_model=CRUDResponse)
async def request_update(key: str, request: Request) -> CRUDResponse[Any]:
    """Ask the project's LINKED sessions to post a progress update.

    One check-in per linked session (``project.sessions`` only — never the
    coordination half), delivered through the shared peer-send core as a
    mailbox drop with ``wake=True``, strictly sequentially, and awaited here so
    the sends complete even if the UI closes (design freeze). Per-session
    outcomes are three-way (``delivered``/``unconfirmed``/``failed``); a batch
    never fails whole. A 60 s per-project cooldown refuses a repeat, with the
    same sentence the UI shows. See :mod:`local_operator.server.request_update`
    for the loop, the frozen message and the outcome vocabulary.

    ``message`` is the governing sentence for the state, so the response is
    built here rather than through ``reply()``: its fixed session-flavoured
    line would replace a cooldown refusal the client must read character for
    character.
    """
    async with errors(request):
        registry = _registry(request)

        def resolve() -> Project:
            found = _find(registry, key)
            if found is None:
                raise _not_found(registry, key)
            return found

        project = await asyncio.to_thread(resolve)
        message, result = await request_updates(registry, project)
        return CRUDResponse(status=200, message=message, result=result)
