"""The per-session file index: what files did this conversation (and its subagents) produce?

Two routes:

- ``GET  /v1/desktop/sessions/{session_id}/files`` - the paged, filtered, sorted list.
- ``POST /v1/desktop/sessions/{session_id}/files/{file_id}/materialise`` - copy a digest-only
  artifact (a generated image or video, which has NO filesystem path) to a real file under the
  session's scratchpad, so "open", "reveal" and "copy path" have something to act on.

WHY THIS IS A BACKEND ROUTE. The renderer used to derive the list by scanning transcript text
for path-shaped strings, which can never see a generated image (the tool result carries a
content-addressed digest, not a path), stops after a fixed number of messages, and loses
anything compaction blanked. The authoritative inventory is derived from the journal by
``session_files`` and cached outside the session directory, exactly as ``code_requests`` does
for pull requests - and this module copies that route's stance:

* **No session bridge.** The index is read from the journal on disk, so a cold session answers
  as fast as a live one and a read never starts, wakes or attaches a runtime. Admission is the
  door's own rule (``DesktopSessions.files_target``): the same 404 for an unknown id, the same
  409 for a silent peer; a registered DRAFT answers an empty ``ready`` page (like ``find`` and
  ``history``, not a 404) and a PEER's conversation answers ``state: "unsupported"`` with no
  rows, because its journal is not on this disk - the renderer falls back to its own scan.
* **GET never blocks past a budget.** A cold scan of a 300 MB journal runs in a worker thread;
  the request waits ~1.5 s, then answers ``state: "building"`` with the rows found so far
  while a single-flight background task finishes the scan (resumable across restarts - the
  cache stores the byte offset). The client re-polls.
* **Absolute paths on the wire.** The route is bearer-gated and the renderer already holds
  absolute paths from its transcript; a mesh or phone surface must NOT reuse this route's
  projection - it calls the service with ``display=True``.

Capability: ``features.session_files == 1`` (``server/features.py``). Absent means the surface
does not exist; the renderer keeps its own scan and must not call the route.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_session_files import (
    FilesListing,
    MaterialiseReceipt,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import errors, host, reply
from local_operator.server.utils.desktop_sessions import contained_descendants
from local_operator.session_files import service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop session files"], dependencies=[Depends(require_desktop)])

#: The session-id shape every desktop session route declares, so a malformed handle is refused
#: by FastAPI (422) before any handler runs.
SessionID = Annotated[str, PathParam(pattern=r"^[a-f0-9]{12}$")]
FileID = Annotated[str, PathParam(pattern=r"^[a-f0-9]{16}$")]

#: In-flight background completions, keyed ``(config_dir, session_id, scope)``. Single-flight:
#: a 1 s client poll that arrives while a scan runs joins it rather than queueing another. The
#: strong reference is the point - a bare ``create_task`` can be collected mid-flight.
_IN_FLIGHT: dict[tuple[str, str, str], "asyncio.Task[Any]"] = {}


def _config_dir(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


def _bad_filter(field: str, value: str) -> HTTPException:
    return HTTPException(
        422,
        _wd.session_files_bad_filter(field=field, value=value).envelope(
            code="session_files_bad_filter"
        ),
    )


async def _family(config_dir: Path, session_id: str, directory: Path) -> service.Family:
    children, skipped = await asyncio.to_thread(contained_descendants, config_dir, session_id)
    return service.Family(
        session_id=session_id,
        session_dir=directory,
        children=tuple(children),
        skipped=skipped,
    )


def _kick(config_dir: Path, session_id: str, scope: str, family: service.Family) -> None:
    """Finish a ``building`` scan in the background, once per (session, scope)."""
    key = (str(config_dir), session_id, scope)
    running = _IN_FLIGHT.get(key)
    if running is not None and not running.done():
        return

    async def _run() -> None:
        try:
            await asyncio.to_thread(
                service.complete_in_background,
                config_dir,
                family,
                scope,
                home=str(Path.home()),
                deny=service.make_deny(config_dir),
            )
        except Exception:  # noqa: BLE001 - a background completion is an optimisation
            logger.debug("session-files background scan failed", exc_info=True)
        finally:
            _IN_FLIGHT.pop(key, None)

    _IN_FLIGHT[key] = asyncio.create_task(_run())


@router.get(
    "/v1/desktop/sessions/{session_id}/files",
    response_model=CRUDResponse[FilesListing],
)
async def list_session_files(
    session_id: SessionID,
    request: Request,
    scope: str = Query(default="family"),
    kind: str | None = Query(default=None, max_length=256),
    origin: str | None = Query(default=None, max_length=256),
    highlighted: bool | None = Query(default=None),
    q: str = Query(default="", max_length=service.MAX_QUERY_CHARS),
    sort: str = Query(default="recent"),
    order: str | None = Query(default=None),
    limit: int = Query(default=service.DEFAULT_LIMIT, ge=1, le=service.MAX_LIMIT),
    cursor: str | None = Query(default=None, max_length=512),
):
    """The files this conversation produced, newest first by default.

    ``scope=family`` (default) includes the conversation's subagents (depth <= 3, <= 64
    children, each PROVEN to be this conversation's by its roster); ``scope=self`` is the
    conversation alone. ``kind``/``origin`` are comma lists; ``highlighted=1`` keeps the files
    the turn-summary "Highlights" flagged; ``q`` is a case-folded substring over name and
    display path. ``sort`` is one of ``recent|name|kind|modified|created|size`` and ``order``
    ``asc|desc`` (default per sort). ``cursor`` is the opaque ``next_cursor`` of the previous
    page; a cursor from another query is 422 ``session_files_bad_cursor`` and one from an
    older revision restarts at the top with ``cursor_reset: true``.

    ``state``: ``ready`` (complete), ``building`` (partial; poll again), ``unsupported`` (a
    peer's conversation). Unknown id: 404. A silent peer might hold the id: 409, as on every
    other desktop read.
    """
    if scope not in service.SCOPES:
        raise _bad_filter("scope", scope)
    if sort not in service.SORTS:
        raise _bad_filter("sort", sort)
    if order is not None and order not in service.ORDERS:
        raise _bad_filter("order", order)
    try:
        query = service.Query(
            scope=scope,
            kinds=service.parse_csv(kind),
            origins=service.parse_csv(origin, vocabulary=service.ORIGINS),
            highlighted=highlighted,
            q=q.strip(),
            sort=sort,
            order=order,
            limit=limit,
            cursor=cursor or None,
        )
    except service.InvalidFilter as error:
        field = "origin" if origin and str(error) in origin.lower() else "kind"
        raise _bad_filter(field, str(error)) from None

    config_dir = _config_dir(request)
    async with errors(request):
        kind_of_target, directory = await host(request).files_target(session_id)
    if kind_of_target == "peer":
        return reply(service.empty_listing(query, state="unsupported"))
    if directory is None:
        return reply(service.empty_listing(query))

    family = await _family(config_dir, session_id, directory)
    try:
        answer = await asyncio.to_thread(
            service.list_files, config_dir, family, query, budget_s=service.DEFAULT_BUDGET_S
        )
    except service.InvalidCursor:
        raise HTTPException(
            422,
            _wd.session_files_bad_cursor().envelope(code="session_files_bad_cursor"),
        ) from None
    if answer["state"] == "building":
        _kick(config_dir, session_id, query.scope, family)
    return reply(answer)


@router.post(
    "/v1/desktop/sessions/{session_id}/files/{file_id}/materialise",
    response_model=CRUDResponse[MaterialiseReceipt],
)
async def materialise_session_file(session_id: SessionID, file_id: FileID, request: Request):
    """Copy one digest-only artifact to ``<session scratchpad>/generated/<name>``.

    POST only and session-confined: the destination is built server-side from the URL
    session's own scratchpad and a sanitised name; the client names a ROW, never a path. A
    row that already has a filesystem path answers 409 ``session_files_not_an_artifact``; an
    id that is not in this conversation's family answers 404 ``session_files_file_not_found``;
    a store entry that was pruned answers 410 ``session_files_artifact_missing``.
    Idempotent: an identical existing copy is reused (``created: false``).
    """
    config_dir = _config_dir(request)
    async with errors(request):
        kind_of_target, directory = await host(request).files_target(session_id)
    if kind_of_target != "local" or directory is None:
        # A draft has no journal to hold a file; a peer's artifact bytes are on the peer.
        raise HTTPException(
            404,
            _wd.session_files_file_not_found().envelope(code="session_files_file_not_found"),
        )
    family = await _family(config_dir, session_id, directory)
    try:
        receipt = await asyncio.to_thread(service.materialise, config_dir, family, file_id)
    except service.MaterialiseError as error:
        status, message = _MATERIALISE_REFUSALS.get(
            error.code, (500, _wd.session_files_materialise_failed(reason=str(error)))
        )
        if error.code == "materialise_failed":
            message = _wd.session_files_materialise_failed(reason=str(error))
        raise HTTPException(status, message.envelope(code=f"session_files_{error.code}")) from None
    return reply(receipt)


_MATERIALISE_REFUSALS = {
    "file_not_found": (404, _wd.session_files_file_not_found()),
    "not_an_artifact": (409, _wd.session_files_not_an_artifact()),
    "artifact_bytes_missing": (410, _wd.session_files_artifact_missing()),
}
