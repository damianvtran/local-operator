"""Desktop Library routes (design §4, minus the later Librarian slice).

Every route is bearer-gated and answers in the usual ``CRUDResponse`` envelope.
Two rules are enforced HERE, at the boundary, rather than trusted from the
service:

* an asset/tag/session handle is shape-checked by FastAPI before a handler runs;
* a content request carries an asset ID ONLY. There is no path parameter that
  could be coerced into a filesystem read, and the service re-checks the
  indexed path against the denylist on every byte.

The relay reuses the same service with ``display=True``; this router is the
desktop projection (absolute ``ref.path``) and must not be mounted on a phone
plane.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Iterator, Literal
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request
from fastapi.responses import StreamingResponse

from local_operator.i18n.keys import wire_errors_desktop as _wd
from local_operator.library import index as library_index
from local_operator.library import service
from local_operator.library.metadata import (
    MAX_TAGS_PER_ASSET,
    MetadataStore,
    MetadataUnavailable,
    TagRefused,
)
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_library import (
    LibraryAssetPatch,
    LibraryChatReceipt,
    LibraryChatWrite,
    LibraryListing,
    LibraryRow,
    LibraryTag,
    LibraryTagCreate,
    LibraryTagMerge,
    LibraryTagPatch,
    LibraryTagsWrite,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop library"], dependencies=[Depends(require_desktop)])

AssetID = Annotated[str, PathParam(pattern=r"^la_[0-9a-f]{16}$")]
TagID = Annotated[str, PathParam(pattern=r"^[a-f0-9]{10}$")]
SessionIDShape = re.compile(r"^[a-f0-9]{12}$")

_CHUNK = 64 * 1024

#: One background finisher per config root. A poll that arrives while the
#: catalog is still building joins the running task instead of starting a
#: second full journal pass — the same single-flight rule as session files.
_IN_FLIGHT: dict[str, asyncio.Task[Any]] = {}


def _config_dir(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


def _store(request: Request) -> MetadataStore:
    return MetadataStore(_config_dir(request))


def _bad_filter(field: str, value: str) -> HTTPException:
    return HTTPException(
        422,
        _wd.library_bad_filter(field=field, value=value).envelope(code="library_bad_filter"),
    )


def _not_found() -> HTTPException:
    return HTTPException(
        404, _wd.library_asset_not_found().envelope(code="library_asset_not_found")
    )


def _unavailable() -> HTTPException:
    return HTTPException(
        503,
        _wd.library_metadata_unavailable().envelope(code="library_metadata_unavailable"),
    )


def _metadata_refusal(error: MetadataUnavailable) -> HTTPException:
    logger.debug("library metadata unavailable: %s", error)
    return _unavailable()


_FIELD_BY_CODE = {
    "bad_organized": "organized",
    "bad_name": "name",
    "name_required": "name",
    "description_invalid": "description",
    "bad_description": "description",
    "asset_locked": "locked",
}


def _field_refusal(error: TagRefused) -> HTTPException:
    """A malformed PATCH field is a client refusal, never a 500.

    ``TagRefused`` is the store's vocabulary; a route that lets it escape turns
    a typo (``organized: "garbage"``) or an explicit null name into an internal
    error, which reads to the client as a broken backend rather than a bad body.
    """
    field = _FIELD_BY_CODE.get(error.code, "body")
    return _bad_filter(field, error.code)


def _tag_refusal(error: TagRefused) -> HTTPException:
    if error.code in ("unknown_tag", "tag_unknown"):
        return HTTPException(404, _wd.library_tag_unknown().envelope(code="library_tag_unknown"))
    if error.code == "tag_limit":
        return HTTPException(
            422,
            _wd.library_tag_limit(max=MAX_TAGS_PER_ASSET).envelope(code="library_tag_limit"),
        )
    if error.code == "bad_category":
        return HTTPException(
            422,
            _wd.library_tag_category(category=str(error)).envelope(code="library_tag_category"),
        )
    # Bad grammar, unknown actor, or an empty list: all client-side refusals.
    return HTTPException(
        422,
        _wd.library_tag_name(name=str(error)).envelope(code="library_tag_name"),
    )


def _kick(config: Path) -> None:
    key = str(config)
    running = _IN_FLIGHT.get(key)
    if running is not None and not running.done():
        return

    async def _run() -> None:
        try:
            await asyncio.to_thread(library_index.complete_in_background, config)
        except Exception:  # noqa: BLE001 - a background finish is an optimisation
            logger.debug("library background build failed", exc_info=True)
        finally:
            _IN_FLIGHT.pop(key, None)

    _IN_FLIGHT[key] = asyncio.create_task(_run())


async def _query_from(
    *,
    q: str,
    category: str | None,
    kind: str | None,
    origin_group: str | None,
    origin: str | None,
    tag: str | None,
    tag_mode: str,
    highlighted: bool | None,
    organized: bool | None,
    favorite: bool | None,
    session: str | None,
    session_id: str | None,
    scope: str,
    sort: str,
    order: str | None,
    limit: int,
    cursor: str | None,
) -> service.Query:
    if tag_mode not in service.TAG_MODES:
        raise _bad_filter("tag_mode", tag_mode)
    if scope not in service.SCOPES:
        raise _bad_filter("scope", scope)
    if sort not in service.SORTS:
        raise _bad_filter("sort", sort)
    if order is not None and order not in service.ORDERS:
        raise _bad_filter("order", order)
    try:
        session_ids = service.parse_csv(session_id)
    except service.InvalidFilter as error:
        raise _bad_filter("session_id", str(error)) from None
    for value in session_ids:
        if not SessionIDShape.match(value):
            raise _bad_filter("session_id", value)
    if session is not None and not SessionIDShape.match(session):
        raise _bad_filter("session", session)
    try:
        return service.Query(
            q=q.strip(),
            categories=service.parse_csv(category, vocabulary=service.CATEGORIES),
            kinds=service.parse_csv(kind, vocabulary=service.KINDS),
            origin_groups=service.parse_csv(origin_group, vocabulary=library_index.ORIGIN_GROUPS),
            origins=service.parse_csv(origin, vocabulary=service.ORIGINS),
            tag=tag or None,
            tag_mode=tag_mode,
            highlighted=highlighted,
            organized=organized,
            favorite=favorite,
            session=session or None,
            session_ids=session_ids,
            scope=scope,
            sort=sort,
            order=order,
            limit=limit,
            cursor=cursor or None,
        )
    except service.InvalidFilter as error:
        field = "origin" if origin and str(error) in origin.lower() else "filter"
        raise _bad_filter(field, str(error)) from None


def _asset_not_found() -> HTTPException:
    return _not_found()


@router.get("/v1/desktop/library/assets", response_model=CRUDResponse[LibraryListing])
async def list_library_assets(
    request: Request,
    q: str = Query(default="", max_length=service.MAX_QUERY_CHARS),
    category: str | None = Query(default=None, max_length=256),
    kind: str | None = Query(default=None, max_length=256),
    origin_group: str | None = Query(default=None, max_length=256),
    origin: str | None = Query(default=None, max_length=256),
    tag: str | None = Query(default=None, max_length=64),
    tag_mode: str = Query(default="all"),
    highlighted: bool | None = Query(default=None),
    organized: bool | None = Query(default=None),
    favorite: bool | None = Query(default=None),
    session: str | None = Query(default=None, max_length=12),
    session_id: str | None = Query(default=None, max_length=512),
    scope: Literal["all", "final"] = Query(default="all"),
    sort: str = Query(default="recent"),
    order: str | None = Query(default=None),
    limit: int = Query(default=service.DEFAULT_LIMIT, ge=1, le=service.MAX_LIMIT),
    cursor: str | None = Query(default=None, max_length=512),
):
    """The global catalog, newest first by default.

    Filters are comma lists where the vocabulary has more than one value
    (``category``, ``kind``, ``origin_group``, ``origin``). ``session`` selects
    one source conversation; ``session_id`` is the same filter as a comma list,
    so a picker can ask for several. ``tag`` is a tag id; ``tag_mode=all|any``
    decides whether every named tag (the default, one id) or any must match.
    ``cursor`` is the opaque ``next_cursor`` of the previous page; a cursor from
    another query is 422 and one from an older revision restarts at the top with
    ``cursor_reset: true``.

    ``scope=final`` narrows the catalog to what the operator kept — highlighted
    assets and generated artifacts — before metadata is materialised, so
    ``total``, ``counts`` and ``tags`` describe that set. The default stays
    ``all`` for API compatibility; the desktop UI asks for ``final``.
    """
    query = await _query_from(
        q=q,
        category=category,
        kind=kind,
        origin_group=origin_group,
        origin=origin,
        tag=tag,
        tag_mode=tag_mode,
        highlighted=highlighted,
        organized=organized,
        favorite=favorite,
        session=session,
        session_id=session_id,
        scope=scope,
        sort=sort,
        order=order,
        limit=limit,
        cursor=cursor,
    )
    config = _config_dir(request)
    try:
        answer = await asyncio.to_thread(service.list_assets, config, query)
    except service.InvalidCursor:
        raise HTTPException(
            422,
            _wd.library_bad_cursor().envelope(code="library_bad_cursor"),
        ) from None
    if answer["state"] == "building":
        _kick(config)
    return reply(answer)


@router.get("/v1/desktop/library/assets/{asset_id}", response_model=CRUDResponse[LibraryRow])
async def get_library_asset(asset_id: AssetID, request: Request):
    try:
        row = await asyncio.to_thread(service.get_asset, _config_dir(request), asset_id)
    except service.AssetNotFound:
        raise _asset_not_found() from None
    return reply(row)


@router.patch("/v1/desktop/library/assets/{asset_id}", response_model=CRUDResponse[LibraryRow])
async def patch_library_asset(asset_id: AssetID, body: LibraryAssetPatch, request: Request):
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        raise _bad_filter("body", "no fields")
    try:
        row = await asyncio.to_thread(
            service.update_asset,
            _config_dir(request),
            asset_id,
            fields,
            actor="user",
            source="user",
        )
    except service.AssetNotFound:
        raise _asset_not_found() from None
    except TagRefused as error:
        raise _field_refusal(error) from None
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply(row)


@router.put(
    "/v1/desktop/library/assets/{asset_id}/tags",
    response_model=CRUDResponse[LibraryRow],
)
async def put_library_asset_tags(asset_id: AssetID, body: LibraryTagsWrite, request: Request):
    try:
        row = await asyncio.to_thread(
            service.set_tags,
            _config_dir(request),
            asset_id,
            body.tags,
            body.primary,
            actor="user",
        )
    except service.AssetNotFound:
        raise _asset_not_found() from None
    except TagRefused as error:
        raise _tag_refusal(error) from None
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply(row)


def _stream(path: Path, start: int, length: int) -> Iterator[bytes]:
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining > 0:
            chunk = handle.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _parse_range(header: str, size: int) -> tuple[int, int] | None:
    """``(start, end)`` inclusive for one satisfiable byte range, else ``None``.

    Only a single ``bytes=`` range is honoured. Multi-range answers are legal
    multipart responses, but no Library surface asks for one and a wrong
    multipart body is worse than a clean 200 whole-file answer.
    """
    if not header.startswith("bytes=") or "," in header:
        return None
    spec = header[6:].strip()
    if "-" not in spec:
        return None
    raw_start, raw_end = spec.split("-", 1)
    try:
        if raw_start == "":
            suffix = int(raw_end)
            if suffix <= 0:
                return None
            start = max(0, size - suffix)
            return start, size - 1
        start = int(raw_start)
        end = int(raw_end) if raw_end else size - 1
    except ValueError:
        return None
    if start < 0 or start > end or start >= size:
        return None
    return start, min(end, size - 1)


@router.get("/v1/desktop/library/assets/{asset_id}/content")
async def library_content(
    asset_id: AssetID,
    request: Request,
    download: bool = Query(default=False),
):
    """Bytes for one asset, resolved from the id and never from a caller path.

    ``Range`` answers 206 for video/audio scrubbing. The disposition is
    ``inline`` by default; ``?download=1`` switches it to ``attachment`` so a
    surface can offer save-a-copy without a second route.
    """
    config = _config_dir(request)
    try:
        content = await asyncio.to_thread(service.resolve_content, config, asset_id)
    except service.AssetNotFound:
        raise _asset_not_found() from None
    except service.ContentUnavailable as error:
        message = (
            _wd.library_content_denied()
            if error.code == "content_denied"
            else _wd.library_content_missing()
        )
        raise HTTPException(
            404 if error.code != "content_denied" else 403,
            message.envelope(code=f"library_{error.code}"),
        ) from None
    try:
        size = content.path.stat().st_size  # type: ignore[union-attr]
    except OSError:
        raise HTTPException(
            404,
            _wd.library_content_missing().envelope(code="library_content_missing"),
        ) from None
    disposition = "attachment" if download else "inline"
    headers = {
        "Content-Disposition": (f"{disposition}; filename*=UTF-8''{quote(content.name, safe='')}"),
        "X-Content-Type-Options": "nosniff",
        "Accept-Ranges": "bytes",
        "Cache-Control": (
            "private, max-age=31536000, immutable" if content.digest else "private, max-age=60"
        ),
    }
    range_header = request.headers.get("range")
    if range_header:
        span = _parse_range(range_header, size)
        if span is None:
            return StreamingResponse(
                iter(()),
                status_code=416,
                headers={**headers, "Content-Range": f"bytes */{size}"},
            )
        start, end = span
        length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(length)
        return StreamingResponse(
            _stream(content.path, start, length),  # type: ignore[arg-type]
            status_code=206,
            media_type=content.content_type,
            headers=headers,
        )
    headers["Content-Length"] = str(size)
    return StreamingResponse(
        _stream(content.path, 0, size),  # type: ignore[arg-type]
        media_type=content.content_type,
        headers=headers,
    )


@router.get("/v1/desktop/library/assets/{asset_id}/thumbnail")
async def library_thumbnail(
    asset_id: AssetID,
    request: Request,
    w: int = Query(default=service.THUMB_MAX_WIDTH, ge=1, le=service.THUMB_MAX_WIDTH),
):
    """Best-effort thumbnail. A non-image is a designed 404, not a blank pane."""
    try:
        content = await asyncio.to_thread(service.thumbnail, _config_dir(request), asset_id, w)
    except service.AssetNotFound:
        raise _asset_not_found() from None
    except service.ContentUnavailable as error:
        message = (
            _wd.library_not_an_image()
            if error.code == "not_an_image"
            else _wd.library_content_missing()
        )
        raise HTTPException(
            404 if error.code == "not_an_image" else 404,
            message.envelope(code=f"library_{error.code}"),
        ) from None
    try:
        data = content.path.read_bytes()  # type: ignore[union-attr]
    except OSError:
        raise HTTPException(
            404,
            _wd.library_content_missing().envelope(code="library_content_missing"),
        ) from None
    return StreamingResponse(
        iter((data,)),
        media_type=content.content_type,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=86400",
        },
    )


@router.put(
    "/v1/desktop/library/assets/{asset_id}/chat",
    response_model=CRUDResponse[LibraryChatReceipt],
)
async def put_library_chat(asset_id: AssetID, body: LibraryChatWrite, request: Request):
    """Record the caller-created conversation this asset is discussed in.

    The caller creates the session through the existing session route first;
    this route only verifies the id and the directory, so there is exactly one
    session-creation path per surface.
    """
    config = _config_dir(request)
    session_id = body.session_id
    if not SessionIDShape.match(session_id) or not (config / "sessions" / session_id).is_dir():
        raise HTTPException(
            422,
            _wd.library_chat_session_missing().envelope(code="library_chat_session_missing"),
        )
    try:
        await asyncio.to_thread(service.set_chat_link, config, asset_id, session_id)
        chat = await asyncio.to_thread(service.chat_link, config, asset_id)
    except service.AssetNotFound:
        raise _asset_not_found() from None
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply({"asset_id": asset_id, "chat": chat, "cleared": False})


@router.delete(
    "/v1/desktop/library/assets/{asset_id}/chat",
    response_model=CRUDResponse[LibraryChatReceipt],
)
async def delete_library_chat(asset_id: AssetID, request: Request):
    try:
        cleared = await asyncio.to_thread(service.clear_chat_link, _config_dir(request), asset_id)
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply({"asset_id": asset_id, "chat": None, "cleared": cleared})


@router.get("/v1/desktop/library/tags", response_model=CRUDResponse[list[LibraryTag]])
async def list_library_tags(request: Request):
    try:
        tags = await asyncio.to_thread(_store(request).list_tags)
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply(tags)


@router.post("/v1/desktop/library/tags", response_model=CRUDResponse[LibraryTag])
async def create_library_tag(body: LibraryTagCreate, request: Request):
    try:
        tag = await asyncio.to_thread(
            _store(request).create_tag, body.name, body.category, actor="user"
        )
    except TagRefused as error:
        raise _tag_refusal(error) from None
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply({**tag, "count": 0})


@router.patch("/v1/desktop/library/tags/{tag_id}", response_model=CRUDResponse[LibraryTag])
async def patch_library_tag(tag_id: TagID, body: LibraryTagPatch, request: Request):
    try:
        tag = await asyncio.to_thread(
            _store(request).rename_tag,
            tag_id,
            body.name if body.name is not None else "",
        )
    except TagRefused as error:
        raise _tag_refusal(error) from None
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply(tag)


@router.delete("/v1/desktop/library/tags/{tag_id}")
async def delete_library_tag(tag_id: TagID, request: Request):
    try:
        deleted = await asyncio.to_thread(_store(request).delete_tag, tag_id)
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    if not deleted:
        raise HTTPException(404, _wd.library_tag_unknown().envelope(code="library_tag_unknown"))
    return reply({"deleted": True})


@router.post("/v1/desktop/library/tags/{tag_id}/merge", response_model=CRUDResponse[LibraryTag])
async def merge_library_tag(tag_id: TagID, body: LibraryTagMerge, request: Request):
    try:
        tag = await asyncio.to_thread(_store(request).merge_tags, tag_id, body.into)
    except TagRefused as error:
        raise _tag_refusal(error) from None
    except MetadataUnavailable as error:
        raise _metadata_refusal(error) from None
    return reply(tag)


__all__ = ["router"]
