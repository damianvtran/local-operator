"""Desktop link-preview surface: three endpoints, all bounded, none trusting the caller.

``POST /v1/desktop/team-bridge/link-preview/resolve``
``GET  /v1/desktop/team-bridge/link-preview/artifact/{artifact_id}``
``GET  /v1/desktop/team-bridge/link-preview/cache``

WHY THESE LIVE HERE AND NOT IN THE UI. The renderer never fetches. It sends the
URL text it was given and receives either a card whose images are artifact IDs on
THIS backend's origin, or one code saying why there is no card. That is the
operator's constraint made structural: no third-party host is contacted by a
surface that holds a session, and a correspondent's links never become a request
the other end can attribute to the reader.

THE THREE THINGS THIS ROUTE REFUSES TO TRUST:

* ``expected_account_id`` — compared against the identity this install actually
  holds (the signed-in OAuth row, read through the same seam every Team Bridge
  read uses). A caller cannot declare whose preview it wants, and the string is
  only ever COMPARED: the scope a preview is cached under is built server-side
  from the verified account and the committed owner row's tenant.
* ``index`` — the per-message cap is enforced server-side, BEFORE any fetch, so a
  client that ignored it cannot turn one message into an unbounded burst of
  outbound requests.
* the artifact-id shape — asserted by the route and again by the store, so the id
  can never become a path even if one of the two checks regresses. The artifact
  route proves OWNERSHIP as well as shape: it serves a blob only while the
  caller's own scope still names that id.

NOTHING HERE LOGS A URL, HOST OR QUERY STRING. Link text routinely carries
capability tokens in its query string, and a log line outlives the message; the
only identifier that appears anywhere on this surface is the artifact id, which
is a scope-derived hash and names nothing on its own.

The client stack (and therefore ``httpx``) is imported INSIDE the handler that
needs it: this module rides ``server.app``'s import path on every ``lop serve``
boot, and a module-scope httpx costs boot time for a surface most boots never
touch (the discipline ``routes/desktop_team_bridge.py`` documents).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Final

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as PathParam
from fastapi import Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from starlette.responses import Response

from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply
from local_operator.team_bridge.link_preview import cache as preview_cache
from local_operator.team_bridge.link_preview.cache import PreviewScope
from local_operator.team_bridge.link_preview.extract import sniff_raster
from local_operator.team_bridge.link_preview.models import LinkPreviewResolution
from local_operator.team_bridge.link_preview.service import resolve_preview
from local_operator.team_bridge.link_preview.store import (
    ARTIFACT_ID_PATTERN,
    InvalidArtifactId,
    PreviewStore,
)
from local_operator.web_fetch.service import load_fetch_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Desktop link preview"], dependencies=[Depends(require_desktop)])

#: The id, shaped at the boundary. ``ARTIFACT_ID_PATTERN`` is the one definition
#: of "an artifact id" and the store re-checks with it, so the route and the disk
#: cannot disagree about what may become a filename.
ArtifactID = Annotated[str, PathParam(pattern=rf"^{ARTIFACT_ID_PATTERN}$")]

#: A URL longer than this is not a link anyone wrote by hand. Bounded here so the
#: fetcher's own normalisation never sees an unbounded body field.
MAX_URL_CHARS: Final = 4096


class ResolveRequest(BaseModel):
    """The renderer's ask: which link, at which position, for which account.

    ``extra="forbid"`` is deliberate. This body has three fields and no forward
    compatibility story worth the risk: a client that sends something else is a
    client built against a different contract, and answering it silently is how
    two surfaces drift.

    ``index`` has a lower bound and NO upper bound: a position past the cap is not
    a shape error, it is the ``cap`` answer, and the route must be able to say so
    without a body-level rejection the renderer would read as broken.
    """

    model_config = ConfigDict(extra="forbid")

    url: Annotated[StrictStr, Field(min_length=1, max_length=MAX_URL_CHARS)]
    index: Annotated[StrictInt, Field(ge=0)]
    expected_account_id: Annotated[StrictStr, Field(min_length=1, max_length=128)]


class LinkPreviewCacheStats(BaseModel):
    """What the preview cache holds, in numbers that cannot name a page.

    camelCased on the wire (``residentBytes``) to match the alias boundary the
    card's own model sets (``siteName`` in ``link_preview.models``), NOT because
    any renderer contract declares a cache shape — none does. This is a
    diagnostics surface: the alias keeps the desktop wire's one spelling
    convention rather than inventing a second one beside it.
    """

    entries: int
    resident_bytes: int = Field(serialization_alias="residentBytes")
    hits: int
    misses: int
    evictions: int
    resolutions: int


def _config_dir(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


async def _require_account(config_dir: Path, expected_account_id: str) -> str:
    """Refuse unless ``expected_account_id`` is the account this install holds.

    The claim is a RECEIPT, never a grant: the account is read from the signed-in
    OAuth row (``current_access`` — the same seam the read path uses, which fails
    closed on a row whose identity cannot be read), and the caller's string is
    only ever compared to it. A renderer that asked for account A's preview while
    the machine is signed in as B gets a refusal, not A's card.

    The grammar is checked first, so a malformed id costs no network call and no
    token refresh.

    RETURNS THE VERIFIED STRING. A caller must build the cache scope from what the
    check compared, not from what it was handed: a grant derived from the
    requester's own claim would not be a receipt any more.
    """
    from local_operator.team_bridge.client import BridgeRefusal, BridgeSignInRequired
    from local_operator.team_bridge.read_context import wire_for
    from local_operator.team_bridge.types import is_account_id

    if not is_account_id(expected_account_id):
        raise HTTPException(422, {"code": "invalid_request"})

    try:
        access = await wire_for(config_dir).current_access()
    except BridgeSignInRequired:
        raise HTTPException(
            # i18n: ignore desktop wire refusal; the renderer switches on `code`
            status.HTTP_401_UNAUTHORIZED,
            {"code": "not_signed_in", "message": "sign in to Radient and try again"},
        ) from None
    except BridgeRefusal as refusal:
        # The code is the machine-readable half and the server's own text the
        # human one, exactly as the read path carries a bridge refusal: this
        # route adds no vocabulary of its own to an upstream refusal.
        upstream = refusal.http_status if refusal.http_status is not None else 502
        raise HTTPException(upstream, {"code": refusal.code, "message": refusal.message}) from None

    account_id = str(getattr(access, "account_id", ""))
    if account_id != expected_account_id:
        raise HTTPException(
            # i18n: ignore desktop wire refusal; the renderer switches on `code`
            status.HTTP_409_CONFLICT,
            {
                "code": "account_mismatch",
                "message": (
                    "this preview was asked for by a different account; "
                    "sign in as that account and try again"
                ),
            },
        )
    return account_id


def _scope_for(config_dir: Path, account_id: str) -> PreviewScope:
    """The cache scope for one request: the VERIFIED account and its tenant.

    THE TENANT IS READ HERE, LOCALLY, FROM THE COMMITTED OWNER ROW — never from
    the request. The body's ``expected_account_id`` is a receipt that has already
    been compared; accepting a tenant from the caller as well would let a caller
    pick its own fence, and a cache keyed by a fence the caller chooses is a cache
    with no fence at all.

    An install with no committed owner row has no tenant, and the account alone is
    then the whole scope: a missing tenant must narrow nothing. The same applies to
    a store that cannot be read — the tenant is the SECOND fence, and losing it
    must not fail a preview the account scope still isolates.

    WHICH FAILURES THAT COVERS, EXACTLY (review R2-2). ``active_owner_readonly``
    already wraps what SQLite raises while reading as ``StoreRefused`` (its
    read-only connection wraps the caller's read), so catching a raw
    ``sqlite3.Error`` here caught nothing. What DOES escape is the row decoder:
    ``_owner_from_row`` calls ``int()`` on the id and generation columns and then
    ``OwnerSnapshot.validated()``, so a HALF-WRITTEN committed row raises
    ``TypeError`` (a ``NULL`` where an int belongs) or ``ValueError``
    (``TeamBridgeTypeError`` is a ``ValueError``; an empty tenant or a relative
    base URL lands there). Those are precisely the case this docstring promises to
    degrade, and catching only the store's own refusal answered them with a 500 on
    both routes.
    """
    from local_operator.team_bridge.store import Store, StoreRefused

    owner = None
    try:
        owner = Store(config_dir).active_owner_readonly()
    except (StoreRefused, TypeError, ValueError):
        owner = None
    tenant_id = "" if owner is None else str(getattr(owner, "tenant_id", "") or "")
    return PreviewScope(account_id=account_id, tenant_id=tenant_id)


@router.post(
    "/v1/desktop/team-bridge/link-preview/resolve",
    response_model=CRUDResponse[LinkPreviewResolution],
    response_model_exclude_none=True,
)
async def resolve_link_preview(body: ResolveRequest, request: Request):
    """Resolve ONE link into a card, a coded absence, or a retryable error.

    A 200 envelopes every *resolution*, including the ones that produce no card:
    "this link has no metadata" is an answer, not a transport failure, and a
    renderer that had to tell 404 from 200 for it would have two failure paths for
    one state. The status codes this route DOES use are about the request being
    unanswerable at all — no bearer, not signed in, another account's receipt, a
    malformed body.
    """
    config_dir = _config_dir(request)
    account_id = await _require_account(config_dir, body.expected_account_id)
    resolution = await resolve_preview(
        url=body.url,
        index=body.index,
        scope=_scope_for(config_dir, account_id),
        settings=load_fetch_settings(request.app.state.config_manager),
        store=PreviewStore.from_config(config_dir),
        cache=preview_cache.get_preview_cache(),
    )
    return reply(resolution)


@router.get(
    "/v1/desktop/team-bridge/link-preview/cache",
    response_model=CRUDResponse[LinkPreviewCacheStats],
)
async def link_preview_cache_stats():
    """Counts and resident bytes for this process's preview cache, and nothing else.

    NO ACCOUNT RECEIPT, DELIBERATELY. Every other route on this surface takes one
    because it answers with something a specific account owns — a card, a blob.
    This answer owns nothing: six integers describing this process's own cache
    (entries held, bytes they occupy, and the hit/miss/eviction/resolution
    counters), so there is no account for a caller to prove. That safety is
    structural rather than promised: the report NEVER carries a URL, a host, a
    title or an account id — :meth:`PreviewCache.stats` cannot produce one — so
    what a caller learns is only that this install previews links at all, which
    the bearer gate has already established.

    Read through the same seam the resolve route writes to, so a test that injects
    a cache sees these numbers move. ``stats()`` is a handful of integer reads (no
    disk, no lock), which is why it runs on the loop instead of in a thread like
    the store's file work does.

    Enveloped in the surface's usual ``CRUDResponse`` so the one desktop client
    helper parses it like every sibling GET. That is a wire-convention choice, not
    a renderer contract: this is the surface's DIAGNOSTIC endpoint — what an
    operator or a test reads to see the cache move — and no renderer draws it, so
    there is no cache shape in any renderer contract to keep in step.
    """
    stats = preview_cache.get_preview_cache().stats()
    return CRUDResponse(
        status=200,
        # i18n: ignore desktop wire diagnostic; the renderer reads the numbers,
        # not the sentence, and nothing here is drawn as prose.
        message="Link preview cache stats.",
        result=LinkPreviewCacheStats(
            entries=stats.entries,
            resident_bytes=stats.resident_bytes,
            hits=stats.hits,
            misses=stats.misses,
            evictions=stats.evictions,
            resolutions=stats.resolutions,
        ),
    )


async def _read_owned_artifact(
    config_dir: Path, scope: PreviewScope, artifact_id: str
) -> tuple[bytes, str] | None:
    """The bytes and sniffed mime for an id THIS scope owns, or ``None``.

    The route's whole read path, factored out so the manual evidence script can
    drive exactly what the endpoint does rather than a copy of it. ``None`` is the
    one refusal: an id the caller's scope does not name, a blob that has gone, and
    a blob whose header no longer parses are all "there is nothing to serve", and
    telling them apart would confirm to a caller which ids exist elsewhere.

    Ownership is checked BEFORE the read: an id this scope cannot name must not
    even cost a filesystem lookup.
    """
    if not preview_cache.get_preview_cache().owns(scope, artifact_id):
        return None
    store = PreviewStore.from_config(config_dir)
    try:
        body = await asyncio.to_thread(store.get, artifact_id)
    except InvalidArtifactId:
        # Unreachable through the shaped path parameter; kept because the store is
        # the layer that would build a filename, and its refusal must reach a
        # status rather than a traceback if this route ever loosens its pattern.
        raise HTTPException(422, {"code": "invalid_request"}) from None
    if body is None:
        return None
    raster = sniff_raster(body)
    if raster is None:
        return None
    return body, raster.mime


@router.get("/v1/desktop/team-bridge/link-preview/artifact/{artifact_id}")
async def link_preview_artifact(
    artifact_id: ArtifactID,
    request: Request,
    expected_account_id: Annotated[StrictStr, Query(min_length=1, max_length=128)],
):
    """The stored bytes for one artifact id, as an image and nothing else.

    THE ID ALONE IS NOT ENOUGH. An artifact id is derived from the account, the
    tenant and the canonical URL it was fetched for, and this route serves a blob
    only while the CALLER's own scope still names that id. A second account asking
    for an id another account's message produced gets a 404 — not the bytes, and
    not the small confirmation a 403 would be.

    ``expected_account_id`` is a required receipt here for the same reason the
    resolve route takes one: the scope a request is judged under is built from the
    identity this install actually holds, and the caller's string is only compared
    with it. A renderer on an install signed in as B cannot read A's bytes even by
    replaying an id it copied from A's card.

    The mime is re-sniffed from the bytes on every read rather than stored
    alongside them: a blob whose header no longer parses is corrupt, and serving
    it as ``application/octet-stream`` would hand the renderer a download instead
    of an image, so it is a miss.
    """
    config_dir = _config_dir(request)
    account_id = await _require_account(config_dir, expected_account_id)
    served = await _read_owned_artifact(config_dir, _scope_for(config_dir, account_id), artifact_id)
    if served is None:
        raise HTTPException(404, {"code": "artifact_not_found"})
    body, mime = served
    return Response(
        content=body,
        media_type=mime,
        headers={
            # A stored blob is always raster (the sniff above is the gate), so this
            # is belt-and-braces against a future format: never let a browser
            # re-interpret these bytes as a document.
            #
            # No ``Cache-Control`` of our own: every ``/v1/desktop/`` response
            # already leaves with ``no-store`` from ``managed_desktop_boundary``,
            # so setting a lifetime here would be a header that never applies. A
            # renderer that re-asks costs one local file read, not a fetch.
            "X-Content-Type-Options": "nosniff",
        },
    )
