"""Desktop link-preview surface: two endpoints, both bounded, neither trusting the caller.

``POST /v1/desktop/team-bridge/link-preview/resolve``
``GET  /v1/desktop/team-bridge/link-preview/artifact/{artifact_id}``

WHY THESE LIVE HERE AND NOT IN THE UI. The renderer never fetches. It sends the
URL text it was given and receives either a card whose images are artifact IDs on
THIS backend's origin, or one code saying why there is no card. That is the
operator's constraint made structural: no third-party host is contacted by a
surface that holds a session, and a correspondent's links never become a request
the other end can attribute to the reader.

THE THREE THINGS THIS ROUTE REFUSES TO TRUST:

* ``expected_account_id`` — compared against the identity this install actually
  holds (the signed-in OAuth row, read through the same seam every Team Bridge
  read uses). A caller cannot declare whose preview it wants.
* ``index`` — the per-message cap is enforced server-side, BEFORE any fetch, so a
  client that ignored it cannot turn one message into an unbounded burst of
  outbound requests.
* the artifact-id shape — asserted by the route and again by the store, so the id
  can never become a path even if one of the two checks regresses.

NOTHING HERE LOGS A URL, HOST OR QUERY STRING. Link text routinely carries
capability tokens in its query string, and a log line outlives the message; the
only identifier that appears anywhere on this surface is the content-addressed
artifact id, which names bytes the origin already served publicly.

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

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi import Path as PathParam
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr
from starlette.responses import Response

from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply
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


def _config_dir(request: Request) -> Path:
    return Path(request.app.state.config_manager.config_dir)


async def _require_account(config_dir: Path, expected_account_id: str) -> None:
    """Refuse unless ``expected_account_id`` is the account this install holds.

    The claim is a RECEIPT, never a grant: the account is read from the signed-in
    OAuth row (``current_access`` — the same seam the read path uses, which fails
    closed on a row whose identity cannot be read), and the caller's string is
    only ever compared to it. A renderer that asked for account A's preview while
    the machine is signed in as B gets a refusal, not A's card.

    The grammar is checked first, so a malformed id costs no network call and no
    token refresh.
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
        raise HTTPException(
            upstream, {"code": refusal.code, "message": refusal.message}
        ) from None

    if str(getattr(access, "account_id", "")) != expected_account_id:
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
    await _require_account(config_dir, body.expected_account_id)
    resolution = await resolve_preview(
        url=body.url,
        index=body.index,
        settings=load_fetch_settings(request.app.state.config_manager),
        store=PreviewStore.from_config(config_dir),
    )
    return reply(resolution)


@router.get("/v1/desktop/team-bridge/link-preview/artifact/{artifact_id}")
async def link_preview_artifact(artifact_id: ArtifactID, request: Request):
    """The stored bytes for one artifact id, as an image and nothing else.

    There is no account check here, deliberately: the id is the sha256 of the
    bytes, so what it names is content this install already fetched from a public
    origin. It carries no message, sender or tenant — the store holds no metadata
    at all — and a second account asking for a favicon another account's message
    linked to learns nothing about that message.

    The mime is re-sniffed from the bytes on every read rather than stored
    alongside them: a blob whose header no longer parses is corrupt, and serving
    it as ``application/octet-stream`` would hand the renderer a download instead
    of an image, so it is a miss.
    """
    store = PreviewStore.from_config(_config_dir(request))
    try:
        body = await asyncio.to_thread(store.get, artifact_id)
    except InvalidArtifactId:
        # Unreachable through the shaped path parameter; kept because the store is
        # the layer that would build a filename, and its refusal must reach a
        # status rather than a traceback if this route ever loosens its pattern.
        raise HTTPException(422, {"code": "invalid_request"}) from None
    if body is None:
        raise HTTPException(404, {"code": "artifact_not_found"})
    raster = sniff_raster(body)
    if raster is None:
        raise HTTPException(404, {"code": "artifact_not_found"})
    return Response(
        content=body,
        media_type=raster.mime,
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
