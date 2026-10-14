"""
Static file hosting endpoints for the Local Operator API.

This module contains the FastAPI route handlers for serving static files.
"""

import logging
import mimetypes
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from local_operator.config import read_config_values
from local_operator.helpers import convert_heic_to_png_file
from local_operator.i18n.keys import wire_errors_files as _wf
from local_operator.server.desktop import require_desktop
from local_operator.server.utils import local_tokens
from local_operator.server.utils.static_roots import (
    StaticPathDenied,
    build_roots,
    connection_mode,
    resolve_servable,
)

# Keep exception tails opaque and English text byte-identical: released previews
# consume status, while desktop error readers accept the additive text envelope.
# StaticPathDenied belongs to the path policy and remains an unchanged passthrough.
router = APIRouter(tags=["Static"])
logger = logging.getLogger("local_operator.server.routes.static")

# List of allowed image MIME types
ALLOWED_IMAGE_TYPES: List[str] = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/svg+xml",
    "image/tiff",
    "image/x-icon",
    "image/heic",
    "image/heif",
    "image/avif",
    "image/pjpeg",
]

# List of allowed video MIME types
ALLOWED_VIDEO_TYPES: List[str] = [
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/mpeg",
    "video/3gpp",
    "video/3gpp2",
    "video/x-flv",
]

# List of allowed audio MIME types
ALLOWED_AUDIO_TYPES: List[str] = [
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/ogg",
    "audio/webm",
    "audio/aac",
    "audio/flac",
    "audio/x-wav",
    "audio/x-m4a",
    "audio/mp4",
    "audio/x-aiff",
    "audio/aiff",
]

# List of allowed HTML MIME types
ALLOWED_HTML_TYPES: List[str] = [
    "text/html",
    "application/xhtml+xml",
]


def _servable_path(request: Request, path: str, route: str) -> Path:
    """Resolve ``path`` to a file this server is allowed to serve, or raise HTTP.

    The one place the four handlers decide WHICH file; the extension/mime check
    stays in each handler because the allowlist differs per route. The policy and
    its reasoning live in :mod:`local_operator.server.utils.static_roots`. The
    roots are built per request (the live-session arm is memoised there) so a
    session started a moment ago, or a ``static.roots`` edit, takes effect
    without a restart. The general/rooted posture is the CONNECTION's (RFC §3.2),
    computed here from ``scope["server"]`` -- never from a header or a query.
    ``route`` is the calling route's id (``images``/``videos``/``audio``/
    ``html``): the signed-URL contract signs one route + path + expiry, so the
    gate that verifies a signature must know which route it is gating.

    ``HTTPException`` rather than the policy's own error so this module keeps the
    FastAPI surface and the policy module stays importable without it.
    """
    state = request.app.state
    config_manager = state.config_manager
    # The file as it is on disk NOW, not ``config_manager.get_config()``: the
    # server's manager holds the snapshot it loaded at startup, so a
    # ``static.roots`` written by the settings page or ``lop config edit`` (another
    # process) would not reach it until a restart -- and the settings row is
    # labelled LIVE. ``read_config_values`` is a stat-memoised read and ``None``
    # (a torn mid-write read) falls back to the manager's copy.
    values = read_config_values(config_manager.config_dir)
    if values is None:
        values = config_manager.get_config().values
    roots = build_roots(config_manager.config_dir, values)
    # ``scope["server"]`` is ``(host, port)`` as uvicorn sets it from the socket
    # the kernel accepted on; a missing or foreign value is rooted (fail-closed),
    # so only a loopback-accepted request reaches the unclamped predicate. The
    # bind half of the condition lives in ``cli.py`` (every non-loopback bind is
    # refused there).
    server = request.scope.get("server")
    host = None
    if isinstance(server, (tuple, list)) and server and isinstance(server[0], str):
        host = server[0]
    general = connection_mode(host) == "general"
    if general:
        # Phase A credential observation (file-serving RFC §4.2): a presented
        # token or signed URL is validated and counted, never refused -- and it
        # runs BEFORE ``resolve_servable`` so the signature check is pure
        # string/HMAC work that can never become a filesystem oracle. Phases
        # B/C will gate on this same point.
        local_tokens.observe_credentials(
            route=route,
            path=path,
            token=request.headers.get(local_tokens.ACCESS_TOKEN_HEADER),
            exp_raw=request.query_params.get("exp"),
            sig_raw=request.query_params.get("sig"),
        )
    try:
        return resolve_servable(path, roots, general=general)
    except StaticPathDenied as denied:
        raise HTTPException(status_code=denied.status, detail=denied.detail)


@router.get(
    "/v1/static/images",
    summary="Serve image file",
    description=(
        "Serves an image file from disk by path. Only image file types are allowed. "
        "HEIC/HEIF files are automatically converted to PNG."
    ),
    response_class=FileResponse,
)
async def get_image(
    request: Request,
    path: str = Query(..., description="Path to the image file on disk"),
) -> FileResponse:
    """
    Serve an image file from disk. HEIC/HEIF files are automatically converted to PNG.

    Args:
        path: Path to the image file on disk

    Returns:
        The image file as a response

    Raises:
        HTTPException: If the file doesn't exist, is not accessible, or is not an image file
    """
    try:
        # Connection-gated, symlink-resolved, regular-file-only; see _servable_path.
        expanded_path = _servable_path(request, path, "images")

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=_wf.image_type(mime_type=mime_type or "unknown").envelope(),
            )

        # Check if this is a HEIC/HEIF file that needs conversion
        if mime_type in ["image/heic", "image/heif"]:
            try:
                # Convert HEIC/HEIF to PNG
                converted_path = convert_heic_to_png_file(expanded_path)

                # Return the converted PNG file with PNG MIME type
                return FileResponse(
                    path=converted_path,
                    media_type="image/png",
                    filename=f"{expanded_path.stem}.png",
                )
            except Exception as e:
                logger.exception(f"Failed to convert HEIC/HEIF file {expanded_path}: {e}")
                raise HTTPException(
                    status_code=500, detail=_wf.heic_conversion(error=str(e)).envelope()
                )

        # Return the file as-is for other image types
        return FileResponse(path=expanded_path, media_type=mime_type, filename=expanded_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving image file: {e}")
        raise HTTPException(status_code=500, detail=_wf.internal_error(error=str(e)).envelope())


@router.get(
    "/v1/static/videos",
    summary="Serve video file",
    description="Serves a video file from disk by path. Only video file types are allowed.",
    response_class=FileResponse,
)
async def get_video(
    request: Request,
    path: str = Query(..., description="Path to the video file on disk"),
) -> FileResponse:
    """
    Serve a video file from disk.

    Args:
        path: Path to the video file on disk

    Returns:
        The video file as a response

    Raises:
        HTTPException: If the file doesn't exist, is not accessible, or is not a video file
    """
    try:
        # Connection-gated, symlink-resolved, regular-file-only; see _servable_path.
        expanded_path = _servable_path(request, path, "videos")

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=_wf.video_type(mime_type=mime_type or "unknown").envelope(),
            )

        # Return the file
        return FileResponse(path=expanded_path, media_type=mime_type, filename=expanded_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving video file: {e}")
        raise HTTPException(status_code=500, detail=_wf.internal_error(error=str(e)).envelope())


@router.get(
    "/v1/static/audio",
    summary="Serve audio file",
    description="Serves an audio file from disk by path. Only audio file types are allowed.",
    response_class=FileResponse,
)
async def get_audio(
    request: Request,
    path: str = Query(..., description="Path to the audio file on disk"),
) -> FileResponse:
    """
    Serve an audio file from disk.

    Args:
        path: Path to the audio file on disk

    Returns:
        The audio file as a response

    Raises:
        HTTPException: If the file doesn't exist, is not accessible, or is not an audio file
    """
    try:
        # Connection-gated, symlink-resolved, regular-file-only; see _servable_path.
        expanded_path = _servable_path(request, path, "audio")

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=_wf.audio_type(mime_type=mime_type or "unknown").envelope(),
            )

        # Return the file
        return FileResponse(path=expanded_path, media_type=mime_type, filename=expanded_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving audio file: {e}")
        raise HTTPException(status_code=500, detail=_wf.internal_error(error=str(e)).envelope())


@router.get(
    "/v1/static/html",
    summary="Serve HTML file",
    description="Serves an HTML file from disk by path. Only HTML file types are allowed.",
    response_class=Response,
)
async def get_html(
    request: Request,
    path: str = Query(..., description="Path to the HTML file on disk"),
) -> Response:
    """
    Serve an HTML file from disk as text/html content.

    Args:
        path: Path to the HTML file on disk

    Returns:
        Response: The HTML file content as a text/html response

    Raises:
        HTTPException: If the file doesn't exist, is not accessible, or is not an HTML file
    """
    try:
        expanded_path = _servable_path(request, path, "html")

        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_HTML_TYPES:
            raise HTTPException(
                status_code=400,
                detail=_wf.html_type(mime_type=mime_type or "unknown").envelope(),
            )

        try:
            content = expanded_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.exception(f"HTML file is not valid UTF-8: {expanded_path}")
            raise HTTPException(
                status_code=400,
                detail=_wf.html_utf8().envelope(),
            )
        except Exception as e:
            logger.exception(f"Error reading HTML file: {e}")
            raise HTTPException(
                status_code=500,
                detail=_wf.internal_error(error=str(e)).envelope(),
            )

        return Response(content=content, media_type=mime_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error serving HTML file: {e}")
        raise HTTPException(status_code=500, detail=_wf.internal_error(error=str(e)).envelope())


# ------------------------------------------------------- signed URLs (Phase A)

#: The route ids a signed URL may name (RFC §4.1: "route" is the static family
#: segment). Kept as the endpoint's own constant rather than derived from the
#: decorators: the SIGNED STRING bakes this value in, so changing a route id is
#: a signed-URL contract change, not a rename.
SIGNABLE_ROUTES: tuple[str, ...] = ("images", "videos", "audio", "html")


class SignRequest(BaseModel):
    """Body of ``POST /v1/static/sign`` (contract: ``docs/DESKTOP_API.md``)."""

    route: str = Field(..., description="One of: images, videos, audio, html")
    path: str = Field(..., description="The path string the signed URL will carry, verbatim")
    ttl_s: int | None = Field(
        None,
        description=(
            f"Lifetime in seconds (default {local_tokens.TTL_DEFAULT_S}, "
            f"max {local_tokens.TTL_MAX_S})"
        ),
    )


class SignResponse(BaseModel):
    """A signed URL for one route + path + expiry, and that expiry."""

    url: str
    exp: int


def _require_sign_auth(request: Request) -> None:
    """The sign endpoint's door: the local access token OR the desktop bearer.

    Token first -- it is the Phase A credential any surface reads from the
    serve record, and a page cannot read the record. Otherwise the desktop
    plane's own door opens this one: its origin allowlist and bearer check
    apply unchanged, and its refusal is normalized to one stable 401, so this
    endpoint cannot be used to tell "wrong bearer" from "closed plane" from
    "foreign origin".
    """
    supplied = request.headers.get(local_tokens.ACCESS_TOKEN_HEADER, "")
    if supplied and local_tokens.process_keys().token_status(supplied) == "valid":
        return
    try:
        require_desktop(request)
    except HTTPException as exc:
        raise HTTPException(401, _wf.sign_auth().envelope()) from exc


@router.post(
    "/v1/static/sign",
    summary="Mint a signed static-file URL",
    description=(
        "Returns a URL carrying route, path, expiry and an HMAC signature, for "
        "src= loads that cannot carry a header. Phase A of the file-serving RFC: "
        "minting works today; enforcement is a later phase. Auth: the local access "
        "token (X-Local-Token header) or the desktop bearer. The signature is "
        "verified before any filesystem resolution when the URL is used."
    ),
)
async def sign_static_url(
    body: SignRequest,
    _auth: None = Depends(_require_sign_auth),
) -> SignResponse:
    """
    Mint a signed URL for one static route + path + expiry.

    The signed bytes are ``f"{route}\\n{path}\\n{exp}"`` (UTF-8) under
    HMAC-SHA256 with the per-boot signing key; the same canonical string is
    verified on GET (``_servable_path`` -> ``local_tokens``). TTL: default
    600 s, cap 3600 s.
    """
    if body.route not in SIGNABLE_ROUTES:
        raise HTTPException(400, _wf.sign_route(routes=", ".join(SIGNABLE_ROUTES)).envelope())
    if not body.path:
        raise HTTPException(400, _wf.sign_path().envelope())
    ttl_s = local_tokens.TTL_DEFAULT_S if body.ttl_s is None else body.ttl_s
    if not 0 < ttl_s <= local_tokens.TTL_MAX_S:
        # An opaque bound string preserves the ungrouped decimal English contract;
        # ICU formats an integer as 3,600, unlike the former f-string's 3600.
        raise HTTPException(400, _wf.sign_ttl(maximum=str(local_tokens.TTL_MAX_S)).envelope())
    url, exp = local_tokens.sign_url(body.route, body.path, ttl_s=ttl_s)
    return SignResponse(url=url, exp=exp)
