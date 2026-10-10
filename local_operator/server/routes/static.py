"""
Static file hosting endpoints for the Local Operator API.

This module contains the FastAPI route handlers for serving static files.
"""

import logging
import mimetypes
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from local_operator.config import read_config_values
from local_operator.helpers import convert_heic_to_png_file
from local_operator.server.utils.static_roots import (
    StaticPathDenied,
    build_roots,
    resolve_servable,
)

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


def _servable_path(request: Request, path: str) -> Path:
    """Resolve ``path`` to a file this server is allowed to serve, or raise HTTP.

    The one place the four handlers decide WHICH file; the extension/mime check
    stays in each handler because the allowlist differs per route. The policy and
    its reasoning live in :mod:`local_operator.server.utils.static_roots`. The
    roots are built per request (the live-session arm is memoised there) so a
    session started a moment ago, or a ``static.roots`` edit, takes effect
    without a restart.

    ``HTTPException`` rather than the policy's own error so this module keeps the
    FastAPI surface and the policy module stays importable without it.
    """
    state = request.app.state
    config_manager = state.config_manager
    agent_cwds = [
        agent.current_working_directory
        for agent in state.agent_registry.list_agents()
        if agent.current_working_directory
    ]
    # The file as it is on disk NOW, not ``config_manager.get_config()``: the
    # server's manager holds the snapshot it loaded at startup, so a
    # ``static.roots`` written by the settings page or ``lop config edit`` (another
    # process) would not reach it until a restart -- and the settings row is
    # labelled LIVE. ``read_config_values`` is a stat-memoised read and ``None``
    # (a torn mid-write read) falls back to the manager's copy.
    values = read_config_values(config_manager.config_dir)
    if values is None:
        values = config_manager.get_config().values
    roots = build_roots(config_manager.config_dir, values, agent_cwds)
    try:
        return resolve_servable(path, roots)
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
        # Root-confined, symlink-resolved, regular-file-only; see _servable_path.
        expanded_path = _servable_path(request, path)

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File is not an allowed image type: {mime_type or 'unknown'}",
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
                    status_code=500, detail=f"Failed to convert HEIC/HEIF file: {str(e)}"
                )

        # Return the file as-is for other image types
        return FileResponse(path=expanded_path, media_type=mime_type, filename=expanded_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving image file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


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
        # Root-confined, symlink-resolved, regular-file-only; see _servable_path.
        expanded_path = _servable_path(request, path)

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_VIDEO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File is not an allowed video type: {mime_type or 'unknown'}",
            )

        # Return the file
        return FileResponse(path=expanded_path, media_type=mime_type, filename=expanded_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving video file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


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
        # Root-confined, symlink-resolved, regular-file-only; see _servable_path.
        expanded_path = _servable_path(request, path)

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File is not an allowed audio type: {mime_type or 'unknown'}",
            )

        # Return the file
        return FileResponse(path=expanded_path, media_type=mime_type, filename=expanded_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


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
        expanded_path = _servable_path(request, path)

        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_HTML_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File is not an allowed HTML type: {mime_type or 'unknown'}",
            )

        try:
            content = expanded_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.exception(f"HTML file is not valid UTF-8: {expanded_path}")
            raise HTTPException(
                status_code=400,
                detail="HTML file is not valid UTF-8 text.",
            )
        except Exception as e:
            logger.exception(f"Error reading HTML file: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal Server Error: {str(e)}",
            )

        return Response(content=content, media_type=mime_type)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error serving HTML file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
