"""
Static file hosting endpoints for the Local Operator API.

This module contains the FastAPI route handlers for serving static files.
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse

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

# List of allowed HTML MIME types
ALLOWED_HTML_TYPES: List[str] = [
    "text/html",
    "application/xhtml+xml",
]


@router.get(
    "/v1/static/images",
    summary="Serve image file",
    description="Serves an image file from disk by path. Only image file types are allowed.",
    response_class=FileResponse,
)
async def get_image(
    path: str = Query(..., description="Path to the image file on disk"),
):
    """
    Serve an image file from disk.

    Args:
        path: Path to the image file on disk

    Returns:
        The image file as a response

    Raises:
        HTTPException: If the file doesn't exist, is not accessible, or is not an image file
    """
    try:
        # Validate the path exists
        file_path = Path(path)

        expanded_path = file_path.expanduser().resolve()

        if not expanded_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        if not expanded_path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {path}")

        # Check if the file is readable
        if not os.access(expanded_path, os.R_OK):
            raise HTTPException(status_code=403, detail=f"File not accessible: {path}")

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(expanded_path)
        if not mime_type or mime_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File is not an allowed image type: {mime_type or 'unknown'}",
            )

        # Return the file
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
    path: str = Query(..., description="Path to the video file on disk"),
):
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
        # Validate the path exists
        file_path = Path(path)

        expanded_path = file_path.expanduser().resolve()

        if not expanded_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        if not expanded_path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {path}")

        # Check if the file is readable
        if not os.access(expanded_path, os.R_OK):
            raise HTTPException(status_code=403, detail=f"File not accessible: {path}")

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
    "/v1/static/html",
    summary="Serve HTML file",
    description="Serves an HTML file from disk by path. Only HTML file types are allowed.",
    response_class=Response,
)
async def get_html(
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
        file_path = Path(path)
        expanded_path = file_path.expanduser().resolve()

        if not expanded_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        if not expanded_path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {path}")

        if not os.access(expanded_path, os.R_OK):
            raise HTTPException(status_code=403, detail=f"File not accessible: {path}")

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
