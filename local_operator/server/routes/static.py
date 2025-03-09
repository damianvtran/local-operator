"""
Static file hosting endpoints for the Local Operator API.

This module contains the FastAPI route handlers for serving static files.
"""

import logging
import mimetypes
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query
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
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {path}")

        # Check if the file is readable
        if not os.access(path, os.R_OK):
            raise HTTPException(status_code=403, detail=f"File not accessible: {path}")

        # Determine the file's MIME type
        mime_type, _ = mimetypes.guess_type(path)
        if not mime_type or mime_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File is not an allowed image type: {mime_type or 'unknown'}",
            )

        # Return the file
        return FileResponse(path=path, media_type=mime_type, filename=file_path.name)

    except HTTPException:
        # Re-raise HTTP exceptions to preserve their status code and detail
        raise
    except Exception as e:
        logger.exception(f"Error serving image file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
