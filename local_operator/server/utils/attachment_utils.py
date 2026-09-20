import base64
import binascii
import logging
import mimetypes
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import HTTPException

from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

BASE64_DATA_URL_PATTERN = re.compile(r"^data:(?P<mime_type>[^;]+);base64,(?P<data>.+)$")


def uploads_dir() -> Path:
    """Where a decoded chat attachment is written: ``<config dir>/uploads``.

    Resolved on every call from :func:`local_operator.paths.config_dir` rather
    than frozen into a module constant. This module used to build its own
    ``Path.home() / ".local-operator" / "uploads"`` — a second copy of the rule
    ``paths`` exists to own — so an isolated run that relocated
    ``LOCAL_OPERATOR_CONFIG_DIR`` still had this module create a directory in
    the operator's REAL home, at import time, as a side effect of importing the
    chat routes. Import-time filesystem work is the other half of that bug: a
    reader of the routes should not need write access to the user's home.
    """
    return config_dir() / "uploads"


def _ensure_uploads_dir_exists(directory: Path) -> Path:
    """Create ``directory`` and return it, at the point of use. See :func:`uploads_dir`."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create uploads directory {directory}: {e}")
        raise RuntimeError(f"Failed to create uploads directory: {directory}") from e
    return directory


def parse_base64_data_url(data_url: str) -> Tuple[str, bytes]:
    """
    Parses a base64 data URL and returns the mime type and decoded data.

    Args:
        data_url: The base64 data URL string.

    Returns:
        A tuple containing the mime type and the decoded binary data.

    Raises:
        HTTPException: If the URL is not a valid base64 data URL or decoding fails.
    """
    match = BASE64_DATA_URL_PATTERN.match(data_url)
    if not match:
        logger.warning(f"Data URL does not match expected pattern: {data_url[:100]}...")
        raise HTTPException(status_code=400, detail="Invalid data URL format for attachment.")

    mime_type = match.group("mime_type")
    base64_data = match.group("data")

    try:
        missing_padding = len(base64_data) % 4
        if missing_padding:
            base64_data += "=" * (4 - missing_padding)
        decoded_data = base64.b64decode(base64_data, validate=True)
        return mime_type, decoded_data
    except binascii.Error as e:
        logger.error(f"Failed to decode base64 data from URL (mime: {mime_type}): {e}")
        raise HTTPException(status_code=400, detail=f"Invalid base64 data in attachment: {e}")
    except Exception as e:
        logger.error(f"Unexpected error decoding base64 data (mime: {mime_type}): {e}")
        raise HTTPException(status_code=500, detail="Error processing base64 attachment data.")


def save_base64_attachment(data_url: str) -> str:
    """
    Saves a base64 encoded attachment to the local uploads directory.

    Args:
        data_url: The base64 data URL string.

    Returns:
        The absolute normalized path (as a string) of the saved attachment.

    Raises:
        HTTPException: If parsing, decoding, or saving fails.
    """
    mime_type, binary_data = parse_base64_data_url(data_url)

    extension = mimetypes.guess_extension(mime_type)
    if not extension:
        logger.warning(
            f"Could not determine file extension for mime type: {mime_type}. Using '.bin'."
        )
        extension = ".bin"

    if extension == ".jpe":
        extension = ".jpg"
    elif extension == ".htm":
        extension = ".html"

    filename = f"{uuid.uuid4()}{extension}"
    file_path = _ensure_uploads_dir_exists(uploads_dir()) / filename

    try:
        with open(file_path, "wb") as f:
            f.write(binary_data)
        logger.info(f"Saved attachment to {file_path}")
        return str(file_path.resolve())
    except IOError as e:
        logger.error(f"Failed to save attachment to {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save attachment file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving attachment to {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error saving attachment file.")


async def process_attachments(attachment_urls: Optional[List[str]]) -> List[str]:
    """
    Processes a list of attachment URLs. If a URL is a base64 data URL,
    it decodes the data, saves it to a local file, and replaces the URL
    with the absolute normalized path to the file.

    Args:
        attachment_urls: An optional list of attachment URLs.

    Returns:
        A list of processed attachment paths. Returns an empty list if input is None or empty.

    Raises:
        HTTPException: If processing any base64 attachment fails (propagated from helper functions).
    """
    if not attachment_urls:
        return []

    processed_urls = []
    for url in attachment_urls:
        if url.startswith("data:") and ";base64," in url:
            file_path = save_base64_attachment(url)
            processed_urls.append(file_path)
        else:
            processed_urls.append(url)
    return processed_urls
