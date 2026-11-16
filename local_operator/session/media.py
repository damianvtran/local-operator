"""Retained session media, addressed by transcript identity rather than paths.

A view grant to one conversation is not a grant to the whole attachment store.
Resolve the published entry and family-specific index first, then its retained
content. Origin paths and source URLs are display metadata, never byte sources.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionMedia:
    mime: str
    media_id: str
    total: int
    path: Path | None = None
    data: bytes | None = None
    fingerprint: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        # Stored/imported MIME is descriptive, not HTTP-header authority. The
        # same resolver serves local files and remote chunks, so enforce this
        # invariant here rather than only on the mesh receiver.
        if not self.mime.isascii() or "\r" in self.mime or "\n" in self.mime:
            object.__setattr__(self, "mime", "application/octet-stream")

    def read(self, offset: int, length: int) -> bytes:
        if self.path is not None:
            import os

            with self.path.open("rb") as handle:
                before = os.fstat(handle.fileno())
                if (before.st_ino, before.st_size, before.st_mtime_ns) != self.fingerprint:
                    raise OSError("retained media changed")
                handle.seek(offset)
                data = handle.read(length)
                after = os.fstat(handle.fileno())
                if (after.st_ino, after.st_size, after.st_mtime_ns) != self.fingerprint:
                    raise OSError("retained media changed")
                return data
        return (self.data or b"")[offset : offset + length]


def _entry(directory: Path, entry_id: str) -> Any:
    from local_operator.session.transcript import read_entry_by_id, read_transcript_page

    entry = read_entry_by_id(directory, entry_id)
    if entry is not None:
        return entry
    # Match both phone tool-row spellings. Locate the assistant prefix rather
    # than splitting once: call ids themselves may contain colons, and the
    # published composite must still name its actual parent call.
    call_id = entry_id[3:] if entry_id.startswith("tc-") else ""
    if not call_id:
        for cut in reversed([position for position, char in enumerate(entry_id) if char == ":"]):
            parent = read_entry_by_id(directory, entry_id[:cut])
            candidate_id = entry_id[cut + 1 :]
            calls = (parent.payload.get("tool_calls") or []) if parent is not None else []
            if not isinstance(calls, list):
                continue
            if any(isinstance(call, dict) and call.get("id") == candidate_id for call in calls):
                call_id = candidate_id
                break
    if not call_id:
        return None
    before = None
    seen: set[str] = set()
    while True:
        if before is not None:
            if before in seen:
                return None  # a concurrently replaced journal reconciled us
            seen.add(before)
        page = read_transcript_page(directory, before_id=before)
        for candidate in reversed(page.entries):
            if (
                candidate.payload.get("role") == "tool"
                and candidate.payload.get("tool_call_id") == call_id
            ):
                return candidate
        if not page.has_more or not page.entries:
            return None
        before = page.entries[0].id


def session_media(
    root: Path, session_id: str, entry_id: str, index: int, kind: str
) -> SessionMedia | None:
    """Resolve image/video/file bytes retained by this exact user conversation.

    Store-backed videos are hashed and read in existing mesh-sized chunks; they
    are never materialized as a whole-video bytes/base64 object. Hash validation
    preserves AttachmentStore's bit-rot contract and makes a changed object fail
    the receiver's expected-media identity on subsequent chunks.
    """
    from local_operator.network.sync import SYNC_CHUNK_BYTES
    from local_operator.resume import is_user_session
    from local_operator.session.attachments import store_for_transcript_dir
    from local_operator.session.compression import has_transcript

    if session_id in ("", ".", "..") or Path(session_id).name != session_id:
        return None
    if type(index) is not int or index < 0 or kind not in ("image", "video", "file"):
        return None
    directory = root / "sessions" / session_id
    if not has_transcript(directory) or not is_user_session(directory):
        return None
    entry = _entry(directory, entry_id)
    if entry is None or entry.type != "message":
        return None
    # A retained pre-compaction row remains addressable, but an explicitly
    # pruned row does not. Scan only prune records, without hydrating media.
    from local_operator.session.compression import transcript_for_read
    from local_operator.session.transcript import TranscriptEntry

    journal = transcript_for_read(directory, read_only=True)
    if journal is None:
        return None
    try:
        with journal.open("rb") as handle:
            for raw in handle:
                if b'"prune"' not in raw:
                    continue
                marker = TranscriptEntry.from_json(raw.decode("utf-8", errors="replace"))
                if (
                    marker is not None
                    and marker.type == "prune"
                    and marker.payload.get("target") == entry.id
                ):
                    return None
    except OSError:
        return None
    content = entry.payload.get("content")
    if not isinstance(content, list):
        return None
    blocks = []
    for block in content:
        if not isinstance(block, dict):
            continue
        # Same family index as mobile projection._image_refs/_video_refs.
        artifact_kind = block.get("kind")
        block_type = block.get("type")
        family = None
        if artifact_kind in ("image", "video", "audio") and any(
            field in block for field in ("content_type", "attachment", "source_url")
        ):
            family = artifact_kind
        elif block_type == "document" or (
            not block_type and "filename" in block and "mime_type" in block
        ):
            family = "file"
        elif block_type == "image" or (
            not block_type
            and not artifact_kind
            and any(field in block for field in ("data", "attachment"))
            and not str(block.get("mime_type", "")).startswith("audio/")
        ):
            # Default ImageContent fields are omitted in durable rows: an
            # inline PNG can be exactly {data: ...}, not {type: image, ...}.
            family = "image"
        if family == kind:
            blocks.append(block)
    if index >= len(blocks):
        return None
    block = blocks[index]
    mime = str(
        block.get("content_type")
        or block.get("mime_type")
        or {"image": "image/png", "video": "video/mp4", "file": "application/octet-stream"}[kind]
    )
    digest = block.get("attachment")
    if digest:
        # The digest came from the retained row, not the phone. Validate it
        # anyway: a damaged journal must not turn a store lookup into traversal.
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{32}", digest) is None:
            return None
        store = store_for_transcript_dir(directory)
        retained_mime = store.mime_type_for(digest)
        if retained_mime is None:
            return None
        mime = str(block.get("content_type") or block.get("mime_type") or retained_mime)
        path = store.path_for(digest)
        try:
            before = path.stat()
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(SYNC_CHUNK_BYTES):
                    hasher.update(chunk)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                return None
            identity = hasher.hexdigest()
            if not identity.startswith(digest):
                return None
            return SessionMedia(
                mime,
                identity,
                after.st_size,
                path=path,
                fingerprint=(after.st_ino, after.st_size, after.st_mtime_ns),
            )
        except OSError:
            return None
    raw = block.get("data")
    if not isinstance(raw, str):
        return None
    try:
        data = base64.b64decode(raw)
    except (ValueError, binascii.Error):
        return None
    return SessionMedia(mime, hashlib.sha256(data).hexdigest(), len(data), data=data)
