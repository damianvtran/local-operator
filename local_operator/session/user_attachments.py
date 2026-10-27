"""Every file and picture the USER attached to one conversation, read from its journal.

WHY THIS EXISTS. The desktop canvas lists a conversation's user attachments (an
"Attachments" section above the agent's files). The renderer cannot derive that list from
what it has loaded: history is paged lazily from the newest row backwards, so a PDF attached
in the first turn of a long conversation is simply not in memory when the canvas opens. An
empty or partial section there would be a false claim ("you attached nothing"), so the
complete inventory has to come from the one place that has every row - the journal on disk.

WHAT IT READS, and why only that:

* ``message`` rows whose payload ``role`` is ``user``. A tool result's image, a generated
  artifact and an assistant's media are the AGENT's output and already belong to the
  session-file index (``session_files``); this module answers only "what did the person
  hand over".
* Rows the harness minted (``provider_payload.harness_injected``) are skipped through the
  shared :func:`~local_operator.harness.rows.is_harness_injection` decision: those were never
  sent by a person, so listing their media as the user's would repeat the attribution error
  that predicate exists to prevent.
* ``document`` blocks (``{type: "document", filename, mime_type, path?, stored_path?, bytes,
  injection?, attachment|data}``) and image blocks (``{mime_type, attachment|data}``; ``type``
  is usually ABSENT on a durable image because the encoder dumps with ``exclude_defaults``,
  so an image is recognised by ``data``/``attachment`` exactly the way
  ``transcript._externalize_attachments`` recognises one). The legacy document shape
  ``{filename, mime_type, attachment}`` with no ``type`` is a document too.

METADATA ONLY. No base64 ever leaves this module: the bytes stay behind the existing
digest-addressed ``…/attachments/{digest}`` route, so there is ONE binary transport for an
attachment and this listing cannot become a second one. An inline image (a row written
without an attachment store, or under the 1 KiB externalisation floor) has no digest on the
row; its digest is looked up READ-ONLY in the store (:meth:`AttachmentStore.locate`, never
``put``) so a picture whose bytes ARE retained is still reachable through that route, and one
whose bytes are not is reported with ``attachment: None`` rather than inlined.

ORIGIN vs RETAINED COPY. ``exists`` stats the origin path and ``stored`` the store copy, so a
surface can name all three outcomes: still at its origin; moved, but a saved copy is kept; and
both gone.

TOLERANCE. A journal is written by many builds and sometimes edited by hand: a malformed
line, a non-dict block or a field of the wrong type drops THAT item, never the listing. The
stored-copy fact (``stored``) is a ``stat`` of the store file, so a surface can tell "origin
moved, retained copy still here" from "both gone" without fetching bytes.

Pure and host-free: it takes a session directory and returns plain dicts; the desktop route
(``server/routes/desktop_session_files.py``) owns admission and the wire envelope.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from pathlib import Path
from typing import Any, Literal

from local_operator.harness.rows import is_harness_injection
from local_operator.session import compression
from local_operator.session.attachments import AttachmentStore, store_for_transcript_dir

logger = logging.getLogger(__name__)

#: Upper bound on rows returned. A person does not attach thousands of files to one
#: conversation; the cap exists so a pathological (scripted, hand-built) journal cannot turn a
#: canvas open into a multi-megabyte response. ``truncated`` tells the client the list is not
#: the whole story, so it never presents a capped list as complete.
MAX_ITEMS = 2000

#: The text a user message row must contain before it is worth a JSON parse. Rows are written
#: compact (``separators=(",", ":")``) by ``TranscriptEntry.to_json``; the spaced spelling is
#: what ``json.dumps`` defaults produce, i.e. a hand-built or older-tool journal.
_USER_NEEDLES = ('"role":"user"', '"role": "user"')

#: Quoted keys, one of which every media block carries. ``"mime_type"`` alone is NOT enough: an
#: inline PNG whose mime defaulted has no ``mime_type`` key on the durable row at all.
_MEDIA_NEEDLES = ('"data"', '"attachment"', '"mime_type"')

#: Keys whose presence marks a content block as MEDIA rather than text. Mirrors
#: ``transcript._externalize_attachments``: identify by ``data``/``attachment``, never by
#: ``type`` alone, because ``type`` is the pydantic default and absent on a durable image.
_MEDIA_KEYS = ("data", "attachment")

Kind = Literal["document", "image"]


def _maybe_user_row(line: str) -> bool:
    """Cheap pre-filter: could this line be a user message carrying media?

    A 300 MB journal is overwhelmingly tool output; parsing every line as JSON to discard it
    is the cost this avoids. Both conditions are necessary for a match below, so a ``False``
    here never hides a real attachment.
    """
    if not any(needle in line for needle in _USER_NEEDLES):
        return False
    return any(needle in line for needle in _MEDIA_NEEDLES)


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> int | None:
    # ``bool`` is an ``int`` subclass; a ``true`` size is malformed, not 1 byte.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _decoded_size(data: str) -> int | None:
    """Decoded byte count of a base64 payload without materialising the bytes twice."""
    try:
        return len(base64.b64decode(data, validate=False))
    except (binascii.Error, ValueError):
        return None


def _block_kind(block: dict[str, Any]) -> Kind | None:
    """``document``, ``image`` or ``None`` (text, audio, artifact, anything else)."""
    block_type = block.get("type")
    if block_type == "document":
        return "document"
    if not any(key in block for key in _MEDIA_KEYS):
        return None
    # The legacy document shape: a filename beside the media keys and no ``type``.
    if block_type is None and isinstance(block.get("filename"), str):
        return "document"
    # An AttachmentContent (``kind`` image/video/audio) is agent output, never a user upload;
    # audio blocks are stamped ``type: "audio"`` by the encoder and are recordings, not files.
    if "kind" in block:
        return None
    if block_type in (None, "image"):
        mime = block.get("mime_type")
        # An image block whose ``mime_type`` defaulted (``image/png``) has it ABSENT on the
        # durable row; an explicit non-image mime here is some other media and is skipped.
        if mime is None or (isinstance(mime, str) and mime.startswith("image/")):
            return "image"
    return None


def _item(
    block: dict[str, Any],
    kind: Kind,
    *,
    message_id: str,
    ts: float,
    index: int,
    store: AttachmentStore,
) -> dict[str, Any]:
    digest = _str_or_none(block.get("attachment"))
    data = block.get("data") if isinstance(block.get("data"), str) else None
    mime = _str_or_none(block.get("mime_type")) or (
        "image/png" if kind == "image" else "application/octet-stream"
    )
    size = _int_or_none(block.get("bytes"))
    if digest is None and data:
        # Read-only lookup: never populate the store from a read path.
        ref = store.locate(data, mime)
        if ref is not None:
            digest = ref.digest
        if size is None:
            size = _decoded_size(data)
    stored_path = _str_or_none(block.get("stored_path"))
    if stored_path is None and digest is not None:
        stored_path = str(store.path_for(digest))
    stored = False
    if stored_path is not None:
        try:
            stat = Path(stored_path).stat()
            stored = True
            # A durable image row records no size (the encoder never wrote one), but the
            # retained copy IS the decoded bytes, so its length is the honest size to show.
            if size is None:
                size = stat.st_size
        except OSError:
            stored = False
    origin = _str_or_none(block.get("path"))
    # Whether the ORIGIN is still where the person attached it from. ``None`` when the row
    # names no origin (a pasted screenshot) - unknown is not "gone". Stat'd here rather than
    # by the renderer so one listing answers every row without a per-row bridge round trip.
    exists: bool | None = None
    if origin is not None:
        try:
            exists = Path(origin).is_file()
        except OSError:
            exists = None
    injection = block.get("injection")
    return {
        "message_id": message_id,
        "ts": ts,
        "index": index,
        "kind": kind,
        "attachment": digest,
        "filename": _str_or_none(block.get("filename")),
        "mime_type": mime,
        "bytes": size,
        "path": origin,
        "exists": exists,
        "stored_path": stored_path,
        "stored": stored,
        "injection": injection if injection in ("native", "extracted", "reference") else None,
    }


def items_from_row(row: Any, store: AttachmentStore) -> list[dict[str, Any]]:
    """The user attachments one raw journal row carries (empty for anything else)."""
    if not isinstance(row, dict) or row.get("type") != "message":
        return []
    payload = row.get("payload")
    if not isinstance(payload, dict) or payload.get("role") != "user":
        return []
    if payload.get("kind") not in (None, "message"):
        return []
    if is_harness_injection(payload):
        return []
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    message_id = row.get("id")
    if not isinstance(message_id, str) or not message_id:
        return []
    ts_raw = row.get("ts")
    ts = float(ts_raw) if isinstance(ts_raw, (int, float)) and not isinstance(ts_raw, bool) else 0.0
    found: list[dict[str, Any]] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        kind = _block_kind(block)
        if kind is None:
            continue
        try:
            found.append(_item(block, kind, message_id=message_id, ts=ts, index=index, store=store))
        except Exception:  # noqa: BLE001 - one odd block must not cost the whole listing
            logger.debug("user-attachments: skipping malformed block", exc_info=True)
    return found


def list_user_attachments(session_dir: Path) -> dict[str, Any]:
    """``{items, truncated}`` for the conversation in ``session_dir``, oldest first.

    Order is journal order (append order), which is send order; ``index`` is the block's
    position in its message's ``content`` so two attachments on one message keep the order
    the person attached them in. A session with no journal answers an empty list. Raises
    :class:`compression.ArchiveError` for an archive this build cannot decode - an honest
    failure the route maps to an error, never an empty "nothing attached".
    """
    session_dir = Path(session_dir)
    if not compression.has_transcript(session_dir):
        return {"items": [], "truncated": False}
    store = store_for_transcript_dir(session_dir)
    items: list[dict[str, Any]] = []
    truncated = False
    for line in compression.iter_lines(session_dir):
        if not _maybe_user_row(line):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        for item in items_from_row(row, store):
            if len(items) >= MAX_ITEMS:
                truncated = True
                break
            items.append(item)
        if truncated:
            break
    return {"items": items, "truncated": truncated}


def find_user_attachment(session_dir: Path, digest: str) -> dict[str, Any] | None:
    """The newest listed attachment carrying ``digest``, or ``None``.

    The materialise route's MEMBERSHIP check: the attachment store is shared across
    conversations, so a digest alone proves nothing about this session. Only a digest the
    person attached HERE may be copied into this session's scratchpad. Newest wins so a file
    attached twice under two names materialises under the name used most recently.
    """
    found = None
    for item in list_user_attachments(session_dir)["items"]:
        if item.get("attachment") == digest:
            found = item
    return found
