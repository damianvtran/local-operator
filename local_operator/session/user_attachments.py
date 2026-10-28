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
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Literal

from local_operator.harness.rows import is_harness_injection
from local_operator.session import compression
from local_operator.session.attachments import AttachmentStore, store_for_transcript_dir

logger = logging.getLogger(__name__)

#: Upper bound on rows returned. A person does not attach thousands of files to one
#: conversation; the cap exists so a pathological (scripted, hand-built) journal cannot turn a
#: canvas open into a multi-megabyte response. ``truncated`` tells the client the list is not
#: the whole story, so it never presents a capped list as complete.
MAX_ITEMS = 2000

#: The same cheap user-row probe as session_files/scan.py. Whitespace is not fixed:
#: hand-built journals and older tools can use either compact or spaced JSON.
_USER_ROW = re.compile(r'"role"\s*:\s*"user"')

#: Quoted keys, one of which every media block carries. ``"mime_type"`` alone is NOT enough: an
#: inline PNG whose mime defaulted has no ``mime_type`` key on the durable row at all.
_MEDIA_NEEDLES = ('"data"', '"attachment"', '"mime_type"')

#: Keys whose presence marks a content block as MEDIA rather than text. Mirrors
#: ``transcript._externalize_attachments``: identify by ``data``/``attachment``, never by
#: ``type`` alone, because ``type`` is the pydantic default and absent on a durable image.
_MEDIA_KEYS = ("data", "attachment")

Kind = Literal["document", "image"]

#: ``(records, truncated)``: the transcript-derived half of a listing, the unit the cache holds.
#: A tuple so the shared cached value cannot be appended to by a caller.
_Inventory = tuple[tuple[dict[str, Any], ...], bool]


def _maybe_user_row(line: str) -> bool:
    """Cheap pre-filter: could this line be a user message carrying media?

    A 300 MB journal is overwhelmingly tool output; parsing every line as JSON to discard it
    is the cost this avoids. Both conditions are necessary for a match below, so a ``False``
    here never hides a real attachment.
    """
    if not _USER_ROW.search(line):
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
    # A user-row kind:"image" is a pasted image in model context, which the file index
    # already owns as a user artifact. Other kind-carrying blocks (audio/video) are not uploads.
    if "kind" in block and block.get("kind") != "image":
        return None
    if block_type in (None, "image"):
        mime = block.get("mime_type") or block.get("content_type")
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
    mime = (
        _str_or_none(block.get("mime_type"))
        or _str_or_none(block.get("content_type"))
        or ("image/png" if kind == "image" else "application/octet-stream")
    )
    size = _int_or_none(block.get("bytes"))
    if digest is None and data:
        # Read-only lookup: never populate the store from a read path.
        ref = store.locate(data, mime)
        if ref is not None:
            digest = ref.digest
        if size is None:
            size = _decoded_size(data)
    # Journal paths are device/config-local. A synced row can name another device's path
    # while this device's content-addressed store has the bytes, so the digest is authoritative.
    stored_path = (
        str(store.path_for(digest))
        if digest is not None
        else _str_or_none(block.get("stored_path"))
    )
    injection = block.get("injection")
    # Transcript-derived only. ``exists``/``stored`` (and a ``bytes`` taken from the store
    # copy) are filesystem facts that change without any journal write, so they are NOT part
    # of this record: :func:`_with_disk_state` stats them on every listing.
    return {
        "message_id": message_id,
        "ts": ts,
        "index": index,
        "kind": kind,
        "attachment": digest,
        "filename": _str_or_none(block.get("filename")) or _str_or_none(block.get("name")),
        "mime_type": mime,
        "bytes": size,
        "path": _str_or_none(block.get("path")),
        "stored_path": stored_path,
        "injection": injection if injection in ("native", "extracted", "reference") else None,
    }


def _with_disk_state(record: dict[str, Any]) -> dict[str, Any]:
    """A fresh wire item: the cached row record plus ``exists``/``stored`` stat'd NOW.

    WHY PER CALL. The operator requirement is that a file which is no longer there SAYS SO
    rather than failing silently, and the canvas derives "Moved or deleted"/"gone" directly
    from these two flags. In a settled conversation nothing is ever appended to the journal,
    so a flag computed at scan time and cached against the journal's identity would never
    turn ``False`` (PR #3042 round-2 R2-1). At most two stats per item (``MAX_ITEMS`` caps the
    list) is cheap beside a journal rescan, and the route already runs this off the event loop.

    Returns a NEW dict every time, so a caller mutating its listing cannot corrupt the cache.
    """
    item = dict(record)
    size = item["bytes"]
    stored = False
    stored_path = item["stored_path"]
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
    # Whether the ORIGIN is still where the person attached it from. ``None`` when the row
    # names no origin (a pasted screenshot) - unknown is not "gone". Stat'd here rather than
    # by the renderer so one listing answers every row without a per-row bridge round trip.
    exists: bool | None = None
    origin = item["path"]
    if origin is not None:
        try:
            exists = Path(origin).is_file()
        except OSError:
            exists = None
    item["bytes"] = size
    item["exists"] = exists
    item["stored"] = stored
    return item


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


#: Settled-turn re-reads and materialise share this small revision cache of TRANSCRIPT-DERIVED
#: records (no ``exists``/``stored``: those are re-stat'd per call). A hit costs one stat of the
#: journal; edits and compression invalidate on the source's physical identity. Nothing is
#: written to disk. The route reads it from ``asyncio.to_thread`` workers, so every
#: get/move/set/evict happens under ``_CACHE_LOCK``: an unlocked ``move_to_end`` racing another
#: thread's eviction raises ``KeyError`` (PR #3042 round-2 R2-2). The scan itself runs outside
#: the lock so one slow journal never serialises every other session's listing.
_CACHE_SIZE = 8
_INVENTORY_CACHE: OrderedDict[tuple[str, str, int, int, int, int], _Inventory] = OrderedDict()
_CACHE_LOCK = threading.Lock()


def _inventory_cache_key(directory: Path) -> tuple[str, str, int, int, int, int] | None:
    path, _codec = compression.transcript_source(directory)
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(directory), str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _inventory(session_dir: Path) -> _Inventory:
    """The cached (or freshly scanned) row records. Shared and read-only: never mutate."""
    cache_key = _inventory_cache_key(session_dir)
    if cache_key is not None:
        with _CACHE_LOCK:
            cached = _INVENTORY_CACHE.get(cache_key)
            if cached is not None:
                _INVENTORY_CACHE.move_to_end(cache_key)
                return cached
    result = _scan_user_attachments(session_dir)
    if cache_key is not None:
        with _CACHE_LOCK:
            _INVENTORY_CACHE[cache_key] = result
            _INVENTORY_CACHE.move_to_end(cache_key)
            while len(_INVENTORY_CACHE) > _CACHE_SIZE:
                _INVENTORY_CACHE.popitem(last=False)
    return result


def list_user_attachments(
    session_dir: Path, *, deny: Callable[[str], bool] | None = None
) -> dict[str, Any]:
    """``{items, truncated}`` for the conversation in ``session_dir``, oldest first.

    Order is journal order (append order), which is send order; ``index`` is the block's
    position in its message's ``content`` so two attachments on one message keep the order
    the person attached them in. A session with no journal answers an empty list. Raises
    :class:`compression.ArchiveError` for an archive this build cannot decode - an honest
    failure the route maps to an error, never an empty "nothing attached". The answer is
    always freshly built (new dicts), with ``exists``/``stored`` stat'd on this call.

    ``deny`` (``session_files.service.make_deny`` for this config dir) adds ``materialisable``
    to every item: ``False`` when the sensitive-path gate refuses the row's origin path, its
    original filename or the name a materialise would write (PR #3042 QA round 3, Q1). The
    row is still LISTED - the person attached it and the transcript shows it - but no copy of
    it may be written. Computed per call like ``exists``: the denylist reads the live config
    dir and resolves symlinks, so it is not a journal fact the cache may hold.
    """
    records, truncated = _inventory(Path(session_dir))
    items = [_with_disk_state(record) for record in records]
    if deny is not None:
        # Materialise is addressed by DIGEST and the store is content-addressed, so the bytes
        # of an attached ``.env`` re-attached as ``notes.txt`` are the same secret under an
        # innocent name. A digest any row of which is denied is refused for EVERY row sharing
        # it, which keeps this flag and the route's verdict (``find_user_attachment``) equal.
        verdicts = [_record_denied(record, deny) for record in records]
        refused = {r.get("attachment") for r, bad in zip(records, verdicts) if bad}
        refused.discard(None)
        for item, bad in zip(items, verdicts):
            item["materialisable"] = not (bad or item.get("attachment") in refused)
    return {"items": items, "truncated": truncated}


def _materialised_name(record: dict[str, Any]) -> str:
    """The name ``…/materialise`` would write, computed exactly as the route computes it."""
    from local_operator.session_files.service import sanitise_name

    return sanitise_name(
        record.get("filename"), record.get("attachment") or "0" * 32, record.get("mime_type")
    )


def _record_denied(record: dict[str, Any], deny: Callable[[str], bool]) -> bool:
    from local_operator.session_files.service import attachment_denied

    return attachment_denied(
        deny,
        origin=record.get("path"),
        filename=record.get("filename"),
        materialised_name=_materialised_name(record),
    )


def _scan_user_attachments(session_dir: Path) -> _Inventory:
    if not compression.has_transcript(session_dir):
        return (), False
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
    return tuple(items), truncated


def find_user_attachment(
    session_dir: Path, digest: str, *, deny: Callable[[str], bool] | None = None
) -> dict[str, Any] | None:
    """The newest listed attachment carrying ``digest``, or ``None``.

    The materialise route's MEMBERSHIP check: the attachment store is shared across
    conversations, so a digest alone proves nothing about this session. Only a digest the
    person attached HERE may be copied into this session's scratchpad. Newest wins so a file
    attached twice under two names materialises under the name used most recently. With
    ``deny``, ``materialisable`` is ``False`` when ANY row carrying the digest is denied.
    """
    found = None
    matches: list[dict[str, Any]] = []
    # Membership needs only the cached records; stat just the one item that is returned.
    for record in _inventory(Path(session_dir))[0]:
        if record.get("attachment") == digest:
            found = record
            matches.append(record)
    if found is None:
        return None
    item = _with_disk_state(found)
    if deny is not None:
        # The same per-digest verdict the listing reports (see ``list_user_attachments``).
        item["materialisable"] = not any(_record_denied(match, deny) for match in matches)
    return item
