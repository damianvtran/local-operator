"""Team Bridge — an AGENT attaches local images to a ``dm`` / ``channel_post``.

WHY THIS MODULE EXISTS. Receiving an image is complete (``attachments.py``: the
fail-closed screener, hash binding, ``verify_and_bound``), but the SEND half had
no attachment surface at all. An agent asked to forward a picture therefore
pasted it into the message TEXT as ``[image …; sha256 …] data:image/jpeg;base64,…``
(the operator's audit screenshot of a multi-hundred-KiB base64 body is exactly
that): the bytes ride the model's own output, the transcript, the reply-guard
bookkeeping, the hub's text store and the audit view, and the receiver never gets
a screened image — only a wall of characters. The fix at the source is a real
attachment path plus a refusal of the workaround (:func:`refuse_inline_image_data`).

THE FLOW, and what is deliberately NOT here:

1. :func:`prepare_attachments` runs BEFORE any hub write and before a DM is even
   opened: validate the named local paths, read and sniff each file, bound it
   exactly as the human composer does (``imaging.bound_image_for_model``: 1024 px
   edge / 1 MiB / 50 MP, so the sender and the receiver's ``verify_and_bound``
   agree), and confirm the hub ADVERTISES ``attachments.agent_upload``. A refusal
   here costs no network write and no reply-guard budget.
2. :func:`stage_attachments` uploads each image AS the agent
   (``POST …/attachments?actor_endpoint_id=…``, image mode) and returns only the
   ``batt_`` ids + metadata. The caller then sends with ``attachment_ids``.
3. :func:`discard_attachments` deletes the staged ids when the send did not land.

Agent FILES (non-image) are out of scope: the server keeps agent file mode
disabled with no production switch (agent-server PR #283, ``agentFilesEnabled``),
and this module never sends ``X-Bridge-Attachment-Mode: file``.

NO BYTES LEAVE THIS MODULE'S BOUNDARY EXCEPT TO THE HUB. The tool result, the
transcript, the team-bridge store and every log line carry the ``batt_`` id, the
sniffed MIME, the size and the dimensions — never the bytes, a base64 form, a
data URI, a digest or a signed URL. :class:`PreparedImage` keeps its payload in a
private slot with a redacted ``repr`` so an accidental ``%r`` or traceback cannot
print it, and nothing here logs. Local filesystem paths are NOT echoed in a
refusal either: the model supplied them, but the refusal text is billed back into
context and persisted, and a basename is all the agent needs to find its error.

THE CAPABILITY IS DEFAULT-OFF. The hub emits ``attachments.agent_upload`` only when
its operator enabled endpoint-scoped image staging, and its own proposal says the
flag must stay off until every replica enforces the actor-aware predicates (an old
replica could bind another actor's staged row). A missing, malformed or ``false``
value therefore reads as OFF, and so does a status read that fails: the safe
answer to "may I stage an agent upload?" when the hub cannot be asked is no. The
flag is read FRESH on every attachment send rather than from the admission snapshot,
because a snapshot taken before an operator turned the flag off would keep
uploading into a rollback-unsafe hub; the cost is one status GET, paid only by
sends that carry attachments.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import stat
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from local_operator import browser_files
from local_operator.team_bridge.client import (
    ATTACHMENT_MAX_BYTES,
    ATTACHMENT_MAX_COUNT,
    ATTACHMENT_MIME_TYPES,
    attachments_capability,
)
from local_operator.team_bridge.types import ATTACHMENT_ID_PATTERN

if TYPE_CHECKING:  # pragma: no cover - typing only; tool.py imports THIS module
    from local_operator.team_bridge.tool import BridgeRuntime

__all__ = [
    "AttachmentRefused",
    "INLINE_IMAGE_DATA_MIN_CHARS",
    "PreparedImage",
    "StagedImage",
    "discard_attachments",
    "prepare_attachments",
    "annotate_receipt",
    "refuse_inline_image_data",
    "send_kwargs",
    "stage_attachments",
]

#: Ops that may carry ``attachments``. ``send``/``reply`` answer a pair-inbox
#: delivery and have no conversation to stage into; every other op is a read.
ATTACHMENT_OPS = ("dm", "channel_post")

#: Read ceiling for the SOURCE file, before bounding. Larger than the hub's 5 MiB
#: upload cap on purpose: a 12 MP camera JPEG is legitimately 6-10 MiB and bounds
#: down to well under 1 MiB, exactly as a human paste does. A pixel bomb is small
#: on disk and is caught by the 50 MP guard inside ``bound_image_for_model``, so
#: this cap is only about not slurping a multi-gigabyte file into memory.
_SOURCE_MAX_BYTES = 25 * 1024 * 1024

#: The signal for an inline image is the run length, not the prefix: a few hundred
#: characters can be a tiny inline icon in prose; ~1 KiB and up is an image payload
#: riding the message text. Counted in SIGNIFICANT characters (base64 alphabet,
#: ``%2B``/``%2F``/``%3D`` escapes, whitespace excluded) so wrapping cannot shrink it.
INLINE_IMAGE_DATA_MIN_CHARS = 1024

#: ``data:`` + image type + any ``;param`` list + ``,``. Deliberately loose around
#: the edges that a model (or a forwarder re-wrapping a paste) varies: case,
#: whitespace after the colon/semicolons/comma, and ``;charset=…``-style parameters.
#: Each repetition must consume a ``;`` so the group cannot backtrack quadratically.
_INLINE_IMAGE_HEADER = re.compile(
    r"data:\s*image/[a-z0-9.+-]+((?:\s*;\s*[^,;\s]*)*)\s*,",
    re.IGNORECASE,
)
_BASE64_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-")
#: The URL-escaped spellings of the base64 specials; a ``%2B`` form must count too.
_PCT_BASE64 = frozenset({"2b", "2f", "3d", "2B", "2F", "3D"})


def _significant_run(text: str, start: int, limit: int) -> int:
    """Base64-looking characters from ``start`` on, whitespace skipped, capped at ``limit``."""

    count = 0
    index = start
    size = len(text)
    while index < size and count < limit:
        char = text[index]
        if char in _BASE64_CHARS:
            count += 1
            index += 1
        elif char.isspace():
            index += 1
        elif char == "%" and text[index + 1 : index + 3] in _PCT_BASE64:
            count += 1
            index += 3
        else:
            break
    return count


_ATTACHMENT_ID = re.compile(ATTACHMENT_ID_PATTERN)


class AttachmentRefused(Exception):
    """A LOCAL refusal (nothing was written to the hub). ``code`` is closed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PreparedImage:
    """One validated, bounded image awaiting upload. Bytes stay private."""

    mime: str
    size: int
    width: int | None
    height: int | None
    # ``repr=False`` + a redacted ``__repr__`` below: the payload must not survive
    # into a traceback, a log line or a test failure diff.
    _payload: bytes = field(repr=False, compare=False, default=b"")

    def __repr__(self) -> str:
        return f"PreparedImage(mime={self.mime!r}, size={self.size}, bytes=<withheld>)"

    def payload(self) -> bytes:
        return self._payload


@dataclass(frozen=True)
class StagedImage:
    """What the model and the transcript may see of an uploaded image."""

    id: str
    mime: str
    size: int
    width: int | None = None
    height: int | None = None

    def as_receipt(self) -> dict[str, Any]:
        row: dict[str, Any] = {"id": self.id, "mime": self.mime, "size": self.size}
        if self.width and self.height:
            row["width"] = self.width
            row["height"] = self.height
        return row


def refuse_inline_image_data(text: str) -> dict[str, str] | None:
    """A refusal when ``text`` carries an inline ``data:image/…;base64,`` payload.

    Applies whether or not the hub advertises attachments: pasting image bytes
    into message text was never a working path, only an expensive one. The error
    names the supported route so the agent corrects itself rather than retrying
    the same paste, and says the route exists only where the hub supports agent
    uploads (the option is hub-gated, so recommending it unconditionally would
    send an agent to a refusal on a hub that has it off). Content-free by
    construction: the matched run is never echoed.
    """

    if "data:" not in text.lower():
        return None
    for match in _INLINE_IMAGE_HEADER.finditer(text):
        if "base64" not in match.group(1).lower():
            continue
        if _significant_run(text, match.end(), INLINE_IMAGE_DATA_MIN_CHARS) >= (
            INLINE_IMAGE_DATA_MIN_CHARS
        ):
            return {
                "code": "inline_image_data",
                "error": (
                    "the message text contains an inline data:image base64 payload "
                    f"(over {INLINE_IMAGE_DATA_MIN_CHARS} characters); nothing was sent. "
                    "Never paste image bytes into text. Where this hub supports agent "
                    "uploads, pass the local image file path(s) in the `attachments` "
                    f"parameter instead (PNG or JPEG, up to {ATTACHMENT_MAX_COUNT} per "
                    "dm or channel_post); where it does not, describe the image in words"
                ),
            }
    return None


# -- metadata stripping -------------------------------------------------------
#
# WHY. ``bound_image_for_model`` forwards an in-bounds image VERBATIM (it exists
# to make an image model-sized, not to sanitise it), so a phone photo would leave
# this machine with its EXIF block (GPS position, device, timestamps) and a PNG
# with its text chunks. A human attaching a photo picks it deliberately; an agent
# forwarding a file it found should not leak what nobody looked at. The strip is
# LOSSLESS (segments/chunks are dropped, pixels are never re-encoded) and runs on
# the final payload whichever rung produced it. EXIF ORIENTATION is not lost:
# the verbatim rung is taken only for images that need no rotation, and every
# rotated image is re-encoded upright by the ladder before this runs.

#: JPEG APPn segments kept, by marker AND payload identifier: the JFIF header, the
#: ICC profile (colour fidelity) and Adobe (colour transform flag — dropping it can
#: invert a CMYK/YCCK image). Matching the identifier, not just the marker, is what
#: drops a JFXX thumbnail (APP0) and the MPF multi-picture index (APP2). Everything
#: else in APP1..APP15 (EXIF, XMP, IPTC, Photoshop, maker notes) and COM is dropped.
_JPEG_KEEP_APP = {0xE0: b"JFIF\x00", 0xE2: b"ICC_PROFILE\x00", 0xEE: b"Adobe"}
#: PNG ancillary chunks that carry free text, EXIF or a timestamp.
_PNG_DROP_CHUNKS = frozenset({b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"})


def _strip_jpeg(data: bytes) -> bytes:
    out = bytearray(b"\xff\xd8")
    index = 2
    size = len(data)
    while index < size:
        if data[index] != 0xFF:
            raise ValueError("malformed jpeg")
        while index < size and data[index] == 0xFF:  # fill bytes
            index += 1
        if index >= size:
            raise ValueError("malformed jpeg")
        marker = data[index]
        index += 1
        if marker == 0xD9:  # EOI
            out += b"\xff\xd9"
            return bytes(out)
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:  # standalone, no length
            out += bytes((0xFF, marker))
            continue
        if index + 2 > size:
            raise ValueError("malformed jpeg")
        (length,) = struct.unpack(">H", data[index : index + 2])
        end = index + length
        if length < 2 or end > size:
            raise ValueError("malformed jpeg")
        if marker == 0xDA:  # SOS: header, then entropy-coded data up to the next real marker
            out += bytes((0xFF, marker)) + data[index:end]
            index = end
            start = index
            # In entropy-coded data every 0xFF is stuffed (FF00), a restart marker
            # (FFD0-D7) or a fill byte (FFFF); anything else is the next segment
            # (progressive JPEGs carry several scans, so this is a loop, not "to EOI").
            while index + 1 < size and not (
                data[index] == 0xFF
                and data[index + 1] not in (0x00, 0xFF)
                and not 0xD0 <= data[index + 1] <= 0xD7
            ):
                index += 1
            if index + 1 >= size:
                raise ValueError("malformed jpeg")
            out += data[start:index]
            continue
        payload = data[index + 2 : end]
        keep_id = _JPEG_KEEP_APP.get(marker)
        if keep_id is not None:
            drop = not payload.startswith(keep_id)
        else:
            drop = marker == 0xFE or 0xE0 <= marker <= 0xEF
        if not drop:
            out += bytes((0xFF, marker)) + data[index:end]
        index = end
    raise ValueError("malformed jpeg")


def _strip_png(data: bytes) -> bytes:
    signature = data[:8]
    out = bytearray(signature)
    index = 8
    size = len(data)
    while index < size:
        if index + 12 > size:
            raise ValueError("malformed png")
        (length,) = struct.unpack(">I", data[index : index + 4])
        kind = data[index + 4 : index + 8]
        end = index + 12 + length
        if end > size:
            raise ValueError("malformed png")
        if kind not in _PNG_DROP_CHUNKS:
            out += data[index:end]
        index = end
        if kind == b"IEND":  # anything after IEND is not part of the image
            return bytes(out)
    raise ValueError("malformed png")


def strip_image_metadata(data: bytes, mime: str) -> bytes:
    """``data`` without EXIF/XMP/IPTC/comment/text metadata; ``ValueError`` if unparseable."""

    if mime == "image/jpeg":
        return _strip_jpeg(data)
    if mime == "image/png":
        return _strip_png(data)
    raise ValueError("unsupported image type")


_POLICY_REFUSAL_PREFIX = "refused: "


def _refusal_from_gate(label: str, name: str, reason: str) -> AttachmentRefused:
    """Map a ``browser_files.check_upload`` reason onto a closed code + a path-free sentence.

    The gate's own text is reused only where it names the basename; its "does not
    exist"/"could not be opened" sentences echo the FULL path (and the nearest
    existing folder), which would turn this tool into a directory-listing oracle
    billed into context and persisted, so those become a fixed sentence.
    """

    body = reason.removeprefix(_POLICY_REFUSAL_PREFIX)
    if body.startswith(("does not exist", "could not be opened", "no path given")):
        return AttachmentRefused("attachment_unreadable", f"{label} ({name}) could not be opened")
    if "could not be read" in body:
        return AttachmentRefused("attachment_unreadable", f"{label} ({name}) could not be read")
    if "config directory" in body:
        # The gate's full text is relocation advice meant for a human-driven tool; an
        # agent that reads "copy it elsewhere" is being taught the bypass.
        return AttachmentRefused(
            "attachment_refused",
            f"{label} ({name}) is inside Local Operator's own config directory; only this "
            "session's own scratchpad can be attached from there",
        )
    if "byte limit" in body:
        return AttachmentRefused("attachment_too_large", f"{label} ({name}) is too large")
    if "not a regular file" in body:
        return AttachmentRefused("attachment_invalid", f"{label} ({name}) is not a regular file")
    if "is empty" in body:
        return AttachmentRefused("attachment_invalid", f"{label} ({name}) is empty")
    if "credential" in body:
        return AttachmentRefused(
            "attachment_refused", f"{label} ({name}) matches the credential deny-list"
        )
    return AttachmentRefused("attachment_invalid", f"{label} ({name}) cannot be attached")


def _read_and_bound(path_text: str, index: int, session_id: str) -> PreparedImage:
    """Read one local file, gate it, sniff it, bound it. Blocking: run in a thread.

    THE PATH GATE IS ``browser_files.check_upload``, and that is the security
    control of this feature, not a convenience. This tool lets a model name any
    host path and then publishes the bytes to a conversation other people (and
    other agents) read, so it is a confused-deputy exfiltration channel unless the
    path is judged first. ``check_upload`` is what the browser ``upload`` tool
    already uses for the same threat and, deliberately, it consults no approval
    policy (an auto-approved call must not be able to widen it): it resolves the
    path (a symlink is judged by its TARGET), refuses anything inside the config
    directory (other sessions' transcripts/screenshots, ``config.yml``, the secret
    store) except THIS session's own scratchpad — where an agent legitimately renders
    its images — rejects credential-named files and credential directories, and
    rejects non-regular files. The RESOLVED path is what is opened below.

    What it is NOT: a workspace allow-list. An image elsewhere on the host that the
    operator's account can read (``~/Pictures``) is attachable, exactly as it is for
    ``upload``/``image``; the approval prompt for a ``bridge`` write shows the raw
    arguments, ``attachments`` paths included, but that is visibility, not a control
    this module relies on.
    """

    from local_operator.imaging import bound_image_for_model
    from local_operator.media import sniff_image

    label = f"attachments[{index}]"
    if not isinstance(path_text, str) or not path_text.strip() or "\x00" in path_text:
        raise AttachmentRefused("attachment_invalid", f"{label} must be a file path")
    stripped = path_text.strip()
    if not stripped.startswith(("/", "~")):
        # No cwd contract exists for this tool, so a relative path would resolve
        # against whatever the server process happens to be in.
        raise AttachmentRefused(
            "attachment_invalid", f"{label} must be an absolute path (or start with ~/)"
        )
    # ``expanduser`` raises RuntimeError for ``~nosuchuser/x`` and a lone surrogate
    # raises UnicodeEncodeError at the first syscall: both are the MODEL's malformed
    # input, so they are a structured refusal, never an uncaught tool crash.
    try:
        name = browser_files.safe_name(Path(stripped).name or "file")
        policy = browser_files.Policy(upload_max_bytes=_SOURCE_MAX_BYTES)
        resolved, reason = browser_files.check_upload(
            stripped, cwd="/", policy=policy, session_id=session_id
        )
    except (RuntimeError, ValueError, OSError):
        raise AttachmentRefused("attachment_invalid", f"{label} is not a usable path") from None
    if resolved is None:
        raise _refusal_from_gate(label, name, reason)
    try:
        # O_NONBLOCK so a FIFO swapped in after the gate cannot hang open();
        # O_NOFOLLOW so a swap to a symlink after the gate is refused rather than
        # followed (the resolved path has no link components when it was judged).
        fd = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except (OSError, ValueError):
        raise AttachmentRefused(
            "attachment_unreadable", f"{label} ({name}) could not be opened"
        ) from None
    try:
        status = os.fstat(fd)
        if not stat.S_ISREG(status.st_mode):
            raise AttachmentRefused("attachment_invalid", f"{label} ({name}) is not a regular file")
        if status.st_size > _SOURCE_MAX_BYTES:
            raise AttachmentRefused(
                "attachment_too_large",
                f"{label} ({name}) is larger than {_SOURCE_MAX_BYTES // (1024 * 1024)} MiB",
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read(_SOURCE_MAX_BYTES + 1)
    except OSError:
        raise AttachmentRefused(
            "attachment_unreadable", f"{label} ({name}) could not be read"
        ) from None
    finally:
        os.close(fd)
    if len(data) > _SOURCE_MAX_BYTES:
        raise AttachmentRefused("attachment_too_large", f"{label} ({name}) is too large")
    if not data:
        raise AttachmentRefused("attachment_invalid", f"{label} ({name}) is empty")

    # The BYTES decide the type, never the extension: a renamed file is still
    # whatever its header says, and only PNG/JPEG are stored by the hub.
    info = sniff_image(data)
    if info is None or info.mime_type not in ATTACHMENT_MIME_TYPES:
        raise AttachmentRefused(
            "attachment_type_unsupported", f"{label} ({name}) is not a PNG or JPEG image"
        )
    try:
        payload, wire_mime, _summary = bound_image_for_model(data, info)
        # Fail closed: an image whose metadata cannot be proven stripped is refused,
        # not shipped with its EXIF/GPS (see the metadata-stripping note above).
        payload = strip_image_metadata(payload, wire_mime)
    except ValueError:
        # Undecodable, truncated, over the 50 MP decode budget, no decoder and too
        # large to forward unresized, or unparseable for stripping. The reason text
        # is not forwarded: it can name host packaging detail the model cannot act on.
        raise AttachmentRefused(
            "attachment_undecodable",
            f"{label} ({name}) could not be decoded as an image within the size limits",
        ) from None
    # Re-sniff what will actually be uploaded: the bounding ladder may have
    # re-encoded (PNG -> JPEG), and the hub identifies type from these bytes too.
    final = sniff_image(payload)
    if final is None or final.mime_type not in ATTACHMENT_MIME_TYPES:
        raise AttachmentRefused(
            "attachment_type_unsupported", f"{label} ({name}) is not a PNG or JPEG image"
        )
    if len(payload) > ATTACHMENT_MAX_BYTES:
        raise AttachmentRefused(
            "attachment_too_large", f"{label} ({name}) is still too large after resizing"
        )
    return PreparedImage(
        mime=final.mime_type,
        size=len(payload),
        width=final.width,
        height=final.height,
        _payload=payload,
    )


async def prepare_attachments(
    runtime: "BridgeRuntime", paths: Sequence[str]
) -> tuple[PreparedImage, ...]:
    """Validate the paths and the hub capability; raise :class:`AttachmentRefused`.

    Local checks run FIRST and need no network, so an obviously wrong call (a
    PDF, a missing file, five images) never costs a status read. The capability
    read comes last for the same reason.
    """

    if not paths:
        return ()
    if len(paths) > ATTACHMENT_MAX_COUNT:
        raise AttachmentRefused(
            "attachment_invalid", f"at most {ATTACHMENT_MAX_COUNT} attachments per message"
        )
    session_id = str(getattr(runtime, "session_id", "") or "")
    prepared: list[PreparedImage] = []
    for index, path in enumerate(paths):
        try:
            prepared.append(await asyncio.to_thread(_read_and_bound, path, index, session_id))
        except AttachmentRefused:
            raise
        except Exception:  # noqa: BLE001 — any other failure on model-named input is a refusal
            # A path the platform cannot even encode (lone surrogate), an
            # unexpected decoder error: refused with a closed code, never an
            # uncaught crash of the tool (dm and channel_post must agree).
            raise AttachmentRefused(
                "attachment_invalid", f"attachments[{index}] could not be processed"
            ) from None
    # The hub binds in send order and the client refuses duplicate ids; identical
    # bytes would come back as ONE staged id, so refuse the repeat up front with a
    # message the agent can act on rather than a late ValueError.
    digests = {hashlib.sha256(item.payload()).digest() for item in prepared}
    if len(digests) != len(prepared):
        raise AttachmentRefused("attachment_invalid", "the same image is attached more than once")

    capability = await _agent_upload_capability(runtime)
    if capability is None:
        raise AttachmentRefused(
            "attachments_unavailable",
            "this hub does not accept agent image attachments right now; nothing was "
            "sent. Send the message without `attachments`",
        )
    max_count = capability.get("max_count")
    if isinstance(max_count, int) and len(prepared) > max_count:
        raise AttachmentRefused(
            "attachment_invalid", f"this hub allows at most {max_count} attachments per message"
        )
    max_bytes = capability.get("max_bytes")
    if isinstance(max_bytes, int) and any(item.size > max_bytes for item in prepared):
        raise AttachmentRefused(
            "attachment_too_large", "an image exceeds this hub's attachment size limit"
        )
    return tuple(prepared)


async def _agent_upload_capability(runtime: "BridgeRuntime") -> dict[str, Any] | None:
    """The normalized capability block iff the hub says ``agent_upload`` is true."""

    try:
        payload = await runtime.client.status()
    except Exception:  # noqa: BLE001 — cannot ask the hub => treat as off (fail closed)
        return None
    capability = attachments_capability(payload)
    if capability is None or capability.get("agent_upload") is not True:
        return None
    return capability


async def stage_attachments(
    runtime: "BridgeRuntime",
    conversation_id: str,
    endpoint_id: str,
    prepared: Sequence[PreparedImage],
) -> list[StagedImage]:
    """Upload each image as the agent endpoint; return ids + metadata only.

    All-or-nothing: if any upload fails, the ones already staged are discarded
    and the original failure propagates (the caller words it). A hub answer that
    does not carry a well-formed ``batt_`` id is treated as a failure too — an id
    that is not minted-shaped must never reach a send body or a URL path.
    """

    staged: list[StagedImage] = []
    try:
        for item in prepared:
            result = await runtime.client.upload_attachment(
                conversation_id,
                item.payload(),
                tenant_id=runtime.tenant_id,
                actor_endpoint_id=endpoint_id,
                expected_account_id=runtime.owner.account_id,
            )
            row = result.get("attachment") if isinstance(result, dict) else None
            attachment_id = row.get("id") if isinstance(row, dict) else None
            if (
                not isinstance(attachment_id, str)
                or _ATTACHMENT_ID.fullmatch(attachment_id) is None
            ):
                raise ValueError("the hub returned no usable attachment id")
            staged.append(
                StagedImage(
                    id=attachment_id,
                    mime=item.mime,
                    size=item.size,
                    width=item.width,
                    height=item.height,
                )
            )
    except BaseException:
        await discard_attachments(runtime, conversation_id, endpoint_id, staged)
        raise
    return staged


#: Strong references to in-flight cleanups: a shielded task that nothing else holds
#: can be garbage-collected mid-run once its awaiter is cancelled.
_CLEANUPS: set["asyncio.Future[None]"] = set()


async def discard_attachments(
    runtime: "BridgeRuntime",
    conversation_id: str,
    endpoint_id: str,
    staged: Sequence[StagedImage],
) -> None:
    """Best-effort delete of the ids THIS call staged after a send that did not land.

    Never raises on a delete failure: this runs on a failure path whose own error
    the caller is about to report, and an unreachable hub must not replace it with
    a second one. The hub deletes only STAGED rows (a bound one stays) and expires
    the rest after 72 hours, so a failed cleanup leaks nothing durable.

    Runs shielded so a cancelled turn (``CancelledError`` is how an interrupt
    arrives) still finishes deleting rather than abandoning the ids half way.

    DEDUPE CAVEAT, stated because it bounds what "only ids this call created" can
    mean: the hub returns the SAME staged id for identical bytes from the same actor
    (agent-server PR #283: "Re-uploading the same bytes as the same actor while staged
    returns the same attachment") and its upload response — ``BridgeAttachmentView``
    ``{id, mime, size, width, height, sha256, state, expires_at}`` — carries no
    "existing"/deduplicated flag, so a client cannot tell a fresh stage from a re-used
    one. Within one call duplicates are refused up front; across calls, deleting on
    failure can remove an id a concurrent send of the same image by the same endpoint
    is about to bind. That send then fails its bind and runs its own cleanup, and the
    agent retries; nothing already BOUND is ever touched (the hub keeps bound rows).
    That is accepted rather than leaking a staged row for every failed send.
    """

    if not staged:
        return

    async def _delete_all() -> None:
        for item in staged:
            try:
                await runtime.client.delete_attachment(
                    conversation_id,
                    item.id,
                    tenant_id=runtime.tenant_id,
                    actor_endpoint_id=endpoint_id,
                    expected_account_id=runtime.owner.account_id,
                )
            except Exception:  # noqa: BLE001 — see docstring
                continue

    task = asyncio.ensure_future(_delete_all())
    _CLEANUPS.add(task)
    task.add_done_callback(_CLEANUPS.discard)
    await asyncio.shield(task)


def send_kwargs(staged: Sequence[StagedImage]) -> dict[str, Any]:
    """Extra ``conversation_send`` kwargs: ``attachment_ids`` only when staged.

    Empty (not ``attachment_ids=[]``) for a text-only send, so the call — and
    therefore the wire — is byte-identical to the pre-attachment one, which is
    what a hub without the capability (and every existing caller) requires.
    """

    return {"attachment_ids": [item.id for item in staged]} if staged else {}


def annotate_receipt(receipt: dict[str, Any], staged: Sequence[StagedImage]) -> None:
    """Add the ids-and-metadata block to a send receipt; no-op without images.

    Mutates the receipt in place so ``_post_receipt`` stays the ONE construction
    site for a conversation write. Only ``batt_`` ids and ``{mime, size, width,
    height}`` are added — the receipt is persisted with the transcript row.
    """

    if not staged:
        return
    receipt["attachments"] = [item.as_receipt() for item in staged]
    noun = "image" if len(staged) == 1 else "images"
    receipt["content"] = (
        f"{receipt.get('content', '')}; attached {len(staged)} {noun} "
        f"({', '.join(item.id for item in staged)})"
    )
