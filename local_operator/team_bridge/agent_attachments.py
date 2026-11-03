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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

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

#: ``data:image/<type>;base64,`` followed by a base64 run (line breaks allowed: a
#: model that wraps its paste still pastes). The run length is the signal — a few
#: hundred characters can be a tiny inline icon in prose; ~1 KiB and up is an
#: image payload riding the message text.
INLINE_IMAGE_DATA_MIN_CHARS = 1024
_INLINE_IMAGE_DATA = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,(?:[A-Za-z0-9+/=_-]|\r?\n){%d,}"
    % INLINE_IMAGE_DATA_MIN_CHARS
)

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
    the same paste. Content-free by construction: the matched run is never
    echoed.
    """

    if "data:image/" not in text:
        return None
    if _INLINE_IMAGE_DATA.search(text) is None:
        return None
    return {
        "code": "inline_image_data",
        "error": (
            "the message text contains an inline data:image base64 payload "
            f"(over {INLINE_IMAGE_DATA_MIN_CHARS} characters); nothing was sent. Do not paste "
            "image bytes into text: pass the local image file path(s) in the "
            f"`attachments` parameter instead (PNG or JPEG, up to {ATTACHMENT_MAX_COUNT} per "
            "message, dm or channel_post)"
        ),
    }


def _read_and_bound(path_text: str, index: int) -> PreparedImage:
    """Read one local file, sniff it, bound it. Blocking: run in a thread."""

    from local_operator.imaging import bound_image_for_model
    from local_operator.media import sniff_image

    label = f"attachments[{index}]"
    if not isinstance(path_text, str) or not path_text.strip() or "\x00" in path_text:
        raise AttachmentRefused("attachment_invalid", f"{label} must be a file path")
    path = Path(path_text.strip()).expanduser()
    if not path.is_absolute():
        # No cwd contract exists for this tool, so a relative path would resolve
        # against whatever the server process happens to be in.
        raise AttachmentRefused(
            "attachment_invalid", f"{label} must be an absolute path (or start with ~/)"
        )
    name = path.name or "file"
    try:
        # O_NONBLOCK so a FIFO named by the model cannot hang the tool at open();
        # the fstat below then refuses anything that is not a regular file.
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    except OSError:
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
        payload, _wire_mime, _summary = bound_image_for_model(data, info)
    except ValueError:
        # Undecodable, truncated, over the 50 MP decode budget, or no decoder and
        # too large to forward unresized. The reason text is not forwarded: it can
        # name host packaging detail the model cannot act on.
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
    prepared = [await asyncio.to_thread(_read_and_bound, p, i) for i, p in enumerate(paths)]
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


async def discard_attachments(
    runtime: "BridgeRuntime",
    conversation_id: str,
    endpoint_id: str,
    staged: Sequence[StagedImage],
) -> None:
    """Best-effort delete of STAGED ids after a send that did not land.

    Never raises: this runs on a failure path whose own error the caller is
    about to report, and an unreachable hub must not replace it with a second
    one. The hub deletes only staged rows (a bound one stays) and expires the
    rest after 72 hours, so a failed cleanup leaks nothing durable.
    """

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
