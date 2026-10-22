"""Team Bridge image attachments — the runtime's verification half.

WHY THIS MODULE EXISTS. A conversation message can carry up to four images
(docs/TEAM-BRIDGE-CONVERSATIONS.md, "Image attachments"). The hub stores them,
sniffs their type and binds the set to the message with ``attachments_hash``;
this runtime must never let an image reach an agent's model context unless a
screen that actually SAW those pixels allowed them. Two hops read bytes:

- the connector's ``purpose=screen`` fetch, whose bytes go ONLY to the isolated
  screening call (no tools, no session context, no disk, no store); and
- the ``bridge`` tool's ``purpose=context`` fetch after ``allow``, whose bytes
  become ``ImageContent`` in the agent's tool result.

Both hops apply the SAME checks, defined once here so they cannot drift:
descriptor shape, per-image sha256 + size against the hub's descriptor, the
recomputed ``attachments_hash``, a magic-byte sniff restricted to png/jpeg, and
the model bound (:func:`local_operator.imaging.bound_image_for_model`: 1024 px
edge, 1 MiB, 50 MP). Nothing here persists or logs bytes, hashes or ids — the
same no-body-cache rule as message text.

The descriptor is the hub's ``{id, mime, size, width, height, sha256}``; a
``state`` other than absent/``bound`` (e.g. ``expired`` after retention) has no
bytes to fetch and is reported, never fetched.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from local_operator.team_bridge.types import ATTACHMENT_ID_PATTERN

__all__ = [
    "ATTACHMENTS_MAX",
    "AttachmentDescriptor",
    "AttachmentRejected",
    "VerifiedImage",
    "attachments_hash",
    "parse_descriptors",
    "verify_and_bound",
    "withheld_marker",
]

#: The hub's per-message bound; a longer list is malformed, not truncated.
ATTACHMENTS_MAX = 4
#: The hub's byte ceiling; descriptors claiming more are malformed.
_MAX_BYTES = 5 * 1024 * 1024
_MIMES = ("image/png", "image/jpeg")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_ATTACHMENT_ID = re.compile(ATTACHMENT_ID_PATTERN)


class AttachmentRejected(ValueError):
    """One attachment failed verification. ``code`` is closed and content-free."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AttachmentDescriptor:
    """The hub's metadata for one bound image. No bytes; safe to render."""

    id: str
    mime: str
    size: int
    width: int
    height: int
    sha256: str
    state: str = "bound"

    @property
    def fetchable(self) -> bool:
        return self.state == "bound"


@dataclass(frozen=True)
class VerifiedImage:
    """One verified, model-bounded image. Bytes on a private carrier only."""

    descriptor: AttachmentDescriptor
    wire_mime: str

    def private_payload(self) -> bytes:
        return self._private_payload  # type: ignore[attr-defined]


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def parse_descriptors(raw: Any) -> tuple[AttachmentDescriptor, ...] | None:
    """The validated descriptor list, ``()`` when absent, ``None`` when malformed.

    Malformed is ANY entry failing the closed shape: the caller then treats
    the whole set as unverifiable rather than screening a subset — a partial
    image set screened as if complete would bind a verdict to pixels the
    screener never saw.
    """

    if raw is None:
        return ()
    if not isinstance(raw, list) or len(raw) > ATTACHMENTS_MAX:
        return None
    out: list[AttachmentDescriptor] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            return None
        attachment_id = entry.get("id")
        mime = entry.get("mime")
        sha = entry.get("sha256")
        size = _positive_int(entry.get("size"))
        width = _positive_int(entry.get("width"))
        height = _positive_int(entry.get("height"))
        state = entry.get("state", "bound")
        if (
            not isinstance(attachment_id, str)
            or _ATTACHMENT_ID.fullmatch(attachment_id) is None
            or mime not in _MIMES
            or not isinstance(sha, str)
            or _SHA256_HEX.fullmatch(sha) is None
            or size is None
            or size > _MAX_BYTES
            or width is None
            or height is None
            or not isinstance(state, str)
        ):
            return None
        out.append(
            AttachmentDescriptor(
                id=attachment_id,
                mime=str(mime),
                size=size,
                width=width,
                height=height,
                sha256=sha,
                state=state,
            )
        )
    if len({d.id for d in out}) != len(out):
        return None
    return tuple(out)


def attachments_hash(descriptors: Sequence[AttachmentDescriptor]) -> str:
    """The hub's binding hash: sha256 over ``"<sha256>:<size>\\n"`` in send order.

    Recomputed locally and compared with the hub's ``attachments_hash`` so the
    set the runtime screens is provably the set the hub bound — the image
    analogue of ``body_hash``.
    """

    canonical = "".join(f"{d.sha256}:{d.size}\n" for d in descriptors)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _verify_and_bound_sync(data: bytes, descriptor: AttachmentDescriptor) -> VerifiedImage:
    from local_operator.imaging import bound_image_for_model
    from local_operator.media import sniff_image

    if len(data) != descriptor.size:
        raise AttachmentRejected("attachment_size_mismatch")
    if hashlib.sha256(data).hexdigest() != descriptor.sha256:
        raise AttachmentRejected("attachment_sha256_mismatch")
    info = sniff_image(data)
    # The type comes from the BYTES, and must also agree with the hub's sniff:
    # png/jpeg only (gif/webp/heif are sendable elsewhere in the harness but
    # are not part of this contract and the hub never stores them).
    if info is None or info.mime_type not in _MIMES or info.mime_type != descriptor.mime:
        raise AttachmentRejected("attachment_type_mismatch")
    try:
        payload, wire_mime, _summary = bound_image_for_model(data, info)
    except ValueError:
        # Undecodable, truncated or over the 50 MP decode budget.
        raise AttachmentRejected("attachment_undecodable") from None
    image = VerifiedImage(descriptor=descriptor, wire_mime=wire_mime)
    object.__setattr__(image, "_private_payload", payload)
    return image


async def verify_and_bound(data: bytes, descriptor: AttachmentDescriptor) -> VerifiedImage:
    """Verify fetched bytes against their descriptor and bound them for a model.

    The decode is CPU-bound (hundreds of ms at the pixel ceiling), so it runs
    in a worker thread; the event loop that drives the connector and the
    session must not stall on it.
    """

    return await asyncio.to_thread(_verify_and_bound_sync, bytes(data), descriptor)


def withheld_marker(count: int) -> str:
    """D5's honest refusal: images exist but no vision screen admitted them."""

    noun = "attachment" if count == 1 else "attachments"
    return f"[{count} image {noun} withheld: screening model has no vision]"
