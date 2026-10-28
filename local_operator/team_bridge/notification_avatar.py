"""Precomposed sender images for Team Bridge OS notifications.

WHY THIS EXISTS. A desktop notification has one image slot (macOS attaches it
as a right-side thumbnail, Windows toasts show it as ``appLogoOverride``,
libnotify hands it to the shell as image-data). To make a bridge notification
read as "from Ben" rather than "from the app", that one slot carries ONE PNG:
the sender's circle-cropped photo -- or an initials placeholder on a stable
per-person colour -- with the Local Operator mark in a bottom-right corner
badge and, for a person's agent (chief of staff), a separate agent marker.

CONSTRAINTS (design contract, Team Bridge agreement, 2026-10-28):

* Composition lives here, in core, so every consumer (desktop main today,
  other surfaces later) gets the same validated bytes, and no consumer ever
  decodes a remote image itself.
* The ONLY photo source is :meth:`TeamBridgeClient.fetch_avatar` -- the closed
  ``/accounts/{id}/avatar`` read on the configured hub, redirects disabled,
  1 MiB cap, revalidated under the current login on every use. This module
  never takes or follows a URL.
* The picture is PRE-CROPPED with transparent corners and the badge sits fully
  inside the canvas, because Windows' optional ``hint-crop="circle"`` would
  clip a badge drawn at the corner of a square image.
* Untrusted bytes are decoded under explicit limits (format allowlist,
  dimension and pixel ceilings checked BEFORE the bitmap is allocated, a
  bounded number of concurrent decodes off the event loop) and re-encoded, so
  no metadata or malformed chunk survives into the output.
* NO COMPOSITE CACHE IN CORE. Every call renders from bytes the hub has
  just (re)authorized under the current login, and the route answers
  ``private, no-store``. Display retention belongs to the desktop main
  process, under the operator's A0 policy (2026-10-28): an in-memory cache
  of at most 15 minutes, keyed viewer+auth epoch+tenant+sender+hash+size,
  cleared on every auth transition, which Team Bridge agreed to. A second,
  core-side copy of the same pixels would need its own purge story, and in
  review round 1 that story failed twice (a cancelled render could re-insert an old login's
  composite after a purge; a ``no-store`` reply discarded only one sibling
  variant). Being stateless removes both by construction: there is
  nothing to purge. A render costs ~5 ms at 64 px and ~18 ms at 256 px
  (measured on an M-series Mac), well inside the desktop's
  250 ms miss budget.

Pillow is not a new dependency: ``pillow-heif`` (a base dependency) requires
it, and it is imported lazily so a server that never renders pays nothing.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    from PIL.Image import Image

Kind = Literal["person", "person_agent"]
Theme = Literal["light", "dark"]

KINDS: frozenset[str] = frozenset({"person", "person_agent"})
THEMES: frozenset[str] = frozenset({"light", "dark"})
SIZES: frozenset[int] = frozenset({64, 256})

#: Decode ceilings for an untrusted avatar. The hub serves 64/256 px
#: derivatives; 1024 px per side leaves headroom for a future 512 without
#: letting a 1 MiB PNG declare a 30k x 30k canvas (a decompression bomb
#: allocates by DECLARED size, so this is checked before ``load``).
MAX_SOURCE_SIDE = 1024
MAX_SOURCE_PIXELS = MAX_SOURCE_SIDE * MAX_SOURCE_SIDE
SOURCE_FORMATS = ("PNG", "JPEG")
#: Output cap; a 256 px RGBA PNG of a photo is ~60-120 KiB.
MAX_OUTPUT_BYTES = 256 * 1024

#: Badge is ~40 % of the canvas, as agreed for the corner mark.
BADGE_RATIO = 0.40
AGENT_MARKER_RATIO = 0.30
#: Transparent gap cut around each overlay so it reads as a separate token on
#: any notification background.
GAP_RATIO = 0.035
#: Initials layout. Measured, not eyeballed: with the badge and agent-marker
#: clearances (plus the circle crop) as the exclusion mask, every two-character
#: pair of A-Z/0-9 in the bundled font loses 0 % of its ink at this size and
#: centre (design review D1). Centred text at 40 % lost up to ~11 % (``CQ``).
#: Accented letters fold to their base letter in ``_drawable``, so the ASCII
#: set covers them. The centre sits up and left, away from the bottom-right
#: badge; ``tests/unit/team_bridge/test_notification_avatar.py`` re-measures it.
INITIALS_RATIO = 0.33
INITIALS_CENTRE = (0.43, 0.45)
#: Supersampling for antialiased circles without a vector renderer.
_SUPERSAMPLE = 4

_MARK_PATH = Path(__file__).resolve().parent.parent / "mobile" / "static" / "mark.png"

# Initials palettes: fixed, mid-saturation tones that hold white text at
# >= 4.5:1 in BOTH themes. That is the normal-text bar: at the intended 32/48
# px appearance, 64 px initials scale to ~10-16 px, which is not large text
# (design review D2). The dark set is the light-theme hues, a step brighter so
# they stay distinct on a dark shell; each was darkened in HLS lightness only
# (hue/saturation kept, so the order and per-owner mapping are unchanged)
# until white reached >= 4.6:1. The
# colour is chosen by a digest of (hub origin, account id) -- never the display
# name -- so renaming someone never recolours them.
_PALETTE_LIGHT = (
    (0x2F, 0x6F, 0xB3),
    (0x8A, 0x4F, 0xB0),
    (0xB0, 0x47, 0x6B),
    (0xA8, 0x55, 0x1E),
    (0x3D, 0x7D, 0x3A),
    (0x1E, 0x7A, 0x78),
    (0x5A, 0x5F, 0xC8),
    (0x7A, 0x63, 0x45),
)
_PALETTE_DARK = (
    (0x39, 0x77, 0xBA),
    (0x96, 0x5B, 0xBE),
    (0xBD, 0x4F, 0x75),
    (0xAE, 0x61, 0x2D),
    (0x46, 0x82, 0x42),
    (0x27, 0x81, 0x7F),
    (0x64, 0x6A, 0xD4),
    (0x88, 0x71, 0x54),
)
_BADGE_STYLE = {
    # (disc fill, mark ink)
    "light": ((0xFF, 0xFF, 0xFF, 0xFF), (0x6F, 0x6A, 0x62, 0xFF)),
    "dark": ((0x2B, 0x2B, 0x2E, 0xFF), (0xE6, 0xE2, 0xDA, 0xFF)),
}
_AGENT_STYLE = {
    "light": ((0x24, 0x24, 0x28, 0xFF), (0xFF, 0xFF, 0xFF, 0xFF)),
    "dark": ((0xF2, 0xF2, 0xF4, 0xFF), (0x24, 0x24, 0x28, 0xFF)),
}

#: At most two decodes/renders at once across the process: a burst of
#: notifications must not fan out into N simultaneous bitmap allocations.
_RENDER_SLOTS = threading.BoundedSemaphore(2)
# Bound queued work as well as active decodes; a flood must not retain arbitrary
# image bytes in executor closures. Overload is a normal app-icon fallback.
_PENDING = threading.BoundedSemaphore(8)


class AvatarImageRejected(ValueError):
    """The authorized bytes are not an image this renderer will decode."""


@dataclass(frozen=True)
class Variant:
    """Everything that changes the rendered pixels."""

    kind: Kind
    theme: Theme
    size: int
    initials: str


def valid_initials(value: str) -> bool:
    """At most two letters/digits after NFC; anything else is refused.

    Closed so a query cannot smuggle a long string, controls or bidi marks
    into the drawing; the desktop derives them from the directory name.
    """

    text = unicodedata.normalize("NFC", value)
    return len(text) <= 2 and all(unicodedata.category(ch)[0] in ("L", "N") for ch in text)


def palette_colour(origin: str, account_id: str, theme: Theme) -> tuple[int, int, int]:
    palette = _PALETTE_DARK if theme == "dark" else _PALETTE_LIGHT
    digest = hashlib.sha256(f"{origin}\n{account_id}".encode()).digest()
    return palette[digest[0] % len(palette)]


def decode_source(data: bytes) -> Image:
    """Decode one untrusted avatar under explicit limits, as RGBA.

    ``Image.open`` reads only the header, so the declared size is checked
    before ``load`` allocates the bitmap. Pillow's own bomb guard is global
    state other code may tune; these ceilings are local and much tighter.
    """

    from PIL import Image as PILImage
    from PIL import UnidentifiedImageError

    if not data or len(data) > 1024 * 1024:
        raise AvatarImageRejected("source byte limit")
    try:
        image = PILImage.open(io.BytesIO(data), formats=list(SOURCE_FORMATS))
    # DecompressionBombError subclasses neither OSError nor ValueError, and
    # Pillow raises it from ``open`` when the declared size is huge.
    except (UnidentifiedImageError, OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise AvatarImageRejected("unsupported image") from exc
    width, height = image.size
    if (
        width < 1
        or height < 1
        or width > MAX_SOURCE_SIDE
        or height > MAX_SOURCE_SIDE
        or width * height > MAX_SOURCE_PIXELS
        or width * height * 4 > max(len(data), 1) * 4096
        or getattr(image, "n_frames", 1) != 1
    ):
        raise AvatarImageRejected("image dimensions out of bounds")
    try:
        image.load()
        return image.convert("RGBA")
    except (OSError, ValueError, PILImage.DecompressionBombError) as exc:
        raise AvatarImageRejected("undecodable image") from exc


@lru_cache(maxsize=1)
def _mark_alpha() -> Image:
    """The LO mark's alpha channel, recoloured per theme at draw time."""

    from PIL import Image as PILImage

    with PILImage.open(_MARK_PATH) as mark:
        return mark.convert("RGBA").getchannel("A").copy()


def _circle_mask(side: int) -> Image:
    from PIL import Image as PILImage
    from PIL import ImageDraw

    big = PILImage.new("L", (side * _SUPERSAMPLE, side * _SUPERSAMPLE), 0)
    ImageDraw.Draw(big).ellipse((0, 0, big.width - 1, big.height - 1), fill=255)
    return big.resize((side, side), PILImage.Resampling.LANCZOS)


def _cover(image: Image, side: int) -> Image:
    """Centre-crop to a square, then scale to ``side`` (object-fit: cover)."""

    from PIL import Image as PILImage

    w, h = image.size
    edge = min(w, h)
    left, top = (w - edge) // 2, (h - edge) // 2
    return image.crop((left, top, left + edge, top + edge)).resize(
        (side, side), PILImage.Resampling.LANCZOS
    )


@lru_cache(maxsize=4)
def _initials_font(px: int):  # noqa: ANN202 - PIL font type is not public
    from PIL import ImageFont

    # Pillow's bundled default (Aileron, FreeType) -- no system font lookup,
    # so the output is identical on every host.
    return ImageFont.load_default(size=px)


def _has_glyphs(font, text: str) -> bool:  # noqa: ANN001
    """False when the bundled font would draw a tofu box for any character."""

    missing = _glyph_bytes(font, "\U0010fffd")
    return all(_glyph_bytes(font, ch) != missing for ch in text)


def _drawable(font, text: str) -> str:  # noqa: ANN001
    """The initials as the bundled font can draw them, or "" for the silhouette.

    Pillow's bundled face covers basic Latin only. Accented Latin folds to its
    base letter (É -> E: still the right initial); any other script yields ""
    so the caller draws a neutral person instead of a tofu box.
    """

    out = []
    for ch in text:
        if not _has_glyphs(font, ch):
            base = unicodedata.normalize("NFKD", ch)[:1]
            if not base or not _has_glyphs(font, base):
                return ""
            ch = base
        out.append(ch)
    return "".join(out)


def _glyph_bytes(font, ch: str) -> bytes:  # noqa: ANN001
    from PIL import Image as PILImage
    from PIL import ImageDraw

    cell = PILImage.new("L", (font.size * 2, font.size * 2), 0)
    ImageDraw.Draw(cell).text((0, 0), ch, font=font, fill=255)
    return cell.tobytes()


def _placeholder(side: int, colour: tuple[int, int, int], initials: str) -> Image:
    from PIL import Image as PILImage
    from PIL import ImageDraw

    tile = PILImage.new("RGBA", (side, side), (*colour, 255))
    draw = ImageDraw.Draw(tile)
    font = _initials_font(max(8, int(side * INITIALS_RATIO)))
    text = _drawable(font, unicodedata.normalize("NFC", initials).upper())
    if text:
        centre = (side * INITIALS_CENTRE[0], side * INITIALS_CENTRE[1])
        draw.text(centre, text, font=font, fill=(255, 255, 255, 255), anchor="mm")
    else:
        # Generic person: no initials, or a script the bundled font lacks.
        # Better a neutral silhouette than a tofu box or a guessed letter.
        head = side * 0.18
        cx, cy = side / 2, side * 0.40
        draw.ellipse((cx - head, cy - head, cx + head, cy + head), fill=(255, 255, 255, 230))
        draw.ellipse(
            (side * 0.22, side * 0.64, side * 0.78, side * 1.10), fill=(255, 255, 255, 230)
        )
    return tile


def _disc(side: int, fill: tuple[int, int, int, int]) -> Image:
    from PIL import Image as PILImage

    disc = PILImage.new("RGBA", (side, side), fill)
    disc.putalpha(_circle_mask(side))
    return disc


def _badge(side: int, theme: Theme) -> Image:
    from PIL import Image as PILImage

    fill, ink = _BADGE_STYLE[theme]
    disc = _disc(side, fill)
    inner = max(1, int(side * 0.74))
    alpha = _mark_alpha().resize((inner, inner), PILImage.Resampling.LANCZOS)
    glyph = PILImage.new("RGBA", (inner, inner), ink)
    glyph.putalpha(alpha)
    offset = (side - inner) // 2
    disc.alpha_composite(glyph, (offset, offset))
    return disc


def _agent_marker(side: int, theme: Theme) -> Image:
    """A four-point spark on a contrasting disc: "an agent", not "the app".

    Deliberately a different shape and colour from the LO badge so the two
    never read as one mark.
    """

    from PIL import Image as PILImage
    from PIL import ImageDraw

    fill, ink = _AGENT_STYLE[theme]
    big_side = side * _SUPERSAMPLE
    big = PILImage.new("RGBA", (big_side, big_side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    draw.ellipse((0, 0, big_side - 1, big_side - 1), fill=fill)
    c, r, w = big_side / 2, big_side * 0.32, big_side * 0.09
    draw.polygon(
        [
            (c, c - r),
            (c + w, c - w),
            (c + r, c),
            (c + w, c + w),
            (c, c + r),
            (c - w, c + w),
            (c - r, c),
            (c - w, c - w),
        ],
        fill=ink,
    )
    return big.resize((side, side), PILImage.Resampling.LANCZOS)


def _punch(canvas: Image, box: tuple[int, int, int], gap: int) -> None:
    """Clear a ring around an overlay so it never fuses with the picture."""

    from PIL import Image as PILImage
    from PIL import ImageChops

    x, y, side = box
    hole = _circle_mask(side + 2 * gap)
    mask = PILImage.new("L", canvas.size, 0)
    mask.paste(hole, (x - gap, y - gap))
    canvas.putalpha(ImageChops.subtract(canvas.getchannel("A"), mask))


def compose(
    variant: Variant,
    *,
    origin: str,
    account_id: str,
    photo: bytes | None,
) -> bytes:
    """Render one notification PNG. Pure and synchronous; call off-loop."""

    from PIL import Image as PILImage
    from PIL import ImageChops

    # Always draw at 256 and downsample, so 64 px is a faithful reduction of
    # the same geometry rather than a second layout.
    side = 256
    if photo is not None:
        face = _cover(decode_source(photo), side)
    else:
        face = _placeholder(
            side, palette_colour(origin, account_id, variant.theme), variant.initials
        )
    canvas = PILImage.new("RGBA", (side, side), (0, 0, 0, 0))
    face.putalpha(ImageChops.multiply(face.getchannel("A"), _circle_mask(side)))
    canvas.alpha_composite(face)

    gap = max(1, int(side * GAP_RATIO))
    badge_side = int(side * BADGE_RATIO)
    badge_at = side - badge_side
    _punch(canvas, (badge_at, badge_at, badge_side), gap)
    canvas.alpha_composite(_badge(badge_side, variant.theme), (badge_at, badge_at))
    if variant.kind == "person_agent":
        marker_side = int(side * AGENT_MARKER_RATIO)
        marker_x = side - marker_side
        _punch(canvas, (marker_x, 0, marker_side), gap)
        canvas.alpha_composite(_agent_marker(marker_side, variant.theme), (marker_x, 0))

    if variant.size != side:
        canvas = canvas.resize((variant.size, variant.size), PILImage.Resampling.LANCZOS)
    out = io.BytesIO()
    # Fresh encode: no source metadata, ICC or text chunks are carried over.
    canvas.save(out, format="PNG", optimize=True)
    data = out.getvalue()
    if len(data) > MAX_OUTPUT_BYTES:
        raise AvatarImageRejected("composite too large")
    return data


async def render(
    variant: Variant,
    *,
    origin: str,
    account_id: str,
    photo: bytes | None,
    check_owner: Callable[[], None] = lambda: None,
) -> bytes:
    """Concurrency-bounded :func:`compose` off the event loop. Stores nothing.

    ``check_owner`` runs before and after the decode inside the worker, so a
    queued render whose login ended in the meantime refuses instead of
    returning pixels. With no cache, a caller that is cancelled mid-render
    simply drops the bytes: there is no store for a late result to land in.
    """

    if not _PENDING.acquire(blocking=False):
        raise AvatarImageRejected("renderer busy")

    def work() -> bytes:
        try:
            with _RENDER_SLOTS:
                check_owner()
                data = compose(variant, origin=origin, account_id=account_id, photo=photo)
                check_owner()
                return data
        finally:
            _PENDING.release()

    # Shield ensures a cancelled HTTP request cannot cancel a queued worker
    # before it releases its pending slot. The worker's result is local to
    # this call, so a cancelled caller leaves nothing behind.
    return await asyncio.shield(asyncio.to_thread(work))
