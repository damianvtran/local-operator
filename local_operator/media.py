"""Image identification from bytes, by header, with no decode.

Pillow IS available at runtime — ``pillow-heif`` is a BASE dependency
(``pyproject.toml`` ``project.dependencies``; the ``[images]`` extra is an
alias kept so hosts can install by name) and it requires ``pillow>=11.1.0``
unconditionally. So this module is not a workaround for a missing library, and
callers that need to RESIZE should decode with Pillow.

What this is for is the question asked far more often than "resize this":
*is this an image, and how big is it*. Answering from the header costs one
short read regardless of file size, cannot be made expensive by a hostile or
corrupt body, and does not pay Pillow's import on a path where nothing is
going to be decoded — the composer asks it on every paste and ``read`` asks it
on every file.

Type is decided by CONTENT, never by extension. A `.png` that is actually HTML
must not be handed to a provider as an image, and a screenshot saved without a
suffix must still work — both arrive routinely from a clipboard.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

#: What providers accept directly. Deliberately not a superset: a format we can
#: name but they cannot read is worse than an honest refusal, because the
#: failure surfaces as a provider 400 halfway through a turn.
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

#: Recognised but NOT directly sendable — a caller must transcode first (see
#: ``helpers.convert_heic_to_png_file``). Named here anyway because "I do not
#: know what this is" and "this is an iPhone screenshot you must convert" are
#: different answers, and only the second one tells the caller what to do.
TRANSCODE_IMAGE_MIME_TYPES = frozenset({"image/heic", "image/heif"})

#: Enough bytes for every header this module parses, including a JPEG whose
#: first frame marker sits behind a large EXIF/ICC segment.
_SNIFF_BYTES = 65_536

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_GIF_MAGICS = (b"GIF87a", b"GIF89a")

#: ISO-BMFF brands that mean HEIF-family stills. ``mif1``/``msf1`` are the
#: generic image/sequence brands Apple also emits, so they are included.
_HEIF_BRANDS = frozenset(
    {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx", b"hevm", b"hevs", b"mif1", b"msf1"}
)

#: JPEG frame markers that carry dimensions. SOF4/SOF8/SOF12 (0xC4/0xC8/0xCC)
#: are excluded on purpose: they are DHT/JPG/DAC tables, not frame headers, and
#: reading them as a frame yields nonsense dimensions.
_JPEG_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)


@dataclass(frozen=True)
class ImageInfo:
    """What a header says about an image."""

    mime_type: str
    #: ``None`` when the format was recognised but its dimensions were not
    #: readable from the header. HEIF keeps them in an ``ispe`` box nested
    #: inside ``meta``, which is a container walk rather than a field read, so
    #: it is left to whoever actually decodes the file. Callers must degrade —
    #: the composer drops to a bare ``[Image #N]`` marker — rather than
    #: printing a zero they would be inventing.
    width: int | None = None
    height: int | None = None

    @property
    def sendable(self) -> bool:
        """Can this go to a provider as-is, without transcoding first?"""
        return self.mime_type in SUPPORTED_IMAGE_MIME_TYPES

    @property
    def dimensions(self) -> str:
        """``WxH``, or ``""`` when the header did not carry them."""
        if self.width is None or self.height is None:
            return ""
        return f"{self.width}x{self.height}"


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    # IHDR is mandated to be the first chunk, so width/height are at a fixed
    # offset: 8 magic + 4 length + 4 type.
    if len(data) < 24:
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _gif_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 10:
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """Walk the segment chain to the first start-of-frame.

    JPEG has no fixed dimension offset: the frame header sits behind however
    many APPn/DQT/DRI segments the encoder emitted, and a phone photo's EXIF
    thumbnail alone can be tens of kilobytes.
    """
    index = 2  # past SOI
    end = len(data)
    while index + 3 < end:
        if data[index] != 0xFF:
            # Fill bytes are legal between segments; anything else means the
            # chain is broken and any dimensions we invented would be fiction.
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2  # standalone markers carry no length
            continue
        if index + 4 > end:
            return None
        length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        if marker in _JPEG_SOF_MARKERS:
            # SOF payload: precision(1) height(2) width(2)
            if index + 9 > end:
                return None
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        if length < 2:
            return None
        index += 2 + length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    """WebP carries three different frame encodings, each with its own header.

    All three are handled because a clipboard hands over whichever the source
    app produced: lossy (VP8), lossless (VP8L), and extended/animated (VP8X).
    """
    if len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8 ":
        # Lossy: 3-byte frame tag, 3-byte sync code, then 14-bit dimensions.
        if data[23:26] != b"\x9d\x01\x2a":
            return None
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        # Lossless: 1-byte signature then 14+14 bits packed little-endian.
        if data[20] != 0x2F:
            return None
        bits = struct.unpack("<I", data[21:25])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X":
        # Extended: canvas size as two 24-bit little-endian values, minus one.
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    return None


def sniff_image(data: bytes) -> ImageInfo | None:
    """Identify ``data`` as an image, or return ``None``.

    ``None`` means "do not treat this as an image", and covers both "not an
    image at all" and "a format whose header we could not parse". Those are
    deliberately the same answer: shipping an image block we could not verify
    is how a provider 400 gets into the conversation history — where, until the
    session layer learned to recover from it, it stayed forever.

    A recognised format with UNREADABLE dimensions is a different answer again,
    and returns an :class:`ImageInfo` with ``width``/``height`` of ``None``:
    the caller can still send it, it just cannot label it.
    """
    if data.startswith(_PNG_MAGIC):
        size = _png_dimensions(data)
        return ImageInfo("image/png", *size) if size else None
    if data.startswith(b"\xff\xd8\xff"):
        size = _jpeg_dimensions(data[:_SNIFF_BYTES])
        return ImageInfo("image/jpeg", *size) if size else None
    if data.startswith(_GIF_MAGICS):
        size = _gif_dimensions(data)
        return ImageInfo("image/gif", *size) if size else None
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        size = _webp_dimensions(data)
        return ImageInfo("image/webp", *size) if size else None
    if data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS:
        # Dimensions live in a `meta`-nested `ispe` box; see ImageInfo.width.
        # `heif`/`mif1` also brand HEIF SEQUENCES, which Pillow will reject on
        # transcode — reported as an image so the caller can say "convert this"
        # rather than the useless "unrecognised file".
        return ImageInfo("image/heic")
    return None


def sniff_image_file(path: str) -> ImageInfo | None:
    """:func:`sniff_image` against a file, reading only the header.

    Bounded at :data:`_SNIFF_BYTES` so pointing this at a multi-gigabyte file
    costs one short read rather than the file.

    ``None`` for anything unreadable, which has to include the paths that do
    not raise ``OSError``: a NUL byte in the name raises ``ValueError:
    embedded null byte``, which is not an ``OSError`` subclass. Callers reach
    this from a PASTE, where every other malformed input degrades to an
    ordinary text paste — so the one shape that escaped instead took down the
    keystroke with Textual's error screen (review round 17).
    """
    try:
        with open(path, "rb") as handle:
            return sniff_image(handle.read(_SNIFF_BYTES))
    except (OSError, ValueError):
        return None
