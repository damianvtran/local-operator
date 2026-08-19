"""Minimal 8-bit grayscale PNG encoder (stdlib only).

Why hand-rolled
---------------
:mod:`local_operator.compaction.snapcompact` renders conversation history onto
plain white-on-black text bitmaps. That is the *only* raster work the agent
does, and it needs exactly one image operation: take a ``width * height``
buffer of 8-bit gray samples and emit a PNG byte string.

Pulling a full imaging library in for that costs ~28 MB of installed weight
plus two compiled wheels, which is the single worst part of the install story
on Windows (no wheel for a new CPython release means an on-the-fly C build).
PNG's baseline encoding path is small, fully specified, and needs nothing
beyond :mod:`zlib` and :mod:`struct` from the standard library, so we own it.

Scope and limits (deliberate)
-----------------------------
* Color type 0 (grayscale), bit depth 8, no interlacing, no ancillary chunks.
* Filter type 0 ("None") on every scanline. Adaptive per-row filter selection
  is what a general-purpose encoder does, but the heuristic costs five passes
  over every byte in pure Python — seconds per frame at our page sizes. Our
  input is a sparse two-tone bitmap (mostly 0x00 with 0xFF ink), which DEFLATE
  already collapses by ~2-3 orders of magnitude with no prefiltering, so the
  filter pass would buy little and cost a lot.
* Encode only. Nothing in the agent reads PNGs back; the model does.

The output is a conformant PNG that any decoder accepts, and it is
byte-for-byte deterministic for a given buffer (a hard requirement: snapcompact
frames must be reproducible so identical history yields identical archives).
"""

from __future__ import annotations

import struct
import zlib

__all__ = ["encode_grayscale_png"]

#: The fixed 8-byte PNG file signature (spec section 5.2). The high bit and the
#: CRLF/LF pair let readers detect corrupting transfers.
_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: DEFLATE level. 6, not 9, and the difference is the user's wait: on a full
#: 1932px frame level 9 spends ~390 ms against level 6's ~37 ms (measured on
#: the raw scanline stream of a typical page) for ~11% fewer bytes. The bytes
#: are the wrong thing to optimize — providers bill images by their pixel
#: dimensions, not their file size, so the smaller file saves request payload
#: only, while the encode time sits directly on the compaction pass the user
#: is watching. The one byte ceiling that matters (FRAME_DATA_BYTES_BUDGET,
#: 3 MB per request) is nowhere near binding at either level: a full archive
#: replay is ~6 frames × ~130 KB base64.
_COMPRESS_LEVEL = 6


def _chunk(kind: bytes, payload: bytes) -> bytes:
    """Frame one PNG chunk: length, type, payload, CRC-32 over type+payload."""
    return b"".join(
        (
            struct.pack(">I", len(payload)),
            kind,
            payload,
            struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF),
        )
    )


def encode_grayscale_png(width: int, height: int, samples: bytes) -> bytes:
    """Encode ``samples`` as an 8-bit grayscale PNG.

    Args:
        width: Image width in pixels; must be positive.
        height: Image height in pixels; must be positive.
        samples: Exactly ``width * height`` gray bytes in row-major order,
            top row first, 0 = black and 255 = white.

    Returns:
        The complete PNG file as bytes.

    Raises:
        ValueError: If the dimensions are not positive or ``samples`` is not
            exactly ``width * height`` bytes long. This is a programming error
            rather than bad user input, so it fails loudly instead of padding.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG dimensions must be positive, got {width}x{height}")
    if len(samples) != width * height:
        raise ValueError(
            f"Expected {width * height} gray samples for a {width}x{height} image, "
            f"got {len(samples)}"
        )

    # IHDR: width, height, bit depth 8, color type 0 (grayscale), compression
    # method 0 (the only one defined), filter method 0, interlace 0 (none).
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)

    # Each scanline is prefixed with its filter type byte; 0 means the raw
    # bytes follow unmodified. Building the whole stream with one join keeps
    # this a single pass over the buffer.
    raw = bytearray()
    for start in range(0, len(samples), width):
        raw.append(0)
        raw += samples[start : start + width]

    return b"".join(
        (
            _SIGNATURE,
            _chunk(b"IHDR", header),
            _chunk(b"IDAT", zlib.compress(bytes(raw), _COMPRESS_LEVEL)),
            _chunk(b"IEND", b""),
        )
    )
