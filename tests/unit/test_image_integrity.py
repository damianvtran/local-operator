"""Structurally damaged PNGs are normalized before a provider sees them.

Measured live against DeepSeek on 2026-11-11 (1-token requests, the matrix is
in the PR): the provider refuses a PNG with ``unsupported image`` when its
CHUNK STRUCTURE is broken — no ``IEND``, a chunk whose length runs past the
end of the file, or a chunk whose CRC does not match — and accepts every
colour type (L, LA, P, 1-bit, I;16, RGB, RGBA), every size from 1x1 to
2100px and aspect ratios to 1:100, and trailing bytes AFTER ``IEND``. Pillow,
meanwhile, decodes the damaged file without complaint (it never checks the
IDAT CRC and stops reading once it has enough rows), so the ``read`` tool's
verbatim rung forwarded the damaged bytes untouched and the session died on
the next request. The observed file was a 310x680 grayscale crop the model had
reassembled from base64: its IDAT length field was short, the bytes past it
read as a chunk claiming 1.5 GB, and there was no ``IEND``.

These tests pin both seams: the creation-time bound
(:func:`bound_image_for_model`) and the render-time history repair
(:func:`rebound_oversize_image`), each with a stable output digest so a
repaired block never churns the prompt-cache prefix.
"""

from __future__ import annotations

import base64
import hashlib
import io
import struct
import zlib

import pytest
from PIL import Image

from local_operator.imaging import (
    _REBOUND_CACHE,
    bound_image_for_model,
    rebound_oversize_image,
)
from local_operator.media import png_structure_fault, sniff_image

IEND = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))


def _clean_png(mode: str = "L", size: tuple[int, int] = (310, 68)) -> bytes:
    image = Image.new(mode, size, 255 if mode == "L" else (255, 255, 255))
    # A few dark rows so the decoded pixels are not trivially uniform and a
    # re-encode that lost them would show in the digest.
    for y in range(0, size[1], 7):
        for x in range(size[0]):
            image.putpixel((x, y), 0 if mode == "L" else (0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    out, offset = [], 8
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        out.append((data[offset + 4 : offset + 8], data[offset : offset + 12 + length]))
        offset += 12 + length
    return out


def _no_iend(data: bytes) -> bytes:
    assert data.endswith(IEND)
    return data[: -len(IEND)]


def _bad_idat_crc(data: bytes) -> bytes:
    rebuilt = data[:8]
    for kind, chunk in _chunks(data):
        if kind == b"IDAT":
            chunk = chunk[:-4] + bytes(b ^ 0xFF for b in chunk[-4:])
        rebuilt += chunk
    return rebuilt


#: The first bytes after the IDAT in the file the episode actually read: read
#: as a chunk header they claim a 1,527,998,571-byte chunk of type
#: ``\x02mM\xa0``. Pillow never gets that far (the IDAT before it already
#: inflates to more rows than the image has), which is why it decoded the file
#: and the provider did not.
OBSERVED_JUNK = bytes.fromhex("5b13686b026d4da0ad09b43581b626d0d604da9a405b13686b026d4da0")


def _observed_shape(data: bytes) -> bytes:
    """The shape of the file the episode actually read: an intact IHDR and
    IDAT followed by junk where the next chunk header should be, no IEND."""
    return _no_iend(data) + OBSERVED_JUNK


@pytest.fixture(autouse=True)
def _fresh_rebound_cache():
    _REBOUND_CACHE.clear()
    yield
    _REBOUND_CACHE.clear()


def test_clean_pngs_have_no_structure_fault():
    for mode in ("L", "LA", "RGB", "RGBA", "P", "1"):
        buffer = io.BytesIO()
        Image.new(mode, (9, 9)).save(buffer, format="PNG")
        assert png_structure_fault(buffer.getvalue(), verify_crc=True) is None, mode


def test_trailing_bytes_after_iend_are_not_a_fault():
    # Measured ACCEPTED by DeepSeek: data after IEND is ignored, so treating it
    # as damage would rewrite blocks the provider takes as they are.
    assert png_structure_fault(_clean_png() + b"\x00junk" * 8, verify_crc=True) is None


@pytest.mark.parametrize(
    ("damage", "needs_crc", "reason"),
    [
        (_no_iend, False, "IEND"),
        (_observed_shape, False, "invalid chunk type"),
        (
            lambda d: _no_iend(d) + struct.pack(">I", 1 << 30) + b"IDAT" + b"\0" * 16,
            False,
            "past the end",
        ),
        (_bad_idat_crc, True, "CRC"),
    ],
)
def test_structure_faults_are_named(damage, needs_crc, reason):
    damaged = damage(_clean_png())
    fault = png_structure_fault(damaged, verify_crc=True)
    assert fault is not None and reason in fault
    if needs_crc:
        # The render-time walk skips CRCs (it runs on every image every
        # render); only the creation-time bound pays for them.
        assert png_structure_fault(damaged, verify_crc=False) is None


@pytest.mark.parametrize("damage", [_no_iend, _observed_shape, _bad_idat_crc])
def test_bound_image_re_encodes_a_damaged_png_deterministically(damage):
    damaged = damage(_clean_png())
    info = sniff_image(damaged)
    assert info is not None and info.mime_type == "image/png"

    payload, mime, summary = bound_image_for_model(damaged, info)

    assert mime == "image/png"
    assert payload != damaged
    assert png_structure_fault(payload, verify_crc=True) is None
    # The model is told the file was damaged, so it does not trust pixels a
    # lenient decoder may have half-invented.
    assert "damaged PNG" in summary
    # Grayscale stays grayscale: widening L to RGB is a token/byte cost with
    # no acceptance benefit (measured: L is accepted).
    assert Image.open(io.BytesIO(payload)).mode == "L"
    again, _, _ = bound_image_for_model(damaged, info)
    assert hashlib.sha256(again).hexdigest() == hashlib.sha256(payload).hexdigest()


def test_bound_image_keeps_a_clean_png_verbatim():
    clean = _clean_png()
    info = sniff_image(clean)
    assert info is not None
    payload, mime, summary = bound_image_for_model(clean, info)
    assert payload == clean and mime == "image/png"
    assert "damaged" not in summary


@pytest.mark.parametrize("damage", [_no_iend, _observed_shape])
def test_history_repair_fixes_a_damaged_png_already_in_the_transcript(damage):
    damaged = damage(_clean_png())
    encoded = base64.b64encode(damaged).decode("ascii")

    repaired = rebound_oversize_image(encoded)

    assert repaired is not None
    data, mime = repaired
    assert mime == "image/png"
    raw = base64.b64decode(data)
    assert png_structure_fault(raw, verify_crc=True) is None
    # Stable across renders: the same block repairs to the same bytes, so the
    # prompt-cache prefix changes once (old build -> this one), never per turn.
    _REBOUND_CACHE.clear()
    assert rebound_oversize_image(encoded) == repaired


def test_history_repair_leaves_clean_in_bounds_png_alone():
    assert rebound_oversize_image(base64.b64encode(_clean_png()).decode("ascii")) is None
