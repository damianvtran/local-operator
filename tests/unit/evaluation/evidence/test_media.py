"""Structural evidence media validation tests."""

from __future__ import annotations

import struct
import zlib

import pytest

from local_operator.evaluation.evidence.media import (
    MediaValidationError,
    validate_media,
)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(4, "big")
        + kind
        + payload
        + (zlib.crc32(kind + payload) & 0xFFFFFFFF).to_bytes(4, "big")
    )


def png() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + _png_chunk(b"IEND", b"")
    )


def jpeg() -> bytes:
    dqt = b"\x00" + b"\x01" * 64
    dht_dc = b"\x00" + b"\x01" + b"\x00" * 15 + b"\x00"
    dht_ac = b"\x10" + b"\x01" + b"\x00" * 15 + b"\x00"
    sof = b"\x08\x00\x01\x00\x01\x01\x01\x11\x00"
    scan = b"\x01\x01\x00\x00\x3f\x00"
    segments = (
        b"\xff\xdb"
        + (len(dqt) + 2).to_bytes(2, "big")
        + dqt
        + b"\xff\xc4"
        + (len(dht_dc) + 2).to_bytes(2, "big")
        + dht_dc
        + b"\xff\xc4"
        + (len(dht_ac) + 2).to_bytes(2, "big")
        + dht_ac
        + b"\xff\xc0"
        + (len(sof) + 2).to_bytes(2, "big")
        + sof
        + b"\xff\xda"
        + (len(scan) + 2).to_bytes(2, "big")
        + scan
    )
    return b"\xff\xd8" + segments + b"\x00\xff\xd9"


def gif() -> bytes:
    return (
        b"GIF89a\x01\x00\x01\x00\x00\x00\x00"
        b"\x2c\x00\x00\x00\x00\x01\x00\x01\x00\x00"
        b"\x02\x02\x4c\x01\x00\x3b"
    )


def webp() -> bytes:
    payload = b"\x20\x00\x00\x9d\x01\x2a\x01\x00\x01\x00\x00"
    chunk = b"VP8 " + len(payload).to_bytes(4, "little") + payload + b"\x00"
    return b"RIFF" + (len(chunk) + 4).to_bytes(4, "little") + b"WEBP" + chunk


@pytest.mark.parametrize(
    ("media_type", "fixture"),
    [
        ("image/png", png),
        ("image/jpeg", jpeg),
        ("image/gif", gif),
        ("image/webp", webp),
    ],
)
def test_minimal_structural_image_fixtures_are_valid(media_type: str, fixture: object) -> None:
    validate_media(fixture(), media_type)  # type: ignore[operator]


@pytest.mark.parametrize(
    ("media_type", "payload"),
    [
        ("image/png", b"\x89PNG\r\n\x1a\n"),
        ("image/jpeg", b"\xff\xd8junk\xff\xd9"),
        ("image/gif", b"GIF89a"),
        ("image/webp", b"RIFF\x04\x00\x00\x00WEBP"),
    ],
)
def test_magic_only_images_are_rejected(media_type: str, payload: bytes) -> None:
    with pytest.raises(MediaValidationError):
        validate_media(payload, media_type)


def test_png_rejects_crc_dimension_truncation_and_trailing_bytes() -> None:
    valid = png()
    corrupt = bytearray(valid)
    corrupt[29] ^= 1
    for payload in (
        bytes(corrupt),
        valid[:-1],
        valid + b"trailing",
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1_000_001, 1, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", b"x")
        + _png_chunk(b"IEND", b""),
    ):
        with pytest.raises(MediaValidationError):
            validate_media(payload, "image/png")


def test_reviewer_header_only_fixtures_are_rejected() -> None:
    empty_scan = jpeg()[:-3] + b"\xff\xd9"
    invalid_gif = gif().replace(b"\x02\x02\x4c\x01", b"\x00\x00")
    vp8x = b"VP8X" + (10).to_bytes(4, "little") + b"\x00" * 10
    header_only_webp = b"RIFF" + (len(vp8x) + 4).to_bytes(4, "little") + b"WEBP" + vp8x
    for media_type, payload in (
        ("image/jpeg", empty_scan),
        ("image/gif", invalid_gif),
        ("image/webp", header_only_webp),
    ):
        with pytest.raises(MediaValidationError):
            validate_media(payload, media_type)


@pytest.mark.parametrize(
    ("media_type", "fixture"),
    [
        ("image/jpeg", jpeg),
        ("image/gif", gif),
        ("image/webp", webp),
    ],
)
def test_other_images_reject_truncation_and_trailing(media_type: str, fixture: object) -> None:
    valid = fixture()  # type: ignore[operator]
    for payload in (valid[:-1], valid + b"trailing"):
        with pytest.raises(MediaValidationError):
            validate_media(payload, media_type)
