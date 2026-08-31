"""Bounded stdlib-only validation for evidence artifact media.

Magic bytes are not structural evidence. These parsers walk every declared
container length, reject trailing bytes, and bound dimensions before an adapter
or downstream report is allowed to treat attacker-controlled bytes as an image.
They intentionally do not decode pixels.
"""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any

MAX_IMAGE_DIMENSION = 1_000_000
MAX_IMAGE_PIXELS = 1_000_000_000


class MediaValidationError(ValueError):
    pass


def _dimensions(width: int, height: int) -> None:
    if (
        width <= 0
        or height <= 0
        or width > MAX_IMAGE_DIMENSION
        or height > MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise MediaValidationError("invalid image dimensions")


def validate_media(data: bytes, media_type: str) -> Any:
    if media_type == "application/json":
        try:
            decoded = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as error:
            raise MediaValidationError("invalid canonical JSON") from error
        encoded = json.dumps(
            decoded,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if encoded != data:
            raise MediaValidationError("invalid canonical JSON")
        return decoded
    if media_type == "text/plain":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise MediaValidationError("invalid UTF-8 text") from error
    if media_type == "image/png":
        _png(data)
    elif media_type == "image/jpeg":
        _jpeg(data)
    elif media_type == "image/gif":
        _gif(data)
    elif media_type == "image/webp":
        _webp(data)
    elif media_type != "application/octet-stream":
        raise MediaValidationError("unsupported media type")
    return data


def _png(data: bytes) -> None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise MediaValidationError("invalid PNG")
    offset = 8
    chunks = 0
    saw_idat = False
    saw_iend = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise MediaValidationError("truncated PNG chunk")
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise MediaValidationError("truncated PNG chunk")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(data[offset + 8 + length : end], "big")
        if zlib.crc32(kind + payload) & 0xFFFFFFFF != expected_crc:
            raise MediaValidationError("invalid PNG CRC")
        if chunks == 0:
            if kind != b"IHDR" or length != 13:
                raise MediaValidationError("PNG IHDR must be first")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            _dimensions(width, height)
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                depth not in valid_depths.get(color, set())
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                raise MediaValidationError("invalid PNG IHDR")
        elif kind == b"IHDR":
            raise MediaValidationError("duplicate PNG IHDR")
        if kind == b"IDAT":
            saw_idat = True
        if kind == b"IEND":
            if length != 0 or not saw_idat or end != len(data):
                raise MediaValidationError("invalid PNG IEND")
            saw_iend = True
        chunks += 1
        offset = end
        if saw_iend:
            break
    if not saw_iend:
        raise MediaValidationError("missing PNG IEND")


def _jpeg(data: bytes) -> None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise MediaValidationError("invalid JPEG")
    offset = 2
    saw_sof = False
    in_scan = False
    while offset < len(data):
        if data[offset] != 0xFF:
            if not in_scan:
                raise MediaValidationError("invalid JPEG marker")
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise MediaValidationError("truncated JPEG marker")
        marker = data[offset]
        offset += 1
        if in_scan and marker == 0x00:
            continue
        if in_scan and 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:
            if not saw_sof or offset != len(data):
                raise MediaValidationError("invalid JPEG EOI")
            return
        in_scan = False
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            raise MediaValidationError("truncated JPEG segment")
        length = int.from_bytes(data[offset : offset + 2], "big")
        if length < 2 or offset + length > len(data):
            raise MediaValidationError("invalid JPEG segment length")
        payload = data[offset + 2 : offset + length]
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if len(payload) < 6:
                raise MediaValidationError("truncated JPEG SOF")
            _dimensions(
                int.from_bytes(payload[3:5], "big"),
                int.from_bytes(payload[1:3], "big"),
            )
            saw_sof = True
        if marker == 0xDA:
            if not saw_sof:
                raise MediaValidationError("JPEG scan precedes SOF")
            in_scan = True
        offset += length
    raise MediaValidationError("missing JPEG EOI")


def _subblocks(data: bytes, offset: int) -> int:
    while True:
        if offset >= len(data):
            raise MediaValidationError("truncated GIF subblocks")
        length = data[offset]
        offset += 1
        if length == 0:
            return offset
        if offset + length > len(data):
            raise MediaValidationError("truncated GIF subblock")
        offset += length


def _gif(data: bytes) -> None:
    if len(data) < 14 or data[:6] not in (b"GIF87a", b"GIF89a"):
        raise MediaValidationError("invalid GIF")
    width, height = struct.unpack("<HH", data[6:10])
    _dimensions(width, height)
    packed = data[10]
    offset = 13
    if packed & 0x80:
        offset += 3 * (2 ** ((packed & 0x07) + 1))
    if offset > len(data):
        raise MediaValidationError("truncated GIF color table")
    saw_image = False
    while offset < len(data):
        introducer = data[offset]
        offset += 1
        if introducer == 0x3B:
            if not saw_image or offset != len(data):
                raise MediaValidationError("invalid GIF trailer")
            return
        if introducer == 0x21:
            if offset >= len(data):
                raise MediaValidationError("truncated GIF extension")
            offset += 1
            offset = _subblocks(data, offset)
        elif introducer == 0x2C:
            if offset + 9 > len(data):
                raise MediaValidationError("truncated GIF image descriptor")
            image_width, image_height = struct.unpack("<HH", data[offset + 4 : offset + 8])
            _dimensions(image_width, image_height)
            image_packed = data[offset + 8]
            offset += 9
            if image_packed & 0x80:
                offset += 3 * (2 ** ((image_packed & 0x07) + 1))
            if offset >= len(data):
                raise MediaValidationError("truncated GIF image")
            offset += 1  # LZW minimum code size
            offset = _subblocks(data, offset)
            saw_image = True
        else:
            raise MediaValidationError("invalid GIF block")
    raise MediaValidationError("missing GIF trailer")


def _webp(data: bytes) -> None:
    if len(data) < 20 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise MediaValidationError("invalid WebP")
    if int.from_bytes(data[4:8], "little") + 8 != len(data):
        raise MediaValidationError("invalid WebP RIFF size")
    offset = 12
    saw_image = False
    while offset < len(data):
        if offset + 8 > len(data):
            raise MediaValidationError("truncated WebP chunk")
        kind = data[offset : offset + 4]
        length = int.from_bytes(data[offset + 4 : offset + 8], "little")
        start = offset + 8
        end = start + length
        padded = end + (length & 1)
        if padded > len(data):
            raise MediaValidationError("truncated WebP chunk")
        payload = data[start:end]
        if kind == b"VP8X":
            if len(payload) != 10:
                raise MediaValidationError("invalid WebP VP8X")
            width = 1 + int.from_bytes(payload[4:7], "little")
            height = 1 + int.from_bytes(payload[7:10], "little")
            _dimensions(width, height)
            saw_image = True
        elif kind == b"VP8L":
            if len(payload) < 5 or payload[0] != 0x2F:
                raise MediaValidationError("invalid WebP VP8L")
            bits = int.from_bytes(payload[1:5], "little")
            _dimensions((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
            saw_image = True
        elif kind == b"VP8 ":
            if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
                raise MediaValidationError("invalid WebP VP8")
            _dimensions(
                int.from_bytes(payload[6:8], "little") & 0x3FFF,
                int.from_bytes(payload[8:10], "little") & 0x3FFF,
            )
            saw_image = True
        offset = padded
    if offset != len(data) or not saw_image:
        raise MediaValidationError("missing WebP image chunk")
