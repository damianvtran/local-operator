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
    components: set[int] = set()
    quant_tables: set[int] = set()
    huffman_dc: set[int] = set()
    huffman_ac: set[int] = set()
    saw_scan_data = False
    in_scan = False
    while offset < len(data):
        if data[offset] != 0xFF:
            if not in_scan:
                raise MediaValidationError("invalid JPEG marker")
            saw_scan_data = True
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
            if not components or not saw_scan_data or offset != len(data):
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
            if len(payload) < 6 or payload[0] not in (8, 12):
                raise MediaValidationError("truncated JPEG SOF")
            count = payload[5]
            if count == 0 or len(payload) != 6 + 3 * count:
                raise MediaValidationError("invalid JPEG components")
            _dimensions(
                int.from_bytes(payload[3:5], "big"),
                int.from_bytes(payload[1:3], "big"),
            )
            components = {payload[6 + index * 3] for index in range(count)}
            if len(components) != count:
                raise MediaValidationError("duplicate JPEG component")
            referenced_quant = {payload[8 + index * 3] for index in range(count)}
            if referenced_quant - quant_tables:
                raise MediaValidationError("undefined JPEG quantization table")
        elif marker == 0xDB:
            cursor = 0
            while cursor < len(payload):
                precision_table = payload[cursor]
                cursor += 1
                size = 64 * (2 if precision_table >> 4 else 1)
                if precision_table >> 4 not in (0, 1) or cursor + size > len(payload):
                    raise MediaValidationError("invalid JPEG DQT")
                quant_tables.add(precision_table & 0x0F)
                cursor += size
        elif marker == 0xC4:
            cursor = 0
            while cursor < len(payload):
                table = payload[cursor]
                cursor += 1
                if cursor + 16 > len(payload):
                    raise MediaValidationError("invalid JPEG DHT")
                symbols = sum(payload[cursor : cursor + 16])
                cursor += 16
                if cursor + symbols > len(payload):
                    raise MediaValidationError("invalid JPEG DHT")
                (huffman_ac if table >> 4 else huffman_dc).add(table & 0x0F)
                cursor += symbols
        if marker == 0xDA:
            if not components or len(payload) < 6:
                raise MediaValidationError("JPEG scan precedes SOF")
            count = payload[0]
            if count == 0 or len(payload) != 1 + 2 * count + 3:
                raise MediaValidationError("invalid JPEG SOS")
            for index in range(count):
                component = payload[1 + 2 * index]
                tables = payload[2 + 2 * index]
                if (
                    component not in components
                    or tables >> 4 not in huffman_dc
                    or tables & 0x0F not in huffman_ac
                ):
                    raise MediaValidationError("undefined JPEG scan table")
            saw_scan_data = False
            in_scan = True
        offset += length
    raise MediaValidationError("missing JPEG EOI")


def _subblocks(data: bytes, offset: int) -> tuple[int, bytes]:
    collected = bytearray()
    while True:
        if offset >= len(data):
            raise MediaValidationError("truncated GIF subblocks")
        length = data[offset]
        offset += 1
        if length == 0:
            return offset, bytes(collected)
        if offset + length > len(data):
            raise MediaValidationError("truncated GIF subblock")
        collected.extend(data[offset : offset + length])
        offset += length


def _gif_lzw(payload: bytes, minimum_code_size: int) -> None:
    if minimum_code_size < 2 or minimum_code_size > 8 or not payload:
        raise MediaValidationError("invalid GIF LZW stream")
    clear = 1 << minimum_code_size
    end = clear + 1
    code_size = minimum_code_size + 1
    next_code = end + 1
    bit_offset = 0
    saw_clear = False
    while bit_offset + code_size <= len(payload) * 8:
        byte_offset = bit_offset // 8
        shift = bit_offset % 8
        value = int.from_bytes(payload[byte_offset : byte_offset + 3], "little")
        code = (value >> shift) & ((1 << code_size) - 1)
        bit_offset += code_size
        if code == clear:
            saw_clear = True
            code_size = minimum_code_size + 1
            next_code = end + 1
            continue
        if code == end:
            if not saw_clear:
                raise MediaValidationError("GIF LZW end precedes clear")
            return
        if not saw_clear or code > next_code:
            raise MediaValidationError("invalid GIF LZW code")
        next_code += 1
        if next_code == (1 << code_size) and code_size < 12:
            code_size += 1
    raise MediaValidationError("missing GIF LZW end code")


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
            offset, _extension = _subblocks(data, offset)
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
            minimum_code_size = data[offset]
            offset += 1
            offset, compressed = _subblocks(data, offset)
            _gif_lzw(compressed, minimum_code_size)
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
    saw_extended = False
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
            if len(payload) != 10 or saw_extended or saw_image:
                raise MediaValidationError("invalid WebP VP8X")
            width = 1 + int.from_bytes(payload[4:7], "little")
            height = 1 + int.from_bytes(payload[7:10], "little")
            _dimensions(width, height)
            saw_extended = True
        elif kind == b"VP8L":
            if len(payload) <= 5 or payload[0] != 0x2F:
                raise MediaValidationError("invalid WebP VP8L")
            bits = int.from_bytes(payload[1:5], "little")
            _dimensions((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)
            saw_image = True
        elif kind == b"VP8 ":
            if len(payload) <= 10 or payload[3:6] != b"\x9d\x01\x2a":
                raise MediaValidationError("invalid WebP VP8")
            frame_tag = int.from_bytes(payload[:3], "little")
            partition_length = frame_tag >> 5
            if frame_tag & 1 or partition_length == 0 or partition_length + 10 > len(payload):
                raise MediaValidationError("invalid WebP VP8 partition")
            _dimensions(
                int.from_bytes(payload[6:8], "little") & 0x3FFF,
                int.from_bytes(payload[8:10], "little") & 0x3FFF,
            )
            saw_image = True
        offset = padded
    if offset != len(data) or not saw_image:
        raise MediaValidationError("missing WebP image chunk")
