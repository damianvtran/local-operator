"""Minimal ``application/vnd.amazon.eventstream`` ENCODER for tests and the fake server.

botocore ships only a decoder (``botocore.eventstream.EventStreamBuffer``), so
the tests and ``scripts/bedrock_fake_server.py`` need this ~30-line encoder to
produce real frames: prelude (total length, headers length, CRC32), string
headers (type 7), payload, message CRC32. Decoding these with botocore's own
parser is what proves the client's framing path, not a hand-rolled reader.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct
from typing import Any


def _headers(headers: dict[str, str]) -> bytes:
    out = b""
    for name, value in headers.items():
        raw_name = name.encode("utf-8")
        raw_value = value.encode("utf-8")
        out += struct.pack("!B", len(raw_name)) + raw_name
        out += struct.pack("!BH", 7, len(raw_value)) + raw_value
    return out


def frame(headers: dict[str, str], payload: bytes) -> bytes:
    encoded_headers = _headers(headers)
    total = 12 + len(encoded_headers) + len(payload) + 4
    prelude = struct.pack("!II", total, len(encoded_headers))
    prelude += struct.pack("!I", binascii.crc32(prelude) & 0xFFFFFFFF)
    body = prelude + encoded_headers + payload
    return body + struct.pack("!I", binascii.crc32(body) & 0xFFFFFFFF)


def chunk(event: dict[str, Any]) -> bytes:
    """One ``chunk`` event frame carrying an Anthropic event."""
    payload = json.dumps(
        {"bytes": base64.b64encode(json.dumps(event).encode("utf-8")).decode("ascii")}
    ).encode("utf-8")
    return frame(
        {":message-type": "event", ":event-type": "chunk", ":content-type": "application/json"},
        payload,
    )


def exception(exception_type: str, message: str) -> bytes:
    return frame(
        {
            ":message-type": "exception",
            ":exception-type": exception_type,
            ":content-type": "application/json",
        },
        json.dumps({"message": message}).encode("utf-8"),
    )


def anthropic_events(
    *,
    text: str = "ok",
    usage: dict[str, int] | None = None,
    output_tokens: int | None = 3,
    metrics: dict[str, int] | None = None,
    extra_blocks: list[list[dict[str, Any]]] | None = None,
    stop_reason: str = "end_turn",
) -> list[dict[str, Any]]:
    """A complete Messages event sequence (text block, optional extra blocks)."""
    message: dict[str, Any] = {"id": "msg_bedrock_1", "model": "claude-haiku-4-5-20251001"}
    if usage is not None:
        message["usage"] = usage
    events: list[dict[str, Any]] = [{"type": "message_start", "message": message}]
    index = 0
    for block_events in extra_blocks or []:
        for event in block_events:
            events.append({**event, "index": index})
        index += 1
    events += [
        {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": index},
    ]
    delta: dict[str, Any] = {"type": "message_delta", "delta": {"stop_reason": stop_reason}}
    if output_tokens is not None:
        delta["usage"] = {"output_tokens": output_tokens}
    events.append(delta)
    stop: dict[str, Any] = {"type": "message_stop"}
    if metrics is not None:
        stop["amazon-bedrock-invocationMetrics"] = metrics
    events.append(stop)
    return events


def stream_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(chunk(event) for event in events)
