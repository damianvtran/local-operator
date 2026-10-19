"""Fixtures for the Library tests: a synthetic config root and real-shaped journals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SESSION = "abcdef123456"
OTHER = "fedcba654321"


def strip_env(monkeypatch) -> None:
    """Isolate a test from the harness and a developer's own config.

    A binding rule, not tidiness: a leaked ``LOCAL_OPERATOR_CONFIG_DIR`` makes a
    test write into the operator's real catalog, and ``CMUX_*``/``LOP_*`` carry
    a session identity that can redirect tool resolution.
    """
    for name in list(os.environ):
        if name.startswith(("CMUX_", "LOP_", "LOCAL_OPERATOR_")):
            monkeypatch.delenv(name, raising=False)


def config_root(monkeypatch, tmp_path: Path) -> Path:
    strip_env(monkeypatch)
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


def session(
    root: Path, session_id: str = SESSION, rows: list[dict[str, Any]] | None = None
) -> Path:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        (directory / "transcript.jsonl").write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    return directory


def user_row(row_id: str, ts: float, text: str, **blocks: Any) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"text": text}]
    if blocks:
        content.append(blocks)
    return {
        "id": row_id,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": content},
    }


def write_rows(call_id: str, ts: float, path: Path, text: str = "saved") -> list[dict[str, Any]]:
    return [
        {
            "id": f"a-{call_id}",
            "ts": ts,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {
                        "id": call_id,
                        "name": "write",
                        "arguments": {"path": str(path), "content": "x"},
                    }
                ],
            },
        },
        {
            "id": f"t-{call_id}",
            "ts": ts + 0.5,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "tool",
                "content": [{"text": text}],
                "tool_call_id": call_id,
                "tool_name": "write",
                "provider_payload": {"details": {"path": str(path)}},
            },
        },
    ]


def image_rows(call_id: str, ts: float, digest: str, name: str = "cat.png") -> list[dict[str, Any]]:
    block = {
        "kind": "image",
        "attachment": digest,
        "content_type": "image/png",
        "name": name,
        "size_bytes": 72,
        "width": 8,
        "height": 6,
    }
    return [
        {
            "id": f"a-{call_id}",
            "ts": ts,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [],
                "tool_calls": [
                    {"id": call_id, "name": "generate_image", "arguments": {"prompt": "cat"}}
                ],
            },
        },
        {
            "id": f"t-{call_id}",
            "ts": ts + 0.5,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "tool",
                "content": [{"text": "ok"}, block],
                "tool_call_id": call_id,
                "tool_name": "generate_image",
                "provider_payload": {
                    "details": {
                        "artifacts": [
                            {
                                "digest": digest,
                                "kind": "image",
                                "name": name,
                                "content_type": "image/png",
                                "size_bytes": 72,
                                "width": 8,
                                "height": 6,
                            }
                        ]
                    }
                },
            },
        },
    ]


def png(width: int = 8, height: int = 6, seed: int = 7) -> bytes:
    """A valid tiny PNG, so Pillow and the attachment store see real bytes."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes([seed]) * (width * 3) for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = struct.pack(">I", len(data)) + kind + data
        return body + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
