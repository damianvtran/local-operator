"""Real-shape journal rows for the session-files tests.

The shapes are copied from a real store (session ``9c59d684d16b``, read 2026-10-14), not
invented: a ``generate_image`` tool row is ``content[caption, AttachmentContent block]`` plus
``provider_payload.details`` with NO digest; a pruned row keeps ``details`` and replaces
``content`` with a notice (``transcript._pruned_entry``). Artifact bytes are registered through
the REAL ``cache_media`` into an isolated config dir, so a digest here is one the store can
actually serve.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

SESSION = "abcdef123456"
CHILD = "fedcba654321"
GRANDCHILD = "0123456789ab"


def png(width: int = 8, height: int = 6, seed: int = 7) -> bytes:
    """A valid, tiny, distinct PNG (distinct ``seed`` -> distinct digest)."""
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


def line(row: dict[str, Any]) -> bytes:
    return (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")


def assistant(row_id: str, ts: float, calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": row_id,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "assistant", "content": [], "tool_calls": calls},
    }


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"id": call_id, "name": name, "arguments": arguments}


def tool(
    row_id: str,
    ts: float,
    call_id: str,
    name: str,
    text: str,
    details: Any = None,
    *,
    extra_content: list[dict[str, Any]] | None = None,
    is_error: bool = False,
    pruned: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "message",
        "role": "tool",
        "content": [{"text": text}, *(extra_content or [])],
        "tool_call_id": call_id,
        "tool_name": name,
        "provider_payload": {"details": details, "useless": False, "duration_s": 0.1},
    }
    if is_error:
        payload["is_error"] = True
    if pruned:
        payload["content"] = [{"text": "[pruned: result no longer needed]"}]
        payload["provider_payload"]["pruned"] = True
    return {"id": row_id, "ts": ts, "type": "message", "payload": payload}


def image_block(digest: str, name: str = "flux-dev-01.png", size: int = 72) -> dict[str, Any]:
    """The ``AttachmentContent`` block ``_generated_result`` puts in ``content``."""
    return {
        "kind": "image",
        "content_type": "image/png",
        "attachment": digest,
        "source_url": "https://v3b.fal.media/files/b/x.png",
        "size_bytes": size,
        "width": 8,
        "height": 6,
        "name": name,
    }


def generate_image_rows(
    n: int,
    ts: float,
    digest: str,
    *,
    name: str = "flux-dev-01.png",
    pruned: bool = False,
    with_artifacts: bool = False,
) -> list[dict[str, Any]]:
    """An assistant call + its ``generate_image`` result, as ``_generated_result`` writes them."""
    call_id = f"call_img_{n}"
    details: dict[str, Any] = {
        "provider": "radient",
        "model": "[redacted]",
        "prompt": "a cat",
        "seed": None,
        "generation_id": f"gen-{n}",
        "cost_usd": 0.03,
    }
    if with_artifacts:
        details["artifacts"] = [
            {
                "digest": digest,
                "kind": "image",
                "name": name,
                "content_type": "image/png",
                "size_bytes": 72,
                "width": 8,
                "height": 6,
                "duration_s": None,
            }
        ]
    return [
        assistant(f"ga{n}", ts, [call(call_id, "generate_image", prompt="a cat", i="draw a cat")]),
        tool(
            f"gt{n}",
            ts + 0.5,
            call_id,
            "generate_image",
            f"Generated 1 image with Radient, 8x6 — attached (digest {digest}). Cost $0.03.",
            details,
            extra_content=[image_block(digest, name)],
            pruned=pruned,
        ),
    ]


def write_journal(session_dir: Path, rows: list[dict[str, Any]]) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "transcript.jsonl"
    path.write_bytes(b"".join(line(row) for row in rows))
    return path


def append_journal(session_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (session_dir / "transcript.jsonl").open("ab") as handle:
        handle.write(b"".join(line(row) for row in rows))
