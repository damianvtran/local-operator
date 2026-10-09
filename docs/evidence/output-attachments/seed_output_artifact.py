"""Seed an isolated config root with a session carrying a real output artifact.

Runs against the harness worktree's REAL code (branch feat/output-attachments):
the artifact bytes are registered through the real ``cache_media`` store call
and the rows are written through the real ``Transcript.append_message`` path —
the same code a tool result takes — so the frames the UI/TUI capture show the
real pipeline, not a hand-built JSONL.

The session holds THREE image shapes, one per durable convention:

  1. an OUTPUT ARTIFACT (kind + content_type + digest + metadata) — the new
     contract, riding a tool result the way ``generate_image`` will return it;
  2. a legacy EXTERNALISED image (``{attachment, mime_type}``) — a user row
     whose inline base64 was moved to the store by the old pass;
  3. a legacy SUB-FLOOR inline image (``{data}``) — small enough to stay
     inline, the oldest shape in the store.

Usage (isolated):
    env -i HOME=$RIG/home LOCAL_OPERATOR_CONFIG_DIR=$RIG/home/.local-operator \
      PATH="$PATH" TERM=xterm-256color \
      .venv/bin/python seed_output_artifact.py "$RIG"
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

RIG = Path(sys.argv[1])
SESSION_ID = "a77ac41f0001"

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from local_operator.harness.types import (  # noqa: E402
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.attachments import cache_media  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _artifact_png() -> bytes:
    """A 640x360 'generated' picture, visually unmistakable in a frame."""
    img = Image.new("RGB", (640, 360), (18, 30, 54))
    draw = ImageDraw.Draw(img)
    for i in range(0, 640, 16):
        draw.rectangle([i, 0, i + 8, 360], fill=(28 + (i * 5) % 120, 70, 150))
    draw.ellipse([420, 40, 600, 220], fill=(250, 190, 60))
    draw.rectangle([0, 250, 640, 360], fill=(12, 14, 22))
    font = _font(34)
    draw.text((26, 285), "OUTPUT ARTIFACT 640x360", fill=(235, 240, 250), font=font)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def _legacy_png() -> bytes:
    """A 320x200 legacy screenshot stand-in."""
    img = Image.new("RGB", (320, 200), (10, 74, 88))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 300, 180], outline=(220, 240, 235), width=3)
    font = _font(20)
    draw.text((42, 80), "LEGACY ATTACHMENT 320x200", fill=(230, 245, 240), font=font)
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def _tiny_png() -> bytes:
    """An 8x8 png under the externalisation floor (stays inline ``{data}``)."""
    img = Image.new("RGB", (8, 8), (240, 90, 90))
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def main() -> None:
    root = config_dir()
    session_dir = root / "sessions" / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)

    artifact_bytes = _artifact_png()
    artifact = cache_media(
        artifact_bytes,
        "image/png",
        name="output-artifact.png",
        source_url="https://provider.example/generated/fox-01.png",
    )
    if artifact is None:
        raise SystemExit("cache_media refused the artifact — store not writable?")
    legacy_b64 = base64.b64encode(_legacy_png()).decode("ascii")
    tiny_b64 = base64.b64encode(_tiny_png()).decode("ascii")

    transcript = Transcript(session_dir)
    import asyncio

    async def seed() -> None:
        await transcript.append_message(
            Message.user("Generate an image of a fox in a forest at dusk.")
        )
        await transcript.append_message(
            Message.assistant(
                "",
                tool_calls=[
                    ToolCall(
                        id="call_a77a_01",
                        name="generate_image",
                        arguments={"prompt": "fox in a forest at dusk"},
                    )
                ],
            )
        )
        await transcript.append_message(
            Message.tool_result(
                ToolResult(
                    tool_call_id="call_a77a_01",
                    tool_name="generate_image",
                    content=[
                        TextContent(text="Generated one image (640x360, %d bytes)." % len(artifact_bytes)),
                        artifact,
                    ],
                    details={
                        "provider": "mock",
                        "model": "mock-image",
                        "prompt": "fox in a forest at dusk",
                        "seed": 7,
                        "generation_id": "gen-01",
                    },
                    duration_s=4.2,
                )
            )
        )
        await transcript.append_message(
            Message.user(
                "Also, here is the earlier screenshot.",
                [ImageContent(data=legacy_b64, mime_type="image/png")],
            )
        )
        await transcript.append_message(
            Message.user("And a tiny inline one.", [ImageContent(data=tiny_b64, mime_type="image/png")])
        )
        await transcript.append_message(
            Message.assistant("Both are on the transcript now.")
        )

    asyncio.run(seed())

    # A title so the sidebar names the session if it is ever listed.
    try:
        from local_operator.resume import write_session_title

        write_session_title(session_dir, "Artifact demo", user_set=True, past_names=[])
    except Exception as exc:  # noqa: BLE001 — optional nicety
        print(f"title skipped: {exc}")

    # -- verify, from disk ------------------------------------------------
    raw = (session_dir / "transcript.jsonl").read_text(encoding="utf-8")
    kinds = []
    for line in raw.splitlines():
        payload = json.loads(line).get("payload", {})
        for block in payload.get("content", []):
            if isinstance(block, dict):
                kinds.append(block.get("kind") or ("legacy-ref" if "attachment" in block else ("inline" if "data" in block else "text")))
    print("block shapes on disk:", kinds)
    digest = artifact.attachment
    store_file = root / "attachments" / f"{digest}.bin"
    print("artifact digest:", digest)
    print("store file exists:", store_file.exists(), store_file.stat().st_size if store_file.exists() else 0)
    history = transcript.build_llm_history()
    from local_operator.harness.types import AttachmentContent

    found = [
        b
        for m in history
        if isinstance(m, Message)
        for b in m.content
        if isinstance(b, AttachmentContent)
    ]
    print("artifacts replayed:", len(found), found[0].model_dump(exclude_defaults=True) if found else None)
    print("SEED OK")


if __name__ == "__main__":
    main()
