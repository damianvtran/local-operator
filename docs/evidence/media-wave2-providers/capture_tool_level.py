"""Tool-level capture: the REAL ``generate_image`` path over a scripted socket.

    ISO=$(mktemp -d)
    env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/.local-operator" \
        PATH="$PATH" TERM=xterm-256color \
        <worktree>/.venv/bin/python docs/evidence/media-wave2-providers/capture_tool_level.py

Hermetic — spend $0.00, no real endpoint is contacted:

- the credential is a SYNTHETIC ``openrouter`` row in a throwaway config root;
- the socket is an ``httpx.MockTransport`` attached at the seam the rung docs
  name (``imagegen.rungs._client_scope``), serving one realistic OpenRouter
  Images response (the shape its docs' Response Format section documents);
- the attachment store is redirected to a scratch dir.

What it drives: resolver → cascade walk → ``_run_route`` → call-time
credential resolution → ``run_openrouter`` → asset decode → ``RungResult`` →
the tool's caption / attachment / details assembly. Nothing in the data path
is stubbed except the socket. Media wave-2, committed as the PR's tool-level
evidence (the wave-1 evidence shape: real code paths, scripted transport).

A second config root (empty) drives the refusal path so the output carries
the full seven-rung remedy sentence.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd4"
    "0000000049454e44ae426082"
)
B64 = base64.b64encode(PNG_1X1).decode("ascii")

SENT: list[dict[str, Any]] = []
UPDATES: list[Any] = []


def _mask(value: str) -> str:
    return f"{value[:12]}...({len(value)} chars, masked)" if value else ""


def _handler(request: httpx.Request) -> httpx.Response:
    SENT.append(
        {
            "method": request.method,
            "url": str(request.url),
            "authorization": _mask(request.headers.get("authorization", "")),
            "content_type": request.headers.get("content-type"),
            "body": json.loads(request.content.decode()) if request.content else None,
        }
    )
    return httpx.Response(
        200,
        json={
            "created": 1748372400,
            "data": [{"b64_json": B64, "media_type": "image/png"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 4175,
                "total_tokens": 4175,
                "cost": 0.04,
            },
        },
    )


async def main() -> None:
    home = os.environ.get("HOME", "")
    config = os.environ.get("LOCAL_OPERATOR_CONFIG_DIR", "")
    assert home and config, "isolation env missing (run under env -i, see the docstring)"
    assert "local-operator-worktrees" not in home, "HOME must be a scratch dir"
    assert not os.environ.get("OPENROUTER_API_KEY"), "an env key would bypass the stored row"
    config_dir = Path(config)
    config_dir.mkdir(parents=True, exist_ok=True)

    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=config_dir / "auth.db", config_dir=config_dir)
    store.upsert_credential(
        "openrouter",
        {"type": "api_key", "source": "login", "key": "SYNTHETIC-not-a-real-key"},
    )
    store.close()

    import local_operator.session.attachments as attachments

    scratch_store = Path(tempfile.mkdtemp(prefix="wave2-attach-"))
    attachments.attachments_dir = lambda: scratch_store  # type: ignore[assignment]

    from local_operator.imagegen import rungs as image_rungs

    @contextlib.asynccontextmanager
    async def scripted_scope(client=None):  # type: ignore[no-untyped-def]
        if client is not None:
            yield client
            return
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as owned:
            yield owned

    image_rungs._client_scope = scripted_scope  # type: ignore[assignment]

    from local_operator.harness.types import ToolContext
    from local_operator.imagegen import cascade
    from local_operator.tools import image_tool

    print("### resolver — the isolated config's only credential is the synthetic row")
    resolution = await cascade.resolve_image_route()
    print(f"route: {resolution.route} | reason: {resolution.reason}")
    print("rungs:", [(str(rung.route), rung.available) for rung in resolution.rungs])

    print()
    print("### the real tool -> REAL cascade -> REAL run_openrouter -> scripted socket")
    tool = image_tool.build_generate_image_tool(ToolContext(cwd="."))
    print("tool built:", tool is not None)
    result = await image_tool.execute_generate_image(
        "call-1",
        {
            "prompt": "a single red apple on a wooden table, soft daylight",
        },
        None,
        UPDATES.append,
        ToolContext(cwd="."),
    )
    print("is_error:", result.is_error)
    for block in result.content:
        if hasattr(block, "text"):
            print("caption:", block.text)
        elif hasattr(block, "attachment"):
            print(
                "attachment:",
                type(block).__name__,
                {"kind": block.kind, "attachment": block.attachment},
            )
    print("details:", json.dumps(result.details, indent=2, default=str))

    from local_operator.session.attachments import AttachmentStore

    digest = next(
        (
            getattr(block, "attachment", None)
            for block in result.content
            if getattr(block, "attachment", None)
        ),
        None,
    )
    stored = AttachmentStore().get_bytes(digest) if digest else None
    print(
        "store round-trip:",
        f"digest {digest} -> {None if stored is None else len(stored[0])} bytes, "
        f"byte-equal: {stored is not None and stored[0] == PNG_1X1}",
    )
    print("progress updates the tool streamed:", len(UPDATES))

    print()
    print("### what the REAL executor sent, and what the scripted socket served")
    print("request:", json.dumps(SENT[0] if SENT else {}, indent=2))
    print("response fields: b64_json length:", len(B64), "| usage.cost: 0.04")

    print()
    print("### the no-provider refusal (a second, EMPTY config root)")
    empty = Path(tempfile.mkdtemp(prefix="wave2-empty-"))
    refusal = await cascade.resolve_image_route(empty)
    print("route:", refusal.route)
    print("reason:", refusal.reason)


if __name__ == "__main__":
    asyncio.run(main())
