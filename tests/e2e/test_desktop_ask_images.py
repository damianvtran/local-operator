"""A queued-ask answer WITH an image, end to end: HTTP body -> model message.

Why this file exists
--------------------

``tests/unit`` pins each hop of the image-answer wire in isolation. What it
cannot show is the property the feature is for: the picture the user attached to
their answer is the picture the MODEL receives, and the text-only path did not
move. Everything between the HTTP body and the provider request is production
code here -- the real app and ``Answer`` model, the real desktop facade dialling
a real ``RuntimeServer`` over loopback (so ``ask-attachments-v1`` is negotiated
from the owner's own record), the real ``ServingSessionHandle`` and ``Session``,
the real ``AskQueue`` and ``asks.jsonl`` and attachment store, and the real
transcript. Only the provider stream is scripted.

It is modelled on ``test_desktop_input_mode.py`` (same rig: the ``.session.pid``
marker is what makes the pool attach to THIS owner instead of spawning one).
"""

import asyncio
import base64
import contextlib
import io
import json
import os
import secrets
import socket
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from local_operator.asks import store as ask_store
from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import ASK_ATTACHMENTS_CAPABILITY
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


async def until(predicate, timeout_s: float = 30.0) -> None:
    """The deadline is a deadlock guard, not the assertion (AGENTS.md, timing)."""
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.005)


def _png_b64() -> str:
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.frombytes("RGB", (48, 48), os.urandom(48 * 48 * 3)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_config(root: Path) -> None:
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )


async def _no_ask(_questions: Any) -> None:
    """The host hook: only its PRESENCE makes the queued arm live."""
    return None


def _questions() -> list[dict[str, Any]]:
    return [
        {
            "id": "q1",
            "question": "Which design do you want?",
            "options": [{"label": "A"}, {"label": "B"}],
            "multi": False,
            "secret": False,
            "persist": False,
            "recommended": None,
        }
    ]


@contextlib.asynccontextmanager
async def _running_app(root: Path, workspace: Path, stream: ScriptedStream, token: str):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = handle = None
    try:
        await until(lambda: server.started)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            created = await client.post(
                "/v1/desktop/sessions",
                json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            session.set_conversation_name("Ask image answer cell", user_set=True)
            session.set_ask_handler(_no_ask)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            # The production handle advertises the capability: the unit cells argue
            # the gate's answers, this asserts the build the desktop runs has it.
            assert ASK_ATTACHMENTS_CAPABILITY in runtime._record.capabilities
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
            yield client, "/v1/desktop/sessions/" + sid, session, sid
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()


def _images_in(request: Any) -> list[Any]:
    return [
        block
        for message in request.messages
        for block in message.content
        if getattr(block, "type", "") == "image"
    ]


@pytest.mark.asyncio
async def test_an_image_answer_reaches_the_durable_row_and_the_model(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    _write_config(root)
    stream = ScriptedStream([text_turn("Saw it."), text_turn("Saw the text.")])
    image_b64 = _png_b64()

    async with _running_app(root, workspace, stream, token) as (client, target, session, sid):
        queue = session.ask_queue()
        assert queue is not None, "the queued arm must be live"
        first = queue.enqueue(_questions(), None)["details"]["ask_id"]

        # 1. THE IMAGE ANSWER, over real HTTP.
        answered = await client.post(
            target + "/answers",
            json={
                "ask_id": first,
                "answers": {"q1": ["the screenshot"]},
                "images": [{"question_id": "q1", "data_b64": image_b64, "mime_type": "image/png"}],
            },
        )
        assert answered.status_code == 200, answered.text
        row_id = ask_store.response_row_id(first)
        await until(lambda: len(stream.requests) >= 1 and not session.is_streaming)

        # 2. THE MODEL GOT THE PICTURE, after the text that points at it.
        request = stream.requests[0]
        assert len(_images_in(request)) == 1, "the answer's image never reached the provider"
        answer_turn = [m for m in request.messages if m.role == "user"][-1]
        assert [b.type for b in answer_turn.content] == ["text", "image"]
        print("model input for the answer turn:", [b.type for b in answer_turn.content])
        print(answer_turn.text)
        assert "the screenshot (+1 image, shown below)" in answer_turn.text

        # 3. THE DURABLE STATE: refs in the ask log, a small journal row, bytes in the store.
        log = (root / "sessions" / sid / "asks.jsonl").read_text()
        assert image_b64 not in log, "base64 must never reach asks.jsonl"
        (event,) = [
            json.loads(line) for line in log.splitlines() if json.loads(line)["kind"] == "answered"
        ]
        (ref,) = event["attachments"]["q1"]
        assert set(ref) == {"attachment", "mime_type", "bytes"}
        assert (root / "attachments" / f"{ref['attachment']}.bin").exists()
        journal = [
            line
            for line in (root / "sessions" / sid / "transcript.jsonl").read_text().splitlines()
            if row_id in line
        ]
        assert len(journal) == 1 and image_b64 not in journal[0]
        assert len(journal[0]) < 4000, "the response row must hold a digest, not the picture"

        # 4. THE TEXT-ONLY PATH DID NOT MOVE: no images in the next request, no
        # attachments key anywhere on its log event or its row.
        second = queue.enqueue(
            [{**_questions()[0], "question": "And the colour?"}],
            None,
        )[
            "details"
        ]["ask_id"]
        text_only = await client.post(
            target + "/answers", json={"ask_id": second, "answers": {"q1": ["A"]}}
        )
        assert text_only.status_code == 200, text_only.text
        await until(lambda: len(stream.requests) >= 2 and not session.is_streaming)
        newest_turn = [m for m in stream.requests[1].messages if m.role == "user"][-1]
        assert [b.type for b in newest_turn.content] == ["text"]
        (event2,) = [
            json.loads(line)
            for line in (root / "sessions" / sid / "asks.jsonl").read_text().splitlines()
            if json.loads(line)["kind"] == "answered" and json.loads(line)["ask_id"] == second
        ]
        assert "attachments" not in event2

        # 5. A BAD IMAGE REFUSES THE WHOLE ANSWER AND LEAVES THE ASK OPEN.
        third = queue.enqueue(
            [{**_questions()[0], "question": "One more?"}],
            None,
        )[
            "details"
        ]["ask_id"]
        refused = await client.post(
            target + "/answers",
            json={
                "ask_id": third,
                "answers": {"q1": ["x"]},
                "images": [
                    {
                        "question_id": "q1",
                        "data_b64": base64.b64encode(b"definitely not an image").decode(),
                        "mime_type": "image/png",
                    }
                ],
            },
        )
        assert refused.status_code == 409, refused.text
        assert "could not be read" in refused.json()["detail"], refused.json()
        record = queue.find(third)
        assert record is not None and record["status"] == ask_store.STATUS_OPEN
        assert len(stream.requests) == 2, "a refused answer reached the provider"

        print(
            "ask image e2e: HTTP image answer -> model request carried [text, image]; "
            "journal row holds a digest; asks.jsonl holds refs only; a text-only answer "
            "carried no image and no attachments key; an undecodable image was refused "
            "409 and the ask stayed open"
        )
