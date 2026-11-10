"""A settings intent against a REAL runtime that is busy, then applied.

The assembled app (uvicorn over ``server.app.app``, so ``lifespan`` builds the
applier) and a real in-process ``RuntimeServer`` serving a session whose turn is
PARKED mid-stream by a scripted model. Only the model's replies are scripted.

What this pins that the unit tests cannot:

* the POST answers 202 while the owner is busy, without waiting on it (the
  route never touches the runtime) — the wall time is printed as evidence;
* the background applier drives the switch through the real bridge
  (``pool.session`` → ``bind_runtime`` → the attach client's ``set_model``
  frame, which carries a paired effort atomically), and the
  session's own journal selection — the applier's confirmation source — moves
  to the target;
* the slot reaches ``applied`` and GET reports it, and the session still
  completes its parked turn afterwards.

Isolation: ``headless_tui_env`` redirects the config dir and HOME; CMUX_* is
stripped; no provider is paid.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import time
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.server.app import app
from local_operator.session.model_selection import read_model_selection
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


class ParkedStream(ScriptedStream):
    """A scripted model whose next call blocks until released."""

    def __init__(self, turns):
        super().__init__(turns)
        self.block = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def __call__(self, request, signal=None):
        if not self.block:
            return super().__call__(request, signal)
        self.requests.append(request)
        self.block = False

        async def parked():
            self.started.set()
            await self.release.wait()
            for event in text_turn("Released answer"):
                yield event

        return parked()


async def until(predicate, seconds: float = 30.0):
    async with asyncio.timeout(seconds):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_model_pick_on_a_busy_runtime_is_recorded_at_once_and_applied(
    headless_tui_env: Path, workspace: Path, monkeypatch
):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = handle = None
    stream = None
    try:
        await until(lambda: server.started)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
            timeout=30,
            headers={"Authorization": "Bearer " + token},
        ) as client:
            features = (await client.get("/v1/capabilities")).json()
            flat = features.get("features") or features.get("result", {}).get("features") or {}
            assert flat.get("settings_intents") == 1, features

            created = await client.post(
                "/v1/desktop/sessions",
                json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            session_dir = root / "sessions" / sid
            stream = ParkedStream([text_turn("Seed answer")] + [text_turn("More")] * 6)
            session = build_session(session_dir, stream, cwd=workspace)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (session_dir / ".session.pid").write_text(str(os.getpid()))

            # BUSY: a real turn parked mid-stream.
            stream.block = True
            admitted = await client.post(
                f"/v1/desktop/sessions/{sid}/messages",
                json={"request_id": str(uuid.uuid4()), "text": "start a long turn"},
            )
            assert admitted.status_code == 200, admitted.text
            await asyncio.wait_for(stream.started.wait(), 30)
            assert session.is_streaming

            started = time.perf_counter()
            posted = await client.post(
                f"/v1/desktop/sessions/{sid}/settings-intents",
                json={"request_id": str(uuid.uuid4()), "kind": "model", "value": "test/other"},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            assert posted.status_code == 202, posted.text
            slot = posted.json()["result"]["intent"]
            assert slot["state"] in {"pending", "applying"}, slot
            # The harness session is born on its own scripted model; base is read
            # from that journal, not from config.
            assert slot["base"] == "test/e2e-model", slot
            print(
                f"POST settings-intents while busy: HTTP202 in {elapsed_ms:.1f} ms, {slot['state']}"
            )

            async def state() -> str | None:
                body = (await client.get("/v1/desktop/settings-intents")).json()["result"]
                found = body["intents"].get(sid, {}).get("model")
                return found and found["state"]

            seen: list[str | None] = []
            async with asyncio.timeout(60):
                while True:
                    current = await state()
                    if not seen or seen[-1] != current:
                        seen.append(current)
                    if current in {"applied", "failed", "superseded"}:
                        break
                    await asyncio.sleep(0.05)
            print(f"GET states over time: {seen}")
            assert seen[-1] == "applied", seen
            selection = await asyncio.to_thread(read_model_selection, session_dir)
            assert selection is not None and selection.selector == "test/other", selection
            print(f"journal selection after apply: {selection.selector}")

            # WARM latency: the first POST also paid one-time imports and the
            # catalogue cache read; a second pick shows the steady-state cost.
            started = time.perf_counter()
            again = await client.post(
                f"/v1/desktop/sessions/{sid}/settings-intents",
                json={"request_id": str(uuid.uuid4()), "kind": "model", "value": "test/third"},
            )
            warm_ms = (time.perf_counter() - started) * 1000
            assert again.status_code == 202, again.text
            assert again.json()["result"]["intent"]["generation"] == 2
            print(f"second POST while busy (warm): HTTP202 in {warm_ms:.1f} ms")

            # The parked turn still completes on the same runtime.
            stream.release.set()
            await until(lambda: not session.is_streaming)
    finally:
        if stream is not None:
            stream.release.set()
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
