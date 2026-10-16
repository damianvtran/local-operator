"""V14: a voice take binds by id, never by timing.

The renderer binds each spoken take to the turn whose per-session stream carries
the USER ``message_start`` with ``message.id == <the take's request_id>``.
Until this file, that equality was an observed behaviour; this pins it as the
contract, over the real wire end to end:

* the real ``lop serve`` app (``server.app``) on a loopback uvicorn, with the
  real ``/v1/desktop/voice*`` and ``/v1/desktop/sessions/*`` routers;
* a real ``Session`` on the ``test`` hosting's own ``MockClient`` (no scripted
  turns), served by the real ``ServingSessionHandle``/``RuntimeServer`` the
  desktop pool attaches to;
* the published ``/events`` SSE stream, read LIVE and again on REPLAY
  (``epoch`` + ``after_seq``), which is how a renderer catches up.

Three admission paths, because each mints the user row at a different place:

1. a take admitted as a PROMPT (the session idle) -- the turn's append point;
2. a take admitted as a STEER mid-turn (``[bash:2]`` holds the turn on a real
   tool) -- delivered at a tool boundary by ``_drain_steering``;
3. a take steered into a turn's TAIL, after its last boundary drain -- carried
   by ``_flush_parked_deliveries`` once the turn ends. The tail is held open by
   parking the turn's ``_maybe_compact`` (the last await before the ``finally``
   that flushes), and the test asserts the flush actually saw the queued row,
   so the cell cannot pass by some other path.

The ``admission.command_id`` the route returns is asserted equal too: it is the
id the receipt journal keys on, and the UI reads it.
"""

from __future__ import annotations

import asyncio
import contextlib
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

from local_operator.providers.clients import MockClient
from local_operator.server.app import app
from local_operator.session import session as session_module
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tools.builtin import build_bash_tool
from tests.e2e.harness import build_session

pytestmark = pytest.mark.e2e

DEADLOCK_GUARD_S = 60.0


async def _pump(client: httpx.AsyncClient, url: str, sink: list[dict[str, Any]], **params):
    async with client.stream("GET", url, params=params or None) as response:
        assert response.status_code == 200
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                sink.append(json.loads(line[6:]))


async def _until(predicate) -> None:
    """Wait on state the code under test publishes; the bound is a deadlock guard."""
    async with asyncio.timeout(DEADLOCK_GUARD_S):
        while not predicate():
            await asyncio.sleep(0.02)


def _user_starts(frames: list[dict[str, Any]]) -> list[str]:
    """The ``message.id`` of every USER ``message_start`` on a stream, in order."""
    return [
        frame["payload"]["message"]["id"]
        for frame in frames
        if frame.get("type") == "event"
        and frame["payload"].get("type") == "message_start"
        and (frame["payload"].get("message") or {}).get("role") == "user"
    ]


def _ends(frames: list[dict[str, Any]]) -> int:
    return sum(
        1
        for frame in frames
        if frame.get("type") == "event" and frame["payload"].get("type") == "agent_end"
    )


@pytest.mark.asyncio
async def test_a_take_binds_to_the_user_message_start_whose_id_is_its_request_id(
    headless_tui_env: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )

    # Hold the FIRST tail-free turn's ``_maybe_compact`` open on request: it is the
    # last await before the turn's ``finally`` (which flushes parked steers), so
    # a steer admitted while it is held lands past the turn's last boundary
    # drain -- the shape ``_flush_parked_deliveries`` exists for.
    hold_tail = asyncio.Event()
    in_tail = asyncio.Event()
    release_tail = asyncio.Event()
    flush_queue_sizes: list[int] = []
    original_compact = session_module.Session._maybe_compact
    original_flush = session_module.Session._flush_parked_deliveries

    async def held_compact(self: Any) -> Any:
        if hold_tail.is_set() and not in_tail.is_set():
            in_tail.set()
            await release_tail.wait()
        return await original_compact(self)

    def observed_flush(self: Any) -> None:
        flush_queue_sizes.append(self._steering_queue.qsize())
        original_flush(self)

    monkeypatch.setattr(session_module.Session, "_maybe_compact", held_compact)
    monkeypatch.setattr(session_module.Session, "_flush_parked_deliveries", observed_flush)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    runtime = handle = None
    pumps: list[asyncio.Task[None]] = []
    try:
        await _until(lambda: server.started or serving.done())
        assert server.started
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}",
            timeout=30,
            headers={"Authorization": f"Bearer {token}"},
        ) as client:
            created = await client.post(
                "/v1/desktop/sessions",
                json={"request_id": str(uuid.uuid4()), "cwd": str(workspace)},
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]

            # The ``test`` hosting's own client, unmodified: ``[bash:N]`` is its
            # marker for a real ``bash`` call that holds the turn N seconds.
            mock = MockClient()

            async def stream(request: Any, signal: object = None):
                async for event in mock.stream(request, None):
                    yield event

            session = build_session(
                root / "sessions" / sid, stream, tools=[build_bash_tool()], cwd=workspace
            )
            # Named up front: an unnamed session fires the auto-naming errand,
            # a side turn this cell does not need on the stream.
            session.set_conversation_name("Voice binding", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            live: list[dict[str, Any]] = []
            pumps.append(
                asyncio.create_task(_pump(client, f"/v1/desktop/sessions/{sid}/events", live))
            )
            await _until(lambda: any(f["type"] == "snapshot" for f in live))
            opened = next(f for f in live if f["type"] == "open")
            enabled = await client.post(
                "/v1/desktop/voice", json={"op": "enable", "session_id": sid}
            )
            assert enabled.status_code == 200, enabled.text

            async def take(text: str) -> tuple[str, dict[str, Any]]:
                request_id = str(uuid.uuid4())
                response = await client.post(
                    "/v1/desktop/voice/utterance",
                    data={"session_id": sid, "request_id": request_id, "text": text},
                )
                assert response.status_code == 200, response.text
                admission = response.json()["result"]["admission"]
                assert admission["command_id"] == request_id
                return request_id, admission

            # (1) PROMPT: the session is idle, so the take opens the turn.
            prompt_id, admission = await take("[bash:2] list files")
            assert admission["detail"] == "prompt admitted"
            await _until(lambda: session.is_streaming)

            # (2) STEER mid-turn: bash holds the turn, the take queues behind it.
            steer_id, admission = await take("and the hidden ones too")
            assert admission["detail"] == "steering queued"
            await _until(lambda: _ends(live) >= 1 and not session.is_streaming)
            assert _user_starts(live) == [prompt_id, steer_id]

            # (3) STEER into the TAIL: open a fresh turn, hold it past its last
            # boundary, and steer while it is held.
            hold_tail.set()
            tail_turn_id, admission = await take("one more check")
            assert admission["detail"] == "prompt admitted"
            await _until(in_tail.is_set)
            assert session.is_streaming, "the held tail is still inside the turn"
            tail_steer_id, admission = await take("thanks, that's all")
            assert admission["detail"] == "steering queued"
            flushes_before = len(flush_queue_sizes)
            release_tail.set()
            await _until(lambda: len(_user_starts(live)) >= 4 and not session.is_streaming)
            # The flush that ended the held turn found the tail steer queued:
            # this is the path under test, not some other drain.
            assert flush_queue_sizes[flushes_before] == 1, flush_queue_sizes

            expected = [prompt_id, steer_id, tail_turn_id, tail_steer_id]
            assert _user_starts(live) == expected

            # REPLAY: a renderer catching up from the open cursor gets the same ids.
            replayed: list[dict[str, Any]] = []
            pumps.append(
                asyncio.create_task(
                    _pump(
                        client,
                        f"/v1/desktop/sessions/{sid}/events",
                        replayed,
                        epoch=opened["epoch"],
                        after_seq=opened["seq"],
                    )
                )
            )
            await _until(lambda: any(f["type"] == "snapshot" for f in replayed))
            replay_open = next(f for f in replayed if f["type"] == "open")
            assert not replay_open["payload"]["gap"], "the replay must not be a gap reset"
            assert _user_starts(replayed) == expected
    finally:
        for task in pumps:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        release_tail.set()
        server.should_exit = True
        with contextlib.suppress(Exception):
            await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
