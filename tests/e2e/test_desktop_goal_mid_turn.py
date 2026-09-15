"""A `/goal <text>` issued while a turn is RUNNING: answered, never parked.

WHY THIS IS ITS OWN CELL. The mid-turn path fails differently from the two
neighbours beside it. The dropped-receipt bug (``test_desktop_sessions.py``) lost
the text and there was no turn to notice; this one consumes the receipt
CORRECTLY — the goal is stored and the turn is queued — and the reply is what
breaks: ``admit_prompt`` acks on the owner's durable transcript append, which the
drain performs only when it reaches the command, so awaiting it parks the reply
for the whole of the turn already running. Past the client's ``ACK_TIMEOUT_S``
(15 s) the caller is told the owner is unavailable, and its retry under the same
request id reads as indeterminate — a 503-then-409 ladder for a command that
worked.

So the property under test is a BOUND, and only a real blocked turn can show it:
the provider stream is held open, and the `/commands` reply arriving while it is
still held is the proof, since an implementation that awaited the ack would not
return at all. Everything else is the production stack — real HTTP, real runtime,
real transcript, real loopback attach — with only the model's replies scripted.
"""

import asyncio
import os
import secrets
import socket
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn

from local_operator.server.app import app
from local_operator.server.routes import desktop_sessions as desktop_routes
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e

GOAL_TEXT = "Preserve one identity"


async def until(predicate) -> None:
    async with asyncio.timeout(15):
        while not predicate():
            await asyncio.sleep(0.001)


def request_id() -> str:
    return str(uuid.uuid4())


class HeldStream(ScriptedStream):
    """Scripted turns whose FIRST call is held open until the test releases it.

    The hold is the whole instrument: the turn is streaming and cannot end until
    ``release()``, so any reply that came back came back without waiting for it.
    Later calls replay the script untouched, which is what lets the cell assert
    that the steered goal text reaches the provider exactly once.
    """

    def __init__(self, turns) -> None:
        super().__init__(turns)
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self._held = False

    def __call__(self, request, signal=None):
        if self._held:
            return super().__call__(request, signal)
        self._held = True
        self.requests.append(request)

        async def held():
            self.started.set()
            await self.released.wait()
            for event in text_turn("The held turn finished."):
                yield event

        return held()

    def release(self) -> None:
        self.released.set()


@pytest.mark.asyncio
async def test_a_goal_mid_turn_is_answered_without_waiting_for_the_turn(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
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
    try:
        await until(lambda: server.started)
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=30
        ) as client:
            client.headers["Authorization"] = f"Bearer {token}"
            created = await client.post(
                "/v1/desktop/sessions", json={"request_id": request_id(), "cwd": str(workspace)}
            )
            assert created.status_code == 200, created.text
            sid = created.json()["result"]["session_id"]
            target = "/v1/desktop/sessions/" + sid

            # Two turns: the held one, and the continuation that has to carry the
            # steered goal text. A third call would be a second turn for a goal
            # that was already delivered, and ``ScriptedStream`` fails loudly on a
            # call past the end rather than answering with a bare stop.
            stream = HeldStream(
                [text_turn("The held turn finished."), text_turn("Carried the steer.")]
            )
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            # Named up front, deliberately: an unnamed conversation fires the
            # owner's separate title-model errand, which is a REAL provider call
            # on this same stream — it would be the call this cell holds open,
            # leaving the turn itself queued behind it and the instrument
            # measuring the wrong thing.
            session.set_conversation_name("Mid-turn goal cell", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))

            # A real turn, streaming and held open.
            started = await client.post(
                target + "/messages",
                json={"request_id": request_id(), "text": "Start a long turn"},
            )
            assert started.status_code == 200, started.text
            await asyncio.wait_for(stream.started.wait(), 15)
            # The turn is streaming and cannot finish: the hold is still set and
            # no second call has been made. Asserted on this cell's own
            # instrument rather than on a session flag, because the flag is the
            # runtime's business and the hold is the fact under test.
            assert not stream.released.is_set()
            assert len(stream.requests) == 1

            # THE ASSERTION THIS FILE EXISTS FOR: the goal command is answered
            # while that turn is still held. The ten seconds are the test's own
            # deadlock guard, not a product bound -- the reply either arrives
            # without the turn or never arrives at all.
            #
            # The bridge is awaited to have SEEN the turn first. That removes a
            # race rather than loosening the cell: the steer choice is read off
            # the bridge's view of the owner, and this cell is about what happens
            # once it has one. Without the wait a lagging frontend update would
            # send the command down the idle path and still pass, which would
            # make the disposition below accidental.
            async with app.state.desktop_sessions.session(sid) as bridge:
                await until(lambda: bool(bridge.remote.is_streaming))
                goal_id = request_id()
                answered = await asyncio.wait_for(
                    client.post(
                        target + "/commands",
                        json={"request_id": goal_id, "command": "goal", "args": GOAL_TEXT},
                    ),
                    timeout=10,
                )
            assert answered.status_code == 200, answered.text
            assert (
                not stream.released.is_set()
            ), "the turn was released early; the bound is untested"
            admission = answered.json()["result"]["result"]["admission"]
            assert admission["status"] == "admitted"
            # Both of these are the STEER disposition: the owner's own ack, or
            # the host's queued notice when even that had not landed inside the
            # prelude. Which one lands is a scheduling detail, so the cell
            # accepts either -- what it does not accept is the idle-path phrase,
            # which would mean the turn in flight was not recognised at all.
            assert admission["detail"] in {
                "steering queued",
                desktop_routes.QUEUED_ADMISSION_DETAIL,
            }, admission["detail"]
            # The goal itself landed on the owner.
            assert session.goal == GOAL_TEXT

            # And the retry under the same id is a REPLAY, not the 409 that the
            # parked path produced once the ack deadline had passed.
            retried = await client.post(
                target + "/commands",
                json={"request_id": goal_id, "command": "goal", "args": GOAL_TEXT},
            )
            assert retried.status_code == 200, retried.text
            assert retried.json()["result"]["replayed"] is True

            # Release the turn: the steered goal text must reach the provider
            # exactly once, in the follow-up call that carries it.
            stream.release()
            await until(lambda: len(stream.requests) >= 2)
            carried = [
                request for request in stream.requests if GOAL_TEXT in request.model_dump_json()
            ]
            assert len(carried) == 1, (
                "the steered goal text must reach the provider exactly once; "
                f"it arrived in {len(carried)} of {len(stream.requests)} calls"
            )
            entries = (await client.get(target + "/history")).json()["result"]["entries"]
            assert (
                sum(GOAL_TEXT in str(entry) for entry in entries) == 1
            ), "one durable user row for one goal submission"
            print(
                "HTTP /goal mid-turn: answered 200 while the turn was still held (no 503, no "
                f"ack-deadline wait); admission detail {admission['detail']!r}; goal stored; "
                "retry replayed; exactly one provider call carried the text and exactly one "
                "durable row recorded it"
            )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()
