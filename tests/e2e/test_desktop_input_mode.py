"""The silent input-metadata carriage, end to end: body -> durable user row.

Covers the frozen semantics of ``input_mode``/``input_path`` (see
``harness.types.Message`` and the PR's field reference):

* a send carrying ``input_mode`` lands it on the durable JSONL row, and a
  PRESENT ``input_path`` (an open string) round-trips beside it;
* a body WITHOUT the fields still works exactly as before — the row carries no
  key at all, the byte-identical property legacy rows are promised;
* an out-of-enum ``input_mode`` is refused at the wire with a 422 before
  anything is admitted (``Prompt`` is ``extra="forbid"`` and the enum is
  enforced there);
* a mid-turn dictation rides the STEER op onto the queued row when the drain
  at the next boundary persists it.

Only the provider stream is scripted (``tests/e2e/harness``); everything else
is the production stack — real HTTP, real RuntimeServer/ServingSessionHandle,
real loopback AttachClient, real transcript on disk.
"""

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

from local_operator.server.app import app
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.runtime.types import INPUT_MODE_CAPABILITY
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


async def until(predicate, timeout_s: float = 20.0) -> None:
    """Poll ``predicate`` until true; the deadline is a deadlock guard, not the assertion.

    AGENTS.md, "Wait on the event, never on the clock": every caller asserts the
    state it waited for afterwards, so a slow box reports an assertion rather
    than a timing failure.
    """
    async with asyncio.timeout(timeout_s):
        while not predicate():
            await asyncio.sleep(0.001)


def request_id() -> str:
    return str(uuid.uuid4())


def _rows(root: Path, sid: str) -> list[dict[str, Any]]:
    """The session's durable JSONL rows, as they are on disk right now."""
    path = root / "sessions" / sid / "transcript.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _row(root: Path, sid: str, entry_id: str) -> dict[str, Any] | None:
    for row in _rows(root, sid):
        if row.get("id") == entry_id:
            return row
    return None


def _introduced_in(stream: ScriptedStream, text: str) -> list[int]:
    """The call indexes where ``text`` FIRST appears — i.e. was delivered.

    Every later turn replays the conversation as history, so the transition is
    what a delivery count reads; a duplicated submission shows as a second
    index.
    """
    seen = False
    indexes = []
    for index, request in enumerate(stream.requests):
        present = text in request.model_dump_json()
        if present and not seen:
            seen = True
            indexes.append(index)
    return indexes


class HeldStream(ScriptedStream):
    """Scripted turns whose FIRST call is held open until ``release()``.

    The hold is what makes a steer mid-turn: the turn is streaming and cannot
    end until released, so a queued steer is exercised against a live turn
    rather than against an idle queue.
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


@contextlib.asynccontextmanager
async def _running_app(root: Path, workspace: Path, stream: ScriptedStream, token: str):
    """Boot the assembled app plus an in-process owner runtime for one session.

    Mirrors ``test_desktop_goal_mid_turn.py``'s rig: the session is real, the
    runtime is a real ``RuntimeServer`` in this process, and the ``.session.pid``
    marker is what makes the desktop pool's facade attach to THIS daemon over
    the real loopback socket instead of spawning one.
    """
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
            session = build_session(root / "sessions" / sid, stream, cwd=workspace)
            # Named up front, deliberately: an unnamed conversation fires the
            # owner's separate title-model errand, which is a REAL provider call
            # on this same stream and would take a scripted turn.
            session.set_conversation_name("Input-mode carriage cell", user_set=True)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(workspace))
            runtime = RuntimeServer(handle, kind="daemon")
            await runtime.start_in_process()
            # The REAL production handle advertises the carriage token: the
            # pinned fakes in ``tests/unit/session/runtime/test_input_mode_capability.py``
            # argue the gate's four answers, and this asserts the build the
            # desktop actually runs carries it.
            assert INPUT_MODE_CAPABILITY in runtime._record.capabilities
            (root / "sessions" / sid / ".session.pid").write_text(str(os.getpid()))
            yield client, target, session, sid
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, 30)
        listener.close()
        if runtime is not None:
            await runtime.aclose()
        if handle is not None:
            await handle.dispose()


def _write_config(root: Path) -> None:
    (root / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
    )


@pytest.mark.asyncio
async def test_input_metadata_rides_the_durable_row_and_the_wire_validates(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    _write_config(root)
    stream = ScriptedStream(
        [text_turn("Answer one."), text_turn("Answer two."), text_turn("Answer three.")]
    )

    async with _running_app(root, workspace, stream, token) as (client, target, session, sid):
        # 1. A dictated send, with the cascade's reserved route slot set: both
        # fields must survive admission and land on the durable row.
        dictated_id = request_id()
        dictated = await client.post(
            target + "/messages",
            json={
                "request_id": dictated_id,
                "text": "Dictated hello",
                "input_mode": "dictated",
                "input_path": "sidecar_transcription",
            },
        )
        assert dictated.status_code == 200, dictated.text
        admission = dictated.json()["result"]
        assert admission["status"] == "admitted", admission
        assert admission["detail"] == "prompt admitted", admission["detail"]
        assert admission["duplicate"] is False, admission

        await until(
            lambda: _row(root, sid, dictated_id) is not None
            and len(stream.requests) >= 1
            and not session.is_streaming
        )
        row = _row(root, sid, dictated_id)
        assert row is not None, "the dictated send never reached the durable transcript"
        assert row["payload"]["role"] == "user"
        assert row["payload"]["input_mode"] == "dictated", row["payload"]
        assert row["payload"]["input_path"] == "sidecar_transcription", row["payload"]

        # 2. A body WITHOUT the fields — the legacy client — must work exactly
        # as before, and its row must carry NO key (that absence is the
        # byte-identical promise).
        legacy_id = request_id()
        legacy = await client.post(
            target + "/messages", json={"request_id": legacy_id, "text": "Legacy hello"}
        )
        assert legacy.status_code == 200, legacy.text
        assert legacy.json()["result"]["status"] == "admitted"

        await until(
            lambda: _row(root, sid, legacy_id) is not None
            and len(stream.requests) >= 2
            and not session.is_streaming
        )
        legacy_row = _row(root, sid, legacy_id)
        assert legacy_row is not None
        assert "input_mode" not in legacy_row["payload"], legacy_row["payload"]
        assert "input_path" not in legacy_row["payload"], legacy_row["payload"]

        # 3. An out-of-enum value is refused at the wire, before the owner is
        # bound and before anything durable exists.
        rejected_id = request_id()
        rejected = await client.post(
            target + "/messages",
            json={"request_id": rejected_id, "text": "Bad enum", "input_mode": "ranting"},
        )
        assert rejected.status_code == 422, rejected.text
        # The app sanitises validation errors to one generic sentence (it does
        # not echo field names), so the DISCRIMINATOR is the status beside the
        # accepted "dictated" send above: same field, same body shape, only the
        # VALUE moved it from admitted to refused.
        assert "invalid fields" in rejected.text, rejected.text
        assert _row(root, sid, rejected_id) is None, "a refused send wrote a durable row"
        assert len(stream.requests) == 2, "a refused send reached the provider"

        print(
            "input carriage e2e: dictated send -> row carries "
            f"input_mode={row['payload']['input_mode']!r} "
            f"input_path={row['payload']['input_path']!r}; a body without the fields"
            " still admitted and its row carries neither key; an out-of-enum value"
            " was refused 422 with no row and no provider call"
        )


@pytest.mark.asyncio
async def test_a_mid_turn_dictation_rides_the_steer_onto_the_row(
    headless_tui_env: Path, workspace: Path, monkeypatch
) -> None:
    root = headless_tui_env
    token = secrets.token_hex(32)
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", token)
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)
    _write_config(root)
    stream = HeldStream(
        [text_turn("unused"), text_turn("Carried the steer."), text_turn("Settled.")]
    )

    async with _running_app(root, workspace, stream, token) as (client, target, session, sid):
        # A real turn, streaming and held open.
        started = await client.post(
            target + "/messages",
            json={"request_id": request_id(), "text": "Start a long turn"},
        )
        assert started.status_code == 200, started.text
        await asyncio.wait_for(stream.started.wait(), 15)
        assert session.is_streaming

        # The dictation arrives mid-turn and rides the steer: the ack is the
        # queue insertion, and it must be answered while the turn is still held.
        steer_id = request_id()
        steered = await asyncio.wait_for(
            client.post(
                target + "/messages",
                json={
                    "request_id": steer_id,
                    "text": "Mid-turn dictation",
                    "mode": "steer",
                    "input_mode": "dictated",
                },
            ),
            timeout=10,
        )
        assert steered.status_code == 200, steered.text
        admission = steered.json()["result"]
        assert admission["status"] == "admitted", admission
        assert admission["detail"] == "steering queued", admission["detail"]
        assert not stream.released.is_set(), "the turn was released early; the steer is untested"

        # Release: the drain at the next boundary persists the queued row and
        # the follow-up model call carries the steered text.
        stream.release()
        await until(lambda: _row(root, sid, steer_id) is not None, timeout_s=20)
        await until(lambda: not session.is_streaming)
        row = _row(root, sid, steer_id)
        assert row is not None, "the steer never reached the durable transcript"
        assert row["payload"]["role"] == "user"
        assert row["payload"]["input_mode"] == "dictated", row["payload"]
        assert "input_path" not in row["payload"], row["payload"]

        delivered = _introduced_in(stream, "Mid-turn dictation")
        assert len(delivered) == 1, (
            "the steered text must reach the provider exactly once, never twice:"
            f" delivered in calls {delivered}"
        )
        assert delivered[0] >= 1, (
            "and NOT in the call that started the held turn, which was already"
            f" streaming when the steer was sent: delivered in call {delivered[0]}"
        )

        print(
            "input carriage e2e (steer): mid-turn dictation answered"
            f" {admission['status']}/{admission['detail']!r} while the turn was held;"
            " the drained row carries input_mode='dictated'; the steer text reached"
            f" the provider exactly once, in call {delivered[0]}"
        )
