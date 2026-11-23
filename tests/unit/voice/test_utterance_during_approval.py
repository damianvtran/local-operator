"""V13, against a real runtime: an utterance at a parked ``bash`` card is refused.

QA round 1 (Q3) measured that a steer cancels an INTERRUPTIBLE tool's approval
card ("Tool call skipped: interrupted by steering."), so the voice route must not
steer into one. This drives the real objects end to end and scripts nothing but
the provider:

* a real ``Session`` on the ``test`` mock provider (``MockClient``), whose
  ``[bash:1]`` marker asks for the REAL ``bash`` tool, with approvals armed;
* the real ``ServingSessionHandle`` and ``RuntimeServer``, and the real
  ``AttachedSession`` facade the desktop pool hands the routes;
* the real ``/v1/desktop/voice*`` and ``/messages`` routers over ASGI.

What it asserts: with the card parked, the text door (the only door, Q10) is
refused 409 ``approval_pending``; the card is still the same pending card with
an unresolved future; and nothing new was admitted. Then the operator denies on
the card (the ``/answers`` path the desktop uses), and the same utterance is
admitted as an ordinary prompt.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.harness.types import ChatRequest
from local_operator.providers.clients import MockClient
from local_operator.server.dependencies import get_provider_auth_store
from local_operator.server.routes import desktop_sessions, desktop_voice
from local_operator.session.attached import AttachedSession
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_bash_tool
from local_operator.voice import slot as voice_slot
from tests.e2e.harness import TEST_MODEL
from tests.unit.session.test_remote import _never_take_over

TOKEN = "voice-v13-token"
SESSION = "d4d4d4d4d4d4"
DEADLOCK_GUARD_S = 30.0


async def _until(predicate) -> None:
    """Wait on state the code under test publishes; the bound is a deadlock guard."""
    async with asyncio.timeout(DEADLOCK_GUARD_S):
        while not predicate():
            await asyncio.sleep(0.01)


def _user_rows(directory: Path) -> list[str]:
    path = directory / "transcript.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line).get("payload") or {}
        if payload.get("role") == "user":
            # A text block is stored WITHOUT its ``type`` (``exclude_defaults``).
            rows.append("".join(block.get("text", "") for block in payload.get("content", [])))
    return rows


class OneBridge:
    """The two bridge seams ``/messages`` calls besides ``remote``; local-session shaped."""

    def __init__(self, remote: AttachedSession) -> None:
        self.remote = remote

    async def prepare_peer_images(self, images: list[dict[str, str]]) -> list[dict[str, str]]:
        return images

    async def refresh_watch(self) -> None:
        return None


class OnePool:
    """``DesktopSessions``-shaped door onto ONE real attached facade."""

    def __init__(self, root: Path, remote: AttachedSession) -> None:
        self.root = root
        self.bridge = OneBridge(remote)

    @contextlib.asynccontextmanager
    async def session(self, session_id: str, *, read: bool = False, allow_draft: bool = False):
        if session_id != SESSION:
            raise KeyError("Unknown session")
        yield self.bridge

    async def announce_admission(self, *_: Any, **__: Any) -> bool:
        return False

    async def announce_admission_failure(self, *_: Any, **__: Any) -> bool:
        return False


@contextlib.asynccontextmanager
async def _parked_bash_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Voice enabled, ``[bash:1]`` admitted through the voice door, its card parked."""
    config = tmp_path / "config"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    directory = config / "sessions" / SESSION
    directory.mkdir(parents=True)
    (config / "config.yml").write_text(
        "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
    )

    # The ``test`` hosting's own client, unmodified: ``[bash:1]`` is its marker
    # for a real ``bash`` call. Side calls (conversation naming) carry no tools
    # and get a plain answer, so they cannot consume the turn.
    mock = MockClient()

    async def stream(request: ChatRequest, signal: object = None):
        async for event in mock.stream(request, None):
            yield event

    session = Session(
        model=TEST_MODEL,
        stream_fn=stream,
        tools=[build_bash_tool()],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
        yolo=False,  # the gate ARMED: the turn must park on a card
        cwd=str(workspace),
    )
    handle = ServingSessionHandle(
        session, asyncio.get_running_loop(), cwd=str(workspace), auto_approve=False
    )
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    remote = await AttachedSession.connect(
        server._record, SESSION, config_dir=config, takeover_factory=_never_take_over
    )

    app = FastAPI()
    app.state.config_manager = ConfigManager(config)
    app.state.env_config = SimpleNamespace(radient_api_base_url="http://radient.invalid")
    app.state.instance_id = "inst-v13"
    app.state.desktop_sessions = OnePool(config, remote)
    app.include_router(desktop_voice.router)
    app.include_router(desktop_sessions.router)

    async def no_store():
        return SimpleNamespace(close=lambda: None)

    # The text door never reaches STT, but the route declares the store.
    app.dependency_overrides[get_provider_auth_store] = no_store

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://localhost",
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as client:
            enabled = await client.post(
                "/v1/desktop/voice", json={"op": "enable", "session_id": SESSION}
            )
            assert enabled.status_code == 200, enabled.text
            assert voice_slot.read_slot(config) is not None

            # The voice door itself starts the turn that will park.
            first = await client.post(
                "/v1/desktop/voice/utterance",
                data={
                    "session_id": SESSION,
                    "request_id": str(uuid.uuid4()),
                    "text": "[bash:1] list files",
                },
            )
            assert first.status_code == 200, first.text
            assert first.json()["result"]["admission"]["status"] == "admitted"

            await _until(lambda: handle._fold.projection.pending is not None)
            card = handle._fold.projection.pending
            assert card is not None and card.kind == "approval" and card.title == "bash"
            gate_future = handle._pending_futures[card.request_id]
            # The facade the route reads must have caught up with the owner.
            await _until(lambda: remote.pending_gate is not None)
            rows_before = _user_rows(directory)
            assert rows_before == ["[bash:1] list files"]

            yield SimpleNamespace(
                client=client,
                handle=handle,
                remote=remote,
                session=session,
                directory=directory,
                card=card,
                gate_future=gate_future,
                rows_before=rows_before,
            )
    finally:
        await remote.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_an_utterance_at_a_parked_bash_card_is_refused_and_the_card_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _parked_bash_card(tmp_path, monkeypatch) as rig:
        client, handle, remote, session = rig.client, rig.handle, rig.remote, rig.session
        directory, card, gate_future = rig.directory, rig.card, rig.gate_future
        rows_before = rig.rows_before
        text_door = await client.post(
            "/v1/desktop/voice/utterance",
            data={"session_id": SESSION, "request_id": str(uuid.uuid4()), "text": "yes"},
        )
        assert text_door.status_code == 409, text_door.text
        detail = text_door.json()["detail"]
        assert detail["code"] == "approval_pending"
        assert detail["text"] == "Answer the question or approval on screen first."
        for _ in range(50):
            await asyncio.sleep(0)

        # The card survived: same request, future unresolved, nothing admitted,
        # and the tool has not been skipped.
        pending = handle._fold.projection.pending
        assert pending is not None and pending.request_id == card.request_id
        assert not gate_future.done(), "the refused utterance settled the card"
        assert _user_rows(directory) == rows_before
        transcript = (directory / "transcript.jsonl").read_text(encoding="utf-8")
        assert "interrupted by steering" not in transcript

        # Settled on screen (the /answers path), voice is accepted again.
        await handle.approval_answer(card.request_id, False, False)
        await _until(lambda: not session.is_streaming)
        await _until(lambda: remote.pending_gate is None and not remote.is_streaming)
        after = await client.post(
            "/v1/desktop/voice/utterance",
            data={"session_id": SESSION, "request_id": str(uuid.uuid4()), "text": "thanks"},
        )
        assert after.status_code == 200, after.text
        await _until(lambda: "thanks" in _user_rows(directory))
        assert gate_future.result() is False


@pytest.mark.asyncio
async def test_the_known_race_admits_a_steer_but_never_approves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V13's accepted race, pinned: the check said "no card" while a card was parked.

    The route's check and the owner's admission are separate steps, so a card
    that parks between them (milliseconds) is not seen; a typed steer has the
    same window. This forces the check to miss (the reviewer's probe) and pins
    what then happens.

    MUST NEVER CHANGE: nothing is approved and the tool does not run. The gate
    never resolves to True (today its future is cancelled: the card is withdrawn,
    not answered), and no bash process is started.

    CURRENT BEHAVIOUR, ALLOWED TO IMPROVE: the steer is admitted and the
    interruptible card is skipped ("Tool call skipped: interrupted by
    steering."). An owner-side "refuse if an approval is pending" check would
    turn this into a 409; when that lands, flip the admission and skip
    assertions below and keep the safety ones.
    """

    async def check_misses(*_: Any, **__: Any) -> bool:
        return False

    async with _parked_bash_card(tmp_path, monkeypatch) as rig:
        ran: list[Any] = []
        original = asyncio.create_subprocess_shell
        original_exec = asyncio.create_subprocess_exec

        async def spy_shell(cmd: Any, *args: Any, **kwargs: Any):
            ran.append(cmd)
            return await original(cmd, *args, **kwargs)

        async def spy_exec(program: Any, *args: Any, **kwargs: Any):
            ran.append((program, *args))
            return await original_exec(program, *args, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_shell", spy_shell)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spy_exec)
        monkeypatch.setattr(desktop_voice, "_card_pending", check_misses)

        response = await rig.client.post(
            "/v1/desktop/voice/utterance",
            data={"session_id": SESSION, "request_id": str(uuid.uuid4()), "text": "yes"},
        )
        await _until(lambda: not rig.session.is_streaming)
        transcript = (rig.directory / "transcript.jsonl").read_text(encoding="utf-8")

        # MUST NEVER CHANGE: the card settled without approval and bash never ran.
        # The steer CANCELS the gate's future (the card is withdrawn, not
        # answered); a result of True is the one outcome that would be an approval.
        assert rig.gate_future.done()
        assert rig.gate_future.cancelled() or rig.gate_future.result() is not True
        assert not any("sleep 1" in str(call) for call in ran), ran
        assert "exit code: 0" not in transcript

        # CURRENT BEHAVIOUR (allowed to improve to a 409 approval_pending).
        assert response.status_code == 200, response.text
        assert response.json()["result"]["admission"]["detail"] == "steering queued"
        assert "interrupted by steering" in transcript
        assert _user_rows(rig.directory) == ["[bash:1] list files", "yes"]
