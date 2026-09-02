"""Production loopback canonical frontend sync and isolation checks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.mobile.attach_client import AttachClient
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from tests.unit.session.runtime.test_server import FakeHandle


async def _record(root: Path):  # noqa: ANN202
    for _ in range(100):
        rows = registry.scan(root)
        if rows and rows[0][1] == "live":
            return rows[0][0]
        await asyncio.sleep(0.02)
    raise AssertionError("record did not publish")


async def _never():
    raise AssertionError("takeover was not expected")


@pytest.mark.asyncio
async def test_owner_two_followers_share_full_1m_state_and_live_updates(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    first = second = None
    try:
        record = await _record(tmp_path)
        assert FRONTEND_CAPABILITY in record.capabilities
        first, second = await asyncio.gather(
            RemoteSession.connect(record, "s1", config_dir=tmp_path, takeover_factory=_never),
            RemoteSession.connect(record, "s1", config_dir=tmp_path, takeover_factory=_never),
        )
        assert first.frontend_state == second.frontend_state
        assert first.effective_model.context_window == 1_000_000

        handle._frontend.mutate(
            context_tokens=402_000,
            cumulative_parent_cost=2.75,
            active_agent="coder",
            active_team="lopdev",
        )
        for _ in range(100):
            if first.frontend_state.sequence == 1 and second.frontend_state.sequence == 1:
                break
            await asyncio.sleep(0.02)
        assert first.frontend_state == second.frontend_state
        assert first.frontend_state.cumulative_cost == 2.75
        assert first.active_agent == "coder"
        assert first.active_team_name == "lopdev"
    finally:
        if first is not None:
            await first.dispose()
        if second is not None:
            await second.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_real_socket_follower_consumes_immutable_subagent_progress_and_settlement(
    tmp_path: Path, monkeypatch
) -> None:
    """RuntimeServer → RemoteSession keeps the follower's full child ledger usable."""
    import time

    from local_operator.session.frontend_state import JobState
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.subagent_panel import SubagentPanel
    from local_operator.tui.widgets.subagent_view import LEDGER_GONE_NOTE, SubagentView
    from tests.unit.tui.test_reconnect_parity import _boot, _remote_factory

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    handle = FakeHandle()
    started = time.time() - 2
    running = JobState(
        id="child",
        type="task",
        label="streaming child",
        status="running",
        start_time=started,
        started_at=started,
        latest_details={"progress": "reading files"},
        trajectory=[
            {"type": "message_start", "message": {"role": "assistant", "id": "m1"}},
            {
                "type": "message_update",
                "message": {"role": "assistant", "id": "m1"},
                "delta": "Inspecting the follower path.",
            },
        ],
    )
    handle._frontend.mutate(jobs=[running])
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    remote = None
    try:
        record = await _record(tmp_path)
        remote = await RemoteSession.connect(
            record, "s1", config_dir=tmp_path, takeover_factory=_never
        )
        app = OperatorApp(_remote_factory(remote))
        async with app.run_test(size=(100, 30)) as pilot:
            await _boot(app, pilot)
            panel = app.query_one(SubagentPanel)
            app._refresh_band()
            panel._tick()
            await pilot.pause()
            assert "reading files" in str(getattr(panel._rows["child"], "content", ""))

            app._open_subagent_view("child")
            for _ in range(20):
                await pilot.pause()
                if "Inspecting the follower path." in " ".join(
                    app.query_one(SubagentView).rendered_rows()
                ):
                    break
            view = app.query_one(SubagentView)
            assert "Inspecting the follower path." in " ".join(view.rendered_rows())

            completed = running.model_copy(
                update={
                    "status": "completed",
                    "settled_at": time.time(),
                    "result_text": "follower path complete",
                    "latest_details": {"progress": "thinking"},
                }
            )
            handle._frontend.mutate(jobs=[completed])
            for _ in range(100):
                await pilot.pause()
                row = remote.jobs.get("child")
                if row is not None and row.status == "completed":
                    break
            assert remote.jobs.get("child") is not None
            assert remote.jobs.get("child").status == "completed"  # type: ignore[union-attr]
            assert "completed" in view.rendered_rows()[0]

            await pilot.press("escape")
            await pilot.pause()
            app._open_subagent_view("child")
            for _ in range(10):
                await pilot.pause()
            reopened = app.query_one(SubagentView)
            rendered = " ".join(reopened.rendered_rows())
            assert LEDGER_GONE_NOTE not in rendered
            assert "completed" in rendered
            assert remote.jobs.get("child") is not None
    finally:
        if remote is not None:
            await remote.dispose()
        registrant.close()


@pytest.mark.asyncio
async def test_daemon_never_receives_frontend_frames(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    writer = None
    try:
        record = await _record(tmp_path)
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(json.dumps({"key": record.control_key}).encode() + b"\n")
        await writer.drain()
        welcome = json.loads((await asyncio.wait_for(reader.readline(), 3)).decode())
        assert welcome["op"] == "projection"
        handle._frontend.mutate(context_tokens=99)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(reader.readline(), 0.15)
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        registrant.close()


@pytest.mark.asyncio
async def test_wrong_key_wrong_session_and_old_protocol_are_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    handle = FakeHandle()
    registrant = RuntimeServer(handle, kind="tui")
    registrant.start()
    try:
        record = await _record(tmp_path)
        reader, writer = await asyncio.open_connection("127.0.0.1", record.control_port)
        writer.write(b'{"key":"wrong","client":"attach","frontend_state":true}\n')
        await writer.drain()
        assert await asyncio.wait_for(reader.readline(), 3) == b""
        writer.close()
        await writer.wait_closed()

        client = AttachClient(lambda state: None, lambda reason: None, frontend_state=True)
        with pytest.raises(ConnectionError, match="another conversation"):
            await client.connect(record, "wrong-session")

        old = record.__class__(**{**record.to_json(), "protocol": 4, "capabilities": []})
        with pytest.raises(ConnectionError, match="lacks tui_state_v1"):
            await RemoteSession.connect(old, "s1", config_dir=tmp_path, takeover_factory=_never)
    finally:
        registrant.close()


@pytest.mark.asyncio
async def test_headless_turn_preserves_a_rich_frontend_checkpoint(tmp_path: Path) -> None:
    """N1: a scheduler/owned turn must never lower the TUI's durable state.

    The headless store starts from the durable checkpoint (never bare), and a
    headless session with no attach subscriber skips the turn-end checkpoint
    entirely — so the richest persisted spend/duration/title survives for the
    next TUI resume/takeover instead of being clobbered by a bare snapshot.
    """
    from local_operator.harness.types import (
        Message,
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
        TextContent,
    )
    from local_operator.session.frontend_state import (
        FRONTEND_CHECKPOINT_CUSTOM_TYPE,
        FrontendSessionState,
        FrontendStateStore,
    )
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="prior")], stop_reason="stop")
    )
    rich = FrontendSessionState(
        session_id="conv",
        epoch="tui-epoch",
        conversation_title="Real title",
        conversation_title_user_set=True,
        cumulative_parent_cost=12.34,
        active_duration_s=300.0,
    )
    await FrontendStateStore(rich).checkpoint(transcript)

    class _Stream:
        def __call__(self, request, signal=None):  # noqa: ANN001
            async def gen():
                yield StreamTextDelta(delta="ok")
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=100_000),
        stream_fn=_Stream(),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["system"],
        has_ui=False,
    )
    try:
        # The headless store itself restored the rich durable state.
        assert session.frontend_state.cumulative_parent_cost == 12.34
        assert session.frontend_state.conversation_title == "Real title"
        await session.prompt("do the scheduled thing")
    finally:
        await session.dispose()

    restored = Transcript(directory).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert isinstance(restored, dict)
    state = FrontendSessionState.model_validate(restored["state"])
    assert state.cumulative_parent_cost == 12.34
    assert state.active_duration_s == 300.0
    assert state.conversation_title == "Real title"


@pytest.mark.asyncio
async def test_real_session_streams_through_registrant_to_two_followers(
    tmp_path: Path, monkeypatch
) -> None:
    """Flagship realism: real Session → real RuntimeServer → two socket followers.

    The canonical seed/updates come from the SESSION's own store (not a test
    double), the turn is a real streamed prompt, and both followers must end
    at identical state with a bounded sequence count — sustained streaming may
    not consume a sequence per token (N4) nor grow follower job state without
    bound (N2).
    """
    from local_operator.harness.types import (
        ModelSpec,
        StreamEndEvent,
        StreamTextDelta,
        Usage,
    )
    from local_operator.mobile.types import SessionProjection
    from local_operator.session.runtime.server import SessionHandle
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    class _Stream:
        def __call__(self, request, signal=None):  # noqa: ANN001
            async def gen():
                for index in range(120):
                    yield StreamTextDelta(delta=f"tok{index} ")
                yield StreamEndEvent(
                    stop_reason="stop",
                    usage=Usage(input_tokens=1_000, output_tokens=120, context_tokens=2_000),
                )

            return gen()

    session = Session(
        model=ModelSpec(provider="test", model_id="m", context_window=1_000_000),
        stream_fn=_Stream(),
        tools=[],
        transcript=Transcript(tmp_path / "sessions" / "conv"),
        system_blocks_provider=lambda: ["system"],
        has_ui=True,
        session_id="conv",
    )

    class _RealSessionHandle(SessionHandle):
        """Production contract implemented directly over the real Session."""

        @property
        def session_projection_seed(self) -> SessionProjection:
            return SessionProjection(
                session_id=session.session_id,
                pid=0,
                kind="tui",
                conversation_name="real",
                cwd=str(tmp_path),
                model_label="test/m",
            )

        def subscribe(self, on_projection):  # noqa: ANN001, ANN202
            return lambda: None

        @property
        def frontend_state_seed(self):  # noqa: ANN202
            return session.frontend_state

        def subscribe_frontend(self, on_update):  # noqa: ANN001, ANN202
            return session.subscribe_frontend(on_update)

        def subscribe_events(self, on_event):  # noqa: ANN001, ANN202
            return session.subscribe(lambda event: on_event(event.model_dump(mode="json")))

        async def prompt(self, text, images=None, command_id=None):  # noqa: ANN001, ANN202
            await session.prompt(text)
            return "ok"

        async def steer(self, text, images=None):  # noqa: ANN001, ANN202
            return "ok"

        async def abort(self):  # noqa: ANN202
            return "ok"

        async def set_model(self, provider, model_id):  # noqa: ANN001, ANN202
            return "ok"

        async def set_effort(self, effort):  # noqa: ANN001, ANN202
            return "ok"

        async def slash(self, command, args):  # noqa: ANN001, ANN202
            return "ok"

        async def new_conversation(self):  # noqa: ANN202
            return "ok"

        async def resume_session(self, session_id):  # noqa: ANN001, ANN202
            return "ok"

        async def approval_answer(self, request_id, approved, remember):  # noqa: ANN001, ANN202
            return "ok"

        async def ask_answer(self, request_id, value, question_index=None):  # noqa: ANN001, ANN202
            return "ok"

        async def refresh(self) -> None:
            return None

    registrant = RuntimeServer(_RealSessionHandle(), kind="tui")
    registrant.start()
    first = second = None
    try:
        record = await _record(tmp_path)
        first, second = await asyncio.gather(
            RemoteSession.connect(record, "conv", config_dir=tmp_path, takeover_factory=_never),
            RemoteSession.connect(record, "conv", config_dir=tmp_path, takeover_factory=_never),
        )
        events: list[str] = []
        first.subscribe(lambda event: events.append(event.type))

        await session.prompt("stream a real turn")

        for _ in range(200):
            if (
                not first.frontend_state.streaming
                and not second.frontend_state.streaming
                and first.frontend_state.sequence == second.frontend_state.sequence
                and first.frontend_state.sequence > 0
                and "agent_end" in events
            ):
                break
            await asyncio.sleep(0.02)
        assert "agent_end" in events, "raw event stream did not deliver the turn"
        assert first.frontend_state == second.frontend_state
        assert first.frontend_state.context_tokens == 2_000
        # 120 token deltas must not consume 120 sequences: canonical state moves
        # on turn edges/summaries, never per token.
        assert first.frontend_state.sequence < 20
    finally:
        if first is not None:
            await first.dispose()
        if second is not None:
            await second.dispose()
        registrant.close()
        await session.dispose()
