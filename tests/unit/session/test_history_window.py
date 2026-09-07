"""Canonical display pages never substitute a truncated context for history."""

import asyncio
import json
from pathlib import Path

import pytest

from local_operator.harness.types import (
    CustomMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.session.history_window import display_window
from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session, seed_transcript, text_turn
from tests.unit.session.test_remote import _never_take_over


def window(transcript: Transcript, **kwargs):  # noqa: ANN003, ANN201
    return display_window(
        transcript,
        conversation_id="window-test",
        owner_epoch="synthetic-epoch",
        through_id=transcript.entries()[-1].id if transcript.entries() else None,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_pages_match_canonical_cut_across_appends_and_reset_on_prune(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path)
    messages = [Message.user(f"row {index}") for index in range(270)]
    await transcript.append_messages(messages)
    first = window(transcript)
    assert first.total_message_count == 270
    assert len(first.messages) == 120
    assert first.start == 150
    assert first.before_token and first.snapshot_token
    await transcript.append_message(Message.assistant("after captured cut"))
    rows = list(first.messages)
    token = first.before_token
    while token:
        page = window(transcript, before=token)
        assert page.through_id == first.through_id
        rows[:0] = page.messages
        token = page.before_token
    assert [m.model_dump() for m in rows] == [m.model_dump() for m in messages]
    await transcript.append_prune(messages[-1].id, "pruned")
    assert window(transcript, before=first.before_token).status == "reset"
    last = transcript.build_llm_history(through_id=first.through_id)[-1]
    assert isinstance(last, Message) and last.text == "row 269"


@pytest.mark.asyncio
async def test_compaction_roles_and_delayed_tool_results_share_canonical_replay(
    tmp_path: Path,
) -> None:
    transcript = Transcript(tmp_path)
    opener = Message.user("preserved verbatim")
    call = Message.assistant("running a tool")
    call.tool_calls = [ToolCall(id="tool-one", name="bash", arguments={"command": "echo hello"})]
    custom = CustomMessage(custom_type="aside", attribution="user", details={"text": "context"})
    result = Message.tool_result(
        ToolResult(tool_call_id="tool-one", tool_name="bash", content=[TextContent(text="hello")])
    )
    await transcript.append_messages([opener, call, custom, result])
    compact = await transcript.append_compaction(
        "summary", call.id, 500, preserved_user_turns=[{"id": opener.id, "text": opener.text}]
    )
    canonical = transcript.build_llm_history()
    assert canonical[0].id == compact.id
    assert canonical[1].id == opener.id
    first = window(transcript, max_messages=3)
    assert [m.id for m in first.messages] == [call.id, custom.id, result.id]
    assert first.before_token
    previous = window(transcript, before=first.before_token, max_messages=3)
    assert [m.model_dump() for m in previous.messages + first.messages] == [
        m.model_dump() for m in canonical
    ]
    anchored = window(transcript, before=first.snapshot_token, anchor="tool:tool-one")
    assert any(m.id == call.id for m in anchored.messages)


@pytest.mark.asyncio
async def test_wire_budget_never_returns_oversized_or_truncated_required_prose(
    tmp_path: Path,
) -> None:
    transcript = Transcript(tmp_path)
    message = Message.assistant("required prose " * 100_000)
    await transcript.append_message(message)
    page = window(transcript)
    assert page.status == "full_required"
    assert page.messages == []
    assert len(page.model_dump_json().encode()) < 1024 * 1024
    replayed = transcript.build_llm_history()[0]
    assert isinstance(replayed, Message) and replayed.text == message.text


@pytest.mark.asyncio
async def test_signed_page_scope_and_anchor_validation(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path)
    await transcript.append_messages([Message.user(str(i)) for i in range(130)])
    first = window(transcript)
    assert first.before_token
    with pytest.raises(ValueError, match="invalid history token"):
        window(transcript, before=first.before_token + "bad")
    with pytest.raises(ValueError, match="another conversation"):
        display_window(
            transcript,
            conversation_id="wrong",
            owner_epoch="synthetic-epoch",
            through_id=first.through_id,
            before=first.before_token,
        )
    assert (
        display_window(
            transcript,
            conversation_id="window-test",
            owner_epoch="new-epoch",
            through_id=first.through_id,
            before=first.before_token,
        ).status
        == "reset"
    )
    assert window(transcript, before=first.snapshot_token, anchor="missing-id").status == "reset"


@pytest.mark.asyncio
async def test_real_attach_pages_without_viewer_journal_parse_and_records_shell_once(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "window-test"
    messages = [Message.user(f"canonical {index}") for index in range(250)]
    await seed_transcript(directory, messages)
    session = build_session(directory, ScriptedStream([text_turn("owner reply")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    remote = None
    await server.start_in_process()
    try:

        async def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise AssertionError("window attach must not parse the viewer journal")

        monkeypatch.setattr(RemoteSession, "_read_transcript", forbidden)
        remote = await RemoteSession.connect(
            server._record,
            "window-test",
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )
        assert not remote.is_cold
        assert remote.history_message_count == 250
        assert len(remote.display_history_window()) == 120
        with pytest.raises(RuntimeError, match="not hydrated"):
            remote.history()
        await remote.seed_history([Message.user("must not seed a nonempty window")])
        assert remote.history_message_count == 250
        old_token = remote.history_before_token
        assert old_token is not None
        await remote.ensure_display_anchor(messages[10].id)
        assert any(m.id == messages[10].id for m in remote.display_history_window())
        assert [m.id for m in await remote.materialize_history()] == [m.id for m in messages]
        result = ToolResult(
            tool_call_id="synthetic-shell",
            tool_name="bash",
            content=[TextContent(text="exit code: 0\nhello")],
        )
        await remote.record_shell("echo hello", result)
        await remote.record_shell("echo hello", result)
        durable = session._transcript.build_llm_history()
        assert sum(m.id == "shell:synthetic-shell:user" for m in durable) == 1
        assert sum(m.id == "shell:synthetic-shell:result" for m in durable) == 1
        assert (
            sum(m.id == "shell:synthetic-shell:user" for m in remote.display_history_window()) == 1
        )
        assert json.loads(remote.frontend_state.model_dump_json())["session_id"] == "window-test"
        received = []
        remote.subscribe(received.append)
        await session._transcript.append_compaction("new canonical cut", messages[-1].id, 500)
        stale = await remote.history_page(old_token)
        assert stale.status == "reset"
        await remote._refresh_display_history()
        assert not remote.is_cold
        assert any(getattr(event, "reset", False) for event in received)
        marker = remote.display_history_window()[0]
        assert isinstance(marker, CustomMessage) and marker.custom_type == "compaction_summary"
        assert not await remote.ensure_display_anchor(messages[0].id)
    finally:
        if remote is not None:
            await remote.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_live_replay_mutations_refresh_without_replacing_the_connection(
    tmp_path, monkeypatch
):
    from local_operator.harness.types import CompactionEndEvent, NoticeEvent

    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "window-live"
    messages = [Message.user(f"canonical {index}") for index in range(250)]
    await seed_transcript(directory, messages)
    session = build_session(directory, ScriptedStream([text_turn("unused")]), cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    assert server._record is not None
    remote = None
    try:
        remote = await RemoteSession.connect(
            server._record,
            "window-live",
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )
        connection = remote._client
        received = asyncio.Event()
        remote.subscribe(
            lambda event: received.set() if isinstance(event, CompactionEndEvent) else None
        )
        old_token = remote.history_before_token
        await session._transcript.append_compaction("canonical summary", messages[-1].id, 1000)
        await session._emit(CompactionEndEvent(reason="manual", success=True))
        await asyncio.wait_for(received.wait(), 3)
        await remote.ensure_display_current()
        assert remote._client is connection
        assert remote.display_history_current
        assert remote.history_message_count == 2
        assert remote._display_history is not None
        assert remote._display_history.history_generation == session._transcript._history_generation
        assert any(
            isinstance(row, CustomMessage) and row.custom_type == "compaction_summary"
            for row in remote.display_history_window()
        )
        assert old_token is not None and (await remote.history_page(old_token)).status == "reset"
        # Non-compaction mutations publish the same generation fence before
        # ordinary events; callers need not page an obsolete token to find out.
        await session._transcript.append_prune(messages[-1].id, "pruned")
        await session._emit(NoticeEvent(text="pruned", kind="info"))
        for _ in range(100):
            if remote.frontend_state.history_generation == session._transcript._history_generation:
                break
            await asyncio.sleep(0.01)
        await remote.ensure_display_current()
        assert remote._client is connection
        assert [row.model_dump() for row in remote.display_history_window()] == [
            row.model_dump() for row in session._transcript.build_llm_history()
        ]
        assert remote._display_history.history_generation == session._transcript._history_generation
        assert len(server._clients) == 1
    finally:
        if remote is not None:
            await remote.dispose()
        server.close()
        await handle.dispose()


@pytest.mark.asyncio
async def test_prompt_and_wait_does_not_complete_on_admission(tmp_path, monkeypatch):
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    directory = config / "sessions" / "window-loop"
    await seed_transcript(directory, [Message.user("initial")])
    started, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def stream(request, signal=None):
        # The session also issues a tool-less naming errand off the first
        # prompt. Only a real agent turn carries the tool schema, and counting
        # the errand would make "how many turns ran" unreadable here.
        if not getattr(request, "tools", None):
            for event in text_turn("named"):
                yield event
            return
        calls.append(len(calls) + 1)
        started.set()
        await release.wait()
        for event in text_turn("completed"):
            yield event

    session = build_session(directory, stream, cwd=tmp_path)
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()
    assert server._record is not None
    remote = None
    task = None
    try:
        remote = await RemoteSession.connect(
            server._record,
            "window-loop",
            config_dir=config,
            takeover_factory=_never_take_over,
            display_window=True,
        )
        task = asyncio.create_task(
            remote.prompt_and_wait("one", message_id="11111111-1111-4111-8111-111111111111")
        )

        async def provider_started():
            while not started.is_set():
                if task.done():
                    await task
                    raise AssertionError("turn completed before the provider started")
                await asyncio.sleep(0.01)

        await asyncio.wait_for(provider_started(), 3)
        await asyncio.sleep(0.05)
        assert not task.done(), "durable admission is not terminal completion"
        assert calls == [1]
        # A refresh can pause UI event delivery. The scheduler still observes
        # the authenticated terminal outcome, not whether a card was painted.
        remote._ready_for_events = False
        release.set()
        await asyncio.wait_for(task, 3)
        remote._ready_for_events = True
        await asyncio.wait_for(
            remote.prompt_and_wait("two", message_id="22222222-2222-4222-8222-222222222222"), 3
        )
        assert calls == [1, 2]
        assert not remote._prompt_completion_waiters
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if remote is not None:
            await remote.dispose()
        server.close()
        await handle.dispose()
