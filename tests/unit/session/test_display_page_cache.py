"""Exact display requests are detached and bounded; misses keep canonical replay."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from local_operator.harness.types import (
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolExecutionEndEvent,
    ToolResult,
)
from local_operator.session.history_window import (
    DISPLAY_PAGE_CACHE_BYTES,
    DISPLAY_PAGE_CACHE_ENTRIES,
    display_window,
)
from local_operator.session.transcript import Transcript
from tests.e2e.harness import ScriptedStream, build_session


@pytest.fixture(autouse=True)
def isolated_owner(tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def page(transcript: Transcript, **kwargs):
    return display_window(
        transcript,
        conversation_id="synthetic-cache",
        owner_epoch="synthetic-epoch",
        through_id=transcript.entries()[-1].id if transcript.entries() else None,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_repeated_tail_older_anchor_and_limits_do_not_replay(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "session")
    messages = [Message.user(f"row {index}") for index in range(270)]
    await transcript.append_messages(messages)
    first = page(transcript)
    variants = [
        {},
        {"before": first.before_token},
        {"before": first.snapshot_token, "anchor": messages[0].id},
        {"max_messages": 4},
    ]
    for kwargs in variants:
        expected = page(transcript, **kwargs).model_dump()
        with patch.object(
            transcript, "build_llm_history", wraps=transcript.build_llm_history
        ) as replay:
            actual = page(transcript, **kwargs)
            assert actual.model_dump() == expected
            replay.assert_not_called()
        message = actual.messages[0]
        assert isinstance(message, Message)
        content = message.content[0]
        assert isinstance(content, TextContent)
        content.text = "caller mutation"
        actual.messages.clear()
        actual.durable_seed_ids.append("caller seed")
        assert page(transcript, **kwargs).model_dump() == expected
    # Model/context consumers still own independent replay objects.
    message = transcript.build_llm_history()[0]
    assert isinstance(message, Message)
    content = message.content[0]
    assert isinstance(content, TextContent)
    content.text = "model mutation"
    fresh = transcript.build_llm_history()[0]
    assert isinstance(fresh, Message)
    assert fresh.text == "row 0"
    # A cached valid token must not outlive its signing authority, even when
    # conversation/epoch/cut are unchanged.
    page(transcript, before=first.snapshot_token)
    transcript._history_page_key = b"replacement-test-signing-key"
    with pytest.raises(ValueError, match="invalid history token"):
        page(transcript, before=first.snapshot_token)


@pytest.mark.asyncio
@pytest.mark.parametrize("retain", [True, False])
async def test_cold_nested_display_payload_cannot_mutate_journal(
    tmp_path: Path, monkeypatch, retain
):
    if not retain:
        monkeypatch.setattr("local_operator.session.history_window.DISPLAY_PAGE_CACHE_BYTES", 1)
    transcript = Transcript(tmp_path / "session")
    await transcript.append_message(
        Message.assistant(
            "call",
            tool_calls=[ToolCall(name="bash", arguments={"nested": {"values": [1]}})],
            provider_payload={"details": {"tags": ["original"]}},
        )
    )
    first = page(transcript).messages[0]
    assert isinstance(first, Message)
    first.tool_calls[0].arguments["nested"]["values"].append(2)
    assert first.provider_payload is not None
    first.provider_payload["details"]["tags"].append("changed")
    for fresh in (transcript.build_llm_history()[0], page(transcript).messages[0]):
        assert isinstance(fresh, Message)
        assert fresh.tool_calls[0].arguments["nested"]["values"] == [1]
        assert fresh.provider_payload == {"details": {"tags": ["original"]}}
    assert transcript._display_window_cache is not None
    assert bool(transcript._display_window_cache.entries) is retain


@pytest.mark.asyncio
async def test_old_cut_hits_after_append_and_regenerates_after_eviction(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "session")
    await transcript.append_messages([Message.user(str(i)) for i in range(140)])
    first = page(transcript)
    previous = page(transcript, before=first.before_token)
    await transcript.append_message(Message.assistant("after old cut"))
    with patch.object(
        transcript, "build_llm_history", wraps=transcript.build_llm_history
    ) as replay:
        assert page(transcript, before=first.before_token).model_dump() == previous.model_dump()
        replay.assert_not_called()
    for i in range(200):
        await transcript.append_message(Message.user(f"new cut {i}"))
        page(transcript)
        cache = transcript._display_window_cache
        assert cache is not None
        assert len(cache.entries) <= DISPLAY_PAGE_CACHE_ENTRIES
        assert cache.retained_bytes + 4096 <= DISPLAY_PAGE_CACHE_BYTES
    with patch.object(
        transcript, "build_llm_history", wraps=transcript.build_llm_history
    ) as replay:
        regenerated = page(transcript, before=first.before_token)
        assert regenerated.model_dump() == previous.model_dump()
        replay.assert_called_once()


@pytest.mark.asyncio
async def test_seed_tools_share_replay_cut_and_are_part_of_request_key(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "session")
    call = Message.assistant(
        "call", tool_calls=[ToolCall(id="tool-one", name="bash", arguments={})]
    )
    result = Message.tool_result(
        ToolResult(tool_call_id="tool-one", content=[TextContent(text="done")])
    )
    await transcript.append_messages([Message.user("start"), call, result])
    with patch.object(
        transcript, "build_llm_history", wraps=transcript.build_llm_history
    ) as replay:
        seeded = page(transcript, durable_seed_tools=frozenset({"tool-one", "not-durable"}))
        assert seeded.durable_seed_tool_ids == ["tool-one"]
        replay.assert_called_once()
    with patch.object(
        transcript, "build_llm_history", wraps=transcript.build_llm_history
    ) as replay:
        assert (
            page(transcript, durable_seed_tools=frozenset({"not-durable", "tool-one"})).model_dump()
            == seeded.model_dump()
        )
        replay.assert_not_called()
    assert page(transcript).durable_seed_tool_ids == []
    # Full-required fallback must still get its seed from that same replay.
    with patch.object(
        transcript, "build_llm_history", wraps=transcript.build_llm_history
    ) as replay:
        too_small = page(transcript, durable_seed_tools=frozenset({"tool-one"}), max_messages=1)
        assert too_small.status == "full_required"
        assert too_small.durable_seed_tool_ids == ["tool-one"]
        replay.assert_called_once()


@pytest.mark.asyncio
async def test_actual_session_subscribe_replays_once_then_reuses_tool_seed_cut(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "config/sessions/aaaaaaaaaaaa"
    transcript = Transcript(directory)
    result = ToolResult(
        tool_call_id="durable-tool", tool_name="bash", content=[TextContent(text="done")]
    )
    await transcript.append_messages(
        [
            Message.user("start"),
            Message.assistant(
                "calling", tool_calls=[ToolCall(id="durable-tool", name="bash", arguments={})]
            ),
            Message.tool_result(result),
        ]
    )
    session = build_session(directory, ScriptedStream([]), cwd=tmp_path)
    try:
        for tool_result in (result, ToolResult(tool_call_id="not-durable", tool_name="bash")):
            session._frontend_state_store._fold_live_event(
                ToolExecutionEndEvent(
                    tool_call_id=tool_result.tool_call_id, tool_name="bash", result=tool_result
                )
            )
        with patch.object(
            session._transcript, "build_llm_history", wraps=session._transcript.build_llm_history
        ) as replay:
            subscription = session.subscribe_frontend(lambda _: None, display_window=True)
            try:
                window = subscription.sync.display_history
                assert window is not None
                assert window.through_id == subscription.sync.live_cursor
                assert window.durable_seed_tool_ids == ["durable-tool"]
                replay.assert_called_once()
            finally:
                subscription.unsubscribe()
        with patch.object(
            session._transcript, "build_llm_history", wraps=session._transcript.build_llm_history
        ) as replay:
            subscription = session.subscribe_frontend(lambda _: None, display_window=True)
            try:
                assert subscription.sync.display_history == window
                replay.assert_not_called()
            finally:
                subscription.unsubscribe()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_prune_compaction_and_fold_discard_cache_and_preserve_token_rules(
    tmp_path: Path,
) -> None:
    transcript = Transcript(tmp_path / "session")
    first = Message.user("preserved opener")
    tool = Message.tool_result(
        ToolResult(tool_call_id="one", content=[TextContent(text="large " * 500)])
    )
    last = Message.user("last")
    await transcript.append_messages([first, tool, last])
    old = page(transcript, max_messages=1)
    await transcript.append_prune(tool.id, "pruned")
    assert transcript._display_window_cache is None
    assert page(transcript, before=old.before_token).status == "reset"
    after = page(transcript)
    assert [m.model_dump() for m in after.messages] == [
        m.model_dump() for m in transcript.build_llm_history()
    ]
    await transcript.append_compaction(
        "summary", last.id, 200, preserved_user_turns=[{"id": first.id, "text": first.text}]
    )
    assert transcript._display_window_cache is None
    compacted = page(transcript)
    assert [m.model_dump() for m in compacted.messages] == [
        m.model_dump() for m in transcript.build_llm_history()
    ]
    await transcript.compact_file(min_reclaim_bytes=0)
    assert transcript._display_window_cache is None
    assert [m.model_dump() for m in page(transcript).messages] == [
        m.model_dump() for m in transcript.build_llm_history()
    ]
    assert page(transcript, before=compacted.snapshot_token).status == "reset"


@pytest.mark.asyncio
async def test_media_and_metadata_admission_stay_within_owner_budget(tmp_path: Path) -> None:
    transcript = Transcript(tmp_path / "session")
    image = ImageContent(data=base64.b64encode(b"synthetic" * 60_000).decode())
    for i in range(8):
        await transcript.append_message(Message.user(str(i), images=[image]))
        result = page(transcript, max_messages=1, max_wire_bytes=2 * 1024 * 1024)
        assert result.status == "ok"
        message = result.messages[0]
        assert isinstance(message, Message)
        content = message.content[1]
        assert isinstance(content, ImageContent)
        assert content.data == image.data
        cache = transcript._display_window_cache
        assert cache is not None
        assert cache.retained_bytes + 4096 <= DISPLAY_PAGE_CACHE_BYTES
        assert len(cache.entries) < DISPLAY_PAGE_CACHE_ENTRIES
    # A request key can itself be huge even when the resulting page is tiny.
    # Include seed identities in admission, not merely message body bytes.
    empty = Transcript(tmp_path / "empty")
    page(empty, durable_seed_tools=frozenset({"x" * DISPLAY_PAGE_CACHE_BYTES}))
    assert empty._display_window_cache is not None
    assert not empty._display_window_cache.entries
