"""Fork copies share the writer's boundary, including cancellation and rewrites."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from local_operator.harness.types import Message, MessageRole, TextContent, ToolCall
from local_operator.session.session import _paired_prefix
from local_operator.session.transcript import Transcript
from local_operator.spawn.policy import fork_mode, parse_fork_args


def message(role: MessageRole, text: str, **kwargs) -> Message:
    return Message(role=role, content=[TextContent(text=text)], **kwargs)


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        ("", (None, "")),
        ("try 'the other' parser", (None, "try 'the other' parser")),
        ("--window try --switch later", ("window", "try --switch later")),
        ("--switch\ttry it", ("switch", "try it")),
        ("-- --window is literal", (None, "--window is literal")),
        ("--window -- --switch", ("window", "--switch")),
    ],
)
def test_destination_flags_preserve_prompt_text(arg, expected) -> None:
    assert parse_fork_args(arg) == expected


@pytest.mark.parametrize("arg", ["--windwo", "--window --switch", "--switch --window"])
def test_invalid_destination_never_becomes_model_input(arg) -> None:
    with pytest.raises(ValueError):
        parse_fork_args(arg)


def test_default_switch_honors_explicit_window() -> None:
    assert fork_mode(None) == fork_mode({"fork": {"mode": "typo"}}) == "switch"
    assert fork_mode({"fork": {"mode": "window"}}) == "window"


@pytest.mark.asyncio
async def test_snapshot_omits_only_incomplete_suffix_and_preserves_raw_rows(tmp_path: Path) -> None:
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    question = message("user", "hello")
    call = message(
        "assistant",
        "",
        tool_calls=[
            ToolCall(id="a", name="bash", arguments={}),
            ToolCall(id="b", name="bash", arguments={}),
        ],
    )
    partial = message("tool", "first finished", tool_call_id="a")
    await parent.append_messages([question, call, partial])
    journal = await parent.append_custom("note", {"message": "keep the journal"})
    before = parent.path.read_bytes()
    fork_id, omitted = await parent.fork_snapshot(message="try another route")
    assert omitted
    fork = Transcript(tmp_path / "sessions" / fork_id)
    assert [m.id for m in fork.build_llm_history()] == [question.id]
    expected = b"".join(
        line
        for line in before.splitlines(keepends=True)
        if call.id.encode() not in line and partial.id.encode() not in line
    )
    assert fork.path.read_bytes() == expected
    assert journal.id in fork.path.read_text()
    assert parent.path.read_bytes() == before
    assert _paired_prefix([question, call, partial]) == [question]
    await parent.append_message(message("tool", "second finished", tool_call_id="b"))
    complete_id, omitted = await parent.fork_snapshot()
    assert not omitted
    assert (
        tmp_path / "sessions" / complete_id / "transcript.jsonl"
    ).read_bytes() == parent.path.read_bytes()


@pytest.mark.asyncio
async def test_snapshot_refuses_removing_compaction_anchor(tmp_path: Path) -> None:
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    call = message(
        "assistant",
        "",
        tool_calls=[
            ToolCall(id="a", name="bash", arguments={}),
            ToolCall(id="b", name="bash", arguments={}),
        ],
    )
    await parent.append_messages(
        [
            message("user", "OLD SUMMARIZED CONTENT"),
            call,
            message("tool", "first finished", tool_call_id="a"),
        ]
    )
    await parent.append_compaction(
        summary="summary",
        first_kept_entry_id=call.id,
        tokens_before=100,
    )
    before = parent.path.read_bytes()
    with pytest.raises(ValueError, match="compaction boundary.*unfinished tool batch"):
        await parent.fork_snapshot()
    assert parent.path.read_bytes() == before
    assert len(list(parent.directory.parent.iterdir())) == 1
    # Refusal is temporary, not a damaged-history dead end: the original's
    # missing result completes the same anchored batch without any repair.
    await parent.append_message(message("tool", "second finished", tool_call_id="b"))
    fork_id, omitted = await parent.fork_snapshot()
    assert not omitted
    fork = Transcript(parent.directory.parent / fork_id)
    assert fork.path.read_bytes() == parent.path.read_bytes()
    assert "OLD SUMMARIZED CONTENT" not in str(fork.build_llm_history())


@pytest.mark.asyncio
async def test_snapshot_refuses_malformed_interior_and_active_compaction(tmp_path: Path) -> None:
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    await parent.append_messages(
        [
            message("assistant", "", tool_calls=[ToolCall(id="a", name="bash", arguments={})]),
            message("user", "interleaved input"),
        ]
    )
    with pytest.raises(ValueError, match="incomplete tool calls before later"):
        await parent.fork_snapshot()
    with pytest.raises(ValueError, match="history is being rewritten"):
        await parent.fork_snapshot(is_compacting=lambda: True)
    assert len(list((tmp_path / "sessions").iterdir())) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_copy_holds_writer_lock_until_worker_settles(
    tmp_path: Path, monkeypatch, cancel: bool
) -> None:
    """An event-controlled syscall, not a sleep, proves append cannot overtake copy."""
    import local_operator.fork as fork_module

    parent = Transcript(tmp_path / "sessions" / "parent000001")
    await parent.append_message(message("user", "committed"))
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = fork_module.fork_session
    replay = parent.build_llm_history
    loop_thread = threading.get_ident()
    replay_threads = []

    def observed_replay():
        replay_threads.append(threading.get_ident())
        return replay()

    monkeypatch.setattr(parent, "build_llm_history", observed_replay)

    def blocked(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(20), "test failed to release its copy worker"
        return original(*args, **kwargs)

    monkeypatch.setattr(fork_module, "fork_session", blocked)
    task = asyncio.create_task(parent.fork_snapshot())
    try:
        await asyncio.wait_for(entered.wait(), 20)
        append = asyncio.create_task(parent.append_message(message("assistant", "later")))
        compact = asyncio.create_task(parent.compact_file(min_reclaim_bytes=0))
        if cancel:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        await asyncio.sleep(0)
        assert parent._lock.locked()
        assert replay_threads and all(thread != loop_thread for thread in replay_threads)
        assert not append.done()
        assert not compact.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            await task
        await append
        await compact
        forks = [p for p in (tmp_path / "sessions").iterdir() if p.name != "parent000001"]
        assert len(forks) == 1
        assert "later" not in (forks[0] / "transcript.jsonl").read_text()
        assert "later" in parent.path.read_text()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_snapshot_copy_error_releases_lock_and_preserves_parent(
    tmp_path: Path, monkeypatch
) -> None:
    import local_operator.fork as fork_module

    parent = Transcript(tmp_path / "sessions" / "parent000001")
    await parent.append_message(message("user", "committed"))
    before = parent.path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(fork_module, "fork_session", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        await parent.fork_snapshot()
    assert not parent._lock.locked()
    assert parent.path.read_bytes() == before
