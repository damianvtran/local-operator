"""Fork copies share the writer's boundary, including cancellation and rewrites."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from local_operator.harness.types import Message, MessageRole, TextContent, ToolCall
from local_operator.session.errors import ForkRefused
from local_operator.session.session import _paired_prefix
from local_operator.session.transcript import ENTRY_COMPACTION, Transcript
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
    fork_id, omitted, _ = await parent.fork_snapshot(message="try another route")
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
    complete_id, omitted, _ = await parent.fork_snapshot()
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
    with pytest.raises(ForkRefused, match="compaction boundary.*unfinished tool batch") as refused:
        await parent.fork_snapshot()
    # The CAUSE, not only the words: the desktop route picks the sentence it
    # renders from this token, so the right sentence with no reason would still
    # mis-render on the wire.
    assert refused.value.reason == "unfinished_batch"
    assert parent.path.read_bytes() == before
    assert len(list(parent.directory.parent.iterdir())) == 1
    # Refusal is temporary, not a damaged-history dead end: the original's
    # missing result completes the same anchored batch without any repair.
    await parent.append_message(message("tool", "second finished", tool_call_id="b"))
    fork_id, omitted, _ = await parent.fork_snapshot()
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
    with pytest.raises(ForkRefused, match="history is being rewritten") as refused:
        await parent.fork_snapshot(is_compacting=lambda: True)
    assert refused.value.reason == "history_rewriting"
    assert len(list((tmp_path / "sessions").iterdir())) == 1


@pytest.mark.asyncio
async def test_cut_through_an_entry_truncates_the_child_and_leaves_the_parent(
    tmp_path: Path,
) -> None:
    """A named cut keeps the prefix through that entry and drops every later row."""
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    asked = message("user", "keep me")
    answered = message("assistant", "kept answer")
    cut_at = message("user", "fork from here")
    dropped = message("assistant", "after the cut")
    await parent.append_messages([asked, answered, cut_at, dropped])
    before = parent.path.read_bytes()

    fork_id, omitted, _ = await parent.fork_snapshot(through_entry_id=cut_at.id)

    child = Transcript(parent.directory.parent / fork_id)
    assert [item.id for item in child.build_llm_history()] == [asked.id, answered.id, cut_at.id]
    # The row is not merely absent from the model's view: it was not copied.
    assert dropped.id.encode() not in child.path.read_bytes()
    assert omitted
    assert parent.path.read_bytes() == before


@pytest.mark.asyncio
async def test_cut_at_the_last_entry_is_todays_whole_copy(tmp_path: Path) -> None:
    """Naming the newest entry equals naming none — the absent target is unchanged."""
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    first = message("user", "one")
    second = message("assistant", "two")
    third = message("user", "three")
    await parent.append_messages([first, second, third])
    before = parent.path.read_bytes()

    whole_id, omitted, _ = await parent.fork_snapshot()
    whole = Transcript(parent.directory.parent / whole_id)
    assert not omitted
    assert whole.path.read_bytes() == before

    cut_id, cut_omitted, _ = await parent.fork_snapshot(through_entry_id=third.id)
    at_last = Transcript(parent.directory.parent / cut_id)
    assert not cut_omitted
    assert at_last.path.read_bytes() == before
    assert parent.path.read_bytes() == before


@pytest.mark.asyncio
async def test_cut_refuses_an_entry_that_is_not_in_this_conversation(tmp_path: Path) -> None:
    """An unknown id must refuse, never silently fork MORE than was asked for."""
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    await parent.append_message(message("user", "hello"))
    before = parent.path.read_bytes()

    with pytest.raises(ForkRefused, match="not part of this conversation") as refused:
        await parent.fork_snapshot(through_entry_id="ffffffffffffffff")
    assert refused.value.reason == "entry_unknown"

    assert parent.path.read_bytes() == before
    assert len(list((tmp_path / "sessions").iterdir())) == 1


@pytest.mark.asyncio
async def test_cut_never_lands_inside_an_unpaired_batch(tmp_path: Path) -> None:
    """A cut aimed at a live batch lands AT-OR-BEFORE it, and is not a refusal."""
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    asked = message("user", "hello")
    call = message(
        "assistant",
        "",
        tool_calls=[
            ToolCall(id="a", name="bash", arguments={}),
            ToolCall(id="b", name="bash", arguments={}),
        ],
    )
    partial = message("tool", "first finished", tool_call_id="a")
    await parent.append_messages([asked, call, partial])

    for target in (call.id, partial.id):
        fork_id, omitted, landed = await parent.fork_snapshot(through_entry_id=target)
        child = Transcript(parent.directory.parent / fork_id)
        assert [item.id for item in child.build_llm_history()] == [asked.id]
        assert omitted
        # The named row could not be honoured exactly, and the caller is told
        # WHERE the copy stopped instead of having to diff the child.
        assert landed == asked.id

    # The cut re-evaluates against the pairing committed NOW, not a cached one,
    # and it still honours "at-or-before": once the batch is complete, a cut on
    # its LAST result carries it, while a cut on its first result does not.
    second = message("tool", "second finished", tool_call_id="b")
    await parent.append_message(second)
    fork_id, omitted, _ = await parent.fork_snapshot(through_entry_id=second.id)
    child = Transcript(parent.directory.parent / fork_id)
    assert [item.id for item in child.build_llm_history()] == [
        asked.id,
        call.id,
        partial.id,
        second.id,
    ]
    assert not omitted

    fork_id, _, _ = await parent.fork_snapshot(through_entry_id=partial.id)
    child = Transcript(parent.directory.parent / fork_id)
    assert [item.id for item in child.build_llm_history()] == [asked.id]


@pytest.mark.asyncio
async def test_cut_refuses_when_it_would_drop_the_compaction_anchor(tmp_path: Path) -> None:
    """A point older than the last summary's anchor refuses, in its own words."""
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    summarized = message("user", "OLD SUMMARIZED CONTENT")
    anchor = message("assistant", "kept after the summary")
    await parent.append_messages([summarized, anchor])
    await parent.append_compaction(
        summary="summary", first_kept_entry_id=anchor.id, tokens_before=100
    )
    after = message("user", "after the summary")
    later = message("assistant", "later still")
    await parent.append_messages([after, later])
    before = parent.path.read_bytes()

    with pytest.raises(ForkRefused, match="before the conversation's last summary") as refused:
        await parent.fork_snapshot(through_entry_id=summarized.id)
    assert refused.value.reason == "before_anchor"
    assert parent.path.read_bytes() == before
    assert len(list((tmp_path / "sessions").iterdir())) == 1

    # Inside the summary's retained prefix the same transcript cuts cleanly: the
    # refusal is about the named POINT, not about the conversation.
    fork_id, _, _ = await parent.fork_snapshot(through_entry_id=after.id)
    child = Transcript(parent.directory.parent / fork_id)
    kept = [item.id for item in child.build_llm_history()]
    # The compaction marker heads the child's replay, then the summary's
    # retained prefix; the summarized row is carried as bytes but never replayed.
    assert kept[1:] == [anchor.id, after.id]
    assert summarized.id not in kept
    assert later.id.encode() not in child.path.read_bytes()


@pytest.mark.asyncio
async def test_cut_inside_the_kept_window_keeps_the_summary(tmp_path: Path) -> None:
    """The ORDINARY cut on a compacted session: the child is the compacted view.

    A compaction is written after the window it preserves, so a cut anywhere in
    that window is *before* the marker's own row while its anchor is *before* the
    cut. Retaining the marker is what stops the child replaying the rows the
    summary replaced — the shape round 1 shipped, where the marker was dropped
    and the summary with it.
    """
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    old = message("user", "OLD SUMMARIZED CONTENT")
    old_answer = message("assistant", "old answer")
    kept = message("user", "KEPT1")
    kept_answer = message("assistant", "kept answer")
    await parent.append_messages([old, old_answer, kept, kept_answer])
    await parent.append_compaction(
        summary="SUMMARY-X", first_kept_entry_id=kept.id, tokens_before=100
    )
    newer = message("user", "NEW1")
    await parent.append_message(newer)
    marker = next(row.id for row in parent.entries() if row.type == ENTRY_COMPACTION)
    before = parent.path.read_bytes()

    for target, expected in ((kept.id, [kept.id]), (kept_answer.id, [kept.id, kept_answer.id])):
        fork_id, omitted, landed = await parent.fork_snapshot(through_entry_id=target)
        child = Transcript(parent.directory.parent / fork_id)
        replayed = [item.id for item in child.build_llm_history()]
        assert replayed == [marker, *expected], replayed
        assert "SUMMARY-X" in str(child.build_llm_history())
        # The summarised rows stay as BYTES and are never replayed — the same
        # shape the no-target fork leaves them in.
        assert old.id not in replayed and old_answer.id not in replayed
        assert landed == target
        assert omitted
        assert newer.id.encode() not in child.path.read_bytes()
    assert parent.path.read_bytes() == before


@pytest.mark.asyncio
async def test_cut_between_two_anchors_replays_the_older_summary(tmp_path: Path) -> None:
    """A cut inside an EARLIER compaction's window is legal, and is not refused.

    Round 1 tested the NEWEST anchor alone, so this cut was refused even though
    the older compaction's marker and anchor are both entirely before it and the
    child replays a coherent summarized prefix. The newer marker must be dropped:
    replay reads the LAST marker it finds, and that one's anchor is after the cut.
    """
    parent = Transcript(tmp_path / "sessions" / "parent000001")
    old = message("user", "OLD1")
    first = message("assistant", "first kept")
    await parent.append_messages([old, first])
    await parent.append_compaction(
        summary="SUMMARY-1", first_kept_entry_id=first.id, tokens_before=10
    )
    second = message("user", "SECOND")
    await parent.append_messages([second, message("assistant", "second answer")])
    await parent.append_compaction(
        summary="SUMMARY-2", first_kept_entry_id=second.id, tokens_before=10
    )
    third = message("user", "THIRD")
    await parent.append_message(third)
    older_marker, newer_marker = [
        row.id for row in parent.entries() if row.type == ENTRY_COMPACTION
    ]
    before = parent.path.read_bytes()

    fork_id, _, landed = await parent.fork_snapshot(through_entry_id=first.id)

    child = Transcript(parent.directory.parent / fork_id)
    assert [item.id for item in child.build_llm_history()] == [older_marker, first.id]
    assert "SUMMARY-1" in str(child.build_llm_history())
    assert landed == first.id
    assert older_marker in child.path.read_text()
    assert newer_marker not in child.path.read_text()
    assert parent.path.read_bytes() == before


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
