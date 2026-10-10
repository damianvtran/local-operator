"""A runtime leaves a CLOSING checkpoint, so the next cold open is bounded.

WHY THIS FILE EXISTS. A journal with no ``frontend_state_checkpoint_v1`` row
gives ``read_replay_suffix`` no compaction boundary to stop at, so it scans to
BOF on every request: 204 ms per call on a 35.5 MB journal, paid again by each
of the snapshot, ``/history`` and the SSE frame. 15 of the 40 largest real
journals carried no checkpoint, and the writer that would have produced one ran
only at turn end and only "for any session with a UI or an attach subscriber" —
so every session driven headlessly (``lop exec``, a scheduled job, a scripted
run) was never anchored at all.

The tests below drive the REAL session: ``prompt`` then ``dispose``, then read
the journal the way the cold paths read it. Asserting on
``Session._write_closing_checkpoint`` directly would prove the method works
while proving nothing about whether teardown reaches it.
"""

from __future__ import annotations

import os

import pytest

from local_operator.harness.types import Message, ModelSpec
from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.session import Session
from local_operator.session.transcript import (
    BOOKKEEPING_CUSTOM_TYPES,
    Transcript,
    TranscriptEntry,
    read_replay_suffix,
)

MODEL = ModelSpec(provider="test", model_id="reads", context_window=100_000, supports_images=False)

#: A stamp far enough in the past that no filesystem timestamp granularity can
#: confuse "restored" with "just written" (the same backdating
#: ``test_transcript_bookkeeping_mtime`` uses).
PAST = 1_700_000_000.0


class ScriptedStream:
    """One reply, whatever is asked — enough to end a turn."""

    def __call__(self, request, signal):
        from local_operator.harness.types import StreamEndEvent, StreamTextDelta

        async def gen():
            yield StreamTextDelta(delta="reply")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(directory) -> Session:
    """A real headless Session (``has_ui`` defaults to False) on ``directory``."""
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream(),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )


def rows(directory) -> list[TranscriptEntry]:
    path = directory / "transcript.jsonl"
    if not path.exists():
        # A session that never wrote has no journal at all, which is an answer
        # to "what is on disk" rather than an error.
        return []
    return [
        entry
        for entry in (
            TranscriptEntry.from_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if entry is not None
    ]


def checkpoint_rows(directory) -> list[TranscriptEntry]:
    return [
        entry
        for entry in rows(directory)
        if entry.type == "custom"
        and entry.payload.get("custom_type") == FRONTEND_CHECKPOINT_CUSTOM_TYPE
    ]


@pytest.mark.asyncio
async def test_a_headless_runtime_anchors_the_journal_at_teardown(tmp_path):
    """The row exists after a headless run — the fix, at its narrowest seam.

    A session with no UI wrote no checkpoint anywhere before this change: the
    only writer was the turn-end one, gated on ``has_ui or has_subscribers``.
    """
    directory = tmp_path / "sess"
    session = make_session(directory)
    await session.prompt("do the thing")
    await session.dispose()

    final = rows(directory)
    assert (
        final[-1].payload.get("custom_type") == FRONTEND_CHECKPOINT_CUSTOM_TYPE
    ), "the last row of a headless run must be the closing checkpoint"


@pytest.mark.asyncio
async def test_the_closing_checkpoint_is_what_bounds_the_next_open(tmp_path):
    """The consequence, measured on the read the cold paths actually make.

    A check: the SAME journal is read twice, with and without the closing row,
    because "the read is bounded" is only evidence of anything if removing the
    row unbounds it. The journal is deliberately larger than one read chunk and
    carries a compaction, which is the shape of every real session this matters
    for: the reader stops at the newest compaction's ``first_kept_entry_id``
    only once the checkpoint it was asked for is also in hand, so a journal
    whose checkpoint is missing scans to BOF — 35.5 MB and 204 ms per request
    on the reference session.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    kept = ""
    for index in range(12):
        user = await transcript.append_message(Message.user(f"turn {index}"))
        await transcript.append_message(Message.assistant("y" * 200_000))
        kept = user.id
    await transcript.append_compaction("summary", kept, tokens_before=100)

    session = make_session(directory)
    await session.prompt("close it out")
    await session.dispose()

    size = (directory / "transcript.jsonl").stat().st_size
    bounded = read_replay_suffix(
        directory, checkpoint_types=(FRONTEND_CHECKPOINT_CUSTOM_TYPE,)
    ).bytes_read
    assert bounded < size, f"the closing row did not bound the read ({bounded} of {size})"

    # A: the same journal with the closing rows removed is unbounded again.
    path = directory / "transcript.jsonl"
    path.write_text(
        "".join(
            line + "\n"
            for line in path.read_text(encoding="utf-8").splitlines()
            if f'"{FRONTEND_CHECKPOINT_CUSTOM_TYPE}"' not in line
        ),
        encoding="utf-8",
    )
    unbounded = read_replay_suffix(
        directory, checkpoint_types=(FRONTEND_CHECKPOINT_CUSTOM_TYPE,)
    ).bytes_read
    assert (
        unbounded == path.stat().st_size
    ), "the control case is not a control: this journal is bounded without the row too"


@pytest.mark.asyncio
async def test_the_closing_checkpoint_does_not_move_the_activity_clock(tmp_path):
    """A runtime closing hours later must not rank the session as just worked.

    The transcript's mtime IS ``retention.session_activity``, the one clock the
    picker and ``session.cleanup`` share. The row is a RECORD about the session,
    so the append restores the mtime — and the size still moves, which is what
    every stat-keyed reader downstream needs to see an append at all.
    """
    directory = tmp_path / "sess"
    session = make_session(directory)
    await session.prompt("do the thing")
    path = directory / "transcript.jsonl"
    # Wind the clock back to a finished conversation, then close the runtime.
    os.utime(path, (PAST, PAST))
    before_size = path.stat().st_size

    await session.dispose()

    stat = path.stat()
    assert stat.st_mtime == pytest.approx(PAST, abs=1e-6), "the closing row restamped the session"
    assert stat.st_size > before_size, "the closing row was not written at all"


@pytest.mark.asyncio
async def test_a_runtime_that_ended_no_turn_writes_nothing(tmp_path):
    """Opening a conversation is not work — the same rule the cold viewer states.

    ``dispose`` is reached by runtimes that were engaged and ended no turn, and
    a session a user only looked at must not gain a row for it.
    """
    directory = tmp_path / "sess"
    session = make_session(directory)
    before = rows(directory)

    await session.dispose()

    assert rows(directory) == before
    assert checkpoint_rows(directory) == []


@pytest.mark.asyncio
async def test_a_child_session_writes_no_closing_checkpoint(tmp_path):
    """A roster of children must not pay one checkpoint each per run.

    Child transcripts are read through page reads (the child panel,
    ``subagent_view``), never through the cold replay this row would bound.
    """
    directory = tmp_path / "child"
    session = make_session(directory)
    session._job_id = "job-1"
    await session.prompt("child work")
    await session.dispose()

    assert checkpoint_rows(directory) == []


def test_the_checkpoint_type_is_bookkeeping_in_the_transcript_vocabulary():
    """The mtime exemption is a property of the TYPE, not of the caller's flag.

    ``_write_entries`` honours ``preserve_mtime`` only for a whole batch of
    ``BOOKKEEPING_CUSTOM_TYPES``, so a type missing from that set loses the
    exemption SILENTLY while every call site still looks right. Pinned against
    the constant as well as by membership, because the transcript module
    restates this literal deliberately (it is a leaf and cannot import
    ``frontend_state``).
    """
    assert FRONTEND_CHECKPOINT_CUSTOM_TYPE in BOOKKEEPING_CUSTOM_TYPES
    entry = TranscriptEntry(
        "closing",
        PAST,
        "custom",
        {"custom_type": FRONTEND_CHECKPOINT_CUSTOM_TYPE, "details": {"state": {}}},
    )
    from local_operator.session.transcript import _is_bookkeeping_batch

    assert _is_bookkeeping_batch([entry]) is True


@pytest.mark.asyncio
async def test_a_closing_row_never_lowers_the_durable_state_it_read(tmp_path):
    """The N1 invariant, now that EVERY runtime writes a closing row.

    This test used to pin the opposite mechanism — the first revision SKIPPED a
    journal that already carried a row, which fixed the lowering and froze the row
    (review round 1, F1: a cold open then painted the FIRST runtime's
    ``context_tokens`` forever, and the read bound decayed with every later
    runtime until a compaction took it back to BOF). The row is written every
    time now and MERGED over the row it read, so the richer durable fields survive
    the write: the title a TUI set, the spend, and the operator's (surface-
    observed) active duration.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("prior work"))
    rich = FrontendSessionState(
        session_id="conv",
        epoch="tui-epoch",
        conversation_title="Real title",
        conversation_title_user_set=True,
        cumulative_parent_cost=12.34,
        active_duration_s=300.0,
    )
    await FrontendStateStore(rich).checkpoint(transcript)

    session = make_session(directory)
    try:
        assert session.frontend_state.conversation_title == "Real title"
        await session.prompt("headless turn")
    finally:
        await session.dispose()

    restored = Transcript(directory).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)
    assert isinstance(restored, dict)
    state = FrontendSessionState.model_validate(restored["state"])
    assert len(checkpoint_rows(directory)) == 2, "the closing row must be written"
    assert state.conversation_title == "Real title", "the closing row lowered the title"
    assert state.cumulative_parent_cost == 12.34
    # The field is the summed duration of the turns that ran, and the merge takes
    # the larger of the two (review round 2, F8): the headless run's own turn time
    # adds to the TUI's 300.0 rather than replacing it. What must never happen is
    # the durable figure being LOWERED, which is what the N1 defect was.
    assert 300.0 <= state.active_duration_s < 301.0, "a headless turn lowered active duration"


@pytest.mark.asyncio
async def test_twelve_runtimes_and_a_compaction_keep_the_read_bounded(tmp_path):
    """F1's other half, and the reason write-once had to go.

    A closing row written ONCE and then skipped forever leaves the newest
    checkpoint where the first runtime wrote it: the reader must reach it, so the
    bound decays with every later runtime, and a compaction landing above it puts
    the reader back at BOF (review round 1 measured `bytes_read` at 100% of a
    5.08 MB journal after 12 runtimes, a compaction and 2 more). This is that
    case, with the two things that matter pinned: the read stays bounded, and the
    tokens the newest row carries are the LAST runtime's reading.

    The turns are bulky on purpose — the read walks in 1 MiB chunks, so a journal
    smaller than one chunk cannot show whether it was bounded.
    """
    directory = tmp_path / "sess"
    last_tokens = 0
    for turn in range(12):
        session = make_session(directory)
        try:
            await session.prompt("x" * 100_000 + f" turn {turn}")
            last_tokens = int(session.frontend_state.context_tokens or 0)
        finally:
            await session.dispose()

    # A compaction that keeps only the recent rows — the shape a real pass leaves
    # behind, and the one that lets the reader stop early.
    transcript = Transcript(directory)
    await transcript.append_compaction(
        "summary of the early turns", transcript.entries()[-1].id, tokens_before=1000
    )
    for turn in range(2):
        session = make_session(directory)
        try:
            await session.prompt("x" * 100_000 + f" post-compaction turn {turn}")
            # THE LAST RUNTIME'S READING, not the largest one seen (review round
            # 2, F7). The max hid the defect this test exists for:
            # ``context_tokens`` is NOT monotonic — the compaction above shrinks
            # it — so taking the largest observed value asserts the stale
            # pre-compaction figure and passes while the newest row carries it.
            last_tokens = int(session.frontend_state.context_tokens or 0)
        finally:
            await session.dispose()

    path = directory / "transcript.jsonl"
    assert path.stat().st_size > (1 << 20), "the fixture must exceed one read chunk"
    suffix = read_replay_suffix(directory, checkpoint_types=(FRONTEND_CHECKPOINT_CUSTOM_TYPE,))
    assert suffix.bytes_read < path.stat().st_size, (
        "the read reached the whole journal again: the newest checkpoint is not at " "the tail"
    )
    assert suffix.checkpoint is not None, "the newest row must carry the checkpoint"
    # ``ReplaySuffix.checkpoint`` is the ROW's details — the same shape
    # ``_restore_cold_details`` reads — so the state is the ``state`` key.
    newest = FrontendSessionState.model_validate(suffix.checkpoint["state"])
    rows = checkpoint_rows(directory)
    assert len(rows) >= 2, "every runtime that ended a turn owes a row"
    assert newest.context_tokens == last_tokens, (
        "the newest row carries a stale context reading: "
        f"{newest.context_tokens} != {last_tokens}"
    )
