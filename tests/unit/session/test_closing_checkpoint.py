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
async def test_a_journal_that_already_carries_a_checkpoint_is_not_re_anchored(tmp_path):
    """A closing row must never LOWER the durable state a richer row already holds.

    Found by CI rather than by design (``test_headless_turn_preserves_a_rich_frontend_checkpoint``):
    the durable row is REPLACEMENT state, so a runtime that writes its own view
    over a richer one hands every reader — who takes the NEWEST row — the poorer
    state. The gate is ``checkpoint_id``: set when a checkpoint is restored and
    when one is written, and by nothing else, so it is the cheap proof that a row
    exists. A session that has one is bounded by it already (and the replay cache
    makes its repeat cost free).
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("prior work"))
    # The durable row names THIS session (the transcript's directory), as a
    # resume leaves it. A row naming another session is a FORK's, and the store
    # rightly clears its ``checkpoint_id`` — see ``_inherited_identity_fixups``.
    rich = FrontendSessionState(
        session_id="sess",
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

    state = FrontendSessionState.model_validate(
        Transcript(directory).latest_custom(FRONTEND_CHECKPOINT_CUSTOM_TYPE)["state"]
    )
    assert state.conversation_title == "Real title", "the closing row lowered the durable title"
    assert state.cumulative_parent_cost == 12.34
    assert state.active_duration_s == 300.0
    assert len(checkpoint_rows(directory)) == 1, "a second anchor was written anyway"
