"""Busy-turn checkpoint: the record rides the real turn-pair commit.

A tiny Session over a fake stream; no provider, no network, no live runtime. The
point is the WIRING: a latched request is recorded at the transcript commit and
readable cold afterwards, and a runtime with no request writes nothing.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import ModelSpec
from local_operator.session.session import Session
from local_operator.session.transcript import (
    CHECKPOINT_CUSTOM_TYPE,
    Transcript,
    read_latest_custom,
)
from tests.unit.session.test_session_footprint import (
    ScriptedStream,
    _final_turn,
    _read_turn,
    read_tool,
)

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


def _session(tmp_path, turns) -> Session:
    return Session(
        model=MODEL,
        stream_fn=ScriptedStream(turns),
        tools=[read_tool()],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
    )


@pytest.mark.asyncio
async def test_no_request_writes_no_checkpoint(tmp_path):
    """An ordinary turn must leave the checkpoint feature completely inert."""
    session = _session(tmp_path, [_read_turn("c1"), _final_turn("done")])
    await session.prompt("do the thing")
    assert read_latest_custom(session._transcript.directory, CHECKPOINT_CUSTOM_TYPE) is None


@pytest.mark.asyncio
async def test_latched_request_is_recorded_at_the_turn_pair_commit(tmp_path):
    """The checkpoint lands with the settled batch and is readable cold."""
    session = _session(tmp_path, [_read_turn("c1"), _final_turn("done")])
    assert session.request_checkpoint(
        turn_id="turn-1",
        boot_build="A",
        target_build="B",
        completed_effects=[{"tool": "read", "call_id": "c1", "result_digest": "d"}],
        inbox_consumed_ids=["inbox-1"],
    )
    await session.prompt("do the thing")

    record = read_latest_custom(session._transcript.directory, CHECKPOINT_CUSTOM_TYPE)
    assert record is not None
    assert record["phase"] == "resume"
    assert record["turn_id"] == "turn-1" and record["target_build"] == "B"
    assert record["inbox_consumed_ids"] == ["inbox-1"]
    assert "c1" in {effect["call_id"] for effect in record["completed_effects"]}
    # The latch is consumed: a second ordinary turn writes no new checkpoint.
    assert session._checkpoint_pending is False


@pytest.mark.asyncio
async def test_unknown_phase_is_not_latched(tmp_path):
    session = _session(tmp_path, [_final_turn("done")])
    assert session.request_checkpoint(phase="bogus") is False
    assert session._checkpoint_pending is False


@pytest.mark.asyncio
async def test_deferred_phase_still_records_but_says_so(tmp_path):
    session = _session(tmp_path, [_final_turn("done")])
    session.request_checkpoint(turn_id="t", phase="deferred")
    await session.prompt("do the thing")
    record = read_latest_custom(session._transcript.directory, CHECKPOINT_CUSTOM_TYPE)
    assert record is not None and record["phase"] == "deferred"
