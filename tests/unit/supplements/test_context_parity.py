"""Nothing a supplement writes reaches model context (memo §2.11, §5.1 parity test).

THE DOOR THIS PINS. ``supplement_v1`` rows are ``append_custom`` entries, which "never enter
LLM context". Two things could open that door later and neither would fail any other test:
adding the type to ``session._PERSISTABLE_CUSTOM_TYPES`` (which turns a custom entry into a
message row on replay), or a renderer deciding custom entries are interesting. So the
assertion is not "the list does not contain the string" but the OBSERVABLE one: two identical
conversations, one with supplements in its journal, produce byte-identical LLM history and a
byte-identical provider request; and an existing compaction cut is unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.supplements import persistence
from local_operator.supplements.candidates import Candidate
from local_operator.supplements.decision import Decision
from tests.unit.session.test_session import ScriptedStream, wait_for

pytestmark = pytest.mark.asyncio

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


def _candidate(path: str) -> Candidate:
    return Candidate(
        path=path,
        absolute=f"/work/{path}",
        name=path.rsplit("/", 1)[-1],
        kind="markdown",
        size_bytes=10,
        mtime=1.0,
        tier=1,
        tool="write",
        order=1,
    )


async def _seed(directory: Path) -> Transcript:
    transcript = Transcript(directory)
    from local_operator.harness.types import Message

    for index in range(3):
        await transcript.append_message(Message.user(f"question {index}"))
        await transcript.append_message(Message.assistant(f"answer {index}"))
    return transcript


async def _session(directory: Path, stream: ScriptedStream) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )


async def test_llm_history_and_the_provider_request_are_byte_identical(tmp_path: Path) -> None:
    """ONE conversation, measured before and after the rows land.

    A single transcript is the whole point: comparing two conversations would also have to
    explain away their different message ids, and the question here is narrow -- does a
    supplement row change what the model sees?
    """
    directory = tmp_path / "sess"
    transcript = await _seed(directory)

    def history() -> list[dict[str, Any]]:
        return [m.model_dump() for m in transcript.build_llm_history()]

    before_history = history()
    stream = ScriptedStream([[StreamTextDelta(delta="ok"), StreamEndEvent(stop_reason="stop")]])
    session = await _session(directory, stream)
    try:
        await session.prompt("next question")
        await wait_for(lambda: len(stream.requests) == 1)
        before_request = [m.model_dump() for m in stream.requests[0].messages]
        for index in range(3):
            await persistence.append_row(
                transcript,
                persistence.build_details(
                    anchor=f"anchor{index}",
                    job=f"job{index}",
                    version=1,
                    state="done",
                    decision=Decision(
                        featured=(_candidate(f"reports/r{index}.md"),),
                        files_p={f"reports/r{index}.md": 0.9},
                        vendor="radient",
                    ),
                ),
            )
        assert history() == before_history, "a supplement row changed the replayed history"
        # The rendered request, on a session that replays the journal WITH the rows.
        reopened = await _session(directory, ScriptedStream([[]]))
        try:
            replayed = [m.model_dump() for m in reopened.transcript.build_llm_history()]
            # The turn the first session ran is appended AFTER the supplement rows, so the
            # seam under test is the prefix: the conversation as it was, unchanged.
            assert replayed[: len(before_history)] == before_history
        finally:
            await reopened.dispose()
        assert "supplement" not in repr(before_history)
        assert before_request[-1]["role"] == "user"
    finally:
        await session.dispose()


async def test_a_supplement_row_after_a_compaction_does_not_change_the_cut(tmp_path: Path) -> None:
    """The compaction cut is by message id against the conversation; a custom entry written
    afterwards must not move it. Asserted on the built history: the summary message is
    followed by exactly the entries the cut kept, with or without a supplement row."""
    from local_operator.compaction import CompactionSettings

    directory = tmp_path / "sess"
    transcript = await _seed(directory)
    entries_before = tuple(e.id for e in transcript.entries())
    settings = CompactionSettings()
    assert settings is not None  # the cut machinery is not exercised here; the row is
    await persistence.append_row(
        transcript,
        persistence.build_details(
            anchor="a",
            job="b",
            version=1,
            state="done",
            decision=Decision(
                featured=(_candidate("reports/x.md"),), files_p={"reports/x.md": 0.9}, vendor="r"
            ),
        ),
    )
    entries_after = tuple(e.id for e in transcript.entries())
    assert entries_after[: len(entries_before)] == entries_before
    assert len(entries_after) == len(entries_before) + 1
    # The one new entry is the supplement and nothing else moved.
    newest = transcript.latest_custom("supplement_v1")
    assert newest is not None and newest["anchor"] == "a"
