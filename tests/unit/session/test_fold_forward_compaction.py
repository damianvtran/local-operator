"""Fold-forward compaction: a pass after a wake folds onto the summary that exists.

Operator report: "the conversation is compact and then the agent wakes and then
it uncompacts and then recompacts when the agent finishes again... On wake, the
conversation up to that point should stay compacted and then once the wake turn
finishes then the remaining messages should join the compaction, instead of
uncompacting and recompacting."

Which reading held was MEASURED first, because the two imply different fixes:
on the operator's 272 MB session (``bda7b76d34e0``: 56 compactions, 664 wake
runs), every post-compaction wake turn starts at the COMPACTED size (172,612 ->
185,947; 97,494 -> 116,901; 86,407 -> 119,078), and the runtime logs carry no
"replaying full history" — the model context never "uncompacts". What
re-derives is the SUMMARY: a chained pass summarises a span whose head is the
rendered marker, re-exposing the previous summary inside ``<conversation>`` as
an ordinary user turn and re-deriving everything from it. These tests pin the
fold that replaces that, and are the first to combine wake + compaction.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.harness.wake import DueWake, WakeSchedule
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript, TranscriptEntry

#: Text-only, so the strategy resolver picks context-full — the branch that
#: folds through ``previous_summary`` (snapcompact folds its own way, via the
#: archive's accumulated text).
TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=100_000, supports_images=False
)

#: Small enough that three short turns leave history outside the kept window,
#: which is what gives ``find_cut_point`` something to summarize.
KEEP_RECENT = 40

#: What the first pass's summarizer call returns; the second pass must reach it
#: through the fold slot and NOT through the conversation.
PRIOR_SUMMARY = "PRIOR-SUMMARY-7f3a"


class ScriptedStream:
    """Replays one reply per call and records the requests it saw."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[object] = []

    def __call__(self, request, signal):
        self.requests.append(request)
        index = len(self.requests) - 1
        reply = self.replies[index] if index < len(self.replies) else "ok"

        async def gen():
            yield StreamTextDelta(delta=reply)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, model=TEXT_MODEL, **kwargs) -> Session:
    settings = kwargs.pop(
        "compaction_settings",
        # ``auto_continue=False`` keeps the replies each turn consumes a fact
        # of the fixture: a post-turn pass would otherwise schedule a
        # continuation prompt and eat the next scripted reply.
        CompactionSettings(keep_recent_tokens=KEEP_RECENT, auto_continue=False),
    )
    return Session(
        model=model,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings,
        **kwargs,
    )


async def talk(session: Session, turns: int = 3) -> None:
    for index in range(turns):
        await session.prompt(f"question {index} " + "detail " * 30)


async def wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def _due_wake(message: str = "check the deploy") -> DueWake:
    schedule = WakeSchedule(id="w1", message=message, next_due_at=0, created_at=0)
    return DueWake(schedule=schedule, occurrence=1, planned_total=1, final=True)


def _compaction_entries(session: Session) -> list[TranscriptEntry]:
    return [e for e in session._transcript.entries() if e.type == "compaction"]


@pytest.mark.asyncio
async def test_a_wake_after_a_pass_leaves_the_compacted_context_and_no_second_pass(tmp_path):
    """The conversation up to the wake stays compacted; the wake only appends.

    After a pass the live context is ``[marker, *kept]``. A wake turn that does
    not cross the trigger must leave exactly that prefix in place — same
    marker object, byte-identical render — and append only its own delivery
    and reply. No second pass, no re-summarisation: the state "uncompacts and
    recompacts" describes is what this pins against.
    """
    stream = ScriptedStream(["reply", "reply", "reply", PRIOR_SUMMARY, "assistant reply " * 60])
    session = make_session(tmp_path, stream)
    await talk(session)

    outcome = await session.compact_now()
    assert outcome.ran is True
    assert outcome.strategy == "context-full"

    compacted = list(session._context.messages)
    assert isinstance(compacted[0], CustomMessage)
    assert compacted[0].custom_type == "compaction_summary"
    assert _compaction_entries(session)[0].payload["summary"] == PRIOR_SUMMARY
    rendered_before = session._render_for_compaction()

    await session._deliver_wake(_due_wake())
    expected = len(compacted) + 2  # the wake delivery and the reply it runs
    await wait_for(lambda: len(session._context.messages) >= expected and not session._is_streaming)

    after = list(session._context.messages)
    assert len(after) == expected
    assert after[: len(compacted)] == compacted
    assert after[0] is compacted[0]
    assert isinstance(after[-2], CustomMessage) and after[-2].custom_type == "wake_prompt"
    assert isinstance(after[-1], Message) and after[-1].role == "assistant"

    # The model context is byte-identical except for the appended messages.
    rendered_after = session._render_for_compaction()
    assert len(rendered_after) == len(rendered_before) + 2
    assert rendered_after[: len(rendered_before)] == rendered_before

    # Below the trigger: the wake turn ended with no second pass.
    assert len(_compaction_entries(session)) == 1

    await session.dispose()


@pytest.mark.asyncio
async def test_a_wake_that_crosses_the_trigger_folds_its_span_onto_the_summary(tmp_path):
    """A wake whose result crosses the trigger folds onto the prior summary.

    The second pass runs automatically at the wake turn's end. Its summarizer
    input must carry the prior summary in the fold slot (``previous_summary``
    -> ``<previous-summary>``) and must NOT re-expose the rendered marker
    inside ``<conversation>`` as an ordinary user turn; the wake's own span is
    the conversation material being joined to the fold.
    """
    stream = ScriptedStream(["reply", "reply", "reply", PRIOR_SUMMARY, "assistant work " * 200])
    session = make_session(tmp_path, stream)
    await talk(session)

    first = await session.compact_now()
    assert first.ran is True
    assert first.strategy == "context-full"
    assert len(_compaction_entries(session)) == 1

    # Capture the second pass's prompt and answer it deterministically.
    captured: dict[str, str] = {}

    async def capture(system: str, prompt: str) -> str:
        captured["prompt"] = prompt
        return "SECOND-SUMMARY-9c21"

    session._one_shot_complete = capture  # type: ignore[method-assign]

    # Bring the trigger down so the wake turn's work crosses it, without
    # letting the talk/first-pass turns fire the automatic gate early.
    session._compaction_settings = session._compaction_settings.model_copy(
        update={"threshold_tokens": 100}
    )

    await session._deliver_wake(_due_wake("wake up for the deploy"))
    await wait_for(lambda: len(_compaction_entries(session)) == 2)

    entries = _compaction_entries(session)
    assert entries[-1].payload["summary"] == "SECOND-SUMMARY-9c21"

    prompt = captured["prompt"]
    fold = prompt.split("<previous-summary>", 1)[1].split("</previous-summary>", 1)[0]
    conversation = prompt.split("<conversation>", 1)[1].split("</conversation>", 1)[0]
    # The prior summary arrives once, through the fold slot...
    assert PRIOR_SUMMARY in fold
    assert prompt.count(PRIOR_SUMMARY) == 1
    # ...the rendered marker does not also ride the conversation...
    assert "<previous-context-summary>" not in conversation
    assert PRIOR_SUMMARY not in conversation
    # ...and the wake's span is the new material being folded in.
    assert "wake up for the deploy" in conversation

    # The rebuilt context is a fresh marker over the kept tail: the pre-pass
    # context (previous summary and all) was replaced by the folded one.
    context = session._context.messages
    assert isinstance(context[0], CustomMessage)
    assert context[0].custom_type == "compaction_summary"
    assert context[0].details["summary"] == "SECOND-SUMMARY-9c21"

    await session.dispose()
