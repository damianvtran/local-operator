"""On-demand compaction: ``Session.compact_now`` — the pass behind ``/compact``.

The command used to print "compaction runs automatically when the context fills
up" and change nothing, so these tests are about the two properties that fix
means:

* the manual trigger runs THE SAME pass as the automatic one — one strategy
  resolver, one cut point, one pair of events — differing only in that the user
  asking IS the trigger, and
* every state a manual trigger can be pressed in that the automatic gate never
  sees comes back as a REFUSAL naming itself. A refusal nobody can see is
  indistinguishable from the bug being fixed.

Real compaction throughout (no stubbed ``compaction.api``): the strategy split
is only interesting if the real ``resolve_strategy`` reads the real
``ModelSpec.supports_images``, and the receipt's numbers are only interesting if
a real estimator produced them.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    AgentEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    CustomMessage,
    ImageContent,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.session.session import Session, _render_compaction_marker
from local_operator.session.transcript import Transcript

#: PNG signature. A replayed frame that does not start with it is our own
#: double-encoding, not a provider being fussy.
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Vision-capable and text-only halves of the same model, so the ONLY input to
#: the strategy split is the capability flag.
VISION_MODEL = ModelSpec(provider="test", model_id="sees", context_window=100_000)
TEXT_MODEL = ModelSpec(
    provider="test", model_id="reads", context_window=100_000, supports_images=False
)

#: Small enough that three short turns leave history outside the kept window,
#: which is what gives ``find_cut_point`` something to summarize. Production
#: keeps 20 000; the arithmetic under test does not depend on the size.
KEEP_RECENT = 40


class ScriptedStream:
    """Replays one event script per call and records the requests it saw."""

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


def make_session(tmp_path, stream, model=VISION_MODEL, **kwargs) -> Session:
    settings = kwargs.pop("compaction_settings", CompactionSettings(keep_recent_tokens=KEEP_RECENT))
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
    """Run real turns so the context AND the transcript carry history.

    Both matter: the cut point is taken over the LLM history, and the kept
    window's first message must be a transcript entry or replay would drop it
    on the next resume.
    """
    for index in range(turns):
        await session.prompt(f"question {index} " + "detail " * 30)


@pytest.mark.asyncio
async def test_a_vision_model_compacts_into_a_snapcompact_archive(tmp_path):
    """``supports_images`` picks snapcompact, and the pass proves it took that
    branch by storing an archive — the same ``preserve_data['snapcompact']``
    payload the automatic pass stores, because it is the same code.

    The pass must also make NO provider call: that is snapcompact's contract
    (the archive replaces a summary), and the call it used to make — the whole
    discarded history shipped out for a caption — is where a manual /compact
    spent 20–50 of its ~60 seconds. ``_one_shot_complete`` raising proves the
    branch never reaches for it."""
    stream = ScriptedStream(["reply"] * 4)
    session = make_session(tmp_path, stream, model=VISION_MODEL)
    await talk(session)

    async def no_llm_calls(system: str, prompt: str) -> str:
        raise AssertionError("snapcompact must not make a provider call")

    session._one_shot_complete = no_llm_calls  # type: ignore[method-assign]
    outcome = await session.compact_now()

    assert outcome.ran is True
    assert outcome.strategy == "snapcompact"
    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert len(entries) == 1
    archive = entries[0].payload["preserve_data"]["snapcompact"]
    # The archive is the branch's fingerprint: a snapcompact pass stores the
    # bounded source text and the frame shape it will render into, and the
    # language pass stores nothing at all (asserted in the sibling test).
    # Rendered PNGs appear once the source outgrows the plain-text edges
    # (2 x 3 frames x 13 916 chars), which this history does NOT: the empty
    # frames list below is a property of the fixture, not coverage of the
    # imaged path — test_an_archive_with_frames_replays_as_valid_pngs covers
    # that, and this assertion is why the doubly-encoded frames shipped.
    assert archive["text"] and archive["shape_id"]
    assert archive["frames"] == []
    marker = session._context.messages[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"
    # The live context replays from the SAME archive the transcript persisted,
    # so a resume and the running session see one history.
    assert marker.details["preserve_data"]["snapcompact"]["text"] == archive["text"]
    await session.dispose()


@pytest.mark.asyncio
async def test_an_archive_with_frames_replays_as_valid_pngs(tmp_path):
    """The sibling above deliberately compacts a history too small to image,
    which is exactly why CI never executed the path that mattered.

    With enough history to render frames, the marker's archive is a dump that
    the converter revives and replays on the VERY NEXT request — in this live
    process, not only after a resume. That round trip encoded base64 twice and
    decoded it never, so every post-compaction request shipped
    ``base64(base64(png))`` as ``image/png``; providers answered 400 and the
    session's image-rejection backstop then dropped the whole compacted
    history. The PNG magic on the replayed block is the assertion that closes
    it (corrupt frames decode to ASCII ``iVBO`` = ``6956424f``).
    """
    stream = ScriptedStream(["reply"] * 8)
    session = make_session(tmp_path, stream, model=VISION_MODEL)
    # Past 2 x HQ_EDGE_FRAMES x 13 916 chars of plain-text edges, so the middle
    # is actually imaged instead of kept verbatim.
    for index in range(6):
        await session.prompt(f"question {index} " + "detail " * 3000)

    outcome = await session.compact_now()

    assert outcome.strategy == "snapcompact"
    entry = [e for e in session._transcript.entries() if e.type == "compaction"][0]
    persisted = entry.payload["preserve_data"]["snapcompact"]
    assert persisted["frames"] and all(isinstance(f, str) for f in persisted["frames"])
    assert base64.b64decode(persisted["frames"][0]).startswith(PNG_MAGIC)

    # The live wire history, i.e. what the next request would actually send.
    replayed = session._render_history(list(session._context.messages))
    images = [
        block
        for message in replayed
        for block in message.content
        if isinstance(block, ImageContent)
    ]
    assert images, "the compacted history must replay as images, not vanish"
    for image in images:
        assert image.mime_type == "image/png"
        assert base64.b64decode(image.data).startswith(PNG_MAGIC)
    await session.dispose()


def test_a_malformed_archive_degrades_to_the_text_summary():
    """A frames list that is not base64 is now a validation error, and the
    marker renderer's degrade path is what keeps that from ending the turn:
    the summary text still reaches the model, minus the frames."""
    marker = CustomMessage(
        custom_type="compaction_summary",
        details={
            "summary": "what happened earlier",
            "preserve_data": {"snapcompact": {"frames": ["not base64 at all!!"], "text": "t"}},
        },
    )

    rendered = _render_compaction_marker(marker)

    assert not any(isinstance(block, ImageContent) for block in rendered.content)
    texts = [block.text for block in rendered.content if isinstance(block, TextContent)]
    assert texts and "what happened earlier" in texts[0]


@pytest.mark.asyncio
async def test_a_text_only_model_compacts_into_a_language_summary(tmp_path):
    """The same history on a model that cannot read images back: a written
    summary, and NO archive — rendering frames it could never see would spend
    the work and lose the history."""
    stream = ScriptedStream(["reply", "reply", "reply", "the summary text"])
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session)

    outcome = await session.compact_now()

    assert outcome.ran is True
    assert outcome.strategy == "context-full"
    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert entries[0].payload.get("preserve_data") is None
    assert "the summary text" in entries[0].payload["summary"]
    await session.dispose()


@pytest.mark.asyncio
async def test_the_manual_strategy_is_the_automatic_one(tmp_path):
    """Neither trigger owns a strategy decision: both read
    ``Session._resolve_strategy``, so the split cannot drift between them."""
    stream = ScriptedStream(["reply"] * 4)
    vision = make_session(tmp_path / "v", stream, model=VISION_MODEL)
    text = make_session(tmp_path / "t", ScriptedStream(["reply"] * 4), model=TEXT_MODEL)

    settings = CompactionSettings(keep_recent_tokens=KEEP_RECENT)
    assert vision._resolve_strategy(settings) == "snapcompact"
    assert text._resolve_strategy(settings) == "context-full"

    await talk(vision, turns=3)
    outcome = await vision.compact_now()
    # The pass reports the strategy the shared resolver named, not one of its own.
    assert outcome.strategy == vision._resolve_strategy(settings)
    await vision.dispose()
    await text.dispose()


@pytest.mark.asyncio
async def test_the_receipt_reports_a_real_reduction(tmp_path):
    """tokens_before is the figure the gate acted on and tokens_after the
    estimate of the rebuilt history, so their difference is a saving a receipt
    can quote — and the end EVENT carries the same pair, which is what the TUI
    notice renders."""
    stream = ScriptedStream(["reply"] * 4)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=4)

    events: list[AgentEvent] = []
    session.subscribe(events.append)
    outcome = await session.compact_now()

    assert outcome.ran is True
    assert outcome.tokens_before > outcome.tokens_after > 0
    starts = [e for e in events if isinstance(e, CompactionStartEvent)]
    ends = [e for e in events if isinstance(e, CompactionEndEvent)]
    # Same vocabulary as the automatic pass, with `manual` naming the trigger.
    assert [e.reason for e in starts] == ["manual"]
    assert len(ends) == 1 and ends[0].success is True
    assert ends[0].reason == "manual"
    assert ends[0].strategy == "context-full"
    assert (ends[0].tokens_before, ends[0].tokens_after) == (
        outcome.tokens_before,
        outcome.tokens_after,
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_the_receipt_quotes_the_figure_the_gate_acted_on(tmp_path):
    """When the provider has reported a context size larger than the local
    estimate, the receipt's "before" is THAT figure — the one on the status
    band the user is comparing against. Quoting the local estimate instead
    made a pass that fired at a provider-reported 600k print "319.4k → …",
    which reads as the band and the receipt disagreeing about what happened.
    """
    from local_operator.harness.types import Usage

    stream = ScriptedStream(["reply"] * 4)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=4)
    # A provider reading far above anything the tiny fixture history could
    # estimate locally, as after a long real session.
    session._last_usage = Usage(input_tokens=1, context_tokens=600_080)

    outcome = await session.compact_now()

    assert outcome.ran is True
    assert outcome.tokens_before == 600_080
    await session.dispose()


@pytest.mark.asyncio
async def test_it_refuses_while_a_turn_is_running(tmp_path):
    """A running turn owns the message list; rebuilding it under the loop is how
    a tool call loses the result it is waiting for."""
    started = asyncio.Event()
    release = asyncio.Event()

    def stream(request, signal):
        async def gen():
            started.set()
            await release.wait()
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return gen()

    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    turn = asyncio.create_task(session.prompt("start something long"))
    await asyncio.wait_for(started.wait(), timeout=2)

    outcome = await session.compact_now()

    assert outcome.ran is False
    assert outcome.reason == "turn_running"
    assert "turn" in outcome.detail
    release.set()
    await turn
    await session.dispose()


@pytest.mark.asyncio
async def test_a_second_concurrent_pass_is_refused_and_a_prompt_says_why(tmp_path):
    """One pass at a time, and the pass HOLDS THE TURN LOCK while it runs.

    Both halves matter. A second pass would rewrite the history the first is
    summarizing; a prompt that started mid-pass would build its request from a
    message list being replaced underneath it. And a user who types during a
    compaction must not be told the session is "already streaming", because
    there is no turn — only this.
    """
    summarizing = asyncio.Event()
    release = asyncio.Event()
    session = make_session(tmp_path, ScriptedStream(["reply"] * 4), model=TEXT_MODEL)
    await talk(session, turns=3)

    async def slow_summary(system: str, prompt: str) -> str:
        summarizing.set()
        await release.wait()
        return "summary"

    session._one_shot_complete = slow_summary  # type: ignore[method-assign]
    first = asyncio.create_task(session.compact_now())
    await asyncio.wait_for(summarizing.wait(), timeout=2)

    second = await session.compact_now()
    assert second.ran is False
    assert second.reason == "already_running"

    with pytest.raises(RuntimeError, match="compaction is running"):
        await session.prompt("meanwhile")

    release.set()
    assert (await first).ran is True
    # The lock is handed back, so the next turn runs normally.
    await session.prompt("after the pass")
    await session.dispose()


@pytest.mark.asyncio
async def test_it_refuses_an_empty_conversation(tmp_path):
    """Nothing said yet: there is no history to summarize, and the refusal says
    so rather than emitting a start/end pair over nothing."""
    session = make_session(tmp_path, ScriptedStream([]), model=TEXT_MODEL)
    events: list[AgentEvent] = []
    session.subscribe(events.append)

    outcome = await session.compact_now()

    assert outcome.ran is False
    assert outcome.reason == "nothing_to_compact"
    assert "empty" in outcome.detail
    assert [e for e in events if e.type.startswith("compaction")] == []
    await session.dispose()


@pytest.mark.asyncio
async def test_it_refuses_a_context_that_is_all_recent(tmp_path):
    """One short turn against production's 20k keep-recent window: compacting
    four messages is worse than useless, and the refusal quotes both figures so
    the answer is checkable rather than a bare "no"."""
    stream = ScriptedStream(["reply"])
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())
    await talk(session, turns=1)

    outcome = await session.compact_now()

    assert outcome.ran is False
    assert outcome.reason == "nothing_to_compact"
    assert "20,000" in outcome.detail and "kept verbatim" in outcome.detail
    assert outcome.tokens_before == outcome.tokens_after > 0
    await session.dispose()


@pytest.mark.asyncio
async def test_it_refuses_a_second_pass_with_nothing_new_to_summarize(tmp_path):
    """Pressed twice in a row: the first pass ran, and the second has only the
    first pass's summary plus the kept window to work with. Re-compressing a
    summary spends a provider call for no headroom, so it is refused — and
    refused with a DIFFERENT sentence from the small-context case, because the
    reason is different."""
    stream = ScriptedStream(["reply"] * 5)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=3)

    first = await session.compact_now()
    assert first.ran is True

    second = await session.compact_now()
    assert second.ran is False
    assert second.reason == "nothing_to_compact"
    assert "already summarized" in second.detail
    # And exactly one compaction entry: the refusal wrote nothing.
    assert len([e for e in session._transcript.entries() if e.type == "compaction"]) == 1
    await session.dispose()


@pytest.mark.asyncio
async def test_it_refuses_when_compaction_is_switched_off(tmp_path):
    """``values.compaction.strategy: off`` is a configured decision, and a
    command that quietly overrode it would make the setting a lie. The refusal
    names the setting so the user can act on it."""
    session = make_session(
        tmp_path,
        ScriptedStream(["reply"] * 3),
        model=TEXT_MODEL,
        compaction_settings=CompactionSettings(strategy="off"),
    )
    await talk(session, turns=2)

    outcome = await session.compact_now()

    assert outcome.ran is False
    assert outcome.reason == "disabled"
    assert "config" in outcome.detail

    disabled = make_session(
        tmp_path / "off",
        ScriptedStream(["reply"] * 3),
        model=TEXT_MODEL,
        compaction_settings=CompactionSettings(enabled=False),
    )
    assert (await disabled.compact_now()).reason == "disabled"
    await session.dispose()
    await disabled.dispose()


@pytest.mark.asyncio
async def test_a_failed_pass_reports_the_failure_and_ends_the_notice(tmp_path):
    """The summarization call raising must not leave the UI on "compacting
    context…" forever: the end event still fires (unsuccessful), and the caller
    gets a reason to print."""
    stream = ScriptedStream(["reply"] * 3)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=3)

    async def boom(system: str, prompt: str) -> str:
        raise RuntimeError("provider exploded")

    session._one_shot_complete = boom  # type: ignore[method-assign]
    events: list[AgentEvent] = []
    session.subscribe(events.append)

    outcome = await session.compact_now()

    assert outcome.ran is False
    assert outcome.reason == "failed"
    assert "provider exploded" in outcome.detail
    ends = [e for e in events if isinstance(e, CompactionEndEvent)]
    assert len(ends) == 1 and ends[0].success is False
    # The context is untouched: a failed pass must not half-rewrite the history.
    assert not isinstance(session._context.messages[0], CustomMessage)
    await session.dispose()
