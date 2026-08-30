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
import dataclasses

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
from local_operator.session.session import (
    Session,
    _CompactionPlan,
    _render_compaction_marker,
)
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
    the summary text still reaches the model, minus the frames — plus the
    archive's plain-text edges. The snapcompact summary is reading
    instructions for the frames, not a digest, so without the salvage the
    fallback replayed a caption describing images that are not there while
    the actual history vanished."""
    marker = CustomMessage(
        custom_type="compaction_summary",
        details={
            "summary": "what happened earlier",
            "preserve_data": {
                "snapcompact": {
                    "frames": ["not base64 at all!!"],
                    "text": "t",
                    "text_head": "OLDEST: the user asked about parsers",
                    "text_tail": "NEWEST: the fix landed in commit abc123",
                }
            },
        },
    )

    rendered = _render_compaction_marker(marker)

    assert not any(isinstance(block, ImageContent) for block in rendered.content)
    texts = [block.text for block in rendered.content if isinstance(block, TextContent)]
    assert texts and "what happened earlier" in texts[0]
    # The real transcript edges survive the frame corruption.
    assert "the user asked about parsers" in texts[0]
    assert "the fix landed in commit abc123" in texts[0]


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
    notice renders.

    The assistant replies carry real bulk: user turns are now preserved
    VERBATIM across a pass (never summarized), so the compressible content is
    the ASSISTANT/tool side. A fixture whose only bulk was user filler would
    show no reduction after the fix — correctly, because there is nothing left
    to compress — which is a property of that fixture, not of the receipt."""
    stream = ScriptedStream(["assistant reply " * 60] * 4)
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

    The "after" is then the SAME provider figure scaled by the ratio the
    history actually shrank by, so both ends of the receipt stay on the
    provider's ruler. It may not be reached by subtracting a locally-measured
    saving from a provider total: the two rulers diverge by ~1.65-1.73x on
    Anthropic, so that subtraction understated every pass by ~140k tokens
    (see the arithmetic's comment in ``Session._run_compaction``).
    """
    from local_operator.harness.types import Usage

    # Bulky assistant replies so there is summarizable content once the user
    # turns are preserved verbatim (see the sibling reduction test).
    stream = ScriptedStream(["assistant reply " * 60] * 4)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=4)
    # A provider reading far above anything the tiny fixture history could
    # estimate locally, as after a long real session.
    session._last_usage = Usage(input_tokens=1, context_tokens=600_080)

    outcome = await session.compact_now()

    assert outcome.ran is True
    assert outcome.tokens_before == 600_080
    # The receipt is PROPORTIONAL, so the after-figure is the provider's
    # before-figure times the fraction of history the pass kept. Asserted as
    # the ratio rather than a literal because the fixture's exact token counts
    # are an implementation detail of the estimator.
    ratio = outcome.tokens_after / outcome.tokens_before
    assert 0 < ratio < 1
    # A pass that compressed bulky assistant replies must show a real
    # reduction; the pre-fix subtraction form put this at >0.98 (the whole
    # provider overhead surviving as if untouched) and hid the saving.
    assert ratio < 0.75
    await session.dispose()


@pytest.mark.asyncio
async def test_the_receipt_scales_the_provider_figure_by_the_history_ratio(tmp_path):
    """The exact arithmetic, pinned: ``after == round(before * ha / hb)``.

    The sibling test above asserts the PROPERTY (a real reduction on the
    provider's ruler); this one pins the FORM, because the two candidate
    formulas differ by ~140k tokens on real data and a property test passes
    under both. ``tokens_before`` on the plan is the pure local estimate over
    the pre-pass history, and the estimator is re-run here over the rebuilt
    history, so both sides of the ratio come from the same ruler the
    implementation uses.
    """
    from local_operator.compaction.tokens import estimate_messages_tokens
    from local_operator.harness.types import Usage

    stream = ScriptedStream(["assistant reply " * 60] * 4)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=4)
    session._last_usage = Usage(input_tokens=1, context_tokens=600_080)

    # The plan carries the local before-estimate the formula divides by.
    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    history_before = plan.tokens_before

    outcome = await session._run_compaction(plan, reason="manual")

    history_after = estimate_messages_tokens(session._render_for_compaction())
    expected = max(history_after, round(600_080 * history_after / history_before))
    assert outcome.tokens_after == expected
    await session.dispose()


@pytest.mark.asyncio
async def test_the_receipt_survives_an_empty_before_history(tmp_path):
    """``history_before == 0`` must not divide by zero.

    The zero-guard makes the proportional form total. It is not reachable
    through a real pass (a plan needs summarizable history to exist at all),
    so the branch is exercised directly on the arithmetic's own inputs — a
    guard nothing can reach is still a guard a future edit can break.
    """
    from local_operator.harness.types import Usage

    stream = ScriptedStream(["assistant reply " * 60] * 4)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=4)
    session._last_usage = Usage(input_tokens=1, context_tokens=600_080)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    # A plan whose local before-estimate is zero: degenerate, but the division
    # must still answer rather than raise. The plan is frozen, so this is a
    # copy rather than a mutation.
    plan = dataclasses.replace(plan, tokens_before=0)

    outcome = await session._run_compaction(plan, reason="manual")

    assert outcome.ran is True
    # No divisor means no ratio, so the receipt reports the size it already
    # knew about rather than inventing one. That is the same answer the
    # shrink guard gives, and it is deliberately NOT zero and not a product.
    assert outcome.tokens_after == plan.context_tokens
    # The invariant that matters everywhere: a receipt never reports growth.
    assert outcome.tokens_after <= outcome.tokens_before
    await session.dispose()


@pytest.mark.asyncio
async def test_the_receipt_is_bounded_by_the_figure_it_reduces_from(tmp_path):
    """``tokens_after`` may never exceed ``tokens_before``.

    The old subtraction form got this for free: ``context_tokens - max(0,
    saved)`` cannot exceed ``context_tokens``. A PRODUCT can, and losing that
    bound is how the proportional form came to report a compaction that grew
    the context (agent review round 2, blocker-1). A receipt above its own
    before-figure is not merely wrong, it is invisible — ``compaction_receipt``
    drops both numbers when ``after >= before`` and prints a bare "context
    compacted", which is the "it did nothing" frame ``compact_now`` exists to
    stop showing.

    Driven with a provider figure BELOW the local estimate, which forces the
    product upward relative to the history; the clamp is the term that has to
    decide the answer.
    """
    stream = ScriptedStream(["assistant reply " * 60] * 4)
    session = make_session(tmp_path, stream, model=TEXT_MODEL)
    await talk(session, turns=4)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    plan = dataclasses.replace(plan, context_tokens=10)

    outcome = await session._run_compaction(plan, reason="manual")

    assert outcome.tokens_after <= outcome.tokens_before
    assert outcome.tokens_after == 10
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


@pytest.mark.asyncio
async def test_the_snapcompact_frame_correction_is_priced_once(tmp_path):
    """The frame correction is added AFTER the ratio, and never above the
    figure the receipt is reporting a reduction FROM.

    Two properties, because round 2 showed the first one alone is not enough.

    PLACEMENT (round 1, major-2): the proportional form is only sound while
    numerator and denominator are on the same LOCAL ruler. The snapcompact
    path corrects the after-figure by the difference between the provider's
    visual-token price for a replayed archive frame and the local flat
    ``IMAGE_TOKEN_ESTIMATE`` — a PROVIDER-scale addend. Folding it in before
    dividing inflated the ratio and multiplied that addend by the provider
    total a second time.

    MAGNITUDE (round 2, blocker-1): pinning placement against a computed
    identity passes at ANY magnitude, so it could not see that the result had
    no upper bound. On snapcompact ``history_after`` routinely EXCEEDS
    ``history_before`` — the pass replaces history with verbatim edges plus
    archive text plus images the local ruler prices flat, so the saving is
    real on the provider ruler while the local estimate of the replacement
    grows — and the receipt reported a compaction that GREW the context
    (measured 70,888 -> 111,594 on a real over-threshold pass). This asserts
    the invariant directly on a real snapcompact pass.
    """
    from local_operator.compaction.snapcompact import frame_token_estimate_for
    from local_operator.compaction.tokens import estimate_messages_tokens
    from local_operator.harness.types import Usage
    from local_operator.session.session import IMAGE_TOKEN_ESTIMATE

    stream = ScriptedStream(["reply"] * 8)
    session = make_session(tmp_path, stream, model=VISION_MODEL)
    # Past the plain-text edges (2 x HQ_EDGE_FRAMES x 13 916 chars), so the
    # middle is actually imaged and the correction has frames to price. A
    # smaller history stores an empty frames list and tests nothing.
    for index in range(6):
        await session.prompt(f"question {index} " + "detail " * 3000)
    session._last_usage = Usage(input_tokens=1, context_tokens=600_080)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    history_before = plan.tokens_before

    outcome = await session._run_compaction(plan, reason="manual")
    assert outcome.strategy == "snapcompact"

    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    frames = entries[-1].payload["preserve_data"]["snapcompact"].get("frames") or []
    per_frame = frame_token_estimate_for(session._model.provider, session._model.model_id)
    correction = len(frames) * (per_frame - IMAGE_TOKEN_ESTIMATE)
    assert correction > 0, "fixture produced no frames — the invariant is untested"

    # THE invariant. A receipt above its own before-figure is dropped entirely
    # by ``compaction_receipt`` and paints an over-window figure on the band.
    assert outcome.tokens_after <= outcome.tokens_before

    # The uncorrected local estimate of the rebuilt history: the ratio's real
    # numerator, and the reason this fixture is interesting — snapcompact does
    # not shrink it.
    history_after = estimate_messages_tokens(session._render_for_compaction())
    if history_after < history_before:
        expected = round(600_080 * history_after / history_before) + correction
        assert outcome.tokens_after == min(600_080, expected)
    else:
        # No ratio worth applying: the receipt reports the size it already
        # knew rather than a product the measurement contradicts.
        assert outcome.tokens_after == 600_080

    await session.dispose()


@pytest.mark.asyncio
async def test_a_snapcompact_receipt_never_reports_growth(tmp_path):
    """The blocker-1 regression, pinned on the path that produced it.

    Swept across history sizes, because the defect's magnitude depended on how
    far the local estimate of the archive-plus-edges replacement overshot the
    original: at 14 turns the shipped-at-round-1 code reported 70,888 ->
    111,594, and larger histories overshot further. Every size must satisfy
    the same invariant, and none may print a bare receipt because its numbers
    were dropped.

    Uses a REAL provider ratio (this PR's measured opus-5 slope of 1.685)
    rather than a pinned constant, so the fixture reflects the ruler
    divergence that makes the multiplication dangerous in the first place.
    """
    from local_operator.compaction.tokens import estimate_messages_tokens
    from local_operator.harness.types import Usage
    from local_operator.tui.app import compaction_receipt
    from local_operator.tui.events import CompactionEnded

    for turns in (8, 14, 20):
        stream = ScriptedStream(["reply"] * (turns * 2))
        session = make_session(tmp_path / f"t{turns}", stream, model=VISION_MODEL)
        for index in range(turns):
            await session.prompt(f"question {index} " + "detail " * 3000)

        local = estimate_messages_tokens(session._render_for_compaction())
        session._last_usage = Usage(input_tokens=1, context_tokens=round(local * 1.685))

        plan = await session._plan_compaction(respect_threshold=False)
        assert isinstance(plan, _CompactionPlan), f"{turns} turns: {plan!r}"
        outcome = await session._run_compaction(plan, reason="threshold")

        assert outcome.strategy == "snapcompact"
        assert outcome.tokens_after <= outcome.tokens_before, (
            f"{turns} turns: receipt reports GROWTH, "
            f"{outcome.tokens_before:,} -> {outcome.tokens_after:,}"
        )
        # And never above the model's own window, which is what the status
        # band paints from this figure.
        assert outcome.tokens_after <= max(VISION_MODEL.context_window, outcome.tokens_before)

        rendered = compaction_receipt(
            CompactionEnded(
                reason="threshold",
                success=True,
                strategy=outcome.strategy,
                tokens_before=outcome.tokens_before,
                tokens_after=outcome.tokens_after,
            )
        )
        assert "context compacted" in rendered
        await session.dispose()


@pytest.mark.asyncio
async def test_a_snapcompact_pass_that_shrinks_takes_the_proportional_arm(tmp_path):
    """The ratio arm, on snapcompact, exercised rather than merely reachable.

    ``test_the_snapcompact_frame_correction_is_priced_once`` branches at
    runtime on ``history_after < history_before``, and its fixture is small
    enough that only the ``else`` (fallback) arm ever runs — so the
    proportional-plus-correction arm, which ALL TEN real passes of the measured
    session take, was asserted only by a dormant ``if`` (agent review round 3,
    minor-3).

    The bulk has to be on the ASSISTANT side. User turns are preserved
    verbatim across a pass, so a fixture whose weight is in its prompts leaves
    nothing summarizable and lands at ``history_after == history_before - 1``
    — which satisfies a naive shrink assertion while exercising no ratio at
    all. With assistant-side bulk this compacts 0.27x, the same shape as the
    real passes (1.8x-8.8x), and the result lands well below the clamp so the
    ratio is what decides it.
    """
    from local_operator.compaction.snapcompact import frame_token_estimate_for
    from local_operator.compaction.tokens import estimate_messages_tokens
    from local_operator.harness.types import Usage
    from local_operator.session.session import IMAGE_TOKEN_ESTIMATE

    stream = ScriptedStream(["assistant reply " * 1200] * 60)
    session = make_session(tmp_path, stream, model=VISION_MODEL)
    for index in range(30):
        await session.prompt(f"question {index}")
    session._last_usage = Usage(input_tokens=1, context_tokens=600_080)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    history_before = plan.tokens_before

    outcome = await session._run_compaction(plan, reason="manual")
    assert outcome.strategy == "snapcompact"

    history_after = estimate_messages_tokens(session._render_for_compaction())
    # The whole point of this fixture: if it stops shrinking, the test has
    # silently gone back to covering the fallback and must be resized.
    assert history_after < history_before, (
        f"fixture no longer reaches the proportional arm "
        f"({history_before:,} -> {history_after:,}); resize it rather than "
        "deleting this assertion"
    )

    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    frames = entries[-1].payload["preserve_data"]["snapcompact"].get("frames") or []
    per_frame = frame_token_estimate_for(session._model.provider, session._model.model_id)
    correction = len(frames) * (per_frame - IMAGE_TOKEN_ESTIMATE)

    # The ratio runs on the UNCORRECTED local figures, the correction is added
    # once afterwards, and the whole result is clamped.
    scaled = max(history_after, round(600_080 * history_after / history_before))
    assert outcome.tokens_after == min(600_080, scaled + correction)
    assert outcome.tokens_after <= outcome.tokens_before
    # The clamp must NOT be what decides this case, or the ratio arm is
    # unobservable and a regression in it would pass unnoticed.
    assert scaled + correction < 600_080, (
        "the clamp is masking the ratio: this fixture no longer tests the " "proportional arm"
    )
    await session.dispose()
