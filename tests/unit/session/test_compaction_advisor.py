"""The compaction advisor wired into a real session (BETA).

The unit-level rejection rules live in ``tests/unit/compaction/test_advisor.py``.
What is checked HERE is the wiring, because that is where a beta flag stops
being inert:

- with the flag OFF the session behaves identically, including that it never
  makes an advisor call at all;
- an accepted hint lowers the trigger and only the trigger, never below the
  advisor floor;
- a hint may only WIDEN the preserve window handed to ``find_cut_point``;
- every advisor failure (timeout, provider error, garbage answer) is a no-op
  the turn cannot observe;
- the anti-thrash cooldown and the non-negotiable kill switch fire.

Real compaction throughout, as in ``test_compaction_trigger.py``: only the
ruler and the advisor's provider call are substituted, never the gate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.compaction import api as compaction_api
from local_operator.compaction.advisor import CompactionHint
from local_operator.compaction.api import CompactionSettings
from local_operator.harness.types import (
    CompactionEndEvent,
    CustomMessage,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.session.protocol import CompactionOutcome
from local_operator.session.session import (
    Session,
    _CompactionPlan,
    _render_compaction_marker,
)
from local_operator.session.transcript import Transcript

BIG_MODEL = ModelSpec(provider="test", model_id="opus-like", context_window=1_000_000)
KEEP_RECENT = 40


class ScriptedStream:
    """Replies to a working turn, and to an advisor call with ``advice``.

    The advisor request is recognised by its ``tool_choice="none"`` plus the
    advisor marker in the last message, which is exactly how it differs on the
    wire — so a test that stops the advisor being SENT also stops this
    counting a call.
    """

    def __init__(self, advice: str | None = None) -> None:
        self.requests: list[object] = []
        self.advisor_calls = 0
        self.advice = advice

    def _is_advisor(self, request) -> bool:
        messages = getattr(request, "messages", []) or []
        if not messages:
            return False
        text = getattr(messages[-1], "text", "") or ""
        return "compaction advisor" in text

    def __call__(self, request, signal):
        self.requests.append(request)
        advisor = self._is_advisor(request)
        if advisor:
            self.advisor_calls += 1
        payload = self.advice if advisor and self.advice is not None else "reply"

        async def gen():
            yield StreamTextDelta(delta=payload)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream=None, **kwargs) -> Session:
    settings = kwargs.pop("compaction_settings", CompactionSettings(keep_recent_tokens=KEEP_RECENT))
    return Session(
        model=BIG_MODEL,
        stream_fn=stream or ScriptedStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings,
        **kwargs,
    )


def advisor_settings(**overrides) -> CompactionSettings:
    base = {
        "keep_recent_tokens": KEEP_RECENT,
        "advisor_enabled": True,
        "advisor_floor_tokens": 200_000,
        "advisor_trigger_tokens": 300_000,
        "advisor_every_n_turns": 1,
    }
    base.update(overrides)
    return CompactionSettings(**base)


def pin_measured_context(monkeypatch, tokens: int) -> None:
    monkeypatch.setattr(compaction_api, "messages_tokens_upper_bound", lambda messages: tokens)
    monkeypatch.setattr(compaction_api, "estimate_messages_tokens", lambda messages: tokens)


async def talk(session: Session, turns: int = 3) -> None:
    for index in range(turns):
        await session.prompt(f"question {index} " + "detail " * 30)


def usage_at(context_tokens: int) -> Usage:
    """Provider usage standing at a given context size.

    A real ``Usage`` rather than an ad-hoc stand-in: the advisor's spawn gate
    reads ``context_tokens`` off it, and a duck-typed double would let a field
    rename pass this suite while breaking production.
    """
    return Usage(context_tokens=context_tokens)


def refusal_reason(result: object) -> str | None:
    return getattr(result, "reason", None)


def as_plan(result: object) -> _CompactionPlan:
    assert isinstance(result, _CompactionPlan), f"expected a plan, got {result!r}"
    return result


def seed_hint(session: Session, *, compact_now: bool = True, preserve: int = KEEP_RECENT) -> None:
    """Park a validated hint the way a completed advisor call would.

    ``preserve`` defaults to the recency budget rather than something large:
    in production ``validate_hint`` measures the window from the REAL
    messages, so it can never exceed the history. A test that seeds a
    preserve window bigger than its own short history is asking to keep
    everything, and the honest answer to that is ``nothing_to_compact``.
    """
    session._advisor_hint = CompactionHint(
        preserve_from_id=session._context.messages[-1].id,
        preserve_tokens=preserve,
        compact_now=compact_now,
        confidence=0.9,
        reason="task boundary reached",
        turn_index=session._generation,
    )


# --- OFF BY DEFAULT -------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_never_calls_the_advisor(tmp_path, monkeypatch):
    """The default session must not make the call at all — not a cheap call,
    not a skipped one: none."""
    stream = ScriptedStream()
    session = make_session(tmp_path, stream=stream)
    await talk(session)
    pin_measured_context(monkeypatch, 500_000)
    session._maybe_spawn_advisor()
    await asyncio.sleep(0)
    assert stream.advisor_calls == 0
    assert session._advisor_hint is None
    await session.dispose()


@pytest.mark.asyncio
async def test_flag_off_trigger_is_unchanged(tmp_path, monkeypatch):
    """A context between the advisor floor and the real threshold does not
    compact with the beta off, even if a hint somehow existed."""
    session = make_session(tmp_path)
    await talk(session)
    seed_hint(session)
    pin_measured_context(monkeypatch, 400_000)
    refused = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(refused) == "below_threshold"
    await session.dispose()


# --- the trigger ----------------------------------------------------------


@pytest.mark.asyncio
async def test_accepted_hint_lowers_the_trigger(tmp_path, monkeypatch):
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session)
    pin_measured_context(monkeypatch, 400_000)

    # Without a hint the ordinary 600k trigger governs.
    assert refusal_reason(await session._plan_compaction(respect_threshold=True)) == (
        "below_threshold"
    )

    seed_hint(session)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    assert plan.advisor_hint is not None
    await session.dispose()


@pytest.mark.asyncio
async def test_hint_cannot_trigger_below_the_advisor_floor(tmp_path, monkeypatch):
    """200k floor: a confident "compact now" at 150k is refused. This is what
    bounds a wrong hint to an early pass rather than a treadmill."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session)
    seed_hint(session)
    pin_measured_context(monkeypatch, 150_000)
    refused = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(refused) == "below_threshold"
    await session.dispose()


@pytest.mark.asyncio
async def test_compact_now_false_does_not_lower_the_trigger(tmp_path, monkeypatch):
    """A hint that says "not yet" is advice too, and must not fire a pass."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session)
    seed_hint(session, compact_now=False)
    pin_measured_context(monkeypatch, 400_000)
    refused = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(refused) == "below_threshold"
    await session.dispose()


@pytest.mark.asyncio
async def test_hint_is_single_use(tmp_path, monkeypatch):
    """One hint is an opinion about one moment. Leaving it in place would keep
    the trigger lowered on every later gate — a stuck threshold."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session)
    seed_hint(session)
    pin_measured_context(monkeypatch, 400_000)
    assert isinstance(await session._plan_compaction(respect_threshold=True), _CompactionPlan)
    assert session._advisor_hint is None
    assert refusal_reason(await session._plan_compaction(respect_threshold=True)) == (
        "below_threshold"
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_stale_hint_is_discarded(tmp_path, monkeypatch):
    """A hint produced more than one advisory interval ago describes a
    conversation that has moved on."""
    session = make_session(tmp_path, compaction_settings=advisor_settings(advisor_every_n_turns=2))
    await talk(session)
    seed_hint(session)
    fresh = session._advisor_hint
    assert fresh is not None
    session._advisor_hint = CompactionHint(
        preserve_from_id=fresh.preserve_from_id,
        preserve_tokens=KEEP_RECENT,
        compact_now=True,
        confidence=0.9,
        reason="old news",
        turn_index=session._generation - 10,
    )
    pin_measured_context(monkeypatch, 400_000)
    assert refusal_reason(await session._plan_compaction(respect_threshold=True)) == (
        "below_threshold"
    )
    await session.dispose()


# --- the cut --------------------------------------------------------------


@pytest.mark.asyncio
async def test_hint_widens_the_preserve_window(tmp_path, monkeypatch):
    """The hint reaches ``find_cut_point`` as a keep-recent floor and nothing
    else. Captured rather than inferred, so a regression that stopped passing
    it would fail here."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session, turns=6)
    captured: list[int] = []
    real = compaction_api.find_cut_point

    def spy(messages, keep_recent_tokens):
        captured.append(keep_recent_tokens)
        return real(messages, keep_recent_tokens)

    monkeypatch.setattr(compaction_api, "find_cut_point", spy)
    pin_measured_context(monkeypatch, 400_000)
    widened = KEEP_RECENT * 3
    seed_hint(session, preserve=widened)
    await session._plan_compaction(respect_threshold=True)
    assert captured and captured[-1] >= widened
    await session.dispose()


@pytest.mark.asyncio
async def test_task_boundary_floor_applies_without_any_advisor(tmp_path, monkeypatch):
    """STEP 0 is independent of the beta: the cut is task-aware for every
    session, advisor or not, because it can only keep MORE history."""
    session = make_session(tmp_path)  # advisor OFF
    await talk(session, turns=6)
    captured: list[int] = []
    real = compaction_api.find_cut_point

    def spy(messages, keep_recent_tokens):
        captured.append(keep_recent_tokens)
        return real(messages, keep_recent_tokens)

    monkeypatch.setattr(compaction_api, "find_cut_point", spy)
    pin_measured_context(monkeypatch, 700_000)
    await session._plan_compaction(respect_threshold=True)
    assert captured and captured[-1] >= KEEP_RECENT
    await session.dispose()


# --- failure is always a no-op -------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "advice",
    [
        "not json at all",
        '```json\n{"preserve_from": "hallucinated", "compact_now": true, '
        '"confidence": 0.9, "reason": "x"}\n```',
        '```json\n{"compact_now": true, "confidence": 0.1, "reason": "x"}\n```',
        "",
    ],
)
async def test_bad_advice_produces_no_hint(tmp_path, monkeypatch, advice):
    stream = ScriptedStream(advice=advice)
    session = make_session(tmp_path, stream=stream, compaction_settings=advisor_settings())
    await talk(session)
    pin_measured_context(monkeypatch, 500_000)
    session._last_usage = usage_at(500_000)
    await session._run_advisor(session._compaction_settings, session._generation)
    assert session._advisor_hint is None
    assert session._advisor_in_flight is False
    await session.dispose()


@pytest.mark.asyncio
async def test_provider_failure_is_swallowed(tmp_path):
    """A turn is running alongside the advisor and must never learn it failed."""

    class Boom:
        def __call__(self, request, signal):
            async def gen():
                raise RuntimeError("provider exploded")
                yield  # pragma: no cover

            return gen()

    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session)
    session._stream_fn = Boom()
    session._last_usage = usage_at(500_000)
    await session._run_advisor(session._compaction_settings, session._generation)
    assert session._advisor_hint is None
    assert session._advisor_in_flight is False
    await session.dispose()


@pytest.mark.asyncio
async def test_timeout_is_swallowed_and_releases_the_latch(tmp_path, monkeypatch):
    session = make_session(tmp_path, compaction_settings=advisor_settings(advisor_timeout_s=0.01))
    await talk(session)

    async def never(*_a, **_k):
        await asyncio.sleep(10)
        return ""

    monkeypatch.setattr(session, "advise_compaction", never)
    session._last_usage = usage_at(500_000)
    await session._run_advisor(session._compaction_settings, session._generation)
    assert session._advisor_hint is None
    assert session._advisor_in_flight is False
    await session.dispose()


# --- rate limiting --------------------------------------------------------


@pytest.mark.asyncio
async def test_only_one_call_in_flight(tmp_path, monkeypatch):
    """Skip, never queue: a second boundary while a call is outstanding does
    not stack another."""
    stream = ScriptedStream()
    session = make_session(tmp_path, stream=stream, compaction_settings=advisor_settings())
    await talk(session)
    session._last_usage = usage_at(500_000)
    session._advisor_in_flight = True
    session._maybe_spawn_advisor()
    assert session._advisor_calls == 0
    await session.dispose()


@pytest.mark.asyncio
async def test_max_calls_ceiling(tmp_path):
    session = make_session(tmp_path, compaction_settings=advisor_settings(advisor_max_calls=1))
    await talk(session)
    session._last_usage = usage_at(500_000)
    session._advisor_calls = 1
    assert session._advisor_settings() is None
    await session.dispose()


@pytest.mark.asyncio
async def test_below_advisor_trigger_no_call(tmp_path):
    """Under ``advisor_trigger_tokens`` there is no decision to inform."""
    stream = ScriptedStream()
    session = make_session(tmp_path, stream=stream, compaction_settings=advisor_settings())
    await talk(session)
    session._last_usage = usage_at(100_000)
    session._maybe_spawn_advisor()
    assert session._advisor_calls == 0
    await session.dispose()


# --- anti-thrash ----------------------------------------------------------


@pytest.mark.asyncio
async def test_cooldown_after_an_advisory_pass(tmp_path, monkeypatch):
    session = make_session(
        tmp_path, compaction_settings=advisor_settings(advisor_cooldown_turns=50)
    )
    await talk(session, turns=6)
    seed_hint(session)
    pin_measured_context(monkeypatch, 400_000)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    outcome = await session._run_compaction(plan, reason="context-window")
    assert outcome.ran is True
    session._settle_advisor(plan, outcome)
    assert session._advisor_cooldown_until > session._generation
    assert session._advisor_settings() is None  # suppressed
    await session.dispose()


@pytest.mark.asyncio
async def test_kill_switch_on_a_pass_that_does_not_clear_the_band(tmp_path, monkeypatch):
    """Non-negotiable: an advisory pass that fails to clear
    ``RECOVERY_BAND * advisor_floor`` disables the advisor for the session.
    A feature that can spend money in a loop must fail closed."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session, turns=6)
    seed_hint(session)
    pin_measured_context(monkeypatch, 400_000)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    # Residual well above 0.8 * 200k: the advice reclaimed nothing.
    bad = CompactionOutcome(ran=True, strategy="snapcompact", tokens_after=190_000)
    session._settle_advisor(plan, bad)
    assert session._advisor_disabled is True
    assert session._advisor_settings() is None
    await session.dispose()


@pytest.mark.asyncio
async def test_ordinary_pass_never_arms_the_kill_switch(tmp_path, monkeypatch):
    """A size-triggered pass is not the advisor's business."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session, turns=6)
    pin_measured_context(monkeypatch, 700_000)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    assert plan.advisor_hint is None
    session._settle_advisor(
        plan, CompactionOutcome(ran=True, strategy="snapcompact", tokens_after=690_000)
    )
    assert session._advisor_disabled is False
    await session.dispose()


# --- observability --------------------------------------------------------


@pytest.mark.asyncio
async def test_receipt_explains_an_advisory_pass(tmp_path, monkeypatch):
    """A pass firing below the configured threshold owes the user a reason, or
    the receipt reads as the trigger misfiring."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session, turns=6)
    events: list[object] = []
    session.subscribe(events.append)
    seed_hint(session)
    pin_measured_context(monkeypatch, 400_000)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    await session._run_compaction(plan, reason="context-window")
    ends = [e for e in events if isinstance(e, CompactionEndEvent) and e.success]
    assert ends and ends[-1].detail == "advisor: task boundary reached"
    await session.dispose()


@pytest.mark.asyncio
async def test_ordinary_pass_carries_no_detail(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    await talk(session, turns=6)
    events: list[object] = []
    session.subscribe(events.append)
    pin_measured_context(monkeypatch, 700_000)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    await session._run_compaction(plan, reason="context-window")
    ends = [e for e in events if isinstance(e, CompactionEndEvent) and e.success]
    assert ends and ends[-1].detail is None
    await session.dispose()


# --- resume after an ADVISOR-triggered snapcompact pass --------------------
#
# All seven real compaction passes on the session that motivated this feature
# used the snapcompact strategy, and a prior bug on that path's image-replay
# destroyed history outright. The advisor multiplies pass frequency roughly
# 2-3x, so it multiplies exposure to exactly that class of bug by the same
# factor. These two tests are therefore not an afterthought: they are the
# reason it is safe to raise the pass rate at all.

VISION_MODEL = ModelSpec(provider="test", model_id="sees", context_window=1_000_000)

#: A distinctive user constraint. If an advisory pass ever paraphrases it away,
#: the substring stops being present and the assertions fail.
CONSTRAINT = "NEVER touch billing.py"

#: Turn count and per-turn filler sized so ``compact_to_archive`` actually
#: RENDERS PNG frames instead of taking its archive-text early return.
#:
#: This is load-bearing, not arbitrary. ``compact_to_archive`` returns
#: ``Archive(frames=[], ...)`` when the serialized history is under
#: ``2 * _edge_chars_for(shape, context_window)``, so a small fixture exercises
#: the text path while asserting ``strategy == "snapcompact"`` — which is how
#: the first version of these tests claimed to cover image replay while
#: measuring ``frames: 0`` (agent review round 1, major-2). Measured at this
#: size: 6 frames and ~155k chars of archive text. Shrinking it silently
#: returns the tests to the text-only path, so ``_assert_frames_rendered``
#: fails loudly rather than letting that pass unnoticed.
FRAME_TURNS = 12
FRAME_WORDS = 2000


def _snapcompact_payload(session: Session) -> dict[str, Any]:
    """The persisted snapcompact archive for the first compaction entry."""
    entries = [e for e in session._transcript.entries() if e.type == "compaction"]
    assert entries, "no compaction entry was written"
    preserve = entries[0].payload.get("preserve_data") or {}
    payload = preserve.get("snapcompact")
    assert isinstance(payload, dict), "the pass did not take the snapcompact path"
    return payload


def _assert_frames_rendered(session: Session) -> int:
    """Assert the pass rendered real PNG frames, and return how many.

    The whole justification for these tests is the image-replay path, so a
    fixture that quietly falls back to archive text has to fail here rather
    than pass as coverage it does not provide.
    """
    payload = _snapcompact_payload(session)
    frames = payload.get("frames") or []
    assert frames, (
        "the snapcompact pass rendered NO frames, so this test is exercising the "
        "archive-text path and not the image replay it exists to pin"
    )
    return len(frames)


async def _talk_to_frame_size(session: Session) -> None:
    """Open with the constraint, then bury it under enough history to image."""
    await session.prompt(f"{CONSTRAINT} " + "detail " * 30)
    for index in range(FRAME_TURNS):
        await session.prompt(f"question {index} " + "detail " * FRAME_WORDS)


@pytest.mark.asyncio
async def test_resume_after_an_advisory_snapcompact_pass(tmp_path, monkeypatch):
    """A session resumed after an ADVISOR-triggered snapcompact pass replays
    its history, constraint intact.

    Same round trip as the vision-model test in
    ``test_compaction_preserves_user_turns.py``, but with the pass fired by the
    advisor below the configured threshold rather than by size.
    """
    stream = ScriptedStream()
    session = Session(
        model=VISION_MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=advisor_settings(),
    )
    await _talk_to_frame_size(session)

    seed_hint(session)
    pin_measured_context(monkeypatch, 400_000)
    plan = as_plan(await session._plan_compaction(respect_threshold=True))
    assert plan.advisor_hint is not None, "the pass must be the advisor's, not a size trigger"
    assert plan.strategy == "snapcompact"
    outcome = await session._run_compaction(plan, reason="context-window")
    assert outcome.ran is True
    # The point of this test: real PNG frames, not the archive-text fallback.
    frame_count = _assert_frames_rendered(session)
    assert frame_count > 0

    directory = session._transcript.directory
    await session.dispose()

    replayed = Transcript(directory).build_llm_history()
    assert replayed, "resume after an advisory snapcompact pass replayed nothing"
    texts = " ".join(m.text for m in replayed if isinstance(m, Message) and m.text)
    assert CONSTRAINT in texts, "advisory snapcompact resume dropped the user constraint"

    # The IMAGE-REPLAY assertion, which is the reason this test exists: the
    # frames the pass rendered must come back as real image blocks on resume.
    # A prior bug on exactly this path double-encoded the base64 and shipped
    # unreadable PNGs, and the advisor multiplies pass frequency 2-3x. Counting
    # the blocks proves the archive was DECODED, not merely stored.
    #
    # Replay is two stages: ``build_llm_history`` returns the marker as a
    # ``CustomMessage`` still carrying ``preserve_data``, and
    # ``_render_compaction_marker`` is what turns the archive into image
    # blocks for the wire. Asserting on the first stage alone would pass
    # against an archive that never decodes, so the render is driven here.
    markers = [
        message
        for message in replayed
        if isinstance(message, CustomMessage) and message.custom_type == "compaction_summary"
    ]
    assert markers, "resume produced no compaction marker to replay"
    rendered = _render_compaction_marker(markers[0])
    replayed_images = [
        block for block in (rendered.content or []) if isinstance(block, ImageContent)
    ]
    assert len(replayed_images) == frame_count, (
        f"resume replayed {len(replayed_images)} image blocks for {frame_count} archived "
        "frames — the image-replay path did not round trip"
    )
    # Each frame must be non-empty PNG data: a present-but-empty block is the
    # shape a decode failure leaves behind.
    for image in replayed_images:
        assert image.mime_type == "image/png"
        assert image.data, "an archived frame replayed with empty data"


@pytest.mark.asyncio
async def test_two_advisory_snapcompact_passes_still_replay(tmp_path, monkeypatch):
    """Frequency is the advisor's whole risk profile, so the multi-generation
    case is pinned too: a second advisory pass over an already-compacted
    history must not lose the first generation's constraint."""
    session = Session(
        model=VISION_MODEL,
        stream_fn=ScriptedStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=advisor_settings(advisor_cooldown_turns=0),
    )
    await _talk_to_frame_size(session)

    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)
    first = as_plan(await session._plan_compaction(respect_threshold=True))
    assert (await session._run_compaction(first, reason="context-window")).ran is True
    _assert_frames_rendered(session)

    await talk(session, turns=5)
    seed_hint(session)
    second = as_plan(await session._plan_compaction(respect_threshold=True))
    assert (await session._run_compaction(second, reason="context-window")).ran is True

    directory = session._transcript.directory
    await session.dispose()

    replayed = Transcript(directory).build_llm_history()
    texts = " ".join(m.text for m in replayed if isinstance(m, Message) and m.text)
    assert CONSTRAINT in texts, "the second advisory pass dropped the first generation's constraint"


# --- the REAL turn-boundary path (agent review round 1, blocker-1) ---------
#
# Every other wiring test in this file calls `_maybe_spawn_advisor()` directly
# or seeds `_advisor_hint`. That is exactly how the original defect hid: the
# spawn sat BELOW the mid-turn cheap pre-gate, which returns as soon as the
# provider figure fails the ORDINARY trigger, so in production the advisor was
# only ever consulted once the context had already passed the line it exists to
# pull down. Measured before the fix: ctx 350k/400k/500k/590k -> 0 advisor
# calls on a 600k trigger; 650k -> 1.
#
# So these tests drive `_on_turn_end` (the real boundary hook the tool loop
# calls) and assert on the SPAWN, never on the helper. A regression that
# reintroduces an early return above the spawn fails here and nowhere else.


class BoundaryStream:
    """A stream that answers a working turn, and a tool-calling turn on demand.

    Only used to give `_on_turn_end` a realistic message list; the advisor's
    own request is recognised and counted the same way `ScriptedStream` does.
    """

    def __init__(self) -> None:
        self.advisor_calls = 0

    def __call__(self, request, signal):
        messages = getattr(request, "messages", []) or []
        text = (getattr(messages[-1], "text", "") if messages else "") or ""
        if "compaction advisor" in text:
            self.advisor_calls += 1

        async def gen():
            yield StreamTextDelta(delta="reply")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


async def _drive_boundary(session: Session, context_tokens: int) -> None:
    """Run the REAL mid-turn hook at a given provider-reported context size.

    `_on_turn_end` reads the provider figure off the last message's usage, the
    way the tool loop hands it over, so the usage rides a real assistant
    message rather than being poked onto the session.
    """
    assistant = Message.assistant("mid-run reply")
    assistant.usage = usage_at(context_tokens)
    await session._on_turn_end([*session._context.messages, assistant])


@pytest.mark.asyncio
@pytest.mark.parametrize("context_tokens", [350_000, 400_000, 500_000, 590_000])
async def test_advisor_fires_in_its_own_band_through_the_real_boundary(
    tmp_path, monkeypatch, context_tokens
):
    """THE blocker-1 regression: between `advisor_trigger_tokens` (300k) and
    the ordinary trigger (600k), a real turn boundary must reach the advisor.

    These four sizes are the exact ones that produced zero calls before the
    fix. Driving `_on_turn_end` rather than `_maybe_spawn_advisor` is the
    entire point: the helper was always reachable, the boundary was not.
    """
    stream = BoundaryStream()
    session = make_session(tmp_path, stream=stream, compaction_settings=advisor_settings())
    await talk(session, turns=3)
    pin_measured_context(monkeypatch, context_tokens)

    await _drive_boundary(session, context_tokens)
    # The call is spawned off-loop and nothing awaits it, so give the task a
    # turn of the loop to run — the same shape the production path has.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if stream.advisor_calls:
            break

    assert stream.advisor_calls == 1, (
        f"the advisor was never consulted at {context_tokens:,} tokens, inside its own "
        "operating band — the mid-turn pre-gate is short-circuiting above the spawn again"
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_a_landed_hint_survives_the_mid_turn_pre_gate(tmp_path, monkeypatch):
    """The spawn alone is not enough: the pre-gate must also let a boundary
    through once a usable hint exists, or the hint is spawned, lands, and is
    then unreachable because the gate returned before the plan gate could read
    it. Asserted as a real compaction pass at 400k, below the 600k trigger.
    """
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session, turns=6)
    events: list[object] = []
    session.subscribe(events.append)

    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)
    await _drive_boundary(session, 400_000)

    ends = [e for e in events if isinstance(e, CompactionEndEvent) and e.success]
    assert ends, "a usable hint did not survive the mid-turn pre-gate at 400k"
    assert ends[-1].detail == "advisor: task boundary reached"
    await session.dispose()


@pytest.mark.asyncio
async def test_the_pre_gate_peek_does_not_consume_the_hint(tmp_path, monkeypatch):
    """The pre-gate PEEKS. If it consumed the hint, the plan gate one step
    later would find nothing and refuse the pass as below_threshold — the
    defect would look fixed at the gate and still not compact."""
    session = make_session(tmp_path, compaction_settings=advisor_settings())
    await talk(session, turns=3)
    pin_measured_context(monkeypatch, 400_000)
    seed_hint(session)

    assert session._has_pending_advisory(session._compaction_settings) is True
    # Peeking twice is still true: it is read-only.
    assert session._has_pending_advisory(session._compaction_settings) is True
    assert session._advisor_hint is not None
    # Consuming clears it.
    assert session._take_advisor_hint(session._compaction_settings) is not None
    assert session._advisor_hint is None
    await session.dispose()


@pytest.mark.asyncio
async def test_flag_off_boundary_never_spawns(tmp_path, monkeypatch):
    """Moving the spawn above the pre-gate must not make a default session pay
    anything: with the beta off, a boundary in the same band spawns nothing."""
    stream = BoundaryStream()
    session = make_session(tmp_path, stream=stream)  # advisor OFF
    await talk(session, turns=3)
    pin_measured_context(monkeypatch, 400_000)
    await _drive_boundary(session, 400_000)
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert stream.advisor_calls == 0
    await session.dispose()


@pytest.mark.asyncio
async def test_hint_preserve_window_is_capped(tmp_path, monkeypatch):
    """major-1: an uncapped hint turned a mandatory pass into
    ``nothing_to_compact`` — the "never compact" failure the cap exists to
    prevent, reached through the guard meant to make hints safe.

    A SMALL-window model is used deliberately. The cap is
    ``max(keep_recent, threshold // 2)``, so it is only meaningful relative to
    the real history: on a 1M window the cap is 300k and any fixture small
    enough to build in a test is swallowed by it whatever the clamp does.
    Here the window is 2,500, so the trigger is 2,000 and the cap
    (``max(keep_recent, threshold // 2)``) is 1,000 against a ~2,460-token
    history — roughly the proportion a 4M-token production session has against
    its own 300k cap, and small enough that ``find_cut_point`` still has
    something to summarize. Without the clamp the 800,340-token hint asks to
    preserve 325x the entire history and the pass is refused outright.
    """
    small = ModelSpec(provider="test", model_id="small", context_window=2_500)
    session = Session(
        model=small,
        stream_fn=ScriptedStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=advisor_settings(advisor_floor_tokens=500),
    )
    for index in range(12):
        await session.prompt(f"question {index} " + "detail " * 200)

    captured: list[int] = []
    real = compaction_api.find_cut_point

    def spy(messages, keep_recent_tokens):
        captured.append(keep_recent_tokens)
        return real(messages, keep_recent_tokens)

    monkeypatch.setattr(compaction_api, "find_cut_point", spy)
    # Above the ordinary trigger (2,000 on a 2,500 window): a pass MUST happen.
    pin_measured_context(monkeypatch, 2_400)
    # A legal but absurdly wide hint — 100x the cap, exceeding the whole window.
    seed_hint(session, preserve=800_340)

    planned = await session._plan_compaction(respect_threshold=True)
    cap = session._advisor_floor_cap(session._compaction_settings)
    assert captured, "find_cut_point was never reached"
    assert (
        captured[-1] <= cap
    ), f"the hint bypassed the preserve-window cap: {captured[-1]:,} > {cap:,}"
    assert isinstance(
        planned, _CompactionPlan
    ), f"a wide hint suppressed a mandatory pass: {planned!r}"
    await session.dispose()
