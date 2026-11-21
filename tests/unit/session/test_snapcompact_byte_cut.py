"""The byte-aware cut: a pass fired by the BYTE trigger must land inside the band.

The defect. ``find_cut_point`` keeps ``keep_recent_tokens`` (20k) of history and
the token ruler prices every image at a flat 1,200, so on a screenshot history
that window is ~16 frames -- ~15 MB of 1280x720 PNG -- against a 16 MB byte
trigger. A pass fired by bytes landed with a fraction of a frame of headroom and
fired again one or two tool groups later (measured live: 62 passes in a
295-request run, bursts of six in a row, the request after each pass reading
27.5% cache-read against ~80% in steady state, ~24% of the run's spend).

These tests drive the REAL gate (``Session._on_turn_end`` -> ``_plan_compaction``
-> ``_run_compaction``, real snapcompact) over a synthetic screenshot run at the
real frame size (a 1280x720 PNG is ~705 KB, 939,876 bytes of base64). Payloads
are filler strings of that length rather than decodable PNGs: the wire ruler
counts ``len(data)`` and the cut never decodes, so the cost of rendering real
PNGs would buy nothing here.
"""

from __future__ import annotations

import pytest

from local_operator.compaction import api
from local_operator.compaction.api import CompactionSettings, estimate_wire_bytes
from local_operator.harness.types import (
    CapturedFrame,
    CompactionEndEvent,
    ImageContent,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
    Usage,
)
from local_operator.session.session import Session, _CompactionPlan
from local_operator.session.transcript import Transcript

#: Base64 length of the 1280x720 PNG the live run observed (705 KB raw).
FRAME_B64 = 939_876

#: Vision model: ``resolve_strategy`` picks snapcompact for it, as in the run.
MODEL = ModelSpec(provider="test", model_id="m", context_window=1_000_000)


def _stream(request, signal):
    async def gen():
        yield StreamTextDelta(delta="r")
        yield StreamEndEvent(stop_reason="stop")

    return gen()


def _make_session(tmp_path, **kwargs) -> Session:
    return Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        **kwargs,
    )


def _group(index: int, *, stamp: bool = False) -> tuple[Message, Message]:
    """One tool group: an assistant click and the screenshot it returned."""
    call_id = f"call-{index}"
    assistant = Message(
        role="assistant",
        content=[TextContent(text=f"step {index}")],
        tool_calls=[ToolCall(id=call_id, name="computer", arguments={"x": index})],
    )
    frame = (
        CapturedFrame(stream_id="screen", revision_id=f"r{index}", producer_scope="test:cut")
        if stamp
        else None
    )
    tool = Message(
        role="tool",
        tool_call_id=call_id,
        tool_name="computer",
        content=[
            TextContent(text=f"ok {index}"),
            ImageContent(data="A" * FRAME_B64, captured_frame=frame),
        ],
    )
    return assistant, tool


async def _grow(session: Session, groups: int, *, stamp: bool = False) -> None:
    await session.seed_history([Message.user("play the game")])
    for index in range(groups):
        assistant, tool = _group(index, stamp=stamp)
        session._context.messages.extend([assistant, tool])
    await session._persist_new_messages(session._context.messages)


def _band(settings: CompactionSettings) -> int:
    """Where a byte-fired pass must land: the LANDING target.

    This was ``WIRE_RECOVERY_BAND`` x trigger (12.8 MB) while the cut aimed at
    the auto-continue gate's acceptance line; the two are separate figures now
    and the cut aims at the lower one (8 MB on the default trigger).
    """
    return api.wire_landing_bytes(settings)


def _acceptance_band(settings: CompactionSettings) -> int:
    return int(api.WIRE_RECOVERY_BAND * api.resolve_wire_bytes_trigger(settings))


def _stamped_images(messages) -> int:
    """Producer-stamped captures only: a snapcompact marker's archive frames
    are images too, and they are not what these tests count.

    ``Message`` only: after a pass the live context opens with the
    ``compaction_summary`` ``CustomMessage``, which has no ``content``."""
    return sum(
        1
        for message in messages
        if isinstance(message, Message)
        for block in message.content
        if isinstance(block, ImageContent) and block.captured_frame is not None
    )


def _dump(messages) -> list[dict[str, object]]:
    return [message.model_dump(mode="json") for message in messages]


@pytest.mark.asyncio
async def test_a_byte_fired_pass_lands_inside_the_recovery_band(tmp_path):
    settings = CompactionSettings()
    session = _make_session(tmp_path, compaction_settings=settings)
    await _grow(
        session, 20
    )  # 20 frames = 18.8 MB: over the 16 MB trigger, far under every token line

    history = session._render_for_compaction()
    assert estimate_wire_bytes(history) > api.resolve_wire_bytes_trigger(
        settings
    ), "the fixture is not over the byte trigger; the test proves nothing"
    token_cut = api.find_cut_point(history, settings.keep_recent_tokens)

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan), f"no pass planned: {plan}"

    assert token_cut is not None and plan.cut > token_cut, (
        "the cut stayed where the TOKEN ruler put it, which keeps ~16 frames "
        "(~15 MB) against a 16 MB trigger"
    )
    outcome = await session._run_compaction(plan, reason="mid-turn")
    assert outcome.ran

    residual = estimate_wire_bytes(session._render_for_compaction())
    assert residual <= _band(settings), (
        f"the pass landed at {residual:,} bytes, outside the {_band(settings):,}-byte band: "
        "the next tool group re-fires it"
    )
    await session.dispose()


@pytest.mark.asyncio
async def test_the_advanced_cut_keeps_the_newest_observation_and_its_call(tmp_path):
    session = _make_session(tmp_path, compaction_settings=CompactionSettings())
    await _grow(session, 20)
    newest_tool = session._context.messages[-1]
    newest_call = session._context.messages[-2]

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    await session._run_compaction(plan, reason="mid-turn")

    ids = [m.id for m in session._context.messages]
    assert newest_tool.id in ids and newest_call.id in ids, "the current observation was summarized"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_user_attached_image_in_the_kept_window_is_never_advanced_past(tmp_path):
    settings = CompactionSettings()
    session = _make_session(tmp_path, compaction_settings=settings)
    await _grow(session, 20)
    # A person pasted a screenshot four groups from the end.
    attached = Message(
        role="user",
        content=[TextContent(text="look at this"), ImageContent(data="A" * FRAME_B64)],
    )
    session._context.messages.insert(-8, attached)
    await session._persist_new_messages(session._context.messages)

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    kept_ids = {m.id for m in plan.llm_history[plan.cut :]}
    assert attached.id in kept_ids, "the cut summarized away an image the user attached"
    await session.dispose()


@pytest.mark.asyncio
async def test_no_byte_pressure_leaves_the_token_ruled_cut_untouched(tmp_path):
    """The inert case: an image-free history, and a manual pass on a small one."""
    settings = CompactionSettings(keep_recent_tokens=200)
    session = _make_session(tmp_path, compaction_settings=settings)
    await session.seed_history([Message.user("start")])
    for index in range(30):
        session._context.messages.append(Message.user(f"question {index} " + "detail " * 40))
        session._context.messages.append(Message.assistant(f"answer {index} " + "word " * 40))
    await session._persist_new_messages(session._context.messages)

    history = session._render_for_compaction()
    expected = api.find_cut_point(history, 200)
    assert expected is not None and estimate_wire_bytes(history) < 1_000_000

    plan = await session._plan_compaction(respect_threshold=False)  # manual /compact
    assert isinstance(plan, _CompactionPlan)
    assert plan.cut == expected, "a session with no byte pressure had its cut moved"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_disabled_byte_trigger_leaves_the_cut_untouched(tmp_path):
    settings = CompactionSettings(wire_bytes_trigger=0)
    session = _make_session(tmp_path, compaction_settings=settings)
    await _grow(session, 20)
    history = session._render_for_compaction()
    expected = api.find_cut_point(history, settings.keep_recent_tokens)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    assert plan.cut == expected
    await session.dispose()


@pytest.mark.asyncio
async def test_a_run_of_screenshots_no_longer_compacts_every_tool_group(tmp_path):
    """The cadence, end to end through the mid-turn gate.

    On base this run fires at groups 17, 19, 21, ... and then EVERY group (the
    residual sits ~0.4 frame under the trigger). With the byte-aware cut the
    passes are spaced by the headroom the band buys.
    """
    session = _make_session(tmp_path, compaction_settings=CompactionSettings())
    events: list[object] = []
    session.subscribe(events.append)
    await session.seed_history([Message.user("play the game")])

    fired: list[int] = []
    for index in range(36):
        assistant, tool = _group(index)
        assistant.usage = Usage(input_tokens=1, output_tokens=1, context_tokens=30_000)
        session._context.messages.extend([assistant, tool])
        before = sum(isinstance(e, CompactionEndEvent) for e in events)
        await session._on_turn_end(list(session._context.messages))
        if sum(isinstance(e, CompactionEndEvent) for e in events) > before:
            fired.append(index)

    assert fired, "a 36-frame run never compacted; the fixture proves nothing"
    gaps = [later - earlier for earlier, later in zip(fired, fired[1:])]
    # Was ``>= 3`` while the cut aimed at the acceptance band (0.2 x 16 MB of
    # room = 3.4 of these frames). At the landing target a pass keeps at most
    # 8 MB, so ~8.5 frames fit before the next one (expected gap: 9). Six is
    # asserted, not nine, to leave room for the archive a pass seats in front.
    assert len(gaps) >= 1, f"only one pass fired, so no cadence was measured: {fired}"
    assert all(gap >= 6 for gap in gaps), f"passes fired too close together: {fired}"
    await session.dispose()


@pytest.mark.asyncio
async def test_image_free_history_over_the_byte_trigger_still_keeps_its_cut(tmp_path):
    """Even an explicitly tiny byte trigger must not change text-only cuts."""
    settings = CompactionSettings(wire_bytes_trigger=100, keep_recent_tokens=20)
    session = _make_session(tmp_path, compaction_settings=settings)
    await session.seed_history([Message.user("goal")])
    history = [Message.assistant("long text " * 100) for _ in range(6)]
    assert estimate_wire_bytes(history) > api.resolve_wire_bytes_trigger(settings)
    assert session._advance_cut_for_wire_band(api, settings, history, 2) == 2
    await session.dispose()


@pytest.mark.asyncio
async def test_advanced_cut_records_the_same_kept_observations_on_resume(tmp_path):
    session = _make_session(tmp_path, compaction_settings=CompactionSettings())
    await _grow(session, 20)
    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    kept_ids = [m.id for m in plan.llm_history[plan.cut :]]
    assert (await session._run_compaction(plan, reason="mid-turn")).ran
    directory = session._transcript.directory
    live_kept = [m.model_dump(mode="json") for m in session._context.messages if m.id in kept_ids]
    await session.dispose()
    replayed = Transcript(directory).build_llm_history()
    resumed_kept = [m.model_dump(mode="json") for m in replayed if m.id in kept_ids]
    assert resumed_kept == live_kept
    assert [m["id"] for m in resumed_kept] == kept_ids


# ---------------------------------------------------------------------------
# The landing target (``thresholds.WIRE_LANDING_BAND``): a byte-fired pass
# lands with real room under the trigger, keeps the newest frame exactly, and
# leaves the prompt prefix alone on every turn where no pass fires.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_byte_fired_pass_leaves_room_for_at_least_eight_more_screenshots(tmp_path):
    """The stated target, at the frame size this file's run observed.

    Aiming at the acceptance band left 3.2 MB: 3.4 of these frames, so the
    trigger re-fired every third or fourth tool group.
    """
    settings = CompactionSettings()
    trigger = api.resolve_wire_bytes_trigger(settings)
    session = _make_session(tmp_path, compaction_settings=settings)
    await _grow(session, 20)

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan), f"no pass planned: {plan}"
    assert (await session._run_compaction(plan, reason="mid-turn")).ran

    residual = estimate_wire_bytes(session._render_history(list(session._context.messages)))
    headroom = trigger - residual
    assert headroom >= 8 * FRAME_B64, (
        f"the pass left {headroom:,} bytes under the trigger: "
        f"{headroom / FRAME_B64:.1f} screenshots before the next pass"
    )
    # ...and it still holds recent raw frames: the target is not "keep nothing".
    kept_frames = sum(
        1
        for message in session._context.messages
        if isinstance(message, Message)
        for block in message.content
        if isinstance(block, ImageContent) and len(block.data) == FRAME_B64
    )
    assert kept_frames >= 4, f"only {kept_frames} raw screenshots survived the pass"
    # A pass that lands on target must clear the auto-continue gate's band.
    assert api.cleared_wire_headroom(residual, settings) is True
    assert session._cleared_wire_headroom(plan) is True
    await session.dispose()


@pytest.mark.asyncio
async def test_the_newest_frame_survives_a_byte_fired_pass_byte_for_byte(tmp_path):
    """Full fidelity, not only presence: no re-encode, no downscale.

    The newest observation is the one the model is about to act on, and a
    visual-precision task reads coordinates off it.
    """
    session = _make_session(tmp_path, compaction_settings=CompactionSettings())
    await _grow(session, 20)
    newest = session._context.messages[-1]
    assert isinstance(newest, Message)
    # A payload no other frame has, so equality cannot be met by a neighbour.
    distinct = "B" * FRAME_B64
    newest.content[1] = ImageContent(data=distinct)
    await session._persist_new_messages(session._context.messages)
    before = newest.model_dump(mode="json")

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    assert (await session._run_compaction(plan, reason="mid-turn")).ran

    survivor = next((m for m in session._context.messages if m.id == newest.id), None)
    assert survivor is not None, "the current observation was summarized"
    assert survivor.model_dump(mode="json") == before
    rendered = session._render_history(list(session._context.messages))
    assert rendered[-1].model_dump(mode="json") == before
    await session.dispose()


@pytest.mark.asyncio
async def test_the_rendered_prefix_is_not_rewritten_between_passes(tmp_path):
    """Cache stability: on a turn where no pass fires, the request is the
    previous request plus the new tool group, byte for byte.

    The landing target buys its room at the pass (where the prefix is rewritten
    anyway), never by re-encoding earlier frames on ordinary turns.
    """
    session = _make_session(tmp_path, compaction_settings=CompactionSettings())
    events: list[object] = []
    session.subscribe(events.append)
    await session.seed_history([Message.user("play the game")])

    def passes() -> int:
        return sum(isinstance(event, CompactionEndEvent) for event in events)

    previous: list[dict[str, object]] | None = None
    quiet_turns_after_first_pass = 0
    for index in range(30):
        assistant, tool = _group(index)
        assistant.usage = Usage(input_tokens=1, output_tokens=1, context_tokens=30_000)
        session._context.messages.extend([assistant, tool])
        before = passes()
        await session._on_turn_end(list(session._context.messages))
        rendered = _dump(session._render_history(list(session._context.messages)))
        if passes() > before:
            previous = rendered  # a pass rewrites the prefix; start a new interval
            continue
        if previous is not None:
            assert (
                rendered[: len(previous)] == previous
            ), f"group {index}: the prefix changed on a turn where no pass fired"
            assert len(rendered) == len(previous) + 2
            if before >= 1:
                quiet_turns_after_first_pass += 1
        previous = rendered

    assert passes() >= 1, "the run never compacted; the fixture proves nothing"
    assert quiet_turns_after_first_pass >= 2, "no two consecutive non-pass turns after a pass"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_token_fired_pass_in_an_image_session_keeps_the_acceptance_band(tmp_path):
    """Byte pressure is the key, not "the session has images".

    A pass the TOKEN term fired, on a request under the byte trigger, must do
    exactly what it did before the landing target existed: the token-ruled cut
    and capture elision against the acceptance band. Its kept window here sits
    between the two figures (over 8 MB, under 12.8 MB), so applying the landing
    target would elide superseded captures the pass keeps today.
    """
    settings = CompactionSettings(threshold_tokens=5_000, keep_recent_tokens=14_000)
    trigger = api.resolve_wire_bytes_trigger(settings)
    session = _make_session(tmp_path, compaction_settings=settings)
    events: list[object] = []
    session.subscribe(events.append)
    await _grow(session, 14, stamp=True)  # 13.2 MB: under the 16 MB byte trigger

    history = session._render_for_compaction()
    assert estimate_wire_bytes(history) < trigger, "the fixture is over the byte trigger"
    token_cut = api.find_cut_point(history, settings.keep_recent_tokens)

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan), f"the token term did not plan a pass: {plan}"
    assert plan.cut == token_cut, "a pass with no byte pressure had its cut moved"
    kept_bytes = estimate_wire_bytes(plan.llm_history[plan.cut :])
    assert _band(settings) < kept_bytes <= _acceptance_band(settings), (
        f"the kept window ({kept_bytes:,} bytes) is not between the landing target and the "
        "acceptance band; the fixture cannot tell the two budgets apart"
    )
    assert session._capture_elision_budget(api, plan) == _acceptance_band(settings)
    kept_captures = _stamped_images(plan.llm_history[plan.cut :])

    assert (await session._run_compaction(plan, reason="context-window")).ran
    assert (
        _stamped_images(session._context.messages) == kept_captures
    ), "a token-fired pass elided captures it kept before the landing target existed"
    end = [event for event in events if isinstance(event, CompactionEndEvent)][-1]
    assert end.trigger == "tokens"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_pass_over_the_byte_trigger_elides_captures_to_the_landing_target(tmp_path):
    """The other half of the key: over the byte trigger, the commit's elision
    budget and the cut's projection of it are the same landing figure."""
    settings = CompactionSettings()
    session = _make_session(tmp_path, compaction_settings=settings)
    await _grow(session, 20, stamp=True)

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    assert session._capture_elision_budget(api, plan) == _band(settings)
    assert session._wire_landing_bytes(api, settings) == _band(settings)
    assert session._wire_band_bytes(api, settings) == _acceptance_band(settings)
    assert (await session._run_compaction(plan, reason="mid-turn")).ran
    residual = estimate_wire_bytes(session._render_for_compaction())
    assert residual <= _band(settings)
    await session.dispose()


def test_a_compaction_module_without_the_landing_resolver_keeps_the_old_band(tmp_path):
    """A partial test double that predates ``wire_landing_bytes`` degrades to
    the acceptance band instead of losing the byte-aware cut."""
    from types import SimpleNamespace

    double = SimpleNamespace(
        resolve_wire_bytes_trigger=lambda settings: 1_000_000, WIRE_RECOVERY_BAND=0.8
    )
    session = Session.__new__(Session)
    assert session._wire_landing_bytes(double, CompactionSettings()) == 800_000
    off = SimpleNamespace(resolve_wire_bytes_trigger=lambda settings: 0, WIRE_RECOVERY_BAND=0.8)
    assert session._wire_landing_bytes(off, CompactionSettings()) is None


# ---------------------------------------------------------------------------
# ``compaction_end`` says which size term fired, and the byte figures.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_end_records_a_byte_fired_pass_and_its_byte_figures(tmp_path):
    settings = CompactionSettings()
    trigger = api.resolve_wire_bytes_trigger(settings)
    session = _make_session(tmp_path, compaction_settings=settings)
    events: list[object] = []
    session.subscribe(events.append)
    await _grow(session, 20)
    planned_bytes = estimate_wire_bytes(session._render_for_compaction())

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    assert (await session._run_compaction(plan, reason="mid-turn")).ran

    end = [event for event in events if isinstance(event, CompactionEndEvent)][-1]
    assert end.success and end.reason == "mid-turn"
    # 20 frames are ~24k tokens against a 400k token trigger: bytes alone fired.
    assert end.trigger == "wire_bytes"
    assert end.wire_bytes_trigger == trigger == 16_000_000
    assert end.wire_bytes_before == planned_bytes > trigger
    assert end.wire_bytes_after == estimate_wire_bytes(
        session._render_history(list(session._context.messages))
    )
    assert end.wire_bytes_after <= _band(settings) < end.wire_bytes_before
    # The token pair is untouched by the new fields.
    assert end.tokens_before == plan.context_tokens and end.strategy == plan.strategy
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_end_trigger_fields_never_re_run_the_decision(tmp_path, monkeypatch):
    """QA round 1, Q2: the diagnostic reads resolved thresholds, it never asks
    ``should_compact`` again. The gate decided once, while planning; a re-run
    is a second evaluation hosts and doubles observe (``test_session.py``'s
    ``test_compaction_runs_when_due`` records every decision) and one free to
    drift from the decision the gate actually made."""
    settings = CompactionSettings(threshold_tokens=5_000, keep_recent_tokens=14_000)
    session = _make_session(tmp_path, compaction_settings=settings)
    events: list[object] = []
    session.subscribe(events.append)
    await _grow(session, 14, stamp=True)  # token-fired, under the byte trigger

    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan)
    real = api.should_compact
    calls: list[int] = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "should_compact", counting)
    fields = session._compaction_trigger_fields(plan)
    assert calls == [], "the trigger diagnostic re-ran the compaction decision"
    assert fields["trigger"] == "tokens"
    await session.dispose()


@pytest.mark.asyncio
async def test_compaction_end_on_a_manual_text_pass_names_no_size_term(tmp_path):
    """A pass neither size term forced reads ``none``, with real byte figures."""
    settings = CompactionSettings(keep_recent_tokens=200)
    session = _make_session(tmp_path, compaction_settings=settings)
    events: list[object] = []
    session.subscribe(events.append)
    await session.seed_history([Message.user("start")])
    for index in range(30):
        session._context.messages.append(Message.user(f"question {index} " + "detail " * 40))
        session._context.messages.append(Message.assistant(f"answer {index} " + "word " * 40))
    await session._persist_new_messages(session._context.messages)

    plan = await session._plan_compaction(respect_threshold=False)  # manual /compact
    assert isinstance(plan, _CompactionPlan)
    assert (await session._run_compaction(plan, reason="manual")).ran

    end = [event for event in events if isinstance(event, CompactionEndEvent)][-1]
    assert end.trigger == "none"
    assert end.wire_bytes_trigger == api.resolve_wire_bytes_trigger(settings)
    assert 0 < end.wire_bytes_before < end.wire_bytes_trigger
    assert end.wire_bytes_after > 0
    await session.dispose()


def test_compaction_end_fields_are_additive_and_default_to_absent_values():
    """A host that builds or reads the event the old way is unaffected."""
    legacy = CompactionEndEvent(reason="manual", success=True, tokens_before=100, tokens_after=50)
    assert (legacy.trigger, legacy.wire_bytes_before, legacy.wire_bytes_after) == ("", 0, 0)
    assert legacy.wire_bytes_trigger == 0
    # A record written before the fields existed still validates.
    old_row = {
        "type": "compaction_end",
        "reason": "mid-turn",
        "success": True,
        "strategy": "snapcompact",
        "tokens_before": 34847,
        "tokens_after": 28642,
        "detail": None,
    }
    assert CompactionEndEvent.model_validate(old_row).trigger == ""
    dumped = CompactionEndEvent(reason="mid-turn", success=False).model_dump()
    assert dumped["trigger"] == "" and dumped["wire_bytes_trigger"] == 0
