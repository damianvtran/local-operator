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
    return int(api.WIRE_RECOVERY_BAND * api.resolve_wire_bytes_trigger(settings))


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
    assert all(gap >= 3 for gap in gaps), f"passes fired on consecutive-ish groups: {fired}"
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
