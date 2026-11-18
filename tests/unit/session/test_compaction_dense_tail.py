"""The live task rides the archive as text instead of widening the raw window.

The task floor (``task_boundary_floor``) exists because a recency-shaped cut
landed inside the task an agent was still executing. Its remedy was to keep the
whole task RAW, up to a 100k cap. Replaying 24 real compaction points through
``Session.compact_now`` put a number on that: the passes the floor widened
kept a median 69k-100k raw tokens, and those were the passes landing at
108k-134k local (roughly 200k+ on an Anthropic tokenizer).

Simply dropping the floor is not the fix, and that was measured too: cutting
back to plain recency left only 38% of the distinctive lines of today's
rebuilt context present, because the archive's fixed 8-page middle cannot
absorb an extra 80k of history. So the task is still protected, just carried
differently: the pass cuts at plain recency and the span the floor would have
kept raw is serialized onto the archive's verbatim TAIL edge (0.67x its raw
size at the median of 15 real spans: tool results past 2,000 characters and
tool arguments past 500 lose their middles; nothing else changes).

These tests drive the real gate and the real pass (``_plan_compaction`` ->
``compact_now`` -> snapcompact -> ``Transcript.build_llm_history``); only the
history is synthetic.
"""

from __future__ import annotations

import pytest

from local_operator.compaction import api, snapcompact
from local_operator.compaction.advisor import CompactionHint
from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.cutpoint import is_rendered_injection
from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import (
    CustomMessage,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
    ToolCall,
)
from local_operator.session.session import Session, _CompactionPlan
from local_operator.session.transcript import Transcript

#: Vision model: ``resolve_strategy`` picks snapcompact for it.
VISION = ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5", context_window=1_000_000)
#: Text-only model: context-full, which has no archive tail to carry a task on.
TEXT_ONLY = ModelSpec(
    provider="test", model_id="reads", context_window=1_000_000, supports_images=False
)

#: Sized so the span the floor protects beyond recency (``FLOOR_CAP -
#: KEEP_RECENT`` raw tokens) serializes WIDER than the archive's default tail
#: edge for this reader: otherwise the default tail already holds it and the
#: fixture would not exercise the widening (the tests assert that control).
KEEP_RECENT = 10_000
#: ``_advisor_floor_cap`` on a 1M window: ``keep_recent * 5`` (the task term).
FLOOR_CAP = KEEP_RECENT * 5

OLD_STEPS = 40
LIVE_STEPS = 60
LIVE_REQUEST = "LIVE TASK: migrate the billing ledger, and do NOT touch invoices.py"


def _stream(request, signal):
    async def gen():
        yield StreamTextDelta(delta="r")
        yield StreamEndEvent(stop_reason="stop")

    return gen()


def _make_session(tmp_path, model: ModelSpec = VISION, name: str = "sess") -> Session:
    return Session(
        model=model,
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(tmp_path / name),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=CompactionSettings(keep_recent_tokens=KEEP_RECENT),
    )


def _step(tag: str, index: int) -> tuple[Message, Message]:
    """One tool group whose output is long enough for the archive to truncate."""
    call_id = f"{tag}-{index}"
    body = "\n".join(f"{tag} {index} line {row}: value={row * 7919 % 10007}" for row in range(110))
    assistant = Message(
        role="assistant",
        content=[TextContent(text=f"{tag} step {index} reasoning")],
        tool_calls=[ToolCall(id=call_id, name="bash", arguments={"command": f"run {tag} {index}"})],
    )
    tool = Message(
        role="tool",
        tool_call_id=call_id,
        tool_name="bash",
        content=[TextContent(text=f"{tag} output {index}\n{body}")],
    )
    return assistant, tool


async def _grow(session: Session, live_steps: int = LIVE_STEPS) -> None:
    """Finished older work, then ONE user request and a long run of tool steps."""
    await session.seed_history([Message.user("earlier question about the report")])
    for index in range(OLD_STEPS):
        session._context.messages.extend(_step("old", index))
    session._context.messages.append(Message.user(LIVE_REQUEST))
    for index in range(live_steps):
        session._context.messages.extend(_step("live", index))
    await session._persist_new_messages(session._context.messages)


async def _more(session: Session, steps: range) -> None:
    """The SAME task carries on: more tool steps, no operator turn."""
    for index in steps:
        session._context.messages.extend(_step("live", index))
    await session._persist_new_messages(session._context.messages)


def _cuts(session: Session) -> tuple[list[Message], int, int]:
    """The planner's history with the cut the floor asks for and the recency cut."""
    history = session._render_for_compaction()
    genuine = {
        m.id for m in session._context.messages if isinstance(m, Message) and m.role == "user"
    }
    floor = api.task_boundary_floor(history, genuine, cap=FLOOR_CAP)
    assert floor == FLOOR_CAP, "fixture: the live task must be wider than the floor cap"
    wide = api.find_cut_point(history, floor)
    narrow = api.find_cut_point(history, KEEP_RECENT)
    assert wide is not None and narrow is not None and narrow > wide
    return history, wide, narrow


def _archive(session: Session) -> snapcompact.Archive:
    entry = session._transcript.latest_entry("compaction")
    assert entry is not None
    return snapcompact.Archive(**entry.payload["preserve_data"]["snapcompact"])


def _live_texts(session: Session) -> str:
    return "\n".join(
        m.text for m in session._context.messages if isinstance(m, Message) and m.role != "tool"
    )


@pytest.mark.asyncio
async def test_the_pass_cuts_at_recency_and_carries_the_task_on_the_archive_tail(tmp_path):
    session = _make_session(tmp_path)
    await _grow(session)
    history, wide, narrow = _cuts(session)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan), f"no pass planned: {plan}"
    assert plan.cut == narrow, "the raw window is the plain recency window"
    assert plan.dense_tail_from == wide, "the span the floor protects starts where it did"

    outcome = await session.compact_now()
    assert outcome.ran, outcome

    archive = _archive(session)
    span = history[wide:narrow]
    first_step = next(m.text for m in span if m.role == "assistant")
    last_step = next(m.text for m in reversed(span) if m.role == "assistant")
    # The tail edge IS the protected span, serialized: from its first role
    # header to its last character, nothing paged and nothing dropped.
    assert archive.text_tail == snapcompact.serialize_for_snapcompact(span)
    assert archive.text_tail.index(first_step) < archive.text_tail.index(last_step)
    assert (
        len(archive.text_tail)
        > snapcompact.HQ_EDGE_FRAMES
        * snapcompact.resolve_shape(VISION.provider, VISION.model_id).capacity
    ), "control: the span is wider than the default tail could have held"

    # The raw window no longer holds those steps...
    raw_tool_outputs = [
        m.text for m in session._context.messages if isinstance(m, Message) and m.role == "tool"
    ]
    assert not any(
        first_step.split(" reasoning")[0].replace("step", "output") in t for t in raw_tool_outputs
    )
    # ...and it is the recency window, not the floor's.
    kept = [m for m in session._render_for_compaction() if m.id in {h.id for h in history[narrow:]}]
    assert api.estimate_messages_tokens(kept) < FLOOR_CAP // 2

    # The operator's own words are untouched by any of this.
    assert LIVE_REQUEST in _live_texts(session)
    assert "earlier question about the report" in _live_texts(session)
    await session.dispose()


@pytest.mark.asyncio
async def test_the_dense_pass_lands_smaller_than_the_raw_floor_pass(tmp_path, monkeypatch):
    """The point of the change, on the product's own ruler: same history, less context."""
    dense = _make_session(tmp_path, name="dense")
    await _grow(dense)
    assert (await dense.compact_now()).ran
    dense_tokens = api.estimate_messages_tokens(dense._render_for_compaction())

    monkeypatch.setattr(snapcompact, "dense_tail_saves", lambda span: False)
    raw = _make_session(tmp_path, name="raw")
    await _grow(raw)
    assert (await raw.compact_now()).ran
    raw_tokens = api.estimate_messages_tokens(raw._render_for_compaction())

    assert dense_tokens < raw_tokens
    await dense.dispose()
    await raw.dispose()


@pytest.mark.asyncio
async def test_a_resumed_session_rebuilds_the_same_tail(tmp_path):
    """Live and replay must agree: the tail rides the marker, not session state."""
    session = _make_session(tmp_path)
    await _grow(session)
    assert (await session.compact_now()).ran
    live_tail = _archive(session).text_tail
    live_kept = [m.id for m in session._context.messages if not isinstance(m, CustomMessage)]

    replayed = Transcript(tmp_path / "sess").build_llm_history()
    marker = next(m for m in replayed if isinstance(m, CustomMessage))
    assert marker.details["preserve_data"]["snapcompact"]["text_tail"] == live_tail
    assert [m.id for m in replayed if not isinstance(m, CustomMessage)] == live_kept
    await session.dispose()


def _step_markers(tag: str, steps: range) -> list[str]:
    """One line per step that only that step's assistant message produces."""
    return [f"{tag} step {index} reasoning" for index in steps]


#: A live task NARROWER than the floor cap (the floor is not clamped), which is
#: the case where today's raw floor still has its anchor at the next pass.
FITTING_STEPS = 30
#: ``Session._dense_tail_cap``: three quarters of the floor cap.
TAIL_CAP = FLOOR_CAP * 3 // 4


def _floor(session: Session) -> int:
    history = session._render_for_compaction()
    genuine = {m.id for m in session._context.messages if isinstance(m, Message)}
    return api.task_boundary_floor(history, genuine, cap=FLOOR_CAP)


@pytest.mark.asyncio
async def test_a_task_still_running_at_the_next_pass_keeps_its_earlier_steps(tmp_path):
    """Protection must outlive ONE pass (agent review round 1, F1).

    The dense pass archives the task's opening request along with the task, so
    the next pass's ``task_boundary_floor`` finds no operator turn to measure
    from and answers 0. Before this was handled the previous protected tail
    then became ordinary middle history after a single pass, where today's
    raw floor (anchor still in the window) went on protecting the task.
    """
    session = _make_session(tmp_path)
    await _grow(session, FITTING_STEPS)
    assert KEEP_RECENT < _floor(session) < FLOOR_CAP, "fixture: widened, not clamped"
    assert (await session.compact_now()).ran
    first = _archive(session)
    assert first.live_tail_chars == len(first.text_tail) > 0
    held = _step_markers("live", range(FITTING_STEPS))
    assert all(m in first.text_tail + "\n" + _live_texts(session) for m in held)

    # Few enough new steps that both spans fit the tail bound: nothing may leave.
    more = range(FITTING_STEPS, FITTING_STEPS + 12)
    await _more(session, more)
    assert _floor(session) == 0, "fixture: the request was archived, the floor has no anchor"

    assert (await session.compact_now()).ran
    second = _archive(session)
    assert second.text_tail.startswith(first.text_tail), "the earlier protected tail was released"
    assert second.live_tail_chars == len(second.text_tail)
    context = second.text_tail + "\n" + _live_texts(session)
    missing = [m for m in held + _step_markers("live", more) if m not in context]
    assert not missing, f"{len(missing)} live steps are no longer verbatim text: {missing[:3]}"
    await session.dispose()


@pytest.mark.asyncio
async def test_a_new_operator_request_releases_the_earlier_task(tmp_path):
    """The earlier tail is protected only while ITS task is the live one."""
    session = _make_session(tmp_path)
    await _grow(session, FITTING_STEPS)
    assert (await session.compact_now()).ran
    first = _archive(session)
    assert first.live_tail_chars > 0

    session._context.messages.append(Message.user("NEW TASK: now audit the refund path"))
    for index in range(FITTING_STEPS):
        session._context.messages.extend(_step("audit", index))
    await session._persist_new_messages(session._context.messages)

    assert (await session.compact_now()).ran
    second = _archive(session)
    assert second.live_tail_chars == len(second.text_tail)
    assert "audit step 0 reasoning" in second.text_tail
    assert "live step 0 reasoning" not in second.text_tail
    assert not second.text_tail.startswith(first.text_tail)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_task_that_outgrows_the_tail_bound_is_carried_once_more_then_not(tmp_path):
    """The continuation has the horizon the raw floor has, and no ratchet.

    Today's floor protects a task while it fits the cap, keeps the newest
    cap's worth on the pass where the task outgrows it, and is plain recency
    after that. The tail does the same with its own bound: the pass that has
    to trim still carries the newest steps on a bounded tail, records nothing
    as continuable, and the pass after it is an ordinary recency pass.
    """
    session = _make_session(tmp_path)
    assert session._dense_tail_cap(session._compaction_settings) == TAIL_CAP
    await _grow(session, FITTING_STEPS)
    assert (await session.compact_now()).ran
    assert _archive(session).live_tail_chars > 0

    # Far more than the bound holds.
    await _more(session, range(FITTING_STEPS, FITTING_STEPS + 80))
    assert (await session.compact_now()).ran
    trimmed = _archive(session)
    tail_tokens = estimate_tokens(Message.assistant(trimmed.text_tail))
    default_tail = _archive_default_tail_chars()
    assert len(trimmed.text_tail) > default_tail, "the bounded tail is still a protected one"
    assert 0.8 * TAIL_CAP <= tail_tokens <= TAIL_CAP
    assert f"live step {FITTING_STEPS + 60} reasoning" in trimmed.text_tail
    assert "live step 0 reasoning" not in trimmed.text_tail
    assert trimmed.live_tail_chars == 0, "a trimmed tail must not be continued"

    await _more(session, range(FITTING_STEPS + 80, FITTING_STEPS + 140))
    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    assert plan.dense_tail_from is None and plan.dense_tail_previous_chars == 0
    assert (await session.compact_now()).ran
    assert len(_archive(session).text_tail) == default_tail
    await session.dispose()


@pytest.mark.asyncio
async def test_a_task_already_wider_than_the_floor_is_not_continued(tmp_path):
    """A clamped floor has already lost its anchor: carry once, do not continue."""
    session = _make_session(tmp_path)
    await _grow(session)
    assert _floor(session) == FLOOR_CAP, "fixture: the task is wider than the floor cap"
    assert (await session.compact_now()).ran
    first = _archive(session)
    assert len(first.text_tail) > _archive_default_tail_chars(), "still carried on this pass"
    assert first.live_tail_chars == 0

    await _more(session, range(LIVE_STEPS, LIVE_STEPS + 40))
    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan)
    assert plan.dense_tail_from is None
    await session.dispose()


def _archive_default_tail_chars() -> int:
    shape = snapcompact.resolve_shape(VISION.provider, VISION.model_id)
    return snapcompact.HQ_EDGE_FRAMES * shape.capacity


@pytest.mark.asyncio
async def test_a_text_only_model_keeps_the_task_raw(tmp_path):
    """Context-full has no verbatim tail: an LLM summary would paraphrase the task."""
    session = _make_session(tmp_path, model=TEXT_ONLY)
    await _grow(session)
    _history, wide, _narrow = _cuts(session)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan), f"no pass planned: {plan}"
    assert plan.strategy == "context-full"
    assert plan.cut == wide
    assert plan.dense_tail_from is None
    await session.dispose()


@pytest.mark.asyncio
async def test_a_task_that_serializes_no_smaller_stays_raw(tmp_path, monkeypatch):
    """Dense carry is a size optimisation: no saving, no change of carrier."""
    monkeypatch.setattr(snapcompact, "dense_tail_saves", lambda span: False)
    session = _make_session(tmp_path)
    await _grow(session)
    _history, wide, _narrow = _cuts(session)

    plan = await session._plan_compaction(respect_threshold=False)
    assert isinstance(plan, _CompactionPlan), f"no pass planned: {plan}"
    assert plan.cut == wide
    assert plan.dense_tail_from is None
    await session.dispose()


@pytest.mark.asyncio
async def test_an_advisor_widened_window_stays_raw(tmp_path, monkeypatch):
    """An advisory is a semantic "this must survive as it is"; it is not re-carried.

    Only the automatic task floor changes carrier. A hint that shaped the
    window keeps the pre-existing behaviour exactly, so the advisor's
    widen-only guarantee and its reclaim bookkeeping mean what they meant.
    The gate is driven with ``respect_threshold=True`` because that is the only
    path that consumes a hint; the context is pinned over the trigger so the
    pass is due.
    """
    session = _make_session(tmp_path)
    session._compaction_settings = CompactionSettings(
        keep_recent_tokens=KEEP_RECENT, advisor_enabled=True, advisor_every_n_turns=1
    )
    await _grow(session)
    _history, wide, narrow = _cuts(session)
    monkeypatch.setattr(api, "messages_tokens_upper_bound", lambda messages: 700_000)
    monkeypatch.setattr(api, "estimate_messages_tokens", lambda messages: 700_000)

    # Control: without a hint this very gate takes the dense cut.
    plain = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plain, _CompactionPlan) and plain.cut == narrow

    session._advisor_hint = CompactionHint(
        preserve_from_id=session._context.messages[-1].id,
        preserve_tokens=KEEP_RECENT,
        compact_now=True,
        confidence=0.9,
        reason="task boundary reached",
        turn_index=session._generation,
    )
    plan = await session._plan_compaction(respect_threshold=True)
    assert isinstance(plan, _CompactionPlan), f"no pass planned: {plan}"
    assert session._advisor_hint is None, "fixture: the hint must have been consumed"
    assert plan.cut == wide
    assert plan.dense_tail_from is None
    await session.dispose()


@pytest.mark.asyncio
async def test_relaunch_briefs_are_collapsed_before_the_block_is_capped(tmp_path):
    """A resumed subagent's re-stamped launch briefs are carried once (see
    ``collapse_relaunch_briefs``); the newest copy stays verbatim."""
    brief = "\n".join(f"Standing rule {i}: keep the gate." for i in range(60))
    session = _make_session(tmp_path)
    launches = [
        Message.user(f"[resume context — attempt {n} ended: paused.]\n{brief}\nFix the test.")
        for n in range(4)
    ]
    for n, launch in enumerate(launches):
        launch.id = f"subagent-launch:job{n}"
    await session.seed_history([launches[0]])
    for n, launch in enumerate(launches[1:], start=1):
        for index in range(12):
            session._context.messages.extend(_step(f"a{n}", index))
        session._context.messages.append(launch)
    for index in range(30):
        session._context.messages.extend(_step("tail", index))
    await session._persist_new_messages(session._context.messages)

    assert (await session.compact_now()).ran
    entry = session._transcript.latest_entry("compaction")
    assert entry is not None
    stored = {t["id"]: t["text"] for t in entry.payload["preserved_user_turns"]}
    assert set(stored) == {launch.id for launch in launches}
    assert stored[launches[-1].id] == launches[-1].text
    for n, launch in enumerate(launches[:-1]):
        assert f"attempt {n} ended" in stored[launch.id]
        assert "Standing rule 7" not in stored[launch.id]
        assert stored[launch.id].endswith(api.RELAUNCH_ELISION_NOTE)
    # The live context carries exactly what the journal recorded.
    live = [
        m.text
        for m in session._context.messages
        if isinstance(m, Message) and m.role == "user" and not is_rendered_injection(m)
    ]
    assert live.count(launches[-1].text) == 1
    assert sum(text.count("Standing rule 7:") for text in live) == 1
    await session.dispose()
