"""When the automatic compaction gate fires — the session side of the trigger.

The bug these tests hold the line on: a session on a 1M-context model
(`anthropic/claude-opus-5`, `context_window=1_000_000`) compacted at 234.8k
tokens, 23% of its window, throwing away three quarters of its usable context
per pass. The cause was a second absolute knob resolving a threshold of its own
next to the percentage one; the fix is that
``compaction.thresholds.resolve_threshold_tokens`` is the only thing that
answers "when", and it answers ``min(percent x window, absolute)``.

Real compaction throughout (no stubbed ``compaction.api``) — a threshold test
against a fake gate proves nothing about the gate. Only the RULER is
substituted: the estimator is pinned so a test can stand at an exact context
size without carrying a megabyte of history.
"""

from __future__ import annotations

import pytest

from local_operator.compaction import api as compaction_api
from local_operator.compaction.api import CompactionSettings
from local_operator.compaction.cutpoint import find_cut_point, prepare_partitions
from local_operator.harness.types import (
    AgentEvent,
    CompactionEndEvent,
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

#: The model from the report: 1M advertised context, vision-capable.
BIG_MODEL = ModelSpec(provider="test", model_id="opus-like", context_window=1_000_000)
#: A 200k-context model, where the 600k absolute default cannot ever be reached
#: and the percentage has to govern instead.
SMALL_MODEL = ModelSpec(provider="test", model_id="sonnet-like", context_window=200_000)

#: Small enough that a few short turns leave history outside the kept window,
#: so ``find_cut_point`` has something to summarize.
KEEP_RECENT = 40


class ScriptedStream:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def __call__(self, request, signal):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="reply")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, model=BIG_MODEL, **kwargs) -> Session:
    settings = kwargs.pop("compaction_settings", CompactionSettings(keep_recent_tokens=KEEP_RECENT))
    return Session(
        model=model,
        stream_fn=ScriptedStream(),
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        compaction_settings=settings,
        **kwargs,
    )


async def talk(session: Session, turns: int = 3) -> None:
    for index in range(turns):
        await session.prompt(f"question {index} " + "detail " * 30)


def pin_measured_context(monkeypatch, tokens: int) -> None:
    """Pin both rulers the gate consults to one figure.

    The gate takes the cheap upper bound first and only pays for the exact
    estimate when the bound clears the threshold, so a test that pinned only
    one of them would exercise only one half of the two-stage trigger.
    """
    monkeypatch.setattr(compaction_api, "messages_tokens_upper_bound", lambda messages: tokens)
    monkeypatch.setattr(compaction_api, "estimate_messages_tokens", lambda messages: tokens)


def refusal_reason(result: object) -> str | None:
    """The gate's refusal reason, or ``None`` when it produced a real plan.

    ``_plan_compaction`` returns a union: a ``_CompactionPlan`` when a pass is
    due, a ``CompactionOutcome`` carrying ``reason`` when it declines. Reading
    both arms through helpers keeps the assertions honest about that — plain
    attribute access type checks against only one arm and hides which one a
    given assertion is really about.
    """
    return getattr(result, "reason", None)


def as_plan(result: object) -> _CompactionPlan:
    """The plan arm of that union, asserted rather than assumed."""
    assert isinstance(result, _CompactionPlan), f"expected a plan, got {result!r}"
    return result


@pytest.mark.asyncio
async def test_a_1m_session_does_not_compact_at_235k(tmp_path, monkeypatch):
    """The regression: 234.8k of a 1M window is 23%, nowhere near due."""
    session = make_session(tmp_path)
    await talk(session)
    pin_measured_context(monkeypatch, 234_800)

    outcome = await session._plan_compaction(respect_threshold=True)

    assert getattr(outcome, "ran", None) is False
    assert refusal_reason(outcome) == "below_threshold"

    # And no pass runs through the real post-turn path either.
    events: list[AgentEvent] = []
    session.subscribe(events.append)
    await session.prompt("another turn")
    assert not [e for e in events if isinstance(e, CompactionEndEvent)]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_1m_session_compacts_just_past_600k(tmp_path, monkeypatch):
    """min(80% x 1M, 600k) = 600k: the absolute ceiling governs a huge window,
    because re-sending 800k tokens on every request is slow and expensive even
    though it fits."""
    session = make_session(tmp_path)
    await talk(session)

    pin_measured_context(monkeypatch, 600_000)  # exactly on the line: stable
    refused = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(refused) == "below_threshold"

    pin_measured_context(monkeypatch, 600_001)
    plan = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(plan) is None  # a plan, not a refusal
    outcome = await session._run_compaction(as_plan(plan), reason="context-window")
    assert outcome.ran is True
    await session.dispose()


@pytest.mark.asyncio
async def test_a_200k_session_still_compacts_at_80_percent(tmp_path, monkeypatch):
    """The 600k absolute default is larger than the whole 200k window. min()
    makes it inert rather than disabling compaction: the percentage governs and
    the pass fires at 160k."""
    session = make_session(tmp_path, model=SMALL_MODEL)
    await talk(session)

    pin_measured_context(monkeypatch, 160_000)
    refused = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(refused) == "below_threshold"

    pin_measured_context(monkeypatch, 160_001)
    plan = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(plan) is None
    outcome = await session._run_compaction(as_plan(plan), reason="context-window")
    assert outcome.ran is True
    await session.dispose()


@pytest.mark.asyncio
async def test_the_gate_measures_the_provider_figure_not_just_the_estimate(tmp_path, monkeypatch):
    """``compaction_context_tokens`` is ``max(provider-reported, local)``, and
    the gate acts on that maximum. The plan carries BOTH figures: the maximum
    (``context_tokens``, what the gate compared and what the receipt now
    quotes as "before") and the bare local estimate (``tokens_before``, the
    transcript-entry bookkeeping). They used to be printed crosswise — the
    receipt quoted the local estimate while the band showed the provider
    figure, so a pass that fired at a provider-reported 600k printed
    "319.4k → …" and read as the two disagreeing about what just happened.
    Pinning the fields apart keeps the split documented.
    """
    session = make_session(tmp_path)
    await talk(session)
    pin_measured_context(monkeypatch, 234_800)
    session._last_usage = Usage(input_tokens=1, context_tokens=600_500)

    plan = as_plan(await session._plan_compaction(respect_threshold=True))

    assert plan.context_tokens == 600_500  # what the gate compared
    assert plan.tokens_before == 234_800  # what the receipt will print
    await session.dispose()


@pytest.mark.asyncio
async def test_config_knobs_move_the_session_gate(tmp_path, monkeypatch):
    """Both knobs are live at the session gate, not just in the resolver."""
    session = make_session(
        tmp_path,
        compaction_settings=CompactionSettings(
            keep_recent_tokens=KEEP_RECENT, threshold_percent=0.20, threshold_tokens=600_000
        ),
    )
    await talk(session)

    # 20% of 1M = 200k now governs (smaller than the 600k ceiling).
    pin_measured_context(monkeypatch, 200_000)
    refused = await session._plan_compaction(respect_threshold=True)
    assert refusal_reason(refused) == "below_threshold"
    pin_measured_context(monkeypatch, 200_001)
    assert refusal_reason(await session._plan_compaction(respect_threshold=True)) is None
    await session.dispose()


# ---------------------------------------------------------------------------
# Mid-run starvation: the gate fires but the pass never lands
# ---------------------------------------------------------------------------


def _tool_call_assistant(index: int) -> Message:
    """An assistant message that issues one tool call, as a real run does."""
    message = Message.assistant(f"step {index}")
    message.tool_calls = [ToolCall(id=f"c{index}", name="echo", arguments={})]
    return message


def _tool_result(index: int, text: str) -> Message:
    return Message(
        role="tool", tool_call_id=f"c{index}", tool_name="echo", content=[TextContent(text=text)]
    )


def _history_captured_mid_run(rounds: int) -> list[Message]:
    """History as it looks at a mid-run tool-loop boundary.

    The distinguishing feature is the TAIL: the newest messages are a tool
    call and its result, with no terminal assistant turn, because the run has
    not finished. That is the shape the mid-turn gate plans against.
    """
    big = "word " * 4000
    messages: list[Message] = [
        Message.user("older turn A " + big),
        Message.assistant("older reply A " + big),
        Message.user("go and do the long thing"),
    ]
    for index in range(rounds):
        messages.append(_tool_call_assistant(index))
        messages.append(_tool_result(index, big))
    return messages


def test_a_history_ending_mid_tool_run_is_compactable():
    """THE bug, at the level of the decision that produced it.

    A long tool run ends in a tool cluster, and every message in that cluster
    was treated as an illegal cut point, so ``find_cut_point`` answered
    ``None`` — "nothing to compact" — no matter how large the context grew.
    The session reported a context far above its configured threshold while
    every mid-turn gate refused, and relief arrived only when the run ended.

    Pinned as a cut point plus a pairing check rather than an index, because
    WHERE the cut lands is an implementation detail and "it lands somewhere
    legal" is the property.
    """
    messages = _history_captured_mid_run(rounds=8)

    cut = find_cut_point(messages, 20_000)

    assert cut is not None, "a mid-run history must be compactable"
    assert messages[cut].role != "tool", "a cut may never land on a tool result"
    to_summarize, kept = prepare_partitions(messages, cut)
    assert to_summarize, "the pass must actually reclaim something"
    # Pairing: no result kept while the call that issued it is summarized.
    # ``prepare_partitions`` hands back ``Message | CustomMessage``; only a
    # ``Message`` carries calls or a role, so narrow rather than assume.
    kept_calls = {c.id for m in kept if isinstance(m, Message) for c in m.tool_calls}
    summarized_calls = {c.id for m in to_summarize if isinstance(m, Message) for c in m.tool_calls}
    for message in kept:
        if isinstance(message, Message) and message.role == "tool":
            assert message.tool_call_id in kept_calls
            assert message.tool_call_id not in summarized_calls


def test_the_cut_keeps_working_as_the_run_grows():
    """Not a one-off: every boundary of a lengthening run stays compactable.

    The first fix attempt made the FIRST mid-run pass possible and the session
    still starved afterwards, because the post-compaction history is itself a
    marker followed by one long tool chain. Sweeping the run length is what
    catches that.
    """
    # From 5 rounds up: below that the whole history (~24k) still fits inside
    # the 20k keep-recent window, where "nothing to compact" is the correct
    # answer rather than the starvation this pins.
    for rounds in range(5, 20):
        messages = _history_captured_mid_run(rounds)
        cut = find_cut_point(messages, 20_000)
        assert cut is not None, f"run of {rounds} tool rounds was uncompactable"
        assert messages[cut].role != "tool"
