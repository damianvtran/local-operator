"""Boundary-respecting cancel: stop after the tool finishes, not mid-tool.

The distinction these tests defend is the whole reason the feature exists. An
``abort`` fires the turn's signal and cancels the running tool task, which is
right for a human pressing Esc. A supervised agent's tools have external side
effects a machine operator cannot repair — a half-transferred ``git push``, a
partly-created merge request — so a supervisor's cancel must land at the
boundary where every call has ALREADY produced a paired result, and before the
next model request is spent. The loop's RECOVERY decisions (the nameless-call
re-ask, the empty-truncation retreat) are next-model safe points too: each of
them must read the same boundary before it spends another request.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.loop import (
    LENGTH_STOP_RECOVERY_PROMPT,
    NAMELESS_CALL_RECOVERY_PROMPT,
    AgentLoop,
    LoopContext,
)
from local_operator.harness.types import (
    AgentEndEvent,
    AgentTool,
    Message,
    ModelSpec,
    NoticeEvent,
    RenderedStreamError,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolExecutionEndEvent,
    ToolResult,
)
from tests.unit.harness.test_loop import make_config


def _tool(name: str = "touch", on_execute: Any = None) -> AgentTool:
    async def execute(  # type: ignore[no-untyped-def]
        tool_call_id, args, signal, on_update, context
    ):
        if on_execute is not None:
            on_execute()
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(name=name, parameters={"type": "object"}, execute=execute)


class _Scripted:
    """One tool call on the first model call, plain text on any later one.

    ``calls`` is the assertion that matters: it proves whether the loop spent
    another provider request after the cancel.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield StreamToolCallDelta(index=0, id="call_1", name="touch")
            yield StreamToolCallDelta(index=0, argument_delta="{}")
            yield StreamEndEvent(stop_reason="toolUse")
            return
        yield StreamEndEvent(stop_reason="stop")


async def _events(tool: AgentTool, **config_kwargs: Any) -> tuple[list[Any], _Scripted]:
    stream = _Scripted()
    context = LoopContext(tools=[tool])
    config = make_config(stream, **config_kwargs)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]
    return events, stream


@pytest.mark.asyncio
async def test_cancel_lands_after_the_tool_completes_not_during() -> None:
    """The in-flight tool RUNS TO COMPLETION and no second model call is made.

    This is the property that protects an external side effect: the tool body
    finishes even though the cancel becomes pending while it is running.
    """
    completed: list[str] = []
    pending = {"cancel": False}

    def during_tool() -> None:
        # The supervisor cancels WHILE the tool is in flight, mirroring a
        # cancel pressed during a push.
        pending["cancel"] = True
        completed.append("tool finished")

    events, stream = await _events(
        _tool(on_execute=during_tool),
        graceful_cancel_requested=lambda: pending["cancel"],
    )

    assert completed == ["tool finished"], "the tool must not be cut mid-flight"
    assert stream.calls == 1, "no model request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True, "the turn must end as aborted"
    # The finished work stays in the transcript as a paired tool result.
    assert any(isinstance(event, ToolExecutionEndEvent) for event in events)


@pytest.mark.asyncio
async def test_no_cancel_runs_to_normal_completion() -> None:
    """Control: with no cancel pending the loop takes its second model call."""
    events, stream = await _events(_tool(), graceful_cancel_requested=lambda: False)

    assert stream.calls == 2
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


@pytest.mark.asyncio
async def test_absent_hook_preserves_existing_behaviour() -> None:
    """``None`` — every existing host — must behave exactly as before."""
    events, stream = await _events(_tool())

    assert stream.calls == 2
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


@pytest.mark.asyncio
async def test_a_raising_hook_does_not_strand_the_turn() -> None:
    """A broken host hook degrades to "no cancel", never kills the run."""

    def explode() -> bool:
        raise RuntimeError("host is broken")

    events, stream = await _events(_tool(), graceful_cancel_requested=explode)

    assert stream.calls == 2, "a raising hook must not stop the turn"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


# ---------------------------------------------------------------------------
# Recovery decisions are next-model safe points too.
#
# The post-tool boundary is not the only place the loop decides to spend
# another provider request: the nameless-call re-ask and the empty-truncation
# retreat each `continue` BEFORE the boundary, so a boundary cancel requested
# on the Nth request used to be outrun by its own recovery -- request N+1 was
# spent after `request_graceful_cancel` had landed (QA step-boundary matrix,
# nameless-recovery: three root rows where the caller's budget admitted two).
# These pins hold every such recovery to the same rule as the boundary itself,
# and hold the controls that the recoveries still fire without a cancel.
# ---------------------------------------------------------------------------


class _NamelessOnSecondCall:
    """A tool batch, then a NAMELESS call, then text.

    The second request's only call carries arguments and no name, so the loop
    drops it at assembly and arms its nameless-call recovery -- the decision
    that used to `continue` straight past the post-tool boundary. ``cancelled``
    flips while that request streams: the harness-level stand-in for a
    supervisor's ``request_graceful_cancel`` landing with request 2 in flight.
    ``calls`` is the assertion that matters: whether a third provider request
    was spent after the boundary was requested.
    """

    def __init__(self, *, cancel_on_nameless: bool) -> None:
        self.calls = 0
        self.cancel_on_nameless = cancel_on_nameless
        self.cancelled = False

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield StreamToolCallDelta(index=0, id="call_1", name="touch")
            yield StreamToolCallDelta(index=0, argument_delta="{}")
            yield StreamEndEvent(stop_reason="toolUse")
            return
        if self.calls == 2:
            yield StreamToolCallDelta(index=0, id="call_nameless", argument_delta="{}")
            if self.cancel_on_nameless:
                self.cancelled = True
            yield StreamEndEvent(stop_reason="toolUse")
            return
        yield StreamTextDelta(delta="done")
        yield StreamEndEvent(stop_reason="stop")


@pytest.mark.asyncio
async def test_nameless_recovery_does_not_spend_a_request_after_the_cancel() -> None:
    """A requested boundary outranks the nameless-call re-ask (the N+1 leak).

    Before this pin the recovery ``continue`` skipped the post-tool boundary,
    so the loop spent request 3 after ``request_graceful_cancel`` had landed on
    request 2 -- one more model call than the caller's budget admitted, with
    the terminal only then aborted. Post-fix the turn ends inside request 2,
    as aborted, and the drop is still told to the user.
    """
    stream = _NamelessOnSecondCall(cancel_on_nameless=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 2, "no third request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True, "the turn must end as aborted"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("without a tool name" in text for text in notices), "the drop is still told"
    assert not any("re-asking" in text for text in notices), "no re-ask was attempted"
    assert all(
        NAMELESS_CALL_RECOVERY_PROMPT not in getattr(message, "text", "")
        for message in ends[-1].messages
    ), "the recovery prompt must not be on the wire"


@pytest.mark.asyncio
async def test_nameless_recovery_still_reasks_without_a_cancel() -> None:
    """Control: with no cancel the re-ask still happens, prompt and all."""
    stream = _NamelessOnSecondCall(cancel_on_nameless=False)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 3, "the recovery re-ask must still be spent"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted
    assert any(
        NAMELESS_CALL_RECOVERY_PROMPT in getattr(message, "text", "")
        for message in ends[-1].messages
    ), "the recovery prompt rides the re-ask"


#: A model WITH a reasoning ladder. The empty-truncation retreat can only
#: exist when ``_lower_effort`` finds a rung below the current one; the
#: ladder-less default model answers ``None``, so the retry arm needs this.
_LADDER_MODEL = ModelSpec(
    provider="test",
    model_id="m",
    reasoning_efforts=("low", "medium", "high"),
    reasoning_effort="high",
)


class _SilentLengthThenText:
    """A SILENT length stop (no text, no calls) on request 1; text later."""

    def __init__(self, *, cancel_on_silent: bool) -> None:
        self.calls = 0
        self.cancel_on_silent = cancel_on_silent
        self.cancelled = False

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            if self.cancel_on_silent:
                self.cancelled = True
            yield StreamEndEvent(stop_reason="length")
            return
        yield StreamTextDelta(delta="recovered")
        yield StreamEndEvent(stop_reason="stop")


@pytest.mark.asyncio
async def test_empty_truncation_retreat_does_not_spend_a_request_after_the_cancel() -> None:
    """The same N+1 rule for the silent-length retreat (the retry arm).

    The retreat's `continue` was the second path that outran the boundary.
    With a cancel requested the turn must end inside the truncated request --
    and the notice must name the boundary rather than claim a ladder rung the
    loop never consulted.
    """
    stream = _SilentLengthThenText(cancel_on_silent=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(
        stream, model=_LADDER_MODEL, graceful_cancel_requested=lambda: stream.cancelled
    )
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 1, "no retreat request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True
    assert config.model.reasoning_effort == "high", "the retreat must not be consumed"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("a stop was requested" in text for text in notices)


@pytest.mark.asyncio
async def test_empty_truncation_retreat_still_retries_without_a_cancel() -> None:
    """Control: the retreat itself still works, at one rung lower (QA GAP)."""
    stream = _SilentLengthThenText(cancel_on_silent=False)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, model=_LADDER_MODEL, graceful_cancel_requested=lambda: False)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 2, "the retreat retry must still be spent"
    assert config.model.reasoning_effort == "medium", "the retry must step one rung down"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("retrying once at effort medium" in text for text in notices)
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


#: A model with NO ladder: ``_lower_effort`` always answers ``None``, so the
#: no-rung RE-ASK is the arm that owns its silent length stops (the
#: ``radient/auto`` sentinel's shape without needing the router registry).
_NO_LADDER_MODEL = ModelSpec(provider="test", model_id="m")


@pytest.mark.asyncio
async def test_the_no_rung_reask_does_not_spend_a_request_after_the_cancel() -> None:
    """The new re-ask arm reads the same boundary as its siblings.

    A silent length stop on a no-rung model is the ONE case the loop
    self-completes (``MAX_LENGTH_SILENT_RECOVERIES``), and a re-ask spends a
    provider request -- so a stop the caller requested must not pay for it,
    exactly as the lower-effort retreat above must not. The boundary is the
    same predicate at the same point (``_graceful_cancel_requested``), read
    before the ``continue`` that would spend request N+1.
    """
    stream = _SilentLengthThenText(cancel_on_silent=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(
        stream, model=_NO_LADDER_MODEL, graceful_cancel_requested=lambda: stream.cancelled
    )
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 1, "no re-ask request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True
    # A REQUESTED stop is not branded as an involuntary cut-off: it ends as the
    # deliberate aborted shape, with no output-limit cause riding it.
    assert ends[-1].cut_off_cause == ""
    assert all(
        LENGTH_STOP_RECOVERY_PROMPT not in getattr(message, "text", "")
        for message in ends[-1].messages
    ), "the recovery prompt must not be on the wire"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    # The silent-arm notice fires (there is nothing on screen to explain the
    # ending), and it must NOT claim a re-ask that was never spent.
    assert any("no visible output" in text for text in notices)
    assert not any("re-asking" in text for text in notices), "no re-ask was attempted"


@pytest.mark.asyncio
async def test_the_no_rung_reask_still_reasks_without_a_cancel() -> None:
    """Control for the cell above: the re-ask is spent when nothing asked for a stop.

    Without this pair the cancel test would also pass on a loop that never
    re-asks at all, which is the exact regression the arm was added to prevent.
    """
    stream = _SilentLengthThenText(cancel_on_silent=False)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, model=_NO_LADDER_MODEL, graceful_cancel_requested=lambda: False)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 2, "the re-ask must be spent"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("re-asking it to continue the work" in text for text in notices)
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


class _MalformedArgumentsOnce:
    """One request carrying a call with unparseable arguments; text later."""

    def __init__(self, *, cancel_on_call: bool) -> None:
        self.calls = 0
        self.cancel_on_call = cancel_on_call
        self.cancelled = False

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield StreamToolCallDelta(index=0, id="call_bad", name="touch")
            yield StreamToolCallDelta(index=0, argument_delta='{"path": ')
            if self.cancel_on_call:
                self.cancelled = True
            yield StreamEndEvent(stop_reason="toolUse")
            return
        yield StreamTextDelta(delta="done")
        yield StreamEndEvent(stop_reason="stop")


@pytest.mark.asyncio
async def test_malformed_arguments_respect_the_boundary() -> None:
    """Unparseable arguments pair as an invalid-arguments result, then the
    boundary cancels -- the call is recorded, not re-asked, no N+1."""
    stream = _MalformedArgumentsOnce(cancel_on_call=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 1, "no second request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True
    assert any(
        "not valid JSON" in getattr(message, "text", "") for message in ends[-1].messages
    ), "the malformed call must leave its paired invalid-arguments record"


# ---------------------------------------------------------------------------
# Failure-response resends are next-model safe points too.
#
# The round-1 review (R1) and the QA matrix (Q1/Q2) reproduced the same leak
# class at three more decisions: the connectivity continuation, the
# reasoning-echo fill, and the thinking-off retreat each `continue`d before
# the post-response boundary, spending request N+1 -- and, for the echo pair,
# mutating the run's model -- after a stop had been requested at request N.
# Each now reads the same protected hook and, when a stop is pending, falls
# through to the failure terminal, which marks the stop as the verdict.
# ---------------------------------------------------------------------------


class _RenderedCut(RenderedStreamError):
    """A mid-stream cut as the provider layer renders one: resumable."""

    connectivity_loss = True


class _CutThenText:
    """A cut mid-answer on request 1; text on later ones."""

    def __init__(self, *, cancel_on_cut: bool) -> None:
        self.calls = 0
        self.cancel_on_cut = cancel_on_cut
        self.cancelled = False

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            yield StreamTextDelta(delta="partial network answer")
            if self.cancel_on_cut:
                self.cancelled = True
            raise _RenderedCut("response stream cut mid-answer")
        yield StreamTextDelta(delta="continued")
        yield StreamEndEvent(stop_reason="stop")


@pytest.mark.asyncio
async def test_connectivity_resume_does_not_spend_a_request_after_the_cancel() -> None:
    """A requested stop outranks the connectivity continuation (R1/Q1).

    Resuming would spend request 2 on a turn the caller stopped at request 1,
    and the resume notice would promise the remainder "is being fetched on the
    next attempt" -- a promise the stop then breaks.
    """
    stream = _CutThenText(cancel_on_cut=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 1, "no resume request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True, "the stop is the terminal verdict"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert not any("resuming" in text for text in notices), "no fake resume claim"


@pytest.mark.asyncio
async def test_connectivity_resume_still_fires_without_a_cancel() -> None:
    """Control: the continuation itself still works (R1 neighbouring case)."""
    stream = _CutThenText(cancel_on_cut=False)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 2, "the resume request must still be spent"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("resuming the turn (1/3)" in text for text in notices)
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


class _EchoRefusalOnce:
    """A reasoning-echo refusal on request 1; text on later ones."""

    def __init__(self, *, cancel_on_refusal: bool) -> None:
        self.calls = 0
        self.cancel_on_refusal = cancel_on_refusal
        self.cancelled = False

    async def __call__(self, request: Any, signal: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            if self.cancel_on_refusal:
                self.cancelled = True
            yield StreamEndEvent(stop_reason="error", error="reasoning_content must be passed back")
            return
        yield StreamTextDelta(delta="done")
        yield StreamEndEvent(stop_reason="stop")


@pytest.mark.asyncio
async def test_echo_fill_resend_does_not_spend_a_request_after_the_cancel() -> None:
    """A requested stop outranks the echo fill, with no model mutation."""
    stream = _EchoRefusalOnce(cancel_on_refusal=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 1, "no resend request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert not any("retrying with the echo" in text for text in notices)
    assert config.model.requires_reasoning_echo is False, "no wire mutation after the stop"


@pytest.mark.asyncio
async def test_echo_fill_resend_still_fires_without_a_cancel() -> None:
    """Control: the fill itself still works, echo flag and all."""
    stream = _EchoRefusalOnce(cancel_on_refusal=False)
    context = LoopContext(tools=[_tool()])
    config = make_config(stream, graceful_cancel_requested=lambda: stream.cancelled)
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 2, "the resend request must still be spent"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("retrying with the echo filled in" in text for text in notices)
    assert config.model.requires_reasoning_echo is True
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted


#: A model that already requires the echo AND has a ladder: the echo fill is
#: inapplicable (the bit is set) and the retreat has a "none" rung to switch to.
_ECHO_LADDER_MODEL = ModelSpec(
    provider="test",
    model_id="m",
    reasoning_efforts=("none", "low", "medium", "high"),
    reasoning_effort="high",
    requires_reasoning_echo=True,
)


@pytest.mark.asyncio
async def test_echo_retreat_does_not_spend_a_request_after_the_cancel() -> None:
    """A requested stop outranks the thinking-off retreat (R1/Q2)."""
    stream = _EchoRefusalOnce(cancel_on_refusal=True)
    context = LoopContext(tools=[_tool()])
    config = make_config(
        stream, model=_ECHO_LADDER_MODEL, graceful_cancel_requested=lambda: stream.cancelled
    )
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 1, "no retreat request may be spent after the cancel"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and ends[-1].aborted is True
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert not any("thinking disabled" in text for text in notices)
    assert config.model.reasoning_effort == "high", "no effort mutation after the stop"


@pytest.mark.asyncio
async def test_echo_retreat_still_fires_without_a_cancel() -> None:
    """Control: the retreat itself still works, one rung to thinking off."""
    stream = _EchoRefusalOnce(cancel_on_refusal=False)
    context = LoopContext(tools=[_tool()])
    config = make_config(
        stream, model=_ECHO_LADDER_MODEL, graceful_cancel_requested=lambda: stream.cancelled
    )
    events = [event async for event in AgentLoop().run([Message.user("go")], context, config, None)]

    assert stream.calls == 2, "the retreat request must still be spent"
    notices = [event.text for event in events if isinstance(event, NoticeEvent)]
    assert any("retrying with thinking disabled" in text for text in notices)
    assert config.model.reasoning_effort == "none"
    ends = [event for event in events if isinstance(event, AgentEndEvent)]
    assert ends and not ends[-1].aborted
