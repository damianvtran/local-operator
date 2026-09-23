"""The progress probe must judge a child LANE's own step, not just its streams.

The sibling file (``test_process_child_model_progress.py``) pins the half that
came first: a FORKED CHILD STREAM awaiting its provider is process progress, read
at O(1) off the shared counter. This file pins the half that counter cannot
express — a step OPEN in the lane itself, its own tool batch or its own on-demand
compaction — because the shape it closes is a manager whose own loop burns CPU
walking the roster projection while its lanes hold steps: no motion, nothing in
flight by the parent's own reading, CPU advancing, and the process cut while it
is working (``stall_watchdog``'s docstring names the shape and the O(N^2) walk
behind it).

REAL STRUCTURES, not a model of them: a real ``Session`` for the manager, a real
``SubagentComms`` with real records holding real child ``Session``s built by the
product's own constructor, attached through the product's own ``record_launch`` /
``attach``, and real ``Message`` tails read by the product's own
``unanswered_tail_call_ids``.

THE COST TABLE in ``stall_watchdog``'s docstring is a CPU measurement (that file
explains why wall time is unusable on this host); what is asserted HERE is the
structural half of it, which is what a test can hold: every live lane is asked
once per sample, a lane holding a step short-circuits the scan, and a settled
lane is never asked at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness.comms import SubagentComms
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    TextContent,
    ToolCall,
)
from local_operator.session.runtime import process
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def _never_streams(
    request: ChatRequest, signal: AbortSignal | None
) -> AsyncIterator[StreamEvent]:
    """These cells never run a turn; the stream function only has to exist."""
    if False:  # pragma: no cover - makes this an async generator
        yield StreamEndEvent(stop_reason="stop")


def _session(tmp_path, name: str) -> Session:
    """A real session, built the way the product builds one."""
    return Session(
        model=MODEL,
        stream_fn=_never_streams,
        tools=[],
        transcript=Transcript(tmp_path / name),
        system_blocks_provider=lambda: ["stable"],
    )


def _lane(comms: SubagentComms, job_id: str) -> Session:
    """The live child SESSION behind a record.

    ``record.child`` is typed as the narrow ``ChildSession`` protocol — the whole
    coupling the comms module needs — while these cells are about the attributes a
    real ``Session`` has and a lane is executing.
    """
    return cast("Session", comms._records[job_id].child)


def _handle(session: Session) -> SimpleNamespace:
    """The runtime's own probe reads the handle's ``_session``; that is all it needs."""
    return SimpleNamespace(_session=session)


def _completed_turn(lane: Session) -> None:
    """A lane whose last step landed: no calls open, i.e. what a parked call looks like."""
    lane._context.messages.append(Message.user("do the thing"))
    lane._context.messages.append(
        Message.assistant(tool_calls=[ToolCall(id="c1", name="bash", arguments={})])
    )
    lane._context.messages.append(
        Message(role="tool", content=[TextContent(text="ok")], tool_call_id="c1", tool_name="bash")
    )
    lane._context.messages.append(Message.assistant("step done"))


def _open_batch(lane: Session) -> None:
    """A lane mid-batch: the live tail is an assistant message whose calls are unanswered.

    The shape ``Session``'s own ``_wire_legal_snapshot`` documents as holding "for
    the whole duration of every tool batch", and the shape a lane parked in a long
    in-process tool is in.
    """
    lane._context.messages.append(Message.user("run the long tool"))
    lane._context.messages.append(
        Message.assistant(tool_calls=[ToolCall(id="open-1", name="bash", arguments={})])
    )


def _manager_with_lanes(tmp_path, lanes: int) -> tuple[Session, SubagentComms]:
    """A real manager whose real registry holds ``lanes`` real, attached lanes.

    The registry is a real ``SubagentComms``, so it has to be built AFTER the
    session it belongs to and published on ``_subagent_comms`` — which is what the
    product's own ``subagent_comms`` property does on first use (see
    :func:`test_the_lane_read_never_mints_comms_state` for why the probe reads the
    attribute rather than calling that property).
    """
    manager = _session(tmp_path, "manager")
    comms = SubagentComms(manager)
    manager._subagent_comms = comms
    for index in range(lanes):
        lane = _session(tmp_path, f"lane-{index}")
        _completed_turn(lane)
        job_id = f"job-{index}"
        comms.record_launch(job_id, f"lane {index}")
        comms.attach(job_id, lane, tmp_path / f"lane-{index}")
    return manager, comms


def test_a_lane_holding_a_tool_batch_holds_the_progress_leg(tmp_path) -> None:
    """The parent's own reading is unchanged; the LANE's open step is what spares it."""
    manager, comms = _manager_with_lanes(tmp_path, 1)
    handle = _handle(manager)
    assert process._step_in_flight(handle) is False
    _open_batch(_lane(comms, "job-0"))
    assert process._step_in_flight(handle) is True


def test_a_lane_compacting_is_a_step_in_flight(tmp_path) -> None:
    """A lane rewriting history has no open batch; its compaction is the step."""
    manager, comms = _manager_with_lanes(tmp_path, 1)
    handle = _handle(manager)
    assert process._step_in_flight(handle) is False
    _lane(comms, "job-0")._compacting = True
    assert process._step_in_flight(handle) is True


def test_a_settled_lane_is_never_asked(tmp_path) -> None:
    """``detach`` releases the child, so a settled lane stops being read."""
    manager, comms = _manager_with_lanes(tmp_path, 2)
    handle = _handle(manager)
    _open_batch(_lane(comms, "job-1"))
    assert process._step_in_flight(handle) is True

    comms.detach("job-1")
    assert comms._records["job-1"].child is None
    # The settled lane's own step is gone from the answer, and the successor
    # shape is what a lane parked in a provider call leaves behind.
    assert process._step_in_flight(handle) is False


def test_each_live_lane_is_asked_once_and_a_held_step_stops_the_scan(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The linear claim in ``stall_watchdog``'s cost table, as a structural fact."""
    manager, comms = _manager_with_lanes(tmp_path, 4)
    handle = _handle(manager)
    asked: list[Session] = []
    real = process._lane_step_in_flight

    def spy(lane: Session) -> bool:
        asked.append(lane)
        return real(lane)

    monkeypatch.setattr(process, "_lane_step_in_flight", spy)
    assert process._step_in_flight(handle) is False
    assert len(asked) == 4
    assert len({id(lane) for lane in asked}) == 4

    # Hold the step on the FIRST lane in the probe's own scan order, captured from
    # the pass above, so this cell does not depend on the registry's iteration
    # order — and the scan then stops there rather than walking the other three.
    first = asked[0]
    _open_batch(first)
    asked.clear()
    assert process._step_in_flight(handle) is True
    assert asked == [first]


def test_a_lane_mid_turn_with_nothing_open_is_not_in_flight(tmp_path) -> None:
    """The lane question is NARROW, and this is the cell that keeps it narrow.

    ``lane._is_streaming`` is true for the whole of a lane's turn — including one
    parked in a provider call that never returns, or in a gate — so using it would
    let a single stuck lane keep its parent's spin unobservable forever. A lane that
    is streaming with no step open is NOT this leg's evidence, and a future widening
    that reaches for it goes red here.
    """
    manager, comms = _manager_with_lanes(tmp_path, 1)
    lane = _lane(comms, "job-0")
    lane._is_streaming = True
    assert lane.is_streaming is True
    assert process._step_in_flight(_handle(manager)) is False


def test_a_session_with_no_roster_keeps_the_pre_widening_answer(tmp_path) -> None:
    """A host with no lane registry answers exactly what it answered before.

    The ordinary case for a session that never delegated, and for a reduced host
    or a double: nothing to read, so nothing to hold.
    """
    manager = _session(tmp_path, "manager")
    handle = _handle(manager)
    assert process._step_in_flight(handle) is False
    manager._context.messages.append(Message.user("run the long tool"))
    manager._context.messages.append(
        Message.assistant(tool_calls=[ToolCall(id="open-1", name="bash", arguments={})])
    )
    assert process._step_in_flight(handle) is True


def test_the_lane_read_never_mints_comms_state(tmp_path) -> None:
    """The sampler thread must not construct a registry as a side effect of a read.

    ``Session.subagent_comms`` MINTS on first use and the mint SUBSCRIBES a frontend
    projector (a real session reaches it during construction, through
    ``_build_tool_context``), so a probe calling that property could mutate the
    session it is only there to observe. The probe reads ``_subagent_comms``
    instead, and this arms the case where there is nothing to read.
    """
    manager = _session(tmp_path, "manager")
    manager._subagent_comms = None
    assert process._step_in_flight(_handle(manager)) is False
    assert manager._subagent_comms is None


def test_an_unreadable_lane_roster_fails_closed(tmp_path) -> None:
    """The lanes are known to be there, so "I could not read them" is not "none"."""

    class _BrokenComms:
        @property
        def _records(self) -> dict[str, Any]:
            raise RuntimeError("roster unavailable")

    manager = _session(tmp_path, "manager")
    manager._subagent_comms = _BrokenComms()  # type: ignore[assignment]
    assert process._step_in_flight(_handle(manager)) is True


def test_an_unreadable_lane_fails_closed(tmp_path) -> None:
    """One lane whose step cannot be read preserves the process, not fires a bound."""

    class _BrokenContext:
        @property
        def messages(self) -> list[Any]:
            raise RuntimeError("context unavailable")

    manager, comms = _manager_with_lanes(tmp_path, 2)
    _lane(comms, "job-1")._context = _BrokenContext()  # type: ignore[assignment]
    assert process._step_in_flight(_handle(manager)) is True


def test_a_lane_with_no_context_is_not_an_unreadable_lane(tmp_path) -> None:
    """Absent state is a fact ("nothing open"); only a RAISE is unreadable."""
    manager, comms = _manager_with_lanes(tmp_path, 2)
    _lane(comms, "job-1")._context = None  # type: ignore[assignment]
    assert process._step_in_flight(_handle(manager)) is False


def test_the_tail_scan_raises_and_the_two_callers_pick_directions() -> None:
    """One rule, two directions, chosen at the seam rather than copied per caller.

    A hand-copied second version of ``unanswered_tail_call_ids``' rule is a defect
    this module has already paid for once, so the scan is shared and the DIRECTION
    is the caller's: this session's own tail may fire a bound on a failed read,
    while a child lane must not.
    """

    class _BrokenContext:
        @property
        def messages(self) -> list[Any]:
            raise RuntimeError("context unavailable")

    broken = SimpleNamespace(_context=_BrokenContext())
    with pytest.raises(RuntimeError):
        process._unanswered_tail_step(broken)
    assert process._tool_batch_in_flight(broken) is False
    assert process._lane_step_in_flight(broken) is True
