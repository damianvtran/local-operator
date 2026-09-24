"""Structural pins for the subagent fan-out cost fixes.

Each test pins one of the four changes by WHAT they do, not by how fast they
run (see AGENTS.md "Prefer a structural invariant to a numeric one"); the
wall/CPU evidence lives in ``scripts/bench_subagent_fanout.py`` and the PR.

1. A child session's roster is its own SUBTREE of the shared comms graph, not
   every sibling's trajectory (was O(N^2) across a fan-out).
2. A child's streamed text is coalesced into one trajectory row per flush
   instead of one row per token.
3. An unobserved child does not fold per-token events into its own store.
4. A job that settles while the parent's turn is streaming is DELIVERED once
   the turn ends, instead of being dropped (the "wedged parent" report).
5. One parent roster tick builds ONE registry pass, not one per job.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.harness import subagent as subagent_module
from local_operator.harness.comms import SubagentComms
from local_operator.harness.jobs import AsyncJob
from local_operator.harness.subagent import _make_relay
from local_operator.harness.types import (
    ChatRequest,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolExecutionStartEvent,
)
from local_operator.session import frontend_state as fs
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.registry import create_tools

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


# -- 1. child roster scope ------------------------------------------------------


def _graph() -> SubagentComms:
    """root -> a -> a1 -> a1x ; root -> b ; plus an alias for a resumed ``a``."""
    comms = SubagentComms(cast(Any, SimpleNamespace(jobs=None, _job_id=None)))
    comms.record_launch("a", "A")
    comms.record_launch("b", "B")
    comms.record_launch("a1", "A1", parent_job_id="a")
    comms.record_launch("a1x", "A1X", parent_job_id="a1")
    return comms


def test_descendant_ids_is_the_strict_subtree() -> None:
    comms = _graph()
    assert comms.descendant_ids("a") == {"a1", "a1x"}
    assert comms.descendant_ids("a1") == {"a1x"}
    assert comms.descendant_ids("b") == set()
    assert comms.descendant_ids("a1x") == set()


def test_descendant_ids_follows_a_resumed_attempt_alias() -> None:
    """A resumed attempt answers for its record, so its subtree survives."""
    comms = _graph()
    comms._aliases["a-attempt-2"] = "a"
    assert comms.descendant_ids("a-attempt-2") == {"a1", "a1x"}


def test_descendant_ids_terminates_on_a_legacy_cycle() -> None:
    comms = _graph()
    comms._records["a"].parent_job_id = "a1x"  # a malformed restored snapshot
    # Must return (not hang); the cycle never reaches ``b``.
    assert "b" not in comms.descendant_ids("a1")


def test_a_child_store_projects_only_its_subtree() -> None:
    """The unscoped read put every sibling on every child's roster."""
    comms = _graph()
    child = SimpleNamespace(_job_id="a", _subagent_comms=comms, jobs=None)
    root = SimpleNamespace(_job_id=None, _subagent_comms=comms, jobs=None)
    assert fs._child_scope(child, comms) == {"a1", "a1x"}
    assert fs._child_scope(root, comms) is None
    # A job id the registry does not know is not a child: keep the old view.
    stranger = SimpleNamespace(_job_id="zzz", _subagent_comms=comms, jobs=None)
    assert fs._child_scope(stranger, comms) is None

    child_rows = {row.id for row in fs.FrontendStateStore._jobs(child)}
    root_rows = {row.id for row in fs.FrontendStateStore._jobs(root)}
    assert child_rows == {"a1", "a1x"}
    assert root_rows == {"a", "b", "a1", "a1x"}


def test_a_scope_failure_falls_back_to_the_full_graph() -> None:
    """Scoping is an optimisation: a raising registry must not hide rows."""
    comms = _graph()

    def boom(_job_id: str) -> set[str]:
        raise RuntimeError("registry unavailable")

    comms.descendant_ids = boom  # type: ignore[method-assign]
    child = SimpleNamespace(_job_id="a", _subagent_comms=comms, jobs=None)
    assert fs._child_scope(child, comms) is None


# -- 2. text coalescing in the relay ---------------------------------------------


def _rows(job: AsyncJob) -> list[dict[str, Any]]:
    """The job's retained trajectory, asserted present (the relay always sets it)."""
    assert job.trajectory is not None
    return job.trajectory


def _relay() -> tuple[Any, AsyncJob, list[str]]:
    job = AsyncJob(id="j1", type="task", status="running", label="child", start_time=1.0)
    job.trajectory = []
    notified: list[str] = []
    manager = SimpleNamespace(
        _notify_roster_change=lambda: notified.append("roster"),
        _notify_transient_job_change=lambda: notified.append("transient"),
    )

    async def _emit(_event: Any) -> None:
        return None

    relay = _make_relay(
        "j1", "child", job, cast(Any, manager), _emit, lambda _t: None, {"text": "", "error": None}
    )
    return relay, job, notified


def _assistant(mid: str = "m1") -> Message:
    return Message(id=mid, role="assistant", content=[TextContent(text="")])


def test_text_deltas_become_one_row_ahead_of_the_next_boundary() -> None:
    relay, job, _ = _relay()

    async def drive() -> None:
        await relay(MessageStartEvent(message=_assistant()))
        for word in ("a ", "b ", "c"):
            await relay(MessageUpdateEvent(message=_assistant(), delta=word))
        await relay(ToolExecutionStartEvent(tool_call_id="t", tool_name="bash", args={}))

    asyncio.run(drive())
    rows = _rows(job)
    assert [row["type"] for row in rows] == [
        "message_start",
        "message_update",
        "tool_execution_start",
    ]
    assert rows[1]["delta"] == "a b c"
    # The stamp stays dense and ordered: nothing coalesced consumed a number.
    assert [row["_lo_seq"] for row in rows] == [0, 1, 2]


def test_a_new_message_id_flushes_the_previous_messages_text() -> None:
    """One coalesced row never spans two messages."""
    relay, job, _ = _relay()

    async def drive() -> None:
        await relay(MessageUpdateEvent(message=_assistant("m1"), delta="one"))
        await relay(MessageUpdateEvent(message=_assistant("m2"), delta="two"))
        await relay(MessageEndEvent(message=_assistant("m2")))

    asyncio.run(drive())
    rows = [(r["type"], r.get("delta"), r["message"]["id"]) for r in _rows(job)]
    assert rows[:2] == [("message_update", "one", "m1"), ("message_update", "two", "m2")]


def test_quiet_text_is_flushed_by_the_timer_and_announced(monkeypatch) -> None:
    """Text followed by silence still reaches the page, and says it did."""
    monkeypatch.setattr(subagent_module, "SUBAGENT_TEXT_FLUSH_S", 0.01)
    relay, job, notified = _relay()

    async def drive() -> None:
        await relay(MessageUpdateEvent(message=_assistant(), delta="still typing"))
        assert _rows(job) == []
        await asyncio.sleep(0.05)

    asyncio.run(drive())
    assert [r["type"] for r in _rows(job)] == ["message_update"]
    assert _rows(job)[0]["delta"] == "still typing"
    assert "transient" in notified


# -- 3 + 4. real sessions ---------------------------------------------------------


def _tool_call(name: str, args: dict[str, Any], call_id: str):
    async def gen():
        yield StreamToolCallDelta(index=0, id=call_id, name=name, argument_delta=json.dumps(args))
        yield StreamEndEvent(stop_reason="toolUse")

    return gen()


def _text(body: str, pieces: int = 1):
    async def gen():
        for _ in range(pieces):
            yield StreamTextDelta(delta=body)
        yield StreamEndEvent(stop_reason="stop")

    return gen()


class _Provider:
    """Parent: launch two children, run one slow tool, then end the turn
    WITHOUT waiting. Children answer in streamed prose."""

    def __init__(self) -> None:
        self.deliveries: list[str] = []

    def __call__(self, request: ChatRequest, signal: Any = None):
        users = [m for m in request.messages if isinstance(m, Message) and m.role == "user"]
        if users and "CHILD" in users[0].text:
            return _text("tok ", pieces=40)
        done = sum(1 for m in request.messages if isinstance(m, Message) and m.role == "assistant")
        last = request.messages[-1]
        body = str(getattr(last, "text", "") or "") + json.dumps(
            getattr(last, "details", {}) or {}, default=str
        )
        if "background job '" in body:
            self.deliveries.append(body)
            return _text("noted")
        if done == 0:
            return _tool_call("task", {"label": "a", "prompt": "CHILD a"}, "c0")
        if done == 1:
            return _tool_call("task", {"label": "b", "prompt": "CHILD b"}, "c1")
        if done == 2:
            return _tool_call("bash", {"command": "sleep 1"}, "c2")
        return _text("I'll wait for the children to report back.")


async def _wait_until(predicate, timeout: float = 20.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_children_that_settle_mid_turn_are_delivered_after_it(
    tmp_path: Path, monkeypatch
) -> None:
    """The wedge: results that landed during a streaming turn were dropped.

    Falsified against the fix: with ``_deliver_deferred_job_results`` removed
    this ends with zero deliveries and an idle parent holding two finished
    children -- the state the operator reported as a session stuck waiting.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    provider = _Provider()
    parent = Session(
        model=MODEL,
        stream_fn=provider,
        tools=create_tools(ToolContext(cwd=str(tmp_path)), enabled=["bash"]),
        transcript=Transcript(tmp_path / "parent"),
        system_blocks_provider=lambda *a, **k: ["parent"],
        cwd=str(tmp_path),
        yolo=True,
    )
    await parent.async_init()
    folded: list[str] = []
    original = fs.FrontendStateStore.observe_event

    def spy(self, session, event):  # type: ignore[no-untyped-def]
        if getattr(session, "_job_id", None):
            folded.append(event.type)
        return original(self, session, event)

    monkeypatch.setattr(fs.FrontendStateStore, "observe_event", spy)
    try:
        await parent.prompt("go")
        jobs = [job for job in parent.jobs.list() if job.type == "task"]
        assert len(jobs) == 2
        await _wait_until(lambda: len(provider.deliveries) == 2)
        assert sorted("'a'" in d for d in provider.deliveries) == [False, True]
        # 3. No child folded a per-token event into its own (unobserved) store,
        # while its boundaries still folded (spend/outcome depend on them).
        assert "message_update" not in folded
        assert "message_end" in folded and "agent_end" in folded
        # 2. Each child's 40 streamed tokens are one or a few rows, not 40.
        for job in jobs:
            updates = [r for r in job.trajectory or [] if r["type"] == "message_update"]
            assert 1 <= len(updates) < 5
            assert "".join(r["delta"] for r in updates) == "tok " * 40
    finally:
        await parent.dispose()


@pytest.mark.asyncio
async def test_a_result_the_turn_collected_with_wait_is_not_delivered_again(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = Session(
        model=MODEL,
        stream_fn=_Provider(),
        tools=[],
        transcript=Transcript(tmp_path / "parent"),
        system_blocks_provider=lambda *a, **k: ["parent"],
        cwd=str(tmp_path),
    )
    delivered: list[str] = []

    def record(job_id: str, text: str, job: Any) -> None:
        delivered.append(job_id)

    parent._deliver_job_result = record  # type: ignore[method-assign]
    job = AsyncJob(id="j", type="task", status="completed", label="x", start_time=1.0)
    parent.jobs._jobs["j"] = job
    parent._is_streaming = True
    await parent._on_job_completed("j", "done", job)
    assert delivered == []
    job.consumed = True  # the turn's own ``wait`` took it
    parent._is_streaming = False
    parent._deliver_deferred_job_results()
    assert delivered == []
    await parent.dispose()


# -- 5. one registry pass per tick ------------------------------------------------


def test_a_roster_tick_builds_one_registry_pass(monkeypatch) -> None:
    comms = _graph()
    passes: list[int] = []
    original = SubagentComms.roster_pass

    def counting(self, now=None):  # type: ignore[no-untyped-def]
        passes.append(1)
        return original(self, now)

    monkeypatch.setattr(SubagentComms, "roster_pass", counting)
    root = SimpleNamespace(_job_id=None, _subagent_comms=comms, jobs=None)
    rows = fs.FrontendStateStore._jobs(root)
    assert {row.id for row in rows} == {"a", "b", "a1", "a1x"}
    # Lineage was stamped from the pass (parent ids present) ...
    assert {row.id: row.parent_job_id for row in rows}["a1x"] == "a1"
    # ... and the four per-job lookups shared ONE walk of the registry.
    assert len(passes) == 1
