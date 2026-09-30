"""The ``patience`` tool: createIf, arm/cancel/list, class and hold gates.

The engine's rules are pinned in ``tests/unit/wakes/test_patience.py``; this
file is about the TOOL surface — what a model can see and call, and the
refusals it gets when the session is not allowed to arm (reactive class, an
engine hold) or the request is malformed. The ``send`` tool's optional
``patience`` param is pinned here too (its schema and its pre-delivery
validation).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.resume import write_session_attachment
from local_operator.tools import builtin


class FakeScheduler:
    def __init__(self, schedules=()):
        self._schedules = list(schedules)
        self.updates: list[list[Any]] = []

    @property
    def schedules(self):
        return tuple(self._schedules)

    async def update(self, schedules):
        self._schedules = list(schedules)
        self.updates.append(list(schedules))


def make_session_dir(tmp_path: Path, agent: str = "aida") -> Path:
    session_dir = tmp_path / "sessions" / "sess00000001"
    session_dir.mkdir(parents=True, exist_ok=True)
    write_session_attachment(session_dir, team="", agent=agent, goal="")
    return session_dir


def make_context(
    tmp_path: Path,
    monkeypatch,
    *,
    session_dir: Path | None = None,
    action_class: str = "proactive",
    scheduler: FakeScheduler | None = None,
    proactive_hold: bool = False,
    sink=None,
) -> ToolContext:
    monkeypatch.setattr("local_operator.paths.config_dir", lambda: tmp_path)
    return ToolContext(
        cwd=str(tmp_path),
        session_id="sess00000001",
        session_dir=str(session_dir) if session_dir is not None else None,
        wake_scheduler=scheduler if scheduler is not None else FakeScheduler(),
        action_class=action_class,
        proactive_hold=proactive_hold,
        patience_sink=sink,
    )


class TestCreateIf:
    def test_proactive_with_a_scheduler_gets_the_tool(self, tmp_path, monkeypatch) -> None:
        context = make_context(tmp_path, monkeypatch)
        tool = builtin.build_patience_tool(context)
        assert tool is not None and tool.name == "patience"

    def test_a_reactive_session_pays_no_schema(self, tmp_path, monkeypatch) -> None:
        context = make_context(tmp_path, monkeypatch, action_class="reactive")
        assert builtin.build_patience_tool(context) is None

    def test_no_scheduler_no_tool(self, tmp_path, monkeypatch) -> None:
        context = make_context(tmp_path, monkeypatch)
        context.wake_scheduler = None
        assert builtin.build_patience_tool(context) is None


async def call_arm(tool, context, args):
    return await tool.execute("call-1", args, None, None, context)


class TestArm:
    @pytest.mark.asyncio
    async def test_arm_writes_a_hidden_row_and_reports_the_wait(
        self, tmp_path, monkeypatch
    ) -> None:
        session_dir = make_session_dir(tmp_path)
        scheduler = FakeScheduler()
        seen: list[str] = []
        context = make_context(
            tmp_path, monkeypatch, session_dir=session_dir, scheduler=scheduler, sink=seen.append
        )
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await call_arm(tool, context, {"op": "arm", "timeout": "5m", "note": "ask"})
        assert result.is_error is False
        row = scheduler.schedules[0]
        assert row.kind == "patience" and row.hidden is True
        assert row.attempt == 1 and row.note == "ask"
        assert row.armed_after == ""  # flushed at turn end via the sink
        assert seen == [row.id]
        assert "Hidden" in result.text or "hidden" in result.text

    @pytest.mark.asyncio
    async def test_an_explicit_after_target_skips_the_turn_end_flush(
        self, tmp_path, monkeypatch
    ) -> None:
        session_dir = make_session_dir(tmp_path)
        scheduler = FakeScheduler()
        seen: list[str] = []
        context = make_context(
            tmp_path, monkeypatch, session_dir=session_dir, scheduler=scheduler, sink=seen.append
        )
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await call_arm(
            tool, context, {"op": "arm", "after": "message:m123", "timeout": "90s"}
        )
        assert result.is_error is False
        assert scheduler.schedules[0].armed_after == "message:m123"
        assert seen == []

    @pytest.mark.asyncio
    async def test_a_bad_duration_is_a_validation_error(self, tmp_path, monkeypatch) -> None:
        context = make_context(tmp_path, monkeypatch, session_dir=make_session_dir(tmp_path))
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await call_arm(tool, context, {"op": "arm", "timeout": "banana"})
        assert result.is_error is True and "invalid timeout" in result.text

    @pytest.mark.asyncio
    async def test_a_bad_after_spelling_is_a_validation_error(self, tmp_path, monkeypatch) -> None:
        context = make_context(tmp_path, monkeypatch, session_dir=make_session_dir(tmp_path))
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await call_arm(tool, context, {"op": "arm", "after": "later"})
        assert result.is_error is True and "message:" in result.text

    @pytest.mark.asyncio
    async def test_switching_the_class_mid_session_refuses_the_arm(
        self, tmp_path, monkeypatch
    ) -> None:
        # createIf was evaluated a turn ago; the ARM path re-reads, which is
        # what makes a switch land without restarting the session.
        from local_operator.action_class import REACTIVE, set_registered_action_class
        from local_operator.agents import AgentRegistry

        session_dir = make_session_dir(tmp_path)
        context = make_context(tmp_path, monkeypatch, session_dir=session_dir)
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        registry = AgentRegistry(tmp_path)
        set_registered_action_class(registry, "aida", REACTIVE)
        context.agent_registry = registry
        result = await call_arm(tool, context, {"op": "arm"})
        assert result.is_error is True and "reactive" in result.text

    @pytest.mark.asyncio
    async def test_the_engine_hold_refuses_new_arms(self, tmp_path, monkeypatch) -> None:
        session_dir = make_session_dir(tmp_path)
        context = make_context(tmp_path, monkeypatch, session_dir=session_dir, proactive_hold=True)
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await call_arm(tool, context, {"op": "arm"})
        assert result.is_error is True and "paused" in result.text

    @pytest.mark.asyncio
    async def test_a_context_without_a_session_directory_refuses_cleanly(
        self, tmp_path, monkeypatch
    ) -> None:
        context = make_context(tmp_path, monkeypatch, session_dir=None)
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await call_arm(tool, context, {"op": "arm"})
        assert result.is_error is True and "session directory" in result.text


class TestCancelAndList:
    @pytest.mark.asyncio
    async def test_list_and_cancel_round_trip(self, tmp_path, monkeypatch) -> None:
        from local_operator.harness.wake_types import WakeSchedule

        now = int(time.time() * 1000)
        row = WakeSchedule(
            id="patience-1",
            message="",
            next_due_at=now + 300_000,
            created_at=now,
            kind="patience",
            hidden=True,
            episode_id="patience-1",
            attempt=1,
            armed_at=now,
        )
        scheduler = FakeScheduler([row])
        context = make_context(
            tmp_path, monkeypatch, session_dir=make_session_dir(tmp_path), scheduler=scheduler
        )
        tool = builtin.build_patience_tool(context)
        assert tool is not None

        listed = await tool.execute("call-1", {"op": "list"}, None, None, context)
        assert listed.is_error is False and "patience-1" in listed.text

        cancelled = await tool.execute(
            "call-2", {"op": "cancel", "id": "patience-1"}, None, None, context
        )
        assert cancelled.is_error is False and "patience-1" in cancelled.text
        assert [r.id for r in scheduler.schedules] == []

        nothing = await tool.execute("call-3", {"op": "cancel"}, None, None, context)
        assert nothing.is_error is False and "No pending" in nothing.text

    @pytest.mark.asyncio
    async def test_cancel_with_an_unknown_id_names_the_known_ones(
        self, tmp_path, monkeypatch
    ) -> None:
        from local_operator.harness.wake_types import WakeSchedule

        now = int(time.time() * 1000)
        row = WakeSchedule(
            id="patience-1",
            message="",
            next_due_at=now + 300_000,
            kind="patience",
            hidden=True,
            episode_id="patience-1",
            attempt=1,
            armed_at=now,
        )
        context = make_context(
            tmp_path,
            monkeypatch,
            session_dir=make_session_dir(tmp_path),
            scheduler=FakeScheduler([row]),
        )
        tool = builtin.build_patience_tool(context)
        assert tool is not None
        result = await tool.execute(
            "call-1", {"op": "cancel", "id": "patience-9"}, None, None, context
        )
        assert result.is_error is True and "patience-1" in result.text


class TestSendPatienceParam:
    def test_the_send_schema_takes_a_duration(self) -> None:
        from local_operator.tools.builtin import SendParams

        params = SendParams(target="builder", message="hi", patience="5m")
        assert params.patience == "5m"

    @pytest.mark.asyncio
    async def test_a_bad_patience_spelling_refuses_before_delivery(
        self, tmp_path, monkeypatch
    ) -> None:
        # No peers exist here, so if the call got PAST validation it would have
        # failed with a target-resolution error instead. The patience sentence
        # is the discriminator: validation runs first, and nothing was sent.
        context = ToolContext(cwd=str(tmp_path), session_id="s")
        result = await builtin.execute_send(
            "call-1",
            {"target": "nobody-at-all", "message": "hi", "patience": "banana"},
            None,
            None,
            context,
        )
        assert result.is_error is True and "invalid patience" in result.text
