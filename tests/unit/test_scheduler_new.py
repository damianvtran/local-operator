"""Unit tests for the rewritten SchedulerService (new engine, CL-07).

Covers: one-time fires once and pops, interval schedules reschedule,
end_time deactivates, past-due one-time replays on load, run failures are
recorded as FAILED without killing the loop, and the schedules.jsonl
persistence round trip. Session creation is faked by monkeypatching
``local_operator.session_factory.create_session``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator import session_factory
from local_operator import scheduler_service as scheduler_module
from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.jobs import JobManager, JobStatus
from local_operator.scheduler_service import SchedulerService
from local_operator.types import Schedule, ScheduleUnit

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSession:
    """Minimal stand-in for the rewritten engine's Session.

    Duck-typed against the parts the scheduler uses: ``subscribe`` /
    ``prompt`` / ``dispose``. Emits one ``agent_end`` per successful prompt.
    """

    def __init__(self, fail: BaseException | None = None):
        self.prompts: list[str] = []
        self.disposed = False
        self._fail = fail
        self._handlers: list[Any] = []

    def subscribe(self, handler):
        self._handlers.append(handler)

        def _unsubscribe():
            if handler in self._handlers:
                self._handlers.remove(handler)

        return _unsubscribe

    async def prompt(self, text: str, attachments=None) -> None:
        self.prompts.append(text)
        if self._fail is not None:
            raise self._fail
        end = SimpleNamespace(
            type="agent_end",
            error=None,
            aborted=False,
            messages=[
                SimpleNamespace(role="user", text=text),
                SimpleNamespace(role="assistant", text=f"scheduled response #{len(self.prompts)}"),
            ],
        )
        for handler in list(self._handlers):
            outcome = handler(end)
            if hasattr(outcome, "__await__"):
                await outcome

    async def dispose(self) -> None:
        self.disposed = True


class FakeSessionFactory:
    """Async stand-in for ``session_factory.create_session``."""

    def __init__(self, fail: BaseException | None = None):
        self.fail = fail
        self.calls: list[argparse.Namespace] = []
        self.sessions: list[FakeSession] = []

    async def __call__(
        self,
        args,
        config_manager,
        credential_manager,
        agent_registry,
        *,
        has_ui=False,
        cwd=None,
    ):
        session = FakeSession(fail=self.fail)
        self.calls.append(args)
        self.sessions.append(session)
        session.cwd = cwd
        return session


class SlowSessionFactory(FakeSessionFactory):
    """Factory whose session's prompt hangs forever (for the timeout test)."""

    async def __call__(
        self, args, config_manager, credential_manager, agent_registry, *, has_ui=False, cwd=None
    ):
        session = FakeSession()

        async def _hanging_prompt(text: str, attachments=None) -> None:
            session.prompts.append(text)
            await asyncio.sleep(3600)

        session.prompt = _hanging_prompt  # type: ignore[method-assign]
        self.calls.append(args)
        self.sessions.append(session)
        return session


class RecordingWebSocketManager:
    """Captures broadcast calls (websocket-manager shaped)."""

    def __init__(self):
        self.broadcasts: list[tuple[str, dict[str, Any]]] = []

    async def broadcast(self, message_id: str, data: dict[str, Any], connection_type=None):
        self.broadcasts.append((message_id, dict(data)))


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    return AgentRegistry(config_dir=tmp_path / "config")


def _make_agent(registry: AgentRegistry, name: str = "sched-agent") -> Any:
    return registry.create_agent(
        AgentEditFields(
            name=name,
            hosting="openai",
            model="gpt-4o",
            current_working_directory=".",
        )
    )


def _add_schedule(registry: AgentRegistry, agent_id: str, **overrides: Any) -> Schedule:
    state = registry.load_agent_state(agent_id)
    schedule = Schedule(
        agent_id=uuid.UUID(agent_id),
        prompt=overrides.pop("prompt", "run the scheduled thing"),
        interval=overrides.pop("interval", 5),
        unit=overrides.pop("unit", ScheduleUnit.MINUTES),
        **overrides,
    )
    state.schedules.append(schedule)
    registry.save_agent_state(agent_id, state)
    return schedule


def _make_service(
    registry: AgentRegistry,
    job_manager: JobManager | None = None,
    websocket_manager: Any = None,
) -> SchedulerService:
    return SchedulerService(
        agent_registry=registry,
        config_manager=ConfigManager(registry.config_dir),
        credential_manager=CredentialManager(registry.config_dir),
        env_config=EnvConfig(),
        operator_type="cli",
        verbosity_level=VerbosityLevel.QUIET,
        job_manager=job_manager or JobManager(),
        websocket_manager=websocket_manager if websocket_manager is not None else object(),
    )


async def _wait_for(predicate, timeout: float = 15.0, interval: float = 0.05) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("timed out waiting for condition")


# ---------------------------------------------------------------------------
# Import-graph contract
# ---------------------------------------------------------------------------


def test_module_import_graph_has_no_langchain():
    import sys

    import local_operator.scheduler_service  # noqa: F401

    leaks = [
        m
        for m in sys.modules
        if m.split(".")[0] in ("langchain", "langchain_community", "langchain_core")
    ]
    assert not leaks, f"scheduler_service module graph must not import langchain: {leaks}"


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_time_fires_once_and_pops(registry, monkeypatch):
    factory = FakeSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    job_manager = JobManager()
    service = _make_service(registry, job_manager, RecordingWebSocketManager())
    agent = _make_agent(registry)
    schedule = _add_schedule(
        registry,
        agent.id,
        one_time=True,
        interval=0,
        start_time_utc=datetime.now(timezone.utc) + timedelta(seconds=0.5),
    )

    await service.start()
    try:
        job_id = str(schedule.id)
        await _wait_for(
            lambda: job_id in job_manager.jobs
            and job_manager.jobs[job_id].status == JobStatus.COMPLETED,
            timeout=20.0,
        )

        # One run only, via the train/yolo session contract
        assert len(factory.sessions) == 1
        args = factory.calls[0]
        assert args.agent_id == agent.id
        assert args.train is True
        assert args.yolo is True
        assert schedule.prompt in factory.sessions[0].prompts[0]
        assert factory.sessions[0].disposed is True

        # Schedule popped from the agent's persisted state
        state = registry.load_agent_state(agent.id)
        assert all(s.id != schedule.id for s in state.schedules)

        # Ledger + broadcast captured the outcome
        job = job_manager.jobs[job_id]
        assert job.result is not None
        assert job.result.response and "scheduled response" in job.result.response
        broadcasts = service.websocket_manager.broadcasts
        assert any(b[0] == job_id and b[1].get("status") == "completed" for b in broadcasts)

        # Extra scheduler ticks: a one-time schedule never fires twice
        await asyncio.sleep(2.0)
        assert len(factory.sessions) == 1
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_interval_reschedules_and_stamps_last_run(registry, monkeypatch):
    factory = FakeSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    job_manager = JobManager()
    service = _make_service(registry, job_manager)
    agent = _make_agent(registry)
    schedule = _add_schedule(
        registry,
        agent.id,
        one_time=False,
        interval=15,
        unit=ScheduleUnit.MINUTES,
        start_time_utc=datetime.now(timezone.utc) + timedelta(minutes=10),
    )

    await service.start()
    try:
        job_id = str(schedule.id)
        armed = service.scheduler.get_job(job_id)
        assert armed is not None, "recurring schedule must be armed in APScheduler"
        assert armed.next_run_time is not None

        # Simulate two consecutive fires: the job reschedules each time.
        for expected_runs in (1, 2):
            await service._trigger_agent_task(agent.id, job_id, schedule.prompt)
            assert len(factory.sessions) == expected_runs
            state = registry.load_agent_state(agent.id)
            item = next((s for s in state.schedules if s.id == schedule.id), None)
            assert item is not None, "recurring schedule must remain after a run"
            assert item.is_active
            assert item.last_run_at is not None
            assert item.last_run_at.tzinfo is not None

        assert service.scheduler.get_job(job_id) is not None, "recurring job must reschedule"
        assert job_manager.jobs[job_id].status == JobStatus.COMPLETED
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_end_time_deactivates_on_load(registry, monkeypatch):
    factory = FakeSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    service = _make_service(registry)
    agent = _make_agent(registry)
    now = datetime.now(timezone.utc)
    schedule = _add_schedule(
        registry,
        agent.id,
        one_time=False,
        interval=1,
        unit=ScheduleUnit.HOURS,
        start_time_utc=now - timedelta(days=2),
        end_time_utc=now - timedelta(hours=1),
    )

    await service.start()
    try:
        # Ended schedules are not armed and are purged from persistence on load
        assert service.scheduler.get_job(str(schedule.id)) is None
        state = registry.load_agent_state(agent.id)
        assert all(s.id != schedule.id for s in state.schedules)
        assert not factory.sessions
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_end_time_deactivates_at_trigger(registry, monkeypatch):
    factory = FakeSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    job_manager = JobManager()
    service = _make_service(registry, job_manager)
    agent = _make_agent(registry)
    now = datetime.now(timezone.utc)

    await service.start()
    try:
        # Schedule added after load, already past its end time
        schedule = _add_schedule(
            registry,
            agent.id,
            one_time=False,
            interval=1,
            unit=ScheduleUnit.MINUTES,
            start_time_utc=now - timedelta(hours=2),
            end_time_utc=now - timedelta(minutes=5),
        )
        await service._trigger_agent_task(agent.id, str(schedule.id), schedule.prompt)

        state = registry.load_agent_state(agent.id)
        item = next(s for s in state.schedules if s.id == schedule.id)
        assert not item.is_active, "end-time trigger must deactivate the schedule"
        assert not factory.sessions, "an ended schedule must not run"
        assert str(schedule.id) not in job_manager.jobs, "no ledger entry for an ended schedule"
        assert service.scheduler.get_job(str(schedule.id)) is None
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_past_due_one_time_replays_on_load(registry, monkeypatch):
    factory = FakeSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    job_manager = JobManager()
    service = _make_service(registry, job_manager)
    agent = _make_agent(registry)
    schedule = _add_schedule(
        registry,
        agent.id,
        one_time=True,
        interval=0,
        start_time_utc=datetime.now(timezone.utc) - timedelta(minutes=30),
    )

    await service.start()
    try:
        job_id = str(schedule.id)
        await _wait_for(
            lambda: job_id in job_manager.jobs
            and job_manager.jobs[job_id].status == JobStatus.COMPLETED,
            timeout=15.0,
        )
        assert len(factory.sessions) == 1
        state = registry.load_agent_state(agent.id)
        assert all(s.id != schedule.id for s in state.schedules)
        assert service.scheduler.get_job(job_id) is None
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_failed_run_records_failed_and_loop_survives(registry, monkeypatch):
    factory = FakeSessionFactory(fail=RuntimeError("provider exploded"))
    monkeypatch.setattr(session_factory, "create_session", factory)
    job_manager = JobManager()
    service = _make_service(registry, job_manager)
    agent = _make_agent(registry)
    schedule = _add_schedule(
        registry, agent.id, one_time=False, interval=5, unit=ScheduleUnit.MINUTES
    )

    await service.start()
    try:
        job_id = str(schedule.id)
        await service._trigger_agent_task(agent.id, job_id, schedule.prompt)

        job = job_manager.jobs[job_id]
        assert job.status == JobStatus.FAILED
        assert job.result is not None
        assert "provider exploded" in (job.result.error or "")

        # Legacy semantics: no last_run_at stamp on failure (retries keep firing)
        state = registry.load_agent_state(agent.id)
        item = next(s for s in state.schedules if s.id == schedule.id)
        assert item.last_run_at is None
        assert item.is_active

        # The scheduler loop is alive: a subsequent run still executes
        factory.fail = None
        await service._trigger_agent_task(agent.id, job_id, schedule.prompt)
        assert len(factory.sessions) == 2
        assert job_manager.jobs[job_id].status == JobStatus.COMPLETED
        assert service.scheduler.running
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_run_timeout_records_failed_and_disposes(registry, monkeypatch):
    factory = SlowSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    monkeypatch.setenv(scheduler_module.RUN_TIMEOUT_ENV_VAR, "0.3")
    job_manager = JobManager()
    service = _make_service(registry, job_manager)
    agent = _make_agent(registry)
    schedule = _add_schedule(
        registry, agent.id, one_time=False, interval=5, unit=ScheduleUnit.MINUTES
    )

    await service.start()
    try:
        job_id = str(schedule.id)
        await service._trigger_agent_task(agent.id, job_id, schedule.prompt)
        job = job_manager.jobs[job_id]
        assert job.status == JobStatus.FAILED
        assert job.result is not None
        assert "timed out" in (job.result.error or "")
        assert factory.sessions[0].disposed is True, "timed-out session must be disposed"
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_websocket_shape_mismatch_degrades_gracefully(registry, monkeypatch):
    factory = FakeSessionFactory()
    monkeypatch.setattr(session_factory, "create_session", factory)
    job_manager = JobManager()
    service = _make_service(registry, job_manager, websocket_manager=object())
    agent = _make_agent(registry)
    schedule = _add_schedule(
        registry, agent.id, one_time=False, interval=5, unit=ScheduleUnit.MINUTES
    )

    await service.start()
    try:
        await service._trigger_agent_task(agent.id, str(schedule.id), schedule.prompt)
        assert job_manager.jobs[str(schedule.id)].status == JobStatus.COMPLETED
    finally:
        await service.shutdown()


def test_schedules_jsonl_round_trip(registry):
    agent = _make_agent(registry)
    now = datetime.now(timezone.utc)
    start = now + timedelta(minutes=10)
    end = start + timedelta(days=1)
    schedule = Schedule(
        agent_id=uuid.UUID(agent.id),
        prompt="round trip",
        interval=30,
        unit=ScheduleUnit.MINUTES,
        start_time_utc=start.replace(tzinfo=None),  # naive input must coerce to UTC
        end_time_utc=end,
        last_run_at=now.replace(tzinfo=None),
        one_time=False,
    )
    state = registry.load_agent_state(agent.id)
    state.schedules.append(schedule)
    registry.save_agent_state(agent.id, state)

    # On disk: one JSON line per schedule in the agent's schedules.jsonl
    schedules_file = registry.config_dir / "agents" / agent.id / "schedules.jsonl"
    lines = schedules_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["prompt"] == "round trip"
    assert payload["interval"] == 30
    assert payload["unit"] == "minutes"

    # Reload through a fresh registry: fields and UTC coercion preserved
    fresh = AgentRegistry(config_dir=registry.config_dir)
    state2 = fresh.load_agent_state(agent.id)
    assert len(state2.schedules) == 1
    loaded = state2.schedules[0]
    assert loaded.id == schedule.id
    assert loaded.prompt == "round trip"
    assert loaded.interval == 30
    assert loaded.unit == ScheduleUnit.MINUTES
    assert loaded.start_time_utc is not None and loaded.start_time_utc.tzinfo == timezone.utc
    assert loaded.last_run_at is not None and loaded.last_run_at.tzinfo == timezone.utc
    assert loaded.start_time_utc.replace(tzinfo=None) == start.replace(tzinfo=None)
