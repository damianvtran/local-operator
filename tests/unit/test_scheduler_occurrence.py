"""Scheduler occurrence inheritance: durable markers and exact-once suppression.

The durable record itself is covered in test_update_operation; this drives the
SchedulerService's use of it through injected callables, so no install root and no
live scheduler are touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel
from local_operator.env import EnvConfig
from local_operator.jobs import JobManager, JobStatus
from local_operator.scheduler_service import SchedulerService
from local_operator.types import OperatorType


@pytest.fixture
def registry(tmp_path: Path) -> AgentRegistry:
    return AgentRegistry(config_dir=tmp_path / "config")


def _fenced_service(registry: Any, **kwargs: Any) -> SchedulerService:
    return SchedulerService(
        agent_registry=registry,
        config_manager=ConfigManager(registry.config_dir),
        env_config=EnvConfig(),
        operator_type=OperatorType.CLI,
        verbosity_level=VerbosityLevel.QUIET,
        job_manager=JobManager(),
        event_broker=None,
        **kwargs,
    )


def test_feature_is_inert_without_a_recorder_or_suppressor(registry) -> None:
    """No injected callables ⇒ nothing is marked and nothing is suppressed."""

    service = _fenced_service(registry)
    service._mark_occurrence("s1", "running")  # must not raise
    assert service._suppressed_now() == set()


def test_suppression_is_reread_each_call_not_cached(registry) -> None:
    seen: list[set[str]] = []

    def suppressor() -> set[str]:
        return {"s1"} if len(seen) < 1 else set()

    service = _fenced_service(registry, occurrence_suppressor=suppressor)
    seen.append(service._suppressed_now())
    assert service._suppressed_now() == set()
    seen.append(service._suppressed_now())


def test_unreadable_suppressor_never_suppresses(registry) -> None:
    """An unreadable record must let the run fire, never stall it."""

    def boom() -> set[str]:
        raise OSError("record unreadable")

    service = _fenced_service(registry, occurrence_suppressor=boom)
    assert service._suppressed_now() == set()


def test_marker_failure_never_kills_a_run(registry) -> None:
    def boom(_id: str, _state: str, _terminal: str) -> None:
        raise OSError("record unwritable")

    service = _fenced_service(registry, occurrence_recorder=boom)
    service._mark_occurrence("s1", "running")  # must swallow, not raise


@pytest.mark.asyncio
async def test_status_push_writes_running_then_terminal_markers(registry) -> None:
    recorded: list[tuple[str, str, str]] = []
    service = _fenced_service(
        registry, occurrence_recorder=lambda i, s, t: recorded.append((i, s, t))
    )
    job = await service.job_manager.create_job(prompt="p", model="m", hosting="openai")
    await service._push_job_status(job.id, JobStatus.PROCESSING)
    await service._push_job_status(job.id, JobStatus.COMPLETED)
    assert recorded[0] == (job.id, "running", "")
    assert recorded[1] == (job.id, "terminal", "done")
