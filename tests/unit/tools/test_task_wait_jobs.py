"""Tests for the background engine tools: task / wait / jobs.

These are the createIf-gated siblings of ``wake``: they only exist when the
ToolContext carries a ``subagent_launcher`` (task) or a job manager
(wait/jobs). The tests drive the real unit wraps against a real
``harness.jobs.AsyncJobManager``, not a mock, so the tool results reflect how
jobs actually settle.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools


async def wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


async def _quick_runner(job_id: str, signal: Any, report_progress) -> str:
    report_progress("halfway")
    return f"done:{job_id}"


async def _slow_runner(job_id: str, signal: Any, report_progress) -> str:
    await asyncio.sleep(30)  # outlive any test window
    return "never"


def _engine_context(tmp_path, manager: AsyncJobManager) -> ToolContext:
    """A context carrying the job manager AND a launcher so both task and
    wait/jobs build. The launcher mimics ``Session._launch_subagent`` by
    registering a ``task``-type job on the same manager."""
    launcher = lambda label, prompt: manager.register(  # noqa: E731
        "task", label, _quick_runner, owner_id=None
    )
    return ToolContext(cwd=str(tmp_path), session_id="s", subagent_launcher=launcher, jobs=manager)


def _tools(context: ToolContext) -> dict[str, AgentTool]:
    return {t.name: t for t in create_tools(context)}


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    tool = tools[name]
    return await tool.execute("call-1", args, None, None, context)  # type: ignore[operator]


@pytest.mark.asyncio
async def test_task_registers_a_job_and_returns_its_id(tmp_path):
    manager = AsyncJobManager()
    context = _engine_context(tmp_path, manager)
    tools = _tools(context)

    before = [j.id for j in manager.list()]
    result = await _call(tools, "task", {"label": "summarize", "prompt": "do it"}, context)

    assert result.is_error is False
    assert result.details is not None
    job_id = result.details["job_id"]
    assert job_id not in before  # a NEW job was registered
    assert f"job {job_id}" in result.text

    job = manager.get(job_id)
    assert job is not None
    assert job.type == "task"
    assert job.label == "summarize"

    def _completed():
        job = manager.get(job_id)
        return job is not None and job.status == "completed"

    # The quick runner completes on its own; wait for settlement.
    await wait_for(_completed)
    job = manager.get(job_id)
    assert job is not None
    assert job.result_text == f"done:{job_id}"
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_returns_final_output_when_job_completes(tmp_path):
    manager = AsyncJobManager()
    context = _engine_context(tmp_path, manager)
    tools = _tools(context)

    result = await _call(tools, "task", {"label": "work", "prompt": "run"}, context)
    assert result.is_error is False
    assert result.details is not None
    job_id = result.details["job_id"]

    waited = await _call(tools, "wait", {"job_id": job_id, "wait_ms": 1000}, context)
    assert waited.is_error is False
    assert waited.details is not None
    assert waited.details["status"] == "completed"
    assert f"done:{job_id}" in waited.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_spills_large_subagent_handoff_without_losing_it(tmp_path, monkeypatch):
    """One verbose child must not consume an unbounded slice of parent context."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    report = "\n".join(f"review finding {index}: {'x' * 120}" for index in range(400))

    async def verbose_runner(job_id: str, signal: Any, report_progress) -> str:
        return report

    manager = AsyncJobManager()
    job_id = manager.register("task", "verbose review", verbose_runner)
    context = ToolContext(cwd=str(tmp_path), session_id="parent", jobs=manager)
    tools = _tools(context)

    waited = await _call(tools, "wait", {"job_id": job_id, "wait_ms": 1000}, context)

    assert waited.is_error is False
    assert waited.details is not None
    spill = waited.details["spill"]
    assert spill["complete"] is True
    assert spill["handle"] in waited.text
    assert len(waited.text) < builtin.TOOL_OUTPUT_LIMIT_CHARS + 1_000
    stored = builtin.get_store().read_lines(spill["handle"])
    assert stored is not None
    lines, _total = stored
    recovered = "\n".join(lines)
    assert "review finding 0:" in recovered
    assert "review finding 399:" in recovered
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_bounded_by_wait_ms_times_out(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager, subagent_launcher=None)
    # Register a job that never settles so wait must time out.
    slow_id = manager.register("task", "slow", _slow_runner)
    tools = _tools(context)

    started = asyncio.get_running_loop().time()
    result = await _call(tools, "wait", {"job_id": slow_id, "wait_ms": 60}, context)

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 1.0  # returned promptly on the bound, did not hang
    assert result.is_error is False
    assert result.details is not None
    assert result.details["status"] == "running"
    assert "still running" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_lists_ids_labels_and_statuses(tmp_path):
    manager = AsyncJobManager()
    quick_id = manager.register("task", "alpha", _quick_runner)
    running_id = manager.register("task", "beta", _slow_runner)
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)

    result = await _call(tools, "jobs", {}, context)
    assert result.is_error is False
    assert result.details is not None
    assert result.details["count"] == 2
    assert "alpha" in result.text
    assert "beta" in result.text
    assert running_id in result.text
    assert "running" in result.text

    def _settled():
        job = manager.get(quick_id)
        return job is not None and job.status != "running"

    # Let the quick job settle before dispose so its runner coroutine is
    # awaited rather than abandoned (the slow one is cancelled by dispose).
    await wait_for(_settled)
    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_empty_manager(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    result = await _call(tools, "jobs", {}, context)
    assert result.is_error is False
    assert result.details is not None
    assert result.details["count"] == 0
    assert "no background jobs" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_unknown_job_is_an_error(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    result = await _call(tools, "wait", {"job_id": "nope", "wait_ms": 10}, context)
    assert result.is_error is True
    assert "unknown job nope" in result.text
    await manager.dispose()


def test_task_not_advertised_without_launcher(tmp_path):
    """createIf: no subagent_launcher -> no task tool, and task execute errors."""
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=AsyncJobManager())
    names = {t.name for t in create_tools(context)}
    assert "task" not in names
    assert builtin.build_task_tool(context) is None


def test_wait_jobs_not_advertised_without_job_manager(tmp_path):
    context = ToolContext(cwd=str(tmp_path), session_id="s")
    names = {t.name for t in create_tools(context)}
    assert "wait" not in names
    assert "jobs" not in names
    assert builtin.build_wait_tool(context) is None
    assert builtin.build_jobs_tool(context) is None
