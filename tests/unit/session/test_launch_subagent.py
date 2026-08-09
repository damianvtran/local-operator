"""Session._launch_subagent: the production caller of the subagent engine.

``_launch_subagent`` registers one one-shot child run on the parent session's
job manager and emits ``subagent_start`` / ``subagent_end`` on the parent
event stream. These tests drive it through a real ``run_subagent`` against a
ScriptedStream, so the child actually runs its own loop and the parent sees
the full lifecycle.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentEvent,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    SubagentEndEvent,
    SubagentStartEvent,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


async def wait_for(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


class OneShotStream:
    """Serves exactly one text-only turn (the child's run)."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="child did the work")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, **kwargs) -> Session:
    transcript = Transcript(tmp_path / "sess")
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=transcript,
        system_blocks_provider=kwargs.pop("system_blocks_provider", lambda: ["stable", "env"]),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_launch_subagent_runs_child_and_emits_lifecycle(tmp_path, monkeypatch):
    """_launch_subagent registers a task job, the child runs via the parent's
    stream_fn, and subagent_start/end land on the parent stream."""
    # The child writes its transcript under config_dir(); keep it hermetic.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    stream = OneShotStream()
    parent = make_session(tmp_path, stream)

    events: list[AgentEvent] = []
    parent.subscribe(events.append)

    job_id = parent._launch_subagent(label="sub", prompt="go do a thing")
    assert isinstance(job_id, str) and job_id  # a non-empty job id

    # The child is registered on the parent's job manager.
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.type == "task"
    assert job.label == "sub"

    # Wait for the child run to settle and the parent stream to see the end.
    await wait_for(lambda: any(e.type == "subagent_end" for e in events))

    starts = [e for e in events if isinstance(e, SubagentStartEvent)]
    ends = [e for e in events if isinstance(e, SubagentEndEvent)]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].job_id == job_id
    assert starts[0].label == "sub"
    assert ends[0].job_id == job_id
    assert ends[0].status == "completed"
    assert "child did the work" in (ends[0].result_text or "")

    # The child actually ran its own provider turn through the shared stream.
    assert len(stream.requests) == 1
    assert stream.requests[0].messages
    assert isinstance(stream.requests[0].messages[0], Message)
    assert stream.requests[0].messages[0].text == "go do a thing"

    await parent.dispose()


@pytest.mark.asyncio
async def test_launch_subagent_is_wired_as_subagent_launcher(tmp_path, monkeypatch):
    """The ToolContext built for a turn carries _launch_subagent as the
    subagent_launcher, so the task tool can call it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = OneShotStream()
    parent = make_session(tmp_path, stream)
    ctx = parent._build_tool_context()
    assert ctx.subagent_launcher is not None
    # The launcher registers on the SAME manager the task/wait/jobs tools see.
    assert ctx.jobs is parent.jobs
    await parent.dispose()


@pytest.mark.asyncio
async def test_launch_subagent_cancels_on_parent_dispose(tmp_path, monkeypatch):
    """A still-running child is cancelled when the parent session disposes,
    because it lives on the parent's job manager."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    class _HangStream:
        """Turns never finish, so the child keeps running until dispose."""

        def __call__(self, request, signal):
            async def gen():
                await asyncio.sleep(30)
                yield StreamEndEvent(stop_reason="stop")

            return gen()

    parent = make_session(tmp_path, _HangStream())
    job_id = parent._launch_subagent(label="slow", prompt="never finish")

    # Give the child a moment to register AND start its turn, so the runner's
    # inner coroutine is genuinely awaited before disposal (cancelling a task
    # that never reached `await coro` leaks an un-awaited coroutine warning).
    def _running():
        job = parent.jobs.get(job_id)
        return job is not None and job.status == "running"

    await wait_for(_running)
    await asyncio.sleep(0.05)
    assert (job := parent.jobs.get(job_id)) is not None
    assert job.status == "running"
    await parent.dispose()
    assert (job := parent.jobs.get(job_id)) is not None
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_child_inherits_parent_compaction_settings(tmp_path, monkeypatch):
    """A long-running review child must not bypass the operator's compaction
    budget. Live finding: a one-shot child ran 48 requests / 1.5M tokens on
    the default 600k-cap threshold while the parent's cap was 250k. The child
    Session must receive the parent's compaction settings."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness import subagent as subagent_mod

    capped = CompactionSettings(max_threshold_tokens=250_000)
    stream = OneShotStream()
    parent = make_session(tmp_path, stream, compaction_settings=capped)

    # Capture the child session constructed by the runner.
    built_children: list[Session] = []
    orig_build = subagent_mod._build_child_session

    async def captured_build(*a, **kw):
        child = await orig_build(*a, **kw)
        built_children.append(child)
        return child

    subagent_mod._build_child_session = captured_build

    job_id = parent._launch_subagent(label="sub", prompt="review")
    await wait_for(lambda: parent.jobs.get(job_id) is not None)
    await wait_for(
        lambda: (job := parent.jobs.get(job_id)) is not None and job.status == "completed"
    )

    assert built_children, "the runner must construct a child session"
    child = built_children[0]
    assert child._compaction_settings is not None
    assert child._compaction_settings.max_threshold_tokens == 250_000
    assert child._compaction_settings == capped
    await parent.dispose()
