"""Launching a subagent in a ROLE: guidance and tool surface reach the child.

The contract under test is the boundary, not the wording: a role's guidance
must arrive ahead of the task on the child's first turn, and a role's tool
allowlist must be a capability the child physically lacks rather than advice it
is asked to respect.
"""

from __future__ import annotations

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from tests.unit.session.test_launch_subagent import MODEL, wait_for


class RecordingStream:
    """Serves one text-only turn and keeps the requests it was handed."""

    def __init__(self) -> None:
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)

        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_parent(tmp_path, stream, **kwargs) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        **kwargs,
    )


async def run_role(parent: Session, role: str, prompt: str = "Do the thing.") -> None:
    job_id = parent._launch_subagent(label=role, prompt=prompt, agent=role)
    await wait_for(lambda: (job := parent.jobs.get(job_id)) is not None and job.status != "running")


@pytest.mark.asyncio
async def test_a_roles_guidance_arrives_ahead_of_the_task(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = make_parent(tmp_path, stream)

    await run_role(parent, "reviewer", "Review PR 42.")

    assert stream.requests
    first_user = next(m for m in stream.requests[0].messages if m.role == "user")
    assert first_user.text.startswith("[role: reviewer]")
    assert "Review PR 42." in first_user.text
    assert first_user.text.index("Review PR 42.") > first_user.text.index("[role: reviewer]")
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_reviewer_child_cannot_edit_but_can_run_commands(tmp_path, monkeypatch) -> None:
    """The capability boundary: a reviewer that could edit would end up
    reviewing a diff it had itself changed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = make_parent(tmp_path, stream)

    await run_role(parent, "reviewer")

    names = {tool.name for tool in stream.requests[0].tools}
    assert "edit" not in names and "write" not in names
    assert "bash" in names and "read" in names
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_non_delegating_role_gets_no_task_tool(tmp_path, monkeypatch) -> None:
    """A reviewer spawning its own children turns one review into a fan-out
    nobody is watching."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = make_parent(tmp_path, stream)

    await run_role(parent, "reviewer")

    names = {tool.name for tool in stream.requests[0].tools}
    assert not names & {"task", "wait", "jobs", "wake"}
    await parent.dispose()


@pytest.mark.asyncio
async def test_an_unknown_role_still_launches_a_full_child(tmp_path, monkeypatch) -> None:
    """A typo in a role name must not lose work the parent already decided to
    delegate."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = make_parent(tmp_path, stream)

    await run_role(parent, "no-such-role", "Still do it.")

    first_user = next(m for m in stream.requests[0].messages if m.role == "user")
    assert first_user.text == "Still do it.", "no role framing should be stamped"
    names = {tool.name for tool in stream.requests[0].tools}
    assert "edit" in names, "an unknown role must not silently restrict the child"
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_operators_own_role_overrides_the_packaged_one(tmp_path, monkeypatch) -> None:
    """Editing the guidance has to actually change what the child is told —
    otherwise the registry is decoration."""
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    registry = AgentRegistry(config)
    agent = registry.create_agent(
        AgentEditFields(
            name="reviewer",
            description="house reviewer",
            tags=["role", "tools:read"],
            categories=["role"],
            security_prompt=None,
            hosting=None,
            model=None,
            last_message=None,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            stop=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            current_working_directory=None,
        )
    )
    registry.set_agent_system_prompt(agent.id, "ONLY CHECK THE MIGRATIONS.")

    stream = RecordingStream()
    parent = make_parent(tmp_path, stream, agent_registry=registry)

    await run_role(parent, "reviewer")

    first_user = next(m for m in stream.requests[0].messages if m.role == "user")
    assert "ONLY CHECK THE MIGRATIONS." in first_user.text
    names = {tool.name for tool in stream.requests[0].tools}
    # The operator's allowlist wins; ``hub`` rides along as the one deliberate
    # exception so a restricted child can still answer its parent's questions.
    assert names == {"read", "hub"}, f"the operator's allowlist should win, got {names}"
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_plain_task_child_is_unchanged(tmp_path, monkeypatch) -> None:
    """The default launch must pay nothing for the role machinery."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = make_parent(tmp_path, stream)

    job_id = parent._launch_subagent(label="plain", prompt="Just do it.")
    await wait_for(lambda: (job := parent.jobs.get(job_id)) is not None and job.status != "running")

    first_user = next(m for m in stream.requests[0].messages if m.role == "user")
    assert first_user.text == "Just do it."
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_single_task_form_accepts_a_role(tmp_path, monkeypatch) -> None:
    """Found live: a model asked for one reviewer the obvious way and the call
    was rejected, costing a round trip to rediscover that a role was only
    reachable through the batch form."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.tools.builtin import TaskParams

    params = TaskParams(label="review", prompt="Review it.", agent="reviewer")
    assert params.agent == "reviewer"


def test_the_batch_form_refuses_a_top_level_role() -> None:
    """Silently ignoring it would leave every child in the batch unroled while
    the caller believed it had asked for one."""
    import pytest as _pytest

    from local_operator.tools.builtin import TaskItem, TaskParams

    with _pytest.raises(ValueError, match="each tasks"):
        TaskParams(tasks=[TaskItem(label="a", prompt="b")], agent="reviewer")
