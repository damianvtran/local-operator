"""``subagents.model_choice=model`` hands the tier picker to the delegating model at EVERY depth.

Why this file exists: PR #2052 closed the picker below the top level after a
costly fan-out (session 3463fc25dade: a depth-1 subagent launched five scouts
with ``effort='hi'`` and they ran on a paid tier model while the session was on
Radient auto). The operator's decision was the opposite: keep tier choice at
every depth and treat the real defect as the cost DISPLAY (the panel rows show
only the depth-1 child's own usage). The gate was reverted; these cases pin the
restored behaviour so a later "fix" of the cost surprise does not quietly close
the picker again.

============================  ================  ==========================
launcher                      ``effort`` given  result
============================  ================  ==========================
depth 0, model_choice=model   ``hi``            tier model
depth >= 1, model_choice=model ``hi``           tier model, field offered
depth >= 1                    none, unpinned    inherits the launcher's model
any depth, model_choice=operator ``hi``         refused (the key still rules)
============================  ================  ==========================

Every case builds a REAL child through ``_build_child_session`` and launches
through the real ``Session._launch_subagent`` / the tool's own executor; nothing
stubs the resolver.
"""

from __future__ import annotations

import pytest
import yaml

from local_operator.harness import subagent as subagent_mod
from local_operator.harness.subagent import model_may_choose_tier, resolve_launch_target
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    TextContent,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

ROOT_MODEL = ModelSpec(provider="radient", model_id="auto", context_window=100_000)
HI = "anthropic/claude-sonnet-5-5"
ROOT_LABEL = "radient/auto"


class OneShotStream:
    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            yield StreamTextDelta(delta="done")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def write_config(config_dir, *, choice: str = "model") -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "values": {
                    "subagents": {
                        "model_choice": choice,
                        "models": {"hi": HI, "lo": "deepseek/deepseek-flash"},
                    }
                }
            }
        )
    )


def make_root(tmp_path) -> Session:
    return Session(
        model=ROOT_MODEL,
        stream_fn=OneShotStream(),
        tools=[],
        transcript=Transcript(tmp_path / "root"),
        system_blocks_provider=lambda: ["stable", "env"],
    )


async def make_child(parent: Session, job_id: str):
    """A child built the way ``run_subagent`` builds it, including the depth stamp."""
    target = resolve_launch_target("task", parent)
    child = await subagent_mod._build_child_session(
        label=job_id,
        prompt="go",
        parent_session=parent,
        model_spec=None,
        job_id=job_id,
        agent="task",
        target=target,
    )
    assert child._delegation_depth == target.depth
    return child


def task_tool(session: Session):
    return next(tool for tool in session._tools if tool.name == "task")


def effort_in_schema(session: Session) -> bool:
    params = task_tool(session).parameters
    return "effort" in params["properties"] or any(
        "effort" in d.get("properties", {}) for d in params.get("$defs", {}).values()
    )


def text_of(result) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextContent))


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    write_config(tmp_path / "config")
    return tmp_path / "config"


def test_the_policy_takes_no_depth(config):
    assert model_may_choose_tier() is True


@pytest.mark.asyncio
async def test_depth_zero_picks_a_tier(tmp_path, config):
    root = make_root(tmp_path)
    assert effort_in_schema(root)
    job = root.jobs.get(root._launch_subagent(label="s", prompt="p", agent="scout", effort="hi"))
    assert job is not None and job.model_label == HI and job.owns_model is True
    await root.dispose()


@pytest.mark.asyncio
async def test_a_subagent_is_offered_effort_and_its_tier_choice_is_honoured(tmp_path, config):
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    assert effort_in_schema(child), "the picker must be offered at depth >= 1 too"

    # Through the child's REAL executor, so the argument gate runs as it does live.
    result = await task_tool(child).execute(
        "c1",
        {"label": "scout", "prompt": "p", "agent": "scout", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert not result.is_error, text_of(result)
    launched = child.jobs.list()
    assert len(launched) == 1
    assert launched[0].model_label == HI and launched[0].owns_model is True
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_depth_two_scout_without_effort_still_inherits(tmp_path, config):
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    job = child.jobs.get(child._launch_subagent(label="s", prompt="p", agent="scout"))
    assert job is not None
    assert job.owns_model is False
    assert job.model_label == ROOT_LABEL
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_subagent_may_write_a_tier_pin_into_a_role(tmp_path, config):
    from local_operator.agents import AgentRegistry
    from local_operator.tools.agent_tool import build_agent_tool

    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    child.agent_registry = AgentRegistry(tmp_path / "agents")
    tool = build_agent_tool(child._build_tool_context())
    assert tool is not None
    members = [v["enum"] for v in tool.parameters["properties"]["effort"]["anyOf"] if "enum" in v][
        0
    ]
    assert "hi" in members and "inherit" in members

    result = await tool.execute(
        "c1",
        {"op": "create", "name": "rev", "description": "d", "instructions": "i", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert not result.is_error, text_of(result)
    spec = child._resolve_subagent_model("rev", None, strict=True)
    assert spec is not None and f"{spec.provider}/{spec.model_id}" == HI
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_the_operator_default_still_refuses_a_tier_at_depth(tmp_path, config):
    write_config(config, choice="operator")
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    assert not effort_in_schema(child)
    result = await task_tool(child).execute(
        "c1",
        {"label": "s", "prompt": "p", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert result.is_error and "subagents.model_choice" in text_of(result)
    assert child.jobs.list() == []
    await child.dispose()
    await root.dispose()
