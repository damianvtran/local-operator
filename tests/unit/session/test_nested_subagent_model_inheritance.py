"""A subagent's own subagents inherit its model; only the top level picks a tier.

The incident (session 3463fc25dade): ``subagents.model_choice=model`` with
``models.hi`` on Claude Sonnet and the session on Radient auto. A depth-1
subagent launched five ``scout`` children with ``effort='hi'``; they ran on
Sonnet (``owns_model=True``) instead of inheriting — about $5 of Sonnet spend
against $0.76 on auto, invisible in the panel rows. Depth-1 children that passed
no ``effort`` inherited correctly, so inheritance itself was never broken: the
leak was that a CHILD model's explicit tier was honoured below the top level.

The policy these pin (``harness.subagent.model_may_choose_tier``):

============================  ================  ==========================
launcher                      ``effort`` given  result
============================  ================  ==========================
depth 0, model_choice=model   ``hi``            tier model (unchanged)
depth >= 1, any model_choice  ``hi``            refused; field not offered
depth >= 1                    none, unpinned    inherits the CHILD's model
depth >= 1                    none, role pinned operator's pin (unchanged)
============================  ================  ==========================

Every case builds a REAL child through ``_build_child_session`` and launches
through the real ``Session._launch_subagent``; nothing here stubs the resolver.
"""

from __future__ import annotations

import asyncio

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
from local_operator.tools.builtin import execute_task

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


async def make_child(parent: Session, job_id: str, *, model_spec: ModelSpec | None = None):
    """A child built the way ``run_subagent`` builds it, including the depth stamp."""
    target = resolve_launch_target("task", parent)
    child = await subagent_mod._build_child_session(
        label=job_id,
        prompt="go",
        parent_session=parent,
        model_spec=model_spec,
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


# -- (a) depth 0 is unchanged -------------------------------------------------


def test_the_policy_function_only_closes_below_the_top(config):
    assert model_may_choose_tier() is True
    assert model_may_choose_tier(0) is True
    assert model_may_choose_tier(1) is False
    assert model_may_choose_tier(2) is False


@pytest.mark.asyncio
async def test_depth_zero_with_model_choice_still_picks_a_tier(tmp_path, config):
    root = make_root(tmp_path)
    assert effort_in_schema(root), "the top-level session must still be offered the tiers"

    job_id = root._launch_subagent(label="s", prompt="p", agent="scout", effort="hi")
    job = root.jobs.get(job_id)
    assert job is not None
    assert job.model_label == HI
    assert job.owns_model is True

    result = await execute_task(
        "c1",
        {"label": "s2", "prompt": "p", "effort": "hi"},
        None,
        None,
        root._build_tool_context(),
    )
    assert not result.is_error, text_of(result)
    await root.dispose()


@pytest.mark.asyncio
async def test_depth_zero_under_the_operator_default_is_still_refused(tmp_path, config):
    write_config(config, choice="operator")
    root = make_root(tmp_path)
    assert not effort_in_schema(root)
    # Through the tool's own executor: a bare ``execute_task`` publishes no build
    # record and is treated as an operator-side caller by design.
    result = await task_tool(root).execute(
        "c1",
        {"label": "s", "prompt": "p", "effort": "hi"},
        None,
        None,
        root._build_tool_context(),
    )
    assert result.is_error and "subagents.model_choice" in text_of(result)
    await root.dispose()


# -- (b) depth >= 1: not offered, refused, inherited --------------------------


@pytest.mark.asyncio
async def test_a_subagent_is_not_offered_effort_even_under_model_choice(tmp_path, config):
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    # Same config, same tool builder: the only difference is who is asking.
    assert effort_in_schema(root) and not effort_in_schema(child)
    assert "only the top-level session picks tiers" in task_tool(child).description
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_subagent_that_passes_effort_anyway_is_refused_and_nothing_launches(
    tmp_path, config
):
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    before = len(child.jobs.list())

    # Through the child's REAL executor (the wrapper publishes the build depth)...
    result = await task_tool(child).execute(
        "c1",
        {"label": "scout", "prompt": "p", "agent": "scout", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    text = text_of(result)
    assert result.is_error
    assert "inherit this session's model" in text
    # ...and it must NOT blame a setting the operator did not set.
    assert "model_choice" not in text
    assert len(child.jobs.list()) == before

    # Even a bare executor call is closed, because the CALL's own context
    # carries the depth: a tool object built before the stamp cannot leak.
    bare = await execute_task(
        "c2",
        {"label": "scout", "prompt": "p", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert bare.is_error and len(child.jobs.list()) == before
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_the_batch_form_is_refused_too(tmp_path, config):
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    result = await task_tool(child).execute(
        "c1",
        {"tasks": [{"label": "a", "prompt": "p"}, {"label": "b", "prompt": "p", "effort": "hi"}]},
        None,
        None,
        child._build_tool_context(),
    )
    assert result.is_error and "tasks.1.effort" in text_of(result)
    assert len(child.jobs.list()) == 0
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_subagent_cannot_write_a_tier_pin_into_a_role(tmp_path, config):
    from local_operator.agents import AgentRegistry
    from local_operator.tools.agent_tool import build_agent_tool

    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    child.agent_registry = AgentRegistry(tmp_path / "agents")
    # The bare test session carries no factory inventory, so build the tool the
    # way the registry would for this child's live context.
    agent_tool = build_agent_tool(child._build_tool_context())
    assert agent_tool is not None
    props = agent_tool.parameters["properties"]["effort"]
    members = [v["enum"] for v in props["anyOf"] if "enum" in v][0]
    assert members == ["inherit"], "a subagent is offered 'inherit' only, never a tier"

    result = await agent_tool.execute(
        "c1",
        {"op": "create", "name": "x", "description": "d", "instructions": "i", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert result.is_error and "cannot pin a role" in text_of(result)
    await child.dispose()
    await root.dispose()


# -- (c) an operator role pin survives at any depth ---------------------------


@pytest.mark.asyncio
async def test_an_operator_role_pin_still_resolves_for_a_nested_launch(tmp_path, config):
    from local_operator.agents import AgentRegistry
    from local_operator.tools.agent_tool import AgentParams, write_profile

    registry = AgentRegistry(tmp_path / "agents")
    # The operator's own surface: no validation context, so no depth gate.
    write_profile(
        registry,
        AgentParams(
            op="create",
            name="reviewer",
            description="Reviews",
            instructions="Review it.",
            effort="hi",
        ),
        creating=True,
    )
    root = make_root(tmp_path)
    root.agent_registry = registry
    child = await make_child(root, "job-1")
    child.agent_registry = registry

    spec = child._resolve_subagent_model("reviewer", None, strict=True)
    assert spec is not None and f"{spec.provider}/{spec.model_id}" == HI

    job = child.jobs.get(child._launch_subagent(label="r", prompt="p", agent="reviewer"))
    assert job is not None and job.model_label == HI and job.owns_model is True
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_recorded_tier_still_re_resolves_for_a_nested_resume(tmp_path, config):
    """The resolver itself is not gated: resume re-prices a RECORDED tier (item 5)."""
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    spec = child._resolve_subagent_model("task", "hi", strict=True)
    assert spec is not None and f"{spec.provider}/{spec.model_id}" == HI
    await child.dispose()
    await root.dispose()


# -- (d) unpinned children inherit the launching CHILD's model -----------------


@pytest.mark.asyncio
async def test_an_unpinned_scout_at_depth_two_inherits_the_session_model(tmp_path, config):
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")

    job = child.jobs.get(child._launch_subagent(label="s", prompt="p", agent="scout"))
    assert job is not None
    assert job.owns_model is False
    assert job.model_label == ROOT_LABEL
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_grandchild_inherits_the_running_model_of_a_tier_pinned_parent(tmp_path, config):
    """Which model "inherit" means: the LAUNCHING child's actual running model.

    A depth-1 child the OPERATOR's pin moved onto the ``hi`` model launches an
    unpinned child; that grandchild runs on the same ``hi`` model, not on the
    root session's Radient auto. This is what existing inheritance already does
    (``run_subagent`` builds an inherit child with ``parent_session.model``);
    the test fixes it so the new gate cannot be "fixed" later into resolving
    against the root instead.
    """
    root = make_root(tmp_path)
    pinned = root._resolve_subagent_model("task", "hi", strict=True)
    assert pinned is not None
    child = await make_child(root, "job-1", model_spec=pinned)
    assert child.model_label == HI

    job = child.jobs.get(child._launch_subagent(label="g", prompt="p"))
    assert job is not None
    assert job.owns_model is False
    assert job.model_label == HI
    await child.dispose()
    await root.dispose()


# -- review round 1 (R1-3, R1-4, R1-5) ----------------------------------------


@pytest.mark.asyncio
async def test_a_recorded_depth_two_tier_resumes_on_its_tier_through_comms(tmp_path, config):
    """Resume is NOT gated: a recorded tier re-resolves at depth 2, end to end.

    The launcher is called directly (the gate lives on the TOOL, so this is the
    shape of a child recorded before the gate existed, or an operator-authored
    launch). The point is the ROW of the resumed child: if resume ever routed
    through the tool-argument gate it would lose its recorded tier and come
    back on the session's model while the panel still said ``hi``.
    """
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    first_id = child._launch_subagent(label="s", prompt="p", agent="scout", effort="hi")
    record = child.subagent_comms._record(first_id)
    assert record is not None and record.depth == 2

    deadline = asyncio.get_running_loop().time() + 10
    while (j := child.jobs.get(first_id)) is None or j.status != "completed":
        assert asyncio.get_running_loop().time() < deadline, "first run never settled"
        await asyncio.sleep(0.01)

    new_id, error = child.subagent_comms.resume(first_id, "carry on")
    assert error is None and new_id is not None
    # The hand-built child has no comms row of its own, so the resume re-parents
    # the JOB onto the comms-owning root (``_launch_parent``); the depth and the
    # tier ride the recorded launch, which is what this pins.
    resumed = root.jobs.get(new_id)
    assert resumed is not None
    assert resumed.model_label == HI and resumed.owns_model is True and resumed.effort == "hi"
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_a_subagent_cannot_update_a_role_pin_to_a_tier(tmp_path, config):
    from local_operator.agents import AgentRegistry
    from local_operator.tools.agent_tool import (
        AgentParams,
        build_agent_tool,
        write_profile,
    )

    registry = AgentRegistry(tmp_path / "agents")
    write_profile(
        registry,
        AgentParams(op="create", name="rev", description="d", instructions="i"),
        creating=True,
    )
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    child.agent_registry = registry
    tool = build_agent_tool(child._build_tool_context())
    assert tool is not None
    result = await tool.execute(
        "c1",
        {"op": "update", "name": "rev", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert result.is_error and "cannot pin a role" in text_of(result)
    # ...and the stored profile did not move.
    profile = registry.get_agent_by_name("rev") if hasattr(registry, "get_agent_by_name") else None
    assert profile is None or not getattr(profile, "effort", None)
    await child.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_depth_zero_may_still_write_a_tier_pin_through_the_agent_tool(tmp_path, config):
    """The other side of the gate: the top-level model under ``model`` can pin."""
    from local_operator.agents import AgentRegistry
    from local_operator.tools.agent_tool import build_agent_tool

    root = make_root(tmp_path)
    root.agent_registry = AgentRegistry(tmp_path / "agents")
    tool = build_agent_tool(root._build_tool_context())
    assert tool is not None
    result = await tool.execute(
        "c1",
        {"op": "create", "name": "rev", "description": "d", "instructions": "i", "effort": "hi"},
        None,
        None,
        root._build_tool_context(),
    )
    assert not result.is_error, text_of(result)
    spec = root._resolve_subagent_model("rev", None, strict=True)
    assert spec is not None and f"{spec.provider}/{spec.model_id}" == HI
    await root.dispose()


@pytest.mark.asyncio
async def test_a_team_launch_at_depth_is_refused_a_tier_and_keeps_the_managers_pin(
    tmp_path, config
):
    """``task(agent='team:pod', effort=...)`` from a subagent: the tier is refused,
    and the launch that does go through runs on the MANAGER role's operator pin
    (teams carry no effort of their own)."""
    from local_operator.agents import AgentRegistry
    from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
    from local_operator.tools.agent_tool import AgentParams, write_profile

    registry = AgentRegistry(tmp_path / "agents")
    write_profile(
        registry,
        AgentParams(
            op="create",
            name="pod-lead",
            description="d",
            instructions="i",
            effort="lo",
            delegate=True,
        ),
        creating=True,
    )
    teams = TeamRegistry(config)
    teams.create_team(
        TeamEditFields(
            name="pod", manager="pod-lead", members=[TeamMember(role="coder")], instructions="x"
        )
    )
    root = make_root(tmp_path)
    root.agent_registry = registry
    root.team_registry = teams
    child = await make_child(root, "job-1")
    child.agent_registry = registry
    child.team_registry = teams

    refused = await task_tool(child).execute(
        "c1",
        {"label": "pod", "prompt": "p", "agent": "team:pod", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    assert refused.is_error and "inherit this session's model" in text_of(refused)

    job_id = child._launch_subagent(label="pod", prompt="p", agent="team:pod")
    job = child.jobs.get(job_id)
    assert job is not None and job.model_label == "deepseek/deepseek-flash"
    assert job.owns_model is True
    await child.dispose()
    await root.dispose()


# -- R1-3: the copy tells the truth under BOTH values of the key ---------------


@pytest.mark.asyncio
async def test_under_operator_a_subagent_gets_the_operator_copy_not_the_nested_copy(
    tmp_path, config
):
    """Under ``operator`` nobody on the model side may pick, so "only the
    top-level session may pick a tier" would be false and the operator's route
    (``subagents.model_choice``) must survive in the message."""
    write_config(config, choice="operator")
    root = make_root(tmp_path)
    child = await make_child(root, "job-1")
    assert "subagents.model_choice=operator" in task_tool(child).description
    assert "top-level" not in task_tool(child).description

    result = await task_tool(child).execute(
        "c1",
        {"label": "s", "prompt": "p", "agent": "scout", "effort": "hi"},
        None,
        None,
        child._build_tool_context(),
    )
    text = text_of(result)
    assert result.is_error
    assert "subagents.model_choice" in text and "top-level" not in text
    await child.dispose()
    await root.dispose()


# -- R1-5: a bool is not a depth -----------------------------------------------


def test_a_bool_in_the_tool_context_is_not_a_delegation_depth():
    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import (
        ADVERTISED_DELEGATION_DEPTH_KEY,
        effort_validation_context,
    )

    plain = ToolContext()
    assert effort_validation_context(plain)[ADVERTISED_DELEGATION_DEPTH_KEY] == 0
    flagged = ToolContext()
    object.__setattr__(flagged, "delegation_depth", True)
    assert effort_validation_context(flagged)[ADVERTISED_DELEGATION_DEPTH_KEY] == 0
    flagged.delegation_depth = 2
    assert effort_validation_context(flagged)[ADVERTISED_DELEGATION_DEPTH_KEY] == 2
