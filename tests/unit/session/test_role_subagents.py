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
    nobody is watching, so the SPAWN/persist tools stay pruned.

    ``jobs`` is the deliberate exception: it only observes and cancels the
    child's OWN background bash jobs, and the reviewer keeps ``bash`` (hence
    ``background``), so pruning ``jobs`` would leave a reviewer that
    backgrounded a long command looping on ``Tool not found: jobs``. See the
    ``_can_background`` invariant in ``harness/subagent.py``."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = RecordingStream()
    parent = make_parent(tmp_path, stream)

    await run_role(parent, "reviewer")

    names = {tool.name for tool in stream.requests[0].tools}
    # The fan-out and persistence tools stay gone: a non-delegating role must
    # not start its own children or arm a wake its one-prompt session can't keep.
    assert not names & {"task", "wait", "wake"}
    # But it keeps ``jobs`` to poll/cancel the background bash jobs it can
    # still produce (it has ``bash``), which is what stops the poll loop.
    assert "jobs" in names
    assert "bash" in names
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
    # The operator's allowlist wins for every capability that can CHANGE
    # something: nothing here can edit, write or execute.
    assert names.isdisjoint(
        {"write", "edit", "bash", "eval", "browser", "task", "todo", "glob", "grep"}
    ), f"the operator's allowlist should win, got {names}"
    # Two deliberate additions ride along on top of it. ``hub`` so a restricted
    # child can still answer its parent's questions, and the read-only network
    # floor (``_with_network_floor``) so a role whose persisted tag list
    # predates those tools is not left unable to reach the web.
    #
    # The floor is applied to any allowlist, because a tag list cannot say
    # whether it omits ``web_search`` deliberately or merely because it was
    # written before the tool existed — and on this machine every installed
    # role was in fact the latter. The cost is recorded here rather than
    # hidden: an operator who narrows a role to ``read`` on purpose still gets
    # retrieval back. That is a read with no local side effect, so it takes
    # nothing away from what the allowlist was drawn to prevent.
    assert names == {"read", "hub", "web_search", "web_fetch"}, names
    await parent.dispose()


@pytest.mark.asyncio
async def test_restricted_role_constructs_only_its_effective_builtin_tools(
    tmp_path, monkeypatch
) -> None:
    """A restricted child must never mint schemas it will immediately discard."""
    from local_operator.tools import registry as tool_registry

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
    del agent

    stream = RecordingStream()
    parent = make_parent(tmp_path, stream, agent_registry=registry)

    constructed: list[str] = []
    factory_calls: list[tuple[str | None, list[str] | None, list[str], list[str]]] = []
    create_tools = tool_registry.create_tools

    def recording_create_tools(context, enabled=None):
        start = len(constructed)
        tools = create_tools(context, enabled=enabled)
        factory_calls.append(
            (
                context.job_id,
                None if enabled is None else list(enabled),
                constructed[start:],
                [tool.name for tool in tools],
            )
        )
        return tools

    monkeypatch.setattr(tool_registry, "create_tools", recording_create_tools)
    for name, builder in tuple(tool_registry.TOOL_BUILDERS.items()):
        def recording_builder(context, *, _name=name, _builder=builder):
            constructed.append(_name)
            return _builder(context)

        monkeypatch.setitem(tool_registry.TOOL_BUILDERS, name, recording_builder)

    await run_role(parent, "reviewer")

    # The allowlist is read plus the deliberate read-only network floor and
    # parent-messaging capability. Inspect the child's construction context,
    # separately from parent calls and Session's later capability refreshes,
    # so this proves which builders the restricted launch invoked.
    child_call = next(call for call in factory_calls if call[0] is not None)
    assert child_call[0]
    assert child_call[1:] == (
        ["read", "web_search", "web_fetch", "hub"],
        ["read", "web_search", "web_fetch", "hub"],
        ["read", "web_search", "web_fetch", "hub"],
    )
    assert [tool.name for tool in stream.requests[0].tools] == [
        "read",
        "web_search",
        "web_fetch",
        "hub",
    ]
    await parent.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_name", "custom_tools"),
    [
        ("reviewer", None),
        ("web-only", ("web_search",)),
        ("scout", None),
        ("task", None),
    ],
)
async def test_restricted_tool_construction_matches_legacy_inventory(
    tmp_path, monkeypatch, agent_name, custom_tools
) -> None:
    """Preselection preserves exact final tools and provider schemas."""
    from local_operator.agent_profiles import READ_ONLY_TOOLS
    from local_operator.harness import subagent as subagent_mod
    from local_operator.session.session import SESSION_CAPABILITY_TOOLS
    from local_operator.tools.registry import TOOL_BUILDERS, create_tools

    config = tmp_path / agent_name
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    registry = AgentRegistry(config)
    if custom_tools is not None:
        registry.create_agent(
            AgentEditFields(
                name=agent_name,
                description="network floor oracle role",
                tags=["role", f"tools:{','.join(custom_tools)}"],
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

    stream = RecordingStream()
    parent = make_parent(tmp_path, stream, agent_registry=registry)
    if agent_name == "scout":
        # Exercise the no-profile fallback even if a packaged scout seed exists.
        monkeypatch.setattr(subagent_mod, "_resolve_role", lambda *_: None)

    contexts = []
    child_sessions = []
    from local_operator.session.session import Session as SessionClass

    original_init = SessionClass.__init__

    def capture_child_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if kwargs.get("job_id"):
            child_sessions.append(self)

    monkeypatch.setattr(SessionClass, "__init__", capture_child_init)
    original_create_tools = create_tools

    def capture_child_context(context, enabled=None):
        if context.job_id:
            contexts.append(context)
        return original_create_tools(context, enabled=enabled)

    monkeypatch.setattr("local_operator.tools.registry.create_tools", capture_child_context)
    await run_role(parent, agent_name)

    assert len(contexts) >= 1
    assert len(child_sessions) == 1
    tool_context = contexts[0]
    profile = None if agent_name == "scout" else subagent_mod._resolve_role(agent_name, parent)
    restricted = (profile is not None and bool(profile.tools)) or agent_name == "scout"

    # Verbatim pre-optimization oracle: mint the whole registry first, then
    # apply the allowlist/network floor, scout fallback and child messaging.
    legacy_inventory = original_create_tools(tool_context)
    hub_tool = next((tool for tool in legacy_inventory if tool.name == "hub"), None)
    if profile is not None and profile.tools:
        allowed_names = set(profile.tools)
        legacy_tools = [tool for tool in legacy_inventory if tool.name in allowed_names]
        present = {tool.name for tool in legacy_tools}
        network_floor = {}
        for tool in legacy_inventory:
            if tool.name in {"web_search", "web_fetch"} and tool.name not in present:
                network_floor.setdefault(tool.name, tool)
        legacy_tools.extend(network_floor.values())
    elif agent_name == "scout":
        legacy_tools = [tool for tool in legacy_inventory if tool.name in READ_ONLY_TOOLS]
    else:
        legacy_tools = list(legacy_inventory)
    if restricted and hub_tool is not None and all(tool.name != "hub" for tool in legacy_tools):
        legacy_tools.append(hub_tool)

    # Session construction merges lifecycle tools from its own context. Re-run
    # those original builders, replace duplicate names and preserve their append
    # order, then mirror the child's existing descendant-boundary pruning.
    lifecycle_context = child_sessions[0]._build_tool_context()
    lifecycle_tools = original_create_tools(
        lifecycle_context, enabled=SESSION_CAPABILITY_TOOLS
    )
    merged = list(legacy_tools)
    for capability in lifecycle_tools:
        if any(tool.name == capability.name for tool in merged):
            merged = [
                capability if tool.name == capability.name else tool for tool in merged
            ]
        else:
            merged.append(capability)
    merged_in = {tool.name for tool in merged} - {tool.name for tool in legacy_tools}
    may_delegate = (
        profile.may_delegate
        if profile is not None
        else any(tool.name == "task" for tool in parent._tools)
    )
    if agent_name == "scout" or not may_delegate:
        drop = merged_in
    else:
        drop = {name for name in merged_in if name == "wake"}
    if subagent_mod._can_background(legacy_tools):
        drop = drop - {"jobs"}
    expected_names = [tool.name for tool in merged if tool.name not in drop]
    actual_names = [tool.name for tool in child_sessions[0]._tools]
    assert actual_names == expected_names, (
        f"{agent_name}: actual={actual_names!r}, expected={expected_names!r}"
    )
    assert [tool.name for tool in stream.requests[0].tools] == expected_names
    assert set(TOOL_BUILDERS) >= set(expected_names)
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
