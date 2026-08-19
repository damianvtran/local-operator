"""Session._launch_subagent: the production caller of the subagent engine.

``_launch_subagent`` registers one one-shot child run on the parent session's
job manager and emits ``subagent_start`` / ``subagent_end`` on the parent
event stream. These tests drive it through a real ``run_subagent`` against a
ScriptedStream, so the child actually runs its own loop and the parent sees
the full lifecycle.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from local_operator.harness.types import (
    AbortSignal,
    AgentEvent,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
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
    assert stream.requests
    assert stream.requests[0].messages
    assert isinstance(stream.requests[0].messages[0], Message)
    assert stream.requests[0].messages[0].text == "go do a thing"

    # Item 12: once the task settles, the idle TOP-LEVEL parent re-wakes with
    # the result instead of polling `jobs`; the shared provider sees a second
    # request carrying that job-result custom message.
    await wait_for(lambda: len(stream.requests) >= 2)
    assert any("background job 'sub' completed" in m.text for m in stream.requests[1].messages)

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
    budget. Live finding: a one-shot child ran 48 requests / 1.5M tokens on the
    default 600k threshold while the parent's absolute trigger was 250k. The
    child Session must receive the parent's compaction settings."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness import subagent as subagent_mod

    capped = CompactionSettings(threshold_tokens=250_000)
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
    assert child._compaction_settings.threshold_tokens == 250_000
    assert child._compaction_settings == capped
    await parent.dispose()


# --- session parity: what a child is actually constructed with -----------------------
#
# Reported live: a subagent "worked around its lack of MCP access by using your
# stored OAuth token directly" — it could not call the Linear MCP tools its
# parent had, so it improvised with raw credentials. These tests pin the
# inheritance decisions that failure exposed.


class FakeMcpManager:
    """The manager surface ``_child_mcp_wiring`` and the ``mcp://`` resolver use.

    Real ``McpManager`` construction needs the SDK, transports and a config
    pass; every behaviour under test here is about which tools land on which
    session, so the fake carries exactly the lookup surface both consume.
    """

    def __init__(self, tools: dict[str, list[str]]) -> None:
        async def never_called(tool_call_id, args, signal, on_update, context):
            raise AssertionError("these tests never dispatch an MCP call")

        self._by_server = {
            server: [
                AgentTool(
                    name=f"mcp__{server}_{raw}",
                    description=f"{raw} on {server}",
                    parameters={"type": "object", "properties": {}},
                    execute=never_called,
                )
                for raw in raws
            ]
            for server, raws in tools.items()
        }
        self._meta = {
            tool.name: {"server_name": server, "mcp_tool_name": raw, "deferred": False}
            for server, raws in tools.items()
            for raw, tool in zip(raws, self._by_server[server])
        }

        self.disconnected = 0

    async def disconnect_all(self) -> None:
        """The real manager tears every server down here. A child calling it
        would end MCP for the rest of the PARENT's session."""
        self.disconnected += 1

    def get_all_server_names(self) -> list[str]:
        return sorted(self._by_server)

    def get_connection_status(self, name: str) -> str:
        return "connected"

    def get_server_config(self, name: str) -> object | None:
        return None

    def get_server_tools(self, name: str) -> list[AgentTool]:
        return sorted(self._by_server.get(name, ()), key=lambda tool: tool.name)

    def rebuild_tools(self) -> None:
        """Replace every AgentTool with a fresh, equivalent object, which is
        what a reconnect's ``_register_tools`` does to the real manager."""
        self.__init__(
            {
                s: [self._meta[t.name]["mcp_tool_name"] for t in ts]
                for s, ts in self._by_server.items()
            }
        )

    def get_tools(self) -> list[AgentTool]:
        out = [tool for tools in self._by_server.values() for tool in tools]
        return sorted(out, key=lambda tool: tool.name)

    def get_tool_meta(self, tool_name: str):
        return self._meta.get(tool_name)


async def build_child(parent, model_spec=None, job_id="job-1", agent="task"):
    from local_operator.harness import subagent as subagent_mod

    return await subagent_mod._build_child_session(
        label="sub",
        prompt="do the thing",
        parent_session=parent,
        model_spec=model_spec,
        job_id=job_id,
        agent=agent,
    )


def attach_manager(parent, manager) -> None:
    """What ``attach_mcp_dispose`` does for the manager handle, without the SDK:
    the parent HOLDS the manager and the child borrows it off that attribute."""
    parent.mcp_manager = manager


def knowledge_tail(session) -> str:
    """The system prompt's volatile tail — where the MCP catalogue rides."""
    return session._system_blocks_provider()[3]


def resolve(session, url: str) -> str | None:
    """Read one internal URL exactly as the ``read`` tool would."""
    resolver = session._skill_resolver
    assert resolver is not None, "the child must always get an internal-URL resolver"
    return resolver(url)


@pytest.mark.asyncio
async def test_child_gets_the_parents_mcp_manager_and_catalogue(tmp_path, monkeypatch):
    """The reported gap: a child had no MCP at all. It must borrow the parent's
    LIVE manager (never run a second discovery pass) and see the catalogue that
    tells it ``read mcp://<server>`` is available."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)

    child = await build_child(parent)

    assert child.mcp_manager is manager
    tail = knowledge_tail(child)
    assert "<mcps>" in tail
    assert "- linear:" in tail
    # BORROWED, not owned. `attach_mcp_dispose` sets `mcp_manager` AND hangs
    # `disconnect_all` on dispose; the obvious "make the child symmetric with
    # the parent" edit would therefore have the first subagent to finish tear
    # down every server for the rest of the parent's session.
    assert child._dispose_hooks == []
    await child.dispose()
    assert manager.disconnected == 0
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_starts_with_the_tools_the_parent_already_activated(tmp_path, monkeypatch):
    """Lazy activation is a token-budget decision the PARENT already made for
    the task it is delegating; the child should not have to re-read the same
    server to get back to where its parent was."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)
    active = manager.get_server_tools("linear")[1]  # mcp__linear_list_teams
    parent.refresh_tools(list(parent._tools) + [active])

    child = await build_child(parent)

    child_mcp = sorted(t.name for t in child._tools if t.name.startswith("mcp__"))
    assert child_mcp == ["mcp__linear_list_teams"]
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_activation_lands_on_the_child_not_the_parent(tmp_path, monkeypatch):
    """``read mcp://linear/list_issues`` inside a child must enable that schema
    on the CHILD. The parent's own resolver chain ends in an activate() bound to
    the parent's inventory, so a child that simply reused it would enable the
    tool on the wrong session and see nothing appear in its own."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    from local_operator.mcp.resources import make_mcp_resolver

    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    # Wired the way the factory wires it: the parent's chain ENDS in an mcp
    # resolver whose activate() targets the parent. Without this the parent's
    # resolver is None, a parent-first chain falls through to the child's, and
    # an inversion of the documented order is undetectable.
    parent_activations: list[tuple[str, str]] = []
    parent = make_session(
        tmp_path,
        OneShotStream(),
        skill_resolver=make_mcp_resolver(
            manager, lambda server, tool: parent_activations.append((server, tool))
        ),
    )
    attach_manager(parent, manager)
    parent_tools_before = [t.name for t in parent._tools]

    child = await build_child(parent)
    assert [t.name for t in child._tools if t.name.startswith("mcp__")] == []

    rendered = resolve(child, "mcp://linear/list_issues")

    assert rendered is not None and "mcp__linear_list_issues" in rendered
    assert [t.name for t in child._tools if t.name.startswith("mcp__")] == [
        "mcp__linear_list_issues"
    ]
    # The parent is untouched: a delegated read must not spend the parent's
    # prompt budget on a schema the parent never asked for, and its activate()
    # must not have been reached at all.
    assert parent_activations == []
    assert [t.name for t in parent._tools] == parent_tools_before
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_resolver_falls_through_to_the_parents_knowledge(tmp_path, monkeypatch):
    """``skill://`` and ``guide://`` are pure lookups over indexes the parent
    has already built. The child does not re-run knowledge SELECTION, but a
    prompt that names a skill must still be able to read it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(
        tmp_path,
        OneShotStream(),
        skill_resolver=lambda url: "SKILL BODY" if url == "skill://deploy" else None,
    )
    attach_manager(parent, FakeMcpManager({"linear": ["list_teams"]}))

    child = await build_child(parent)

    assert resolve(child, "skill://deploy") == "SKILL BODY"
    assert resolve(child, "skill://missing") is None
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_without_parent_mcp_still_builds(tmp_path, monkeypatch):
    """MCP unconfigured, SDK missing, or a host that never wired a manager: the
    child must build with an empty knowledge tail rather than degrade to an
    error, exactly as it did before MCP inheritance existed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    assert parent.mcp_manager is None

    child = await build_child(parent)

    assert child.mcp_manager is None
    assert knowledge_tail(child) == "<skills/>"
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_inherits_a_mid_session_model_override(tmp_path, monkeypatch):
    """``/model`` swaps the spec through ``set_model``. A child spawned after
    that must run the model the operator is now on, not the one the session
    booted with — otherwise a delegated task silently uses a different (and
    differently priced) model than everything around it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    switched = ModelSpec(provider="test", model_id="switched", context_window=42_000)
    parent.set_model(switched)

    child = await build_child(parent)

    assert child.model == switched
    # An explicit per-launch override still wins over the session's spec.
    override = ModelSpec(provider="test", model_id="override", context_window=7_000)
    other = await build_child(parent, model_spec=override)
    assert other.model == override
    await other.dispose()
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_of_top_level_session_keeps_delegation_but_not_wake(tmp_path, monkeypatch):
    """Depth is two: a child of a TOP-LEVEL session keeps task/wait/jobs (its
    own job manager is observable through its own tools and is disposed with
    it, so grandchildren cannot outlive their lineage), while ``wake`` never
    crosses any boundary — a child session ends after one prompt, so a wake
    armed there would be silently lost."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    child = await build_child(parent)

    names = {tool.name for tool in child._tools}
    assert {"task", "wait", "jobs"} <= names
    assert "wake" not in names
    # The prune must not take the ordinary inventory with it.
    assert {"bash", "read", "edit", "write"} <= names
    # ...and the loop sees the same list the session does.
    assert [t.name for t in child._context.tools] == [t.name for t in child._tools]
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_grandchild_never_advertises_session_capability_tools(tmp_path, monkeypatch):
    """One level deeper the whole set goes: a grandchild's children would
    register on a job manager nothing observes and that dies mid-turn."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    # A parent that is itself a child is recognisable by its _job_id.
    parent._job_id = "job-parent"

    child = await build_child(parent)

    names = {tool.name for tool in child._tools}
    assert names.isdisjoint({"task", "wait", "jobs", "wake"})
    assert {"bash", "read", "edit", "write"} <= names
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_scout_child_is_read_only_and_cannot_delegate(tmp_path, monkeypatch):
    """The scout tier: tool inventory filtered to lookups (allowlist, not
    tier — no bash, no eval-style execution, no MCP calls, no delegation),
    its prompt stamped with the scout preamble, and the capability tools
    pruned entirely: a read-only agent that delegates autonomous work is not
    read-only."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())

    child = await build_child(parent, agent="scout")

    names = {tool.name for tool in child._tools}
    assert names <= {"read", "glob", "grep", "list_variables", "read_variable"}
    assert {"bash", "edit", "write", "task", "wait", "jobs", "wake"}.isdisjoint(names)
    assert (
        "scout mode" in getattr(child._context.messages[0], "text", "")
        if child._context.messages
        else True
    )
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_inherits_cwd_approval_handler_and_never_yolo(tmp_path, monkeypatch):
    """The approval HANDLER carries over (the parent's UI surface is the only
    place a human can answer), the ``yolo`` FLAG never does. A parent running
    ``--yolo`` still auto-approves inside the child, because that is what its
    handler returns — the flag's job is to stop the child from bypassing a gate
    the parent is subject to, not to re-gate what the operator already opened."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))

    async def approve(tool_name: str, description: str) -> bool:
        return True

    parent = make_session(
        tmp_path, OneShotStream(), cwd="/tmp/workspace", request_approval=approve, yolo=True
    )

    child = await build_child(parent)

    assert child._cwd == "/tmp/workspace"
    assert child._request_approval is approve
    assert child._yolo is False
    assert child._build_tool_context().request_approval is approve
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_child_carries_the_sessions_standing_goal(tmp_path, monkeypatch):
    """``/goal`` is a constraint the operator set on the whole session, and a
    delegated slice is exactly where an unstated constraint gets violated. It
    is read off the parent's live holder on every provider call, so an edit
    reaches a child that is ALREADY RUNNING — a snapshot taken at build time
    would pass every weaker version of this test."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    parent.set_goal("never touch production")

    child = await build_child(parent)
    assert "never touch production" in knowledge_tail(child)

    parent.set_goal("staging only")
    # The child built BEFORE the edit sees it too: that is the live read, and
    # it is the property a build-time snapshot would fail.
    assert "staging only" in knowledge_tail(child)
    later = await build_child(parent)
    assert "staging only" in knowledge_tail(later)
    await later.dispose()
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_childs_approval_reaches_the_host_tagged_with_its_job(tmp_path, monkeypatch):
    """A background child's approval must be distinguishable from the parent
    turn's. Reproduced live before this existed: a subagent running past the
    end of its parent's turn inherited that turn's approval state and had its
    tools denied with no prompt shown to anyone, because the host had no way to
    tell the two apart. Driven through the real loop and the real ``write``
    tool, so the provenance is checked on the path a tool call actually takes,
    not on the context the session was constructed with."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    target = tmp_path / "child-wrote-this.txt"
    asks: list[tuple[str, str | None]] = []

    async def gate(tool_name: str, description: str, job_id: str | None) -> bool:
        asks.append((tool_name, job_id))
        return False  # denial is the interesting answer: it must be scopeable

    class WriteOnce:
        """One write attempt, then a plain answer."""

        def __init__(self) -> None:
            self.turn = 0

        def __call__(self, request: ChatRequest, signal: AbortSignal | None):
            self.turn += 1
            first = self.turn == 1

            async def gen():
                if first:
                    yield StreamToolCallDelta(
                        index=0,
                        id="call_1",
                        name="write",
                        argument_delta=json.dumps({"path": str(target), "content": "x"}),
                    )
                    yield StreamEndEvent(stop_reason="toolUse")
                else:
                    yield StreamTextDelta(delta="denied, stopping")
                    yield StreamEndEvent(stop_reason="stop")

            return gen()

    parent = make_session(tmp_path, WriteOnce(), request_approval=gate)
    job_id = parent._launch_subagent(label="writer", prompt="write the file")
    await wait_for(
        lambda: (job := parent.jobs.get(job_id)) is not None
        and job.status in ("completed", "failed")
    )

    assert asks == [("write", job_id)], f"the host saw {asks}"
    assert not target.exists(), "a denied write must not have run"
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_reconnect_swaps_the_managers_tools_under_a_running_child(tmp_path, monkeypatch):
    """The child deliberately does NOT install ``set_on_tools_changed`` — it is
    a single slot the parent holds, and taking it would freeze the parent's
    inventory for the rest of the session. The cost is that a reconnect leaves
    the child holding stale ``AgentTool`` objects. That is only acceptable
    because a stale object still routes (its execute closes over the manager
    and the server/tool pair, not over a connection) and because the child's
    NEXT activation picks up the fresh schemas. Both pinned here; confirmed
    live against the real Linear server before this test existed."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    manager = FakeMcpManager({"linear": ["list_teams", "list_issues"]})
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, manager)
    child = await build_child(parent)
    resolve(child, "mcp://linear/list_teams")
    stale = next(t for t in child._tools if t.name == "mcp__linear_list_teams")

    manager.rebuild_tools()  # what a reconnect / tools_list_changed does
    assert manager.get_tools()[0] is not stale, "the fake must actually swap objects"
    # The child keeps the object it already had: nothing refreshed it, and
    # nothing had to — the call still routes through the manager.
    assert next(t for t in child._tools if t.name == "mcp__linear_list_teams") is stale
    # The next activation rebuilds the selection from the manager's CURRENT
    # tools, so the child converges on the fresh schemas without a callback.
    resolve(child, "mcp://linear/list_issues")
    refreshed = {t.name: t for t in child._tools if t.name.startswith("mcp__")}
    assert sorted(refreshed) == ["mcp__linear_list_issues", "mcp__linear_list_teams"]
    assert refreshed["mcp__linear_list_teams"] is not stale
    assert [t.name for t in parent._tools if t.name.startswith("mcp__")] == []
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_parent_with_no_configured_servers_gives_an_empty_tail(tmp_path, monkeypatch):
    """A manager exists but discovery found nothing. The child must not carry
    an empty ``<mcps>`` block into every prompt, and ``mcp://`` must still
    answer rather than raise — the resolver is installed either way."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    attach_manager(parent, FakeMcpManager({}))

    child = await build_child(parent)

    assert knowledge_tail(child) == "<skills/>"
    assert resolve(child, "mcp://nope") == "Unknown MCP server: nope. Available: (none)"
    await child.dispose()
    await parent.dispose()


@pytest.mark.asyncio
async def test_the_launch_prompt_is_recorded_on_the_job(tmp_path, monkeypatch):
    """A reader opening a child's transcript needs the instruction it was given,
    and `label` is only a short summary the launcher wrote for a status line.
    Nothing else can supply it: `Session.prompt` feeds the text straight into
    the turn pipeline without emitting an event, so it never reaches
    `job.trajectory`. Recorded at registration, so a job still QUEUED behind
    the capacity gate — which has no trajectory and may never run — can still
    say what it was asked to do."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, OneShotStream())
    instruction = "review the diff at /tmp/x.diff and report blockers only"

    job_id = parent._launch_subagent(label="review", prompt=instruction)
    job = parent.jobs.get(job_id)
    assert job is not None
    assert job.prompt == instruction
    assert job.label == "review", "the summary must stay a separate field"
    # Verbatim, not summarised or truncated: the point is the exact instruction.
    assert job.prompt is not None and len(job.prompt) == len(instruction)

    await wait_for(
        lambda: (settled := parent.jobs.get(job_id)) is not None and settled.status == "completed"
    )
    settled = parent.jobs.get(job_id)
    assert settled is not None and settled.prompt == instruction, "must survive settlement"
    await parent.dispose()


@pytest.mark.asyncio
async def test_scout_preamble_reaches_the_provider_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = OneShotStream()
    parent = make_session(tmp_path, stream)
    job_id = parent._launch_subagent(label="research", prompt="Map the repo.", agent="scout")

    def settled() -> bool:
        job = parent.jobs.get(job_id)
        return job is not None and job.status != "running"

    await wait_for(settled)
    assert stream.requests
    first_user = next(message for message in stream.requests[0].messages if message.role == "user")
    # The ROLE framing reaches the turn ahead of the task, and the task itself
    # survives verbatim. Asserted by behaviour rather than by the seed's exact
    # wording: the scout guidance now lives in an editable profile
    # (``local_operator/agent_seeds/scout.md``), so pinning its prose here
    # would make every improvement to it a test failure.
    assert first_user.text.startswith("[role: scout]"), first_user.text[:80]
    assert "READ-ONLY" in first_user.text
    assert "Map the repo." in first_user.text
    assert first_user.text.index("Map the repo.") > first_user.text.index("READ-ONLY")
    await parent.dispose()


class LongStream:
    """A child turn that streams many deltas, yielding the loop between each.

    A real provider suspends on network I/O between chunks. Reproducing that
    suspension is what makes this a concurrency test rather than a CPU
    benchmark: without it the generator never yields and nothing else could
    run no matter how the harness behaved.
    """

    def __init__(self, deltas: int = 2000, chunk_chars: int = 1200) -> None:
        self.deltas = deltas
        self.chunk = "lorem ipsum dolor " * max(1, chunk_chars // 18)

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            for _ in range(self.deltas):
                await asyncio.sleep(0)
                yield StreamTextDelta(delta=self.chunk)
            yield StreamEndEvent(stop_reason="stop")

        return gen()


@pytest.mark.asyncio
async def test_the_loop_stays_responsive_while_several_subagents_run(tmp_path, monkeypatch):
    """Concurrent children must not stall the event loop they share.

    THE regression this guards. One asyncio loop serves the parent session,
    every subagent and the TUI's repaint, and the per-turn compaction gate
    used to tokenize the whole history INLINE on it. With several children
    streaming at once that made one child's threshold check freeze all of
    them: measured at 2.5 s of maximum loop stall (worst single stall 860 ms,
    with 116 of 121 stall samples inside the tokenizer). The user-visible
    symptom was an agent reporting that its subagents "only run when I yield".

    The measurement is a watchdog that asks to be woken every 5 ms and records
    how late it actually was. That overshoot IS the window in which nothing
    else on the loop could be serviced, which is exactly what a starved
    sibling experiences. Asserting on the watchdog rather than on wall-clock
    time keeps this about responsiveness and not about machine speed.

    The workload is calibrated, not arbitrary. It has to build a history big
    enough that the gate's tokenizer pass is expensive, because that is the
    stretch under test; at a smaller size both variants pass and the test
    proves nothing. Measured on this workload against the pre-fix tree and
    this one: worst stall 1353 ms before, 139 ms after. The 1 s bound sits an
    order of magnitude above the fixed behaviour (so a loaded CI box does not
    flake it) and comfortably below the broken one.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    parent = make_session(tmp_path, LongStream())

    stop = asyncio.Event()
    lateness: list[float] = []

    async def watchdog() -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            before = loop.time()
            await asyncio.sleep(0.005)
            lateness.append(loop.time() - before - 0.005)

    watcher = asyncio.create_task(watchdog())
    await asyncio.sleep(0.05)
    lateness.clear()

    job_ids = [parent._launch_subagent(label=f"c{i}", prompt="do the work") for i in range(6)]

    def all_settled() -> bool:
        jobs = [parent.jobs.get(job_id) for job_id in job_ids]
        return all(job is not None and job.status != "running" for job in jobs)

    try:
        await wait_for(all_settled, timeout=60.0)
    finally:
        stop.set()
        await watcher

    assert lateness, "the watchdog never ran — the measurement itself is broken"
    worst = max(lateness)
    assert worst < 1.0, (
        f"the event loop stalled for {worst * 1000:.0f} ms while subagents ran; "
        "a synchronous stretch is starving concurrent children"
    )
    await parent.dispose()


@pytest.mark.asyncio
async def test_a_launched_subagent_stays_out_of_the_resume_picker(tmp_path, monkeypatch):
    """The reported bug, end to end: `/resume` listed the machine's own runs.

    A child session is an ephemeral directory under ``sessions/`` with exactly
    the shape of a real conversation, so nothing on disk told them apart and
    the picker named each delegated run by its role preamble. This drives the
    production launch path and then asks the same function the picker calls,
    rather than asserting on the marker file — the marker is the mechanism, and
    the row count is the promise.
    """
    from local_operator.resume import (
        ORIGIN_SUBAGENT,
        recent_session_rows,
        session_origin,
    )

    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))

    # The parent is a user session: an ephemeral directory under sessions/,
    # which is what session_factory hands a TUI run.
    parent_dir = config / "sessions" / "parentsession"
    transcript = Transcript(parent_dir)
    await transcript.append_message(Message.user("fix the resume picker"))
    parent = Session(
        model=MODEL,
        stream_fn=OneShotStream(),
        tools=[],
        transcript=transcript,
        system_blocks_provider=lambda: ["stable", "env"],
    )

    events: list[AgentEvent] = []
    parent.subscribe(events.append)
    parent._launch_subagent(label="reviewer", prompt="[role: reviewer] review the diff")
    await wait_for(lambda: any(e.type == "subagent_end" for e in events))

    sessions = sorted(p.name for p in (config / "sessions").iterdir())
    assert len(sessions) == 2, f"expected the parent and its child on disk, got {sessions}"
    child_dir = next(p for p in (config / "sessions").iterdir() if p.name != "parentsession")
    # The nonzero control: the child really did write a transcript, so an empty
    # picker row set below cannot be a directory that was never created.
    assert (child_dir / "transcript.jsonl").is_file()

    # The promise first, then the mechanism that delivers it: a reader who
    # only trusts one line should trust the row set.
    rows = recent_session_rows(config, limit=50)
    assert [row.id for row in rows] == ["parentsession"]
    assert [row.name for row in rows] == ["fix the resume picker"]
    assert session_origin(child_dir) == ORIGIN_SUBAGENT

    await parent.dispose()
