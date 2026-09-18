"""A session's DECLARED tool inventory is enforced, not advised.

The capability under test is what an unattended embedder needs and could not
previously express: "this run may reach exactly these tools and nothing else".
Before ``Session.set_tool_inventory`` the only way a write/exec-tier call could
succeed with no tty to answer the gate was ``--yolo``, which also unlocks
``bash``/``write``/``edit``/``eval`` — so "reach my MCP tools" and "be unable to
shell out" were not jointly expressible.

The assertions are written against what the PROVIDER is sent and what the loop
can RESOLVE, never against a private attribute, because those are the two halves
the security property is made of: a declaration that only trimmed the advertised
schema would leave the tool reachable by name (see
``test_an_excluded_tool_is_unreachable_by_name_through_the_resolver``).
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from local_operator.harness.approval import ask_approval
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolResult,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


class ScriptedStream:
    """Replays per-call event scripts; records requests (harness convention)."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def make_session(tmp_path, stream, tools: Sequence[AgentTool] | None = None, **kwargs) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=list(tools or ()),
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable", "env"],
        **kwargs,
    )


def echo_tool(executed: list[str], name: str = "echo") -> AgentTool:
    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append(name)
        return ToolResult(
            tool_call_id=tool_call_id, tool_name=name, content=[TextContent(text="ok")]
        )

    return AgentTool(name=name, parameters={"type": "object", "properties": {}}, execute=execute)


def tool_call_turns(name: str) -> list[list[StreamEvent]]:
    """One turn that calls ``name``, then one that answers in text."""
    return [
        [
            StreamToolCallDelta(index=0, id="c1", name=name, argument_delta="{}"),
            StreamEndEvent(stop_reason="toolUse"),
        ],
        [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
    ]


def surfaced_tools(stream: ScriptedStream, index: int = 0) -> list[str]:
    return [tool.name for tool in stream.requests[index].tools]


def tool_result_texts(session: Session) -> dict[str, str]:
    """``tool_call_id -> text`` for every tool result the MODEL was given.

    Read off the transcript rather than off ``ToolExecutionEndEvent``, and that
    is not a convenience: a call that never resolved emits no execution events at
    all — the loop parks a synthetic "Tool not found" result without them — so an
    event-shaped assertion would be blind to exactly the case this file exists to
    pin.
    """
    texts: dict[str, str] = {}
    for message in session._context.messages:
        # ``messages`` also carries ``CustomMessage`` rows (harness notices), which
        # have neither a role nor a tool call to key on.
        if not isinstance(message, Message) or message.role != "tool" or not message.tool_call_id:
            continue
        texts[message.tool_call_id] = "".join(getattr(part, "text", "") for part in message.content)
    return texts


async def _always_deny(tool_name: str, description: str) -> bool:
    return False


@pytest.mark.asyncio
async def test_declared_inventory_bounds_what_the_provider_is_sent(tmp_path):
    """The declaration is the session's whole surface: an excluded tool is gone
    from the schema list AND from the session's own answer about its reach."""
    executed: list[str] = []
    stream = ScriptedStream([[StreamTextDelta(delta="hi"), StreamEndEvent(stop_reason="stop")]])
    session = make_session(
        tmp_path,
        stream,
        tools=[
            echo_tool(executed, name="read"),
            echo_tool(executed, name="bash"),
            echo_tool(executed, name="write"),
        ],
    )
    session.set_tool_inventory(["read"])
    await session.prompt("go")

    assert surfaced_tools(stream) == ["read"]
    assert session.tool_inventory == ("read",)
    await session.dispose()


@pytest.mark.asyncio
async def test_capability_tools_arriving_after_the_declaration_are_subject_to_it(tmp_path):
    """``task``/``wait``/``jobs``/``wake``/``hub``/``ask`` reach the inventory
    through a merge that runs AFTER construction. A declaration that only
    filtered the factory's builtins would leave this session able to delegate to
    an unrestricted child — the excluded set one hop away, which is not
    excluded."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[])
    # Sanity: undeclared, the merge really does hand the session delegation.
    assert "task" in session.tool_inventory

    session.set_tool_inventory(["read"])
    assert "task" not in session.tool_inventory
    assert "bash" not in session.tool_inventory
    await session.dispose()


@pytest.mark.asyncio
async def test_an_excluded_tool_is_unreachable_by_name_through_the_resolver(tmp_path):
    """The half that is easy to miss. The loop asks the fallback resolver for any
    name NOT in ``context.tools`` before reporting an unknown tool — that is how
    a lazily discovered MCP tool is dispatchable — so an excluded MCP tool would
    stay reachable by name even with its schema withheld."""
    executed: list[str] = []
    resolved: list[str] = []
    deferred = echo_tool(executed, name="mcp__vendor_screen")
    stream = ScriptedStream(tool_call_turns("mcp__vendor_screen"))
    session = make_session(tmp_path, stream, tools=[echo_tool(executed, name="read")])

    def resolver(name: str) -> AgentTool | None:
        resolved.append(name)
        return deferred if name == "mcp__vendor_screen" else None

    session.set_tool_inventory(["read"])
    session.set_fallback_tool_resolver(resolver)

    await session.prompt("go")

    assert executed == []
    # The resolver is not even consulted: an excluded name must not reach the
    # dispatch path at all, rather than being resolved and then denied.
    assert resolved == []
    assert "Tool not found" in tool_result_texts(session)["c1"]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_declared_deferred_tool_still_resolves(tmp_path):
    """The negative control for the test above: a declared name that is not
    materialized yet is exactly what the resolver is for, and must keep working."""
    executed: list[str] = []
    deferred = echo_tool(executed, name="mcp__vendor_screen")
    stream = ScriptedStream(tool_call_turns("mcp__vendor_screen"))
    session = make_session(tmp_path, stream, tools=[echo_tool(executed, name="read")])
    session.set_tool_inventory(["read", "mcp__vendor_screen"], unattended=True)
    session.set_fallback_tool_resolver(
        lambda name: deferred if name == "mcp__vendor_screen" else None
    )

    await session.prompt("go")
    assert executed == ["mcp__vendor_screen"]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_declared_tool_runs_unattended_where_the_gate_would_have_denied(tmp_path):
    """The other half of the gap. With the excluded set unreachable, a gate that
    refuses a PERMITTED tool is refusing the only thing the run was allowed to
    do — so an unattended declaration stands as the approval for its members."""
    executed: list[str] = []
    stream = ScriptedStream(tool_call_turns("echo"))
    session = make_session(
        tmp_path, stream, tools=[echo_tool(executed)], request_approval=_always_deny
    )
    session.set_tool_inventory(["echo"], unattended=True)
    await session.prompt("go")

    assert executed == ["echo"]
    await session.dispose()


@pytest.mark.asyncio
async def test_the_same_declaration_stays_gated_when_the_host_is_attended(tmp_path):
    """``unattended`` is deliberately NOT implied by narrowing: a host with a
    human at the gate narrows a role's reach and still wants that human asked
    per call. Same declaration, opposite outcome — so the flag is what decides,
    not the declaration."""
    executed: list[str] = []
    stream = ScriptedStream(tool_call_turns("echo"))
    session = make_session(
        tmp_path, stream, tools=[echo_tool(executed)], request_approval=_always_deny
    )
    session.set_tool_inventory(["echo"])
    await session.prompt("go")

    assert executed == []
    await session.dispose()


@pytest.mark.asyncio
async def test_declared_approval_never_widens_to_an_undeclared_name(tmp_path):
    """Asked directly, because this is the invariant the whole design rests on:
    the gate that a declaration installs decides its MEMBERS. A gate that
    approved whatever it was handed would put the session back where ``--yolo``
    leaves it the first time some future writer reached the inventory by a route
    the filter does not cover."""

    async def deny(tool_name: str, description: str) -> bool:
        return False

    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[echo_tool([], name="declared")],
        request_approval=deny,
    )
    session.set_tool_inventory(["declared"], unattended=True)
    gate = session._tool_approval_gate()
    assert gate is not None

    # Through ``ask_approval``, the shared arity resolver, because that is how the
    # loop itself calls a gate — a gate's accepted shape is the union of the two
    # host signatures and only that helper knows which one applies.
    assert await ask_approval(gate, "declared", "d") is True
    assert await ask_approval(gate, "bash", "d") is False
    await session.dispose()


@pytest.mark.asyncio
async def test_a_name_that_matches_nothing_is_unreachable_and_not_an_error(tmp_path):
    """Fails CLOSED, like a role allow-list: a declaration naming a tool this
    build does not have (an MCP server not connected on this host, a renamed
    builtin) matches nothing rather than raising."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[echo_tool([], name="read")])
    session.set_tool_inventory(["no_such_tool"])
    await session.prompt("go")

    assert surfaced_tools(stream) == []
    assert session.tool_inventory == ()
    await session.dispose()


@pytest.mark.asyncio
async def test_narrowing_is_one_way_for_the_life_of_the_session(tmp_path):
    """A bounded reach cannot be lifted mid-run by whoever can call the setter.
    Documented rather than incidental: a host that needs a different set starts a
    session with it.

    Asserted against the REACH the declaration still produces, never against the
    declaration's own bookkeeping and not against the materialized view on its
    own: the defect this pins was a setter that quietly accepted the wider set
    (and a ``refresh_tools`` that then re-derived the inventory from it), so an
    assertion on ``tool_inventory`` taken straight after the setter observed
    "the filter ran" while the widening attempt was never exercised. Here the
    widening is attempted, the session is then handed the FULL candidate set —
    the same route a lazily discovered MCP tool takes — and both the reach and
    the provider's schema list must still exclude it. Remove the guard in
    ``set_tool_inventory`` and this test fails on the raise, on ``bash`` coming
    back, and on what the provider is sent.
    """
    all_tools = [echo_tool([], name=name) for name in ("read", "bash", "write")]
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=all_tools)
    session.set_tool_inventory(["read"])

    with pytest.raises(ValueError):
        session.set_tool_inventory(["read", "bash"])

    # The inventory write that follows: ``bash`` must not come back through it.
    session.refresh_tools(all_tools)
    assert session.tool_inventory == ("read",)
    await session.prompt("go")
    assert surfaced_tools(stream) == ["read"]
    await session.dispose()


@pytest.mark.asyncio
async def test_a_declaration_in_force_is_not_lifted_by_none(tmp_path):
    """``names=None`` is the value an ABSENT declaration has, not the statement
    "unrestricted". Read as a reset it restored the session's full builtin reach
    — ``write`` and ``bash`` included — which is the same bound lifted by the
    same setter, so the invariant above must cover it too (the docstring already
    claimed it does)."""
    all_tools = [echo_tool([], name=name) for name in ("read", "write")]
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=all_tools)
    session.set_tool_inventory(["read"])

    session.set_tool_inventory(None)

    session.refresh_tools(all_tools)
    assert session.tool_inventory == ("read",)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_later_declaration_may_still_tighten(tmp_path):
    """The mirror of the invariant, and the reason it is a SUBSET check rather
    than "one declaration per session": a host may narrow further mid-run, and a
    guard that refused every second call would break that while satisfying the
    test above."""
    all_tools = [echo_tool([], name=name) for name in ("read", "bash")]
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=all_tools)
    session.set_tool_inventory(["read", "bash"])

    session.set_tool_inventory(["read"])

    session.refresh_tools(all_tools)
    assert session.tool_inventory == ("read",)
    await session.dispose()


@pytest.mark.asyncio
async def test_a_narrowing_call_cannot_turn_the_declarations_approval_back_on(tmp_path):
    """The approval half is one-way in the same direction the reach is.

    Declaring a set with a human at the gate is that human's decision to keep
    deciding; the *allowed* subset call must not be able to lift it to
    auto-approval, or the loosening direction of the invariant is open to
    exactly the caller the widening refusal above is written against. Driven
    through the GATE rather than read off the flag, because that is what the
    loop calls and what decides whether the tool runs.

    Before the fix this sequence left the declared gate answering ``True`` for
    ``echo`` without consulting the base gate at all — the reach stayed bounded
    and the approval did not — so the base-gate denial asserted below is the
    discriminator, not the declaration's bookkeeping.
    """
    executed: list[str] = []
    stream = ScriptedStream(tool_call_turns("echo"))
    session = make_session(
        tmp_path, stream, tools=[echo_tool(executed)], request_approval=_always_deny
    )
    session.set_tool_inventory(["echo", "bash"])

    session.set_tool_inventory(["echo"], unattended=True)

    gate = session._tool_approval_gate()
    assert gate is not None
    assert await ask_approval(gate, "echo", "d") is False
    await session.prompt("go")
    assert executed == []
    await session.dispose()


@pytest.mark.asyncio
async def test_a_later_call_may_still_turn_the_declarations_approval_off(tmp_path):
    """The other half of the one-way flag, and the trade-off it accepts.

    ``unattended=False`` on a later call is a TIGHTENING and stays honoured even
    though the first declaration was auto-approved: a host that narrows and
    wants the human asked again gets that. It fails closed — where nobody can
    answer, the base gate refuses the remaining calls — which is the direction
    to fail in, and the alternative (freezing the first value) would silently
    ignore the request.
    """
    executed: list[str] = []
    stream = ScriptedStream(tool_call_turns("echo"))
    session = make_session(
        tmp_path, stream, tools=[echo_tool(executed)], request_approval=_always_deny
    )
    session.set_tool_inventory(["echo", "bash"], unattended=True)

    session.set_tool_inventory(["echo"], unattended=False)

    await session.prompt("go")
    assert executed == []
    await session.dispose()


@pytest.mark.asyncio
async def test_an_undeclared_session_is_untouched(tmp_path):
    """The negative case the change must not disturb: no declaration, no
    narrowing, and the ordinary gate still decides."""
    executed: list[str] = []
    tool = echo_tool(executed, name="echo")
    stream = ScriptedStream(tool_call_turns("echo"))
    session = make_session(tmp_path, stream, tools=[tool])

    await session.prompt("go")
    assert "echo" in surfaced_tools(stream)
    assert "task" in session.tool_inventory
    assert executed == ["echo"]
    await session.dispose()


@pytest.mark.asyncio
async def test_yolo_and_a_declaration_are_independent(tmp_path):
    """``--yolo`` is an APPROVAL override and a declaration is a REACH bound, so
    a yolo run with a declaration still cannot reach an excluded tool. Treating
    them as one thing would make the bound evaporate on exactly the runs that
    need it most."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[echo_tool([], name="read")], yolo=True)
    session.set_tool_inventory(["read"])
    await session.prompt("go")

    assert surfaced_tools(stream) == ["read"]
    assert session._tool_approval_gate() is None  # --yolo still skips the gate object
    await session.dispose()


@pytest.mark.asyncio
async def test_the_attached_roles_allow_list_is_recorded_on_the_session(tmp_path, monkeypatch):
    """``attach_agent_profile`` stamps INSTRUCTIONS only — the role's ``tools:``
    allow-list was enforced solely where the profile is launched as a subagent.
    It is recorded on the session here so a headless host can honour it, and
    recorded at attach time because re-resolving later would have to go back
    through a DISPLAY name."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[echo_tool([], name="read")])

    assert session.attached_profile_tools == ()
    assert session.attach_agent_profile("reviewer") == "reviewer"
    reviewer_tools = session.attached_profile_tools
    assert "read" in reviewer_tools
    assert "edit" not in reviewer_tools

    session.clear_agent_profile()
    assert session.attached_profile_tools == ()
    await session.dispose()


@pytest.mark.asyncio
async def test_a_roles_allow_list_bounds_a_session_that_honours_it(tmp_path, monkeypatch):
    """End-to-end of the pair above: the reviewer seed's allow-list is what makes
    a headless reviewer unable to ``edit`` the diff it was asked to review."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(
        tmp_path,
        stream,
        tools=[
            echo_tool([], name="read"),
            echo_tool([], name="edit"),
            echo_tool([], name="bash"),
        ],
    )
    assert session.attach_agent_profile("reviewer") == "reviewer"
    session.set_tool_inventory(session.attached_profile_tools)
    await session.prompt("go")

    surfaced = set(surfaced_tools(stream))
    assert "read" in surfaced
    assert "edit" not in surfaced
    # ``bash`` IS in the reviewer seed's allow-list, deliberately — the role's
    # whole point is that it can run the tests while being unable to change what
    # it reviews. Asserted so nobody reads this test as "a role surface is a hard
    # lockdown": a caller that needs no local execution at all declares
    # ``--tools`` by name rather than reaching for a packaged role.
    assert "bash" in surfaced
    await session.dispose()


class FakeMcpManager:
    """The one accessor the declaration consults, and nothing else."""

    def __init__(self, tools: Sequence[AgentTool]) -> None:
        self._tools = list(tools)

    def get_tools(self) -> list[AgentTool]:
        return list(self._tools)


def attach_manager(session, manager) -> None:
    """What ``attach_mcp_dispose`` does for the manager handle, without the SDK.

    Untyped params on purpose, mirroring the subagent suite's helper of the same
    name: the fake carries only the lookup surface, so it cannot satisfy
    ``McpManager``'s annotation, and the point of the test is which tools land on
    the session — not the manager's own type.
    """
    session.mcp_manager = manager


@pytest.mark.asyncio
async def test_a_declared_mcp_tool_is_granted_its_schema(tmp_path):
    """MCP tools are lazy: the model is expected to activate one from the server's
    catalogue, which it does with ``read`` — a tool a bounded runtime may
    deliberately not have. Measured live before this: a declaration naming an MCP
    tool reached NOTHING, and the model (shown no tools at all) emitted a tool call
    as prose. A declaration is that decision already made, so the named tool is
    granted its schema with no activation step."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[echo_tool([], name="read")])
    attach_manager(session, FakeMcpManager([echo_tool([], name="mcp__vendor_screen")]))
    session.set_tool_inventory(["mcp__vendor_screen"], unattended=True)
    await session.prompt("go")

    assert surfaced_tools(stream) == ["mcp__vendor_screen"]
    assert session.unresolved_declared_tools() == ()
    await session.dispose()


@pytest.mark.asyncio
async def test_a_manager_does_not_grant_an_undeclared_mcp_tool(tmp_path):
    """The negative half: the manager is not a licence to reach its whole
    surface. Only what the caller enumerated is granted."""
    stream = ScriptedStream([[StreamEndEvent(stop_reason="stop")]])
    session = make_session(tmp_path, stream, tools=[echo_tool([], name="read")])
    attach_manager(
        session,
        FakeMcpManager(
            [echo_tool([], name="mcp__vendor_screen"), echo_tool([], name="mcp__vendor_delete")]
        ),
    )
    session.set_tool_inventory(["read"])
    await session.prompt("go")

    assert surfaced_tools(stream) == ["read"]
    await session.dispose()


@pytest.mark.asyncio
async def test_only_never_arriving_names_are_reported_unresolved(tmp_path):
    """The report exists so a typo in a security control does not look like a
    harness fault. It is answered from the live inventory, so a name that DID
    arrive — including one granted from the MCP manager — is never reported.

    Meaningful once discovery has settled, which is why the exec host asks at the
    end of the run rather than at startup: a server still connecting and a typo
    are the same observation until then."""
    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[echo_tool([], name="read")],
    )
    session.set_tool_inventory(["read", "reed", "mcp__vendor_screen"])
    assert session.unresolved_declared_tools() == ("mcp__vendor_screen", "reed")

    # A declared MCP tool that the manager CAN supply is not unresolved. Granted
    # by the settle path — ``session_factory`` calls this again once discovery
    # settles, which is the route a lazily arriving server takes in production —
    # rather than by re-declaring a wider set: a declaration is one-way, so the
    # second ``set_tool_inventory`` this test used to make is no longer a
    # supported way to say this (see the invariant test above).
    attach_manager(session, FakeMcpManager([echo_tool([], name="mcp__vendor_screen")]))
    session.materialize_declared_tools()
    assert session.unresolved_declared_tools() == ("reed",)
    await session.dispose()


@pytest.mark.asyncio
async def test_materializing_with_no_manager_is_a_no_op(tmp_path):
    """Most sessions have no MCP manager wired at all, and the settle path calls
    this unconditionally — so absent must read as 'nothing to grant', never raise."""
    session = make_session(
        tmp_path,
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[echo_tool([], name="read")],
    )
    session.set_tool_inventory(["read"])
    assert session.materialize_declared_tools() == ()
    await session.dispose()
