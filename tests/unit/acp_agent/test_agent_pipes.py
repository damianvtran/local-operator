"""``LopAcpAgent`` over a REAL ACP connection: two ``os.pipe`` pairs.

The agent side is ``acp.run_agent`` exactly as ``lop acp`` runs it; the client
side is the SDK's ``connect_to_agent`` with a fake client that records what it
was sent and answers permission requests from a script. The lop session is a
fake owning session (injected through the opener seam), so these tests pin the
PROTOCOL behaviour — ids, stop reasons, the permission round trip, cancel,
refusals — without a provider. The assembled path is the e2e stage's job.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

acp = pytest.importorskip("acp")

from acp.schema import (  # noqa: E402
    AllowedOutcome,
    DeniedOutcome,
    HttpHeader,
    HttpMcpServer,
    McpServerStdio,
    RequestPermissionResponse,
)

from local_operator.acp_agent.agent import LopAcpAgent  # noqa: E402
from local_operator.harness.approval import ApprovalUnavailableError  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    AgentEndEvent,
    Message,
    MessageUpdateEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)

GUARD_S = 20.0


class FakeSession:
    """The owning-session surface ``LopAcpAgent`` uses, driven by a script."""

    def __init__(self, session_id: str, gate: Callable[..., Any]) -> None:
        self.session_id = session_id
        self._gate = gate
        self._handlers: list[Callable[[Any], None]] = []
        self.is_streaming = False
        self.aborted: list[str] = []
        self.subagents_cancelled: list[str] = []
        self.prompts: list[str] = []
        self.gate_outcomes: list[Any] = []
        self._abort = asyncio.Event()
        self.disposed = False

    def subscribe(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    def _emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)

    def abort(self, reason: str = "interrupted") -> None:
        self.aborted.append(reason)
        self._abort.set()

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        self.subagents_cancelled.append(reason)
        return 0

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)
        self.is_streaming = True
        self._abort.clear()
        try:
            await self._turn(text)
        finally:
            self.is_streaming = False

    async def _turn(self, text: str) -> None:
        message = Message(role="assistant", content=[TextContent(text="")], id="m1")
        if "[bash]" in text or "[hang-bash]" in text:
            self._emit(ToolExecutionStartEvent(tool_call_id="call-1", tool_name="bash"))
            try:
                approved = await self._gate("bash", "Run: ls", None)
            except ApprovalUnavailableError as exc:
                self.gate_outcomes.append(exc)
                approved = False
            else:
                self.gate_outcomes.append(approved)
            self._emit(
                ToolExecutionEndEvent(
                    tool_call_id="call-1",
                    tool_name="bash",
                    result=ToolResult(
                        tool_call_id="call-1",
                        content=[TextContent(text="ran" if approved else "denied")],
                        is_error=not approved,
                    ),
                )
            )
            if self._abort.is_set():
                self._emit(AgentEndEvent(aborted=True))
                return
        if "[slow]" in text:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._abort.wait(), timeout=GUARD_S)
            self._emit(AgentEndEvent(aborted=True))
            return
        if "[error]" in text:
            self._emit(AgentEndEvent(error="provider exploded"))
            return
        for delta in ("Hello", " there"):
            self._emit(MessageUpdateEvent(message=message, delta=delta))
        self._emit(AgentEndEvent())


class FakeClient:
    """Records updates; answers permission requests with the next scripted reply."""

    def __init__(self) -> None:
        self.updates: list[Any] = []
        self.permission_requests: list[Any] = []
        self.replies: list[Any] = []
        self.permission_asked = asyncio.Event()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)

    async def request_permission(
        self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any
    ) -> RequestPermissionResponse:
        self.permission_requests.append((tool_call, options))
        self.permission_asked.set()
        reply = self.replies.pop(0) if self.replies else "hang"
        if reply == "hang":
            await asyncio.Event().wait()
        if reply == "error":
            raise acp.RequestError(-32603, "client cannot ask")
        if reply == "deny":
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=reply)
        )


class Rig:
    def __init__(self) -> None:
        self.client = FakeClient()
        self.sessions: list[FakeSession] = []
        self.opened: list[tuple[str, dict[str, Any]]] = []
        self.conn: Any = None


@asynccontextmanager
async def rig() -> AsyncIterator[Rig]:
    state = Rig()

    @asynccontextmanager
    async def opener(cwd: str, gate: Callable[..., Any], servers: dict[str, Any]) -> AsyncIterator[Any]:
        state.opened.append((cwd, servers))
        session = FakeSession(f"sess{len(state.sessions):08x}", gate)
        state.sessions.append(session)
        try:
            yield session
        finally:
            session.disposed = True

    agent = LopAcpAgent(opener, permission_timeout_s=GUARD_S)
    loop = asyncio.get_running_loop()
    c2a_r, c2a_w = os.pipe()
    a2c_r, a2c_w = os.pipe()

    async def streams(read_fd: int, write_fd: int) -> tuple[Any, Any]:
        reader = asyncio.StreamReader()
        await loop.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(read_fd, "rb", 0)
        )
        from acp.stdio import _WritePipeProtocol

        protocol = _WritePipeProtocol()
        transport, _ = await loop.connect_write_pipe(lambda: protocol, os.fdopen(write_fd, "wb", 0))
        return reader, asyncio.StreamWriter(transport, protocol, None, loop)

    agent_reader, agent_writer = await streams(c2a_r, a2c_w)
    client_reader, client_writer = await streams(a2c_r, c2a_w)

    async def serve() -> None:
        try:
            await acp.run_agent(agent, agent_writer, agent_reader)
        finally:
            await agent.shutdown()

    server = asyncio.create_task(serve())
    state.conn = acp.connect_to_agent(state.client, client_writer, client_reader)
    try:
        yield state
    finally:
        client_writer.close()  # EOF on the agent's stdin: the clean-exit path
        await asyncio.wait_for(server, timeout=GUARD_S)
        await state.conn.close()


async def _new(state: Rig, **kwargs: Any) -> str:
    await state.conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
    response = await state.conn.new_session(cwd="/work", mcp_servers=[], **kwargs)
    return response.session_id


def _kinds(updates: list[Any]) -> list[str]:
    return [u.session_update for u in updates]


@pytest.mark.asyncio
async def test_initialize_advertises_the_poc_capabilities() -> None:
    async with rig() as state:
        response = await state.conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
    caps = response.agent_capabilities
    assert response.protocol_version == acp.PROTOCOL_VERSION
    assert caps.load_session is False
    assert caps.prompt_capabilities.image is False
    assert caps.prompt_capabilities.audio is False
    assert caps.prompt_capabilities.embedded_context is False
    # D8 (revised): the Gateway refuses a backend without HTTP MCP support.
    assert caps.mcp_capabilities.http is True
    assert caps.mcp_capabilities.sse is False
    assert response.agent_info.name == "local-operator"


@pytest.mark.asyncio
async def test_new_session_id_is_the_lop_session_id() -> None:
    async with rig() as state:
        session_id = await _new(state)
        assert session_id == state.sessions[0].session_id
    assert state.sessions[0].disposed, "stdin EOF disposes every session"


@pytest.mark.asyncio
async def test_prompt_streams_chunks_and_ends_the_turn() -> None:
    async with rig() as state:
        session_id = await _new(state)
        response = await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("hi")])
        assert response.stop_reason == "end_turn"
        assert state.sessions[0].prompts == ["hi"]
        chunks = [u.content.text for u in state.client.updates]
        assert chunks == ["Hello", " there"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "approved", "status"),
    [
        ("allow_once", True, "completed"),
        ("reject_once", False, "failed"),
        ("deny", False, "failed"),
    ],
)
async def test_permission_round_trip(reply: str, approved: bool, status: str) -> None:
    async with rig() as state:
        state.client.replies.append(reply)
        session_id = await _new(state)
        response = await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[bash]")])
        assert response.stop_reason == "end_turn"
        [(tool_call, options)] = state.client.permission_requests
        assert tool_call.tool_call_id == "call-1", "matched to the started call"
        assert tool_call.status == "pending"
        assert [o.option_id for o in options] == ["allow_once", "reject_once"]
        assert state.sessions[0].gate_outcomes == [approved]
        ends = [u for u in state.client.updates if u.session_update == "tool_call_update"]
        assert ends[-1].status == status


@pytest.mark.asyncio
async def test_a_client_error_raises_approval_unavailable() -> None:
    async with rig() as state:
        state.client.replies.append("error")
        session_id = await _new(state)
        await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[bash]")])
        [outcome] = state.sessions[0].gate_outcomes
        assert isinstance(outcome, ApprovalUnavailableError)


@pytest.mark.asyncio
async def test_cancel_mid_prompt_returns_cancelled_and_settles_the_permission() -> None:
    async with rig() as state:
        session_id = await _new(state)
        turn = asyncio.ensure_future(
            state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[hang-bash]")])
        )
        await asyncio.wait_for(state.client.permission_asked.wait(), timeout=GUARD_S)
        await state.conn.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=GUARD_S)
        assert response.stop_reason == "cancelled"
        session = state.sessions[0]
        assert session.gate_outcomes == [False], "the pending permission settled as denied"
        assert session.aborted and session.subagents_cancelled, "turn AND subagents stopped"


@pytest.mark.asyncio
async def test_cancel_without_a_permission_returns_cancelled() -> None:
    async with rig() as state:
        session_id = await _new(state)
        turn = asyncio.ensure_future(
            state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[slow]")])
        )
        while not state.sessions[0].is_streaming:
            await asyncio.sleep(0.01)
        await state.conn.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=GUARD_S)
        assert response.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_a_second_concurrent_prompt_and_an_unknown_id_are_refused() -> None:
    async with rig() as state:
        session_id = await _new(state)
        first = asyncio.ensure_future(
            state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[slow]")])
        )
        while not state.sessions[0].is_streaming:
            await asyncio.sleep(0.01)
        with pytest.raises(acp.RequestError) as second:
            await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("again")])
        assert second.value.code == -32600
        with pytest.raises(acp.RequestError) as unknown:
            await state.conn.prompt(session_id="nope", prompt=[acp.text_block("x")])
        assert unknown.value.code == -32602
        await state.conn.cancel(session_id=session_id)
        await asyncio.wait_for(first, timeout=GUARD_S)


@pytest.mark.asyncio
async def test_a_turn_error_is_a_json_rpc_error() -> None:
    async with rig() as state:
        session_id = await _new(state)
        with pytest.raises(acp.RequestError) as error:
            await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[error]")])
        assert error.value.code == -32603
        assert "provider exploded" in str(error.value)


@pytest.mark.asyncio
async def test_http_mcp_servers_reach_the_opener_and_stdio_ones_are_ignored(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with rig() as state:
        await state.conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        with caplog.at_level("WARNING"):
            await state.conn.new_session(
                cwd="/work",
                mcp_servers=[
                    HttpMcpServer(
                        type="http",
                        name="gateway",
                        url="http://127.0.0.1:1/mcp",
                        headers=[HttpHeader(name="Authorization", value="Bearer t")],
                    ),
                    McpServerStdio(name="local-tool", command="x", args=[], env=[]),
                ],
            )
    [(cwd, servers)] = state.opened
    assert cwd == "/work"
    assert servers == {
        "gateway": {
            "type": "http",
            "url": "http://127.0.0.1:1/mcp",
            "headers": {"Authorization": "Bearer t"},
        }
    }
    assert "local-tool (stdio)" in caplog.text
