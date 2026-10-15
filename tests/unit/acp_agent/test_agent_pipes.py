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
from typing import Any, cast

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

from local_operator.acp_agent.agent import (  # noqa: E402
    MCP_SETTLE_TIMEOUT_S,
    LopAcpAgent,
)
from local_operator.harness.approval import ApprovalUnavailableError  # noqa: E402
from local_operator.harness.types import (  # noqa: E402
    AgentEndEvent,
    AgentStartEvent,
    Message,
    MessageUpdateEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolResult,
)

GUARD_S = 20.0

#: A directory that exists on every host, for session/new's cwd check.
WORK = os.path.realpath(os.path.dirname(__file__))


class FakeSession:
    """The owning-session surface ``LopAcpAgent`` uses, driven by a script.

    ``prompt`` models the REAL ``Session.prompt`` admission order, because the
    cancel race (review R1) lives in it: ``@`` references are expanded first
    (awaited — one yield per ``@`` token here, to model the file reads), then
    the turn lock is taken and the sticky abort flag is CLEARED ("a fresh user
    prompt supersedes any earlier interrupt request"), and only then is the
    turn live with a signal an abort can reach. An abort before that point
    sets a flag the admission wipes, exactly like the real session.
    """

    def __init__(self, session_id: str, gate: Callable[..., Any]) -> None:
        self.session_id = session_id
        self._gate = gate
        self._handlers: list[Callable[[Any], None]] = []
        self.is_streaming = False
        self.aborted: list[str] = []
        self.subagents_cancelled: list[str] = []
        self.prompts: list[str] = []
        self.gate_outcomes: list[Any] = []
        self._abort_requested = False
        self._signal: asyncio.Event | None = None
        self.disposed = False
        self.mcp_manager: Any = None
        self.admission_yields = 0
        #: When set, admission waits for it: the window between the prompt
        #: starting and its turn going live, held open so a test can land a
        #: cancel in it deterministically (the real window is the ``@``
        #: expansion's file reads and the turn-lock wait).
        self.admission: asyncio.Event | None = None

    def subscribe(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    @property
    def subscribers(self) -> int:
        return len(self._handlers)

    def _emit(self, event: Any) -> None:
        for handler in list(self._handlers):
            handler(event)

    def abort(self, reason: str = "interrupted") -> None:
        self.aborted.append(reason)
        self._abort_requested = True
        if self._signal is not None:
            self._signal.set()

    def cancel_subagents(self, reason: str = "interrupted") -> int:
        self.subagents_cancelled.append(reason)
        return 0

    async def prompt(self, text: str) -> None:
        self.prompts.append(text)
        for _ in range(text.count("@") + self.admission_yields):
            await asyncio.sleep(0)  # reference expansion: real awaits
        if self.admission is not None:
            await self.admission.wait()
        self._abort_requested = False  # the real admission clears the flag
        await self._run_turn(text)

    async def _run_turn(self, text: str) -> None:
        self.is_streaming = True
        self._signal = asyncio.Event()
        if self._abort_requested:
            self._signal.set()
        self._emit(AgentStartEvent())
        try:
            await self._turn(text, self._signal)
        finally:
            self.is_streaming = False
            self._signal = None

    def background_turn(self, text: str) -> asyncio.Task[None]:
        """A turn the SESSION opens (a job or subagent reporting back): no prompt."""
        return asyncio.ensure_future(self._run_turn(text))

    async def _turn(self, text: str, signal: asyncio.Event) -> None:
        message = Message(role="assistant", content=[TextContent(text="")], id="m1")
        if "[hang]" in text:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(signal.wait(), timeout=GUARD_S)
            self._emit(AgentEndEvent(aborted=signal.is_set()))
            return
        if "[bash]" in text or "[hang-bash]" in text or "[sub-bash]" in text:
            call_id = "call-1"
            if "[sub-bash]" not in text:
                self._emit(ToolExecutionStartEvent(tool_call_id=call_id, tool_name="bash"))
            job_id = "job-1" if "[sub-bash]" in text else None
            try:
                approved = await self._gate("bash", "Run: ls", job_id)
            except ApprovalUnavailableError as exc:
                self.gate_outcomes.append(exc)
                approved = False
            else:
                self.gate_outcomes.append(approved)
            if "[sub-bash]" not in text:
                self._emit(
                    ToolExecutionEndEvent(
                        tool_call_id=call_id,
                        tool_name="bash",
                        result=ToolResult(
                            tool_call_id=call_id,
                            content=[TextContent(text="ran" if approved else "denied")],
                            is_error=not approved,
                        ),
                    )
                )
            if signal.is_set():
                self._emit(AgentEndEvent(aborted=True))
                return
        if "[slow]" in text:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(signal.wait(), timeout=GUARD_S)
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
        #: Installed as each new session's ``mcp_manager`` (None: no MCP).
        self.manager: Any = None


@asynccontextmanager
async def rig(*, permission_timeout_s: float = GUARD_S) -> AsyncIterator[Rig]:
    state = Rig()

    @asynccontextmanager
    async def opener(
        cwd: str, gate: Callable[..., Any], servers: dict[str, Any]
    ) -> AsyncIterator[Any]:
        state.opened.append((cwd, servers))
        session = FakeSession(f"sess{len(state.sessions):08x}", gate)
        session.mcp_manager = state.manager
        state.sessions.append(session)
        try:
            yield session
        finally:
            session.disposed = True

    agent = LopAcpAgent(opener, permission_timeout_s=permission_timeout_s)
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
    response = await state.conn.new_session(cwd=WORK, mcp_servers=[], **kwargs)
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
        # Q4: the copy names the ACP client, not exec's remedies.
        assert "ACP client" in str(outcome)
        assert "--yolo" not in str(outcome) and "--tools" not in str(outcome)


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
                cwd=WORK,
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
    assert cwd == WORK
    assert servers == {
        "gateway": {
            "type": "http",
            "url": "http://127.0.0.1:1/mcp",
            "headers": {"Authorization": "Bearer t"},
            "preloadTools": True,
        }
    }
    assert "local-tool (stdio)" in caplog.text


# --- R1: a cancel that lands before the turn is live -------------------------


async def _wait_for(predicate: Callable[[], bool]) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=GUARD_S)


@pytest.mark.asyncio
@pytest.mark.parametrize("yields", [0, 1, 2, 3, 5, 8, 20])
@pytest.mark.parametrize(
    "blocks",
    [
        pytest.param(lambda: [acp.text_block("[hang]")], id="text"),
        pytest.param(
            lambda: [
                acp.text_block("[hang] look at"),
                acp.resource_link_block(name="notes", uri="/work/notes.md"),
            ],
            id="resource_link",
        ),
    ],
)
async def test_a_cancel_before_the_turn_is_live_still_cancels_it(
    yields: int, blocks: Callable[[], list[Any]]
) -> None:
    """The reviewer's repro: the cancel is handled while the prompt is still
    being admitted, then admission completes N loop yields later.

    ``FakeSession`` clears its abort flag on admission, as the real one does,
    so the abort the cancel issued is wiped; the prompt's own token must
    survive and re-abort once the turn is live. Without that, every case here
    hangs to the guard (checked by disabling the re-abort: all 14 fail).
    """
    async with rig() as state:
        session_id = await _new(state)
        session = state.sessions[0]
        session.admission = asyncio.Event()
        turn = asyncio.ensure_future(state.conn.prompt(session_id=session_id, prompt=blocks()))
        await _wait_for(lambda: bool(session.prompts))
        await state.conn.cancel(session_id=session_id)
        await _wait_for(lambda: bool(session.aborted))  # handled pre-admission
        for _ in range(yields):
            await asyncio.sleep(0)
        session.admission.set()
        response = await asyncio.wait_for(turn, timeout=GUARD_S)
        assert response.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_a_cancel_sent_before_the_agent_even_starts_the_prompt() -> None:
    """Prompt and cancel written back to back: the agent may see both before
    the prompt task's first step. The prompt's own token must survive."""
    async with rig() as state:
        session_id = await _new(state)
        state.sessions[0].admission_yields = 3
        turn = asyncio.ensure_future(
            state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[hang]")])
        )
        await asyncio.sleep(0)
        await state.conn.cancel(session_id=session_id)
        response = await asyncio.wait_for(turn, timeout=GUARD_S)
        assert response.stop_reason == "cancelled"


# --- D11: turns the session opens by itself ----------------------------------


@pytest.mark.asyncio
async def test_a_background_opened_turn_streams_to_the_client() -> None:
    async with rig() as state:
        session_id = await _new(state)
        session = state.sessions[0]
        await asyncio.wait_for(session.background_turn("a job reported back"), GUARD_S)
        await _wait_for(lambda: len(state.client.updates) >= 2)
        assert [u.content.text for u in state.client.updates] == ["Hello", " there"]
        # The next prompt is unaffected and owns only its own updates.
        before = len(state.client.updates)
        response = await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("hi")])
        assert response.stop_reason == "end_turn"
        assert len(state.client.updates) == before + 2


@pytest.mark.asyncio
async def test_a_prompt_while_a_background_turn_runs_is_refused() -> None:
    async with rig() as state:
        session_id = await _new(state)
        background = state.sessions[0].background_turn("[slow]")
        await _wait_for(lambda: state.sessions[0].is_streaming)
        with pytest.raises(acp.RequestError) as refused:
            await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("hi")])
        assert refused.value.code == -32600
        assert "started itself" in str(refused.value.data)
        await state.conn.cancel(session_id=session_id)
        await asyncio.wait_for(background, GUARD_S)


@pytest.mark.asyncio
async def test_a_cancel_during_a_background_turn_aborts_it() -> None:
    async with rig() as state:
        session_id = await _new(state)
        session = state.sessions[0]
        background = session.background_turn("[slow]")
        await _wait_for(lambda: session.is_streaming)
        await state.conn.cancel(session_id=session_id)
        await asyncio.wait_for(background, GUARD_S)
        assert session.aborted and session.subagents_cancelled


@pytest.mark.asyncio
async def test_a_background_gate_asks_on_a_synthetic_id_and_closes_it() -> None:
    """R5 + D11: a permission with no prompt in flight, for a call this
    session's stream never announced, is opened and then closed."""
    async with rig() as state:
        state.client.replies.append("allow_once")
        await _new(state)
        await asyncio.wait_for(state.sessions[0].background_turn("[sub-bash]"), GUARD_S)
        [(tool_call, _options)] = state.client.permission_requests
        assert tool_call.tool_call_id.startswith("perm-")
        await _wait_for(
            lambda: any(
                u.session_update == "tool_call_update" and u.status is not None
                for u in state.client.updates
            )
        )
        ids = [
            (u.session_update, u.status)
            for u in state.client.updates
            if getattr(u, "tool_call_id", None) == tool_call.tool_call_id
        ]
        assert ids == [("tool_call", "pending"), ("tool_call_update", "completed")]


@pytest.mark.asyncio
async def test_the_subscriber_is_released_on_eof() -> None:
    async with rig() as state:
        await _new(state)
        session = state.sessions[0]
        assert session.subscribers == 1, "one subscriber per session, not per prompt"
        await state.conn.prompt(session_id=session.session_id, prompt=[acp.text_block("hi")])
        assert session.subscribers == 1
    assert session.subscribers == 0 and session.disposed


@pytest.mark.asyncio
async def test_a_missing_cwd_is_invalid_params() -> None:
    async with rig() as state:
        await state.conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        with pytest.raises(acp.RequestError) as error:
            await state.conn.new_session(cwd="/definitely/not/here-0000", mcp_servers=[])
        assert error.value.code == -32602
    assert state.opened == []


# --- Q4 / Q5: approval copy --------------------------------------------------


@pytest.mark.asyncio
async def test_an_unanswered_permission_is_unavailable_not_denied() -> None:
    async with rig(permission_timeout_s=0.2) as state:
        state.client.replies.append("hang")
        session_id = await _new(state)
        await state.conn.prompt(session_id=session_id, prompt=[acp.text_block("[bash]")])
        [outcome] = state.sessions[0].gate_outcomes
        assert isinstance(outcome, ApprovalUnavailableError)
        text = str(outcome)
        assert "no answer within" in text and "denied" not in text.lower()
        assert "--yolo" not in text and "--control" not in text


def test_the_permission_timeout_fits_a_spoken_answer() -> None:
    from local_operator.acp_agent.agent import PERMISSION_TIMEOUT_S

    assert PERMISSION_TIMEOUT_S == 120.0
    agent = LopAcpAgent(cast(Any, None))
    assert agent._permission_timeout_s == PERMISSION_TIMEOUT_S


class _SettlingManager:
    """The one ``McpManager`` method ``session/new`` waits on."""

    def __init__(self, settles: bool) -> None:
        self.settles = settles
        self.waited: list[float] = []

    async def wait_settled(self, timeout_s: float) -> bool:
        self.waited.append(timeout_s)
        return self.settles


@pytest.mark.asyncio
@pytest.mark.parametrize("with_servers", [True, False])
async def test_session_new_waits_for_the_clients_mcp_servers(with_servers: bool) -> None:
    """Q1, the timing half: a preloaded tool that lands after the 250 ms gate
    reaches the tools array only from the next turn, so ``session/new`` waits
    (bounded) for the client's servers — and only when there are some."""
    manager = _SettlingManager(settles=True)
    async with rig() as state:
        state.manager = manager
        await state.conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        servers = (
            [HttpMcpServer(type="http", name="gw", url="http://127.0.0.1:1/mcp", headers=[])]
            if with_servers
            else []
        )
        await state.conn.new_session(cwd=WORK, mcp_servers=servers)
    assert manager.waited == ([MCP_SETTLE_TIMEOUT_S] if with_servers else [])


@pytest.mark.asyncio
async def test_a_slow_client_server_does_not_fail_session_new(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with rig() as state:
        state.manager = _SettlingManager(settles=False)
        await state.conn.initialize(protocol_version=acp.PROTOCOL_VERSION)
        with caplog.at_level("WARNING"):
            response = await state.conn.new_session(
                cwd=WORK,
                mcp_servers=[
                    HttpMcpServer(type="http", name="gw", url="http://127.0.0.1:1/mcp", headers=[])
                ],
            )
        assert response.session_id
    assert "still connecting" in caplog.text
