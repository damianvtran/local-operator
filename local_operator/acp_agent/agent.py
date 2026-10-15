"""``LopAcpAgent``: the ACP method handlers, over in-process lop sessions.

One ACP session is one ``sdk.open_session(...)`` (D1). The design decisions are
numbered in ``docs/design/acp-agent.md``; the ones that shape this file:

* D2 — approvals go to the client as ``session/request_permission`` with
  ``allow_once``/``reject_once`` only; a failed or closed connection raises
  ``ApprovalUnavailableError`` (nobody could answer) rather than a denial.
* D6 — the ACP ``sessionId`` IS the lop session id; one prompt at a time.
* D7 — ``session/cancel`` aborts the turn AND its subagents, and settles any
  pending permission request as denied.
* D8 (revised) — HTTP ``mcpServers`` from ``session/new`` are wired into that
  session only; others are logged and ignored.
* D9 — ``stopReason`` from the turn's ``agent_end``.

WHY EACH SESSION HAS A HOST TASK. ``open_session`` is an async context manager,
and the MCP transports it builds hold anyio cancel scopes that must be exited
in the task that entered them. A JSON-RPC request is its own task, so entering
the context in ``session/new`` and leaving it on shutdown would cross tasks.
Each session therefore lives inside one long-lived task that opens it, signals
readiness, waits for the close signal and disposes it — in the same task.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, RequestError
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    Implementation,
    InitializeResponse,
    McpCapabilities,
    NewSessionResponse,
    PermissionOption,
    PromptCapabilities,
    PromptResponse,
    ToolCallUpdate,
)

from local_operator.acp_agent.mapping import (
    EventMapper,
    TurnError,
    mcp_server_configs,
    stop_reason_for,
    tool_kind,
)

logger = logging.getLogger(__name__)

#: The option ids offered on every permission request (D2). ``allow_always``
#: is never offered: it would flip the session to auto-approve, which is the
#: authority-increasing transition docs/design/approval-authority.md §1 names.
ALLOW_ONCE = "allow_once"
REJECT_ONCE = "reject_once"
PERMISSION_OPTIONS = [
    PermissionOption(option_id=ALLOW_ONCE, name="Allow once", kind="allow_once"),
    PermissionOption(option_id=REJECT_ONCE, name="Reject", kind="reject_once"),
]

#: Opens one lop session for ``(cwd, approval gate, mcp servers)``; injected so
#: the pipe tests can substitute a fake owning session for the real SDK.
SessionOpener = Callable[
    [str, Callable[..., Any], dict[str, dict[str, Any]]], AbstractAsyncContextManager[Any]
]


def _permission_timeout_s() -> float:
    """How long a permission request may wait: serving's card timeout (D2)."""
    from local_operator.session.runtime.serving import PENDING_REQUEST_TIMEOUT_S

    return PENDING_REQUEST_TIMEOUT_S


@dataclass
class _AcpSession:
    """Everything this agent holds for one ACP session id."""

    session: Any
    mapper: EventMapper
    host: asyncio.Task[None]
    close: asyncio.Event
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: set[asyncio.Future[Any]] = field(default_factory=set)
    cancelled: bool = False


class LopAcpAgent:
    """The ACP ``Agent``: ``initialize``, ``session/new``, ``prompt``, ``cancel``.

    Structurally an ``acp.Agent`` (the SDK dispatches by method name); methods
    the POC does not offer — ``session/load``, modes, config options — are
    simply absent, which the SDK answers with "method not found".
    """

    def __init__(
        self,
        open_session: SessionOpener,
        *,
        permission_timeout_s: float | None = None,
    ) -> None:
        self._open_session = open_session
        self._permission_timeout_s = permission_timeout_s
        self._conn: Any = None
        self._sessions: dict[str, _AcpSession] = {}

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

    # --- lifecycle ------------------------------------------------------------

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        from local_operator.update import installed_version

        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=False,
                prompt_capabilities=PromptCapabilities(
                    image=False, audio=False, embedded_context=False
                ),
                # D8 (revised): the Qwen Gateway refuses a backend that does not
                # declare HTTP MCP support, because its coordination tools ride
                # an HTTP MCP server handed over in session/new.
                mcp_capabilities=McpCapabilities(http=True, sse=False),
            ),
            agent_info=Implementation(
                name="local-operator", title="Local Operator", version=installed_version()
            ),
            auth_methods=[],
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        servers, ignored = mcp_server_configs(mcp_servers or [])
        if ignored:
            logger.warning("acp: ignoring unsupported MCP servers: %s", ", ".join(ignored))
        if servers:
            logger.info("acp: session MCP servers: %s", ", ".join(sorted(servers)))
        if not Path(cwd).is_absolute():
            raise RequestError.invalid_params({"cwd": "must be an absolute path"})

        # The gate needs the session id, which exists only once the session is
        # built; the cell is filled before any turn can reach the gate.
        cell: dict[str, str] = {}

        async def gate(tool_name: str, description: str, job_id: str | None = None) -> bool:
            return await self._request_permission(cell["id"], tool_name, description, job_id)

        ready: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        close = asyncio.Event()

        async def host() -> None:
            try:
                async with self._open_session(cwd, gate, servers) as session:
                    ready.set_result(session)
                    await close.wait()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                    return
                raise

        task = asyncio.create_task(host(), name="acp-session-host")
        try:
            session = await ready
        except Exception as exc:
            logger.warning("acp: session/new failed", exc_info=True)
            raise RequestError.internal_error({"details": str(exc)}) from exc
        session_id = str(session.session_id)
        cell["id"] = session_id
        self._sessions[session_id] = _AcpSession(
            session=session, mapper=EventMapper(), host=task, close=close
        )
        return NewSessionResponse(session_id=session_id)

    async def shutdown(self) -> None:
        """Dispose every session (stdin EOF): settle waits, end each host task."""
        entries = list(self._sessions.values())
        self._sessions.clear()
        for entry in entries:
            self._settle_pending(entry)
            if entry.session.is_streaming:
                entry.session.abort("acp client disconnected")
            entry.close.set()
        for entry in entries:
            try:
                await asyncio.wait_for(entry.host, timeout=30)
            except Exception:  # noqa: BLE001 — teardown must reach every session
                logger.warning("acp: session dispose failed", exc_info=True)

    # --- turns ------------------------------------------------------------------

    def _entry(self, session_id: str) -> _AcpSession:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise RequestError.invalid_params({"sessionId": f"unknown session {session_id!r}"})
        return entry

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        entry = self._entry(session_id)
        if entry.prompt_lock.locked():
            raise RequestError.invalid_request(
                {"details": "a prompt is already running on this session"}
            )
        async with entry.prompt_lock:
            entry.cancelled = False
            text = _prompt_text(prompt)
            ended: list[Any] = []
            sends: list[asyncio.Task[None]] = []

            def on_event(event: Any) -> None:
                # Synchronous by contract (the session's dispatcher awaits
                # async handlers on the turn path, and a slow client must not
                # stall the turn); each update is queued in order on its own
                # task, and the prompt drains them before answering.
                if getattr(event, "type", None) == "agent_end":
                    ended.append(event)
                    return
                for update in entry.mapper.updates_for(event):
                    sends.append(asyncio.ensure_future(self._send_update(session_id, update)))

            unsubscribe = entry.session.subscribe(on_event)
            try:
                await entry.session.prompt(text)
            except Exception as exc:
                raise RequestError.internal_error({"details": str(exc)}) from exc
            finally:
                unsubscribe()
                if sends:
                    await asyncio.gather(*sends, return_exceptions=True)
            if entry.cancelled:
                return PromptResponse(stop_reason="cancelled")
            if not ended:
                return PromptResponse(stop_reason="end_turn")
            try:
                return PromptResponse(stop_reason=stop_reason_for(ended[-1]))
            except TurnError as exc:
                raise RequestError(-32603, str(exc)) from exc

    async def _send_update(self, session_id: str, update: Any) -> None:
        try:
            await self._conn.session_update(session_id=session_id, update=update)
        except Exception:  # noqa: BLE001 — a lost update must not fail the turn
            logger.debug("acp: session/update send failed", exc_info=True)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        entry = self._sessions.get(session_id)
        if entry is None:
            return
        entry.cancelled = True
        self._settle_pending(entry)
        # ``abort`` stops only this session's turn; ACP asks to stop ALL work,
        # so the subagents it started are cancelled too (D7).
        entry.session.abort("cancelled by the ACP client")
        entry.session.cancel_subagents("cancelled by the ACP client")

    def _settle_pending(self, entry: _AcpSession) -> None:
        for waiter in list(entry.pending):
            waiter.cancel()

    # --- approvals --------------------------------------------------------------

    async def _request_permission(
        self, session_id: str, tool_name: str, description: str, job_id: str | None
    ) -> bool:
        from local_operator.harness.approval import ApprovalUnavailableError

        entry = self._sessions.get(session_id)
        if entry is None or entry.cancelled:
            return False
        call_id, known = entry.mapper.permission_call_id(tool_name, job_id)
        tool_call = ToolCallUpdate(
            tool_call_id=call_id,
            title=description.splitlines()[0][:200] if description else tool_name,
            kind=tool_kind(tool_name),
            status="pending",
        )
        if not known:
            # A call the client has not seen on the stream (a subagent's, or a
            # name the mapper could not match) is announced first, so the
            # permission request refers to something the client can render.
            await self._send_update(
                session_id,
                _start_from(tool_call),
            )
        waiter = asyncio.ensure_future(
            self._conn.request_permission(
                session_id=session_id, tool_call=tool_call, options=PERMISSION_OPTIONS
            )
        )
        entry.pending.add(waiter)
        timeout = self._permission_timeout_s
        if timeout is None:
            timeout = _permission_timeout_s()
        try:
            response = await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout)
        except asyncio.CancelledError:
            if waiter.cancelled():
                # Settled by session/cancel (D7): the user stopped the turn.
                return False
            waiter.cancel()
            raise
        except asyncio.TimeoutError:
            waiter.cancel()
            logger.warning("acp: permission request for %s timed out; denied", tool_name)
            return False
        except Exception as exc:
            raise ApprovalUnavailableError(
                tool_name, f"the ACP client could not answer the permission request ({exc})"
            ) from exc
        finally:
            entry.pending.discard(waiter)
        outcome = response.outcome
        return isinstance(outcome, AllowedOutcome) and outcome.option_id == ALLOW_ONCE


def _start_from(tool_call: ToolCallUpdate) -> Any:
    from acp import helpers

    return helpers.start_tool_call(
        tool_call.tool_call_id,
        tool_call.title or "tool",
        kind=tool_call.kind,
        status="pending",
    )


def _prompt_text(blocks: list[Any]) -> str:
    """The prompt as one text: text blocks verbatim, resource links by URI."""
    parts: list[str] = []
    for block in blocks:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(block.text)
        elif kind == "resource_link":
            parts.append(f"@{block.uri}" if block.uri.startswith("/") else block.uri)
    return "\n".join(parts)
