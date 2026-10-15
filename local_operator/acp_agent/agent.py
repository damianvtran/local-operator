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
* D11 — one subscriber per session for its whole life: turns the session opens
  by itself stream to the client as plain ``session/update``s.

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

from acp import PROTOCOL_VERSION, RequestError, helpers
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
from local_operator.harness.approval import ApprovalUnavailableError

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


#: How long a permission request may wait for the client before the call is
#: refused (D2). Longer than serving's 30 s card timeout on purpose: the first
#: client is a VOICE gateway, and a spoken "yes, run it" — hearing the question,
#: deciding, answering — routinely takes longer than 30 s. Unanswered is not a
#: denial: it is refused as "nobody answered" (Q5).
PERMISSION_TIMEOUT_S = 120.0

#: How long ``session/new`` waits for the client's MCP servers to connect, so
#: their tools are in the first prompt's tools array (D8). MCP's own request
#: timeout is 30 s; a connect slower than this one is left to finish late.
MCP_SETTLE_TIMEOUT_S = 10.0


#: The reason given when a gate call finds its session already closed.
ACP_CLIENT_GONE = "the ACP client has closed this session"


class AcpApprovalUnavailable(ApprovalUnavailableError):
    """``ApprovalUnavailableError`` worded for an ACP session (Q4).

    The base class renders ``APPROVAL_UNAVAILABLE_NOTICE``, whose remedies are
    ``lop exec`` flags (``--yolo``, ``--tools``, ``--control``) that mean
    nothing to a model or a person on the other end of an ACP client. A
    subclass keeps the type every call site catches, so the call is still
    refused as "nobody could answer" rather than "the user denied it", with
    copy that names the real cause and the real remedy.
    """

    def __str__(self) -> str:
        return (
            f"Approval unavailable for '{self.tool_name}': {self.reason}, so the call was "
            "not run. Ask the user to answer the permission request in their ACP "
            "client, then try again."
        )


@dataclass
class _PromptRun:
    """One ``session/prompt`` in flight: its own cancel token (R1).

    A token per prompt, rather than a flag on the session reset when the prompt
    starts: a ``session/cancel`` handled before the prompt's first step would
    otherwise be wiped by that reset. ``ended`` collects the turn's
    ``agent_end``.
    """

    cancelled: bool = False
    turn_live: bool = False
    ended: list[Any] = field(default_factory=list)


@dataclass
class _AcpSession:
    """Everything this agent holds for one ACP session id."""

    session: Any
    mapper: EventMapper
    host: asyncio.Task[None]
    close: asyncio.Event
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    pending: set[asyncio.Future[Any]] = field(default_factory=set)
    #: The prompt in flight, or ``None`` between prompts. Events that arrive
    #: while it is ``None`` belong to a turn the SESSION opened (D11).
    run: _PromptRun | None = None
    #: In-order delivery of this session's updates (see ``_on_event``).
    sends: list[asyncio.Task[None]] = field(default_factory=list)
    unsubscribe: Callable[[], None] | None = None


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
        permission_timeout_s: float = PERMISSION_TIMEOUT_S,
        mcp_settle_timeout_s: float = MCP_SETTLE_TIMEOUT_S,
    ) -> None:
        self._open_session = open_session
        self._permission_timeout_s = permission_timeout_s
        self._mcp_settle_timeout_s = mcp_settle_timeout_s
        self._conn: Any = None
        self._sessions: dict[str, _AcpSession] = {}
        self._closed = False

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
        # Validated before anything is logged or built: a bad cwd is the
        # client's error (-32602), not a failure inside session construction.
        if not Path(cwd).is_absolute():
            raise RequestError.invalid_params({"cwd": "must be an absolute path"})
        if not Path(cwd).is_dir():
            raise RequestError.invalid_params({"cwd": f"not an existing directory: {cwd}"})
        servers, ignored = mcp_server_configs(mcp_servers or [])
        if ignored:
            logger.warning("acp: ignoring unsupported MCP servers: %s", ", ".join(ignored))
        if servers:
            logger.info("acp: session MCP servers: %s", ", ".join(sorted(servers)))

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
        if servers:
            await self._await_client_servers(session)
        entry = _AcpSession(session=session, mapper=EventMapper(), host=task, close=close)
        # D11: ONE subscription for the session's whole life, not one per
        # prompt. A session opens turns by itself — a background job or a
        # subagent settling, a wake, a monitor — and those turns' text, tool
        # calls and permission requests must reach the client too. Released in
        # ``_close_entry``, so it cannot outlive the connection or the session.
        entry.unsubscribe = session.subscribe(lambda event: self._on_event(session_id, event))
        self._sessions[session_id] = entry
        return NewSessionResponse(session_id=session_id)

    async def _await_client_servers(self, session: Any) -> None:
        """Let the client's MCP servers connect before ``session/new`` answers.

        The session opens past MCP's 250 ms startup gate, and an HTTP connect
        routinely outlasts it. Preloaded tools (D8) that land after the gate
        reach the tools array only from the NEXT turn, because the array is
        published once per turn — so the client's first prompt, typically the
        one that needs its coordination tools, would not see them. Bounded by
        :data:`MCP_SETTLE_TIMEOUT_S`; a slower server keeps connecting and its
        tools arrive for a later turn, never failing the session.
        """
        manager = getattr(session, "mcp_manager", None)
        if manager is None:
            return
        try:
            settled = await manager.wait_settled(self._mcp_settle_timeout_s)
        except Exception:  # noqa: BLE001 — enrichment: never a reason to fail session/new
            logger.debug("acp: waiting for client MCP servers failed", exc_info=True)
            return
        if not settled:
            logger.warning(
                "acp: client MCP servers still connecting after %.0f s; their tools "
                "will reach a later turn",
                self._mcp_settle_timeout_s,
            )

    async def shutdown(self) -> None:
        """Dispose every session (stdin EOF): settle waits, end each host task."""
        self._closed = True
        entries = list(self._sessions.values())
        self._sessions.clear()
        for entry in entries:
            self._close_entry(entry)
            if entry.session.is_streaming:
                entry.session.abort("acp client disconnected")
            entry.close.set()
        for entry in entries:
            try:
                await asyncio.wait_for(entry.host, timeout=30)
            except Exception:  # noqa: BLE001 — teardown must reach every session
                logger.warning("acp: session dispose failed", exc_info=True)

    def _close_entry(self, entry: _AcpSession) -> None:
        if entry.unsubscribe is not None:
            entry.unsubscribe()
            entry.unsubscribe = None
        self._settle_pending(entry)

    # --- events -----------------------------------------------------------------

    def _on_event(self, session_id: str, event: Any) -> None:
        """The session's one subscriber: map, then queue the updates in order.

        Synchronous by contract (the session awaits async handlers on the turn
        path, and a slow client must not stall the turn). Each update is sent
        on its own task, chained to the previous one by ``_send_after`` so the
        client receives them in emission order; a prompt drains the chain
        before it answers, so no update of a turn trails its stop reason.
        """
        entry = self._sessions.get(session_id)
        if entry is None or self._closed:
            return
        run = entry.run
        if run is not None and not run.turn_live and entry.session.is_streaming:
            # The first event once the prompt's turn is live. ``Session.prompt``
            # clears its sticky abort flag when it admits the prompt, so an
            # abort issued before this point is lost; from here on the turn's
            # signal exists and an abort reaches it (R1). Keyed on
            # ``is_streaming`` rather than on "any first event", because a
            # notice can be emitted before the prompt is admitted.
            run.turn_live = True
            if run.cancelled:
                entry.session.abort("cancelled by the ACP client")
        if getattr(event, "type", None) == "agent_end":
            if run is not None:
                run.ended.append(event)
            return
        for update in entry.mapper.updates_for(event):
            self._queue_update(entry, session_id, update)

    def _queue_update(self, entry: _AcpSession, session_id: str, update: Any) -> None:
        previous = entry.sends[-1] if entry.sends else None
        task = asyncio.ensure_future(self._send_after(previous, session_id, update))
        entry.sends.append(task)
        task.add_done_callback(lambda done: _discard(entry.sends, done))

    async def _send_after(
        self, previous: asyncio.Task[None] | None, session_id: str, update: Any
    ) -> None:
        if previous is not None:
            await asyncio.gather(previous, return_exceptions=True)
        await self._send_update(session_id, update)

    async def _drain(self, entry: _AcpSession) -> None:
        if entry.sends:
            await asyncio.gather(*list(entry.sends), return_exceptions=True)

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
        if entry.session.is_streaming:
            # A turn the session opened by itself (D11) is running; a prompt
            # now would queue behind it and adopt the tail of its stream.
            raise RequestError.invalid_request(
                {
                    "details": "this session is busy with a turn it started itself "
                    "(a background job or subagent reported back); wait for its "
                    "updates to end, or send session/cancel, then prompt again"
                }
            )
        async with entry.prompt_lock:
            run = _PromptRun()
            entry.run = run
            text = _prompt_text(prompt)
            try:
                await entry.session.prompt(text)
            except Exception as exc:
                raise RequestError.internal_error({"details": str(exc)}) from exc
            finally:
                entry.run = None
                await self._drain(entry)
            if run.cancelled:
                return PromptResponse(stop_reason="cancelled")
            if not run.ended:
                return PromptResponse(stop_reason="end_turn")
            try:
                return PromptResponse(stop_reason=stop_reason_for(run.ended[-1]))
            except TurnError as exc:
                raise RequestError(-32603, str(exc)) from exc

    async def _send_update(self, session_id: str, update: Any) -> None:
        if self._closed:
            return
        try:
            await self._conn.session_update(session_id=session_id, update=update)
        except Exception:  # noqa: BLE001 — a lost update must not fail the turn
            logger.debug("acp: session/update send failed", exc_info=True)

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        entry = self._sessions.get(session_id)
        if entry is None:
            return
        if entry.run is not None:
            # Recorded on THIS prompt's token, so a cancel that lands before
            # the session's turn starts survives until ``_on_event`` sees the
            # turn go live and re-issues the abort (R1).
            entry.run.cancelled = True
        self._settle_pending(entry)
        # ``abort`` stops only this session's turn; ACP asks to stop ALL work,
        # so the subagents it started are cancelled too (D7). Between prompts
        # this stops a turn the session opened by itself (D11).
        entry.session.abort("cancelled by the ACP client")
        entry.session.cancel_subagents("cancelled by the ACP client")

    def _settle_pending(self, entry: _AcpSession) -> None:
        for waiter in list(entry.pending):
            waiter.cancel()

    # --- approvals --------------------------------------------------------------

    async def _request_permission(
        self, session_id: str, tool_name: str, description: str, job_id: str | None
    ) -> bool:
        entry = self._sessions.get(session_id)
        if entry is None:
            raise AcpApprovalUnavailable(tool_name, ACP_CLIENT_GONE)
        if entry.run is not None and entry.run.cancelled:
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
            # permission request refers to something the client can render —
            # and closed after the answer (R5), since no tool_execution_end of
            # this session will ever name it.
            self._queue_update(entry, session_id, _start_from(tool_call))
        # The request must not overtake the ``tool_call`` it refers to, which
        # is still on the update chain.
        await self._drain(entry)
        approved = False
        try:
            approved = await self._ask_client(entry, session_id, tool_name, tool_call)
        finally:
            if not known:
                self._queue_update(
                    entry,
                    session_id,
                    helpers.update_tool_call(call_id, status="completed" if approved else "failed"),
                )
        return approved

    async def _ask_client(
        self, entry: _AcpSession, session_id: str, tool_name: str, tool_call: ToolCallUpdate
    ) -> bool:
        waiter = asyncio.ensure_future(
            self._conn.request_permission(
                session_id=session_id, tool_call=tool_call, options=PERMISSION_OPTIONS
            )
        )
        entry.pending.add(waiter)
        try:
            response = await asyncio.wait_for(
                asyncio.shield(waiter), timeout=self._permission_timeout_s
            )
        except asyncio.CancelledError:
            if waiter.cancelled():
                # Settled by session/cancel (D7): the user stopped the turn.
                return False
            waiter.cancel()
            raise
        except asyncio.TimeoutError:
            waiter.cancel()
            logger.warning(
                "acp: permission request for %s unanswered after %.0f s; refused",
                tool_name,
                self._permission_timeout_s,
            )
            # Unanswered is not "the user denied it" (Q5): nobody answered.
            raise AcpApprovalUnavailable(
                tool_name,
                f"the permission request got no answer within "
                f"{self._permission_timeout_s:.0f} s",
            ) from None
        except Exception as exc:
            raise AcpApprovalUnavailable(
                tool_name, f"the ACP client could not answer the permission request ({exc})"
            ) from exc
        finally:
            entry.pending.discard(waiter)
        outcome = response.outcome
        return isinstance(outcome, AllowedOutcome) and outcome.option_id == ALLOW_ONCE


def _discard(tasks: list[asyncio.Task[None]], done: asyncio.Task[None]) -> None:
    try:
        tasks.remove(done)
    except ValueError:
        pass


def _start_from(tool_call: ToolCallUpdate) -> Any:
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
