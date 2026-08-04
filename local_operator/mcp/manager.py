"""MCP connection manager: fast-startup gate, deferred tools, reconnect breaker.

Ports the omp manager semantics (``src/mcp/manager.ts``) onto the official
``mcp`` Python SDK:

- A 250 ms startup gate (``asyncio.wait(timeout=0.25)``): servers that finish
  the gate contribute live tools; servers still pending WITH a tool-cache hit
  contribute deferred tools whose execute awaits the connection first;
  pending servers without a cache hit contribute nothing until a background
  continuation swaps them in via the ``on_tools_changed`` callback.
- Reconnect on transport close with backoff ``[0.5, 1, 2, 4]`` s and a
  circuit breaker: more than 5 attempts in a sliding 30 s window suspends
  auto-reconnect (manual reconnect resets the history). An epoch counter
  incremented on ``disconnect_all`` prevents a late reconnect from
  resurrecting a dead connection.
- Server-initiated ``notifications/tools/list_changed`` refreshes that
  server's tools and fires the tools-changed callback.

SDK imports are lazy where feasible so config-only callers never pay for the
transport machinery.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from local_operator.harness.types import AgentTool, TextContent, ToolResult
from local_operator.mcp.config import (
    MCPHttpServerConfig,
    MCPServerConfig,
    MCPSseServerConfig,
    MCPStdioServerConfig,
    load_all_mcp_configs,
    validate_server_config,
)
from local_operator.mcp.tool_bridge import (
    build_agent_tool,
    create_mcp_tool_name,
    format_mcp_result,
    is_retriable_connection_error,
    prepare_outbound_args,
)
from local_operator.mcp.tool_cache import McpToolCache

logger = logging.getLogger(__name__)

# Fast-startup gate: how long discovery blocks before deferring slow servers.
STARTUP_GATE_MS = 250

# Reconnect policy (omp): escalating backoff, sliding-window circuit breaker.
RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 4.0)
RECONNECT_BURST_WINDOW_S = 30.0
RECONNECT_BURST_LIMIT = 5

# Default per-request timeout; ``LOCAL_OPERATOR_MCP_TIMEOUT_MS`` overrides,
# config ``timeout`` (ms) refines, ``0`` disables.
DEFAULT_MCP_TIMEOUT_MS = 30_000.0


class McpConnectionError(RuntimeError):
    """Raised when a server cannot be reached (deferred execute path)."""


def resolve_mcp_timeout_s(cfg: MCPServerConfig | None) -> float | None:
    """Resolve the client-side request timeout in seconds (``None`` = off).

    Precedence: ``LOCAL_OPERATOR_MCP_TIMEOUT_MS`` env > ``config.timeout`` >
    30 s default; ``0`` disables the timeout entirely (omp).
    """
    env_raw = os.environ.get("LOCAL_OPERATOR_MCP_TIMEOUT_MS")
    if env_raw is not None:
        try:
            ms = float(env_raw)
        except ValueError:
            ms = DEFAULT_MCP_TIMEOUT_MS
    elif cfg is not None and cfg.timeout is not None:
        ms = float(cfg.timeout)
    else:
        ms = DEFAULT_MCP_TIMEOUT_MS
    return None if ms <= 0 else ms / 1000.0


# ---------------------------------------------------------------------------
# stdio transport with omp's platform spawn rules
# ---------------------------------------------------------------------------

# Argument bytes cmd.exe delivers unchanged without quoting (omp CMD_SAFE_ARG).
_CMD_SAFE_ARG_RE = re.compile(r"^[A-Za-z0-9#$*+\-./:?@\\_]+$")
_WINDOWS_BATCH_EXTENSIONS = {".cmd", ".bat"}


def _assert_cmd_batch_token(value: str, kind: str) -> None:
    """Reject bytes that cannot round-trip a ``cmd.exe /c`` command line."""
    if any(ch in value for ch in "\0\r\n"):
        raise ValueError(f"Windows batch MCP {kind} cannot contain NUL, CR, or LF characters")


def escape_cmd_quoted_interior(value: str) -> str:
    """Escape the interior of a cmd.exe-quoted token (BatBadBut, CVE-2024-24576).

    Percent becomes ``%%cd:~,%`` (expands to a literal ``%``), quotes are
    doubled, and backslash runs preceding a quote — including the caller's
    closing quote — are doubled so ``CommandLineToArgvW`` delivers them
    literally. Ported verbatim from omp ``escapeCmdQuotedInterior``.
    """
    out: list[str] = []
    backslashes = 0
    for ch in value:
        if ch == "\\":
            backslashes += 1
            out.append(ch)
        elif ch == '"':
            out.append("\\" * backslashes)
            out.append('""')
            backslashes = 0
        elif ch == "%":
            out.append("%%cd:~,%")
            backslashes = 0
        else:
            backslashes = 0
            out.append(ch)
    out.append("\\" * backslashes)  # keep a trailing run literal before the closing quote
    return "".join(out)


def escape_cmd_batch_arg(arg: str) -> str:
    """Escape one argument for cmd.exe's pre-parse; quotes only when needed."""
    _assert_cmd_batch_token(arg, "argument")
    needs_quotes = len(arg) == 0 or arg.endswith("\\") or not _CMD_SAFE_ARG_RE.match(arg)
    return f'"{escape_cmd_quoted_interior(arg)}"' if needs_quotes else arg


def build_cmd_exe_argv(comspec: str, command: str, args: list[str]) -> list[str]:
    """Build the ``cmd.exe /d /e:ON /v:OFF /c "<line>"`` argv for batch shims.

    ``/e:ON`` keeps extensions on (required for the ``%%cd:~,%`` percent
    trick); ``/v:OFF`` disables delayed expansion. Ported from omp
    ``buildCmdExeArgv``.
    """
    _assert_cmd_batch_token(command, "command")
    line = f'""{escape_cmd_quoted_interior(command)}"'
    for arg in args:
        line += f" {escape_cmd_batch_arg(arg)}"
    line += '"'
    return [comspec, "/d", "/e:ON", "/v:OFF", "/c", line]


def build_stdio_argv(command: str, args: list[str]) -> list[str]:
    """Resolve the argv for a Windows stdio server; identity on other platforms.

    Batch files (``.cmd``/``.bat``) and unresolvable bare commands go through
    ``cmd.exe`` with BatBadBut escaping; everything else launches directly.
    POSIX returns ``[command, *args]`` unchanged — the platform rule there is
    the session-detach flag, handled at spawn time.
    """
    if sys.platform != "win32":
        return [command, *args]

    resolved = shutil.which(command)
    if resolved is None:
        for ext in (".cmd", ".bat", ".exe", ".ps1"):
            if path := shutil.which(f"{command}{ext}"):
                resolved = path
                break
    needs_cmd_exe = resolved is None or Path(resolved).suffix.lower() in _WINDOWS_BATCH_EXTENSIONS
    if not needs_cmd_exe:
        return [resolved, *args]
    comspec = os.environ.get("COMSPEC") or "cmd.exe"
    return build_cmd_exe_argv(comspec, resolved or command, args)


def stdio_start_new_session() -> bool:
    """Whether stdio servers spawn detached (their own session).

    POSIX except macOS: ``True`` (setsid) so terminal job control (Ctrl+Z
    SIGTSTP, background-read SIGTTIN) cannot stop the server and block the
    read loop on silent pipes. macOS: ``False`` — LaunchServices/TCC
    attributes Apple Events automation to the responsible terminal only while
    the child stays in the inherited session (omp issue #4987). Windows:
    ``False`` (Job Objects own tree termination there).
    """
    return sys.platform not in ("win32", "darwin")


@asynccontextmanager
async def _stdio_transport(cfg: MCPStdioServerConfig, on_close: Callable[[], None]):
    """Spawn an MCP stdio server and pump newline-delimited JSON-RPC.

    An SDK-shaped transport context manager (yields ``(read_stream,
    write_stream)``) built directly on ``anyio.open_process`` so we control
    the platform spawn rules the SDK hardcodes differently (see
    :func:`stdio_start_new_session`). ``on_close`` fires once when the
    connection can no longer carry traffic (process exit or pump failure).
    """
    import anyio
    import mcp.types as mcp_types
    from mcp.client.stdio import get_default_environment
    from mcp.shared.message import SessionMessage

    argv = build_stdio_argv(cfg.command, list(cfg.args))
    env = get_default_environment() | dict(cfg.env or {})
    cwd = cfg.cwd or None

    kwargs: dict[str, Any] = {"env": env, "stderr": None, "cwd": cwd}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = stdio_start_new_session()

    process = await anyio.open_process(argv, **kwargs)
    assert process.stdin is not None and process.stdout is not None

    read_writer, read_stream = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_stream, write_reader = anyio.create_memory_object_stream[SessionMessage](0)

    closed_fired = False

    def _fire_close() -> None:
        nonlocal closed_fired
        if not closed_fired:
            closed_fired = True
            with suppress(Exception):
                on_close()

    async def _stdout_pump() -> None:
        from anyio.streams.text import TextReceiveStream

        assert process.stdout is not None
        stdout = TextReceiveStream(process.stdout, encoding="utf-8", errors="replace")
        try:
            async with read_writer:
                buffer = ""
                async for chunk in stdout:
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        if not line.strip():
                            continue
                        try:
                            message = mcp_types.jsonrpc_message_adapter.validate_json(line, by_name=False)
                            await read_writer.send(SessionMessage(message))
                        except ValueError as exc:
                            await read_writer.send(exc)
        except Exception:
            logger.debug("stdio stdout pump ended for %r", cfg.command, exc_info=True)
        finally:
            _fire_close()

    async def _stdin_pump() -> None:
        assert process.stdin is not None
        try:
            async with write_reader:
                async for session_message in write_reader:
                    data = session_message.message.model_dump_json(by_alias=True, exclude_unset=True)
                    await process.stdin.send((data + "\n").encode("utf-8"))
        except Exception:
            logger.debug("stdio stdin pump ended for %r", cfg.command, exc_info=True)
        finally:
            _fire_close()

    async def _stop() -> None:
        """Close stdin, give the server a grace window, then kill the tree."""
        with suppress(Exception):
            await process.stdin.aclose()
        exited = False
        for _ in range(20):
            if process.returncode is not None:
                exited = True
                break
            await asyncio.sleep(0.1)
        if not exited:
            for signal_name in ("terminate", "kill"):
                with suppress(Exception):
                    getattr(process, signal_name)()
                for _ in range(20):
                    if process.returncode is not None:
                        break
                    await asyncio.sleep(0.1)
                if process.returncode is not None:
                    break

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_stdout_pump)
            tg.start_soon(_stdin_pump)
            try:
                yield read_stream, write_stream
            finally:
                with suppress(Exception):
                    await _stop()
                tg.cancel_scope.cancel()
    finally:
        _fire_close()


# ---------------------------------------------------------------------------
# Connection state
# ---------------------------------------------------------------------------


@dataclass
class ServerConnection:
    """One live MCP connection: session plus the resources that own it."""

    name: str
    config: MCPServerConfig
    session: Any  # mcp ClientSession (Any keeps fakes viable in tests)
    tools: list[Any] = field(default_factory=list)
    stack: AsyncExitStack | None = None
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)
    source: str = ""


@dataclass
class McpLoadResult:
    """Outcome of one discovery round."""

    tools: list[AgentTool] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    connected_servers: list[str] = field(default_factory=list)


ToolsChangedCallback = Callable[[list[AgentTool]], Awaitable[None] | None]


class _AbortedError(Exception):
    """Raised by ``_race_abort`` when the abort signal beats the tool call.

    Distinct from ``CancelledError`` so an outer task cancellation (dispose)
    still propagates instead of being mapped to a tool result.
    """

def _tool_to_cache_entry(tool: Any) -> dict[str, Any]:
    """Serialize one SDK ``Tool`` (or dict) into the cache JSON shape."""
    if isinstance(tool, dict):
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", "") or "",
            "inputSchema": tool.get("inputSchema") or tool.get("input_schema") or {},
        }
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", "") or "",
        "inputSchema": getattr(tool, "input_schema", None) or {},
    }


class McpManager:
    """Owns every MCP connection for one working directory.

    A top-level session owns the manager it creates and must call
    ``disconnect_all`` on dispose; borrowers (future subagents) must not.
    """

    def __init__(self, cwd: str | os.PathLike[str], tool_cache: McpToolCache | None = None) -> None:
        self.cwd = str(cwd)
        self.tool_cache = tool_cache
        self._configs: dict[str, MCPServerConfig] = {}
        self._sources: dict[str, str] = {}
        self._connections: dict[str, ServerConnection] = {}
        self._tools_by_server: dict[str, list[AgentTool]] = {}
        self._connect_futures: dict[str, asyncio.Future[ServerConnection]] = {}
        self._pending_reconnects: dict[str, asyncio.Task[None]] = {}
        self._pending_continuations: set[asyncio.Task[None]] = set()
        self._watchers: set[asyncio.Task[None]] = set()
        self._reconnect_history: dict[str, deque[float]] = {}
        self._reconnect_suspended: set[str] = set()
        self._on_tools_changed: ToolsChangedCallback | None = None
        self._tool_meta: dict[str, dict[str, Any]] = {}
        self._epoch = 0
        self._disposed = False

    # --- public API --------------------------------------------------------

    def set_on_tools_changed(self, callback: ToolsChangedCallback | None) -> None:
        """Install the callback fired whenever the tool list changes."""
        self._on_tools_changed = callback

    def get_tools(self) -> list[AgentTool]:
        """All registered tools, sorted by name for stability."""
        tools: list[AgentTool] = []
        for server_tools in self._tools_by_server.values():
            tools.extend(server_tools)
        return sorted(tools, key=lambda tool: tool.name)

    def get_connection(self, name: str) -> ServerConnection | None:
        return self._connections.get(name)

    def get_connection_status(self, name: str) -> str:
        """``connected`` | ``connecting`` | ``disconnected`` for one server."""
        if name in self._connections:
            return "connected"
        if name in self._connect_futures or name in self._pending_reconnects:
            return "connecting"
        return "disconnected"

    def get_connected_servers(self) -> list[str]:
        return sorted(self._connections)

    def get_all_server_names(self) -> list[str]:
        return sorted(self._configs)

    def get_source(self, name: str) -> str | None:
        return self._sources.get(name)

    def get_server_config(self, name: str) -> MCPServerConfig | None:
        return self._configs.get(name)

    async def discover_and_connect(self) -> McpLoadResult:
        """Discover configs, race connects against the 250 ms gate.

        After the gate: settled connects are live tools; rejected connects are
        error entries (others continue); pending connects with a cache hit
        become deferred tools; pending without cache contribute nothing until
        the background continuation swaps them in and fires on_tools_changed.
        """
        configs, sources = load_all_mcp_configs(self.cwd)
        self._configs = configs
        self._sources = sources
        self._disposed = False

        result = McpLoadResult()
        tasks: dict[str, asyncio.Task[ServerConnection]] = {}
        for name, cfg in configs.items():
            errors = validate_server_config(name, cfg)
            if errors:
                result.errors[name] = "; ".join(errors)
                continue
            tasks[name] = asyncio.get_running_loop().create_task(self._connect_server(name, cfg))

        if not tasks:
            return result

        done, _pending = await asyncio.wait(set(tasks.values()), timeout=STARTUP_GATE_MS / 1000.0)

        done_names = {name for name, task in tasks.items() if task in done}
        for name in done_names:
            task = tasks[name]
            try:
                conn = task.result()
            except Exception as exc:
                result.errors[name] = str(exc)
                logger.warning("MCP server %r failed to connect: %s", name, exc)
                continue
            self._register_connection(conn)
            result.connected_servers.append(name)

        for name, task in tasks.items():
            if name in done_names:
                continue
            # Still pending at the gate: defer from cache, or contribute nothing.
            cached = self.tool_cache.get(name) if self.tool_cache is not None else None
            if cached:
                self._tools_by_server[name] = [
                    self._build_tool(name, entry, deferred=True) for entry in cached
                ]
            # Deferred executes await this future; the continuation settles it.
            self._connect_futures[name] = asyncio.get_running_loop().create_future()
            continuation = asyncio.get_running_loop().create_task(self._finish_pending(name, task))
            self._pending_continuations.add(continuation)
            continuation.add_done_callback(self._pending_continuations.discard)

        result.tools = self.get_tools()
        return result

    async def wait_for_connection(self, name: str) -> ServerConnection:
        """Block until ``name`` has a live connection (deferred tool path)."""
        conn = self._connections.get(name)
        if conn is not None:
            return conn
        future = self._connect_futures.get(name)
        if future is not None:
            return await asyncio.shield(future)
        raise McpConnectionError(f"MCP server {name!r} is not connected")

    async def refresh_server_tools(self, name: str) -> None:
        """Re-list one server's tools, update registration + cache, notify."""
        conn = self._connections.get(name)
        if conn is None:
            return
        try:
            tools = await self._list_all_tools(conn.session)
        except Exception as exc:
            logger.warning("MCP tools refresh failed for %r: %s", name, exc)
            return
        conn.tools = tools
        self._register_tools(name, tools)
        if self.tool_cache is not None:
            self.tool_cache.put(name, [_tool_to_cache_entry(tool) for tool in tools])
        self._fire_tools_changed()

    async def reconnect_server(self, name: str) -> ServerConnection | None:
        """Manual reconnect: resets the breaker history for ``name``."""
        self._reconnect_history.pop(name, None)
        self._reconnect_suspended.discard(name)
        pending = self._pending_reconnects.pop(name, None)
        if pending is not None:
            pending.cancel()
        await self._teardown_connection(name)
        cfg = self._configs.get(name)
        if cfg is None:
            return None
        try:
            conn = await self._connect_server(name, cfg)
        except Exception as exc:
            logger.warning("Manual MCP reconnect failed for %r: %s", name, exc)
            return None
        self._register_connection(conn)
        self._fire_tools_changed()
        return conn

    async def disconnect_server(self, name: str) -> None:
        """Tear down one server and drop its tools."""
        pending = self._pending_reconnects.pop(name, None)
        if pending is not None:
            pending.cancel()
        await self._teardown_connection(name)
        self._tools_by_server.pop(name, None)
        self._fire_tools_changed()

    async def disconnect_all(self) -> None:
        """Tear everything down; bumps the epoch so late reconnects die."""
        self._epoch += 1
        self._disposed = True
        for task in list(self._pending_reconnects.values()):
            task.cancel()
        self._pending_reconnects.clear()
        for task in list(self._pending_continuations):
            task.cancel()
        self._pending_continuations.clear()
        for future in self._connect_futures.values():
            if not future.done():
                future.cancel()
        self._connect_futures.clear()
        for name in list(self._connections):
            await self._teardown_connection(name)
        for watcher in list(self._watchers):
            watcher.cancel()
        self._watchers.clear()
        self._tools_by_server.clear()
        self._connections.clear()

    # --- connection lifecycle ----------------------------------------------

    async def _connect_server(self, name: str, cfg: MCPServerConfig) -> ServerConnection:
        """Open transport + session, initialize, list tools, update cache.

        This is the seam tests override: it returns a :class:`ServerConnection`
        without touching a real server.
        """
        timeout_s = resolve_mcp_timeout_s(cfg)
        stack = AsyncExitStack()
        try:
            conn = await self._open_transport_and_session(stack, name, cfg, timeout_s)
        except BaseException:
            await stack.aclose()
            raise

        try:
            tools = await self._list_all_tools(conn.session)
        except BaseException:
            await stack.aclose()
            raise

        conn.tools = tools
        if self.tool_cache is not None:
            self.tool_cache.put(name, [_tool_to_cache_entry(tool) for tool in tools])
        return conn

    async def _open_transport_and_session(
        self,
        stack: AsyncExitStack,
        name: str,
        cfg: MCPServerConfig,
        timeout_s: float | None,
    ) -> ServerConnection:
        """Enter the transport + ClientSession context managers on ``stack``."""
        import mcp.types as mcp_types
        from mcp.client.session import ClientSession

        conn = ServerConnection(
            name=name,
            config=cfg,
            session=None,
            stack=stack,
            source=self._sources.get(name, ""),
        )

        if isinstance(cfg, MCPStdioServerConfig):
            streams_cm = _stdio_transport(cfg, lambda: conn.closed_event.set())
        elif isinstance(cfg, MCPHttpServerConfig):
            from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

            http_client = create_mcp_http_client(
                headers=dict(cfg.headers) or None,
                auth=self._build_oauth_auth(cfg.url, cfg),
            )
            streams_cm = streamable_http_client(cfg.url, http_client=http_client)
        elif isinstance(cfg, MCPSseServerConfig):
            from mcp.client.sse import sse_client

            streams_cm = sse_client(cfg.url, headers=dict(cfg.headers) or None)
        else:  # pragma: no cover - validation rejects unknown shapes
            raise McpConnectionError(f"unsupported MCP transport for {name!r}")

        read_stream, write_stream = await stack.enter_async_context(streams_cm)

        def _message_handler(message: Any) -> Awaitable[None]:
            return self._on_session_message(name, conn, message)

        session = ClientSession(
            read_stream=read_stream,
            write_stream=write_stream,
            read_timeout_seconds=timeout_s,
            message_handler=_message_handler,
            client_info=mcp_types.Implementation(name="local-operator", version="2.0"),
        )
        await stack.enter_async_context(session)
        await session.initialize()
        conn.session = session
        return conn

    def _build_oauth_auth(self, url: str, cfg: MCPServerConfig) -> Any | None:
        """Build an ``OAuthClientProvider`` for configs with ``auth.type=oauth``."""
        auth = getattr(cfg, "auth", None)
        if auth is None or auth.type != "oauth":
            return None
        try:
            from mcp.client.auth import OAuthClientProvider

            from local_operator.mcp.auth import wire_oauth_auth

            return OAuthClientProvider(**wire_oauth_auth(url, cfg))
        except Exception:
            logger.warning("OAuth wiring unavailable for %r; connecting unauthenticated", url, exc_info=True)
            return None

    async def _on_session_message(self, name: str, conn: ServerConnection, message: Any) -> None:
        """Session message_handler: notifications + transport faults.

        ``tools/list_changed`` refreshes the server's tools; transport-level
        exceptions mark the connection closed and schedule a reconnect.
        """
        if isinstance(message, BaseException):
            if self._connections.get(name) is conn and not conn.closed_event.is_set():
                conn.closed_event.set()
                self._handle_disconnect(name)
            return
        method = getattr(message, "method", "")
        if method == "notifications/tools/list_changed":
            await self.refresh_server_tools(name)

    async def _finish_pending(self, name: str, task: asyncio.Task[ServerConnection]) -> None:
        """Background continuation for a server still connecting at the gate."""
        future = self._connect_futures.get(name)
        try:
            conn = await task
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("MCP server %r failed to connect after the gate: %s", name, exc)
            if future is not None and not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved; waiters still see the raise
            self._connect_futures.pop(name, None)
            self._tools_by_server.pop(name, None)  # drop the deferred slice
            self._fire_tools_changed()
            return
        if self._disposed:
            if conn.stack is not None:
                with suppress(Exception):
                    await conn.stack.aclose()
            return
        self._register_connection(conn)  # settles the waiter future
        self._fire_tools_changed()

    def _register_connection(self, conn: ServerConnection) -> None:
        """Install a live connection: registry, tools, waiter, watcher."""
        old = self._connections.get(conn.name)
        if old is not None and old is not conn:
            old.closed_event.set()
            if old.stack is not None:
                with suppress(Exception):
                    asyncio.get_running_loop().create_task(old.stack.aclose())
        self._connections[conn.name] = conn
        self._register_tools(conn.name, conn.tools)
        future = self._connect_futures.pop(conn.name, None)
        if future is not None and not future.done():
            future.set_result(conn)
        watcher = asyncio.get_running_loop().create_task(self._watch_connection(conn.name, conn))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)

    def _register_tools(self, name: str, tools: list[Any]) -> None:
        """Build AgentTools for one server's tool list (live, not deferred)."""
        self._tools_by_server[name] = [self._build_tool(name, tool, deferred=False) for tool in tools]

    def _build_tool(self, server_name: str, tool: Any, *, deferred: bool) -> AgentTool:
        """Wrap one tool (SDK model or cached dict) with the manager call path."""

        if isinstance(tool, dict):
            mcp_tool_name = tool.get("name", "")
        else:
            mcp_tool_name = getattr(tool, "name", "") or ""

        async def _call(
            tool_call_id: str,
            args: dict[str, Any],
            signal: Any,
            on_update: Any,
            context: Any,
        ) -> ToolResult:
            return await self._execute_tool_call(
                server_name, mcp_tool_name, tool_call_id, args, signal, deferred=deferred
            )

        agent_tool = build_agent_tool(server_name, tool, _call)
        self._tool_meta[agent_tool.name] = {
            "server_name": server_name,
            "mcp_tool_name": mcp_tool_name,
            "deferred": deferred,
        }
        return agent_tool

    async def _execute_tool_call(
        self,
        server_name: str,
        mcp_tool_name: str,
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any,
        *,
        deferred: bool,
    ) -> ToolResult:
        """One tools/call with arg hygiene, abort racing, and one retry."""
        tool_label = create_mcp_tool_name(server_name, mcp_tool_name)
        try:
            if deferred or server_name not in self._connections:
                conn = await self.wait_for_connection(server_name)
            else:
                conn = self._connections[server_name]
        except Exception as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=tool_label,
                content=[TextContent(text=f"MCP error: {exc}")],
                is_error=True,
            )

        properties, required, additional = self._schema_parts(conn, mcp_tool_name)
        outbound = prepare_outbound_args(args, properties, required, additional)

        async def _call_once() -> Any:
            timeout_s = resolve_mcp_timeout_s(conn.config)
            if timeout_s is not None:
                return await conn.session.call_tool(mcp_tool_name, outbound, read_timeout_seconds=timeout_s)
            return await conn.session.call_tool(mcp_tool_name, outbound)
        try:
            result = await self._race_abort(_call_once(), signal)
        except _AbortedError:
            return self._error_result(tool_call_id, tool_label, RuntimeError("aborted"))
        except Exception as exc:
            if is_retriable_connection_error(exc):
                # One reconnect + one retry at the call site (omp policy).
                new_conn = await self._reconnect_for_call(server_name)
                if new_conn is not None:
                    conn = new_conn
                    try:
                        result = await self._race_abort(_call_once(), signal)
                    except Exception as retry_exc:
                        return self._error_result(tool_call_id, tool_label, retry_exc)
                else:
                    return self._error_result(tool_call_id, tool_label, exc)
            else:
                return self._error_result(tool_call_id, tool_label, exc)
        return format_mcp_result(result, tool_call_id, tool_label)

    async def _race_abort(self, coro: Awaitable[Any], signal: Any) -> Any:
        """Run ``coro`` racing the abort signal; abort wins with cancellation.

        The call is wrapped in a task; a racing ``signal.wait()`` task decides
        the winner. When the abort lands first the work task is cancelled and
        the method raises ``CancelledError`` (the caller maps that to an
        aborted tool result). A ``None`` signal runs the coroutine inline.
        """
        if signal is None:
            return await coro
        task = asyncio.get_running_loop().create_task(coro)
        abort_task = asyncio.get_running_loop().create_task(signal.wait())
        try:
            done, _pending = await asyncio.wait(
                {task, abort_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if abort_task in done and task not in done:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                raise _AbortedError("aborted")
            return task.result()
        finally:
            abort_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task

    def _schema_parts(self, conn: ServerConnection, mcp_tool_name: str) -> tuple[dict[str, Any], list[str], Any]:
        """Extract (properties, required, additionalProperties) for arg hygiene."""
        for tool in conn.tools:
            name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
            if name != mcp_tool_name:
                continue
            schema = (
                tool.get("inputSchema") or tool.get("input_schema")
                if isinstance(tool, dict)
                else getattr(tool, "input_schema", None)
            ) or {}
            properties = schema.get("properties") if isinstance(schema, dict) else None
            required = schema.get("required") if isinstance(schema, dict) else None
            additional = schema.get("additionalProperties") if isinstance(schema, dict) else None
            return (
                properties if isinstance(properties, dict) else {},
                required if isinstance(required, list) else [],
                additional,
            )
        return {}, [], None

    @staticmethod
    def _error_result(tool_call_id: str, tool_name: str, exc: BaseException) -> ToolResult:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=[TextContent(text=f"MCP error: {exc}")],
            is_error=True,
        )

    # --- pagination ----------------------------------------------------------

    @staticmethod
    async def _list_all_tools(session: Any) -> list[Any]:
        """Follow ``tools/list`` pagination (nextCursor) to completion."""
        from mcp.types import PaginatedRequestParams

        tools: list[Any] = []
        cursor: str | None = None
        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor is not None else None
            result = await session.list_tools(params=params)
            tools.extend(result.tools)
            cursor = result.next_cursor
            if cursor is None:
                return tools

    # --- reconnect / circuit breaker -----------------------------------------

    async def _watch_connection(self, name: str, conn: ServerConnection) -> None:
        """Wait for the connection's closed event, then handle the disconnect."""
        await conn.closed_event.wait()
        self._handle_disconnect(name, expected=conn)

    def _handle_disconnect(self, name: str, expected: ServerConnection | None = None) -> None:
        """Transport closed: drop the live connection, schedule a reconnect.

        Tools stay registered (deferred executes await the reconnect). An
        ``expected`` connection guards against a stale watcher firing after a
        replacement took the slot.
        """
        if self._disposed:
            return
        current = self._connections.get(name)
        if expected is not None and current is not expected:
            return  # superseded connection reporting in; ignore
        conn = self._connections.pop(name, None)
        if conn is None:
            return
        future = self._connect_futures.get(name)
        if future is not None and future.done():
            if not future.cancelled():
                future.exception()  # consume to avoid "never retrieved" warnings
            self._connect_futures.pop(name, None)
        # Deferred executes must now await the reconnect: install a fresh waiter.
        if name not in self._connect_futures:
            self._connect_futures[name] = asyncio.get_running_loop().create_future()
        self._schedule_reconnect(name)

    def reconnect_suspended(self, name: str) -> bool:
        """Whether the circuit breaker currently suspends auto-reconnect."""
        return name in self._reconnect_suspended

    def get_tool_meta(self, tool_name: str) -> dict[str, Any] | None:
        """MCP origin metadata for a minted tool name (server, tool, deferred)."""
        return self._tool_meta.get(tool_name)

    def _record_reconnect_attempt(self, name: str) -> bool:
        """Account one attempt in the sliding window; False trips the breaker."""
        now = asyncio.get_running_loop().time()
        history = self._reconnect_history.setdefault(name, deque())
        while history and now - history[0] > RECONNECT_BURST_WINDOW_S:
            history.popleft()
        if len(history) >= RECONNECT_BURST_LIMIT:
            self._reconnect_suspended.add(name)
            return False
        history.append(now)
        return True

    def _schedule_reconnect(self, name: str) -> None:
        """Queue a backoff-delayed reconnect unless the breaker is tripped."""
        if self._disposed or name in self._reconnect_suspended:
            return
        if not self._record_reconnect_attempt(name):
            logger.warning(
                "MCP reconnect breaker tripped for %r: >%d attempts in %ds; "
                "auto-reconnect suspended (manual reconnect resets)",
                name,
                RECONNECT_BURST_LIMIT,
                int(RECONNECT_BURST_WINDOW_S),
            )
            return
        history = self._reconnect_history[name]
        attempt = len(history) - 1
        delay = RECONNECT_BACKOFF_S[min(attempt, len(RECONNECT_BACKOFF_S) - 1)]
        epoch = self._epoch
        task = asyncio.get_running_loop().create_task(self._reconnect(name, delay, epoch))
        previous = self._pending_reconnects.pop(name, None)
        if previous is not None and previous is not task:
            previous.cancel()
        self._pending_reconnects[name] = task

        def _discard(done: asyncio.Task[None]) -> None:
            if self._pending_reconnects.get(name) is done:
                self._pending_reconnects.pop(name, None)

        task.add_done_callback(_discard)

    async def _reconnect(self, name: str, delay: float, epoch: int) -> None:
        """One reconnect attempt after ``delay``; the epoch guards resurrection."""
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        if epoch != self._epoch or self._disposed:
            return  # disconnect_all ran meanwhile; never resurrect
        cfg = self._configs.get(name)
        if cfg is None:
            return
        await self._teardown_connection(name)
        future: asyncio.Future[ServerConnection] = asyncio.get_running_loop().create_future()
        self._connect_futures[name] = future
        try:
            conn = await self._connect_server(name, cfg)
        except Exception as exc:
            logger.warning("MCP reconnect attempt failed for %r: %s", name, exc)
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved; waiters still see the raise
            self._connect_futures.pop(name, None)
            self._schedule_reconnect(name)
            return
        self._register_connection(conn)
        self._reconnect_history.pop(name, None)
        self._fire_tools_changed()

    async def _reconnect_for_call(self, name: str) -> ServerConnection | None:
        """Synchronous reconnect for the call-site retry (no backoff wait)."""
        cfg = self._configs.get(name)
        if cfg is None:
            return None
        await self._teardown_connection(name)
        try:
            conn = await self._connect_server(name, cfg)
        except Exception as exc:
            logger.warning("MCP call-site reconnect failed for %r: %s", name, exc)
            return None
        self._register_connection(conn)
        return conn

    async def _teardown_connection(self, name: str) -> None:
        """Close one connection's stack without touching registries."""
        conn = self._connections.pop(name, None)
        if conn is None:
            return
        conn.closed_event.set()
        if conn.stack is not None:
            with suppress(Exception):
                await conn.stack.aclose()

    # --- notifications -------------------------------------------------------

    def _fire_tools_changed(self) -> None:
        """Invoke on_tools_changed with the full sorted tool list."""
        callback = self._on_tools_changed
        if callback is None:
            return
        tools = self.get_tools()
        try:
            outcome = callback(tools)
            if asyncio.iscoroutine(outcome):
                asyncio.get_running_loop().create_task(outcome)
        except Exception:
            logger.exception("on_tools_changed callback raised")


# Re-export for callers that serialize cache entries.
__all__ = [
    "McpManager",
    "McpLoadResult",
    "McpConnectionError",
    "ServerConnection",
    "STARTUP_GATE_MS",
    "RECONNECT_BACKOFF_S",
    "RECONNECT_BURST_WINDOW_S",
    "RECONNECT_BURST_LIMIT",
    "resolve_mcp_timeout_s",
    "build_stdio_argv",
    "stdio_start_new_session",
    "build_cmd_exe_argv",
    "escape_cmd_batch_arg",
    "escape_cmd_quoted_interior",
]
