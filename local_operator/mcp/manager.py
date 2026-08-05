"""MCP connection manager: fast-startup gate, deferred tools, reconnect breaker.

Ports the established MCP manager semantics onto the official
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

NOTE (session integration): this module deliberately does NOT wire itself
into the harness session loop — the ExecCli stream owns that integration
(``discover_and_load_mcp_tools`` + ``set_on_tools_changed`` rebinding). This
package exports only the manager surface; consumers must drive lifecycle.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    TextContent,
    ToolContext,
    ToolResult,
)
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
from local_operator.optional import missing_extra_error

if TYPE_CHECKING:
    # Annotation-only SDK references: the extra may be absent, and even when
    # it is installed the real imports stay at their call sites so config-only
    # callers never pay for the transport machinery.
    from mcp.client.auth import OAuthClientProvider
    from mcp.client.session import IncomingMessage
    from mcp.client.streamable_http import TransportStreams
    from mcp.types import CallToolResult, ListToolsResult, PaginatedRequestParams, Tool

    from local_operator.mcp.auth import ManagedAuthStore

logger = logging.getLogger(__name__)

# Result of one tool call raced against an abort; ``_race_abort`` is
# transparent to whatever the call itself resolves to.
_RaceT = TypeVar("_RaceT")


def _sdk_available() -> bool:
    """Whether the MCP client SDK is importable.

    The SDK is an optional extra: it drags in a TLS stack, a JSON Schema
    validator, and (on Windows) the pywin32 bundle, none of which the core
    agent needs. Probing with :func:`importlib.util.find_spec` avoids paying
    the import cost just to answer the question — the real imports stay at
    their call sites.
    """
    return importlib.util.find_spec("mcp") is not None


# Fast-startup gate: how long discovery blocks before deferring slow servers.
STARTUP_GATE_MS = 250

# Reconnect policy: escalating backoff, sliding-window circuit breaker.
RECONNECT_BACKOFF_S = (0.5, 1.0, 2.0, 4.0)
RECONNECT_BURST_WINDOW_S = 30.0
RECONNECT_BURST_LIMIT = 5

# Reconnect attempts are accounted in one sliding window per server
# (``_reconnect_history``); the backoff ladder position is separate state
# (``_backoff_index``) so a successful reconnect resets the LADDER but never
# clears the window — a flapping server still trips the breaker (MCP-07).

# Default per-request timeout; ``LOCAL_OPERATOR_MCP_TIMEOUT_MS`` overrides,
# config ``timeout`` (ms) refines, ``0`` disables.
DEFAULT_MCP_TIMEOUT_MS = 30_000.0


class McpConnectionError(RuntimeError):
    """Raised when a server cannot be reached (deferred execute path)."""


def _settle_future_error(future: asyncio.Future[ServerConnection] | None, exc: Exception) -> None:
    """Set an exception on ``future`` and consume it so waiters raise cleanly.

    ``None`` is a no-op on purpose: every caller reaches for the waiter with
    ``dict.get``/``dict.pop``, and "there was nobody parked on this server"
    is the ordinary case, not an error.
    """
    if future is not None and not future.done():
        future.set_exception(exc)
        future.exception()  # mark retrieved; waiters still see the raise


def resolve_mcp_timeout_s(cfg: MCPServerConfig | None) -> float | None:
    """Resolve the client-side request timeout in seconds (``None`` = off).

    Precedence: ``LOCAL_OPERATOR_MCP_TIMEOUT_MS`` env > ``config.timeout`` >
    30 s default; ``0`` disables the timeout entirely (established behavior).
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
# stdio transport with the established platform spawn rules
# ---------------------------------------------------------------------------

# Argument bytes cmd.exe delivers unchanged without quoting (CMD_SAFE_ARG).
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
    literally. Implements the standard cmd.exe interior-quoting rule.
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
    trick); ``/v:OFF`` disables delayed expansion.
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


def win32_process_target(argv: list[str]) -> str:
    """Single-string command line for ``anyio.open_process`` on Windows (MCP-10).

    Passing a STRING (not a list) makes the spawn use the raw command line
    and skip ``list2cmdline`` re-escaping — the only way the BatBadBut
    escaping in a ``cmd.exe /c`` payload reaches ``CreateProcess`` verbatim.
    Tokens needing quotes were already quoted by the escaper; ``argv[0]`` is
    always quoted defensively.
    """
    return f'"{argv[0]}" {" ".join(argv[1:])}'


def stdio_start_new_session() -> bool:
    """Whether stdio servers spawn detached (their own session).

    POSIX except macOS: ``True`` (setsid) so terminal job control (Ctrl+Z
    SIGTSTP, background-read SIGTTIN) cannot stop the server and block the
    read loop on silent pipes. macOS: ``False`` — LaunchServices/TCC
    attributes Apple Events automation to the responsible terminal only while
    the child stays in the inherited session (a real macOS automation bug). Windows:
    ``False`` (Job Objects own tree termination there).
    """
    return sys.platform not in ("win32", "darwin")


@asynccontextmanager
async def _stdio_transport(
    cfg: MCPStdioServerConfig, on_close: Callable[[], None]
) -> AsyncIterator[TransportStreams]:
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
    target: str | list[str]
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # The pre-escaped cmd.exe command line must reach CreateProcess
        # verbatim: passing a single string bypasses list2cmdline, which
        # would re-quote the BatBadBut escaping in the ``/c`` payload.
        # Deferred: npm cmd-shims whose fallback interpreter is node are not
        # yet bypassed straight to node.
        target = win32_process_target(argv)
    else:
        kwargs["start_new_session"] = stdio_start_new_session()
        target = argv

    process = await anyio.open_process(target, **kwargs)
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
                            message = mcp_types.jsonrpc_message_adapter.validate_json(
                                line, by_name=False
                            )
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
                    data = session_message.message.model_dump_json(
                        by_alias=True, exclude_unset=True
                    )
                    await process.stdin.send((data + "\n").encode("utf-8"))
        except Exception:
            logger.debug("stdio stdin pump ended for %r", cfg.command, exc_info=True)
        finally:
            _fire_close()

    async def _stop() -> None:
        """Close stdin, give the server a grace window, then kill the tree."""
        stdin = process.stdin
        if stdin is not None:
            with suppress(Exception):
                await stdin.aclose()
        exited = False
        for _ in range(20):
            if process.returncode is not None:
                exited = True
                break
            await asyncio.sleep(0.1)
        if not exited:
            for stop_process in (process.terminate, process.kill):
                with suppress(Exception):
                    stop_process()
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


class McpSession(Protocol):
    """The ``ClientSession`` slice this manager drives.

    Structural on purpose: the SDK's ``ClientSession`` satisfies it, and so
    do the in-process fakes that stand in for a server in tests, without
    either side depending on the other. Only the two request methods the
    manager actually issues are declared.
    """

    async def list_tools(self, *, params: PaginatedRequestParams | None = None) -> ListToolsResult:
        """One page of ``tools/list``."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> CallToolResult:
        """Invoke one tool and wait for its result."""
        ...


@dataclass
class ServerConnection:
    """One live MCP connection: session plus the resources that own it."""

    name: str
    config: MCPServerConfig
    # ``None`` only during the window in which the transport callbacks close
    # over this object while its session is still being constructed; use
    # :attr:`live_session` everywhere else.
    session: McpSession | None = None
    tools: list[Tool] = field(default_factory=list)
    stack: AsyncExitStack | None = None
    closed_event: asyncio.Event = field(default_factory=asyncio.Event)
    source: str = ""

    @property
    def live_session(self) -> McpSession:
        """The session, which every caller past the open handshake has."""
        session = self.session
        if session is None:
            raise McpConnectionError(f"MCP server {self.name!r} has no live session")
        return session


@dataclass
class McpLoadResult:
    """Outcome of one discovery round."""

    tools: list[AgentTool] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    connected_servers: list[str] = field(default_factory=list)


ToolsChangedCallback = Callable[[list[AgentTool]], Awaitable[None] | None]


class McpToolMeta(TypedDict, total=False):
    """Origin bookkeeping for one minted tool name.

    ``agent_name`` is filled in by :meth:`McpManager._rebuild_agent_names`
    once collision suffixing has resolved, so it is absent from the entry a
    tool is first registered with.
    """

    server_name: str
    mcp_tool_name: str
    deferred: bool
    agent_name: str


# Abort semantics: an abort racing a tool call raises ``asyncio.CancelledError``
# (MCP-16, "abort stays abort") so it propagates like any outer cancellation and
# is NEVER mapped to a tool error result by the call path.


def _tool_to_cache_entry(tool: Tool) -> dict[str, Any]:
    """Serialize one SDK ``Tool`` into the cache JSON shape."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "inputSchema": tool.input_schema or {},
    }


class McpManager:
    """Owns every MCP connection for one working directory.

    A top-level session owns the manager it creates and must call
    ``disconnect_all`` on dispose; borrowers (future subagents) must not.
    """

    def __init__(
        self,
        cwd: str | os.PathLike[str],
        tool_cache: McpToolCache | None = None,
        auth_store: ManagedAuthStore | None = None,
    ) -> None:
        self.cwd = str(cwd)
        self.tool_cache = tool_cache
        # The session's AuthStore, when injected: every OAuth MCP server's
        # token storage shares it instead of opening its own SQLite
        # connection (which nothing closed — one leaked WAL handle per
        # reconnect). None means the manager constructs and owns one, closed
        # in disconnect_all.
        self.auth_store: ManagedAuthStore | None = auth_store
        self._owns_auth_store = auth_store is None
        self._configs: dict[str, MCPServerConfig] = {}
        self._sources: dict[str, str] = {}
        self._connections: dict[str, ServerConnection] = {}
        self._tools_by_server: dict[str, list[AgentTool]] = {}
        self._connect_futures: dict[str, asyncio.Future[ServerConnection]] = {}
        self._pending_reconnects: dict[str, asyncio.Task[None]] = {}
        # Gate continuations, per server: (continuation task, raw gate task).
        self._pending_continuations: dict[
            str, tuple[asyncio.Task[None], asyncio.Task[ServerConnection]]
        ] = {}
        self._watchers: set[asyncio.Task[None]] = set()
        # Fire-and-forget tools/list_changed refreshes (MCP-05): never awaited
        # inline on the SDK receive path.
        self._notify_tasks: set[asyncio.Task[None]] = set()
        self._reconnect_history: dict[str, deque[float]] = {}
        self._reconnect_suspended: set[str] = set()
        # Backoff ladder position is separate from the breaker window (MCP-07):
        # a successful reconnect resets the ladder but keeps the window intact,
        # so a flapping server still trips the breaker.
        self._backoff_index: dict[str, int] = {}
        self._on_tools_changed: ToolsChangedCallback | None = None
        # Tool-name collision state keyed by stable origin key (MCP-09):
        # (server name, original tool name), never registration order.
        self._tool_meta: dict[str, McpToolMeta] = {}
        self._tool_by_origin: dict[tuple[str, str], AgentTool] = {}
        self._meta_by_origin: dict[tuple[str, str], McpToolMeta] = {}
        self._origins_by_server: dict[str, set[tuple[str, str]]] = {}
        # First-connect security surface (MCP-12): one warning per server.
        self._security_logged: set[str] = set()
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
        self._drop_removed_servers(configs)
        return await self._connect_round(configs, sources)

    async def reload(self) -> McpLoadResult:
        """Re-discover and reconnect in place (``/mcp reload`` semantics).

        Bumps the epoch so in-flight reconnects and gate continuations die,
        cancels pending reconnects, tears down every live connection, drops
        servers that left the config (tools, meta, cache), and reconnects the
        rest from fresh configs. The manager object is reused, so callbacks
        installed via ``set_on_tools_changed`` survive (MCP-17).
        """
        self._epoch += 1
        for task in list(self._pending_reconnects.values()):
            task.cancel()
        self._pending_reconnects.clear()
        for continuation, gate_task in list(self._pending_continuations.values()):
            continuation.cancel()
            gate_task.cancel()
        self._pending_continuations.clear()
        configs, sources = load_all_mcp_configs(self.cwd)
        self._drop_removed_servers(configs)
        loop = asyncio.get_running_loop()
        for name in list(self._connections):
            # Deferred executes must wait out the reconnect, not fail.
            future = self._connect_futures.get(name)
            if future is None or future.done():
                self._connect_futures[name] = loop.create_future()
            await self._teardown_connection(name)
        return await self._connect_round(configs, sources)

    def _drop_removed_servers(self, configs: dict[str, MCPServerConfig]) -> None:
        """Drop all state for servers that left the config (MCP-17)."""
        gone = (set(self._tools_by_server) | set(self._configs)) - set(configs)
        for name in gone:
            self._tools_by_server.pop(name, None)
            self._unregister_origins(name)
            self._reconnect_history.pop(name, None)
            self._reconnect_suspended.discard(name)
            self._backoff_index.pop(name, None)
            future = self._connect_futures.pop(name, None)
            _settle_future_error(
                future, McpConnectionError(f"MCP server {name!r} removed from config")
            )
            if self.tool_cache is not None:
                with suppress(Exception):
                    self.tool_cache.delete(name)

    async def _connect_round(
        self, configs: dict[str, MCPServerConfig], sources: dict[str, str]
    ) -> McpLoadResult:
        """Race connects for ``configs`` against the 250 ms startup gate."""
        self._configs = configs
        self._sources = sources
        self._disposed = False

        result = McpLoadResult()
        if configs and not _sdk_available():
            # One actionable line instead of the same opaque
            # "No module named 'mcp'" repeated per configured server. The
            # session treats MCP as enrichment, so this surfaces as a warning
            # and the turn proceeds with zero MCP tools.
            message = missing_extra_error("mcp", "Connecting to MCP servers")
            result.errors.update({name: message for name in configs})
            return result

        tasks: dict[str, asyncio.Task[ServerConnection]] = {}
        for name, cfg in configs.items():
            errors = validate_server_config(name, cfg)
            if errors:
                result.errors[name] = "; ".join(errors)
                continue
            tasks[name] = asyncio.get_running_loop().create_task(self._connect_server(name, cfg))

        if not tasks:
            result.tools = self.get_tools()
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
                # A parked waiter (reload) must fail, not hang (MCP-08).
                waiter = self._connect_futures.pop(name, None)
                _settle_future_error(waiter, exc)
                continue
            self._register_connection(conn)
            result.connected_servers.append(name)

        for name, task in tasks.items():
            if name in done_names:
                continue
            # Still pending at the gate: defer from cache, or contribute nothing.
            cached = self.tool_cache.get(name) if self.tool_cache is not None else None
            if cached:
                self._unregister_origins(name)
                self._tools_by_server[name] = [
                    self._build_tool(name, entry, deferred=True) for entry in cached
                ]
                self._rebuild_agent_names()
            # Deferred executes await this future; the continuation settles it.
            # Reuse a live waiter installed by reload() rather than stranding
            # its waiters behind a fresh future (MCP-08/MCP-17).
            future = self._connect_futures.get(name)
            if future is None or future.done():
                self._connect_futures[name] = asyncio.get_running_loop().create_future()
            continuation = asyncio.get_running_loop().create_task(
                self._finish_pending(name, task, self._epoch)
            )
            self._pending_continuations[name] = (continuation, task)

            def _discard_continuation(done: asyncio.Task[None], _name: str = name) -> None:
                entry = self._pending_continuations.get(_name)
                if entry is not None and entry[0] is done:
                    self._pending_continuations.pop(_name, None)

            continuation.add_done_callback(_discard_continuation)

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
            tools = await self._list_all_tools(conn.live_session)
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
        self._backoff_index.pop(name, None)
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
        continuation = self._pending_continuations.pop(name, None)
        if continuation is not None:
            continuation[0].cancel()
            continuation[1].cancel()  # kill the underlying connect too
        # A pending deferred-connect waiter must fail, not hang (MCP-19).
        future = self._connect_futures.pop(name, None)
        _settle_future_error(future, McpConnectionError(f"MCP server {name!r} disconnected"))
        await self._teardown_connection(name)
        self._tools_by_server.pop(name, None)
        self._unregister_origins(name)
        self._rebuild_agent_names()
        self._fire_tools_changed()

    async def disconnect_all(self) -> None:
        """Tear everything down; bumps the epoch so late reconnects die."""
        self._epoch += 1
        self._disposed = True
        for task in list(self._pending_reconnects.values()):
            task.cancel()
        self._pending_reconnects.clear()
        for continuation, gate_task in list(self._pending_continuations.values()):
            continuation.cancel()
            gate_task.cancel()
        self._pending_continuations.clear()
        for name, future in self._connect_futures.items():
            # Settle with an error (never cancel): a cancelled future would
            # surface as CancelledError in waiters; deferred executes need a
            # real McpConnectionError they can turn into a tool result (MCP-08).
            _settle_future_error(future, McpConnectionError("MCP manager disposed"))
        self._connect_futures.clear()
        for name in list(self._connections):
            await self._teardown_connection(name)
        for watcher in list(self._watchers):
            watcher.cancel()
        for task in list(self._notify_tasks):
            task.cancel()
        self._notify_tasks.clear()
        self._watchers.clear()
        self._tools_by_server.clear()
        self._connections.clear()
        if self._owns_auth_store and self.auth_store is not None:
            try:
                self.auth_store.close()
            except Exception:  # noqa: BLE001 — teardown must not raise
                logger.debug("closing manager-owned auth store failed", exc_info=True)
            self.auth_store = None

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
            tools = await self._list_all_tools(conn.live_session)
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
            stack=stack,
            source=self._sources.get(name, ""),
        )

        if isinstance(cfg, MCPStdioServerConfig):
            streams_cm = _stdio_transport(cfg, lambda: conn.closed_event.set())
        elif isinstance(cfg, MCPHttpServerConfig):
            from mcp.client.streamable_http import (
                create_mcp_http_client,
                streamable_http_client,
            )

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

        async def _message_handler(message: IncomingMessage) -> None:
            await self._on_session_message(name, conn, message)

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

    def _effective_auth_store(self) -> ManagedAuthStore | None:
        """The injected session store, or one this manager owns and closes."""
        if self.auth_store is not None:
            return self.auth_store
        try:
            from local_operator.providers.auth_store import AuthStore

            self.auth_store = AuthStore()
            self._owns_auth_store = True
            return self.auth_store
        except Exception:  # pragma: no cover - environment dependent
            logger.debug("providers.auth_store unavailable", exc_info=True)
            return None

    def _build_oauth_auth(self, url: str, cfg: MCPServerConfig) -> OAuthClientProvider | None:
        """Build an ``OAuthClientProvider`` for configs with ``auth.type=oauth``."""
        auth = cfg.auth
        if auth is None or auth.type != "oauth":
            return None
        try:
            from mcp.client.auth import OAuthClientProvider

            from local_operator.mcp.auth import wire_oauth_auth

            return OAuthClientProvider(
                **wire_oauth_auth(url, cfg, store=self._effective_auth_store())
            )
        except Exception:
            logger.warning(
                "OAuth wiring unavailable for %r; connecting unauthenticated", url, exc_info=True
            )
            return None

    async def _on_session_message(
        self, name: str, conn: ServerConnection, message: IncomingMessage
    ) -> None:
        """Session message_handler: notifications + transport faults.

        ``tools/list_changed`` refreshes the server's tools; transport-level
        exceptions mark the connection closed and schedule a reconnect.
        """
        if isinstance(message, BaseException):
            if self._connections.get(name) is conn and not conn.closed_event.is_set():
                conn.closed_event.set()
                self._handle_disconnect(name)
            return
        if message.method == "notifications/tools/list_changed":
            # NEVER await a tools/list round trip inline here: the SDK invokes
            # this handler while holding its read loop (mcp/client/session.py
            # ~1430-1453), so an inline refresh deadlocks in-process servers.
            task = asyncio.get_running_loop().create_task(self.refresh_server_tools(name))
            self._notify_tasks.add(task)
            task.add_done_callback(self._notify_tasks.discard)

    async def _finish_pending(
        self, name: str, task: asyncio.Task[ServerConnection], epoch: int
    ) -> None:
        """Background continuation for a server still connecting at the gate."""
        try:
            conn = await task
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("MCP server %r failed to connect after the gate: %s", name, exc)
            # Re-fetch the waiter: a reload during the await may have swapped
            # it, and settling the stale one would strand the current waiters.
            _settle_future_error(self._connect_futures.get(name), exc)
            self._connect_futures.pop(name, None)
            self._tools_by_server.pop(name, None)  # drop the deferred slice
            self._unregister_origins(name)
            self._rebuild_agent_names()
            self._fire_tools_changed()
            return
        if self._disposed or epoch != self._epoch:
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
        self._log_first_connect_security(conn)
        self._register_tools(conn.name, conn.tools)
        future = self._connect_futures.pop(conn.name, None)
        if future is not None and not future.done():
            future.set_result(conn)
        watcher = asyncio.get_running_loop().create_task(self._watch_connection(conn.name, conn))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)

    def _log_first_connect_security(self, conn: ServerConnection) -> None:
        """WARNING surface for project-sourced stdio servers (MCP-12).

        Once per server per manager lifetime: name the contributing config
        file and the command being spawned. A committed ``mcp.json`` is
        trusted input — stdio entries run arbitrary commands — so the user
        must have seen exactly what this project is launching.
        """
        name = conn.name
        if name in self._security_logged:
            return
        cfg = conn.config
        if not isinstance(cfg, MCPStdioServerConfig):
            return
        source = self._sources.get(name, "")
        try:
            project_sourced = bool(source) and Path(source).is_relative_to(Path(self.cwd))
        except (ValueError, OSError):
            project_sourced = False
        if not project_sourced:
            return
        self._security_logged.add(name)
        logger.warning(
            "MCP: spawning project-configured stdio server %r: command=%r args=%r "
            "(configured by %s) — a project's mcp.json is trusted input; "
            "review it before opening a repo under a credentialed profile",
            name,
            cfg.command,
            list(cfg.args),
            source,
        )

    def _register_tools(self, name: str, tools: list[Tool]) -> None:
        """Build AgentTools for one server's tool list (live, not deferred)."""
        self._unregister_origins(name)
        self._tools_by_server[name] = [
            self._build_tool(name, tool, deferred=False) for tool in tools
        ]
        self._rebuild_agent_names()

    def _build_tool(
        self, server_name: str, tool: Tool | dict[str, Any], *, deferred: bool
    ) -> AgentTool:
        """Wrap one tool (SDK model or cached dict) with the manager call path.

        The tool is recorded under its stable origin key ``(server_name,
        original tool name)`` — never under its minted name — so reconnect
        ordering cannot flip ownership. Minted names (incl. collision
        suffixing) are resolved centrally in :meth:`_rebuild_agent_names`.
        """
        if isinstance(tool, dict):
            mcp_tool_name: str = tool.get("name", "") or ""
        else:
            mcp_tool_name = tool.name
        origin = (server_name, mcp_tool_name)

        async def _call(
            tool_call_id: str,
            args: dict[str, Any],
            signal: AbortSignal | None,
            on_update: Callable[[AgentToolUpdate], None] | None,
            context: ToolContext,
        ) -> ToolResult:
            return await self._execute_tool_call(
                server_name, mcp_tool_name, tool_call_id, args, signal, deferred=deferred
            )

        agent_tool = build_agent_tool(server_name, tool, _call)
        self._tool_by_origin[origin] = agent_tool
        self._meta_by_origin[origin] = {
            "server_name": server_name,
            "mcp_tool_name": mcp_tool_name,
            "deferred": deferred,
        }
        self._origins_by_server.setdefault(server_name, set()).add(origin)
        return agent_tool

    def _unregister_origins(self, server_name: str) -> None:
        """Drop every origin recorded for ``server_name`` (re-register/reload)."""
        origins = self._origins_by_server.pop(server_name, set())
        for origin in origins:
            self._tool_by_origin.pop(origin, None)
            self._meta_by_origin.pop(origin, None)

    def _rebuild_agent_names(self) -> None:
        """Mint collision-free tool names, deterministic by origin key.

        Two distinct origins can sanitize to the same agent name (e.g. server
        ``my-server`` + tool ``a_b`` and server ``my`` + tool ``server_a_b``
        both mint ``mcp__my_server_a_b``). The origin that sorts FIRST keeps
        the base name; each later colliding origin is suffixed ``_2``, ``_3``,
        ... and logged. Keying by origin (not registration order) means a
        reconnect or a tools/list_changed can never flip who owns a name.
        """
        self._tool_meta.clear()
        owners: dict[str, tuple[str, str]] = {}
        for origin in sorted(self._tool_by_origin):
            agent_tool = self._tool_by_origin[origin]
            base = create_mcp_tool_name(origin[0], origin[1])
            name = base
            if name in owners:
                suffix = 2
                while f"{base}_{suffix}" in owners:
                    suffix += 1
                name = f"{base}_{suffix}"
                logger.warning(
                    "MCP tool-name collision: %r/%r mints %r, already owned by %r/%r; using %r",
                    origin[0],
                    origin[1],
                    base,
                    owners[base][0],
                    owners[base][1],
                    name,
                )
            owners[name] = origin
            agent_tool.name = name
            meta: McpToolMeta = {**self._meta_by_origin.get(origin, {}), "agent_name": name}
            self._tool_meta[name] = meta

    async def _execute_tool_call(
        self,
        server_name: str,
        mcp_tool_name: str,
        tool_call_id: str,
        args: dict[str, Any],
        signal: AbortSignal | None,
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

        async def _call_once() -> CallToolResult:
            timeout_s = resolve_mcp_timeout_s(conn.config)
            if timeout_s is not None:
                return await conn.live_session.call_tool(
                    mcp_tool_name, outbound, read_timeout_seconds=timeout_s
                )
            return await conn.live_session.call_tool(mcp_tool_name, outbound)

        try:
            result = await self._race_abort(_call_once(), signal)
        except asyncio.CancelledError:
            raise  # abort stays abort (MCP-16): never converted to an error result
        except Exception as exc:
            if is_retriable_connection_error(exc):
                # One reconnect + one retry at the call site (established policy).
                new_conn = await self._reconnect_for_call(server_name)
                if new_conn is not None:
                    conn = new_conn
                    try:
                        result = await self._race_abort(_call_once(), signal)
                    except asyncio.CancelledError:
                        raise  # abort stays abort
                    except Exception as retry_exc:
                        return self._error_result(tool_call_id, tool_label, retry_exc)
                else:
                    return self._error_result(tool_call_id, tool_label, exc)
            else:
                return self._error_result(tool_call_id, tool_label, exc)
        return format_mcp_result(result, tool_call_id, tool_label)

    async def _race_abort(
        self, coro: Coroutine[Any, Any, _RaceT], signal: AbortSignal | None
    ) -> _RaceT:
        """Run ``coro`` racing the abort signal; abort wins with cancellation.

        The call is wrapped in a task; a racing ``signal.wait()`` task decides
        the winner. When the abort lands first the work task is cancelled and
        this method raises ``asyncio.CancelledError`` — real cancellation,
        which the call path propagates instead of mapping to a tool result
        ("abort stays abort", MCP-16). A ``None`` signal runs the coroutine
        inline.
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
                raise asyncio.CancelledError("aborted")
            return task.result()
        finally:
            abort_task.cancel()
            with suppress(asyncio.CancelledError):
                await abort_task

    def _schema_parts(
        self, conn: ServerConnection, mcp_tool_name: str
    ) -> tuple[dict[str, Any], list[str], bool | dict[str, Any] | None]:
        """Extract (properties, required, additionalProperties) for arg hygiene."""
        for tool in conn.tools:
            if tool.name != mcp_tool_name:
                continue
            schema = tool.input_schema or {}
            properties = schema.get("properties")
            required = schema.get("required")
            additional = schema.get("additionalProperties")
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
    async def _list_all_tools(session: McpSession) -> list[Tool]:
        """Follow ``tools/list`` pagination (nextCursor) to completion."""
        from mcp.types import PaginatedRequestParams

        tools: list[Tool] = []
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

    def get_tool_meta(self, tool_name: str) -> McpToolMeta | None:
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

    def _abandon_reconnect(self, name: str, reason: str) -> None:
        """Auto-reconnect is over for ``name``: fail waiters instead of hanging.

        A deferred execute parked on ``_connect_futures[name]`` must get a real
        ``McpConnectionError`` when the breaker trips (MCP-08); otherwise it
        awaits a future nobody will ever settle.
        """
        future = self._connect_futures.pop(name, None)
        _settle_future_error(
            future, McpConnectionError(f"MCP server {name!r} unavailable: {reason}")
        )

    def _schedule_reconnect(self, name: str) -> None:
        """Queue a backoff-delayed reconnect unless the breaker is tripped."""
        if self._disposed:
            return
        if name in self._reconnect_suspended:
            self._abandon_reconnect(name, "auto-reconnect suspended (breaker tripped)")
            return
        if not self._record_reconnect_attempt(name):
            logger.warning(
                "MCP reconnect breaker tripped for %r: >%d attempts in %ds; "
                "auto-reconnect suspended (manual reconnect resets)",
                name,
                RECONNECT_BURST_LIMIT,
                int(RECONNECT_BURST_WINDOW_S),
            )
            self._abandon_reconnect(
                name,
                f"reconnect breaker tripped (>{RECONNECT_BURST_LIMIT} in "
                f"{int(RECONNECT_BURST_WINDOW_S)}s)",
            )
            return
        # Backoff ladder position is independent of the breaker window (MCP-07).
        index = self._backoff_index.get(name, 0)
        delay = RECONNECT_BACKOFF_S[min(index, len(RECONNECT_BACKOFF_S) - 1)]
        self._backoff_index[name] = index + 1
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
        # Reuse the waiter installed by _handle_disconnect: replacing it
        # would strand any deferred execute parked on it (MCP-08).
        future = self._connect_futures.get(name)
        if future is None or future.done():
            future = asyncio.get_running_loop().create_future()
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
        # Success resets the backoff LADDER only; the breaker window stays
        # intact so a flapping server still trips (MCP-07).
        self._backoff_index[name] = 0
        self._fire_tools_changed()

    async def _reconnect_for_call(self, name: str) -> ServerConnection | None:
        """Synchronous reconnect for the call-site retry (no backoff wait).

        Guarded like every other reconnect path (MCP-06): disposed/epoch
        mismatch and a tripped breaker short-circuit BEFORE reconnecting, and
        the attempt is recorded in the breaker window so a call-site retry on
        a dead server counts against the burst budget instead of resurrecting
        forever after ``disconnect_all``.
        """
        if self._disposed:
            return None
        cfg = self._configs.get(name)
        if cfg is None:
            return None
        if name in self._reconnect_suspended:
            return None
        if not self._record_reconnect_attempt(name):
            logger.warning("MCP reconnect breaker tripped for %r (call-site attempt)", name)
            return None
        epoch = self._epoch
        await self._teardown_connection(name)
        try:
            conn = await self._connect_server(name, cfg)
        except Exception as exc:
            logger.warning("MCP call-site reconnect failed for %r: %s", name, exc)
            return None
        if epoch != self._epoch or self._disposed:
            # disconnect_all ran while we were connecting: never resurrect.
            if conn.stack is not None:
                with suppress(Exception):
                    await conn.stack.aclose()
            return None
        self._register_connection(conn)
        self._backoff_index[name] = 0
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
    "RECONNECT_BACKOFF_S",
    "RECONNECT_BURST_LIMIT",
    "RECONNECT_BURST_WINDOW_S",
    "STARTUP_GATE_MS",
    "McpConnectionError",
    "McpLoadResult",
    "McpManager",
    "ServerConnection",
    "build_cmd_exe_argv",
    "build_stdio_argv",
    "escape_cmd_batch_arg",
    "escape_cmd_quoted_interior",
    "resolve_mcp_timeout_s",
    "stdio_start_new_session",
    "win32_process_target",
]
