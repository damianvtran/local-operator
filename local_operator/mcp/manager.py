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
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar

from local_operator.ansi import strip_control_sequences
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.mcp.auth import McpAuthRequiredError
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


#: What every configured server's error says when the SDK is absent. A module
#: constant, not an inline string, because the session layer collapses these N
#: identical entries back into ONE report and compares against this exact value
#: rather than sniffing the text — the alternative is a substring match that goes
#: quietly wrong the day the wording changes.
MCP_SDK_MISSING_ERROR = missing_extra_error("mcp", "Connecting to MCP servers")

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


def _unwrap_auth_required(exc: BaseException) -> BaseException:
    """Surface a :class:`McpAuthRequiredError` wrapped in an ``ExceptionGroup``.

    The streamable-HTTP transport runs its auth flow inside an anyio TaskGroup,
    which wraps any exception the redirect handler raises in an
    ``ExceptionGroup`` — and that group can itself be nested inside the
    ``ClientSession`` task group's own group, so the auth error may arrive at
    ANY depth. Callers that need to RECOGNISE an auth requirement (the startup
    toast, the reconnect breaker, ``/mcp login``) would otherwise see only
    ``"unhandled errors in a TaskGroup"`` and treat a recoverable grant as an
    opaque transport failure. This walks the group's LEAVES (``subgroup``
    preserves nesting structure, so ``matches.exceptions[0]`` can be another
    group) and returns the first auth error found, else the original exception
    unchanged.
    """
    if isinstance(exc, McpAuthRequiredError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        matches = exc.subgroup(McpAuthRequiredError)
        if matches is not None:
            # A connect raises at most ONE auth error, so the first leaf is the
            # whole story; flattening keeps the re-raise a single clean type.
            stack: list[BaseException] = [matches]
            while stack:
                candidate = stack.pop()
                if isinstance(candidate, McpAuthRequiredError):
                    return candidate
                if isinstance(candidate, BaseExceptionGroup):
                    stack.extend(candidate.exceptions)
    return exc


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


# ---------------------------------------------------------------------------
# Child output containment
# ---------------------------------------------------------------------------

#: Environment that asks a stdio child not to decorate its output.
#:
#: This is the BRACES, not the belt, and the measurement says so plainly.
#: MEASURED 2026-08-11 against ``workspace-mcp`` 1.23.1 — the server the defect
#: was reported on — with its stderr on a real PTY:
#:
#: ===============================  ==========  =======
#: child stderr                     bytes       artwork
#: ===============================  ==========  =======
#: PTY, ``TERM=xterm-256color``     3289        yes
#: PTY, all of ``CHILD_QUIET_ENV``  2391        yes
#: PIPE                            **303**      **no**
#: ===============================  ==========  =======
#:
#: So these variables do NOT stop this server drawing its logo: it decides on
#: ``isatty()``, and only redirecting the stream (see ``_stdio_transport``)
#: actually silences it. What they buy is the ~900 bytes of colour escapes
#: between rows one and two, which would otherwise be written into the log
#: file and corrupt ``less``/``tail`` there instead of on the frame.
#:
#: Kept for that, and because the next third-party server will differ — one
#: that renders unconditionally is exactly the case the stream redirect
#: handles and the environment does not, and vice versa.
#:
#: EVERY ENTRY HERE TURNS COLOUR OFF BY ITS OWN CONVENTION, and that is a
#: harder rule than it looks. ``FORCE_COLOR`` and ``CLICOLOR_FORCE`` were in
#: this dict set to ``"0"``, which reads like an opt-out and is not one:
#: force-color.org senses PRESENCE, so Rich takes ``FORCE_COLOR=0`` as "yes, I
#: am a terminal" (``rich/console.py``: ``if force_color is not None: return
#: force_color != ""``). MEASURED on a pipe — where the child had already
#: given up on colour — adding ``FORCE_COLOR=0`` to the rest of this dict
#: turned ``b'LOGO\n'`` back into ``b'\x1b[1mLOGO\x1b[0m\n'`` for a server
#: whose config restores ``TERM``; ``test_the_quiet_env_actually_silences_rich``
#: keeps that measurement running. ``CLICOLOR_FORCE=0`` is inert in Rich by the
#: same measurement and by bixense's spec (force only when non-zero), but it is
#: presence-sensed elsewhere and buys nothing here, so it goes with it.
#: Both are gone, and leaving them UNSET is the whole of the fix: this dict is
#: not merged into the operator's own environment but into
#: ``get_default_environment()``, which copies an allowlist
#: (``DEFAULT_INHERITED_ENV_VARS``: ``HOME``, ``LOGNAME``, ``PATH``, ``SHELL``,
#: ``TERM``, ``USER``) and nothing else, so a ``FORCE_COLOR`` in the shell that
#: launched us cannot reach a server either. What is left here is only opt-OUT
#: switches: ``NO_COLOR`` (https://no-color.org, presence disables),
#: ``TERM=dumb``, ``CLICOLOR=0`` (bixense.com/clicolors), and ``PY_COLORS=0``.
#: Before adding another, check that its documented sense is "off" and not
#: "force" — and that its off value is a VALUE and not an absence.
#:
#: A server that wants any of them back sets it in its config ``env``, which is
#: merged last and wins.
CHILD_QUIET_ENV: dict[str, str] = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "CLICOLOR": "0",
    "PY_COLORS": "0",
}

#: How long teardown waits for the stderr pump after the child has exited. The
#: pipe is normally at EOF already and this returns instantly; the bound only
#: bites when a surviving descendant still holds the write end, and there is
#: nothing more of the SERVER's own output to wait for in that case. Kept
#: short even though ``disconnect_all`` now closes connections concurrently:
#: any single stdio server still pays it inline on ``/mcp reconnect`` and
#: ``disconnect_server``, where nothing overlaps it.
STDERR_DRAIN_GRACE_S = 0.25

#: Grace a stdio server gets at each rung of teardown (stdin EOF, then
#: SIGTERM, then SIGKILL) before escalation to the next rung. The wait is
#: event-driven — ``process.wait()`` wakes the moment the child exits — so a
#: prompt server pays nothing and only a stubborn one sits out a rung. The
#: value preserves the budget of the 0.1 s polling loop it replaced (20 ticks
#: per rung); what changed is the wake latency, not the patience.
STDIO_EXIT_GRACE_S = 2.0

#: Bound on closing ONE connection's exit stack (see
#: :meth:`McpManager._teardown_connection`). Bounded at all because a remote
#: transport's close is network I/O — the streamable-HTTP transport DELETEs
#: its session on an HTTP client whose connect timeout alone is 30 s — and a
#: dead network must not be able to hold quit hostage for that long. Five
#: seconds matches the session's other dispose bounds (turn abort, browser
#: close, title flush): comfortably above any healthy close, and a pause a
#: person will sit through once rather than a hang they will kill -9.
CONNECTION_TEARDOWN_TIMEOUT_S = 5.0

#: Stderr lines retained per stdio server for the failure report. A Python
#: traceback plus the banner that preceded it fits inside this; a chatty server
#: cannot pin more than this many lines of memory per connection.
STDERR_TAIL_LINES = 50

#: Longest stderr line kept whole. A server that writes without ever emitting a
#: newline must not turn into an unbounded log record.
STDERR_LINE_LIMIT = 2_000

#: How many trailing stderr lines are quoted into a failed connect's error
#: text, and how many characters of them. That message is rendered whole by the
#: transcript notice and by ``/mcp`` — only the toast truncates — so a server
#: whose last words are a 2000-character line must not become a wall of text
#: where a reason belongs. The full tail is in the log.
STDERR_QUOTED_LINES = 2
STDERR_QUOTED_CHARS = 200


class McpServerStderr:
    """One stdio child's stderr: into the session log, never onto the terminal.

    A stdio MCP server speaks protocol on stdout, so the SDK captures that. Its
    stderr used to be inherited (``stderr=None`` on the spawn), which put every
    byte the child logged straight onto the file descriptor Textual is painting
    — reproduced as 2508 bytes of Rich artwork tearing the boot splash in half.

    Discarding it is not an option either: a missing credential, a bad config
    or a crash on startup is reported on exactly this stream, and it is the
    only answer a user has to "why did my server not start". So it follows the
    convention the rest of the app already uses for output that cannot go on
    screen (``local_operator.logger.file_logging``): every line becomes a log
    record, under a per-server logger name so ``grep`` finds one server's
    output, and the tail is retained so a failure can be reported WITH its
    cause instead of as a bare transport error.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        # Per-server child logger: a record's own name says which server wrote
        # it, and `logging` levels can then silence one noisy server without
        # silencing the manager's own diagnostics.
        self._log = logging.getLogger(f"local_operator.mcp.server.{name}")
        self._tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        self._reported = False

    def feed(self, line: str) -> None:
        """Record one line of the child's stderr."""
        # Stripped: the child may ignore CHILD_QUIET_ENV, and a raw CSI in the
        # log file corrupts `less`/`tail` the same way it corrupted the frame.
        text = strip_control_sequences(line).rstrip()
        if not text:
            return
        if len(text) > STDERR_LINE_LIMIT:
            text = text[:STDERR_LINE_LIMIT] + " …[truncated]"
        self._tail.append(text)
        # INFO, not WARNING: an ordinary server's startup chatter is not a
        # problem, and the TUI's log file is opened at INFO
        # (`configure_cli_logging`), so this is visible without being alarming.
        # The failure path re-reports the tail at ERROR below.
        self._log.info("%s", text)

    @property
    def captured(self) -> bool:
        """Whether the child said anything at all on stderr."""
        return bool(self._tail)

    def tail_text(self) -> str:
        """The retained tail as one block of text."""
        return "\n".join(self._tail)

    def quoted_tail(self, lines: int = STDERR_QUOTED_LINES) -> str:
        """The last few lines, joined and bounded, for a one-line error message."""
        text = " / ".join(list(self._tail)[-lines:])
        if len(text) <= STDERR_QUOTED_CHARS:
            return text
        return text[:STDERR_QUOTED_CHARS].rstrip() + "…"

    def report_failure(self, reason: str) -> None:
        """Log the retained tail at ERROR, once, when the server failed.

        Once: the connect path and the transport teardown both notice the same
        dead child, and one failure deserves one report.
        """
        if self._reported or not self._tail:
            return
        self._reported = True
        self._log.error(
            "MCP server %r %s; its last %d stderr line(s) follow:\n%s",
            self.name,
            reason,
            len(self._tail),
            self.tail_text(),
        )

    def explain(self, exc: Exception) -> Exception:
        """``exc``, restated with the child's own last words when it has any.

        A stdio server that dies during the handshake surfaces as a transport
        error whose text ("") says nothing at all, while the reason it died is
        sitting in the tail. This is what puts that reason in
        ``McpStartupOutcome.failures`` and therefore in the TUI's notice.
        """
        if not self._tail:
            return exc
        detail = str(exc).strip() or type(exc).__name__
        return McpConnectionError(f"{detail}: {self.quoted_tail()}")


@asynccontextmanager
async def _stdio_transport(
    cfg: MCPStdioServerConfig,
    on_close: Callable[[], None],
    stderr_log: McpServerStderr,
) -> AsyncIterator[TransportStreams]:
    """Spawn an MCP stdio server and pump newline-delimited JSON-RPC.

    An SDK-shaped transport context manager (yields ``(read_stream,
    write_stream)``) built directly on ``anyio.open_process`` so we control
    the platform spawn rules the SDK hardcodes differently (see
    :func:`stdio_start_new_session`). ``on_close`` fires once when the
    connection can no longer carry traffic (process exit or pump failure).
    ``stderr_log`` receives everything the child writes to stderr; see
    :class:`McpServerStderr` for why it may not reach the terminal.
    """
    import anyio
    import mcp.types as mcp_types
    from mcp.client.stdio import get_default_environment
    from mcp.shared.message import SessionMessage

    argv = build_stdio_argv(cfg.command, list(cfg.args))
    env = get_default_environment() | CHILD_QUIET_ENV | dict(cfg.env or {})
    cwd = cfg.cwd or None

    # stderr on a PIPE, never inherited. `stderr=None` handed the child the
    # parent's own fd 2, so a server's startup banner landed on top of the
    # Textual frame (the reported defect). The pump below drains it into the
    # session log; leaving the pipe unread would eventually block the child on
    # a full 64 KiB buffer, which is why the pump is not optional.
    kwargs: dict[str, Any] = {"env": env, "stderr": subprocess.PIPE, "cwd": cwd}
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

    stderr_drained = anyio.Event()

    async def _stderr_pump() -> None:
        """Drain the child's stderr into ``stderr_log``, line by line.

        Deliberately does NOT fire ``on_close``: stderr reaching EOF says
        nothing about whether the protocol channel still carries traffic, and a
        server that closes stderr early would otherwise be treated as dead.
        """
        from anyio.streams.text import TextReceiveStream

        stderr = process.stderr
        if stderr is None:  # pragma: no cover - PIPE always yields one
            stderr_drained.set()
            return
        text_stream = TextReceiveStream(stderr, encoding="utf-8", errors="replace")
        try:
            buffer = ""
            async for chunk in text_stream:
                lines = (buffer + chunk).split("\n")
                buffer = lines.pop()
                # A single unterminated line must not grow without bound: a
                # server printing a progress bar with \r and no \n would
                # otherwise accumulate in this buffer for the whole session.
                if len(buffer) > STDERR_LINE_LIMIT:
                    lines.append(buffer)
                    buffer = ""
                for line in lines:
                    stderr_log.feed(line)
            if buffer:
                stderr_log.feed(buffer)
        except Exception:
            logger.debug("stdio stderr pump ended for %r", cfg.command, exc_info=True)
        finally:
            stderr_drained.set()

    async def _stop() -> None:
        """Close stdin, give the server a grace window, then kill the tree.

        EVENT-DRIVEN: each rung awaits ``process.wait()`` under a deadline
        rather than polling ``returncode`` on a 0.1 s tick. The polling loop
        this replaced charged up to a full tick of pure latency per rung on
        every quit — a child that exited 1 ms after a poll still cost 99 ms —
        and teardown is exactly the path where that latency is user-visible
        (the terminal is already released; the user is watching the prompt).

        CANCELLATION-SAFE on purpose: ``_teardown_connection`` bounds the
        stack close, and the cancel it delivers on timeout lands on whichever
        await is current in here. Absorbing that cancel without killing the
        child would leak the process past the session — on Linux
        ``start_new_session`` detaches it from our process group entirely (see
        :func:`stdio_start_new_session`), so it would not even die with us.
        Kill first, then let the cancellation propagate.
        """
        try:
            stdin = process.stdin
            if stdin is not None:
                with suppress(Exception):
                    await stdin.aclose()
            with anyio.move_on_after(STDIO_EXIT_GRACE_S):
                await process.wait()
                return
            for stop_process in (process.terminate, process.kill):
                with suppress(Exception):
                    stop_process()
                with anyio.move_on_after(STDIO_EXIT_GRACE_S):
                    await process.wait()
                    return
        except asyncio.CancelledError:
            # The bounded teardown gave up on waiting; the child must not
            # outlive the session that owns it. ``kill`` is synchronous (one
            # signal), so this cannot itself block the cancellation.
            with suppress(Exception):
                process.kill()
            raise

    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_stdout_pump)
            tg.start_soon(_stdin_pump)
            tg.start_soon(_stderr_pump)
            try:
                yield read_stream, write_stream
            finally:
                # Sampled BEFORE `_stop` closes stdin, because closing stdin is
                # how we ASK the server to leave: an exit status observed after
                # that is a response to the request, not a fault. Reading it
                # afterwards manufactured a failure on every clean quit — a
                # server that treats stdin EOF as an error exits non-zero on
                # being asked politely, and on Windows `Popen.terminate` is
                # `TerminateProcess(handle, 1)` so EVERY kill reports status 1
                # (found in review, reproduced).
                #
                # Non-zero rather than positive: a status we did not cause is
                # worth reporting whichever sign it has. A child SIGKILLed by
                # the OOM killer arrives as -9 and is exactly the death whose
                # last stderr lines someone will come looking for.
                died_unbidden = process.returncode
                with suppress(Exception):
                    await _stop()
                # Let the stderr pump finish before the cancel below kills it.
                # Without this wait a server that died DURING the handshake
                # loses the last lines of its own stderr — exactly the ones
                # saying why: `_stop` observes the exit the moment it happens,
                # which can be BEFORE the pump has consumed what is still
                # sitting in the pipe, and `_connect_server` quotes that tail
                # into the error the user is shown.
                #
                # Bounded because EOF is not guaranteed: the write end is held
                # by every process that inherited it, so a server that spawned
                # a grandchild (`uvx` doing the real work in a subprocess, for
                # one) keeps the pipe open past its own death, and MEASURED in
                # review that teardown then runs to the full bound (0.101 s
                # plain child vs 0.602 s with a surviving grandchild). Kept
                # short for that reason: by this point the direct child has
                # exited, so anything still holding the pipe is a descendant
                # whose output was never the server's own.
                with anyio.move_on_after(STDERR_DRAIN_GRACE_S):
                    await stderr_drained.wait()
                if died_unbidden is not None and died_unbidden != 0:
                    stderr_log.report_failure(f"exited with status {died_unbidden}")
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
        # Session-installed sink for model-visible MCP breaker incidents.
        self.on_incident: Callable[[str, str], None] | None = None
        # UI-installed sink fired when a server needs an OAuth login. The
        # startup toast only covers failures that land INSIDE the 250 ms gate;
        # HTTP OAuth servers connect AFTER it, so their auth failures (and
        # mid-session expiry) need this dedicated hook to reach the user as a
        # toast. Signature: (server_name, message).
        self.on_auth_required: Callable[[str, str], None] | None = None
        # UI/session-installed sink fired ONCE per discovery round when every
        # server that missed the 250 ms startup gate has reached a terminal
        # state (connected or failed). It exists because the boot report is a
        # SNAPSHOT taken at the gate: OAuth HTTP servers do PRM/ASM discovery and
        # a possible token refresh before their transport opens, so they almost
        # always miss the gate and are still connecting when the snapshot is
        # taken. A single server that fails FAST would otherwise flip the report
        # to "N of M up — failed: X" while the slow successes are still in
        # flight, reporting the momentary authed count as if it were final. This
        # fires when the round actually settles so the front end can re-report
        # the COMBINED outcome once. Signature: (). Never fires when no server
        # was deferred (the gate snapshot was already final).
        self.on_startup_settled: Callable[[], None] | None = None
        # Combined per-server startup failures for the CURRENT round, accumulated
        # across both the gate pass and the background continuations, so the
        # settled outcome names every failure and not just the ones fast enough
        # to lose the race to the gate. Reset at the start of each round; a
        # server that later connects is cleared from it.
        self._startup_failures: dict[str, str] = {}
        # Servers deferred past the gate this round and not yet settled. When it
        # drains, ``on_startup_settled`` fires. Emptiness is also how the front
        # end tells "the boot snapshot was final" from "still connecting".
        self._startup_deferred: set[str] = set()
        # Servers already toasted for an auth requirement, so a dead grant that
        # a tool call keeps retrying does not re-raise the toast on every
        # attempt. Cleared per server when it connects again.
        self._auth_toasted: set[str] = set()
        # Tool-name collision state keyed by stable origin key (MCP-09):
        # (server name, original tool name), never registration order.
        self._tool_meta: dict[str, McpToolMeta] = {}
        self._tool_by_origin: dict[tuple[str, str], AgentTool] = {}
        self._meta_by_origin: dict[tuple[str, str], McpToolMeta] = {}
        self._origins_by_server: dict[str, set[tuple[str, str]]] = {}
        # First-connect security surface (MCP-12): one warning per server.
        self._security_logged: set[str] = set()
        # Discovered OAuth endpoints per server URL, populated by the proactive
        # refresh in ``_connect_server`` and handed to the provider so an
        # in-flow (mid-session) refresh targets the real token endpoint instead
        # of the SDK's ``<server_base>/token`` guess. Keyed by URL because the
        # provider is rebuilt per connect while discovery is per server.
        self._oauth_endpoints: dict[str, Any] = {}
        # Loopback flows of in-flight/last OAuth grants, keyed by server URL.
        # An abandoned grant crosses the transport as a raw CancelledError;
        # this is how the connect error path finds the flow whose
        # ABANDONED_GRANTS ledger entry re-voices it.
        self._oauth_flows: dict[str, Any] = {}
        self._epoch = 0
        self._disposed = False

    # --- public API --------------------------------------------------------

    def set_on_tools_changed(self, callback: ToolsChangedCallback | None) -> None:
        """Install the callback fired whenever the tool list changes."""
        self._on_tools_changed = callback

    @property
    def on_tools_changed(self) -> ToolsChangedCallback | None:
        """The installed callback, so a second consumer can CHAIN it.

        The slot is deliberately single: the composition root owns it and uses
        it to keep the session's tool inventory in step. A front end that also
        wants to hear about connect/disconnect (the TUI's live MCP counter)
        therefore has to read the incumbent and call it from its own wrapper —
        reaching for ``_on_tools_changed`` to do that would make the inventory
        merge depend on a private attribute, and silently dropping it would
        leave the agent's tool list frozen at boot.
        """
        return self._on_tools_changed

    def get_tools(self) -> list[AgentTool]:
        """All registered tools, sorted by name for stability."""
        tools: list[AgentTool] = []
        for server_tools in self._tools_by_server.values():
            tools.extend(server_tools)
        return sorted(tools, key=lambda tool: tool.name)

    def get_server_tools(self, name: str) -> list[AgentTool]:
        """One server's registered tools, sorted without exposing internals.

        Lazy MCP discovery uses this public view to render ``mcp://`` resources
        and activate selected schemas. Returning a copy prevents resolver code
        from mutating the manager's live inventory.
        """
        return sorted(self._tools_by_server.get(name, ()), key=lambda tool: tool.name)

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

    async def connect_configured_server(
        self, name: str, *, timeout_ms: float | None = None, interactive: bool = True
    ) -> ServerConnection:
        """Connect one configured server without touching unrelated entries.

        Explicit CLI login must wait for one OAuth exchange rather than start
        every configured server through the session's 250 ms startup gate.
        ``timeout_ms`` lets that interactive flow outlive the normal 30-second
        request budget without weakening ordinary session connections.

        ``interactive`` defaults to ``True`` because this is the explicit login
        path: it may open a browser to complete a grant. Ordinary startup and
        reconnects go through ``_connect_round``/``_reconnect``, which stay
        non-interactive.
        """
        configs, sources = load_all_mcp_configs(self.cwd)
        cfg = configs.get(name)
        if cfg is None:
            raise McpConnectionError(f"MCP server {name!r} is not configured")
        errors = validate_server_config(name, cfg)
        if errors:
            raise McpConnectionError("; ".join(errors))

        # The PRISTINE config is what persists: the login-widened timeout below
        # must scope to the one interactive connect, or every later tool call
        # and reconnect on this server would inherit a 10-minute request budget
        # (ServerConnection.config and _configs both feed resolve_mcp_timeout_s).
        self._configs[name] = cfg
        self._sources[name] = sources[name]
        self._disposed = False
        connect_cfg = (
            cfg.model_copy(update={"timeout": timeout_ms}) if timeout_ms is not None else cfg
        )
        conn = await self._connect_server(name, connect_cfg, interactive=interactive)
        # The live connection must carry the pristine config too — tool calls
        # read their timeout from conn.config, not from _configs.
        conn.config = cfg
        # The widened budget also became the SESSION's default read timeout
        # (ClientSession(read_timeout_seconds=...) baked in at connect), which
        # requests WITHOUT an explicit per-call timeout — tools/list refreshes
        # — would inherit for the session's whole life. Reset it to the
        # pristine config's budget. Private attribute, set under suppress: the
        # SDK offers no setter, and a rename would merely leave the widened
        # default in place (the pre-fix behavior), never break the login.
        with suppress(Exception):
            conn.live_session._session_read_timeout_seconds = (  # type: ignore[attr-defined]
                resolve_mcp_timeout_s(cfg)
            )
        # An explicit login is the documented recovery from an auth-suspended
        # breaker (see _reconnect's McpAuthRequiredError arm): clear the breaker
        # state so the server's NEXT disconnect auto-reconnects again instead of
        # being abandoned by the suspension this login just resolved.
        self._reconnect_history.pop(name, None)
        self._reconnect_suspended.discard(name)
        self._backoff_index.pop(name, None)
        self._register_connection(conn)
        # A mid-session login adds tools the session booted without; notify
        # subscribers exactly like a successful reconnect does. A no-op when no
        # callback is installed (the CLI login path).
        self._fire_tools_changed()
        return conn

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
        # Fresh accumulators for this round: the settled outcome is built from
        # these, not from the gate snapshot alone (see on_startup_settled).
        self._startup_failures = {}
        self._startup_deferred = set()

        result = McpLoadResult()
        if configs and not _sdk_available():
            # One actionable line instead of the same opaque
            # "No module named 'mcp'" repeated per configured server. The
            # session treats MCP as enrichment, so this surfaces as a warning
            # and the turn proceeds with zero MCP tools. Every server carries the
            # SAME message, which is what lets the session layer recognise the
            # cause and report it once (see MCP_SDK_MISSING_ERROR).
            result.errors.update({name: MCP_SDK_MISSING_ERROR for name in configs})
            return result

        tasks: dict[str, asyncio.Task[ServerConnection]] = {}
        for name, cfg in configs.items():
            errors = validate_server_config(name, cfg)
            if errors:
                message = "; ".join(errors)
                result.errors[name] = message
                self._startup_failures[name] = message
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
            except McpAuthRequiredError as exc:
                # Actionable, not raw: the startup toast is where a user first
                # learns a server needs a login, and "run /mcp login <name>" is
                # the one thing they can do about it.
                message = self._auth_required_text(name, exc)
                result.errors[name] = message
                self._startup_failures[name] = message
                logger.info("MCP server %r needs authorization: %s", name, exc)
                waiter = self._connect_futures.pop(name, None)
                _settle_future_error(waiter, exc)
                continue
            except Exception as exc:
                result.errors[name] = str(exc)
                self._startup_failures[name] = str(exc)
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
            # Deferred past the gate: its terminal state is not yet known, so it
            # joins the settle set. When this set drains (every deferred server
            # connected or failed), the round is settled and the front end can
            # re-report the combined outcome instead of the gate snapshot.
            self._startup_deferred.add(name)
            # Still pending at the gate: defer from cache, or contribute nothing.
            cached = self.tool_cache.get(name) if self.tool_cache is not None else None
            if cached:
                self._unregister_origins(name)
                self._tools_by_server[name] = [
                    self._build_tool(name, entry, deferred=True)
                    for entry in cached
                    if self._tool_is_enabled(name, self._raw_tool_name(entry))
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
        except McpAuthRequiredError as exc:
            logger.info("Manual MCP reconnect needs authorization for %r", name)
            self._fire_auth_required(name, exc)
            return None
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
        # CONCURRENTLY, not one at a time: teardown is per-connection I/O — a
        # remote session-terminate round trip, a child process exit plus its
        # stderr drain grace — and paying it serially made quit latency grow
        # with the server count (measured 1.6 s across seven servers where the
        # slowest single one needed 0.95 s). The teardowns share no state:
        # each pops its own entry from ``_connections`` and closes its own
        # stack, so the only thing serial order bought was the wait.
        names = list(self._connections)
        if names:
            await asyncio.gather(
                *(self._teardown_connection(name) for name in names),
                return_exceptions=True,
            )
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

    async def _connect_server(
        self, name: str, cfg: MCPServerConfig, *, interactive: bool = False
    ) -> ServerConnection:
        """Open transport + session, initialize, list tools, update cache.

        This is the seam tests override: it returns a :class:`ServerConnection`
        without touching a real server.

        ``interactive`` controls OAuth: an ordinary startup or auto-reconnect
        passes ``False`` and must never open a browser — an unrefreshable grant
        surfaces as an actionable :class:`McpAuthRequiredError` instead. Only
        an explicit ``/mcp login`` (``connect_configured_server``) runs with
        ``True``.

        Before opening the transport, an OAuth server gets a PROACTIVE refresh
        (:func:`~local_operator.mcp.auth.ensure_mcp_oauth_fresh`): it spends a
        stored refresh token against the DISCOVERED token endpoint, race-free
        across concurrently starting sessions, so a day-old access token never
        forces a browser grant on startup.
        """
        timeout_s = resolve_mcp_timeout_s(cfg)
        await self._ensure_oauth_fresh(name, cfg)
        stack = AsyncExitStack()
        # One collector per connect ATTEMPT, so a retry never quotes the
        # previous attempt's stderr as this one's reason. Made unconditionally:
        # a remote transport spawns nothing, so its collector stays empty and
        # both methods below are then no-ops by construction rather than by a
        # branch someone has to keep in step with the transport list.
        stderr_log = McpServerStderr(name)
        try:
            conn = await self._open_transport_and_session(
                stack, name, cfg, timeout_s, stderr_log, interactive=interactive
            )
            tools = await self._list_all_tools(conn.live_session)
        except BaseException as exc:
            # Tear down FIRST: for a stdio child this stops the process and
            # drains its stderr so the tail is complete by the time ``explain``
            # quotes it; for the streamable-HTTP transport this is where the
            # anyio task group's scope exits, which is ITSELF where a failure
            # raised inside the auth flow surfaces — anyio delivers that
            # failure to the awaiting task as a CancelledError first, and the
            # group's ``__aexit__`` then re-raises it as an ExceptionGroup.
            close_exc: BaseException | None = None
            try:
                await stack.aclose()
            except BaseException as ce:  # noqa: BLE001 — examined below, never lost
                close_exc = ce
            # A GENUINE external cancellation (dispose/reload/esc) keeps its
            # priority even when the teardown surfaced a grouped auth error:
            # the task itself was asked to cancel (``cancelling() > 0``), and
            # converting that into an auth failure would toast + suspend a
            # server for what was actually the user leaving. anyio's own
            # internal delivery — the auth flow failing inside the transport's
            # task group — raises CancelledError WITHOUT marking this task as
            # cancelling, which is exactly what lets the two be told apart.
            current = asyncio.current_task()
            externally_cancelled = (
                current is not None
                and current.cancelling() > 0
                and (
                    isinstance(exc, asyncio.CancelledError)
                    # The cancel can also land DURING ``stack.aclose()`` — then
                    # it rides ``close_exc`` while ``exc`` carries the original
                    # failure, and converting would swallow the pending
                    # cancellation (F12).
                    or isinstance(close_exc, asyncio.CancelledError)
                )
            )
            if externally_cancelled:
                if isinstance(exc, asyncio.CancelledError):
                    raise
                assert isinstance(close_exc, asyncio.CancelledError)
                raise close_exc
            # An abandoned grant (browser closed, consent left unanswered,
            # idle guard fired) arrives as a bare CancelledError with NO
            # cancelling count: the flow raises it raw precisely because the
            # SDK's transport swallows ordinary auth exceptions (see
            # ``LoopbackAuthFlow.callback_handler``). The ABANDONED_GRANTS
            # ledger is the channel the message actually travelled on; the
            # externally_cancelled guard above has already kept a REAL task
            # cancellation's priority, so a recorded abandonment here can be
            # re-voiced as the receipt the user reads.
            if isinstance(exc, asyncio.CancelledError):
                from local_operator.mcp.auth import (
                    ABANDONED_GRANTS,
                    McpLoginCancelledError,
                )

                url = getattr(cfg, "url", None)
                flow = self._oauth_flows.pop(url, None) if isinstance(url, str) else None
                if flow is not None and ABANDONED_GRANTS.pop(flow):
                    raise McpLoginCancelledError(
                        "the browser never completed the authorization — the login "
                        "was probably cancelled (tab closed, or the consent left "
                        "unfinished). Run /mcp login again to retry."
                    ) from exc
            # An OAuth grant requirement is not a transport failure: surface it
            # as the clean type so callers (startup toast, reconnect breaker,
            # /mcp login) can recognise it. It can ride EITHER the original
            # exception or the group the task raised on close, so check both —
            # otherwise the actionable error reads as an opaque TaskGroup
            # failure or, worse, a bare CancelledError.
            for candidate in (exc, close_exc):
                if candidate is None:
                    continue
                auth_exc = _unwrap_auth_required(candidate)
                if isinstance(auth_exc, McpAuthRequiredError):
                    raise auth_exc from candidate
            if isinstance(exc, asyncio.CancelledError):
                raise  # a real cancellation (dispose/esc): never converted
            if not isinstance(exc, Exception):
                raise  # KeyboardInterrupt & co. propagate unchanged
            stderr_log.report_failure(f"failed to connect: {exc}")
            raise stderr_log.explain(exc) from exc

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
        stderr_log: McpServerStderr,
        *,
        interactive: bool = False,
    ) -> ServerConnection:
        """Enter the transport + ClientSession context managers on ``stack``.

        ``stderr_log`` is what the stdio transport writes the child's stderr to
        instead of the terminal. The remote transports spawn nothing and leave
        it untouched.

        ``interactive`` is forwarded to the OAuth provider builder: only an
        explicit login may open a browser.
        """
        import mcp.types as mcp_types
        from mcp.client.session import ClientSession

        conn = ServerConnection(
            name=name,
            config=cfg,
            stack=stack,
            source=self._sources.get(name, ""),
        )

        if isinstance(cfg, MCPStdioServerConfig):
            streams_cm = _stdio_transport(cfg, lambda: conn.closed_event.set(), stderr_log)
        elif isinstance(cfg, MCPHttpServerConfig):
            from mcp.client.streamable_http import (
                create_mcp_http_client,
                streamable_http_client,
            )

            oauth_provider = self._build_oauth_auth(cfg.url, cfg, interactive=interactive)
            http_client = create_mcp_http_client(
                headers=dict(cfg.headers) or None,
                auth=oauth_provider,
            )
            streams_cm = streamable_http_client(cfg.url, http_client=http_client)
        elif isinstance(cfg, MCPSseServerConfig):
            from mcp.client.sse import sse_client

            # SSE wires no auth into its client here (pre-existing gap), but
            # the provider is still built so an OAuth config's loopback flow
            # is recorded: the abandoned-grant check in ``_connect_server``
            # reads it off the manager's per-URL map.
            self._build_oauth_auth(cfg.url, cfg, interactive=interactive)
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

    def _build_oauth_auth(
        self, url: str, cfg: MCPServerConfig, *, interactive: bool = False
    ) -> OAuthClientProvider | None:
        """Build an ``OAuthClientProvider`` for configs with ``auth.type=oauth``.

        ``interactive`` decides whether the flow may open a browser (only an
        explicit login). The provider is primed with the endpoints the
        proactive refresh discovered, so a mid-session in-flow refresh targets
        the real token endpoint.
        """
        auth = cfg.auth
        if auth is None or auth.type != "oauth":
            return None
        try:
            from local_operator.mcp.auth import build_oauth_provider

            provider = build_oauth_provider(
                url,
                cfg,
                store=self._effective_auth_store(),
                interactive=interactive,
                endpoints=self._oauth_endpoints.get(url),
            )
        except Exception:
            logger.warning(
                "OAuth wiring unavailable for %r; connecting unauthenticated",
                url,
                exc_info=True,
            )
            return None
        # Record the grant's flow by server URL: an abandoned grant crosses
        # the transport as a raw CancelledError (the SDK swallows ordinary
        # auth exceptions), and ``_connect_server``'s error path cannot reach
        # the provider from there — this map is how it finds the flow to
        # consult the ABANDONED_GRANTS ledger.
        flow = getattr(provider, "_loopback_flow", None)
        if flow is not None:
            self._oauth_flows[url] = flow
        return provider

    async def _ensure_oauth_fresh(self, name: str, cfg: MCPServerConfig) -> None:
        """Proactively refresh an OAuth grant before connecting (best-effort).

        Spends a stored refresh token against the DISCOVERED token endpoint so
        a day-old access token never forces a browser grant on startup, and
        caches the discovered endpoints for the provider. Never raises: a
        failed refresh simply leaves the stored token as-is, and the provider's
        non-interactive redirect handler is what turns the resulting grant
        attempt into an actionable error instead of a login tab.
        """
        auth = getattr(cfg, "auth", None)
        url = getattr(cfg, "url", None)
        if auth is None or getattr(auth, "type", None) != "oauth" or not url:
            return
        try:
            from local_operator.mcp.auth import ensure_mcp_oauth_fresh

            endpoints = await ensure_mcp_oauth_fresh(url, cfg, store=self._effective_auth_store())
        except Exception:  # noqa: BLE001 — refresh is best-effort; degrade, don't fail
            logger.debug("MCP proactive refresh failed for %r", name, exc_info=True)
            return
        if endpoints is not None:
            self._oauth_endpoints[url] = endpoints

    @staticmethod
    def _auth_required_text(name: str, exc: McpAuthRequiredError) -> str:
        """The startup-toast wording for a server that needs an OAuth login.

        Leads with the COMMAND that fixes it rather than the diagnosis. The
        toast renders this after a ``failed: <name> — `` prefix and then clamps
        to the card width, so the tail is what gets truncated: putting
        ``run /mcp login <name>`` first keeps the one actionable thing on screen
        even on a narrow terminal, where ``needs authorization — run /mcp login
        <name>`` used to sever the command mid-word (design review D1). One
        ``—`` only, so the composed line is not a chain of dashes (D4). The same
        string lands in the durable transcript notice and in ``/mcp``, so one
        helper keeps all three surfaces agreeing.
        """
        return f"run /mcp login {name} to authorize"

    def _fire_auth_required(self, name: str, exc: McpAuthRequiredError) -> None:
        """Notify the UI that ``name`` needs an OAuth login (best-effort).

        Fired for auth failures that land OUTSIDE the startup gate (the common
        case for HTTP OAuth servers) and for mid-session expiry, so the user
        sees a toast rather than only a transcript incident. Deduped per
        server until it connects again, so a dead grant a tool call keeps
        retrying does not re-raise the toast on every attempt. Never raises: a
        broken UI hook must not take down the connect/reconnect machinery.
        """
        if name in self._auth_toasted:
            return
        sink = self.on_auth_required
        if sink is None:
            return
        try:
            sink(name, self._auth_required_text(name, exc))
            self._auth_toasted.add(name)
        except Exception:  # noqa: BLE001 — UI hooks must never break the manager
            logger.debug("on_auth_required sink raised", exc_info=True)

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

    def _settle_deferred(self, name: str) -> None:
        """Mark one deferred server settled; fire the round callback when last.

        Called from the background continuation once ``name`` reaches a terminal
        state (connected or failed), whatever the outcome. When the deferred set
        empties, the discovery round is fully settled and ``on_startup_settled``
        fires so the front end re-reports the COMBINED outcome — the gate
        snapshot it reported first was taken while these servers were still
        connecting. Fires at most once per round (the set is only refilled by a
        new ``_connect_round``); a manager with nothing deferred never arms it.
        """
        self._startup_deferred.discard(name)
        if self._startup_deferred:
            return
        sink = self.on_startup_settled
        if sink is None:
            return
        try:
            sink()
        except Exception:  # noqa: BLE001 — a UI hook must never break the manager
            logger.debug("mcp on_startup_settled sink raised", exc_info=True)

    def startup_failures(self) -> dict[str, str]:
        """The combined per-server startup failures for the current round.

        Accumulated across the gate pass and the background continuations, so it
        names every failure and not only the ones that lost the race to the
        250 ms gate. This is what the settled re-report reads instead of a
        second, momentary ``get_connected_servers`` snapshot.
        """
        return dict(self._startup_failures)

    def startup_settling(self) -> bool:
        """True while servers deferred past the startup gate are still settling.

        The boot report reads this to mark its snapshot provisional: a settling
        outcome suppresses the failure/success surface until
        ``on_startup_settled`` fires with the complete tally. False means the
        gate snapshot was already final (nothing was deferred, or every deferred
        server has since settled)."""
        return bool(self._startup_deferred)

    async def _finish_pending(
        self, name: str, task: asyncio.Task[ServerConnection], epoch: int
    ) -> None:
        """Background continuation for a server still connecting at the gate."""
        try:
            conn = await task
        except asyncio.CancelledError:
            # A cancelled continuation is teardown/reload, not a settled server:
            # the deferred set is reset wholesale by the next round, so do NOT
            # fire the settle callback off a cancellation (it would report a
            # half-torn-down round).
            return
        except Exception as exc:
            logger.warning("MCP server %r failed to connect after the gate: %s", name, exc)
            # An OAuth grant requirement that lands AFTER the startup gate (the
            # common case for HTTP servers, which are slow) never reaches the
            # startup toast. Fire the incident sink so the failure is recorded
            # durably and the agent knows the tools are gone until a login.
            auth_exc = _unwrap_auth_required(exc)
            if isinstance(auth_exc, McpAuthRequiredError):
                sink = getattr(self, "on_incident", None)
                if sink is not None:
                    try:
                        sink(
                            name,
                            f"OAuth authorization expired; run /mcp login {name} to "
                            "restore its tools",
                        )
                    except Exception:  # noqa: BLE001 — incidents must never break the manager
                        logger.debug("mcp incident sink raised", exc_info=True)
                # The startup toast has already been dismissed by the time an
                # after-gate connect fails, so raise a fresh one via the UI hook.
                self._fire_auth_required(name, auth_exc)
            # Whether this continuation still belongs to the CURRENT round. A
            # reload()/dispose() during the await bumps the epoch, and a stale
            # continuation must not write the new round's startup accounting —
            # the success arm below already guards on this, and the failure arm
            # needs the same guard for the accumulator and the settle callback
            # (F2). The waiter-settling and auth-required surfacing below are
            # NOT guarded: a parked waiter must fail rather than hang, and the
            # incident/toast reflect a real failure regardless of which round
            # owns it.
            current_round = not self._disposed and epoch == self._epoch
            if current_round:
                # Record the failure into the round accumulator (an auth
                # requirement as its actionable text, else the raw reason) and
                # settle this deferred server BEFORE firing tools-changed, so the
                # front end that re-reports on settle sees the complete map.
                if isinstance(auth_exc, McpAuthRequiredError):
                    self._startup_failures[name] = self._auth_required_text(name, auth_exc)
                else:
                    self._startup_failures[name] = str(exc)
            # Re-fetch the waiter: a reload during the await may have swapped
            # it, and settling the stale one would strand the current waiters.
            _settle_future_error(self._connect_futures.get(name), exc)
            self._connect_futures.pop(name, None)
            self._tools_by_server.pop(name, None)  # drop the deferred slice
            self._unregister_origins(name)
            self._rebuild_agent_names()
            if current_round:
                self._settle_deferred(name)
            self._fire_tools_changed()
            return
        if self._disposed or epoch != self._epoch:
            if conn.stack is not None:
                with suppress(Exception):
                    await conn.stack.aclose()
            # A disposed/superseded round is not a settled one: leave the settle
            # callback to the round that is actually current.
            return
        # A late success clears any earlier failure recorded for this server and
        # settles it. Order matches the failure arm: accounting first, then the
        # tools-changed fire that may trigger the settle re-report.
        self._startup_failures.pop(name, None)
        self._register_connection(conn)  # settles the waiter future
        self._settle_deferred(name)
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
        # A successful (re)connect clears the auth-toast latch, so a grant that
        # expires AGAIN later gets its own toast instead of staying silent.
        self._auth_toasted.discard(conn.name)
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
            self._build_tool(name, tool, deferred=False)
            for tool in tools
            if self._tool_is_enabled(name, self._raw_tool_name(tool))
        ]
        self._rebuild_agent_names()

    @staticmethod
    def _raw_tool_name(tool: Tool | dict[str, Any]) -> str:
        return str(tool.get("name", "")) if isinstance(tool, dict) else str(tool.name)

    def _tool_is_enabled(self, server_name: str, tool_name: str) -> bool:
        """Per-server MCP tool filter.

        ``disabledTools`` wins; a non-empty ``enabledTools`` is an allowlist;
        both accept exact names or glob patterns. Filtering happens before
        name minting for BOTH cached/deferred and live tools, so a reconnect
        or tools/list_changed cannot resurrect a denied schema into the
        provider tools array (the context-cost and trust guarantees are the
        same at startup and after recovery).
        """
        cfg = self._configs.get(server_name)
        if cfg is None:
            return True
        denied = getattr(cfg, "disabled_tools", []) or []
        if any(fnmatchcase(tool_name, pattern) for pattern in denied):
            return False
        allowed = getattr(cfg, "enabled_tools", []) or []
        return not allowed or any(fnmatchcase(tool_name, pattern) for pattern in allowed)

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
                server_name,
                mcp_tool_name,
                tool_call_id,
                args,
                signal,
                deferred=deferred,
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
            meta: McpToolMeta = {
                **self._meta_by_origin.get(origin, {}),
                "agent_name": name,
            }
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
            # Model-visible incident (session installs the sink): the agent
            # must know the server's tools are GONE, or it hammers them in a
            # tight loop. Fire-and-forget so a raising sink cannot stall the
            # reconnect machinery.
            sink = getattr(self, "on_incident", None)
            if sink is not None:
                try:
                    sink(
                        name,
                        f"auto-reconnect suspended after >{RECONNECT_BURST_LIMIT} "
                        f"attempts in {int(RECONNECT_BURST_WINDOW_S)}s; its tools are "
                        "unavailable until a reconnect succeeds",
                    )
                except Exception:  # noqa: BLE001 — incidents must never break the manager
                    logger.debug("mcp incident sink raised", exc_info=True)
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
        except McpAuthRequiredError as exc:
            # An expired grant will not heal by retrying: auto-reconnect is
            # non-interactive by design, so further attempts would only burn the
            # breaker window. Abandon with an actionable reason; ``/mcp login``
            # (which resets the breaker) is the recovery path.
            logger.info("MCP reconnect needs authorization for %r", name)
            # Model-visible incident: the agent must know the server's tools are
            # gone until a login, or it hammers them. Same fire-and-forget guard
            # as the breaker path.
            sink = getattr(self, "on_incident", None)
            if sink is not None:
                try:
                    sink(
                        name,
                        f"OAuth authorization expired; run /mcp login {name} to "
                        "restore its tools",
                    )
                except Exception:  # noqa: BLE001 — incidents must never break the manager
                    logger.debug("mcp incident sink raised", exc_info=True)
            # Mid-session expiry happens long after the startup toast, so raise a
            # fresh one via the UI hook.
            self._fire_auth_required(name, exc)
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark retrieved; waiters still see the raise
            self._connect_futures.pop(name, None)
            # Suspend durably (like the breaker) so a call-site retry also stops
            # hammering a grant that cannot heal without a login.
            self._reconnect_suspended.add(name)
            self._abandon_reconnect(name, str(exc))
            return
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
        except McpAuthRequiredError as exc:
            logger.info("MCP call-site reconnect needs authorization for %r", name)
            self._fire_auth_required(name, exc)
            return None
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
        """Close one connection's stack without touching registries.

        BOUNDED (:data:`CONNECTION_TEARDOWN_TIMEOUT_S`): a remote transport's
        close sends a session-terminate request over the network, and a dead
        network or wedged server must not hold session dispose — the path the
        user experiences as "quit hangs" — for the HTTP client's own 30 s
        connect timeout. On timeout the close is CANCELLED: the stdio
        transport responds by killing its child (see ``_stop``), and a
        cancelled remote close simply drops the connection on the floor,
        which is what a dead network leaves anyway. ``TimeoutError`` is an
        ``Exception``, so the existing suppress covers it; a real
        cancellation from above still propagates.
        """
        conn = self._connections.pop(name, None)
        if conn is None:
            return
        conn.closed_event.set()
        if conn.stack is not None:
            with suppress(Exception):
                await asyncio.wait_for(conn.stack.aclose(), CONNECTION_TEARDOWN_TIMEOUT_S)

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
    "CHILD_QUIET_ENV",
    "RECONNECT_BACKOFF_S",
    "RECONNECT_BURST_LIMIT",
    "RECONNECT_BURST_WINDOW_S",
    "STARTUP_GATE_MS",
    "STDERR_TAIL_LINES",
    "McpConnectionError",
    "McpLoadResult",
    "McpManager",
    "McpServerStderr",
    "ServerConnection",
    "build_cmd_exe_argv",
    "build_stdio_argv",
    "escape_cmd_batch_arg",
    "escape_cmd_quoted_interior",
    "resolve_mcp_timeout_s",
    "stdio_start_new_session",
    "win32_process_target",
]
