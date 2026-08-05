"""Builtin tools for the new harness.

Why this module exists
----------------------
Tools declare as classes implementing ``AgentTool`` with per-tool
parameter schemas; the old local-operator instead injected Python callables
into executed code and rendered prose signatures into the prompt (the
``prompts.py`` reflection generator — the audit flagged that as the main
thing keeping the prompt at 176 KB). The rewrite adopts that shape: each
tool is an :class:`local_operator.harness.types.AgentTool` with a JSON Schema
derived from a pydantic parameter model, executed via native provider tool
calling.

Conventions every tool here follows:

- Parameter schema: a module-level pydantic model per tool;
  ``model_json_schema()`` output becomes ``AgentTool.parameters``. Field
  ``description`` strings are the model's only documentation the LLM sees.
- Parameter validation failures are returned as a clean
  ``invalid arguments:`` list — never a traceback — so the model can fix its
  call; truly unexpected exceptions are caught by the shared ``_guard``
  wrapper and returned as an error result carrying the traceback tail, so a
  buggy tool can never kill the turn.
- ``useless`` flags contextually worthless results (zero matches) so
  compaction may elide them once consumed; it is never combined with
  ``is_error``, and a useless result always carries
  ``details={"useless": True}`` so compaction can trust the payload.
- Approval tiers follow the read/write/exec model; the host's approval
  callback on ``ToolContext`` gates mutating side effects. Paths are always
  resolved before the prompt is built, so the user approves the exact file
  that will change; paths outside the workspace are flagged in the approval
  text and always require approval, even for read-tier tools.

The ``wake`` tool delegates to ``local_operator.harness.wake``; the import is
deferred to execute time so this module has no hard dependency on the wake
subsystem being importable (the session may run without wakes).
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import os
import re
import signal as signal_module
import time
import traceback
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    TextContent,
    ToolContext,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Shared limits and helpers
# ---------------------------------------------------------------------------

#: Single combined budget for captured stdout+stderr (chars, not bytes, since
#: the output is one decoded transcript): the whole result must fit the
#: ~30k-char per-tool budget in docs/REWRITE.md, and truncation is
#: head-then-tail with one marker so both ends survive.
BASH_OUTPUT_LIMIT_CHARS = 50 * 1024
BASH_TRUNCATION_MARKER = "\n\n... [output truncated] ...\n\n"

#: Maximum number of matches returned by grep.
GREP_MATCH_LIMIT = 200
#: Maximum number of paths returned by glob.
GLOB_RESULT_LIMIT = 500
#: Default timeout for bash commands (seconds).
BASH_DEFAULT_TIMEOUT_SECONDS = 120.0
#: Hard cap on the per-command timeout; longer runs are a session bug, not a
#: tool feature.
BASH_MAX_TIMEOUT_SECONDS = 3600.0
#: Number of trailing traceback characters kept in an error result.
TRACEBACK_TAIL_CHARS = 2000

#: Files larger than this are refused by read (serve 2MB+ blobs through bash
#: with head/tail instead); the cap serves the per-tool output budget.
READ_FILE_LIMIT_BYTES = 2 * 1024 * 1024
#: Maximum lines read renders; larger files show the head plus a footer
#: telling the model to continue with a line range.
READ_LINE_CAP = 2000
#: Per-file size cap for grep; bigger files are skipped and counted.
GREP_FILE_LIMIT_BYTES = 1 * 1024 * 1024

#: Directory names never worth walking during grep (VCS internals, vendored
#: trees, build output). Dotdirs are pruned wholesale in addition.
_GREP_PRUNE_DIRS = frozenset({"__pycache__", "node_modules", "dist", "build", ".git", ".venv"})
#: Marker prefix on approval descriptions for targets outside the workspace.
OUTSIDE_WORKSPACE_MARKER = "[outside workspace]"

#: Environment overrides that make common CLIs non-interactive.
NON_INTERACTIVE_ENV: dict[str, str] = {
    # Disable pagers so commands don't block on interactive views.
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "MANPAGER": "cat",
    "SYSTEMD_PAGER": "cat",
    "BAT_PAGER": "cat",
    "DELTA_PAGER": "cat",
    "GH_PAGER": "cat",
    "GLAB_PAGER": "cat",
    "PSQL_PAGER": "cat",
    "MYSQL_PAGER": "cat",
    "AWS_PAGER": "",
    "HOMEBREW_PAGER": "cat",
    "LESS": "FRX",
    # Disable terminal features that can block the process.
    "TERM": "dumb",
    "NO_COLOR": "1",
    "PYTHONUNBUFFERED": "1",
    # Disable editor and terminal credential prompts.
    "GIT_EDITOR": "true",
    "VISUAL": "true",
    "EDITOR": "true",
    "GIT_TERMINAL_PROMPT": "0",
    "SSH_ASKPASS": "/usr/bin/false",
    "CI": "1",
    # Package manager defaults for unattended execution.
    "npm_config_yes": "true",
    "npm_config_update_notifier": "false",
    "npm_config_fund": "false",
    "npm_config_audit": "false",
    "npm_config_progress": "false",
    "PNPM_DISABLE_SELF_UPDATE_CHECK": "true",
    "PNPM_UPDATE_NOTIFIER": "false",
    "YARN_ENABLE_TELEMETRY": "0",
    "YARN_ENABLE_PROGRESS_BARS": "0",
    # Cross-language/tooling non-interactive defaults.
    "CARGO_TERM_PROGRESS_WHEN": "never",
    "DEBIAN_FRONTEND": "noninteractive",
    "PIP_NO_INPUT": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "TF_INPUT": "0",
    "TF_IN_AUTOMATION": "1",
    "GH_PROMPT_DISABLED": "1",
    "COMPOSER_NO_INTERACTION": "1",
    "CLOUDSDK_CORE_DISABLE_PROMPTS": "1",
}


def truncate_output(text: str, limit: int = BASH_OUTPUT_LIMIT_CHARS) -> str:
    """Keep the head and tail of ``text`` when it exceeds ``limit``.

    Mirrors the established output truncation: the middle is replaced by a marker so the
    model sees both the beginning (banners, command echo) and the end (actual
    error) of a large output without blowing up the transcript. The result is
    at most ``limit`` chars — the marker lives inside the budget.
    """
    if len(text) <= limit:
        return text
    budget = limit - len(BASH_TRUNCATION_MARKER)
    head = budget // 2
    tail = budget - head
    return text[:head] + BASH_TRUNCATION_MARKER + text[len(text) - tail :]


def _safe_cwd(context: ToolContext | None) -> str:
    return context.cwd if context and context.cwd else "."


def _resolve_workspace_path(raw: str, cwd: str) -> tuple[Path, bool]:
    """Resolve a tool-supplied path to an absolute ``Path``.

    ``~`` is expanded, relative paths join onto ``cwd``, and the result is
    fully resolved. Returns ``(path, inside)`` where ``inside`` is True when
    the resolved path stays within the resolved workspace root — approval
    prompts always show the resolved path, and ``outside`` targets escalate
    approval even for read-tier tools.
    """
    root = Path(cwd).expanduser().resolve()
    candidate = Path(raw).expanduser()
    path = candidate if candidate.is_absolute() else root / candidate
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return path, False
    return path, True


def _approval_description(path: Path, inside: bool, action: str) -> str:
    """Approval prompt text for ``action`` on a RESOLVED path (the user must
    approve the exact target, not the raw string the model typed)."""
    marker = "" if inside else f"{OUTSIDE_WORKSPACE_MARKER} "
    return f"{marker}{action}: {path}"


def _error(tool_call_id: str, tool_name: str, message: str) -> ToolResult:
    """Build a non-throwing error result (loop never raises into the model)."""
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=message)],
        is_error=True,
    )


def _text(
    tool_call_id: str,
    tool_name: str,
    text: str,
    *,
    useless: bool = False,
    details: Any = None,
) -> ToolResult:
    """Build a plain-text result; ``details`` carries structured payload for
    renderers and compaction pruning (e.g. ``path`` for file tools)."""
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
        details=details,
        useless=useless,
    )


def _validation_error(tool_call_id: str, tool_name: str, exc: ValidationError) -> ToolResult:
    """One ``invalid arguments:`` line per field — no traceback. The model can
    correct its call from the message; the stack trace could not."""
    lines = [
        f"- {'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
        for err in exc.errors()
    ]
    return _error(tool_call_id, tool_name, "invalid arguments:\n" + "\n".join(lines))


def _guard(tool_name: str) -> Callable[..., Any]:
    """Wrap an execute coroutine so unexpected exceptions become error results.

    The harness contract is that tools never throw into the loop: provider
    error paths (Anthropic rejects empty is_error blocks) and retry logic all
    assume a ToolResult comes back. The traceback tail is included so the
    model can self-correct and we can debug from transcripts.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(
            tool_call_id: str,
            args: dict[str, Any],
            signal: AbortSignal | None = None,
            on_update: Callable[[AgentToolUpdate], None] | None = None,
            context: ToolContext | None = None,
        ) -> ToolResult:
            try:
                return await fn(tool_call_id, args, signal, on_update, context)
            except Exception:  # noqa: BLE001 — boundary: nothing may escape
                return _error(
                    tool_call_id,
                    tool_name,
                    f"Tool '{tool_name}' failed unexpectedly:\n"
                    f"{traceback.format_exc()[-TRACEBACK_TAIL_CHARS:]}",
                )

        wrapper.__name__ = f"execute_{tool_name}"
        wrapper.__qualname__ = wrapper.__name__
        return wrapper

    return decorator


async def _check_approval(context: ToolContext | None, tier: str, description: str) -> bool:
    """Ask the host for approval; True means proceed.

    No approval hook installed -> auto-approved (CLI --yolo and headless tests
    rely on this). A hook returning False denies the action without error
    state beyond a plain refusal message.
    """
    request_approval = getattr(context, "request_approval", None) if context else None
    if request_approval is None:
        return True
    return bool(await request_approval(tier, description))


async def _run_with_abort(
    coro: Awaitable[Any],
    signal: AbortSignal | None,
    on_abort: Callable[[], None],
) -> tuple[Any, bool]:
    """Race ``coro`` against the abort signal.

    Returns ``(result_or_None, aborted)``. On abort ``on_abort`` runs (e.g.
    process kill) before the coroutine is cancelled, so resources held by
    the awaited call are released deterministically instead of being
    abandoned. A signal already aborted at entry STILL runs ``on_abort`` and
    closes the pending coroutine — the old early return skipped both, which
    leaked the spawned child and raised "coroutine was never awaited"
    (RT-01). Callers that must not spawn at all should check
    ``signal.aborted`` before creating the coroutine.
    """
    if signal is not None and signal.aborted:
        on_abort()
        if asyncio.iscoroutine(coro):
            coro.close()
        return None, True
    if signal is None:
        return await coro, False
    waiter = asyncio.create_task(signal.wait())
    work = asyncio.ensure_future(coro)
    done, _pending = await asyncio.wait({waiter, work}, return_when=asyncio.FIRST_COMPLETED)
    if work in done:
        waiter.cancel()
        with contextlib.suppress(BaseException):
            await waiter
        return work.result(), False
    on_abort()
    work.cancel()
    with contextlib.suppress(BaseException):
        await work
    return None, True


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class BashParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    command: str = Field(description="Shell command to run (executed via /bin/sh -c).")
    timeout: float = Field(
        default=BASH_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=BASH_MAX_TIMEOUT_SECONDS,
        description="Max seconds before the command is killed.",
    )


def _bash_output_summary(stdout: str, stderr: str) -> str:
    """The shared 'stdout/stderr' body used by updates and the final result."""
    parts = [
        f"--- stdout ---\n{stdout}" if stdout else "--- stdout ---\n(empty)",
        f"--- stderr ---\n{stderr}" if stderr else "--- stderr ---\n(empty)",
    ]
    return "\n".join(parts)


@_guard("bash")
async def execute_bash(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Run a shell command non-interactively and capture its output.

    Output is read incrementally by per-stream reader tasks (kept referenced,
    so they are never orphaned): accumulated output streams to ``on_update``
    roughly every 500 ms while the command runs, and partial output survives
    both abort and timeout.
    """
    try:
        params = BashParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "bash", exc)
    if not params.command.strip():
        return _error(tool_call_id, "bash", "command must be a non-empty string")
    # Approval for write/exec tiers is the LOOP's gate (it fires after
    # tool_execution_start so the UI shows the pending call). A second gate
    # here made the user answer twice per action, with the tier name rendered
    # as the tool. Read-tier outside-workspace escalations still use
    # _check_approval in execute_read/execute_grep.

    # Pre-aborted signal: never spawn a child there is no intention to run.
    if signal is not None and signal.aborted:
        return _error(
            tool_call_id,
            "bash",
            f"aborted ({signal.reason or 'aborted'}): {params.command}",
        )

    env = os.environ.copy()
    env.update(NON_INTERACTIVE_ENV)

    process = await asyncio.create_subprocess_exec(
        "/bin/sh",
        "-c",
        params.command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=_safe_cwd(context),
        env=env,
        start_new_session=True,
    )

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def _pump(stream: asyncio.StreamReader, sink: list[bytes]) -> None:
        try:
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                sink.append(chunk)
        except (ConnectionResetError, BrokenPipeError):
            pass

    # Hold the tasks ourselves so the readers are never abandoned mid-run.
    stdout_task = asyncio.create_task(_pump(process.stdout, stdout_chunks))
    stderr_task = asyncio.create_task(_pump(process.stderr, stderr_chunks))
    readers = (stdout_task, stderr_task)

    def _kill() -> None:
        # Kill the whole session group so children (sh -c spawns) die too.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal_module.SIGKILL)

    def _emit_update() -> None:
        if on_update is None:
            return
        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        on_update(
            AgentToolUpdate(
                content=[TextContent(text=_bash_output_summary(stdout, stderr))],
                details={"tool_name": "bash", "running": True},
            )
        )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + params.timeout
    wait_task = asyncio.create_task(process.wait())
    abort_waiter = asyncio.create_task(signal.wait()) if signal is not None else None

    timed_out = False
    aborted = False
    next_update = loop.time() + 0.5

    while True:
        waiters: list[asyncio.Task[object]] = [wait_task, stdout_task, stderr_task]
        if abort_waiter is not None:
            waiters.append(abort_waiter)
        if wait_task.done():
            break  # finished already — never misreport as timeout
        remaining = deadline - loop.time()
        if remaining <= 0:
            timed_out = True
            _kill()
            break
        done, _pending = await asyncio.wait(waiters, timeout=min(0.25, remaining))
        if wait_task in done:
            break
        if abort_waiter is not None and abort_waiter in done:
            aborted = True
            _kill()
            break
        if loop.time() >= next_update:
            _emit_update()
            next_update = loop.time() + 0.5

    # Bounded drain: the kill above EOFs both pipes; give the readers 250 ms
    # to consume what is already buffered so partial output survives.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(asyncio.gather(*readers, return_exceptions=True), timeout=0.25)
    for task in readers:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task

    # Reap the process and release the transport so no ResourceWarning fires.
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), timeout=1.0)
    if process.returncode is None:
        _kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)
    transport = getattr(process, "_transport", None)
    if transport is not None:
        transport.close()

    if abort_waiter is not None and not abort_waiter.done():
        abort_waiter.cancel()
        with contextlib.suppress(BaseException):
            await abort_waiter

    if aborted:
        partial = _bash_output_summary(
            b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        )
        return _error(
            tool_call_id,
            "bash",
            f"aborted ({(signal.reason or 'aborted') if signal else 'aborted'}): "
            f"{params.command}\n{partial}",
        )

    stdout_raw = b"".join(stdout_chunks).decode("utf-8", errors="replace")
    stderr_raw = b"".join(stderr_chunks).decode("utf-8", errors="replace")
    # Both streams may end up carrying a marker, so reserve room for two.
    budget = BASH_OUTPUT_LIMIT_CHARS - 2 * len(BASH_TRUNCATION_MARKER)
    if len(stdout_raw) + len(stderr_raw) > budget:
        total = len(stdout_raw) + len(stderr_raw)
        stdout_budget = max(budget * len(stdout_raw) // total, 1)
        stderr_budget = max(budget - stdout_budget, 1)
        stdout = truncate_output(stdout_raw, stdout_budget)
        stderr = truncate_output(stderr_raw, stderr_budget)
    else:
        stdout, stderr = stdout_raw, stderr_raw

    return_code = process.returncode if process.returncode is not None else -1
    parts = [f"exit code: {return_code}", _bash_output_summary(stdout, stderr)]
    if timed_out:
        parts.insert(0, f"TIMEOUT after {params.timeout}s (process killed)")
    return _text(tool_call_id, "bash", "\n".join(parts))


def build_bash_tool() -> AgentTool:
    return AgentTool(
        name="bash",
        label="Shell",
        description=("Run a shell command and return its exit code, stdout and stderr."),
        parameters=BashParams.model_json_schema(),
        approval_tier="exec",
        # bash runs shared when non-pty; models batch independent
        # commands, and exclusive would serialize the common case.
        concurrency="shared",
        interruptible=True,
        execute=execute_bash,
    )


# ---------------------------------------------------------------------------
# read
# ---------------------------------------------------------------------------


class ReadParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        description=(
            "File path (absolute or relative to the working directory), or an "
            "internal URL such as skill://<name>."
        )
    )
    range: str | None = Field(
        default=None,
        description=(
            "Optional 1-based inclusive line range 'start-end' (e.g. '10-40') "
            "or 'start-' to read to the end. Ignored for internal URLs."
        ),
    )


_LINE_RANGE_RE = re.compile(r"^(\d+)\s*-\s*(\d+)?$")


def _parse_line_range(spec: str) -> tuple[int, int | None]:
    match = _LINE_RANGE_RE.match(spec.strip())
    if not match:
        raise ValueError(f"invalid line range '{spec}' (expected 'start-end' or 'start-')")
    start = int(match.group(1))
    if start < 1:
        raise ValueError(f"invalid line range '{spec}': start must be >= 1")
    end = int(match.group(2)) if match.group(2) else None
    if end is not None and end < start:
        raise ValueError(f"invalid line range '{spec}': end must be >= start")
    return start, end


def _number_lines(lines: list[str], start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{start + i:>{width}}| {line}" for i, line in enumerate(lines))


@_guard("read")
async def execute_read(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Read a file (with optional line range) or resolve an internal URL."""
    try:
        params = ReadParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "read", exc)
    target = params.path.strip()
    if not target:
        return _error(tool_call_id, "read", "path must be a non-empty string")

    # Internal URLs (skill://...) go through the session-installed resolver.
    if "://" in target and not target.startswith(("http://", "https://", "file://")):
        resolver = getattr(context, "resolve_internal_url", None) if context else None
        if resolver is None:
            return _error(
                tool_call_id,
                "read",
                f"Cannot resolve '{target}': no internal URL resolver is available.",
            )
        content = resolver(target)
        if content is None:
            return _error(
                tool_call_id,
                "read",
                f"Cannot resolve '{target}': the resolver does not handle this URL.",
            )
        return _text(tool_call_id, "read", content, details={"url": target})

    cwd = _safe_cwd(context)
    path, inside = _resolve_workspace_path(target, cwd)
    if not path.exists():
        return _error(tool_call_id, "read", f"Path does not exist: {path}")

    # Outside-workspace reads escalate to an approval prompt regardless of
    # the read tier auto-approval the host normally applies.
    if not inside:
        description = _approval_description(path, inside, "read")
        if not await _check_approval(context, "read", description):
            return _error(tool_call_id, "read", "User declined to read this file.")

    if path.is_dir():
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return _text(
            tool_call_id,
            "read",
            f"Directory listing of {path} ({len(entries)} entries):\n" + "\n".join(entries),
            details={"path": str(path)},
        )

    # Stat BEFORE reading: refuse oversized files instead of loading them,
    # then read only up to the line cap's worth of bytes.
    size = path.stat().st_size
    if size > READ_FILE_LIMIT_BYTES:
        return _error(
            tool_call_id,
            "read",
            f"File too large to read ({size} bytes; limit "
            f"{READ_FILE_LIMIT_BYTES} bytes): {path}. Use bash (head/tail) "
            "or a 'range' on a smaller file.",
        )

    data = path.read_bytes()
    if b"\x00" in data[:8000]:
        return _error(tool_call_id, "read", f"Binary file not readable as text: {path}")

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()

    if params.range:
        try:
            start, end = _parse_line_range(params.range)
        except ValueError as exc:
            return _error(tool_call_id, "read", str(exc))
        selected = lines[start - 1 : end]
        if not selected:
            return _text(
                tool_call_id,
                "read",
                f"(range {params.range} is beyond end of file {path})",
                useless=True,
                details={"path": str(path), "useless": True},
            )
        return _text(
            tool_call_id,
            "read",
            _number_lines(selected, start),
            # The range rides in details: compaction's supersede key must
            # distinguish ranged reads of the same file, or a read of lines
            # 900-1000 blanks an unrelated 1-100 read as "superseded".
            details={"path": str(path), "range": params.range},
        )

    if len(lines) > READ_LINE_CAP:
        body = _number_lines(lines[:READ_LINE_CAP], 1)
        remaining = len(lines) - READ_LINE_CAP
        return _text(
            tool_call_id,
            "read",
            f"{body}\n\n[{remaining} more lines in file. Use range to continue]",
            details={"path": str(path)},
        )
    return _text(
        tool_call_id,
        "read",
        _number_lines(lines, 1) if lines else "(empty file)",
        details={"path": str(path)},
    )


def build_read_tool() -> AgentTool:
    return AgentTool(
        name="read",
        label="Read",
        description="Read a file, a line range, or an internal URL like skill://<name>.",
        parameters=ReadParams.model_json_schema(),
        approval_tier="read",
        # read model: parallel reads are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_read,
    )


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------


class WriteParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File path to create or overwrite.")
    content: str = Field(description="Full file content to write.")


@_guard("write")
async def execute_write(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Create or overwrite a file, creating parent directories as needed."""
    try:
        params = WriteParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "write", exc)
    if not params.path.strip():
        return _error(tool_call_id, "write", "path must be a non-empty string")
    # Write-tier approval is the loop's gate; see execute_bash.
    path, inside = _resolve_workspace_path(params.path, _safe_cwd(context))

    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params.content, encoding="utf-8")
    verb = "Overwrote" if existed else "Created"
    return _text(
        tool_call_id,
        "write",
        f"{verb} {path} ({len(params.content)} chars).",
        details={"path": str(path)},
    )


def build_write_tool() -> AgentTool:
    return AgentTool(
        name="write",
        label="Write",
        description="Create or overwrite a file (parents are created automatically).",
        parameters=WriteParams.model_json_schema(),
        approval_tier="write",
        # write model: concurrent writes to the same file race silently;
        # an exclusive tool makes the last-writer outcome deterministic.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_write,
    )


# ---------------------------------------------------------------------------
# edit
# ---------------------------------------------------------------------------


class EditParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="File to edit.")
    old_text: str = Field(description="Exact text to find (must match verbatim).")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring exactly one.",
    )


@_guard("edit")
async def execute_edit(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Exact-match string replacement in a file.

    Ambiguity is an error, not a guess: with ``old_text`` matching more than
    once and ``replace_all`` unset the tool refuses, because silently editing
    the first occurrence is how edits corrupt the wrong site.
    """
    try:
        params = EditParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "edit", exc)
    if not params.path.strip():
        return _error(tool_call_id, "edit", "path must be a non-empty string")
    if params.old_text == "":
        return _error(tool_call_id, "edit", "old_text must be a non-empty string")

    path, inside = _resolve_workspace_path(params.path, _safe_cwd(context))
    if not path.is_file():
        return _error(tool_call_id, "edit", f"File does not exist: {path}")

    content = path.read_text(encoding="utf-8")
    occurrences = content.count(params.old_text)
    if occurrences == 0:
        return _error(
            tool_call_id,
            "edit",
            "old_text not found in the file. Re-read the file to get the exact "
            "current text (whitespace included) and retry.",
        )
    if occurrences > 1 and not params.replace_all:
        return _error(
            tool_call_id,
            "edit",
            f"old_text matches {occurrences} places; include more surrounding "
            "context to make it unique, or set replace_all=true.",
        )
    # Write-tier approval is the loop's gate; see execute_bash.

    if params.replace_all:
        updated = content.replace(params.old_text, params.new_text)
    else:
        updated = content.replace(params.old_text, params.new_text, 1)
    path.write_text(updated, encoding="utf-8")
    replaced = occurrences if params.replace_all else 1
    return _text(
        tool_call_id,
        "edit",
        f"Edited {path}: replaced {replaced} occurrence(s) of old_text.",
        details={"path": str(path)},
    )


def build_edit_tool() -> AgentTool:
    return AgentTool(
        name="edit",
        label="Edit",
        description="Replace exact text in a file (errors on missing or ambiguous matches).",
        parameters=EditParams.model_json_schema(),
        approval_tier="write",
        # edit model: two concurrent edits on one file corrupt each
        # other's match anchors; exclusive serializes the read-modify-write.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_edit,
    )


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


class GlobParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        description="Glob pattern relative to the working directory ('**/*.py' supported)."
    )


@_guard("glob")
async def execute_glob(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """List paths matching a glob pattern (files and directories)."""
    try:
        params = GlobParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "glob", exc)
    pattern = params.pattern.strip()
    if not pattern:
        return _error(tool_call_id, "glob", "pattern must be a non-empty string")
    if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        return _error(
            tool_call_id,
            "glob",
            "pattern must be a relative glob within the working directory "
            "(no absolute paths, no '..').",
        )

    root = Path(_safe_cwd(context))
    # An unbounded ``**`` walk is filesystem work that can freeze the session;
    # off the event loop and raced against abort like the grep scan.
    matches, aborted = await _run_with_abort(
        asyncio.to_thread(_glob_walk, root, pattern),
        signal,
        lambda: None,
    )
    if aborted:
        return _error(tool_call_id, "glob", "Glob aborted.")
    if not matches:
        return _text(
            tool_call_id,
            "glob",
            f"No paths matched pattern '{params.pattern}'.",
            useless=True,
            details={"useless": True},
        )
    capped = len(matches) > GLOB_RESULT_LIMIT
    matches = matches[:GLOB_RESULT_LIMIT]
    header = f"{len(matches)} match(es) for '{params.pattern}'"
    if capped:
        header += f" (capped at {GLOB_RESULT_LIMIT})"
    return _text(tool_call_id, "glob", header + ":\n" + "\n".join(matches))


def build_glob_tool() -> AgentTool:
    return AgentTool(
        name="glob",
        label="Glob",
        description="Find files and directories by glob pattern ('**' supported).",
        parameters=GlobParams.model_json_schema(),
        approval_tier="read",
        # Read-only listing; parallel globs are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_glob,
    )


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


class GrepParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="Python regular expression to search for.")
    path: str = Field(
        default=".",
        description="Directory or file to search (relative to the working directory).",
    )
    include: str | None = Field(
        default=None,
        description=(
            "Optional glob filter applied to file names/basenames " "(e.g. '*.py', '**/*.ts')."
        ),
    )
    case: bool = Field(default=True, description="Case-sensitive matching.")


def _glob_walk(root: Path, pattern: str) -> list[str]:
    """The walk half of execute_glob, run in a worker thread."""
    return sorted(
        p.relative_to(root).as_posix() + ("/" if p.is_dir() else "") for p in root.glob(pattern)
    )


def _glob_matches(rel_path: str, pattern: str) -> bool:
    """Match ``rel_path`` against ``pattern`` (basename fallback for bare globs)."""
    if fnmatch.fnmatch(rel_path, pattern):
        return True
    name = rel_path.rsplit("/", 1)[-1]
    return fnmatch.fnmatch(name, pattern)


def _walk_files(root: Path) -> list[Path]:
    """Walk ``root`` depth-first, pruning VCS/vendor/build trees and every
    dotdir (.git history and node_modules are noise the model never wants)."""
    files: list[Path] = []

    def _walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.is_symlink():
                continue  # never follow links: cycles and out-of-tree escapes
            if entry.is_dir():
                if entry.name in _GREP_PRUNE_DIRS or entry.name.startswith("."):
                    continue
                _walk(entry)
            elif entry.is_file():
                files.append(entry)

    _walk(root)
    return files


#: Wall-clock cap for one grep scan. Bounds the pathological-regex case
#: (backtracking patterns on large lines) without classifying regexes; a
#: scan that hits it returns what it has so far.
GREP_SCAN_DEADLINE_S = 30.0


def _grep_scan(
    files: list[Path],
    base: Path,
    regex: re.Pattern[str],
    include: str | None,
) -> tuple[list[str], int, int]:
    """The filesystem+regex half of execute_grep, run in a worker thread.

    Returns ``(matches, files_searched, files_skipped)``. Kept synchronous and
    self-contained so ``asyncio.to_thread`` can carry it off the event loop;
    the deadline bounds a backtracking pattern without touching the loop.
    """
    deadline = time.monotonic() + GREP_SCAN_DEADLINE_S
    matches: list[str] = []
    files_searched = 0
    files_skipped = 0
    for file_path in files:
        if time.monotonic() > deadline:
            break
        rel = (
            file_path.relative_to(base).as_posix()
            if base in file_path.parents or file_path == base
            else file_path.as_posix()
        )
        if include and not _glob_matches(rel, include):
            continue
        try:
            if file_path.stat().st_size > GREP_FILE_LIMIT_BYTES:
                files_skipped += 1
                continue
            data = file_path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8000]:
            continue  # binary file
        files_searched += 1
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                matches.append(f"{rel}:{lineno}:{line}")
                if len(matches) >= GREP_MATCH_LIMIT:
                    break
        if len(matches) >= GREP_MATCH_LIMIT:
            break
    return matches, files_searched, files_skipped


@_guard("grep")
async def execute_grep(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Regex search over files; ripgrep-free pure-Python implementation."""
    try:
        params = GrepParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "grep", exc)
    try:
        regex = re.compile(params.pattern, 0 if params.case else re.IGNORECASE)
    except re.error as exc:
        return _error(tool_call_id, "grep", f"invalid regex '{params.pattern}': {exc}")

    cwd = _safe_cwd(context)
    target, inside = _resolve_workspace_path(params.path, cwd)
    if not target.exists():
        return _error(tool_call_id, "grep", f"Path does not exist: {target}")

    # Outside-workspace searches escalate to an approval prompt regardless
    # of the read tier auto-approval the host normally applies.
    if not inside:
        description = _approval_description(target, inside, "grep")
        if not await _check_approval(context, "read", description):
            return _error(tool_call_id, "grep", "User declined to search this path.")

    if target.is_file():
        files: list[Path] = [target]
        base = target.parent
    else:
        base = target
        files = _walk_files(target)

    # The scan is FILESYSTEM + REGEX work on model-controlled input; running
    # it on the event loop would pin the CPU on a backtracking pattern or a
    # large tree and make Ctrl+C unprocessable. It runs in a worker thread
    # raced against the abort signal, with a wall-clock cap bounding the
    # pathological-regex case (regexes are not classified).
    scan_result, aborted = await _run_with_abort(
        asyncio.to_thread(_grep_scan, files, base, regex, params.include),
        signal,
        lambda: None,
    )
    if aborted:
        return _error(tool_call_id, "grep", "Search aborted.")
    matches, files_searched, files_skipped = scan_result

    if not matches:
        skipped_note = (
            f" ({files_skipped} file(s) skipped over the 1MB cap)" if files_skipped else ""
        )
        return _text(
            tool_call_id,
            "grep",
            f"No matches for '{params.pattern}' in {files_searched} " f"file(s){skipped_note}.",
            useless=True,
            details={"useless": True},
        )
    header = f"{len(matches)} match(es) for '{params.pattern}'"
    if len(matches) >= GREP_MATCH_LIMIT:
        header += f" (capped at {GREP_MATCH_LIMIT})"
    if files_skipped:
        header += f" ({files_skipped} file(s) skipped over the 1MB cap)"
    return _text(tool_call_id, "grep", header + ":\n" + "\n".join(matches))


def build_grep_tool() -> AgentTool:
    return AgentTool(
        name="grep",
        label="Grep",
        description="Regex search across files, returning 'path:line:text' matches.",
        parameters=GrepParams.model_json_schema(),
        approval_tier="read",
        # Read-only search; parallel greps are the common batch shape.
        concurrency="shared",
        interruptible=False,
        execute=execute_grep,
    )


# ---------------------------------------------------------------------------
# todo
# ---------------------------------------------------------------------------

#: In-memory todo lists keyed by NON-EMPTY session id. The host may attach a
#: durable store to the ToolContext (``todos`` dict) — we prefer that so
#: transcripts can replay todo state — but a bare context still works via
#: this table (keyed by the context object's id when no session id exists).
TODO_STORE: dict[str, list[dict[str, str]]] = {}
#: Fallback store for contexts without a session id, so their lists never
#: collide under the shared "" key.
_CONTEXT_TODO_STORE: dict[int, list[dict[str, str]]] = {}


class TodoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["init", "done", "view"] = Field(
        description="init: set the list; done: mark one item done; view: show the list."
    )
    items: list[str] = Field(
        default_factory=list,
        description="Todo texts (required for 'init', item text for 'done').",
    )


def _todo_store_and_key(
    context: ToolContext | None,
) -> tuple[dict[Any, list[dict[str, str]]], Any]:
    """Resolve ``(store, key)`` for this context. An attached ``todos`` dict
    wins; otherwise the module table keyed by session id; a context with NO
    session id gets its own slot keyed by object id, never the shared "" key.
    """
    store = getattr(context, "todos", None) if context else None
    if isinstance(store, dict):
        session_id = context.session_id if context else ""
        return store, session_id or id(context)
    if context is not None and context.session_id:
        return TODO_STORE, context.session_id
    return _CONTEXT_TODO_STORE, id(context)


@_guard("todo")
async def execute_todo(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Maintain a visible task list so progress survives compaction."""
    try:
        params = TodoParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "todo", exc)
    store, key = _todo_store_and_key(context)

    if params.op == "init":
        if not params.items:
            return _error(tool_call_id, "todo", "'init' requires a non-empty items list")
        store[key] = [{"text": item, "status": "pending"} for item in params.items]
        return _text(
            tool_call_id,
            "todo",
            f"Todo list initialized with {len(params.items)} item(s).",
        )

    current = store.get(key, [])
    if params.op == "done":
        if not params.items:
            return _error(tool_call_id, "todo", "'done' requires items with the item text")
        target = params.items[0]
        for item in current:
            if item["text"] == target and item["status"] != "done":
                item["status"] = "done"
                done = sum(1 for i in current if i["status"] == "done")
                return _text(
                    tool_call_id,
                    "todo",
                    f"Marked done: {target} ({done}/{len(current)} complete).",
                )
        return _error(
            tool_call_id,
            "todo",
            f"No pending todo matching '{target}'. Use todo view to see current items.",
        )

    # op == "view"
    if not current:
        return _text(
            tool_call_id,
            "todo",
            "No todos recorded yet.",
            useless=True,
            details={"useless": True},
        )
    marks = {"done": "x", "pending": " "}
    lines = [f"- [{marks.get(item['status'], ' ')}] {item['text']}" for item in current]
    return _text(tool_call_id, "todo", "\n".join(lines))


def build_todo_tool() -> AgentTool:
    return AgentTool(
        name="todo",
        label="Todo",
        description="Track a visible task list (init / done / view).",
        parameters=TodoParams.model_json_schema(),
        # read tier exemption: todo mutates only session-local bookkeeping
        # (no files, no autonomous turns), so it stays auto-approved.
        approval_tier="read",
        # init rewrites the whole list; concurrent calls would lose one,
        # so the tool runs exclusive despite being cheap.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_todo,
    )


# ---------------------------------------------------------------------------
# wake
# ---------------------------------------------------------------------------


class WakeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["create", "list", "cancel"] = Field(
        description="create: schedule a wake; list: show schedules; cancel: remove one."
    )
    message: str | None = Field(
        default=None, description="Message delivered when the wake fires (create only)."
    )
    # Scheduling (create) — one of `in` or `at` selects the first due time.
    field_in: str | None = Field(
        default=None,
        alias="in",
        description="Delay before first fire: '45s'|'30m'|'2h'|'7d'|'1w'.",
    )
    at: str | None = Field(
        default=None,
        description="First fire time: 'HH:MM', '+<duration>', or ISO datetime.",
    )
    every: str | None = Field(default=None, description="Repeat interval duration, e.g. '1h'.")
    until: str | None = Field(default=None, description="Retire after this time (ISO datetime).")
    limit: int | None = Field(default=None, ge=1, description="Max number of fires.")
    id: str | None = Field(default=None, description="Schedule id (cancel; from wake list).")


def _wake_due_label(schedule: Any) -> str:
    from local_operator.harness.wake import _format_duration

    due = datetime.fromtimestamp(schedule.next_due_at / 1000, tz=UTC)
    every = f" every {_format_duration(schedule.every_ms)}" if schedule.every_ms else ""
    fired = f" (fired {schedule.fired_count}x)" if schedule.fired_count else ""
    return f"next at {due.isoformat()}{every}{fired}"


async def _wake_list(tool_call_id: str, scheduler: Any) -> ToolResult:
    schedules = list(getattr(scheduler, "schedules", []))
    if not schedules:
        return _text(
            tool_call_id,
            "wake",
            "No wake schedules.",
            useless=True,
            details={"useless": True},
        )
    lines = [f'- {s.id}: "{s.message}" {_wake_due_label(s)}' for s in schedules]
    return _text(tool_call_id, "wake", f"{len(schedules)} wake schedule(s):\n" + "\n".join(lines))


async def _wake_create(
    tool_call_id: str, params: WakeParams, scheduler: Any, now_ms: int
) -> ToolResult:
    # Deferred import: wakes are optional for the session; a missing module
    # must not break tool import at startup.
    from local_operator.harness import wake as wake_module

    existing = list(getattr(scheduler, "schedules", []))
    request: dict[str, Any] = {
        "message": params.message or "",
        "in": params.field_in,
        "at": params.at,
        "every": params.every,
        "until": params.until,
        "limit": params.limit,
    }
    outcome = wake_module.build_wake_schedule(request, existing, now_ms)
    if "error" in outcome:
        return _error(tool_call_id, "wake", outcome["error"])
    schedule = outcome["schedule"]
    updated = [s for s in existing if s.id != schedule.id] + [schedule]
    await scheduler.update(updated)
    due = datetime.fromtimestamp(schedule.next_due_at / 1000, tz=UTC)
    return _text(
        tool_call_id,
        "wake",
        f"Scheduled wake '{schedule.id}' at {due.isoformat()}: \"{schedule.message}\"",
    )


async def _wake_cancel(tool_call_id: str, params: WakeParams, scheduler: Any) -> ToolResult:
    if not params.id:
        return _error(tool_call_id, "wake", "'cancel' requires the schedule id (see wake list)")
    existing = list(getattr(scheduler, "schedules", []))
    remaining = [s for s in existing if s.id != params.id]
    if len(remaining) == len(existing):
        ids = ", ".join(s.id for s in existing) or "none"
        return _error(
            tool_call_id,
            "wake",
            f"No wake schedule with id '{params.id}' (known: {ids})",
        )
    await scheduler.update(remaining)
    return _text(tool_call_id, "wake", f"Cancelled wake schedule '{params.id}'.")


def build_wake_tool(context: ToolContext) -> AgentTool | None:
    """CreateIf builder: the tool only exists when the context carries a wake
    scheduler. A session without wakes must not advertise a tool whose every
    call errors (the createIf convention)."""
    if getattr(context, "wake_scheduler", None) is None:
        return None
    return AgentTool(
        name="wake",
        label="Wake",
        description="Schedule a future wake (create/list/cancel), e.g. 'in 30m'.",
        parameters=WakeParams.model_json_schema(),
        # write tier: wake create persists schedules and arms unattended
        # future agent turns — the only tool that creates autonomous
        # execution, so it prompts like a mutation (the loop gates write/exec).
        approval_tier="write",
        # create/cancel rewrite the whole schedule list; two concurrent
        # calls would lose one, so the tool runs exclusive.
        concurrency="exclusive",
        interruptible=False,
        execute=execute_wake,
    )


@_guard("wake")
async def execute_wake(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Create, list, or cancel scheduled wakes via the session's scheduler."""
    try:
        params = WakeParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "wake", exc)
    scheduler = getattr(context, "wake_scheduler", None) if context else None
    if scheduler is None:
        return _error(
            tool_call_id,
            "wake",
            "Wake scheduling is not available in this session (no scheduler attached).",
        )
    now_ms = int(time.time() * 1000)
    if params.op == "list":
        return await _wake_list(tool_call_id, scheduler)
    if params.op == "create":
        if not params.message or not params.message.strip():
            return _error(tool_call_id, "wake", "'create' requires a non-empty message")
        if not params.field_in and not params.at:
            return _error(tool_call_id, "wake", "'create' requires 'in' or 'at'")
        return await _wake_create(tool_call_id, params, scheduler, now_ms)
    return await _wake_cancel(tool_call_id, params, scheduler)


# ---------------------------------------------------------------------------
# variables — list / read session variables (values never enter the prompt)
# ---------------------------------------------------------------------------


class ListVariablesParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadVariableParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Variable name to read.")


#: Safety cap on a single variable value returned to the model. Keeps an
#: accidental read of a huge value from blowing up context; oversize values
#: are elided with a marker rather than dumped in full.
MAX_VARIABLE_VALUE_CHARS = 4000


def _variable_store(context: ToolContext | None) -> Any:
    """The session's VariableStore, or a fresh env-only store as fallback.

    A session attaches its store (config variables + project file + env) to
    ``context.variables``; when absent (bare tool tests) we fall back to a
    store over the process environment so the tools still answer."""
    if context is not None and getattr(context, "variables", None) is not None:
        return context.variables
    from local_operator.variables import VariableStore

    return VariableStore(cwd=_safe_cwd(context))


@_guard("list_variables")
async def execute_list_variables(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Return variable NAMES only (never values) so the agent can pick what
    to read without pulling everything into context. One compact line each."""
    try:
        ListVariablesParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "list_variables", exc)
    names = _variable_store(context).names()
    shown = names if len(names) <= 100 else names[:100] + ["…"]
    body = "\n".join(shown) if shown else "(no variables defined)"
    return _text(
        tool_call_id,
        "list_variables",
        f"{len(names)} variable(s) available:\n{body}",
        details={"count": len(names)},
    )


@_guard("read_variable")
async def execute_read_variable(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Read ONE variable value on demand; unknown names return a not-found
    error (the loop surfaces it, the caller can list_variables)."""
    try:
        params = ReadVariableParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "read_variable", exc)
    if not params.name.strip():
        return _error(tool_call_id, "read_variable", "name must be a non-empty string")
    store = _variable_store(context)
    if params.name not in store.names():
        return _error(
            tool_call_id, "read_variable", f"unknown variable: {params.name} (see list_variables)"
        )
    try:
        value = store.read(params.name)
    except KeyError:
        return _error(tool_call_id, "read_variable", f"unknown variable: {params.name}")
    if value is None:
        value = ""
    if len(value) > MAX_VARIABLE_VALUE_CHARS:
        value = value[:MAX_VARIABLE_VALUE_CHARS] + f"\n[… {len(value)} chars elided …]"
    return _text(
        tool_call_id,
        "read_variable",
        value,
        details={"name": params.name, "chars": len(value)},
    )


def build_list_variables_tool() -> AgentTool:
    return AgentTool(
        name="list_variables",
        label="List variables",
        description="List available variable names (never their values).",
        parameters=ListVariablesParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_list_variables,
    )


def build_read_variable_tool() -> AgentTool:
    return AgentTool(
        name="read_variable",
        label="Read variable",
        description="Read the value of one named variable.",
        parameters=ReadVariableParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        interruptible=False,
        execute=execute_read_variable,
    )
