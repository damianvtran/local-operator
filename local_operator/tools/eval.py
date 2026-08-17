"""Persistent Python kernel tool (``eval``) — a session-scoped REPL.

WHY a kernel tool
-----------------
Modelling work constantly needs small computations — counts, sums, data
reshaping, checking a hypothesis about a library's behaviour — and paying a
``bash`` one-shot (spawn, import the interpreter fresh, recompute everything
from scratch on the next question) for each one is the expensive way. This
tool keeps ONE Python process per session whose namespace survives across
calls, so ``import``s amortise and intermediate state stays resident.

WHY these safety properties
---------------------------
* Arbitrary code execution, so ``approval_tier="exec"`` like bash.
* ``concurrency="exclusive"``: the namespace is STATE, and two parallel calls
  writing it would interleave variables the model believes are sequential.
  The registry never runs two ``eval`` calls at once.
* ``interruptible=True``: abort kills the worker. The code may be in any
  state mid-run, so the kernel is DISCARDED, not reused — a namespace that
  half-executed an abort is a corruptible witness. The same applies to a
  timeout and to a worker crash: every one of those paths reports honestly
  that session state was lost and the next call starts fresh.

Worker lifecycle is owned HERE, keyed by session: one kernel per
``context.session_id`` (object id as the fallback for hosts that pass no
id), an on-access idle reaper (no background timers to keep alive), an LRU
cap so one process cannot farm kernels without bound, and ``spill_truncate``
for the 8 KiB result budget — the same contract every other tool offers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal as process_signal
import sys
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import (
    TOOL_OUTPUT_LIMIT_CHARS,
    TRACEBACK_TAIL_CHARS,
    _bash_output_summary,
    _display_target,
    _error,
    _guard,
    _safe_cwd,
    _text,
    _validation_error,
    spill_truncate,
)

#: Per-call wall clock. Kernel calls are expected to be short (they replace
#: one-shot bash snippets, not training runs); 30s catches the runaway case
#: without punishing a slow first ``import``.
EVAL_DEFAULT_TIMEOUT_SECONDS = 30.0
#: Hard cap on the caller-chosen timeout — same role as bash's: a call longer
#: than this is a session bug, not a tool feature.
EVAL_MAX_TIMEOUT_SECONDS = 300.0
#: A kernel untouched for this long is closed on the NEXT eval call. On-access
#: reaping rather than a background task: a timer would have to be owned by an
#: event loop that outlives sessions, and 5 minutes of staleness costs nothing
#: compared to the complexity of keeping one alive correctly.
KERNEL_IDLE_SECONDS = 5 * 60.0
#: Most kernels ever resident. Each is a full interpreter (tens of MB RSS);
#: 4 concurrent sessions with live kernels is already an unusual host, and the
#: LRU eviction below makes room rather than refusing the call.
MAX_KERNELS = 4
#: Stderr kept from a crashed worker: enough for the fatal exception, the
#: rest is gone with the process. Same value the guard uses for its
#: tracebacks, for the same reason.
_CRASH_STDERR_TAIL_CHARS = TRACEBACK_TAIL_CHARS


class EvalParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    code: str = Field(description="Python code to run in the persistent session kernel.")
    timeout: float = Field(
        default=EVAL_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=EVAL_MAX_TIMEOUT_SECONDS,
        description="Max seconds before the kernel is killed (state is lost).",
    )


class _Kernel:
    """One live worker process plus its LRU bookkeeping."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.last_used = time.monotonic()


#: Session key -> kernel, least-recently-used first. Process-wide on purpose:
#: the worker's namespace is per SESSION, and sessions outlive the tool
#: objects the registry builds for each of them.
_KERNELS: OrderedDict[str, _Kernel] = OrderedDict()

#: References to in-flight reap tasks. A bare task can be garbage-collected
#: before it runs, which would strand the kill half-done — the same reason
#: bash keeps its reader tasks referenced.
_CLOSING: set[asyncio.Task[None]] = set()


class _WorkerCrash(Exception):
    """The worker died before answering a request; carries its stderr tail."""

    def __init__(self, stderr: str) -> None:
        super().__init__(stderr.strip() or "worker exited without a response")
        self.stderr = stderr


def _session_key(context: ToolContext | None) -> str:
    """Registry key for one session's kernel.

    ``id(context)`` is the documented fallback for hosts that pass no session
    id — and because the context object is rebuilt every turn, that fallback
    means a fresh kernel per call. Hosts that want persistence set
    ``session_id``; the fallback only has to be deterministic within a call.
    """
    if context is None:
        return "no-context"
    return context.session_id or f"ctx-{id(context):x}"


async def _close_kernel(kernel: _Kernel) -> None:
    """Kill/reap the worker process GROUP, then release its transport.

    Evaluated code can spawn descendants. The worker is a process-group leader
    by construction, so killing only the interpreter would leave those
    descendants running after a timeout, abort, eviction, or session close.
    """
    process = kernel.process
    killed_group = False
    if hasattr(os, "killpg"):
        try:
            # Attempt this even when the leader has already exited: its
            # descendants keep the original process group alive.
            os.killpg(process.pid, process_signal.SIGKILL)
            killed_group = True
        except (ProcessLookupError, PermissionError):
            pass
    if process.returncode is None and not killed_group:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1.0)
    transport = getattr(process, "_transport", None)
    if transport is not None:
        transport.close()


def _retire(kernel: _Kernel) -> None:
    """Discard a kernel without blocking the call that outgrew it.

    Registry mutations are synchronous (single-threaded loop, no awaits
    between read and write), so the kill itself must not become an await
    here — a slow reap during a timeout path would stall the very call being
    rescued. The await happens in a referenced background task instead.
    """
    task = asyncio.ensure_future(_close_kernel(kernel))
    _CLOSING.add(task)
    task.add_done_callback(_CLOSING.discard)


def _reap_idle(now: float) -> None:
    """Close kernels unused for :data:`KERNEL_IDLE_SECONDS` (on access)."""
    stale = [
        key for key, kernel in _KERNELS.items() if now - kernel.last_used > KERNEL_IDLE_SECONDS
    ]
    for key in stale:
        _retire(_KERNELS.pop(key))


def _remember(key: str, kernel: _Kernel) -> None:
    """Re-insert a healthy kernel as most-recently-used, evicting past the cap."""
    _KERNELS[key] = kernel
    _KERNELS.move_to_end(key)
    while len(_KERNELS) > MAX_KERNELS:
        _key, evicted = _KERNELS.popitem(last=False)
        # Safe by construction: this call's kernel was just moved to the
        # MRU end, and eviction only runs while more than one entry is
        # resident, so the LRU end it pops can never be the kernel in use.
        _retire(evicted)


async def _spawn(cwd: str) -> _Kernel:
    """Start a worker in the session's working directory.

    ``start_new_session`` detaches the kernel from the harness's process
    group: a terminal Ctrl-C aimed at the TUI must not SIGINT the kernel and
    silently destroy the persistence this tool exists to provide (the tool's
    own timeout/abort paths kill it explicitly).
    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-m",
        "local_operator.tools.eval_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        start_new_session=True,
    )
    return _Kernel(process)


async def _read_crash_stderr(kernel: _Kernel) -> str:
    """Best-effort stderr tail from a worker that stopped answering."""
    stream = kernel.process.stderr
    if stream is None:
        return ""
    try:
        raw = await asyncio.wait_for(stream.read(), timeout=1.0)
    except (TimeoutError, ConnectionResetError):
        return ""
    return raw.decode("utf-8", errors="replace")


async def _exchange(kernel: _Kernel, request: dict[str, Any], request_id: str) -> dict[str, Any]:
    """Send one request line, await its response line by id.

    Lines that are not this request's response are skipped, not fatal: user
    code can write to fd 1 directly (the worker only intercepts the
    ``sys.stdout`` OBJECT), and a protocol that dies on stray bytes would
    lose a healthy kernel over noise. EOF before a response is a crash.
    """
    stdin = kernel.process.stdin
    assert stdin is not None  # piped at spawn
    try:
        stdin.write((json.dumps(request) + "\n").encode())
        await stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        raise _WorkerCrash(await _read_crash_stderr(kernel)) from None

    stdout = kernel.process.stdout
    assert stdout is not None  # piped at spawn
    while True:
        try:
            line = await stdout.readline()
        except (ValueError, asyncio.LimitOverrunError, OSError) as exc:
            # ``readline`` raises ValueError when untrusted fd-1 output exceeds
            # StreamReader's line limit. Turn every protocol read failure into
            # the crash path so the popped kernel is killed rather than lost
            # outside the registry.
            raise _WorkerCrash(f"eval protocol stream failed: {type(exc).__name__}: {exc}") from exc
        if not line:
            raise _WorkerCrash(await _read_crash_stderr(kernel))
        try:
            response = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        if isinstance(response, dict) and response.get("id") == request_id:
            return response


def _lost_state_error(tool_call_id: str, reason: str) -> ToolResult:
    """The honest result for every path that had to kill the kernel.

    The model must not believe the namespace survived: the next call's
    ``NameError`` would otherwise read as a tool bug rather than the
    documented consequence, and the model would start avoiding the tool.
    """
    return _error(
        tool_call_id,
        "eval",
        f"{reason}\nALL session state (variables, imports) was lost — "
        "the kernel restarts fresh on the next call.",
    )


@_guard("eval")
async def execute_eval(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Run Python code in the session's persistent kernel."""
    try:
        params = EvalParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, "eval", exc)
    if not params.code.strip():
        return _error(tool_call_id, "eval", "code must be a non-empty string")

    # Pre-aborted signal: never spawn (or disturb a resident kernel) for a
    # call there is no intention to run.
    if signal is not None and signal.aborted:
        return _error(
            tool_call_id,
            "eval",
            f"aborted ({signal.reason or 'aborted'}): code not run",
        )

    key = _session_key(context)
    _reap_idle(time.monotonic())

    kernel = _KERNELS.pop(key, None)
    if kernel is not None and kernel.process.returncode is not None:
        # Dead husk (prior crash whose retire task has since reaped it) —
        # never send a request to a corpse.
        kernel = None
    if kernel is None:
        try:
            kernel = await _spawn(_safe_cwd(context))
        except OSError as exc:
            return _error(tool_call_id, "eval", f"failed to start Python kernel: {exc}")
    kernel.last_used = time.monotonic()

    request_id = uuid.uuid4().hex
    exchange = asyncio.create_task(
        _exchange(kernel, {"id": request_id, "code": params.code}, request_id)
    )
    abort_waiter = asyncio.create_task(signal.wait()) if signal is not None else None

    response: dict[str, Any] | None = None
    crash: _WorkerCrash | None = None
    timed_out = False
    aborted = False
    externally_cancelled = False
    try:
        waiters: list[asyncio.Task[Any]] = [exchange]
        if abort_waiter is not None:
            waiters.append(abort_waiter)
        done, _pending = await asyncio.wait(waiters, timeout=params.timeout)
        if exchange in done:
            try:
                response = exchange.result()
            except _WorkerCrash as exc:
                crash = exc
            except Exception as exc:  # noqa: BLE001 - protocol failures retire state
                crash = _WorkerCrash(f"eval protocol exchange failed: {type(exc).__name__}: {exc}")
        elif abort_waiter is not None and abort_waiter in done:
            aborted = True
        else:
            timed_out = True
    except asyncio.CancelledError:
        externally_cancelled = True
        raise
    finally:
        if response is None:
            exchange.cancel()
            with contextlib.suppress(BaseException):
                await exchange
        if abort_waiter is not None and not abort_waiter.done():
            abort_waiter.cancel()
            with contextlib.suppress(BaseException):
                await abort_waiter
        if externally_cancelled:
            _retire(kernel)

    if aborted or timed_out:
        # The code may be mid-run inside the kernel: reuse is not an option.
        _retire(kernel)
        if aborted:
            return _lost_state_error(
                tool_call_id,
                f"aborted ({signal.reason if signal else 'aborted'}): kernel killed mid-run",
            )
        return _lost_state_error(
            tool_call_id, f"TIMEOUT after {params.timeout}s: kernel killed mid-run"
        )
    if crash is not None:
        _retire(kernel)
        tail = crash.stderr[-_CRASH_STDERR_TAIL_CHARS:]
        detail = tail.strip() or "(no stderr)"
        return _error(
            tool_call_id,
            "eval",
            "Python kernel crashed while running the code (fatal signal or "
            "os._exit — it cannot be caught in-process).\n"
            f"Session state was lost — the kernel restarts fresh on the next "
            f"call.\n--- stderr (tail) ---\n{detail}",
        )

    # Success: the kernel stays resident for the next call.
    kernel.last_used = time.monotonic()
    _remember(key, kernel)
    return _render(tool_call_id, response or {}, context, on_update)


def _render(
    tool_call_id: str,
    response: dict[str, Any],
    context: ToolContext | None,
    on_update: Callable[[AgentToolUpdate], None] | None,
) -> ToolResult:
    """Build the ToolResult from one worker response.

    ``display()`` output goes to the update stream and ``details`` only —
    updates render in the UI and never enter the message history, and
    ``details`` is never serialized to providers, so the human sees it and
    the model does not. Everything the model reads is routed through
    ``spill_truncate`` for the shared 8 KiB budget with a spill handle.
    """
    ok = bool(response.get("ok"))
    stdout = str(response.get("stdout") or "")
    stderr = str(response.get("stderr") or "")
    result_repr = response.get("result")
    result_repr = str(result_repr) if result_repr is not None else None
    display = [str(item) for item in response.get("display") or []]

    if display and on_update is not None:
        on_update(
            AgentToolUpdate(
                content=[TextContent(text="\n".join(display))],
                details={"tool_name": "eval", "display": True},
            )
        )

    # The result leads so head-biased truncation keeps it: it is the one line
    # the call existed to produce.
    if ok:
        body = "\n".join(
            ([f"result: {result_repr}"] if result_repr is not None else [])
            + [_bash_output_summary(stdout, stderr)]
        )
        text, spill_details = spill_truncate(body, "eval", context, TOOL_OUTPUT_LIMIT_CHARS)
        details = spill_details
        if display:
            details = {**(details or {}), "display": display}
        return _text(tool_call_id, "eval", text, details=details)

    error = str(response.get("error") or "(no error reported)")
    body = "\n".join([error, _bash_output_summary(stdout, stderr)])
    text, spill_details = spill_truncate(body, "eval", context, TOOL_OUTPUT_LIMIT_CHARS)
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="eval",
        content=[TextContent(text=text)],
        details=spill_details,
        is_error=True,
    )


def _describe_eval_approval(args: dict[str, Any], cwd: str) -> str:
    """The approval sentence: the code's first line, not a JSON dump.

    ``exec``-tier prompts must show the decision-relevant argument, and for a
    REPL that is the code itself. Only the first line fits a prompt; the
    rest is in the transcript the reviewer already has open.
    """
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        return ""
    first = code.strip().splitlines()[0]
    if len(first) > 80:
        first = first[:77] + "..."
    return f"eval: {_display_target(first)}"


def build_eval_tool() -> AgentTool:
    return AgentTool(
        name="eval",
        label="Python",
        describe_approval=_describe_eval_approval,
        description=(
            "Run Python in a persistent per-session kernel. State (variables, "
            "imports, functions) survives across calls — build on earlier calls "
            "instead of recomputing. Compute facts, transform data, verify "
            "library behaviour: cheaper and safer than bash one-shots. The "
            "trailing expression's value is returned. display(value) shows "
            "output to the user only, excluded from your context; use it "
            "instead of print() when only the human needs to see something."
        ),
        parameters=EvalParams.model_json_schema(),
        approval_tier="exec",
        # The namespace is state: parallel calls would interleave variables
        # the model believes are sequential. Exclusive serialises them.
        concurrency="exclusive",
        # Abort kills the worker mid-run; the next call starts a fresh
        # kernel (state is honestly reported lost — see _lost_state_error).
        interruptible=True,
        execute=execute_eval,
    )
