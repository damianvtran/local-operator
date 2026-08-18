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
import subprocess
import sys
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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
#: Cap for a BACKGROUND run. The foreground cap exists to stop one call
#: blocking a turn for minutes; a background job blocks nothing, so that reason
#: does not apply and holding it there would have contradicted the mode's own
#: description ("training, large fetches, polling loops" — none of which fit in
#: five minutes). Matches bash's background ceiling so the two modes agree.
EVAL_MAX_BACKGROUND_TIMEOUT_SECONDS = 3600.0
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
        le=EVAL_MAX_BACKGROUND_TIMEOUT_SECONDS,
        description=(
            "Max seconds before the kernel is killed (state is lost). "
            f"Foreground calls are capped at {EVAL_MAX_TIMEOUT_SECONDS:.0f}s; "
            f"background=true allows up to "
            f"{EVAL_MAX_BACKGROUND_TIMEOUT_SECONDS:.0f}s."
        ),
    )

    @model_validator(mode="after")
    def _timeout_within_mode_cap(self) -> "EvalParams":
        """Hold foreground calls to the lower cap.

        The field bound has to admit the larger background ceiling, so the
        foreground limit is enforced here instead of being quietly dropped.
        Rejecting is the honest outcome: silently clamping would run the call
        for a different duration than the caller asked for.
        """
        if not self.background and self.timeout > EVAL_MAX_TIMEOUT_SECONDS:
            raise ValueError(
                f"timeout must be <= {EVAL_MAX_TIMEOUT_SECONDS:.0f}s for a "
                f"foreground call; pass background=true to run up to "
                f"{EVAL_MAX_BACKGROUND_TIMEOUT_SECONDS:.0f}s as a job."
            )
        return self

    background: bool = Field(
        default=False,
        description=(
            "Run in a DEDICATED kernel and return a job id immediately instead "
            "of waiting. Use for long work (training, large fetches, polling "
            "loops); follow it with jobs(op='peek') to read new output as it "
            "prints, and jobs(op='cancel') to stop it. The background kernel is "
            "separate, so it neither sees nor changes the session namespace."
        ),
    )


class _Kernel:
    """One live worker process plus its process-tree ownership and LRU state."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        windows_job: int | None = None,
    ) -> None:
        self.process = process
        # On Windows, closing this Job Object is the process-tree equivalent of
        # POSIX killpg. It is assigned before any user code can run.
        self.windows_job = windows_job
        self.last_used = time.monotonic()


#: Session key -> kernel, least-recently-used first. Process-wide on purpose:
#: the worker's namespace is per SESSION, and sessions outlive the tool
#: objects the registry builds for each of them.
_KERNELS: OrderedDict[str, _Kernel] = OrderedDict()

#: References to in-flight reap tasks. A bare task can be garbage-collected
#: before it runs, which would strand the kill half-done — the same reason
#: bash keeps its reader tasks referenced.
_CLOSING: set[asyncio.Task[None]] = set()

#: Live background kernels by job id. These are deliberately NOT in
#: ``_KERNELS``: that registry is the per-SESSION namespace cache with its own
#: LRU and idle reaping, and a background job's kernel is owned by the job, not
#: the session. Tracked anyway so a background worker is never an untracked
#: process — the entry is removed as soon as the kernel is retired, on every
#: path (completion, timeout, cancel, pre-start teardown).
_BACKGROUND_KERNELS: dict[str, asyncio.subprocess.Process] = {}


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


def _create_windows_kill_job(pid: int) -> int:
    """Own ``pid`` with a kill-on-close Windows Job Object.

    The worker initially blocks on stdin, so assignment happens before it can
    execute user code or spawn descendants. Refusing eval startup when this
    setup fails is safer than advertising a timeout that leaks child processes.
    """
    import ctypes
    from ctypes import wintypes

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    last_error = getattr(ctypes, "get_last_error")
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(last_error(), "CreateJobObjectW failed")
    try:
        limits = ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise OSError(last_error(), "SetInformationJobObject failed")

        process_handle = kernel32.OpenProcess(
            0x0100 | 0x0001,  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            False,
            pid,
        )
        if not process_handle:
            raise OSError(last_error(), "OpenProcess failed")
        try:
            if not kernel32.AssignProcessToJobObject(job, process_handle):
                raise OSError(last_error(), "AssignProcessToJobObject failed")
        finally:
            kernel32.CloseHandle(process_handle)
    except BaseException:
        kernel32.CloseHandle(job)
        raise
    return int(job)


def _close_windows_job(job: int) -> None:
    """Terminate the Job tree and close its kill-on-close native handle."""
    import ctypes

    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    last_error = getattr(ctypes, "get_last_error")
    kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = ctypes.c_void_p(job)
    terminated = bool(kernel32.TerminateJobObject(handle, 1))
    terminate_error = last_error()
    closed = bool(kernel32.CloseHandle(handle))
    close_error = last_error()
    # Either operation is sufficient to terminate the owned tree:
    # TerminateJobObject does it directly; successful CloseHandle triggers the
    # KILL_ON_JOB_CLOSE limit. Both failing means containment was not honored.
    if not terminated and not closed:
        raise OSError(
            terminate_error or close_error,
            "TerminateJobObject and CloseHandle(job) both failed",
        )


async def _taskkill_windows_tree(pid: int) -> bool:
    """Last-resort Windows tree termination if a native Job handle fails."""
    taskkill = os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"),
        "System32",
        "taskkill.exe",
    )
    try:
        killer = await asyncio.create_subprocess_exec(
            taskkill,
            "/PID",
            str(pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await asyncio.wait_for(killer.wait(), timeout=5.0) == 0
    except (OSError, TimeoutError):
        return False


async def _close_kernel(kernel: _Kernel) -> None:
    """Kill/reap the worker process GROUP, then release its transport.

    Evaluated code can spawn descendants. The worker is a process-group leader
    by construction, so killing only the interpreter would leave those
    descendants running after a timeout, abort, eviction, or session close.
    """
    process = kernel.process
    killed_group = False
    if kernel.windows_job is not None:
        try:
            _close_windows_job(kernel.windows_job)
            killed_group = True
        except OSError:
            killed_group = await _taskkill_windows_tree(process.pid)
        finally:
            # Never close/reuse a native handle twice, including when close
            # itself reports an error.
            kernel.windows_job = None
    elif hasattr(os, "killpg"):
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
    """Start a worker with platform-native process-tree ownership.

    POSIX uses a new session/process group. Windows assigns the worker to a
    kill-on-close Job Object before the first request is sent.
    """
    spawn_options: dict[str, Any] = {
        "stdin": asyncio.subprocess.PIPE,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": cwd,
    }
    if sys.platform == "win32":
        spawn_options["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        spawn_options["start_new_session"] = True
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-m",
        "local_operator.tools.eval_worker",
        **spawn_options,
    )
    windows_job: int | None = None
    if sys.platform == "win32":
        try:
            windows_job = _create_windows_kill_job(process.pid)
        except OSError:
            # No request has been sent, so the worker cannot have descendants.
            # Reap this leader and refuse unsafe eval startup.
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=2.0)
            transport = getattr(process, "_transport", None)
            if transport is not None:
                transport.close()
            raise
    return _Kernel(process, windows_job=windows_job)


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


async def _exchange(
    kernel: _Kernel,
    request: dict[str, Any],
    request_id: str,
    on_stream: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """Send one request line, await its response line by id.

    Lines that are not this request's response are skipped, not fatal: user
    code can write to fd 1 directly (the worker only intercepts the
    ``sys.stdout`` OBJECT), and a protocol that dies on stray bytes would
    lose a healthy kernel over noise. EOF before a response is a crash.

    ``on_stream`` receives intermediate ``stream`` frames (channel, text) when
    the request asked for them. They are progress only: the response still
    carries the complete captured streams, so this callback never has to be
    reconciled with the final result.
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
        if not isinstance(response, dict) or response.get("id") != request_id:
            continue
        channel = response.get("stream")
        if channel:
            # A progress frame, not the answer: publish it and keep reading.
            if on_stream is not None:
                with contextlib.suppress(Exception):
                    on_stream(str(channel), str(response.get("text") or ""))
            continue
        return response


async def _run_in_background(
    tool_call_id: str,
    params: EvalParams,
    context: ToolContext | None,
) -> ToolResult:
    """Run ``code`` as a background job in a kernel of its own.

    A DEDICATED kernel rather than the session's, for the same reason the tool
    is ``exclusive`` in the first place: the namespace is state, and a job that
    keeps mutating it for the next half hour would interleave with every
    foreground call the model makes meanwhile. Isolation is the honest trade —
    the background run cannot see session variables, and the tool description
    says so, which is far better than a shared namespace that silently
    corrupts. The kernel is never registered in ``_KERNELS``: it belongs to the
    job, and it is killed when the job settles, times out, or is cancelled.
    """
    jobs = context.jobs if context is not None else None
    if jobs is None:
        return _error(
            tool_call_id,
            "eval",
            "background=true needs a job manager, which this session has not "
            "attached; re-run without background.",
        )
    try:
        kernel = await _spawn(_safe_cwd(context))
    except OSError as exc:
        return _error(tool_call_id, "eval", f"failed to start Python kernel: {exc}")

    code = params.code
    timeout = params.timeout

    async def _runner(job_id: str, job_signal: Any, report_progress: Any) -> str:
        appender = getattr(jobs, "append_output", None)

        def _publish(channel: str, text: str) -> None:
            if appender is not None:
                appender(job_id, text)
            line = text.strip().splitlines()
            if line:
                report_progress(line[-1][:200])

        request_id = uuid.uuid4().hex
        exchange = asyncio.create_task(
            _exchange(
                kernel,
                {"id": request_id, "code": code, "stream": True},
                request_id,
                _publish,
            )
        )
        abort_waiter = asyncio.create_task(job_signal.wait()) if job_signal is not None else None
        waiters: list[asyncio.Task[Any]] = [exchange]
        if abort_waiter is not None:
            waiters.append(abort_waiter)
        try:
            done, _pending = await asyncio.wait(
                waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if exchange in done:
                try:
                    response = exchange.result()
                except _WorkerCrash as exc:
                    tail = exc.stderr[-_CRASH_STDERR_TAIL_CHARS:].strip() or "(no stderr)"
                    return f"Python kernel crashed.\n--- stderr (tail) ---\n{tail}"
                return _background_summary(response, context)
            if abort_waiter is not None and abort_waiter in done:
                return "CANCELLED: background kernel killed mid-run"
            return f"TIMEOUT after {timeout}s: background kernel killed mid-run"
        finally:
            # The kernel is this job's alone; it dies with the job on every
            # path, including cancellation, so a killed job cannot leave an
            # interpreter (and whatever it spawned) running untracked.
            for task in (exchange, abort_waiter):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task
            _forget_kernel()
            _retire(kernel)

    # Filled in once ``register`` returns the id; the runner and the teardown
    # hook both need it to drop their tracking entry.
    job_slot: dict[str, str | None] = {"id": None}

    def _forget_kernel() -> None:
        job_id = job_slot["id"]
        if job_id is not None:
            _BACKGROUND_KERNELS.pop(job_id, None)

    def _kill_unstarted_kernel() -> None:
        """Teardown for a cancel that lands before the runner is ever entered.

        The kernel is spawned above, BEFORE ``register``, so between those two
        points it is owned by nothing the manager can see. ``register`` only
        schedules the runner, so a cancel in that same event-loop turn would
        settle the row without the runner's ``finally`` ever running and leave
        this interpreter alive, reparented to init. The manager drops this the
        moment the runner starts, so the kernel is never killed twice.
        """
        _forget_kernel()
        _retire(kernel)

    try:
        job_id = cast(Any, jobs).register(
            # ``"bash"``, not a new ``"eval"`` literal, and that is
            # load-bearing: ``session`` only auto-delivers jobs whose type is
            # in ``("task", "bash")``, so a third literal would silently strip
            # background evals of the completion message that is the whole
            # point of running them detached. ``JobType`` would have to gain
            # the member and every reader branch on it before this can change;
            # the label carries the distinction meanwhile.
            "bash",
            f"eval: {code.strip().splitlines()[0][:60] if code.strip() else 'code'}",
            _runner,
            # The owner is whoever asked for the work: a subagent's background
            # job must stay inside that subagent's scope, or owner-scoped
            # ``cancel``/``list`` stop containing it and its completion routes
            # to the parent's fallback sink instead of the child's.
            owner_id=context.job_id if context is not None else None,
            on_cancel=_kill_unstarted_kernel,
        )
    except Exception:  # noqa: BLE001 — no slot for the job: kill, don't leak
        _retire(kernel)
        raise
    job_slot["id"] = job_id
    _BACKGROUND_KERNELS[job_id] = kernel.process
    return _text(
        tool_call_id,
        "eval",
        f"started in the background as job {job_id} (use jobs op='peek' for new "
        f"output, op='cancel' to stop it); its result auto-delivers when it "
        f"finishes.\nNOTE: a background run uses its own kernel, so it does not "
        f"share the session namespace.",
        details={"job_id": job_id, "backgrounded": True},
    )


def _background_summary(response: dict[str, Any], context: ToolContext | None) -> str:
    """Render a finished background run the way the foreground result reads.

    Passed through ``spill_truncate`` like every other eval and bash result:
    the worker caps each stream at 1 MB, and a job that produced that much
    would otherwise store it whole in ``result_text`` and hand all of it to the
    model when the completion auto-delivers. The full text stays reachable
    through the spill handle, which is the same bargain the foreground path
    already makes.
    """
    stdout = str(response.get("stdout") or "")
    stderr = str(response.get("stderr") or "")
    error = response.get("error")
    result = response.get("result")
    parts: list[str] = []
    if error:
        parts.append(str(error))
    elif result is not None:
        parts.append(f"result: {result}")
    parts.append(f"--- stdout ---\n{stdout if stdout else '(empty)'}")
    parts.append(f"--- stderr ---\n{stderr if stderr else '(empty)'}")
    text, _spill_details = spill_truncate("\n".join(parts), "eval", context)
    return text


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

    if params.background:
        return await _run_in_background(tool_call_id, params, context)

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
        # FIRST_COMPLETED is load-bearing, not a micro-optimisation. asyncio.wait
        # defaults to ALL_COMPLETED, and ``abort_waiter`` only completes if the
        # session is aborted — so with a signal attached (which is every real
        # call: the loop hands one to every tool) the default made this await
        # block for the WHOLE ``timeout`` even after the kernel had already
        # answered. The response was still correct, so the bug was invisible in
        # the output and visible only as wall clock: a call whose code raised a
        # NameError in microseconds sat for the full 300s the caller allowed.
        # That also inverted the parameter's contract — ``timeout`` is a MAX, and
        # raising it must never make a fast call slower.
        done, _pending = await asyncio.wait(
            waiters, timeout=params.timeout, return_when=asyncio.FIRST_COMPLETED
        )
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
