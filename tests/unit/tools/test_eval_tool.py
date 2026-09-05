"""End-to-end tests for the ``eval`` persistent-kernel tool.

Each acceptance property of the tool is exercised against the REAL worker
subprocess (spawned exactly as the tool spawns it): persistence across
calls, trailing-expression results, output budget/spill, display() routing,
clean syntax errors, honest state-loss on timeout/crash with fresh restart,
and per-session kernel isolation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import pytest_asyncio

from local_operator.harness.types import (
    AbortSignal,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.tools import builtin
from local_operator.tools import eval as eval_tool


@pytest_asyncio.fixture(autouse=True)
async def _clean_kernel_registry():
    """Isolate the module-level kernel registry, killing any workers a test
    leaves behind — a leaked interpreter outlives the test that spawned it."""
    eval_tool._KERNELS.clear()
    eval_tool._LOST_KERNELS.clear()
    eval_tool._ACTIVE_KERNELS.clear()
    eval_tool._CLOSE_ON_RETURN.clear()
    yield
    for kernel in list(eval_tool._KERNELS.values()):
        await eval_tool._close_kernel(kernel)
    eval_tool._KERNELS.clear()
    for task in list(eval_tool._CLOSING):
        task.cancel()


@pytest.fixture
def context(tmp_path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="eval-unit")


async def _call(
    context: ToolContext,
    code: str,
    *,
    timeout: float | None = None,
    signal: AbortSignal | None = None,
    on_update: Any = None,
) -> ToolResult:
    """Invoke the tool the way the loop does (fresh builder per call is fine;
    the kernel state lives in the module registry, not the tool object)."""
    tool = eval_tool.build_eval_tool()
    args: dict[str, Any] = {"code": code}
    if timeout is not None:
        args["timeout"] = timeout
    return await tool.execute("call-1", args, signal, on_update, context)  # type: ignore[operator]


def _updates_to_list() -> tuple[list[AgentToolUpdate], Any]:
    captured: list[AgentToolUpdate] = []
    return captured, (lambda update: captured.append(update))


# ---------------------------------------------------------------------------
# persistence + result contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_persists_across_calls(context) -> None:
    first = await _call(context, "x = 41")
    assert first.is_error is False
    second = await _call(context, "x + 1")
    assert second.is_error is False
    assert "result: 42" in second.text


@pytest.mark.asyncio
async def test_imports_persist(context) -> None:
    await _call(context, "import json")
    result = await _call(context, "json.dumps({'a': 1})")
    assert result.is_error is False
    assert "result: '{\"a\": 1}'" in result.text


@pytest.mark.asyncio
async def test_trailing_expression_result_and_stdout(context) -> None:
    result = await _call(context, "print('hello')\n2 ** 10")
    assert result.is_error is False
    assert "result: 1024" in result.text
    assert "hello" in result.text
    assert "--- stdout ---" in result.text


@pytest.mark.asyncio
async def test_statement_only_call_has_no_result_line(context) -> None:
    result = await _call(context, "y = 5")
    assert result.is_error is False
    assert "result:" not in result.text


@pytest.mark.asyncio
async def test_worker_runs_in_session_cwd(context, tmp_path) -> None:
    result = await _call(context, "open('probe.txt', 'w').write('hi')\n'written'")
    assert result.is_error is False
    assert (tmp_path / "probe.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_namespace_docstring_explains_persistence(context) -> None:
    result = await _call(context, "__doc__")
    assert result.is_error is False
    assert "SURVIVES" in result.text


# ---------------------------------------------------------------------------
# display routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_display_lands_in_details_and_updates_not_model_text(context) -> None:
    captured, on_update = _updates_to_list()
    result = await _call(
        context,
        "display('for the user')\nprint('printed')\n'for the model'",
        on_update=on_update,
    )
    assert result.is_error is False
    # Model-visible text: the result and stdout, never display().
    assert "for the model" in result.text
    assert "printed" in result.text
    assert "for the user" not in result.text
    # details carries it for renderers/transcripts (never sent to providers).
    assert result.details is not None
    assert "for the user" in result.details["display"][0]
    # update stream shows the human what display() produced.
    assert captured, "display output should be streamed via on_update"
    assert "for the user" in captured[-1].content[0].text  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_display_is_per_call_not_sticky(context) -> None:
    await _call(context, "display('first call')")
    result = await _call(context, "'second call'")
    assert "first call" not in result.text
    assert result.details is None or "display" not in result.details


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_syntax_error_is_clean_error_result(context) -> None:
    result = await _call(context, "def broken(:")
    assert result.is_error is True
    assert "SyntaxError" in result.text
    # Clean: a compile failure has no frames worth showing.
    assert "Traceback" not in result.text


@pytest.mark.asyncio
async def test_runtime_error_reports_traceback_and_kernel_survives(context) -> None:
    await _call(context, "keeper = 'kept'")
    result = await _call(context, "raise ValueError('boom')")
    assert result.is_error is True
    assert "ValueError: boom" in result.text
    assert "Traceback" in result.text
    after = await _call(context, "keeper")
    assert after.is_error is False
    assert "result: 'kept'" in after.text


@pytest.mark.asyncio
async def test_invalid_arguments_and_empty_code(context) -> None:
    tool = eval_tool.build_eval_tool()
    invalid = await tool.execute("call-1", {}, None, None, context)  # type: ignore[operator]
    assert invalid.is_error is True
    assert "invalid arguments" in invalid.text
    empty = await _call(context, "   ")
    assert empty.is_error is True
    assert "non-empty" in empty.text


# ---------------------------------------------------------------------------
# timeout / abort / crash: state honestly lost, fresh kernel next call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_reports_lost_state_and_next_call_is_fresh(context) -> None:
    await _call(context, "x = 1")
    timed_out = await _call(context, "while True:\n    pass", timeout=0.5)
    assert timed_out.is_error is True
    assert "TIMEOUT" in timed_out.text
    assert "state" in timed_out.text
    assert "fresh" in timed_out.text
    # The next call runs on a NEW kernel: it succeeds, and the old namespace
    # is genuinely gone.
    recovered = await _call(context, "1 + 1")
    assert recovered.is_error is False
    assert "result: 2" in recovered.text
    gone = await _call(context, "x")
    assert gone.is_error is True
    assert "NameError" in gone.text


@pytest.mark.asyncio
async def test_fast_call_returns_immediately_despite_a_long_timeout(context) -> None:
    """``timeout`` is a MAX, not a fixed wall clock the call is held against.

    Regression for the ALL_COMPLETED default in ``asyncio.wait``: with a live
    (never-aborted) signal attached, the tool waited on the abort task too, so
    a call that had already answered was still held for the entire timeout.
    Raising ``timeout`` must never make a fast call slower, so this asserts on
    elapsed time an order of magnitude under the timeout it allows.
    """
    signal = AbortSignal()
    await _call(context, "warm = 1", signal=signal)  # pay the spawn cost first
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await _call(context, "1 + 1", timeout=30, signal=signal)
    elapsed = loop.time() - started
    assert result.is_error is False
    assert "result: 2" in result.text
    assert elapsed < 5.0, f"fast call held for {elapsed:.1f}s of its 30s timeout"


@pytest.mark.asyncio
async def test_failing_code_returns_immediately_despite_a_long_timeout(context) -> None:
    """The reported symptom: code that raises returns its traceback at once.

    A ``NameError`` is raised in microseconds, but the hang was in the parent's
    wait rather than in the worker, so an erroring call burned the full timeout
    before showing a traceback that had been ready the whole time.
    """
    signal = AbortSignal()
    await _call(context, "warm = 1", signal=signal)
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await _call(context, "undefined_name_here()", timeout=30, signal=signal)
    elapsed = loop.time() - started
    assert result.is_error is True
    assert "NameError" in result.text
    assert elapsed < 5.0, f"failing call held for {elapsed:.1f}s of its 30s timeout"


@pytest.mark.asyncio
async def test_timeout_still_fires_with_a_live_signal_attached(context) -> None:
    """The fix must not cost the timeout itself: genuinely slow code is killed.

    Pairs with the two tests above — together they pin ``timeout`` as an upper
    bound that is enforced but never waited out.
    """
    signal = AbortSignal()
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await _call(context, "while True:\n    pass", timeout=0.5, signal=signal)
    elapsed = loop.time() - started
    assert result.is_error is True
    assert "TIMEOUT" in result.text
    assert elapsed < 10.0, f"timeout kill took {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_abort_mid_run_still_interrupts_with_a_long_timeout(context) -> None:
    """Abort must still win the race it shares with the exchange.

    ``FIRST_COMPLETED`` is what makes whichever of the two finishes first the
    outcome; this asserts the abort branch is still reachable and prompt, so
    the fix cannot be mistaken for "just drop the abort waiter".
    """
    signal = AbortSignal()
    await _call(context, "warm = 1", signal=signal)
    loop = asyncio.get_running_loop()

    async def abort_soon() -> None:
        await asyncio.sleep(0.2)
        signal.abort("user steering")

    started = loop.time()
    aborter = asyncio.create_task(abort_soon())
    result = await _call(context, "import time\ntime.sleep(60)", timeout=120, signal=signal)
    elapsed = loop.time() - started
    await aborter
    assert result.is_error is True
    assert "aborted" in result.text
    assert elapsed < 10.0, f"abort took {elapsed:.1f}s to interrupt"


@pytest.mark.asyncio
async def test_worker_crash_reports_stderr_tail_and_restarts(context) -> None:
    await _call(context, "x = 1")
    crashed = await _call(context, "import os\nos._exit(3)")
    assert crashed.is_error is True
    assert "crashed" in crashed.text
    after = await _call(context, "2 * 21")
    assert after.is_error is False
    assert "result: 42" in after.text
    # The pre-crash namespace died with the process.
    gone = await _call(context, "x")
    assert gone.is_error is True
    assert "NameError" in gone.text


@pytest.mark.asyncio
async def test_preaborted_signal_never_spawns(context) -> None:
    signal = AbortSignal()
    signal.abort("user steering")
    result = await _call(context, "1", signal=signal)
    assert result.is_error is True
    assert "aborted" in result.text
    assert eval_tool._KERNELS == {}


# ---------------------------------------------------------------------------
# session isolation + lifecycle caps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_sessions_get_independent_kernels(tmp_path) -> None:
    session_a = ToolContext(cwd=str(tmp_path), session_id="session-a")
    session_b = ToolContext(cwd=str(tmp_path), session_id="session-b")
    await _call(session_a, "shared = 'A'")
    # B has its own kernel: A's variable is not there.
    missing = await _call(session_b, "shared")
    assert missing.is_error is True
    assert "NameError" in missing.text
    await _call(session_b, "shared = 'B'")
    kept_a = await _call(session_a, "shared")
    assert kept_a.is_error is False
    assert "result: 'A'" in kept_a.text
    assert len(eval_tool._KERNELS) == 2


@pytest.mark.asyncio
async def test_kernel_lru_cap_evicts_oldest_session(tmp_path) -> None:
    contexts = [ToolContext(cwd=str(tmp_path), session_id=f"cap-{i}") for i in range(5)]
    for index, ctx in enumerate(contexts):
        await _call(ctx, f"n = {index}")
    assert len(eval_tool._KERNELS) == eval_tool.MAX_KERNELS
    assert "cap-0" not in eval_tool._KERNELS
    assert "cap-4" in eval_tool._KERNELS
    # A namespace reset is surfaced before code can take a wrong branch or
    # repeat a side effect against an unexpectedly empty namespace.
    sentinel = tmp_path / "must-not-run"
    reset = await _call(contexts[0], f"open({str(sentinel)!r}, 'w').write('bad')")
    assert reset.is_error
    assert (reset.details or {}).get("kernel_reset")
    assert (reset.details or {}).get("code_executed") is False
    assert not sentinel.exists()
    rebuilt = await _call(contexts[0], "n = 123; n")
    assert not rebuilt.is_error
    assert "result: 123" in rebuilt.text


@pytest.mark.asyncio
async def test_idle_kernel_budget_accounts_for_live_work(context, monkeypatch) -> None:
    await _call(context, "n = 1")
    kernel = eval_tool._KERNELS[context.session_id]
    monkeypatch.setattr(eval_tool, "_ACTIVE_KERNELS", {"busy-1", "busy-2", "busy-3"})
    other = await eval_tool._spawn(str(context.cwd))
    eval_tool._remember("other", other)
    assert len(eval_tool._KERNELS) == 1
    assert context.session_id not in eval_tool._KERNELS
    assert "evicted" in eval_tool._LOST_KERNELS[context.session_id]
    await eval_tool._close_kernel(kernel)


@pytest.mark.asyncio
async def test_eval_tool_bridge_uses_this_calls_dispatch(context) -> None:
    seen = []

    async def dispatch(name, arguments):
        seen.append((name, arguments))
        return {"is_error": False, "text": "answer"}

    context.dispatch_tool = dispatch
    result = await _call(context, "tool('read', path='a')['text']")
    assert not result.is_error, result.text
    assert "answer" in result.text
    assert seen == [("read", {"path": "a"})]
    # The same persistent kernel must not retain the prior turn's callback.
    context.dispatch_tool = None
    unavailable = await _call(context, "tool('read', path='a')")
    assert unavailable.is_error
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_session_dispose_retires_idle_kernel(context) -> None:
    await _call(context, "n = 1")
    kernel = eval_tool._KERNELS[context.session_id]
    await eval_tool.close_session_kernel(context.session_id)
    assert context.session_id not in eval_tool._KERNELS
    assert kernel.process.returncode is not None


@pytest.mark.asyncio
async def test_dispose_during_exchange_cannot_repopulate_idle_pool(context, monkeypatch) -> None:
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    original = eval_tool._exchange

    async def exchange(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(eval_tool, "_exchange", exchange)
    work = asyncio.create_task(_call(context, "n = 1"))
    await entered.wait()
    await eval_tool.close_session_kernel(context.session_id)
    release.set()
    assert not (await work).is_error
    assert context.session_id not in eval_tool._KERNELS
    assert context.session_id not in eval_tool._CLOSE_ON_RETURN


@pytest.mark.asyncio
async def test_dead_idle_kernel_reports_reset_before_any_new_code(context, tmp_path) -> None:
    await _call(context, "state = 123")
    kernel = eval_tool._KERNELS[context.session_id]
    kernel.process.kill()
    await kernel.process.wait()
    sentinel = tmp_path / "must-not-execute"
    result = await _call(context, f"open({str(sentinel)!r}, 'w').write('bad')")
    assert result.is_error
    assert (result.details or {}).get("code_executed") is False
    assert "idle Python kernel exited" in result.text
    assert not sentinel.exists()
    recovered = await _call(context, "'state' in globals()")
    assert not recovered.is_error
    assert "False" in recovered.text


@pytest.mark.asyncio
async def test_idle_kernel_is_reaped_on_access(context) -> None:
    await _call(context, "a = 1")
    key = next(iter(eval_tool._KERNELS))
    stale_pid = eval_tool._KERNELS[key].process.pid
    # Simulate the 5-minute idle window without waiting for it.
    eval_tool._KERNELS[key].last_used -= eval_tool.KERNEL_IDLE_SECONDS + 1
    result = await _call(context, "a = 2")
    assert result.is_error is True
    assert (result.details or {}).get("code_executed") is False
    assert "idle kernel expired" in result.text
    result = await _call(context, "a = 2")
    assert result.is_error is False
    fresh = eval_tool._KERNELS[key]
    assert fresh.process.pid != stale_pid
    # The reaped kernel's state is gone too — a NEW process answered.
    assert "result:" not in result.text


# ---------------------------------------------------------------------------
# output budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stdout_budget_capped_with_recovery_route(context) -> None:
    result = await _call(context, "print('x' * 20_000)\n'the answer'")
    assert result.is_error is False
    # Model-visible text stays inside the 8 KiB budget (plus the bounded
    # footer a spill/truncation adds)…
    assert len(result.text) <= builtin.TOOL_OUTPUT_LIMIT_CHARS + 700
    # …truncation is ANNOUNCED, not silent…
    assert "truncated" in result.text or "SAVED at spill://" in result.text
    # …and the head (result line) survives the clip.
    assert "result: 'the answer'" in result.text


@pytest.mark.asyncio
async def test_render_does_not_block_the_event_loop(context, monkeypatch) -> None:
    """A spilled result must build off the loop Textual renders on.

    Regression for the operator's "TUI freezes on eval" report. Building an
    oversized result runs ``spill_truncate``, whose store-eviction sweep is
    O(entries) synchronous disk I/O; run inline on the event loop it froze the
    render loop. The fix offloads that tail with ``asyncio.to_thread``.

    The proof is DETERMINISTIC, not a wall-clock threshold that depends on how
    slow the host's disk is (an earlier version of this test used a 50 ms bound
    that the inline path slipped under on fast hardware, so it did not actually
    guard the fix). We replace ``spill_truncate`` with a wrapper that blocks for
    a FIXED ``block_s`` before doing the real work, then measure the longest gap
    between 5 ms heartbeat callbacks scheduled on the same loop while the eval
    runs. If the blocking call executes ON the loop, every heartbeat during that
    fixed window is delayed and ``max_gap`` is forced to at least ``block_s``; if
    it executes in a worker thread, the loop keeps ticking and ``max_gap`` stays
    near the heartbeat interval. Asserting ``max_gap`` well below ``block_s``
    therefore fails on the inline path and passes on the offloaded path,
    independent of disk speed. (Verified: reverting ``_render`` to call
    ``_build_render_result`` inline makes this test fail; the fix makes it pass.)
    """
    import time as _time

    from local_operator.tools import builtin as _builtin

    block_s = 0.3  # fixed, far above any scheduling jitter — the whole point
    real_spill_truncate = _builtin.spill_truncate

    def _blocking_spill_truncate(*args, **kwargs):
        # A synchronous stand-in for the O(entries) eviction sweep: a fixed
        # sleep the loop can only tolerate if this runs off the loop thread.
        _time.sleep(block_s)
        return real_spill_truncate(*args, **kwargs)

    # Patch the name the render path actually calls (eval imports it by name).
    monkeypatch.setattr(eval_tool, "spill_truncate", _blocking_spill_truncate)

    loop = asyncio.get_running_loop()
    interval = 0.005
    state = {"last": loop.time(), "max_gap": 0.0, "on": True}

    def _beat() -> None:
        now = loop.time()
        state["max_gap"] = max(state["max_gap"], now - state["last"])
        state["last"] = now
        if state["on"]:
            loop.call_later(interval, _beat)

    loop.call_later(interval, _beat)
    await asyncio.sleep(0.05)  # let the heartbeat settle before measuring
    state["max_gap"] = 0.0
    state["last"] = loop.time()

    # ~30 KB of stdout: over the 8 KiB spill threshold, so the render path goes
    # through spill_truncate (our blocking stand-in) rather than returning inline.
    result = await _call(context, "print('lorem ipsum ' * 2000)")
    await asyncio.sleep(0.05)
    state["on"] = False

    assert result.is_error is False
    # Inline execution forces max_gap >= block_s (0.3 s); off-loop keeps it near
    # the 5 ms interval. Half of block_s cleanly separates the two outcomes and
    # leaves generous room for scheduler jitter.
    assert state["max_gap"] < block_s / 2, (
        f"event loop stalled {state['max_gap'] * 1000:.0f} ms during eval render "
        f"(block_s={block_s * 1000:.0f} ms ran on the loop, not off it)"
    )


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------


def test_tool_shape() -> None:
    tool = eval_tool.build_eval_tool()
    assert tool.name == "eval"
    assert tool.label == "Python"
    assert tool.approval_tier == "exec"
    assert tool.concurrency == "exclusive"
    assert tool.interruptible is True
    properties = tool.parameters["properties"]
    assert "code" in properties
    # The `i` intent field is the registry's injection, not ours.
    assert "i" not in properties
    assert "timeout" in properties
    # The SCHEMA bound is the background ceiling, because one field serves both
    # modes; the lower foreground cap is enforced by the model validator, which
    # is asserted by test_background_timeout_may_exceed_the_foreground_cap.
    assert (
        tool.parameters["properties"]["timeout"]["maximum"]
        == eval_tool.EVAL_MAX_BACKGROUND_TIMEOUT_SECONDS
    )
    assert eval_tool.EVAL_MAX_TIMEOUT_SECONDS < eval_tool.EVAL_MAX_BACKGROUND_TIMEOUT_SECONDS


def test_describe_approval_shows_code_first_line() -> None:
    sentence = eval_tool._describe_eval_approval({"code": "x = 1\ny = 2"}, "/ws")
    assert sentence.startswith("eval: ")


# ---------------------------------------------------------------------------
# Disclosure-gated argv redaction in subprocess error rendering
# ---------------------------------------------------------------------------
# A cell that spawns a subprocess can put a credential in argv; the stdlib
# re-renders that argv into CalledProcessError/TimeoutExpired tracebacks. The
# worker must re-render an argument ONLY when its exact bytes already appear in
# code the model wrote, replacing everything else with a length-only marker.
# These tests drive the real ``_handle`` with a reset ledger.


@pytest.fixture
def _fresh_argv_ledger():
    """Reset the worker's disclosure ledger so each test controls disclosure."""
    from local_operator.tools import eval_worker

    eval_worker._ARGV_LEDGER._entries.clear()
    eval_worker._ARGV_LEDGER._bytes = 0
    yield
    eval_worker._ARGV_LEDGER._entries.clear()
    eval_worker._ARGV_LEDGER._bytes = 0


def _run_cell(code: str) -> dict[str, Any]:
    from local_operator.tools.eval_worker import _handle

    return _handle({}, {"id": "argv", "code": code})


def test_an_undisclosed_argv_secret_is_redacted_from_the_traceback(
    _fresh_argv_ledger, tmp_path
) -> None:
    """The leak this guard exists for: a credential read from a file (never in
    source) must not appear in the model-visible CalledProcessError."""
    secret = "postgres://u:Sup3rSecretPW@db:5432/prod"
    dsn_file = tmp_path / "dsn.txt"
    dsn_file.write_text(secret)
    payload = _run_cell(
        "import subprocess\n"
        f"dsn = open({str(dsn_file)!r}).read()\n"
        "subprocess.run(['false', 'db', dsn], check=True)\n"
    )
    assert payload["ok"] is False
    assert secret not in payload["error"]
    assert "<redacted:" in payload["error"]
    # argv[0] is preserved verbatim and the exit status is untouched.
    assert "'false'" in payload["error"]
    assert "exit status" in payload["error"]


def test_a_disclosed_command_renders_byte_identical(_fresh_argv_ledger) -> None:
    """An all-disclosed invocation is invisible to the guard."""
    payload = _run_cell(
        "import subprocess\nsubprocess.run(['false', 'git', 'status', '--porcelain'], check=True)\n"
    )
    assert "'false', 'git', 'status', '--porcelain'" in payload["error"]
    assert "<redacted:" not in payload["error"]


def test_a_literal_from_an_earlier_cell_counts_as_disclosed(_fresh_argv_ledger) -> None:
    """A persistent kernel keeps state: a literal in cell 1 may reach argv in
    cell 5. Checking only the failing cell would redact a value plainly visible
    in the transcript."""
    _run_cell('MARKER = "literal-in-source-abc123"')
    payload = _run_cell(
        "import subprocess\n"
        "subprocess.run(\n"
        "    ['false', 'h', 'Authorization: Bearer literal-in-source-abc123'], check=True\n"
        ")\n"
    )
    assert "literal-in-source-abc123" in payload["error"]


def test_an_env_derived_value_is_redacted(_fresh_argv_ledger) -> None:
    """A value pulled from the environment was never in source, so it is
    undisclosed even though the agent chose to pass it."""
    import os

    home = os.environ["HOME"]
    payload = _run_cell(
        "import subprocess, os\nsubprocess.run(['false', 'h', os.environ['HOME']], check=True)\n"
    )
    assert home not in payload["error"]
    assert "<redacted:" in payload["error"]


def test_timeout_expired_also_redacts_argv(_fresh_argv_ledger, tmp_path) -> None:
    """The worst case: a timing-out child has no output, so the argv is the
    entire content of the error."""
    secret = "deadbeef-timeout-secret"
    secret_file = tmp_path / "s.txt"
    secret_file.write_text(secret)
    # ``env`` keeps the secret in argv while ``sleep`` runs long enough to hit
    # the timeout (passing it as sleep's interval would just exit 1 instead).
    payload = _run_cell(
        "import subprocess\n"
        f"token = open({str(secret_file)!r}).read()\n"
        "subprocess.run(['env', 'S=' + token, 'sleep', '5'], timeout=0.2, check=True)\n"
    )
    assert payload["ok"] is False
    assert secret not in payload["error"]
    assert "TimeoutExpired" in payload["error"]


def test_a_caught_error_is_an_explicit_disclosure_and_untouched(
    _fresh_argv_ledger, tmp_path
) -> None:
    """Deliberate disclosure still works: a cell that catches the error and
    prints it made an explicit choice. Only the UNCAUGHT rendering is filtered;
    the live exception object still carries the real argv."""
    secret = "caught-explicit-secret"
    sf = tmp_path / "s.txt"
    sf.write_text(secret)
    payload = _run_cell(
        "import subprocess\n"
        f"token = open({str(sf)!r}).read()\n"
        "try:\n"
        "    subprocess.run(['false', 'h', token], check=True)\n"
        "except subprocess.CalledProcessError as e:\n"
        "    print('CAUGHT:', e.cmd)\n"
    )
    # The cell chose to print e.cmd, so the real argv is in stdout (an explicit
    # disclosure), and the call SUCCEEDS because the error was caught.
    assert payload["ok"] is True
    assert secret in payload["stdout"]


def test_a_context_chain_also_redacts_the_inner_process_error(_fresh_argv_ledger, tmp_path) -> None:
    """format_exception renders the WHOLE __context__/__cause__ chain, so
    redacting only the outermost exception leaks the secret in the inner one.
    A bare ``raise`` inside ``except`` is the common shape."""
    secret = "context-chain-secret"
    sf = tmp_path / "s.txt"
    sf.write_text(secret)
    payload = _run_cell(
        "import subprocess\n"
        f"tok = open({str(sf)!r}).read()\n"
        "try:\n"
        "    subprocess.run(['false', 'h', tok], check=True)\n"
        "except subprocess.CalledProcessError:\n"
        "    raise ValueError('wrapper')\n"
    )
    assert secret not in payload["error"]
    # The wrapper and the inner error are both rendered, but the secret is not.
    assert "ValueError" in payload["error"] and "CalledProcessError" in payload["error"]


def test_an_exception_group_leaf_also_redacts_argv(_fresh_argv_ledger, tmp_path) -> None:
    """format_exception renders ExceptionGroup leaves: a CalledProcessError
    raised inside an asyncio.TaskGroup task is part of the model-visible
    traceback, so the walk must descend into .exceptions, not stop at the
    group."""
    secret = "taskgroup-secret"
    sf = tmp_path / "s.txt"
    sf.write_text(secret)
    payload = _run_cell(
        "import subprocess, asyncio\n"
        f"tok = open({str(sf)!r}).read()\n"
        "async def job():\n"
        "    subprocess.run(['false', 'h', tok], check=True)\n"
        "async def main():\n"
        "    async with asyncio.TaskGroup() as tg:\n"
        "        tg.create_task(job())\n"
        "asyncio.run(main())\n"
    )
    assert "ExceptionGroup" in payload["error"]
    assert secret not in payload["error"]


def test_raise_from_redacts_the_cause_chain(_fresh_argv_ledger, tmp_path) -> None:
    """``raise ... from e`` names the cause explicitly; its argv is redacted."""
    secret = "raise-from-secret"
    sf = tmp_path / "s.txt"
    sf.write_text(secret)
    payload = _run_cell(
        "import subprocess\n"
        f"tok = open({str(sf)!r}).read()\n"
        "try:\n"
        "    subprocess.run(['false', 'h', tok], check=True)\n"
        "except subprocess.CalledProcessError as e:\n"
        "    raise KeyError('k') from e\n"
    )
    assert secret not in payload["error"]


def test_an_unparseable_string_cmd_is_ledger_gated_not_passed_through(
    _fresh_argv_ledger,
) -> None:
    """A string cmd shlex cannot parse (one unbalanced quote) must not be
    rendered verbatim with the secret inside: the whole string is disclosed or
    collapsed to a length marker."""
    import subprocess as sp

    from local_operator.tools import eval_worker

    undisclosed = "curl -H 'Authorization: Bearer unbalanced-secret"
    exc = sp.CalledProcessError(1, undisclosed)
    eval_worker._redact_process_exception(exc)
    assert exc.cmd == "<redacted:%dc>" % len(undisclosed)

    # ...but a string the model literally wrote stays readable.
    eval_worker._ARGV_LEDGER.record(undisclosed)
    exc2 = sp.CalledProcessError(1, undisclosed)
    eval_worker._redact_process_exception(exc2)
    assert exc2.cmd == undisclosed


def test_a_plain_traceback_without_a_command_is_untouched(_fresh_argv_ledger) -> None:
    """The guard must not mangle ordinary errors. A ValueError whose message is
    prose (no process invocation) renders word for word."""
    payload = _run_cell("raise ValueError('plain words with spaces and no command')\n")
    assert "plain words with spaces and no command" in payload["error"]
    assert "<redacted" not in payload["error"]


def test_worker_caps_stdout_stderr_display_and_repr_before_json() -> None:
    """Containment is worker-side: huge output never reaches the parent or
    json.dumps unbounded, even before the eval tool's 8KiB spill layer."""
    from local_operator.tools.eval_worker import (
        DISPLAY_CHAR_LIMIT,
        STREAM_CHAR_LIMIT,
        _handle,
    )

    payload = _handle(
        {},
        {
            "id": "cap",
            "code": (
                "print('x' * 10_000_000)\n"
                "import sys\n"
                "print('e' * 10_000_000, file=sys.stderr)\n"
                "display('d' * 10_000_000)\n"
                "'r' * 10_000_000"
            ),
        },
    )
    assert len(payload["stdout"]) <= STREAM_CHAR_LIMIT + 100
    assert len(payload["stderr"]) <= STREAM_CHAR_LIMIT + 100
    assert sum(len(item) for item in payload["display"]) <= DISPLAY_CHAR_LIMIT + 100
    assert len(payload["result"]) < 10_000
    assert "truncated" in payload["stdout"]
    # reprlib truncates one huge value before it reaches the display-channel
    # aggregate cap; either path is bounded without building a giant payload.
    assert "..." in payload["display"][0]


@pytest.mark.asyncio
async def test_timeout_kills_eval_descendant_process_group(context, tmp_path) -> None:
    marker = tmp_path / "descendant-leaked.txt"
    child = (
        "import time; from pathlib import Path; "
        f"time.sleep(1); Path({str(marker)!r}).write_text('leaked')"
    )
    code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        "time.sleep(60)"
    )
    result = await _call(context, code, timeout=0.2)
    assert result.is_error is True
    assert "TIMEOUT" in result.text
    if eval_tool._CLOSING:
        await asyncio.gather(*list(eval_tool._CLOSING))
    await asyncio.sleep(1.1)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_raw_stdout_protocol_overflow_retires_kernel_and_recovers(
    context, monkeypatch
) -> None:
    # Exercise the actual reader boundary without allocating the production
    # 32 MiB JSON envelope in every test worker. Large valid bridge requests
    # separately run through the default ceiling in test_efficiency.py.
    monkeypatch.setattr(eval_tool, "_PROTOCOL_FRAME_LIMIT", 64 * 1024)
    result = await _call(
        context,
        "import os\nos.write(1, b'x' * 100_000 + b'\\n')\n42",
        timeout=5,
    )
    assert result.is_error is True
    assert "protocol" in result.text.lower()
    assert not eval_tool._KERNELS
    if eval_tool._CLOSING:
        await asyncio.gather(*list(eval_tool._CLOSING))

    fresh = await _call(context, "1 + 1")
    assert fresh.is_error is False
    assert "result: 2" in fresh.text


@pytest.mark.asyncio
async def test_windows_spawn_assigns_kill_on_close_job_before_use(tmp_path, monkeypatch) -> None:
    class Transport:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Process:
        pid = 4242
        returncode = None
        _transport = Transport()

        async def wait(self) -> int:
            return 0

        def kill(self) -> None:
            raise AssertionError("Job Object should own termination")

    process = Process()
    spawn_options: dict[str, Any] = {}
    closed: list[int] = []

    async def spawn(*_args: object, **kwargs: Any) -> Process:
        spawn_options.update(kwargs)
        return process

    monkeypatch.setattr(eval_tool.sys, "platform", "win32")
    monkeypatch.setattr(eval_tool.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(eval_tool, "_create_windows_kill_job", lambda pid: 77)
    monkeypatch.setattr(eval_tool, "_close_windows_job", closed.append)

    kernel = await eval_tool._spawn(str(tmp_path))
    assert kernel.windows_job == 77
    assert "start_new_session" not in spawn_options
    assert "creationflags" in spawn_options

    await eval_tool._close_kernel(kernel)
    assert closed == [77]
    assert kernel.windows_job is None
    assert process._transport.closed is True


# ---------------------------------------------------------------------------
# background mode: long work observed through the job, not by blocking a turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_returns_a_job_id_without_waiting(tmp_path) -> None:
    """The point of the mode: a long run costs the caller no wall clock.

    The code below sleeps far longer than this call is allowed to take, so a
    fast return is only possible if the work really was handed to a job.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg", jobs=manager)
    tool = eval_tool.build_eval_tool()
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await tool.execute(  # type: ignore[operator]
        "call-bg",
        {"code": "import time\ntime.sleep(30)", "background": True, "timeout": 60},
        None,
        None,
        context,
    )
    elapsed = loop.time() - started
    assert result.is_error is False
    assert elapsed < 5.0, f"background eval blocked for {elapsed:.1f}s"
    job_id = str((result.details or {})["job_id"])
    assert manager.get(job_id) is not None
    await manager.cancel(job_id)
    await manager.dispose()


@pytest.mark.asyncio
async def test_background_streams_output_while_it_runs(tmp_path) -> None:
    """Output is observable BEFORE the job settles.

    A run whose output only appeared at the end would leave the caller blind
    for exactly as long as the work is interesting, which is the failure this
    mode exists to fix.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg-stream", jobs=manager)
    tool = eval_tool.build_eval_tool()
    code = "import time\nfor i in range(20):\n    print('tick', i, flush=True)\n    time.sleep(0.3)"
    result = await tool.execute(  # type: ignore[operator]
        "call-bg2", {"code": code, "background": True, "timeout": 60}, None, None, context
    )
    job_id = str((result.details or {})["job_id"])

    # Poll until output appears while the job is still running.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    seen = ""
    while loop.time() < deadline:
        window = manager.read_output(job_id)
        assert window is not None
        seen = window[0]
        live = manager.get(job_id)
        if "tick 0" in seen and live is not None and live.status == "running":
            break
        await asyncio.sleep(0.1)
    assert "tick 0" in seen, "no streamed output observed while the job ran"
    live = manager.get(job_id)
    assert live is not None and live.status == "running"

    await manager.cancel(job_id)
    await manager.dispose()


@pytest.mark.asyncio
async def test_background_uses_its_own_kernel_and_leaves_the_session_alone(tmp_path) -> None:
    """Isolation is the documented trade for not blocking the turn.

    A background run sharing the session namespace would mutate it under every
    foreground call the model makes meanwhile — the exact interleaving the
    tool's ``exclusive`` concurrency exists to prevent.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg-iso", jobs=manager)
    tool = eval_tool.build_eval_tool()
    execute = tool.execute  # type: ignore[operator]
    await execute("s1", {"code": "session_only = 'here'"}, None, None, context)
    bg = await execute(
        "s2",
        {"code": "import time\ntime.sleep(20)", "background": True, "timeout": 60},
        None,
        None,
        context,
    )
    # The session kernel is untouched by the background run.
    kept = await execute("s3", {"code": "session_only"}, None, None, context)
    assert kept.is_error is False
    assert "result: 'here'" in kept.text
    await manager.cancel(str((bg.details or {})["job_id"]))
    await manager.dispose()


@pytest.mark.asyncio
async def test_background_without_a_job_manager_is_refused(context) -> None:
    """Refusing beats silently running in the foreground for 5 minutes."""
    tool = eval_tool.build_eval_tool()
    result = await tool.execute(  # type: ignore[operator]
        "call-nojobs", {"code": "1", "background": True}, None, None, context
    )
    assert result.is_error is True
    assert "job manager" in result.text


@pytest.mark.asyncio
async def test_background_cancelled_before_start_kills_the_kernel(tmp_path) -> None:
    """Cancel with ZERO intervening awaits must still kill the worker.

    The kernel is spawned before ``register``, and ``register`` only schedules
    the runner — so a cancel in the same event-loop turn never enters it and
    never reaches its ``finally``. Without a pre-start teardown the interpreter
    survives, reparented to init, owned by nothing.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg-leak", jobs=manager)
    tool = eval_tool.build_eval_tool()
    result = await tool.execute(  # type: ignore[operator]
        "c1",
        {"code": "import time\ntime.sleep(300)", "background": True, "timeout": 600},
        None,
        None,
        context,
    )
    job_id = str((result.details or {})["job_id"])
    process = eval_tool._BACKGROUND_KERNELS.get(job_id)
    assert process is not None, "the background kernel should be tracked while unstarted"
    assert process.returncode is None, "precondition: the kernel is alive"

    await manager.cancel(job_id)  # no awaits in between
    await asyncio.sleep(1.0)
    assert process.returncode is not None, "kernel survived a cancel before the runner started"
    await manager.dispose()


@pytest.mark.asyncio
async def test_background_timeout_may_exceed_the_foreground_cap(tmp_path) -> None:
    """The mode advertises training runs and polling loops, so its ceiling has
    to allow them; the foreground cap still applies to blocking calls."""
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg-cap", jobs=manager)
    tool = eval_tool.build_eval_tool()
    long_bg = await tool.execute(  # type: ignore[operator]
        "c1", {"code": "1", "background": True, "timeout": 1800}, None, None, context
    )
    assert long_bg.is_error is False

    too_long_fg = await tool.execute(  # type: ignore[operator]
        "c2", {"code": "1", "timeout": 1800}, None, None, context
    )
    assert too_long_fg.is_error is True
    assert "background=true" in too_long_fg.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_background_job_stays_deliverable_inside_a_subagent(tmp_path) -> None:
    """A background job must be registered UNOWNED, even in a child session.

    Setting ``owner_id`` looks like scoping and is not: the manager routes an
    owned completion exclusively through that owner's registered delivery sink,
    nothing in this codebase registers one, so the completion is dead-lettered
    and the caller is never told its job finished. That silently defeats the
    entire mode, so the ownership is pinned here rather than left to look like
    an oversight.
    """
    from local_operator.harness.jobs import AsyncJobManager

    delivered: list[str] = []

    async def fallback(job_id: str, text: str, job: object) -> None:
        delivered.append(job_id)

    manager = AsyncJobManager(on_job_complete=fallback)
    context = ToolContext(
        cwd=str(tmp_path), session_id="bg-own", jobs=manager, job_id="child-job-1"
    )
    tool = eval_tool.build_eval_tool()
    result = await tool.execute(  # type: ignore[operator]
        "c1", {"code": "'done'", "background": True, "timeout": 60}, None, None, context
    )
    job_id = str((result.details or {})["job_id"])
    row = manager.get(job_id)
    assert row is not None and row.owner_id is None

    for _ in range(100):
        await asyncio.sleep(0.05)
        settled = manager.get(job_id)
        if settled is not None and settled.status != "running":
            break
    await asyncio.sleep(0.2)
    assert delivered == [job_id], "the child was never told its background job finished"
    await manager.dispose()
