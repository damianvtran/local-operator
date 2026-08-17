"""End-to-end tests for the ``eval`` persistent-kernel tool.

Each acceptance property of the tool is exercised against the REAL worker
subprocess (spawned exactly as the tool spawns it): persistence across
calls, trailing-expression results, output budget/spill, display() routing,
clean syntax errors, honest state-loss on timeout/crash with fresh restart,
and per-session kernel isolation.
"""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_idle_kernel_is_reaped_on_access(context) -> None:
    await _call(context, "a = 1")
    key = next(iter(eval_tool._KERNELS))
    stale_pid = eval_tool._KERNELS[key].process.pid
    # Simulate the 5-minute idle window without waiting for it.
    eval_tool._KERNELS[key].last_used -= eval_tool.KERNEL_IDLE_SECONDS + 1
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
    assert tool.parameters["properties"]["timeout"]["maximum"] == eval_tool.EVAL_MAX_TIMEOUT_SECONDS


def test_describe_approval_shows_code_first_line() -> None:
    sentence = eval_tool._describe_eval_approval({"code": "x = 1\ny = 2"}, "/ws")
    assert sentence.startswith("eval: ")
