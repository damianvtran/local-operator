"""Tests for the background engine tools: task / wait / jobs.

These are the createIf-gated siblings of ``wake``: they only exist when the
ToolContext carries a ``subagent_launcher`` (task) or a job manager
(wait/jobs). The tests drive the real unit wraps against a real
``harness.jobs.AsyncJobManager``, not a mock, so the tool results reflect how
jobs actually settle.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.jobs import OUTPUT_TAIL_CHARS, AsyncJobManager
from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools


async def wait_for(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.005)


def _status(manager: AsyncJobManager, job_id: str) -> str | None:
    """A job's status, narrowed for predicates that poll for a transition."""
    job = manager.get(job_id)
    return job.status if job is not None else None


async def _quick_runner(job_id: str, signal: Any, report_progress) -> str:
    report_progress("halfway")
    return f"done:{job_id}"


async def _slow_runner(job_id: str, signal: Any, report_progress) -> str:
    await asyncio.sleep(30)  # outlive any test window
    return "never"


def _engine_context(tmp_path, manager: AsyncJobManager) -> ToolContext:
    """A context carrying the job manager AND a launcher so both task and
    wait/jobs build. The launcher mimics ``Session._launch_subagent`` by
    registering a ``task``-type job on the same manager."""

    def launcher(label, prompt, *, agent="task", effort=None):
        return manager.register("task", label, _quick_runner, owner_id=None)

    return ToolContext(cwd=str(tmp_path), session_id="s", subagent_launcher=launcher, jobs=manager)


def _tools(context: ToolContext) -> dict[str, AgentTool]:
    return {t.name: t for t in create_tools(context)}


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    tool = tools[name]
    return await tool.execute("call-1", args, None, None, context)  # type: ignore[operator]


@pytest.mark.asyncio
async def test_task_registers_a_job_and_returns_its_id(tmp_path):
    manager = AsyncJobManager()
    context = _engine_context(tmp_path, manager)
    tools = _tools(context)

    before = [j.id for j in manager.list()]
    result = await _call(tools, "task", {"label": "summarize", "prompt": "do it"}, context)

    assert result.is_error is False
    assert result.details is not None
    job_id = result.details["job_id"]
    assert job_id not in before  # a NEW job was registered
    assert f"job {job_id}" in result.text

    job = manager.get(job_id)
    assert job is not None
    assert job.type == "task"
    assert job.label == "summarize"

    def _completed():
        job = manager.get(job_id)
        return job is not None and job.status == "completed"

    # The quick runner completes on its own; wait for settlement.
    await wait_for(_completed)
    job = manager.get(job_id)
    assert job is not None
    assert job.result_text == f"done:{job_id}"
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_returns_final_output_when_job_completes(tmp_path):
    manager = AsyncJobManager()
    context = _engine_context(tmp_path, manager)
    tools = _tools(context)

    result = await _call(tools, "task", {"label": "work", "prompt": "run"}, context)
    assert result.is_error is False
    assert result.details is not None
    job_id = result.details["job_id"]

    waited = await _call(tools, "wait", {"job_id": job_id, "wait_ms": 1000}, context)
    assert waited.is_error is False
    assert waited.details is not None
    assert waited.details["status"] == "completed"
    assert f"done:{job_id}" in waited.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_spills_large_subagent_handoff_without_losing_it(tmp_path, monkeypatch):
    """One verbose child must not consume an unbounded slice of parent context."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    report = "\n".join(f"review finding {index}: {'x' * 120}" for index in range(400))

    async def verbose_runner(job_id: str, signal: Any, report_progress) -> str:
        return report

    manager = AsyncJobManager()
    job_id = manager.register("task", "verbose review", verbose_runner)
    context = ToolContext(cwd=str(tmp_path), session_id="parent", jobs=manager)
    tools = _tools(context)

    waited = await _call(tools, "wait", {"job_id": job_id, "wait_ms": 1000}, context)

    assert waited.is_error is False
    assert waited.details is not None
    spill = waited.details["spill"]
    assert spill["complete"] is True
    assert spill["handle"] in waited.text
    assert len(waited.text) < builtin.TOOL_OUTPUT_LIMIT_CHARS + 1_000
    stored = builtin.get_store().read_lines(spill["handle"])
    assert stored is not None
    lines, _total = stored
    recovered = "\n".join(lines)
    assert "review finding 0:" in recovered
    assert "review finding 399:" in recovered
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_bounded_by_wait_ms_times_out(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager, subagent_launcher=None)
    # Register a job that never settles so wait must time out.
    slow_id = manager.register("task", "slow", _slow_runner)
    tools = _tools(context)

    started = asyncio.get_running_loop().time()
    result = await _call(tools, "wait", {"job_id": slow_id, "wait_ms": 60}, context)

    elapsed = asyncio.get_running_loop().time() - started
    assert elapsed < 1.0  # returned promptly on the bound, did not hang
    assert result.is_error is False
    assert result.details is not None
    assert result.details["status"] == "running"
    assert "still running" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_lists_ids_labels_and_statuses(tmp_path):
    manager = AsyncJobManager()
    quick_id = manager.register("task", "alpha", _quick_runner)
    running_id = manager.register("task", "beta", _slow_runner)
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)

    result = await _call(tools, "jobs", {}, context)
    assert result.is_error is False
    assert result.details is not None
    assert result.details["count"] == 2
    assert "alpha" in result.text
    assert "beta" in result.text
    assert running_id in result.text
    assert "running" in result.text

    def _settled():
        job = manager.get(quick_id)
        return job is not None and job.status != "running"

    # Let the quick job settle before dispose so its runner coroutine is
    # awaited rather than abandoned (the slow one is cancelled by dispose).
    await wait_for(_settled)
    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_reports_how_long_a_running_job_has_been_going(tmp_path):
    """The elapsed column read ``now - now`` and printed 0.0s for every live
    job, whatever its real age.

    That is the one number this tool exists to report: a caller polling
    ``jobs`` uses it to tell a subagent that is progressing from one that is
    wedged, and a six-minute child was indistinguishable from one launched a
    second ago. Asserted against a job whose ``start_time`` is backdated,
    because a fresh job is genuinely ~0s and cannot tell the two apart.
    """
    manager = AsyncJobManager()
    running_id = manager.register("task", "beta", _slow_runner)
    job = manager.get(running_id)
    assert job is not None
    job.start_time -= 373.0  # a child that has been running 6m13s

    # Let the runner actually start before dispose cancels it; registration
    # only SCHEDULES it (``ensure_future``), so without a yield its coroutine
    # is never awaited and the test leaves a RuntimeWarning behind.
    await asyncio.sleep(0)

    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    result = await _call(tools, "jobs", {}, context)

    row = next(line for line in result.text.splitlines() if running_id in line)
    seconds = float(row.split()[2].rstrip("s"))
    assert seconds >= 373.0, f"a long-running job reported {seconds}s: {row!r}"
    # The number is useless without which quantity it is: this column also
    # carries "settled N seconds ago" for a settled row, in the same shape.
    assert row.split()[3] == "up", f"a running job's age is not marked as uptime: {row!r}"

    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_says_which_quantity_each_age_is(tmp_path):
    """One column, two facts: a running row's age is uptime, a settled row's
    is time since it settled.

    Adjacent rows were identical in presentation — ``running  373.0s`` and
    ``failed  287.5s`` — while the first counts up from launch and the second
    counts up from the end. A reader comparing a settled row against the
    subagent panel (which reports the job's own DURATION) got two numbers with
    no way to reconcile them.
    """
    manager = AsyncJobManager()
    running_id = manager.register("task", "alpha", _slow_runner)
    settled_id = manager.register("task", "beta", _quick_runner)

    def _settled() -> bool:
        job = manager.get(settled_id)
        return job is not None and job.status != "running"

    await wait_for(_settled)
    settled = manager.get(settled_id)
    assert settled is not None and settled.settled_at is not None
    settled.settled_at -= 287.5  # finished nearly five minutes ago

    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    result = await _call(_tools(context), "jobs", {}, context)
    rows = {line.split()[0]: line for line in result.text.splitlines()}

    assert rows[running_id].split()[3] == "up"
    assert rows[settled_id].split()[3] == "ago"
    assert float(rows[settled_id].split()[2].rstrip("s")) >= 287.5

    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_never_says_up_beside_a_settled_status(tmp_path):
    """The sense follows the STATUS, not the clock.

    Sharing one test let a settled row with no ``settled_at`` print ``up``
    beside a ``completed`` or ``cancelled`` — a contradiction no reader can
    reconcile. It is reachable through the real manager, in the window inside
    ``cancel()``'s await where the status is set and the settle stamp is not
    yet, so this drives exactly that: a cancelled row with the stamp withheld.
    """
    manager = AsyncJobManager()
    job_id = manager.register("task", "probe", _slow_runner)
    # Let the runner begin, then cancel; the row is settled the moment cancel
    # returns, and the settle stamp arrives from the runner's own teardown.
    await asyncio.sleep(0)
    await manager.cancel(job_id)
    job = manager.get(job_id)
    assert job is not None and job.status == "cancelled"
    job.settled_at = None  # the window, held open

    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    result = await _call(_tools(context), "jobs", {}, context)
    row = next(line for line in result.text.splitlines() if line.startswith(job_id))

    assert "cancelled" in row
    assert row.split()[3] != "up", row

    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_says_wait_for_a_job_that_has_not_been_admitted(tmp_path):
    """A parked job is ``running`` with ``queued=True`` and a runner that has
    never been entered, so ``up`` presented its wait as uptime — the same
    misreport this PR was filed to stop, on the third surface."""
    manager = AsyncJobManager(max_running=1)
    gate = asyncio.Event()

    async def blocked(job_id, signal, report_progress):
        await gate.wait()
        return "ok"

    running_id = manager.register("task", "alpha", blocked)
    parked_id = manager.register("task", "parked", blocked, queued=True)
    await asyncio.sleep(0)
    parked = manager.get(parked_id)
    assert parked is not None and parked.queued and parked.started_at is None
    parked.start_time -= 215.0

    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    result = await _call(_tools(context), "jobs", {}, context)
    rows = {line.split()[0]: line for line in result.text.splitlines()}
    row = rows[parked_id]

    assert "215.0s" in row or float(row.split()[2].rstrip("s")) >= 215.0, row
    assert row.split()[3] == "wait", row
    # ``wait`` is four cells where every other sense is at most three. The
    # field must budget for the widest vocabulary entry or the one parked row
    # shears its label one cell right of the entire table.
    assert row.index("parked") == rows[running_id].index("alpha"), rows

    gate.set()
    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_keeps_its_columns_aligned_at_any_age(tmp_path):
    """A day-old running job must not shear the grid.

    ``6.1f`` fits through ``9999.9s`` and overflows at 2h46m40s, pushing that
    row's label one cell right of every other row's and its decimal point out
    of the column. An overnight ``bash`` watcher reaches that in an ordinary
    session — retention only sweeps SETTLED rows — and before the age column
    carried a real number for running jobs it was unreachable.
    """
    manager = AsyncJobManager()
    ages = (0.0, 12.3, 373.0, 3600.0, 86400.0)
    ids = [manager.register("task", f"job {age}", _slow_runner) for age in ages]
    await asyncio.sleep(0)
    for job_id, age in zip(ids, ages):
        job = manager.get(job_id)
        assert job is not None
        job.start_time -= age

    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    result = await _call(_tools(context), "jobs", {}, context)
    lines = result.text.splitlines()

    label_columns = {line.index("job ") for line in lines}
    assert len(label_columns) == 1, f"the label column sheared: {lines!r}"
    decimals = {line.index(".", 12) for line in lines}
    assert len(decimals) == 1, f"the decimal point sheared: {lines!r}"

    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_says_unknown_rather_than_zero_for_a_row_with_no_clock(tmp_path):
    """``JobManagerProtocol.list()`` is typed ``list[Any]``, so an embedder may
    hand this tool a row with no ``start_time``.

    Rendering that as ``0.0s`` is byte-identical to a job launched this
    instant — the exact unreadable reading this column was fixed to stop
    printing, and worse for being a number a caller will act on.
    """

    class NoClock:
        id = "aaaaaaaaaaaa"
        type = "task"
        status = "running"
        settled_at = None
        label = "embedder row"

    class OneRow(AsyncJobManager):
        def list(self, *, owner_id: str | None = None) -> list[Any]:
            return [NoClock()]

    manager = OneRow()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    result = await _call(_tools(context), "jobs", {}, context)

    row = result.text.splitlines()[0]
    assert result.is_error is False
    assert "unknown" in row, f"a row with no clock reported a number: {row!r}"
    assert "0.0s" not in row
    await manager.dispose()


@pytest.mark.asyncio
async def test_jobs_empty_manager(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    result = await _call(tools, "jobs", {}, context)
    assert result.is_error is False
    assert result.details is not None
    assert result.details["count"] == 0
    assert "no background jobs" in result.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_wait_unknown_job_is_an_error(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    result = await _call(tools, "wait", {"job_id": "nope", "wait_ms": 10}, context)
    assert result.is_error is True
    assert "unknown job nope" in result.text
    await manager.dispose()


def test_task_not_advertised_without_launcher(tmp_path):
    """createIf: no subagent_launcher -> no task tool, and task execute errors."""
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=AsyncJobManager())
    names = {t.name for t in create_tools(context)}
    assert "task" not in names
    assert builtin.build_task_tool(context) is None


def test_wait_jobs_not_advertised_without_job_manager(tmp_path):
    context = ToolContext(cwd=str(tmp_path), session_id="s")
    names = {t.name for t in create_tools(context)}
    assert "wait" not in names
    assert "jobs" not in names
    assert builtin.build_wait_tool(context) is None
    assert builtin.build_jobs_tool(context) is None


# ---------------------------------------------------------------------------
# task: batch form, shared context, agent/effort tiers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_task_batch_launches_concurrent_children_with_shared_context(tmp_path):
    """One call, three jobs: the batch form is the fan-out economics — N
    independent slices cost N round trips when launched one per call and one
    when launched together. The shared context is prepended to every prompt
    verbatim so the delegator's contract cannot drift between children."""
    launched: list[tuple[str, str, str, str | None]] = []

    def launcher(label, prompt, *, agent="task", effort=None):
        launched.append((label, prompt, agent, effort))
        return f"job-{len(launched)}"

    context = ToolContext(
        cwd=str(tmp_path), session_id="s", subagent_launcher=launcher, jobs=AsyncJobManager()
    )
    tools = _tools(context)
    result = await _call(
        tools,
        "task",
        {
            "context": "Goal: ship the release. Constraint: no breaking API changes.",
            "tasks": [
                {"label": "Docs", "prompt": "Update the changelog."},
                {"label": "Scan", "prompt": "Audit imports.", "agent": "scout"},
                {"label": "Build", "prompt": "Run the release build.", "effort": "hi"},
            ],
        },
        context,
    )
    assert result.is_error is False
    assert len(launched) == 3
    labels = [entry[0] for entry in launched]
    assert labels == ["Docs", "Scan", "Build"]
    for _label, prompt, _agent, _effort in launched:
        assert prompt.startswith("Goal: ship the release.")
        assert "Your task (" in prompt
    assert launched[1][2] == "scout"
    assert launched[2][3] == "hi"
    assert "3 subagent(s)" in result.text
    assert result.details is not None
    assert result.details["jobs"][0]["job_id"] == "job-1"


@pytest.mark.asyncio
async def test_task_single_form_still_works_and_forwards_defaults(tmp_path):
    launched: list[tuple[str, str, str, str | None]] = []

    def launcher(label, prompt, *, agent="task", effort=None):
        launched.append((label, prompt, agent, effort))
        return "job-1"

    context = ToolContext(
        cwd=str(tmp_path), session_id="s", subagent_launcher=launcher, jobs=AsyncJobManager()
    )
    tools = _tools(context)
    result = await _call(tools, "task", {"label": "Solo", "prompt": "one thing"}, context)
    assert result.is_error is False
    assert launched == [("Solo", "one thing", "task", None)]


@pytest.mark.asyncio
async def test_task_rejects_mixed_and_half_forms(tmp_path):
    context = _engine_context(tmp_path, AsyncJobManager())
    tools = _tools(context)
    both = await _call(
        tools,
        "task",
        {"label": "x", "prompt": "y", "tasks": [{"label": "z", "prompt": "w"}]},
        context,
    )
    assert both.is_error is True
    assert both.text.startswith("invalid arguments:")
    half = await _call(tools, "task", {"label": "x"}, context)
    assert half.is_error is True
    empty = await _call(tools, "task", {}, context)
    assert empty.is_error is True


@pytest.mark.asyncio
async def test_task_partial_batch_failure_reports_survivors(tmp_path):
    calls = {"n": 0}

    def launcher(label, prompt, *, agent="task", effort=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("engine hiccup")
        return f"job-{calls['n']}"

    context = ToolContext(
        cwd=str(tmp_path), session_id="s", subagent_launcher=launcher, jobs=AsyncJobManager()
    )
    tools = _tools(context)
    result = await _call(
        tools,
        "task",
        {"tasks": [{"label": "a", "prompt": "p"}, {"label": "b", "prompt": "p"}]},
        context,
    )
    assert result.is_error is False  # 1 of 2 launched: not a total failure
    assert "1 subagent(s)" in result.text
    assert "failed to launch: b" in result.text


# ---------------------------------------------------------------------------
# jobs(op="peek") / jobs(op="cancel") — observing and stopping live work
# ---------------------------------------------------------------------------


async def _emitting_runner(job_id: str, signal: Any, report_progress) -> str:
    """A job that prints, then blocks — the shape peek exists to observe."""
    await asyncio.sleep(30)
    return "never"


@pytest.mark.asyncio
async def test_peek_returns_only_new_output_and_advances_the_cursor(tmp_path):
    """Polling stays cheap: each peek costs what the job newly produced.

    The second peek is the assertion that matters — a peek that re-sent the
    whole tail would make watching a long job cost the same bytes repeatedly.
    """
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    job_id = manager.register("bash", "watched", _emitting_runner)

    manager.append_output(job_id, "line one\n")
    first = await _call(tools, "jobs", {"op": "peek", "job_id": job_id}, context)
    assert first.is_error is False
    assert "line one" in first.text
    cursor = int((first.details or {})["seq"])

    # Nothing new: the reply carries no output at all, only status.
    quiet = await _call(tools, "jobs", {"op": "peek", "job_id": job_id, "since": cursor}, context)
    assert (quiet.details or {})["new_chars"] == 0
    assert "line one" not in quiet.text, "a peek must not re-send what was already read"
    assert "no new output" in quiet.text

    manager.append_output(job_id, "line two\n")
    second = await _call(tools, "jobs", {"op": "peek", "job_id": job_id, "since": cursor}, context)
    assert "line two" in second.text
    assert "line one" not in second.text
    assert int((second.details or {})["seq"]) > cursor

    await manager.cancel(job_id)
    await manager.dispose()


@pytest.mark.asyncio
async def test_peek_reports_a_settled_job_and_flags_dropped_output(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    job_id = manager.register("bash", "done-soon", _quick_runner)
    await wait_for(lambda: _status(manager, job_id) != "running")

    settled = await _call(tools, "jobs", {"op": "peek", "job_id": job_id}, context)
    # Peek does not duplicate the result body; it says where to get it.
    assert "finished" in settled.text
    assert "wait" in settled.text

    # A cursor whose bytes were evicted is warned, never quietly patched over.
    gap_job = manager.register("bash", "chatty", _emitting_runner)
    manager.append_output(gap_job, "X")
    gap_row = manager.get(gap_job)
    assert gap_row is not None
    stale = gap_row.output_seq
    manager.append_output(gap_job, "Y" * (OUTPUT_TAIL_CHARS + 50))
    gapped = await _call(tools, "jobs", {"op": "peek", "job_id": gap_job, "since": stale}, context)
    assert (gapped.details or {})["gap"] is True
    assert "not contiguous" in gapped.text
    await manager.cancel(gap_job)
    await manager.dispose()


@pytest.mark.asyncio
async def test_cancel_stops_a_running_job_and_is_honest_about_settled_ones(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    job_id = manager.register("bash", "long", _slow_runner)

    cancelled = await _call(tools, "jobs", {"op": "cancel", "job_id": job_id}, context)
    assert cancelled.is_error is False
    assert (cancelled.details or {})["cancelled"] is True
    await wait_for(lambda: _status(manager, job_id) == "cancelled")

    # Cancelling again is not an error: the caller's intent is already true.
    again = await _call(tools, "jobs", {"op": "cancel", "job_id": job_id}, context)
    assert again.is_error is False
    assert (again.details or {})["cancelled"] is False
    assert "cancelled" in again.text
    await manager.dispose()


@pytest.mark.asyncio
async def test_peek_and_cancel_reject_missing_or_unknown_ids(tmp_path):
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="s", jobs=manager)
    tools = _tools(context)
    missing = await _call(tools, "jobs", {"op": "peek"}, context)
    assert missing.is_error is True
    assert "requires job_id" in missing.text
    unknown = await _call(tools, "jobs", {"op": "cancel", "job_id": "nope"}, context)
    assert unknown.is_error is True
    assert "unknown job" in unknown.text
    # The default op is still a plain listing, so existing callers are unaffected.
    listed = await _call(tools, "jobs", {}, context)
    assert listed.is_error is False
    await manager.dispose()


@pytest.mark.asyncio
async def test_peek_and_cancel_reach_jobs_the_listing_shows(tmp_path):
    """Whatever `op="list"` shows, `peek` and `cancel` must be able to address.

    Regression: scoping these two ops by `context.job_id` (and leaving `list`
    unscoped) made the tool contradict itself inside a child session — it
    listed a grandchild `task` job and then called that same id "unknown job",
    because `run_subagent` registers those with `owner_id=None`.
    """
    manager = AsyncJobManager()
    grandchild = manager.register("task", "grandchild", _slow_runner)
    # A CHILD session's context: it carries a job_id of its own.
    context = ToolContext(cwd=str(tmp_path), session_id="child", jobs=manager, job_id="child-job-1")
    tools = _tools(context)

    listed = await _call(tools, "jobs", {}, context)
    assert grandchild in listed.text, "precondition: the listing shows the job"

    peeked = await _call(tools, "jobs", {"op": "peek", "job_id": grandchild}, context)
    assert peeked.is_error is False, "peek could not address a job the listing showed"

    cancelled = await _call(tools, "jobs", {"op": "cancel", "job_id": grandchild}, context)
    assert cancelled.is_error is False, "cancel could not address a job the listing showed"
    await manager.dispose()
