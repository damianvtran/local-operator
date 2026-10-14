"""The restart / persistence contract (acceptance bar; design §10.7, §7.2).

The operator's bar, verbatim: monitors "stably come back up" across session
and app restarts "and [are] able to report on diffs after the restart vs the
last state" — the baseline must never be reset or replaced by an artifact.

Pinned here, beyond the consolidated-delta rig in ``test_integration``:

- the check's cwd is captured ONCE, absolutely, and a restart keeps running
  from it even when the reopening session stands elsewhere (``spec.cwd`` is
  "captured at arm; the check runs with it" — and a relative capture would
  silently re-resolve against whichever process hosts the next check);
- a check cancelled at teardown can never write the baseline: the class
  from the field (m28: a disposed check's steering-detach NOTICE — "job …:
  steering interrupted …" — was adopted as the snapshot and the post-restart
  delta ran against it);
- a check that outlives its deadline is a FAILURE, not a notice;
- an unusable captured cwd fails loudly through the ladder instead of
  silently watching nothing forever.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    AgentTool,
    ModelSpec,
    StreamEndEvent,
    TextContent,
    ToolResult,
)
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.settings import MonitorSettings
from local_operator.monitors.spec import MonitorSpec
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_bash_tool
from tests.unit.monitors.support import DRAIN_BACKSTOP_S, drain_checks

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


def _stream(request: Any, signal: Any):  # noqa: ANN202
    async def gen():
        yield StreamEndEvent(stop_reason="stop")

    return gen()


def _bash_session(transcript_dir: Path, *, cwd: str | None = None) -> Session:
    """A real Session owning the real bash tool, at ``cwd`` or the test root."""
    return Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[build_bash_tool()],
        transcript=Transcript(transcript_dir),
        system_blocks_provider=lambda: [],
        cwd=cwd or str(transcript_dir.parent),
    )


def _clean_child_env(**extra: str) -> dict[str, str]:
    """A child env with the inherited loop/monitor prefixes stripped.

    ``CMUX_*`` and ``LOP_*`` are read by the child product (see AGENTS.md
    "Isolating a run"), so a rig must not leak its session's values into the
    child it spawns.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith(("CMUX_", "LOP_"))}
    env.update(extra)
    return env


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


# ---------------------------------------------------------------------------
# The restart rigs (child process dies; the reopened session must carry on)
# ---------------------------------------------------------------------------

CHILD_RESTART_SCRIPT = """
import asyncio, os, sys

from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import build_bash_tool

SESS = os.environ["SESS_DIR"]
CWD = os.environ["ARM_CWD"]
COMMAND = os.environ["ARM_COMMAND"]


async def stream(request, signal):
    yield StreamEndEvent(stop_reason="stop")


async def main():
    session = Session(
        model=ModelSpec(provider="t", model_id="m", context_window=1000),
        stream_fn=stream,
        tools=[build_bash_tool()],
        transcript=Transcript(SESS),
        system_blocks_provider=lambda: [],
        cwd=CWD,
    )
    outcome = await session._monitors.create(
        {"name": "watch", "tool": "bash", "arguments": {"command": COMMAND}, "every": "60s"},
        cwd=os.environ.get("ARM_CAPTURE_CWD", CWD),
    )
    assert outcome.get("created"), outcome
    if os.environ.get("WAIT_FOR_BASELINE") == "1":
        # Poll until the first check has landed the baseline on disk, then
        # die. Deterministic where a fixed sleep is not: under fleet load the
        # 1-3 s arm delay plus a cold bash spawn can exceed any constant.
        import time

        from local_operator.monitors import state as monitor_state
        from local_operator.paths import config_dir

        deadline = time.time() + 30
        while time.time() < deadline:
            counters = monitor_state.read_counters(config_dir(), "sess", "m1")
            if counters and counters.get("content_hash"):
                break
            await asyncio.sleep(0.2)
        else:
            os._exit(8)  # diagnosable: the baseline never landed
    os._exit(9)


asyncio.run(main())
"""


def _run_child(script: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=_clean_child_env(**extra_env),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 9, completed.stderr[-2000:]
    return completed


@pytest.mark.asyncio
async def test_a_restart_diffs_against_the_pre_restart_baseline(
    tmp_path: Path, config_dir: Path
) -> None:
    """The strict reading of "diffs after the restart vs the last state".

    The pre-restart baseline is ``A\\nB`` (plus bash's exit-code line); the
    post-restart check sees ``A\\nC``. The consolidated delta must name ``B``
    as a REMOVAL — a baseline that had been reset would deliver nothing (no
    diff) or a full dump.
    """
    watch = tmp_path / "watched.txt"
    watch.write_text("A\nB\n", encoding="utf-8")
    session_dir = tmp_path / "sess"
    command = f"cat '{watch}'"
    _run_child(
        CHILD_RESTART_SCRIPT,
        SESS_DIR=str(session_dir),
        ARM_CWD=str(tmp_path),
        ARM_COMMAND=command,
        LOCAL_OPERATOR_CONFIG_DIR=str(config_dir),
        WAIT_FOR_BASELINE="1",
    )

    # The crash left the durable baseline behind (transcript row + blob).
    entry = monitor_store.read_entry(config_dir, "sess")
    assert entry is not None and entry["monitors"][0]["id"] == "m1"
    counters_path = config_dir / "monitors" / "state" / "sess" / "m1.json"
    counters = json.loads(counters_path.read_text(encoding="utf-8"))
    assert counters["checks"] >= 1, "the first check must have landed before the kill"
    assert counters["content_hash"].startswith("sha256:")

    reopened = _bash_session(session_dir, cwd=str(tmp_path))
    clock = [int(counters["last_check_at"]) + 4 * 60_000]
    reopened._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    events: list[Any] = []

    async def record(event: Any) -> None:
        events.append(event)

    reopened._emit = record  # type: ignore[method-assign]
    watch.write_text("A\nC\n", encoding="utf-8")
    try:
        await reopened._monitors.pump(now_ms=clock[0] + 30_000)
        await drain_checks(reopened._monitors, reopened)
        deltas = [event for event in events if getattr(event, "type", "") == "monitor_delta"]
        assert len(deltas) == 1, [getattr(event, "type", "") for event in events]
        text = deltas[0].text
        # The REMOVED baseline line is the pin: the diff ran against what the
        # pre-restart state actually held.
        assert "- B" in text, text
        assert "+ C" in text, text
        match = re.search(r"(\d+) skipped while the session was down", text)
        assert match is not None, text
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_a_restart_keeps_running_checks_from_the_captured_cwd(
    tmp_path: Path, config_dir: Path
) -> None:
    """The captured cwd survives the restart even when the session's own cwd moved.

    A marker file exists ONLY in the arm directory. The reopened session is
    constructed with a DIFFERENT cwd; the check must still find the marker —
    ``spec.cwd`` is the persisted contract, not "whatever directory hosts the
    next runtime".
    """
    arm_dir = tmp_path / "arm-dir"
    arm_dir.mkdir()
    marker = "cwd-marker-9f2a81"
    (arm_dir / "where.txt").write_text(marker + "\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    session_dir = tmp_path / "sess"
    _run_child(
        CHILD_RESTART_SCRIPT,
        SESS_DIR=str(session_dir),
        ARM_CWD=str(arm_dir),
        ARM_COMMAND="cat where.txt",
        LOCAL_OPERATOR_CONFIG_DIR=str(config_dir),
    )

    reopened = _bash_session(session_dir, cwd=str(elsewhere))
    assert reopened._monitors.monitors[0].cwd == str(arm_dir)
    clock = [int(time.time() * 1000) + 10_000]
    reopened._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        await reopened._monitors.pump(now_ms=clock[0])
        await drain_checks(reopened._monitors)
        blob = monitor_state.read_snapshot(config_dir, "sess", "m1")
        assert blob is not None and marker in blob["snapshot"], blob
    finally:
        await reopened.dispose()


@pytest.mark.asyncio
async def test_a_relative_captured_cwd_is_resolved_once_at_arm(
    tmp_path: Path, config_dir: Path
) -> None:
    """A relative capture must be made absolute AT ARM, or restarts move it.

    The child stands in ``arm_dir`` and arms with cwd ``"."`` — the shape a
    session whose own cwd is relative would capture. The stored spec must
    hold the absolute directory the arm meant; the reopened check (a
    different process, a different cwd) must run there.
    """
    arm_dir = tmp_path / "child-arm"
    arm_dir.mkdir()
    marker = "relative-marker-4c1d77"
    (arm_dir / "where.txt").write_text(marker + "\n", encoding="utf-8")
    session_dir = tmp_path / "sess"
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD_RESTART_SCRIPT.replace(
                "    session = Session(",
                "    os.chdir(CWD)\n    session = Session(",
            ).replace('cwd=os.environ.get("ARM_CAPTURE_CWD", CWD)', 'cwd="."'),
        ],
        env=_clean_child_env(
            SESS_DIR=str(session_dir),
            ARM_CWD=str(arm_dir),
            ARM_COMMAND="cat where.txt",
            LOCAL_OPERATOR_CONFIG_DIR=str(config_dir),
        ),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert child.returncode == 9, child.stderr[-2000:]

    reopened = _bash_session(session_dir, cwd=str(tmp_path))
    stored = reopened._monitors.monitors[0].cwd
    assert stored.startswith("/"), f"captured cwd is not absolute: {stored!r}"
    assert os.path.realpath(stored) == os.path.realpath(str(arm_dir)), stored
    clock = [int(time.time() * 1000) + 10_000]
    reopened._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
    try:
        await reopened._monitors.pump(now_ms=clock[0])
        await drain_checks(reopened._monitors)
        blob = monitor_state.read_snapshot(config_dir, "sess", "m1")
        assert blob is not None and marker in blob["snapshot"], blob
    finally:
        await reopened.dispose()


# ---------------------------------------------------------------------------
# A cancellation must never write the baseline
# ---------------------------------------------------------------------------


def _blocking_fifo(tmp_path: Path) -> tuple[Path, int]:
    """A fifo plus a held writer: ``cat`` on it blocks until the fd closes."""
    fifo = tmp_path / "block.fifo"
    os.mkfifo(fifo)
    writer = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    return fifo, writer


@pytest.mark.asyncio
async def test_a_check_cancelled_at_teardown_never_writes_the_baseline(
    tmp_path: Path, config_dir: Path
) -> None:
    """The m28 class: a disposed check must not adopt its cancel as output.

    Pre-fix, the disposed check's bash call swallowed the cancellation into
    the steering-detach branch and RETURNED a "job … steering interrupted"
    notice, which ``_apply_success`` then wrote as the snapshot and counters
    — the baseline the post-restart delta ran against. Post-fix the call dies
    and the scheduler refuses to apply anything after dispose.
    """
    fifo, writer = _blocking_fifo(tmp_path)
    session = _bash_session(tmp_path / "sess", cwd=str(tmp_path))
    try:
        clock = [int(time.time() * 1000)]
        session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
        outcome = await session._monitors.create(
            {"name": "blocked", "tool": "bash", "arguments": {"command": f"cat {fifo}"}},
            cwd=str(tmp_path),
        )
        assert outcome.get("created"), outcome
        clock[0] += 10_000
        await session._monitors.pump(now_ms=clock[0])
        await asyncio.sleep(0.5)  # the check is inside the blocked ``cat``
        task = next(iter(session._monitors._check_tasks), None)
        assert task is not None and not task.done(), "the check should be parked on the fifo"

        await session.dispose()
        await asyncio.wait([task], timeout=DRAIN_BACKSTOP_S)
        assert task.done() and task.cancelled(), "a teardown cancel must terminate the check"

        assert monitor_state.read_snapshot(config_dir, "sess", "m1") is None
        counters = monitor_state.read_counters(config_dir, "sess", "m1")
        assert counters is None or not counters.get("checks")
        assert [job for job in session.jobs.list() if job.type == "bash"] == []
    finally:
        os.close(writer)
        if not session._disposed:
            await session.dispose()


@pytest.mark.asyncio
async def test_a_check_past_its_deadline_fails_rather_than_recording_a_notice(
    tmp_path: Path, config_dir: Path
) -> None:
    """A timed-out check is a FAILURE (the ladder's contract, §5.4).

    Pre-fix the deadline's cancellation took the steering-detach branch: the
    call returned a notice as its "success", no strike was charged, the
    baseline became the notice, and the command ran on as a leaked job.
    """
    fifo, writer = _blocking_fifo(tmp_path)
    session = _bash_session(tmp_path / "sess", cwd=str(tmp_path))
    try:
        session._monitors._settings = MonitorSettings(run_timeout_ms=400)
        clock = [int(time.time() * 1000)]
        session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
        outcome = await session._monitors.create(
            {"name": "blocked", "tool": "bash", "arguments": {"command": f"cat {fifo}"}},
            cwd=str(tmp_path),
        )
        assert outcome.get("created"), outcome
        clock[0] += 10_000
        await session._monitors.pump(now_ms=clock[0])
        await drain_checks(session._monitors)

        counters = monitor_state.read_counters(config_dir, "sess", "m1")
        assert counters is not None, "a failed check still writes its health counters"
        assert counters["consecutive_failures"] == 1, counters
        assert "timed out" in str(counters.get("last_error")), counters
        assert monitor_state.read_snapshot(config_dir, "sess", "m1") is None
        assert [job for job in session.jobs.list() if job.type == "bash"] == []
    finally:
        os.close(writer)
        if not session._disposed:
            await session.dispose()


@pytest.mark.asyncio
async def test_a_swallowed_cancellation_cannot_apply_after_teardown(
    tmp_path: Path, config_dir: Path
) -> None:
    """The last-resort guard: a tool that swallows its cancel still cannot write.

    This is the shape the bash tool had (and any future tool could have): the
    call catches ``CancelledError`` and returns a value. The scheduler's own
    ``_disposed`` check must drop that result — ``dispose`` promises "a check
    in flight cannot apply its result after teardown".
    """
    started = asyncio.Event()

    async def execute(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> ToolResult:
        started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return ToolResult(
                tool_call_id=tool_call_id,
                content=[TextContent(text="swallowed-cancel")],
            )
        return ToolResult(tool_call_id=tool_call_id, content=[TextContent(text="slept")])

    slow = AgentTool(
        name="slowread",
        approval_tier="read",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        execute=execute,
    )
    session = Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[slow],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        cwd=str(tmp_path),
    )
    try:
        # The guard is a tick-side contract: bypass arm validation to arm the
        # spec the way a pre-fix store could already hold it.
        session._validate_monitor_call = lambda *a, **k: None  # type: ignore[method-assign]
        clock = [int(time.time() * 1000)]
        session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
        await session.set_monitor_schedules(
            [
                MonitorSpec(
                    id="m1",
                    name="slow",
                    tool="slowread",
                    arguments={"path": "x"},
                    every_ms=60_000,
                    created_at=1,
                    cwd=str(tmp_path),
                )
            ]
        )
        clock[0] += 10_000
        await session._monitors.pump(now_ms=clock[0])
        await asyncio.wait_for(started.wait(), DRAIN_BACKSTOP_S)
        task = next(iter(session._monitors._check_tasks), None)
        assert task is not None and not task.done()

        await session.dispose()
        await asyncio.wait([task], timeout=DRAIN_BACKSTOP_S)
        assert task.done(), "the swallowed check still settles"

        assert monitor_state.read_snapshot(config_dir, "sess", "m1") is None
        counters = monitor_state.read_counters(config_dir, "sess", "m1")
        assert counters is None or not counters.get("checks")
    finally:
        if not session._disposed:
            await session.dispose()


# ---------------------------------------------------------------------------
# A backgrounded call is never a check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_backgrounded_check_is_refused_at_arm(tmp_path: Path, config_dir: Path) -> None:
    """``background: true`` cannot be watched: the call would return a job id."""
    session = _bash_session(tmp_path / "sess", cwd=str(tmp_path))
    try:
        outcome = await session._monitors.create(
            {
                "name": "bg",
                "tool": "bash",
                "arguments": {"command": "cat x", "background": True},
            },
            cwd=str(tmp_path),
        )
        assert "error" in outcome, outcome
        assert "background" in outcome["error"], outcome
        assert session._monitors.monitors == ()
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_job_notice_result_is_never_recorded_as_output(
    tmp_path: Path, config_dir: Path
) -> None:
    """The belt: if a check's call comes back handed to a background job,
    fail the tick and cancel the escaped job — never adopt the notice."""
    fifo, writer = _blocking_fifo(tmp_path)
    session = _bash_session(tmp_path / "sess", cwd=str(tmp_path))
    try:
        # A spec bypassing the arm gate (armed before that gate existed).
        session._validate_monitor_call = lambda *a, **k: None  # type: ignore[method-assign]
        spec = MonitorSpec(
            id="m1",
            name="bg",
            tool="bash",
            arguments={"command": f"cat {fifo}", "background": True},
            every_ms=60_000,
            created_at=1,
            cwd=str(tmp_path),
        )
        outcome = await session._run_monitor_check(spec)
        assert outcome.get("error"), outcome
        assert outcome.get("kind") == "fatal", outcome
        jobs = [job for job in session.jobs.list() if job.type == "bash"]
        assert jobs and all(job.status == "cancelled" for job in jobs), [
            (job.id, job.status) for job in jobs
        ]
    finally:
        os.close(writer)
        await session.dispose()


# ---------------------------------------------------------------------------
# An unusable captured cwd fails loudly (the pin for that half of the class)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unusable_captured_cwd_fails_the_ladder_loudly(
    tmp_path: Path, config_dir: Path
) -> None:
    """A captured cwd that no longer exists must disable the monitor with the
    reason, not watch nothing in silence.

    Pins the deliberate non-change: the check keeps the captured cwd (a
    re-resolution to some other directory would silently diff different
    content), and the bash tool's spawn guard turns the missing directory
    into an ordinary failed check — five strikes, a legible disable.
    """
    missing = tmp_path / "gone-dir"
    session = _bash_session(tmp_path / "sess", cwd=str(tmp_path))
    try:
        clock = [int(time.time() * 1000)]
        session._monitors._now = lambda: clock[0]  # type: ignore[method-assign]
        outcome = await session._monitors.create(
            {"name": "gone", "tool": "bash", "arguments": {"command": "cat x.txt"}},
            cwd=str(missing),
        )
        assert outcome.get("created"), outcome
        for _ in range(5):
            clock[0] += 16 * 60_000
            await session._monitors.pump(now_ms=clock[0])
            await drain_checks(session._monitors)
        counters = monitor_state.read_counters(config_dir, "sess", "m1")
        assert counters is not None
        assert counters["disabled"] is True, counters
        assert counters["consecutive_failures"] >= 5, counters
        assert str(missing) in counters["disabled_reason"], counters
    finally:
        await session.dispose()
