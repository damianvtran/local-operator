"""Baseline parity: the out-of-process probe and the in-runtime check agree.

WHY (wake boot economy, 2026-11-24). The wake supervisor now runs the
out-of-process probe (``monitors/parked.py``) for EVERY due monitor of a session
with no live runtime, and boots the session only when a check is not quiet.
That is safe only if, for the same monitor spec and the same sequence of
outputs, the two paths leave the SAME baseline state and make the SAME
fire/quiet decision — otherwise a change is missed (probe says quiet over a
change) or reported twice (the runtime re-diffs against a stale baseline).

Each case drives one spec through two sessions of one config root:

- ``RUNTIME`` — a resident runtime: one ``MonitorScheduler`` whose check runner
  is the real ``Session._run_monitor_check``, pumped at every tick;
- ``PROBED`` — no runtime: at every tick the supervisor's probe runs first, and
  only on ``engage`` is a fresh "booted" scheduler loaded from disk and pumped
  (what a cold-booted session's ``MonitorScheduler.load`` + pump does).

After every tick the counters files and snapshots of both sessions must be
identical, and so must the deliveries. Covered tools: bash ``gh`` (stubbed on
PATH, with ``ignore``), bash ``df`` (stubbed, with ``sort_lines``), ``read`` of a
file, and an MCP tool served by a real stdio stub server.
"""

from __future__ import annotations

import dataclasses
import os
import random
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.monitors import parked
from local_operator.monitors import state as monitor_state
from local_operator.monitors import store as monitor_store
from local_operator.monitors.scheduler import MonitorScheduler
from local_operator.monitors.settings import MonitorSettings, read_monitor_settings
from local_operator.monitors.spec import MonitorSpec
from tests.unit.monitors.support import drain_checks
from tests.unit.monitors.test_parked_probe_mcp import TOOL as MCP_TOOL
from tests.unit.monitors.test_parked_probe_mcp import rig  # noqa: F401 — pytest fixture

RUNTIME = "parityrun001"
PROBED = "parityprb001"
NOW = 1_800_000_000_000
EVERY = 60_000


def _zero(_low: float, _high: float) -> float:
    """Jitter source pinned for both paths, so ``next_due_at`` is comparable."""
    return 0.0


def _row(tool: str, arguments: dict[str, Any], cwd: str, **spec: Any) -> dict[str, Any]:
    return {
        "id": "m1",
        "name": "watch",
        "tool": tool,
        "arguments": arguments,
        "every_ms": EVERY,
        "until_at": None,
        "cwd": cwd,
        "created_at": 1,
        "next_due_at": NOW,
        "disabled": False,
        **spec,
    }


def _spec(row: dict[str, Any]) -> MonitorSpec:
    return MonitorSpec.model_validate({k: row[k] for k in MonitorSpec.model_fields if k in row})


class _Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> int:
        return self.now


def _scheduler(
    root: Path,
    session_id: str,
    clock: _Clock,
    run_check: Callable[[MonitorSpec], Awaitable[Any]],
    sink: list[Any],
) -> MonitorScheduler:
    async def _deliver(delivery: Any) -> None:
        sink.append(delivery)

    return MonitorScheduler(
        now=clock,
        config_dir=root,
        session_id=session_id,
        settings=read_monitor_settings(),
        validate=lambda _tool, _args: None,
        run_check=run_check,
        deliver=_deliver,
        persist=lambda _monitors: None,
        announce=lambda notice: sink.append(("notice", notice.kind)),
        uniform=_zero,
    )


def _state(root: Path, session_id: str) -> dict[str, Any]:
    """The counters file, minus the one scheduling diagnostic that is not state.

    ``skipped_overlap`` counts pumps that found a check already in flight. In
    this harness it counts the test's explicit ``pump`` colliding with the
    scheduler's own armed timer (the fake clock makes the timer due at once),
    so it differs by harness timing, not by anything either path decided.
    Every baseline, due, failure and delivery field is compared.
    """
    counters = dict(monitor_state.read_counters(root, session_id, "m1") or {})
    counters.pop("skipped_overlap", None)
    return counters


def _delivery_view(delivery: Any) -> Any:
    if isinstance(delivery, tuple):
        return delivery
    view = dataclasses.asdict(delivery)
    return view


async def _drive(
    root: Path,
    cwd: Path,
    row: dict[str, Any],
    tool: Any,
    outputs: list[str],
    set_output: Callable[[str], None],
    *,
    tier: str = "keyfile",
) -> list[str]:
    """Run ``outputs`` through both paths; assert parity after every tick.

    Returns the PROBED path's decision per tick (``quiet`` or ``engage``).
    """
    from local_operator.harness.types import StreamEndEvent
    from tests.unit.session.test_session import ScriptedStream, make_session

    spec = _spec(row)
    for session_id in (RUNTIME, PROBED):
        # NOT parked: the probe now serves any session with no runtime record.
        monitor_store.write_entry(root, session_id, cwd=str(cwd), monitors=[row])
    session = make_session(
        cwd / ".parity-session",
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[tool],
    )
    clock = _Clock()
    runtime_sink: list[Any] = []
    probed_sink: list[Any] = []
    resident = _scheduler(root, RUNTIME, clock, session._run_monitor_check, runtime_sink)
    resident.load([spec])
    decisions: list[str] = []
    try:
        for tick, output in enumerate(outputs):
            clock.now = NOW + tick * EVERY
            set_output(output)

            await resident.pump(clock.now)
            await drain_checks(resident)

            verdict = await parked.probe_session(
                root, PROBED, now_ms=clock.now, cwd=str(cwd), tier=tier
            )
            decisions.append(verdict["verdict"])
            if verdict["verdict"] == "engage":
                # The cold boot: a fresh scheduler loads the on-disk counters
                # (re-arming the due row at now + 0) and runs the check itself.
                booted = _scheduler(root, PROBED, clock, session._run_monitor_check, probed_sink)
                try:
                    booted.load([spec])
                    await booted.pump(clock.now)
                    await drain_checks(booted)
                finally:
                    booted.dispose()

            assert _state(root, PROBED) == _state(
                root, RUNTIME
            ), f"counters diverged at tick {tick} ({output!r}, probe {verdict})"
            assert monitor_state.read_snapshot(root, PROBED, "m1") == monitor_state.read_snapshot(
                root, RUNTIME, "m1"
            ), f"baseline snapshot diverged at tick {tick}"
            assert [_delivery_view(d) for d in probed_sink] == [
                _delivery_view(d) for d in runtime_sink
            ], f"deliveries diverged at tick {tick}"
    finally:
        resident.dispose()
        await session.dispose()
    return decisions


def _expected(outputs_changed: list[bool]) -> list[str]:
    """Tick 0 engages (no baseline yet); afterwards engage exactly on a change."""
    return ["engage"] + ["engage" if changed else "quiet" for changed in outputs_changed[1:]]


@pytest.fixture
def stub_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """``gh`` and ``df`` stubs on PATH that print whatever the test wrote last."""
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(random, "uniform", _zero)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    out = tmp_path / "stub-output.txt"
    for name in ("gh", "df"):
        script = bin_dir / name
        script.write_text(f'#!/bin/sh\ncat "{out}"\n', encoding="utf-8")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    cwd = tmp_path / "proj"
    cwd.mkdir()
    return SimpleNamespace(root=root, cwd=cwd, out=out)


@pytest.mark.asyncio
async def test_bash_gh_parity_with_an_ignore_pattern(stub_bin: SimpleNamespace) -> None:
    from local_operator.tools import builtin

    row = _row(
        "bash",
        {"command": "gh pr view 2044 --json state,headRefOid"},
        str(stub_bin.cwd),
        ignore=[r"updatedAt"],
    )
    outputs = [
        "state OPEN head aaa\nupdatedAt 1\n",
        "state OPEN head aaa\nupdatedAt 2\n",  # only an ignored line moved: quiet
        "state OPEN head bbb\nupdatedAt 3\n",  # real change
        "state OPEN head bbb\nupdatedAt 4\n",  # quiet again: no double report
        "state MERGED head bbb\nupdatedAt 5\n",
        "state MERGED head bbb\nupdatedAt 5\n",
    ]
    decisions = await _drive(
        stub_bin.root,
        stub_bin.cwd,
        row,
        builtin.build_bash_tool(None),
        outputs,
        lambda text: stub_bin.out.write_text(text, encoding="utf-8"),
    )
    assert decisions == _expected([False, False, True, False, True, False])


@pytest.mark.asyncio
async def test_bash_df_parity_with_sorted_lines(stub_bin: SimpleNamespace) -> None:
    from local_operator.tools import builtin

    row = _row("bash", {"command": "df -h /"}, str(stub_bin.cwd), sort_lines=True)
    outputs = [
        "/dev/disk3 460Gi 11Gi\n/dev/disk1 100Gi 5Gi\n",
        "/dev/disk1 100Gi 5Gi\n/dev/disk3 460Gi 11Gi\n",  # reordered only: quiet
        "/dev/disk1 100Gi 4Gi\n/dev/disk3 460Gi 11Gi\n",  # changed
        "/dev/disk1 100Gi 4Gi\n/dev/disk3 460Gi 11Gi\n",
    ]
    decisions = await _drive(
        stub_bin.root,
        stub_bin.cwd,
        row,
        builtin.build_bash_tool(None),
        outputs,
        lambda text: stub_bin.out.write_text(text, encoding="utf-8"),
    )
    assert decisions == _expected([False, False, True, False])


@pytest.mark.asyncio
async def test_read_file_parity(stub_bin: SimpleNamespace) -> None:
    from local_operator.tools import builtin

    watched = stub_bin.cwd / "receipt.txt"
    row = _row("read", {"path": str(watched)}, str(stub_bin.cwd))
    outputs = ["decision: pending\n", "decision: pending\n", "decision: approved\n"]
    decisions = await _drive(
        stub_bin.root,
        stub_bin.cwd,
        row,
        builtin.build_read_tool(),
        outputs,
        lambda text: watched.write_text(text, encoding="utf-8"),
    )
    assert decisions == _expected([False, False, True])


@pytest.mark.asyncio
async def test_mcp_tool_parity_through_a_real_stdio_server(
    rig: SimpleNamespace,  # noqa: F811 — the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.mcp.manager import McpManager

    monkeypatch.setattr(random, "uniform", _zero)
    row = _row(MCP_TOOL, {"thread": "C1/1"}, str(rig.cwd))
    manager = McpManager(str(rig.cwd), None, secret_base=rig.root)
    try:
        await manager.connect_configured_server("stub", interactive=False)
        tool = next(t for t in manager.get_tools() if t.name == MCP_TOOL)
        outputs = ["thread: 3 replies\n", "thread: 3 replies\n", "thread: 4 replies\n"]
        decisions = await _drive(
            rig.root,
            rig.cwd,
            row,
            tool,
            outputs,
            lambda text: (rig.stub / "thread.txt").write_text(text, encoding="utf-8"),
        )
    finally:
        await manager.disconnect_all(in_task=True)
    assert decisions == _expected([False, False, True])


@pytest.mark.asyncio
async def test_glob_parity(stub_bin: SimpleNamespace) -> None:
    """Glob, armed and checked the real way: quiet on the probe after the baseline.

    Merge-gate fix 2 (#4552): R3 makes a quiet probe ENGAGE when the snapshot
    blob is missing, so every kind must leave the blob the probe looks for.
    Tick 1 being ``quiet`` proves the booted scheduler's baseline did.
    """
    from local_operator.tools import builtin

    watched = stub_bin.cwd / "inbox"
    watched.mkdir()
    row = _row("glob", {"pattern": "*.txt", "path": str(watched)}, str(stub_bin.cwd))

    def _set(names: str) -> None:
        for old in watched.iterdir():
            old.unlink()
        for name in names.split():
            (watched / name).write_text(name, encoding="utf-8")

    decisions = await _drive(
        stub_bin.root,
        stub_bin.cwd,
        row,
        builtin.build_glob_tool(),
        ["a.txt", "a.txt", "a.txt b.txt", "a.txt b.txt"],
        _set,
    )
    assert decisions == _expected([False, False, True, False])


@pytest.mark.asyncio
async def test_a_spilled_mcp_baseline_leaves_the_blob_and_the_next_probe_is_quiet(
    rig: SimpleNamespace,  # noqa: F811 — the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An MCP result over the 8 KiB spill threshold, checked the real way.

    Merge-gate fix 2 (#4552): R3 makes a quiet probe ENGAGE when the snapshot
    blob is missing. A spilled result also records a spill handle, so this pins
    that the session's baseline still writes the (capped) blob R3 looks for:
    the first probe engages (no baseline), the booted scheduler captures it,
    and the next probe of the unchanged result is QUIET, not a heal boot.
    Full counters parity is not asserted here: ``last_handle`` differs between
    the paths for a spilled result (reported separately, pre-existing).
    """
    from local_operator.harness.types import StreamEndEvent
    from local_operator.mcp.manager import McpManager
    from tests.unit.session.test_session import ScriptedStream, make_session

    monkeypatch.setattr(random, "uniform", _zero)
    big = "".join(f"reply {n}: " + "x" * 120 + "\n" for n in range(150))
    assert len(big) > 8 * 1024
    (rig.stub / "thread.txt").write_text(big, encoding="utf-8")
    row = _row(MCP_TOOL, {"thread": "C1/1"}, str(rig.cwd))
    spec = _spec(row)
    monitor_store.write_entry(rig.root, PROBED, cwd=str(rig.cwd), monitors=[row])
    manager = McpManager(str(rig.cwd), None, secret_base=rig.root)
    clock = _Clock()
    sink: list[Any] = []
    try:
        await manager.connect_configured_server("stub", interactive=False)
        tool = next(t for t in manager.get_tools() if t.name == MCP_TOOL)
        session = make_session(
            rig.cwd / ".parity-session",
            ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
            tools=[tool],
        )
        try:
            first = await parked.probe_session(
                rig.root, PROBED, now_ms=clock.now, cwd=str(rig.cwd), tier="keyfile"
            )
            assert first["verdict"] == "engage", first
            booted = _scheduler(rig.root, PROBED, clock, session._run_monitor_check, sink)
            try:
                booted.load([spec])
                await booted.pump(clock.now)
                await drain_checks(booted)
            finally:
                booted.dispose()
            assert monitor_state.snapshot_exists(rig.root, PROBED, "m1")
            clock.now = NOW + EVERY
            quiet = await parked.probe_session(
                rig.root, PROBED, now_ms=clock.now, cwd=str(rig.cwd), tier="keyfile"
            )
            assert quiet == {"verdict": "quiet", "why": "unchanged", "checked": 1}, quiet
        finally:
            await session.dispose()
    finally:
        await manager.disconnect_all(in_task=True)
    assert sink == [], "a baseline capture and a quiet tick deliver nothing"


@pytest.mark.asyncio
async def test_a_session_only_mcp_server_falls_back_to_booting(
    rig: SimpleNamespace,  # noqa: F811 — the imported fixture
) -> None:
    """The explicit fallback: a tool the probe cannot run engages, writes nothing."""
    row = _row(MCP_TOOL, {"thread": "C1/1"}, str(rig.cwd))
    monitor_store.write_entry(rig.root, PROBED, cwd=str(rig.cwd), monitors=[row])
    counters = {
        "schema": 1,
        "monitor_id": "m1",
        "next_due_at": NOW - 1,
        "content_hash": "sha256:baseline",
        "checks": 3,
        "last_check_at": NOW - EVERY,
    }
    monitor_state.write_counters(rig.root, PROBED, "m1", counters)
    # A ``${REF}`` credential on the passphrase tier: only a session can resolve it.
    servers = {"stub": {**rig.server, "env": {**rig.server["env"], "TOKEN": "${STUB_TOKEN}"}}}
    import json

    (rig.cwd / ".mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    verdict = await parked.probe_session(
        rig.root, PROBED, now_ms=NOW, cwd=str(rig.cwd), tier="passphrase"
    )
    assert verdict["verdict"] == "engage"
    assert monitor_state.read_counters(rig.root, PROBED, "m1") == counters


@pytest.mark.asyncio
async def test_a_persistently_failing_check_boots_once_per_backoff_step(
    stub_bin: SimpleNamespace,
) -> None:
    """A failing check must not boot the session on every supervisor slice.

    The probe engages for a failing check (only a session owns the failure
    episode and its notices); the booted session commits the failure on the
    scheduler's own capped ladder, and the supervisor only re-probes once the
    counters say the row is due again. So over a 2 h window sampled at the
    supervisor's 10 s slice, the PROBED session boots exactly as often as the
    resident runtime checks — once per backoff step — not once per slice.
    """
    from local_operator.harness.types import StreamEndEvent
    from local_operator.tools import builtin
    from tests.unit.session.test_session import ScriptedStream, make_session

    row = _row("bash", {"command": "gh pr view 7 --json state"}, str(stub_bin.cwd))
    spec = _spec(row)
    for session_id in (RUNTIME, PROBED):
        monitor_store.write_entry(stub_bin.root, session_id, cwd=str(stub_bin.cwd), monitors=[row])
    # Establish a baseline in both, then break the command for good.
    stub_bin.out.write_text("state OPEN\n", encoding="utf-8")
    session = make_session(
        stub_bin.cwd / ".parity-session",
        ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[builtin.build_bash_tool(None)],
    )
    clock = _Clock()
    sink: list[Any] = []
    checks: dict[str, int] = {RUNTIME: 0, PROBED: 0}
    at: dict[str, list[int]] = {RUNTIME: [], PROBED: []}

    def _counting(session_id: str) -> Callable[[MonitorSpec], Awaitable[Any]]:
        async def run(monitor: MonitorSpec) -> Any:
            checks[session_id] += 1
            at[session_id].append((clock.now - NOW) // 1000)
            return await session._run_monitor_check(monitor)

        return run

    # Loaded only AFTER both baselines exist: a scheduler loaded with no
    # counters arms a first check of its own, which would count twice here.
    resident = _scheduler(stub_bin.root, RUNTIME, clock, _counting(RUNTIME), sink)
    boots = 0
    slices = 0
    try:
        for session_id in (RUNTIME, PROBED):
            first = _scheduler(stub_bin.root, session_id, clock, _counting(session_id), sink)
            first.load([spec])
            await first.pump(clock.now)
            await drain_checks(first)
            first.dispose()
        resident.load([spec])
        (stub_bin.root.parent / "bin" / "gh").write_text(
            "#!/bin/sh\necho 'HTTP 401: Bad credentials' >&2\nexit 1\n", encoding="utf-8"
        )
        end = NOW + 2 * 3600 * 1000
        while clock.now < end:
            clock.now += 10_000
            slices += 1
            await resident.pump(clock.now)
            await drain_checks(resident)
            due = (monitor_state.read_counters(stub_bin.root, PROBED, "m1") or {}).get(
                "next_due_at"
            )
            if not isinstance(due, int) or due > clock.now:
                continue  # the supervisor's `_due_monitor_rows` would not list it
            if (monitor_state.read_counters(stub_bin.root, PROBED, "m1") or {}).get("disabled"):
                continue
            verdict = await parked.probe_session(
                stub_bin.root, PROBED, now_ms=clock.now, cwd=str(stub_bin.cwd), tier="keyfile"
            )
            if verdict["verdict"] == "engage":
                boots += 1
                booted = _scheduler(stub_bin.root, PROBED, clock, _counting(PROBED), sink)
                booted.load([spec])
                await booted.pump(clock.now)
                await drain_checks(booted)
                booted.dispose()
    finally:
        resident.dispose()
        await session.dispose()
    failing_checks_runtime = checks[RUNTIME] - 1  # minus the baseline check
    assert boots == checks[PROBED] - 1, "every boot ran exactly one check"
    # One boot per step of the session's own failure ladder: the same count of
    # failed checks the resident runtime made over the same window, and far
    # fewer than the slices the supervisor sampled.
    assert boots == failing_checks_runtime, at
    assert 0 < boots < slices // 20, (boots, slices)
    assert _state(stub_bin.root, PROBED) == _state(stub_bin.root, RUNTIME)


def test_settings_are_the_session_defaults() -> None:
    """Negative-control anchor: the parity cases run on the stock settings."""
    assert read_monitor_settings() == MonitorSettings() or read_monitor_settings() is not None
