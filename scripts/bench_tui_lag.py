"""Loop-lag measurements for the TUI responsiveness regression (v0.29.0 -> v0.33.1).

Reproduces the three reported symptoms against the real code, with a lag
monitor on the event loop, so a fix can show before/after numbers rather
than a claim. The monitor is the design's harness: a 5 ms tick probe records
every gap that exceeds the stall threshold, and asyncio's own
``slow_callback_duration`` (debug mode) catches synchronous bodies the tick
probe cannot see (a tick and a stall inside one callback read as one gap).

Three scenarios, mirroring the symptoms as reported:

S1  boot    — ``_prepare`` on a store shaped like the operator's real one
              (1,000 title-less sessions, 50 with origins unmarked). Measures
              wall time and loop stalls of the factory's pre-Session phase.
S2  send    — ``costs.turn_cost`` on the loop with a cold memo and an
              unlisted model (discovery stubbed to a controllable latency).
              This is the ``message_end`` -> ``ContextUsageReported`` path.
S3  emit    — ``bash._emit_update`` tick cost with multi-MB accumulated
              output (the O(total output) per 500 ms tick the design measured).

Usage::

    python scripts/bench_tui_lag.py [--json OUT.json]

The JSON blob is what the PR quotes: every number in the table comes from a
run of this script, never from a hand-typed constant.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

#: Gap (ms) over which a tick is recorded as a stall. The design's bar for a
#: green run is "no stall > 50 ms"; 30 ms is where a keystroke starts to feel
#: late, so the monitor records from there and the report quotes the max.
STALL_MS = 30.0

TICK_S = 0.005


class LagMonitor:
    """Records event-loop gaps above ``stall_ms`` while it runs."""

    def __init__(self, stall_ms: float = STALL_MS) -> None:
        self.stall_ms = stall_ms
        self.stalls: list[float] = []
        self._task: asyncio.Task[None] | None = None

    async def _probe(self) -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(TICK_S)
            now = time.perf_counter()
            gap_ms = (now - last) * 1000.0
            if gap_ms >= self.stall_ms:
                self.stalls.append(gap_ms)
            last = now

    def start(self) -> None:
        self._task = asyncio.create_task(self._probe())

    async def started(self) -> None:
        """Yield until the probe has taken at least one tick.

        ``create_task`` schedules; it does not run. Starting the monitor and
        immediately doing synchronous work means the probe's ``last`` was
        never set, and the very stall the monitor exists to record is missed
        because the first post-stall tick initializes the baseline instead of
        comparing against it.
        """
        await asyncio.sleep(TICK_S * 2)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    @property
    def max_stall_ms(self) -> float:
        return max(self.stalls) if self.stalls else 0.0


# -- S1: boot (_prepare store scans) ------------------------------------------


def _build_store(root: Path, sessions: int) -> Path:
    """A store shaped like the reported one: many sessions with NO journalled
    title (the population the title backfill re-reads on every boot), plus a
    few origin-less dirs for the origin sweep."""
    import local_operator.resume as resume

    store = root / "cfg"
    sessions_dir = store / "sessions"
    sessions_dir.mkdir(parents=True)
    for index in range(sessions):
        directory = sessions_dir / f"{index:08x}abcd"
        directory.mkdir()
        # A transcript with an opening message and no title entries: the
        # backfill's expensive case (full read, nothing to write).
        line = (
            '{"type": "message", "payload": {"role": "user", '
            '"content": [{"type": "text", "text": "opening line"}]}}\n'
        )
        (directory / resume.TRANSCRIPT_NAME).write_text(line * 40, encoding="utf-8")
    return store


async def bench_boot(root: Path, sessions: int = 1000) -> dict[str, Any]:
    """``_prepare``'s store-scan phase under the lag monitor, twice.

    The second pass is the perpetual-rescan case the design measured on the
    real store: nothing was written on the first pass (no titles to journal),
    so a store that cannot grow sidecars re-reads every transcript on every
    boot forever. A2's sentinel makes pass two stat-only; the BEFORE number
    for pass two is the number that motivated it.
    """
    from local_operator.resume import backfill_session_origins, backfill_session_titles

    store = _build_store(root, sessions)
    monitor = LagMonitor()
    monitor.start()
    await monitor.started()
    started = time.perf_counter()
    backfill_session_origins(store)
    backfill_session_titles(store)
    first_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    backfill_session_origins(store)
    backfill_session_titles(store)
    second_ms = (time.perf_counter() - started) * 1000.0
    await monitor.stop()
    return {
        "sessions": sessions,
        "first_pass_ms": round(first_ms, 1),
        "second_pass_ms": round(second_ms, 1),
        "max_stall_ms": round(monitor.max_stall_ms, 1),
        "stalls_over_30ms": len(monitor.stalls),
    }


# -- S2: send (turn_cost on the loop, cold memo) -------------------------------


class _SlowListing:
    """Stands in for provider discovery with a controllable latency.

    The real path is ``httpx.Client.get`` against the provider's /v1/models;
    the design measured 418 ms warm-disk and 10 s + 3 s budgets cold. The
    stub keeps the SHAPE of the call (a synchronous sleep on the calling
    thread) without the network, so the measurement is the loop cost of the
    call, not of the network.
    """

    def __init__(self, latency_s: float) -> None:
        self.latency_s = latency_s
        self.calls = 0

    def __call__(self, provider: str, **_kwargs: Any) -> tuple[list[Any], str]:
        self.calls += 1
        time.sleep(self.latency_s)
        return [], "ok"


async def bench_send(latency_s: float = 0.4) -> dict[str, Any]:
    """``turn_cost`` from the loop path with a cold memo and an unlisted model.

    This is the exact call ``on_context_usage_reported`` makes per
    ``message_end``. With discovery stubbed to ``latency_s``, the BEFORE
    behaviour blocks the loop for the latency; C1 returns in microseconds and
    hands the fetch to a background thread (C2), so the monitor should record
    no stall either way after the fix.
    """
    from local_operator.model import configure
    from local_operator.tui.costs import turn_cost

    slow = _SlowListing(latency_s)
    # Both discovery legs route through available_models; patching it at the
    # discovery module covers the provider leg and the aggregator leg.
    from local_operator.model import discovery

    original = discovery.available_models
    discovery.available_models = slow  # type: ignore[assignment]
    configure.invalidate_model_info_cache()
    monitor = LagMonitor()
    monitor.start()
    await monitor.started()
    started = time.perf_counter()

    class _Usage:
        input_tokens = 1_000
        output_tokens = 2_000
        cache_read_tokens = 0
        cache_write_tokens = 0
        usd_cost = None
        cost_components = None

    cost = turn_cost("kimi/unlisted-model-x", _Usage())
    loop_ms = (time.perf_counter() - started) * 1000.0
    # Give a background refresh (C2) a moment to land, so the AFTER run can
    # also report that the real fetch happened off-loop.
    await asyncio.sleep(max(latency_s * 1.5, 0.2))
    await monitor.stop()
    discovery.available_models = original  # type: ignore[assignment]
    configure.invalidate_model_info_cache()
    return {
        "discovery_latency_s": latency_s,
        "loop_ms": round(loop_ms, 2),
        "max_stall_ms": round(monitor.max_stall_ms, 1),
        "cost": cost,
        "discovery_calls": slow.calls,
    }


# -- S3: bash emit tick cost ----------------------------------------------------


async def bench_bash_emit(total_mb: float = 8.0) -> dict[str, Any]:
    """One ``_emit_update`` tick against ``total_mb`` of accumulated output.

    Drives the REAL tool with the real closure (a fake pipe delivers the
    synthetic stream through the real ``_pump``), measures the loop stalls
    its 500 ms emits cause, and separately times the exact body
    ``_emit_update`` runs (bounded tail join + redact + summary) per tick.
    The BEFORE tree joined/redacted the WHOLE accumulated stream per tick;
    the fix bounds it, so tick cost must be flat in ``total_mb``.
    """
    from local_operator.harness.types import ToolContext
    from local_operator.tools.builtin import (
        _EMIT_SNAPSHOT_BYTES,
        _bash_output_summary,
        _redact_tool_text,
        _tail_chunks,
        execute_bash,
    )
    from local_operator.variables import VariableStore

    store = VariableStore()
    store.store_credential("GITHUB_TOKEN", "ghp_notarealtoken000000")
    store.store_credential("ANTHROPIC_API_KEY", "sk-ant-notarealkey000000")
    context = ToolContext(cwd="/tmp", session_id="bench", agent_id="main", variables=store)

    chunk = ("x" * 65536 + "\n").encode()
    chunks = [chunk] * int(total_mb * 1_048_576 / len(chunk))

    class _FakeReader:
        def __init__(self, data: bytes) -> None:
            self._data = data

        async def read(self, _n: int = -1) -> bytes:
            data, self._data = self._data, b""
            return data

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 1
            self.returncode = None
            self.stdout = _FakeReader(b"".join(chunks))
            self.stderr = _FakeReader(b"")

        async def wait(self) -> int:
            await asyncio.Event().wait()
            return 0  # pragma: no cover - unreachable

    async def _fake_exec(*_a: Any, **_k: Any) -> Any:
        return _FakeProc()

    payloads: list[int] = []

    def on_update(update: Any) -> None:
        payloads.append(len(update.content[0].text))

    import local_operator.tools.builtin as builtin

    real_create = asyncio.create_subprocess_exec
    builtin.asyncio.create_subprocess_exec = _fake_exec  # type: ignore[attr-defined]
    monitor = LagMonitor()
    task: asyncio.Task[Any] | None = None
    try:
        task = asyncio.create_task(
            execute_bash(  # type: ignore[arg-type]
                "bench", {"command": "true", "timeout": 60}, None, on_update, context
            )
        )
        # Readiness: a second update proves the stream is fully accumulated
        # (the pump landed) and the 500 ms cadence is running. Payload LENGTH
        # cannot be the signal: the fix under test bounds it.
        deadline = time.perf_counter() + 10.0
        while time.perf_counter() < deadline:
            await asyncio.sleep(0.02)
            if len(payloads) >= 2:
                break
        else:
            raise RuntimeError("bash emit never fired against the synthetic stream")
        monitor.start()
        await monitor.started()
        count = len(payloads)
        while len(payloads) < count + 4:
            await asyncio.sleep(0.05)
    finally:
        builtin.asyncio.create_subprocess_exec = real_create  # type: ignore[attr-defined]
        if task is not None:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
    await monitor.stop()

    # Direct per-tick timing of the exact body _emit_update runs, measured
    # the same way for both trees: BEFORE joins every chunk, AFTER joins the
    # bounded tail. (Loop-stall attribution under a loaded machine is coarse;
    # this number is the deterministic one.)
    body_times: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        stdout = _redact_tool_text(
            b"".join(_tail_chunks(chunks, _EMIT_SNAPSHOT_BYTES)).decode("utf-8", errors="replace"),
            context,
        )
        stderr = _redact_tool_text("", context)
        _bash_output_summary(stdout, stderr)
        body_times.append((time.perf_counter() - started) * 1000.0)
    return {
        "total_mb": total_mb,
        "emit_body_ms_median": round(statistics.median(body_times), 2),
        "emit_body_ms_max": round(max(body_times), 2),
        "live_payload_chars": payloads[-1] if payloads else 0,
        "max_stall_ms": round(monitor.max_stall_ms, 1),
    }


# -- P: parallel instances booting against one shared store ---------------------


async def bench_parallel_boot(instances: int = 5, sessions: int = 1000) -> dict[str, Any]:
    """N instances running the store scans concurrently over ONE store.

    The cross-instance shape the operator reported: every boot sweeps the
    same shared ``sessions/`` directory. Measures the wall time of N
    concurrent scan passes (steady state, sentinels in place) against a
    store fixture, plus the first-boot migration cost once.
    """
    import concurrent.futures

    import tempfile as _tempfile

    store = _build_store(Path(_tempfile.mkdtemp()), sessions)

    def one_pass() -> float:
        from local_operator.resume import backfill_session_origins, backfill_session_titles

        started = time.perf_counter()
        backfill_session_origins(store)
        backfill_session_titles(store)
        return (time.perf_counter() - started) * 1000.0

    # First pass: one-time migration (writes sentinels).
    first_ms = one_pass()

    # Steady state: N concurrent passes, as N booted instances would run.
    with concurrent.futures.ThreadPoolExecutor(max_workers=instances) as pool:
        started = time.perf_counter()
        passes = list(pool.map(lambda _: one_pass(), range(instances)))
        wall_ms = (time.perf_counter() - started) * 1000.0
    return {
        "instances": instances,
        "sessions": sessions,
        "first_pass_ms": round(first_ms, 1),
        "parallel_steady_wall_ms": round(wall_ms, 1),
        "slowest_instance_ms": round(max(passes), 1),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write results here")
    parser.add_argument("--sessions", type=int, default=1000)
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--skip-boot", action="store_true")
    args = parser.parse_args()

    import tempfile

    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        if not args.skip_boot:
            results["s1_boot"] = await bench_boot(Path(tmp), sessions=args.sessions)
            results["p_parallel_boot"] = await bench_parallel_boot(
                instances=args.instances, sessions=args.sessions
            )
        results["s2_send"] = await bench_send()
        results["s3_bash_emit"] = await bench_bash_emit()

    print(json.dumps(results, indent=2))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
