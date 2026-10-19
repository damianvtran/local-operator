"""Diagnostic clocks must explain a failure without changing its verdict."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from tests.loop_callback_spans import CallbackSpans
from tests.unit.test_tui_responsiveness import StallRecorder


class Clocks:
    def __init__(self) -> None:
        self.cpu = 0.0
        self.wall = 0.0

    def advance(self, cpu: float, wall: float) -> None:
        self.cpu += cpu
        self.wall += wall


def run_handle(loop: asyncio.AbstractEventLoop, callback: Callable[[], None]) -> None:
    asyncio.Handle(callback, (), loop)._run()


def test_many_callbacks_and_one_block_have_equal_cpu_but_different_attribution() -> None:
    loop = asyncio.new_event_loop()
    clocks = Clocks()
    try:
        with CallbackSpans(
            loop, cpu_clock=lambda: clocks.cpu, wall_clock=lambda: clocks.wall
        ) as spans:

            def small() -> None:
                clocks.advance(0.010, 0.010)

            for _ in range(12):
                run_handle(loop, small)
            many = spans.take_window()

            def block() -> None:
                clocks.advance(0.120, 0.120)

            run_handle(loop, block)
            single = spans.take_window()
        assert many.count == 12
        assert many.cpu_ms == pytest.approx(single.cpu_ms)
        assert many.top_cpu[0].cpu_ms == pytest.approx(10)
        assert single.count == 1
        assert single.top_cpu[0].cpu_ms == pytest.approx(120)
        assert single.top_cpu[0].name.endswith(".block")
    finally:
        loop.close()


def test_wall_only_block_is_retained_and_storage_is_bounded() -> None:
    loop = asyncio.new_event_loop()
    clocks = Clocks()
    try:
        with CallbackSpans(
            loop, cpu_clock=lambda: clocks.cpu, wall_clock=lambda: clocks.wall
        ) as spans:

            def sleep() -> None:
                clocks.advance(0, 5)

            run_handle(loop, sleep)
            for _ in range(100):
                run_handle(loop, lambda: clocks.advance(0.001, 0.001))
            window = spans.take_window()
        assert window.count == 101
        assert window.top_wall[0].name.endswith(".sleep")
        assert window.top_wall[0].wall_ms == pytest.approx(5000)
        assert len(window.top_cpu) <= spans.TOP_N
        assert len(window.top_wall) <= spans.TOP_N
        assert spans.take_window().count == 0
    finally:
        loop.close()


def test_exception_restores_global_patch_and_preserves_original_callback_error() -> None:
    loop = asyncio.new_event_loop()
    original = asyncio.Handle._run
    errors: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    try:
        with pytest.raises(RuntimeError, match="test failure"):
            with CallbackSpans(loop) as spans:

                def fail() -> None:
                    raise ValueError("callback failure")

                run_handle(loop, fail)
                assert spans.take_window().count == 1
                raise RuntimeError("test failure")
        assert asyncio.Handle._run is original
        assert isinstance(errors[0]["exception"], ValueError)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_real_small_cpu_callback_is_named_and_measured() -> None:
    loop = asyncio.get_running_loop()
    done: asyncio.Future[None] = loop.create_future()

    def cpu_block() -> None:
        started = time.thread_time()
        while time.thread_time() - started < 0.020:
            pass
        done.set_result(None)

    with CallbackSpans(loop) as spans:
        loop.call_soon(cpu_block)
        await done
        window = spans.take_window()
    assert window.top_cpu[0].name.endswith(".cpu_block")
    assert window.top_cpu[0].cpu_ms >= 20
    assert window.cpu_ms >= 20


@pytest.mark.parametrize("cpu,wall", [(1600, 0), (0, 5000)])
def test_hostile_stalls_still_fail_at_maximum_calibration(cpu: float, wall: float) -> None:
    recorder = StallRecorder()
    recorder.cpu_calibration_ratio = 3
    recorder.cpu_stalls = [cpu]
    recorder.wall_stalls = [wall]
    with pytest.raises(AssertionError):
        recorder.assert_no_stall_loaded(noise_floor_ms=10000)


def test_failure_reports_bounded_named_windows_without_callback_payloads() -> None:
    loop = asyncio.new_event_loop()
    other_loop = asyncio.new_event_loop()
    clocks = Clocks()

    class SensitiveCallback:
        def __repr__(self) -> str:
            raise AssertionError("callback repr must not be inspected")

        def __call__(self) -> None:
            clocks.advance(0.020, 0.040)

    try:
        with CallbackSpans(
            loop, cpu_clock=lambda: clocks.cpu, wall_clock=lambda: clocks.wall
        ) as spans:
            run_handle(other_loop, SensitiveCallback())
            assert spans.take_window().count == 0
            for _ in range(20):
                run_handle(loop, SensitiveCallback())
                spans.observe_window(20, 40)
            recorder = StallRecorder(callbacks=spans)
            recorder.cpu_stalls = [600]
            with pytest.raises(AssertionError, match="between probe ticks") as failure:
                recorder.assert_no_stall_loaded()
        summary = spans.summary()
        cpu_windows = summary["worst_cpu_windows"]
        wall_windows = summary["worst_wall_windows"]
        assert isinstance(cpu_windows, list) and len(cpu_windows) == spans.WINDOW_N
        assert isinstance(wall_windows, list) and len(wall_windows) == spans.WINDOW_N
        assert "SensitiveCallback" in str(failure.value)
        assert "loop_thread_id" in str(failure.value)
        assert "worst_wall_windows" in str(failure.value)
    finally:
        loop.close()
        other_loop.close()


@pytest.mark.asyncio
async def test_probe_and_global_hook_are_restored_when_work_raises() -> None:
    original = asyncio.Handle._run
    recorder: StallRecorder | None = None
    with pytest.raises(RuntimeError, match="turn failed"):
        with CallbackSpans(asyncio.get_running_loop()) as spans:
            recorder = StallRecorder(callbacks=spans)
            try:
                await recorder.start()
                raise RuntimeError("turn failed")
            finally:
                await recorder.stop()
    assert asyncio.Handle._run is original
    assert recorder is not None and recorder._task is None


def test_cpu_time_bucket_is_not_a_fixed_work_throughput_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tests.unit.test_tui_responsiveness as module

    work_counts = []
    for cpu_cost_per_clock_read in (0.010, 0.025):
        cpu = 0.0
        work = 0

        def cpu_clock() -> float:
            nonlocal cpu
            cpu += cpu_cost_per_clock_read
            return cpu

        def work_unit(_values: range) -> int:
            nonlocal work
            work += 1
            return 0

        # A model of slower fixed work, not host stress: both clocks advance
        # together (no descheduling), while fewer work units fit the CPU bucket.
        monkeypatch.setattr(
            module, "time", SimpleNamespace(thread_time=cpu_clock, perf_counter=lambda: cpu)
        )
        monkeypatch.setattr(module, "sum", work_unit, raising=False)
        assert StallRecorder().calibrate_cpu() == pytest.approx(1)
        work_counts.append(work)
    assert work_counts[0] > work_counts[1]


def test_passing_measurement_evidence_is_junit_serializable(
    record_property: Callable[[str, object], None],
) -> None:
    loop = asyncio.new_event_loop()
    try:
        with CallbackSpans(loop) as spans:
            run_handle(loop, lambda: None)
            spans.observe_window(12.5, 18)
            recorder = StallRecorder(callbacks=spans)
            recorder.cpu_stalls = [12.5]
            recorder.wall_stalls = [18]
            recorder.noise_floor_ms = 7
            recorder.cpu_calibration_ratio = 1.06
            recorder.assert_no_stall_loaded(noise_floor_ms=7)
            encoded = json.dumps(recorder.measurement_summary())
            record_property("s2_loop_liveness_control", encoded)
        result = json.loads(encoded)
        assert result["worst_cpu_ms"] == 12.5
        assert result["worst_wall_ms"] == 18
        assert result["noise_floor_ms"] == 7
        assert result["wall_cpu_descheduling_ratio"] == 1.06
        assert result["callbacks"]["worst_cpu_windows"][0]["callbacks"]["count"] == 1
    finally:
        loop.close()
