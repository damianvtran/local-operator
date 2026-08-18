"""Event-loop liveness under concurrent tool output: the anti-freeze suite.

The TUI renders on the SAME asyncio loop the tool coroutines ride, so any
synchronous stretch inside a tool — a multi-megabyte decode, a PIL image
re-encode, a spill write — is a stretch the frame does not animate. Reported
as "the TUI intermittently freezes on multiple concurrent tool calls": a
batch of reads finishing together put several hundred milliseconds of
blocking work back-to-back on the loop and the whole screen stood still
(measured: 649 ms for three concurrent image reads).

The fixes moved those stretches into ``asyncio.to_thread`` (see
``execute_read``/``execute_write``/``execute_edit``/``execute_bash``/
``execute_grep``). These tests hold that property two ways at once, from a
single run of the real workload:

* structurally — the function that moved off the loop is observed running on
  a thread that is not the loop's. That is the contract stated exactly, and
  no amount of machine load can perturb it.
* temporally — a heartbeat task wakes every 20 ms and records how much CPU
  time THE LOOP THREAD accumulated since its last wake. A synchronous
  stretch on the loop is, by definition, the loop thread executing without
  yielding, so it shows up as one sample the size of the stretch.

The temporal half deliberately measures loop-thread CPU rather than the
wall-clock gap between wakes, because the wall-clock gap does not measure
what it looks like it measures. It goes wide whenever the loop thread is not
SCHEDULED, which is a different thing from the loop thread being BUSY — and
the leading cause of the loop thread not being scheduled is the fix itself:
three CPU-bound Pillow threads in-process delay the main thread's next wake
through GIL handoff and allocator contention. Measured on an idle 14-core
M3 Max with the decode correctly threaded, the widest wall gap ranged from
156 ms to 375 ms run to run, and a control of three 220 ms pure-CPU spins in
``asyncio.to_thread`` — nothing at all on the loop — already produced gaps
of 86-130 ms. A wall-clock bound tight enough to catch a re-introduced stall
therefore fails on the fixed tree roughly one run in two; the old 0.12 s one
did, which is what sent this file red on load and often without it.

Loop-thread CPU separates the two worlds by two orders of magnitude and does
not move with load, because a busy machine steals wall time from the loop
thread without handing it CPU time. Measured over 25 repetitions each: the
widest sample is 4.3 ms for the image workload and 12.7 ms for the bash one,
against 512-725 ms with the image decode put back on the loop, and 2.4-4.1 ms
for the image workload under eight competing CPU hogs — i.e. load moves this
statistic by a millisecond where it moved the wall gap by 300.

Which half is decisive differs by the SHAPE of the stretch, which is why both
are here rather than either alone.

* CPU on the loop — the shape PR #129 fixed. The temporal half owns it: the
  decode is pure CPU, so putting it back on the loop lands 512-725 ms on one
  sample (a single image's worth, 147 ms, is still caught). It owns it for the
  whole path too, including stretches nobody thought to spy on.
* A BLOCKING CALL on the loop — a synchronous file read, the spill's write.
  Those cost the loop thread wall time and no CPU at all, so no CPU bound can
  see them; measured, the file snapshot put back on the loop moves the widest
  sample only to 6.7 ms. The structural half owns that shape, which is why it
  watches every hop on the path and not just the expensive one.

:data:`MAX_LOOP_CPU_S` is set from those measurements: 6x clear of the widest
legitimate sample, so a machine six times slower than this one still passes,
and 6x under the cheapest re-introduced stretch actually observed. A slower
machine only widens the loop-bound side.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

import local_operator.tools.builtin as builtin
from local_operator.tools.builtin import execute_bash, execute_read

# Cadence of the liveness probe.
HEARTBEAT_S = 0.02
#: The most CPU time the loop thread may accumulate between two heartbeat
#: wakes. Servicing the tools' completion callbacks costs it up to 12.7 ms per
#: sample; running the image workload on the loop costs it 512-725 ms.
MAX_LOOP_CPU_S = 0.08


class LoopCpuProbe:
    """Heartbeat that records the loop thread's CPU time between its wakes.

    ``time.thread_time`` is per-thread and excludes time asleep or waiting on
    the GIL, so a sample is large only when the loop thread genuinely ran
    without yielding — never merely because the machine was busy.
    """

    def __init__(self) -> None:
        self.samples: list[float] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        last = time.thread_time()
        while not self._stop.is_set():
            await asyncio.sleep(HEARTBEAT_S)
            now = time.thread_time()
            self.samples.append(now - last)
            last = now

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stop.set()
        assert self._task is not None
        await self._task

    @property
    def worst(self) -> float:
        return max(self.samples)


class OffLoopSpy:
    """Records the threads a to_thread'd function was actually called on.

    A spy, not a stub: the real function still runs, so the same workload
    feeds the structural and the temporal assertion at once.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        self.threads: dict[str, list[int]] = {name: [] for name in names}
        # Constructed from inside the test coroutine, so this IS the loop's
        # thread — asyncio offers no other way to name it.
        self.loop_thread = threading.get_ident()
        for name in names:
            monkeypatch.setattr(builtin, name, self._wrap(name, getattr(builtin, name)))

    def _wrap(self, name: str, real: Callable[..., Any]) -> Callable[..., Any]:
        def spy(*args: Any, **kwargs: Any) -> Any:
            self.threads[name].append(threading.get_ident())
            return real(*args, **kwargs)

        return spy

    def assert_all_off_loop(self) -> None:
        for name, idents in self.threads.items():
            assert idents, (
                f"{name} never ran — the work it carries no longer goes through it, "
                "so this test is no longer watching anything"
            )
            on_loop = [i for i in idents if i == self.loop_thread]
            assert not on_loop, (
                f"{name} ran on the event-loop thread {len(on_loop)} of "
                f"{len(idents)} times — the asyncio.to_thread hop is gone and the "
                "work is back on the render loop"
            )


def _noisy_png(path: Path) -> None:
    """A photographic-noise PNG that costs ~200 ms to decode and re-encode.

    Built with ``effect_noise`` + ``quantize`` because both run in C: the
    pixel loop a pure-Python builder needs costs more than the assertion is
    worth. The noise is what makes the PNG expensive to decode — a flat
    colour image of the same dimensions round-trips in milliseconds.
    """
    image = Image.effect_noise((2600, 2600), 42).quantize(16).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=6)
    path.write_bytes(buffer.getvalue())


@pytest.mark.asyncio
async def test_concurrent_image_reads_keep_the_loop_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three simultaneous image reads must not stop the frame from animating."""
    png = tmp_path / "noise.png"
    _noisy_png(png)

    # Both hops an image read makes: the blocking file snapshot and the
    # decode+re-encode. A blocking syscall put back on the loop costs the loop
    # thread wall time but no CPU, so the structural half is the only one that
    # can see it, and it must therefore watch the whole path — not just the
    # expensive end of it.
    spy = OffLoopSpy(monkeypatch, "_read_file_snapshot", "bound_image_for_model")
    probe = LoopCpuProbe()
    probe.start()
    try:
        results = await asyncio.gather(
            *(execute_read(f"read-{i}", {"path": str(png)}, None, None, None) for i in range(3))
        )
    finally:
        await probe.stop()

    assert not any(r.is_error for r in results), "reads failed; liveness proved nothing"
    spy.assert_all_off_loop()
    assert probe.samples, "heartbeat never woke"
    assert probe.worst < MAX_LOOP_CPU_S, (
        f"the loop thread burned {probe.worst:.3f}s of CPU without yielding while "
        "the reads ran — a synchronous stretch is back on the render loop"
    )


@pytest.mark.asyncio
async def test_oversized_bash_output_keeps_the_loop_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command that prints megabytes settles without freezing the frame.

    The oversized tail (join, decode, spill write, elide) is the part that
    moved into a thread, and the fixture size is what makes it expensive: 20 MB
    costs 80 ms of CPU plus a spill write. Most of that is the write, so on
    this path the structural assertion is the one with teeth — see the module
    docstring on which half owns which shape.
    """
    big = tmp_path / "big.txt"
    big.write_text("x" * (20 * 1024 * 1024) + "\n")

    spy = OffLoopSpy(monkeypatch, "_decode_chunks", "_bash_oversized_streams")
    probe = LoopCpuProbe()
    probe.start()
    try:
        result = await execute_bash(
            "bash-1",
            {"command": f"cat '{big}'", "timeout": 60},
            None,
            None,
            None,
        )
    finally:
        await probe.stop()

    assert not result.is_error, result.text
    spy.assert_all_off_loop()
    assert probe.samples, "heartbeat never woke"
    assert probe.worst < MAX_LOOP_CPU_S, (
        f"the loop thread burned {probe.worst:.3f}s of CPU without yielding while "
        "bash settled — the oversized-output tail is back on the render loop"
    )
