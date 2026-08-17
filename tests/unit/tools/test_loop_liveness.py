"""Event-loop liveness under concurrent tool output: the anti-freeze suite.

The TUI renders on the SAME asyncio loop the tool coroutines ride, so any
synchronous stretch inside a tool — a multi-megabyte decode, a PIL image
re-encode, a spill write — is a stretch the frame does not animate. Reported
as "the TUI intermittently freezes on multiple concurrent tool calls": a
batch of reads finishing together put several hundred milliseconds of
blocking work back-to-back on the loop and the whole screen stood still.

The fixes moved those stretches into ``asyncio.to_thread`` (see
``execute_read``/``execute_write``/``execute_edit``/``execute_bash``/
``execute_grep``). These tests hold that property with a heartbeat: a task
that wakes every 20 ms and records the gap between wakes. While the tools
run, a loop-bound stretch shows up as one gap the size of the stretch; a
threaded one leaves the cadence untouched.

The bounds sit between the two worlds with margin on both sides: the
heartbeat's own cadence plus scheduler noise is tens of milliseconds, while
the workloads here block for 200 ms+ when run on the loop (measured: 214 ms
for the image fixture's decode+re-encode alone). A slower machine widens
only the loop-bound side, so the assertions stay decisive.
"""

from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import pytest
from PIL import Image

from local_operator.tools.builtin import execute_bash, execute_read

# Cadence of the liveness probe, and the bound a single wake may exceed it by.
HEARTBEAT_S = 0.02
#: The largest acceptable gap between wakes while tools are running. Chosen
#: against the measured loop-bound cost of the fixtures (200 ms+ each) with
#: an order of magnitude of headroom over the cadence itself.
MAX_GAP_S = 0.12


async def _heartbeat(stop: asyncio.Event, gaps: list[float]) -> None:
    """Record inter-wake gaps until told to stop; the probe itself is trivial."""
    last = time.monotonic()
    while not stop.is_set():
        await asyncio.sleep(HEARTBEAT_S)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


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
async def test_concurrent_image_reads_keep_the_loop_live(tmp_path: Path) -> None:
    """Three simultaneous image reads must not stop the frame from animating."""
    png = tmp_path / "noise.png"
    _noisy_png(png)

    stop = asyncio.Event()
    gaps: list[float] = []
    probe = asyncio.create_task(_heartbeat(stop, gaps))
    try:
        results = await asyncio.gather(
            *(execute_read(f"read-{i}", {"path": str(png)}, None, None, None) for i in range(3))
        )
    finally:
        stop.set()
        await probe

    assert not any(r.is_error for r in results), "reads failed; liveness proved nothing"
    assert gaps, "heartbeat never woke"
    assert max(gaps) < MAX_GAP_S, (
        f"event loop blocked for {max(gaps):.3f}s while tools ran — "
        "a synchronous stretch is back on the render loop"
    )


@pytest.mark.asyncio
async def test_oversized_bash_output_keeps_the_loop_live(tmp_path: Path) -> None:
    """A command that prints megabytes settles without freezing the frame.

    The oversized tail (join, decode, spill write, elide) is the part that
    moved into a thread; the fixture size is what makes that tail expensive
    enough to see from the heartbeat.
    """
    big = tmp_path / "big.txt"
    big.write_text("x" * (20 * 1024 * 1024) + "\n")

    stop = asyncio.Event()
    gaps: list[float] = []
    probe = asyncio.create_task(_heartbeat(stop, gaps))
    try:
        result = await execute_bash(
            "bash-1",
            {"command": f"cat '{big}'", "timeout": 60},
            None,
            None,
            None,
        )
    finally:
        stop.set()
        await probe

    assert not result.is_error, result.text
    assert gaps, "heartbeat never woke"
    assert max(gaps) < MAX_GAP_S, (
        f"event loop blocked for {max(gaps):.3f}s while bash settled — "
        "the oversized-output tail is back on the render loop"
    )
