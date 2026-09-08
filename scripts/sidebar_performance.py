"""Measure assembled sidebar rendering and the real off-loop catalog collector.

Examples (no live configuration, sessions, providers or sockets are used)::

    .venv/bin/python scripts/sidebar_performance.py --output /tmp/sidebar-after
    .venv/bin/python scripts/sidebar_performance.py --mode frames --output /tmp/frames
    .venv/bin/python scripts/sidebar_performance.py --mode catalog --output /tmp/catalog

For an interleaved before/after comparison, --source-root points at a clean
baseline worktree. The SAME harness drives both trees; source provenance is
recorded in every result. Matrix cells use synthetic catalog summaries and
suppress prewarm ONLY, separating render cost from session attach cost. Catalog
cells separately exercise real transcript/registry/attention reads in a worker.
Timings are evidence, not CI ceilings: regression tests assert work counts.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent.parent)
PARSER.add_argument("--output", type=Path, required=True)
PARSER.add_argument("--mode", choices=("matrix", "frames", "catalog"), default="matrix")
PARSER.add_argument("--samples", type=int, default=40)
ARGS = PARSER.parse_args()
if ARGS.samples < 1:
    PARSER.error("--samples must be positive")
ARGS.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ARGS.source_root.resolve()))
# Multiplexer variables are independent of HOME/config. Even headless pilots
# must not inherit identifiers that could rename the operator's real workspace.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        del os.environ[_key]

import scripts.probe_isolation as isolation  # noqa: E402

# isort: split
# Isolation must precede even the package root import.
import local_operator  # noqa: E402
from local_operator.harness.types import Message  # noqa: E402
from local_operator.resume import SessionRow  # noqa: E402
from local_operator.session.runtime.types import SessionRecord  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.session_catalog import CatalogEntry, load_catalog  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def render_cell(count: int, opened: bool, streaming: bool) -> dict[str, Any]:
    entries = [
        CatalogEntry(
            SessionRow(
                f"{i + 1:012x}",
                1788700000 - i * 60,
                f"Background session {i}: investigate timeouts",
                live_state="busy" if streaming else "idle",
            )
        )
        for i in range(count)
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    with (
        patch("local_operator.tui.session_catalog.load_catalog", return_value=entries),
        patch.object(app, "_prewarm_sidebar"),
    ):
        async with app.run_test(size=(150, 40)) as pilot:
            await pilot.pause()
            app._append_block(UserBlock("Investigate responsiveness with background sessions."))
            block = AssistantBlock()
            block.update_text("Profiling sidebar updates.")
            app._append_block(block)
            app._set_sidebar_open(opened)
            await pilot.pause()
            sidebar = app._session_sidebar
            name = f"{count}-{int(opened)}-{int(streaming)}"
            # Stable captures do not suppress animation during measurement.
            if sidebar._timer is not None:
                sidebar._timer.pause()
            sidebar._frame = 0
            sidebar.refresh()
            await pilot.pause()
            save_capture(app, str(ARGS.output / f"{name}-before.svg"))
            if ARGS.mode == "frames":
                # Fixed work, not fixed time: every tick must reach a painted
                # frame before the next. This removes profiler/scheduler noise.
                if app._sidebar_timer is not None:
                    app._sidebar_timer.pause()
                app._sidebar_refresh_generation += 1
            else:
                sidebar._sync_animation()
            profiler = cProfile.Profile()
            if ARGS.mode == "matrix":
                profiler.enable()
            with patch.object(sidebar, "render", wraps=sidebar.render) as renders:
                cpu_start, wall_start = time.thread_time(), time.monotonic()
                for i in range(ARGS.samples):
                    if ARGS.mode == "frames":
                        if streaming:
                            sidebar._advance_spinner()
                        else:
                            sidebar.set_entries(entries)
                        await pilot.pause()
                    else:
                        if streaming:
                            block.update_text(
                                "Profiling sidebar updates. "
                                + "A live update remains visible. " * i
                            )
                        await asyncio.sleep(0.1)
                cpu = time.thread_time() - cpu_start
                wall = time.monotonic() - wall_start
                render_count = renders.call_count
            profiler.disable()
            if ARGS.mode == "matrix":
                profiler.dump_stats(str(ARGS.output / f"{name}.prof"))
            if ARGS.mode == "frames" and streaming:
                save_capture(app, str(ARGS.output / f"{name}-live.svg"))
                sidebar._advance_spinner()
                await pilot.pause()
                save_capture(app, str(ARGS.output / f"{name}-next.svg"))
            if sidebar._timer is not None:
                sidebar._timer.pause()
            sidebar._frame = 0
            sidebar.refresh()
            await pilot.pause()
            save_capture(app, str(ARGS.output / f"{name}-after.svg"))
            result = {
                "mode": ARGS.mode,
                "sessions": count,
                "sidebar": opened,
                "streaming": streaming,
                "samples": ARGS.samples,
                "loop_cpu_s": cpu,
                "wall_s": wall,
                "full_sidebar_renders": render_count,
                "source": str(local_operator.__file__),
            }
            print(json.dumps(result), flush=True)
            return result


async def catalog_cell(count: int) -> dict[str, Any]:
    root = isolation.SANDBOX / f"catalog-{count}"
    run = root / "run/mobile"
    run.mkdir(parents=True)
    for i in range(count):
        session_id = f"{i + 1:012x}"
        transcript = Transcript(root / "sessions" / session_id)
        await transcript.append_messages(
            [
                Message.user(f"Session {i} turn {j}: " + "representative history " * 20)
                for j in range(1000)
            ]
        )
        # All files reference THIS probe's pid, so registry.scan executes real
        # liveness syscalls without probing unrelated processes. No control
        # server is created; port zero cannot attach to an operator session.
        record = SessionRecord(
            os.getpid(), "tui", session_id, f"Fixture {i}", str(root), "synthetic", 0, "fixture"
        )
        (run / f"{i}.json").write_text(json.dumps(record.to_json()))
    cpu_samples, wall_samples = [], []
    for _ in range(ARGS.samples):
        cpu, wall = time.thread_time(), time.monotonic()
        rows = await asyncio.to_thread(load_catalog, root)
        cpu_samples.append(time.thread_time() - cpu)
        wall_samples.append(time.monotonic() - wall)
        assert len(rows) == count
    result = {
        "mode": "catalog",
        "sessions": count,
        "messages_each": 1000,
        "samples": ARGS.samples,
        "median_wall_s": statistics.median(wall_samples),
        "max_wall_s": max(wall_samples),
        "median_loop_cpu_s": statistics.median(cpu_samples),
        "max_loop_cpu_s": max(cpu_samples),
        "source": str(local_operator.__file__),
    }
    print(json.dumps(result), flush=True)
    return result


async def main() -> None:
    results = []
    for count in (1, 10, 50):
        if ARGS.mode == "catalog":
            results.append(await catalog_cell(count))
        else:
            for streaming in (False, True):
                for opened in ((True,) if ARGS.mode == "frames" else (False, True)):
                    results.append(await render_cell(count, opened, streaming))
    (ARGS.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
