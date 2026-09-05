"""Capture missing page families in the production CSS host with synthetic data.

Usage: python scripts/pages_shot.py OUT.svg PAGE [COLSxROWS] [THEME]
These are rendering fixtures, not claims to have called live providers. Empty
sidebars intentionally collapse: forcing them visible would misrepresent lop.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.analytics.store import UsageAggregate  # noqa: E402
from local_operator.resume import SessionRow  # noqa: E402
from local_operator.tui import theme  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.session_picker import SessionPickerScreen  # noqa: E402
from local_operator.tui.widgets.todo_panel import TodoPanel  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock  # noqa: E402
from tests.unit.tui.test_analytics_panel import _agg  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_subagent_view import TRAJECTORY, _job_with  # noqa: E402
from tests.unit.tui.test_usage_panel import _percent, _report  # noqa: E402

PAGES = [
    "welcome",
    "specimen",
    "transcript",
    "transcript-loading",
    "transcript-error",
    "transcript-overflow",
    "composer-picker",
    "composer-multiline",
    "copy-empty",
    "resume-empty",
    "resume-populated",
    "analytics-empty",
    "analytics-populated",
    *[f"usage-{state}" for state in ("empty", "loading", "error", "populated")],
    *[f"aside-{state}" for state in ("empty", "loading", "error", "populated")],
    *[f"todo-{state}" for state in ("empty", "populated", "overflow")],
    *[f"wake-{state}" for state in ("empty", "populated", "overflow")],
    *[f"subagent-{state}" for state in ("empty", "populated", "running")],
]


def grid(value: str) -> tuple[int, int]:
    try:
        columns, rows = (int(n) for n in value.split("x"))
        if not (20 <= columns <= 400 and 10 <= rows <= 150):
            raise ValueError
        return columns, rows
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "grid must be COLSxROWS within 20..400 by 10..150"
        ) from exc


async def capture(path: Path, page: str, size: tuple[int, int], palette: str) -> None:
    session: Any = FakeSession()
    job: Any = None
    if page.startswith("subagent"):
        from tests.unit.tui.test_band_panels import _fake_jobs

        job = _job_with(
            [] if page.endswith("empty") else TRAJECTORY,
            status="running" if page.endswith("running") else "completed",
        )
        session.jobs = _fake_jobs(job)
    # Stable cwd is fixture data, not a rewrite of the user's status-band layout.
    os.environ["HOME"] = str(Path(os.environ["HOME"]).resolve())
    os.chdir(os.environ["HOME"])
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        for _ in range(100):
            await pilot.pause()
            if app._session is not None:
                break
        if app._session is None:
            raise RuntimeError("capture session did not finish booting")
        app._apply_theme(palette)
        # Session assignment precedes the welcome's polling tick. Refresh via
        # its public seam so a fast pilot cannot capture a stale connecting label.
        assert app._welcome is not None
        app._welcome.refresh_info()
        await pilot.pause()
        if page.startswith("resume"):
            rows = (
                []
                if page.endswith("empty")
                else [
                    SessionRow(
                        id=f"session-{i}",
                        name=f"Investigate rendering sample {i}",
                        mtime=1000 - i * 60,
                    )
                    for i in range(12)
                ]
            )
            app.push_screen(SessionPickerScreen(rows, now=1100))
        elif page.startswith("analytics"):
            app.push_screen(AnalyticsScreen(UsageAggregate() if page.endswith("empty") else _agg()))
        elif page.startswith("usage"):
            panel = app._usage_panel()
            assert panel is not None
            panel.start_fetch()
            if page.endswith("empty"):
                panel.show_reports([])
            elif page.endswith("error"):
                panel.show_error("Synthetic provider unavailable; retry is safe")
            elif page.endswith("populated"):
                panel.show_reports([_report(_percent("weekly", "Weekly", 36))])
            panel.focus()
        elif page.startswith("aside"):
            panel = app._open_aside()
            assert panel is not None
            if not page.endswith("empty"):
                generation = panel.ask("Why are these cells spaced this way?")
                if page.endswith("populated"):
                    panel.settle_answer(
                        generation, "The **cell grid** controls layout. Font pixels are separate."
                    )
                elif page.endswith("error"):
                    panel.fail_answer(generation, "Synthetic provider unavailable")
        elif page.startswith("todo"):
            from local_operator.session.frontend_state import (
                TodoItemState,
                TodoPhaseState,
            )

            count = 0 if page.endswith("empty") else (24 if page.endswith("overflow") else 3)
            session.frontend_state = SimpleNamespace(
                todos=(
                    [
                        TodoPhaseState(
                            name="Validation",
                            items=[
                                TodoItemState(text=f"Inspect sample {i}", status="pending")
                                for i in range(count)
                            ],
                        )
                    ]
                    if count
                    else []
                )
            )
            app._refresh_band()
            if page.endswith("overflow"):
                app.query_one(TodoPanel).toggle_expanded()
                app._refresh_band()
        elif page.startswith("wake"):
            count = 0 if page.endswith("empty") else (20 if page.endswith("overflow") else 2)
            session.wake_scheduler = SimpleNamespace(
                schedules=[
                    SimpleNamespace(
                        id=f"wake-{i}",
                        next_due_at=1800000000000,
                        every_ms=60000,
                        message=f"Check the capture artifacts {i}",
                    )
                    for i in range(count)
                ]
            )
            app._refresh_band()
        elif page.startswith("subagent"):
            assert job is not None
            app._open_subagent_view(job.id)
        elif page == "copy-empty":
            app._run_slash_command("/copy")
        elif page == "composer-picker":
            await pilot.press("slash", "m", "o")
        elif page == "composer-multiline":
            app.query_one(Editor).text = (
                "Inspect the gallery\nand record the measurements\nwithout resizing the app."
            )
        elif page != "welcome":
            app._append_block(UserBlock("Inspect terminal cell fidelity"))
            block = AssistantBlock()
            if page == "specimen":
                text = (
                    "```text\n0123456789 ABCDEFGHIJ\nilll WWWW gypq j_\n"
                    "┌────────┬────────┐\n│ cells  │ 界界   │\n└────────┴────────┘\n"
                    "café café → ✓ × …\n👩‍💻X 👨‍👩‍👧‍👦X 界́X ❤️X\n```"
                )
            else:
                text = (
                    "**Rendered transcript** with `inline code`.\n\n```python\n"
                    "print('fixed cell advances')\n```\n\n1. Capture native pixels\n"
                    "2. Inspect layout and typography"
                )
            block.update_text(text)
            if page != "transcript-loading":
                block.finalize_text()
            app._append_block(block)
            if page == "transcript-error":
                app._append_block(NoticeBlock("Synthetic request failed; retry is safe", "warning"))
            if page == "transcript-overflow":
                for i in range(50):
                    app._append_block(NoticeBlock(f"Retained event {i}", "info"))
        await pilot.pause()
        save_capture(app, path.with_name(path.stem + ".first.svg"))
        await pilot.pause()
        await pilot.pause()
        save_capture(app, path)
        print(f"{page}: {size[0]}x{size[1]} theme={palette} session=ready -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("page", choices=PAGES)
    parser.add_argument("size", nargs="?", default="100x30", type=grid)
    parser.add_argument("theme", nargs="?", default="dark", choices=theme.available_themes())
    args = parser.parse_args()
    # Resolve before the fixture chdir so caller-relative output stays put.
    asyncio.run(capture(args.output.resolve(), args.page, args.size, args.theme))


if __name__ == "__main__":
    main()
