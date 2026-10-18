#!/usr/bin/env python3
"""Capture the real ``/import`` TUI page for one state and terminal size.

The evidence is NOT committed: pass an output path under the operator's
scratchpad. The script isolates HOME/config before importing the app, seeds a
synthetic omp fixture (or nothing for the empty state), drives the REAL
``OperatorApp`` with ``run_test`` and exports through ``save_capture``.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
      scripts/harness_import_shot.py STATE OUT.svg 100x30 [--before]

STATES: empty, loading, sources, sessions, preview, progress, cancel, error, result
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.harness_import.service import HarnessImportService  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.harness_import_view import (  # noqa: E402
    HarnessImportView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "unit" / "harness_import" / "fixtures"


def seed_omp() -> None:
    root = Path.home() / ".omp" / "agent"
    root.mkdir(parents=True)
    shutil.copytree(FIXTURES / "omp" / "sessions", root / "sessions")


def seed_large_omp() -> None:
    """A session large enough that a cancel lands before any target write.

    The ordinary fixture imports in a few milliseconds, so a scripted cancel
    would race completion and capture ``done``. This is synthetic text only:
    the same header the fixture uses, then repeated unique turns.
    """
    source = FIXTURES / "omp" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    root = Path.home() / ".omp" / "agent" / "sessions" / "--tmp--"
    root.mkdir(parents=True)
    with (root / "omp-synthetic.jsonl").open("w", encoding="utf-8") as handle:
        for record in records[:3]:
            handle.write(json.dumps(record) + "\n")
        parent = "m1"
        for index in range(200_000):
            handle.write(
                json.dumps(
                    {
                        "type": "message",
                        "id": f"u{index}",
                        "parentId": parent,
                        "timestamp": "2026-01-03T00:00:02.000Z",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": f"question {index}"}],
                        },
                    }
                )
                + "\n"
            )
            handle.write(
                json.dumps(
                    {
                        "type": "message",
                        "id": f"a{index}",
                        "parentId": f"u{index}",
                        "timestamp": "2026-01-03T00:00:03.000Z",
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": f"answer {index}"}],
                        },
                    }
                )
                + "\n"
            )
            parent = f"a{index}"


def seed_metadata_codex() -> None:
    """One Codex thread whose rollout body is absent: the index-only journey."""
    root = Path.home() / ".codex"
    root.mkdir(parents=True)
    writer = sqlite3.connect(root / "state_5.sqlite")
    try:
        writer.execute(
            "CREATE TABLE threads (id TEXT, rollout_path TEXT, created_at INTEGER, "
            "updated_at INTEGER, cwd TEXT, title TEXT, archived INTEGER, "
            "cli_version TEXT, first_user_message TEXT, model TEXT, model_provider TEXT)"
        )
        writer.execute(
            "INSERT INTO threads VALUES ('index-only', '', 1, 2, '/tmp', "
            "'Index-only Codex thread', 0, '', '', '', '')"
        )
        writer.commit()
    finally:
        writer.close()


def seed_long_omp() -> None:
    """The fixture with a title at the service's own TITLE_MAX cap."""
    source = FIXTURES / "omp" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    records[0]["title"] = "A very long synthetic title " * 8
    records[1]["id"] = "omp-long"
    root = Path.home() / ".omp" / "agent" / "sessions" / "--long--"
    root.mkdir(parents=True)
    with (root / "omp-long.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def seed_many_omp(count: int = 12) -> None:
    """Synthetic sessions, for the list's scrollbar/truncation/End checks."""
    source = FIXTURES / "omp" / "sessions" / "--tmp--" / "omp-synthetic.jsonl"
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    root = Path.home() / ".omp" / "agent" / "sessions" / "--many--"
    root.mkdir(parents=True)
    for index in range(count):
        copy = [dict(record) for record in records]
        copy[0]["title"] = f"Synthetic omp session {index:02d}"
        copy[1]["id"] = f"omp-synthetic-{index:02d}"
        with (root / f"omp-synthetic-{index:02d}.jsonl").open("w", encoding="utf-8") as handle:
            for record in copy:
                handle.write(json.dumps(record) + "\n")


async def settle(pilot, predicate, turns: int = 300) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
    raise RuntimeError("state did not settle")


async def open_view(pilot) -> HarnessImportView:
    pilot.app._run_slash_command("/import")
    await pilot.pause()
    view = pilot.app.query_one(HarnessImportView)
    await settle(pilot, lambda: view._state != "loading" or view._sources)
    return view


async def open_omp(pilot) -> HarnessImportView:
    view = await open_view(pilot)
    await settle(pilot, lambda: view._state == "sources")
    for _ in range(len(view._sources)):
        if view._sources[view._source_index].source == "omp":
            break
        await pilot.press("down")
    await pilot.press("enter")
    await settle(pilot, lambda: view._state in ("sessions", "error"))
    return view


async def select_and_preview(pilot, view: HarnessImportView) -> None:
    await pilot.press("space")
    await pilot.press("enter")
    await settle(pilot, lambda: view._state in ("preview", "error"))


async def capture(pilot, state: str, out: Path) -> None:
    view = None
    if state == "empty":
        shutil.rmtree(Path.home() / ".omp", ignore_errors=True)
        shutil.rmtree(Path.home() / ".claude", ignore_errors=True)
        shutil.rmtree(Path.home() / ".codex", ignore_errors=True)
        view = await open_view(pilot)
    elif state == "absent":
        shutil.rmtree(Path.home() / ".omp", ignore_errors=True)
        shutil.rmtree(Path.home() / ".claude", ignore_errors=True)
        shutil.rmtree(Path.home() / ".codex", ignore_errors=True)
        view = await open_view(pilot)
        await pilot.press("enter")
        await settle(pilot, lambda: view._state == "empty")
    elif state == "loading":
        original = HarnessImportService.sources

        async def slow_sources(self):
            await asyncio.sleep(30)
            return await original(self)

        HarnessImportService.sources = slow_sources  # type: ignore[assignment]
        pilot.app._run_slash_command("/import")
        await pilot.pause()
        view = pilot.app.query_one(HarnessImportView)
        await settle(pilot, lambda: view._state == "loading")
    elif state == "error":
        pilot.app._run_slash_command("/import preview nope")
        await pilot.pause()
        view = pilot.app.query_one(HarnessImportView)
        await settle(pilot, lambda: view._state == "error")
    else:
        if state == "cancel":
            seed_large_omp()
        elif state == "many":
            seed_many_omp()
        elif state == "long":
            seed_long_omp()
        elif state == "long_end":
            seed_many_omp(41)
        elif state in ("metadata_preview", "metadata_result"):
            seed_metadata_codex()
        else:
            seed_omp()
        if state == "sources":
            view = await open_view(pilot)
        elif state in ("metadata_preview", "metadata_result"):
            view = await open_view(pilot)
            for _ in range(len(view._sources)):
                if view._sources[view._source_index].source == "codex":
                    break
                await pilot.press("down")
            await pilot.press("enter")
            await settle(pilot, lambda: view._state == "sessions")
            await pilot.press("space")
            await pilot.press("enter")
            await settle(pilot, lambda: view._state == "preview")
            if state == "metadata_result":
                await pilot.press("enter")
                await settle(pilot, lambda: view._state in ("progress", "result"))
                await settle(pilot, lambda: view._state == "result")
        else:
            view = await open_omp(pilot)
            if state == "checked":
                await pilot.press("space")
            if state == "long_end":
                view.action_jump(1)
            if state in ("preview", "progress", "cancel", "result"):
                if state == "progress":
                    original_import = HarnessImportService._import_one

                    def slow_import(self, *args, **kwargs):
                        time.sleep(1.2)
                        return original_import(self, *args, **kwargs)

                    HarnessImportService._import_one = slow_import  # type: ignore[assignment]
                await select_and_preview(pilot, view)
                await settle(pilot, lambda: view._state == "preview")
                if state == "preview":
                    pass
                else:
                    await pilot.press("enter")
                    if state == "progress":
                        await settle(pilot, lambda: view._state == "progress")
                    elif state == "cancel":
                        await settle(pilot, lambda: view._state == "progress")
                        await pilot.press("c")
                        await settle(pilot, lambda: view._state == "result")
                    else:
                        await settle(pilot, lambda: view._state == "result")
    assert view is not None
    if state == "sessions":
        assert view._state == "sessions"
    if state in ("many", "long", "long_end", "checked"):
        assert view._state == "sessions"
    if state == "metadata_preview":
        assert view._state == "preview"
    if state == "metadata_result":
        assert view._state == "result"
    if state == "sources":
        assert view._state == "sources"
    await pilot.pause()
    save_capture(pilot.app, out)
    job = getattr(view, "_job", None)
    print(
        json.dumps(
            {
                "state": state,
                "grid": list(pilot.app.size),
                "out": str(out),
                "job_state": getattr(getattr(job, "progress", None), "state", None),
                "job_messages": getattr(getattr(job, "result", None), "messages", None),
                "job_outcomes": [
                    row.outcome for row in getattr(getattr(job, "result", None), "sessions", [])
                ],
            }
        )
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state")
    parser.add_argument("out")
    parser.add_argument("size", nargs="?", default="100x30")
    parser.add_argument(
        "--before", action="store_true", help="base-commit frame: /import is absent"
    )
    parser.add_argument("--theme", default="dark", help="OperatorApp theme name")
    args = parser.parse_args()
    columns, rows = (int(value) for value in args.size.lower().split("x"))
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name=args.theme)
    async with app.run_test(size=(columns, rows)) as pilot:
        await pilot.pause()
        if args.before:
            seed_omp()
            app._run_slash_command("/import")
            await pilot.pause()
            save_capture(app, Path(args.out))
            print(json.dumps({"state": "before", "grid": [columns, rows], "out": args.out}))
            return
        await capture(pilot, args.state, Path(args.out))
    svg = Path(args.out)
    png = svg.with_suffix(".png")
    subprocess.run(["rsvg-convert", str(svg), "-o", str(png)], check=True)


if __name__ == "__main__":
    asyncio.run(main())
