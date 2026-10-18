#!/usr/bin/env python3
"""Real-path ``/import`` smoke: a copied omp session through the TUI.

Runs the REAL ``OperatorApp`` in an isolated HOME, copies one real omp session
from the operator's store into it, and drives the page with keys exactly as a
user would: open → choose omp → select → preview → import → result. Then
re-runs the same selection and requires the core's ``skipped_already_imported``
outcome, while hashing the source before/after and checking no sidecars were
created. Nothing from the session body is printed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture  # noqa: E402

ORIGINAL_HOME = Path(os.environ["HOME"])
isolate_capture()

from local_operator.harness_import.jobs import JOBS  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.session.transcript import Transcript  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.harness_import_view import (  # noqa: E402
    HarnessImportView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_real_session() -> Path:
    root = ORIGINAL_HOME / ".omp" / "agent" / "sessions"
    candidates = sorted(root.rglob("*.jsonl"), key=lambda path: path.stat().st_size)
    if not candidates:
        raise SystemExit("no real omp session available")
    # Large enough to contain a real conversation, small enough to keep the
    # smoke fast; never the operator's newest/live transcript.
    return next((path for path in candidates if path.stat().st_size > 10_000), candidates[-1])


async def settle(pilot, predicate, turns: int = 400) -> None:
    for _ in range(turns):
        if predicate():
            return
        await pilot.pause()
    raise RuntimeError("state did not settle")


async def open_omp(pilot) -> HarnessImportView:
    pilot.app._run_slash_command("/import")
    await pilot.pause()
    view = pilot.app.query_one(HarnessImportView)
    await settle(pilot, lambda: view._state == "sources")
    for _ in range(len(view._sources)):
        if view._sources[view._source_index].source == "omp":
            break
        await pilot.press("down")
    await pilot.press("enter")
    await settle(pilot, lambda: view._state == "sessions")
    return view


async def one_import(pilot) -> tuple[str, str]:
    view = await open_omp(pilot)
    await pilot.press("space")
    await pilot.press("enter")
    await settle(pilot, lambda: view._state == "preview")
    await pilot.press("enter")
    await settle(pilot, lambda: view._state in ("progress", "result"))
    await settle(pilot, lambda: view._state == "result")
    assert view._job is not None and view._job.result is not None
    result = view._job.result
    target = result.sessions[0].target_session_id if result.sessions else ""
    return result.state, target


async def main() -> None:
    real = pick_real_session()
    encoded = "--copied-real--"
    source = Path.home() / ".omp" / "agent" / "sessions" / encoded / real.name
    source.parent.mkdir(parents=True)
    shutil.copy2(real, source)
    before = sha(source)
    before_sidecars = sorted(path.name for path in source.parent.glob(source.name + "-*"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        first_state, target = await one_import(pilot)
        assert first_state == "done", first_state
        directory = config_dir() / "sessions" / target
        assert directory.is_dir(), directory
        history = Transcript(directory).build_llm_history()
        assert history, "imported target has no replayable history"
        # Back to the source list, then the same selection again.
        await pilot.press("enter")
        await settle(pilot, lambda: app.query_one(HarnessImportView)._state == "sources")
        second_state, second_target = await one_import(pilot)
        assert second_target == target, (second_target, target)
        view = app.query_one(HarnessImportView)
        assert view._job is not None and view._job.result is not None
        outcomes = [row.outcome for row in view._job.result.sessions]
        assert outcomes == ["skipped_already_imported"], outcomes
        await pilot.press("escape")
        await pilot.pause()
        import_mode_closed = not app.screen.has_class("settings")

    after = sha(source)
    after_sidecars = sorted(path.name for path in source.parent.glob(source.name + "-*"))
    print(
        json.dumps(
            {
                "state": second_state,
                "target": target,
                "history_messages": len(history),
                "source_hash_unchanged": before == after,
                "sidecars_before": before_sidecars,
                "sidecars_after": after_sidecars,
                "rerun_outcome": outcomes,
                "import_mode_closed": import_mode_closed,
                "job_kept": JOBS.latest(session_key="tui") is not None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
