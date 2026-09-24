"""Capture the TUI with a BUSY follower roster, driven through the canonical stream.

Run from the worktree root (or point ``--repo`` at another checkout for a before
frame; the script's own imports follow ``--repo``):

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/busy_roster_shot.py \\
        OUT.svg [COLSxROWS] [--repo CHECKOUT] [--expanded]

WHAT IT IS FOR. The roster-apply path on a viewer (``AttachedSession`` ->
``FrontendStateStore.apply_update`` -> ``OperatorApp._on_frontend_update`` ->
band and subagent panel) was re-cut for cost, with the explicit requirement that
NOTHING VISIBLE changes. This frame is the evidence: a real ``OperatorApp`` with
its production stylesheet, attached to a production ``AttachedSession`` whose
store is fed real ``FrontendUpdate`` deltas -- a 252-row roster (12 running
lanes with progress text, 240 settled children), one of them then settling and
several changing their progress -- exactly the stream a loaded parent sends. A
before/after pair taken from two checkouts must differ in nothing.

Pinned so a pair differs only where the code does: ``time.time`` (the rows'
elapsed clocks and the band's duration), the spinner frame, the update probe,
and the session ids. The ``.geometry.json`` beside the SVG carries every widget
box; the script also prints the dock's numbers.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any


def _repo_from_argv() -> Path:
    for index, arg in enumerate(sys.argv):
        if arg == "--repo" and index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1]).resolve()
    return Path(__file__).resolve().parent.parent


REPO = _repo_from_argv()
# The CAPTURE helpers come from this script's own tree (they may not exist in
# an older checkout); the APP comes from --repo, inserted first so it wins.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()
sys.path.insert(0, str(REPO))

NOW = 1_790_000_000.0
time.time = lambda: NOW  # type: ignore[assignment]

import local_operator  # noqa: E402
import local_operator.update as _update  # noqa: E402


class _NotBehind:
    """The update probe pinned to "not behind" (see ``dock_band_shot.py``)."""

    behind = False
    latest: str | None = None


_update.check_latest = lambda *a, **k: _NotBehind()  # type: ignore[assignment]

from local_operator.session.attached import AttachedSession  # noqa: E402
from local_operator.session.frontend_state import (  # noqa: E402
    FrontendModelSpec,
    FrontendSessionState,
    FrontendUpdate,
)
from local_operator.tui.app import OperatorApp  # noqa: E402

SESSION = "shotroster001"


def _lane(i: int, progress: str) -> dict[str, Any]:
    return {
        "id": f"lane{i:02d}",
        "type": "task",
        "status": "running",
        "label": f"lane-{i}",
        "agent_role": "coder",
        "start_time": NOW - 60 * (i + 1),
        "latest_details": {"progress": progress},
        "session_dir": f"/nowhere/lane{i:02d}",
    }


def _settled(i: int) -> dict[str, Any]:
    return {
        "id": f"old{i:05d}",
        "type": "task",
        "status": "completed",
        "label": f"settled child {i}",
        "agent_role": "reviewer",
        "start_time": NOW - 7200,
        "settled_at": NOW - 3600,
        "result_text": "ok " * 160,
        "restored": True,
        "session_dir": f"/nowhere/old{i:05d}",
    }


def _roster(step: int) -> list[dict[str, Any]]:
    lanes = [_lane(i, f"reading files step {step + i}") for i in range(12)]
    if step >= 3:
        lanes[4] = {
            **lanes[4],
            "status": "failed",
            "error_text": "provider error: overloaded",
            "settled_at": NOW - 5,
        }
    return lanes + [_settled(i) for i in range(240)]


async def main(out: str, size: tuple[int, int], expanded: bool) -> None:
    async def never() -> Any:
        raise RuntimeError("no takeover in a capture")

    viewer = AttachedSession(
        config_dir=Path(local_operator.__file__).parent, session_id=SESSION, takeover_factory=never
    )
    viewer._install_frontend(
        FrontendSessionState(
            session_id=SESSION,
            epoch="shot",
            cwd="/work/project",
            selected_model=FrontendModelSpec(
                provider="anthropic", model_id="claude-sonnet-4-5", context_window=200_000
            ),
        )
    )

    def push(**changes: Any) -> None:
        store = viewer._frontend_store
        assert store is not None
        update = FrontendUpdate(epoch="shot", sequence=store._state.sequence + 1, changes=changes)
        viewer._on_frontend_update(update.model_dump(mode="json"))

    async def factory() -> Any:
        return viewer

    app = OperatorApp(factory)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        for step in range(6):
            push(jobs=_roster(step), context_tokens=40_000 + step * 1000)
            await pilot.pause()
            await asyncio.sleep(0.3)
        panel = app._subagent_panel
        if expanded and panel is not None:
            panel.toggle_expanded()
        # Settle: let every spaced paint, band pass and panel tick land.
        await asyncio.sleep(2.5)
        await pilot.pause()
        # The two TIME-driven cells, pinned last so a pair differs only where
        # the code does: the splash's rotating tip (``_tip_tick`` on a timer)
        # and the dock's spinner frame (``_tick`` at 12.5 fps).
        welcome = app._welcome
        if welcome is not None:
            welcome._stop_tip_timer()
            welcome._tip_index = 0
            welcome.refresh_info()
        if panel is not None:
            panel._stop_spinner()
            panel._spinner_index = 0
            panel._paint_all(reread_stats=False)
        await pilot.pause()
        save_capture(app, out)
        band = app.query_one("#band")
        print(
            json.dumps(
                {
                    "tree": local_operator.__file__,
                    "band_region": list(band.region),
                    "subagent_panel": list(panel.region) if panel is not None else None,
                    "panel_rows": panel.predicted_rows() if panel is not None else None,
                    "summary": panel.summary_text() if panel is not None else None,
                    "status_segments": getattr(app._status, "_subagents", None),
                }
            )
        )


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--repo" in sys.argv:
        repo_value = sys.argv[sys.argv.index("--repo") + 1]
        args = [a for a in args if a != repo_value]
    target = args[0]
    cols, rows = (int(x) for x in (args[1] if len(args) > 1 else "120x40").split("x"))
    asyncio.run(main(target, (cols, rows), "--expanded" in sys.argv))
