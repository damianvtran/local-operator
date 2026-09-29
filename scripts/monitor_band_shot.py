"""Capture the dock band's scheduled-work panel against the real OperatorApp.

The band is one slot the subagent, todo and wake panels share (U7), so a frame
of it has to come from the REAL app — a bare test host declares no CSS_PATH and
would show none of the band's stylesheet. This script boots ``OperatorApp``,
attaches duck-typed schedulers to the session, and repaints through the app's
own ``_refresh_band`` (the method the 1 Hz poll calls), then exports via
``scripts.visual_capture``.

Run from the worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/monitor_band_shot.py OUT.svg [COLSxROWS] [SHAPE] [ICONS]

``SHAPE``:
  - ``wakes``    — two wakes, no monitors (the pre-monitor band);
  - ``both``     — two wakes and two monitors, one of them disabled (the band
    the monitor work adds);
  - ``overflow`` — four wakes and four monitors, so both sections overflow and
    the marker lines are in frame.

``ICONS`` is ``nerd`` (default) or ``plain``; same gate and same reasoning as
``wake_shot.py``. The monitor stubs are duck-typed on purpose: this script must
run against a tree BEFORE the panel learns to read ``monitor_scheduler`` (the
before-frame) as well as after (the after-frames).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import (  # noqa: E402
    isolate_capture,
    save_capture,
    settle_status_line,
)

isolate_capture()


def _seed_env(mode: str) -> None:
    """Force the icon gate into ``mode`` BEFORE the app is imported.

    Same marker list and same reasoning as ``wake_shot.py``: the gate reads the
    environment at row-build time and this capture runs under an isolated HOME.
    """
    for var in (
        "GHOSTTY_RESOURCES_DIR",
        "GHOSTTY_BIN",
        "KITTY_WINDOW_ID",
        "WEZTERM_PANE",
        "WEZTERM_EXECUTABLE",
        "TERM_PROGRAM",
        "LOCAL_OPERATOR_NO_NERD_ICONS",
    ):
        os.environ.pop(var, None)
    if mode == "nerd":
        os.environ["GHOSTTY_BIN"] = "/usr/local/bin/ghostty"
    else:
        os.environ["TERM_PROGRAM"] = "Apple_Terminal"


_seed_env(sys.argv[4] if len(sys.argv) > 4 else "nerd")

from local_operator.harness.wake import WakeSchedule  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.wake_panel import WakePanel  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

NOW_MS = int(time.time() * 1000)


def _wake(wake_id: str, message: str, *, every_ms: int | None, due_in_s: int) -> WakeSchedule:
    return WakeSchedule(
        id=wake_id,
        message=message,
        next_due_at=NOW_MS + due_in_s * 1000,
        every_ms=every_ms,
        created_at=NOW_MS - 3_600_000,
    )


class _WakeScheduler:
    """The slice ``WakePanel.sync`` reads: ``schedules``."""

    def __init__(self, schedules: list[WakeSchedule]) -> None:
        self.schedules = tuple(schedules)


class _MonitorScheduler:
    """The slice ``WakePanel.sync`` reads: ``monitors`` + ``runtime(id)``.

    Duck-typed so the same script runs before the panel reads
    ``monitor_scheduler`` at all (the before-frame) and after.
    """

    def __init__(self, rows: list[tuple[SimpleNamespace, dict[str, object]]]) -> None:
        self._rows = rows

    @property
    def monitors(self) -> tuple[SimpleNamespace, ...]:
        return tuple(spec for spec, _ in self._rows)

    def runtime(self, monitor_id: str) -> SimpleNamespace:
        for spec, counters in self._rows:
            if spec.id == monitor_id:
                return SimpleNamespace(spec=spec, counters=dict(counters))
        return SimpleNamespace(spec=None, counters={})


def _monitor(
    monitor_id: str,
    name: str,
    *,
    every_ms: int = 60_000,
    due_in_s: int | None = 30,
    disabled: bool = False,
    disabled_reason: str = "",
    failures: int = 0,
) -> tuple[SimpleNamespace, dict[str, object]]:
    spec = SimpleNamespace(
        id=monitor_id,
        name=name,
        tool="bash",
        every_ms=every_ms,
        until_at=None,
        description="",
        created_at=NOW_MS - 3_600_000,
    )
    counters: dict[str, object] = {
        "next_due_at": None if due_in_s is None else NOW_MS + due_in_s * 1000,
        "last_check_at": NOW_MS - 30_000,
        "checks": 12,
        "deliveries": 1,
        "consecutive_failures": failures,
        "disabled": disabled,
        "disabled_reason": disabled_reason,
    }
    return spec, counters


def _shapes(
    shape: str,
) -> tuple[list[WakeSchedule], list[tuple[SimpleNamespace, dict[str, object]]]]:
    wakes = [
        _wake("w1", "check the backup", every_ms=3_600_000, due_in_s=1_800),
        _wake("w2", "re-run the nightly audit", every_ms=None, due_in_s=7_200),
    ]
    monitors = [
        _monitor("m1", "watch the deploy queue", every_ms=60_000, due_in_s=42),
        _monitor("m2", "watch the issue tracker", every_ms=300_000, due_in_s=120),
    ]
    if shape == "wakes":
        return wakes, []
    if shape == "both":
        monitors[1] = _monitor(
            "m2",
            "watch the issue tracker",
            every_ms=300_000,
            due_in_s=None,
            disabled=True,
            disabled_reason="the tool stopped being read-only after 5 failed checks",
            failures=5,
        )
        return wakes, monitors
    if shape == "overflow":
        wakes = [
            _wake("w1", "check the backup", every_ms=3_600_000, due_in_s=1_800),
            _wake("w2", "re-run the nightly audit", every_ms=None, due_in_s=7_200),
            _wake("w3", "sweep the staging logs", every_ms=600_000, due_in_s=900),
            _wake("w4", "poke the slow endpoint", every_ms=300_000, due_in_s=300),
            _wake("w5", "flush the metrics cache", every_ms=900_000, due_in_s=600),
        ]
        monitors = [
            _monitor("m1", "watch the deploy queue", every_ms=60_000, due_in_s=42),
            _monitor("m2", "watch the issue tracker", every_ms=300_000, due_in_s=120),
            _monitor("m3", "watch the error budget", every_ms=120_000, due_in_s=60),
            _monitor("m4", "watch the queue depth", every_ms=30_000, due_in_s=15),
        ]
        return wakes, monitors
    raise SystemExit(f"unknown shape {shape!r} (wakes|both|overflow)")


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))
    shape = sys.argv[3] if len(sys.argv) > 3 else "wakes"

    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        # The app boot is a worker: `self._session` lands a frame or two after
        # mount, and a band sync before it paints the empty band. Wait on the
        # app's own attribute rather than a fixed pause count.
        for _ in range(20):
            await pilot.pause()
            if app._session is not None:
                break
        target = app._session if app._session is not None else session
        wakes, monitors = _shapes(shape)
        target.wake_scheduler = _WakeScheduler(wakes)  # type: ignore[attr-defined]
        target.monitor_scheduler = _MonitorScheduler(monitors)  # type: ignore[attr-defined]
        # Through the app's OWN refresh path (the 1 Hz poll's method), so the
        # frame is the band the poll produces rather than a hand-painted one.
        app._refresh_band()
        await pilot.pause()

        await pilot.pause()
        await settle_status_line(pilot, app)
        panel = app.query_one(WakePanel)
        content = str(panel._body.content)
        print(
            f"shape={shape} size={app.screen.size} virtual={app.screen.virtual_size} "
            f"vscroll={app.screen.show_vertical_scrollbar}",
            file=sys.stderr,
        )
        print(
            f"panel display={bool(panel.display)} rows={panel.predicted_rows()} "
            f"body_budget={panel._body_rows()} painted_lines={len(content.splitlines())}",
            file=sys.stderr,
        )
        for line in content.splitlines():
            print(f"  | {line}", file=sys.stderr)
        save_capture(app, out)


asyncio.run(main())
