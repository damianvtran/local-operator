"""Capture the two SIBLING full-page modes opened from the boot/splash screen.

The subagent view and the org chart mount exactly the way ``/settings`` does
(hide the transcript, mount before ``#input-dock``, add the mode's class), so
they shared its collision with the boot layout. When
``_sync_boot_layout_class`` was generalised to cover every full-page mode
rather than just ``/settings`` (review round 1, F2), the condition for taking
that route over merely correcting the docstring was that both siblings be
PHOTOGRAPHED correct from the splash rather than asserted. This is that
capture; the frames live in ``docs/evidence/sibling-modes-boot-layout/``.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \\
        scripts/sibling_shot.py OUT.svg COLSxROWS {subagent|org}

Drives the real ``OperatorApp`` (the only host that loads
``local_operator.tcss``) against a scratch config dir, exactly as
``scripts/settings_shot.py`` does, and seeds NOTHING into the transcript so the
splash stays up -- which is the whole point, because a mode captured over a
conversation retires the splash and cannot show this collision at all.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SCRATCH = tempfile.mkdtemp(prefix="lo-sibling-shot-")
os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = _SCRATCH

from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_band_panels import FakeSession as BandSession  # noqa: E402
from tests.unit.tui.test_band_panels import (  # noqa: E402
    _async_factory,
    _fake_jobs,
    _Job,
)
from tests.unit.tui.test_team_chart import _nested_registry  # noqa: E402


def _geometry(app: OperatorApp, view, mode: str, size) -> str:
    screen = app.screen
    dock = app.query_one("#input-dock")
    shell = app.query_one("#input-shell")
    return (
        f"mode={mode} size={size} "
        f"boot={screen.has_class('boot')} boot-card={screen.has_class('boot-card')} "
        f"dock.height={dock.size.height} dock.outer={dock.outer_size.height} "
        f"shell.width={shell.size.width} shell.x={shell.region.x} "
        f"view.height={view.size.height} view.width={view.size.width} "
        f"screen.size={tuple(screen.size)} "
        f"virtual<=size={screen.virtual_size.height <= screen.size.height}"
    )


async def main() -> None:
    out = sys.argv[1]
    cols, rows = sys.argv[2].split("x")
    size = (int(cols), int(rows))
    mode = sys.argv[3]

    if mode == "subagent":
        session = BandSession()
        session.jobs = _fake_jobs(_Job("sub-1", "audit the ingest path"))
        app = OperatorApp(_async_factory(session))
    else:
        session = FakeSession()
        session.team_registry = _nested_registry()
        app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=size) as pilot:
        for _ in range(80):
            await pilot.pause()
            if app._session is not None:
                break
        # NOTHING appended: the splash is up, which is the state the collision
        # needs. Assert it rather than hope for it.
        assert app.screen.has_class("boot"), "premise: the boot layout must be up"
        if mode == "subagent":
            app._refresh_band()
            await pilot.pause()
            app._open_subagent_view("sub-1")
        else:
            app._open_org_chart_view("org")
        await pilot.pause()
        await pilot.pause()
        view = app._subagent_view if mode == "subagent" else app._org_chart_view
        assert view is not None
        app.save_screenshot(out)
        print(_geometry(app, view, mode, size))


asyncio.run(main())
