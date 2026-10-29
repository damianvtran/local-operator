"""Capture the composer band with Saver OFF and ON, plus the geometry proving the
rest of the row does not move.

Run from the repo root:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/saver_command_shot.py <outdir>
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture

isolate_capture()

from rich.cells import cell_len

from local_operator.saver import command
from local_operator.tui.app import OperatorApp, _saver_label
from local_operator.tui.widgets.status_line import ICON_SAVER
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The Saver segment's reserved VALUE width, in cells (``badge`` is 5 in en).
RESERVED_VALUE_CELLS = 5
MODEL_LABEL = "openai/gpt-5.4"


async def _settle(app: OperatorApp, pilot) -> None:
    """Repaint the band and let the compositor settle.

    ``StatusLine`` renders from its own pushed fields, so a still taken without
    this would show the PREVIOUS state's row — the first version of this script
    produced two byte-identical SVGs for exactly that reason.
    """
    assert app._status is not None
    app._status.refresh()
    app.query_one("#status-band").refresh()
    await pilot.pause()
    await asyncio.sleep(0.05)
    await pilot.pause()


async def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    # A signed-in, opted-in owner is the only state the badge can appear in.
    command.signed_in = lambda: True
    session = FakeSession()
    session.routing_settings = {"radient": {"saver": {"mode": "opt_in"}}}
    session.radient_saver_choice = False
    session._transcript = type(
        "Journal", (), {"append_custom": lambda self, kind, details: asyncio.sleep(0)}
    )()
    app = OperatorApp(lambda: _factory(session))
    geometry: dict[str, object] = {}
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert app._status is not None
        # A short cwd keeps the row wide enough for the reserved segment in BOTH states;
        # a long path sheds it on the ladder and the pair would compare two
        # different layouts. The shed itself is pinned by the status-line suite.
        app._status.update(
            model_label=MODEL_LABEL,
            cwd="~/proj",
            saver=_saver_label(session),
        )
        await _settle(app, pilot)
        off_row = app._status.render_text(100).plain
        assert app._status.is_showing("saver") is True, "reserved segment must be ON the row"
        assert f"{ICON_SAVER} saver" not in off_row, off_row
        save_capture(app, out / "saver-off.svg")
        geometry["off"] = {
            "band_cells": cell_len(off_row),
            "model_cell": off_row.index(MODEL_LABEL),
            "badge_cells": 0,
        }
        session.radient_saver_choice = True
        app._status.update(saver=_saver_label(session))
        await _settle(app, pilot)
        on_row = app._status.render_text(100).plain
        assert app._status.is_showing("saver") is True, "badge must be ON the row"
        assert f"{ICON_SAVER} saver" in on_row, on_row
        save_capture(app, out / "saver-on.svg")
        geometry["on"] = {
            "band_cells": cell_len(on_row),
            "model_cell": on_row.index(MODEL_LABEL),
            "badge_cells": cell_len("saver"),
        }
    geometry["reserved_badge_value_cells"] = RESERVED_VALUE_CELLS
    geometry["model_reflow_cells"] = int(geometry["on"]["model_cell"]) - int(
        geometry["off"]["model_cell"]
    )
    (out / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    print(json.dumps(geometry, indent=2))


asyncio.run(main())
