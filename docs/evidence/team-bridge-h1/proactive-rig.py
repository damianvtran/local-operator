"""R10/Q5 pane frames: the populated proactive pane on the REAL OperatorApp.

Same approach QA used for the before-frames (real app + stylesheet, supported
SettingsView.load synthetic roster, action_pane), plus settled pairs (two
frames after idle, byte-compared), effective theme, geometry sidecars from
save_capture and the direct helper arithmetic (the number half of the fix:
45 == 45 cells at the caller budget after the pending-indent reset).

Run from the repo root:  .venv/bin/python docs/evidence/team-bridge-h1/proactive-rig.py
writes into $LOCAL_OPERATOR_SCRATCHPAD/proactive-r2/.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[3]  # repo root
ROOT = Path(os.environ["LOCAL_OPERATOR_SCRATCHPAD"]) / "proactive-r2"
ROOT.mkdir(exist_ok=True)
sys.path.insert(0, str(WT))

# Inherited live-host identifiers must not reach the TUI boot below: the
# harness reads CMUX_*/LOP_* around import/boot time, and an inherited
# CMUX_WORKSPACE_ID let a headless boot rename the operator's real cmux
# workspaces (team incident). An isolated HOME, run_test and synthetic
# sessions neutralise FILESYSTEM and identity state — never ENV-carried
# identifiers; only this strip does (review round 3, R12/Q7). The
# LOCAL_OPERATOR_* prefix is deliberately NOT matched: this rig's own
# scratch/config variables are distinct from the LOP_ harness namespace.
for k in list(os.environ):
    if k.startswith(("CMUX_", "LOP_")):
        os.environ.pop(k, None)

home = ROOT / "home"
cfg = home / ".local-operator"
cfg.mkdir(parents=True, exist_ok=True)
os.environ.update(
    HOME=str(home),
    LOCAL_OPERATOR_CONFIG_DIR=str(cfg),
    TERM="xterm-256color",
    LOCAL_OPERATOR_NO_SHIMMER="1",
    LOCAL_OPERATOR_NO_NOTIFICATIONS="1",
    LOCAL_OPERATOR_NO_DESKTOP_LAUNCH="1",
)
os.environ.pop("NO_COLOR", None)

from rich.cells import cell_len  # noqa: E402
from rich.style import Style  # noqa: E402

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.tui import theme  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.settings import settings_reload  # noqa: E402
from local_operator.tui.widgets.settings_view import _paint_facts  # noqa: E402
from scripts.settings_repaint_probe import open_settings  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

FACTS = "proactive · medium · 3 tools · model/name"
NAME = "Synthetic proactive agent"
SUMMARY = "Synthetic populated roster snapshot through supported SettingsView.load"

results = []


async def shot(theme_name: str) -> None:
    cm = ConfigManager(cfg)
    settings_io.write_setting(cm, settings_io.BY_KEY["tui.theme"], theme_name)
    settings_reload()
    theme.set_theme(theme_name)
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name=theme_name)
    async with app.run_test(size=(150, 40)) as pilot:
        await pilot.pause()
        view = await open_settings(app, pilot)
        view.load(agents=[(NAME, FACTS, SUMMARY)])
        view.action_pane(1)
        await pilot.pause()
        await pilot.pause()
        stem = f"r2-proactive-150x40-{theme_name}"
        save_capture(app, ROOT / f"{stem}-1.svg")
        await pilot.pause()
        save_capture(app, ROOT / f"{stem}-2.svg")
        settled = (ROOT / f"{stem}-1.svg").read_bytes() == (ROOT / f"{stem}-2.svg").read_bytes()
        # The number half: the fixed helper against the caller's own budget.
        muted = Style(color=theme.semantic_color("muted"))
        faint = Style(color=theme.semantic_color("faint"))
        for room in (80, 27):
            line = _paint_facts(FACTS, room, muted=muted, faint=faint)
            plain = "".join(part for part, _style in line)
            expected = "    " + FACTS
            fits = room + 4 >= cell_len(expected)
            # A room that fits must render EXACTLY (the R10 regression showed
            # +12 cells here); a room that cannot fit must stay within the
            # caller's budget with the marker intact - never re-indent a part.
            ok = (
                (plain == expected if fits else plain.startswith("    proactive"))
                and cell_len(plain) <= room + 4
                and line[0][1] == muted
                and all(style is faint for _text, style in line[1:])
            )
            board = {
                "theme_requested": theme_name,
                "theme_effective": theme.current_theme(),
                "facts_source": FACTS,
                "helper_room": room,
                "fits_in_room": fits,
                "painted_plain": plain,
                "expected_prior_plain": expected,
                "cell_width_actual": cell_len(plain),
                "cell_width_expected": cell_len(expected),
                "extra_cells": cell_len(plain) - cell_len(expected),
                "marker_class": line[0][1] == muted,
                "pane_size": list(view._pane_view.size),
                "body_size": list(view._body.size),
                "app_size": list(app.size),
                "settled_pair_identical": settled,
                "result": "PASS" if ok else "FAIL",
            }
            results.append(board)
            print(json.dumps(board), flush=True)
    (ROOT / "r2-proactive-results.json").write_text(json.dumps(results, indent=2))


async def main():
    await shot("dark")
    await shot("light")


asyncio.run(main())
