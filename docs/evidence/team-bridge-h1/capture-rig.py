"""H1 remediation frames — actual OperatorApp, stylesheet loaded, isolated home.

Posture is REPLAYED from the LIVE wire captures (wireproof-captures.json): the
availability record carries the same visible/state/read_only/note the desktop
route returned, so the read-only sentence shown here is the one the wire
published. Settled pairs: two frames per case after the app is idle; geometry
and effective theme are logged per case, never inferred from filenames.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

SCRATCH = Path(os.environ["LOCAL_OPERATOR_SCRATCHPAD"])
OUT = SCRATCH / "frames-r1"  # set LOCAL_OPERATOR_SCRATCHPAD; run from the repo root
OUT.mkdir(exist_ok=True)
WT = Path(__file__).resolve().parents[3]  # repo root
sys.path.insert(0, str(WT))

for k in list(os.environ):
    if k.startswith(("CMUX_", "LOP_")):
        os.environ.pop(k, None)
home = SCRATCH / "frames-home-r1"
cfg = home / ".local-operator"
home.mkdir(exist_ok=True)
cfg.mkdir(exist_ok=True)
os.environ.update(
    HOME=str(home),
    LOCAL_OPERATOR_CONFIG_DIR=str(cfg),
    TERM="xterm-256color",
    LOCAL_OPERATOR_NO_SHIMMER="1",
    LOCAL_OPERATOR_NO_NOTIFICATIONS="1",
    LOCAL_OPERATOR_NO_DESKTOP_LAUNCH="1",
)
os.environ.pop("NO_COLOR", None)

from local_operator import settings_io  # noqa: E402
from local_operator.config import ConfigManager  # noqa: E402
from local_operator.team_bridge import availability  # noqa: E402
from local_operator.tui import theme  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.settings import settings_reload  # noqa: E402
from scripts.settings_repaint_probe import open_settings  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

CAP = json.loads((Path(__file__).parent / "wire-captures.json").read_text())
LONG_SESSION = "synthetic-cos-session-with-long-identifier-0000000000000000000000000000000000000000"


def posture(name: str) -> dict:
    if name == "absent":
        return {"visible": False, "state": "A", "checked_at": 1.0, "read_only": False, "note": None}
    r = CAP[name]["result"]
    return {
        "visible": r["visibility"]["settings"],
        "state": r["surface_state"],
        "checked_at": 1.0,
        "read_only": r["read_only"],
        "note": r["state_note"],
    }


results = []


async def capture(name, size, theme_name, key, *, configured=False, press_space=False):
    cm = ConfigManager(cfg)
    settings_io.write_setting(cm, settings_io.BY_KEY["tui.theme"], theme_name)
    settings_io.write_setting(
        cm, settings_io.BY_KEY["team_bridge.session"], LONG_SESSION if configured else ""
    )
    settings_reload()
    theme.set_theme(theme_name)
    availability.record(cfg, **posture(name))
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name=theme_name)
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        view = await open_settings(app, pilot)
        row_info = {}
        if key is not None:
            index = next(i for i, r in enumerate(view._rows) if r.setting and r.setting.key == key)
            view._selected = index
            view._land()
            await pilot.pause()
            row = view._rows[index]
            row_info = {
                "selected": key,
                "detail_plain": view._detail_text.plain,
                "detail_two_line": bool(getattr(view, "_detail_two_line", False)),
                "row_value": view._row_text(row, index, view._list.size.width).plain,
                "readonly": bool(view._current_is_readonly()),
            }
        stem = f"r1-{name}-{size[0]}x{size[1]}-{theme_name}"
        if configured:
            stem += "-cosconfigured"
        if key and key.endswith("enabled"):
            stem += "-enabled"
        for frame in (1, 2):
            save_capture(app, OUT / f"{stem}-{frame}.svg")
            await pilot.pause()
        tokens = theme.get_tokens()
        surface = tokens.get("surface") or tokens.get("background") or tokens.get("bg")
        detail = {
            "state_source": (
                "LIVE wire capture replayed into availability (wireproof-captures.json)"
            ),
            "state": name,
            "size": list(size),
            "theme_requested": theme_name,
            "theme_effective": theme.current_theme(),
            "muted": theme.semantic_color("muted"),
            "faint": theme.semantic_color("faint"),
            "dim": theme.semantic_color("dim"),
            "surface": surface,
            "contrast_muted_surface": (
                theme.contrast(theme.semantic_color("muted"), surface) if surface else None
            ),
            "contrast_faint_surface": (
                theme.contrast(theme.semantic_color("faint"), surface) if surface else None
            ),
            "app_size": list(app.size),
            "body_size": list(view._body.size),
            "body_virtual_size": list(view._body.virtual_size),
            "rows": len(view._rows),
            "scroll_y": view._body.scroll_y,
            "config_session": ConfigManager(cfg).get_nested_value(("team_bridge", "session")),
            **row_info,
        }
        if press_space:
            before = ConfigManager(cfg).get_nested_value(("team_bridge", "enabled"), False)
            await pilot.press("space")
            await pilot.pause()
            after = ConfigManager(cfg).get_nested_value(("team_bridge", "enabled"), False)
            save_capture(app, OUT / f"{stem}-after-space.svg")
            detail["space_enabled_before_after"] = [before, after]
            detail["editing_after_space"] = str(view._editing)
        results.append(detail)
        (OUT / "r1-frames-results.json").write_text(json.dumps(results, indent=2, default=str))
        print(json.dumps(detail, default=str), flush=True)


async def main():
    await capture("B", (80, 24), "dark", "team_bridge.enabled")
    await capture("B", (80, 24), "light", "team_bridge.enabled")
    await capture("B", (80, 24), "dark", "team_bridge.session", configured=True)
    await capture("B", (60, 20), "dark", "team_bridge.enabled")
    await capture("B", (60, 20), "light", "team_bridge.enabled")
    await capture("F_unknown", (80, 24), "dark", "team_bridge.enabled", press_space=True)
    await capture("F_unknown", (80, 24), "light", "team_bridge.enabled")
    await capture("F_unknown", (60, 20), "dark", "team_bridge.enabled")
    await capture("F_kill", (80, 24), "dark", "team_bridge.enabled", press_space=True)
    await capture("absent", (80, 24), "dark", None)
    print("FRAMES DONE", len(results), flush=True)


asyncio.run(main())
