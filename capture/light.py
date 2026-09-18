"""D2 on the LIGHT ramp — the gap both round 1 and its remediation left open.

Renders the `/links` card on the shipped `light` (Operator Light) ramp and
prints the ink of its meta layer plus the contrast against the card's own
ground there.

    LO_ROOT=<tree> .venv/bin/python light.py OUTDIR
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(os.environ.get("LO_ROOT", "/Users/damian/lo-wt/open-links-r3")).resolve()
sys.path.insert(0, str(ROOT))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui import theme as theme_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.link_picker import LinkPickerScreen  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

ANSWER = "Sources: https://one.test/a, https://two.test/b and https://three.test/c."


def _style_hex(style) -> str | None:
    color = getattr(style, "color", None)
    if color is None:
        return None
    try:
        return "#%02x%02x%02x" % color.get_truecolor()
    except Exception:  # noqa: BLE001
        return str(color)


def _lum(hex_color: str) -> float:
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    chan = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]


def _ratio(fg: str, bg: str) -> float:
    a, b = _lum(fg), _lum(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    app = OperatorApp(lambda: _factory(FakeSession()), theme_name="light")
    with patch("local_operator.mcp.auth.open_browser_quietly"):
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(60):
                await pilot.pause()
                if app._session is not None:
                    break
            app._append_block(UserBlock("how do I roll this out?", fold_width=100))
            block = AssistantBlock()
            block.update_text(ANSWER)
            block.finalize_text()
            app._append_block(block)
            await pilot.pause()
            editor = app.query_one(Editor)
            editor.text = "/links"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, LinkPickerScreen)
            out = outdir / "light.svg"
            save_capture(app, str(out))
            report = {
                "theme": theme_mod.current_theme_name() if hasattr(theme_mod, "current_theme_name") else "light?",
                "size": tuple(app.screen.size),
                "virtual": tuple(app.screen.virtual_size),
                "vscroll": app.screen.show_vertical_scrollbar,
                "card_region": [screen._body.parent.region.x, screen._body.parent.region.y,
                                screen._body.parent.region.width, screen._body.parent.region.height],
                "rows": [],
            }
            for line in screen._card_text().split("\n"):
                report["rows"].append(
                    {
                        "plain": line.plain,
                        "runs": [
                            {"text": line.plain[s.start : s.end], "ink": _style_hex(s.style)}
                            for s in line.spans
                        ],
                    }
                )
            print(json.dumps(report, indent=1))

    light = theme_mod.BRAND_TOKENS["light"]
    ground = light["overlay"]
    print(f"\nlight ramp: dim {light['dim']} on ground {ground} = {_ratio(light['dim'], ground):.2f}:1")
    print(f"light ramp: faint {light['faint']} on ground {ground} = {_ratio(light['faint'], ground):.2f}:1")
    print(f"light ramp: muted {light['muted']} on ground {ground} = {_ratio(light['muted'], ground):.2f}:1")
    print(f"light ramp: ink {light['ink']} on ground {ground} = {_ratio(light['ink'], ground):.2f}:1")


asyncio.run(main())
