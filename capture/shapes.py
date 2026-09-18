"""The delta's other half, looked at: what a row PAINTS once a body's brackets
are tracked as a pair.

    LO_ROOT=<tree> .venv/bin/python shapes.py OUTDIR
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

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.link_picker import LinkPickerScreen  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

SHAPES = {
    "ipv6": "It is https://[2001:db8::1]/path on the internal net.",
    "wiki_bracket": "See [https://en.wikipedia.org/wiki/Foo_(bar)] for the case.",
    "bracket_then_prose": "Sources: [https://one.test/a] and [https://two.test/b] here.",
    "citation_marker": "See [1] at https://a.test/x for the table.",
    "bare_then_cite": "From https://a.test/y [2] we know the rest.",
    "raw_query_bracket": "The feed is https://a.test/x?q=a]b today.",
    "md_bracket_target": "Read [the case](https://a.test/a[b]) once more.",
    "squared_paren": "Deep: https://a.test/a_(b_[c])_d and then stopped.",
}


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, answer in SHAPES.items():
        app = OperatorApp(lambda: _factory(FakeSession()))
        with patch("local_operator.mcp.auth.open_browser_quietly"):
            async with app.run_test(size=(100, 30)) as pilot:
                for _ in range(60):
                    await pilot.pause()
                    if app._session is not None:
                        break
                app._append_block(UserBlock("which one?", fold_width=100))
                block = AssistantBlock()
                block.update_text(answer)
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
                if isinstance(screen, LinkPickerScreen):
                    save_capture(app, str(outdir / f"shape-{name}.svg"))
                    rows = [
                        line.plain
                        for line in screen._card_text().split("\n")
                        if line.plain.startswith(("❯", " "))
                        and "─" not in line.plain
                    ]
                    report[name] = {
                        "answer": answer,
                        "targets": [t.url for t in screen._targets],
                        "rows": rows,
                    }
                else:
                    report[name] = {"answer": answer, "targets": None, "rows": None}
        print(f"[{name}] {json.dumps(report[name])}")
    (outdir / "shapes.json").write_text(json.dumps(report, indent=1))


asyncio.run(main())
