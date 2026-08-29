"""Render the composer's click-selection states from the REAL OperatorApp.

Uses the real app (CSS_PATH applied) driven by `pilot.click(times=...)`, which
is the gesture the bug is about: a double-click to select a word, a
triple-click to select the line, and the Ctrl+C that should copy what they
selected.

Usage: shot_composer_copy.py <outdir>
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else "/tmp/lop-composer-copy")

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.markdown_theme import install_markdown_theme  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.toast import Toast  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

DRAFT = "summarise the ingest path please"


async def shot(outdir: Path, name: str, times: int, press_copy: bool) -> None:
    install_markdown_theme()
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text(DRAFT)
        await pilot.pause()
        await pilot.pause()
        app._clipboard = "PREEXISTING"

        # Column 18 sits inside the word "ingest".
        await pilot.click(editor, offset=(editor.gutter.left + 18, 0), times=times)
        await pilot.pause()
        await pilot.pause()

        if press_copy:
            await pilot.press("ctrl+c")
            await pilot.pause()
            await pilot.pause()

        app.save_screenshot(str(outdir / f"{name}.svg"))
        toast = app.query_one(Toast)
        print(
            f"{name}: selection={editor.selection} "
            f"selected_text={editor.selected_text!r} "
            f"clipboard={app._clipboard!r} toast={toast.message!r} "
            f"draft={editor.text!r}"
        )


async def main() -> None:
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    await shot(outdir, "01-double-click", 2, False)
    await shot(outdir, "02-double-click-ctrl-c", 2, True)
    await shot(outdir, "03-triple-click", 3, False)
    await shot(outdir, "04-triple-click-ctrl-c", 3, True)


asyncio.run(main())
