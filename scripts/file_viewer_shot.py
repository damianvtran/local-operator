"""Capture the real file-viewer modes at both review grids, without live config.

Usage: .venv/bin/python scripts/file_viewer_shot.py <scratch-evidence-directory>
Frames are PR evidence, never committed. The fixture session is synthetic; only
its boot protocol is faked, not the app, production CSS, routing or viewer.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.widgets.supplement_block import SupplementBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_file_viewer_app import make_app, settled  # noqa: E402


async def capture(output: Path) -> None:
    fixtures = Path.home() / "viewer-fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    specimens = {
        "text": (
            "notes.txt",
            (
                "Local file preview\n\nA read-only page keeps the draft and transcript intact. "
                "Long plain-text paragraphs wrap to the viewport, including at narrow widths.\n"
            )
            * 3,
        ),
        "code": (
            "example.py",
            "# Bounded preview: source stays local\n"
            "def describe(path: str) -> str:\n"
            '    return "preview: " + path  # ' + "long source line " * 7 + "\n\n"
            'print(describe("report.txt"))\n' * 18,
        ),
        "markdown": (
            "guide.md",
            "# Local document\n\n**Read-only preview**, not an editor.\n\n"
            "- Keeps the conversation draft\n- Never writes the source\n"
            "- Reload explicitly to read changes\n\n## Limits\n\n"
            "Text is bounded before decoding.\n\n```python\nprint('safe preview')\n```\n",
        ),
        "truncated": ("large.txt", "A bounded line in a large document.\n" * 20_005),
        "binary": ("binary.txt", b"not a text file\x00payload"),
        "missing": ("missing.txt", None),
        "empty": ("empty.txt", ""),
    }
    for name, body in specimens.values():
        if isinstance(body, bytes):
            (fixtures / name).write_bytes(body)
        elif body is not None:
            (fixtures / name).write_text(body)
    for size in ((100, 30), (60, 24)):
        app = make_app(fixtures)
        grid = f"{size[0]}x{size[1]}"
        async with app.run_test(size=size) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            app._append_block(
                UserBlock("Inspect the generated report without leaving the conversation.")
            )
            app._editor().load_text("draft kept while viewing")
            await pilot.pause()
            # A delayed local read makes the real loading frame reproducible;
            # closing it also exercises worker cancellation in this capture.
            from local_operator.tui.widgets import file_viewer as viewer_module

            original_read = viewer_module.read_document_async
            gate = asyncio.Event()

            async def waiting_read(*args, **kwargs):
                await gate.wait()
                return await original_read(*args, **kwargs)

            viewer_module.read_document_async = waiting_read
            try:
                app._run_slash_command("/view notes.txt")
                await pilot.pause()
                save_capture(app, output / f"loading-{grid}.svg")
            finally:
                viewer_module.read_document_async = original_read
                app._close_file_viewer()
                await pilot.pause()
            for state, (name, _) in specimens.items():
                app._run_slash_command(f"/view {name}")
                await pilot.pause()
                await settled(pilot, app)
                await pilot.pause()
                # Both export locations use calibrated cell geometry. The raw/
                # companion is the first capture, not legacy Rich geometry;
                # save_capture writes its matching geometry sidecar as well.
                raw = output / "raw"
                raw.mkdir(parents=True, exist_ok=True)
                save_capture(app, raw / f"{state}-{grid}.svg")
                save_capture(app, output / f"{state}-{grid}.svg")
                await pilot.pause()
                save_capture(app, output / f"{state}-{grid}-settled.svg")
                app._close_file_viewer()
                await pilot.pause()
            # Same callout immediately before and after its real click route.
            block = SupplementBlock(
                {
                    "state": "done",
                    "files": [
                        {"path": name, "name": name, "size_bytes": (fixtures / name).stat().st_size}
                        for name in ("guide.md", "notes.txt", "example.py")
                    ],
                    "components": [],
                    "anchor": "synthetic-viewer-answer",
                    "version": 1,
                },
                cwd=str(fixtures),
                hyperlinks=True,
            )
            app._append_block(block)
            await pilot.pause()
            uri = (fixtures / "guide.md").as_uri()
            hits = [
                (x, y)
                for y in range(block.size.height)
                for x in range(block.size.width)
                if block.link_at(x, y) == uri
            ]
            if not hits:
                raise RuntimeError("fixture did not paint a file hyperlink")
            save_capture(app, output / f"callout-before-{grid}.svg")
            await pilot.click(block, offset=hits[0])
            await pilot.pause()
            await settled(pilot, app)
            await pilot.pause()
            save_capture(app, output / f"callout-after-{grid}.svg")
            print(
                f"{grid}: text/code/markdown/truncated/binary/missing/empty "
                "+ callout before/after"
            )


if __name__ == "__main__":
    asyncio.run(capture(Path(sys.argv[1]).resolve()))
