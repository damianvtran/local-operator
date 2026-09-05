"""Capture the long-session tool-ledger frame used for TUI performance evidence."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Resolve the checkout from the script rather than requiring the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    NoticeBlock,
    TranscriptView,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def _capture(output: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        with view.batch_append():
            for index in range(80):
                view.append_block(
                    NoticeBlock(
                        f"Long-session event {index + 1}: retained context remains readable.",
                        "info",
                    )
                )
            for index in range(12):
                card = ToolCard(f"tool-{index}", "read", {"path": f"src/module_{index}.py"})
                card.mark_done("ok")
                view.append_block(card)
        await pilot.pause()
        await pilot.pause()
        view.scroll_end(animate=False)
        await pilot.pause()
        save_capture(app, str(output))
        print(
            f"blocks={len(view.blocks())} content_size={view.size} "
            f"virtual_size={view.virtual_size} screen_size={app.screen.size} "
            f"screen_virtual={app.screen.virtual_size} "
            f"scrollbar={view.show_vertical_scrollbar}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    asyncio.run(_capture(args.output))


if __name__ == "__main__":
    main()
