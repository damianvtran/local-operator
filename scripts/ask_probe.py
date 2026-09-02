"""Measure the ask card under the REAL app: height, grants, reveal, reach.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_probe.py

Prints one row per (fixture, size, cursor, reveal state) with the numbers the
F3 decision turns on -- card height, per-row description grants, the reveal
block's height, and the PREFIX REACH of the selected row's description, i.e.
how many characters of it the user can actually read off the frame.

Not a test: this is the measurement harness the F3 report quotes. Kept in
`scripts/` beside the shot scripts because it answers the same question they
do, in numbers rather than in pixels.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.harness.types import AskOption, AskQuestion  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.ask_picker import AskPickerScreen  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.transcript import UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from ask_user_repro import QUESTION as REPRO_QUESTION  # noqa: E402

SHORT_QUESTION = AskQuestion(
    id="stale",
    question="What should happen to the stale analytics rows?",
    options=[
        AskOption(label="Drop the rows", description="nothing reads the column any more"),
        AskOption(label="Backfill from the audit log", description="slower, keeps history"),
        AskOption(label="Dual-write for a week", description="safest, needs a follow-up MR"),
    ],
    recommended=1,
)

APPROVAL_QUESTION = AskQuestion(
    id="approve",
    question="Run `rm -rf /Users/x/project/build`?",
    options=[
        AskOption(label="Allow", description="runs the command once, this turn only"),
        AskOption(label="Deny", description="the tool call is refused and reported"),
        AskOption(label="Allow all", description="stop asking for this session"),
    ],
)


def prefix_reach(rows: list[str], full: str) -> int:
    """Longest prefix of ``full`` the frame carries -- the test file's measure."""
    joined = " ".join(" ".join(line.split()) for line in rows)
    low, high = 0, len(full)
    while low < high:
        mid = (low + high + 1) // 2
        if full[:mid] in joined:
            low = mid
        else:
            high = mid - 1
    return low


def _card(app: OperatorApp) -> AskPickerScreen:
    return app.query_one(AskPickerScreen)


async def measure(
    question: AskQuestion, size: tuple[int, int], row: int, reveal: bool
) -> dict[str, object]:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._append_block(UserBlock("give me an ask tool call where the description is long"))
        prose = AssistantBlock()
        prose.update_text("Sure -- here is one with paragraph-length options.")
        app._append_block(prose)
        await pilot.pause()

        task = asyncio.create_task(app.request_user_choice([question]))
        for _ in range(10):
            await pilot.pause()
        for _ in range(row):
            await pilot.press("down")
            await pilot.pause()
        if reveal:
            await pilot.press("ctrl+e")
            await pilot.pause()

        card = _card(app)
        lines = card.render_lines_for_test()
        layout = card._layout()
        selected = card.state.selected
        description = card._row_description(selected)
        result = {
            "height": len(lines),
            "grants": [layout.description_rows.get(i, 0) for i in range(card.row_count)],
            "reveal_rows": layout.reveal_rows,
            "show_title": layout.show_title,
            "descriptions": layout.show_descriptions,
            "offers_e": card._reveal_hint() is not None,
            "reach": prefix_reach(lines, description) if description else 0,
            "full": len(description),
            "lines": lines,
        }
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return result


FIXTURES = {
    "repro": REPRO_QUESTION,
    "short": SHORT_QUESTION,
    "approval": APPROVAL_QUESTION,
}

SIZES = [(190, 50), (150, 40), (130, 30), (100, 30)]


async def main() -> None:
    out: dict[str, dict[str, object]] = {}
    for name, question in FIXTURES.items():
        for size in SIZES:
            for row in (0, 1, 2):
                for reveal in (False, True):
                    key = f"{name}|{size[0]}x{size[1]}|row{row}|{'rev' if reveal else 'def'}"
                    out[key] = await measure(question, size, row, reveal)
    Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe.json").write_text(json.dumps(out))
    for key, value in out.items():
        print(
            f"{key:40s} h={value['height']:2d} grants={value['grants']} "
            f"rev={value['reveal_rows']} ^e={'Y' if value['offers_e'] else '-'} "
            f"reach={value['reach']}/{value['full']}"
        )


asyncio.run(main())
