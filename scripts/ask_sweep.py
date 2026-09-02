"""Sweep the ask card across sizes and cursor positions, on the REAL app.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/ask_sweep.py OUT.json

Walks every (fixture, size, cursor, reveal state) and records the frame, the
plan and the selected row's prefix reach. Written for the F3 decision, where
the question is whether a drawing change costs any frame any text; a sweep is
the only honest answer, because the reveal's arithmetic turns over between
sizes and the defect it removes is visible at two of them.

Sizes are the four the F3 brief names plus the 34-row band D6 was found in,
where the block can afford exactly one line after the column reserve.
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
from ask_probe import APPROVAL_QUESTION, SHORT_QUESTION, prefix_reach  # noqa: E402
from ask_user_repro import QUESTION as REPRO_QUESTION  # noqa: E402

# One long option beside one-line ones: the shape where the reveal block is
# reserved for the TALL row and then retargeted onto a row the list already
# draws in full. That is the case a continuation rule has to answer for.
MIXED_QUESTION = AskQuestion(
    id="mixed",
    question="How should the recorder handle a ledger written by a newer release?",
    options=[
        AskOption(
            label="Refuse to open it",
            description=(
                "The recorder checks the schema version on open and refuses outright when the"
                " ledger reports a shape this release does not know how to read, which is the"
                " only behaviour that cannot corrupt a file a newer release is still writing"
                " to. The cost is that a user who rolls back after an upgrade finds their"
                " history apparently gone, with no way to reach it short of upgrading again,"
                " and nothing in the message explains that the data is intact."
            ),
        ),
        AskOption(
            label="Open it read-only",
            description="history stays visible, writes are refused",
        ),
        AskOption(label="Open it anyway", description="fastest, risks silent corruption"),
    ],
    recommended=1,
)

FIXTURES = {
    "repro": REPRO_QUESTION,
    "mixed": MIXED_QUESTION,
    "short": SHORT_QUESTION,
    "approval": APPROVAL_QUESTION,
}

SIZES = [(190, 50), (150, 40), (140, 34), (130, 30), (100, 34), (100, 30)]


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

        card = app.query_one(AskPickerScreen)
        lines = card.render_lines_for_test()
        layout = card._layout()
        selected = card.state.selected
        description = card._row_description(selected)
        result = {
            "height": len(lines),
            "grants": [layout.description_rows.get(i, 0) for i in range(card.row_count)],
            "body_line_budget": layout.body_line_budget,
            "show_title": layout.show_title,
            "descriptions": layout.show_descriptions,
            "show_position": layout.show_position,
            "offers_e": card._reveal_hint() is not None,
            "hint": card._reveal_hint(),
            "revealed": card.state.revealed,
            "reach": prefix_reach(lines, " ".join(description.split())) if description else 0,
            "full": len(" ".join(description.split())),
            "blank_in_card": sum(1 for line in lines if not line.strip()),
            "lines": lines,
        }
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return result


async def main() -> None:
    out: dict[str, dict[str, object]] = {}
    for name, question in FIXTURES.items():
        for size in SIZES:
            for row in range(4):
                for reveal in (False, True):
                    key = f"{name}|{size[0]}x{size[1]}|row{row}|{'rev' if reveal else 'def'}"
                    out[key] = await measure(question, size, row, reveal)
                    print(key, out[key]["reach"], "/", out[key]["full"], flush=True)
    Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sweep.json").write_text(json.dumps(out))


asyncio.run(main())
