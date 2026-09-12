"""Capture the tool-ledger DENSITY ladder: every rung between flush and comfortable.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ledger_density_shot.py OUT.svg [COLSxROWS] [RUNG] [PALETTE]
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/ledger_density_shot.py --geometry [COLSxROWS]

The report this exists for was about the ledger's RHYTHM, not about one card:
"there's essentially no padding between lines". A single card is unreadable as
evidence of that — the defect only appears in a RUN of consecutive actions,
where the ink pitch (summary row to next summary row) is what the eye reads as
crowding. So every rung seeds six real ``ToolCard``s under an ``AssistantBlock``
through the real ``OperatorApp``, which is also what makes the frames comparable
to each other and to a live session: the lightweight hosts in the test files
declare no ``CSS_PATH`` and would not show a padding change at all (``AGENTS.md``,
"Visual validation").

The rungs, in ink-pitch order. Each is a DIFFERENT way to buy the same row of
air, and they do not look alike on a filled frame even though the arithmetic
says they cost the same:

* ``default``     — 1-row card, 1 ground row. Pitch 2. The reported state.
* ``pad-below``   — 2-row card, the pad row BELOW the summary. Pitch 3.
* ``pad-above``   — 2-row card, the pad row ABOVE the summary. Pitch 3.
* ``double-gap``  — 1-row card, TWO ground rows between neighbours. Pitch 3.
* ``comfortable`` — ``display.comfortable_rows``: 3-row card. Pitch 4.

``pad-below``/``pad-above``/``double-gap`` all cost one row per action and are
told apart only by WHOSE fill the extra row carries — the card's own
``$lo-surface`` on the first two, the transcript's ``$lo-bg`` on the third.
That distinction is the whole design question here (see the ``.comfortable-rows``
comment in the stylesheet on a fill that "terminated on an inked line"), and it
is invisible in a height number, which is why this script rasterizes.

Rungs other than the tree's own default are applied as INLINE styles on the
widgets rather than by editing the sheet, so one process can render the whole
ladder and a rung is never confused with what the repository ships. Inline
styles beat the sheet's rules, so what is captured for a rung is that rung —
but ``default`` is genuinely untouched, and therefore tracks the sheet.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.probe_isolation  # noqa: E402, F401  -- must precede app imports
from local_operator.tui.app import COMFORTABLE_ROWS_CLASS, OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    GAP_CLASS,
    TranscriptView,
    UserBlock,
)
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: A realistic opening run: the shapes an agent actually emits back to back.
#: Mixed tools so the icon/name columns vary, mixed durations so the status
#: column is not a single repeated string, and summaries long enough to reach
#: the ellipsis at 100 columns — a ledger of short rows understates crowding
#: because the eye has whitespace to rest in that a real one does not give it.
CALLS: list[tuple[str, dict[str, object], float]] = [
    ("bash", {"command": "cd ~/local-operator && git log --oneline -3 && git status --short"}, 0.1),
    ("read", {"path": "~/local-operator/AGENTS.md"}, 0.0),
    ("bash", {"command": "ls local_operator && ls scripts && echo '---' && ls tests"}, 0.0),
    ("grep", {"pattern": "Visual validation", "path": "AGENTS.md", "context_lines": 6}, 0.2),
    ("read", {"path": "local_operator/tui/local_operator.tcss", "range": "355-505"}, 0.0),
    ("bash", {"command": "grep -rn 'comfortable_rows' --include='*.py' . | head -40"}, 1.0),
]

#: Twelve calls is the second measurement the density argument needs: the cost
#: of a rung is per-action, so it is a screen's worth of ledger that says
#: whether a rung fits a working session rather than a demo.
LONG_RUN = 12

RUNGS = ("default", "pad-below", "pad-above", "double-gap", "comfortable")


def _apply_rung(app: OperatorApp, rung: str) -> None:
    """Put the app into ``rung``'s density, on top of whatever the sheet says.

    Inline styles rather than a stylesheet edit: the whole ladder has to be
    renderable from one tree so the frames differ in density and in nothing
    else (font, theme, fixture, Textual version). ``height`` and ``padding``
    are set together for the reason the sheet's own comment gives — Textual
    sizes a fixed-height widget INCLUDING its padding, so a pad row without
    the matching height clips the summary instead of bracketing it.
    """
    if rung == "comfortable":
        app.screen.add_class(COMFORTABLE_ROWS_CLASS)
        return
    for card in app.query(ToolCard):
        if rung == "pad-below":
            card.styles.height = 2
            card.styles.padding = (0, 0, 1, 0)
        elif rung == "pad-above":
            card.styles.height = 2
            card.styles.padding = (1, 0, 0, 0)
        elif rung == "double-gap":
            # The rung that buys its row from the GROUND instead of the card.
            # Only rows that already carry the adaptive gap are widened, so
            # this stays a change to the ledger's rhythm rather than a margin
            # on every block (the "blank row between everything" regression
            # the base selectors are kept margin-free to prevent).
            if card.has_class(GAP_CLASS):
                card.styles.margin = (2, 0, 0, 0)


async def _seed(app: OperatorApp, pilot, count: int) -> None:
    app.query_one(Editor).cursor_blink = False
    app._append_block(UserBlock("review the tool call lines for padding"))
    block = AssistantBlock()
    block.update_text("I'll start by reading the ground-truth docs before touching anything.")
    app._append_block(block)
    await pilot.pause()
    for index in range(count):
        name, args, seconds = CALLS[index % len(CALLS)]
        card = ToolCard(f"call-{index}", name, dict(args))
        card.mark_done("ok", measured_s=seconds)
        app._append_block(card)
    for _ in range(4):
        await pilot.pause()


def _report(app: OperatorApp, rung: str, size: tuple[int, int]) -> dict[str, object]:
    """The numbers behind the frame (``AGENTS.md``, "Check the numbers").

    Pitch is measured summary-row to summary-row on the COMPOSED frame rather
    than derived from the declarations, because that is the quantity the report
    was about and the one a padding/margin/height mix-up gets wrong.
    """
    view = app.query_one(TranscriptView)
    cards = [c for c in app.query(ToolCard) if c.region.height]
    rows: list[str] = []
    for child in view.children:
        classes = " ".join(sorted(child.classes)) or "-"
        rows.append(
            f"    y={child.region.y:>3} h={child.region.height} "
            f"fill={child.styles.background} {type(child).__name__} [{classes}]"
        )
    tops = [c.region.y for c in cards]
    pitches = [b - a for a, b in zip(tops, tops[1:])]
    heights = sorted({c.region.height for c in cards})
    card_heights = (c.region.height for c in cards)
    gaps = sorted({b - (a + h) for a, b, h in zip(tops, tops[1:], card_heights)})
    span = (cards[-1].region.bottom - cards[0].region.y) if cards else 0
    print(f"  rung={rung} size={size[0]}x{size[1]} calls={len(cards)}")
    print("\n".join(rows))
    print(f"    card heights={heights} gap rows={gaps} pitch={pitches} span={span}")
    return {
        "rung": rung,
        "height": heights[0] if heights else 0,
        "gap": gaps[0] if gaps else 0,
        "pitch": pitches[0] if pitches else 0,
        "span": span,
    }


async def _run(rung: str, size: tuple[int, int], palette: str, count: int, out: str | None):
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        app._apply_theme(palette)
        await _seed(app, pilot, count)
        _apply_rung(app, rung)
        for _ in range(4):
            await pilot.pause()
        summary = _report(app, rung, size)
        if out is not None:
            save_capture(app, out)
    return summary


async def _geometry(size: tuple[int, int]) -> None:
    table: list[dict[str, object]] = []
    for rung in RUNGS:
        short = await _run(rung, size, "dark", len(CALLS), None)
        # A tall frame so twelve cards are all laid out rather than clipped by
        # the viewport — the span being measured is the ledger's, not the
        # screen's.
        long_run = await _run(rung, (size[0], 60), "dark", LONG_RUN, None)
        short["span12"] = long_run["span"]
        table.append(short)
    print(f"\n| rung | card h | gap rows | ink pitch | span {len(CALLS)} calls | span {LONG_RUN} |")
    print("|---|---|---|---|---|---|")
    for row in table:
        print(
            f"| {row['rung']} | {row['height']} | {row['gap']} | {row['pitch']} "
            f"| {row['span']} | {row['span12']} |"
        )


def main() -> None:
    if sys.argv[1] == "--geometry":
        size = (100, 30)
        if len(sys.argv) > 2:
            cols, rows = sys.argv[2].lower().split("x")
            size = (int(cols), int(rows))
        asyncio.run(_geometry(size))
        return
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].lower().split("x")
        size = (int(cols), int(rows))
    rung = sys.argv[3] if len(sys.argv) > 3 else "default"
    palette = sys.argv[4] if len(sys.argv) > 4 else "dark"
    if rung not in RUNGS:
        raise SystemExit(f"unknown rung {rung!r}; expected one of {', '.join(RUNGS)}")
    asyncio.run(_run(rung, size, palette, len(CALLS), out))


main()
