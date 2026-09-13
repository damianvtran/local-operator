"""Capture /session and /analytics carrying web-search spend.

Usage: python scripts/search_spend_shot.py OUTDIR [WxH]

The fixture is the existing session-report one (reused, not re-invented): a real
ledger shape on a real app, so the frames differ from the pre-change ones only by
the search-spend surfaces under test. Search spend is seeded through the same
process-wide ledger the ``web_search`` tool writes, with one priced provider, one
free provider, one read (a distinguishable provider key), and one provider whose
price is unknown -- the four states the section has to spell differently.

Run this on the branch for the AFTER frames and run the sibling scripts
(``session_report_shot.py populated``, ``analytics_label_shot.py``) in a base
worktree for the BEFORE frames; the base does not have this surface at all.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.web_search.cost import SEARCH_SPEND  # noqa: E402
from local_operator.web_search.models import SearchCost  # noqa: E402
from scripts.session_report_shot import DiagnosticSession, seed_populated  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402


def seed_search_spend(session_id: str, scenario: str = "mixed") -> None:
    """Search spend in the shapes the section must render differently.

    ``read-only`` is its own scenario because it is the case the section used to
    vanish for: reads carry money and no search count, so a guard on searches
    drew nothing while the status band kept the figure.
    """
    SEARCH_SPEND.reset()
    if scenario == "read-only":
        SEARCH_SPEND.record(
            session_id,
            "deepseek:read",
            SearchCost(
                usd=0.002, basis="tokens at list price (estimate), off-peak", priced_from_usage=True
            ),
            kind="read",
        )
        return
    # Priced per search (a keyed paid engine).
    SEARCH_SPEND.record(
        session_id, "brave", SearchCost(usd=0.004, basis="published per-search rate")
    )
    SEARCH_SPEND.record(
        session_id, "brave", SearchCost(usd=0.004, basis="published per-search rate")
    )
    # Token-priced, an estimate, the expensive one.
    SEARCH_SPEND.record(
        session_id,
        "deepseek",
        SearchCost(
            usd=0.0049, basis="tokens at list price (estimate), peak", priced_from_usage=True
        ),
    )
    # Free, and known to be free.
    SEARCH_SPEND.record(session_id, "duckduckgo", SearchCost(usd=0.0, basis="free"))
    # A read from captured pages: its own provider key AND its own kind, so the
    # money lands in the total without inflating the search count -- and so the
    # row can say "read" rather than claiming to be a search.
    SEARCH_SPEND.record(
        session_id,
        "deepseek:read",
        SearchCost(
            usd=0.002, basis="tokens at list price (estimate), off-peak", priced_from_usage=True
        ),
        kind="read",
    )
    # Unknown price: must not read as $0.00.
    SEARCH_SPEND.record(
        session_id, "future-engine", SearchCost(usd=None, basis="no published rate")
    )


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    size_arg = sys.argv[2] if len(sys.argv) > 2 else "120x40"
    cols, rows = size_arg.split("x")
    size = (int(cols), int(rows))

    scenario = sys.argv[3] if len(sys.argv) > 3 else "mixed"
    session = DiagnosticSession()
    # A conversation-like name: a fixture titled after the surface under test
    # reads as a section heading in the frames it produces.
    session.set_conversation_name("Investigate retrieval costs")
    seed_populated(session.session_id)
    seed_search_spend(session.session_id, scenario)

    app = OperatorApp(lambda: _factory(session))
    widths = [(int(cols), int(rows)), (60, 24), (80, 24)]
    for index, size in enumerate(widths):
        suffix = "" if index == 0 else f"-{size[0]}x{size[1]}"
        app = OperatorApp(lambda: _factory(session))
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            await _submit(pilot, app, "/session")
            await pilot.pause()
            save_capture(app, str(out / f"session-with-search-spend{suffix}.svg"))
            # The search-spend block sits below the ledger sections on this screen,
            # so the frame that proves it renders is a PAGED one: a top-of-page shot
            # of a scrollable body is evidence about the fold, not the block. Paging
            # rather than a press count, because the number of presses that reaches
            # the block depends on the terminal height (the sibling report script's
            # idiom, for the same reason).
            for page in range(1, 6):
                await pilot.press("pagedown")
                await pilot.pause()
                save_capture(app, str(out / f"session-search-spend-page-{page}{suffix}.svg"))
                scroll = getattr(app.screen, "_scroll", None)
                if scroll is not None and scroll.scroll_offset.y >= scroll.max_scroll_y:
                    break
            await pilot.press("escape")
            await pilot.pause()
            await _submit(pilot, app, "/analytics")
            await pilot.pause()
            save_capture(app, str(out / f"analytics-with-search-spend{suffix}.svg"))
    print(f"wrote frames for {', '.join(f'{w}x{h}' for w, h in widths)} into {out}")


if __name__ == "__main__":
    asyncio.run(main())
