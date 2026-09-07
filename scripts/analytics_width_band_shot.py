"""Capture the /analytics session table across the width BAND, not at samples.

Usage: python scripts/analytics_width_band_shot.py OUTDIR [PAGES]

Sibling of ``analytics_label_shot.py`` (what the rows SAY) and
``analytics_collision_shot.py`` (which row is WHICH). This one is about whether
a row FITS, and it exists because sampling three widths could not see the
defect it was written for.

Design review D8: the name budget stepped 30 -> 48 the instant the content box
reached 96 while the rest of a row still needed 51-55 cells, so the widest row
jumped to 103 against a 96-cell box and terminal widths 114-120 clipped the
``% cache`` column off every row. Nothing about the frame said so — a row
ending in ``cach`` still looks like a row — and rounds captured at 71/100/140
straddled the breakpoint without landing on it.

So the artifact here is a SWEEP rather than a still: every terminal width in a
range is driven through the real app, and each one reports its content box, the
widest row the report composes, and the resulting gutter. A negative gutter is
a clipped column. Two frames (114 and 120) are exported as images because that
is what a reader can look at; the table is what proves the band is closed.

The fixture is deliberately several sessions with one >=48-character name. A
LONE long row does not reproduce the defect: the name column is sized to the
widest label present, so a single long name is cut to the cap and the row lands
exactly on the box edge. It takes a second row holding the column at the full
cap for the overrun to appear — the ordinary shape of this table, and the shape
of the ledger the review bisected.

No provider request, live session or operator config is used; ``probe_isolation``
re-homes HOME and the config dir before any application import, and every
capture asserts its own ledger was untouched.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import: a headless
# pilot must not rename the operator's real workspace through inherited CMUX IDs.
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402

from rich.cells import cell_len  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from local_operator.tui.widgets.tool_card import truncate_cells  # noqa: E402
from scripts.analytics_label_shot import LedgerSession, _svg_rows  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: Enough expensive rows to fill the frame, with two names at or past the 48-cell
#: cap so the name column is actually held open at its maximum — see the module
#: docstring on why one long row is not enough to reproduce D8.
#:
#: The third element is the CALL COUNT, and it is why this fixture was rebuilt
#: for design review D11. The previous seed rendered ``16 calls`` and ``2
#: calls`` — 1-2 digits against what was then a hard-coded 4-cell allowance — so
#: the exported frames could not show the column that actually overran, and a
#: real defect passed a visual review. These magnitudes are taken from the
#: operator's own ledger (largest session 25,445 calls; the ``anthropic``
#: provider row 317,977) so the frames exercise the 5- and 6-digit widths that
#: exist in practice, alongside small counts so the column is ragged in the way
#: real data makes it. A fixture whose magnitudes are all toy-sized is evidence
#: only about toy data.
BAND_ROWS = [
    ("8f21ac93bb04", "Toggleable Sidebar for Session Switching in the TUI", 3433.96, 25445),
    ("2b77de10c9a5", "coder · Improve Support for Local Model Providers", 2957.68, 21927),
    ("d40c81ff7a13", "reviewer · Article-search-svc schema review", 2950.37, 12804),
    ("6b1d0e97ca42", "Harden Against Credential Leak Paths in the runner", 2210.11, 8324),
    ("0f3a8d2e5c19", "eval osworld/chrome-0421 · claude-sonnet-4-6", 1980.44, 1712),
    ("77c4e0a1b8d3", "qa-tester · Update Provider Onboarding and OAuth", 1755.02, 430),
    ("c19b4f7d3a08", "short one", 1502.75, 96),
    ("98bfe7686ffe", "", 1200.10, 8),
]

#: The per-session call counts above sum to the provider row's count, and the
#: ledger's real provider total is an order of magnitude past its largest
#: session (317,977 vs 25,445). Seeding one row per call would mean ~70k inserts
#: for a screenshot, so the provider column is topped up with a single
#: cheap-to-seed session carrying the remainder; the point of the fixture is the
#: WIDTH of the rendered numbers, not their provenance.
PROVIDER_CALLS = 317_977

#: The band to sweep. Chosen to bracket the 96-cell content-box breakpoint with
#: room on both sides: at 0.9x-6 (``_card_width``) terminals 108-126 map to
#: boxes 91-107, and the defect lived at boxes 96-98 (terminals 114-116).
BAND = range(104, 131)

#: The two frames exported as images, because they are what the review asked to
#: see: 114 is the first clipping width, 120 is the ordinary terminal in the band.
FRAME_WIDTHS = (114, 120)


def seed(store: AnalyticsStore) -> None:
    """Write the band fixture into the isolated ledger, names included.

    ``UsageAggregate.calls`` is ``COUNT(*)`` over the ledger, so a row's call
    count is literally how many snapshots it carries — there is no field to set.
    The D11 magnitudes therefore cost real inserts (~390k), which is why the
    spend and token totals are divided across a row's calls rather than repeated:
    the rendered ``$`` and ``tokens`` figures must stay the ones the earlier
    rounds captured, so that only the CALLS column changes between the round-2
    and round-3 frames and a reader comparing them sees one variable move.
    """
    snapshots = []
    index = 0
    for session_id, name, usd, calls in BAND_ROWS:
        for part in range(calls):
            snapshots.append(
                replace(
                    _snap(
                        session_id=session_id,
                        provider="anthropic",
                        model_id="claude-sonnet-4-6",
                        context=(120_000 + index * 900) // calls,
                        input_tokens=40_000 // calls,
                        cache_read=70_000 // calls,
                        cache_write=2_000 // calls,
                        output_tokens=6_000 // calls,
                        reasoning=1_200 // calls,
                        cost_micro=int(usd * 1_000_000 / calls),
                        chars={"conversation": 4000, "tool_results": 2200},
                        ts_ms=1788602400000 + index * 61000 + part,
                        ok=True,
                    ),
                    request_id=f"req-{index:04d}-{part}",
                    purpose="turn",
                    outcome="ok",
                    duration_ms=2400,
                )
            )
        index += 1
        if name:
            store.upsert_session_name(session_id, name)

    # Top the provider row up to the ledger's real 6-digit magnitude. Carries
    # negligible spend and tokens so it changes only the CALLS width — the
    # column under audit — and lands in a session too cheap to appear in the
    # top-of-table rows the frames show.
    filler = PROVIDER_CALLS - sum(row[3] for row in BAND_ROWS)
    for part in range(max(0, filler)):
        snapshots.append(
            replace(
                _snap(
                    session_id="f1lle4c0unt5",
                    provider="anthropic",
                    model_id="claude-sonnet-4-6",
                    context=1,
                    input_tokens=1,
                    cache_read=0,
                    cache_write=0,
                    output_tokens=1,
                    reasoning=0,
                    cost_micro=1,
                    chars={"conversation": 1},
                    ts_ms=1788602400000 + part,
                    ok=True,
                ),
                request_id=f"req-fill-{part}",
                purpose="turn",
                outcome="ok",
                duration_ms=1,
            )
        )
    store.upsert_session_name("f1lle4c0unt5", "routine background polling")
    store.record_batch(snapshots)


def _make_session() -> LedgerSession:
    session = LedgerSession()
    session.set_conversation_name("Analytics width band")
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="capture",
        generation=1,
        context_tokens=28400,
        context_is_estimate=False,
        context_window=200000,
    )
    return session


async def measure(width: int, out: Path | None, pages: int) -> dict[str, object]:
    """Drive the real app at ``width`` and report whether its rows fit.

    Exports a frame when ``out`` is given. Three consecutive captures with a
    pause between them, so a post-paint reflow would show up as two differing
    SVGs rather than being averaged away by a single lucky shot.
    """
    app = OperatorApp(lambda: _factory_for(_make_session()))
    async with app.run_test(size=(width, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()

        # Narrowed for the type checker: ``app.screen`` is typed as the base
        # ``Screen``, and the width maths under audit lives on this subclass.
        screen = app.screen
        assert isinstance(screen, AnalyticsScreen), f"expected /analytics, got {type(screen)}"
        for _ in range(pages):
            await pilot.press("pagedown")
            await pilot.pause()
        await pilot.pause()

        frames: list[str] = []
        if out is not None:
            for shot in range(3):
                path = out / f"w{width}-frame{shot}.svg"
                save_capture(app, str(path))
                frames.append(hashlib.sha256(path.read_bytes()).hexdigest())
                await pilot.pause()

        box = screen._card_width()
        render_lines = getattr(screen, "render_lines_for_test", None)
        body = "\n".join(render_lines()) if render_lines is not None else ""
        rows = [
            line.rstrip()
            for line in body.split("By session", 1)[-1].splitlines()
            if " tokens" in line
        ]
        widest = max((cell_len(line) for line in rows), default=0)
        # What the reader actually sees: the row painted into the box. A row
        # wider than the box loses its rightmost column, and that column is
        # ``% cache`` — the number that says whether a session was cheap.
        painted = [truncate_cells(line, box) for line in rows]
        clipped = [line for line in painted if not line.endswith(" cache")]
        result: dict[str, object] = {
            "terminal": width,
            "content_box": box,
            "widest_row": widest,
            "gutter": box - widest,
            "rows": len(rows),
            "rows_clipped": len(clipped),
            "sample_tail": painted[0][-14:] if painted else "",
        }
        if out is not None:
            result["frames_identical"] = len(set(frames)) == 1
            visible = [ln for ln in _svg_rows(out / f"w{width}-frame0.svg") if " tokens" in ln]
            result["rows_on_screen"] = len(visible)
            result["on_screen"] = visible
        return result


def _factory_for(session):
    from tests.unit.tui.test_app_pilot import _factory

    return _factory(session)


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    store = AnalyticsStore()
    seed(store)
    store.close()

    ledger = default_db_path()
    before = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None

    sweep = []
    for width in BAND:
        sweep.append(await measure(width, out if width in FRAME_WIDTHS else None, pages))

    after = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None
    report = {
        "source": str(Path(__file__).resolve().parents[1]),
        "band": [BAND.start, BAND.stop - 1],
        "frames": list(FRAME_WIDTHS),
        "pages_scrolled": pages,
        "session_rows_seeded": len(BAND_ROWS),
        "ledger_unchanged": before == after,
        "widths_with_clipped_rows": [s["terminal"] for s in sweep if s["rows_clipped"]],
        "sweep": sweep,
    }
    (out / "band.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'term':>5} {'box':>4} {'widest':>7} {'gutter':>7} {'clipped':>8}  tail")
    for row in sweep:
        print(
            f"{row['terminal']:>5} {row['content_box']:>4} {row['widest_row']:>7} "
            f"{row['gutter']:>7} {row['rows_clipped']:>8}  {row['sample_tail']!r}"
        )
    print()
    print("ledger_unchanged:", report["ledger_unchanged"])
    print("widths with clipped rows:", report["widths_with_clipped_rows"] or "NONE")


if __name__ == "__main__":
    asyncio.run(main())
