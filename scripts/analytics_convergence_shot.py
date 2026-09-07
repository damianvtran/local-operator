"""Capture /analytics where BOTH the nesting and the label budgeting apply.

Usage: python scripts/analytics_convergence_shot.py OUTDIR [LO HI]

Sibling of ``analytics_width_band_shot.py``, and it exists because that script's
fixture is flat. Two changes landed on this table from different branches — one
made the rows a nested root/child forest (#716), the other made them id-keyed
with budgeted, condensed labels and measured column widths (#717) — and each was
verified against a fixture that did not exercise the other. A flat fixture
cannot see the interaction, because the interaction IS the indent:

- ``_row_prefix`` prepends ``└ `` to a nested row AFTER its label is composed,
  while the budget sizes labels to fill ``name_cap`` exactly. Compose to the
  full budget, then prepend two cells, and the row is two cells over the column
  it is padded into — so ``_group_section``'s truncation cuts the tail back off
  WITHOUT a marker, at exactly the widths where the label already fitted. That
  is an unmarked mid-word cut, the defect #717 exists to remove, reintroduced by
  #716's indent.
- The rows a nested table paints carry SUBTREE totals, which are strictly larger
  than the per-session figures ``by_session`` holds. Budgeting the cost and
  calls columns against the flat map therefore understates the very columns
  ``_row_overhead`` measures, which is the D8/D11 clipping one rollup later.

So this fixture is the band fixture plus parent edges: named roots, several
children per root (including a root whose children are numerous enough to force
label disambiguation among themselves), and one grandchild so depth 2 is on
screen. The sweep reports, per width, whether any row overruns the content box
and whether any label was cut without a marker — the two properties that can
only fail once both slices are present.

No provider request, live session or operator config is used; ``probe_isolation``
re-homes HOME and the config dir before any application import, and the capture
asserts its own ledger was untouched.
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
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402

from rich.cells import cell_len  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from local_operator.tui.widgets.tool_card import truncate_cells  # noqa: E402
from scripts.analytics_label_shot import LedgerSession, _svg_rows  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: ``(session_id, name, usd, calls, parent_id)``. The roots reuse the band
#: fixture's magnitudes so a frame from here is comparable with one from there;
#: what is new is the fifth element.
#:
#: The children of ``8f21ac93bb04`` deliberately share ONE composed name. Real
#: subagent labels are ``<role> · <parent title>``, so every sibling delegated
#: under one parent with one role composes byte-identically — the operator's
#: ledger has parents with 46, 29 and 24 such children. That is what forces
#: ``session_table_labels`` to disambiguate WITHIN a depth, and it is only
#: reachable when nesting and naming are both present.
CONV_ROWS = [
    ("8f21ac93bb04", "Toggleable Sidebar for Session Switching in the TUI", 2100.00, 25445, None),
    (
        "aa01000000c1",
        "reviewer · Toggleable Sidebar for Session Switching",
        480.10,
        5120,
        "8f21ac93bb04",
    ),
    (
        "aa02000000c2",
        "reviewer · Toggleable Sidebar for Session Switching",
        361.55,
        3980,
        "8f21ac93bb04",
    ),
    (
        "aa03000000c3",
        "reviewer · Toggleable Sidebar for Session Switching",
        274.30,
        2210,
        "8f21ac93bb04",
    ),
    ("aa04000000c4", "", 190.75, 640, "8f21ac93bb04"),
    # depth 2: a subagent that itself delegated.
    ("bb01000000g1", "qa-tester · nested regression sweep", 96.40, 310, "aa01000000c1"),
    ("2b77de10c9a5", "coder · Improve Support for Local Model Providers", 2957.68, 21927, None),
    (
        "cc01000000c1",
        "designer · Improve Support for Local Model Providers",
        402.12,
        1904,
        "2b77de10c9a5",
    ),
    ("d40c81ff7a13", "reviewer · Article-search-svc schema review", 2950.37, 12804, None),
    ("6b1d0e97ca42", "Harden Against Credential Leak Paths in the runner", 2210.11, 8324, None),
    ("0f3a8d2e5c19", "eval osworld/chrome-0421 · claude-sonnet-4-6", 1980.44, 1712, None),
    ("77c4e0a1b8d3", "qa-tester · Update Provider Onboarding and OAuth", 1755.02, 430, None),
    ("c19b4f7d3a08", "short one", 1502.75, 96, None),
    ("98bfe7686ffe", "", 1200.10, 8, None),
]

#: Matches the band fixture, for the same reason: the ledger's provider row is an
#: order of magnitude past its largest session, and the calls column's width is
#: what the D11 finding turned on.
PROVIDER_CALLS = 317_977

#: Brackets the 96-cell content-box breakpoint on both sides, as the band script
#: does, so a regression that lives in one band is visible next to widths that
#: are fine.
BAND = range(104, 131)

#: Exported as images: 114 is the width the D8 clipping lived at, 120 is an
#: ordinary terminal, and both are the ones earlier rounds captured.
FRAME_WIDTHS = (114, 120)


def seed(store: AnalyticsStore) -> None:
    """Write the convergence fixture, parent edges included.

    ``UsageAggregate.calls`` is ``COUNT(*)`` over the ledger, so a row's call
    count is literally how many snapshots it carries; spend and tokens are
    divided across those calls so the rendered ``$``/``tokens`` figures stay
    comparable with the band fixture's frames.
    """
    snapshots = []
    for index, (session_id, name, usd, calls, parent) in enumerate(CONV_ROWS):
        for part in range(calls):
            snap = replace(
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
            if parent:
                snap = dataclasses.replace(snap, parent_session_id=parent)
            snapshots.append(snap)
        if name:
            store.upsert_session_name(session_id, name)

    filler = PROVIDER_CALLS - sum(row[3] for row in CONV_ROWS)
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
    session.set_conversation_name("Analytics convergence")
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="capture",
        generation=1,
        context_tokens=28400,
        context_is_estimate=False,
        context_window=200000,
    )
    return session


def _factory_for(session):
    from tests.unit.tui.test_app_pilot import _factory

    return _factory(session)


async def measure(width: int, out: Path | None, pages: int) -> dict[str, object]:
    """Drive the real app at ``width`` and report whether its rows fit.

    Three consecutive captures with a pause between them, so a post-paint
    reflow shows up as two differing SVGs rather than being averaged away.
    """
    app = OperatorApp(lambda: _factory_for(_make_session()))
    async with app.run_test(size=(width, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()

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
        painted = [truncate_cells(line, box) for line in rows]
        clipped = [line for line in painted if not line.endswith(" cache")]
        # A row that had to be truncated by the PAINT is a budget failure: the
        # label was composed against a budget the frame then refused to honour.
        # This is the check a flat fixture cannot make, because only a nested
        # row carries a prefix the budget has to account for.
        overrun = [line for line, p in zip(rows, painted) if line != p]
        nested = [line for line in painted if "└" in line]
        result: dict[str, object] = {
            "terminal": width,
            "content_box": box,
            "widest_row": widest,
            "gutter": box - widest,
            "rows": len(rows),
            "nested_rows": len(nested),
            "rows_clipped": len(clipped),
            "rows_truncated_by_paint": len(overrun),
            "sample_tail": painted[0][-14:] if painted else "",
        }
        if out is not None:
            result["frames_identical"] = len(set(frames)) == 1
            visible = [ln for ln in _svg_rows(out / f"w{width}-frame0.svg") if " tokens" in ln]
            result["rows_on_screen"] = len(visible)
            result["on_screen"] = visible
        return result


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else BAND.start
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else BAND.stop - 1

    ledger = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / "analytics.db"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    store = AnalyticsStore(ledger)
    seed(store)
    store.close()
    before = hashlib.sha256(ledger.read_bytes()).hexdigest()

    rows = []
    for width in range(lo, hi + 1):
        rows.append(await measure(width, out if width in FRAME_WIDTHS else None, pages=3))

    print(
        f"{'term':>5} {'box':>4} {'widest':>7} {'gut':>4} {'rows':>5} {'nest':>5} "
        f"{'clip':>5} {'cut':>4}"
    )
    for row in rows:
        print(
            f"{row['terminal']:>5} {row['content_box']:>4} {row['widest_row']:>7} "
            f"{row['gutter']:>4} {row['rows']:>5} {row['nested_rows']:>5} "
            f"{row['rows_clipped']:>5} {row['rows_truncated_by_paint']:>4}"
        )

    bad = [r["terminal"] for r in rows if r["rows_truncated_by_paint"]]
    clipped = [r["terminal"] for r in rows if r["rows_clipped"]]
    print(f"\nwidths where the paint had to truncate a row: {bad}")
    print(f"widths with a clipped rightmost column:        {clipped}")

    (out / "convergence-sweep.json").write_text(
        json.dumps({"band": [lo, hi], "sweep": rows}, indent=1)
    )
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before, "the fixture ledger moved"
    print(f"\nwrote {out}/convergence-sweep.json and frames for {FRAME_WIDTHS}")


if __name__ == "__main__":
    asyncio.run(main())
