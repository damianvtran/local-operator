"""PROTOTYPE — THROWAWAY. The D16 invariant, measured rather than derived.

D16 (round-2 BLOCKER): widening the terminal 139 -> 140 cut variant A's name
field 62% and took truncation 0% -> 33%, because the breakpoint was set from
"when does side-by-side fit the p75 name" and nothing checked that side-by-side
was actually BETTER than the stacked layout it replaced.

The invariant this asserts, from the review:

    for any width W, the name field must NOT be narrower in side-by-side than
    it WOULD be in stacked at that same W.

Both sides are measured by rendering variant A at W and reading the name field
the screen actually computed — the counterfactual layout is forced by moving
STACK_BELOW_COLS, so the "would be" branch is the real code path, not algebra.

Usage:
    python scripts/prototype_resume_d16.py [W ...]
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REAL_STORE = Path(
    os.environ.get("LOCAL_OPERATOR_CONFIG_DIR") or (Path.home() / ".local-operator")
)

import scripts.probe_isolation  # noqa: F401,E402

from local_operator.tui.widgets import prototype_resume_variants as variants  # noqa: E402
from local_operator.tui.widgets.prototype_resume_data import PreviewData  # noqa: E402
from local_operator.tui.widgets.prototype_resume_picker import (  # noqa: E402
    PrototypeResumeApp,
)

#: 180 is the width round 2 never captured, and the one that proves the fix
#: holds ABOVE the breakpoint rather than only at it.
WIDTHS = (120, 140, 155, 160, 180, 200)


async def _measure(data: PreviewData, width: int, force_stacked: bool | None) -> dict[str, int]:
    """Render A at `width` and read back the field it computed.

    `force_stacked` overrides the breakpoint so the same geometry can be
    measured in BOTH layouts — that comparison is the whole invariant.
    """
    app = PrototypeResumeApp(data, "A", "")
    async with app.run_test(size=(width, 40)) as pilot:
        await pilot.pause()
        screen = app.screen
        if force_stacked is not None:
            original = variants.STACK_BELOW_COLS
            variants.STACK_BELOW_COLS = 10_000 if force_stacked else 0
            try:
                # Defeat the restyle guard, then settle: the pane widths the
                # name field is derived from are only correct after layout.
                screen._stacked = None
                screen._applied_pane_rows = None
                screen.refresh_view()
                await pilot.pause()
                screen.refresh_view()
                await pilot.pause()
            finally:
                variants.STACK_BELOW_COLS = original
        page = screen._rows[screen._top : screen._top + screen._drawn]
        name_w = screen._name_w()
        return {
            "name_w": name_w,
            "rows": screen._drawn,
            "truncated": sum(1 for row in page if len(row.name or row.id) > name_w),
        }


async def main(widths: tuple[int, ...]) -> int:
    data = PreviewData(REAL_STORE)
    print(f"NAME_MAX={variants.NAME_MAX}  STACK_BELOW_COLS={variants.STACK_BELOW_COLS}  ")
    print(f"split={variants.LIST_FR}fr list / {variants.PREVIEW_FR}fr preview\n")
    header = (
        f"{'W':>5} {'layout':>12} {'stacked_nw':>11} {'sbs_nw':>7} "
        f"{'drawn_nw':>9} {'rows':>5} {'trunc':>6} {'D16':>5}"
    )
    print(header)
    print("-" * len(header))
    failures = 0
    for width in widths:
        stacked = await _measure(data, width, True)
        sbs = await _measure(data, width, False)
        natural = await _measure(data, width, None)
        live = "stacked" if width < variants.STACK_BELOW_COLS else "side-by-side"
        # Only a width that ACTUALLY renders side-by-side can violate the
        # invariant; below the breakpoint the stacked field is what is drawn.
        ok = live == "stacked" or sbs["name_w"] >= stacked["name_w"]
        failures += not ok
        print(
            f"{width:>5} {live:>12} {stacked['name_w']:>11} {sbs['name_w']:>7} "
            f"{natural['name_w']:>9} {natural['rows']:>5} {natural['truncated']:>6} "
            f"{'OK' if ok else 'FAIL':>5}"
        )
    # Monotonicity as the user experiences it: drag the edge one column at a
    # time and the field must never shrink. The 139->140 cliff was invisible to
    # a spot check of round numbers, so this sweeps every column.
    print("\nmonotonicity sweep, 80..240 (the field must never shrink as W grows):")
    previous = 0
    regressions: list[str] = []
    for width in range(80, 241):
        drawn = await _measure(data, width, None)
        if drawn["name_w"] < previous:
            regressions.append(f"W={width}: {previous} -> {drawn['name_w']}")
        previous = drawn["name_w"]
    if regressions:
        failures += len(regressions)
        for line in regressions:
            print(f"  SHRANK  {line}")
    else:
        print("  0 shrink events across 161 widths")
    print(f"\n{'D16 HOLDS' if not failures else f'D16 VIOLATED ({failures})'}")
    return 1 if failures else 0


if __name__ == "__main__":
    args = tuple(int(a) for a in sys.argv[1:]) or WIDTHS
    raise SystemExit(asyncio.run(main(args)))
