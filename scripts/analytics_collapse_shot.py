"""Capture the ``/analytics`` session table's collapsed/expanded states.

Usage: python scripts/analytics_collapse_shot.py OUTDIR

Sibling of ``analytics_width_band_shot.py`` (does a row FIT?) and
``analytics_label_shot.py`` (what does a row SAY?). This one is about how much
of the table is on screen at all, and it exists for the collapsible-rows change:
a session with 90 subagents used to paint all 91 rows whether or not anyone
wanted them, and the frames here are what shows that the default view is now the
roots alone and that expanding one brings its children back.

Runs against the SAME script on both sides of the change, deliberately. On the
pre-change code the expand keys do not exist, so the "collapsed" and "expanded"
frames come out identical — which is the before-state stated as an artifact
rather than as a claim. Three consecutive captures per state, so a post-paint
reflow shows up as two differing SVGs instead of being averaged away by one
lucky shot.

The fixture is seeded rather than read from a real ledger: the frames have to
show a root with MANY children beside a root with NONE (the two shapes whose
rendering differs), and a real ledger cannot be relied on to put both near the
top of a cost-sorted table. Magnitudes follow ``analytics_width_band_shot``'s
lead — real-sized call counts, because a fixture whose numbers are all toy-sized
is evidence only about toy data.

No provider request, live session or operator config is touched:
``probe_isolation`` re-homes HOME and the config dir before any application
import, and every ``CMUX_*`` identifier is cleared first (a headless pilot that
inherits ``CMUX_WORKSPACE_ID`` renames the operator's real cmux workspaces).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Clear every multiplexer identifier BEFORE any application import (see above).
for _key in tuple(os.environ):
    if _key.startswith("CMUX_"):
        os.environ.pop(_key)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from dataclasses import replace  # noqa: E402

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.analytics_panel import AnalyticsScreen  # noqa: E402
from scripts.analytics_label_shot import LedgerSession, _svg_rows  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: ``(session_id, name, usd, calls, parent_id)``. Ordered by spend because the
#: table is, so the frame's first rows are these rows.
#:
#: The shape matters more than the numbers. Row 1 is an expensive root with a
#: dozen subagents — the case the user reported, scaled down to what a 40-row
#: frame can show at once. Row 2 is an expensive root with NONE, which is the
#: majority case on the real ledger (492 of 595 roots are childless) and the one
#: that must not grow a disclosure glyph or otherwise look interactive.
FIXTURE: list[tuple[str, str, float, int, str]] = [
    ("8f21ac93bb04", "Toggleable Sidebar for Session Switching in the TUI", 900.00, 4210, ""),
    ("2b77de10c9a5", "Improve Support for Local Model Providers", 640.00, 3180, ""),
    ("d40c81ff7a13", "Harden Against Credential Leak Paths in the runner", 420.00, 812, ""),
    ("6b1d0e97ca42", "eval osworld/chrome-0421 · claude-sonnet-4-6", 310.00, 96, ""),
    ("c19b4f7d3a08", "routine background polling", 120.00, 41, ""),
    ("98bfe7686ffe", "", 60.00, 8, ""),
]

#: Subagent children of the first root: the "hundreds of rows of subagent cost"
#: the report is about. Twelve is enough to overflow the frame from ONE root
#: while leaving the childless roots visible beneath it in the collapsed shot.
CHILD_ROLES = [
    "coder",
    "reviewer",
    "qa-tester",
    "designer",
    "architect",
    "ux-reviewer",
    "scout",
    "manager",
    "coder",
    "reviewer",
    "qa-tester",
    "scout",
]
CHILD_PARENT = FIXTURE[0][0]
#: A second, shallower parent, so the frame shows that "has children" is not a
#: property of the single most expensive row.
SECOND_PARENT = FIXTURE[2][0]
SECOND_CHILDREN = ["reviewer", "qa-tester"]


def _children() -> list[tuple[str, str, float, int, str]]:
    rows: list[tuple[str, str, float, int, str]] = []
    for index, role in enumerate(CHILD_ROLES):
        rows.append(
            (
                f"c{index:02d}a1b2c3d4e5"[:12],
                f"{role} · Toggleable Sidebar for Session Switching",
                42.00 - index,
                120 + index * 7,
                CHILD_PARENT,
            )
        )
    for index, role in enumerate(SECOND_CHILDREN):
        rows.append(
            (
                f"d{index:02d}f6e5d4c3b2"[:12],
                f"{role} · Harden Against Credential Leak Paths",
                18.00 - index,
                64 + index * 5,
                SECOND_PARENT,
            )
        )
    return rows


def seed(store: AnalyticsStore) -> None:
    """Write the fixture into the isolated ledger, parent edges included.

    ``UsageAggregate.calls`` is ``COUNT(*)`` over the ledger — there is no field
    to set — so a row's call count costs that many inserts. Spend and tokens are
    therefore divided across a row's calls, keeping the rendered ``$`` figures
    fixed while the call counts stay real-sized.
    """
    snapshots = []
    for index, (session_id, name, usd, calls, parent) in enumerate(FIXTURE + _children()):
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
                    parent_session_id=parent,
                )
            )
        if name:
            store.upsert_session_name(session_id, name)
    store.record_batch(snapshots)


def _make_session() -> LedgerSession:
    session = LedgerSession()
    session.set_conversation_name("Analytics collapsible sessions")
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


async def _capture(app, pilot, out: Path, state: str) -> dict[str, object]:
    """Three consecutive frames of one state, plus what they actually contain."""
    digests = []
    for shot in range(3):
        path = out / f"{state}-frame{shot}.svg"
        save_capture(app, str(path))
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        await pilot.pause()
    rows = [line for line in _svg_rows(out / f"{state}-frame0.svg") if " tokens" in line]
    return {
        "state": state,
        "frames_identical": len(set(digests)) == 1,
        "session_rows_on_screen": len(rows),
        "rows": rows,
    }


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    store = AnalyticsStore()
    seed(store)
    store.close()

    ledger = default_db_path()
    before = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None

    app = OperatorApp(lambda: _factory_for(_make_session()))
    states: list[dict[str, object]] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()

        screen = app.screen
        assert isinstance(screen, AnalyticsScreen), f"expected /analytics, got {type(screen)}"

        # Page down to the session table, which sits below the totals, the two
        # charts and the input attribution.
        for _ in range(4):
            await pilot.press("pagedown")
            await pilot.pause()
        await pilot.pause()
        states.append(await _capture(app, pilot, out, "collapsed"))

        # Move the cursor onto the first expandable row and open it. On the
        # pre-change code these keys do not exist, so the frames repeat the
        # previous state — which is the point of running one script on both sides.
        #
        # TWO presses, deliberately: the first Enter PLACES the cursor (there is
        # no "that row" to expand until one exists) and the second acts on it.
        # A single press captured the cursor arriving on a still-collapsed row,
        # which is a real state but not the one this frame is for.
        await pilot.press("home")
        await pilot.pause()
        for _ in range(4):
            await pilot.press("pagedown")
            await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        states.append(await _capture(app, pilot, out, "expanded"))

        # A repaint must not silently close what the user opened: ``t`` rebuilds
        # the whole report, so it is the cheapest proof that expansion state
        # survives a rebuild.
        await pilot.press("t")
        await pilot.pause()
        await pilot.pause()
        states.append(await _capture(app, pilot, out, "expanded-after-toggle"))

    after = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None
    report = {
        "source": str(Path(__file__).resolve().parents[1]),
        "ledger_unchanged_by_capture": before == after,
        "states": states,
    }
    (out / "states.json").write_text(json.dumps(report, indent=2) + "\n")

    for state in states:
        print(
            f"{state['state']:<24} rows_on_screen={state['session_rows_on_screen']:>3} "
            f"frames_identical={state['frames_identical']}"
        )
    print("ledger unchanged by capture:", report["ledger_unchanged_by_capture"])


if __name__ == "__main__":
    asyncio.run(main())
