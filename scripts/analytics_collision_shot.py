"""Capture the /analytics per-session table at a COLLISION GROUP.

Usage: python scripts/analytics_collision_shot.py OUTDIR 140x45

Round-1 review found the original evidence only captured the TOP of the table,
where the expensive uniquely-named sessions sit — so the collisions, which live
further down among the sibling subagent rows, sat below the fold and the "after"
frame looked clean. This script scrolls to the affected region and captures it,
which is the only frame that shows what the fix does.

The fixture is a REAL ledger shape reduced to a readable frame: a handful of
uniquely-named parents plus one large group of sibling subagents that compose a
byte-identical label (``reviewer · Article-search-svc schema review``), which is
exactly what the backfill mints. Copy this script into a preserved base worktree
to capture the pre-fix frame with the SAME fixture. No provider request, live
session or operator config is used.
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

import scripts.probe_isolation  # noqa: E402, F401
from local_operator.analytics.store import AnalyticsStore, default_db_path  # noqa: E402
from local_operator.harness.types import ModelSpec  # noqa: E402
from local_operator.session.frontend_state import FrontendSessionState  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402

#: Uniquely-named sessions, as (session_id, name, cost_usd). These are the rows
#: the ORIGINAL evidence captured: they sit at the top by spend and they render
#: correctly both before and after, which is why a top-of-table frame could not
#: show the bug.
UNIQUE_ROWS = [
    ("8f21ac93bb04", "OSWorld benchmark evaluation and scoring", 3433.96),
    ("2b77de10c9a5", "Article search service integration", 2957.68),
    ("d40c81ff7a13", "Revise Agent Risk Assessments PDF export", 2950.37),
]

#: The COLLISION GROUP: sibling subagents delegated by one parent under one
#: role, so the backfill composes the identical label for every one of them.
#: This is the shape that made 355 sessions vanish on the real ledger.
COLLIDING_LABEL = "reviewer · Article-search-svc schema review"
COLLIDING_ROWS = [
    ("4fa0e21c7b93", 11.60),
    ("d677b3f9012a", 10.85),
    ("1a24c8e70bf6", 7.47),
    ("2c39ab5410de", 6.78),
    ("4a5e77c2b901", 6.70),
    ("9b13fe4a2c78", 5.94),
    ("c802d51b3ae6", 5.31),
    ("71c9042fbd85", 4.88),
]


class LedgerSession(FakeSession):
    frontend_state: FrontendSessionState

    @property
    def session_id(self) -> str:
        return "a3f9c21b7e40"

    @property
    def model(self) -> ModelSpec:
        return ModelSpec(provider="anthropic", model_id="claude-sonnet-4-6", context_window=200000)

    @property
    def model_label(self) -> str:
        return "anthropic/claude-sonnet-4-6"


def seed(store: AnalyticsStore) -> None:
    """Write the fixture into the isolated ledger, names included."""
    snapshots = []
    index = 0
    rows: list[tuple[str, str, float]] = list(UNIQUE_ROWS)
    rows += [(sid, COLLIDING_LABEL, usd) for sid, usd in COLLIDING_ROWS]
    for session_id, name, usd in rows:
        # A couple of calls per session so `calls` is not uniformly 1; the
        # dollar figure is what this frame is read for.
        for part in range(2):
            snapshots.append(
                replace(
                    _snap(
                        session_id=session_id,
                        provider="anthropic",
                        model_id="claude-sonnet-4-6",
                        context=120_000 + index * 900,
                        input_tokens=40_000,
                        cache_read=70_000,
                        cache_write=2_000,
                        output_tokens=6_000,
                        reasoning=1_200,
                        cost_micro=int(usd * 1_000_000 / 2),
                        chars={"conversation": 4000, "tool_results": 2200},
                        ts_ms=1788602400000 + index * 61000,
                        ok=True,
                    ),
                    request_id=f"req-{index:04d}-{part}",
                    purpose="turn",
                    outcome="ok",
                    duration_ms=2400,
                )
            )
            index += 1
        store.upsert_session_name(session_id, name)
    store.record_batch(snapshots)


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cols, rows_ = sys.argv[2].split("x")
    size = (int(cols), int(rows_))

    store = AnalyticsStore()
    seed(store)
    store.close()

    session = LedgerSession()
    session.set_conversation_name("Analytics session naming")
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="capture",
        generation=1,
        context_tokens=28400,
        context_is_estimate=False,
        context_window=200000,
    )
    ledger = default_db_path()
    before_digest = hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        await _submit(pilot, app, "/analytics")
        await app.workers.wait_for_complete()
        await pilot.pause()
        # The frame the original evidence stopped at: the top of the table.
        save_capture(app, str(out / "top.svg"))

        screen = app.screen
        body = ""
        # Scroll to the BOTTOM, where the sibling rows live. `end` is the
        # binding the screen exposes for exactly this.
        await pilot.press("end")
        await pilot.pause()
        await pilot.pause()
        save_capture(app, str(out / "collisions.svg"))

        # The screen's own accessor for "what a user reads", so the metrics
        # describe the rendered frame rather than a re-derived one. Reached
        # through getattr because ``app.screen`` is typed as the base
        # ``Screen``, which does not declare the analytics screen's helpers.
        render_lines = getattr(screen, "render_lines_for_test", None)
        body = "\n".join(render_lines()) if render_lines is not None else ""
        scroll = getattr(screen, "_scroll", None)
        session_block = body.split("By session", 1)[-1]
        table_rows = [ln.strip() for ln in session_block.splitlines() if "tokens" in ln]
        metrics = {
            "source": str(Path(__file__).resolve().parents[1]),
            "size": size,
            "screen": type(screen).__name__,
            "session_rows_rendered": len(table_rows),
            "session_rows_seeded": len(UNIQUE_ROWS) + len(COLLIDING_ROWS),
            "distinct_rendered_rows": len(set(table_rows)),
            "colliding_group_size": len(COLLIDING_ROWS),
            "colliding_rows_on_screen": sum(
                1 for ln in table_rows if ln.startswith("reviewer · Article")
            ),
            "ledger_unchanged": before_digest
            == (hashlib.sha256(ledger.read_bytes()).hexdigest() if ledger.exists() else None),
            "scroll": (
                {
                    "size": list(scroll.size),
                    "virtual_size": list(scroll.virtual_size),
                    "max_y": scroll.max_scroll_y,
                }
                if scroll
                else None
            ),
            "rows": table_rows,
        }
        (out / "result.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "rows"}, indent=2))
        for row in table_rows:
            print("   ", row)


if __name__ == "__main__":
    asyncio.run(main())
