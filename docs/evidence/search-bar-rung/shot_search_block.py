"""Capture one Search-spend frame from a chosen local-operator tree.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python shot_search_block.py \\
        <repo-root> <out-dir> {session,analytics} <terminal-width> [height]

Written for the #1073 before/after pair. ``repo-root`` is explicit because the
BEFORE frames have to come from a throwaway worktree carrying the pre-fix
``analytics_panel.py`` while the harness and the ledger stay identical on both
sides; this script self-corrects its own imports off ``repo-root`` rather than
off the venv it is run with (see AGENTS.md, "Every feature worktree owns its own
venv"), so the same file drives both trees.

The ledger is fixed and shared by both screens -- 5 free ``duckduckgo``
searches, 3 free ``deepseek:read`` reads, 1 priced ``brave`` search -- plus one
analytics-store snapshot for session ``sess``, which is what makes ``/analytics``
draw the ``This session`` reference row whose 43-cell rung the issue is about.
The store file is REMOVED before every run: an accumulating ledger would make
each frame a different ledger and the pair incomparable.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO, OUT, SCREEN, WIDTH = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
HEIGHT = int(sys.argv[5]) if len(sys.argv) > 5 else 60

sys.path.insert(0, REPO)

import scripts.probe_isolation  # noqa: E402,F401

# Statement, not a blank line: it holds the two import blocks apart, so isort
# cannot sort ``local_operator`` above the probe. The probe re-homes HOME and the
# config dir ON IMPORT -- the incident it exists for (PR #645) was an ad-hoc
# probe that imported the app first and ran a migration against the operator's
# live config -- so the ordering here is load-bearing.
assert "local_operator" not in sys.modules, "the isolation probe must import first"

import local_operator.analytics.store as store_mod  # noqa: E402
from local_operator.analytics.store import AnalyticsStore  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.web_search.cost import SEARCH_SPEND  # noqa: E402
from local_operator.web_search.models import SearchCost  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.analytics.test_store import _snap  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession  # noqa: E402
from tests.unit.tui.test_band_panels import _async_factory  # noqa: E402
from tests.unit.tui.test_cost_aggregation import _settle_boot  # noqa: E402
from tests.unit.tui.test_slash_echo import _submit  # noqa: E402


def seed(db: Path) -> None:
    SEARCH_SPEND.reset()
    db.unlink(missing_ok=True)
    store_mod.default_db_path = lambda: db
    store = AnalyticsStore(db)
    store.record_batch([replace(_snap(session_id="sess"), request_id="req")])
    store.close()
    for _ in range(5):
        SEARCH_SPEND.record("sess", "duckduckgo", SearchCost(usd=0.0, basis="free"))
    for _ in range(3):
        SEARCH_SPEND.record("sess", "deepseek:read", SearchCost(usd=0.0, basis="free"), kind="read")
    SEARCH_SPEND.record("sess", "brave", SearchCost(usd=0.0069, basis="per-search rate"))


async def main() -> None:
    # A scratch store: it is an input to the frame, not an artifact of it.
    seed(Path(tempfile.mkdtemp()) / "ledger.db")
    session = FakeSession()
    app = OperatorApp(_async_factory(session))
    async with app.run_test(size=(WIDTH, HEIGHT)) as pilot:
        await _settle_boot(pilot, app, session)
        await _submit(pilot, app, "/analytics" if SCREEN == "analytics" else "/session")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()
        save_capture(app, OUT / f"{SCREEN}-{WIDTH}x{HEIGHT}.svg")


asyncio.run(main())
