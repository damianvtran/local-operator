"""The TUI must not get slower the longer it runs with the sidebar open.

The operator reported two symptoms that ``/reload`` cures: the TUI slows down
after running a long time with the session sidebar open, and sidebar switches
that are fast at first get gradually slower with switch count. A root-cause
pass on the assembled app (real ``OperatorApp`` under ``run_test``, real
in-process runtime owners, real ``RemoteSession`` viewers over loopback) found
three independent accumulations rather than one leak:

* **F1** — ``_report_startup_cleanup`` self-schedules a 1 s recheck chain for
  30 s, and ``_adopt_session`` seeded a fresh chain on EVERY adoption, which a
  sidebar switch re-enters. The chains did not know about each other, so live
  timers grew 9 → 60 over 50 switches and the callback ran 6462 times over
  150 — each a disk read on the event loop, each timer a live asyncio task.
* **F2** — ``RETAINED_PRESENTATIONS`` was 4, smaller than a real working set,
  so a user who had touched more than four conversations paid a cold rebuild
  (connect, window, replay, MOUNT, layout wait, teardown) on most switches.
* **F3** — ``_prewarm_sidebar`` admitted a candidate and then immediately
  evicted an older presentation to respect the same bound, making the evicted
  session a candidate again on the next 2 s poll: a socket connected and
  disposed per poll, forever, whenever live sessions exceeded the bound.

The assertions here are STRUCTURAL — timer counts, socket connects, cache
hits — never wall-clock or CPU bounds, per AGENTS.md "Prefer a structural
invariant to a numeric one". Each was run against the pre-fix tree and
failed there; the PR carries that proof.
"""

from __future__ import annotations

import pytest

from local_operator.tui.app import (
    STARTUP_CLEANUP_RECHECK_WINDOW_S,
    OperatorApp,
)
from tests.e2e.harness import wait_for_adoption
from tests.unit.tui.test_app_pilot import FakeSession, _factory


def _cleanup_timers(app: OperatorApp) -> list[object]:
    """Every live Textual timer whose callback is the startup-cleanup recheck.

    Named by callback rather than counted in aggregate so the assertion is
    about the site (F1) and cannot be satisfied or broken by an unrelated
    interval timer coming or going.
    """
    return [
        timer
        for timer in list(app._timers)
        if "_report_startup_cleanup" in repr(getattr(timer, "_callback", None))
    ]


@pytest.mark.asyncio
async def test_the_startup_cleanup_recheck_chain_does_not_accumulate_across_adoptions() -> None:
    """Re-adopting N times leaves at most ONE recheck chain alive, not N.

    ``_adopt_session`` is exactly what a sidebar switch calls, so this is the
    switch-count growth stated at its site. The pre-fix tree fails with 21
    live chains after 20 adoptions.

    The bound is ``<= 1`` rather than ``== 1``: the chain is a one-shot that
    re-arms itself from inside its own callback, so between a tick firing and
    the next ``set_timer`` there is momentarily no live timer, and a count of
    exactly one would race that gap. What must never be true is "more than
    one", which is the accumulation.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await wait_for_adoption(app, pilot)
        await pilot.pause()
        # PRECONDITION: the boot adoption armed the chain at all, so the
        # assertion below is about overlap and not about a chain that never
        # starts (which would also pass, vacuously).
        assert len(_cleanup_timers(app)) == 1, "boot did not arm the recheck chain"
        assert STARTUP_CLEANUP_RECHECK_WINDOW_S > 0
        timers_before = len(list(app._timers))

        for _ in range(20):
            app._adopt_session(FakeSession(), replay_history=False)
            await pilot.pause()

        live = _cleanup_timers(app)
        assert len(live) <= 1, (
            f"{len(live)} startup-cleanup recheck chains alive after 20 adoptions: "
            "each adoption seeded a new chain without stopping the previous one"
        )
        # And the app's whole timer census did not grow with adoption count —
        # the symptom the operator sees, stated once at the aggregate so a
        # second per-adoption repeater added later is caught here too.
        timers_after = len(list(app._timers))
        assert timers_after <= timers_before, (
            f"app timers grew {timers_before} -> {timers_after} over 20 adoptions"
        )
