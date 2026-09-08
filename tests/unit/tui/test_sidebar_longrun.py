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

import asyncio
import contextlib
import json
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.session.runtime.owned import OwnedSessionHandle
from local_operator.session.runtime.server import RuntimeServer
from local_operator.tui.app import (
    RETAINED_PRESENTATIONS,
    STARTUP_CLEANUP_RECHECK_WINDOW_S,
    OperatorApp,
)
from tests.e2e.harness import (
    ScriptedStream,
    assistant_message,
    build_session,
    seed_transcript,
    user_message,
    wait_for_adoption,
)
from tests.unit.harness.test_comms import DEADLOCK_GUARD_S, MAX_PUMP_TURNS
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


# --- Assembled-app fixture: N live in-process owners behind one sidebar ------


@asynccontextmanager
async def live_owners(
    config: Path, count: int, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[list[str], Callable[[str], Awaitable[RemoteSession]]]]:
    """``count`` real ``RuntimeServer`` owners, each discoverable by the catalog.

    Production runs one session per process, so ``registry.publish`` keys the
    discovery record by PID. Every owner here shares this test's PID, so
    without re-keying, N servers overwrite ONE file, the catalog reports a
    single live row, prewarm never selects anything, and every assertion
    below passes vacuously against a cold path. Keyed by ``(pid, session_id)``
    instead; ``scan`` globs ``*.json`` and checks ``pid_alive``, so the
    records stay truthful for all of them.
    """
    from local_operator.session.runtime import registry

    def publish(record: Any, root: Path | None = None) -> Path:
        directory = registry.run_dir(root)
        record.heartbeat_at = time.time()
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".x.", suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            json.dump(record.to_json(), handle)
        target = directory / f"{record.pid}-{record.session_id}.json"
        os.replace(tmp, target)
        return target

    monkeypatch.setattr(registry, "publish", publish)
    monkeypatch.setattr(registry, "unpublish", lambda pid, root=None: None)

    servers: dict[str, RuntimeServer] = {}
    handles: list[OwnedSessionHandle] = []
    ids: list[str] = []
    try:
        for i in range(count):
            sid = f"longrun{i:02d}"
            ids.append(sid)
            directory = config / "sessions" / sid
            await seed_transcript(
                directory,
                [user_message(f"{sid} question"), assistant_message(f"{sid} saved answer")],
            )
            owner = build_session(directory, ScriptedStream([]), cwd=config)
            handle = OwnedSessionHandle(owner, asyncio.get_running_loop(), cwd=str(config))
            handles.append(handle)
            server = RuntimeServer(handle, kind="daemon")
            await server.start_in_process()
            servers[sid] = server

        def find(_directory: Path, sid: str) -> tuple[Any, Any]:
            server = servers.get(sid)
            return (server._record, server._record.pid) if server else (None, None)

        async def never() -> Any:
            raise AssertionError("view navigation must never take execution ownership")

        async def resume(sid: str) -> RemoteSession:
            return await RemoteSession.connect(
                servers[sid]._record,
                sid,
                config_dir=config,
                takeover_factory=never,
                display_window=True,
            )

        monkeypatch.setattr("local_operator.mobile.attach_client.find_owner_record", find)
        monkeypatch.setattr(OperatorApp, "_check_for_update", lambda self: None)
        yield ids, resume
    finally:
        for server in servers.values():
            await server.aclose()
        for handle in handles:
            await handle.dispose()


async def one_poll(app: OperatorApp, pilot: Any) -> None:
    """One catalog poll + the prewarm it launches, driven to completion.

    Waits on the app's own state (``_sidebar_refresh_pending`` cleared, then
    the prefetch worker's ``wait()``) rather than on the clock, per AGENTS.md
    "Wait on the event, never on the clock".
    """
    app._refresh_sidebar()
    for _ in range(MAX_PUMP_TURNS):
        if not app._sidebar_refresh_pending:
            break
        await pilot.pause()
    else:
        raise AssertionError("the catalog refresh never settled")
    prefetch = app._sidebar_prefetch
    if prefetch is not None:
        await asyncio.wait_for(asyncio.shield(prefetch.wait()), DEADLOCK_GUARD_S)
    await pilot.pause()


async def one_switch(app: OperatorApp, sid: str) -> None:
    await asyncio.wait_for(app._sidebar_navigation.select(sid), DEADLOCK_GUARD_S)
    connection = app._interaction.connection_task
    if connection is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.shield(connection), DEADLOCK_GUARD_S)


@pytest.mark.asyncio
async def test_alternating_within_the_working_set_hits_the_cache_every_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user alternating between N <= bound conversations never pays a rebuild.

    F2, stated structurally: after every conversation has been visited once,
    each further switch takes the reveal path (a hit in
    ``_sidebar_presentation_current``) rather than ``_prepare_sidebar_session``'s
    rebuild. Eight conversations is the smallest working set the operator's
    report describes; the pre-fix bound of 4 fails this with 12/32 hits.

    Counted at the acceptance predicate, not by timing: a rebuild that got
    faster would still be a rebuild.
    """
    config = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    n = 8
    assert n <= RETAINED_PRESENTATIONS, "the working set must fit the bound for this claim"
    async with live_owners(config, n, monkeypatch) as (ids, resume):
        hits = 0
        original = OperatorApp._sidebar_presentation_current

        def counted(cached: Any, source: Any, gate: Any) -> bool:
            nonlocal hits
            ok = original(cached, source, gate)
            hits += int(ok)
            return ok

        monkeypatch.setattr(OperatorApp, "_sidebar_presentation_current", staticmethod(counted))
        app = OperatorApp(lambda: resume(ids[0]), resume_factory=resume)
        async with app.run_test(size=(120, 36)) as pilot:
            await wait_for_adoption(app, pilot)
            app._set_sidebar_open(True)
            assert app._sidebar_timer is not None
            app._sidebar_timer.pause()  # the test drives polls; no background prewarm races
            # First lap: every conversation visited once, so each is either
            # parked or the live one. Misses here are expected and not counted.
            for sid in ids[1:]:
                await one_switch(app, sid)
            hits = 0
            switches = 0
            # Two more laps: every switch must reveal a parked presentation.
            for _ in range(2):
                for sid in ids:
                    if sid == app._session.session_id:
                        continue
                    await one_switch(app, sid)
                    switches += 1
            assert switches >= 2 * (n - 1)
            assert hits == switches, (
                f"{switches - hits} of {switches} switches over a working set of {n} rebuilt "
                f"the presentation instead of revealing the parked one "
                f"(RETAINED_PRESENTATIONS={RETAINED_PRESENTATIONS})"
            )
            assert len(app._sidebar_presentations) == n - 1
