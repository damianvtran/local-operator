#!/usr/bin/env python3
"""The fail-closed probe, driven through the REAL poll and the REAL predicate.

``server/retire.py``'s probe helpers used to fail OPEN — a raising ``stats()``
read as "no streams", a raising accessor as no sockets, a raising desktop probe
as an idle plane — so a probe that BROKE let the daemon exit under a live stream.
That is the failure the module's own rule forbids ("an exit under a live stream
is an interruption nobody undoes"), and review round 1 asked for the direction to
be pinned.

It cannot be provoked in an unprivileged real daemon: making ``stats()`` raise
inside a live process needs a fault injection the harness does not have. So this
drives the same code the lifespan starts — ``retire.retirement_poll`` over the
real ``in_flight`` predicate, a real ``DesktopSessions`` pool holding a REAL
attached bridge — with exactly one thing broken: the probe accessor. Everything
printed is the real predicate's answer, the real record write and the real log.

Run from anywhere; ``run.sh`` calls it, and it stands alone too:

    .venv/bin/python docs/evidence/serve-build-retire/inject.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from local_operator import buildwatch
from local_operator import update as update_mod
from local_operator.server import registry as serve_registry
from local_operator.server import retire
from local_operator.server.registry import ServeRecord
from local_operator.server.utils.desktop_sessions import DesktopSessions
from local_operator.update import BuildStamp

OLD = BuildStamp(version="0.54.39", source_ref="1111111")
NEW = BuildStamp(version="0.54.39", source_ref="2222222")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main() -> int:
    with TemporaryDirectory() as raw:
        root = Path(raw)
        state = {"build": OLD}

        # The same two fakes tests/unit/session/runtime's refresh tests use:
        # "the install on disk" and "how long its marker has sat there".
        update_mod.installed_build = lambda *_a, **_k: state["build"]  # type: ignore[assignment]
        update_mod.build_marker_age_s = lambda *_a, **_k: 999.0  # type: ignore[assignment]
        buildwatch.BUILD_CHECK_S = 0.2  # test-only; the daemon uses 5.0 s
        buildwatch.BUILD_STAGGER_S = 0.0

        # A REAL desktop pool with a REAL held bridge: the plane really is busy
        # while the probe that would report it cannot be read.
        pool = DesktopSessions(root)
        session_id = await pool.create(str(root))
        context = pool.session(session_id)
        bridge = await context.__aenter__()
        print(f"real desktop bridge held: session={session_id} users={bridge.users}")

        record = ServeRecord(
            pid=4242,
            host="127.0.0.1",
            port=53421,
            instance_id="injected",
            version=OLD.version,
            source_ref=OLD.source_ref,
            prefix=str(root),
            install_kind="uv-tool",
            desktop=True,
        )
        publisher = serve_registry.publisher(record, root=root)

        broken = {"on": True}
        honest = type(pool).in_flight_reason

        def probe() -> str | None:
            if broken["on"]:
                raise OSError("the desktop probe could not be read")
            return honest(pool)

        pool.in_flight_reason = probe  # type: ignore[method-assign]

        class App:
            state = type("S", (), {})()

        app = App()
        app.state.desktop_sessions = pool  # type: ignore[attr-defined]
        exits: list[bool] = []
        stop = asyncio.Event()
        task = asyncio.create_task(
            retire.retirement_poll(
                app, publisher, stop=stop, exit_process=lambda: exits.append(True), boot=OLD
            )
        )

        t0 = time.monotonic()
        state["build"] = NEW  # the install moves under the daemon
        for _ in range(200):
            if record.retiring_to:
                break
            await asyncio.sleep(0.02)
        print(
            f"t={time.monotonic() - t0:5.3f}s record announced: "
            f"retiring_from={record.retiring_from!r} retiring_to={record.retiring_to!r}"
        )

        await asyncio.sleep(1.0)  # several check intervals with the probe broken
        print(
            f"t={time.monotonic() - t0:5.3f}s with the probe UNREADABLE: "
            f"latched={retire.retiring(app)} exits={len(exits)} "
            f"poll_alive={not task.done()}"
        )
        print(f"          verdict(probe broken): {retire.in_flight(app)!r}")
        assert record.retiring_to == NEW.label(), "the announcement must still be written"
        assert retire.retiring(app) is False, "an unreadable probe must not latch"
        assert exits == [], "an unreadable probe must not release the daemon"
        assert not task.done(), "an unreadable probe must not kill the poll"
        print("          FAIL-CLOSED: announced, still serving, not latched, NOT exited")

        broken["on"] = False  # the probe reads again; the plane is still attached
        await asyncio.sleep(1.0)
        print(
            f"t={time.monotonic() - t0:5.3f}s with the probe readable again: "
            f"reason={retire.in_flight(app)!r} exits={len(exits)}"
        )
        assert exits == [], "a live desktop attachment must still hold the daemon"

        await context.__aexit__(None, None, None)  # the viewer lets go
        await asyncio.wait_for(task, 5.0)
        print(
            f"t={time.monotonic() - t0:5.3f}s after the viewer lets go: "
            f"latched={retire.retiring(app)} exits={len(exits)}"
        )
        assert exits == [True], "the daemon must still retire once the drain is empty"
        await pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
