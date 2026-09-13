"""The mount engage's bounded failure, with a real child that constructs slowly.

**What this covers that the unit suite cannot.** Every existing mount-engage
test stubs ``engage_runtime``, so none of them ever runs the thing whose
ceiling is the defect: the real engage loop, its 30 s deadline, its poll
regime, and a REAL candidate process that is alive but has not published a
record when the deadline expires. That is the measured shape on the operator's
machine, and it is where the band's `starting…` went away leaving nothing
behind — ``session.mcp_startup`` was unset because MCP never got that far, and
the failure was logged at DEBUG only, so the screen and the log file both said
nothing at all.

The app behaved as designed ("the real prompt reports the failure"). What the
design missed is that the prompt is not the only way out of the wait: a user
who reads the band, watches it clear, and then types a letter gets the band
re-armed through ``_warm_runtime_for_draft`` and reads the whole thing as
permanent. So the assertion here is not merely "the band clears in bounded
time" — it is that the transcript says WHICH bounded thing happened.

Two deviations from production, both named so a reviewer can weigh them:

* the candidate is a wrapper that sleeps before exec'ing the real runtime
  entry point, so the slow construction is deterministic rather than a matter
  of host load. It is spawned by ``_spawn_runtime``'s replacement with
  production's own argv shape, environment and ``start_new_session``;
* the wrapper's stdio goes to ``DEVNULL`` rather than a capture file. Nothing
  under test reads that file unless the candidate DIES, and this one is alive
  throughout — the property the engage branches on.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

#: How long the wrapped candidate sleeps before it execs the real runtime entry
#: point. Past the engage's own ``DEFAULT_DEADLINE_S`` (30 s) with margin,
#: which is what makes the failure deterministic instead of a race: the
#: engagement gives up while the candidate it spawned is still alive.
CONSTRUCT_DELAY_S = 40.0

#: The band must be clear by here. Above the deadline (measured 30.0-31.5 s on
#: this host) plus the app's own dispatch and unwind, and well below the delay
#: above — a band still up at 35 s is waiting on something other than the
#: bounded engage, which is the class of bug this budget exists to catch.
BAND_CLEAR_BUDGET_S = 35.0

#: How long to give the mount engage to appear at all. Not a performance
#: claim: the engage is dispatched from adoption, and this only has to outlast
#: the app's own boot under a loaded CI runner.
ENGAGE_START_BUDGET_S = 20.0

#: The notice the app now owes the user when a warm engage gives up. Matched on
#: the half that names the FAILURE rather than the half that names the next
#: step, so the copy can be reworded without silently un-pinning the
#: behaviour — but not so loosely that any notice would satisfy it.
FAILURE_NOTICE_FRAGMENT = "no runtime came up for this session"

#: The candidate the engage spawns, in place of the runtime entry point.
#:
#: A ``-c`` program rather than a module dropped on ``PYTHONPATH``: the spawn
#: passes ``-P`` (``interpreter.SAFE_PATH_FLAG``), so a wrapper file on disk
#: would not be importable and the candidate would die instead of sleeping —
#: a DIFFERENT failure, reported by ``RuntimeStartupError`` rather than by the
#: deadline. Exec'ing the real module afterwards keeps the wrapper honest about
#: what it stands in for: it is the real runtime, just late.
_WRAPPER = (
    "import runpy, sys, time\n"
    "time.sleep(float(sys.argv[1]))\n"
    "sys.argv = [sys.argv[2]]\n"
    'runpy.run_module("local_operator.session.runtime.process", run_name="__main__",'
    " alter_sys=True)\n"
)


def _configure_provider(config_dir: Path) -> None:
    """Make the isolated config look like a configured machine.

    The mount engage is gated on this (``OperatorApp._runtime_can_start``): an
    empty config is the onboarding screen, where the engage must NOT fire. The
    child this spawns is the sleeper, so the provider is never dialled — it
    only has to be NAMED.
    """
    from local_operator.config import ConfigManager

    ConfigManager(config_dir=config_dir).update_config({"hosting": "test", "model_name": "mock"})


@pytest.fixture
def slow_runtime_child(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    """Replace ``_spawn_runtime`` with the same process, born late.

    Returns the list of candidates so the test can terminate them: the engage
    deliberately does not kill a candidate it gives up waiting for ("what is
    surrendered is the waiting, not the runtime"), so a test that never reaped
    them would leave sleeping children behind.
    """
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.session.runtime import launch

    spawned: list[subprocess.Popen[bytes]] = []

    def spawn(
        session_id: str,
        cwd: str,
        *,
        defer_materialise: bool,
        initial_model: Any = None,
        model_selection_override: bool = False,
    ) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        # Production's routing environment, built the same way: the wrapper
        # execs the real entry point, which reads these.
        env["LOP_MOBILE_CHILD_CWD"] = cwd
        env["LOP_MOBILE_CHILD_RESUME"] = session_id
        if defer_materialise:
            env["LOP_RUNTIME_DEFER_MATERIALISE"] = "1"
        else:
            env.pop("LOP_RUNTIME_DEFER_MATERIALISE", None)
        process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [
                sys.executable,
                SAFE_PATH_FLAG,
                "-c",
                _WRAPPER,
                str(CONSTRUCT_DELAY_S),
                "local_operator.session.runtime.process",
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        spawned.append(process)
        return process

    monkeypatch.setattr(launch, "_spawn_runtime", spawn)
    return spawned


@pytest.fixture
def reap_candidates(slow_runtime_child: list[subprocess.Popen[bytes]]) -> Any:
    """Terminate every candidate the engage spawned, however the test ended."""
    yield
    for process in slow_runtime_child:
        with contextlib.suppress(Exception):
            process.terminate()
    for process in slow_runtime_child:
        with contextlib.suppress(Exception):
            process.wait(timeout=5)


@pytest.mark.asyncio
async def test_a_slow_runtime_start_clears_the_band_and_says_what_happened(
    headless_tui_env: Path, workspace: Path, reap_candidates: Any
) -> None:
    """The band's two wait states must stop rendering identically.

    On the defect this test fails at the last assertion, with an empty
    transcript and ``is_cold`` still True — the exact frame pair the diagnosis
    captured (`band_mid_engage.svg` during the wait, `construct40.svg` after
    it, differing only in the `starting…` segment being gone).
    """
    from local_operator.session.attached import AttachedSession
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.transcript import NoticeBlock

    _configure_provider(headless_tui_env)
    session_id = uuid.uuid4().hex[:12]

    async def _never_take_over() -> Any:
        raise AssertionError("a viewer must never take over a session")

    async def factory() -> Any:
        return await AttachedSession.cold(
            session_id,
            config_dir=headless_tui_env,
            cwd=str(workspace),
            takeover_factory=_never_take_over,
        )

    app = OperatorApp(factory)
    viewer = None
    try:
        async with app.run_test(size=(100, 30)) as pilot:
            # The mount engage, and the band it puts up. Waiting for the band
            # BEFORE timing the clear is what makes the measurement the
            # engage's rather than the app's boot.
            started = time.monotonic()
            while time.monotonic() - started < ENGAGE_START_BUDGET_S:
                await pilot.pause()
                if app._starting_runtime:
                    break
                await asyncio.sleep(0.02)
            assert app._starting_runtime, "the mount engage never put the band up"
            # ``OperatorApp._session`` is typed as the protocol every host satisfies;
            # this test drives the real ``AttachedSession.cold`` facade, whose
            # ``is_cold`` is the property under assertion.
            viewer: Any = app._session
            assert viewer is not None and viewer.is_cold

            band_up_at = time.monotonic()
            while time.monotonic() - band_up_at < BAND_CLEAR_BUDGET_S:
                await pilot.pause()
                if not app._starting_runtime:
                    break
                await asyncio.sleep(0.05)
            cleared_after = time.monotonic() - band_up_at

            assert not app._starting_runtime, (
                f"the band was still up {cleared_after:.1f}s after the engage "
                "started; the bounded engage is supposed to give up well inside "
                f"{BAND_CLEAR_BUDGET_S}s"
            )
            # Nothing bound: the deadline expired against a candidate that had
            # not published a record, which is the failure being reported.
            assert viewer.is_cold, "the viewer bound despite the candidate never publishing"

            notices = [str(block._text) for block in app.query(NoticeBlock)]
            assert any(FAILURE_NOTICE_FRAGMENT in text for text in notices), (
                "the band cleared with no word in the transcript: " f"notices={notices!r}"
            )
    finally:
        if viewer is not None:
            with contextlib.suppress(Exception):
                await viewer.dispose()
