"""``is_cold`` means "no runtime reachable", NOT "never bound".

The distinction is load-bearing rather than semantic. ``ViewerSessionProtocol``
declared the opposite for one round ("it is NOT 'the runtime is unreachable' --
there is no runtime to reach") while the implementation was exactly that, and a
declaration is what later call sites get written against: substituting a type
for a duck-probe against a false declaration propagates the error into every
branch that trusts it. ``app.py:23470`` already depends in writing on
``is_cold`` covering the drop ("a superset of the drop: it is also true while
the facade is redialing an owner that died"), so narrowing it would silently
re-open the #625 shape there.

A docstring cannot fail, so the corrected claim is pinned here instead. Drives
the real facade over a real loopback socket through:

  1. never bound                          -> expect is_cold True
  2. attached, having WATCHED A REAL TURN  -> expect is_cold False
  3. that same viewer after the runtime dies -> expect is_cold True  (disputed)

If (3) ever reads False, either the implementation was narrowed or the
declaration is wrong again; the assertion says which. Built on the repo's own
e2e harness rather than a double, because a stub that declares the member is
exactly what hid the last outage.
"""

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime import registry
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.e2e.harness import ScriptedStream, build_session, text_turn

pytestmark = pytest.mark.e2e


async def _never_take_over() -> Any:
    raise AssertionError("a viewer must never take over a session")


async def _wait_for_record(config_dir: Path, session_id: str, timeout: float = 10.0) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        for record, _state in registry.scan(config_dir):
            if getattr(record, "session_id", "") == session_id:
                return record
        await asyncio.sleep(0.05)
    raise AssertionError(f"no record published for {session_id} within {timeout}s")


@pytest.mark.asyncio
async def test_is_cold_covers_a_runtime_that_died_not_only_one_never_bound(
    headless_tui_env: Path, workspace: Path
) -> None:
    from local_operator.session.attached import AttachedSession

    directory = headless_tui_env / "sessions" / "iscoldsess01"
    directory.mkdir(parents=True)

    # --- state 1: cold, never bound to any runtime -------------------------
    cold = await AttachedSession.cold(
        "iscold-never-bound",
        config_dir=headless_tui_env,
        cwd=str(workspace),
        takeover_factory=_never_take_over,
    )
    try:
        never_bound = cold.is_cold
        never_bound_pid = cold.runtime_pid
    finally:
        await cold.dispose()
    print(f"1. never bound            is_cold={never_bound} runtime_pid={never_bound_pid}")

    stream = ScriptedStream([text_turn("Hello from the runtime.")])
    session = build_session(directory, stream)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(directory))
    server = RuntimeServer(handle, kind="daemon")
    await server.start_in_process()

    viewer = None
    try:
        record = await _wait_for_record(headless_tui_env, session.session_id)
        viewer = await AttachedSession.connect(
            record,
            session.session_id,
            config_dir=headless_tui_env,
            takeover_factory=_never_take_over,
        )
        attached = viewer.is_cold
        print(f"2. attached, quiescent    is_cold={attached} runtime_pid={viewer.runtime_pid}")

        # --- state 2: watch a REAL turn run on the runtime ------------------
        events: list[Any] = []
        viewer.subscribe(lambda e: events.append(e))
        await viewer.prompt("hello there")

        transcript_path = directory / "transcript.jsonl"
        for _ in range(200):
            if transcript_path.exists() and "Hello from the runtime." in transcript_path.read_text(
                encoding="utf-8"
            ):
                break
            await asyncio.sleep(0.05)
        else:  # pragma: no cover
            raise AssertionError("the runtime never wrote the reply; the turn did not run")

        watched = viewer.is_cold
        watched_pid = viewer.runtime_pid
        print(
            f"3. after WATCHING a turn  is_cold={watched} runtime_pid={watched_pid} "
            f"events={len(events)} (reply on the runtime's own disk)"
        )

        # --- state 3: the runtime dies under the attached viewer ------------
        server.close()
        await session.dispose()
        for _ in range(200):
            if viewer.is_cold:
                break
            await asyncio.sleep(0.05)
        died = viewer.is_cold
        died_pid = viewer.runtime_pid
        print(f"4. runtime DIED           is_cold={died} runtime_pid={died_pid}")
    finally:
        if viewer is not None:
            await viewer.dispose()
        with contextlib.suppress(Exception):
            server.close()
        with contextlib.suppress(Exception):
            await session.dispose()

    assert never_bound is True, "a never-bound viewer must read cold"
    assert attached is False, "an attached viewer must not read cold"
    assert watched is False and len(events) > 0, "a viewer watching a turn must not read cold"
    assert watched_pid is not None, "an attached viewer must report the runtime pid"

    # THE DISPUTED CASE. True here means `is_cold` is "no runtime reachable",
    # covering both never-bound and bound-then-lost -- which is what the
    # corrected declaration says and what app.py:23470 depends on.
    assert died is True, (
        "is_cold must be True once the runtime is gone: app.py:23470 relies on it "
        "being a superset of the drop. If this fails, the round-1 docstring was "
        "right and the implementation needs splitting instead."
    )
    assert died_pid is None, "runtime_pid must clear when the runtime is lost"
