"""Disposing a Session ends the process groups ITS bash calls left running.

WHY: ``server & echo $!`` leaves the server alive after the shell leader exits,
and a finished subagent child shares its parent's ``lop`` process, so no
process-exit hook ever reaps it. ``Session.dispose`` (which ``_dispose_child``
awaits for every child) is the one seam both a top-level end and a child pass
through. Real ``execute_bash`` + real subprocesses: a mocked ledger would prove
the call is made, not that the server dies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import time

import pytest

from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    ToolContext,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import builtin

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000, supports_images=False)

_LOG = logging.getLogger(__name__)

#: How long each survivor process is told to live. Over an hour, so nothing but the reap
#: can end it inside this test — the ``sleep`` argument, NOT a port: neither this file nor
#: ``group_reaper`` opens a socket, so there is no fixed port to make ephemeral here.
_SURVIVOR_SECONDS = 4251

#: The backstop for "the survivor is gone", never the assertion (AGENTS.md, "Wait on the
#: event, never on the clock"). The old form was a 5 s slice of wall clock, and this host
#: runs ~25 concurrent worktrees at load 50-110: a kill that landed at 5.2 s reported
#: "dispose does not reap" for a reap that had already happened.
_REAP_BUDGET_S = 60.0


def _stream(request, signal):
    async def gen():
        yield StreamTextDelta(delta="ok")
        yield StreamEndEvent(stop_reason="stop")

    return gen()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_gone(pid: int, budget_s: float) -> tuple[bool, float]:
    """Wait until ``pid`` is gone; report whether it went, and the measured time.

    The measurement is the point: a teardown at 0.2 s and one at 20 s are both green, but
    only the second says the margin is shrinking — and reporting it on failure separates
    "slow host" from "leaked process", which the old fixed slice could not.
    """
    started = time.monotonic()
    while True:
        if not _alive(pid):
            elapsed = time.monotonic() - started
            _LOG.debug("survivor %d gone %.2fs after dispose returned", pid, elapsed)
            return True, elapsed
        elapsed = time.monotonic() - started
        if elapsed >= budget_s:
            return False, elapsed
        await asyncio.sleep(0.05)


async def _start_server(session_id: str, seconds: int) -> int:
    result = await builtin.execute_bash(
        "t",
        {"command": f"sleep {seconds} >/dev/null 2>&1 & echo pid=$!", "timeout": 20},
        None,
        None,
        ToolContext(session_id=session_id),
    )
    match = re.search(r"pid=(\d+)", json.dumps(result.model_dump(mode="json")))
    assert match
    return int(match.group(1))


@pytest.mark.asyncio
async def test_dispose_reaps_its_own_survivor_and_spares_a_siblings(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    directory = tmp_path / "sess-mine"
    session = Session(
        model=MODEL,
        stream_fn=_stream,
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: ["stable"],
    )
    mine = await _start_server(directory.name, _SURVIVOR_SECONDS)
    theirs = await _start_server("some-other-session", _SURVIVOR_SECONDS)
    try:
        assert _alive(mine) and _alive(theirs)
        await session.dispose()
        gone, teardown_s = await _wait_gone(mine, _REAP_BUDGET_S)
        assert gone, (
            f"session.dispose left its own survivor (pid {mine}) alive {teardown_s:.1f}s "
            f"after it returned, past the {_REAP_BUDGET_S:.0f}s backstop — a leaked process "
            "group, not a slow machine: a loaded host still clears it inside the backstop"
        )
        assert _alive(theirs), "dispose reaped the OTHER session's survivor"
        await session.dispose()  # idempotent
    finally:
        for pid in (mine, theirs):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
