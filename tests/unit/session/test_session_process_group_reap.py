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


async def _start_server(session_id: str, seconds: int) -> int:
    result = await builtin.execute_bash(
        "t",
        {"command": f"sleep {seconds} >/dev/null 2>&1 & echo pid=$!", "timeout": 20},
        context=ToolContext(session_id=session_id),
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
    mine = await _start_server(directory.name, 4251)
    theirs = await _start_server("some-other-session", 4252)
    try:
        assert _alive(mine) and _alive(theirs)
        await session.dispose()
        deadline = time.monotonic() + 5
        while _alive(mine) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert not _alive(mine)
        assert _alive(theirs)
        await session.dispose()  # idempotent
    finally:
        for pid in (mine, theirs):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
