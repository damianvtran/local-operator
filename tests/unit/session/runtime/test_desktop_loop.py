"""Owner loop invariants supplemental to the assembled HTTP/runtime exercise."""

import asyncio
from typing import Any

import pytest

from local_operator.session.frontend_state import (
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.runtime.owned import OwnedSessionHandle
from tests.unit.session.runtime.test_owned import FakeSession


class LoopSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.goal = "Finish the fixture"
        self.aborted = 0
        self._frontend_state_store: Any = None

    def abort(self, reason="cancelled"):
        self.aborted += 1
        self.prompt_release.set()


async def until(predicate):
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_cancelling_queued_loop_does_not_abort_manual_turn(tmp_path):
    session = LoopSession()
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        await handle.prompt("manual")
        await until(lambda: session.prompt_calls == ["manual"])
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        await until(lambda: len(handle._prompt_queue) == 2)
        await driver.cancel()
        assert driver.state["status"] == "cancelled"
        assert session.aborted == 0
        assert len(handle._prompt_queue) == 1
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
        assert session.prompt_calls == ["manual"]
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_replacement_owner_marks_active_loop_interrupted(tmp_path):
    session = LoopSession()
    session._frontend_state_store = FrontendStateStore(
        FrontendSessionState(
            session_id="abcdef123456", epoch="old", loop={"status": "running", "completed": 2}
        )
    )
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        assert session._frontend_state_store.state.loop["status"] == "interrupted"
        driver = handle._loop_driver()
        assert not driver.running
        assert driver.state == {"status": "interrupted", "completed": 2}
        assert not session.prompt_calls
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["0", "26", "3e", "-1", "1.5"])
async def test_invalid_loop_does_not_start_work(tmp_path, argument):
    session = LoopSession()
    handle = OwnedSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "error"
        assert result["data"]["code"] == "loop_invalid"
        assert not handle._loop_driver().running
        assert not session.prompt_calls
    finally:
        await handle.dispose()
