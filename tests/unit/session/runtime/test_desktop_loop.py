"""Owner loop invariants supplemental to the assembled HTTP/runtime exercise."""

import asyncio
from typing import Any

import pytest

from local_operator.session.frontend_state import (
    FRONTEND_CHECKPOINT_CUSTOM_TYPE,
    FrontendSessionState,
    FrontendStateStore,
)
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession


class LoopSession(FakeSession):
    def __init__(self):
        super().__init__()
        self.goal = "Finish the fixture"
        self.aborted = 0
        self._frontend_state_store: Any = None
        #: The `append_custom` seam `FrontendStateStore.checkpoint` writes through.
        #: Declared (as ``None``) so the durability test can substitute a recorder
        #: and the type checker still sees the attribute on the real shape.
        self._transcript: Any = None

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
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
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
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
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
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "error"
        assert result["data"]["code"] == "loop_invalid"
        assert not handle._loop_driver().running
        assert not session.prompt_calls
    finally:
        await handle.dispose()


class _RecordingTranscript:
    """The `append_custom` seam `FrontendStateStore.checkpoint` writes through."""

    def __init__(self) -> None:
        self.customs: list[tuple[str, Any]] = []

    async def append_custom(self, custom_type: str, details: Any) -> None:
        self.customs.append((custom_type, details))


def _finished_loop_session() -> LoopSession:
    """A session whose published loop state is a FINISHED run.

    The state a dismissed desktop surface is left holding: a terminal status
    with the goal and reason of the run that produced it, which is exactly what
    a merge (`publish(**values)`) would carry into the cleared snapshot.
    """
    session = LoopSession()
    session._frontend_state_store = FrontendStateStore(
        FrontendSessionState(
            session_id="abcdef123456",
            epoch="e1",
            loop={
                "status": "done",
                "completed": 3,
                "goal": "Finish the fixture",
                "reason": "the judge saw the goal met",
            },
        )
    )
    session._transcript = _RecordingTranscript()
    return session


@pytest.mark.asyncio
async def test_loop_clear_flag_resets_a_finished_loops_published_state(tmp_path):
    """`/loop --clear` leaves ``{"status": "idle", "completed": 0}`` published.

    REPLACED, not merged: the previous run's goal and reason must not survive,
    or the surface the clear exists to dismiss would still describe an objective
    nobody is looping toward.
    """
    session = _finished_loop_session()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        result = await handle.run_slash_authoritative("loop", "--clear", [])
        assert result["kind"] == "block"
        assert result["data"]["type"] == "loop"
        assert result["data"]["status"] == "idle"
        assert result["data"]["completed"] == 0
        assert session._frontend_state_store.state.loop == {"status": "idle", "completed": 0}
        assert not session.prompt_calls
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_loop_clear_is_durable_in_the_frontend_checkpoint(tmp_path):
    """The clear rides the same checkpoint as every other loop transition.

    Without it the reset would hold only until something re-checkpointed the
    session, and a replaced runtime — which restores `store.state.loop` — would
    bring the dismissed run back.
    """
    session = _finished_loop_session()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        await handle.run_slash_authoritative("loop", "--clear", [])
        written = [
            details
            for custom_type, details in session._transcript.customs
            if custom_type == FRONTEND_CHECKPOINT_CUSTOM_TYPE
        ]
        assert written, "the cleared state never reached a checkpoint"
        assert written[-1]["state"]["loop"] == {"status": "idle", "completed": 0}
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_loop_clear_is_refused_while_a_loop_runs(tmp_path):
    """`--clear` never cancels live work, and names the word that does."""
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        # Numeric mode iterates toward the STANDING goal with the shared loop
        # prompt, so the first iteration is in flight once the fake has been
        # handed ANY turn — and it stays in flight on the release gate, which
        # is what makes `driver.running` a real observation here.
        await until(lambda: len(session.prompt_calls) == 1)
        result = await handle.run_slash_authoritative("loop", "--clear", [])
        assert result["kind"] == "error"
        assert result["data"]["code"] == "loop_running"
        assert "/loop --stop" in result["text"]
        # Still driving: the refusal is a refusal, not a quiet cancel.
        assert driver.running
        assert session.aborted == 0
        await driver.cancel()
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("argument", ["--stop", "stop", "cancel", "abort"])
async def test_loop_stop_forms_cancel_the_driver(tmp_path, argument):
    """The flag and the three bare words all reach the same cancel."""
    session = LoopSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    try:
        driver = handle._loop_driver()
        driver.start("2", session.goal)
        # Numeric mode iterates toward the STANDING goal with the shared loop
        # prompt, so the first iteration is in flight once the fake has been
        # handed ANY turn — and it stays in flight on the release gate, which
        # is what makes `driver.running` a real observation here.
        await until(lambda: len(session.prompt_calls) == 1)
        result = await handle.run_slash_authoritative("loop", argument, [])
        assert result["kind"] == "block"
        assert not driver.running
        assert driver.state["status"] == "cancelled"
        session.prompt_release.set()
        await until(lambda: not handle._prompt_queue)
    finally:
        await handle.dispose()
