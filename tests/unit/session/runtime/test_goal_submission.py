"""Every owner-side /goal entry point submits once, or only changes status.

Legacy/mobile clients cannot consume receipts. Current terminals explicitly
claim goal_set and own the ordinary submit, so the owner must not also send.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.session.goal import MAX_GOAL_CHARS, GoalState
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession


class GoalSession(FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.goal_state = GoalState()
        self.prompt_release.set()

    @property
    def goal(self) -> str:
        return self.goal_state.text

    def set_goal(self, text: str) -> str:
        return self.goal_state.set(text)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
@pytest.mark.parametrize("busy", [False, True])
async def test_goal_starts_or_steers_once(entry, busy):
    session = GoalSession()
    session.is_streaming = busy
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", "ship it", None)
            assert result["data"]["type"] == "goal_set"
        else:
            result = await getattr(handle, entry)("goal", "ship it")
            assert result == "goal set"
        await asyncio.sleep(0.05)
        assert session.goal == "ship it"
        assert session.prompt_calls == ([] if busy else ["ship it"])
        assert session.steer_calls == (["ship it"] if busy else [])
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("consumers", [None, [], ["team_attached"], ["goal_set"]])
async def test_goal_receipt_honors_exact_declared_consumer_and_keeps_full_request(consumers):
    session = GoalSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    text = "x" * (MAX_GOAL_CHARS + 100)
    try:
        result = await handle.run_slash_authoritative("goal", text, None, consumers=consumers)
        await asyncio.sleep(0.05)
        assert session.goal == text[:MAX_GOAL_CHARS]
        assert result["data"] == {"type": "goal_set", "stored": session.goal, "request": text}
        assert result["style"] == "warning"
        assert session.prompt_calls == ([] if consumers == ["goal_set"] else [text])
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
@pytest.mark.parametrize("arg", ["", "clear", "none", "reset", "CLEAR", "--clear"])
async def test_goal_status_and_clear_never_submit(entry, arg):
    session = GoalSession()
    session.set_goal("existing goal")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", arg, None)
            assert not result.get("data")
        else:
            await getattr(handle, entry)("goal", arg)
        await asyncio.sleep(0)
        assert session.goal == ("existing goal" if not arg else "")
        assert session.prompt_calls == []
        assert session.steer_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
async def test_the_clear_flag_never_becomes_the_goal_body(entry):
    """``--clear`` is a FLAG: the literal string must never be stored.

    Asserted on the session's own goal rather than on the receipt, because the
    failure this guards renders as an ordinary "goal set" — the state is the
    only thing that tells a clear from a goal named ``--clear``.
    """
    session = GoalSession()
    session.set_goal("existing goal")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", "--clear", None)
            assert result["text"] == "goal cleared"
            assert not result.get("data")
        else:
            assert await getattr(handle, entry)("goal", "--clear") == "goal cleared"
        await asyncio.sleep(0)
        assert session.goal == ""
        assert session.prompt_calls == []
        assert session.steer_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_a_goal_body_opening_with_the_flag_is_still_a_goal():
    """The flag is the WHOLE argument: free text that starts with it is text.

    Eating the tail of `/goal --clear the flaky job` would be silent data loss
    in the one command whose argument the model is told.
    """
    session = GoalSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        result = await handle.run_slash_authoritative("goal", "--clear the flaky job", None)
        assert result["data"]["type"] == "goal_set"
        assert session.goal == "--clear the flaky job"
    finally:
        await handle.dispose()
