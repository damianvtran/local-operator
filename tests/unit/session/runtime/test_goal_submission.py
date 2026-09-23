"""Every owner-side /goal entry point submits once, or only changes status.

Legacy/mobile clients cannot consume receipts. Current terminals explicitly
claim goal_set and own the ordinary submit, so the owner must not also send.
"""

from __future__ import annotations

import asyncio
from typing import Any

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

    # The judged-goal record's surface, delegated to a REAL holder so the flag
    # forms are asserted against state that actually moves (`/goal --done` from
    # `/goal <text>` is a state difference, not a string difference).
    def arm_goal(self, text: str) -> str:
        return self.goal_state.arm(text)

    @property
    def goal_status(self) -> str:
        return self.goal_state.status

    @property
    def goal_judge(self) -> dict[str, Any] | None:
        if not self.goal_state.text:
            return None
        return self.goal_state.judge.to_wire()

    @property
    def goal_history(self) -> list[dict[str, Any]]:
        return self.goal_state.history_view()

    def history_view(self, limit: int | None = None) -> list[dict[str, Any]]:
        return self.goal_state.history_view(limit)

    def mark_goal_done(self, reason: str = "") -> Any:
        return self.goal_state.mark_done(reason)

    def delete_goal(self) -> str:
        return self.goal_state.delete()

    def dismiss_goal(self) -> bool:
        return self.goal_state.dismiss()

    def note_goal_judge(self, **changes: Any) -> None:
        judge = self.goal_state.judge
        for name, value in changes.items():
            setattr(judge, name, value)


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
            # The receipt NAMES what went, on this host too: the runtime's answer
            # is rendered by a viewer that may have no other sight of the goal
            # (design D4 / UX U3).
            assert result["text"] == "goal cleared: existing goal"
            assert not result.get("data")
        else:
            assert await getattr(handle, entry)("goal", "--clear") == "goal cleared: existing goal"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
@pytest.mark.parametrize("arg", ["--stop", "--reset", "--cli"])
async def test_a_bare_unknown_flag_is_refused_instead_of_becoming_the_goal(entry, arg):
    """Two flag vocabularies are taught, so mixing them is the expected mistake.

    The flags are matched as the WHOLE argument, so an unrecognised one used to
    be stored: `/goal --stop` made the standing objective `--stop` and submitted
    a turn carrying it (round 1: UX U6, reviewer NIT-5). The refusal names the
    forms, and the goal is left alone.
    """
    session = GoalSession()
    session.set_goal("existing goal")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", arg, None)
            assert result["kind"] == "notice"
            assert result["style"] == "warning"
            assert f"unknown flag {arg}" in result["text"]
        else:
            receipt = await getattr(handle, entry)("goal", arg)
            assert f"unknown flag {arg}" in receipt
        await asyncio.sleep(0)
        assert session.goal == "existing goal"
        assert session.prompt_calls == []
        assert session.steer_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
async def test_done_marks_the_goal_and_admits_no_turn(entry):
    """``/goal --done`` settles the ACTIVE goal: struck, recorded, no turn.

    The record is what makes completion a state rather than a deletion, so the
    assertions are on the STATE (status, the entry, the text that stays) and on
    the absence of a provider request — a mark-done that also submitted a turn
    would spend tokens on a goal the user just finished.
    """
    session = GoalSession()
    session.arm_goal("ship it")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", "--done", None)
            receipt = result["text"]
            assert result["kind"] == "notice"
            assert not result.get("data")
        else:
            receipt = await getattr(handle, entry)("goal", "--done")
        await asyncio.sleep(0)
        assert receipt == "goal done: ship it"
        assert session.goal == "ship it", "the text stays until the chip is dismissed"
        assert session.goal_status == "done"
        assert [(row["text"], row["status"]) for row in session.history_view()] == [
            ("ship it", "done")
        ]
        assert session.history_view()[0]["reason"] == "", "a person's mark-done has no verdict"
        assert session.prompt_calls == []
        assert session.steer_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
async def test_history_answers_rows_in_data_and_a_count_in_the_notice(entry):
    """The one new form whose answer is a LIST, in the ``team_list`` shape.

    The rows ride ``data`` because the TUI and the desktop both already render a
    block of wire rows; the NOTICE stays one line, because a multi-row payload
    dump in a transcript is what the receipt discipline exists to prevent.
    """
    session = GoalSession()
    session.arm_goal("first thing")
    session.mark_goal_done("the judge said so")
    session.dismiss_goal()
    session.arm_goal("second thing")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", "--history", None)
        else:
            receipt = await getattr(handle, entry)("goal", "--history")
            result = {"text": receipt}
        # Newest first, and no turn for any of it.
        assert result["text"] == "1 settled goal — newest first"
        if entry == "authoritative":
            assert result["kind"] == "block"
            assert result["data"]["type"] == "goal_history"
            assert result["data"]["items"] == [
                ["first thing", "done · " + session.history_view()[0]["settled_at"] + " · the judge said so"]
            ]
        assert session.prompt_calls == []
        assert session.steer_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
async def test_dismiss_drops_the_done_chip_and_says_when_there_is_none():
    session = GoalSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        # Nothing to dismiss: the flag is a no-op unless the chip is up, and the
        # receipt must not report a change that did not happen.
        first = await handle.run_slash_authoritative("goal", "--dismiss", None)
        assert first["text"] == "nothing to dismiss"
        session.arm_goal("ship it")
        session.mark_goal_done("done")
        second = await handle.run_slash_authoritative("goal", "--dismiss", None)
        assert second["text"] == "goal dismissed"
        await asyncio.sleep(0)
        assert session.goal == ""
        assert session.goal_status == ""
        # The settled row is NOT the chip's: dismissing keeps the record.
        assert len(session.history_view()) == 1
        assert session.prompt_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["slash", "slash_images", "authoritative"])
async def test_delete_is_a_bare_alias_of_clear_and_records_nothing(entry):
    """``delete`` joins the clear set: a DELETE records nothing, unlike --done."""
    session = GoalSession()
    session.arm_goal("ship it")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        if entry == "authoritative":
            result = await handle.run_slash_authoritative("goal", "delete", None)
            receipt = result["text"]
        else:
            receipt = await getattr(handle, entry)("goal", "delete")
        await asyncio.sleep(0)
        assert receipt == "goal cleared: ship it"
        assert session.goal == ""
        assert session.goal_status == ""
        assert session.history_view() == [], "a delete is not a settle"
        assert session.prompt_calls == []
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["--done the report", "done the report", "--history of it"])
async def test_a_new_flag_with_a_tail_stays_a_goal_body(arg):
    """The whole-argument rule holds for the new flags too.

    Anything else would eat the tail of a real objective, which is silent data
    loss in the one command whose argument the MODEL is told.
    """
    session = GoalSession()
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        result = await handle.run_slash_authoritative("goal", arg, None)
        await asyncio.sleep(0.05)
        assert result["data"]["type"] == "goal_set"
        assert session.goal == arg
        assert session.prompt_calls == [arg]
    finally:
        await handle.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("arg", ["--donex", "--historyx", "--dismissx"])
async def test_a_flag_of_this_command_that_does_not_exist_is_refused(arg):
    """The known map must stay honest: a token it does not know is an unknown flag.

    Without the new flags in that map each of these would be STORED as the
    standing objective, which is the failure the refusal exists to prevent.
    """
    session = GoalSession()
    session.arm_goal("existing goal")
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd="/tmp")
    try:
        result = await handle.run_slash_authoritative("goal", arg, None)
        assert result["kind"] == "notice"
        assert result["style"] == "warning"
        assert f"unknown flag {arg}" in result["text"]
        assert result["text"].endswith("--clear/--done/--history")
        await asyncio.sleep(0)
        assert session.goal == "existing goal"
        assert session.prompt_calls == []
    finally:
        await handle.dispose()
