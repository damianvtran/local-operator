"""The five ``/goal`` forms on both TUI hosts, byte-for-byte alike.

``/goal`` is implemented twice in this app — the local handler and the routed
one a follower calls — and the two produce the string the user reads. A host that
names a different state, or reports a settle that did not happen, is the
host-disagreement class the shared vocabulary in ``session/goal.py`` exists to
remove, so the assertions here are made on BOTH hosts and compared.

The picker half is asserted on the rows the widget actually derived, because
those rows are the only place ``/goal`` teaches anything.
"""

from __future__ import annotations

import pytest

from local_operator.session.goal import CLEARED_GOAL_ECHO_CHARS
from local_operator.tui.app import OperatorApp

from .test_app_pilot import FakeSession, _factory
from .test_slash_goal_loop_flags import (
    _boot,
    _draft,
    _notice_texts,
    _row_names,
    _settle,
    _submit,
)

GOAL = "land the OAuth refresh fix"


def _armed() -> FakeSession:
    """A fake with a REAL record, armed exactly as ``/goal <text>`` arms it."""
    session = FakeSession()
    session.arm_goal(GOAL)
    return session


async def _local(pilot, app: OperatorApp, text: str) -> list[str]:
    await _submit(pilot, app, text)
    return _notice_texts(app)


@pytest.mark.asyncio
async def test_done_changes_state_and_starts_no_turn_on_either_host() -> None:
    local, routed = _armed(), _armed()
    app = OperatorApp(lambda: _factory(local))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        local_notices = await _local(pilot, app, "/goal --done")
    app2 = OperatorApp(lambda: _factory(routed))
    async with app2.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app2)
        result = await app2.run_slash_authoritative("goal", "--done")

    assert local_notices == [f"goal done: {GOAL}"]
    assert result["kind"] == "notice"
    assert result["text"] == local_notices[0], "the two hosts must read alike"
    assert not result.get("data"), "a mark-done is not an action receipt"
    for session in (local, routed):
        assert session.goal == GOAL, "the text stays until the chip is dismissed"
        assert session.goal_status == "done"
        assert [row["text"] for row in session.history_view()] == [GOAL]
        assert session.prompts == [], "a mark-done starts no turn"
        assert session.queued_steering() == []


@pytest.mark.asyncio
async def test_done_twice_says_already_done_rather_than_settling_again() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _local(pilot, app, "/goal --done")
        again = await _local(pilot, app, "/goal --done")

    assert again == ["goal already done — /goal --dismiss clears it"]
    # Exactly one row: a second mark-done must not duplicate the record, and the
    # receipt must not claim a settle that did not happen.
    assert len(session.history_view()) == 1


@pytest.mark.asyncio
async def test_history_carries_the_rows_in_data_and_one_count_line_on_either_host() -> None:
    local, routed = _armed(), _armed()
    for session in (local, routed):
        session.mark_goal_done("the judge said so")
        session.dismiss_goal()
    app = OperatorApp(lambda: _factory(local))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        local_notices = await _local(pilot, app, "/goal --history")
    app2 = OperatorApp(lambda: _factory(routed))
    async with app2.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app2)
        result = await app2.run_slash_authoritative("goal", "--history")

    # The notice is ONE logical line and a COUNT: the entries ride `data`, since
    # a multi-row payload dump in the transcript is what the receipt discipline
    # exists to prevent.
    assert local_notices == ["1 settled goal — newest first"]
    assert len(local_notices[0]) <= CLEARED_GOAL_ECHO_CHARS
    assert result["kind"] == "block"
    assert result["data"]["type"] == "goal_history"
    assert result["text"] == local_notices[0]
    assert [row[0] for row in result["data"]["items"]] == [GOAL]
    assert local.prompts == [] and routed.prompts == [], "a listing starts no turn"


@pytest.mark.asyncio
async def test_dismiss_drops_the_chip_and_keeps_the_record() -> None:
    session = _armed()
    session.mark_goal_done("done")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notices = await _local(pilot, app, "/goal --dismiss")

    assert notices == ["goal dismissed"]
    assert session.goal == "" and session.goal_status == ""
    assert [row["status"] for row in session.history_view()] == ["done"]
    assert session.prompts == []


@pytest.mark.asyncio
async def test_dismiss_with_nothing_up_says_so_on_either_host() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notices = await _local(pilot, app, "/goal --dismiss")
        result = await app.run_slash_authoritative("goal", "--dismiss")

    # An ACTIVE goal is not a done chip: dismissing it would be a delete wearing
    # another word, and the receipt must not report a change that did not happen.
    assert notices == ["nothing to dismiss"]
    assert result["text"] == notices[0]
    assert session.goal == GOAL


@pytest.mark.asyncio
async def test_the_bare_report_names_the_state_on_either_host() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        active = await _local(pilot, app, "/goal")
        result = await app.run_slash_authoritative("goal", "")
        session.mark_goal_done("done")
        done = await _local(pilot, app, "/goal")

    assert active == [f"goal: {GOAL} — active"]
    assert result["text"] == active[0]
    # A settled goal has no live work to do, so the report names the way out.
    assert done == [f"goal: {GOAL} — done; /goal --dismiss clears it"]


@pytest.mark.asyncio
async def test_a_new_flag_with_a_tail_is_still_a_goal_body() -> None:
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/goal --done the report")

    # The whole-argument rule: eating the tail would be silent data loss in the
    # one command whose argument the MODEL is told.
    assert session.goal == "--done the report"
    assert session.prompts == ["--done the report"]
    assert session.history_view() == []


@pytest.mark.asyncio
async def test_an_unknown_flag_of_this_command_is_still_refused() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notices = await _local(pilot, app, "/goal --donex")

    assert notices == ["unknown flag --donex — /goal <text> sets it, --clear/--done/--history"]
    assert session.goal == GOAL
    assert session.prompts == []


@pytest.mark.asyncio
async def test_the_picker_offers_only_the_acts_the_record_allows() -> None:
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        editor = await _draft(app, pilot, "/goal ")
        # `--clear` is LAST so that a single pre-selected match on the bare
        # command — Enter, the keystroke that reports the goal — can never run
        # the destructive row (round 1: design D1, UX U1, reviewer MAJOR-1).
        assert _row_names(editor) == ["--done", "--clear"]

        session.mark_goal_done("done")
        editor.load_text("/goal ")
        editor.move_cursor(editor._end_of_buffer())
        await _settle(pilot, app)
        # A DONE goal offers the dismissal and the record, and no longer offers
        # a mark-done it cannot perform.
        assert _row_names(editor) == ["--history", "--dismiss", "--clear"]
