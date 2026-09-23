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

from typing import cast

import pytest
from rich.text import Text

from local_operator.session.goal import CLEARED_GOAL_ECHO_CHARS
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.goal_panel import GoalPanel

from .test_app_pilot import FakeSession, _factory
from .test_slash_goal_loop_flags import (
    _boot,
    _draft,
    _notice_texts,
    _row_names,
    _submit,
)

GOAL = "land the OAuth refresh fix"


def _armed() -> FakeSession:
    """A fake with a REAL record, armed exactly as ``/goal <text>`` arms it."""
    session = FakeSession()
    session.arm_goal(GOAL)
    return session


async def _local(pilot, app: OperatorApp, text: str) -> list[str]:
    """The notices THIS submit added, and nothing the boot had already painted.

    ``_armed()`` deliberately boots a session that ALREADY carries a goal, and
    adopting such a session paints one ``goal restored`` system notice
    (``OperatorApp._report_attachment_restore``). That notice is the feature
    working — invisible standing state being announced once on adopt — so it
    must not be suppressed; but it lands in the transcript before any submit,
    and the exact-equality assertions below are about the submit's own words.
    Slicing from a snapshot keeps that meaning intact.
    """
    before = len(_notice_texts(app))
    await _submit(pilot, app, text)
    return _notice_texts(app)[before:]


def _card_text(app: OperatorApp) -> str:
    """The goal card's whole body, as plain text.

    Typed here rather than at each call site: ``GoalPanel`` subclasses ``Static``
    without overriding ``render()``, so the declared return is the broad
    ``RenderableType`` union while the renderable it actually holds is a
    ``Text`` — the same ``cast`` the repo's other widget tests use.
    """
    return cast(Text, app.query_one(GoalPanel).render()).plain


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
    """The no-argument form names the state — on the card here, in the receipt there.

    RULINGS R6 makes the TUI's no-argument form the OVERLAY, so THIS host answers a
    READ with the card and paints no receipt at all (``_open_goal_panel``: a receipt
    is the record of an ACT, and the card is the state a read leaves behind). The
    routed host keeps the shared vocabulary's one-liner, which is what a FOLLOWER
    reads — it cannot be handed a card whose keys write the owner's record. So the
    two hosts name the same state in deliberately different shapes, and the
    assertions below pin each host's own shape rather than comparing them word for
    word; the routed string stays the ONE wording every non-TUI host answers with
    (``serving.py::_goal_slash``).
    """
    session = _armed()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert await _local(pilot, app, "/goal") == [], "a read prints no receipt"
        active_card = _card_text(app)
        active = await app.run_slash_authoritative("goal", "")
        session.mark_goal_done("done")
        assert await _local(pilot, app, "/goal") == [], "still no receipt when done"
        done_card = _card_text(app)
        done = await app.run_slash_authoritative("goal", "")

    assert GOAL in active_card and "— active" in active_card
    assert active["text"] == f"goal: {GOAL} — active"
    # A settled goal has no live work to do, so the report names the way out — and
    # the card carries the same state, struck, while it is up.
    assert GOAL in done_card and "— done" in done_card
    assert done["text"] == f"goal: {GOAL} — done; /goal --dismiss clears it"


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
async def test_a_flag_form_on_a_viewer_that_cannot_route_says_where_the_act_lives() -> None:
    """A host with no record of its own is told where the act lives.

    ``_goal_record()`` is ``None`` on anything that is not the session's OWNER
    (``owns_the_session``), and a cold viewer is exactly that: it advertises no
    capabilities, so ``_run_slash_command`` has nothing to route ``/goal`` to and
    the command runs HERE. The four FLAG forms write the OWNER's record, so such
    a host has nothing local to act on — and the honest answer says so, rather
    than reporting an act it did not perform or telling the user the session is
    "still starting" long after it did.

    The SET path deliberately stays open on this same shape —
    ``test_a_followers_loop_turn_holds_working_between_iterations`` pins that,
    because the viewer's own LOCAL ``/loop`` iterates toward ``session.goal``.
    The refusal here is about the record, not about the objective.
    """
    session = _armed()
    # The locality every attached viewer reports, cold or not.
    session.runtime_locality = "this-machine"
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        notices = await _local(pilot, app, "/goal --clear")

    assert notices == [
        "the goal record belongs to the session's owner — "
        "/goal --clear, --done and --dismiss run there"
    ]
    assert session.goal == GOAL, "a refused act changes nothing"
    assert session.goal_status == "active"
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

    # A DONE goal offers the dismissal and the record, and no longer offers a
    # mark-done it cannot perform. Built as its own session rather than mutated
    # mid-test so the rows are derived from the state the app booted with, which
    # is what the reported path does.
    settled = _armed()
    settled.mark_goal_done("done")
    app2 = OperatorApp(lambda: _factory(settled))
    async with app2.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app2)
        editor = await _draft(app2, pilot, "/goal ")
        assert _row_names(editor) == ["--history", "--dismiss", "--clear"]
