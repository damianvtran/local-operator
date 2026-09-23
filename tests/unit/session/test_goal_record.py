"""The judged-goal record: arming, settling, deleting and the payload round trip.

What this file pins, and why each one is here rather than in a host's test: the
record is shared by the runtime and both TUI hosts, so a rule stated once here is
the rule all four implement. The two that matter most are the DELETE-VERSUS-DONE
distinction (a delete records nothing; that is the whole difference between the
two commands) and the ORDERING in ``arm`` (a second ``/goal`` marks the first one
superseded BEFORE the new text is stored — without that order ``/goal B`` destroys
``/goal A`` silently, the loss class ``cleared_goal_receipt`` exists to prevent).
"""

from __future__ import annotations

from local_operator.session.goal import (
    GOAL_HISTORY_MAX,
    GOAL_HISTORY_TEXT_CHARS,
    GOAL_JUDGE_STATES,
    MAX_GOAL_CHARS,
    GoalHistoryEntry,
    GoalJudgeState,
    GoalState,
)


def _entries(state: GoalState) -> list[tuple[str, str]]:
    """``(text, status)`` for every settled entry, in stored order."""
    return [(entry.text, entry.status) for entry in state.history]


def test_arm_marks_active_and_arms_the_judge():
    state = GoalState()

    stored = state.arm("ship the thing")

    assert stored == "ship the thing"
    assert state.text == "ship the thing"
    assert state.status == "active"
    assert state.token, "a goal with no token cannot have an in-flight verdict matched"
    assert state.judge.state == "waiting"
    assert state.created_at, "the entry a settle writes needs the arm's own stamp"
    assert state.history == []


def test_arm_over_a_second_goal_records_the_first_as_superseded():
    state = GoalState()
    state.arm("A")

    state.arm("B")

    # EXACTLY one entry, carrying the OLD text, and nothing for the new goal —
    # a supersede records what was lost, never what replaced it.
    assert _entries(state) == [("A", "superseded")]
    assert state.text == "B"
    assert state.status == "active"


def test_arm_over_a_done_goal_records_nothing_new():
    state = GoalState()
    state.arm("A")
    state.mark_done("because")

    state.arm("B")

    # A is already in the history: the chip window is a DISPLAY state, not a
    # second settle, so re-arming must not duplicate its row. And the chip goes,
    # which is what carrying a done goal and a new one at once would mean.
    assert _entries(state) == [("A", "done")]
    assert state.text == "B"
    assert state.status == "active"


def test_arm_mints_a_new_token_per_arming():
    state = GoalState()
    state.arm("the same words")
    first = state.token

    state.arm("the same words")

    # The identity `text` cannot provide: the user may set the same objective
    # twice, and the second arming is NEW work whose in-flight verdict must not
    # be the first one's.
    assert state.token and state.token != first


def test_mark_done_strikes_the_goal_and_keeps_its_text():
    state = GoalState()
    state.arm("A")

    entry = state.mark_done("the judge said so")

    assert entry is not None
    assert entry.status == "done"
    assert entry.reason == "the judge said so"
    # The text STAYS until dismissal: that window is what lets every surface
    # strike the objective it is still showing.
    assert state.text == "A"
    assert state.status == "done"
    assert state.judge.state == "done"


def test_mark_done_twice_records_one_entry():
    state = GoalState()
    state.arm("A")
    state.mark_done()

    assert state.mark_done() is None
    assert len(state.history) == 1


def test_mark_done_with_no_goal_is_a_no_op():
    state = GoalState()

    assert state.mark_done() is None
    assert state.history == []
    # Not even a status: a session with no goal has nothing to settle, and
    # `done` with no text would render as a struck empty row.
    assert state.status == ""


def test_a_user_marked_done_carries_no_model_reason():
    state = GoalState()
    state.arm("A")

    entry = state.mark_done()

    assert entry is not None
    assert entry.reason == ""
    assert state.judge.verdict == "", "a person's judgement is not a model verdict"


def test_dismiss_clears_the_chip_and_says_whether_there_was_one():
    state = GoalState()
    state.arm("A")
    state.mark_done("done")

    assert state.dismiss() is True
    assert state.text == ""
    assert state.status == ""
    assert state.token == ""
    assert state.judge.state == "idle"
    # The settled row is NOT the chip's: dismissing the chip keeps the record.
    assert _entries(state) == [("A", "done")]
    # Nothing left to dismiss — and saying so is what stops a host printing a
    # no-op that reads like a success.
    assert state.dismiss() is False


def test_dismiss_refuses_an_active_goal():
    state = GoalState()
    state.arm("A")

    # Only the done chip is dismissible. Dismissing an ACTIVE goal would be a
    # delete wearing a different word, and the delete path records nothing.
    assert state.dismiss() is False
    assert state.text == "A"
    assert state.status == "active"


def test_delete_blanks_the_goal_and_records_nothing():
    state = GoalState()
    state.arm("A")
    before = len(state.history)

    went = state.delete()

    assert went == "A"
    assert state.text == ""
    assert state.status == ""
    assert state.token == ""
    assert state.judge == GoalJudgeState()
    # THE delete-versus-done distinction, asserted as a count rather than as a
    # comment: an explicit delete is the user saying this objective is not worth
    # keeping, so recording it would make the two commands differ only in wording.
    assert len(state.history) == before == 0


def test_delete_keeps_the_history_of_earlier_goals():
    state = GoalState()
    state.arm("A")
    state.mark_done("done")
    state.dismiss()
    state.arm("B")

    state.delete()

    # `/goal --clear` clears the ACTIVE slot; the settled record is a different
    # thing, and `/goal --history` is how the user reads it back.
    assert _entries(state) == [("A", "done")]


def test_history_evicts_the_oldest_past_the_cap():
    state = GoalState()
    for index in range(GOAL_HISTORY_MAX + 5):
        state.arm(f"goal {index}")
        state.mark_done(f"reason {index}")
        state.dismiss()

    texts = [entry.text for entry in state.history]
    assert len(state.history) == GOAL_HISTORY_MAX
    # Newest first, and the eviction takes the OLDEST: the recent rows are the
    # ones a reader acts on, which is the argument the entry cap is set by.
    assert texts[0] == f"goal {GOAL_HISTORY_MAX + 4}"
    assert "goal 0" not in texts


def test_history_view_is_newest_first_and_bounded_by_limit():
    state = GoalState()
    state.arm("A")
    state.mark_done("first")
    state.dismiss()
    state.arm("B")
    state.mark_done("second")

    rows = state.history_view()

    assert [row["text"] for row in rows] == ["B", "A"]
    assert rows[0]["reason"] == "second"
    assert [row["text"] for row in state.history_view(limit=1)] == ["B"]
    assert state.history_view(limit=0) == []


def test_entry_text_is_clipped_at_construction():
    entry = GoalHistoryEntry(
        id="i",
        text="x" * (GOAL_HISTORY_TEXT_CHARS + 500),
        status="done",
        created_at="c",
        settled_at="s",
    )

    # AT CONSTRUCTION, not at read: a later reader cannot forget the bound, and
    # every copy of the entry is already the bounded one.
    assert len(entry.text) == GOAL_HISTORY_TEXT_CHARS
    short = GoalHistoryEntry(id="i", text="y" * 5, status="done", created_at="", settled_at="")
    assert short.text == "y" * 5


def test_a_recorded_entry_clips_the_goal_it_settles():
    state = GoalState()
    state.arm("x" * MAX_GOAL_CHARS)

    state.mark_done()

    assert len(state.history[0].text) == GOAL_HISTORY_TEXT_CHARS


def test_the_wire_shape_is_the_four_decided_keys():
    judge = GoalJudgeState(state="continuing", run=3, verdict="continue", reason="r", failures=2)

    # `failures` is the breaker's own counter: publishing it would invite a
    # surface to render an internal number whose only meaning is `stalled`.
    assert judge.to_wire() == {
        "state": "continuing",
        "run": 3,
        "verdict": "continue",
        "reason": "r",
    }
    # The SIDECAR is not the wire: the counter has to survive a restart or a
    # resumed session hands a broken provider a fresh set of strikes every boot.
    assert judge.to_payload()["failures"] == 2


def test_every_judge_member_is_one_of_the_declared_vocabulary():
    # The vocabulary is closed and named once, so a host can branch on it and a
    # reader can tolerate a member it does not know.
    assert GOAL_JUDGE_STATES == {
        "idle",
        "judging",
        "continuing",
        "waiting",
        "done",
        "stalled",
    }
    state = GoalState()
    state.arm("A")
    assert state.judge.state in GOAL_JUDGE_STATES


def test_reset_judge_moves_only_the_state():
    state = GoalState()
    state.arm("A")
    state.judge.run = 4
    state.judge.failures = 2

    state.reset_judge(state="stalled")

    assert state.judge.state == "stalled"
    assert state.judge.run == 4
    assert state.judge.failures == 2


def test_payload_round_trip_is_lossless():
    state = GoalState()
    state.arm("A")
    state.mark_done("the first one")
    state.dismiss()
    state.arm("B")
    state.judge = GoalJudgeState(state="judging", run=2, verdict="continue", reason="r", failures=1)

    restored = GoalState.from_payload(state.to_payload())

    assert restored.text == "B"
    assert restored.status == "active"
    assert restored.token == state.token
    assert restored.created_at == state.created_at
    assert restored.judge == state.judge
    assert [entry.to_wire() for entry in restored.history] == [
        entry.to_wire() for entry in state.history
    ]
    assert restored.judge.to_payload() == state.judge.to_payload()


def test_from_payload_tolerates_an_absent_or_unusable_document():
    for payload in (None, [], "nonsense", 7):
        state = GoalState.from_payload(payload)
        assert state.text == ""
        assert state.status == ""
        assert state.history == []
        assert state.token == ""


def test_from_payload_reads_a_goal_with_no_status_as_active():
    # The same migration rule the frontend fold implements: a document that has
    # an objective but lost (or never carried) a status is ACTIVE, because
    # reading it as absent would silently un-set standing work.
    state = GoalState.from_payload({"goal": "still going", "status": "gone"})

    assert state.text == "still going"
    assert state.status == "active"


def test_from_payload_drops_only_the_unusable_rows():
    payload = {
        "goal": "A",
        "status": "active",
        "history": [
            {"id": "1", "text": "kept", "status": "done", "created_at": "", "settled_at": ""},
            42,
            {"id": "3", "text": "no status", "status": ""},
        ],
    }

    state = GoalState.from_payload(payload)

    assert [entry.text for entry in state.history] == ["kept"]


def test_from_payload_maps_an_unknown_status_member_to_active():
    state = GoalState.from_payload({"goal": "A", "status": "archived-by-a-newer-build"})

    # A newer writer's member is not a crash and not an empty goal: the text is
    # plainly standing work, so it reads as active.
    assert state.status == "active"


def test_history_entry_wire_shape_has_exactly_the_six_contract_keys():
    entry = GoalHistoryEntry(
        id="i", text="t", status="superseded", created_at="c", settled_at="s"
    )

    assert set(entry.to_wire()) == {
        "id",
        "text",
        "status",
        "created_at",
        "settled_at",
        "reason",
    }


def test_the_fold_reads_a_goal_with_no_status_as_active():
    """The migration default, in the ONE place it lives.

    A session restored from a build that predates the record has a goal text and
    nothing else, and reading that as absent would silently un-set an objective
    the user is still expecting to be pursued. The rule is implemented in the
    frontend fold rather than in each reader, so this asserts the fold's own
    answer for all four combinations.
    """
    from local_operator.session.frontend_state import _fold_goal_status

    class _Session:
        pass

    inherited = _Session()
    inherited.goal = "still going"
    assert _fold_goal_status(inherited) == "active"

    recorded_active = _Session()
    recorded_active.goal = "still going"
    recorded_active.goal_status = "active"
    assert _fold_goal_status(recorded_active) == "active"

    recorded_done = _Session()
    recorded_done.goal = "finished"
    recorded_done.goal_status = "done"
    assert _fold_goal_status(recorded_done) == "done"

    empty = _Session()
    assert _fold_goal_status(empty) == ""


def test_ensure_token_mints_one_for_a_goal_that_never_had_it():
    """Agent review MAJOR-3: the goal every surface called `active` and ran on.

    The judge refuses to run without a token — that is its staleness guard — and
    `arm` was the only minter, so a goal that arrived any other way (a plain
    `set_goal`, or a restore from a build that predates the record) was silently
    unjudgeable. Minting is the R9 fold's other half.
    """
    state = GoalState()
    state.set("A")
    assert state.token == "", "a plain set really does mint nothing"

    assert state.ensure_token() is True
    assert state.token, "the judge can now run on it"
    # Idempotent: the goal's identity must not change under a running judge.
    assert state.ensure_token() is False
    assert state.token


def test_ensure_token_leaves_an_empty_and_a_settled_goal_alone():
    empty = GoalState()
    assert empty.ensure_token() is False
    assert empty.token == ""

    done = GoalState()
    done.arm("A")
    token = done.token
    done.mark_done()
    # A done goal is RETAINED so the surfaces can show what was achieved; arming
    # the judge against work that is already finished would be the opposite.
    assert done.ensure_token() is False
    assert done.token == token


def test_a_person_marked_done_clears_the_stale_judge_reason():
    """QA-Q4: the card printed a live-looking quote about a closed goal.

    `mark_done` keeps `text` (the chip strikes through WHAT was done) and set the
    judge to `done`, but the previous tick's REASON stayed on the record — so the
    surface that renders both fields together read
    `judge: achieved · 2/12 · — still drafting` about work the user had just
    closed by hand.
    """
    state = GoalState()
    state.arm("A")
    state.reset_judge(state="continuing")
    state.judge.reason = "still drafting"

    state.mark_done()

    assert state.judge.state == "done"
    assert state.judge.reason == ""


def test_a_verdict_marked_done_leaves_the_live_reason_to_the_publish():
    """...and that clear is for a PERSON's settle only.

    A verdict's reason rides the HISTORY entry, which is the record of what the
    model said when it closed the goal; the live judge reason beside it is
    written by the driver's own publish immediately after the settle. So
    `mark_done` must leave that field alone rather than blanking a value the
    publish is about to set.
    """
    state = GoalState()
    state.arm("A")
    state.reset_judge(state="judging")
    state.judge.reason = "the tick's own words"

    entry = state.mark_done("the artifact exists and the goal is met")

    assert entry is not None
    assert entry.reason == "the artifact exists and the goal is met"
    assert state.judge.reason == "the tick's own words"
