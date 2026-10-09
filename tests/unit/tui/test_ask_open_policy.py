"""The TUI's open-by-default policy for queued asks — the pure decision matrix.

``local_operator/tui/ask_open_policy.py`` keeps the six-clause contract the four
ask surfaces share (the clauses are numbered in that module's docstring and cited
here by number). It is a PURE module on purpose: no Textual, no app. That is what
lets the whole matrix be pinned in microseconds, state by state, without booting a
UI — and the app-level behaviour (what really mounts, where focus goes, what a
session swap does to a draft) is pinned separately in ``test_ask_open_default.py``
against the real ``OperatorApp``.

THE FOUR STATES THE OPERATOR NAMED, and the test that owns each:

* no asks              -> closed                      ``test_state_no_asks_*``
* pending on open      -> open (once)                 ``test_state_pending_on_open_*``
* all addressed        -> closed, and stays closed    ``test_state_all_addressed_*``
* dismissed w/ pending -> stays closed after a swap   ``test_state_dismissed_*``
  (and the dismissal is keyed by ASK ID: held while any waved-off ask is still
  outstanding, forgotten once none is — ``test_a_dismissal_*``)

Only the second can fail against ``origin/main``, where nothing ever opens on its
own; the other three are NEGATIVE CONTROLS, and a negative control that passes on
the old code proves nothing about the new. They earn their place by mutation: each
one is the test that goes red when the corresponding clause is removed from the
policy (the PR body lists the mutations and the red test each produced).
"""

from __future__ import annotations

from collections.abc import Collection

from local_operator.tui.ask_open_policy import (
    ARRIVAL_SKEW_MS,
    OPEN_WINDOW_S,
    AskOpenPolicy,
    OpenDecision,
    QueueReading,
    names_every_outstanding,
    read_queue,
)

#: A view that began at t=1000 s. Milliseconds are what an ask's ``created_at`` uses.
T0 = 1000.0
T0_MS = int(T0 * 1000)
OLD = T0_MS - 60_000  # an ask that existed a minute before the view began
NEW = T0_MS + 60_000  # an ask that arrived a minute after it began


def _policy(**kwargs: object) -> AskOpenPolicy:
    # Explicitly enabled: these tests are about the policy itself, and must not
    # depend on the module's AUTO_OPEN seam that other suites flip.
    kwargs.setdefault("enabled", True)
    return AskOpenPolicy(**kwargs)  # type: ignore[arg-type]


def _decide(
    policy: AskOpenPolicy,
    conversation: str,
    reading: QueueReading,
    *,
    at: float = T0,
    occupied: bool = False,
    surface_open: bool = False,
    ids: Collection[str] | None = None,
) -> OpenDecision:
    """One frame's decision. ``ids`` are the outstanding ask ids the frame shows.

    ``None`` is the helper's default AND the policy's: a frame that cannot name them
    all. It only matters to a conversation with a dismissal on record, which is why
    the pre-existing tests (no dismissals) never pass it.
    """
    return policy.decide(
        conversation,
        reading,
        now=at,
        occupied=occupied,
        surface_open=surface_open,
        outstanding_ids=ids,
    )


# -- reading a snapshot ------------------------------------------------------


def test_read_queue_names_every_shape_of_frame() -> None:
    """The five readings, and the one rule that separates a guess from an answer."""

    def read(published: int, created: list[int], tally: int | None) -> QueueReading:
        return read_queue(
            published=published,
            outstanding_created_ms=created,
            tally=tally,
            opened_at_ms=T0_MS,
        )

    # Rows carry the answer when there are any.
    assert read(2, [OLD], 1) is QueueReading.PENDING
    assert read(2, [], 0) is QueueReading.SETTLED
    assert read(1, [NEW], 1) is QueueReading.ARRIVED
    # A live queue with nothing in it says so, in the tally and only there.
    assert read(0, [], 0) is QueueReading.EMPTY
    # Absence is not an answer (clause 5): an unsupported runtime, and a frame the
    # wire bound reduced to its tally, are both "I cannot tell".
    assert read(0, [], None) is QueueReading.UNRESOLVED
    assert read(0, [], 3) is QueueReading.UNRESOLVED
    # Rows that are all settled beside a POSITIVE tally: the outstanding ask is the one
    # the text bound dropped (it keeps newer settled rows over an older timed-out one).
    # "Everything addressed" would be a claim the frame itself contradicts.
    assert read(5, [], 1) is QueueReading.UNRESOLVED
    # A runtime that publishes no tally cannot contradict its own rows.
    assert read(2, [], None) is QueueReading.SETTLED


def test_read_queue_one_old_ask_makes_the_view_pending_even_beside_new_ones() -> None:
    """Pending on open is about ANY outstanding ask that predates the view."""
    reading = read_queue(
        published=3, outstanding_created_ms=[NEW, OLD, NEW], tally=3, opened_at_ms=T0_MS
    )
    assert reading is QueueReading.PENDING


def test_read_queue_tolerates_clock_skew_in_the_pending_direction() -> None:
    """``created_at`` is the owner's clock; a few seconds of skew must not hide an ask.

    The failure being closed is a genuinely pending ask read as an arrival (and so
    left behind a closed panel) because the machine that stamped it runs a clock a
    little ahead of this one. Anything inside the tolerance reads as pending; the
    first millisecond past it is an arrival.
    """

    def read(created: int) -> QueueReading:
        return read_queue(
            published=1, outstanding_created_ms=[created], tally=1, opened_at_ms=T0_MS
        )

    assert read(T0_MS + ARRIVAL_SKEW_MS) is QueueReading.PENDING
    assert read(T0_MS + ARRIVAL_SKEW_MS + 1) is QueueReading.ARRIVED


def test_read_queue_zero_created_at_counts_as_old() -> None:
    """A row with no stamp reads as one that has been there a while, not as an arrival."""
    reading = read_queue(published=1, outstanding_created_ms=[0], tally=1, opened_at_ms=T0_MS)
    assert reading is QueueReading.PENDING


# -- the four states ---------------------------------------------------------


def test_state_no_asks_stays_closed_and_a_later_ask_does_not_force_it_open() -> None:
    """Clause 1 + the tail of clause 3: nothing pending on open means nothing opens."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)

    assert _decide(policy, "conv-a", QueueReading.EMPTY) is OpenDecision.SKIP
    # An ask arrives afterwards. The bar/chip covers it; the panel does not jump up.
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 5) is OpenDecision.SKIP
    assert _decide(policy, "conv-a", QueueReading.ARRIVED, at=T0 + 6) is OpenDecision.SKIP


def test_state_pending_on_open_opens_once() -> None:
    """Clause 2: pending asks on open open the surface — exactly once per view."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)

    assert _decide(policy, "conv-a", QueueReading.PENDING) is OpenDecision.OPEN
    # Every later frame of the same view is a no-op: a re-render, a queue refresh,
    # an ask arriving or changing must not ask the question again.
    for step in range(1, 6):
        assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + step) is OpenDecision.SKIP
    assert not policy.awaiting("conv-a")


def test_state_all_addressed_on_open_stays_closed_and_never_reopens() -> None:
    """Clause 3: every ask already settled on open -> closed, and no later frame reopens."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)

    assert _decide(policy, "conv-a", QueueReading.SETTLED) is OpenDecision.SKIP
    # Even a fresh outstanding ask in the same view stays behind the bar.
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 3) is OpenDecision.SKIP


def test_state_dismissed_while_pending_stays_closed_across_swap_and_back() -> None:
    """Clause 4, the one that needs the app lifetime to hold a fact.

    Open conversation A with a pending ask (the policy opens it), the user closes
    it while the ask remains, switch to B, switch back to A. The second view of A
    sees the very same pending ask and must NOT open — and neither may any
    re-render of it.
    """
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert _decide(policy, "conv-a", QueueReading.PENDING) is OpenDecision.OPEN

    policy.note_user_closed("conv-a", pending_ids={"a1"})
    assert policy.is_dismissed("conv-a")

    # Away and back.
    policy.begin_view("conv-b", now=T0 + 10)
    assert _decide(policy, "conv-b", QueueReading.EMPTY, at=T0 + 10) is OpenDecision.SKIP
    policy.begin_view("conv-a", now=T0 + 20)
    for step in range(5):  # first frame, then re-renders of the same view
        assert (
            _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 20 + step, ids={"a1"})
            is OpenDecision.SKIP
        )


def test_dismissal_is_per_conversation() -> None:
    """Closing A's surface says nothing about B (the record is keyed by id)."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    policy.note_user_closed("conv-a", pending_ids={"a1", "a2"})

    policy.begin_view("conv-b", now=T0 + 1)
    assert _decide(policy, "conv-b", QueueReading.PENDING, at=T0 + 1) is OpenDecision.OPEN


def test_closing_a_surface_with_nothing_outstanding_is_not_a_dismissal() -> None:
    """Clause 4 is about asks that REMAIN: shutting a list of history refuses nothing."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    policy.note_user_closed("conv-a", pending_ids=())
    assert not policy.is_dismissed("conv-a")

    policy.begin_view("conv-b", now=T0 + 1)
    policy.begin_view("conv-a", now=T0 + 2)
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 2) is OpenDecision.OPEN


def test_a_fresh_policy_forgets_dismissals() -> None:
    """Clause 4's "a fresh app start may auto-open again": the record is in memory only."""
    first = _policy()
    first.begin_view("conv-a", now=T0)
    first.note_user_closed("conv-a", pending_ids={"a1"})

    second = _policy()  # a new process
    second.begin_view("conv-a", now=T0)
    assert _decide(second, "conv-a", QueueReading.PENDING, ids={"a1"}) is OpenDecision.OPEN


# -- clause 4, keyed by ask id ------------------------------------------------
#
# The manager's final wording: a dismissal records the ASK IDS it waved off; it is
# FORGOTTEN once none of them is still outstanding. Each test below is one of the
# three effects that wording names (or the fail-closed edge it needs), driven the
# way the app drives the policy: a view begins, a frame names the outstanding ids.


def _dismissed_ab() -> AskOpenPolicy:
    """A policy whose user closed conversation A while asks a and b were pending."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, ids={"a", "b"}) is OpenDecision.OPEN
    ), "premise: the surface opened"
    policy.note_user_closed("conv-a", pending_ids={"a", "b"})
    return policy


def _away_and_back(policy: AskOpenPolicy, *, at: float) -> None:
    """Leave conversation A for B and return: two new views, A's one at ``at``."""
    policy.begin_view("conv-b", now=at - 5)
    policy.begin_view("conv-a", now=at)


def test_a_dismissal_is_held_while_any_waved_off_ask_is_still_outstanding() -> None:
    """Effect 1: dismiss {a, b}; resolve a but not b; leave and return -> still closed."""
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)

    # a was answered while away; b was not. The refusal still has something to be about.
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 60, ids={"b"}) is OpenDecision.SKIP
    )
    assert policy.is_dismissed("conv-a"), "the record must survive a partial resolution"

    # ...and it keeps holding on the NEXT visit too, because the record was not spent.
    _away_and_back(policy, at=T0 + 120)
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 120, ids={"b"}) is OpenDecision.SKIP
    )


def test_a_dismissal_is_forgotten_once_every_waved_off_ask_is_gone_and_a_new_batch_opens() -> None:
    """Effect 2: resolve both; leave and return; a new batch arrives -> a fresh view opens.

    The new asks (c, d) are pending on open for this view (they were created while
    the user was away), and nothing the user refused is among them.
    """
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)

    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 60, ids={"c", "d"})
        is OpenDecision.OPEN
    )
    assert not policy.is_dismissed("conv-a"), "the dismissal must be forgotten, not merely skipped"


def test_a_queue_that_emptied_and_refilled_while_away_needs_no_empty_observation() -> None:
    """Effect 3: the empty-and-refill-while-away hole is closed by the same rule.

    While the user is away the conversation is not on screen, so no frame for it is
    ever seen: nobody can witness "the queue was empty". The rule never asks to. The
    first frame on return names the outstanding ids, and none of them is one the user
    waved off, which is all it takes.
    """
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)
    # Exactly one frame, and it is already the refilled queue.
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 60, ids={"c"}) is OpenDecision.OPEN
    )


def test_returning_to_an_emptied_queue_forgets_the_dismissal_for_the_next_batch() -> None:
    """The other half of effect 3: an EMPTY frame on return releases the record too.

    Nothing opens on it (clause 1), but the dismissal was about asks that are gone,
    so it must not outlive them: the next view that finds pending asks opens.
    """
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)
    assert _decide(policy, "conv-a", QueueReading.EMPTY, at=T0 + 60, ids=set()) is OpenDecision.SKIP
    assert not policy.is_dismissed("conv-a")

    _away_and_back(policy, at=T0 + 120)
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 120, ids={"e"}) is OpenDecision.OPEN
    )


def test_a_new_ask_beside_a_waved_off_one_does_not_force_the_surface_open() -> None:
    """A genuinely new ask arriving later does not override a refusal still in force.

    The bar/chip covers it (clause 4). Only once the OLD asks are gone does the
    conversation stop being held, and then the new one opens with the next view.
    """
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)
    # b is still pending and c is new: held.
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 60, ids={"b", "c"})
        is OpenDecision.SKIP
    )
    # Later b is resolved; only c remains. The next visit opens.
    _away_and_back(policy, at=T0 + 120)
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 120, ids={"c"}) is OpenDecision.OPEN
    )


def test_a_frame_that_cannot_name_every_outstanding_ask_never_releases_a_dismissal() -> None:
    """Fail closed (clause 5): a wire-truncated frame cannot prove the waved-off asks left.

    ``ids=None`` is a frame that does not list them all. The ask the wire bound
    dropped may be exactly the one still outstanding, and releasing on that guess
    would open a panel the user refused.
    """
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 60, ids=None) is OpenDecision.SKIP
    )
    assert policy.is_dismissed("conv-a"), "an incomplete frame spent the record"


def test_a_dismissal_waits_for_a_resolved_frame_to_be_judged() -> None:
    """Judging "is anything I waved off still pending?" needs a frame that can say.

    An unresolved frame waits like any other decision; it neither opens nor releases.
    """
    policy = _dismissed_ab()
    _away_and_back(policy, at=T0 + 60)
    assert (
        _decide(policy, "conv-a", QueueReading.UNRESOLVED, at=T0 + 61, ids=set())
        is OpenDecision.WAIT
    )
    assert policy.is_dismissed("conv-a")
    assert policy.awaiting("conv-a")
    # The resolved frame that follows decides, and sees the refilled queue.
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 62, ids={"c"}) is OpenDecision.OPEN
    )


def test_a_second_dismissal_adds_to_the_record() -> None:
    """Closing again records MORE ids; it never forgets the ones already waved off.

    The case that tells "add" from "replace": the second close happens over a
    TRUNCATED list (the wire bound dropped ask ``a``, which is still outstanding),
    so its pending set is just ``{b}``. On return only ``a`` is left. A replacing
    record would be ``{b}``, find nothing it refused still pending, and open the
    surface over the very ask the user waved off at the first close.
    """
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    policy.note_user_closed("conv-a", pending_ids={"a"})
    policy.note_user_closed("conv-a", pending_ids={"b"})  # a prefix: ``a`` was cut off

    _away_and_back(policy, at=T0 + 60)
    # b was answered; ``a`` — waved off by the FIRST close — is what remains.
    assert (
        _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 60, ids={"a"}) is OpenDecision.SKIP
    )
    assert policy.is_dismissed("conv-a")


def test_closing_with_no_pending_ids_records_nothing() -> None:
    """The boundary of the record: an empty pending set (history only) is not a refusal."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    policy.note_user_closed("conv-a", pending_ids=set())
    policy.note_user_closed("", pending_ids={"a"})  # an unkeyed view cannot be remembered
    assert not policy.is_dismissed("conv-a")
    assert not policy.is_dismissed("")


def test_a_held_dismissal_is_judged_again_on_every_view_not_only_the_first() -> None:
    """The record outlives the view that read it, for as long as its asks do."""
    policy = _dismissed_ab()
    for visit, at in enumerate((T0 + 60, T0 + 120, T0 + 180), start=1):
        _away_and_back(policy, at=at)
        assert (
            _decide(policy, "conv-a", QueueReading.PENDING, at=at, ids={"a", "b"})
            is OpenDecision.SKIP
        ), f"visit {visit} opened a surface the user refused"
    assert policy.is_dismissed("conv-a")


# -- never open on a guess ---------------------------------------------------


def test_unresolved_frames_wait_and_the_first_resolved_one_decides() -> None:
    """Clause 5: a tally-only or not-yet-loaded frame is not "pending asks"."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)

    assert _decide(policy, "conv-a", QueueReading.UNRESOLVED, at=T0 + 1) is OpenDecision.WAIT
    assert _decide(policy, "conv-a", QueueReading.UNRESOLVED, at=T0 + 2) is OpenDecision.WAIT
    assert policy.awaiting("conv-a")
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 3) is OpenDecision.OPEN


def test_an_unresolved_view_never_opens_past_the_window() -> None:
    """A frame that resolves minutes into a view is not "pending on open" (clause 3)."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)

    late = T0 + OPEN_WINDOW_S + 1
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=late) is OpenDecision.SKIP
    assert not policy.awaiting("conv-a")


def test_an_unsupported_runtime_never_opens_anything() -> None:
    """UNRESOLVED for the whole window leaves the surface exactly as it was."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    decisions = {
        _decide(policy, "conv-a", QueueReading.UNRESOLVED, at=T0 + step)
        for step in (0, 10, 20, 30, 40)
    }
    assert decisions == {OpenDecision.WAIT}
    assert _decide(policy, "conv-a", QueueReading.UNRESOLVED, at=T0 + 60) is OpenDecision.SKIP


def test_asks_that_arrived_after_the_view_began_do_not_open_it() -> None:
    """Clause 3 for the first resolved frame: an arrival is not "pending on open"."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert _decide(policy, "conv-a", QueueReading.ARRIVED, at=T0 + 2) is OpenDecision.SKIP
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 3) is OpenDecision.SKIP


# -- never take the keyboard, never fight the user ---------------------------


def test_a_busy_user_is_never_interrupted_even_later() -> None:
    """Clause 5: occupied at the decision means skipped FOR GOOD.

    A user typing (or holding a restored draft, a live prompt, an aside) when the
    conversation opened is not interrupted a minute later by a panel that appears
    once they stop — that would be the focus theft arriving on a delay.
    """
    policy = _policy()
    policy.begin_view("conv-a", now=T0)

    assert _decide(policy, "conv-a", QueueReading.PENDING, occupied=True) is OpenDecision.SKIP
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 2) is OpenDecision.SKIP


def test_an_unresolved_frame_does_not_spend_the_decision_on_a_busy_user() -> None:
    """Busy beats nothing until there is something to open: only a resolved frame decides."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert (
        _decide(policy, "conv-a", QueueReading.UNRESOLVED, occupied=True, at=T0 + 1)
        is OpenDecision.WAIT
    )
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 2) is OpenDecision.OPEN


def test_a_surface_the_user_already_opened_is_left_alone() -> None:
    """Clause 6: the door got there first, so the policy has nothing to add."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert _decide(policy, "conv-a", QueueReading.PENDING, surface_open=True) is OpenDecision.SKIP
    assert not policy.awaiting("conv-a")


def test_opening_by_hand_settles_a_view_still_waiting_on_a_frame() -> None:
    """The user pressed the door while the policy waited: the late frame must not re-open it."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert _decide(policy, "conv-a", QueueReading.UNRESOLVED, at=T0 + 1) is OpenDecision.WAIT

    policy.note_user_opened("conv-a")
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 2) is OpenDecision.SKIP


def test_opening_by_hand_does_not_erase_a_dismissal() -> None:
    """Having closed it once is the fact the contract asks to respect; the record is sticky."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    policy.note_user_closed("conv-a", pending_ids={"a1"})
    policy.note_user_opened("conv-a")
    assert policy.is_dismissed("conv-a")


# -- what a "view" is ---------------------------------------------------------


def test_a_decision_for_another_conversation_is_ignored() -> None:
    """A late frame for a conversation that is no longer on screen decides nothing."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    policy.begin_view("conv-b", now=T0 + 1)
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 1) is OpenDecision.SKIP
    assert policy.awaiting("conv-b")


def test_switching_back_to_a_conversation_begins_a_fresh_view() -> None:
    """Clause 2's "that view of that conversation": each visit gets its own decision."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert _decide(policy, "conv-a", QueueReading.PENDING) is OpenDecision.OPEN

    policy.begin_view("conv-b", now=T0 + 5)
    policy.begin_view("conv-a", now=T0 + 10)
    assert policy.awaiting("conv-a")
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 10) is OpenDecision.OPEN


def test_the_arrival_cutoff_is_the_instant_the_current_view_began() -> None:
    """``opened_at_ms`` moves with each view, so "predates the view" is per visit."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert policy.opened_at_ms == T0_MS
    policy.begin_view("conv-b", now=T0 + 90)
    assert policy.opened_at_ms == int((T0 + 90) * 1000)
    assert policy.view_id == "conv-b"


def test_beginning_the_same_conversation_again_is_not_a_new_view() -> None:
    """A takeover/reload/refresh swaps the session object, not the conversation."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert _decide(policy, "conv-a", QueueReading.PENDING) is OpenDecision.OPEN

    policy.begin_view("conv-a", now=T0 + 30)  # same id: the decision stays spent
    assert not policy.awaiting("conv-a")
    assert policy.opened_at_ms == T0_MS
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 31) is OpenDecision.SKIP


def test_a_settling_switch_waits_instead_of_spending_the_decision() -> None:
    """The composer still holds the previous draft mid-switch: neither "typing" nor
    "empty" is a fact yet, so the view waits for the switch to end."""
    policy = _policy()
    policy.begin_view("conv-a", now=T0)
    assert (
        policy.decide(
            "conv-a",
            QueueReading.PENDING,
            now=T0 + 1,
            occupied=False,
            surface_open=False,
            settling=True,
        )
        is OpenDecision.WAIT
    )
    assert policy.awaiting("conv-a")
    assert _decide(policy, "conv-a", QueueReading.PENDING, at=T0 + 2) is OpenDecision.OPEN


def test_a_view_without_a_conversation_id_is_never_armed() -> None:
    """Its dismissal could not be keyed, so opening it would break clause 4."""
    policy = _policy()
    policy.begin_view("", now=T0)
    assert not policy.awaiting("")
    assert _decide(policy, "", QueueReading.PENDING) is OpenDecision.SKIP


def test_a_disabled_policy_arms_no_view() -> None:
    """The inert switch the pre-existing surface tests and captures use."""
    policy = _policy(enabled=False)
    policy.begin_view("conv-a", now=T0)
    assert not policy.awaiting("conv-a")
    assert _decide(policy, "conv-a", QueueReading.PENDING) is OpenDecision.SKIP


# -- is a frame complete? (what a dismissal is judged on) -----------------------


def test_a_frame_is_complete_exactly_when_its_rows_cover_the_tally() -> None:
    """``asks_open`` counts outstanding rows BEFORE the wire bound; the rows may be fewer.

    The rule is the tally's, and ``asks_truncated`` is not an input at all: the flag stays
    set after every ask is answered (the wire's text budget keeps dropping the settled
    rows), so a rule built on it could never be satisfied again on that conversation.
    """
    # Everything outstanding is named (a dropped SETTLED row is invisible here, which is the
    # point: nothing the user can act on is missing).
    assert names_every_outstanding(tally=3, named_outstanding=3)
    assert names_every_outstanding(tally=0, named_outstanding=0)
    # An outstanding row was dropped: the tally says more than the rows can.
    assert not names_every_outstanding(tally=8, named_outstanding=5)
    # A runtime that does not publish the tally cannot attest to anything.
    assert not names_every_outstanding(tally=None, named_outstanding=0)
    assert not names_every_outstanding(tally=None, named_outstanding=4)


def test_the_completeness_test_has_no_input_for_the_sticky_truncation_flag() -> None:
    """The signature IS the guarantee: the predicate cannot be wedged by a flag it never sees."""
    import inspect

    assert set(inspect.signature(names_every_outstanding).parameters) == {
        "tally",
        "named_outstanding",
    }
