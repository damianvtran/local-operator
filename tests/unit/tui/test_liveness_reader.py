"""The TUI's liveness reader: a stale or absent stamp must not read as live.

Every test here runs WITHOUT a runtime, because the property under test is a
classification of state the session already holds, not a round trip. That is also
the design constraint the task names: the read path answers in 170.7-350.6 ms and
the warm receipt in 6.3-100.6 ms today, so a reader that dialled to look honest
would spend the operator's latency to buy truthfulness he can have for free.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from local_operator.session.runtime.types import LIVE_FRESHNESS_BUDGET_S
from local_operator.tui.liveness import (
    LIVENESS_PROBE_BUDGET_S,
    LIVENESS_PROBE_EVERY_S,
    LIVENESS_TEXT,
    LivenessProbe,
    OwnerLiveness,
    liveness_text,
    owner_liveness,
)


class Owner:
    """A session as the reader sees it: two attributes, no methods called."""

    def __init__(self, *, verified_at: float | None = None, attaching: bool = False) -> None:
        self.verified_at = verified_at
        self.attaching = attaching
        self.touched: list[str] = []

    def __getattr__(self, name: str) -> Any:
        # Any attribute actually READ by the reader is recorded, so a test can
        # prove the reader consulted nothing else -- in particular, nothing that
        # would dial.
        self.__dict__.setdefault("touched", []).append(name)
        raise AttributeError(name)


NOW = 1_000_000.0


# --------------------------------------------------------------------------
# The acceptance criteria: stale and absent are NOT live; attaching is "coming".
# --------------------------------------------------------------------------


def test_a_fresh_stamp_is_live():
    session = Owner(verified_at=NOW - 1.0)
    assert owner_liveness(session, now=NOW) is OwnerLiveness.LIVE


def test_a_stamp_older_than_the_budget_is_not_live():
    """THE MEASURED RESIDUAL: a 21.0-21.1 s old stamp on a live frame.

    1.40-1.41x the 15 s budget, and it used to read ``cold:false`` with nothing
    degrading it. On the TUI it must not read live.
    """
    session = Owner(verified_at=NOW - 21.05)
    assert owner_liveness(session, now=NOW) is OwnerLiveness.STALE
    assert LIVE_FRESHNESS_BUDGET_S < 21.05, "the residual is measured past the budget"


def test_an_absent_stamp_is_never_live():
    """No answered round trip, nothing arriving: never, and not live."""
    assert owner_liveness(Owner(), now=NOW) is OwnerLiveness.NEVER


def test_attaching_reads_as_coming_rather_than_never():
    """The token's whole point: "coming" is distinguishable from "never"."""
    assert owner_liveness(Owner(attaching=True), now=NOW) is OwnerLiveness.COMING


def test_attaching_with_a_stale_stamp_still_reads_as_coming():
    session = Owner(verified_at=NOW - 60.0, attaching=True)
    assert owner_liveness(session, now=NOW) is OwnerLiveness.COMING


def test_a_resync_on_a_freshly_answered_owner_is_live_not_absent():
    """RR1-5's trap in the reader: ``attaching`` is not a liveness term.

    A mid-refresh viewer whose owner answered a moment ago is LIVE -- treating it
    as absent is the conflation ``owner_reachable`` exists to prevent.
    """
    session = Owner(verified_at=NOW - 0.2, attaching=True)
    assert owner_liveness(session, now=NOW) is OwnerLiveness.LIVE


def test_the_budget_boundary_is_inclusive_and_measured_against_the_reader_clock():
    """Age is checked by the READER, never asserted by the writer (§6 rule 2)."""
    assert owner_liveness(Owner(verified_at=NOW - LIVE_FRESHNESS_BUDGET_S), now=NOW) is (
        OwnerLiveness.LIVE
    )
    just_past = Owner(verified_at=NOW - LIVE_FRESHNESS_BUDGET_S - 0.001)
    assert owner_liveness(just_past, now=NOW) is OwnerLiveness.STALE


def test_the_clock_is_read_when_not_injected():
    """The production call site passes no clock; the default must be ``now``."""
    fresh = Owner(verified_at=time.time())
    assert owner_liveness(fresh) is OwnerLiveness.LIVE


# --------------------------------------------------------------------------
# No round trip on the paint path.
# --------------------------------------------------------------------------


def test_the_reader_never_touches_the_owner():
    """It reads the two attributes and NOTHING else -- no method, no dial.

    ``Owner.__getattr__`` records and raises, so a reader that reached for a
    client, a socket or a probe would fail here rather than merely be slower.
    """
    session = Owner(verified_at=NOW - 1.0)
    assert owner_liveness(session, now=NOW) is OwnerLiveness.LIVE
    assert not any(
        name.startswith(("verify", "_client", "connect", "dial")) for name in session.touched
    ), session.touched


def test_a_session_with_neither_attribute_is_never_not_an_error():
    """The status band can be asked to render an owner-side ``Session``."""

    class Bare:
        pass

    assert owner_liveness(Bare(), now=NOW) is OwnerLiveness.NEVER


# --------------------------------------------------------------------------
# The copy seam: one table, and every state has a word in it.
# --------------------------------------------------------------------------


def test_every_state_has_exactly_one_word_and_it_comes_from_the_table():
    for state in OwnerLiveness:
        assert state in LIVENESS_TEXT, f"{state} would render as nothing"
        assert liveness_text(state) == LIVENESS_TEXT[state]
    assert len(set(LIVENESS_TEXT.values())) == len(OwnerLiveness), "two states share a word"


def test_the_words_are_the_designers_to_choose_not_the_readers():
    """A guard on the SEAM, not on the copy.

    It fails only if someone starts spelling liveness words outside this module,
    which is what makes a reword a substitution rather than a hunt.
    """
    source = __import__("pathlib").Path(__file__).parent.parent.parent / "local_operator" / "tui"
    offenders = []
    for path in source.rglob("*.py"):
        if path.name == "liveness.py":
            continue
        text = path.read_text(encoding="utf-8")
        for word in LIVENESS_TEXT.values():
            if f'"{word}"' in text or f"'{word}'" in text:
                offenders.append(f"{path.name}:{word}")
    assert not offenders, "liveness words spelled outside the copy table: " + ", ".join(offenders)


def test_live_is_the_only_state_that_renders_as_an_empty_string():
    """REPLACES an earlier test that asserted every state renders something.

    That test was written against my provisional copy, where `LIVE` had the word
    "live". The designer's decision makes `LIVE` render NOTHING -- a live claim is
    up to a budget stale by construction, so painting one would re-introduce the
    fresh confident-wrong statement this reader exists to remove. The assertion is
    inverted here rather than deleted, so the reversal is visible in the diff.
    """
    empty = [state for state in OwnerLiveness if not liveness_text(state).strip()]
    assert empty == [OwnerLiveness.LIVE], empty
    for state in OwnerLiveness:
        if state is not OwnerLiveness.LIVE:
            assert liveness_text(state).strip(), state


# --------------------------------------------------------------------------
# The designer's copy, and the seam that carries it.
# --------------------------------------------------------------------------


def test_the_copy_is_the_designers_and_live_paints_nothing():
    """The literal strings, pinned so a reword is a deliberate act."""
    assert LIVENESS_TEXT[OwnerLiveness.LIVE] == "", "a live claim must paint no cell"
    assert LIVENESS_TEXT[OwnerLiveness.COMING] == "Connecting…"
    assert LIVENESS_TEXT[OwnerLiveness.STALE] == "Not answering"
    assert LIVENESS_TEXT[OwnerLiveness.NEVER] == "No owner"


def test_the_row_is_present_if_and_only_if_the_reader_is_not_live():
    """The designer's checkable form, as a test.

    A LIVE state paints the ordinary band byte-identically to today; every other
    state takes the row. This is the property the ux round judges, so it is
    pinned here rather than described.
    """
    for state in OwnerLiveness:
        painted = bool(liveness_text(state, now=NOW, verified_at=NOW - 999))
        assert painted is (state is not OwnerLiveness.LIVE), state


def test_stale_carries_its_age_and_the_head_comes_first():
    """`Not answering · 4m` -- head first, so the row's ellipsis eats the age."""
    text = liveness_text(OwnerLiveness.STALE, now=NOW, verified_at=NOW - 240)
    assert text.startswith("Not answering"), text
    assert text == f"Not answering · {text.split(' · ')[1]}"
    assert text.split(" · ")[1], "the age must not be empty"


def test_the_stale_age_is_bounded_for_the_narrow_row():
    """Head + age <= 22 cells (the designer's hard bound), at any age."""
    for age in (1, 59, 60, 3_600, 86_400, 30 * 86_400):
        text = liveness_text(OwnerLiveness.STALE, now=NOW, verified_at=NOW - age)
        assert len(text) <= 22, (age, text, len(text))


def test_stale_without_a_stamp_to_measure_from_degrades_to_its_head():
    """No invented durations: a made-up age is the same class of lie."""
    assert liveness_text(OwnerLiveness.STALE) == "Not answering"


def test_only_stale_grows_an_age():
    for state in OwnerLiveness:
        if state is OwnerLiveness.STALE:
            continue
        assert liveness_text(state, now=NOW, verified_at=NOW - 240) == LIVENESS_TEXT[state], state


# --------------------------------------------------------------------------
# The precondition: someone must have asked for STALE to mean anything.
# --------------------------------------------------------------------------


def test_the_probe_cadence_is_inside_the_budget_it_feeds():
    """A healthy owner must be re-verified before its stamp can expire.

    At a 15 s budget a probe every 5 s leaves two missed rounds of slack; a
    cadence at or above the budget would let a FINE session age into STALE,
    which is the confident-wrong statement this reader exists to avoid.
    """
    assert LIVENESS_PROBE_EVERY_S < LIVE_FRESHNESS_BUDGET_S
    assert LIVENESS_PROBE_EVERY_S * 3 <= LIVE_FRESHNESS_BUDGET_S + 1e-9
    assert LIVENESS_PROBE_BUDGET_S <= 2.0, "the probe must not outlive a paint frame"


def test_the_probe_is_due_on_a_cadence_and_immediately_when_never_run():
    probe = LivenessProbe()
    assert probe.due(None, now=NOW) is True
    assert probe.due(NOW - 1.0, now=NOW) is False
    assert probe.due(NOW - LIVENESS_PROBE_EVERY_S, now=NOW) is True


def test_a_probe_that_cannot_answer_is_false_and_raises_nothing():
    """`verify_live` absent, or raising, is a False -- never a crash on a tick."""

    class NoProbe:
        pass

    class Exploding:
        async def verify_live(self, timeout: float) -> bool:
            raise TimeoutError("owner froze")

    class Answering:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        async def verify_live(self, timeout: float) -> bool:
            self.timeouts.append(timeout)
            return True

    probe = LivenessProbe()
    assert asyncio.run(probe.tick(NoProbe())) is False
    assert asyncio.run(probe.tick(Exploding())) is False
    owner = Answering()
    assert asyncio.run(probe.tick(owner)) is True
    assert owner.timeouts == [LIVENESS_PROBE_BUDGET_S], "the bound must be the constructor's"


def test_a_probe_answer_makes_a_quiet_session_read_live_instead_of_stale():
    """The precondition, end to end at the reader's level.

    A session whose owner answered 4 minutes ago is STALE; the probe answering
    moves the stamp and the same session becomes LIVE. Without the probe -- the
    shipped state -- the first reading is what a HEALTHY quiet session would have
    painted.
    """
    quiet = Owner(verified_at=NOW - 240)
    assert owner_liveness(quiet, now=NOW) is OwnerLiveness.STALE
    quiet.verified_at = NOW  # what verify_live's stamp does
    assert owner_liveness(quiet, now=NOW) is OwnerLiveness.LIVE
    assert liveness_text(OwnerLiveness.LIVE) == ""


# --------------------------------------------------------------------------
# THE FEATURE, DRIVEN THE WAY THE REVIEWER DROVE IT: the app's own timer.
#
# The tests above are at the right level for the CLASSIFICATION and were the
# wrong level for the FEATURE. Both the reader and the band rendered correctly
# while the row could not reach the screen at all: the repaint gate watched the
# verdict ENUM, a failed probe deliberately changes nothing, so `before is after
# is STALE` and six probes over 1800 s produced zero repaints. Green tests at
# the wrong level are how that shipped, so this drives `OperatorApp` itself.
#
# AND IT USES A REAL `AttachedSession`, not a double. The first draft handed the
# pilot a `SidebarRemote`, which does not satisfy `_is_viewer` — and a test that
# has to DEFEAT `_is_viewer` to pass is testing something the app would never do.
# The guard is correct; the double was the wrong object. `BoundClient` is the
# known-good way (the task-6 SIGSTOP tests use it): a real facade, made to look
# bound, with an owner that does not answer.
# --------------------------------------------------------------------------


class BoundClient:
    """A connected dial whose owner never answers — a frozen owner's wire shape."""

    connected = True

    def __init__(self, *, answers: bool = False) -> None:
        self.answers = answers
        self.ops: list[str] = []

    def close(self) -> None:
        pass

    async def request_ack_with_duplicate(self, op, **kwargs):
        self.ops.append(op)
        if not self.answers:
            raise TimeoutError("owner did not answer")
        return ("pong", False)


async def _viewer_facade(tmp_path, *, quiet_for: float, answers: bool = False):
    """A real viewer facade that last heard from its owner ``quiet_for`` ago."""
    from local_operator.session.attached import AttachedSession

    session = await AttachedSession.cold(
        "s1",
        config_dir=tmp_path,
        cwd=str(tmp_path),
        takeover_factory=lambda *a, **k: None,
    )
    session._client = BoundClient(answers=answers)  # type: ignore[assignment]
    session._ready_for_events = True
    session._verified_at = time.time() - quiet_for
    assert session.is_cold is False, "the premise: the facade must look bound"
    assert callable(getattr(session, "verify_live", None)), "the probe must exist"
    return session


@pytest.mark.asyncio
async def test_the_apps_own_row_text_for_a_real_silent_facade(tmp_path, monkeypatch):
    """The harness, and what it proves TODAY.

    `_is_viewer` is True for a real `AttachedSession` made to look bound (it was
    False for the `SidebarRemote` double, which is why the first draft could not
    work: the guard was right and the double was the wrong object), and the app's
    own reader turns that facade into the row it would paint. What this does NOT
    yet prove is that the app REPAINTS — see the note at the top of this file.
    """
    from local_operator.tui.app import OperatorApp, _is_viewer
    from tests.unit.tui.test_app_pilot import _factory
    from tests.unit.tui.test_sidebar_swap_reset import SidebarRemote

    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    app = OperatorApp(lambda: _factory(SidebarRemote("home00000000")))
    async with app.run_test(size=(120, 36)) as pilot:
        for _ in range(10):
            await pilot.pause()
        session = await _viewer_facade(tmp_path, quiet_for=60.0)
        app._interaction.session = session
        assert _is_viewer(session) is True, "a real bound facade must pass the viewer guard"
        row = app._liveness_row_text(session)
        assert row.startswith("Not answering"), row
        assert "\u00b7" in row, "STALE always carries its measured tail"
