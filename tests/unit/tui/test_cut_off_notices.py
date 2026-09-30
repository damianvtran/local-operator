"""A failure is stated ONCE, and an error row NAMES its cause.

Two defects, one seam. Both live in the pair of producers that can announce a
turn's outcome: ``_finalize_turn`` / ``on_turn_ended`` paints it the moment the
turn ends, and the attention poller reads the same outcome back out of the
durable store a tick later. The poller's only dedupe is
``completion_anchor_id``, and the live row cannot carry one (the anchor does not
exist until the session publishes), so the two announced one fact twice.

The aborted branch fixed that for ``interrupted``. The error branch never got
the fix — ``_own_interrupt_notice`` was only ever set when ``aborted`` — so
every turn that ended with a provider error painted ``✗ <error>`` above
``· Stopped with an error``. These tests pin both branches, including the
pre-existing provider-error duplicate, and pin the copy: the poller's row now
carries the harness-authored reason.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attention import AttentionStore
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_attention import _notice_texts  # the established helper

CUT_OFF = (
    "turn cut off — the runtime retired so the next engage would run a newer build. "
    "The transcript holds what it wrote before that and nothing after."
)
REASON = "the runtime retired so the next engage would run a newer build"


class DeliberateStopSession(FakeSession):
    """A fake owner that reports whether it was told the stop was deliberate.

    Bare ``/stop`` on a TUI-owned session ends in ``Session.dispose()``, and a
    real session's dispose notes ``disposed`` — the involuntary default. So the
    one thing this route must do is record the user's verdict first, and this
    is the flag a test can see it on (``FakeSession`` has no taxonomy of its
    own).
    """

    def __init__(self) -> None:
        super().__init__()
        self.deliberate_stops = 0

    def note_deliberate_stop(self) -> None:
        self.deliberate_stops += 1


class OutcomeSession(FakeSession):
    """A fake owner that publishes one outcome, controllable per test."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.store = AttentionStore(path)
        self.identity = "session/cutoff"
        self.last_token = ""

    def publish(self, kind: str, *, cause: str = "", reason: str = "") -> str:
        token = str(uuid.uuid4())
        self.last_token = token
        self.store.publish(
            self.identity,
            token,
            f"completion-{token}",
            kind,
            reason=reason,
            cause=cause,
        )
        return token

    async def refresh_attention(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.state, self.identity)

    async def acknowledge_attention(self, token: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.store.acknowledge, self.identity, token)


async def _start_and_end_turn(
    app: OperatorApp,
    pilot: Any,
    *,
    aborted: bool,
    error: str | None,
    cut_off_cause: str = "",
) -> None:
    """Drive a turn to the point where the app has painted its own end row."""
    from local_operator.tui.events import TurnEnded, TurnStarted
    from local_operator.tui.widgets.editor import Editor

    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    editor = app.query_one(Editor)
    editor.focus()
    editor.text = "run the long job"
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    app.post_message(TurnStarted())
    await pilot.pause()
    app.post_message(TurnEnded(aborted, error, cut_off_cause=cut_off_cause))
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_a_cut_off_turn_paints_the_reason_and_never_interrupted(
    tmp_path, monkeypatch
) -> None:
    """Design test 15: one row, the error sentence, with the cause in it.

    The classifier rewrote the end event to ``aborted=False, error=<notice>``,
    so the app's aborted branch must NOT also paint ``interrupted`` — the
    vocabulary mismatch the design round reviews.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _start_and_end_turn(app, pilot, aborted=False, error=CUT_OFF)
        notices = _notice_texts(app)
        assert len(notices) == 1, notices
        assert notices[0] == CUT_OFF
        assert "Interrupted" not in notices


@pytest.mark.asyncio
async def test_a_provider_error_is_also_stated_once(tmp_path, monkeypatch) -> None:
    """The pre-existing duplicate, on the branch that never got the fix.

    Before this change the error branch left ``_own_interrupt_notice`` unset, so
    the poller appended its own row for the same outcome: the user read the
    failure twice. The two-line check the design asks for is
    ``_adopt_own_interrupt_notice("error", anchor) is True`` with a live error
    row held.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        endpoint = "api refused the request: 502 Bad Gateway"
        await _start_and_end_turn(app, pilot, aborted=False, error=endpoint)
        assert _notice_texts(app) == [endpoint]

        session.publish("error", reason=endpoint)
        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._adopt_own_interrupt_notice("error", anchor) is True
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == [endpoint], "the failure must not be restated"


@pytest.mark.asyncio
async def test_a_retired_outcome_restates_the_live_cut_row_instead_of_doubling_it(
    tmp_path, monkeypatch
) -> None:
    """The ONE adopted row that is RESTATED (retire-for-build arm, 2026-09-29).

    The live row for a cut turn is painted by the error branch — the classified
    end arrives as ``aborted=False, error=<cut sentence>`` — so a ``retired``
    outcome adopts THAT row (the kind correspondence is the arm's own) and
    restates it: leaving the live cut sentence in the error tier would keep
    exactly the failure framing the arm exists to remove. One cut, one row.

    Since design round 2 D2 the live row is ALSO painted in the warning tier at
    the classifier (``cut_off_cause``), so there is no ~1 s of danger ink
    waiting for the restate; the two assertions below pin both halves.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    from local_operator.harness.rows import RETIRED_NOTICE_TEXT

    async with app.run_test(size=(100, 30)) as pilot:
        sentence = "the runtime retired so the next engage would run a newer build"
        await _start_and_end_turn(
            app, pilot, aborted=False, error=sentence, cut_off_cause="runtime-retired"
        )
        assert _notice_texts(app) == [sentence]
        # D2: warning ink from the FIRST frame, never the ~1 s of danger the
        # restate used to correct.
        assert all(block._token != "danger" for block in _notice_blocks(app))

        session.publish("retired", cause="runtime-retired")
        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._adopt_own_interrupt_notice("retired", anchor) is True
        await app._poll_completion_attention()
        await pilot.pause()

        texts = _notice_texts(app)
        assert texts == [RETIRED_NOTICE_TEXT], texts
        assert not any("Stopped with an error" in text for text in texts), texts
        assert all(block._token != "danger" for block in _notice_blocks(app))


@pytest.mark.asyncio
async def test_a_closed_outcome_restates_an_aborted_live_row_into_the_info_tier(
    tmp_path, monkeypatch
) -> None:
    """THE CARRIED CUT'S ONE ROW (design D1 on #1829, 2026-09-30).

    A carried run cancelled in the one-shot handoff paints the live ABORT row
    (``! interrupted``, the aborted branch — the end arrives ``aborted=True``,
    which is the pairing v2's captures missed by checking ``aborted=False``)
    and the session publishes ``closed``. Pre-change the poller REFUSED the
    kind, so the live ``! interrupted`` and the poller's dim closure row stood
    together: two rows for one cut, one of them reading as an interruption that
    had been superseded.

    The adoption restates in place — the closure's own words
    (``CLOSED_NOTICE_TEXT``) in the info tier — so the frame carries ONE row
    for the cut, and it is the receipt the outcome (and both other surfaces)
    call neutral. The retired cell above stays the warning-tier control.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    from local_operator.harness.rows import CLOSED_NOTICE_TEXT

    async with app.run_test(size=(100, 30)) as pilot:
        await _start_and_end_turn(app, pilot, aborted=True, error=None)
        # The live row the handoff paints, held for adoption.
        assert _notice_texts(app) == ["interrupted"]

        session.publish("closed", cause="disposed")
        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._adopt_own_interrupt_notice("closed", anchor) is True
        await app._poll_completion_attention()
        await pilot.pause()

        texts = _notice_texts(app)
        assert texts == [CLOSED_NOTICE_TEXT], texts
        # The info tier, POSITIVELY: `info` is the `dim` token — the same ink
        # the poller's own closure row uses — where the adopted row was warning
        # `interrupted` before the restate.
        assert [block._token for block in _notice_blocks(app)] == ["dim"]
        assert not any("interrupted" in text for text in texts), texts


@pytest.mark.asyncio
async def test_the_closed_adoption_declines_a_mismatched_live_row(tmp_path, monkeypatch) -> None:
    """The correspondence stays a guard, not a blanket (design D1).

    `closed` supersedes the ABORT row; a live row for a DIFFERENT fact must
    still refuse, so a held ``error`` row keeps its own wording and the
    poller's closure row is not suppressed in its favour.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))

    async with app.run_test(size=(100, 30)) as pilot:
        endpoint = "api refused the request: 502 Bad Gateway"
        await _start_and_end_turn(app, pilot, aborted=False, error=endpoint)
        assert _notice_texts(app) == [endpoint]

        session.publish("closed", cause="disposed")
        anchor = session.store.state(session.identity)["anchor_id"]
        assert app._adopt_own_interrupt_notice("closed", anchor) is False
        hold = app._own_interrupt_notice
        assert hold is not None
        assert hold._text == endpoint


@pytest.mark.asyncio
async def test_the_poller_names_the_cause_for_a_cut_off_it_never_painted(
    tmp_path, monkeypatch
) -> None:
    """The away case keeps its row — and gains the reason.

    A user returning to a session cut off while they were elsewhere has no live
    row to adopt, so the poller's announcement is the only one; it must carry
    the cause rather than a bare class marker.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        session.publish("error", cause="runtime-retired", reason=REASON)
        await app._poll_completion_attention()
        await pilot.pause()

        assert _notice_texts(app) == [f"Stopped with an error — {REASON}"]


@pytest.mark.asyncio
async def test_an_interrupted_outcome_keeps_its_own_spelling(tmp_path, monkeypatch) -> None:
    """The regression guard: a real stop still reads as one, with no reason."""
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        session.publish(
            "interrupted", cause="user-stop", reason="the session was stopped by the user"
        )
        await app._poll_completion_attention()
        await pilot.pause()
        # The deliberate stop keeps the existing spelling: the reason is for a
        # cut-off the user did not ask for, and appending "stopped by the user"
        # to a row that already says Interrupted is noise.
        assert _notice_texts(app) == ["Interrupted"]


def _notice_blocks(app: OperatorApp) -> list[Any]:
    from local_operator.tui.widgets.transcript import NoticeBlock

    return [b for b in app._transcript_view().blocks() if isinstance(b, NoticeBlock)]


@pytest.mark.asyncio
async def test_the_poller_paints_a_cut_off_in_the_danger_tier(tmp_path, monkeypatch) -> None:
    """D1: one cut-off must not be an alarm live and a whisper on return.

    The poller's error branch passed no kind, so it took ``NoticeBlock``'s
    ``info`` default — the same ``·`` glyph in the same dim fill as the routine
    ``Interrupted`` control one branch away, for an outcome the live surface
    paints ``✗`` danger. The row decision now comes from ``harness/rows.py``,
    which is why the assertion is on the tier rather than on the words.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        session.publish("error", cause="runtime-retired", reason=REASON)
        await app._poll_completion_attention()
        await pilot.pause()

        (block,) = _notice_blocks(app)
        assert block._glyph == "✗"
        assert block._token == "danger"


@pytest.mark.asyncio
async def test_the_pollers_interrupted_control_stays_quiet(tmp_path, monkeypatch) -> None:
    """And the deliberate stop keeps the tier the design round signed off.

    Its louder ``warning`` ink belongs to the LIVE row, which is a statement
    about the turn the user just ended; a replayed receipt for the same stop is
    not, and promoting it would put two weights on one fact.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        session.publish("interrupted", cause="user-stop", reason="the session was stopped")
        await app._poll_completion_attention()
        await pilot.pause()

        (block,) = _notice_blocks(app)
        assert block._glyph == "·"
        assert block._token == "dim"


@pytest.mark.asyncio
async def test_the_tui_stop_route_records_the_deliberate_verdict(tmp_path, monkeypatch) -> None:
    """BLOCKER-1 through the APP: bare ``/stop`` notes the stop before disposing.

    ``Session.dispose()`` notes ``disposed`` unconditionally, so this call is
    what stands between the user's own cancel and a published
    ``kind=error, cause=disposed``. Asserted on the session the app actually
    disposes, which is the route the unit guard and the e2e cell both missed.
    """
    session = DeliberateStopSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        assert app._session is session
        await app._stop_local_session()
        await pilot.pause()

    assert session.deliberate_stops == 1, "the dispose route lost the user's verdict"
    assert session.disposed is True


@pytest.mark.asyncio
async def test_a_cut_off_turns_live_cards_are_retired_as_cut_off(tmp_path, monkeypatch) -> None:
    """D2 end to end through the app: the end event's verdict reaches the row.

    ``TurnEnded`` is where the fact that this was an involuntary stop exists
    (the classifier reports it as ``aborted=False, error=<notice>``, so nothing
    downstream can infer it), and ``_finalize_turn`` is the only turn-death path
    that owns the stranded cards. Without the flag carried across that seam the
    ledger said ``⊘ interrupted`` under a ``✗ turn cut off`` notice — the two
    words disagreeing about one death, on one screen.
    """
    from local_operator.tui.events import (
        ToolExecutionStartEvent,
        ToolStarted,
        TurnEnded,
        TurnStarted,
    )
    from local_operator.tui.widgets.editor import Editor

    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(200):
            if app._session is not None:
                break
            await pilot.pause()
            await asyncio.sleep(0.01)
        editor = app.query_one(Editor)
        editor.focus()
        editor.text = "run the long job"
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        app.post_message(TurnStarted())
        await pilot.pause()
        app.post_message(
            ToolStarted(
                ToolExecutionStartEvent(
                    tool_call_id="c-cut", tool_name="bash", args={"command": "sleep 60"}
                )
            )
        )
        await pilot.pause()
        card = app._tool_cards["c-cut"]
        assert "cut off" not in card._build_row(100).plain

        app.post_message(TurnEnded(False, CUT_OFF, cut_off=True))
        await pilot.pause()
        await pilot.pause()

        row = card._build_row(100).plain
        assert "cut off" in row, row
        assert "interrupted" not in row, row


@pytest.mark.asyncio
async def test_a_settled_run_paints_no_error_card_and_the_poller_stays_quiet(
    tmp_path, monkeypatch
) -> None:
    """The settle arm's surfaces: no verdict means no card — live or on return.

    Session 664a234ec561's disposal settled a zero-work delivery run with an
    ``eligible:False`` marker and NO store row (``Session.dispose``), leaving
    the completed turn's own ``complete`` as the latest record. The surfaces
    must read that as nothing to say: the live empty aborted end — the turn
    row the abort persisted ([775]: an assistant message with
    ``stop_reason="aborted"`` and no content) — keeps its standing
    ``interrupted`` row (``test_attention.py`` pins that for any aborted
    turn) and must paint NO error card, and the poller, reading the store the
    settled session left behind, must not synthesise one from the absence.

    The LAST phase is the canary: it publishes the row the fix removes
    (``error | cause=disposed``) and requires the same instrument to report
    it, so the quiet readings above cannot come from a dead probe.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _start_and_end_turn(app, pilot, aborted=True, error=None)
        notices = _notice_texts(app)
        assert not any("Stopped with an error" in text for text in notices), notices
        assert all(
            block._token != "danger" for block in _notice_blocks(app)
        ), "the settled run's end must not wear the danger tier"

        # The store a settled session leaves: the work turn's own `complete`
        # is still the latest row — the disposal published nothing on top.
        session.publish("complete")
        before = _notice_texts(app)
        await app._poll_completion_attention()
        await pilot.pause()
        assert _notice_texts(app) == before, "the poller must stay quiet on the settle"

        # CANARY for the instrument above.
        session.publish(
            "error",
            cause="disposed",
            reason="the session was disposed while this turn was running",
        )
        await app._poll_completion_attention()
        await pilot.pause()
        assert any(
            "Stopped with an error" in text for text in _notice_texts(app)
        ), "the canary must be reported, or the quiet readings above are dead"


@pytest.mark.asyncio
async def test_the_poller_paints_a_closed_run_neutrally_and_never_in_danger(
    tmp_path, monkeypatch
) -> None:
    """THE v2 NEUTRAL ROW: a closure is a receipt, not a failure.

    Session 23fc556c3799 (2026-09-29): the disposal published ``error|disposed``
    for a zero-work run and the operator's completed turn read "Stopped with an
    error" on both surfaces. The disposal publishes ``closed`` now; the poller
    must paint its sentence in the info tier — never the danger tier — with the
    words from the shared row decision (``harness/rows.py``), so the TUI, the
    phone and the desktop cannot drift. The error and interrupted cells above
    stay the still-paints-in-their-own-tier controls.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    from local_operator.harness.rows import CLOSED_NOTICE_TEXT

    async with app.run_test(size=(100, 30)) as pilot:
        await _start_and_end_turn(app, pilot, aborted=False, error=None)
        session.publish("closed", cause="disposed")
        await app._poll_completion_attention()
        await pilot.pause()
        texts = _notice_texts(app)
        assert any(CLOSED_NOTICE_TEXT in text for text in texts), texts
        assert not any("Stopped with an error" in text for text in texts), texts
        assert all(
            block._token != "danger" for block in _notice_blocks(app)
        ), "a closure must never paint in the danger tier"


def test_the_closed_closure_reads_completed_and_keeps_the_info_tier() -> None:
    """The copy and the tier, pinned at the one decision both surfaces read."""
    from local_operator.harness.rows import CLOSED_NOTICE_TEXT, completion_notice

    text, severity = completion_notice(
        "closed", "the session was disposed while this turn was running"
    )
    assert text == CLOSED_NOTICE_TEXT == "Completed — runtime retired/disposed"
    assert severity == "info", "the closure must never wear the danger tier"


def test_the_retired_row_reads_retired_for_an_update_and_keeps_the_warning_tier() -> None:
    """The retire-for-build arm's copy and tier, at the shared decision point.

    Seed 7e797aaaf6e7 (2026-09-29): the bound-cut row must stay TRUTHFUL (the
    turn was cut) but DISTINGUISHED FROM A FAILURE — warning ink, its own
    sentence, never danger, and never the info whisper a closure gets.
    """
    from local_operator.harness.rows import RETIRED_NOTICE_TEXT, completion_notice

    text, severity = completion_notice(
        "retired", "the runtime retired so the next engage would run a newer build"
    )
    assert (
        text
        == RETIRED_NOTICE_TEXT
        == ("Retired for an update — a turn was in flight and was cut; its earlier output is kept")
    )
    assert severity == "warning", "a cut for an update is warning, never danger"


@pytest.mark.asyncio
async def test_the_poller_paints_a_retired_run_in_warning_and_never_in_danger(
    tmp_path, monkeypatch
) -> None:
    """THE RETIRE-FOR-BUILD ROW on the returning surface.

    Session 7e797aaaf6e7 (2026-09-29): a bound-expired build drain cut a live
    turn and the returned-to row read "Stopped with an error". The arm
    publishes ``retired`` now; the poller must paint its sentence in the
    warning tier — never danger, never the info whisper a closure gets — with
    the words from the shared row decision (``harness/rows.py``), so the TUI,
    the phone and the desktop cannot drift. The error and interrupted cells
    above stay the still-paints-in-their-own-tier controls.
    """
    session = OutcomeSession(tmp_path / "attention.db")
    monkeypatch.setattr("local_operator.tui.attention.terminal_is_foreground", lambda: True)
    app = OperatorApp(lambda: _factory(session))
    from local_operator.harness.rows import RETIRED_NOTICE_TEXT

    async with app.run_test(size=(100, 30)) as pilot:
        await _start_and_end_turn(app, pilot, aborted=False, error=None)
        session.publish("retired", cause="runtime-retired")
        await app._poll_completion_attention()
        await pilot.pause()
        texts = _notice_texts(app)
        assert any(RETIRED_NOTICE_TEXT in text for text in texts), texts
        assert not any("Stopped with an error" in text for text in texts), texts
        assert all(
            block._token != "danger" for block in _notice_blocks(app)
        ), "a cut for an update must never paint in the danger tier"


def test_the_returned_to_turn_notice_carries_an_escalated_stops_attribution() -> None:
    """Design round 1, D1 on the row the TUI poller and the phone SHARE.

    ``rows.completion_notice`` builds this row for both surfaces, and it was
    kind-gated: an ``interrupted`` outcome returned the bare word whatever the
    reason said, so a rung-3 kill and a rung-1 request painted the same eleven
    cells on the TUI and on the phone. Only the ESCALATED rung appends, because
    ``Interrupted`` already means the operator stopped it — what they cannot
    learn from the word is that the ladder had to signal the runtime to make it
    stop, and that is the fact this PR's marker records.
    """
    from local_operator.harness.rows import completion_notice
    from local_operator.incidents import (
        CUT_OFF_UNKNOWN,
        DELIBERATE_CUT_OFF_CAUSE,
        render_cut_off_reason,
        render_stop_attribution,
    )

    def stop(rung: str, command: str) -> str:
        return render_cut_off_reason(
            DELIBERATE_CUT_OFF_CAUSE,
            detail=render_stop_attribution(rung=rung, command=command, killer_pid=40609),
        )

    assert completion_notice("interrupted", stop("sigkill", "/stop --all")) == (
        "Interrupted — killed by /stop --all",
        "info",
    )
    assert completion_notice("interrupted", stop("sigterm", "/stop")) == (
        "Interrupted — stopped with a signal by /stop",
        "info",
    )
    # Rung 1, a pre-attribution record, and the no-evidence sentence all read
    # exactly as they did before this change.
    for quiet in (stop("socket", "/stop"), CUT_OFF_UNKNOWN, ""):
        assert completion_notice("interrupted", quiet) == ("Interrupted", "info")
    # The error arm is untouched: it prints the reason whole, and its tier is
    # still the loud one.
    assert completion_notice("error", "boom") == ("Stopped with an error — boom", "error")
