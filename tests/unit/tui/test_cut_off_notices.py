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
    app: OperatorApp, pilot: Any, *, aborted: bool, error: str | None
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
    app.post_message(TurnEnded(aborted, error))
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
