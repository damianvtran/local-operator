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
