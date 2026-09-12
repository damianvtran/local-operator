"""A send that could not reach a runtime hands the message back, and says why ONCE.

Two failures share this row because they are one fact to the user: a prompt
whose ``prompt()`` raised on a dead socket, and — since QA round 2 (Q-1) — a
queued STEER whose bind was refused after the recovery give-up released it (that
half is pinned in ``test_queued_steer_receipt.py``, where the held-steer harness
lives). In both the runtime is not there and the text is back in the composer.

What this file adds is the RETRY shape U6 measured: while a stale record still
claims a live owner, the resend is refused again in ~0.4 s, so a user who does
what the row says presses twice or three times. Each press used to leave its own
message row and its own copy of the warning — one message became three rows and
the screen grew two identical amber blocks. The message row comes down with the
withdrawal (it was never sent), and the warning is a STATE, so it is painted
once while it stands. It is not a quiet retry: the state is named, and the row
now names the way out that works in that window.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.tui.app import UNSENT_RUNTIME_NOTICE, OperatorApp
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory

#: The refusal a real facade raises for a record that is on disk, claims a live
#: owner, and answers nothing — the shape the U6 measurement planted.
DEAD_RUNTIME = ConnectionError("owner socket unreachable: [Errno 61] Connect call failed")


def _dead_session() -> FakeSession:
    session = FakeSession()

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> None:
        raise DEAD_RUNTIME

    session.prompt = prompt  # type: ignore[assignment]
    return session


def _blocks(app: OperatorApp) -> list[Any]:
    return list(app.query_one(TranscriptView).blocks())


def _notice_texts(app: OperatorApp) -> list[str]:
    return [block._text for block in _blocks(app) if isinstance(block, NoticeBlock)]


def _user_texts(app: OperatorApp) -> list[str]:
    return [block.text() for block in _blocks(app) if isinstance(block, UserBlock)]


async def _boot(pilot: Any, app: OperatorApp) -> Editor:
    """Wait for the session and focus the composer, as this suite's tests do."""
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    assert app._session is not None, "the session never booted"
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    return editor


async def _send(pilot: Any, editor: Editor, text: str) -> None:
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    # The prompt runs in a worker; the failure path paints after it settles.
    for _ in range(100):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if editor.text:
            return


@pytest.mark.asyncio
async def test_a_prompt_on_a_dead_runtime_comes_back_with_one_named_reason() -> None:
    """The first refusal: text back, no row claiming delivery, one reason."""
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "are you there?")

        assert editor.text == "are you there?", "the user's text was not handed back"
        assert _user_texts(app) == [], "a row stood for a message nobody received"
        assert _notice_texts(app) == [UNSENT_RUNTIME_NOTICE]


@pytest.mark.asyncio
async def test_pressing_again_does_not_pile_up_rows_or_warnings() -> None:
    """U6's measurement, reproduced: two presses, one message row, one warning.

    Before this, press two added a second message row AND a second copy of the
    warning. The first press's row is already gone with its own withdrawal, so
    what the user ends up with is one standing state rather than a tally of
    their attempts.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "are you there?")
        await _send(pilot, editor, "are you there?")

        assert editor.text == "are you there?"
        assert _user_texts(app) == [], _user_texts(app)
        assert _notice_texts(app) == [UNSENT_RUNTIME_NOTICE], _notice_texts(app)


@pytest.mark.asyncio
async def test_the_row_names_the_way_out_the_retry_cannot_take() -> None:
    """The copy has to be honest for the shape that shows it.

    `send it again to start a new one` is false while a record still claims a
    live owner — the measured refusals at t+9.87 s and t+13.47 s, served only at
    t+51 s once the planted heartbeat aged out. The row therefore names `/resume`
    as well, which is the lever that works in that window.
    """
    assert "/resume" in UNSENT_RUNTIME_NOTICE, UNSENT_RUNTIME_NOTICE
    assert "back in the composer" in UNSENT_RUNTIME_NOTICE, UNSENT_RUNTIME_NOTICE
