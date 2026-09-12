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
once while it stands.

ROUND 3 CHANGED BOTH HALVES OF THAT, and the cells at the bottom pin them. The
row no longer names `/resume`, because QA MINOR-4 and UX U2 measured it as a
lever this shape cannot use (it reopens onto `owner socket unreachable`, and
`/resume` typed into the composer holding the returned text is consumed as a
command that takes the message with it). And the row STANDS ONLY WHILE IT IS
TRUE: U1 measured it still asserting `your message is back in the composer` at
t+40, with an empty composer and the answered exchange below it, so the served
send now retires it — and a draft with no caret of its own lands at the END of
the restored text, which is what stops the next input being glued in front of
it (U2).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.tui.app import UNSENT_RUNTIME_NOTICE, OperatorApp
from local_operator.tui.events import UserMessageStart
from local_operator.tui.session_interaction import SessionDraft, SessionInteraction
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


async def _serving_prompt(text: str, images: Any = None, **kwargs: Any) -> None:
    """A send that reaches a runtime: the composer draft goes out and stays out."""
    return None


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
async def test_the_row_does_not_name_a_lever_the_shape_cannot_use() -> None:
    """The copy has to be honest for the shape that shows it (round 3, U1 + MINOR-4).

    The `/resume` clause this replaces was measured as a second promise that same
    shape cannot keep: in the live-but-silent window `/resume` reopens the session
    and immediately surfaces `owner socket unreachable: [Errno 61] Connect call
    failed ('127.0.0.1', 1)` — reopening does not clear the live-pid claim that
    refuses the bind — and a `/resume` typed into the composer holding the
    returned text dispatches as a command and takes the message with it. What is
    left is the one move that does work, and the row promises no timing for it.
    """
    assert "/resume" not in UNSENT_RUNTIME_NOTICE, UNSENT_RUNTIME_NOTICE
    # Both halves that ARE true stay: where the text is, and the act that starts
    # a runtime again.
    assert "back in the composer" in UNSENT_RUNTIME_NOTICE, UNSENT_RUNTIME_NOTICE
    assert "send it again" in UNSENT_RUNTIME_NOTICE, UNSENT_RUNTIME_NOTICE


@pytest.mark.asyncio
async def test_the_row_is_retired_once_the_message_it_describes_is_served() -> None:
    """UX round 3, U1: the row is a STATE, so being painted once is not enough.

    Measured on the real app: hand-back at ~t+8.5, a refused resend at t+12.2,
    the ghost record gone at t+20.1, the resend SERVED at t+25.4 — and at t+40
    the row still read `your message is back in the composer` over an empty
    composer with the answered exchange directly below it. The served send is
    the first moment the claim is false, so that is where it is taken down.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "are you there?")
        assert _notice_texts(app) == [UNSENT_RUNTIME_NOTICE]

        # The runtime comes back and the message goes out: the draft the row
        # describes leaves the composer.
        session.prompt = _serving_prompt  # type: ignore[assignment]
        editor.text = "are you there?"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if app._pending_user_echoes:
                break
        # The session announces the prompt back the way a real one does, which
        # is the event that consumes the echo and retires the row.
        app.post_message(UserMessageStart("are you there?", 0))
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if not _notice_texts(app):
                break

        assert UNSENT_RUNTIME_NOTICE not in _notice_texts(app), _notice_texts(app)
        assert app._unsent_runtime_notice is None


@pytest.mark.asyncio
async def test_a_refused_press_leaves_the_row_standing() -> None:
    """The other half of the same rule: a refusal does NOT retire it.

    The state is unchanged by a refused press — the text is back in the composer
    and nothing can still carry it — so the row that names it must survive, or
    U6's fix is undone in the opposite direction and the user is left with a
    refusal and no explanation.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "are you there?")
        await _send(pilot, editor, "are you there?")

        assert _notice_texts(app) == [UNSENT_RUNTIME_NOTICE], _notice_texts(app)
        assert app._unsent_runtime_notice is not None


@pytest.mark.asyncio
async def test_a_hidden_conversations_stored_row_is_retired_by_text() -> None:
    """The second life of the row, and the door identity alone cannot close.

    A hand-back raised while its conversation was NOT on screen is stored in
    `source.notices` and painted on adoption, so the copy on screen is a
    different block object than the field holds. Retiring only the held block
    would leave the identical sentence standing; the stored entries go too.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        hidden = SessionInteraction(session=None)
        hidden.notices.append((UNSENT_RUNTIME_NOTICE, "warning"))
        hidden.notices.append(("an unrelated notice", "info"))

        app._retire_unsent_runtime_notice(hidden)

        assert hidden.notices == [("an unrelated notice", "info")], hidden.notices


@pytest.mark.asyncio
async def test_a_draft_with_no_caret_of_its_own_lands_at_the_end() -> None:
    """UX round 3, U2: the restored draft must not eat the user's next input.

    Serialised from the seat that measured it. Step A/B: the user's next words
    used to be glued IN FRONT of the returned message (measured as one message
    reading `second message [bash:2]are you there?`). Step C: the row's own
    `/resume` advice typed there was consumed as a command and took the message
    with it, leaving the screen with no composer text at all. With the caret at
    the end, the words append (the message is intact and first) and a `/resume`
    typed the same way does not start the line, so it cannot dispatch — which is
    the property, independent of the copy above moving off `/resume` entirely.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "are you there?")

        assert editor.text == "are you there?"
        assert editor.selection.end == (0, len("are you there?")), editor.selection

        # Step A/B — the glued-input case.
        editor.insert("second message ")
        await pilot.pause()
        assert editor.text == "are you there?second message ", editor.text

        # Step C — the advice case, on a clean hand-back so the state is exactly
        # the one the row describes: the seam the refusal uses does the restore.
        await _send(pilot, editor, "are you there?")
        app._load_editor_draft(SessionDraft(text="are you there?"))
        await pilot.pause()
        editor.insert("/resume silent-owner-drive")
        await pilot.pause()
        assert editor.text == "are you there?/resume silent-owner-drive", editor.text
