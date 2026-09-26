"""A send that could not reach a runtime KEEPS its row, and says why ONCE.

Two failures share this file because they are one fact to the user: a prompt
whose ``prompt()`` raised on a dead socket, and — since QA round 2 (Q-1) — a
queued STEER whose bind was refused after the recovery give-up released it (that
half is pinned in ``test_queued_steer_receipt.py``, which still drives
``UNSENT_RUNTIME_NOTICE``, the steer path's own row). The dead-socket half was
REWRITTEN by the boundary rule (see the S1 review round): a post-paint failure
keeps its row and is resolved ON the failure record — ``send again`` / ``edit``
— instead of withdrawing the row and returning the payload to the composer, so
the cells below assert the record's row, its one notice, and its verbs.

What this file still adds is the RETRY shape U6 measured: while a stale record
still claims a live owner, a resend is refused again in ~0.4 s, so the screen
must hold ONE row and ONE notice however many attempts are made — never a tally.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.tui.app import RESTORE_SEAM, UNSENT_RUNTIME_NOTICE, OperatorApp
from local_operator.tui.session_interaction import SessionDraft, SessionInteraction
from local_operator.tui.session_presentation import SendFailureNotice
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


async def _send(pilot: Any, app: OperatorApp, editor: Editor, text: str) -> None:
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    # The refusal is painted by the prompt worker; under the boundary rule the
    # landing is its failure record, and the composer stays empty.
    for _ in range(100):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if app._interaction.turn.failed_sends:
            return


async def _pump(pilot: Any, predicate, *, turns: int = 200) -> bool:
    for _ in range(turns):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if predicate():
            return True
    return bool(predicate())


def _failure_notice(app: OperatorApp) -> SendFailureNotice:
    (notice,) = [block for block in _blocks(app) if isinstance(block, SendFailureNotice)]
    return notice


@pytest.mark.asyncio
async def test_a_prompt_on_a_dead_runtime_keeps_its_row_and_names_the_reason() -> None:
    """The first refusal: the row stays, the composer stays empty, one reason."""
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "are you there?")

        assert editor.text == "", "the payload returned to the composer by itself"
        assert _user_texts(app) == ["are you there?"], "the row was withdrawn"
        (notice,) = _notice_texts(app)
        assert "this session's runtime stopped" in notice, notice
        assert "your message was not sent" in notice, notice
        assert "send again enter · edit e" in " ".join(notice.split()), notice
        assert UNSENT_RUNTIME_NOTICE not in notice, notice


@pytest.mark.asyncio
async def test_pressing_again_does_not_pile_up_rows_or_warnings() -> None:
    """U6's measurement restated for the boundary rule: attempts do not tally.

    Before this, press two added a second message row AND a second copy of the
    warning. Under the boundary rule the composer stays EMPTY after the refusal
    (the payload lives on the failure record), so a stray press sends nothing;
    the retry is the notice's own `send again`, which retires its predecessor in
    the same handler that resubmits (J1) — the screen holds one row and one
    notice, never a tally of attempts.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "are you there?")

        # A bare press on the empty composer: nothing to send, nothing may grow.
        await pilot.press("enter")
        assert await _pump(pilot, lambda: False, turns=20) is False  # settle a beat
        assert _user_texts(app) == ["are you there?"], _user_texts(app)
        assert len(_notice_texts(app)) == 1, _notice_texts(app)

        # The notice's own retry: retire-then-resubmit, one row again.
        notice = _failure_notice(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(
            pilot, lambda: len(app._interaction.turn.failed_sends) == 1 and bool(_notice_texts(app))
        ), (_notice_texts(app), app._interaction.turn.failed_sends)
        assert _user_texts(app) == ["are you there?"], _user_texts(app)
        assert len(_notice_texts(app)) == 1, _notice_texts(app)
        assert editor.text == "", editor.text


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
async def test_the_notice_is_retired_when_the_payload_is_resent() -> None:
    """U1 restated: the notice is a STATE, and its resolution retires it.

    SUPERSEDES the retire-on-served rule, which belonged to the old
    "back in the composer" row: this notice is the failure RECORD's, and its
    resolution points are its own verbs — `send again` retires it at the press
    (retire-then-resubmit, one handler), so a delivered resend leaves no notice
    standing over the message it described.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "are you there?")
        assert len(_notice_texts(app)) == 1, _notice_texts(app)

        # The runtime comes back and the notice's own retry goes out: the row
        # it describes is resolved with the resubmit (one row, no notice).
        session.prompt = _serving_prompt  # type: ignore[assignment]
        notice = _failure_notice(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(pilot, lambda: not _notice_texts(app)), _notice_texts(app)

        assert _notice_texts(app) == [], _notice_texts(app)
        assert _user_texts(app) == ["are you there?"], "one row for the delivered resend"


@pytest.mark.asyncio
async def test_a_refused_resend_leaves_a_notice_standing() -> None:
    """The other half of the same rule: a refusal keeps a notice up.

    The state is unchanged by a refused resend — the message is still unsent —
    so a notice must still name it. Under the boundary rule it is the RESEND's
    own record (one record per failed row): the first row is retired with the
    resubmit, the resend's refusal draws its own notice, and the screen still
    holds exactly one row and one notice.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "are you there?")

        notice = _failure_notice(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(
            pilot, lambda: len(app._interaction.turn.failed_sends) == 1 and bool(_notice_texts(app))
        ), (_notice_texts(app), app._interaction.turn.failed_sends)

        (standing,) = _notice_texts(app)
        assert "your message was not sent" in standing, standing
        assert _user_texts(app) == ["are you there?"], _user_texts(app)
        assert app._interaction.turn.failed_sends[0].failure_class == "runtime-gone"


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
    """UX round 3, U2 restated for `edit`: the restored draft must not eat the next input.

    Serialised from the seat that measured it. Step A/B: the user's next words
    used to be glued IN FRONT of the returned message (measured as one message
    reading `second message [bash:2]are you there?`); the restore (now the
    user's own `edit`) lands behind a SEAM (a blank line), so the next words are
    their own paragraph. Step C: a `/resume` typed below the draft does not
    start the line, so it cannot dispatch — the property, independent of the
    copy having moved off `/resume` entirely.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "are you there?")

        notice = _failure_notice(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("e")
        assert await _pump(pilot, lambda: bool(editor.text)), "the payload never loaded"
        assert editor.text == "are you there?" + RESTORE_SEAM
        # The caret is on the line BELOW the seam, not at the end of the drafted
        # sentence, which is what makes the two separable.
        assert editor.selection.end == (2, 0), editor.selection

        # Step A/B — the next-thought case: it lands as its own paragraph.
        editor.insert("second message ")
        await pilot.pause()
        assert editor.text == "are you there?" + RESTORE_SEAM + "second message ", editor.text

        # Step C — the advice case, on a clean hand-back so the state is exactly
        # the one the row describes.
        app._load_editor_draft(SessionDraft(text="are you there?"))
        await pilot.pause()
        editor.insert("/resume silent-owner-drive")
        await pilot.pause()
        assert editor.text == "are you there?/resume silent-owner-drive", editor.text
