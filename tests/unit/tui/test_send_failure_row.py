"""What a send that failed AFTER its row was painted leaves on the transcript.

THE BOUNDARY RULE this suite pins: a failure raised before the row was painted
keeps the text in the composer and paints nothing (the editor gate, pinned in
the composer suites); a failure raised after keeps its row and gets
``send again`` / ``edit`` on a notice beneath it. This SUPERSEDES the reviewed
withdraw-the-row-and-return-the-payload behaviour the removed cells in
``test_unsent_runtime_notice.py`` / ``test_retiring_refusal.py`` /
``test_stop_command.py`` asserted — those were updated knowingly, not deleted —
because the operator's ask is explicit: the retry is one keystroke on the row,
the payload has one home, and a message that never left the machine is not
erased from the conversation it was typed into.

Cells, per the design's S1 test list: every post-paint class keeps its row and
one notice; ``send again`` replays the payload under the ordinary path; ``edit``
returns it and retires the row for provably-not-delivered classes; a second
failure restates rather than stacks; an unknown-delivery row survives ``edit``;
a failure that lands while another conversation is in front projects into its
OWN view on return, never the one in front.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from local_operator.mobile.attach_client import OversizedRequest
from local_operator.session.errors import RuntimeRetiring
from local_operator.tui.app import RESTORE_SEAM, OperatorApp
from local_operator.tui.session_presentation import SendFailureNotice
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: The one sentence every post-paint class must carry now that the composer no
#: longer receives the payload. Asserted as a substring because the classes'
#: own sentences wrap it differently (the drain's halves, the transport's line).
NOT_SENT = "your message was not sent"


@pytest.fixture(autouse=True)
def isolate_failure_row(tmp_path, monkeypatch):
    # Headless apps must never touch the caller's real config or rename a real
    # multiplexer workspace (the same pins the sibling TUI suites carry).
    for key in tuple(os.environ):
        if key.startswith(("CMUX_", "LOP_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(OperatorApp, "_check_for_update", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_terminal_title", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_multiplexer_broadcast", lambda _self: None)
    monkeypatch.setattr(OperatorApp, "_start_herdr_reporter", lambda _self: None)


def _raising_session(error_factory) -> FakeSession:
    """A session whose every send raises ``error_factory()``'s exception.

    Every CALL is recorded on ``session.prompt_calls`` — a failed send never
    reaches the fake's ``prompts`` list, which is for ADMITTED sends, and a
    replay test needs the attempt itself as its proof.
    """
    session = FakeSession()
    session.prompt_calls = []  # type: ignore[attr-defined]

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> None:
        session.prompt_calls.append(text)  # type: ignore[attr-defined]
        raise error_factory()

    session.prompt = prompt  # type: ignore[assignment]
    return session


def _dead_session() -> FakeSession:
    """The U6 measurement's shape: a record claims a live owner and answers nothing."""
    return _raising_session(
        lambda: ConnectionError("owner socket unreachable: [Errno 61] Connect call failed")
    )


async def _boot(pilot: Any, app: OperatorApp) -> Editor:
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


async def _pump(pilot: Any, predicate, *, turns: int = 300) -> bool:
    for _ in range(turns):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if predicate():
            return True
    return bool(predicate())


async def _submit(pilot: Any, editor: Editor, text: str) -> None:
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")


def _view(app: OperatorApp) -> TranscriptView:
    return app._transcript_view()


def _notices(app: OperatorApp) -> list[Any]:
    return [block for block in _view(app).blocks() if isinstance(block, SendFailureNotice)]


def _all_notice_texts(app: OperatorApp) -> list[str]:
    return [block._text for block in _view(app).blocks() if isinstance(block, NoticeBlock)]


def _user_texts(app: OperatorApp) -> list[str]:
    return [block.text() for block in _view(app).blocks() if isinstance(block, UserBlock)]


def _records(app: OperatorApp) -> list[Any]:
    return list(app._interaction.turn.failed_sends)


_POST_PAINT_CLASSES = [
    (
        "oversize",
        lambda: _raising_session(
            lambda: OversizedRequest(
                "this message is 2.0 MB and the limit is 1.0 MB; shorten it and send again"
            )
        ),
        "shorten it and send again",
        NOT_SENT,
    ),
    (
        "retiring",
        lambda: _raising_session(lambda: RuntimeRetiring(RuntimeRetiring.BUILD)),
        "switching to a newer build",
        "Your message was not sent",
    ),
    ("runtime-gone", _dead_session, "this session's runtime stopped", NOT_SENT),
    (
        "generic",
        lambda: _raising_session(lambda: RuntimeError("mcp auth failed")),
        "mcp auth failed",
        # The ONE class that must make NO delivery claim: an attached session's
        # transport error may have reached the owner (design OQ2), so neither
        # "not sent" nor "was sent" is something this notice may assert.
        None,
    ),
]


@pytest.mark.parametrize(
    ("failure_class", "make_session", "sentence_fragment", "state_fragment"),
    _POST_PAINT_CLASSES,
    ids=[entry[0] for entry in _POST_PAINT_CLASSES],
)
@pytest.mark.asyncio
async def test_a_post_paint_failure_keeps_its_row_and_offers_the_verbs(
    failure_class: str, make_session: Any, sentence_fragment: str, state_fragment: str | None
) -> None:
    """The boundary rule's row side, once per class.

    The row stands; exactly one notice states the fate; the composer shows none
    of it; and the notice names both verbs. The old behaviour — the row gone and
    the text back in the composer — is what this cell reddens against.
    """
    app = OperatorApp(lambda: _factory(make_session()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app))), "no failure record landed"

        assert _user_texts(app) == ["are you there?"], "the row was withdrawn"
        notices = _notices(app)
        assert len(notices) == 1, _all_notice_texts(app)
        label = notices[0]._text
        assert sentence_fragment in label, label
        if state_fragment is None:
            assert "not sent" not in label.lower(), label
        else:
            assert state_fragment in label, label
        assert "send again ⏎ · edit e" in label, label
        assert "back in the composer" not in label, label
        assert editor.text == "", "the payload leaked back into the composer"
        records = _records(app)
        assert len(records) == 1
        assert records[0].failure_class == failure_class
        assert records[0].can_send_again is True
        assert records[0].unknown_delivery is False


@pytest.mark.asyncio
async def test_the_stopped_class_offers_edit_alone_and_enter_edits() -> None:
    """No session to send into, so the notice must not offer a dead `send again`.

    Enter still has to do the only thing that can work: load the payload into the
    composer. An Enter that answered with nothing would be a focus stop that
    lies (the notice's own docstring).
    """
    app = OperatorApp(lambda: _factory(_raising_session(lambda: ConnectionError("stopped"))))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        app._stopped_session_id = "sess"
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app))), "no failure record landed"

        (notice,) = _notices(app)
        assert "edit e" in notice._text
        assert "send again" not in notice._text, notice._text
        assert _records(app)[0].can_send_again is False

        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(pilot, lambda: bool(editor.text)), "Enter did not edit"
        assert editor.text == "are you there?" + RESTORE_SEAM
        assert _user_texts(app) == [], "a provably-undelivered row outlived the edit"
        assert _notices(app) == []
        assert _records(app) == []


@pytest.mark.asyncio
async def test_the_attach_behind_class_keeps_its_row_too() -> None:
    """A message whose own bind failed during a paint-first attach.

    The row keeps the message — the old shape withdrew it ('echo down, draft
    back') — and the class's own sentence says what happened, in the product's
    words rather than the transport's.
    """
    session = _raising_session(lambda: ConnectionError("owner did not send its state"))
    session.attach_behind = True  # type: ignore[attr-defined]
    session.is_cold = True  # type: ignore[attr-defined]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app))), "no failure record landed"

        assert _user_texts(app) == ["are you there?"]
        (notice,) = _notices(app)
        assert "did not answer" in notice._text, notice._text
        assert NOT_SENT in notice._text, notice._text
        assert editor.text == ""
        assert _records(app)[0].failure_class == "attach-behind"


@pytest.mark.asyncio
async def test_send_again_replays_the_payload_and_leaves_one_row() -> None:
    """The notice's primary verb re-enters the ordinary submit path.

    The replay must reach the session's ``prompt`` with the same payload, and
    the retirement must happen in the same handler as the resubmit so the frame
    never shows two rows for one message (J1) — the final state is one row, one
    notice, one record, because the replay failed again the same way.
    """
    session = _dead_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        # The replay reached the session (the dead fake records the CALL before
        # raising, so two recorded calls ARE the proof the message went out
        # again — the first attempt included).
        assert await _pump(
            pilot,
            lambda: session.prompt_calls == ["are you there?"] * 2,  # type: ignore[attr-defined]
        ), session.prompt_calls  # type: ignore[attr-defined]
        # ...and its own failure lands on a fresh record.
        assert await _pump(pilot, lambda: len(_records(app)) == 1 and bool(_notices(app))), (
            _records(app),
            _all_notice_texts(app),
        )

        assert _user_texts(app) == ["are you there?"], "one row per payload, never two"
        assert len(_notices(app)) == 1
        assert editor.text == ""


@pytest.mark.asyncio
async def test_a_re_entry_for_the_same_rows_restates_the_notice_in_place() -> None:
    """The restate guard: one record per failed row, never a second notice.

    A second `_mark_send_failed` for the SAME rows (the state reached twice —
    the `_notice_unsent_runtime` discipline) must re-word the standing notice
    rather than append another: a sentence under the same row is the stacking
    defect the U6 measurement recorded.
    """
    app = OperatorApp(lambda: _factory(_dead_session()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        source = app._interaction
        record = _records(app)[0]
        app._mark_send_failed(
            source,
            blocks=record.blocks,
            text=record.text,
            sent=record.sent,
            typed=record.typed,
            images=record.images,
            accepted=record.accepted,
            failure_class="runtime-gone",
            sentence="this session's runtime stopped again — your message was not sent",
            kind="warning",
        )
        await pilot.pause()
        assert len(_records(app)) == 1, "a second record was stacked"
        (notice,) = _notices(app)
        assert notice._text.startswith("this session's runtime stopped again"), notice._text
        assert _user_texts(app) == ["are you there?"]


@pytest.mark.asyncio
async def test_edit_returns_the_payload_and_retires_a_provably_undelivered_row() -> None:
    """``edit`` is the old restore funnel, now triggered by the user, not automatic."""
    app = OperatorApp(lambda: _factory(_dead_session()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("e")
        assert await _pump(pilot, lambda: bool(editor.text)), "the payload did not load"
        assert editor.text == "are you there?" + RESTORE_SEAM
        assert _user_texts(app) == [], "the row should go with the payload move"
        assert _notices(app) == []
        assert _records(app) == []


@pytest.mark.asyncio
async def test_a_second_failure_restates_rather_than_stacks() -> None:
    """Two failures, two attempts, ONE standing row and ONE standing notice.

    The U6 defect this pins (rows and warnings piling up per attempt): with the
    record resolving on each attempt's own resolution and the replay retiring
    before it resubmits, the visible count cannot grow.
    """
    app = OperatorApp(lambda: _factory(_dead_session()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        for _ in range(2):
            (notice,) = _notices(app)
            notice.focus()
            await pilot.pause()
            await pilot.press("enter")
            assert await _pump(pilot, lambda: len(_records(app)) == 1 and bool(_notices(app)))

        assert _user_texts(app) == ["are you there?"], _user_texts(app)
        assert len(_notices(app)) == 1, _all_notice_texts(app)
        assert len(_records(app)) == 1


@pytest.mark.asyncio
async def test_an_unknown_delivery_keeps_its_row_under_edit() -> None:
    """The one class that may have reached the owner: its row is the fate statement.

    A generic transport error on an ATTACHED session (design OQ2) — the write
    crosses a socket and the outcome was never observed — so `edit` moves the
    payload but the row stays, because retiring it would erase the only record
    that the message may exist.
    """
    session = _raising_session(lambda: TimeoutError("no answer to the prompt frame"))
    session.owns_runtime = False  # type: ignore[attr-defined]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        (record,) = _records(app)
        assert record.failure_class == "generic"
        assert record.unknown_delivery is True

        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("e")
        assert await _pump(pilot, lambda: bool(editor.text)), "the payload did not load"
        assert editor.text == "are you there?" + RESTORE_SEAM
        assert _user_texts(app) == ["are you there?"], "the fate statement was erased"
        assert _notices(app) == [], "the spent offer should go"
        assert _records(app) == []
