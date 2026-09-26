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
one notice; ``send again`` replays the payload under the ordinary path and
retires the failed rows for EVERY class — the unknown-delivery one included,
whose ``edit`` keeps its row because only a resend makes a second attempt;
``edit`` returns it and retires the row for provably-not-delivered classes; a
second failure restates rather than stacks; the failure record names the send
that FAILED when a second message was submitted while the first was in flight;
the ``FAILED_SEND_BOUND`` park moves the oldest payload back through the edit
funnel; the two long class sentences do not restate the control the label
wears. The CARRIAGE cells — a failure that lands while another conversation is
in front projects into its OWN view on return, never the one in front — live in
``test_resume_connect_retry.py`` (S2's cross-view arms; this file's docstring
used to list that cell among its own, which is the claim agent review round 1
the NIT flagged).
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

#: The controls, as the label spells them once NBSP-joined (D1) and with the
#: key in words (D3). Matched against a space-normalised copy of the label so
#: the assertions read as copy rather than as whitespace codepoints.
CONTROLS = "send again enter \u00b7 edit e"


def _flat(text: str) -> str:
    """``text`` with every whitespace run (NBSP included) as one space."""
    return " ".join(text.split())


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


def _raising_id_session(error_factory) -> FakeSession:
    """``_raising_session`` whose prompt also NAMES the correlation keyword.

    ``_echo_message_id`` probes the signature for a ``message_id`` parameter —
    a ``**kwargs`` catch-all does not count — so only a session that names it
    registers its echo (and its failure record) under a real id. The restate
    guard's no-rows fallback matches on that id (R-MINOR-5).
    """
    session = FakeSession()
    session.prompt_calls = []  # type: ignore[attr-defined]

    async def prompt(text: str, images: Any = None, message_id: str = "") -> None:
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
        "shorten it \u2014 your message was not sent",
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
        assert CONTROLS in _flat(label), label
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
        assert "edit e" in _flat(notice._text), notice._text
        assert "send again" not in _flat(notice._text), notice._text
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
        # U4 (UX round 1): the verb hands the keyboard back to the composer, the
        # way both sibling paths (`DraftRecoveryNotice.Requested`,
        # `_edit_failed_send`) already do — the notice was the focused row when
        # Enter fired it, and leaving focus on the transcript leaves the
        # composer dark until the next keystroke re-lights it.
        assert app.focused is editor, "focus stayed on the transcript after send again"


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
async def test_a_re_entry_without_the_rows_still_restates() -> None:
    """R-MINOR-5: the guard also holds when the re-entry carries no rows.

    A failure can arrive with ``blocks is None`` — a later submit's ``finally``
    can clear the slot before an earlier failure lands — and the row-identity
    guard alone then skipped the loop and appended a SECOND record and notice
    under one row, the stacking the docstring forbids. The message id is the
    fallback identity, which is why this cell needs a session that carries one.
    """
    app = OperatorApp(lambda: _factory(_raising_id_session(lambda: ConnectionError("gone"))))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        source = app._interaction
        record = _records(app)[0]
        assert record.message_id, "the cell needs the id identity to exercise"
        app._mark_send_failed(
            source,
            blocks=None,
            text=record.text,
            sent=record.sent,
            typed=record.typed,
            images=record.images,
            accepted=record.accepted,
            failure_class="runtime-gone",
            sentence="this session's runtime stopped again — your message was not sent",
            kind="warning",
            message_id=record.message_id,
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
        # U3 (UX round 1): the resend of this class can DUPLICATE the message,
        # and the copy says so without claiming either delivery outcome.
        flat = _flat(notice._text)
        assert "sending again may send it twice" in flat, notice._text
        assert "was not sent" not in flat.lower(), "a delivery claim on the unknown class"
        notice.focus()
        await pilot.pause()
        await pilot.press("e")
        assert await _pump(pilot, lambda: bool(editor.text)), "the payload did not load"
        assert editor.text == "are you there?" + RESTORE_SEAM
        assert _user_texts(app) == ["are you there?"], "the fate statement was erased"
        assert _notices(app) == [], "the spent offer should go"
        assert _records(app) == []


@pytest.mark.asyncio
async def test_send_again_on_an_unknown_delivery_supersedes_its_row() -> None:
    """R-MAJOR-1: a resend retires the failed rows for EVERY class, this one too.

    The unknown-delivery class keeps its row under ``edit`` — an edit makes no
    second attempt, so the row is the only statement that a copy may exist.
    A RESEND is that second attempt, and its successor row is the standing
    statement; two rows for one payload is the count J1 forbids. Measured
    before the fix (reviewer's probe, ``owns_runtime=False`` + a transport
    timeout): after Enter on the notice, user rows
    ``['are you there?', 'are you there?']`` — the first standing with no
    notice, the second carrying it.
    """
    session = _raising_session(lambda: TimeoutError("no answer to the prompt frame"))
    session.owns_runtime = False  # type: ignore[attr-defined]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_records(app)))

        (record,) = _records(app)
        assert record.unknown_delivery is True
        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        assert await _pump(
            pilot, lambda: len(session.prompt_calls) == 2  # type: ignore[attr-defined]
        ), "the resend never reached the session"
        assert await _pump(pilot, lambda: len(_records(app)) == 1 and bool(_notices(app)))

        assert _user_texts(app) == ["are you there?"], (
            "the resend left the failed row standing beside its successor",
            _user_texts(app),
        )
        (fresh,) = _records(app)
        (message,) = _notices(app)
        assert message.record is fresh, "the notice outlived the record it explains"
        assert CONTROLS in _flat(message._text), message._text


class _GatedFirstSession(FakeSession):
    """``prompt`` parks the FIRST call and fails it on release (R-MAJOR-2's interleaving).

    The second call is admitted immediately: it is what proves the first
    failure did not bind the second's rows.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prompt_calls: list[str] = []
        self.gate = asyncio.Event()

    async def prompt(self, text: str, images: Any = None, **kwargs: Any) -> None:
        self.prompt_calls.append(text)
        if text == "first":
            await self.gate.wait()
            raise ConnectionError("owner socket unreachable")
        return None


@pytest.mark.asyncio
async def test_a_second_submit_does_not_steal_the_first_sends_failure() -> None:
    """R-MAJOR-2: the record binds the send that FAILED, not the newest submit.

    Measured before the fix (reviewer's interleaving): send ``first`` (its
    prompt parks), submit ``second`` while the first is pending (its row
    paints, its prompt queues), then let the first fail — ONE record whose
    ``text`` was ``second`` while its ``sent`` was ``first``: a chimera. ``e``
    then returned ``first`` while retiring the SECOND's row — the delivered
    message's row erased, the un-delivered send's left standing.
    """
    session = _GatedFirstSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        await _submit(pilot, editor, "first")
        assert await _pump(pilot, lambda: session.prompt_calls == ["first"])

        await _submit(pilot, editor, "second")
        assert await _pump(pilot, lambda: _user_texts(app) == ["first", "second"])

        session.gate.set()
        assert await _pump(pilot, lambda: bool(_records(app)))

        (record,) = _records(app)
        assert record.text == "first", ("the failure bound the wrong send", record.text)
        assert record.sent == "first", record.sent
        assert record.unknown_delivery is False

        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("e")
        assert await _pump(pilot, lambda: bool(editor.text)), "the payload did not load"
        assert editor.text == "first" + RESTORE_SEAM
        assert _user_texts(app) == ["second"], (
            "edit erased the delivered message's row (or kept the failed one)",
            _user_texts(app),
        )


@pytest.mark.asyncio
async def test_the_bound_parks_the_oldest_payload_through_the_edit_funnel() -> None:
    """R-MINOR-3: ``FAILED_SEND_BOUND`` is load-bearing; pin the park it forces.

    Five failing sends leave FOUR records (the bound) and the oldest payload
    parked through the same ``edit`` funnel the user's ``e`` uses — its fast
    path (composer empty, no aside) loads the editor directly and resolves the
    record, so the fifth failure does not stack an unbounded list. The park
    states nothing extra on the transcript (design OQ3's copy call); what this
    cell protects is that the bound cannot be deleted or silently stop parking.
    """
    app = OperatorApp(lambda: _factory(_dead_session()))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        for index in range(1, 5):
            await _submit(pilot, editor, f"message {index}")
            assert await _pump(pilot, lambda n=index: len(_records(app)) == n), (
                f"failure {index} never landed",
                [r.text for r in _records(app)],
            )

        await _submit(pilot, editor, "message 5")
        assert await _pump(pilot, lambda: len(_records(app)) == 4 and editor.text), (
            "the bound did not park the oldest payload",
            [r.text for r in _records(app)],
            editor.text,
        )
        assert [record.text for record in _records(app)] == [
            "message 2",
            "message 3",
            "message 4",
            "message 5",
        ], [record.text for record in _records(app)]
        assert editor.text == "message 1" + RESTORE_SEAM, editor.text
        assert app._interaction.unsent == [], "the fast path must not also park an offer"
        assert not any("restore unsent prompt" in text for text in _all_notice_texts(app))
        assert _user_texts(app) == ["message 2", "message 3", "message 4", "message 5"]


@pytest.mark.asyncio
async def test_the_long_classes_do_not_restate_the_control_they_wear() -> None:
    """D2: the sentence drops the retry clause the label's controls carry.

    Rendered before the fix: oversize read ``…shorten it and send again — your
    message was not sent — send again ⏎ · edit e`` (the imperative twice, one
    clause apart) and retiring read ``… send it again once the new build is
    up. — send again ⏎ · edit e`` (a duplicate clause plus the mid-label full
    stop the reused head+tail produced). The trim is composition-only — the
    transport's numbers stay its own words — and the controls become the only
    imperative.
    """
    oversize = OperatorApp(
        lambda: _factory(
            _raising_session(
                lambda: OversizedRequest(
                    "this message is 2.0 MB and the limit is 1.0 MB; shorten it and send again"
                )
            )
        )
    )
    async with oversize.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, oversize)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_notices(oversize)))
        flat = _flat(_notices(oversize)[0]._text)
        assert "shorten it — your message was not sent" in flat, flat
        assert flat.count("send again") == 1, flat
        assert ". —" not in flat, flat

    retiring = OperatorApp(
        lambda: _factory(_raising_session(lambda: RuntimeRetiring(RuntimeRetiring.BUILD)))
    )
    async with retiring.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, retiring)
        await _submit(pilot, editor, "are you there?")
        assert await _pump(pilot, lambda: bool(_notices(retiring)))
        flat = _flat(_notices(retiring)[0]._text)
        assert "sending can resume once the new build is up" in flat, flat
        assert "send it again" not in flat, flat
        assert flat.count("send again") == 1, flat
        assert ". —" not in flat, flat
