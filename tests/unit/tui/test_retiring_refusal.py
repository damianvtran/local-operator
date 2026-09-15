"""A message refused by a DRAINING runtime is handed back, and the drain is announced.

Two findings, one window. Design round 1 (D1) and UX round 1 (U1) independently
measured the same harm: once a busy runtime latches its drain, a prompt is
refused, the composer's text is dropped, and the row painted for it stands in
the transcript looking delivered — while the refusal tells the user to "send it
again". Following that advice inside the window produced a second standing row
and a second refusal for one message nobody ever received. The oversize and
runtime-gone branches two `elif`s above already call
``_withdraw_user_echo_for`` + ``_restore_unsent_for`` for exactly this reason;
the retiring refusal is the case that fell past them, so these cells pin the
same contract on the drain's own path.

UX round 1 (U2) measured the other half: nothing announced the handover. The
drain is announced with one ``note``, and rounds 3 measured two ways the first
attempt at that failed: it was painted when the socket CLOSED (26 s into a 26 s
drain, i.e. after the last refusal it existed to warn about) and its gate asked
the viewer's own ``runtime_idle``, which ``_go_cold`` has already made cold in
both hands — so every ordinary idle refresh got a row about refusals that never
came (UX round 3, U1; QA round 3, Q-1). The row is now painted from the
runtime's ``retiring`` frame, at the moment it is sent and only when the frame
says ``draining``; the cells below drive that seam rather than stubbing its
input, and the idle handover's silence is asserted on the callback the real
transport actually reaches.

The second round-3 finding is the handback's own seam (UX round 3, U2): the
restore lands inside the submit, so the operator's NEXT thought used to weld
onto their returned draft — ``summarise the build staleness fixand the deploy
notes``, sent as one message. A blank line is the boundary the composer shows.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.session.errors import RuntimeRetiring, admission_error
from local_operator.tui.app import (
    DRAIN_NOTICE,
    RESTORE_SEAM,
    OperatorApp,
    _is_retiring_refusal,
    _retiring_notice_text,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock

from .test_app_pilot import FakeSession, _factory


def _retiring_session() -> FakeSession:
    """A session whose runtime has committed to leaving: every prompt refused."""
    session = FakeSession()

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> None:
        raise RuntimeRetiring()

    session.prompt = prompt  # type: ignore[assignment]
    return session


def _blocks(app: OperatorApp) -> list[Any]:
    return list(app.query_one(TranscriptView).blocks())


def _notices(app: OperatorApp) -> list[NoticeBlock]:
    return [b for b in _blocks(app) if isinstance(b, NoticeBlock)]


def _user_texts(app: OperatorApp) -> list[str]:
    return [b.text() for b in _blocks(app) if isinstance(b, UserBlock)]


#: The refusal sentence raised by the build that was RESIDENT BEFORE this
#: PR's category exists — verbatim from `serving.py` at `4802dc45a`. It is the
#: mixed-build case this whole change is about: a viewer that has just been
#: updated still binds the runtime it started with, and no `error_code` can
#: cross from a build that has never heard of one.
_LEGACY_RETIRING_REFUSAL = (
    "the session runtime is retiring (runtime-retired); the message was not "
    "admitted — send it again and the next engage runs the new build"
)


def _legacy_retiring_session() -> FakeSession:
    """A session whose runtime refuses with the pre-category wording."""
    session = FakeSession()

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> None:
        raise RuntimeError(_LEGACY_RETIRING_REFUSAL)

    session.prompt = prompt  # type: ignore[assignment]
    return session


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


async def _send(pilot: Any, editor: Editor, text: str) -> None:
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    # The refusal is painted by the prompt worker, after it settles.
    for _ in range(100):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if editor.text:
            return


@pytest.mark.asyncio
async def test_a_drain_refusal_hands_the_message_back_and_retracts_its_row() -> None:
    """D1/U1: the text survives, and nothing claims it was delivered."""
    session = _retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "now summarise the build staleness fix")

        assert editor.text == "now summarise the build staleness fix" + RESTORE_SEAM, (
            "text was dropped, or came back without the seam that keeps the next "
            "thought separable"
        )
        assert _user_texts(app) == [], "a row stood for a message nobody received"

        notices = _notices(app)
        assert len(notices) == 1, [n._text for n in notices]
        # THE PRODUCT COMPOSES THIS: identical to the writer the viewer calls,
        # and the properties the round-3 copy finding turned on are pinned
        # separately so a future edit has to argue with them (design round 3,
        # D1): one dash, and a terminal clause — never the word `composer`
        # alone on the last row at 60 columns.
        assert notices[0]._text == _retiring_notice_text(RuntimeRetiring())
        assert notices[0]._text.count("\u2014") == 1, notices[0]._text
        assert notices[0]._text.endswith(RuntimeRetiring.TAIL), notices[0]._text
        assert "Your message is back in the composer" in notices[0]._text
        # Amber `!`, not the red ✗ of a terminal failure: this state resolves
        # itself and the message cost the user nothing (design round 1, D3).
        # `_token`/`_glyph` are where `NoticeBlock` keeps the resolved kind and
        # the mark it paints, which is what a reader actually sees.
        assert (notices[0]._token, notices[0]._glyph) == ("warning", "!"), (
            notices[0]._token,
            notices[0]._glyph,
        )


@pytest.mark.asyncio
async def test_an_uncategorised_refusal_from_an_older_runtime_is_recovered_too() -> None:
    """D1/U1 across a version skew: the wording has to be recognised as well.

    The viewer and the runtime are separate builds for the whole window this PR
    is about, so the recovery cannot be conditional on BOTH ends sending the
    typed category. Without this path the operator's draft is dropped and their
    row stands as if delivered, in exactly the handover the change exists for.
    """
    session = _legacy_retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "is the build still moving?")

        assert editor.text == "is the build still moving?" + RESTORE_SEAM, "text was dropped"
        assert _user_texts(app) == [], "a row stood for a message nobody received"
        notices = _notices(app)
        assert [n._text for n in notices] == [
            f"{_LEGACY_RETIRING_REFUSAL}. Your message is back in the composer."
        ], [n._text for n in notices]


def test_the_retiring_predicate_answers_the_type_and_the_old_wording() -> None:
    """The category when it survives the transport, the sentence when it cannot.

    A false positive hands the draft back for SOME other failure, which is why
    the marker is the whole legacy sentence's stable prefix rather than any
    word in it, and why a transport loss is asserted NOT to match.
    """
    assert _is_retiring_refusal(RuntimeRetiring()) is True
    assert _is_retiring_refusal(RuntimeError(_LEGACY_RETIRING_REFUSAL)) is True
    assert _is_retiring_refusal(RuntimeError("owner socket unreachable")) is False


@pytest.mark.asyncio
async def test_following_the_refusal_inside_the_window_does_not_stack_rows() -> None:
    """U1's measurement: two presses used to leave TWO rows for one message.

    The refusals themselves repeat, because the user was told to send it again
    and the window is the runtime's to close — but each press now withdraws its
    own echo before returning the draft, so the transcript never accumulates
    rows for messages that were not admitted.
    """
    session = _retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "the build is moving")
        await _send(pilot, editor, "the build is moving")

        # ONE restore, seam and all: the second press finds the composer
        # non-empty, so it parks behind the recovery row instead of refilling —
        # which is exactly the branch that was unreachable while the refill beat
        # the keystroke (UX round 3, U2).
        assert editor.text == "the build is moving" + RESTORE_SEAM, editor.text
        assert _user_texts(app) == [], _user_texts(app)
        assert session.prompts == [], "a refused message reached the runtime"


def test_the_refusal_copy_speaks_to_the_operator_and_decodes_over_the_wire() -> None:
    """D2/D3 + U3: no log token, no machinery, and the category round-trips.

    The sentence is the ONLY thing a user reads about this whole mechanism, and
    the owner's previous wording named an internal token (`runtime-retired`) and
    described the next engage rather than the session. It crosses the transport
    as a category, so the client rebuilds it locally — that is what lets the
    viewer branch on it at all.
    """
    refusal = RuntimeRetiring()
    text = str(refusal)
    assert "runtime-retired" not in text, text
    assert "the next engage" not in text, text
    assert "send it again" in text, "the one act the user can take has to be named"

    decoded = admission_error(RuntimeRetiring.code)
    assert isinstance(decoded, RuntimeRetiring)
    # THE CATCH SHAPE IS PART OF THE CONTRACT: the same refusal has been raised
    # as a bare RuntimeError by these gates since they existed, so both bases
    # are needed — a ValueError-only class would change the handler that sees it.
    assert isinstance(refusal, RuntimeError) and isinstance(refusal, ValueError)


@pytest.mark.asyncio
async def test_the_next_thought_cannot_weld_onto_the_returned_draft() -> None:
    """U2: the restore beats the keystroke, so the composer must show a seam.

    The refill lands INSIDE the submit (one frame after the press, measured),
    so an operator who waited out the refusal and then typed a new thought had
    it appended to their returned draft with nothing between them —
    ``summarise the build staleness fixand the deploy notes``, sent as one
    message that reads like a typo the user did not make (UX round 3, U2). The
    park branch that exists for that case never ran, because the composer was
    refilled before any human could type. The seam is what makes the two
    separable: the next thought lands in its own paragraph.
    """
    session = _retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "summarise the build staleness fix")
        assert editor.text.endswith(RESTORE_SEAM), editor.text

        await pilot.press("a", "n", "d")
        assert (
            editor.text == "summarise the build staleness fix" + RESTORE_SEAM + "and"
        ), "the next thought welded onto the returned draft: " + repr(editor.text)


@pytest.mark.asyncio
async def test_a_draining_announcement_paints_the_notice(monkeypatch: Any, tmp_path: Any) -> None:
    """U1/U2: the row that says the handover is happening, and no other.

    Driven through the method the FACADE calls on the runtime's ``retiring``
    frame (:meth:`OperatorApp._on_runtime_draining`). The pin this replaces
    stubbed ``session.runtime_idle`` and called the disconnect callback — a gate
    whose input the real transport could never answer that way, because
    ``_go_cold`` clears the client before the callback runs and ``is_cold`` is
    the first term of ``runtime_idle`` (QA round 3, Q-1: ``runtime_idle False,
    is_cold True, client NoneType``). The fact now comes from the frame, so the
    pin drives the frame's side of the seam.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)

        app._on_runtime_draining()
        await pilot.pause()
        notices = _notices(app)
        assert [n._text for n in notices] == [DRAIN_NOTICE], [n._text for n in notices]
        assert (notices[0]._token, notices[0]._glyph) == ("muted", "\u00b7"), (
            "the drain is a quiet note, not a warning",
            notices[0]._token,
            notices[0]._glyph,
        )

        # THE CLOSE IS NOT THE ANNOUNCEMENT: the same handover's disconnect
        # adds nothing, because the row was painted at the frame (26 s earlier
        # in a real drain) and would otherwise duplicate on every close.
        app._on_runtime_refreshed()
        await pilot.pause()
        assert [n._text for n in _notices(app)] == [DRAIN_NOTICE], [n._text for n in _notices(app)]


@pytest.mark.asyncio
async def test_an_idle_handover_paints_nothing(monkeypatch: Any, tmp_path: Any) -> None:
    """The deliberate silence stays for the refresh nobody can lose anything to.

    An idle runtime leaves in about a second and refuses nothing, so its frame
    carries ``draining`` false and never reaches
    :meth:`OperatorApp._on_runtime_draining`; the disconnect callback it does
    reach paints nothing. Announcing it would turn every ordinary refresh into
    a notice, which is the behaviour this PR must not change
    (design-runtime-autorefresh \u00a73.3) — and the version this replaced *did*
    do that, because it asked the viewer's own state, which is cold in both
    cases (QA round 3, Q-1).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._on_runtime_refreshed()
        await pilot.pause()

        assert _notices(app) == []


def test_the_facade_only_acts_on_a_draining_frame() -> None:
    """The frame's own field decides, and an older runtime's frame is silent.

    ``AttachedSession._on_retiring_frame`` is the seam between the wire and the
    notice: the idle rung sends the same op with ``draining`` false, and a
    runtime from before the field sends no ``draining`` at all. Both must stay
    silent — a viewer that guessed from its own state called them all draining
    (QA round 3, Q-1).
    """
    from local_operator.session.attached import AttachedSession

    facade = AttachedSession.__new__(AttachedSession)
    fired: list[str] = []
    facade._drain_callback = lambda: fired.append("drain")  # type: ignore[method-assign]

    facade._on_retiring_frame({"op": "retiring", "draining": True})
    assert fired == ["drain"]

    facade._on_retiring_frame({"op": "retiring", "draining": False})
    facade._on_retiring_frame({"op": "retiring"})  # older runtime: no field
    assert fired == ["drain"], fired
