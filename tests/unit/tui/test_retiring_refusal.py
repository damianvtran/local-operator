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
from local_operator.session.runtime.inbox import SPOOL_RECEIPT_PROMPT
from local_operator.session.runtime.types import LEAVING_FOR_BUILD, LEAVING_ON_SIGNAL
from local_operator.tui.app import (
    DRAIN_NOTICE,
    DRAIN_NOTICE_OTHER,
    QUEUED_PROMPT_MISSED_NOTICE,
    QUEUED_PROMPT_TAKEN_BACK_NOTICE,
    RESTORE_SEAM,
    SIGNAL_DRAIN_NOTICE,
    OperatorApp,
    _is_retiring_refusal,
    _retiring_notice_text,
)
from local_operator.tui.events import UserMessageStart
from local_operator.tui.session_presentation import DraftRecoveryNotice
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import (
    QUEUED_ROW_TEXT,
    NoticeBlock,
    TranscriptView,
    UserBlock,
)

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
    and the window is the runtime's to close — but each press withdraws its own
    echo before returning the draft, so the transcript never accumulates rows
    for messages nobody received.

    AND WHAT THE COMPOSER DOES, in two corrections. Round 4 (UX U1): this path
    does NOT park on the second press — it cannot, because the submit clears the
    editor and the restore always finds it empty and refills. The cell that
    claimed otherwise ASSIGNED ``editor.text`` before each press, so it never
    reached the state its comment named; the presses below are real. Round 5
    (design D1): refilling must not append a seam that is already there, or
    following the notice's own "send it again" grows the draft by a blank line
    per press — the composer went from 1 row to 3 and the transcript paid a row
    for each attempt. So the assertion below is that the composer is EXACTLY the
    same after every press, which is the cell that fails if the guard is
    removed.
    """
    session = _retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, editor, "the build is moving")
        assert editor.text == "the build is moving" + RESTORE_SEAM, editor.text

        # Pressed for real: the composer already holds the returned draft, so
        # this IS the operator following "send it again" — and the seam must not
        # accumulate (design round 5, D1). The submit's own strip takes the
        # trailing seam out of the MESSAGE; the accepted snapshot keeps it, and
        # the restore is what must not add a second.
        for _ in range(2):
            await pilot.press("enter")
            for _ in range(100):
                await pilot.pause()
                await asyncio.sleep(0.01)
                if editor.text == "the build is moving" + RESTORE_SEAM:
                    break
            assert (
                editor.text == "the build is moving" + RESTORE_SEAM
            ), "the seam accumulated: " + repr(editor.text)
            assert not [
                b for b in _blocks(app) if isinstance(b, DraftRecoveryNotice)
            ], "this path cannot park: the refill always wins"

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

        app._on_runtime_draining(LEAVING_FOR_BUILD)
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
async def test_the_notice_states_the_trigger_the_frame_named(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """D6: a signalled runtime is not told about a build that does not exist.

    Both triggers commit through one seam and both announce ``draining`` true,
    so the flag alone cannot pick the sentence — and the one it used to pick
    promised a newer build to a runtime that had been terminated mid-turn, where
    there is no newer build and no successor coming. The frame now carries the
    trigger's own words, and the three cells below are the whole rule: the
    signal gets its own sentence, the build keeps the one written for it, and a
    phrase this build cannot place (a newer runtime's) gets neither.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)

        app._on_runtime_draining(LEAVING_ON_SIGNAL)
        await pilot.pause()
        assert [n._text for n in _notices(app)] == [SIGNAL_DRAIN_NOTICE], _notices(app)
        assert "newer build" not in SIGNAL_DRAIN_NOTICE, SIGNAL_DRAIN_NOTICE


@pytest.mark.asyncio
async def test_an_unplaceable_phrase_gets_no_other_triggers_words(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A phrase this build cannot place gets the neutral sentence.

    Two ways to arrive here, one answer. A phrase this build cannot place is a
    NEWER runtime's trigger; an EMPTY one is a frame that named no trigger at
    all, which — because ``AttachedSession._on_retiring_frame`` reads the frame's
    own ``reason``/``to`` first (design round 4, D9) — means the frame itself
    said nothing about why. Painting either the build notice is the falsehood D6
    filed one version over, and the build sentence was in fact the old answer for
    the empty case: it was true of a released runtime, whose only drain is the
    build handover, and FALSE of this branch's own intermediate builds, which
    signal-drained into it (agent review round 4, MAJOR-1; UX round 4, U13). The
    neutral sentence is the only one that is never false, because it says only
    what ``draining`` establishes.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)

        app._on_runtime_draining("")
        await pilot.pause()
        assert [n._text for n in _notices(app)] == [DRAIN_NOTICE_OTHER], _notices(app)

        app._on_runtime_draining("leaving because a trigger this build has never heard of")
        await pilot.pause()
        texts = [n._text for n in _notices(app)]
        assert texts[-1] == DRAIN_NOTICE_OTHER, texts
        assert DRAIN_NOTICE_OTHER != DRAIN_NOTICE


@pytest.mark.asyncio
async def test_each_trigger_keeps_its_own_sentence_at_the_notice_seam(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """D6's pair, plus the derivation that stands in for a missing phrase.

    The app is handed ONE phrase and chooses from it, so the interesting cells
    are the ones where the phrase is absent on the wire: every runtime from
    ``main`` announces the build handover that way, and every build of this
    branch before the key was added announces BOTH triggers that way. The frame's
    own ``reason``/``to`` decide (design round 4, D9), and the frames below are
    the shapes those runtimes actually send — including the released one, whose
    only drain is the stale-build handover.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)

        app._on_runtime_draining(LEAVING_ON_SIGNAL)
        app._on_runtime_draining(LEAVING_FOR_BUILD)
        await pilot.pause()
        assert [n._text for n in _notices(app)] == [SIGNAL_DRAIN_NOTICE, DRAIN_NOTICE], _notices(
            app
        )

        # The derivation itself, at the seam that owns it.
        from local_operator.session.attached import AttachedSession

        facade = AttachedSession.__new__(AttachedSession)
        fired: list[str] = []

        def drain(leaving: str) -> None:
            fired.append(leaving)

        facade._drain_callback = drain
        facade._on_retiring_frame(
            {"op": "retiring", "reason": "shutdown-drain", "to": "", "draining": True}
        )
        facade._on_retiring_frame(
            {"op": "retiring", "reason": "stale-build", "to": "0.55.6@46a4e9b", "draining": True}
        )
        facade._on_retiring_frame(
            {"op": "retiring", "reason": "retiring for 0.55.6@46a4e9b", "draining": True}
        )
        assert fired == [LEAVING_ON_SIGNAL, LEAVING_FOR_BUILD, LEAVING_FOR_BUILD], fired


@pytest.mark.asyncio
async def test_the_refusal_face_agrees_with_the_notice_on_a_pre_key_runtime(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """U14/M-1/D11: two faces of ONE drain tell one story, across versions.

    The acceptance statement this round is about, at the seam the operator
    reads. A build of this branch from before the refusal's ``error_trigger``
    field — ``8dd605365`` and the twelve rungs around it — signal-drains and
    refuses an admission, so a ``prompt_and_wait`` is answered by a raiser that
    cannot name its own departure. The viewer already holds what the raiser
    cannot say: the phrase it derived from the frame it painted the notice from
    a moment earlier.

    Rounds through 4 asked only that the NOTICE stop claiming a build; the
    refusal sat underneath it still inheriting the build sentence, which is the
    same state told two ways in one window (agent review round 5, MINOR-1; UX
    round 5, U14; design round 5, D11). Both frames here are the wire shapes
    measured in rounds 4 and 5, and the refusal is built the way the far side
    builds it: ``admission_error`` over the category the older raiser does send,
    with the phrase as the only evidence of which departure it is.
    """
    from local_operator.session.errors import admission_error
    from local_operator.session.runtime.types import drain_phrase_for_frame

    pre_key_signal_frame = {
        "op": "retiring",
        "reason": "shutdown-drain",
        "to": "",
        "draining": True,
    }
    phrase = drain_phrase_for_frame(pre_key_signal_frame)
    assert phrase == LEAVING_ON_SIGNAL, phrase
    refusal = admission_error(RuntimeRetiring.code, None, None, phrase)
    assert isinstance(refusal, RuntimeRetiring), refusal

    session = FakeSession()

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> None:
        raise refusal

    session.prompt = prompt  # type: ignore[assignment]

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._on_runtime_draining(phrase)
        await pilot.pause()
        await _send(pilot, editor, "a message sent mid-drain")

        rows = [n._text for n in _notices(app)]
        assert rows[0] == SIGNAL_DRAIN_NOTICE, rows
        assert rows[1] == _retiring_notice_text(refusal), rows
        assert rows[1].startswith(RuntimeRetiring.HEAD_SIGNALLED), rows
        assert all("newer build" not in text for text in rows), rows
        assert editor.text == "a message sent mid-drain" + RESTORE_SEAM, editor.text
        assert _user_texts(app) == [], _user_texts(app)


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
    (QA round 3, Q-1). The phrase is forwarded verbatim (design round 3, D6), and
    a frame that carries none has its trigger read off its own ``reason``/``to``
    (design round 4, D9) — only a frame that names no trigger at all reaches the
    host as ``""``.
    """
    from local_operator.session.attached import AttachedSession

    facade = AttachedSession.__new__(AttachedSession)
    fired: list[str] = []
    facade._drain_callback = lambda leaving: fired.append(leaving)  # type: ignore[method-assign]

    facade._on_retiring_frame({"op": "retiring", "draining": True, "leaving": LEAVING_ON_SIGNAL})
    assert fired == [LEAVING_ON_SIGNAL], fired

    facade._on_retiring_frame({"op": "retiring", "draining": True, "leaving": LEAVING_FOR_BUILD})
    assert fired == [LEAVING_ON_SIGNAL, LEAVING_FOR_BUILD], fired

    # A frame that names no trigger: this is the population the old fallback
    # mislabelled. A pre-``leaving`` build of THIS branch signal-drains through
    # the same shape as a released build's stale-build handover, and they differ
    # in ``reason`` — so both are asserted, and so is the one that says neither.
    facade._on_retiring_frame(
        {"op": "retiring", "draining": True, "reason": "shutdown-drain", "to": ""}
    )
    facade._on_retiring_frame(
        {"op": "retiring", "draining": True, "reason": "stale-build", "to": "0.55.6@46a4e9b"}
    )
    facade._on_retiring_frame({"op": "retiring", "draining": True})
    assert fired == [
        LEAVING_ON_SIGNAL,
        LEAVING_FOR_BUILD,
        LEAVING_ON_SIGNAL,
        LEAVING_FOR_BUILD,
        "",
    ], fired

    facade._on_retiring_frame({"op": "retiring", "draining": False})
    facade._on_retiring_frame({"op": "retiring"})  # older runtime: no field
    assert fired == [
        LEAVING_ON_SIGNAL,
        LEAVING_FOR_BUILD,
        LEAVING_ON_SIGNAL,
        LEAVING_FOR_BUILD,
        "",
    ], fired


# -- the THIRD outcome: the message is queued for the build replacing this one -----
#
# The refusal above is the fallback, not the ordinary path any more. A draining
# runtime spools the user's own prompt for its successor (memo §4.2), and the
# answer it gives is a receipt rather than an exception — so the two behaviours
# the refusal branch exists for are exactly WRONG here: there is nothing to hand
# back (the successor has the message) and nothing to retract (the row stands for
# a message that will run).
#
# WHAT THE QUEUED STATE IS, AND WHERE IT ENDS (design round 1, D2/D3/D4; UX round
# 1, U1/U3). The state is painted on the message's own row and taken down by the
# successor's announcement of that message — one statement, with an end — rather
# than as a receipt row below it, which could only accumulate one identical line
# per send and could never stop asserting the queue after the message had run.
# The cells below drive: the receipt (row marked, nothing handed back, no notice),
# the settlement (marker off when the message is announced), and the recall
# (withdrawn from the spool, or an honest "too late").


class _QueuedSession(FakeSession):
    """A session whose runtime queued the message for the build taking over.

    It accepts ``message_id`` exactly as both in-tree sessions do. That keyword
    is the correlation id the app mints, and it is the ONLY key that survives the
    handover — a fake without it sends every cell down the id-less path, where no
    marker can exist and no announcement can be matched.
    """

    def __init__(self) -> None:
        super().__init__()
        self.queued_ids: list[str] = []

    async def prompt(  # type: ignore[override] — the fake widens nothing but the return
        self, text: str, images: Any = None, *, message_id: str = ""
    ) -> Any:
        self.prompts.append(text)
        self.prompt_images.append(list(images or []))
        self.queued_ids.append(message_id)
        return SPOOL_RECEIPT_PROMPT


def _queued_session() -> _QueuedSession:
    return _QueuedSession()


def _user_rows(app: OperatorApp) -> list[str]:
    """Every painted row of every user block, marker rows included.

    ``UserBlock.text()`` is the PROMPT, deliberately — the receipt rows are the
    app talking and are excluded from a copy — so the marker has to be read off
    the authored rows, which is what the frame paints.
    """
    rows: list[str] = []
    for block in _blocks(app):
        if isinstance(block, UserBlock):
            rows.extend(block._rows(40))
    return rows


def _spool_row(app: OperatorApp, command_id: str, text: str) -> None:
    """Write the row the draining runtime would have spooled for this message."""
    from local_operator.paths import config_dir
    from local_operator.session.runtime.inbox import (
        SOURCE_USER,
        InboxLine,
        append_inbox,
    )

    session = app._session
    assert session is not None, "the app has not booted a session"
    directory = config_dir() / "sessions" / str(session.session_id)
    assert append_inbox(
        directory,
        InboxLine(text=text, sender={}, source=SOURCE_USER, command_id=command_id, wake=True),
    ), "the spool row could not be written"


def _admitting_session() -> FakeSession:
    """The ordinary case, which must stay silent."""
    session = FakeSession()

    async def prompt(text: str, images: Any = None, **kwargs: Any) -> str:
        return "prompt admitted"

    session.prompt = prompt  # type: ignore[assignment]
    return session


async def _send_queued(pilot: Any, editor: Editor, text: str, session: Any) -> None:
    """Submit, then wait for the prompt worker to record the queued receipt.

    ``_send``'s own wait is on the composer refilling, which is precisely what
    must NOT happen for a queued message — so the wait is on the session having
    seen the send, which is the fact the assertions below then read.
    """
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    for _ in range(100):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if session.queued_ids:
            return
    raise AssertionError("the queued receipt never reached the session")


@pytest.mark.asyncio
async def test_a_queued_prompt_keeps_its_row_and_carries_the_queued_state() -> None:
    """The receipt is not a refusal, and the state it leaves is ON the row.

    Everything the refusal branch does here would be a falsehood: the draft is
    not returned (the successor holds the message, and a composer copy invites
    the user to send it twice) and the echo row is not withdrawn (the message is
    real and the successor runs it).

    AND NO NOTICE IS APPENDED (design round 1, D4/U3): the row itself carries
    :data:`QUEUED_ROW_TEXT`, so three messages sent during one drain cost three
    rows rather than three identical receipts plus the standing drain row.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "deploy the fix", session)

        assert editor.text == "", "a queued message was handed back as unsent"
        assert _user_texts(app) == ["deploy the fix"], "the row for a queued message was retracted"
        assert _notices(app) == [], [n._text for n in _notices(app)]
        rows = _user_rows(app)
        assert QUEUED_ROW_TEXT in rows, rows
        # The refusal's own copy must not appear anywhere: this message is not
        # coming back, and the drain's sentence is the standing notice's job.
        assert not [row for row in rows if "back in the composer" in row], rows
        # Matched by IDENTITY against the constant the runtime answers with: the
        # message id travelled with the send, which is what the settlement and
        # the recall both key on.
        assert session.queued_ids and session.queued_ids[-1], session.queued_ids


@pytest.mark.asyncio
async def test_the_queued_marker_comes_down_when_the_successor_runs_the_message() -> None:
    """D2: the state has an END, and the successor's own announcement is it.

    The marker is the only evidence the message exists — it is in the spool and
    in no transcript — so a marker that never comes down reads identically
    before the successor has served the message and after it has. The
    announcement is matched on the id the app sent, which is the same id the
    successor's turn carries.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "deploy the fix", session)
        message_id = session.queued_ids[-1]
        assert QUEUED_ROW_TEXT in _user_rows(app)

        app.post_message(UserMessageStart("deploy the fix", 0, message_id))
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if QUEUED_ROW_TEXT not in _user_rows(app):
                break

        assert QUEUED_ROW_TEXT not in _user_rows(app), _user_rows(app)
        assert session.session_id  # the row, and only the marker, changed


@pytest.mark.asyncio
async def test_esc_takes_a_queued_prompt_back_out_of_the_spool() -> None:
    """U1: the queued message is recallable, and the recall says what happened.

    Enter committed the user's words to a process that may run them hours later,
    so the cancel key has to be able to take them back — the neighbouring steer
    channel already works this way (Esc lifts the newest queued steer into the
    composer). The message lives in a FILE, which is why this needs no op: no
    runtime has read the row, and that is exactly what makes it recallable.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "deploy the fix", session)
        command_id = session.queued_ids[-1]
        _spool_row(app, command_id, "deploy the fix")

        await pilot.press("escape")
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if editor.text:
                break

        assert editor.text == "deploy the fix", editor.text
        assert _user_texts(app) == [], "the row for a withdrawn message stayed up"
        from local_operator.paths import config_dir
        from local_operator.session.runtime.inbox import peek_inbox

        remaining = peek_inbox(config_dir() / "sessions" / str(session.session_id))
        assert remaining == [], remaining
        assert [n._text for n in _notices(app)] == [QUEUED_PROMPT_TAKEN_BACK_NOTICE], [
            n._text for n in _notices(app)
        ]


@pytest.mark.asyncio
async def test_esc_says_so_when_the_successor_has_already_taken_the_message() -> None:
    """U1: losing the race is an ANSWER, never a silent no-op.

    Once the successor has drained the spool the row is gone and the message WILL
    run. A recall that quietly did nothing would leave the user believing they had
    taken it back — so the miss says exactly that, and the marker stays up because
    the message is still queued, only out of reach.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "deploy the fix", session)

        await pilot.press("escape")
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if _notices(app):
                break

        assert editor.text == "", "a message already in the successor's hands was handed back"
        assert [n._text for n in _notices(app)] == [QUEUED_PROMPT_MISSED_NOTICE], [
            n._text for n in _notices(app)
        ]
        assert QUEUED_ROW_TEXT in _user_rows(app), "the message is still queued; only out of reach"


@pytest.mark.asyncio
async def test_an_admitted_prompt_paints_no_handover_row() -> None:
    """The receipt is matched by IDENTITY, so the ordinary answer stays silent.

    ``prompt admitted`` is the durable append and needs no explanation; a viewer
    that painted the handover row for it (or for any other receipt) would tell the
    user their message was deferred when it was already in the history.
    """
    session = _admitting_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = "an ordinary message"
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)

        assert _notices(app) == [], [n._text for n in _notices(app)]
        assert _user_texts(app) == ["an ordinary message"]
