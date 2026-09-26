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
    QUEUED_ELSEWHERE_NOTICE,
    QUEUED_PROMPT_MISSED_NOTICE,
    QUEUED_PROMPT_TAKEN_BACK_NOTICE,
    RESTORE_SEAM,
    SIGNAL_DRAIN_NOTICE,
    OperatorApp,
    _is_retiring_refusal,
    _retiring_notice_text,
)
from local_operator.tui.events import UserMessageStart
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.transcript import (
    QUEUED_ROW_TEXT,
    QUEUED_ROW_TEXT_OLDER,
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


async def _send(pilot: Any, app: OperatorApp, editor: Editor, text: str) -> None:
    """Type + Enter, then wait for the refusal to land on its failure record.

    Under the boundary rule the landing is NOT the composer refilling (the
    superseded preference) — the payload stays on the record — so the record
    itself is what says the prompt worker has settled.
    """
    editor.text = text
    await pilot.pause()
    await pilot.press("enter")
    for _ in range(100):
        await pilot.pause()
        await asyncio.sleep(0.01)
        if app._interaction.turn.failed_sends:
            return


@pytest.mark.asyncio
async def test_a_drain_refusal_keeps_its_row_and_offers_the_two_verbs() -> None:
    """D1/U1 under the boundary rule: the row survives and the fate is stated.

    SUPERSEDES the withdraw-and-return this cell used to pin. The refusal is
    real, but the message is the user's and keeps its place; what the round-3
    findings still own is the SENTENCE — recomposed by the same writer
    (``_retiring_notice_text``) with the composer claim swapped for the fact
    that is now true ("Your message was not sent") — and the notice adds the two
    verbs, so the payload moves only when the user says so (``edit``).
    """
    session = _retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "now summarise the build staleness fix")

        assert editor.text == "", "the payload returned to the composer by itself"
        assert _user_texts(app) == [
            "now summarise the build staleness fix"
        ], "the row for a refused message was withdrawn (superseded preference)"

        notices = _notices(app)
        assert len(notices) == 1, [n._text for n in notices]
        # THE PRODUCT COMPOSES THIS: the sentence is identical to the writer the
        # viewer calls, and the composer claim it used to carry is now the fact
        # the boundary rule makes true.
        sentence = _retiring_notice_text(RuntimeRetiring())
        assert notices[0]._text.startswith(sentence), notices[0]._text
        assert notices[0]._text.endswith("send again \u23ce · edit e"), notices[0]._text
        assert "Your message was not sent" in notices[0]._text
        assert "back in the composer" not in notices[0]._text, notices[0]._text
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
        await _send(pilot, app, editor, "is the build still moving?")

        assert editor.text == "", "the payload returned to the composer by itself"
        assert _user_texts(app) == [
            "is the build still moving?"
        ], "the row for a refused message was withdrawn (superseded preference)"
        notices = _notices(app)
        assert [n._text for n in notices] == [
            f"{_LEGACY_RETIRING_REFUSAL}. Your message was not sent."
            " — send again \u23ce · edit e"
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
async def test_following_the_refusal_does_not_stack_rows_or_seams() -> None:
    """U1's measurement, restated for the boundary rule.

    Two presses used to leave TWO rows for one message and, after round 5's fix,
    a seam per press. Under the boundary rule the composer stays EMPTY after the
    refusal — the retry lives on the row — so the accumulation surface is the
    transcript: pressing Enter with nothing to send adds nothing, and the row's
    own `send again` retires-then-resubmits in one handler (J1), so a second
    failed attempt lands on the same single row.

    THE SEAM GUARD IS STILL EXERCISED — by `edit` in the next-thought cell
    below, and by the resend replay in `test_send_failure_row.py`.
    """
    session = _retiring_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send(pilot, app, editor, "the build is moving")
        assert editor.text == "", editor.text
        assert _user_texts(app) == ["the build is moving"]
        assert len(_notices(app)) == 1

        # Pressed for real with an empty composer: nothing to send, nothing may
        # grow (the superseded shape refilled the composer, and this press is
        # how the old cell made it accumulate).
        for _ in range(2):
            await pilot.press("enter")
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.01)
        assert editor.text == "", editor.text
        assert _user_texts(app) == ["the build is moving"], _user_texts(app)
        assert len(_notices(app)) == 1, [n._text for n in _notices(app)]

        # And the row's own route: `send again` on the notice.
        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if app._interaction.turn.failed_sends:
                break
        assert _user_texts(app) == ["the build is moving"], (
            "the replay's own failure stacked a second row",
            _user_texts(app),
        )
        assert len(_notices(app)) == 1, [n._text for n in _notices(app)]
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
        await _send(pilot, app, editor, "summarise the build staleness fix")
        assert editor.text == ""

        # The restore is now the user's deliberate act (`e` on the notice) —
        # and it still lands behind the seam, because the weld hazard is
        # unchanged in kind: a returned draft and the next thought must stay
        # separable.
        (notice,) = _notices(app)
        notice.focus()
        await pilot.pause()
        await pilot.press("e")
        for _ in range(100):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if editor.text:
                break
        assert editor.text == "summarise the build staleness fix" + RESTORE_SEAM, editor.text

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
        windows: list[str] = []

        def drain(leaving: str, *, updating: str = "") -> None:
            fired.append(leaving)
            windows.append(updating)

        facade.set_drain_callback(drain)
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
        assert windows == ["", "", ""], "a drain carries no update window"


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
        await _send(pilot, app, editor, "a message sent mid-drain")

        rows = [n._text for n in _notices(app)]
        assert rows[0] == SIGNAL_DRAIN_NOTICE, rows
        assert rows[1].startswith(_retiring_notice_text(refusal)), rows
        assert rows[1].endswith("send again \u23ce · edit e"), rows
        assert rows[1].startswith(RuntimeRetiring.HEAD_SIGNALLED), rows
        assert all("newer build" not in text for text in rows), rows
        assert editor.text == "", editor.text
        assert _user_texts(app) == ["a message sent mid-drain"], _user_texts(app)


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
    windows: list[str] = []

    def drain(leaving: str, *, updating: str = "") -> None:
        fired.append(leaving)
        windows.append(updating)

    # ``set_drain_callback``, not a direct attribute write: the facade resolves
    # whether the host takes the ``updating`` keyword when the callback is SET, so a
    # cell that bypasses the setter is testing a state the runtime cannot produce.
    facade.set_drain_callback(drain)

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

    # AN UPDATE WINDOW SPEAKS EVEN THOUGH NOBODY IS DRAINING, and that is the new
    # half of this seam rather than a fourth drain. The idle rung announces with
    # ``draining`` false — it is not finishing work, it is moving, and its messages
    # are QUEUED rather than refused — so before the ``updating`` key that frame
    # reached this host not at all, and the one handover that holds the operator's
    # message was the one they were told nothing about (``types.UPDATING``, the
    # 2026-09-19 incident).
    facade._on_retiring_frame(
        {
            "op": "retiring",
            "draining": False,
            "reason": "stale-build",
            "to": "0.59.11@ead71b6",
            "updating": "0.59.9 → 0.59.11@ead71b6",
        }
    )
    assert fired[-1] == LEAVING_FOR_BUILD, fired
    assert windows[-1] == "0.59.9 → 0.59.11@ead71b6", windows
    assert set(windows[:-1]) == {""}, "only a window frame may carry a pair"


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
    """One string per user block: its painted rows, marker rows included.

    ``UserBlock.text()`` is the PROMPT, deliberately — the receipt rows are the
    app talking and are excluded from a copy — so the marker has to be read off
    the authored rows, which is what the frame paints.

    JOINED WITH ONE SPACE, because the marker WRAPS (design round 2, D6) and the
    tests care about the sentence, not the width the harness happened to lay the
    frame out at: ``wrap_cells`` splits on spaces and rebuilds with a single one,
    so the join reproduces the text the app painted.
    """
    return [" ".join(block._rows(40)) for block in _blocks(app) if isinstance(block, UserBlock)]


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
        assert any(QUEUED_ROW_TEXT in row for row in rows), rows
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
        assert any(QUEUED_ROW_TEXT in row for row in _user_rows(app))

        app.post_message(UserMessageStart("deploy the fix", 0, message_id))
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if QUEUED_ROW_TEXT not in _user_rows(app):
                break

        assert not any(QUEUED_ROW_TEXT in row for row in _user_rows(app)), _user_rows(app)
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
        from local_operator.session.runtime.inbox import (
            SOURCE_RECALL,
            inbox_path,
            peek_inbox,
        )

        directory = config_dir() / "sessions" / str(session.session_id)
        # NOT DELIVERABLE — which is the whole contract — rather than "gone from
        # the file": the recall is an append-only MARKER (see `withdraw_inbox`),
        # so the row is still on disk until the next drain consumes the batch,
        # and every reader drops it in the meantime.
        assert peek_inbox(directory) == [], peek_inbox(directory)
        raw = inbox_path(directory).read_text(encoding="utf-8")
        assert SOURCE_RECALL in raw, raw
        assert command_id in raw, raw
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
        assert any(
            QUEUED_ROW_TEXT in row for row in _user_rows(app)
        ), "the message is still queued; only out of reach"


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


def _editor(app: OperatorApp) -> Any:
    """The app's composer, wherever the interaction hides it."""
    return app._editor()


def _durable_row_for(app: OperatorApp, command_id: str) -> None:
    """Write the row the successor appends when it RUNS a spooled message.

    The real shape: a user row carrying ``producer_command_id`` is what puts the
    id into the transcript's append-only index, which is the evidence
    ``Transcript.has_admitted_command`` answers from — and the same index the
    runtime uses to avoid running a twice-spooled message.
    """
    import json

    from local_operator.paths import config_dir
    from local_operator.session.transcript import TRANSCRIPT_FILENAME

    session = app._session
    assert session is not None, "the app has not booted a session"
    directory = config_dir() / "sessions" / str(session.session_id)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / TRANSCRIPT_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "id": command_id,
                    "ts": 1,
                    "type": "message",
                    "payload": {
                        "kind": "message",
                        "role": "user",
                        "content": [],
                        "producer_command_id": command_id,
                    },
                }
            )
            + "\n"
        )


@pytest.mark.asyncio
async def test_the_marker_comes_down_on_the_durable_row_not_only_on_the_announcement() -> None:
    """U2 (UX round 2): on the REAL handover, the settlement is the transcript.

    The boot drain admits the spooled row BEFORE the successor's control socket
    exists, so a viewer that binds afterwards never receives that message's
    ``USER-MESSAGE-START`` — measured on the live handover: the marker was still
    up 60 s after the successor had answered, and the recall was still offered
    for a message that had already run. The previous cell fed the announcement by
    hand (``app.post_message``), which is the seam the real handover cannot
    supply. This one writes the DURABLE ROW the successor writes, and lets the
    bind-time settle find it.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "deploy the fix", session)
        command_id = session.queued_ids[-1]
        assert any(QUEUED_ROW_TEXT in row for row in _user_rows(app))

        _durable_row_for(app, command_id)
        assert app._interaction is not None
        app._settle_handed_over_queues(app._interaction)
        await pilot.pause()

        assert not any(QUEUED_ROW_TEXT in row for row in _user_rows(app)), _user_rows(app)
        assert app._interaction.turn.queued_prompts == {}, app._interaction.turn.queued_prompts
        # AND THE RECALL IS NO LONGER OFFERED for a message the spool no longer
        # holds: the press falls through to Esc's ordinary meaning rather than
        # answering about a message that has run.
        await pilot.press("escape")
        await pilot.pause()
        assert [n._text for n in _notices(app)] == [], [n._text for n in _notices(app)]


@pytest.mark.asyncio
async def test_a_recall_never_stops_the_turn_in_flight() -> None:
    """U1 (UX round 2): the key the row advertises must do ONE thing.

    With a turn streaming, one Esc that recalled a queued message used to fall
    through to ``action_stop`` and abort that turn — and since a drain's liveness
    IS the work in flight, killing it ended the drain, booted the successor and
    put every other queued message beyond the recall's reach about three seconds
    later. The row promises an inert undo of the user's own words.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        # The message is QUEUED first (the drain's receipt), and the turn is
        # live by the time the user reaches for the key — which is the incident's
        # state, not a second submit: a submit with a live turn is routed to a
        # steer and never reaches the spool.
        await _send_queued(pilot, editor, "deploy the fix", session)
        command_id = session.queued_ids[-1]
        _spool_row(app, command_id, "deploy the fix")
        session.streaming = True
        await pilot.pause()

        await pilot.press("escape")
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if editor.text:
                break

        assert editor.text == "deploy the fix", editor.text
        assert session.aborts == [], f"the recall stopped the turn: {session.aborts}"


@pytest.mark.asyncio
async def test_a_recall_reaches_a_queued_message_while_children_are_running() -> None:
    """U3 (UX round 2): with children up the press used to be spent on them.

    The recall sat behind ``if not children:``, so press 1 offered to stop the
    subagents, press 2 STOPPED them, and the queued message the row was
    advertising keeps sitting there untouched — at the incident's own shape
    (``subagents_running=3``). The recall is inert and safe, so it answers first.
    """
    session = _queued_session()
    session.running_children = 3
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
        assert (
            session.subagent_cancels == []
        ), f"the recall stopped the children: {session.subagent_cancels}"


@pytest.mark.asyncio
async def test_only_the_newest_queued_message_offers_the_recall_key() -> None:
    """U4 (UX round 2): two rows offering, one honouring.

    The recall lifts one message at a time, so an older queued row carrying the
    same offer promises a press that will decline — the user has to learn the
    rule by pressing. Only the newest advertises it, and the offer moves back down
    the queue when the newest leaves it.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "first queued", session)
        first_id = session.queued_ids[-1]
        await _send_queued(pilot, editor, "second queued", session)

        # Compared on the END of the row, not by containment: the offer-less
        # marker is a PREFIX of the one carrying the offer, so `in` would read
        # the newest row as both.
        def _offers(app_rows: list[str]) -> int:
            return sum(1 for row in app_rows if row.endswith(QUEUED_ROW_TEXT))

        def _offerless(app_rows: list[str]) -> int:
            return sum(1 for row in app_rows if row.endswith(QUEUED_ROW_TEXT_OLDER))

        rows = _user_rows(app)
        assert len(rows) == 2, rows
        assert _offers(rows) == 1, rows
        assert _offerless(rows) == 1, rows
        assert rows[-1].endswith(QUEUED_ROW_TEXT), rows

        # Recall the newest: the offer moves to the one still queued.
        _spool_row(app, session.queued_ids[-1], "second queued")
        await pilot.press("escape")
        for _ in range(50):
            await pilot.pause()
            await asyncio.sleep(0.01)
            if _offers(_user_rows(app)) == 1 and len(_user_rows(app)) == 1:
                break

        rows = _user_rows(app)
        assert len(rows) == 1, rows
        assert _offers(rows) == 1, rows
        assert first_id not in " ".join(rows), "the recalled row should be gone"


@pytest.mark.asyncio
async def test_the_joiner_row_is_taken_down_when_the_spool_empties() -> None:
    """D7 (design round 2): the joiner's row had no end either.

    ``QUEUED_ELSEWHERE_NOTICE`` announced messages the spool held and could never
    stop announcing them, so it sat directly above the answer it described. It
    goes on the same bind-time settle as the marker, when the spool holds no
    owner row this surface is not already showing.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        assert app._interaction is not None
        _spool_row(app, "f" * 32, "a message from the other front end")
        app._on_runtime_draining(LEAVING_FOR_BUILD)
        await pilot.pause()
        # The standing drain notice is on the frame too — the row under test is
        # the queued-messages one.
        assert QUEUED_ELSEWHERE_NOTICE in [n._text for n in _notices(app)], [
            n._text for n in _notices(app)
        ]

        from local_operator.paths import config_dir
        from local_operator.session.runtime.inbox import drain_inbox

        directory = config_dir() / "sessions" / str(session.session_id)
        assert drain_inbox(directory) != []
        app._settle_handed_over_queues(app._interaction)
        await pilot.pause()

        assert QUEUED_ELSEWHERE_NOTICE not in [n._text for n in _notices(app)], [
            n._text for n in _notices(app)
        ]


@pytest.mark.asyncio
async def test_a_bind_with_an_empty_queue_replays_no_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAJOR-3 (review round 3): the settle runs on EVERY bind, so it must be free.

    ``Transcript(directory)`` replays the whole journal eagerly — measured on this
    store's own 252.7 MB / 22,334-row journal at 3,916 ms synchronously on the
    event loop — and the settle is called after every successful bind, which is
    the join path the operator's requirement is about. With nothing queued there
    is nothing to settle, so nothing may be read.
    """
    from local_operator.paths import config_dir
    from local_operator.session import transcript as transcript_mod
    from local_operator.session.transcript import TRANSCRIPT_FILENAME

    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await _boot(pilot, app)
        assert app._interaction is not None
        # THE SEED IS WHAT MAKES THIS CELL A TEST. Without a durable row the
        # session directory does not exist yet, and `_handover_admitted_ids`'s
        # own `is_file()` guard returns before any construction — so the
        # assertion below holds on a tree with the empty-queue guard DELETED, and
        # round 3 shipped exactly that vacuous cell (agent review round 4, QA
        # Q-2). A real journal on disk makes the guard the only thing standing
        # between the bind and a whole-journal replay.
        _durable_row_for(app, "seed-row")
        assert (
            config_dir() / "sessions" / str(session.session_id) / TRANSCRIPT_FILENAME
        ).is_file(), "the seed must have created the journal this cell is about"
        built: list[tuple[Any, ...]] = []
        real = transcript_mod.Transcript

        class Counting(real):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                built.append(args)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(transcript_mod, "Transcript", Counting)

        app._settle_handed_over_queues(app._interaction)
        await pilot.pause()
        assert built == [], "an empty queue must not replay the journal"

        # NOT A REMOVAL OF THE FEATURE: with something queued and a durable row
        # for it, the read happens and the settle lands.
        await _send_queued(pilot, editor=_editor(app), text="deploy the fix", session=session)
        command_id = session.queued_ids[-1]
        _durable_row_for(app, command_id)
        app._settle_handed_over_queues(app._interaction)
        await pilot.pause()
        assert built, "a queued message must still be settled from the transcript"
        assert app._interaction.turn.queued_prompts == {}


@pytest.mark.asyncio
async def test_a_recall_that_missed_does_not_stop_the_turn() -> None:
    """UX U1 (round 4): the losing press must not kill the turn it just spoke about.

    The miss is the arm this delta widened: the successor's batch already has the
    row, the notice says so ("the next runtime already has that message"), and
    that sentence means the message WILL run. Falling through to the stop ladder
    then aborted the turn in flight on the same press — measured on the real flow
    as ``end_cause='user-stop'``, an extra ``interrupted`` row, and an answer that
    never arrives.
    """
    session = _queued_session()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _send_queued(pilot, editor, "deploy the fix", session)
        # The successor's batch took it: the spool no longer holds the row, which
        # is what makes this press a MISS rather than a recall.
        session.streaming = True
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        assert QUEUED_PROMPT_MISSED_NOTICE in [n._text for n in _notices(app)], [
            n._text for n in _notices(app)
        ]
        assert (
            session.aborts == []
        ), f"the losing recall stopped the turn it said would run: {session.aborts}"


def test_a_one_argument_drain_callback_still_hears_the_phrase() -> None:
    """Agent review round 1 (NIT 3): the keyword must not cost an old host its notice.

    The frame callback is passed ``updating=`` through a ``Callable[..., Any]`` inside
    a blanket ``except Exception``, so a host whose callback predates the keyword would
    raise ``TypeError`` INSIDE that guard and lose the whole notice — including the
    drain sentence it used to receive — with nothing but a ``logger.debug`` to show for
    it. The pre-change behaviour for such a host is the phrase and no window, which is
    what it now gets.
    """
    from local_operator.session.attached import AttachedSession

    facade = AttachedSession.__new__(AttachedSession)
    fired: list[str] = []

    def drain(leaving: str) -> None:
        fired.append(leaving)

    facade.set_drain_callback(drain)
    frame = {
        "op": "retiring",
        "reason": "stale-build",
        "to": "0.55.6@46a4e9b",
        "draining": True,
        "updating": "0.55.6 → 0.59.11@ead71b6",
    }
    facade._on_retiring_frame(frame)

    assert fired == [LEAVING_FOR_BUILD], fired


def test_the_callback_probe_reads_the_signature_it_needs() -> None:
    """The three shapes a host can present, and the answer for each."""
    from local_operator.session.attached import AttachedSession, _accepts_updating

    def one(_leaving: str) -> None: ...

    def keyword(_leaving: str, *, updating: str = "") -> None: ...

    def splatted(**kwargs: Any) -> None: ...

    assert _accepts_updating(keyword) is True
    assert _accepts_updating(splatted) is True
    assert _accepts_updating(one) is False
    # The state, not just the answer: ``set_drain_callback`` is what the facade calls.
    facade = AttachedSession.__new__(AttachedSession)
    facade.set_drain_callback(one)
    assert facade._drain_callback_takes_updating is False
