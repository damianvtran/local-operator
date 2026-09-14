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
``retiring`` frame is handled (it re-engages), but it painted nothing, so the
refusal WAS the discovery — in a window measured at ~26 s of ordinary reachable
state. The drain is therefore announced with one ``note`` on that same frame,
and only when the runtime is still busy; an idle handover, which costs nothing
and lasts about a second, keeps its silence.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.session.errors import RuntimeRetiring, admission_error
from local_operator.tui.app import DRAIN_NOTICE, OperatorApp, _is_retiring_refusal
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

        assert editor.text == "now summarise the build staleness fix", "text was dropped"
        assert _user_texts(app) == [], "a row stood for a message nobody received"

        notices = _notices(app)
        assert len(notices) == 1, [n._text for n in notices]
        assert notices[0]._text == f"{RuntimeRetiring()} — your message is back in the composer"
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

        assert editor.text == "is the build still moving?", "text was dropped"
        assert _user_texts(app) == [], "a row stood for a message nobody received"
        notices = _notices(app)
        assert [n._text for n in notices] == [
            f"{_LEGACY_RETIRING_REFUSAL} — your message is back in the composer"
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

        assert editor.text == "the build is moving"
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
async def test_a_busy_handover_is_announced_once_on_the_retiring_frame(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """U2/m1: the drain is announced, in the app's own handover vocabulary."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    session = FakeSession()
    session.runtime_idle = lambda: False  # a runtime still finishing work
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._on_runtime_refreshed()
        await pilot.pause()

        notices = _notices(app)
        assert [n._text for n in notices] == [DRAIN_NOTICE], [n._text for n in notices]
        assert (notices[0]._token, notices[0]._glyph) == ("muted", "·"), (
            "the drain is a quiet note, not a warning",
            notices[0]._token,
            notices[0]._glyph,
        )


@pytest.mark.asyncio
async def test_an_idle_handover_keeps_its_silence(monkeypatch: Any, tmp_path: Any) -> None:
    """The deliberate silence stays for the refresh nobody can lose anything to.

    An idle runtime leaves in about a second; announcing it would turn every
    ordinary refresh into a notice, which is the behaviour this PR must not
    change (design-runtime-autorefresh §3.3).
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    session = FakeSession()
    session.runtime_idle = lambda: True
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "_start_runtime_engage", lambda *, reason: None)
        app._on_runtime_refreshed()
        await pilot.pause()

        assert _notices(app) == []
