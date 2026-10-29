"""``/saver`` and ``/radient saver`` typed into the REAL composer.

Round 1 shipped a blocker that only a typed key could find: ``/saver on``
opened the PROVIDER list (the argument handler had no branch for these
commands), ``on`` fuzzy-matched a row, and Enter rewrote the buffer to
``/saver openrouter`` — the documented enable form could not be typed at all.
Every form below is typed keystroke-by-keystroke into the real ``OperatorApp``
composer at the shipped stylesheet, and the assertions are on what the
dispatcher RECEIVED and on the journal rows actually on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.saver import command as saver_command
from local_operator.saver.gate import Availability
from local_operator.session.transcript import Transcript
from local_operator.tui.app import OperatorApp
from tests.unit.tui.test_app_pilot import FakeSession, _factory

SETTINGS = {"radient": {"saver": {"mode": "opt_in"}}}
#: The badge as the band paints it: the compression mark plus the word.
BADGE = "\u2296 saver"


def _journalled(directory: Path) -> list[Any]:
    return [
        entry.payload["details"]["enabled"]
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == saver_command.SAVER_CUSTOM_TYPE
    ]


@pytest.fixture(autouse=True)
def _gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(saver_command, "signed_in", lambda: True)

    async def positive() -> Availability:
        return Availability(signed_in=True, positive_balance=True)

    monkeypatch.setattr(saver_command, "read_availability", positive)


async def _boot(pilot, app: OperatorApp) -> None:
    for _ in range(60):
        await pilot.pause()
        if app._session is not None:
            return
    raise AssertionError("the app never bound a session")


async def _typed(pilot, app: OperatorApp, text: str) -> str:
    """Type ``text`` one key at a time and submit it; return what ran."""
    received: list[str] = []
    inner = app._run_slash_command

    def spy(*args, **kwargs):
        received.append(args[0] if args else kwargs.get("command", ""))
        return inner(*args, **kwargs)

    app._run_slash_command = spy  # type: ignore[method-assign]
    app._editor().text = ""
    await pilot.pause()
    await pilot.press(*["space" if char == " " else char for char in text])
    await pilot.pause()
    # Escape closes the picker WITHOUT touching the buffer (an asserted
    # behaviour of its own), so the Enter below submits the text rather than
    # completing the row under the cursor.
    await pilot.press("escape")
    await pilot.pause()
    await pilot.press("enter")
    for _ in range(40):
        await pilot.pause()
    return received[-1] if received else ""


def _notices(app: OperatorApp) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_every_documented_form_runs_as_typed(tmp_path: Path) -> None:
    """The four argument forms, plus the bare toggle and the usage answer."""
    session = FakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)

        assert await _typed(pilot, app, "/saver status") == "/saver status"
        assert _journalled(tmp_path / "sessions" / "saver") == []

        # The blocker: this used to arrive as `/saver openrouter`.
        assert await _typed(pilot, app, "/saver on") == "/saver on"
        assert session.radient_saver_choice is True
        assert _journalled(tmp_path / "sessions" / "saver") == [True]
        assert any("is on for this session" in text for text in _notices(app))

        assert await _typed(pilot, app, "/radient saver off") == "/radient saver off"
        assert session.radient_saver_choice is False
        assert _journalled(tmp_path / "sessions" / "saver") == [True, False]

        assert await _typed(pilot, app, "/saver") == "/saver"
        assert session.radient_saver_choice is True

        assert await _typed(pilot, app, "/saver inherit") == "/saver inherit"
        assert session.radient_saver_choice is None
        assert _journalled(tmp_path / "sessions" / "saver") == [True, False, True, None]

        assert await _typed(pilot, app, "/radient") == "/radient"
        assert any("/radient saver" in text and "toggles" in text for text in _notices(app))


@pytest.mark.asyncio
async def test_the_argument_list_offers_saver_words_not_providers(tmp_path: Path) -> None:
    """The rows that made the enable form unrunnable are gone."""
    session = FakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        for text, expected in (
            ("/saver ", {"status", "on", "off", "inherit"}),
            # U9: ONE row — the word the handler parses. The compound rows made
            # Enter complete four tokens the user never typed.
            ("/radient ", {"saver"}),
        ):
            app._editor().text = ""
            await pilot.pause()
            await pilot.press(*["space" if char == " " else char for char in text])
            await pilot.pause()
            rows = {row.name for row in app._editor().picker._choices}
            assert rows == expected, (text, rows)
            # And the first row is the READ-ONLY word where the vocabulary has
            # one, so a stray Enter reports instead of spending credit.
            first_row = app._editor().picker._choices[0].name
            assert first_row.endswith("status") or first_row == "saver"


@pytest.mark.asyncio
async def test_the_badge_appears_iff_saver_is_on(tmp_path: Path) -> None:
    """Design D1 + UX U4: one rule, and no phantom mark when off."""
    session = FakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        for width in (80, 100, 120, 140):
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            await _typed(pilot, app, "/saver inherit")
            off = app._status.render_text(width).plain
            # The icon is the mark; the cwd in these tests is the worktree path
            # (`.../saver-cmd`), so a bare substring test would pass on the path.
            assert BADGE not in off, (width, off)
            assert "\n" not in off

            await _typed(pilot, app, "/saver on")
            on = app._status.render_text(width).plain
            assert BADGE in on, (width, on)
            assert "\n" not in on


def test_a_routed_session_paints_the_published_state() -> None:
    """Review round 2, M2-2: a follower has ONLY the published bool to read.

    ``FrontendSessionState`` carries no routing settings and no resolver
    attributes, so the previous version — which called the local resolver
    unconditionally — painted nothing for a routed session whose owner was on,
    and could paint a bare glyph with no word beside it.
    """
    from local_operator.session.frontend_state import FrontendSessionState
    from local_operator.tui.app import _saver_label

    assert _saver_label(FrontendSessionState(session_id="s", epoch="e",  saver_on=True)) == "saver"
    assert _saver_label(FrontendSessionState(session_id="s", epoch="e",  saver_on=False)) == ""
    # An older backend omits the field and the model defaults it to False, so the
    # follower paints nothing rather than guessing from state it cannot read.
    assert _saver_label(FrontendSessionState(session_id="s", epoch="e")) == ""


def test_the_badge_never_paints_an_empty_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """A glyph with no word is unreadable, not compact."""
    from types import SimpleNamespace

    import local_operator.i18n.text as text_mod
    from local_operator.tui.app import _saver_label

    from local_operator.session.frontend_state import FrontendSessionState

    monkeypatch.setattr(text_mod, "tr", lambda namespace: (lambda message: "   "))
    assert _saver_label(FrontendSessionState(session_id="s", epoch="e",  saver_on=True)) == ""


@pytest.mark.asyncio
async def test_the_routed_path_paints_through_apply_frontend_state(tmp_path: Path) -> None:
    """UX round 2, U4: the badge never painted in the real app.

    The real path is ``_apply_frontend_state`` → ``saver=_saver_label(state)``
    with a ``FrontendSessionState``. The shipped test read
    ``app._status.render_text()`` on a LOCAL session, so it never exercised this
    call and stayed green while every routed band was blank. This drives the
    real method and asserts the band, so a regression fails here.
    """
    from local_operator.session.frontend_state import FrontendSessionState

    session = FakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        for width in (80, 100, 120, 140):
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            for published in (True, False):
                state = FrontendSessionState(
                    session_id="s", epoch="e", saver_on=published
                )
                app._apply_frontend_state(state)
                # Assert THIS call's own effect, before any later repaint from a
                # different path: `_status._saver` is what `_apply_frontend_state`
                # pushed, and it is the value the band renders. A longer settle
                # would let the local session's repaint answer the question and
                # hide the very path this test exists for.
                assert app._status._saver == ("saver" if published else ""), width
                band = app._status.render_text(width).plain
                assert ("\u2296 saver" in band) is published, (width, band)
                assert "\n" not in band


@pytest.mark.asyncio
async def test_one_enter_on_a_bare_radient_fills_only_what_was_typed(tmp_path: Path) -> None:
    """UX round 2, U9: Enter used to expand `/radient` into `saver status`.

    The old test pressed Escape first, which hid the expansion. This presses
    Enter on the first row and asserts the buffer is exactly `/radient saver` —
    the word the handler parses — and that a second Enter is what runs it.
    """
    session = FakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._editor().text = ""
        await pilot.pause()
        await pilot.press(*["space" if char == " " else char for char in "/radient "])
        await pilot.pause()
        assert [row.name for row in app._editor().picker._choices] == ["saver"]

        received: list[str] = []
        inner = app._run_slash_command

        def spy(*args, **kwargs):
            received.append(args[0] if args else kwargs.get("command", ""))
            return inner(*args, **kwargs)

        app._run_slash_command = spy  # type: ignore[method-assign]
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
        # The ONE word the row offered, and nothing the user did not choose:
        # this used to arrive as `/radient saver status` (UX round 2, U9).
        assert received and received[-1] == "/radient saver", received[-3:]


@pytest.mark.asyncio
async def test_signed_out_refuses_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(saver_command, "signed_in", lambda: False)
    session = FakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert await _typed(pilot, app, "/saver on") == "/saver on"
        assert session.radient_saver_choice is None
        assert _journalled(tmp_path / "sessions" / "saver") == []
        assert any("/login radient" in text for text in _notices(app))
