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


class SaverFakeSession(FakeSession):
    """The shared pilot fake plus the three Saver fields these tests drive.

    Declared here rather than on ``FakeSession``: only this suite reads them, and
    the shared fake is pinned to ``SessionProtocol`` by its own tests. Declaring
    them is what lets pyright check every assignment and read below instead of
    each one being an undeclared attribute on the base class.
    """

    routing_settings: dict[str, Any] | None = None
    radient_saver_choice: bool | None = None
    _transcript: Any = None


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
    session = SaverFakeSession()
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
    session = SaverFakeSession()
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
    session = SaverFakeSession()
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
            assert app._status is not None  # built on mount, which _boot awaited
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

    assert _saver_label(FrontendSessionState(session_id="s", epoch="e", saver_on=True)) == "saver"
    assert _saver_label(FrontendSessionState(session_id="s", epoch="e", saver_on=False)) == ""
    # An older backend omits the field and the model defaults it to False, so the
    # follower paints nothing rather than guessing from state it cannot read.
    assert _saver_label(FrontendSessionState(session_id="s", epoch="e")) == ""


def test_the_badge_never_paints_an_empty_word(monkeypatch: pytest.MonkeyPatch) -> None:
    """A glyph with no word is unreadable, not compact."""
    import local_operator.i18n.text as text_mod
    from local_operator.session.frontend_state import FrontendSessionState
    from local_operator.tui.app import _saver_label

    monkeypatch.setattr(text_mod, "tr", lambda namespace: (lambda message: "   "))
    assert _saver_label(FrontendSessionState(session_id="s", epoch="e", saver_on=True)) == ""


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

    session = SaverFakeSession()
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
                state = FrontendSessionState(session_id="s", epoch="e", saver_on=published)
                app._apply_frontend_state(state)
                assert app._status is not None  # built on mount
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
async def test_bare_radient_never_changes_state_however_many_enters(tmp_path: Path) -> None:
    """QA round 3, Q1: two Enters on a bare `/radient` turned Saver ON.

    `/radient` completes to a subject-less `/radient saver`, and that used to land
    on the bare-toggle path — a money-affecting feature enabled by return keys
    alone, from a fresh session where the user had typed one word. It now reports:
    the assertion is on the OUTCOME and the JOURNAL, not the intermediate buffer,
    because a buffer assertion is exactly what let the previous version pass while
    the feature flipped (QA Q2).
    """
    session = SaverFakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    journal_dir = tmp_path / "sessions" / "saver"
    session._transcript = Transcript(journal_dir)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._editor().text = ""
        await pilot.pause()
        await pilot.press(*["space" if char == " " else char for char in "/radient"])
        await pilot.pause()
        for _ in range(2):
            await pilot.press("enter")
            for _ in range(40):
                await pilot.pause()

        # NOTHING WAS WRITTEN: no choice, no journal row.
        assert session.radient_saver_choice is None
        rows = [
            entry.payload["details"]["enabled"]
            for entry in Transcript(journal_dir).entries()
            if entry.payload.get("custom_type") == "radient_saver_choice_v1"
        ]
        assert rows == [], rows
        # AND THE USER WAS TOLD WHAT IS TRUE, rather than shown an enable.
        from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

        text = "\n".join(
            block._text
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, NoticeBlock)
        )
        assert "Radient Saver: off for this session" in text, text
        assert "is on for this session" not in text, text


@pytest.mark.asyncio
async def test_an_explicit_radient_saver_value_still_writes(tmp_path: Path) -> None:
    """The rule is about the BARE form; an explicit value is honoured."""
    session = SaverFakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    session._transcript = Transcript(tmp_path / "sessions" / "saver")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        app._editor().text = ""
        await pilot.pause()
        await pilot.press(*["space" if char == " " else char for char in "/radient saver on"])
        await pilot.pause()
        await pilot.press("escape")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
        assert session.radient_saver_choice is True


@pytest.mark.asyncio
async def test_signed_out_refuses_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(saver_command, "signed_in", lambda: False)
    session = SaverFakeSession()
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


@pytest.mark.asyncio
async def test_the_app_host_agrees_with_the_serving_host_on_every_input(tmp_path: Path) -> None:
    """Round 4: the hosts disagreed on a whitespace argument, and the disagreement wrote.

    `rest or "status"` is a truthiness test, so `/radient saver␣␣` was truthy and
    reached the toggle arm. The app host was safe only because its own parse
    stripped first; the serving host forwards raw wire arguments and wrote Saver
    ON. Both hosts now strip, and this table is the same one
    `tests/unit/session/runtime/test_serving_saver_slash.py` runs against the
    serving dispatch, so the pair cannot drift apart again.
    """
    session = SaverFakeSession()
    session.routing_settings = SETTINGS
    session.radient_saver_choice = None
    journal_dir = tmp_path / "sessions" / "saver"
    session._transcript = Transcript(journal_dir)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)

        def rows() -> list[bool | None]:
            return [
                entry.payload["details"]["enabled"]
                for entry in Transcript(journal_dir).entries()
                if entry.payload.get("custom_type") == "radient_saver_choice_v1"
            ]

        cases = [
            ("/radient", []),
            ("/radient ", []),
            ("/radient saver", []),
            ("/radient saver  ", []),
            ("/radient saver \t", []),
            ("/radient saver --", []),
            ("/radient saver on", [True]),
            ("/radient saver off", [False]),
        ]
        for typed, expected in cases:
            before = len(rows())
            app._editor().text = ""
            await pilot.pause()
            app._run_slash_command(typed)
            for _ in range(40):
                await pilot.pause()
            written = rows()[before:]
            assert written == expected, (typed, written)
