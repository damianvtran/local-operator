"""What a RESUMED conversation is called on the band and on the terminal tab.

Resuming used to lose the name entirely. The session's title lived in memory
only, so ``--resume``/``/resume`` rebuilt the conversation, replayed its whole
history onto the screen, and then labelled it with the working directory —
turning a sidebar of resumed sessions back into the row of identical ``lo ›
<dir>`` entries the terminal title feature exists to prevent.

Two halves, both covered here:

* A session whose transcript CARRIES a journalled title is adopted already
  named, and ``_adopt_session`` paints that name onto the band and the tab.
* A session whose transcript carries none — every transcript written before
  titles were journalled, and any session closed before its naming call landed
  — falls back to its own opening message, the same text ``/resume``'s picker
  labels its rows with, so the row picked and the tab landed on agree.

The fallback is a DISPLAY stand-in and must stay one: written into the session
it would tell the naming errand this conversation is already named and retire
the one call that could give it a real title.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.naming import ConversationName
from local_operator.tui.app import OperatorApp
from local_operator.tui.terminal_title import TerminalTitle
from tests.unit.tui.test_app_pilot import FakeSession, _factory

OPENER = "Investigate why the nightly importer drops rows silently"


class _ResumedSession(FakeSession):
    """A session as a resume hands one over: prior history, maybe a title.

    ``title`` is applied through the real :class:`ConversationName` holder the
    fake already owns, which is what a resumed ``Session`` does when it replays
    its own ``conversation_name`` entry.
    """

    def __init__(self, *, title: str = "", user_set: bool = False, history: bool = True) -> None:
        super().__init__()
        if title:
            self._name_state = ConversationName(text=title, user_set=user_set, requested=True)
        self._history = (
            [
                Message(role="user", content=[TextContent(text=OPENER)]),
                Message(role="assistant", content=[TextContent(text="Looking now.")]),
            ]
            if history
            else []
        )

    def history(self) -> list[Any]:
        return list(self._history)


async def _boot(session: _ResumedSession, pilot: Any) -> None:  # noqa: ANN401
    """Pump until the session is adopted (see test_conversation_naming._ready)."""
    for _ in range(200):
        if pilot.app._session is not None:
            return
        await pilot.pause()
    raise AssertionError("the session was never adopted")


def _tab_title(app: OperatorApp) -> str:
    """The OSC 0 label this app would put on the terminal tab.

    Read through a TerminalTitle attached to the band, because the pilot runs
    headless and the app installs no writer of its own — there is no terminal.
    ``current`` renders the same string ``emit`` would have written.
    """
    band = app._status
    assert band is not None
    writer = TerminalTitle(lambda _text: None)
    band.set_terminal_title(writer)
    return writer.current


@pytest.mark.asyncio
async def test_a_resumed_session_wears_its_stored_title() -> None:
    """THE bug: the name was on disk and the band showed the cwd anyway."""
    session = _ResumedSession(title="Reduce agent RAM usage")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(session, pilot)
        assert app._status is not None
        assert app._status._conversation_name == "Reduce agent RAM usage"
        assert _tab_title(app) == "lo › Reduce agent RAM usage"
        # The stored title is the store of record, so no stand-in was adopted
        # beside it — a later `/rename` has one thing to supersede, not two.
        assert app._provisional_name == ""


@pytest.mark.asyncio
async def test_a_resume_without_a_stored_title_falls_back_to_the_opener() -> None:
    """Legacy transcripts carry no title; the conversation still has a subject.

    Without this the band read `lo › <cwd>` for a conversation whose opening
    message was on screen two rows up.
    """
    session = _ResumedSession(title="")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(session, pilot)
        assert app._status is not None
        assert app._status._conversation_name.startswith("Investigate why the nightly importer")
        assert _tab_title(app).startswith("lo › Investigate why the nightly importer")
        # DISPLAY only: the store stays empty so the naming errand still fires
        # and can replace the quote with an actual title.
        assert session.conversation_name == ""
        assert app._provisional_name != ""


@pytest.mark.asyncio
async def test_a_stored_title_is_never_displaced_by_the_opener_fallback() -> None:
    """The fallback is for the nameless case only.

    A resumed session carrying a user's own rename must not have it replaced by
    a quote of the first message — the rename outranks everything, forever.
    """
    session = _ResumedSession(title="Q3 billing migration", user_set=True)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(session, pilot)
        assert app._status is not None
        assert app._status._conversation_name == "Q3 billing migration"
        assert app._provisional_name == ""


@pytest.mark.asyncio
async def test_a_fresh_session_is_not_given_a_name_by_the_resume_path() -> None:
    """No history, no name: an ordinary new session keeps its cwd fallback.

    The restore runs on every adoption, not only on a resume, so the empty case
    is the one that proves it cannot invent a label for a conversation that has
    not started.
    """
    session = _ResumedSession(title="", history=False)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(session, pilot)
        assert app._status is not None
        assert app._status._conversation_name == ""
        assert app._provisional_name == ""
        # The band's own cwd fallback is what names the tab, unchanged.
        assert _tab_title(app).startswith("lo › ")


@pytest.mark.asyncio
async def test_the_restored_name_survives_the_first_repaint() -> None:
    """The band re-renders constantly; the name must not be a one-frame flash."""
    session = _ResumedSession(title="Reduce agent RAM usage")
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        await _boot(session, pilot)
        assert app._status is not None
        app._status.refresh()
        await pilot.pause()
        assert app._status._conversation_name == "Reduce agent RAM usage"
        assert _tab_title(app) == "lo › Reduce agent RAM usage"
