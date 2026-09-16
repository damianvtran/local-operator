"""The LF byte submits the composer, exactly as Enter (CR) does.

sidekick.nvim, tmux ``send-keys``, ``expect`` and every editor integration end
a line with LF (0x0a). textual 8.2.8 decodes ``"\\x0a"`` to the key NAME
``ctrl+j`` (``textual/_ansi_sequences.py``, ``textual/keys.py``), and the
composer gated every Enter meaning on the literal ``"enter"`` — so the byte did
nothing at all. Measured on a real pty against 0.56.1: ``"alpha"`` + ``\\n``
left ``alpha`` in the buffer, unsubmitted; ``"alpha"`` + ``\\r`` submitted.

The seam is ``XTermParser().feed(raw)`` → ``event.set_sender(app)`` →
``driver.send_message(event)``, borrowed from ``test_word_caret`` rather than
re-implemented. Feeding BYTES rather than pressing a key name is the point:
``pilot.press`` presses a KEY NAME, which proves routing but would go on
passing if Textual changed the encoding it decodes into that name — the whole
question here is what a terminal's bytes turn into. ``_feed`` forwards only
``events.Key`` instances, so a ``Paste`` event never travels this seam.

Every test installs a ``post_message`` spy on the editor AND reads
``session.prompts``, so each claim is about the real app path and not about the
widget in isolation. ``session.prompts`` counts user prompts only; the system
prompt is ignored.
"""

from __future__ import annotations

import pytest
from textual import events

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import EditorSubmitted
from tests.unit.tui.test_app_pilot import FakeSession, _factory
from tests.unit.tui.test_word_caret import _boot, _feed


def _spy_submissions(editor) -> list[str]:  # type: ignore[no-untyped-def]
    """Record the text of every ``EditorSubmitted`` the editor posts.

    The ``test_inline_credential`` idiom: keep the bound method and delegate,
    so the message still reaches the app and the count stays honest.
    """
    submitted: list[str] = []
    original = editor.post_message

    def _spy(message):  # type: ignore[no-untyped-def]
        if isinstance(message, EditorSubmitted):
            submitted.append(message.text)
        return original(message)

    editor.post_message = _spy  # type: ignore[method-assign]
    return submitted


async def _type(pilot, text: str) -> None:  # type: ignore[no-untyped-def]
    """Press ``text`` one character at a time, as a user types."""
    for char in text:
        await pilot.press(char)


@pytest.mark.asyncio
async def test_an_lf_byte_submits_the_composer_like_enter() -> None:
    """The RED test: an LF byte is Enter, all the way through the app."""
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        submitted = _spy_submissions(editor)

        await _type(pilot, "alpha")
        await pilot.pause()

        await _feed(app, "\n")
        for _ in range(3):
            await pilot.pause()

        assert submitted == ["alpha"], "exactly one submit, not zero and not two"
        assert session.prompts == ["alpha"], "the agent actually received the turn"
        assert editor.text == "", "the buffer cleared"
        assert editor.prompt_history()[-1] == "alpha", "it joins the recall history"


@pytest.mark.asyncio
async def test_a_cr_byte_still_submits_the_composer() -> None:
    """The control, and test 1's independent source of truth for its values.

    CR is what a terminal sends for the Enter key. If this ever fails, the
    "LF is indistinguishable from Enter" claim has no referent.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        submitted = _spy_submissions(editor)

        await _type(pilot, "alpha")
        await pilot.pause()

        await _feed(app, "\r")
        for _ in range(3):
            await pilot.pause()

        assert submitted == ["alpha"]
        assert session.prompts == ["alpha"]
        assert editor.text == ""
        assert editor.prompt_history()[-1] == "alpha"


@pytest.mark.asyncio
async def test_an_lf_byte_accepts_an_open_picker_row_like_enter() -> None:
    """What decides the design: LF must mean Enter where Enter means *pick*.

    A submit-only rewrite would leave the picker open here and send nothing,
    so the row would need a second gesture that Enter does not need.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        submitted = _spy_submissions(editor)

        await _type(pilot, "/credential")
        for _ in range(4):
            await pilot.pause()
        assert editor._picker.is_open(), "precondition: the row is offered"

        await _feed(app, "\n")
        await pilot.pause()

        assert editor.text == "/credential ", "the row was accepted and armed"
        assert editor.credential_typing() is True
        assert submitted == [], "accepting a row is not a submit"


@pytest.mark.asyncio
async def test_an_lf_byte_during_a_masked_capture_mints_instead_of_submitting() -> None:
    """LF must mean Enter where Enter means *mint the chip*.

    The capture is armed with CR, never with LF: the arm has to be delivered
    the way the operator delivers it, so an LF cannot be what put us here.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        submitted = _spy_submissions(editor)

        await _type(pilot, "/credential")
        for _ in range(4):
            await pilot.pause()
        assert editor._picker.is_open(), "precondition: the row is offered"

        await pilot.press("enter")
        await pilot.pause()
        await _type(pilot, "12345")
        await pilot.pause()
        assert (
            editor.credential_typing() and "12345" not in editor.text
        ), "precondition: the typed secret is masked and still open"

        await _feed(app, "\n")
        await pilot.pause()

        assert submitted == [], "minting is not a submit"
        assert editor.text == "[Credential #1, 5 chars] ", "the chip replaced the mask"
        assert not editor.credential_typing(), "the capture closed"


@pytest.mark.asyncio
async def test_a_shift_enter_byte_still_inserts_a_newline_without_submitting() -> None:
    """Shift+Enter keeps meaning a line break — the fix must not eat it.

    A name-level pin already exists in ``test_app_pilot``; this one pins the
    BYTE, so the two cannot drift apart in encoding.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        submitted = _spy_submissions(editor)

        await pilot.press("a")
        await _feed(app, "\x1b[13;2u")
        await pilot.press("b")
        await pilot.pause()

        assert editor.text == "a\nb"
        assert submitted == []


@pytest.mark.asyncio
async def test_a_bracketed_paste_with_a_newline_does_not_submit() -> None:
    """A pasted block keeps its newlines instead of firing a turn per line.

    Posted to the APP, not the widget: the widget's own handler would deliver
    the paste a second time.
    """
    session = FakeSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        editor = await _boot(pilot, app)
        submitted = _spy_submissions(editor)

        await _type(pilot, "alpha")
        await pilot.pause()
        app.post_message(events.Paste("beta\ngamma\n"))
        await pilot.pause()
        await pilot.pause()

        assert editor.text == "alphabeta\ngamma\n"
        assert submitted == []
