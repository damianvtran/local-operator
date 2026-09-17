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

The composer is not the only reader of the byte. ``Editor._on_key`` normalises
the name it gates on, but it also hands the EVENT to the app's live-prompt
router (``OperatorApp.route_key_to_live_prompt``), which compares ``event.key``
itself — so a rewrite of the local name alone left an LF spelled ``ctrl+j``
there. With an answer key held that fell through to
``editor.insert(held.character)``, and this method then submitted the restored
character as a CHAT PROMPT while the question stayed up unanswered: measured as
``prompts == ["y"]`` with the approval card still mounted, against CR's
answered prompt. That is a regression on the pre-fix behaviour, where the inert
byte left the hold timer to commit the answer, and it is deterministic for
integration delivery — both bytes of one ``"y\\n"`` write land inside the hold
window by construction, not by timing luck. So the byte is normalised on the
EVENT as well, and the router is pinned here beside the composer.
"""

from __future__ import annotations

import asyncio

import pytest
from textual import events

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.approval import ApprovalPrompt
from local_operator.tui.widgets.editor import Editor, EditorSubmitted
from tests.unit.tui.test_app_pilot import FakeSession, _factory

# The hold fixture is IMPORTED rather than rebuilt, for the reason
# `test_paste_collapse` imports `skill_root`: a second definition of the
# stretch is how two files come to disagree about how long "held" is. It is
# activated with `usefixtures` rather than taken as a parameter, because a
# parameter of the same name shadows this import (F811) and none of these tests
# reads what it returns — they read the app's own `_held_answer_key`.
from tests.unit.tui.test_steering_approval import (  # noqa: F401
    SteerableSession,
    _booted_gate,
    _focus_composer,
    unraceable_answer_hold,
)
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


async def _answered_from_bytes(deliver) -> dict:  # type: ignore[no-untyped-def]
    """Run a live approval with the composer focused, then deliver BYTES.

    ``deliver(pilot, app, editor)`` is the only difference between the cases:
    the answer key pressed and then a terminator, or the whole write at once.
    Both go through the same ``_feed`` seam as the composer tests, so what is
    measured is what the terminal writes. The hold is stretched by
    ``unraceable_answer_hold``, which makes "the deadline never fired first"
    true by construction rather than by timing luck.
    """
    session = SteerableSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 30)) as pilot:
        ask = await _booted_gate(pilot, session)
        pending = asyncio.ensure_future(ask("bash", "run: rm -rf ./build"))
        for _ in range(100):
            if app.query(ApprovalPrompt):
                break
            await pilot.pause(0.02)
        await pilot.pause()
        await _focus_composer(pilot, app)
        editor = app.query_one(Editor)
        submitted = _spy_submissions(editor)

        await deliver(pilot, app, editor)
        for _ in range(4):
            await pilot.pause()

        try:
            resolved = await asyncio.wait_for(asyncio.shield(pending), 2)
        except asyncio.TimeoutError:
            resolved = None
        pending.cancel()
        return {
            "resolved": resolved,
            "submitted": submitted,
            "prompts": list(session.prompts),
            "card_still_up": bool(app.query(ApprovalPrompt)),
            "editor_text": editor.text,
        }


async def _held_answer_after_byte(byte: str) -> dict:
    """Hold the `y` answer key, then deliver ``byte`` from the driver."""

    async def deliver(pilot, app, editor) -> None:  # type: ignore[no-untyped-def]
        await pilot.press("y")
        await pilot.pause()
        assert app._held_answer_key is not None, "precondition: the answer key is held"
        await _feed(app, byte)

    return await _answered_from_bytes(deliver)


async def _single_write_after(text: str) -> dict:
    """Deliver the answer key AND its terminator in ONE write.

    This is the shape an integration produces: sidekick.nvim, tmux and every
    editor plugin hand the terminal a whole line in one write, so both bytes
    land inside the hold window by construction. There is no composition step
    in which a human's inter-key interval could intervene.
    """

    async def deliver(pilot, app, editor) -> None:  # type: ignore[no-untyped-def]
        await _feed(app, text)

    return await _answered_from_bytes(deliver)


@pytest.mark.usefixtures("unraceable_answer_hold")
@pytest.mark.asyncio
async def test_an_lf_byte_takes_a_held_answer_key_like_enter() -> None:
    """The byte must be Enter at the ROUTER too, not only in the composer.

    Rewriting the composer's local name left the event itself spelled `ctrl+j`,
    and the router reads the event: the held `y` was restored into the buffer
    and then submitted as a chat prompt while the question sat there waiting.
    That regressed the pre-fix behaviour, so the byte is normalised on the
    event as well.
    """
    obs = await _held_answer_after_byte("\n")

    assert obs["resolved"] is True, f"the LF never answered the question: {obs}"
    assert obs["submitted"] == [], f"the held character became a prompt: {obs}"
    assert obs["prompts"] == [], f"the answer went to the agent: {obs}"
    assert obs["card_still_up"] is False, f"the card never resolved: {obs}"
    assert obs["editor_text"] == "", f"the answer key was left in the buffer: {obs}"


@pytest.mark.usefixtures("unraceable_answer_hold")
@pytest.mark.asyncio
async def test_an_lf_that_ends_one_write_answers_like_cr() -> None:
    """The integration form: `y` and the terminator in the SAME write.

    The pressed-key test above has a gap between the two events that a write
    does not: here the answer key is armed by the byte stream itself, so the
    hold is committed by the very next event in the same parse pass. This is
    the delivery every editor integration actually makes.
    """
    obs = await _single_write_after("y\n")

    assert obs["resolved"] is True, f"the LF never answered the question: {obs}"
    assert obs["submitted"] == [], f"the answer became a prompt: {obs}"
    assert obs["prompts"] == [], f"the answer went to the agent: {obs}"
    assert obs["card_still_up"] is False, f"the card never resolved: {obs}"


@pytest.mark.usefixtures("unraceable_answer_hold")
@pytest.mark.asyncio
async def test_a_cr_single_write_answers_the_question() -> None:
    """The single-write control, so the LF claim above has a referent."""
    obs = await _single_write_after("y\r")

    assert obs["resolved"] is True, f"CR never answered the question: {obs}"
    assert obs["submitted"] == [], f"a stray prompt: {obs}"
    assert obs["prompts"] == [], f"the answer went to the agent: {obs}"
    assert obs["card_still_up"] is False, f"the card never resolved: {obs}"


@pytest.mark.usefixtures("unraceable_answer_hold")
@pytest.mark.asyncio
async def test_a_cr_byte_takes_a_held_answer_key() -> None:
    """The control, and the LF case's independent source of truth.

    Identical to the LF test but for the byte, so a difference between the two
    is the defect and a failure here means the LF claim has no referent.
    """
    obs = await _held_answer_after_byte("\r")

    assert obs["resolved"] is True, f"CR never answered the question: {obs}"
    assert obs["submitted"] == [], obs
    assert obs["prompts"] == [], obs
    assert obs["card_still_up"] is False, obs
    assert obs["editor_text"] == "", obs
