"""``/copy`` and its ``ctrl+o`` chord: what reaches the clipboard, and when.

The command answers a gesture users already had for a DRAG (release copies,
``on_text_selected``) but not for the common case — "give me that whole answer"
— which previously meant dragging a multi-screen message from its first row to
its last. Codex CLI's ``/copy`` is the shape being matched: the latest
*completed* response, a ``ctrl+o`` equivalent, and no partial mid-task.

What is asserted here is the PAYLOAD and the REFUSALS, because those are the two
halves a plausible implementation gets wrong in opposite directions:

* the payload must be the block's markdown SOURCE, not the flattened frame the
  user is looking at. The frame carries the wrap of whatever width the terminal
  happened to be plus Rich's own decoration, so pasting it puts box-drawing and
  hard line breaks into the user's document. ``test_transcript_selection.py``
  makes the same claim for a partial selection; this file makes it for the
  whole-message copy, which reaches the source by a different route
  (``AssistantBlock.text()`` rather than ``_copy_markdown.slice_markdown``).
* the refusals must SPEAK. ``_put_on_clipboard`` returns silently on an empty
  payload, which is right for a zero-width drag and wrong for a command the user
  typed — a typed command that no-ops silently reads as broken.
"""

from __future__ import annotations

import pytest

from local_operator.tui.app import SLASH_COMMANDS, OperatorApp
from local_operator.tui.events import (
    AssistantDelta,
    AssistantMessageEnd,
    AssistantMessageStart,
)
from local_operator.tui.widgets.editor import Editor
from local_operator.tui.widgets.toast import Toast
from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView, UserBlock
from tests.unit.tui.test_app_pilot import FakeSession, _factory

#: An answer with the two constructs a rendered frame destroys: bold markers,
#: which the frame paints as a style rather than as characters, and a fenced
#: code block, whose fence lines the frame drops entirely (``IslandCodeBlock``
#: renders a bare ``Syntax``). If the clipboard has these, it has the source.
ANSWER = """Here is the **plan**, in short.

```python
def f(x):
    return x + 1
```

That is all.
"""


async def _boot(pilot, app: OperatorApp) -> None:
    """Settle until the session exists.

    Load-bearing for the mid-stream test: ``_turn_is_live`` reads
    ``session.is_streaming``, so a test that staged the flag before the session
    was attached would exercise the no-session path while reading like it
    tested the guard.
    """
    for _ in range(40):
        await pilot.pause()
        if app._session is not None:
            return


async def _stream(pilot, app: OperatorApp, text: str, *, finish: bool = True) -> None:
    """Paint one agent message through the real event path.

    Posted as events rather than by constructing an ``AssistantBlock`` directly,
    because the thing under test is which block ``/copy`` FINDS — and the mount,
    the finalize and the ``_streaming_block`` handle are all owned by these
    handlers. A hand-mounted block would prove the walk works on a transcript
    the app never builds.
    """
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    app.post_message(AssistantDelta(text))
    await pilot.pause()
    if finish:
        app.post_message(AssistantMessageEnd(text))
        await pilot.pause()
    await pilot.pause()


async def _submit(pilot, app: OperatorApp, text: str) -> None:
    """Type a line into the real editor and press Enter — the reported path.

    Copied in shape from ``test_slash_echo.py``: calling ``_run_slash_command``
    directly would skip the editor and the submit handler, which is the pair
    that decides whether a user row is written. Esc first because Enter on an
    open picker completes the highlighted row and submits THAT.
    """
    editor = app.query_one(Editor)
    editor.text = text
    await pilot.pause()
    if editor._picker.is_open():
        await pilot.press("escape")
        await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


def _notices(app: OperatorApp) -> list[str]:
    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


def _user_rows(app: OperatorApp) -> list[str]:
    return [
        block.text()
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, UserBlock)
    ]


# -- the payload --------------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_puts_the_markdown_source_on_the_clipboard() -> None:
    """The source, not the frame. Both markers the renderer consumes survive."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/copy")
        copied = app._clipboard

    assert copied == ANSWER
    assert "**plan**" in copied  # bold keeps its markers
    assert "```python\ndef f(x):\n    return x + 1\n```" in copied  # the fence, verbatim
    # The frame's own furniture must not be in the paste. `─` is the rule Rich
    # draws around a code block; `▌` is the blockquote bar. Neither is source.
    assert "─" not in copied and "▌" not in copied


@pytest.mark.asyncio
async def test_copy_takes_the_last_message_not_the_first() -> None:
    """The LAST message is the one at the bottom of the frame — append order,
    which is also the order a replayed history arrives in."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the first answer")
        await _stream(pilot, app, "the second answer")
        await _submit(pilot, app, "/copy")
        copied = app._clipboard

    assert copied == "the second answer"


# -- the receipt --------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_receipt_is_the_shared_toast_in_the_shared_format() -> None:
    """One clipboard write, one vocabulary. A bespoke message here would make
    the receipt evidence about WHICH gesture the user used — the drift
    ``on_editor_copied`` exists to prevent.

    The count is ``splitlines()`` of the payload, which is the SOURCE's line
    count and not the frame's row count: the two differ the moment a paragraph
    wraps, and the number a user checks their paste against is the former.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/copy")
        message = app.query_one(Toast).message

    assert message == f"copied {len(ANSWER.splitlines())} lines"


@pytest.mark.asyncio
async def test_a_single_line_answer_is_reported_in_characters() -> None:
    """The unit follows the SHAPE of what was taken, exactly as a drag does —
    `_put_on_clipboard` owns that rule and this command inherits it rather than
    carrying a second copy of it."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "short answer")
        await _submit(pilot, app, "/copy")
        message = app.query_one(Toast).message

    assert message == "copied 12 characters"


@pytest.mark.asyncio
async def test_the_receipt_appears_once() -> None:
    """One gesture, one card. A second toast would mean a second clipboard
    write, which is the drift the shared helper prevents — and the count is
    read from the toast's generation because two identically worded cards are
    indistinguishable by text (the trap ``Toast.generation`` documents)."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        toast = app.query_one(Toast)
        before = toast.generation
        await _submit(pilot, app, "/copy")
        after = toast.generation

    assert after - before == 1, "the copy raised more than one card"


# -- the echo policy ----------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_writes_no_user_row() -> None:
    """The receipt names what landed on the clipboard, which is strictly more
    than the typed word — so a user row above it would restate a row that says
    less. Pinned in ``ECHO_POLICY``; asserted here through the real submit
    path, which is what actually writes (or does not write) the row."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/copy")
        rows = _user_rows(app)

    assert rows == [], rows


def test_the_registry_entry_takes_no_argument() -> None:
    """``/copy me`` and ``/copy <n>`` are deliberately not built, so the entry
    must not advertise an argument list a handler would discard — the rule
    ``/provider`` is held to."""
    from local_operator.tui.autocomplete import ArgumentMode

    entry = next(command for command in SLASH_COMMANDS if command.name == "copy")
    assert entry.arguments is ArgumentMode.NONE
    assert entry.echo is False
    assert entry.consumes_prompt is False


# -- the refusals -------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_agent_message_yet_says_so_rather_than_no_opping() -> None:
    """``_put_on_clipboard`` is silent on empty, which is right for a
    zero-width drag and wrong for a typed command: an unexplained no-op reads
    as broken. No crash, no write, and a notice that names the reason."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/copy")
        notices = _notices(app)
        clipboard = app._clipboard
        toast = app.query_one(Toast).message

    assert clipboard == "", "nothing was copied, so nothing may reach the clipboard"
    assert toast == "", "a refusal must not raise the success receipt"
    assert any("nothing to copy" in text for text in notices), notices


@pytest.mark.asyncio
async def test_mid_stream_copies_the_last_settled_message_not_the_partial() -> None:
    """The failure mode Codex's "unavailable mid-task" rule exists to avoid: a
    clipboard holding a half-streamed answer that then grows, which the user
    cannot tell from a complete one.

    This copies the previous SETTLED message instead of refusing. That keeps the
    guarantee (never a partial) without the cost — a user who asks for the last
    answer while the next turn is running gets the last answer.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the settled answer")
        # A turn in flight: a block mounted and streaming, never finalized.
        await _stream(pilot, app, "a partial answer still arri", finish=False)
        assert app._streaming_block is not None, "the fixture must leave a live block"
        assert app._turn_is_live(), "the app must consider this turn live"
        await _submit(pilot, app, "/copy")
        copied = app._clipboard

    assert copied == "the settled answer"
    assert "still arri" not in copied


@pytest.mark.asyncio
async def test_mid_stream_with_nothing_settled_refuses_and_says_why() -> None:
    """The only genuinely empty case during a turn. It must not tell the user
    to press esc: stopping the turn does not produce the message they asked
    for, and ``_live_turn_refuse_copy``'s wording ("esc first") is written for
    ``/update``/``/reload``, which really cannot proceed until the turn ends."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the first answer, still arriving", finish=False)
        await _submit(pilot, app, "/copy")
        notices = _notices(app)
        clipboard = app._clipboard

    assert clipboard == ""
    assert any("nothing to copy yet" in text for text in notices), notices
    assert not any("esc first" in text for text in notices), notices


# -- the chord ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ctrl_o_copies_with_the_composer_focused() -> None:
    """WHERE THE USER ACTUALLY PRESSES IT, and the reason this is a test rather
    than an assumption.

    ``Screen.BINDINGS`` sits BETWEEN the focused widget and the App in
    ``_binding_chain``, and this app's composer consumes several chords in
    ``Editor._on_key`` before app level is reached — that layering is why
    Textual's stock ``ctrl+c`` copy had to be removed from ``TranscriptScreen``
    to get the interrupt back. So "``ctrl+o`` is unbound" is not sufficient
    evidence that ``ctrl+o`` ARRIVES; only driving it is.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        assert app.focused is editor, "the fixture must press the key from the composer"
        await pilot.press("ctrl+o")
        await pilot.pause()
        copied = app._clipboard
        composer = editor.text

    assert copied == ANSWER
    # The chord must not also type: a key that copies AND inserts a character
    # is worse than one that does neither, because the draft is now wrong.
    assert composer == "", composer


@pytest.mark.asyncio
async def test_ctrl_o_copies_with_the_composer_unfocused() -> None:
    """The other half of the binding's reach: the key is on the App, so it must
    work whatever holds focus — including nothing."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        app.set_focus(None)
        await pilot.pause()
        await pilot.press("ctrl+o")
        await pilot.pause()
        copied = app._clipboard

    assert copied == ANSWER


@pytest.mark.asyncio
async def test_ctrl_o_and_the_typed_command_answer_identically_when_empty() -> None:
    """One handler behind both, including the refusal. The notices are the half
    a second implementation of the chord would most easily get wrong, because
    the happy path would still look right."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await pilot.press("ctrl+o")
        await pilot.pause()
        from_chord = _notices(app)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/copy")
        from_command = _notices(app)

    assert from_chord == from_command
    assert from_chord, "the fixture proved nothing if neither path said anything"
