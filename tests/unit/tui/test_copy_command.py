"""``/copy``: what reaches the clipboard, and when.

The command answers a gesture users already had for a DRAG (release copies,
``on_text_selected``) but not for the common case — "give me that whole answer"
— which previously meant dragging a multi-screen message from its first row to
its last.

There is deliberately NO keyboard chord. One shipped (``ctrl+o``) and was
withdrawn with the picker: a global chord that opens a modal is a different
gesture from one that copies in place, and the keymap surface was not worth
spending twice. Everything here therefore drives the TYPED command, which is
the only way in.

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
from local_operator.tui.widgets.assistant import AssistantBlock
from local_operator.tui.widgets.copy_picker import CopyPickerScreen
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


async def _copy_latest(pilot, app: OperatorApp) -> None:
    """Run ``/copy`` and take the whole most-recent message out of the picker.

    The gesture these tests were written for — "give me that answer" — now costs
    one more keystroke: `/copy` opens a chooser whose cursor starts on the most
    recent message, and Enter on that row copies it whole. Routing every payload
    assertion through here keeps them asserting the PAYLOAD rather than silently
    becoming assertions about the picker's default cursor; the cursor itself is
    pinned once, by ``test_the_picker_opens_on_the_most_recent_message``.

    Driven through the real screen rather than by calling the dismiss callback,
    because the push and the callback are the pair that decides whether anything
    reaches the clipboard at all.
    """
    await _submit(pilot, app, "/copy")
    if isinstance(app.screen, CopyPickerScreen):
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()


def _picker(app: OperatorApp) -> CopyPickerScreen | None:
    """The open picker, or ``None`` — never an ``isinstance`` assert at the call
    site, so a test that expects NO picker reads as plainly as one that does."""
    screen = app.screen
    return screen if isinstance(screen, CopyPickerScreen) else None


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
        await _copy_latest(pilot, app)
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
        await _copy_latest(pilot, app)
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
        await _copy_latest(pilot, app)
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
        await _copy_latest(pilot, app)
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
        await _copy_latest(pilot, app)
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
        await _copy_latest(pilot, app)
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


# -- the write itself ---------------------------------------------------------
#
# Everything above reads ``app._clipboard``, the attribute Textual sets BEFORE
# it emits anything. That attribute is true of a copy that never left the
# process: `copy_to_clipboard` assigns it and then returns early when
# ``_driver`` is None. So the assertions above cannot distinguish "the answer
# reached the user's clipboard" from "the app remembered the answer", and OSC 52
# is the whole reason this survives ssh and a multiplexer. These tests read the
# escape sequence off the DRIVER instead, which is the byte the terminal sees.


def _tap_driver(app: OperatorApp) -> list[str]:
    """Record every raw driver write, so OSC 52 can be counted and decoded."""
    sink: list[str] = []
    driver = app._driver
    assert driver is not None, "no driver: the pilot would prove nothing about the write"
    original = driver.write

    def write(data: str) -> None:
        sink.append(data)
        return original(data)

    driver.write = write  # type: ignore[method-assign]
    return sink


def _osc52_payloads(sink: list[str]) -> list[str]:
    """The clipboard payloads decoded back out of the OSC 52 sequences."""
    import base64
    import re

    pattern = re.compile(r"\x1b]52;c;([A-Za-z0-9+/=]*)\a")
    return [
        base64.b64decode(match.group(1)).decode("utf-8")
        for chunk in sink
        for match in pattern.finditer(chunk)
    ]


@pytest.mark.asyncio
async def test_the_osc52_write_happens_once_and_carries_the_source() -> None:
    """The escape sequence the terminal actually receives, decoded back.

    Asserted on the DRIVER rather than on ``app._clipboard`` because that
    attribute is set before the write and survives its absence — it is equally
    true of a copy that never left the process.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _copy_latest(pilot, app)
        payloads = _osc52_payloads(sink)

    assert payloads == [ANSWER], "one write, byte-identical to the block's source"


@pytest.mark.asyncio
async def test_a_refusal_writes_no_escape_sequence_at_all() -> None:
    """The silent half of the refusal. A notice plus an empty OSC 52 write
    would still clobber whatever the user had on their clipboard — the copy
    they took before asking for one that does not exist."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _submit(pilot, app, "/copy")
        payloads = _osc52_payloads(sink)

    assert payloads == [], "nothing to copy must not overwrite the real clipboard"


@pytest.mark.asyncio
async def test_each_copy_writes_again_rather_than_deduplicating() -> None:
    """A second ``/copy`` is a second write. Users re-copy after clobbering the
    clipboard elsewhere, so an implementation that skipped the repeat because
    the payload was unchanged would leave them with the other application's
    text and a toast claiming otherwise."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, "the answer")
        await _copy_latest(pilot, app)
        await _copy_latest(pilot, app)
        payloads = _osc52_payloads(sink)

    assert payloads == ["the answer", "the answer"]


# -- payload shapes the renderer would destroy --------------------------------


@pytest.mark.asyncio
async def test_every_markdown_construct_survives_verbatim() -> None:
    """The frame is a lossy projection of the source, and each construct here
    is lost in a DIFFERENT way: headings lose their ``#``, list markers are
    repainted as bullets, table pipes become box-drawing, the blockquote bar
    replaces ``>``. ``ANSWER`` covers bold and a fence; this covers the rest,
    so a regression that reached for the flattened rows fails on whichever
    construct it mangles first rather than only on the two already pinned.
    """
    source = (
        "# Heading\n\n"
        "Body with **bold**, *emphasis* and `inline code`.\n\n"
        "- first item\n"
        "- second item\n\n"
        "1. numbered one\n"
        "2. numbered two\n\n"
        "> a blockquote line\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "A [link](https://example.com).\n"
    )
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, source)
        await _copy_latest(pilot, app)
        copied = app._clipboard

    assert copied == source
    for marker in ("# Heading", "- first item", "1. numbered one", "> a blockquote line"):
        assert marker in copied, marker
    assert "| a | b |" in copied
    assert "[link](https://example.com)" in copied
    # None of the furniture Rich draws for those constructs may be in the paste.
    for glyph in "\u2500\u2502\u250c\u2510\u2514\u2518\u258c\u256d\u256f":
        assert glyph not in copied, glyph


@pytest.mark.asyncio
async def test_a_message_that_is_only_a_code_fence_keeps_its_fences() -> None:
    """The worst case for the frame: ``IslandCodeBlock`` renders a bare
    ``Syntax``, so the fence lines are not merely restyled, they are DROPPED.
    A message that is nothing but a fence would therefore paste as naked code
    the receiving markdown renders as prose."""
    source = "```python\nprint('hi')\n```\n"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, source)
        await _copy_latest(pilot, app)
        copied = app._clipboard

    assert copied == source
    assert copied.startswith("```python")
    assert copied.rstrip("\n").endswith("```")


@pytest.mark.asyncio
async def test_trailing_blank_lines_reach_the_clipboard_unchanged() -> None:
    """No stripping. The walk uses ``strip()`` only to DECIDE whether a block
    counts as a message; a version that also copied the stripped value would
    silently reshape a payload whose trailing structure the user may rely on
    when pasting into a document."""
    source = "answer body\n\n\n   \n"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, source)
        await _copy_latest(pilot, app)
        copied = app._clipboard
        message = app.query_one(Toast).message

    assert copied == source, "the payload must not be stripped on the way out"
    assert message == f"copied {len(source.splitlines())} lines"


@pytest.mark.asyncio
async def test_a_very_long_answer_is_copied_whole_in_one_write() -> None:
    """Multi-screen answers are the case the command exists for — dragging one
    from its first row to its last is the gesture it replaces. Asserted on the
    escape sequence so a chunked or truncated write fails here."""
    source = "\n\n".join(f"Paragraph {index} with **bold** text." for index in range(500))
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, source)
        await _copy_latest(pilot, app)
        payloads = _osc52_payloads(sink)
        message = app.query_one(Toast).message

    assert payloads == [source]
    assert message == f"copied {len(source.splitlines())} lines"


@pytest.mark.asyncio
async def test_unicode_and_tabs_survive_the_base64_round_trip() -> None:
    """OSC 52 is base64 of UTF-8, so a payload that is not pure ASCII exercises
    an encode/decode the ASCII fixtures never reach."""
    source = "emoji \U0001f389 and CJK \u65e5\u672c\u8a9e and \u00e9 plus a tab\there\n"
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, source)
        await _copy_latest(pilot, app)
        payloads = _osc52_payloads(sink)

    assert payloads == [source]


# -- which block the walk lands on --------------------------------------------


@pytest.mark.asyncio
async def test_a_user_row_below_the_answer_does_not_become_the_payload() -> None:
    """The most recent BLOCK is routinely not the most recent agent message:
    the user speaks last every time they ask something. Copying their own
    prompt back to them is the failure this pins."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the agent answer")
        await _submit(pilot, app, "a plain user prompt")
        await _copy_latest(pilot, app)
        copied = app._clipboard

    assert copied == "the agent answer"
    assert "user prompt" not in copied


@pytest.mark.asyncio
async def test_a_tool_card_below_the_answer_does_not_stop_the_walk() -> None:
    """A tool card is the most recent block for the whole of any turn that
    ends in tool work, which is most of them. It is not an ``AssistantBlock``,
    so the walk must pass over it rather than treat it as the message."""
    from local_operator.tui.widgets.tool_card import ToolCard

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the agent answer before the tool")
        app._append_block(ToolCard("call-1", "bash", {"command": "ls -la"}))
        await pilot.pause()
        await _copy_latest(pilot, app)
        copied = app._clipboard

    assert copied == "the agent answer before the tool"


@pytest.mark.asyncio
async def test_an_empty_block_above_the_answer_does_not_stop_the_walk() -> None:
    """An abort before the first delta leaves a mounted block that is not a
    message. Stopping on it would report "nothing to copy" with a real answer
    sitting two rows up — the case the walk's ``strip()`` guard is for."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the real answer")
        await _stream(pilot, app, "   \n\n  \n")
        await _copy_latest(pilot, app)
        copied = app._clipboard

    assert copied == "the real answer"


@pytest.mark.asyncio
async def test_copy_after_clear_declines_rather_than_copying_a_wiped_answer() -> None:
    """``/clear`` empties the SCREEN, and the transcript is what this command
    reads. So the answer that was on it is gone for copying purposes, and the
    refusal must be the not-found one — the user is looking at an empty
    surface, so a receipt claiming a copy would describe nothing they can see.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, "the answer before the clear")
        await _submit(pilot, app, "/clear")
        await _submit(pilot, app, "/copy")
        payloads = _osc52_payloads(sink)
        notices = _notices(app)

    assert payloads == [], "a cleared transcript has nothing to put on the clipboard"
    assert any("nothing to copy" in text for text in notices), notices
    assert not any("still coming" in text for text in notices), notices


# -- the mid-stream contract, held across a growing partial -------------------


@pytest.mark.asyncio
async def test_the_partial_never_reaches_the_clipboard_as_it_grows() -> None:
    """The failure mode the guard exists for, exercised over TIME rather than
    at a single instant: the coder's fixture copies once mid-stream, which a
    naive implementation could pass by luck of when the delta landed. Here the
    partial grows between two copies and the answer must not move."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, "the settled answer")
        await _stream(pilot, app, "the partial so far", finish=False)
        assert app._turn_is_live(), "the fixture must actually stage a live turn"
        await _copy_latest(pilot, app)
        app.post_message(AssistantDelta("the partial so far, now longer"))
        await pilot.pause()
        await _copy_latest(pilot, app)
        payloads = _osc52_payloads(sink)

    assert payloads == ["the settled answer", "the settled answer"]
    assert not any("partial" in payload for payload in payloads)


# -- the follower ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_follower_copies_locally_instead_of_routing_to_the_owner() -> None:
    """The scope claim in ``_FRONTEND_LOCAL_SLASHES``, driven end to end.

    The existing coverage for that entry is a set-membership assertion, which
    proves the name was added and not that the routing seam honours it. Both
    halves of this command are local — the transcript is painted here and the
    OSC 52 goes out THIS terminal — so a routed ``/copy`` would put the answer
    on a clipboard belonging to a host nobody is sitting at, and would emit no
    escape sequence to the user who typed it.
    """
    from local_operator.session.frontend_state import (
        FrontendSessionState,
        _slash_capabilities,
    )

    routed: list[tuple[str, str]] = []

    class RoutedSession(FakeSession):
        frontend_state: FrontendSessionState

        async def route_shared_slash(self, command: str, args: str, images=()):  # noqa: ANN001
            routed.append((command, args))
            return "routed"

    session = RoutedSession()
    # The owner's capability list is built by the production helper rather than
    # hand-written, so this cannot drift from what a real owner advertises.
    session.frontend_state = FrontendSessionState(
        session_id=session.session_id,
        epoch="owner",
        slash_capabilities=_slash_capabilities(),
    )

    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(80, 24)) as pilot:
        sink = _tap_driver(app)
        await _boot(pilot, app)
        await _stream(pilot, app, "the answer on the follower")
        await _copy_latest(pilot, app)
        payloads = _osc52_payloads(sink)

    assert routed == [], "a frontend-local command must not reach the owner"
    assert payloads == ["the answer on the follower"], "and it must write to THIS terminal"


# -- an aborted answer is not a complete one ---------------------------------
#
# Review round 1, MAJOR-1. `is_finalized()` is the FINALIZED-BLOCK protocol's
# "this block is immutable" and says nothing about whether the model finished
# talking: `on_assistant_message_end` with empty authoritative text — the abort
# path — calls `finalize_text()` on whatever had streamed, so a TRUNCATED answer
# was indistinguishable from a settled one and `/copy` handed the user a half
# sentence that reads as a short complete reply.
#
# The fix marks the block at the source (`AssistantBlock.mark_truncated`, set on
# that branch) rather than pattern-matching the text downstream. These pin BOTH
# directions, because a flag that is never cleared would silently condemn every
# clean answer and the happy-path tests above would still pass.


async def _abort(pilot, app: OperatorApp, partial: str) -> None:
    """Stream ``partial``, then end the turn with no authoritative text.

    That empty end IS the abort signal on this path: the controller falls back
    to its own buffer only when the text is ``None``, so a user interrupt and a
    provider that stops mid-sentence both arrive here as ``""``.
    """
    app.post_message(AssistantMessageStart())
    await pilot.pause()
    app.post_message(AssistantDelta(partial))
    await pilot.pause()
    app.post_message(AssistantMessageEnd(""))
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_an_aborted_answer_is_copied_but_announced_as_cut_off() -> None:
    """The user gets the text they were looking at AND is told it is partial.

    Copying it rather than skipping back to the previous complete message is
    deliberate: someone who stops a long answer and copies usually wants the
    part they stopped, which is what is on screen. Substituting an older message
    would hand them a different document with nothing saying so — the caller can
    warn about a truncated payload, but it cannot warn about a silent swap.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "an older complete answer")
        await _abort(pilot, app, "a partial answer still arri")
        before = len(_notices(app))
        await _copy_latest(pilot, app)
        copied = app._clipboard
        new_notices = _notices(app)[before:]

    assert copied == "a partial answer still arri", "the user's own screen is what they meant"
    assert any("cut off" in text for text in new_notices), new_notices


@pytest.mark.asyncio
async def test_the_abort_marks_the_block_rather_than_the_command_guessing() -> None:
    """The flag is set where the abort is KNOWN, not inferred later.

    Asserted on the widget because that is the contract the fix rests on: a
    downstream heuristic (short text, no trailing period) would be a guess that
    misfires on a genuinely terse answer, and `/copy` is only one of the
    consumers that needs "did the model finish".
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _abort(pilot, app, "cut off here")
        blocks = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, AssistantBlock)
        ]
        finalized = blocks[-1].is_finalized()
        truncated = blocks[-1].is_truncated()

    # BOTH, and that is the point: the block is frozen (immutable) AND
    # incomplete. Conflating the two is what the defect was.
    assert finalized is True
    assert truncated is True


@pytest.mark.asyncio
async def test_a_completed_answer_is_never_flagged_as_cut_off() -> None:
    """The other direction. A flag that is set for everything is not a flag,
    and it would put a false caveat on every ordinary copy."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        blocks = [
            block
            for block in app.query_one(TranscriptView).blocks()
            if isinstance(block, AssistantBlock)
        ]
        truncated = blocks[-1].is_truncated()
        before = len(_notices(app))
        await _submit(pilot, app, "/copy")
        new_notices = _notices(app)[before:]

    assert truncated is False
    assert new_notices == [], new_notices


@pytest.mark.asyncio
async def test_a_completed_answer_after_an_aborted_one_copies_clean() -> None:
    """The recovery path a user actually walks: abort, ask again, copy.

    The second answer is a different block, so the first one's flag must not
    reach it — a per-block flag read off the wrong block would put a permanent
    caveat on the rest of the session.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _abort(pilot, app, "the abandoned attempt")
        await _stream(pilot, app, "the answer that finished")
        before = len(_notices(app))
        await _copy_latest(pilot, app)
        copied = app._clipboard
        new_notices = _notices(app)[before:]

    assert copied == "the answer that finished"
    assert new_notices == [], new_notices


# -- the picker: what /copy now opens -----------------------------------------
#
# The command changed shape here: it used to take the last message outright and
# now opens a chooser. These pin the WIRING — that a picker is pushed at all,
# what it is built from, what the callback does with the answer, and what a
# cancel does not do. What the picker DRAWS and how it navigates belongs to the
# screen's own tests; asserting frame content here would pin the same rows twice
# and make an innocent layout change fail in two files.


@pytest.mark.asyncio
async def test_copy_opens_the_picker_rather_than_copying_outright() -> None:
    """The shape change, stated once. Nothing may reach the clipboard until the
    user has chosen — a command that copied AND opened a chooser would put an
    unasked-for payload on the clipboard of anyone who cancelled."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        clipboard = app._clipboard
        toast = app.query_one(Toast).message

    assert picker is not None, "/copy must open the picker"
    assert clipboard == "", "nothing may be copied before the user chooses"
    assert toast == "", "and no receipt may be raised for a copy that has not happened"


@pytest.mark.asyncio
async def test_the_picker_opens_on_the_most_recent_message() -> None:
    """Where the cursor starts, pinned once and relied on by every payload test
    above (they press Enter on this row). The most recent answer is the one the
    reader is looking at, so it is the row the common gesture must land on."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the older answer")
        await _stream(pilot, app, "the newest answer")
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        target = picker.selected_target()

    assert target is not None
    assert target.content == "the newest answer"


@pytest.mark.asyncio
async def test_a_cancelled_picker_copies_nothing_and_says_nothing() -> None:
    """Esc is not an event. A notice on cancel would report a non-action, and a
    clipboard write would be the payload the user just declined."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        before = len(_notices(app))
        await _submit(pilot, app, "/copy")
        assert _picker(app) is not None, "the fixture must actually open the picker"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        still_open = _picker(app)
        clipboard = app._clipboard
        toast = app.query_one(Toast).message
        new_notices = _notices(app)[before:]

    assert still_open is None, "esc must dismiss the picker"
    assert clipboard == "", "a cancelled picker must not write"
    assert toast == ""
    assert new_notices == [], new_notices


@pytest.mark.asyncio
async def test_a_code_block_can_be_copied_out_of_a_message() -> None:
    """The reason the picker exists. Getting one fence out of an answer meant
    dragging it row by row; now it is a child node, and what lands is the fence
    BODY without its markers — the thing that pastes into a file."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, ANSWER)
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        await pilot.press("down")
        await pilot.pause()
        target = picker.selected_target()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        copied = app._clipboard

    assert target is not None and target.id == "msg:1:code:0"
    assert copied == "def f(x):\n    return x + 1"
    assert "```" not in copied, "the fence markers are frame, not payload"


@pytest.mark.asyncio
async def test_the_picker_is_a_snapshot_and_does_not_move_under_the_cursor() -> None:
    """A message settling while the picker is open must not re-rank the rows.

    New answers insert at the TOP of a most-recent-first list, so a live rebuild
    would shift every row down by one — including the one the user is already
    aiming at, which is how you copy the wrong thing. The tree is therefore
    built once, at push time; the cost is that the new answer is not listed
    until the picker is reopened, which this also pins so the trade-off is
    visible rather than assumed.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the answer that was there first")
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None
        rows_before = picker.render_lines_for_test()

        await _stream(pilot, app, "an answer that landed during the picker")
        rows_after = picker.render_lines_for_test()
        assert rows_before == rows_after, "the open picker re-ranked itself"

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        copied = app._clipboard

    assert copied == "the answer that was there first"
    assert "during the picker" not in copied


@pytest.mark.asyncio
async def test_the_new_message_is_listed_once_the_picker_is_reopened() -> None:
    """The other half of the snapshot: it is a snapshot per OPEN, not a cache.
    A tree built once per app would strand the user on stale rows forever."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the first answer")
        await _submit(pilot, app, "/copy")
        await pilot.press("escape")
        await pilot.pause()
        await _stream(pilot, app, "the second answer")
        await _copy_latest(pilot, app)
        copied = app._clipboard

    assert copied == "the second answer"


@pytest.mark.asyncio
async def test_mid_stream_lists_what_has_settled_instead_of_refusing() -> None:
    """Mid-stream opens the picker; it does not refuse.

    Refusing would regress behaviour this command already shipped, and the
    refusal it would have to borrow says "esc first" — an instruction that does
    not produce the message the user asked for, because stopping the turn does
    not settle it. The in-flight block is excluded by ``is_finalized`` rather
    than by any check here, so what is listed is exactly what is immutable.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(80, 24)) as pilot:
        await _boot(pilot, app)
        await _stream(pilot, app, "the settled answer")
        await _stream(pilot, app, "the partial still arriving", finish=False)
        assert app._turn_is_live(), "the fixture must actually stage a live turn"
        await _submit(pilot, app, "/copy")
        picker = _picker(app)
        assert picker is not None, "mid-stream must open the picker, not refuse"
        rows = picker.render_lines_for_test()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        copied = app._clipboard
        notices = _notices(app)

    assert copied == "the settled answer"
    assert not any("still arriving" in row for row in rows), "the in-flight block was listed"
    assert not any("esc first" in text for text in notices), notices
