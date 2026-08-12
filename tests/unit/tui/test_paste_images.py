"""Pasting an image into the composer.

The mechanism, because it is not the obvious one: Textual's ``Paste`` event
carries TEXT only, so an image never arrives here as bytes. What arrives is a
PATH — Ghostty writes a clipboard image to ``$TMPDIR/clipboard-<stamp>.png``
and bracketed-pastes the filename, and Finder's Cmd+C and a drag-and-drop land
the same way. So the composer hooks paste and loads the file, rather than
binding a key to read the system clipboard.

Every paste here is posted to the APP, not to the widget. ``App.on_event``
forwards a non-forwarded ``Paste`` to the focused widget (``app.py:4142``), so
posting straight to the widget delivers it twice — once directly and once via
the bubble — and every assertion about "inserted once" silently doubles. That
cost a real debugging detour; the control test below pins it.
"""

from __future__ import annotations

import base64
import io
import os
import signal
from pathlib import Path

import pytest
from PIL import Image
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import TextArea
from textual.widgets.text_area import Selection

from local_operator.harness.types import ImageContent
from local_operator.tui.widgets.editor import (
    IMAGE_MARKER,
    MAX_ATTACHMENT_BYTES,
    Editor,
    EditorSubmitted,
    _pasted_paths,
)


def _png(path, width: int = 1568, height: int = 200) -> str:
    Image.new("RGB", (width, height), (30, 30, 40)).save(path)
    return str(path)


def _escaped(path: str) -> str:
    """A path the way a terminal hands it over — spaces backslash-escaped."""
    return path.replace(" ", "\\ ")


class Host(App[None]):
    def compose(self) -> ComposeResult:
        yield Editor()


class BareHost(App[None]):
    def compose(self) -> ComposeResult:
        yield TextArea()


async def _paste(app: App[None], pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    await pilot.pause()
    await pilot.pause()


# -- the parser ---------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "see screenshot.png for details",
        "check src/foo.py and screenshot.png",
        "https://example.com/a.png",
        "hello",
        "",
        "   ",
        "a paste with 'unbalanced quotes",
    ],
)
def test_ordinary_text_is_never_read_as_a_path_list(text: str) -> None:
    """The expensive failure mode is a false positive: prose swallowed and
    replaced by a marker. Requiring EVERY segment to start with a separator is
    what keeps "see screenshot.png" text — one of its two segments has no `/`.
    """
    assert _pasted_paths(text) == []


def test_a_pasted_essay_is_not_shlex_parsed(tmp_path) -> None:
    """This runs on the keystroke that pasted, so a long paste must bail on a
    length check rather than tokenising the whole thing first."""
    assert _pasted_paths("/tmp/a.png " * 1000) == []


def test_terminal_quoting_of_a_spaced_filename_is_understood(tmp_path) -> None:
    """macOS screenshots are named ``Screenshot 2026-08-11 at 4.48.41 PM.png``.

    Terminals escape or quote that, and hand-rolled unescaping is how one path
    with spaces becomes four paths that do not exist — so ``shlex`` does it,
    since it is the grammar they are quoting for.
    """
    spaced = tmp_path / "Screenshot 2026-08-11 at 4.48.41 PM.png"
    spaced.touch()
    assert _pasted_paths(_escaped(str(spaced))) == [str(spaced)]
    assert _pasted_paths(f"'{spaced}'") == [str(spaced)]
    assert _pasted_paths(f'"{spaced}"') == [str(spaced)]


# -- the control --------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_plain_paste_inserts_exactly_once() -> None:
    """The control for every other test in this file.

    Textual invokes each ``_on_paste`` up the MRO, so an override that also
    calls ``super()._on_paste`` inserts twice — which is what the composer did
    first, and what this pins shut. Compared against a bare ``TextArea`` so the
    expected count comes from Textual rather than from this file's opinion.
    """
    bare = BareHost()
    async with bare.run_test() as pilot:
        widget = bare.query_one(TextArea)
        widget.focus()
        await pilot.pause()
        await _paste(bare, pilot, "XY")
        expected = widget.text

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "XY")
        assert editor.text == expected == "XY"
        assert editor.referenced_images() == []


# -- attaching ----------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_pasted_image_becomes_an_attachment_and_a_marker(tmp_path) -> None:
    """The reported gap: the path went in as literal text and the model got
    nothing it could look at."""
    path = _png(tmp_path / "clipboard-2026-08-11-164841-45DD7A9B.png")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        assert editor.text == "[Image #1, 1568x200] "
        assert len(editor.referenced_images()) == 1
        image = editor.referenced_images()[0]
        assert image.mime_type == "image/png"
        # The real bytes, not the path: this is what reaches the provider.
        assert base64.b64decode(image.data)[:8] == b"\x89PNG\r\n\x1a\n"
        assert Image.open(io.BytesIO(base64.b64decode(image.data))).size == (1568, 200)


@pytest.mark.asyncio
async def test_markers_number_in_the_order_they_were_attached(tmp_path) -> None:
    """``[Image #N]`` is positional — the Nth marker is the Nth attachment —
    because nothing looks an image up by name."""
    first = _png(tmp_path / "one.png", 10, 20)
    second = _png(tmp_path / "two.png", 30, 40)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        await _paste(app, pilot, second)

        assert editor.text == "[Image #1, 10x20] [Image #2, 30x40] "
        sizes = [
            Image.open(io.BytesIO(base64.b64decode(image.data))).size
            for image in editor.referenced_images()
        ]
        assert sizes == [(10, 20), (30, 40)]


@pytest.mark.asyncio
async def test_a_multi_file_drop_attaches_all_of_them(tmp_path) -> None:
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b b.png", 30, 40)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, f"{first} {_escaped(second)}")
        assert editor.text == "[Image #1, 10x20] [Image #2, 30x40] "
        assert len(editor.referenced_images()) == 2


# -- refusing -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_mixed_drop_attaches_nothing_and_pastes_the_paths(tmp_path) -> None:
    """All-or-nothing across one paste.

    Attaching two of three files and pasting the third as text is a result the
    user cannot see and cannot correct — the composer would look like it had
    worked. Falling back to plain text for the whole paste is visible.
    """
    image = _png(tmp_path / "a.png")
    other = tmp_path / "notes.txt"
    other.write_text("hi")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, f"{image} {other}")
        assert editor.referenced_images() == []
        assert editor.text == f"{image} {other}"


@pytest.mark.asyncio
async def test_a_path_that_is_not_really_an_image_is_pasted_as_text(tmp_path) -> None:
    """Typed by CONTENT. A `.png` holding HTML must not reach a provider as an
    image — which is a 400 mid-turn, not a local error."""
    liar = tmp_path / "screenshot.png"
    liar.write_bytes(b"<!doctype html><html>not a png</html>")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(liar))
        assert editor.referenced_images() == []
        assert editor.text == str(liar)


@pytest.mark.asyncio
async def test_a_missing_path_is_pasted_as_text(tmp_path) -> None:
    """A stale clipboard path must leave the user something they can read,
    not an error dialog on a keystroke."""
    missing = str(tmp_path / "gone.png")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, missing)
        assert editor.referenced_images() == []
        assert editor.text == missing


@pytest.mark.asyncio
async def test_an_oversized_image_is_refused_here_not_by_the_provider(tmp_path) -> None:
    """Refused in the composer, where it is one visible path the user can act
    on, rather than as a provider 400 halfway through a turn — which lands in
    the history and, before the session learned to recover, stayed there."""
    big = tmp_path / "huge.png"
    Image.new("RGB", (64, 64)).save(big)
    big.write_bytes(big.read_bytes() + b"\x00" * (MAX_ATTACHMENT_BYTES + 1))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(big))
        assert editor.referenced_images() == []
        assert editor.text == str(big)


# -- submitting ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_submitting_carries_the_attachments_and_then_forgets_them(tmp_path) -> None:
    """Attachments belong to the text that referenced them.

    A draft cleared without dropping them would send the previous prompt's
    screenshots along with the next, unrelated question.
    """
    path = _png(tmp_path / "a.png", 8, 9)
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("what is this")
        await pilot.press("enter")
        await pilot.pause()

        assert len(submitted) == 1
        assert submitted[0].text == "[Image #1, 8x9] what is this"
        assert len(submitted[0].images) == 1
        # Cleared for the NEXT prompt, and the sent list is a copy of it.
        assert editor.referenced_images() == []
        assert editor.text == ""


@pytest.mark.asyncio
async def test_deleting_a_marker_drops_its_attachment(tmp_path) -> None:
    """The marker in the text is the AUTHORITY on what gets sent.

    Built the other way first — marker as a label over a list that always
    sent — and the repo owner reported it immediately: paste three, delete two,
    and all three still went. Deleting a reference is the user changing their
    mind, and sending the image anyway is both surprising and expensive.

    Resolving from the text is also what avoids the thing that argued for the
    old design: nothing renumbers under the cursor, because numbers are keys
    rather than positions and a deleted #2 simply leaves a gap.
    """
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)
    third = _png(tmp_path / "c.png", 50, 60)
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for path in (first, second, third):
            await _paste(app, pilot, path)
        assert editor.text == "[Image #1, 10x20] [Image #2, 30x40] [Image #3, 50x60] "

        # Keep the middle one only, exactly as a user editing the line would.
        editor.text = "look at [Image #2, 30x40] please"
        await pilot.press("enter")
        await pilot.pause()

        assert len(submitted[0].images) == 1
        kept = base64.b64decode(submitted[0].images[0].data)
        assert Image.open(io.BytesIO(kept)).size == (30, 40), "the wrong image survived"


@pytest.mark.asyncio
async def test_attachments_are_sent_in_the_order_the_text_cites_them(tmp_path) -> None:
    """Order comes from the text too, so moving a marker moves its image."""
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        await _paste(app, pilot, second)
        editor.text = "[Image #2, 30x40] before [Image #1, 10x20]"
        await pilot.press("enter")
        await pilot.pause()

    sizes = [
        Image.open(io.BytesIO(base64.b64decode(image.data))).size for image in submitted[0].images
    ]
    assert sizes == [(30, 40), (10, 20)]


@pytest.mark.asyncio
async def test_a_marker_naming_nothing_sends_nothing(tmp_path) -> None:
    """Typed by hand, or left over from a previous prompt's numbering. The
    number NAMES an attachment; text that names nothing carries nothing."""
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("[Image #7, 99x99] what is this")
        await pilot.press("enter")
        await pilot.pause()

    assert submitted[0].images == []


# -- reaching the model -------------------------------------------------------
@pytest.mark.asyncio
async def test_a_pasted_image_reaches_the_session_with_the_prompt(tmp_path) -> None:
    """The whole point of the feature, asserted at the seam that matters.

    Everything upstream can work and the model still see nothing: the composer
    held the bytes, and until this call carried them the prompt arrived as text
    with a marker in it referring to an attachment that was never sent.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    path = _png(tmp_path / "a.png", 12, 34)
    sent: list[tuple[str, list[ImageContent]]] = []

    class Recording(FakeSession):
        async def prompt(self, text, images=None):  # type: ignore[override]
            sent.append((text, list(images or [])))

    app = OperatorApp(lambda: _factory(Recording()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("what is this")
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if sent:
                break

    assert sent, "the prompt never reached the session"
    text, images = sent[0]
    assert text == "[Image #1, 12x34] what is this"
    assert len(images) == 1
    assert images[0].mime_type == "image/png"
    assert base64.b64decode(images[0].data)[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_a_screenshot_with_no_words_is_a_prompt_in_itself(tmp_path) -> None:
    """ "What is wrong with this?" is a real question to ask with a screenshot
    and nothing else, so a paste followed straight by Enter must send.

    It submits because the MARKER is text. An attachment cannot outlive its
    marker now — clearing the buffer clears the attachment — so "an image with
    an empty prompt" is not a reachable state, and the app does not carry a
    branch pretending otherwise.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    path = _png(tmp_path / "a.png", 11, 22)
    sent: list[tuple[str, list[ImageContent]]] = []

    class Recording(FakeSession):
        async def prompt(self, text, images=None):  # type: ignore[override]
            sent.append((text, list(images or [])))

    app = OperatorApp(lambda: _factory(Recording()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if sent:
                break

    assert sent, "a screenshot-only prompt was swallowed"
    assert sent[0][0] == "[Image #1, 11x22]"
    assert len(sent[0][1]) == 1


@pytest.mark.asyncio
async def test_clearing_the_buffer_clears_the_attachments(tmp_path) -> None:
    """The invariant that makes the empty-text branch unnecessary: no marker,
    no attachment. Pinned so a future change cannot reintroduce a draft holding
    images the text does not mention."""
    path = _png(tmp_path / "a.png")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        assert editor.referenced_images()
        editor.text = ""
        assert editor.referenced_images() == []


# -- the marker as an atomic token --------------------------------------------
@pytest.mark.asyncio
async def test_backspace_at_the_end_of_a_marker_removes_the_whole_marker(tmp_path) -> None:
    """The reported bug: it removed the closing bracket and left
    ``[Image #1, 1568x20`` hanging — neither prose the user meant nor a
    reference anything can resolve, and the attachment silently orphaned."""
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        # Paste leaves a trailing space; step back onto the marker's `]`.
        await pilot.press("left")
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == " ", editor.text
        assert editor.referenced_images() == []


@pytest.mark.asyncio
async def test_backspace_inside_a_marker_removes_the_whole_marker(tmp_path) -> None:
    """A caret in the middle of ``[Image #2, 15|68x200]`` is not editing text
    anyone wrote — there is nothing meaningful to change one character of."""
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.selection = Selection.cursor((0, 8))  # inside `[Image #1, 10x20]`
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == " ", editor.text
        assert editor.referenced_images() == []


@pytest.mark.asyncio
async def test_delete_at_the_start_of_a_marker_removes_the_whole_marker(tmp_path) -> None:
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.selection = Selection.cursor((0, 0))
        await pilot.press("delete")
        await pilot.pause()

        assert editor.text == " ", editor.text
        assert editor.referenced_images() == []


@pytest.mark.asyncio
async def test_ordinary_backspace_is_untouched(tmp_path) -> None:
    """The control. Character deletion elsewhere on the line must behave
    exactly as before, or the atomic rule has eaten normal editing."""
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("hello")
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == "[Image #1, 10x20] hell"
        assert len(editor.referenced_images()) == 1


@pytest.mark.asyncio
async def test_a_selection_ending_on_a_marker_deletes_only_the_selection(tmp_path) -> None:
    """A real selection is the user's own range and must never be widened to a
    token boundary — that would delete text they did not highlight.

    The selection has to END on the marker's closing bracket to test anything.
    Written first against a range that merely sat NEXT to the marker, it passed
    with the guard deleted, because the atomic rule was never consulted.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        assert editor.text == "[Image #1, 10x20] "
        # `[Image #1, 10x20]` is columns 0..17; select the last seven cells of
        # it, so the caret sits exactly on the closing bracket.
        editor.selection = Selection((0, 10), (0, 17))
        await pilot.press("backspace")
        await pilot.pause()

        # Only the highlighted cells go. The marker is broken as a REFERENCE by
        # the user's own deliberate edit, which is theirs to make.
        assert editor.text == "[Image #1, ", editor.text
        assert editor.referenced_images() == []


@pytest.mark.asyncio
async def test_deleting_a_marker_releases_the_image_bytes(tmp_path) -> None:
    """Not just unreferenced — GONE.

    ``referenced_images`` reads the text, so a retained entry would never be
    sent and the leak would be invisible. It is still a leak: base64 for a
    screenshot is megabytes, and a long editing session pasting and deleting
    would hold every one of them for the life of the draft.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        assert editor._attachments, "nothing was attached"
        await pilot.press("left")
        await pilot.press("backspace")
        await pilot.pause()

        assert editor._attachments == {}, "the image bytes outlived the marker"


# -- the marker and its image must never disagree -----------------------------
@pytest.mark.asyncio
async def test_recalling_a_previous_prompt_does_not_borrow_this_draft_s_image(tmp_path) -> None:
    """Review round 17, P1, reproduced with real keystrokes only.

    Marker numbers restart at #1 on every submit, so a recalled prompt's
    ``[Image #1]`` and the live draft's ``[Image #1]`` are different images
    with the same name. Recalling the text while leaving the attachments alone
    therefore sent the SECOND screenshot under the FIRST one's label — silent,
    and the model answers about a picture the user never attached here.
    """
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        await pilot.press("enter")
        await pilot.pause()

        await _paste(app, pilot, second)
        await pilot.press("up")
        await pilot.pause()
        assert editor.text.rstrip() == "[Image #1, 10x20]", editor.text
        await pilot.press("enter")
        await pilot.pause()

    # The recalled prompt carries NO image: the first one went with the message
    # that was sent, and the second belongs to the draft that was set aside.
    assert submitted[1].images == [], "a recalled marker resolved to the wrong image"


@pytest.mark.asyncio
async def test_coming_back_from_history_restores_the_draft_s_own_images(tmp_path) -> None:
    """Down-arrow past the newest entry returns the unsent draft, which is the
    same message it was — so its attachments come back with it."""
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            pass

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        await pilot.press("enter")
        await pilot.pause()

        await _paste(app, pilot, second)
        draft = editor.text
        await pilot.press("up")
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()

        assert editor.text == draft
        sizes = [
            Image.open(io.BytesIO(base64.b64decode(image.data))).size
            for image in editor.referenced_images()
        ]
        assert sizes == [(30, 40)], "the draft came back without its attachment"


@pytest.mark.asyncio
async def test_a_pasted_path_with_a_nul_byte_is_a_text_paste_not_a_crash(tmp_path) -> None:
    """Every other malformed path degrades to an ordinary text paste. A NUL in
    the name raised `ValueError` (not an `OSError`), which escaped onto the
    keystroke and surfaced as Textual's error screen (review round 17)."""
    hostile = f"{tmp_path}/a\x00.png"
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, hostile)

        assert editor.text == hostile
        assert editor.referenced_images() == []


@pytest.fixture
def no_full_reads(monkeypatch):
    """Record every path whose CONTENTS the composer pulled into memory.

    Instruments ``pathlib.Path.read_bytes`` specifically. An earlier version
    counted bytes through ``builtins.open`` and was vacuous: ``read_bytes``
    goes through ``io.open``, so removing the size gate changed nothing the
    test could see, and both mutations passed.
    """
    read: list[str] = []
    real = Path.read_bytes

    def recording(self):  # noqa: ANN001 - patching a bound method
        read.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", recording)
    return read


@pytest.mark.asyncio
async def test_an_oversized_file_is_refused_before_it_is_read(tmp_path, no_full_reads) -> None:
    """The cap has to bound the READ, not report on it afterwards.

    It ran on ``len(data)``, so a 601 MB file behind a valid PNG header took
    peak RSS to 618 MB before the cap fired — allocated synchronously on the
    keystroke that pasted it (review round 17). Asserted on whether the file
    was opened at all, which is the property, rather than on timing or memory,
    which are flaky.
    """
    big = tmp_path / "huge.png"
    Image.new("RGB", (64, 64)).save(big)
    big.write_bytes(big.read_bytes() + b"\x00" * (MAX_ATTACHMENT_BYTES + 1))
    no_full_reads.clear()  # the fixture setup above is not the composer

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(big))

        assert editor.referenced_images() == []
        assert editor.text == str(big)
    assert no_full_reads == [], f"read a file it was going to refuse: {no_full_reads}"


@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="POSIX alarm only")
@pytest.mark.asyncio
async def test_a_fifo_is_refused_without_being_opened(tmp_path, no_full_reads) -> None:
    """``open()`` on a FIFO blocks until a writer appears, and this runs inline
    on the event loop — so a named pipe was a hung APP, not a failed paste.

    Two mechanisms, because either alone is a bad test:

    - Asserted as "never opened", not "returned quickly". Timing is flaky, and
      the property under test is that the file is never touched.
    - Guarded by ``SIGALRM``, because the failure this defends against is an
      INFINITE BLOCK inside a synchronous ``open`` on the event loop. Without
      the alarm, deleting the guard does not fail this test — it hangs the
      whole suite, which reports nothing and cannot be mutation-verified.
      Confirmed: the mutation run reported ``HUNG >120s`` until this was added.
    """
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)

    def _blocked(signum, frame):  # noqa: ANN001 - signal handler signature
        raise AssertionError("the composer blocked on a FIFO instead of refusing it")

    previous = signal.signal(signal.SIGALRM, _blocked)
    signal.alarm(10)
    try:
        app = Host()
        async with app.run_test() as pilot:
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            await _paste(app, pilot, str(fifo))

            assert editor.referenced_images() == []
            assert editor.text == str(fifo)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    assert no_full_reads == [], f"opened a FIFO on the event loop: {no_full_reads}"


@pytest.mark.asyncio
async def test_the_aside_gives_the_draft_back_with_its_attachments(tmp_path) -> None:
    """Review round 17, P1. The aside borrows the composer by clearing it, and
    `clear_content` drops attachments by design — so the draft came back with
    its markers resolving to nothing and Enter sent the words alone. Worse, an
    image pasted INSIDE the aside took number 1, so the restored marker
    resolved to the aside's image instead.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    main_image = _png(tmp_path / "a.png", 10, 20)
    aside_image = _png(tmp_path / "b.png", 30, 40)

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, main_image)
        editor.insert("explain this")

        app._open_aside()
        await pilot.pause()
        # An image pasted while the aside owns the composer numbers from #1 too.
        await _paste(app, pilot, aside_image)
        app._close_aside()
        await pilot.pause()

        assert editor.text == "[Image #1, 10x20] explain this"
        sizes = [
            Image.open(io.BytesIO(base64.b64decode(image.data))).size
            for image in editor.referenced_images()
        ]
        assert sizes == [(10, 20)], "the draft came back citing the wrong image"


# -- what the chip claims, and what the clipboard takes ------------------------
@pytest.mark.asyncio
async def test_a_marker_with_no_image_behind_it_is_not_painted_as_a_chip(tmp_path) -> None:
    """Design round 16, D1. The chip is a CLAIM that an image is attached.

    Painted from the text pattern alone it drew a full chip for a marker typed
    by hand, and for one brought back by undo after its attachment had been
    dropped — visually identical to a real attachment, submitting nothing.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(80, 10)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        real = editor._marker_cells(0)
        assert real, "a real attachment should paint"

        editor.text = "[Image #9, 900x900] typed by hand"
        await pilot.pause()
        assert editor._marker_cells(0) == [], "an unresolvable marker painted as a chip"
        assert editor.referenced_images() == []


@pytest.mark.asyncio
async def test_the_attachment_receipt_never_reaches_the_clipboard(tmp_path) -> None:
    """Design round 16, D3. `↑ 1 image attached` is the app talking.

    Copying a prompt must paste the prompt. The receipt says something the user
    did not write, so a drag over the block must not carry it into whatever
    they were quoting into — the same rule `copy_gutter` already applies to the
    left rule, one dimension over.
    """
    from textual.selection import Selection as ScreenSelection

    from local_operator.tui.widgets.transcript import UserBlock

    block = UserBlock("look at this", attachments=1)
    rows = block._rows(60)
    assert rows[-1] == "↑ 1 image attached", rows

    whole = ScreenSelection(None, None)
    copied = block.get_selection(whole)
    assert copied is not None
    assert "image attached" not in copied[0], copied[0]
    assert "look at this" in copied[0]


@pytest.mark.asyncio
async def test_a_paste_after_recall_cannot_steal_a_recalled_marker_s_number(tmp_path) -> None:
    """Review round 18, P1, reproduced with keystrokes only.

    Marker numbers restart at #1 per prompt AND every history entry numbers
    from #1, so recalling one and then pasting issued a number already standing
    in the recalled text: two chips, one image, and the recalled marker
    advertising 10x20 while resolving to the freshly pasted 30x40.

    The counter is derived from the BUFFER at every seam that replaces the text
    wholesale, which is the only source that knows what numbers are on screen.
    """
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            submitted.append(message)

    app = Capturing()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        await _paste(app, pilot, second)

        numbers = [int(m.group(1)) for m in IMAGE_MARKER.finditer(editor.text)]
        assert len(numbers) == len(set(numbers)), f"a number was issued twice: {editor.text}"
        # The recalled marker resolves to nothing; only the new paste is sent.
        images = editor.referenced_images()
        assert len(images) == 1
        assert Image.open(io.BytesIO(base64.b64decode(images[0].data))).size == (30, 40)


@pytest.mark.asyncio
async def test_an_unresolvable_marker_does_not_shift_the_others(tmp_path) -> None:
    """Review round 18, P1. Restores carry IDENTITY, not position.

    `referenced_images()` skips markers that do not resolve; re-keying that
    list positionally against every marker in the text therefore shifted each
    image one marker left as soon as one marker was dead — so every marker sent
    a picture it did not name. Reproduced through the aside, which is an
    ordinary keystroke away, via delete-then-undo to strand a marker.
    """
    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    paths = [
        _png(tmp_path / "a.png", 10, 20),
        _png(tmp_path / "b.png", 30, 40),
        _png(tmp_path / "c.png", 50, 60),
    ]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.25)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for path in paths:
            await _paste(app, pilot, path)

        # Strand marker #1: its text survives, its attachment does not.
        editor.selection = Selection.cursor((0, 17))
        await pilot.press("backspace")
        await pilot.pause()
        editor.text = "[Image #1, 10x20] [Image #2, 30x40] [Image #3, 50x60]"
        await pilot.pause()
        assert 1 not in editor._attachments, "the fixture stopped stranding #1"

        app._open_aside()
        await pilot.pause()
        app._close_aside()
        await pilot.pause()

        # Per MARKER, not as a list: a positional re-key shifts every image one
        # marker left, which leaves the resolved LIST identical and only the
        # bindings wrong. Asserting the list cannot see that (it did not).
        def cited(text: str) -> list[tuple[int, int]]:
            editor.text = text
            return [
                Image.open(io.BytesIO(base64.b64decode(image.data))).size
                for image in editor.referenced_images()
            ]

        assert cited("[Image #3, 50x60]") == [(50, 60)], "#3 names the wrong picture"
        assert cited("[Image #2, 30x40]") == [(30, 40)], "#2 names the wrong picture"
        assert cited("[Image #1, 10x20]") == [], "#1 was resurrected by the restore"


@pytest.mark.asyncio
async def test_clearing_the_buffer_releases_the_stashed_draft_images(tmp_path) -> None:
    """Review round 18, P3. `_draft_attachments` outlived the draft it belonged
    to and pinned a multi-megabyte screenshot for the session. Not reachable as
    a wrong send, but the repo holds itself to this one seam over."""
    path = _png(tmp_path / "a.png")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        await pilot.press("enter")
        await pilot.pause()
        await _paste(app, pilot, path)
        await pilot.press("up")
        await pilot.pause()
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert editor._attachments == {}
        assert editor._draft_attachments == {}, "a stashed draft image outlived its draft"


@pytest.mark.asyncio
async def test_a_marker_arriving_as_text_does_not_get_its_number_reissued(tmp_path) -> None:
    """Design round 18 D4 / review round 19. Markers arrive as TEXT.

    Drag-copying a prompt out of the transcript and pasting it back to re-run
    it is a gesture this branch built, and the copy carries `[Image #1, ...]`
    verbatim. No replacement seam sees that text, so issuance was the one
    consumer of the counter that could hand out a number already on screen: the
    chip landed on last turn's marker and the real attachment rendered as prose
    advertising the wrong dimensions.
    """
    path = _png(tmp_path / "a.png", 30, 40)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        # Exactly what the transcript copy puts on the clipboard.
        await _paste(app, pilot, "[Image #1, 10x20] rerun this")
        await pilot.pause()
        assert editor.referenced_images() == [], "pasted text attached something"
        await _paste(app, pilot, path)

        numbers = [int(m.group(1)) for m in IMAGE_MARKER.finditer(editor.text)]
        assert numbers == [1, 2], f"the number was re-issued: {editor.text}"
        images = editor.referenced_images()
        assert len(images) == 1
        assert Image.open(io.BytesIO(base64.b64decode(images[0].data))).size == (30, 40)


@pytest.mark.asyncio
async def test_a_paste_during_a_compaction_is_queued_with_its_image(tmp_path) -> None:
    """Review round 19, P1, introduced by the previous commit.

    `Editor._submit` clears the buffer synchronously right after posting, and
    Textual delivers on a later tick — so the handler that held the prompt for
    a compaction re-read an already-empty widget and queued the text alone.
    The app announced "sends when compaction finishes" and then sent it without
    the screenshot, which the code's own comment calls worse than not queueing.
    """
    path = _png(tmp_path / "a.png", 30, 40)
    submitted: list[EditorSubmitted] = []

    class Capturing(Host):
        def on_editor_submitted(self, message: EditorSubmitted) -> None:
            # Read on the DELIVERY tick, which is where the app reads it and
            # where the composer has already cleared itself.
            submitted.append(message)

    app = Capturing()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("hold this")
        await pilot.press("enter")
        await pilot.pause()

        assert editor.referenced_images() == [], "the fixture did not clear the composer"

    message = submitted[0]
    assert list(message.attachments) == [1], "the hand-back map was empty at delivery"
    assert len(message.images) == 1


@pytest.mark.asyncio
async def test_deleting_one_citation_keeps_the_image_the_other_still_names(tmp_path) -> None:
    """Design round 18, D6. "No longer cites" is a question about the BUFFER.

    A number can be written twice. Dropping the image because one citation went
    detached the picture the surviving citation still named — and after D4 the
    surviving one is the chipped one, so the frame showed an attachment that
    was no longer attached.
    """
    path = _png(tmp_path / "a.png", 30, 40)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.strip()
        # On a SECOND LINE deliberately: with both citations on one line the
        # test cannot tell the shipped whole-buffer predicate from a
        # line-scoped one, and the line-scoped version is a real defect
        # that left the whole TUI suite green (review round 20).
        editor.text = f"{marker} and\nsee {marker} twice"
        await pilot.pause()

        # The FIRST citation is the chipped one, so it is the atomic token.
        # Deleting it leaves the duplicate behind, still naming #1.
        editor.selection = Selection.cursor((0, editor.text.index("]") + 1))
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text.count("[Image #1") == 1, "the atomic delete did not fire"
        assert len(editor.referenced_images()) == 1, "the surviving citation lost its image"


@pytest.mark.asyncio
async def test_a_marker_painted_as_prose_is_edited_as_prose(tmp_path) -> None:
    """Design round 18, D7. Gesture follows paint.

    A marker the painter renders as prose is text as far as the frame is
    concerned, so backspace must take one character of it rather than
    swallowing all nineteen as an atomic token.
    """
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.text = "[Image #9, 900x900]"
        editor.selection = Selection.cursor((0, len(editor.text)))
        await pilot.pause()
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == "[Image #9, 900x900", "an unresolvable marker deleted atomically"

        # And from INSIDE it. That is a different branch of `_marker_span` -
        # the one that makes a LIVE marker atomic - and prose has to lose it
        # too, or the paint and the gesture disagree at that caret position.
        # Reordering the filter past this branch left the suite green.
        editor.text = "[Image #9, 900x900]"
        editor.selection = Selection.cursor((0, 12))
        await pilot.pause()
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == "[Image #9, 00x900]", "a caret inside prose ate the whole token"


@pytest.mark.asyncio
async def test_text_pasted_above_the_marker_cannot_steal_its_chip(tmp_path) -> None:
    """Design round 19, D4 residual. Uniqueness at issuance is not enough.

    The counter stops a number being ISSUED twice, but a copy of the app's
    marker can still arrive as text afterwards - paste an image, press Home,
    paste a prompt drag-copied out of the transcript - and picking the chip by
    document order then hands it to whatever landed first. The impostor got the
    chip, the dimensions affordance named last turn's screenshot, and through
    the atomic-token gate it took the editing behaviour with it.

    `cite()` prefers the marker text the app actually wrote, so position in the
    draft cannot decide which citation is the app's.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        mine = editor.text.strip()
        editor.selection = Selection.cursor((0, 0))
        await _paste(app, pilot, "[Image #1, 1568x200] rerun this")
        await pilot.pause()

        assert editor.text.startswith("[Image #1, 1568x200]"), "the fixture did not paste above"
        painted = {
            column
            for y in range(3)
            for start, end, _ in editor._marker_cells(y)
            for column in range(start - editor.gutter_width, end - editor.gutter_width)
        }
        impostor = set(range(0, len("[Image #1, 1568x200]")))
        real = editor.text.index(mine)
        assert painted & set(range(real, real + len(mine))), "the app's own marker lost its chip"
        assert not painted & impostor, "the pasted text stole the chip"


@pytest.mark.asyncio
async def test_the_app_s_own_marker_stays_the_atomic_token(tmp_path) -> None:
    """Design round 19, D4k. A stolen chip stole editability with it.

    `_marker_span` gates on the same citation the painter uses, so when the
    impostor won, backspace at the end of the app's OWN marker took one
    character and left `[Image #1, 10x20` - verbatim the fragment the
    docstring calls the reported bug.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        mine = editor.text.strip()
        editor.selection = Selection.cursor((0, 0))
        await _paste(app, pilot, "[Image #1, 1568x200] ")
        await pilot.pause()

        end = editor.text.index(mine) + len(mine)
        editor.selection = Selection.cursor((0, end))
        await pilot.press("backspace")
        await pilot.pause()

        assert mine not in editor.text, "the app's marker was not atomic"
        assert "[Image #1, 10x2" not in editor.text, f"left a fragment: {editor.text!r}"
        assert editor.referenced_images() == [], "the image outlived its marker"
