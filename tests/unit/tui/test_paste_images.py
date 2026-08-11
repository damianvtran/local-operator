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

import pytest
from PIL import Image
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import TextArea
from textual.widgets.text_area import Selection

from local_operator.harness.types import ImageContent
from local_operator.tui.widgets.editor import (
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
