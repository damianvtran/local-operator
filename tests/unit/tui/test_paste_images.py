"""Pasting an image into the composer.

The mechanism, because it is not the obvious one: Textual's ``Paste`` event
carries TEXT only, so an image never arrives here as bytes. This file covers
the route where a PATH arrives instead — a drag-and-drop on any terminal, and a
clipboard image under **cmux**, which watches the pasteboard, writes
``$TMPDIR/clipboard-<stamp>-<hash>.png`` and bracket-pastes that filename.

That is a cmux feature and not a terminal one, which is what made issue #372
invisible for so long: in Ghostty, Terminal.app or iTerm2 the same ``Cmd+V``
delivers an EMPTY paste and this route never fires. The empty-paste route, and
the clipboard read behind it, are covered in ``test_paste_clipboard.py``.

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
import threading
import time
from pathlib import Path

import pytest
from PIL import Image
from textual import events
from textual.app import App, ComposeResult
from textual.widgets import TextArea
from textual.widgets.text_area import Selection

from local_operator.clipboard import MAX_CLIPBOARD_READ_BYTES
from local_operator.harness.types import ImageContent
from local_operator.imaging import IMAGE_MAX_EDGE
from local_operator.tui.widgets import editor as editor_module
from local_operator.tui.widgets.editor import (
    IMAGE_MARKER,
    RESIZED_MARK,
    Editor,
    EditorSubmitted,
    _pasted_paths,
)


def _png(path, width: int = 1000, height: int = 200) -> str:
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


async def _settle(pilot, done, timeout: float = 10.0) -> None:
    """Pump the event loop until ``done()`` or a wall-clock deadline.

    A fixed turn count (`for _ in range(40)`) is a race: it passes in isolation
    and loses under full-suite load, which costs an afternoon every time
    someone reads the failure as a product bug. A deadline cannot silently
    become too small when the app grows one more boot step.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if done():
            return


async def _booted(app, pilot) -> None:
    """Wait for the session to EXIST before submitting anything.

    Not a nicety: `_submit_prompt` refuses outright when `_session is None`,
    appending "session is still starting…" and returning, so a prompt sent one
    tick early is never delivered and no amount of waiting afterwards recovers
    it. A fixed `pilot.pause(0.25)` here lost roughly one full-suite run in
    four and always looked like a product bug at the assert (review round 24
    measured the intermittency; this is why it was never a timing race).
    """
    await _settle(pilot, lambda: app._session is not None)
    assert app._session is not None, "the session never booted"


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

        assert editor.text == "[Image #1, 1000x200] "
        assert len(editor.referenced_images()) == 1
        image = editor.referenced_images()[0]
        assert image.mime_type == "image/png"
        # The real bytes, not the path: this is what reaches the provider.
        assert base64.b64decode(image.data)[:8] == b"\x89PNG\r\n\x1a\n"
        assert Image.open(io.BytesIO(base64.b64decode(image.data))).size == (1000, 200)


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
    # Over the INGEST ceiling, so it is refused at the stat gate. A file merely
    # over the ATTACHMENT cap is no longer refused here — it is bounded down
    # and attached, which is D12's fix and is covered in test_paste_clipboard.
    big = tmp_path / "huge.png"
    Image.new("RGB", (64, 64)).save(big)
    header = big.read_bytes()
    with big.open("wb") as handle:
        handle.write(header)
        handle.truncate(MAX_CLIPBOARD_READ_BYTES + 1)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(big))
        assert editor.referenced_images() == []
        assert editor.text == str(big)


# -- bounding what gets attached ----------------------------------------------
@pytest.mark.asyncio
async def test_a_pasted_screenshot_is_bounded_before_it_is_attached(tmp_path) -> None:
    """The session-wedging bug, reproduced at its source.

    A provider refuses an image over 2000 pixels on its long edge as soon as a
    request carries more than twenty images. The composer used to attach
    whatever the screen produced, so a 2206x266 paste sat harmlessly in the
    history until the twenty-first screenshot arrived and then wedged the
    session PERMANENTLY: the block is in the history, so every later request —
    including the ``/compact`` that is supposed to be the escape hatch — re-sent
    it and earned the same 400.

    2206x266 is the real screenshot that did it, not a round number.
    """
    path = _png(tmp_path / "wide.png", 2206, 266)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        (image,) = editor.referenced_images()
        width, height = Image.open(io.BytesIO(base64.b64decode(image.data))).size
        assert max(width, height) <= IMAGE_MAX_EDGE
        # Below the strict many-image ceiling with room to spare, which is the
        # property that actually keeps the session alive.
        assert max(width, height) < 2000
        # Aspect ratio survives the resize; a squashed screenshot is unreadable.
        assert width / height == pytest.approx(2206 / 266, rel=0.01)


@pytest.mark.asyncio
async def test_the_marker_reports_what_was_attached_not_what_is_on_disk(tmp_path) -> None:
    """The marker's dimensions are the user's only receipt for an attachment.

    Once the paste path started resizing, carrying the SOURCE dimensions into
    the marker would print ``[Image #1, 2560x1440]`` beside a 1024x576
    attachment — a receipt for something that was never sent.
    """
    path = _png(tmp_path / "retina.png", 2560, 1440)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        (image,) = editor.referenced_images()
        delivered = Image.open(io.BytesIO(base64.b64decode(image.data))).size
        assert editor.text == f"[Image #1, {delivered[0]}x{delivered[1]}{RESIZED_MARK}] "


@pytest.mark.asyncio
async def test_a_resized_marker_says_so_and_stays_one_atomic_token(tmp_path) -> None:
    """Design round 1, D1.

    Every 16:9 screenshot bounds to the same 1024x576, so three different
    captures pasted together would read as three identical markers and the label
    would stop doing the job it exists for — telling one paste from another. The
    mark restores that and explains why the number is not the size pasted.

    It must not break the marker's grammar: ``IMAGE_MARKER`` still has to match
    the whole thing, or the chip stops painting and the token stops deleting
    atomically.
    """
    wide = _png(tmp_path / "wide.png", 2560, 1440)
    tall = _png(tmp_path / "tall.png", 1440, 2560)
    small = _png(tmp_path / "small.png", 800, 600)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, wide)
        await _paste(app, pilot, tall)
        await _paste(app, pilot, small)

        # Resized ones are marked; the in-bounds one is not, because nothing
        # about it changed and the mark would be a lie.
        assert editor.text == (
            f"[Image #1, 1024x576{RESIZED_MARK}] "
            f"[Image #2, 576x1024{RESIZED_MARK}] "
            "[Image #3, 800x600] "
        )
        # The grammar still holds: three whole markers, numbered in order.
        assert [m.group(1) for m in IMAGE_MARKER.finditer(editor.text)] == ["1", "2", "3"]
        assert len(editor.referenced_images()) == 3


@pytest.mark.asyncio
async def test_an_image_already_inside_the_bounds_is_attached_verbatim(tmp_path) -> None:
    """No re-encode can improve an image the model sees at its original size,
    and PNG round-tripping routinely makes files BIGGER. The common case — a
    small crop, an already-bounded frame — must stay lossless and cheap."""
    path = _png(tmp_path / "small.png", 800, 600)
    source = Path(path).read_bytes()
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        (image,) = editor.referenced_images()
        assert base64.b64decode(image.data) == source
        assert editor.text == "[Image #1, 800x600] "


@pytest.mark.asyncio
async def test_a_decompression_bomb_is_pasted_as_text_not_attached(tmp_path) -> None:
    """A bomb is small on disk by construction, so the 4 MB cap cannot see it
    coming — only the dimensions can. The paste stays TEXT so the user can see
    what happened; a silently dropped attachment is the shape nobody notices
    until the model answers about nothing."""
    path = _png(tmp_path / "bomb.png", 9000, 9000)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        assert editor.referenced_images() == []
        assert editor.text == path


@pytest.mark.asyncio
async def test_the_decode_runs_off_the_event_loop_thread(tmp_path, monkeypatch) -> None:
    """The bound runs on the keystroke that pasted it, and a 20 MP screenshot
    measures ~315 ms on an M3 Max — a visibly frozen composer if it runs inline.

    Asserted as a THREAD IDENTITY rather than as elapsed time or loop progress.
    Both of those pass trivially on the blocking implementation: a synchronous
    handler that never awaits also never lets the loop fall behind, so the
    obvious 'did the loop keep ticking' test is green both ways and pins
    nothing. Where the work runs is the actual invariant, and it is the thing
    that breaks if someone later drops the ``to_thread``.
    """
    path = _png(tmp_path / "big.png", 2600, 1400)
    seen: list[str] = []
    real = editor_module.bound_image_for_model

    def spy(data, info):
        seen.append(threading.current_thread().name)
        return real(data, info)

    monkeypatch.setattr(editor_module, "bound_image_for_model", spy)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        assert editor.referenced_images(), "the image was not attached at all"
        assert seen, "the paste never bounded the image"
        assert seen[0] != threading.main_thread().name


@pytest.mark.asyncio
async def test_a_rotation_alone_is_not_marked_as_a_downscale(tmp_path) -> None:
    """Review round 2, F8.

    A portrait phone photo inside the bounds is EXIF-rotated on the way in, so
    its dimensions change (700x900 from 900x700) while not one pixel is lost.
    The mark is a claim about FIDELITY, so comparing the ``WxH`` strings made
    every such photo assert a shrink that never happened.

    The fixture sits inside IMAGE_INGEST_MAX_EDGE on both edges deliberately: a
    photo the ingest bound would also RESIZE earns the mark legitimately, and
    the test would then pass for the wrong reason.
    """
    path = str(tmp_path / "portrait.jpg")
    image = Image.new("RGB", (900, 700), (30, 30, 40))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif)

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        assert editor.referenced_images(), "the photo was not attached"
        assert RESIZED_MARK not in editor.text
        # The rotated dimensions are still reported, because that IS what was
        # attached — only the shrink claim was wrong.
        assert editor.text == "[Image #1, 700x900] "


@pytest.mark.asyncio
async def test_an_unlabelled_bound_falls_back_to_the_source_dimensions(
    tmp_path, monkeypatch
) -> None:
    """The marker's label is a convenience and must never cost an attachment.

    ``_bounded_dimensions`` reads the delivered size back with a header sniff.
    If that ever fails to answer — a format whose header we cannot parse, or a
    future encoder — the marker degrades to the source dimensions rather than
    dropping the image or printing a number it invented. Pinned because the
    fallback is otherwise unreachable and would rot silently (review round 1,
    F6).

    The stub answers for the SOURCE bytes and only fails for the bound's
    output. ``sniff_image`` has two callers on this path and they ask different
    questions: ``_attach_image_bytes`` asks "is this an image at all", which is
    the gate that admits a clipboard payload and must keep working, while
    ``_bounded_dimensions`` asks "what size did the bound deliver", which is
    the label this test degrades. A blanket ``lambda: None`` disables both and
    tests nothing, since no attachment survives to carry a label.
    """
    path = _png(tmp_path / "shot.png", 2560, 1440)
    source = Path(path).read_bytes()
    real_sniff = editor_module.sniff_image
    monkeypatch.setattr(
        editor_module,
        "sniff_image",
        lambda payload: real_sniff(payload) if payload == source else None,
    )
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        (image,) = editor.referenced_images()
        # Still bounded, still attached — only the LABEL degraded.
        assert max(Image.open(io.BytesIO(base64.b64decode(image.data))).size) <= IMAGE_MAX_EDGE
        assert editor.text == "[Image #1, 2560x1440] "


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
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("what is this")
        await pilot.press("enter")
        # A DEADLINE, not a turn budget. Forty event-loop turns is a race
        # against `OperatorApp` boot: under full-suite load it lost 2 runs in 8
        # and never once in isolation, which reads as a product bug and is not
        # (review round 24 measured it on both this commit and its parent).
        await _settle(pilot, lambda: bool(sent))

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
        await _booted(app, pilot)
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        await pilot.press("enter")
        await _settle(pilot, lambda: bool(sent))

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
    ``[Image #1, 1000x20`` hanging — neither prose the user meant nor a
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
    # Sized against the INGEST ceiling, which is what the stat gate bounds:
    # the attachment cap applies after `bound_image_for_model`, because a file
    # that resizes under it is attachable and refusing it early was D12
    # (a valid screenshot refused by the file route and blamed on its format).
    # Sparse, so this costs bytes on disk rather than 64 MB of them.
    big = tmp_path / "huge.png"
    Image.new("RGB", (64, 64)).save(big)
    header = big.read_bytes()
    with big.open("wb") as handle:
        handle.write(header)
        handle.truncate(MAX_CLIPBOARD_READ_BYTES + 1)
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
        await _booted(app, pilot)
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
        await _booted(app, pilot)
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
        await _paste(app, pilot, "[Image #1, 1000x200] rerun this")
        await pilot.pause()

        assert editor.text.startswith("[Image #1, 1000x200]"), "the fixture did not paste above"
        painted = {
            column
            for y in range(3)
            for start, end, _ in editor._marker_cells(y)
            for column in range(start - editor.gutter_width, end - editor.gutter_width)
        }
        impostor = set(range(0, len("[Image #1, 1000x200]")))
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
        await _paste(app, pilot, "[Image #1, 1000x200] ")
        await pilot.pause()

        end = editor.text.index(mine) + len(mine)
        editor.selection = Selection.cursor((0, end))
        await pilot.press("backspace")
        await pilot.pause()

        assert mine not in editor.text, "the app's marker was not atomic"
        assert "[Image #1, 10x2" not in editor.text, f"left a fragment: {editor.text!r}"
        # The image goes with it. Prose the user pasted must not become a live
        # chipped attachment because something ELSE was deleted - reorder the
        # two keystrokes and that is a typed marker resurrecting an image,
        # which is the same D1 violation one door over (review round 23).
        assert editor.referenced_images() == [], "a foreign citation inherited the image"
        assert editor._first_citation_columns(0) == set(), "a foreign citation was chipped"


@pytest.mark.asyncio
async def test_deleting_a_foreign_citation_leaves_the_live_image_alone(tmp_path) -> None:
    """Design round 20, D11 - a regression the previous commit introduced.

    When `cite()` falls back it can chip a copy of another prompt's marker, and
    the atomic gate follows the chip - so one backspace deleted that stale
    reference and took THIS turn's screenshot with it, while the marker naming
    it was still standing in the buffer. The parent commit recovered from this
    state; guarding the pop on the marker text alone did not.

    The release is a union of two rules, and this is the case the second one
    must NOT fire on: the token deleted was not the app's marker, and the app's
    marker still resolves, so the image stays.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.selection = Selection.cursor((0, 0))
        await _paste(app, pilot, "look at [Image #1, 1000x200] ")
        await pilot.pause()

        # Edit the tail of the app's OWN marker, so cite() falls back and the
        # stale copy takes the chip.
        own = editor.text.rindex("[Image #1")
        editor.selection = Selection.cursor((0, own + len("[Image #1, 1")))
        await pilot.press("9")
        await pilot.pause()
        stale_end = editor.text.index("]") + 1
        editor.selection = Selection.cursor((0, stale_end))
        await pilot.press("backspace")
        await pilot.pause()

        assert "1000x200" not in editor.text, "the stale reference was not deleted"
        assert len(editor.referenced_images()) == 1, "deleting stale text dropped the live image"


@pytest.mark.asyncio
async def test_a_half_deleted_stale_marker_cannot_swallow_a_live_one(tmp_path) -> None:
    """Design round 20, D12. A marker's tail cannot contain another marker.

    The stale copy is prose, so backspace takes one character - and the first
    character is its closing bracket. With `[` allowed in the tail, the
    unterminated `[Image #1, 1000x200` then matched all the way through the
    LIVE marker's bracket as one token, whose start is nowhere `cite()` points:
    the chip vanished for ten keystrokes of an ordinary cleanup while the image
    stayed attached and on the wire, and the live marker left the atomic set.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.selection = Selection.cursor((0, 0))
        await _paste(app, pilot, "look at [Image #1, 1000x200] ")
        await pilot.pause()

        # Sweep the whole cleanup. The chip must never disappear while the
        # image is still being sent - that is the commit's headline invariant.
        stale = "[Image #1, 1000x200]"
        editor.selection = Selection.cursor((0, editor.text.index(stale) + len(stale)))
        await pilot.pause()
        for keystroke in range(1, len(stale) + 1):
            # The caret walks left on its own; re-seeking `]` would find the
            # LIVE marker's bracket once the stale one is gone.
            await pilot.press("backspace")
            await pilot.pause()
            painted = [span for y in range(3) for span in editor._marker_cells(y)]
            sent = len(editor.referenced_images())
            assert bool(painted) == bool(sent), (
                f"after {keystroke} backspaces: chipped {bool(painted)}, sent {sent} "
                f"- {editor.text!r}"
            )


@pytest.mark.asyncio
async def test_editing_the_dimensions_does_not_orphan_the_draft(tmp_path) -> None:
    """`cite()` falls back to the number on purpose: the tail is a label for the
    user, matched loosely so retyping it cannot strand the attachment. Keying
    only on the exact text would reverse that, and nothing else notices -
    reducing `cite()` to `find` left all 1285 TUI tests green (review round 21).
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.text = "[Image #1, 10x21] edited by hand"
        await pilot.pause()

        assert len(editor.referenced_images()) == 1, "an edited tail orphaned the attachment"
        assert editor._first_citation_columns(0) == {0}, "an edited tail lost its chip"


@pytest.mark.asyncio
async def test_the_chip_lands_on_the_row_and_column_that_carries_it(tmp_path) -> None:
    """`cite()` answers in whole-buffer offsets and the painter wants columns,
    so `_first_citation_columns` converts. Every other marker test sits on
    line 0, where that conversion is the identity - so both terms of the
    arithmetic were unpinned (review round 21).
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        mine = editor.text.strip()
        editor.text = f"line one\nline two\nabc {mine} here"
        await pilot.pause()

        assert editor._first_citation_columns(0) == set(), "a chip leaked onto an earlier row"
        assert editor._first_citation_columns(1) == set(), "a chip leaked onto an earlier row"
        assert editor._first_citation_columns(2) == {4}, "the chip landed on the wrong column"


@pytest.mark.asyncio
async def test_an_edited_marker_deleted_cannot_be_resurrected_by_typing(tmp_path) -> None:
    """Review round 22 - a regression the previous commit introduced.

    Guarding the pop on the deleted TOKEN matching the app's marker meant a
    tail edit first made the token differ, so the pop was skipped: the marker
    went, the attachment stayed in the map, and hand-typing any `[Image #1]`
    afterwards brought the picture back - chipped, sent, receipted. That is
    precisely what design round 16's D1 exists to forbid, since the user typed
    that marker and nothing was ever attached to it.

    The release now asks the one question the chip and the send ask: can the
    buffer still cite this attachment?
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        # Type inside the tail - insertion is not gated, so this is reachable.
        editor.selection = Selection.cursor((0, len("[Image #1, 10x2")))
        await pilot.press("9")
        await pilot.pause()
        assert "10x290" in editor.text, "the fixture did not edit the tail"

        editor.selection = Selection.cursor((0, editor.text.index("]") + 1))
        await pilot.press("backspace")
        await pilot.pause()
        assert "[Image #" not in editor.text, "the marker was not deleted"
        assert editor._attachments == {}, "the attachment leaked past its marker"

        editor.insert("[Image #1]")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker resurrected the image"
        assert editor._first_citation_columns(0) == set(), "a typed marker was chipped"


@pytest.mark.asyncio
async def test_a_crlf_buffer_puts_the_chip_on_the_marker(tmp_path) -> None:
    """Review round 22. `self.text` joins with the DOCUMENT's separator.

    The offset-to-column conversion assumed one character per line break, so a
    CRLF buffer - which a paste can carry in - shifted the chip one cell per
    preceding line. Two lines above the marker put it two cells off, painting
    the prose beside it and leaving the marker's own opening bracket bare.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        mine = editor.text.strip()
        saved = editor.attachments()
        editor.load_text(f"line one\r\nline two\r\nabc {mine} here")
        editor.adopt_attachments(saved)
        await pilot.pause()

        assert editor.document.newline == "\r\n", "the fixture did not produce a CRLF buffer"
        assert editor._first_citation_columns(2) == {4}, "the chip is off the marker"
        assert editor._first_citation_columns(0) == set()
        assert editor._first_citation_columns(1) == set()


@pytest.mark.asyncio
async def test_a_typed_marker_cannot_inherit_an_image_by_deleting_the_real_one(tmp_path) -> None:
    """Review round 23, P1 - the reorder that exposed an over-wide release.

    `cite()` falls back to any citation of the number, so using it alone as the
    release rule handed the image, the chip and the send to whatever else
    mentioned that number when the app's own marker was deleted. Typed first
    and deleted second, that is a bare `[Image #1]` the user wrote acquiring a
    picture - the same resurrection the atomic path forbids, two keystrokes
    apart.
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
        editor.insert("[Image #1] ")
        await pilot.pause()
        assert editor.text.startswith("[Image #1] "), "the fixture did not type a bare marker"

        end = editor.text.index(mine) + len(mine)
        editor.selection = Selection.cursor((0, end))
        await pilot.press("backspace")
        await pilot.pause()

        assert mine not in editor.text, "the app's marker was not deleted"
        assert editor.referenced_images() == [], "the typed marker inherited the image"
        assert editor._first_citation_columns(0) == set(), "the typed marker was chipped"


@pytest.mark.asyncio
async def test_deleting_a_selected_marker_releases_it_too(tmp_path) -> None:
    """Review round 23, P3. The atomic path stands aside for a real selection,
    so a selected marker was removed without the release ever being asked -
    leaving the image held and resurrectable by typing `[Image #1]`. Same D1
    violation as the atomic path had, through the door the selection guard
    deliberately leaves open.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        mine = editor.text.strip()

        editor.selection = Selection((0, 0), (0, len(mine)))
        await pilot.press("backspace")
        await pilot.pause()

        assert mine not in editor.text, "the selection was not deleted"
        assert editor._attachments == {}, "a selected marker left its image held"

        editor.insert("[Image #1]")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker resurrected the image"


@pytest.mark.asyncio
async def test_ctrl_w_at_the_caret_a_paste_leaves_takes_the_whole_marker(tmp_path) -> None:
    """Design round 22, D13 - the third door into round 16's D1.

    A paste inserts `[Image #1, 10x20] ` WITH a trailing space, so the caret it
    leaves is one column past the marker and the atomic check correctly finds
    nothing there. Textual's word-delete then ate `] ` and stopped: the hanging
    fragment this mechanism exists to prevent, plus an orphaned attachment that
    any later `[Image #1]` revived, chipped and sent and receipted.

    Pre-existing rather than a regression, and it never reaches
    `_delete_marker`, so the release guard alone could not close it.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        assert editor.text.endswith(" "), "the fixture lost the trailing space"

        await pilot.press("ctrl+w")
        await pilot.pause()

        assert "[Image #" not in editor.text, f"left a fragment: {editor.text!r}"
        assert editor._attachments == {}, "ctrl+w orphaned the attachment"

        editor.insert("what is in [Image #1] please")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker revived the image"
        assert editor._first_citation_columns(0) == set(), "a typed marker was chipped"


@pytest.mark.asyncio
async def test_ctrl_w_still_takes_a_word_when_no_marker_is_behind_it(tmp_path) -> None:
    """The whitespace-crossing rule is for ctrl+w only, and only when a marker
    is actually what it would reach. Ordinary prose must delete a word."""
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("hello world ")
        await pilot.pause()
        await pilot.press("ctrl+w")
        await pilot.pause()
        assert editor.text == "hello ", f"ctrl+w did not take one word: {editor.text!r}"


@pytest.mark.asyncio
async def test_backspace_past_a_marker_s_space_still_takes_the_space(tmp_path) -> None:
    """Backspace does NOT cross whitespace. At the caret a paste leaves, the
    character before really is a space, and eating the whole marker instead
    would be a surprise - only a word-delete has said it will cross a run."""
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.rstrip()

        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text == marker, "backspace ate more than the trailing space"
        assert len(editor.referenced_images()) == 1, "backspace released a cited image"


@pytest.mark.asyncio
async def test_a_delete_elsewhere_cannot_destroy_a_marker_being_repaired(tmp_path) -> None:
    """Review round 24, P0 - a regression the previous commit introduced.

    A marker is transiently unparseable while the user repairs it: type a stray
    `[` into the tail and it stops matching. The first sweep released on that,
    so one backspace thirty columns away destroyed the image - and removing the
    stray `[` then restored a perfectly-formed `[Image #1, 10x20]` citing
    nothing, with the picture unrecoverable.

    The sweep asks whether the number is MENTIONED, not whether it parses.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("what is this")
        await pilot.pause()

        # Break the marker mid-repair, then delete unrelated text far away.
        editor.selection = Selection.cursor((0, len("[Image #1, 1")))
        editor.insert("[")
        await pilot.pause()
        assert editor.referenced_images() == [], "the fixture did not break the marker"

        editor.selection = Selection.cursor((0, len(editor.text)))
        await pilot.press("backspace")
        await pilot.pause()

        # Repair it. The image must come back with the text.
        editor.selection = Selection.cursor((0, len("[Image #1, 1[")))
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text.startswith("[Image #1, 10x20]"), f"repair failed: {editor.text!r}"
        assert len(editor.referenced_images()) == 1, "a delete elsewhere destroyed the image"


@pytest.mark.asyncio
async def test_cutting_a_clicked_marker_releases_it(tmp_path) -> None:
    """Review round 24, P1. Clicking a chip selects the whole marker - this
    branch's own feature - so ctrl+x on it is a more natural gesture than the
    backspace the previous round closed, and it reached neither the atomic gate
    nor the sweep. The sweep now hangs off every edit that REMOVES a range,
    which is all eight text-removing bindings rather than three.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.rstrip()

        editor.selection = Selection((0, 0), (0, len(marker)))
        await pilot.press("ctrl+x")
        await pilot.pause()

        assert marker not in editor.text, "the cut did not remove the marker"
        assert editor._attachments == {}, "ctrl+x orphaned the attachment"

        editor.insert("[Image #1]")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker revived the cut image"


@pytest.mark.asyncio
async def test_typing_over_a_selected_marker_releases_it(tmp_path) -> None:
    """Same rule from the other common gesture: select the marker and type."""
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.rstrip()

        editor.selection = Selection((0, 0), (0, len(marker)))
        await pilot.press("x")
        await pilot.pause()

        assert editor._attachments == {}, "typing over a marker orphaned the attachment"


@pytest.mark.asyncio
async def test_ctrl_w_takes_the_spaces_it_crossed(tmp_path) -> None:
    """Review round 24, P3. `ctrl+w` removes the word AND the run it crossed -
    asserted, because the previous test only checked the marker had gone, which
    a leftover run of spaces also satisfies."""
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("before ")
        await _paste(app, pilot, path)
        await pilot.pause()
        assert editor.text.endswith("] "), "the fixture lost the trailing space"

        await pilot.press("ctrl+w")
        await pilot.pause()

        assert editor.text == "before ", f"ctrl+w left the crossed spaces: {editor.text!r}"


@pytest.mark.asyncio
async def test_ctrl_w_over_a_word_selection_takes_the_word(tmp_path) -> None:
    """ctrl+w with an ordinary word selected takes the selection and leaves the
    marker alone.

    Retitled after review round 25: this does NOT defend the crossing path's
    selection guard - the selected word sits where that code declines anyway,
    and the mutation survives here. `test_ctrl_w_over_a_selection_ending_in_
    spaces_takes_only_it` is the one that pins it. Kept because the ordinary
    gesture is worth holding still, not because it guards anything.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.rstrip()
        editor.insert("tail")
        await pilot.pause()

        start = len(marker) + 1
        editor.selection = Selection((0, start), (0, start + 4))
        await pilot.press("ctrl+w")
        await pilot.pause()

        assert marker in editor.text, "ctrl+w ate the marker instead of the selection"
        assert len(editor.referenced_images()) == 1, "the marker's image was released"


@pytest.mark.asyncio
async def test_an_insertion_never_releases_an_attachment(tmp_path) -> None:
    """The sweep is gated on a REMOVAL, and that gate is load-bearing.

    Design round 23 measured it: all 96 printable characters break citeability
    when inserted into the marker's prefix. A typo there stops `[Image #1` from
    appearing at all, so the mention guard cannot save it - only the gate can.
    Sweeping on every edit would destroy a real attachment on one stray
    keystroke, with retyping the marker forbidden by design round 16's D1.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        # Inside the PREFIX, not the tail: this breaks the mention as well as
        # the parse, so nothing but the removal gate is holding the image.
        editor.selection = Selection.cursor((0, len("[Ima")))
        await pilot.press("x")
        await pilot.pause()
        assert "[Image #1" not in editor.text, "the fixture did not break the prefix"
        assert editor._attachments, "an insertion destroyed the attachment"

        await pilot.press("backspace")
        await pilot.pause()
        assert editor.text.startswith("[Image #1, 10x20]"), f"repair failed: {editor.text!r}"
        assert len(editor.referenced_images()) == 1, "the repaired marker lost its image"


@pytest.mark.asyncio
async def test_a_higher_numbered_marker_does_not_hold_a_lower_one_alive(tmp_path) -> None:
    """The mention guard's negative lookahead: `#1` must not be kept alive by
    `#10`. Without it, deleting #1 entirely leaves its image held - and a typed
    `[Image #1]` then revives a picture the buffer never mentions."""
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)
        mine = editor.text.rstrip()

        # Force the next issued number to #10 so both are in one draft.
        editor.insert("[Image #9, 1x1] ")
        await pilot.pause()
        await _paste(app, pilot, second)
        await pilot.pause()
        assert "[Image #10," in editor.text, f"the fixture did not reach #10: {editor.text!r}"

        editor.selection = Selection((0, 0), (0, len(mine)))
        await pilot.press("backspace")
        await pilot.pause()

        assert 1 not in editor._attachments, "#10 held #1's image alive"
        assert 10 in editor._attachments, "#10 lost its own image"


@pytest.mark.asyncio
async def test_ctrl_w_over_a_selection_ending_in_spaces_takes_only_it(tmp_path) -> None:
    """The whitespace-crossing path stands aside for a real selection.

    The earlier guard test selects a word, where the crossing code declines
    anyway. This one selects the run of spaces itself, so the crossing code
    WOULD reach the marker behind them - and must not, because a real selection
    is the user's own range.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.rstrip()
        editor.selection = Selection.cursor((0, len(editor.text)))
        editor.insert("  tail")
        await pilot.pause()

        # Select exactly the run of spaces after the marker.
        editor.selection = Selection((0, len(marker)), (0, len(marker) + 3))
        await pilot.press("ctrl+w")
        await pilot.pause()

        assert marker in editor.text, "ctrl+w crossed into the marker instead of the selection"
        assert len(editor.referenced_images()) == 1, "the marker's image was released"


@pytest.mark.asyncio
async def test_prefix_damage_survives_a_delete_elsewhere(tmp_path) -> None:
    """Review round 25, P1. The repair window has to cover the whole marker.

    The mention guard protected damage AFTER `#N`, but a stray character inside
    the `[Image #` prefix breaks the mention as well as the parse - so a delete
    twenty-six columns away still destroyed the image, and removing the stray
    character left a perfectly formed marker citing nothing, unrecoverable
    because a retyped marker is forbidden from reviving it.

    The sweep now only adjudicates attachments the removal actually touched, so
    how the marker is damaged stops mattering.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.insert("what is this")
        await pilot.pause()

        editor.selection = Selection.cursor((0, 4))
        await pilot.press("x")
        await pilot.pause()
        assert "[Image #1" not in editor.text, "the fixture did not damage the prefix"

        editor.selection = Selection.cursor((0, len(editor.text)))
        await pilot.press("backspace")
        await pilot.pause()

        editor.selection = Selection.cursor((0, 5))
        await pilot.press("backspace")
        await pilot.pause()

        assert editor.text.startswith("[Image #1, 10x20]"), f"repair failed: {editor.text!r}"
        assert len(editor.referenced_images()) == 1, "a delete elsewhere destroyed the image"


@pytest.mark.asyncio
async def test_an_unrelated_fragment_cannot_keep_a_deleted_image_alive(tmp_path) -> None:
    """Review round 25, P2 - the cost of the whole-buffer mention guard.

    Any text containing `[Image #1` kept the attachment alive after its real
    marker was removed, so a later typed marker bound to a picture the user had
    deleted. Prose explaining the syntax is enough to do it.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        marker = editor.text.rstrip()
        editor.selection = Selection.cursor((0, len(editor.text)))
        editor.insert("to cite one you type [Image #1 and close it")
        await pilot.pause()

        editor.selection = Selection((0, 0), (0, len(marker)))
        await pilot.press("backspace")
        await pilot.pause()
        assert editor._attachments == {}, "an unrelated fragment held the deleted image"

        editor.insert(" [Image #1]")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker revived the deleted image"


@pytest.mark.asyncio
async def test_clearing_away_a_damaged_marker_holds_but_never_sends_it(tmp_path) -> None:
    """A damaged marker cleared away leaves its image held, uncited and unsent.

    This asserted a release until review round 26 showed the rule that produced
    it - "the cut text names #N" - cannot tell a damaged marker from prose that
    says `#1`, and destroyed repairable attachments irreversibly. The rule is
    gone and the orphan is the accepted cost: invisible, never sent, and
    cleared with the draft. Reachable only through `cite`'s documented
    fallback, which is the residual already recorded on that function.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        editor.selection = Selection.cursor((0, 4))
        await pilot.press("x")
        await pilot.pause()
        assert editor._attachments, "the fixture released too early"

        damaged = editor.text.rstrip()
        editor.selection = Selection((0, 0), (0, len(damaged)))
        await pilot.press("backspace")
        await pilot.pause()

        assert editor._attachments, "the orphan was destroyed rather than held"
        assert editor.referenced_images() == [], "an uncited orphan was sent"
        assert editor._first_citation_columns(0) == set(), "an uncited orphan was chipped"


@pytest.mark.asyncio
async def test_cutting_a_higher_number_does_not_release_a_damaged_lower_one(tmp_path) -> None:
    """The lookahead on the CUT text, not just on the buffer.

    The "already uncitable, and the cut names its number" clause asks whether
    the removed text mentions `#N`. Without `(?![0-9])`, deleting `[Image #10,
    30x40]` reads as a mention of `#1`, so a damaged `#1` elsewhere is released
    - and its image is unrecoverable, because a retyped marker may not revive
    it.
    """
    first = _png(tmp_path / "a.png", 10, 20)
    second = _png(tmp_path / "b.png", 30, 40)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, first)

        # Damage #1 so it is uncitable but repairable.
        editor.selection = Selection.cursor((0, 4))
        await pilot.press("x")
        await pilot.pause()
        assert editor.referenced_images() == [], "the fixture did not damage #1"

        editor.selection = Selection.cursor((0, len(editor.text)))
        editor.insert("[Image #9, 1x1] ")
        await pilot.pause()
        await _paste(app, pilot, second)
        await pilot.pause()
        tenth = "[Image #10, 30x40]"
        assert tenth in editor.text, f"the fixture did not reach #10: {editor.text!r}"

        start = editor.text.index(tenth)
        editor.selection = Selection((0, start), (0, start + len(tenth)))
        await pilot.press("backspace")
        await pilot.pause()

        assert 1 in editor._attachments, "cutting #10 released the damaged #1"
        editor.selection = Selection.cursor((0, 5))
        await pilot.press("backspace")
        await pilot.pause()
        assert len(editor.referenced_images()) == 1, "#1 could not be repaired"


@pytest.mark.asyncio
async def test_deleting_a_marker_s_tail_releases_it(tmp_path) -> None:
    """Design round 24, D16. The cut has to OVERLAP the citation, not contain
    its first cell.

    Dragging over the tail and pressing backspace leaves `[Image #1` - which
    still mentions the number and still starts at offset 0, outside the cut -
    so the attachment was never adjudicated. It stayed held with no chip, and a
    typed `[Image #1]` then chipped and sent a picture the buffer no longer
    describes.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        # A real range over the tail only: from the comma to the bracket.
        marker = editor.text.rstrip()
        editor.selection = Selection((0, marker.index(",")), (0, len(marker)))
        await pilot.press("backspace")
        await pilot.pause()
        assert editor.text.startswith("[Image #1"), f"the fixture cut too much: {editor.text!r}"

        assert editor._attachments == {}, "a tail delete left the image held"
        editor.insert("] see [Image #1] please")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker revived the image"


@pytest.mark.asyncio
async def test_a_lengthened_tail_is_measured_at_its_real_width(tmp_path) -> None:
    """Review round 26, P1. The citation's span comes from `cite`, not from
    `len(attachment.marker)`.

    Editing the tail longer is a gesture `cite` deliberately protects, and it
    makes the citation in the buffer wider than the recorded marker. Measuring
    with the recorded width stopped the span short, so a cut in the gap escaped
    the overlap test - reopening D16 exactly: held, unchipped, and revived by a
    typed marker.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)

        # Lengthen the tail: '10x20' -> '10x200'.
        editor.selection = Selection.cursor((0, editor.text.index("]")))
        await pilot.press("0")
        await pilot.pause()
        assert "[Image #1, 10x200]" in editor.text, f"fixture: {editor.text!r}"
        assert len(editor.referenced_images()) == 1, "the longer tail orphaned the draft"

        # Cut ONLY the closing bracket - past the recorded marker's end.
        closing = editor.text.index("]")
        editor.selection = Selection((0, closing), (0, closing + 1))
        await pilot.press("backspace")
        await pilot.pause()

        assert editor._attachments == {}, "a cut past the recorded width was not seen"
        editor.insert("] see [Image #1] please")
        await pilot.pause()
        assert editor.referenced_images() == [], "a typed marker revived the image"


@pytest.mark.asyncio
async def test_deleting_prose_that_says_the_number_spares_a_damaged_marker(tmp_path) -> None:
    """Review round 26, P2. `#1` is ordinary prose in a draft about images.

    While a marker is damaged and repairable, deleting unrelated text that
    happens to contain its number destroyed the image - and the repair then
    produced a perfectly formed marker citing nothing, with no way back.
    """
    path = _png(tmp_path / "a.png", 10, 20)
    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, path)
        prose = "compare with screenshot #1 from yesterday"
        editor.selection = Selection.cursor((0, len(editor.text)))
        editor.insert(prose)
        await pilot.pause()

        editor.selection = Selection.cursor((0, 4))
        await pilot.press("x")
        await pilot.pause()
        assert editor.referenced_images() == [], "the fixture did not damage the marker"

        start = editor.text.index("screenshot #1")
        editor.selection = Selection((0, start), (0, len(editor.text)))
        await pilot.press("backspace")
        await pilot.pause()
        assert editor._attachments, "deleting prose that says #1 destroyed the image"

        editor.selection = Selection.cursor((0, 5))
        await pilot.press("backspace")
        await pilot.pause()
        assert editor.text.startswith("[Image #1, 10x20]"), f"repair failed: {editor.text!r}"
        assert len(editor.referenced_images()) == 1, "the repaired marker lost its image"
