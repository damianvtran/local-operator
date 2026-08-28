"""Pasting an image the terminal could NOT hand over as text (issue #372).

The companion to ``test_paste_images.py``, which covers the path branch. This
file covers the other route: an EMPTY bracketed paste, which is what a terminal
sends when ``Cmd+V`` had nothing textual to give — the macOS pasteboard held
image bytes, or a Finder file URL, and no text at all.

Why the gap existed is worth restating, because it is what these tests protect
against recurring: the composer only ever received a path because **cmux**
writes the clipboard image to ``$TMPDIR/clipboard-<stamp>-<hash>.png`` and
bracket-pastes that name. That is a cmux feature, not a terminal one. In
Ghostty, Terminal.app or iTerm2, ``Cmd+V`` on a screenshot delivered
``Paste("")`` and the composer inserted an empty string — a keystroke
indistinguishable from a dead key.

The clipboard READ itself is faked here (``local_operator.clipboard`` has its
own suite, and its real-tooling evidence is in the PR). What is under test is
the composer's routing: that an empty paste consults the clipboard at all, that
the bytes go through the same bound and produce the same marker as the path
branch, that it happens off the event loop, and that an ordinary text paste is
still inserted verbatim exactly once.

Every paste is posted to the APP, not the widget, for the reason
``test_paste_images.py`` documents: ``App.on_event`` forwards a non-forwarded
``Paste`` to the focused widget, so posting to the widget delivers it twice.
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading

import pytest
from PIL import Image
from textual import events
from textual.app import App, ComposeResult

from local_operator.clipboard import ClipboardImage
from local_operator.tui.widgets import editor as editor_module
from local_operator.tui.widgets.editor import Editor, EditorPasteEmpty


def _png_bytes(width: int = 1568, height: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 40)).save(buffer, "PNG")
    return buffer.getvalue()


class Host(App[None]):
    """Records the notices the editor posts, so the feedback is assertable
    without standing up the whole app's toast plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.empty_notices: list[EditorPasteEmpty] = []

    def compose(self) -> ComposeResult:
        yield Editor()

    def on_editor_paste_empty(self, message: EditorPasteEmpty) -> None:
        self.empty_notices.append(message)


async def _paste(app: App[None], pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    for _ in range(20):
        await pilot.pause()


def _stub_clipboard(monkeypatch, *, image=None, paths=None) -> dict[str, int]:
    """Replace both clipboard reads; count the calls so routing is assertable."""
    counts = {"image": 0, "paths": 0}

    def read_image(max_bytes: int):
        counts["image"] += 1
        return image

    def read_paths():
        counts["paths"] += 1
        return list(paths or [])

    monkeypatch.setattr(editor_module, "read_clipboard_image", read_image)
    monkeypatch.setattr(editor_module, "read_clipboard_file_paths", read_paths)
    return counts


# -- the reported bug ---------------------------------------------------------
@pytest.mark.asyncio
async def test_an_empty_paste_attaches_the_image_on_the_clipboard(monkeypatch) -> None:
    """The whole issue: a native macOS screenshot, ``Cmd+V``, and before this
    the composer inserted an empty string and said nothing."""
    data = _png_bytes()
    _stub_clipboard(monkeypatch, image=ClipboardImage(data, "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == "[Image #1, 1568x200] "
        assert len(editor.referenced_images()) == 1
        image = editor.referenced_images()[0]
        assert image.mime_type == "image/png"
        # The real bytes, not a path: this is what reaches the provider.
        assert base64.b64decode(image.data)[:8] == b"\x89PNG\r\n\x1a\n"
        assert app.empty_notices == [], "an attachment must not also raise the notice"


@pytest.mark.asyncio
async def test_the_clipboard_marker_is_identical_to_the_path_branch_s(
    monkeypatch, tmp_path
) -> None:
    """The two routes are ONE gesture from the user's side — ``Cmd+V`` on a
    screenshot — that merely takes a different road depending on the terminal.
    A marker that differed between them would make the composer's receipt
    depend on which emulator was running, so both go through
    ``_attach_image_bytes``.
    """
    data = _png_bytes(1000, 300)
    path = tmp_path / "shot.png"
    path.write_bytes(data)

    _stub_clipboard(monkeypatch, image=ClipboardImage(data, "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")
        from_clipboard = editor.text
        clipboard_image = editor.referenced_images()[0]

    monkeypatch.setattr(editor_module, "read_clipboard_image", lambda max_bytes: None)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, str(path))
        from_path = editor.text
        path_image = editor.referenced_images()[0]

    assert from_clipboard == from_path == "[Image #1, 1000x300] "
    assert clipboard_image.data == path_image.data
    assert clipboard_image.mime_type == path_image.mime_type


@pytest.mark.asyncio
async def test_a_whitespace_only_paste_also_consults_the_clipboard(monkeypatch) -> None:
    """Some terminals wrap the empty payload in a newline. That is still "the
    terminal had no text for me", and inserting the whitespace instead would
    put a character in the draft the user never typed."""
    counts = _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "\n")

        assert counts["image"] == 1
        assert editor.text == "[Image #1, 1568x200] "


# -- the Finder Cmd+C route ---------------------------------------------------
@pytest.mark.asyncio
async def test_a_finder_copy_attaches_through_the_path_branch(monkeypatch, tmp_path) -> None:
    """Finder's ``Cmd+C`` puts only a ``public.file-url`` flavor on the
    pasteboard — no text and no image bytes — so it arrives as an empty paste
    too. Routing it into the path branch is what makes a copied file behave
    exactly like the same file dragged in.
    """
    path = tmp_path / "screenshot.png"
    path.write_bytes(_png_bytes(640, 480))
    _stub_clipboard(monkeypatch, image=None, paths=[str(path)])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == "[Image #1, 640x480] "
        assert len(editor.referenced_images()) == 1


@pytest.mark.asyncio
async def test_a_copied_file_whose_name_has_spaces_still_attaches(monkeypatch, tmp_path) -> None:
    """The paths come from an API, not a terminal, so they are quoted on the
    way into the path branch — which exists to UNDO shell quoting. Without
    that, ``Screen Shot 2026.png`` splits into three paths that do not exist.
    """
    path = tmp_path / "Screen Shot 2026 at 10.11.12.png"
    path.write_bytes(_png_bytes(320, 240))
    _stub_clipboard(monkeypatch, image=None, paths=[str(path)])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == "[Image #1, 320x240] "


@pytest.mark.asyncio
async def test_image_bytes_win_over_file_urls(monkeypatch, tmp_path) -> None:
    """A pasteboard can carry both. The bytes are what the user copied most
    recently in the reported gesture, and reading them costs no filesystem
    access, so they are tried first and the URL read never happens."""
    path = tmp_path / "other.png"
    path.write_bytes(_png_bytes(100, 100))
    counts = _stub_clipboard(
        monkeypatch, image=ClipboardImage(_png_bytes(800, 600), "image/png"), paths=[str(path)]
    )
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == "[Image #1, 800x600] "
        assert counts["paths"] == 0


# -- the control: ordinary text is untouched ----------------------------------
@pytest.mark.asyncio
async def test_a_plain_text_paste_is_inserted_once_and_never_reads_the_clipboard(
    monkeypatch,
) -> None:
    """The regression the ``_on_paste`` MRO note warns about, re-pinned for the
    new branch.

    Textual invokes every ``_on_paste`` up the MRO, so the base handler runs on
    its own and a mistaken ``prevent_default`` (or a mistaken absence of one)
    shows up as text pasted twice or not at all. The new empty-paste branch
    calls ``prevent_default`` unconditionally, so this proves the branch is not
    entered for text that has content.

    It also proves the common case costs nothing: pasting a paragraph must not
    spawn ``osascript``.
    """
    counts = _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "some ordinary prose")

        assert editor.text == "some ordinary prose"
        assert editor.referenced_images() == []
        assert counts["image"] == 0, "a text paste must not touch the clipboard"
        assert app.empty_notices == []


# -- feedback -----------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_paste_that_attaches_nothing_says_so(monkeypatch) -> None:
    """The last part of the report: ``Cmd+V`` producing NO response at all is
    indistinguishable from a broken key, which is why this was filed as a paste
    failure rather than as a missing capability."""
    _stub_clipboard(monkeypatch, image=None, paths=[])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == ""
        assert len(app.empty_notices) == 1


@pytest.mark.asyncio
async def test_the_notice_fires_once_per_keypress(monkeypatch) -> None:
    """A notice per press, not per backend consulted. The clipboard read and
    the file-URL read both come back empty on an empty clipboard, and two cards
    for one keystroke is the noise this feature could easily have become."""
    _stub_clipboard(monkeypatch, image=None, paths=[])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")
        await _paste(app, pilot, "")

        assert len(app.empty_notices) == 2


@pytest.mark.asyncio
async def test_an_unattachable_clipboard_image_reports_rather_than_inserting(
    monkeypatch,
) -> None:
    """Bytes were there and could not be attached — an unsupported format, a
    decode failure, past the size cap. Different event, same answer to the
    user: nothing attached, and the notice says so rather than leaving the
    keystroke silent."""
    _stub_clipboard(monkeypatch, image=ClipboardImage(b"\x00\x01not an image", "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == ""
        assert editor.referenced_images() == []
        assert len(app.empty_notices) == 1


@pytest.mark.asyncio
async def test_an_empty_paste_never_inserts_its_payload(monkeypatch) -> None:
    """``prevent_default`` is unconditional on this branch. Letting the base
    handler run when the clipboard turned out to be empty would insert the
    whitespace payload the terminal sent, putting a character in the draft that
    the user did not type."""
    _stub_clipboard(monkeypatch, image=None, paths=[])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("draft")
        await _paste(app, pilot, "  \n  ")

        assert editor.text == "draft"


# -- the event loop -----------------------------------------------------------
@pytest.mark.asyncio
async def test_the_clipboard_read_runs_off_the_event_loop_thread(monkeypatch) -> None:
    """Same invariant as the decode, and for a sharper reason: this one shells
    out. A wedged clipboard daemon inline on the keystroke handler is a frozen
    composer, and the measured macOS read is ~265 ms even when everything is
    healthy.

    Asserted as a THREAD IDENTITY, not as elapsed time: a synchronous handler
    that never awaits also never lets the loop fall behind, so a timing test is
    green both ways and pins nothing.
    """
    seen: list[str] = []

    def read_image(max_bytes: int):
        seen.append(threading.current_thread().name)
        return ClipboardImage(_png_bytes(), "image/png")

    monkeypatch.setattr(editor_module, "read_clipboard_image", read_image)
    monkeypatch.setattr(editor_module, "read_clipboard_file_paths", lambda: [])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.referenced_images(), "the image was not attached at all"
        assert seen and seen[0] != threading.main_thread().name


@pytest.mark.asyncio
async def test_a_slow_clipboard_read_does_not_stall_the_loop(monkeypatch) -> None:
    """The thread's actual payoff: the app keeps painting and other widgets
    keep responding while the read is outstanding. ``to_thread`` is what buys
    that; an inline ``time.sleep`` would not."""
    started = threading.Event()

    def read_image(max_bytes: int):
        started.set()
        threading.Event().wait(0.4)
        return ClipboardImage(_png_bytes(), "image/png")

    monkeypatch.setattr(editor_module, "read_clipboard_image", read_image)
    monkeypatch.setattr(editor_module, "read_clipboard_file_paths", lambda: [])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        app.post_message(events.Paste(""))

        # The loop must reach this WHILE the read is still blocked in its
        # thread. A synchronous read would not let these awaits run at all.
        for _ in range(5):
            await pilot.pause()
            await asyncio.sleep(0.01)
        assert started.is_set()

        for _ in range(40):
            await pilot.pause()
            if editor.referenced_images():
                break
        assert editor.text == "[Image #1, 1568x200] "
