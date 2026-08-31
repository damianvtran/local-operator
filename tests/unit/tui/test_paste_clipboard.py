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
import random
import threading
import time
from contextlib import contextmanager

import pytest
from PIL import Image
from textual import events
from textual.app import App, ComposeResult
from textual.widgets.text_area import Selection

from local_operator.clipboard import ClipboardContents, ClipboardImage
from local_operator.tui.widgets import editor as editor_module
from local_operator.tui.widgets.editor import (
    Editor,
    EditorPasteAttached,
    EditorPasteEmpty,
)


def _png_bytes(width: int = 1568, height: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 40)).save(buffer, "PNG")
    return buffer.getvalue()


@contextmanager
def monkeypatch_context(target, name, value):
    """Patch an attribute for a block, restoring it afterwards.

    pytest's `monkeypatch` unwinds at teardown, which is too late for a test
    that has to leave the patch in place only while the app is running and then
    assert on samples taken during it.
    """
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


class Host(App[None]):
    """Records the notices the editor posts, so the feedback is assertable
    without standing up the whole app's toast plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.empty_notices: list[EditorPasteEmpty] = []
        self.attached_notices: list[EditorPasteAttached] = []
        #: Samples of the reading card's state, as ``(raised, when)``. The Host
        #: stands in for the app hook the composer calls; see
        #: `Editor.PASTE_READING_HOOK`.
        self.reading_calls: list[bool] = []

    def compose(self) -> ComposeResult:
        yield Editor()

    def on_editor_paste_empty(self, message: EditorPasteEmpty) -> None:
        self.empty_notices.append(message)

    def on_editor_paste_attached(self, message: EditorPasteAttached) -> None:
        self.attached_notices.append(message)

    def show_clipboard_reading_notice(self, reading: bool) -> None:
        self.reading_calls.append(reading)


async def _paste(app: App[None], pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    for _ in range(20):
        await pilot.pause()


def _stub_clipboard(
    monkeypatch, *, image=None, paths=None, text="", refused_remote=False
) -> dict[str, int]:
    """Replace the clipboard read; record the calls so routing is assertable."""
    counts = {"reads": 0}

    def read_clipboard(*args, **kwargs):
        counts["reads"] += 1
        return ClipboardContents(
            image=image,
            paths=tuple(paths or ()),
            text=text,
            refused_remote=refused_remote,
        )

    monkeypatch.setattr(editor_module, "read_clipboard", read_clipboard)
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

    monkeypatch.setattr(editor_module, "read_clipboard", lambda *a, **k: ClipboardContents())
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


@pytest.mark.parametrize("payload", ["    ", "\t", "\n", "\n\n"])
@pytest.mark.asyncio
async def test_copied_whitespace_wins_over_an_image_on_the_clipboard(
    payload: str, monkeypatch
) -> None:
    """Whitespace the user copied is inserted even when an image IS readable.

    Round 2 (D9): the D1 fix keyed on the payload but still let a successful
    clipboard read override it, so pasting a four-space indent with a PNG on
    the pasteboard replaced the indent with `[Image #1, 1568x200]` — an image
    the user did not ask for on that keypress, silently, with the indent gone.
    Same defect as D1 at one tenth the reach.

    The clipboard is not consulted at all here, which is also what stops an
    ordinary indent paste paying a multi-second read it never needed (U7).
    """
    counts = _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("X")
        await _paste(app, pilot, payload)

        assert editor.text == f"X{payload}", "the copied whitespace was replaced"
        assert editor.referenced_images() == [], "an image was attached uninvited"
        assert counts["reads"] == 0, "an ordinary whitespace paste must not read the clipboard"
        assert app.empty_notices == []


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
    recently in the reported gesture, so they win and the paths are ignored."""
    path = tmp_path / "other.png"
    path.write_bytes(_png_bytes(100, 100))
    _stub_clipboard(
        monkeypatch, image=ClipboardImage(_png_bytes(800, 600), "image/png"), paths=[str(path)]
    )
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.text == "[Image #1, 800x600] "


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
        assert counts["reads"] == 0, "a text paste must not touch the clipboard"
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


@pytest.mark.parametrize(
    ("payload", "label"),
    [
        ("    ", "a four-space indent"),
        ("  ", "a two-space indent"),
        (" ", "a single space"),
        ("\t", "a tab"),
        ("\n", "a blank line"),
        ("\n\n", "two blank lines"),
        ("\n    \n", "an indented blank line"),
    ],
)
@pytest.mark.asyncio
async def test_pasted_whitespace_is_inserted_verbatim_exactly_once(
    payload: str, label: str, monkeypatch
) -> None:
    """Whitespace the USER copied must paste, even though it reaches this
    branch (review round 1, F1/D1).

    This is the regression the first version of the feature shipped: it treated
    every whitespace-only payload as the terminal's empty-paste signal and
    consumed the event unconditionally, so pasting an indent into the composer
    silently discarded it and raised a toast about images. That is the same
    class of failure as #372 on a gesture that has nothing to do with images,
    and the test that used to live here asserted the broken behaviour.

    The composer takes multi-line prompts, so pasting a run of indentation, a
    tab, or a blank line between paragraphs is ordinary. The payload is
    indistinguishable from the synthesised one, so the branch stops guessing:
    it consults the clipboard, and consumes the event only if that attached
    something.

    ``exactly once`` is the other half. This branch calls ``prevent_default``
    on the success path, and the MRO note on ``_on_paste`` records that getting
    that wrong duplicates the insert.
    """
    _stub_clipboard(monkeypatch, image=None, paths=[])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("X")
        await _paste(app, pilot, payload)

        assert editor.text == f"X{payload}", f"{label} was not inserted verbatim"
        assert app.empty_notices == [], (
            "a whitespace paste that inserted its payload has already succeeded; "
            "a notice about images the user was not pasting is noise"
        )


@pytest.mark.asyncio
async def test_a_genuinely_empty_paste_inserts_nothing(monkeypatch) -> None:
    """The terminal-synthesised payload for an image-only clipboard.

    Falls through to the base handler like any other unattachable paste, which
    inserts ``""`` — a no-op nobody can see. That is what lets the branch above
    stop discriminating on the payload's content: the empty case needs no
    special handling to look right.
    """
    _stub_clipboard(monkeypatch, image=None, paths=[])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        editor.insert("draft")
        await _paste(app, pilot, "")

        assert editor.text == "draft"
        assert len(app.empty_notices) == 1


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

    def read_clipboard(*args, **kwargs):
        seen.append(threading.current_thread().name)
        return ClipboardContents(image=ClipboardImage(_png_bytes(), "image/png"))

    monkeypatch.setattr(editor_module, "read_clipboard", read_clipboard)
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

    def read_clipboard(*args, **kwargs):
        started.set()
        threading.Event().wait(0.4)
        return ClipboardContents(image=ClipboardImage(_png_bytes(), "image/png"))

    monkeypatch.setattr(editor_module, "read_clipboard", read_clipboard)
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


# -- what the notice is allowed to claim --------------------------------------
@pytest.mark.asyncio
async def test_an_ssh_refusal_is_not_reported_as_an_empty_clipboard(monkeypatch) -> None:
    """Over SSH the clipboard is never read, so "no image on the clipboard" is
    a statement about something nobody looked at (review round 1, D2/U2).

    The user's screenshot really is on their local clipboard; told it is not,
    the only move they can think of is to re-copy it, which cannot help. The
    reason code is what lets the app name the refusal and the workaround
    instead.
    """
    _stub_clipboard(monkeypatch, image=None, paths=[], refused_remote=True)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert [n.reason for n in app.empty_notices] == ["remote"]


@pytest.mark.asyncio
async def test_an_image_that_cannot_be_attached_says_so(monkeypatch) -> None:
    """An image WAS found and refused, which is a different answer to the user
    than an empty clipboard: cropping fixes one and nothing fixes the other."""
    _stub_clipboard(monkeypatch, image=ClipboardImage(b"\x00\x01not an image", "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert [n.reason for n in app.empty_notices] == ["unattachable"]


@pytest.mark.asyncio
async def test_an_empty_clipboard_keeps_the_deliberately_vague_reason(monkeypatch) -> None:
    """The one case that stays collapsed. An empty clipboard, a text-only one,
    a missing ``xclip`` and a wedged daemon are indistinguishable by design, so
    the reason must not pretend to know which."""
    _stub_clipboard(monkeypatch, image=None, paths=[])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert [n.reason for n in app.empty_notices] == ["nothing"]


@pytest.mark.asyncio
async def test_a_successful_attach_announces_itself(monkeypatch) -> None:
    """The event that lets the app retire a paste notice still held behind an
    actionable card (review round 1, D3). Without it that notice surfaces when
    the slot frees and contradicts a composer holding the image."""
    _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.referenced_images()
        assert len(app.attached_notices) == 1
        assert app.empty_notices == []


@pytest.mark.asyncio
async def test_a_finder_copy_that_attaches_also_announces_itself(monkeypatch, tmp_path) -> None:
    """The file-URL route reaches the same success, so it must retire a held
    notice too."""
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes(320, 240))
    _stub_clipboard(monkeypatch, image=None, paths=[str(path)])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert len(app.attached_notices) == 1


@pytest.mark.asyncio
async def test_the_read_is_not_capped_at_the_attachment_budget(monkeypatch) -> None:
    """The blocker from review round 1 (U1), pinned at the seam that caused it.

    The composer passed ``MAX_ATTACHMENT_BYTES`` to the clipboard read, so the
    4 MB attachment budget was applied to the RAW pasteboard bytes — before
    ``bound_image_for_model``, whose entire job is to shrink them. A real
    ``Cmd+Shift+Ctrl+4`` on a Retina display puts 8.4-8.5 MB on the pasteboard
    and bounds to 0.28 MB, so the reported gesture still attached nothing and
    still blamed the clipboard.

    Asserted as "the read is not given the attachment cap", which is the
    mistake itself, rather than by round-tripping an 8 MB fixture through the
    encoder on every run.
    """
    seen: list[object] = []

    def read_clipboard(*args, **kwargs):
        seen.append(args[0] if args else kwargs.get("max_bytes"))
        return ClipboardContents(image=ClipboardImage(_png_bytes(), "image/png"))

    monkeypatch.setattr(editor_module, "read_clipboard", read_clipboard)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        assert editor.referenced_images(), "the image was not attached"
        assert seen and seen[0] is None, (
            "the composer must not hand the read a byte ceiling of its own; the "
            "ingest default is MAX_CLIPBOARD_READ_BYTES and the attachment cap "
            "is applied after bounding"
        )


@pytest.mark.asyncio
async def test_a_large_source_image_attaches_because_the_cap_follows_the_bound(
    monkeypatch,
) -> None:
    """The U1 blocker at its real seam, in the direction that was broken.

    A source image LARGER than ``MAX_ATTACHMENT_BYTES`` must still attach when
    the bound brings it under, because that is what the bound is for. The first
    version gated on the source bytes in two places — the clipboard read and
    the shared attachment tail — so a real Retina screenshot (8.4-8.5 MB on the
    pasteboard, 0.28 MB bounded) was thrown away twice over before the resize
    could run.

    The fixture is deliberately high-entropy: a flat-colour PNG of any
    dimensions compresses to a few KB and would sit under the cap by accident,
    which is exactly why the original testing missed this.
    """
    from local_operator.tui.widgets.editor import MAX_ATTACHMENT_BYTES

    buffer = io.BytesIO()
    random.seed(11)
    image = Image.new("RGB", (2400, 1600))
    image.putdata(
        [
            (random.randrange(256), random.randrange(256), random.randrange(256))
            for _ in range(2400 * 1600)
        ]
    )
    image.save(buffer, "PNG")
    source = buffer.getvalue()
    assert len(source) > MAX_ATTACHMENT_BYTES, "fixture must exceed the attachment cap"

    _stub_clipboard(monkeypatch, image=ClipboardImage(source, "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for _ in range(3):
            await _paste(app, pilot, "")
            if editor.referenced_images():
                break

        assert editor.referenced_images(), "a bounded-down image must still attach"
        attached = base64.b64decode(editor.referenced_images()[0].data)
        assert len(attached) <= MAX_ATTACHMENT_BYTES, "the cap must hold on what is SENT"
        assert app.empty_notices == []


@pytest.mark.asyncio
async def test_a_held_paste_notice_is_retired_by_a_later_successful_paste(monkeypatch) -> None:
    """Design round 1 (D3), reproduced against the real app and pinned.

    The captured sequence: an MCP failure holds the slot, an empty paste defers
    the notice, the user then pastes a screenshot successfully, and when the
    failure expires the deferred card is promoted — so "couldn't attach an
    image" paints over a composer visibly holding ``[Image #1, ...]``, seconds
    later, with no keypress to explain it.

    Uses the real ``OperatorApp`` rather than this file's ``Host``, because the
    behaviour under test is the app's toast ownership, not the widget's.
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.toast import TOAST_FAILURE_MS, Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    state: dict[str, ClipboardImage | None] = {"image": None}
    monkeypatch.setattr(
        editor_module,
        "read_clipboard",
        lambda *a, **k: ClipboardContents(image=state["image"]),
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and app._session is None:
            await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        toast = app.query_one(Toast)

        # An actionable notice claims the slot, so the paste notice must defer.
        toast.show("MCP github failed: command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        await _paste(app, pilot, "")
        assert toast._deferred is not None, "the notice should be held, not shown"

        state["image"] = ClipboardImage(_png_bytes(), "image/png")
        for _ in range(3):
            await _paste(app, pilot, "")
            if editor.referenced_images():
                break
        assert editor.referenced_images(), "the second paste should have attached"
        assert toast._deferred is None, (
            "a successful attach must retire the held notice; otherwise it "
            "surfaces later contradicting the composer"
        )

        toast.dismiss_toast()
        for _ in range(6):
            await pilot.pause()
        assert toast.message == "", "nothing stale may be promoted into the freed slot"


@pytest.mark.asyncio
async def test_a_path_paste_also_retires_a_held_notice(monkeypatch, tmp_path) -> None:
    """Round 2 (D8/D3): the path route attaches without retiring the notice.

    D3's fix covered the clipboard route only, so the stale card still surfaced
    through the route cmux users hit — and, worse, through the exact gesture
    the notice recommends: "Paste a file path instead." The user follows the
    advice, it works, and the card that gave the advice reappears to deny it.
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.toast import TOAST_FAILURE_MS, Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes(400, 100))
    monkeypatch.setattr(
        editor_module, "read_clipboard", lambda *a, **k: ClipboardContents(refused_remote=True)
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and app._session is None:
            await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        toast = app.query_one(Toast)

        toast.show("MCP github failed: command not found: gh", duration_ms=TOAST_FAILURE_MS)
        await pilot.pause()
        await _paste(app, pilot, "")
        assert toast._deferred is not None, "the SSH notice should be held"

        # The remedy the notice just told the user to perform.
        await _paste(app, pilot, str(path))
        assert editor.referenced_images(), "the path paste should have attached"
        assert toast._deferred is None, (
            "following the notice's own advice must retire it; otherwise the "
            "card reappears to deny the thing it recommended"
        )


@pytest.mark.asyncio
async def test_a_showing_paste_notice_is_retired_by_the_attach_that_answers_it(
    monkeypatch, tmp_path
) -> None:
    """Round 3 (D13): a card already on screen survived the paste answering it.

    D8's fix used `drop_deferred`, which only covers the HELD card. The attach
    lands 45-57 ms after the notice is raised, so the showing card then sat for
    the rest of its duration above a composer the user can see is populated —
    denying, in the same breath, the remedy it had just recommended.
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes(400, 100))
    monkeypatch.setattr(
        editor_module, "read_clipboard", lambda *a, **k: ClipboardContents(refused_remote=True)
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and app._session is None:
            await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        toast = app.query_one(Toast)

        await _paste(app, pilot, "")
        assert toast.message, "the SSH notice should be showing, not held"

        # The remedy the card just recommended.
        await _paste(app, pilot, str(path))
        assert editor.referenced_images(), "the path paste should have attached"
        assert toast.message == "", (
            "a card claiming the app could not attach an image must not remain "
            "above a composer that visibly holds one"
        )


@pytest.mark.asyncio
async def test_the_vague_notice_does_not_outrank_a_copy_receipt(monkeypatch) -> None:
    """Round 2 (D11), a consequence of the D6 duration change.

    ``Toast`` derives actionability from duration, so putting the vague notice
    at ``TOAST_FAILURE_MS`` made it hold the slot against every courtesy notice
    for 10 s — while naming nothing to act on, which is precisely the test
    ``toast.py`` documents. The two variants that carry a remedy keep the
    failure duration; the one that does not takes the default.
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    cases = [
        (ClipboardContents(), False),
        (ClipboardContents(refused_remote=True), True),
    ]
    for contents, expected_actionable in cases:
        monkeypatch.setattr(editor_module, "read_clipboard", lambda *a, _c=contents, **k: _c)
        app = OperatorApp(lambda: _factory(FakeSession()))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and app._session is None:
                await pilot.pause()
            app.query_one(Editor).focus()
            await pilot.pause()
            await _paste(app, pilot, "")

            toast = app.query_one(Toast)
            assert toast.message, "a notice should be showing"
            assert toast._actionable is expected_actionable, (
                f"{toast.message!r} actionable={toast._actionable}, expected "
                f"{expected_actionable}; a notice naming no action must not "
                "suppress a copy receipt for a gesture performed afterwards"
            )


@pytest.mark.asyncio
async def test_one_image_attaches_the_same_way_through_both_routes(monkeypatch, tmp_path) -> None:
    """Round 3 (D12): the two routes disagreed about one valid image.

    A large screenshot attached via `Cmd+V` (the clipboard route bounds before
    applying the attachment cap, which is U1's fix) and was REFUSED via Finder
    `Cmd+C`, which reaches the path branch and used to stat against the 4 MB
    attachment budget. Two of this feature's own paths, one image, contradictory
    answers — and the refusal blamed the format of a file that was a valid PNG.

    The path branch now stats against the INGEST ceiling, so the resize decides
    for both routes and `_attach_image_bytes` remains the single authority on
    what may be sent.

    The fixture is high-entropy on purpose: a flat-colour PNG of any dimensions
    compresses under the cap and would pass this test without exercising it.
    """
    from local_operator.tui.widgets.editor import MAX_ATTACHMENT_BYTES

    buffer = io.BytesIO()
    random.seed(5)
    image = Image.new("RGB", (2600, 900))
    image.putdata(
        [
            (random.randrange(256), random.randrange(256), random.randrange(256))
            for _ in range(2600 * 900)
        ]
    )
    image.save(buffer, "PNG")
    source = buffer.getvalue()
    assert len(source) > MAX_ATTACHMENT_BYTES, "fixture must exceed the attachment cap"

    path = tmp_path / "screenshot.png"
    path.write_bytes(source)

    # Route 1: the bytes arrive on the clipboard (Cmd+V on a screenshot).
    _stub_clipboard(monkeypatch, image=ClipboardImage(source, "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for _ in range(3):
            await _paste(app, pilot, "")
            if editor.referenced_images():
                break
        from_clipboard = editor.text

    # Route 2: the same bytes arrive as a copied FILE (Finder Cmd+C).
    _stub_clipboard(monkeypatch, image=None, paths=[str(path)])
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        for _ in range(3):
            await _paste(app, pilot, "")
            if editor.referenced_images():
                break
        from_path = editor.text
        assert editor.referenced_images(), (
            "a valid image the clipboard route attaches must not be refused by "
            "the file route; that refusal was also blamed on its format"
        )
        assert app.empty_notices == []

    assert from_clipboard == from_path


@pytest.mark.asyncio
async def test_the_file_notice_does_not_assert_a_format_problem(monkeypatch, tmp_path) -> None:
    """Round 3 (D12), the copy half.

    One branch is reached by a non-image file, a HEIC, an unreadable path and a
    mixed selection, so a sentence naming formats is a guess — and it was
    reaching a user holding a valid PNG, for whom the only implied remedy was
    converting a PNG to a PNG.
    """
    from local_operator.tui.app import OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    not_an_image = tmp_path / "notes.txt"
    not_an_image.write_text("hello")
    monkeypatch.setattr(
        editor_module,
        "read_clipboard",
        lambda *a, **k: ClipboardContents(paths=(str(not_an_image),)),
    )

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and app._session is None:
            await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        await _paste(app, pilot, "")

        message = app.query_one(Toast).message
        assert message, "a file that would not attach should still be reported"
        assert "PNG" not in message and "WebP" not in message, (
            f"{message!r} names formats, but this branch cannot establish that "
            "the cause was the format"
        )


# -- ctrl+v: THE route that actually fires outside cmux -----------------------
#
# Everything above this line exercises the EMPTY-PASTE branch, which is what
# PR #376 shipped and what it tested. That branch is unreachable on the
# terminals the bug was filed against: with an image-only pasteboard,
# Terminal.app and Ghostty deliver ZERO bytes on Cmd+V and beep, so no `Paste`
# event is ever synthesised (measured with a raw-mode PTY probe; the captures
# are in the PR). `Ctrl+V` delivers `\x16` on both, so it is the only paste
# keystroke a TUI can observe, and these tests cover it.
#
# They press the REAL KEY through the pilot rather than calling the action, so
# the binding override is exercised on every one of them: `TextArea` already
# binds ctrl+v to `action_paste` (which pastes Textual's internal buffer), and
# a test that called `action_system_paste` directly would pass just as happily
# with the wrong action wired to the key.
async def _ctrl_v(app: App[None], pilot) -> None:
    await pilot.press("ctrl+v")
    for _ in range(20):
        await pilot.pause()


@pytest.mark.asyncio
async def test_ctrl_v_attaches_the_image_on_the_system_clipboard(monkeypatch) -> None:
    """The bug #376 did not fix: a screenshot on the pasteboard, and the only
    keystroke the terminal actually delivers."""
    counts = _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == "[Image #1, 1568x200] "
        assert counts["reads"] == 1, "ctrl+v must read the SYSTEM clipboard"
        images = editor.referenced_images()
        assert len(images) == 1 and images[0].mime_type == "image/png"
        assert base64.b64decode(images[0].data)[:8] == b"\x89PNG\r\n\x1a\n"
        assert app.empty_notices == [], "an attachment must not also raise the notice"


@pytest.mark.asyncio
async def test_ctrl_v_overrides_textareas_own_paste_binding(monkeypatch) -> None:
    """``TextArea`` binds ctrl+v to ``action_paste``, which inserts
    ``App.clipboard`` — Textual's INTERNAL buffer, which a copy made in another
    application never fills. Left in place it makes ctrl+v a key that silently
    does nothing on the gesture users mean by it.

    Both halves are asserted: the resolved binding map names only this
    widget's action, and pressing the key does not run the base one. The map
    alone is not enough — Textual merges subclass BINDINGS with its bases', and
    "merged" versus "overridden" for a COLLIDING key is exactly the detail this
    fix depends on (verified against textual 8.2.8).
    """
    _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    base_paste_calls: list[int] = []
    monkeypatch.setattr(
        Editor, "action_paste", lambda self: base_paste_calls.append(1), raising=False
    )

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        bound = editor._bindings.key_to_bindings.get("ctrl+v") or []
        assert [binding.action for binding in bound] == ["system_paste"], (
            "ctrl+v must resolve to the system paste alone; TextArea's "
            "action_paste inserts a buffer the system clipboard never fills"
        )

        await _ctrl_v(app, pilot)
        assert base_paste_calls == [], "TextArea.action_paste must not run"
        assert editor.text == "[Image #1, 1568x200] "


@pytest.mark.asyncio
async def test_ctrl_v_inserts_clipboard_text(monkeypatch) -> None:
    """Text is the ORDINARY thing on a clipboard, and ctrl+v is the user asking
    for the system one by name. Dropping it would leave the key useless in the
    common case — which is the defect the inherited binding already had."""
    _stub_clipboard(monkeypatch, text="from the system clipboard")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == "from the system clipboard"
        assert editor.referenced_images() == []
        assert app.empty_notices == [], "text was pasted, so nothing failed"
        assert app.attached_notices == [], (
            "EditorPasteAttached retires a notice claiming an IMAGE could not "
            "be attached, and pasting text does not falsify that claim"
        )


@pytest.mark.asyncio
async def test_ctrl_v_text_lands_at_the_caret_inside_existing_text(monkeypatch) -> None:
    """A paste is an insertion at the caret, not an append. Pinned because the
    text shape returns through the same path the image MARKERS take, and a
    marker is always issued at the caret too."""
    _stub_clipboard(monkeypatch, text="MIDDLE")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.insert("start end")
        editor.move_cursor((0, 5))
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == "startMIDDLE end"


@pytest.mark.asyncio
async def test_ctrl_v_routes_file_urls_through_the_path_branch(monkeypatch, tmp_path) -> None:
    """Finder's Cmd+C puts a ``public.file-url`` flavor on the pasteboard. It
    must attach exactly as the same file dragged in does, because from the
    user's side it is one gesture."""
    copied = tmp_path / "shot.png"
    copied.write_bytes(_png_bytes(320, 200))
    _stub_clipboard(monkeypatch, paths=(str(copied),))

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == "[Image #1, 320x200] "
        assert len(editor.referenced_images()) == 1
        assert len(app.attached_notices) == 1, "an attach must retire a held notice"


@pytest.mark.asyncio
async def test_ctrl_v_on_an_empty_clipboard_says_so(monkeypatch) -> None:
    """The notice still fires — a keystroke that produces nothing visible is
    the original bug."""
    _stub_clipboard(monkeypatch)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert [notice.reason for notice in app.empty_notices] == ["nothing"]


@pytest.mark.asyncio
async def test_the_paste_notice_is_not_a_discovery_surface(monkeypatch) -> None:
    """No route appends "Try ctrl+v" to the empty-clipboard notice, and this
    pins the reason rather than the wording.

    An earlier revision hung discoverability on this card, gated to the
    empty-`Paste` route. That route requires bytes the terminal never sends
    outside cmux, so the hint reached only users whose `Cmd+V` already worked —
    and even there `reason="nothing"` means the clipboard held nothing
    attachable, so `ctrl+v` would return the same empty answer and the advice
    could not have helped (design/ux round 1, D1/U1).

    `ctrl+v` is taught ambiently instead (`welcome.TIPS`, `/help`), which does
    not depend on an event that never arrives. Both routes are asserted to
    report the identical reason, so neither can quietly grow a route-specific
    hint again.
    """
    for use_ctrl_v in (True, False):
        _stub_clipboard(monkeypatch)
        app = Host()
        async with app.run_test() as pilot:
            app.query_one(Editor).focus()
            await pilot.pause()
            if use_ctrl_v:
                await _ctrl_v(app, pilot)
            else:
                await _paste(app, pilot, "")

            assert [notice.reason for notice in app.empty_notices] == ["nothing"]
            assert not hasattr(app.empty_notices[0], "suggest_system_paste"), (
                "the route-gated hint fired only where it could not help; "
                "discoverability belongs on an ambient surface"
            )


@pytest.mark.asyncio
async def test_ctrl_v_replaces_a_live_selection_like_every_other_paste(monkeypatch) -> None:
    """Select, then paste over it. Universal text-editing behaviour, what
    `Cmd+V` does through `TextArea._on_paste`, and what stock
    `TextArea.action_paste` does.

    `ctrl+v` used to `insert` at the caret without removing the selection, so
    selecting `WORD` and pasting `NEW` gave `keep WORDNEW keep` against
    `keep NEW keep` everywhere else — a corrupted buffer the user then has to
    repair by hand (ux round 1, U2). Both shapes are covered: the image route
    had the same defect.
    """
    _stub_clipboard(monkeypatch, text="NEW")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.insert("keep WORD keep")
        editor.selection = Selection((0, 5), (0, 9))
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == "keep NEW keep"
        # The caret lands AFTER the pasted text, so the next keystroke
        # continues past it rather than typing in front of it.
        editor.insert("!")
        assert editor.text == "keep NEW! keep"


@pytest.mark.asyncio
async def test_ctrl_v_replaces_a_live_selection_with_an_image_marker(monkeypatch) -> None:
    """The image shape obeys the same rule as the text shape."""
    _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.insert("keep WORD keep")
        editor.selection = Selection((0, 5), (0, 9))
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == "keep [Image #1, 1568x200]  keep"
        assert len(editor.referenced_images()) == 1


@pytest.mark.asyncio
async def test_a_path_paste_replaces_a_live_selection_like_ctrl_v(tmp_path) -> None:
    """#424 U7: the cmux/path route inserted at the caret without consuming
    the selection, so selecting `WORD` and pasting a file path gave
    `keep WORD[Image #1]  keep` against `keep [Image #1]  keep` on ``ctrl+v``.

    Posted to the APP, not the widget: ``App.on_event`` forwards a
    non-forwarded ``Paste`` to the focused widget, so posting to the widget
    delivers it twice (see this module's docstring).
    """
    path = tmp_path / "shot.png"
    path.write_bytes(_png_bytes())
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.insert("keep WORD keep")
        editor.selection = Selection((0, 5), (0, 9))
        await pilot.pause()
        await _paste(app, pilot, str(path))

        assert editor.text == "keep [Image #1, 1568x200]  keep"
        assert len(editor.referenced_images()) == 1
        # The caret lands AFTER the marker, so the next keystroke continues
        # past it rather than typing in front of it — same as ctrl+v.
        editor.insert("!")
        assert editor.text == "keep [Image #1, 1568x200] ! keep"


@pytest.mark.asyncio
async def test_a_timeout_is_reported_as_a_timeout_not_an_empty_clipboard(monkeypatch) -> None:
    """A read that never finished cannot report what was on the clipboard.

    Under CPU load 8 of 10 reads hit the 2 s ceiling, and every one told a user
    holding a valid screenshot that their clipboard was empty — the same
    wrong-diagnosis class the round-3 D12 correction exists to prevent (ux
    round 1, U3). A retry is the move that helps here and the move that cannot
    help for a genuinely empty clipboard, so the two cannot share a sentence.
    """
    monkeypatch.setattr(
        editor_module,
        "read_clipboard",
        lambda *a, **k: ClipboardContents(timed_out=True),
    )
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert [notice.reason for notice in app.empty_notices] == ["timeout"]


@pytest.mark.asyncio
async def test_a_slow_read_says_it_is_working_and_a_fast_one_stays_silent(monkeypatch) -> None:
    """The composer stops accepting input while the read is in flight, so a
    read the user can perceive owes them an explanation — but the fast read is
    the common one and a card flickering through the shared slot on every
    paste is worse than the pause it describes (ux round 1, U3).

    The card must also be RETIRED on the way out, on every shape. Text is the
    one that used to leak: it raises no `EditorPasteAttached`, so nothing
    withdrew the progress card and it sat over a completed paste for its full
    duration (design round 2, D7).
    """
    monkeypatch.setattr(editor_module, "PASTE_READING_NOTICE_DELAY_S", 0.05)

    def slow(*args, **kwargs):
        time.sleep(0.35)
        return ClipboardContents(text="late")

    monkeypatch.setattr(editor_module, "read_clipboard", slow)
    app = Host()
    async with app.run_test() as pilot:
        app.query_one(Editor).focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)
        assert app.reading_calls == [True, False], (
            "a slow read must raise the card and then retire it; a trailing "
            "True is a progress card left over a finished paste (D7)"
        )

    _stub_clipboard(monkeypatch, text="quick")
    app = Host()
    async with app.run_test() as pilot:
        app.query_one(Editor).focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)
        assert app.reading_calls == [], "a fast read must not flash a card"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contents",
    [
        pytest.param(ClipboardContents(text="late"), id="text"),
        pytest.param(ClipboardContents(), id="empty"),
        pytest.param(ClipboardContents(timed_out=True), id="timeout"),
        pytest.param(ClipboardContents(refused_remote=True), id="remote"),
    ],
)
async def test_every_outcome_retires_the_reading_card(monkeypatch, contents) -> None:
    """A progress card is retired by ANY outcome, not only by ones that
    contradict it.

    The image and file-URL routes were always clean because they post
    `EditorPasteAttached`, which withdraws the shared card. Text, empty,
    timeout and remote post no such message, and text — the commonest ctrl+v
    shape — left "Reading the clipboard…" on screen over a completed paste
    (design round 2, D7). Retirement now lives on the one path every outcome
    passes through, so this is parametrised over all of them rather than
    pinning the single branch that was reported.
    """
    monkeypatch.setattr(editor_module, "PASTE_READING_NOTICE_DELAY_S", 0.05)

    def slow(*args, **kwargs):
        time.sleep(0.3)
        return contents

    monkeypatch.setattr(editor_module, "read_clipboard", slow)
    app = Host()
    async with app.run_test() as pilot:
        app.query_one(Editor).focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert app.reading_calls and app.reading_calls[0] is True
        assert app.reading_calls[-1] is False, "the card outlived its own read"


@pytest.mark.asyncio
async def test_the_reading_card_is_on_screen_while_the_read_is_still_running() -> None:
    """THE TEST THAT WOULD HAVE CAUGHT U3, and the reason it is shaped this way.

    The previous version asserted the notice was RECEIVED
    (`len(reading_notices) == 1`). That was true while the card was invisible
    to every user: `ctrl+v` is an awaited binding action holding the Editor's
    message pump, so a message posted from the timer could not be delivered
    until the action returned, and the card entered the DOM 2-3 ms before the
    paste it was meant to narrate (ux round 2, U3/U8). Asserting delivery
    proved the branch works once entered, not that the user ever reaches it —
    the same shape of gap that let #376 ship.

    So this samples the REAL `Toast` in the REAL `OperatorApp` from inside the
    clipboard read itself — the one place that is genuinely concurrent with the
    stall, because the read runs on a worker thread while the action awaits it.
    Sampling from a timer does not work and that is itself the finding: every
    scheduled callback on either pump is queued behind the blocked handler, so
    a timer-based sample reports the post-read state no matter when it was
    scheduled.

    It fails if the card is not up while the read is running, whatever the
    delivery mechanism, which is precisely what the old
    `len(reading_notices) == 1` assertion could not detect.
    """
    from local_operator.tui.app import CLIPBOARD_READING_NOTICE, OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    read_seconds = 0.6
    samples: list[tuple[float, bool, str]] = []
    toast_box: list[Toast] = []

    def slow(*args, **kwargs):
        # Runs on the worker thread `asyncio.to_thread` hands the read to, so
        # this executes WHILE the action is suspended. Reading the widget's
        # attributes is safe: `show` has already mutated them synchronously on
        # the loop thread before this sampling window opens.
        started = time.monotonic()
        while time.monotonic() - started < read_seconds:
            time.sleep(0.1)
            toast = toast_box[0]
            samples.append((round(time.monotonic() - started, 2), toast.display, toast.message))
        return ClipboardContents(text="done")

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        editor = app.query_one(Editor)
        editor.focus()
        toast_box.append(app.query_one(Toast))
        with (
            monkeypatch_context(editor_module, "read_clipboard", slow),
            monkeypatch_context(editor_module, "PASTE_READING_NOTICE_DELAY_S", 0.05),
        ):
            await pilot.press("ctrl+v")
            for _ in range(40):
                await pilot.pause()

    assert samples, "the read never sampled the card"
    # The first sample is taken at ~0.1 s, before the 0.05 s delay has been
    # crossed by the timer on some runs, so the assertion is that the card is
    # up for the REST of the stall rather than from its first instant.
    during = [sample for sample in samples if sample[0] >= 0.25]
    assert during, "no sample landed inside the stall"
    for at, visible, message in during:
        assert visible and message == CLIPBOARD_READING_NOTICE, (
            f"at {at}s of a {read_seconds}s read the card was {message!r} "
            f"(visible={visible}); it must be on screen DURING the stall, not "
            "delivered after the read it explains has finished"
        )


@pytest.mark.asyncio
async def test_ctrl_v_over_ssh_refuses_the_read_and_reports_it(monkeypatch) -> None:
    """The clipboard on the far end is the SERVER's. Reported as its own
    outcome because "no image on the clipboard" would describe a clipboard
    nobody looked at."""
    _stub_clipboard(monkeypatch, refused_remote=True)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert [notice.reason for notice in app.empty_notices] == ["remote"]


@pytest.mark.asyncio
async def test_ctrl_v_reports_an_image_it_cannot_attach(monkeypatch) -> None:
    """An image WAS on the clipboard and could not be attached — a different
    outcome from "no image", because the two lead to different moves."""
    _stub_clipboard(
        monkeypatch, image=ClipboardImage(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png")
    )
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert [notice.reason for notice in app.empty_notices] == ["unattachable"]


@pytest.mark.asyncio
async def test_ctrl_v_refuses_an_image_too_large_even_after_bounding(monkeypatch) -> None:
    """``MAX_ATTACHMENT_BYTES`` is applied AFTER the resize, which is the only
    place it belongs — but an image that is still over it there must be
    refused rather than sent."""
    data = _png_bytes()
    monkeypatch.setattr(
        editor_module,
        "bound_image_for_model",
        lambda payload, info: (b"\x00" * (editor_module.MAX_ATTACHMENT_BYTES + 1), "image/png", ""),
    )
    _stub_clipboard(monkeypatch, image=ClipboardImage(data, "image/png"))

    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert editor.referenced_images() == []
        assert [notice.reason for notice in app.empty_notices] == ["unattachable"]


@pytest.mark.asyncio
async def test_ctrl_v_reads_the_clipboard_off_the_event_loop(monkeypatch) -> None:
    """The read shells out to osascript/wl-paste/xclip/PowerShell and this is a
    keystroke handler; a 0.6 s Retina read inline is a visible freeze."""
    reader_threads: list[int] = []

    def read_clipboard(*args, **kwargs):
        reader_threads.append(threading.get_ident())
        time.sleep(0.05)
        return ClipboardContents(image=ClipboardImage(_png_bytes(), "image/png"))

    monkeypatch.setattr(editor_module, "read_clipboard", read_clipboard)
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        loop_thread = threading.get_ident()
        await _ctrl_v(app, pilot)

        assert (
            reader_threads and loop_thread not in reader_threads
        ), "the clipboard read must not run on the event loop"
        assert editor.text == "[Image #1, 1568x200] "


@pytest.mark.asyncio
async def test_ctrl_v_is_inert_on_a_read_only_composer(monkeypatch) -> None:
    """The base action honours ``read_only``; rebinding the key must not be a
    way to make a read-only composer writable."""
    counts = _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.read_only = True
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert counts["reads"] == 0, "a read-only composer must not even read the clipboard"


@pytest.mark.asyncio
async def test_ctrl_v_and_the_empty_paste_produce_the_same_marker(monkeypatch) -> None:
    """The two routes share ``_attach_clipboard_image`` precisely so they
    cannot drift: same bound, same marker, same all-or-nothing rule. Two
    implementations is how one route quietly starts attaching unbounded bytes.
    """
    data = _png_bytes(1200, 400)

    async def marker_for(use_ctrl_v: bool) -> str:
        _stub_clipboard(monkeypatch, image=ClipboardImage(data, "image/png"))
        app = Host()
        async with app.run_test() as pilot:
            editor = app.query_one(Editor)
            editor.focus()
            await pilot.pause()
            if use_ctrl_v:
                await _ctrl_v(app, pilot)
            else:
                await _paste(app, pilot, "")
            return editor.text

    assert await marker_for(True) == await marker_for(False)


@pytest.mark.asyncio
async def test_leaving_a_mode_restores_the_resting_placeholder(monkeypatch) -> None:
    """A mode that owns the composer is showing its own voice, and returning to
    rest must go through `resting_placeholder` rather than hardcoding a string.

    The composer no longer advertises the paste key at all (the native Cmd+V
    works wherever the terminal forwards it, and `/help` carries the fallback),
    so the resting copy has one value again. This still asserts the round trip:
    the modes are the reason `resting_placeholder` exists as one authority, and
    a mode exit that invented its own string would be the drift it prevents.
    """
    _stub_clipboard(monkeypatch, text="pasted")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        editor.set_shell_mode(True)
        assert editor.placeholder == editor_module.SHELL_PLACEHOLDER
        editor.set_shell_mode(False)
        assert editor.placeholder == editor_module.DEFAULT_PLACEHOLDER

        await _ctrl_v(app, pilot)
        assert (
            editor.placeholder == editor_module.DEFAULT_PLACEHOLDER
        ), "a paste must not change what the resting composer says"


@pytest.mark.asyncio
async def test_over_budget_clipboard_text_names_its_own_cause(monkeypatch) -> None:
    """A user who copied a huge log and pressed ctrl+v was told the clipboard
    was empty (code round 2, F7). Declining the payload is deliberate — a
    silently truncated paste is damage nobody sees until they read back what
    they sent — but the report has to name what happened."""
    monkeypatch.setattr(
        editor_module,
        "read_clipboard",
        lambda *a, **k: ClipboardContents(text_too_large=True),
    )
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()
        await _ctrl_v(app, pilot)

        assert editor.text == ""
        assert [notice.reason for notice in app.empty_notices] == ["too_large"]


@pytest.mark.asyncio
async def test_every_paste_notice_fits_the_toast_on_one_line() -> None:
    """Every card this feature can raise must fit the toast's content box.

    Two of them did not: `unattachable` (59 cells) and `unreadable` (64)
    wrapped to two lines and clipped the ASCII logo behind the card — the same
    defect D4 was raised for on the SSH copy. They were left in round 1 as
    untouched copy, but this PR is what made them REACHABLE: on base both were
    raised only from the zero-byte paste path that never fires outside cmux
    (design round 2, D8).

    Pinned as a property of the whole family rather than as three string
    assertions, so a future notice cannot reintroduce the wrap.
    """
    from rich.cells import cell_len

    from local_operator.tui.app import CLIPBOARD_READING_NOTICE, OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    reasons = ["nothing", "too_large", "timeout", "remote", "unattachable", "unreadable"]
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        toast = app.query_one(Toast)
        budget = toast.content_cells
        for reason in reasons:
            app.on_editor_paste_empty(EditorPasteEmpty(reason=reason))
            await pilot.pause()
            assert cell_len(toast.message) <= budget, (
                f"{reason!r} notice is {cell_len(toast.message)} cells against a "
                f"{budget}-cell box: it wraps and clips the logo behind it"
            )
        assert cell_len(CLIPBOARD_READING_NOTICE) <= budget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_seconds",
    [
        # Straddles the whole [DELAY, DELAY + MIN) window that D14 lived in,
        # plus a control on either side. The bug was invisible outside it.
        pytest.param(0.20, id="below-the-delay"),
        pytest.param(0.42, id="just-inside-the-window"),
        pytest.param(0.60, id="mid-window"),
        pytest.param(0.72, id="the-0ms-case"),
        pytest.param(0.90, id="past-the-floor"),
    ],
)
async def test_the_minimum_display_floor_never_suppresses_the_outcome_notice(
    monkeypatch, read_seconds
) -> None:
    """THE D10/D14 INTERACTION, pinned as an interaction rather than a constant.

    `PASTE_READING_NOTICE_MIN_S` holds the progress card up for a minimum time
    so a read landing just past the delay does not flash it for ~42 ms (D10).
    That defers the retirement — and the retirement is a `Toast.withdraw`,
    which matches on OWNER. While the progress card and the failure notices
    shared `COMPOSER_PASTE_NOTICE`, a read finishing inside
    `[DELAY, DELAY + MIN)` let the deferred withdrawal fire *after* the failure
    card had taken the slot and pull down a card it was never meant to touch.

    Measured before the fix, with a 5 s courtesy duration: the failure card
    lived 384 ms at a 0.36 s read, 12 ms at 0.70 s, and **0 ms at 0.74 s**
    (design round 3, D14). That is issue #372's original symptom restored — a
    keypress that produces a visible pause and then says nothing.

    So this asserts the property the two mechanisms have to satisfy TOGETHER:
    whatever the read duration, the user ends up looking at the outcome. It is
    parametrised across the window rather than asserting the constant, because
    the constant was never the bug; the interaction between the floor and the
    shared owner was, and a test on the number alone would have stayed green
    through it.
    """
    monkeypatch.setattr(editor_module, "PASTE_READING_NOTICE_DELAY_S", 0.35)
    monkeypatch.setattr(editor_module, "PASTE_READING_NOTICE_MIN_S", 0.4)

    def slow(*args, **kwargs):
        time.sleep(read_seconds)
        return ClipboardContents()

    monkeypatch.setattr(editor_module, "read_clipboard", slow)

    from local_operator.tui.app import CLIPBOARD_READING_NOTICE, OperatorApp
    from local_operator.tui.widgets.toast import Toast
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.query_one(Editor).focus()
        await pilot.pause()
        toast = app.query_one(Toast)

        await _ctrl_v(app, pilot)
        # Well past the floor's own deadline, so a deferred withdrawal has had
        # every chance to fire. Sampled rather than checked once: the defect
        # was a card that appeared and was then destroyed, which a single
        # reading taken too early would have called a pass.
        deadline = time.monotonic() + editor_module.PASTE_READING_NOTICE_MIN_S + 0.3
        while time.monotonic() < deadline:
            await pilot.pause()
            await asyncio.sleep(0.02)

        assert toast.display, (
            f"a {read_seconds}s read left NOTHING on screen: the deferred "
            "minimum-display retirement destroyed the outcome notice, which is "
            "the dead-keypress symptom this feature exists to remove (D14)"
        )
        assert (
            toast.message != CLIPBOARD_READING_NOTICE
        ), "the progress card outlived the read it describes"
        assert toast.message == "Nothing on the clipboard to paste."


@pytest.mark.asyncio
async def test_the_reading_card_and_the_outcome_notice_do_not_share_an_owner() -> None:
    """The structural half of D14, pinned so the fix cannot be undone by
    tidying two owners back into one.

    `Toast.withdraw` matches on owner and nothing else, so a DEFERRED
    withdrawal is only safe if it cannot name a card raised by someone else.
    Separate owners make D14 unrepresentable rather than merely fixed: no
    timing, however unlucky, lets the progress card's retirement reach the
    failure notice that replaced it.
    """
    from local_operator.tui.app import COMPOSER_PASTE_NOTICE, COMPOSER_READING_NOTICE

    assert (
        COMPOSER_READING_NOTICE is not COMPOSER_PASTE_NOTICE
    ), "the progress card's deferred retirement would reach the outcome notice"
