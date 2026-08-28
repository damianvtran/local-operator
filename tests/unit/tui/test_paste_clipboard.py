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

import pytest
from PIL import Image
from textual import events
from textual.app import App, ComposeResult

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


class Host(App[None]):
    """Records the notices the editor posts, so the feedback is assertable
    without standing up the whole app's toast plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.empty_notices: list[EditorPasteEmpty] = []
        self.attached_notices: list[EditorPasteAttached] = []

    def compose(self) -> ComposeResult:
        yield Editor()

    def on_editor_paste_empty(self, message: EditorPasteEmpty) -> None:
        self.empty_notices.append(message)

    def on_editor_paste_attached(self, message: EditorPasteAttached) -> None:
        self.attached_notices.append(message)


async def _paste(app: App[None], pilot, text: str) -> None:
    app.post_message(events.Paste(text))
    for _ in range(20):
        await pilot.pause()


def _stub_clipboard(monkeypatch, *, image=None, paths=None, refused_remote=False) -> dict[str, int]:
    """Replace the clipboard read; record the calls so routing is assertable."""
    counts = {"reads": 0}

    def read_clipboard(*args, **kwargs):
        counts["reads"] += 1
        return ClipboardContents(
            image=image,
            paths=tuple(paths or ()),
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

        assert counts["reads"] == 1
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
