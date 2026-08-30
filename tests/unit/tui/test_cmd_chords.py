"""The macOS-native clipboard chords: ``Cmd+V`` and ``Cmd+C`` in the composer.

Both arrive as ``super+v``/``super+c`` on terminals that implement the kitty
keyboard protocol, and not at all on terminals that do not (Terminal.app
swallows them whole). The app binds them IN ADDITION to the portable ``Ctrl+``
chords, never instead of them, so these tests always assert the pair.

The byte-level captures behind that claim live in
``docs/evidence/cmd-chords/MEASURED.md``. The parser test below is the one that
keeps this file honest across a Textual upgrade: everything else here presses a
KEY NAME, which proves routing but would go on passing if Textual changed the
encoding it decodes into that name.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image
from textual.app import App, ComposeResult
from textual.widgets.text_area import Selection

from local_operator.clipboard import ClipboardContents, ClipboardImage
from local_operator.tui.widgets import editor as editor_module
from local_operator.tui.widgets.editor import Editor, EditorCopied, InterruptRequested


def _png_bytes(width: int = 1568, height: int = 200) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (30, 30, 40)).save(buffer, "PNG")
    return buffer.getvalue()


class Host(App[None]):
    """Records what the editor posts, so the copy receipt and the interrupt are
    both assertable without standing up the whole app."""

    def __init__(self) -> None:
        super().__init__()
        self.copied: list[str] = []
        self.interrupts: int = 0

    def compose(self) -> ComposeResult:
        yield Editor()

    def on_editor_copied(self, message: EditorCopied) -> None:
        self.copied.append(message.text)

    def on_interrupt_requested(self, message: InterruptRequested) -> None:
        self.interrupts += 1


def _stub_clipboard(monkeypatch, *, image=None, text="") -> dict[str, int]:
    counts = {"reads": 0}

    def read_clipboard(*args, **kwargs):
        counts["reads"] += 1
        return ClipboardContents(image=image, paths=(), text=text, refused_remote=False)

    monkeypatch.setattr(editor_module, "read_clipboard", read_clipboard)
    return counts


async def _press(pilot, key: str) -> None:
    await pilot.press(key)
    for _ in range(20):
        await pilot.pause()


# -- the encoding this whole feature rests on ---------------------------------
@pytest.mark.parametrize(
    ("sequence", "expected"),
    (
        ("\x1b[118;9u", "super+v"),
        ("\x1b[99;9u", "super+c"),
        ("\x1b[118;5u", "ctrl+v"),
        ("\x1b[99;5u", "ctrl+c"),
    ),
)
def test_the_csi_u_bytes_decode_to_the_keys_the_composer_binds(sequence, expected) -> None:
    """Pin the terminal encoding to the key NAME, so a Textual upgrade that
    changes it fails here rather than silently unbinding the Mac chords.

    Every other test in this file presses a key by name, which proves the app
    routes that name correctly and proves nothing about which bytes produce it.
    These are the exact sequences captured off a real pty from Ghostty with the
    kitty keyboard protocol enabled: ``CSI <codepoint> ; <modifiers> u`` with
    modifiers ``1 + bitmask`` and Super=8, so 118/9 is literally Cmd+v.
    """
    from textual._xterm_parser import XTermParser

    events = list(XTermParser().feed(sequence))
    assert [getattr(event, "key", None) for event in events] == [expected]


# -- Cmd+V --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_super_v_attaches_the_image_on_the_system_clipboard(monkeypatch) -> None:
    """The reported bug: Cmd+V did nothing because nothing was bound to it.

    The key reaches the app on a kitty-protocol terminal (measured: an
    image-only pasteboard makes Ghostty deliver ``ESC[118;9u``), so the only
    thing missing was the binding.
    """
    counts = _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        await _press(pilot, "super+v")

        assert editor.text == "[Image #1, 1568x200] "
        assert counts["reads"] == 1, "super+v must read the SYSTEM clipboard"


@pytest.mark.asyncio
async def test_both_paste_chords_resolve_to_the_same_action(monkeypatch) -> None:
    """Neither chord replaces the other: Ctrl+V is the portable baseline that
    every terminal delivers, and Cmd+V is bound beside it for the terminals
    that forward it. One action serves both so they cannot drift."""
    _stub_clipboard(monkeypatch, image=ClipboardImage(_png_bytes(), "image/png"))
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        for key in ("ctrl+v", "super+v"):
            bound = editor._bindings.key_to_bindings.get(key) or []
            assert [binding.action for binding in bound] == ["system_paste"], (
                f"{key} must resolve to the system paste alone; TextArea's "
                "action_paste inserts a buffer the system clipboard never fills"
            )


@pytest.mark.asyncio
async def test_a_text_paste_is_not_doubled_by_the_super_v_binding(monkeypatch) -> None:
    """The load-bearing measurement, pinned.

    A terminal with TEXT on the clipboard handles Cmd+V itself and
    bracket-pastes the text without forwarding the key (measured: 21 bytes of
    ``ESC[200~…ESC[201~`` and no key event). The two cases are disjoint, so the
    binding cannot double-paste. This asserts the other half — that the paste
    EVENT alone inserts exactly once and does not also run the clipboard read —
    so the disjointness is the terminal's only job.
    """
    from textual import events as textual_events

    counts = _stub_clipboard(monkeypatch, text="HELLOTEXT")
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        app.post_message(textual_events.Paste("HELLOTEXT"))
        for _ in range(20):
            await pilot.pause()

        assert editor.text == "HELLOTEXT"
        assert counts["reads"] == 0, "a bracketed text paste must not read the clipboard"


# -- Cmd+C --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_super_c_copies_a_live_range_and_emits_the_receipt() -> None:
    """Cmd+C copies and SAYS SO. The receipt matters as much as the clipboard
    write: the field report was that the feedback was missing."""
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("hello world")
        editor.selection = Selection((0, 0), (0, 5))
        await pilot.pause()

        await _press(pilot, "super+c")

        assert app.copied == ["hello"], "super+c must post the copy the app turns into a toast"


@pytest.mark.asyncio
async def test_both_copy_chords_collapse_a_click_chain_selection() -> None:
    """THE defect this change fixes for Cmd+C, which already copied.

    ``TextArea`` binds ``ctrl+c,super+c`` to ``copy``, so Cmd+C already reached
    ``action_copy`` and already produced a receipt. But ``_on_key`` had no
    ``super+c`` branch to consume the key, so the press fell straight through
    to that binding and skipped the click-chain collapse that hands the key
    back to the draft and interrupt rungs. (``_on_key`` runs FIRST; a binding
    fires only on the keys it does not consume. An earlier wording here had
    that rule inverted — code round 1 F2, code round 2 F6.) Measured before the
    fix: ctrl+c left the selection collapsed and super+c left it live, which is
    R1-2's bug reintroduced for the other chord.

    Parametrised over both keys deliberately: the assertion is that the two
    chords are ONE behaviour, and a test that only pressed super+c would pass
    against a fix that broke ctrl+c.
    """
    for key in ("ctrl+c", "super+c"):
        app = Host()
        async with app.run_test() as pilot:
            editor = app.query_one(Editor)
            editor.focus()
            editor.load_text("hello world")
            # The state a double-click leaves: a real range AND the click-chain
            # claim that makes it reflexive rather than deliberate.
            editor._click_selection = Selection((0, 6), (0, 11))
            editor.selection = Selection((0, 6), (0, 11))
            await pilot.pause()

            await _press(pilot, key)

            assert app.copied == ["world"], f"{key} must copy the range"
            assert editor.selection.start == editor.selection.end, (
                f"{key} must collapse a click-chain range after copying, or the "
                "key can never reach the draft and interrupt rungs again"
            )


@pytest.mark.asyncio
async def test_ctrl_c_still_interrupts_when_no_range_is_live() -> None:
    """The interrupt is Ctrl+C's alone and must survive this change: it is the
    first rung of the exit ladder."""
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("draft text")
        editor.selection = Selection((0, 10), (0, 10))
        await pilot.pause()

        await _press(pilot, "ctrl+c")

        assert app.interrupts == 1, "ctrl+c with no selection must still interrupt"


@pytest.mark.asyncio
async def test_super_c_never_interrupts_and_never_clears_the_draft() -> None:
    """Cmd+C is not an interrupt gesture on macOS, and must not become one.

    A Cmd+C that fell through to the interrupt rung would be a new way to lose
    a draft — the exact class of failure D17 and the exit ladder's ordering
    exist to prevent. With nothing selected the chord does nothing at all.
    """
    app = Host()
    async with app.run_test() as pilot:
        editor = app.query_one(Editor)
        editor.focus()
        editor.load_text("draft text")
        editor.selection = Selection((0, 10), (0, 10))
        await pilot.pause()

        await _press(pilot, "super+c")

        assert app.interrupts == 0, "super+c must not carry the interrupt meaning"
        assert editor.text == "draft text", "super+c must never clear the draft"
        assert app.copied == [], "an empty selection is not a copy"
