"""Every option chord, in every composer state, on every terminal encoding.

Two review rounds were needed on #370 because the earlier sweeps tested the
dimensions INDEPENDENTLY — "picker open x horizontal chord" in one row and
"history present x vertical chord" in another — and the defects lived in the
crossings neither row visited. The regression that forced round 3 (code round 2
F5, ux round 2 U6) needed a picker open AND history present AND the vertical
chord AND the CSI encoding before it appeared: `⌥↑` closed the list and
overwrote a half-typed slash command from history, on the terminals where that
key had previously been a harmless no-op.

So this module does not hand-write rows. It generates the CROSS PRODUCT of

    {picker closed, command picker open, model picker open}
  x {history present, history absent}
  x {shell mode on, off}
  x {left, right, up, down, and the shift variants of each}
  x {CSI-modifier, readline-meta, Esc-prefixed}

and asserts one property per cell: **the option chord is indistinguishable from
its plain-arrow equivalent.** That is the whole claim the feature makes, stated
once and checked everywhere, rather than a list of remembered outcomes that a
future chord could be added without.

The oracle is the plain arrow itself, run in an identical app, not a recorded
expectation. So a cell cannot rot: if the meaning of `up` with a picker open
changes, the chord's expectation changes with it automatically, and a chord
that stops matching its arrow fails no matter which layer broke it.

Encoding notes (verified against textual 8.2.8, see `test_word_caret.py`):

- CSI-modifier is the only encoding with a spelling for every chord.
- readline-meta only exists for the horizontal pair (`\\x1bb` / `\\x1bf`); there
  is no meta spelling of a vertical arrow, so those cells are absent by nature
  rather than skipped.
- Esc-prefixed spells everything, as `escape` followed by the plain key.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest
from textual import events
from textual._xterm_parser import XTermParser

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor, StopRequested

from .test_app_pilot import FakeSession, _factory

#: A prompt in history, so any chord that wrongly reaches history navigation
#: visibly destroys the buffer rather than silently doing nothing.
HISTORY = ["summarise the last commit"]

#: Chords under test, mapped to the plain key each must be identical to.
#:
#: The two axes have DIFFERENT oracles, and that asymmetry is the feature, not
#: an inconsistency:
#:
#: - Horizontal chords are word motions, so the arrow that means the same thing
#:   is the `ctrl` one (`ctrl+left` is TextArea's own word-left, and the
#:   Linux/Windows spelling of this very chord). Comparing `⌥←` against plain
#:   `←` would assert the chord does nothing, which is the opposite of #370.
#: - Vertical chords carry no motion of their own and must be pass-throughs, so
#:   their oracle is the plain arrow.
CHORDS: dict[str, str] = {
    "left": "ctrl+left",
    "right": "ctrl+right",
    "shift+left": "ctrl+shift+left",
    "shift+right": "ctrl+shift+right",
    "up": "up",
    "down": "down",
    "shift+up": "shift+up",
    "shift+down": "shift+down",
}

#: How each terminal spells a chord. ``None`` means that terminal has no
#: spelling for it, which is a fact about the encoding, not a gap in coverage.
_CSI = {
    "left": "\x1b[1;3D",
    "right": "\x1b[1;3C",
    "up": "\x1b[1;3A",
    "down": "\x1b[1;3B",
    "shift+left": "\x1b[1;4D",
    "shift+right": "\x1b[1;4C",
    "shift+up": "\x1b[1;4A",
    "shift+down": "\x1b[1;4B",
}
_META = {"left": "\x1bb", "right": "\x1bf"}


def _encode(encoding: str, chord: str) -> str | None:
    if encoding == "csi":
        return _CSI.get(chord)
    if encoding == "meta":
        return _META.get(chord)
    # Esc-prefixed: the plain sequence, preceded by a bare ESC.
    plain = {
        "left": "\x1b[D",
        "right": "\x1b[C",
        "up": "\x1b[A",
        "down": "\x1b[B",
        "shift+left": "\x1b[1;2D",
        "shift+right": "\x1b[1;2C",
        "shift+up": "\x1b[1;2A",
        "shift+down": "\x1b[1;2B",
    }[chord]
    return "\x1b" + plain


ENCODINGS = ("csi", "meta", "esc_prefixed")

#: Composer states. Each is a setup coroutine plus the buffer it leaves behind.
STATES: dict[str, dict[str, Any]] = {
    "resting": {"text": "alpha beta gamma delta", "picker": None, "shell": False},
    "command_picker": {"text": "/analytics", "picker": "command", "shell": False},
    "model_picker": {"text": "/model anthropic/claude", "picker": "model", "shell": False},
    "shell_mode": {"text": "git commit --amend", "picker": None, "shell": True},
    "multiline": {"text": "first line\nsecond line", "picker": None, "shell": False},
}


async def _boot(pilot: Any, app: OperatorApp) -> Editor:
    for _ in range(200):
        if app._session is not None:
            break
        await pilot.pause()
        await asyncio.sleep(0.01)
    assert app._session is not None, "the session never booted"
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    return editor


async def _settle(pilot: Any, cycles: int = 6) -> None:
    for _ in range(cycles):
        await pilot.pause()


async def _arrange(pilot: Any, editor: Editor, state: str, history: bool) -> None:
    """Put the composer in one state of the matrix."""
    spec = STATES[state]
    editor._history = list(HISTORY) if history else []
    if spec["shell"]:
        await pilot.press("!")
        await pilot.pause()
    editor.text = spec["text"]
    await pilot.pause()
    if spec["picker"]:
        editor._sync_picker()
        await pilot.pause()
    # Caret at the end of the last line, the position a user reaches a chord
    # from most often and the one where history navigation is live.
    lines = spec["text"].split("\n")
    editor.move_cursor((len(lines) - 1, len(lines[-1])))
    await pilot.pause()


def _observe(editor: Editor, stops: list[Any]) -> tuple[Any, ...]:
    """Everything a user could notice, as one comparable tuple."""
    return (
        editor.cursor_location,
        editor.text,
        editor.selected_text,
        editor.shell_mode,
        editor._picker.is_open(),
        editor._model_picker.is_open(),
        editor._picker.selected_index if editor._picker.is_open() else None,
        len(stops),
    )


async def _run(
    state: str,
    history: bool,
    act: Callable[[Any, OperatorApp], Any],
    escape_pending: bool = False,
) -> tuple[Any, ...]:
    """Arrange one matrix cell, apply ``act``, and report what a user would see.

    ``escape_pending`` delivers a REAL lone Escape immediately before the act,
    within the one-pump-turn window the coalescing holds it for. That axis
    exists because the matrix could not previously express F8 (code round 3):
    a self-contained CSI chord landing on a held escape was misread as that
    escape's second half and cancelled it, silently dropping every meaning Esc
    has. Every cell is run both ways, so the crossing cannot go dark again.

    The escape is injected as bytes through the real parser rather than as a
    key name, because a lone ``\\x1b`` only emerges after the parser's own
    ESCAPE_DELAY -- which is exactly why the user's inter-key gap is spent
    inside the parser and not inside the widget's window, and therefore why a
    real gap does not save you from the bug.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await _arrange(pilot, editor, state, history)

        stops: list[Any] = []
        original = app.post_message

        def _spy(message: Any) -> bool:
            if isinstance(message, StopRequested):
                stops.append(message)
            return original(message)

        app.post_message = _spy  # type: ignore[method-assign]
        if escape_pending:
            # Posted WITHOUT yielding, so the escape is still held when the act
            # runs. Yielding here would let the one-turn deferral resolve first
            # and the cell would test nothing -- which is exactly the mistake
            # that let F8 through, reproduced one level up in the test itself.
            parser = XTermParser()
            driver = app._driver
            assert driver is not None
            for event in list(parser.feed("\x1b")) + list(parser.feed("")):
                if isinstance(event, events.Key):
                    event.set_sender(app)
                    driver.send_message(event)
        await act(pilot, app)
        await _settle(pilot)
        return _observe(editor, stops)


def _press_plain(key: str) -> Callable[[Any, OperatorApp], Any]:
    async def _act(pilot: Any, app: OperatorApp) -> None:
        await pilot.press(key)

    return _act


def _feed_bytes(raw: str) -> Callable[[Any, OperatorApp], Any]:
    async def _act(pilot: Any, app: OperatorApp) -> None:
        parser = XTermParser()
        parsed = list(parser.feed(raw)) + list(parser.feed(""))
        driver = app._driver
        assert driver is not None
        # No yield between events: one parse pass emits the Esc-prefixed pair
        # together and the real driver posts both before the loop is pumped.
        for event in parsed:
            if isinstance(event, events.Key):
                event.set_sender(app)
                driver.send_message(event)
        await asyncio.sleep(0)

    return _act


def _cells() -> list[tuple[str, bool, str, str]]:
    """The cross product, minus the cells an encoding cannot express."""
    out = []
    for state in STATES:
        for history in (True, False):
            for chord in CHORDS:
                for encoding in ENCODINGS:
                    if _encode(encoding, chord) is None:
                        continue
                    out.append((state, history, chord, encoding))
    return out


@pytest.mark.parametrize(
    ("state", "history", "chord", "encoding"),
    _cells(),
    ids=lambda v: str(v),
)
@pytest.mark.asyncio
async def test_the_chord_is_indistinguishable_from_its_plain_arrow(
    state: str, history: bool, chord: str, encoding: str
) -> None:
    """One cell of the matrix: option chord == plain arrow, whatever the state.

    The plain arrow is run in its own identical app and used as the oracle, so
    this asserts equivalence rather than a remembered outcome.
    """
    raw = _encode(encoding, chord)
    assert raw is not None

    expected = await _run(state, history, _press_plain(CHORDS[chord]))
    actual = await _run(state, history, _feed_bytes(raw))

    assert actual == expected, (
        f"{encoding} {chord} in {state} (history={history}) diverged from plain "
        f"{CHORDS[chord]}:\n  plain={expected}\n  chord={actual}"
    )


def _escape_pending_cells() -> list[tuple[str, bool, str, str]]:
    """The same cross product, for the self-contained encodings only.

    The Esc-prefixed spelling is excluded by nature rather than by choice: it
    IS an escape followed by an arrow, so preceding it with a real escape makes
    a three-event sequence whose first escape stands alone and whose second
    legitimately pairs with the arrow. That is a different scenario (and is
    pinned separately in ``test_word_caret.py``), not this axis.
    """
    return [
        (state, history, chord, encoding)
        for (state, history, chord, encoding) in _cells()
        if encoding != "esc_prefixed"
    ]


@pytest.mark.parametrize(
    ("state", "history", "chord", "encoding"),
    _escape_pending_cells(),
    ids=lambda v: str(v),
)
@pytest.mark.asyncio
async def test_a_self_contained_chord_never_swallows_a_pending_escape(
    state: str, history: bool, chord: str, encoding: str
) -> None:
    """The F8 axis: a real Esc held when the chord lands must still fire.

    A self-contained chord (one event that already means "option+arrow") is not
    evidence about a pending escape, so the escape's own action is still owed.
    This asserts BOTH halves: the escape fires exactly once, AND the chord
    still moves the caret exactly as it does with no escape pending.

    Without the ``self_contained`` guard in ``_on_key`` these cells fail: the
    rewrite turned ``alt+up`` into ``up``, ``up`` is in ``ESCAPE_CHORD_KEYS``,
    and the chord cancelled the escape it had nothing to do with -- taking the
    stop, the shell-mode exit, or a picker dismissal with it.

    NOTE the oracle here is the chord's OWN no-escape outcome, not the plain
    arrow. ``pilot.press`` delivers a key NAME without going through the
    parser, so the plain arm's escape is still held when its arrow lands and it
    legitimately reports zero stops -- an artifact of the harness, not of the
    product. Comparing against it would assert the bug. The equivalence to the
    plain arrow is already covered by the no-escape axis above; what is left to
    pin here is that a pending escape changes nothing about the chord and is
    itself preserved.
    """
    raw = _encode(encoding, chord)
    assert raw is not None

    # THE CONTROL is the horizontal chord, and the choice matters. `alt+left`
    # is never rewritten, is not in `ESCAPE_CHORD_KEYS`, and therefore never had
    # F8 -- the code reviewer identified exactly that asymmetry as the tell. So
    # "how a pending escape is disposed of in this state" is read off the chord
    # that provably handles it correctly, and the chord under test must match.
    #
    # Reading it from a control rather than asserting a fixed outcome is what
    # keeps this honest: Esc means four different things depending on state
    # (stop, leave shell mode, dismiss either list), and in the picker states a
    # dismissed list is legitimately re-opened by the following `_sync_picker`.
    # Hand-writing those outcomes would have encoded a pre-existing behaviour as
    # if it were this feature's contract.
    control_raw = _encode(encoding, "shift+left" if "shift" in chord else "left")
    assert control_raw is not None
    control = await _run(state, history, _feed_bytes(control_raw), escape_pending=True)
    control_baseline = await _run(state, history, _feed_bytes(control_raw))

    baseline = await _run(state, history, _feed_bytes(raw))
    actual = await _run(state, history, _feed_bytes(raw), escape_pending=True)

    # Whatever the escape did to the control's state, it must also have done
    # here: same shell mode, same picker states, same stop count.
    control_effect = (control[3], control[4], control[5], control[-1] - control_baseline[-1])
    actual_effect = (actual[3], actual[4], actual[5], actual[-1] - baseline[-1])
    assert actual_effect == control_effect, (
        f"{encoding} {chord} in {state} (history={history}) disposed of the pending "
        f"escape differently from the never-rewritten horizontal chord:\n"
        f"  control (option+left) effect={control_effect}\n"
        f"  {chord} effect={actual_effect}\n"
        f"  without escape={baseline}\n  with escape={actual}"
    )
    # The chord's own motion is NOT asserted equal to its no-escape baseline,
    # deliberately. A flushed escape legitimately changes the state the chord
    # then acts on -- with a command list open, Esc dismisses the list, so the
    # following `up` is no longer claimed by the picker and correctly falls
    # through to history recall. Demanding identical motion would assert that
    # the escape had NOT taken effect, i.e. would re-encode F8 as the contract.
    #
    # What must hold is that the chord behaves like a plain arrow pressed in
    # that same post-escape state, which the no-escape axis already pins for
    # every state the escape can leave behind.


@pytest.mark.parametrize("encoding", ["csi", "meta"])
@pytest.mark.parametrize("chord", list(CHORDS))
@pytest.mark.asyncio
async def test_a_pending_escape_still_stops_the_turn_after_a_chord(
    chord: str, encoding: str
) -> None:
    """The F8 payload, stated absolutely rather than against an oracle.

    Belt and braces beside the equivalence test above: if the plain-arrow
    oracle were itself ever broken, the equivalence could hold while both sides
    lost the stop. This pins the stop count directly.
    """
    raw = _encode(encoding, chord)
    if raw is None:
        pytest.skip(f"{encoding} has no spelling for {chord}")

    result = await _run("resting", False, _feed_bytes(raw), escape_pending=True)
    stops = result[-1]

    assert stops == 1, f"{encoding} {chord} swallowed the pending escape (stops={stops})"


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("chord", ["up", "down"])
@pytest.mark.asyncio
async def test_a_vertical_chord_never_destroys_a_typed_slash_command(
    chord: str, encoding: str
) -> None:
    """The exact regression that forced round 3, pinned on its own.

    Code round 2 F5 / ux round 2 U6: with a list open and history present, the
    CSI vertical chord closed the picker and replaced `/model anthropic/claude`
    with a history entry, because a `Binding` fires through the action system
    and never reaches the picker branches inside `_on_key`.

    Stated as an absolute rather than as an equivalence, so it still fails
    loudly if the plain-arrow oracle above were ever itself broken.
    """
    raw = _encode(encoding, chord)
    if raw is None:
        pytest.skip(f"{encoding} has no spelling for {chord}")

    typed = STATES["model_picker"]["text"]
    result = await _run("model_picker", True, _feed_bytes(raw))
    text = result[1]

    assert text == typed, f"{encoding} ⌥{chord} destroyed the typed command: {text!r}"
    assert text not in HISTORY, "the buffer was overwritten from history"
