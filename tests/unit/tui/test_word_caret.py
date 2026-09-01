"""Word-wise caret movement works on every terminal encoding of option+arrow.

Issue #370. There is no single byte sequence for ⌥←: which one a user's
terminal emits is a preference in that terminal, not a property of the
platform, and the composer has to handle all of them. These tests feed the
LITERAL BYTES each emulator writes through Textual's real ``XTermParser`` into
the real ``OperatorApp``, so "this works in Ghostty" is checked against
Ghostty's actual encoding rather than against a key name someone assumed it
produces.

The encodings, and who sends them (verified against textual 8.2.8):

===============  =========================  =====================================
bytes            parses to                  terminals
===============  =========================  =====================================
``\\x1b[1;3D``    ``alt+left``               Ghostty, kitty, WezTerm, iTerm2 in
                                            CSI mode
``\\x1bb``        ``ctrl+left``              iTerm2's default ⌥← preset,
                                            Terminal.app "Use Option as Meta"
``\\x1b\\x1b[D``   ``escape`` THEN ``left``   iTerm2 "Esc+", Terminal.app "Esc+",
                                            and any terminal once
                                            ``TEXTUAL_DISABLE_KITTY_KEY`` is set
``\\x1b[98;3u``   ``alt+b``                  Ghostty on its DEFAULT settings,
                                            once the kitty keyboard protocol
                                            is negotiated
===============  =========================  =====================================

The third row is the defect #370/#375 fixed: the composer used to run its
escape action (post ``StopRequested`` — abort the agent's turn) and then move
the caret one character, so ⌥← to fix a typo killed the running turn. The
regression guard is ``stop_requested`` staying empty on those rows.

The fourth row is issue #518, and it is not a terminal preference like the
others — it is Ghostty's factory default. Ghostty ships
``keybind = alt+arrow_left=esc:b``, rewriting ⌥← into readline's meta-b and
DESTROYING the fact that it was an arrow before any application sees the key.
Under the legacy encoding that damage is invisible here, because the resulting
``\\x1bb`` happens to parse to ``ctrl+left``, which ``TextArea`` binds (row 2).
Textual negotiates the kitty protocol, though, and the same chord then arrives
as ``\\x1b[98;3u`` → ``alt+b``, which nothing bound — so word motion silently
did nothing on a stock Ghostty. Note ⌥⇧← was unaffected (Ghostty has no
``alt+shift+arrow`` default to corrupt), which is why selection-by-word worked
while movement-by-word did not.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from textual import events
from textual._xterm_parser import XTermParser
from textual.widget import Widget

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor, StopRequested

from .test_app_pilot import FakeSession, _factory

#: Four words, so a single word-move is unambiguous and lands on a known column.
SAMPLE = "alpha beta gamma delta"
#: Column of the "d" in "delta" — where one word-left from the end must land.
DELTA_START = SAMPLE.index("delta")
#: Column of the "g" in "gamma" — where a second word-left lands.
GAMMA_START = SAMPLE.index("gamma")


async def _boot(pilot: Any, app: OperatorApp) -> Editor:
    """Wait for the session and focus the composer.

    Waits on the boot WORKER, not on a 2 s frame budget. ``on_mount`` hands
    ``_boot_session`` to ``run_worker(group="session")`` so first paint does
    not wait on the factory, and every assertion below reads through that
    session. The previous 200 × 10 ms loop was a bet that 2 s outlasts the
    factory, which is the same class #461 converted in ``test_subagent_stats``.
    Under contention that budget lost as a stylesheet lookup against an
    unmounted editor (``KeyError: No 'text-area--gutter' key``, #463).
    """
    workers = [w for w in app.workers if w.group == "session"]
    if workers:
        await app.workers.wait_for_complete(workers)
    assert app._session is not None, (
        "no session worker is pending and no session was adopted — the "
        "boot worker never ran, so waiting here would have waited on nothing"
    )
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    return editor


async def _feed(app: OperatorApp, raw: str) -> None:
    """Parse ``raw`` with the real parser and inject it as the driver would.

    ``pilot.press`` takes key NAMES, which is exactly the layer these tests
    exist to get underneath: the whole question is what a terminal's bytes turn
    into, so the bytes have to go through ``XTermParser`` rather than a name
    chosen by the test author.
    """
    parser = XTermParser()
    parsed = list(parser.feed(raw)) + list(parser.feed(""))
    driver = app._driver
    assert driver is not None
    # Sent WITHOUT yielding between events, which is what the real driver does:
    # one parse pass emits the chord's `escape` and `left` together and posts
    # both before the loop is pumped. Yielding here would let the escape resolve
    # before its own arrow was even sent — testing a sequence no terminal emits.
    for event in parsed:
        if isinstance(event, events.Key):
            event.set_sender(app)
            driver.send_message(event)
    await asyncio.sleep(0)


async def _settle(pilot: Any, cycles: int = 5) -> None:
    """Pump the message loop so a one-turn-deferred escape action has fired.

    Only a handful of turns are needed: the escape action is deferred by a
    single ``call_later``, not by a wall-clock window.
    """
    for _ in range(cycles):
        await pilot.pause()


def _watch_stops(app: OperatorApp) -> list[StopRequested]:
    """Record every ``StopRequested`` the composer posts."""
    seen: list[StopRequested] = []
    original = app.post_message

    def _spy(message: Any) -> bool:
        if isinstance(message, StopRequested):
            seen.append(message)
        return original(message)

    app.post_message = _spy  # type: ignore[method-assign]
    return seen


@pytest.mark.parametrize(
    ("raw", "expected_column", "terminals"),
    [
        ("\x1b[1;3D", DELTA_START, "Ghostty / kitty / WezTerm / iTerm2 CSI mode"),
        ("\x1bb", DELTA_START, "iTerm2 default preset / Terminal.app Option-as-Meta"),
        ("\x1b\x1b[D", DELTA_START, "iTerm2 Esc+ / Terminal.app Esc+"),
        ("\x1b[98;3u", DELTA_START, "Ghostty default esc:b under the kitty protocol"),
    ],
    ids=["csi-modifier-3", "readline-meta", "escape-prefixed", "kitty-csi-u-meta"],
)
@pytest.mark.asyncio
async def test_option_left_moves_one_word_on_every_encoding(
    raw: str, expected_column: int, terminals: str
) -> None:
    """⌥← moves the caret one word left, whichever bytes the terminal sends."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        editor.move_cursor((0, len(SAMPLE)))
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, raw)
        await _settle(pilot)

        assert editor.cursor_location == (0, expected_column), terminals
        # The regression guard: on the Esc-prefixed encoding this used to abort
        # the agent's turn before moving the caret.
        assert stops == [], f"⌥← must not stop the turn ({terminals})"


@pytest.mark.parametrize(
    ("raw", "terminals"),
    [
        ("\x1b[1;3C", "Ghostty / kitty / WezTerm / iTerm2 CSI mode"),
        ("\x1bf", "iTerm2 default preset / Terminal.app Option-as-Meta"),
        ("\x1b\x1b[C", "iTerm2 Esc+ / Terminal.app Esc+"),
        ("\x1b[102;3u", "Ghostty default esc:f under the kitty protocol"),
    ],
    ids=["csi-modifier-3", "readline-meta", "escape-prefixed", "kitty-csi-u-meta"],
)
@pytest.mark.asyncio
async def test_option_right_moves_one_word_on_every_encoding(raw: str, terminals: str) -> None:
    """⌥→ moves the caret one word right, whichever bytes the terminal sends."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        editor.move_cursor((0, 0))
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, raw)
        await _settle(pilot)

        assert editor.cursor_location[1] > 0, terminals
        assert stops == [], f"⌥→ must not stop the turn ({terminals})"


@pytest.mark.parametrize(
    ("raw", "terminals"),
    [
        ("\x1b[1;4D", "Ghostty / kitty / WezTerm / iTerm2 CSI mode"),
        ("\x1b\x1b[1;2D", "iTerm2 Esc+ / Terminal.app Esc+"),
    ],
    ids=["csi-modifier-4", "escape-prefixed"],
)
@pytest.mark.asyncio
async def test_option_shift_left_selects_one_word(raw: str, terminals: str) -> None:
    """⌥⇧← extends a SELECTION one word left rather than moving a bare caret."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        editor.move_cursor((0, len(SAMPLE)))
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, raw)
        await _settle(pilot)

        assert editor.selected_text == "delta", terminals
        assert stops == [], f"⌥⇧← must not stop the turn ({terminals})"


@pytest.mark.asyncio
async def test_repeated_option_left_walks_word_by_word() -> None:
    """Two chords move two words: the coalescing does not swallow the second."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        editor.move_cursor((0, len(SAMPLE)))
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, "\x1b\x1b[D")
        await _settle(pilot)
        assert editor.cursor_location == (0, DELTA_START)

        await _feed(app, "\x1b\x1b[D")
        await _settle(pilot)
        assert editor.cursor_location == (0, GAMMA_START)
        assert stops == []


@pytest.mark.asyncio
async def test_a_lone_escape_still_stops_the_turn() -> None:
    """The window expiring with no arrow runs the escape action unchanged."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        await pilot.pause()
        stops = _watch_stops(app)

        await pilot.press("escape")
        await _settle(pilot)

        assert len(stops) == 1, "a real Esc must still stop the agent"


@pytest.mark.asyncio
async def test_two_escapes_produce_two_stops() -> None:
    """Esc-Esc is the subagent-cancel ladder; collapsing it would drop a rung."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        await pilot.pause()
        stops = _watch_stops(app)

        # Back to back, with no settle between them, so the second arrives while
        # the first is still held.
        await _feed(app, "\x1b")
        await _feed(app, "\x1b")
        await _settle(pilot)

        assert len(stops) == 2, "each press owes its own action"


@pytest.mark.asyncio
async def test_escape_then_an_ordinary_key_still_stops_the_turn() -> None:
    """A non-arrow key ends the window: the escape stood alone and is owed."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, "\x1b")
        await pilot.press("a")
        await _settle(pilot)

        assert len(stops) == 1, "the escape still meant stop"
        assert editor.text == "a", "and the character was still typed"


@pytest.mark.asyncio
async def test_losing_focus_flushes_a_held_escape() -> None:
    """No arrow is coming once focus leaves, so the escape action is owed.

    Driven through ``_defer_escape`` rather than by feeding bytes: the hold now
    lasts a single pump turn, so there is no way to interleave a blur with it
    from outside. The contract under test is the cleanup hook's, and this calls
    it at exactly the boundary that owns it.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await pilot.pause()
        fired: list[str] = []

        editor._defer_escape(lambda: fired.append("escape"))
        assert editor._pending_escape is not None, "the escape is held"

        editor._on_blur(events.Blur())
        assert fired == ["escape"], "a held escape must not be lost on blur"
        assert editor._pending_escape is None

        # And the pump turn that follows does not run it a second time.
        await pilot.pause()
        assert fired == ["escape"]


@pytest.mark.asyncio
async def test_real_teardown_settles_a_held_escape_and_leaks_no_callback() -> None:
    """Drive REAL teardown and pin what actually happens (code round 1, F3).

    An earlier version of this test called ``_on_unmount()`` by hand and
    asserted the action was dropped. That pinned a path users never take: on a
    real ``await editor.remove()`` the widget is blurred first, and
    ``_on_blur`` FLUSHES before ``_on_unmount`` could drop. So a focused
    composer being torn down runs its held escape, and ``_on_unmount`` is the
    backstop for the unfocused case rather than the usual path.

    What matters either way is the invariant this asserts: teardown always
    settles the slot, so no ``call_later`` callback can survive into a widget
    that is gone.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await pilot.pause()

        fired: list[str] = []
        editor._defer_escape(lambda: fired.append("escape"))
        assert editor._pending_escape is not None, "the escape is held"

        await editor.remove()
        # Wait on the REAL unmount conditions, not on a handful of pauses.
        # ``_settle`` of 5 turns was enough on an idle box and lost under CI
        # load (#463): a CSS refresh timer then restyled the removed editor
        # (``KeyError: text-area--gutter``) on the way out of ``run_test``.
        for _ in range(80):
            if not editor.is_attached and editor._pending_escape is None:
                break
            await pilot.pause()
        else:
            raise AssertionError(
                f"teardown never settled (attached={editor.is_attached}, "
                f"pending={editor._pending_escape is not None})"
            )

        assert fired == ["escape"], "blur wins the race and flushes"
        assert editor._pending_escape is None, "nothing is left pending"


@pytest.mark.asyncio
async def test_unmount_without_focus_drops_a_held_escape() -> None:
    """The backstop: an unfocused teardown drops rather than flushes.

    Reached when the composer never had focus, so no blur precedes the unmount.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.blur()
        await pilot.pause()

        fired: list[str] = []
        editor._defer_escape(lambda: fired.append("escape"))
        editor._on_unmount()

        assert editor._pending_escape is None
        assert fired == [], "a torn-down composer stops nothing"


@pytest.mark.asyncio
async def test_a_lone_escape_still_dismisses_a_picker() -> None:
    """A real Esc with a list open still closes the list, one pump turn later.

    The dismissal is deferred like every other escape meaning (ux round 1, U3),
    but a deferral of one pump turn is not observable: by the time the press
    returns, the list is gone.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = "/mod"
        await pilot.pause()
        editor._sync_picker()
        await pilot.pause()
        assert editor._picker.is_open(), "the command list is up"
        stops = _watch_stops(app)

        await pilot.press("escape")
        assert not editor._picker.is_open(), "the list closed"

        await _settle(pilot)
        assert stops == [], "dismissing a list never stops the turn"


@pytest.mark.asyncio
async def test_a_lone_escape_still_leaves_shell_mode() -> None:
    """A real Esc in bang-mode still leaves the mode, one pump turn later."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await pilot.press("!")
        await pilot.pause()
        assert editor.shell_mode, "bang entered shell mode"
        stops = _watch_stops(app)

        await pilot.press("escape")
        assert not editor.shell_mode, "the mode ended"

        await _settle(pilot)
        assert stops == [], "leaving the mode never stops the turn"


@pytest.mark.asyncio
async def test_the_parser_resolves_the_ambiguity_before_the_widget_sees_it() -> None:
    """The premise the one-turn deferral rests on, pinned against the parser.

    A lone ``\\x1b`` emits NOTHING until the parser has waited out its own
    ``ESCAPE_DELAY``, so a bare ``escape`` reaching the widget is already proof
    nothing followed it. The chord's two events come out of a SINGLE parse pass
    and therefore queue back to back. That is why the widget only has to yield
    one message-pump turn rather than wait a duration — if this stops holding,
    the deferral needs rethinking, not retuning.
    """
    parser = XTermParser()
    assert list(parser.feed("\x1b")) == [], "a lone Esc is held by the parser"

    chord = [e for e in XTermParser().feed("\x1b\x1b[D") if isinstance(e, events.Key)]
    assert [e.key for e in chord] == ["escape", "left"], "one pass, both events"


@pytest.mark.asyncio
async def test_escape_to_stop_costs_one_pump_turn_not_a_wall_clock_window() -> None:
    """The stop must land within a couple of pump turns, with no sleeping.

    Written with no ``asyncio.sleep`` anywhere so it fails if the deferral ever
    goes back to a wall-clock timer: a 100 ms window cannot satisfy this.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = SAMPLE
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, "\x1b")
        await pilot.pause()
        await pilot.pause()

        assert len(stops) == 1, "the stop landed within two pump turns"


@pytest.mark.asyncio
async def test_the_cleanup_hooks_do_not_double_dispatch_the_base_handler() -> None:
    """``_on_blur`` must not chain to ``super()`` — Textual already runs it.

    Textual dispatches every matching handler in the MRO, so an override that
    also calls ``super()._on_blur()`` runs the base handler twice and posts a
    second ``DescendantBlur``. That broke the app's focus tracking (a booted
    session rendered as "session is still starting"), which is a failure a long
    way from the composer, so it is pinned here at the cause.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        calls: list[int] = []
        base = Widget._on_blur

        def _spy(self: Any, event: events.Blur) -> None:
            if self is editor:
                calls.append(1)
            base(self, event)

        with patch.object(Widget, "_on_blur", _spy):
            editor.blur()
            await pilot.pause()

        assert calls == [1], "the base blur handler ran exactly once"


@pytest.mark.parametrize(
    ("raw", "terminals"),
    [
        ("\x1b[1;3D", "Ghostty / kitty / WezTerm / iTerm2 CSI mode"),
        ("\x1bb", "iTerm2 default preset / Terminal.app Option-as-Meta"),
        ("\x1b\x1b[D", "iTerm2 Esc+ / Terminal.app Esc+"),
    ],
    ids=["csi-modifier-3", "readline-meta", "escape-prefixed"],
)
@pytest.mark.asyncio
async def test_option_left_in_shell_mode_keeps_the_mode_on_every_encoding(
    raw: str, terminals: str
) -> None:
    """⌥← while editing a bang-mode command moves by word and KEEPS the mode.

    Code round 1 F1 / ux round 1 U1. The escape-prefixed encoding used to eject
    the user from shell mode as a side effect of nudging the caret, and the
    ejection was invisible — the mode's only indicator is the placeholder, which
    is hidden whenever the buffer has text. The user's next Enter then sent the
    command to the model as a prompt instead of running it, so this asserts the
    mode as well as the caret.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        await pilot.press("!")
        await pilot.pause()
        assert editor.shell_mode, "bang entered shell mode"
        editor.text = "git commit --amend"
        editor.move_cursor((0, len("git commit --amend")))
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, raw)
        await _settle(pilot)

        assert editor.shell_mode, f"⌥← must not leave shell mode ({terminals})"
        # Start of "amend", i.e. one word left rather than one character.
        assert editor.cursor_location == (0, len("git commit --")), terminals
        assert stops == [], f"⌥← must not stop the turn ({terminals})"


@pytest.mark.parametrize(
    ("raw", "terminals"),
    [
        ("\x1b[1;3D", "Ghostty / kitty / WezTerm / iTerm2 CSI mode"),
        ("\x1b\x1b[D", "iTerm2 Esc+ / Terminal.app Esc+"),
    ],
    ids=["csi-modifier-3", "escape-prefixed"],
)
@pytest.mark.asyncio
async def test_option_left_with_a_picker_open_keeps_the_list(raw: str, terminals: str) -> None:
    """⌥← with a command list open moves by word and leaves the list up.

    Ux round 1 U3: the same physical chord used to mean two different things
    depending on whether a list happened to be open.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = "/analytics"
        await pilot.pause()
        editor._sync_picker()
        await pilot.pause()
        assert editor._picker.is_open(), "the command list is up"
        editor.move_cursor((0, len("/analytics")))
        await pilot.pause()

        await _feed(app, raw)
        await _settle(pilot)

        assert editor._picker.is_open(), f"the list must survive ⌥← ({terminals})"
        assert editor.cursor_location == (0, 1), terminals


@pytest.mark.parametrize(
    ("raw", "terminals"),
    [
        ("\x1b[1;3A", "Ghostty / kitty / WezTerm / iTerm2 CSI mode"),
        ("\x1b\x1b[A", "iTerm2 Esc+ / Terminal.app Esc+"),
    ],
    ids=["csi-modifier-3", "escape-prefixed"],
)
@pytest.mark.asyncio
async def test_option_up_never_stops_the_turn(raw: str, terminals: str) -> None:
    """⌥↑ behaves as plain ↑ and never aborts the turn.

    Ux round 1 U2. On the escape-prefixed encoding this used to stop the turn
    AND overwrite the draft from history — strictly worse than the bug #370
    fixed, since it lost the turn and the draft. ⌥↑ is a real macOS chord, so a
    user who has just learned ⌥← works will try it.

    The chord maps to a pass-through rather than to a paragraph motion: this
    composer's ↑ already carries history navigation, and a second competing
    meaning for the same key would be the defect, not the fix. So this asserts
    the ordinary ↑ outcome — history recall — with no stop.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor._history = ["an earlier prompt"]
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, raw)
        await _settle(pilot)

        assert stops == [], f"⌥↑ must never stop the turn ({terminals})"
        assert editor.text == "an earlier prompt", "it behaves as plain ↑"


@pytest.mark.asyncio
async def test_option_up_does_not_eat_a_draft_mid_buffer() -> None:
    """With the caret inside the text, ⌥↑ moves the caret and keeps the draft."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        editor.text = "first line\nsecond line"
        editor.move_cursor((1, 6))
        await pilot.pause()
        stops = _watch_stops(app)

        await _feed(app, "\x1b\x1b[A")
        await _settle(pilot)

        assert stops == [], "no stop"
        assert editor.text == "first line\nsecond line", "the draft survives"
        assert editor.cursor_location[0] == 0, "the caret moved up a line"


@pytest.mark.asyncio
async def test_the_ctrl_bindings_are_kept_alongside_the_alt_ones() -> None:
    """This is additive: the Linux/Windows chord keeps working."""
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 24)) as pilot:
        editor = await _boot(pilot, app)
        keys = set(editor._bindings.key_to_bindings)
        assert {"alt+left", "alt+right", "alt+shift+left", "alt+shift+right"} <= keys
        assert {"ctrl+left", "ctrl+right", "ctrl+shift+left", "ctrl+shift+right"} <= keys
