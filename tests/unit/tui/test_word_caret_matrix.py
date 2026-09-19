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

WHY THE CELLS OF A GROUP SHARE ONE APP. A boot is what a pilot test costs, not
a keystroke: measured in this worktree, `run_test` to a ready session is ~0.74 s
against ~0.13 s for arranging a state and settling one act, and 14 of the 15
seconds the original arms spent per cell were boot. The cross product is 180
cells and the equivalence property is checked against TWO apps per cell — the
chord and its plain-arrow oracle — so the module as first written booted 783
`OperatorApp`s for 303 tests and spent ~570 s of its 572 s doing it (the file's
recorded weight, and the largest single item in `tests/durations.json`). Each
group below (one state, one history value) now boots ONE app and drives every
cell of the group through it, with `_SharedApp.run` restoring the composer
between arms and ASSERTING the restored state against the fingerprint the
group's own boot produced; a reset that missed anything fails there by name.
The matrix, the oracle and every cell are unchanged — the group tests still run
each cell's two arms and still compare chord against arrow — and
`test_the_matrix_still_covers_the_whole_cross_product` pins the cell count that
`--collect-only` no longer shows. `test_the_shared_app_still_behaves_like_a_boot_of_its_own`
is what keeps the reuse honest: it re-runs the first and last cell of every
group on a boot of its own and requires identical observations, because a
restructured test that cannot fail on the bug it was written to catch is a
regression.

Encoding notes (verified against textual 8.2.8, see `test_word_caret.py`):

- CSI-modifier is the only encoding with a spelling for every chord.
- readline-meta only exists for the horizontal pair (`\\x1bb` / `\\x1bf`); there
  is no meta spelling of a vertical arrow, so those cells are absent by nature
  rather than skipped.
- Esc-prefixed spells everything, as `escape` followed by the plain key.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import pytest
from textual import events
from textual._xterm_parser import XTermParser

from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.editor import Editor, StopRequested

from .test_app_pilot import FakeSession, _factory
from .test_word_caret import _watch_stops
from .waiting import MessageWaiter

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


async def _boot(pilot: Any, app: OperatorApp, messages: MessageWaiter) -> Editor:
    await messages.wait_for(lambda: app._session is not None, description="matrix session adopted")
    assert app._session is not None, "the session never booted"
    editor = app.query_one(Editor)
    editor.focus()
    await pilot.pause()
    return editor


async def _settle(pilot: Any, editor: Editor, messages: MessageWaiter) -> None:
    """Observe the deferred Escape, not six guesses at an idle frame.

    The two queue barriers are what carry the observation: the first delivers
    the raw driver events, the second delivers the resulting
    StopRequested/picker messages before ``_observe`` reads them. They inspect
    editor state rather than settled animation geometry, and the zero delay
    keeps the real queue barriers without CPU-idle detection per frame.

    The ``_pending_escape is None`` wait between them is a GUARD, not the
    barrier these cells are observed through. Measured, the deferral arms AND
    retires inside the first barrier, so the predicate is already true at its
    first test on every escape-bearing cell (574/574 in the QA round's
    instrumentation, 32/32 in the review round's F8 cells) and the wait never
    blocks there -- the stop reaches observation through the two barriers
    regardless. It is kept because a future change that made the deferral
    asynchronous would otherwise pass silently; its live case is exercised
    deliberately by
    ``test_settle_waits_for_deferred_escape_before_observing_its_stop``, which
    holds the callback open. Do not read this wait as load-bearing here.
    """
    await pilot.pause(0)
    await messages.wait_for(
        lambda: editor._pending_escape is None, description="deferred Escape retired"
    )
    await pilot.pause(0)


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


def _hold_pending_escape(app: OperatorApp) -> None:
    """Deliver a REAL lone Escape that is still held when the act runs.

    Posted WITHOUT yielding, so the escape is still held when the act runs.
    Yielding here would let the one-turn deferral resolve first and the cell
    would test nothing -- which is exactly the mistake that let F8 through,
    reproduced one level up in the test itself.

    Injected as bytes through the real parser rather than as a key name, because
    a lone ``\\x1b`` only emerges after the parser's own ESCAPE_DELAY -- which is
    exactly why the user's inter-key gap is spent inside the parser and not
    inside the widget's window, and therefore why a real gap does not save you
    from the bug.
    """
    parser = XTermParser()
    driver = app._driver
    assert driver is not None
    for event in list(parser.feed("\x1b")) + list(parser.feed("")):
        if isinstance(event, events.Key):
            event.set_sender(app)
            driver.send_message(event)


async def _run(
    state: str,
    history: bool,
    act: Callable[[Any, OperatorApp], Any],
    escape_pending: bool = False,
) -> tuple[Any, ...]:
    """One matrix cell, run in a boot of its own.

    This is the reference the shared app is checked against, and it is kept for
    exactly that reason: the parity test needs an unfaked boot to compare with,
    and a harness that reimplemented the boot would only be checking itself.
    The hot path no longer runs cells this way -- one boot per cell became one
    boot per group (see ``_SharedApp``) -- with the arm's own semantics
    unchanged: arrange, install the stop spy, hold an escape if this cell wants
    one, act, settle, observe.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    messages = MessageWaiter()
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        editor = await _boot(pilot, app, messages)
        await _arrange(pilot, editor, state, history)

        stops: list[Any] = []
        original = app.post_message

        def _spy(message: Any) -> bool:
            if isinstance(message, StopRequested):
                stops.append(message)
            return original(message)

        app.post_message = _spy  # type: ignore[method-assign]
        if escape_pending:
            _hold_pending_escape(app)
        await act(pilot, app)
        await _settle(pilot, editor, messages)
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


def _groups() -> list[tuple[str, bool]]:
    """The (state, history) groups the cross product partitions into.

    One boot serves a whole group, so this is also the list of the boots the
    module spends: one per group per test that walks the group.
    """
    return [(state, history) for state in STATES for history in (True, False)]


def _cells_for(state: str, history: bool) -> list[tuple[str, str]]:
    """One group's (chord, encoding) cells, in ``_cells``'s own order."""
    return [(chord, encoding) for (s, h, chord, encoding) in _cells() if (s, h) == (state, history)]


def _spellable_cells(
    encodings: Sequence[str] = ENCODINGS, chords: Sequence[str] = tuple(CHORDS)
) -> list[tuple[str, str]]:
    """Every (chord, encoding) pair these terminals can actually spell.

    ``_encode`` returning ``None`` is a fact about the terminal, not a gap in
    coverage: no encoding here has a meta spelling of a vertical arrow, so those
    pairs are absent from the matrix rather than skipped inside it.
    """
    return [
        (chord, encoding)
        for encoding in encodings
        for chord in chords
        if _encode(encoding, chord) is not None
    ]


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


def _fingerprint(editor: Editor, app: OperatorApp, stops: list[Any]) -> dict[str, Any]:
    """Every piece of state a cell can leave behind, as one comparable mapping.

    A reset that silently missed one of these is the failure mode the shared app
    has to survive: the next cell would run from a state no user can reach, and
    both of its arms would share that wrong state, so ``chord == arrow`` would
    still hold and the cell would still pass. Comparing the reset against the
    group's own boot is what turns a miss into a failure that names the field.

    NAMED FIELDS rather than a reflected snapshot of ``vars()``: the objects here
    carry layout, paint and window internals that differ between two boots of the
    same code and mean nothing to a cell, so a reflected comparison would be
    noise. The list is therefore curated for meaning, and the parity test is the
    backstop for anything it does not name.

    The pickers' row lists are deliberately absent: ``_sync_picker`` re-derives
    them from the buffer, so a stale row set shows up here as the buffer, mode or
    query that produced it.
    """
    return {
        "text": editor.text,
        "cursor": editor.cursor_location,
        "selection": editor.selected_text,
        "shell_mode": editor.shell_mode,
        "history": tuple(editor._history),
        "history_index": editor._history_index,
        # The buffer a history step restores on its way OUT of navigation. A
        # fresh boot has it empty; the reset has to empty it too, or the next
        # cell that leaves history navigation gets a previous cell's buffer.
        "draft_stash": editor._draft,
        "pending_escape": editor._pending_escape is not None,
        "picker_open": editor._picker.is_open(),
        "picker_mode": str(editor._picker.mode),
        "picker_selected": editor._picker.selected_index,
        "picker_dismissed": editor._picker.is_dismissed(),
        "picker_key": editor._picker_key_at_last_sync,
        "argument_command": editor._argument_command,
        "argument_subcommand": editor._argument_subcommand,
        "skill_requested": editor._skill_choices_requested,
        "file_requested": editor._file_choices_requested,
        "model_open": editor._model_picker.is_open(),
        "model_selected": editor._model_picker.selected_index,
        "model_query": editor._model_picker.query_text,
        "model_dismissed": editor._model_picker.is_dismissed(),
        "stops": len(stops),
        # The two ladders the app carries ACROSS presses rather than within one:
        # the armed "esc again" stop offer, and the screen stack a cell can push
        # without popping. Neither is in `_observe`, and both are inherited by
        # the next cell if a reset forgets them.
        "stop_offered": app._stop_offered_at is not None,
        "screens": len(app.screen_stack),
    }


async def _reset(pilot: Any, editor: Editor) -> None:
    """Return the composer to the state a fresh boot leaves it in.

    Each line is a state a cell can move, and each one is here because leaving
    it would survive the buffer swap that follows:

    - ``clear_content`` is what ``/clear`` does -- buffer, history-navigation
      index, attachments, armed-secret latch -- so the buffer is emptied through
      the product's own funnel rather than by assigning ``text``.
    - ``set_shell_mode`` is bang-mode's own leave path; the mode is not a
      property of the buffer, so emptying the buffer does not leave it.
    - ``_pending_escape`` is a LIVE deferred callback. Carried into the next
      cell it would let a stop from the previous cell's Escape be counted
      against this one's, which is precisely the axis the F8 test measures.
    - ``close()`` on either picker drops its rows AND zeroes the highlight and
      window offset pointing into them. That is the part a bare buffer swap
      leaves behind: ``_apply`` only re-zeroes those on a MODE change, so a list
      reopened into the same mode would inherit the previous cell's highlight.
    - ``_draft`` is the buffer a history step restores when it leaves
      navigation; a fresh boot has it empty.

    Nothing here is trusted to be complete: ``_SharedApp.run`` compares the
    result against the group's boot fingerprint and fails by field name.
    """
    editor.clear_content()
    editor.set_shell_mode(False)
    editor._pending_escape = None
    editor._picker.close()
    editor._model_picker.close()
    editor._draft = ""
    # One turn so the messages these calls post (the picker close, the shell-mode
    # change) are delivered BEFORE the fingerprint is read.
    await pilot.pause()


class _SharedApp:
    """One booted app, driving every cell of one ``(state, history)`` group.

    The group's first arrange runs on the state the BOOT left, so the
    fingerprint every later arm is checked against is one a real boot produced
    rather than another reset's output -- a reset compared against a reset would
    agree with itself while both drifted.

    What this cannot prove by construction is that the shared app still behaves
    like a boot of its own in the parts ``_fingerprint`` does not name. That is
    measured, not assumed: see
    ``test_the_shared_app_still_behaves_like_a_boot_of_its_own``.
    """

    def __init__(
        self,
        pilot: Any,
        app: OperatorApp,
        editor: Editor,
        messages: MessageWaiter,
        stops: list[Any],
    ) -> None:
        self._pilot = pilot
        self._app = app
        self._editor = editor
        self._messages = messages
        self._stops = stops
        self._state = ""
        self._history = False
        self._baseline: dict[str, Any] = {}

    async def arrange_group(self, state: str, history: bool) -> None:
        """Arrange the group's state and record what every later reset must reproduce."""
        self._state, self._history = state, history
        await _arrange(self._pilot, self._editor, state, history)
        self._stops.clear()
        self._baseline = _fingerprint(self._editor, self._app, self._stops)

    async def run(
        self, act: Callable[[Any, OperatorApp], Any], escape_pending: bool = False
    ) -> tuple[Any, ...]:
        """Reset to the group's state, apply ``act``, and report what a user would see.

        The reset is asserted before the act rather than after: a cell that
        started from a state the boot never produced is not a cell of this
        matrix, whatever it then observes.
        """
        await _reset(self._pilot, self._editor)
        await _arrange(self._pilot, self._editor, self._state, self._history)
        self._stops.clear()
        restored = _fingerprint(self._editor, self._app, self._stops)
        drift = {
            key: (self._baseline[key], restored[key])
            for key in self._baseline
            if restored[key] != self._baseline[key]
        }
        assert not drift, (
            "app reuse leaked state into the next cell: the reset did not reproduce the "
            f"state this app's boot produced for {self._state} (history={self._history}). "
            f"Field: (boot, after reset)\n  {drift}"
        )
        if escape_pending:
            _hold_pending_escape(self._app)
        await act(self._pilot, self._app)
        await _settle(self._pilot, self._editor, self._messages)
        return _observe(self._editor, self._stops)


@asynccontextmanager
async def _boot_shared(state: str, history: bool) -> AsyncIterator[_SharedApp]:
    """Boot ONE app for a group, with the stop spy installed for its whole life.

    The spy is the app's own ``post_message``, so installing it once per boot
    instead of once per arm is the same observation with a fraction of the
    wrapping -- and ``_SharedApp.run`` clears the stop list before each act, so
    a cell still counts only its own stops.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    messages = MessageWaiter()
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        editor = await _boot(pilot, app, messages)
        stops: list[Any] = []
        original = app.post_message

        def _spy(message: Any) -> bool:
            if isinstance(message, StopRequested):
                stops.append(message)
            return original(message)

        app.post_message = _spy  # type: ignore[method-assign]
        shared = _SharedApp(pilot, app, editor, messages, stops)
        await shared.arrange_group(state, history)
        yield shared


@pytest.mark.parametrize(("state", "history"), _groups(), ids=lambda v: str(v))
@pytest.mark.asyncio
async def test_the_chord_is_indistinguishable_from_its_plain_arrow(
    state: str, history: bool
) -> None:
    """One group of the matrix: option chord == plain arrow, whatever the state.

    The plain arrow is run in its own arm of the SAME app and used as the
    oracle, so this asserts equivalence rather than a remembered outcome. The
    app is what a group's cells share; the arm is not -- ``run`` puts the
    composer back in the group's boot state and fails the cell if it cannot.

    Every failing cell in the group is reported, not just the first, so one
    broken crossing does not hide the rest.
    """
    cells = _cells_for(state, history)
    assert cells, f"no cells for {state} (history={history})"
    failures: list[str] = []
    async with _boot_shared(state, history) as shared:
        for chord, encoding in cells:
            raw = _encode(encoding, chord)
            assert raw is not None

            expected = await shared.run(_press_plain(CHORDS[chord]))
            actual = await shared.run(_feed_bytes(raw))

            if actual != expected:
                failures.append(
                    f"{encoding} {chord} in {state} (history={history}) diverged from plain "
                    f"{CHORDS[chord]}:\n  plain={expected}\n  chord={actual}"
                )
    assert not failures, f"{len(failures)} of {len(cells)} cells diverged:\n" + "\n".join(failures)


@pytest.mark.parametrize(("state", "history"), _groups(), ids=lambda v: str(v))
@pytest.mark.asyncio
async def test_a_self_contained_chord_never_swallows_a_pending_escape(
    state: str, history: bool
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
    cells = [
        (chord, encoding)
        for (s, h, chord, encoding) in _escape_pending_cells()
        if (s, h) == (state, history)
    ]
    assert cells, f"no escape-pending cells for {state} (history={history})"
    failures: list[str] = []
    async with _boot_shared(state, history) as shared:
        for chord, encoding in cells:
            raw = _encode(encoding, chord)
            assert raw is not None

            # THE CONTROL is the horizontal chord, and the choice matters.
            # `alt+left` is never rewritten, is not in `ESCAPE_CHORD_KEYS`, and
            # therefore never had F8 -- the code reviewer identified exactly that
            # asymmetry as the tell. So "how a pending escape is disposed of in
            # this state" is read off the chord that provably handles it
            # correctly, and the chord under test must match.
            #
            # Reading it from a control rather than asserting a fixed outcome is
            # what keeps this honest: Esc means four different things depending
            # on state (stop, leave shell mode, dismiss either list), and in the
            # picker states a dismissed list is legitimately re-opened by the
            # following `_sync_picker`. Hand-writing those outcomes would have
            # encoded a pre-existing behaviour as if it were this feature's
            # contract.
            control_raw = _encode(encoding, "shift+left" if "shift" in chord else "left")
            assert control_raw is not None
            control = await shared.run(_feed_bytes(control_raw), escape_pending=True)
            control_baseline = await shared.run(_feed_bytes(control_raw))

            baseline = await shared.run(_feed_bytes(raw))
            actual = await shared.run(_feed_bytes(raw), escape_pending=True)

            # Whatever the escape did to the control's state, it must also have
            # done here: same shell mode, same picker states, same stop count.
            control_effect = (
                control[3],
                control[4],
                control[5],
                control[-1] - control_baseline[-1],
            )
            actual_effect = (actual[3], actual[4], actual[5], actual[-1] - baseline[-1])
            if actual_effect != control_effect:
                failures.append(
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
    assert not failures, f"{len(failures)} of {len(cells)} cells diverged:\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_a_pending_escape_still_stops_the_turn_after_a_chord() -> None:
    """The F8 payload, stated absolutely rather than against an oracle.

    Belt and braces beside the equivalence test above: if the plain-arrow
    oracle were itself ever broken, the equivalence could hold while both sides
    lost the stop. This pins the stop count directly.
    """
    cells = _spellable_cells(("csi", "meta"))
    assert cells, "no self-contained spelling for any chord"
    failures: list[str] = []
    async with _boot_shared("resting", False) as shared:
        for chord, encoding in cells:
            raw = _encode(encoding, chord)
            assert raw is not None

            result = await shared.run(_feed_bytes(raw), escape_pending=True)
            stops = result[-1]
            if stops != 1:
                failures.append(f"{encoding} {chord} swallowed the pending escape (stops={stops})")
    assert not failures, "\n".join(failures)


@pytest.mark.asyncio
async def test_a_vertical_chord_never_destroys_a_typed_slash_command() -> None:
    """The exact regression that forced round 3, pinned on its own.

    Code round 2 F5 / ux round 2 U6: with a list open and history present, the
    CSI vertical chord closed the picker and replaced `/model anthropic/claude`
    with a history entry, because a `Binding` fires through the action system
    and never reaches the picker branches inside `_on_key`.

    Stated as an absolute rather than as an equivalence, so it still fails
    loudly if the plain-arrow oracle above were ever itself broken.
    """
    typed = STATES["model_picker"]["text"]
    cells = _spellable_cells(ENCODINGS, chords=("up", "down"))
    assert cells, "no spelling for a vertical chord"
    failures: list[str] = []
    async with _boot_shared("model_picker", True) as shared:
        for chord, encoding in cells:
            raw = _encode(encoding, chord)
            assert raw is not None

            result = await shared.run(_feed_bytes(raw))
            text = result[1]
            if text != typed:
                failures.append(f"{encoding} ⌥{chord} destroyed the typed command: {text!r}")
            if text in HISTORY:
                failures.append("the buffer was overwritten from history")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(("state", "history"), _groups(), ids=lambda v: str(v))
@pytest.mark.asyncio
async def test_the_shared_app_still_behaves_like_a_boot_of_its_own(
    state: str, history: bool
) -> None:
    """App reuse is only sound while it is indistinguishable from a fresh boot.

    Every other test here runs its cells on an app that has already served other
    cells. ``_SharedApp.run`` proves the reset reproduces the state the group's
    own boot produced, but that is a comparison between two states of the SAME
    app: it cannot see a way in which reuse itself differs, and the app carries
    plenty a fresh boot has not been through -- a picker resync key, row caches,
    an escape ladder, a transcript with blocks in it.

    So the claim is measured against a real boot of its own, on the FIRST and
    LAST cell of every group: the state before any reuse, and the state after a
    whole group's worth of it. Both arms must agree exactly, observation for
    observation. This is the check that keeps the restructure from becoming a
    test that can no longer fail on what it was written to catch -- run
    ``test_the_chord_is_indistinguishable_from_its_plain_arrow`` against the
    #370 defect and these cells are what pins the reused path to the same
    verdict.
    """
    cells = _cells_for(state, history)
    assert cells, f"no cells for {state} (history={history})"

    fresh: list[tuple[Any, ...]] = []
    for chord, encoding in (cells[0], cells[-1]):
        raw = _encode(encoding, chord)
        assert raw is not None
        fresh.append(await _run(state, history, _press_plain(CHORDS[chord])))
        fresh.append(await _run(state, history, _feed_bytes(raw)))

    reused: list[tuple[Any, ...]] = []
    async with _boot_shared(state, history) as shared:
        for chord, encoding in (cells[0], cells[-1]):
            raw = _encode(encoding, chord)
            assert raw is not None
            reused.append(await shared.run(_press_plain(CHORDS[chord])))
            reused.append(await shared.run(_feed_bytes(raw)))
    assert reused == fresh, (
        f"the shared app and a boot of its own disagree in {state} (history={history}):\n"
        f"  boot   ={fresh}\n  shared ={reused}"
    )


def test_the_matrix_still_covers_the_whole_cross_product() -> None:
    """The cross product IS the feature, so its size is pinned in one place.

    Amortizing the boots moved these cells out of pytest ids and into loops
    inside a group's test, which means ``--collect-only`` no longer counts them
    and cannot be the evidence that none was dropped. This is that evidence,
    derived from the DECLARED axes -- states x history x the pairs each terminal
    can spell -- rather than from the generator, so a generator that quietly
    stopped emitting cells would fail here rather than shrink the matrix in
    silence.
    """
    # The encoding tables are the one place a chord could go missing silently:
    # a chord with no CSI spelling is absent from the matrix by nature, and this
    # is where that is required to be true of the TERMINAL rather than of the
    # table.
    assert set(_CSI) == set(CHORDS), "the CSI table must spell every chord"
    assert set(_META) <= set(CHORDS), "the meta table is a subset of the chords"

    # csi spells every chord, meta only the horizontal pair, and esc-prefixed
    # spells everything (a bare ESC followed by the plain sequence).
    spellable = len(CHORDS) + len(_META) + len(CHORDS)
    assert len(_cells()) == len(STATES) * 2 * spellable
    assert len(_escape_pending_cells()) == len(STATES) * 2 * (len(CHORDS) + len(_META))
    assert len(_spellable_cells(("csi", "meta"))) == len(CHORDS) + len(_META)
    assert len(_spellable_cells(ENCODINGS, chords=("up", "down"))) == 4

    # ...and every group carries the full set, which is what makes "no cell was
    # dropped" a property of the groups rather than of a total: a partition
    # that lost one cell from each group would still satisfy a total, a
    # per-group count would not.
    assert len(_groups()) == len(STATES) * 2
    for group in _groups():
        assert len(_cells_for(*group)) == spellable, f"group {group} lost a cell"


@pytest.mark.asyncio
async def test_settle_waits_for_deferred_escape_before_observing_its_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued key is not completion: hold its real deferred callback open.

    A fixed queue barrier alone returns while this worker is parked. The
    observable-condition wait must stay pending, then deliver the StopRequested
    the real callback posts, without depending on a parser or wall-clock delay.

    The composer's resting state is enough: a lone Escape there defers the real
    ``post_message(StopRequested())``, which is the behaviour this pins.
    """
    messages = MessageWaiter()
    app = OperatorApp(lambda: _factory(FakeSession()))
    release = asyncio.Event()
    armed = asyncio.Event()
    waiting = asyncio.Event()
    async with app.run_test(size=(100, 24), message_hook=messages.on_message) as pilot:
        editor = await _boot(pilot, app, messages)
        stops = _watch_stops(app)
        original_call_later = editor.call_later

        def delayed(callback: Any, *args: Any, **kwargs: Any) -> bool:
            if callback == editor._flush_escape:

                async def deliver() -> None:
                    armed.set()
                    await release.wait()
                    original_call_later(callback, *args, **kwargs)

                app.run_worker(deliver())
                return True
            return original_call_later(callback, *args, **kwargs)

        original_wait = messages.wait_for

        async def observed(predicate: Any, *, description: str, timeout: float = 30.0) -> None:
            # Identify the guard by BEHAVIOUR, not by its description prose: a
            # rename inside ``_settle`` must not degrade this pin into a failure
            # carrying a misleading message. ``_settle`` makes exactly one wait
            # while the deferred flush is parked, so a wait whose predicate is
            # False at entry IS that guard -- and the queue-barrier-only
            # regression reaches no wait at all, which is what the assertion at
            # ``waiting.is_set()`` is there to catch.
            if not predicate():
                waiting.set()
            await original_wait(predicate, description=description, timeout=timeout)

        monkeypatch.setattr(editor, "call_later", delayed)
        monkeypatch.setattr(messages, "wait_for", observed)
        await pilot.press("escape")
        await armed.wait()
        task = asyncio.create_task(_settle(pilot, editor, messages))
        # WHICH of these two settles first is the whole discrimination: a wait on
        # the deferred callback (correct), or the settle task returning without
        # one (the queue-barrier-only regression). The bound is a diagnostic
        # backstop for the broken arm, so its expiry can never read as a pass.
        try:
            waited = asyncio.create_task(waiting.wait())
            try:
                await asyncio.wait(
                    {task, waited}, timeout=10.0, return_when=asyncio.FIRST_COMPLETED
                )
                assert waiting.is_set(), "settle returned without waiting on the deferred callback"
                assert not task.done(), "settle returned before the deferred callback"
                assert stops == []
                assert editor._pending_escape is not None
            finally:
                waited.cancel()
        finally:
            release.set()
            await task
        assert editor._pending_escape is None
        assert len(stops) == 1, "the completion barrier lost the downstream stop"
