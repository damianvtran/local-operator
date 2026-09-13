"""The pixel-mouse LATCH gate and the mid-session mode re-clean, end to end.

Scope, stated because the sibling file owns the other half. ``test_pixel_mouse_latch.py``
pins the boot-time pair (the ``CSI ?2048l`` reset and the ``TEXTUAL_SMOOTH_SCROLL``
guard) and Textual's own latch behaviour. This file owns the two things that were
still missing after that pair landed, and the last test here is the only one that
proves the user-visible claim on a real terminal:

- **The latch gate** (``terminal_modes.install_pixel_mouse_gate``). The guard can
  stop us ASKING for mode 2048, but mode 2048 is per-VT state shared with every
  other process on the tty, and Textual's latch (``_xterm_parser.py:281-282``) is
  one-way: a delivered report sets ``mouse_pixels`` and ``parse_mouse_code``
  (``:94-102``) then divides every later position by that report's cell size,
  forever, whatever we negotiated. The gate forces the divisor off for the
  configuration the guard selected, and is installed ONLY there.
- **The mid-session re-clean** (``terminal_modes.InBandResizeReclaimer``), which
  re-asserts the reset on ``Resize`` and focus-in through the app's driver
  writer, so a co-tenant's dirty mode is closed within one interaction instead of
  at the next boot.

``THE PTY TESTS ARE THE ONES THAT MATTER``
(``test_a_dirty_mode_mid_session_does_not_move_the_pointer`` and
``test_a_compliant_vt_reports_cells_once_we_clear_1016``).
Everything above them asserts our own contract; those drive the REAL driver,
the REAL ``XTermParser`` and a REAL Textual app on a pty, dirty the mode
mid-process the way a co-tenant does, and assert on what the app's pointer
actually became. The first carries its own control arm — the same harness, same
run, gate not installed — so a green result is evidence about the gate rather
than about the harness; the second carries the arms of the two terminal classes
the fix has to serve (a VT that honours mode 1016, told cells by our reset, and
the pre-fix wire that never told it). Measurements on this machine (textual
8.2.8, macOS):

    gate OFF: [[40, 44, 40, 44], [7, 1, 7, 1]]   <- (60, 24) collapsed to (7, 1)
    gate ON : [[40, 44, 40, 44], [60, 24, 60, 24]]

Parser-side tests set ``constants.SMOOTH_SCROLL`` through ``monkeypatch`` rather
than the environment, for the reason the sibling file documents: it freezes from
the environment at ``textual`` import time.

Every subprocess here gets an ISOLATED HOME and ``LOCAL_OPERATOR_CONFIG_DIR``,
and its environment is built from scratch rather than inherited, so no ``CMUX_*``
variable can reach it — an inherited ``CMUX_WORKSPACE_ID`` lets a booted TUI
rename workspaces in the operator's live multiplexer. Nothing in this file starts
a ``lop`` session, so there is no session id to synthesise.

All references pinned to textual 8.2.8: ``_xterm_parser.py`` is
``textual/_xterm_parser.py``, ``app.py`` is ``textual/app.py`` and ``driver``
means textual 8.2.8's ``drivers/linux_driver.py``.
"""

from __future__ import annotations

import json
import os
import select
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import pytest
from textual import constants, events
from textual._xterm_parser import XTermParser

from local_operator.tui.app import OperatorApp
from local_operator.tui.terminal_modes import (
    InBandResizeReclaimer,
    guard_pixel_mouse_latch,
    install_pixel_mouse_gate,
    pixel_mouse_gate_installed,
    pixel_mouse_negotiation_open,
    uninstall_pixel_mouse_gate,
)
from tests.unit.tui.test_app_pilot import (
    FakeSession,
    _factory,
    _isolate_tui_settings,
    _spy_driver_writes,
)

#: An SGR mouse move at cell (40, 44) — the wire is 1-based, the event 0-based.
MOUSE_MOVE_1 = "\x1b[<35;41;45M"

#: An SGR mouse move at cell (60, 24). The second position is what makes the
#: collapse visible: (60, 24) divided by a measured (8.00, 16.00) cell is (7, 1),
#: which is where the bug puts the pointer.
MOUSE_MOVE_2 = "\x1b[<35;61;25M"

#: ``CSI 48;rows;cols;pxH;pxW t`` for a 44x133 cell frame measuring 1064x704
#: pixels — a 8x16 cell, i.e. the report that latches the divisor.
IN_BAND_REPORT = "\x1b[48;44;133;704;1064t"

#: Where the divisor puts the second pointer once it has latched (measured).
COLLAPSED_SECOND_POSITION = (7, 1)

#: The re-clean's wire bytes, as literal text: ``?2048l`` (the mode the report
#: reveals) then ``?1016l`` (the pixel-mouse mode a co-tenant sets with it), one
#: write. Literal rather than imported so the assertion is about the wire and so
#: this file still imports against a build that predates the 1016 half — which
#: is how the pre-fix failure is demonstrated.
RESET_PAIR = "\x1b[?2048l\x1b[?1016l"

#: Where the pointer actually is.
TRUE_FIRST_POSITION = (40, 44)
TRUE_SECOND_POSITION = (60, 24)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _positions(parser: XTermParser, data: str) -> list[tuple[int, int]]:
    """Feed ``data`` and return the (x, y) of every mouse event it yields."""
    return [
        (event.x, event.y)
        for event in list(parser.feed(data))
        if isinstance(event, events.MouseEvent)
    ]


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Install the REAL gate for one test and guarantee it is uninstalled.

    The precondition is pinned rather than inherited: ``install_pixel_mouse_gate``
    refuses while textual may still negotiate pixel mouse, and both names it
    reads are frozen from the environment at ``textual`` import time — so a
    developer running the suite with ``TEXTUAL_SMOOTH_SCROLL=1`` exported would
    otherwise see these tests fail for their own shell.

    Unconditionally uninstalled, and in a fixture rather than at the end of each
    test: the gate patches a class shared by the whole worker process, and a
    leaked gate would flip ``mouse_pixels`` off for every other file xdist
    happens to run in that worker — including the sibling test that asserts the
    COLLAPSED behaviour.
    """
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", False)
    assert pixel_mouse_negotiation_open() is False
    installed = install_pixel_mouse_gate()
    try:
        # The fixture installs onto a clean class by contract, so a False here
        # means a leaked gate from another test in this worker, which is worth
        # failing loudly on rather than papering over.
        assert installed is True
        yield None
    finally:
        uninstall_pixel_mouse_gate()


# -- the gate: its precondition, install, idempotence, reversal ---------------


@pytest.mark.parametrize(
    ("smooth_scroll", "is_iterm", "negotiates"),
    [
        # (SMOOTH_SCROLL, IS_ITERM) -> does textual still negotiate pixel mouse?
        # This mirrors textual's own gate on the mode-report branch
        # (`_xterm_parser.py:319-322`), one row per combination.
        pytest.param(True, False, True, id="smooth-scrolling-on"),
        pytest.param(True, True, False, id="smooth-scrolling-on-iterm"),
        pytest.param(False, False, False, id="smooth-scrolling-off"),
        pytest.param(False, True, False, id="smooth-scrolling-off-iterm"),
    ],
)
def test_the_gate_applies_exactly_when_textual_stops_negotiating(
    monkeypatch: pytest.MonkeyPatch, smooth_scroll: bool, is_iterm: bool, negotiates: bool
) -> None:
    """The precondition, pinned across Textual's own gate's whole truth table.

    A report is evidence about our coordinates only while the app may have asked
    for pixel ones, so the gate installs iff the mode-report branch is shut. The
    iTerm rows are the ones a booleans-only reading gets wrong: iTerm does not
    take this path in textual 8.2.8 (``IS_ITERM`` closes it by itself), so the
    in-band report cannot be solicited there whatever ``SMOOTH_SCROLL`` says.

    ``TEXTUAL_SMOOTH_SCROLL=1`` on a non-iTerm terminal is the row that keeps
    upstream behaviour: no gate, so that user's pixel coordinates keep working.
    """
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", smooth_scroll)
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", is_iterm)

    assert pixel_mouse_negotiation_open() is negotiates

    installed = install_pixel_mouse_gate()
    try:
        assert installed is (not negotiates)
        assert pixel_mouse_gate_installed() is (not negotiates)
    finally:
        uninstall_pixel_mouse_gate()


def test_the_gate_accepts_an_inherited_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inherited ``TEXTUAL_SMOOTH_SCROLL=0`` must still install the gate.

    This is the shape the operator's own runtime exports, the shape a
    ``replace_self`` re-entry inherits, and the shape our documentation tells
    users to set. The guard DEFERS to it (it is an integer Textual honours), so a
    gate keyed on the guard's return value would silently drop both halves of
    this fix on precisely the recommended configuration.
    """
    env: dict[str, str] = {"TEXTUAL_SMOOTH_SCROLL": "0"}
    assert guard_pixel_mouse_latch(env) is False, "the guard defers to an honoured integer"
    assert env == {"TEXTUAL_SMOOTH_SCROLL": "0"}

    # What the deferred guard leaves behind is what the constants read.
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", False)
    assert pixel_mouse_negotiation_open() is False
    assert install_pixel_mouse_gate() is True
    assert pixel_mouse_gate_installed() is True
    uninstall_pixel_mouse_gate()


def test_the_gate_installs_once_and_restores_the_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotent and reversible, which is what lets a test own the class.

    A second install must not wrap its own wrapper (the wrapper would then run
    twice per mouse code — harmless today, a trap the moment the wrapper does
    anything more than clear one flag), and uninstall must put back the ORIGINAL
    function object rather than something that merely behaves like it.
    """
    original = XTermParser.parse_mouse_code
    # The precondition first, so this test is about install/uninstall rather
    # than about the developer's terminal.
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", False)
    assert pixel_mouse_gate_installed() is False

    assert install_pixel_mouse_gate() is True
    assert pixel_mouse_gate_installed() is True
    assert XTermParser.parse_mouse_code is not original

    assert install_pixel_mouse_gate() is False, "a second install must not wrap again"
    assert pixel_mouse_gate_installed() is True

    assert uninstall_pixel_mouse_gate() is True
    assert XTermParser.parse_mouse_code is original
    assert pixel_mouse_gate_installed() is False
    assert uninstall_pixel_mouse_gate() is False


def test_the_gate_stops_a_delivered_report_from_scaling_coordinates(
    monkeypatch: pytest.MonkeyPatch, gated: None
) -> None:
    """The gate's whole job, on the real parser, with its own control arm.

    The control arm is the point: the SAME bytes collapse the pointer when the
    gate is not installed, so a green assertion above cannot be a parser that
    never latched in the first place. ``SMOOTH_SCROLL`` is pinned False so the
    run describes the guarded configuration.
    """
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)

    guarded = XTermParser()
    assert _positions(guarded, MOUSE_MOVE_1) == [TRUE_FIRST_POSITION]
    list(guarded.feed(IN_BAND_REPORT))
    # The report still latches the flag — the gate clears it at PARSE time, which
    # is the only place a coordinate can still be rescued. Asserting the latch is
    # still set is what keeps this test honest about what the gate does.
    assert guarded.mouse_pixels is True
    assert _positions(guarded, MOUSE_MOVE_2) == [TRUE_SECOND_POSITION]

    uninstall_pixel_mouse_gate()
    control = XTermParser()
    assert _positions(control, MOUSE_MOVE_1) == [TRUE_FIRST_POSITION]
    list(control.feed(IN_BAND_REPORT))
    assert _positions(control, MOUSE_MOVE_2) == [COLLAPSED_SECOND_POSITION]


def test_the_gate_leaves_textual_resize_bookkeeping_alone(gated: None) -> None:
    """Narrow by construction: only the latch is touched, not the sizes.

    ``terminal_size``/``terminal_pixel_size`` are what the divisor divides BY,
    and Textual's ``Resize`` handling is what the app re-fits its layout from.
    A gate that also cleared those would change resize behaviour to fix a mouse
    bug, so they are pinned here as unchanged.
    """
    parser = XTermParser()
    tokens = list(parser.feed(IN_BAND_REPORT))
    resizes = [token for token in tokens if isinstance(token, events.Resize)]
    assert len(resizes) == 1
    assert resizes[0].size.width == 133
    assert resizes[0].size.height == 44
    assert parser.terminal_size == (133, 44)
    assert parser.terminal_pixel_size == (1064, 704)


# -- the re-clean: the object's own contract ----------------------------------


def test_the_reclaimer_writes_the_reset_through_its_sink() -> None:
    writes: list[str] = []
    reclaimer = InBandResizeReclaimer(writes.append)

    assert reclaimer.reclaim() is True
    assert writes == [RESET_PAIR] == ["\x1b[?2048l\x1b[?1016l"]


def test_the_reclaimer_honours_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LOCAL_OPERATOR_NO_MODE_RESET`` suppresses the write, read per call.

    Per call and not at construction: the boot reset's switch can be exported by
    whoever launched the app, but a switch set mid-session has to bite on the
    next resize or the two halves of this fix disagree about what it means.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_MODE_RESET", raising=False)
    writes: list[str] = []
    reclaimer = InBandResizeReclaimer(writes.append)

    assert reclaimer.reclaim() is True
    monkeypatch.setenv("LOCAL_OPERATOR_NO_MODE_RESET", "1")
    assert reclaimer.reclaim() is False
    assert writes == [RESET_PAIR]


def test_a_dead_writer_cannot_turn_a_resize_into_an_exception() -> None:
    """A writer thread that has already stopped must not break resizing.

    Same trade ``reset_in_band_resize`` makes for a detached stderr: the escape
    is cosmetic, the resize is not.
    """

    def dead_writer(_data: str) -> None:
        raise OSError("the writer thread is gone")

    assert InBandResizeReclaimer(dead_writer).reclaim() is False


# -- the app: which configuration builds a re-closer, and where it fires ------
#
# Driven through the REAL app and its REAL handlers, following `test_app_pilot`'s
# `_start_terminal_title` tests: `run_test` gives the app Textual's headless
# driver, so the terminal-dependent gates are opened with an `is_headless`
# override and the writes are read off the REAL driver through a spy that still
# forwards them. A test that called `reclaim()` itself would pass with the
# wiring connected to nothing.
#
# The gate is installed for exactly the arms that exercise the configuration
# where `run_tui` installs it. The "did not apply" arm is the opposite state of
# the same class the production predicate reads, not a stubbed predicate.


@pytest.mark.asyncio
async def test_the_app_re_closes_the_mode_on_resize_and_on_focus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gated: None
) -> None:
    """Both wiring points, on the app's own handlers, with the driver's writes.

    The reclaimer is rebuilt over the spy after mount for the reason the title
    tests rebuild theirs: it captured ``driver.write`` at mount time, so a spy
    installed afterwards would see nothing and the assertions would be about an
    object nobody called.
    """
    writes: list[str] = []
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._mode_reclaimer is not None, "mount with a terminal and the gate builds one"

        _spy_driver_writes(app, writes)
        app._mode_reclaimer = None
        app._start_mode_reclaimer()
        assert app._mode_reclaimer is not None

        app.post_message(events.Resize(app.size, app.size))
        await pilot.pause()
        assert writes.count(RESET_PAIR) == 1

        app.post_message(events.AppFocus())
        await pilot.pause()
        assert writes.count(RESET_PAIR) == 2


@pytest.mark.asyncio
async def test_a_headless_app_builds_no_re_closer(gated: None) -> None:
    """No driver, then a headless driver: neither has terminal state to close.

    Both halves matter. The first is the app that has not started (``exec`` and
    the headless REPL reach the boot with no driver at all), the second is the
    one ``run_test`` itself uses. Writing through either would put an escape on
    a wire nobody is reading, at best.
    """
    app = OperatorApp(lambda: _factory(FakeSession()))
    assert app._driver is None
    app._start_mode_reclaimer()
    assert app._mode_reclaimer is None

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._mode_reclaimer is None


@pytest.mark.asyncio
async def test_no_re_closer_when_the_negotiation_is_still_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configuration that still negotiates pixel mouse keeps its pointer.

    The gate is refused here (``SMOOTH_SCROLL`` on, non-iTerm — the state
    ``run_tui`` leaves a user who asked for pixel coordinates in), so nothing
    installed it and the app must build no re-closer: writing ``?2048l`` would
    switch the mode off underneath that user on every resize and focus gain,
    which is the exact behaviour this half exists to avoid.

    Both names the decision reads are pinned, because both freeze from the
    environment at ``textual`` import time and the developer's own terminal must
    not decide this test.
    """
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", True)
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", False)
    assert install_pixel_mouse_gate() is False, "the gate refuses while the mode is negotiated"
    assert pixel_mouse_gate_installed() is False
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert app._mode_reclaimer is None


@pytest.mark.asyncio
async def test_the_mid_session_reset_honours_the_kill_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, gated: None
) -> None:
    """``LOCAL_OPERATOR_NO_MODE_RESET`` suppresses the mid-session write too.

    The switch already suppressed the boot reset, so a reader running `lop`
    under something capturing raw terminal output expects no ``?2048l`` from us
    at any point in the session, not just the first one.
    """
    writes: list[str] = []
    _isolate_tui_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(OperatorApp, "is_headless", property(lambda self: False))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_MODE_RESET", "1")
    app = OperatorApp(lambda: _factory(FakeSession()))

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        _spy_driver_writes(app, writes)
        app._mode_reclaimer = None
        app._start_mode_reclaimer()
        assert app._mode_reclaimer is not None, "the switch suppresses the write, not the object"

        app.post_message(events.Resize(app.size, app.size))
        await pilot.pause()
        assert writes.count(RESET_PAIR) == 0


# -- the pty end-to-end: the real driver, a real app, a real pointer ----------
#
# Everything above drives our contract or a stub driver. This section is the
# evidence for the user-visible claim, and it is the only place the bug is
# reproduced end to end: a REAL Textual app on a REAL pty, whose input is the
# REAL driver's parser, dirtied mid-process by a report exactly as a co-tenant
# leaves behind — followed by mouse moves, asserting what the APP's pointer
# became.
#
# The environment is built from scratch rather than inherited. That is the same
# isolation the harness needs anyway, and it is also the CMUX_* guard: an
# inherited CMUX_WORKSPACE_ID lets a booted TUI rename workspaces in the
# operator's live multiplexer, so nothing here may depend on remembering to
# unset it. HOME and LOCAL_OPERATOR_CONFIG_DIR both point at the test's
# tmp_path, and no `lop` session exists in these children, so there is no
# session id to synthesise.

#: A bare Textual app whose whole job is to report what the pointer became.
#:
#: The two calls mirror `run_tui`'s order — the guard sets the environment
#: Textual reads, then (in the gated arm only) the parser gate is installed —
#: and the boot reset is deliberately absent: a pty implements no terminal modes
#: and answers no queries, so there is nothing for it to reset.
_POINTER_PROBE = """
import json, os, sys

from local_operator.tui.terminal_modes import guard_pixel_mouse_latch

# Boot-time reset, only for the arm that needs a wire for a compliant VT to
# observe (ARML_RESET unset keeps the other arms byte-identical to before).
# `pair` is what the fixed head's reset writes; `2048-only` is the PRE-FIX
# build's wire bytes, written literally because a build without the ?1016l half
# cannot be asked to produce them.
if os.environ.get("ARML_RESET") == "pair":
    from local_operator.tui.terminal_modes import reset_in_band_resize

    reset_in_band_resize()
elif os.environ.get("ARML_RESET") == "2048-only":
    sys.stderr.write("\\x1b[?2048l")
    sys.stderr.flush()

guard_pixel_mouse_latch()
if os.environ["ARML_GATE"] == "1":
    from local_operator.tui.terminal_modes import install_pixel_mouse_gate

    install_pixel_mouse_gate()

from textual.app import App

OUT = os.environ["ARML_LOG"]


class Probe(App):
    def on_mount(self) -> None:
        # Backstop only: the parent kills us. It stops a wedged child from
        # holding the pty open and turning a failure into a hang.
        self.set_timer(60.0, self.exit)

    def on_mouse_move(self, event) -> None:
        with open(OUT, "a") as fh:
            fh.write(
                json.dumps([event.x, event.y, self.mouse_position.x, self.mouse_position.y]) + "\\n"
            )


Probe().run()
"""

#: Two physical cells in the two scales a VT can report them in. Cell (5, 2) is
#: the reviewer's own example on review round 1; the wire is 1-based, so it is
#: (6, 3) as cells and (41, 33) as pixels (a measured 8x16 cell). Both cells are
#: in the top two rows ON PURPOSE: read as pixels they land at y 32 and 16 in a
#: 45-row screen, so the misread shows up as a WRONG POSITION on both moves —
#: one row lower and Textual swallows the event past the screen edge
#: (screen.py:1929-1931), and an arm whose second move vanished would be
#: measuring the swallow rather than the misread.
_VT_MOVES = {
    "cells": ("\x1b[<35;6;3M", "\x1b[<35;8;2M"),
    "pixels": ("\x1b[<35;41;33M", "\x1b[<35;57;17M"),
}

#: What the app records for those two cells when the VT reports them correctly,
#: and what it records when PIXEL numbers are read as cells (the pre-fix arm).
_VT_CELLS = [[5, 2, 5, 2], [7, 1, 7, 1]]
_VT_PIXELS_READ_AS_CELLS = [[40, 32, 40, 32], [56, 16, 56, 16]]

#: An app that wires the reclaimer the way the product does — built at mount,
#: fired from `on_resize` — with a spy between it and the driver so the test can
#: compare what we COUNTED with what actually reached the pty.
_RECLAIM_PROBE = """
import json, os

from local_operator.tui.terminal_modes import InBandResizeReclaimer

from textual.app import App

OUT = os.environ["ARML_LOG"]
# The pair the re-clean hands the writer, pinned as literal bytes rather than
# imported: this is the wire contract, and the test above compares what the
# child counted here against what the pty received. Both modes, because a
# co-tenant sets 1016 with 2048 (linux_driver.py:480-482) and it is 1016 that
# decides whether the numbers the terminal sends are cells.
RESET = "\\x1b[?2048l\\x1b[?1016l"


class Probe(App):
    def on_mount(self) -> None:
        self.writes = 0
        original = self._driver.write

        def spy(data: str) -> None:
            if data == RESET:
                self.writes += 1
            original(data)

        self._driver.write = spy
        # Built from the SPY, so a write that never reaches the driver is not
        # counted — the comparison below is only meaningful that way.
        self.reclaimer = InBandResizeReclaimer(self._driver.write)
        self.set_timer(60.0, self.exit)

    def on_resize(self, event) -> None:
        reclaimer = getattr(self, "reclaimer", None)
        if reclaimer is None:
            return
        reclaimer.reclaim()
        with open(OUT, "a") as fh:
            fh.write(json.dumps(self.writes) + "\\n")


Probe().run()
"""


class _PtyChild:
    """A child interpreter on its own pty, driven and read by the parent.

    ``fcntl``/``termios`` are imported in ``__init__`` rather than at module
    scope: they do not exist on Windows, and a module-level import would fail
    collection before this file's ``skipif`` could skip it, erroring the whole
    file. Repo precedent: ``test_pixel_mouse_latch.py:_capture_boot_bytes``.
    """

    def __init__(self, child_source: str, env_extra: dict[str, str], tmp_path: Path) -> None:
        import fcntl
        import struct
        import termios

        self._fcntl = fcntl
        self._struct = struct
        self._termios = termios
        self.log = tmp_path / (env_extra.get("ARML_TAG", "probe") + "-probe.jsonl")
        #: 45 rows x 160 columns: enough cells for the two probe positions, and
        #: a size the app can lay out in.
        master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 45, 160, 0, 0))

        env = {
            "TERM": "xterm-256color",
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(tmp_path),
            "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / ".local-operator"),
            "ARML_LOG": str(self.log),
            **env_extra,
        }

        self._master = master
        self._chunks: list[bytes] = []
        pid = os.fork()
        if pid == 0:  # pragma: no cover - replaced by execve
            try:
                os.setsid()
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
                os.dup2(slave, 0)
                os.dup2(slave, 1)
                os.dup2(slave, 2)
                if slave > 2:
                    os.close(slave)
                os.close(master)
                os.execve(sys.executable, [sys.executable, "-c", child_source], env)
            except BaseException:
                os._exit(127)
        os.close(slave)
        self._pid = pid

    def drain(self, seconds: float) -> None:
        """Read whatever the child wrote for ``seconds``."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            readable, _, _ = select.select([self._master], [], [], 0.05)
            if not readable:
                continue
            try:
                data = os.read(self._master, 65536)
            except OSError:
                # EIO: the slave side closed, i.e. the child is gone.
                return
            if not data:
                return
            self._chunks.append(data)

    def wait_for_output(self, timeout: float = 60.0) -> None:
        """Block until the child has painted something."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._chunks:
            self.drain(0.1)
        assert self._chunks, "the child never painted a frame; the pty harness is broken"

    def send(self, data: str) -> None:
        os.write(self._master, data.encode())

    def resize(self, rows: int, columns: int) -> None:
        """Change the pty's window size, which SIGWINCHes the child."""
        self._fcntl.ioctl(
            self._master, self._termios.TIOCSWINSZ, self._struct.pack("HHHH", rows, columns, 0, 0)
        )

    def lines(self) -> list[Any]:
        """The child's JSONL log, which is how it reports what it observed."""
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def wait_for_lines(self, count: int, timeout: float = 60.0) -> list[Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            lines = self.lines()
            if len(lines) >= count:
                return lines
            self.drain(0.1)
        raise AssertionError(
            f"the child logged {len(self.lines())} of {count} expected events; "
            f"captured {len(b''.join(self._chunks))} bytes of terminal output"
        )

    def output(self) -> bytes:
        self.drain(0.3)
        return b"".join(self._chunks)

    def close(self) -> None:
        try:
            os.kill(self._pid, 9)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(self._pid, 0)
        except ChildProcessError:
            pass
        os.close(self._master)


def _pointer_positions(tmp_path: Path, *, gate: bool) -> list[list[int]]:
    """Run one arm of the pointer probe and return what the app saw.

    The report is sent BETWEEN the two moves, which is the whole point: it
    simulates the resize that reveals a mode a co-tenant enabled — the state a
    boot-time reset cannot see, arriving while the app is running.
    """
    child = _PtyChild(
        _POINTER_PROBE,
        {"ARML_GATE": "1" if gate else "0", "ARML_TAG": "gated" if gate else "plain"},
        tmp_path,
    )
    try:
        child.wait_for_output()
        # Past the first frame by a comfortable margin: a mouse event delivered
        # before the app has a screen to route it through is not evidence.
        child.drain(0.7)

        child.send(MOUSE_MOVE_1)
        child.wait_for_lines(1)

        child.send(IN_BAND_REPORT)  # the co-tenant's dirty mode, mid-process
        child.drain(0.4)

        child.send(MOUSE_MOVE_2)
        return child.wait_for_lines(2)
    finally:
        child.close()


@pytest.mark.skipif(sys.platform == "win32", reason="pty semantics are POSIX-only")
def test_a_dirty_mode_mid_session_does_not_move_the_pointer(tmp_path: Path) -> None:
    """The reported bug, end to end, and the gate that stops it.

    Both arms run in ONE test on purpose. The control arm is the same harness,
    process shape and timings with the gate not installed, and it must collapse
    the second position to ``(7, 1)``: without it, a green gated arm could mean
    "this harness never latched anything" rather than "the gate held". It is
    also the arm that fails on a tree without the gate, which is what makes the
    pre-fix failure recorded on the PR a real measurement rather than a claim.

    Measured here (textual 8.2.8, macOS, 45x160 pty):
        gate OFF: [[40, 44, 40, 44], [7, 1, 7, 1]]
        gate ON : [[40, 44, 40, 44], [60, 24, 60, 24]]
    """
    plain = _pointer_positions(tmp_path, gate=False)
    gated = _pointer_positions(tmp_path, gate=True)

    # Control first: it is the assertion that makes the one below evidence.
    assert plain == [
        [TRUE_FIRST_POSITION[0], TRUE_FIRST_POSITION[1], *TRUE_FIRST_POSITION],
        [COLLAPSED_SECOND_POSITION[0], COLLAPSED_SECOND_POSITION[1], *COLLAPSED_SECOND_POSITION],
    ], f"the harness did not reproduce the collapse without the gate: {plain}"

    assert gated == [
        [TRUE_FIRST_POSITION[0], TRUE_FIRST_POSITION[1], *TRUE_FIRST_POSITION],
        [TRUE_SECOND_POSITION[0], TRUE_SECOND_POSITION[1], *TRUE_SECOND_POSITION],
    ], f"the pointer did not survive the dirty mode with the gate installed: {gated}"


def _compliant_vt_positions(tmp_path: Path, *, reset: str) -> tuple[str, list[Any]]:
    """One arm of the compliant-VT model: the scale the VT used, and what the app saw.

    THE PARENT IS THE MODEL OF THE VT, and it behaves like one that honours
    Textual's mode 1016: it reads the CHILD's own wire output and switches to
    reporting coordinates in CELLS once it has seen our ``CSI ?1016l``. That is
    the mechanism under test rather than an account of it — with the 1016 half
    on the wire the VT switches, and on a build that never sends it the VT keeps
    reporting pixels, which is the arm review round 1 (MAJOR-1) measured against
    this harness with its own scratch probe.

    ``reset`` selects what the child puts on the wire at boot: ``"pair"`` is
    what the fixed head's ``reset_in_band_resize()`` writes, ``"2048-only"`` is
    the PRE-FIX build's bytes. The scale the VT then chose is returned next to
    the child's log, so the test can assert the arm took the path it claims to
    rather than inferring it from the coordinates.
    """
    child = _PtyChild(
        _POINTER_PROBE,
        {"ARML_GATE": "1", "ARML_RESET": reset, "ARML_TAG": f"vt-{reset}"},
        tmp_path,
    )
    try:
        child.wait_for_output()
        child.drain(0.7)
        # The VT's own decision, taken from our wire exactly as a terminal takes
        # it — and taken BEFORE any mouse byte is sent, or the arm would be
        # measuring the wrong scale.
        scale = "cells" if b"\x1b[?1016l" in child.output() else "pixels"
        first, second = _VT_MOVES[scale]

        child.send(first)
        child.wait_for_lines(1)

        child.send(IN_BAND_REPORT)  # the co-tenant's dirty mode, mid-process
        child.drain(0.4)

        child.send(second)
        return scale, child.wait_for_lines(2)
    finally:
        child.close()


@pytest.mark.skipif(sys.platform == "win32", reason="pty semantics are POSIX-only")
def test_a_compliant_vt_reports_cells_once_we_clear_1016(tmp_path: Path) -> None:
    """The VT class the divisor-only gate inverted, and the reset pair that serves it.

    Why this arm exists: clearing Textual's divisor is only correct where the
    coordinates really are cells. A VT that honours mode 1016 sends genuine
    PIXELS, so pre-fix Textual's division was CORRECT there and a gate that only
    cleared the divisor read pixel numbers as cells — cell (5, 2) arriving as the
    wire's (41, 33) and being read as (40, 32), with anything past the screen
    edge swallowed outright (review round 1, MAJOR-1). The fix asserts
    ``?1016l`` beside ``?2048l``, so that VT is told to report cells instead.

    Both arms run in ONE test, as the sibling pty test does: the control is the
    same harness, process shape, gate and timings with the PRE-FIX build's wire
    bytes (``?2048l`` and nothing else), and it must reproduce the misread. The
    child is not told which scale to use — the parent decides it from what the
    child actually put on the wire — so the fixed arm's expectation is what
    fails on a build that lacks the 1016 half, which is the pre-fix measurement
    recorded on the PR.

    Measured here (textual 8.2.8, macOS, 45x160 pty, 8x16 cell):
        reset pair (fixed head): VT switches to cells -> [[5, 2, 5, 2], [7, 1, 7, 1]]
        ?2048l only (pre-fix)  : VT stays in pixels -> pixel numbers read as cells
                                 -> [[40, 32, 40, 32], [56, 16, 56, 16]]
    """
    fixed_scale, fixed = _compliant_vt_positions(tmp_path, reset="pair")
    prefix_scale, prefix = _compliant_vt_positions(tmp_path, reset="2048-only")

    assert (
        fixed_scale == "cells"
    ), "the fixed arm's VT never saw a ?1016l, so the pair is not on the wire at boot"
    assert fixed == _VT_CELLS, f"a compliant VT's cells were not read as cells: {fixed}"

    assert prefix_scale == "pixels", "the control arm's VT switched without a reset to observe"
    assert (
        prefix == _VT_PIXELS_READ_AS_CELLS
    ), f"the pre-fix arm did not reproduce the misread this fix removes: {prefix}"


@pytest.mark.skipif(sys.platform == "win32", reason="pty semantics are POSIX-only")
def test_a_resize_re_closes_the_mode_through_the_real_driver(tmp_path: Path) -> None:
    """One 16-byte write per delivered Resize, and it reaches the wire.

    This is the `B` half's real-path evidence and the measurement behind the
    "does not coalesce" decision in ``terminal_modes``: every SIGWINCH-derived
    Resize produces exactly one re-clean through the driver, so the rate bound
    is the terminal's SIGWINCH rate and not something larger. The comparison of
    the child's own count with the bytes the pty received is what proves the
    write went through the driver's serialised writer rather than into a queue
    that dropped it.

    The read of the delivery list POLLS (review round 1, MINOR-2): reading
    ``child.lines()`` once after three fixed drains made this test fail 1 of 7
    back-to-back runs with ``assert []`` on a loaded host — the child had
    painted and been resized but had not logged yet — where its sibling polls
    and is stable. The number of deliveries is still deliberately not pinned at
    three (see below); only the wait for at least two is.
    """
    child = _PtyChild(_RECLAIM_PROBE, {"ARML_TAG": "reclaim"}, tmp_path)
    try:
        child.wait_for_output()
        child.drain(0.7)
        baseline = child.lines()
        assert baseline, "the app never reported a resize; the probe is broken"
        start = baseline[-1]

        for step in range(1, 4):
            child.resize(45 + step, 160)
            child.drain(0.8)

        # Each LOGGED line is one delivered Resize, and the claim under test is
        # the per-event bound: exactly one write for each of them. The NUMBER of
        # events is deliberately not pinned at three — a resize storm can be
        # coalesced by the OS before the child ever sees a signal, so asserting
        # against that would make this test about the kernel rather than about
        # the re-clean. At least two of the three land even so.
        # POLL rather than read once: a loaded host can deliver the resize and
        # paint the frame before the child's own log write lands.
        delivered = child.wait_for_lines(len(baseline) + 2)[len(baseline) :]
        assert delivered, f"no resize reached the child after the booting ones: {baseline}"
        assert len(delivered) >= 2, f"only one delivery of three resizes: {delivered}"
        counts = [start] + delivered
        assert all(
            after - before == 1 for before, after in zip(counts, counts[1:])
        ), "more than one re-close per delivered Resize: " + repr(delivered)
        # The pair, contiguously: this is what proves the ?1016l half reached the
        # wire on the same write as ?2048l rather than being counted and dropped.
        assert child.output().count(RESET_PAIR.encode()) >= counts[-1]
    finally:
        child.close()
