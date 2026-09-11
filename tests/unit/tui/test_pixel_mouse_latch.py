"""The mouse-coordinate collapse after a resize, and the two halves that stop it.

These tests drive the REAL installed ``XTermParser`` rather than a model of it.
The behaviour under test is Textual's, not ours: a fake parser would assert that
our understanding is self-consistent, which is exactly the thing that was wrong
before this fix existed.

THE FIX HAS TWO HALVES AND EACH HAS A TEST PROVING THE OTHER DOES NOT COVER IT.
``test_env_guard_alone_does_not_stop_a_delivered_report`` kills the
"just set ``TEXTUAL_SMOOTH_SCROLL=0``" approach: the report arrives unsolicited
from the terminal and latches the divisor regardless of what we negotiated, so
the mode reset is load-bearing. ``test_reply_two_would_re_enable_the_mode``
pins the converse: a bare reset leaves the mode supported-but-disabled, and
that is precisely the reply Textual's driver answers by turning the mode back
on, so the environment guard is load-bearing too.

``parser.feed(...)`` is a GENERATOR. Every call here is drained with ``list()``;
a test that forgets this feeds nothing to the parser and asserts nothing.

Parser-side tests set ``constants.SMOOTH_SCROLL`` and ``IS_ITERM`` through
``monkeypatch`` attribute patching, never by mutating the environment
in-process: both freeze from the environment at ``textual`` import time —
``SMOOTH_SCROLL`` is a ``Final`` read once in ``textual.constants``, and
``IS_ITERM`` (``_xterm_parser.py:49-52``) is a module global read from
``LC_TERMINAL``/``TERM_PROGRAM`` — so an env write here would be read by
nothing and would leak into sibling suites. Every env-frozen term the gate
under test reads has to be pinned, or the developer's terminal decides the
result.

OUT OF SCOPE, deliberately: nothing here asserts behaviour under a terminal that
genuinely honours mode 1016 and sends true pixel coordinates. On such a terminal
the divisor is correct and this fix trades it away for a cell-accurate pointer;
that trade is a product decision recorded in ``terminal_modes``, not a property
this file tests.

All references pinned to textual 8.2.8.
"""

from __future__ import annotations

import os
import select
import sys
import time
from pathlib import Path

import pytest
from textual import constants, events, messages
from textual._xterm_parser import XTermParser

from local_operator.tui.terminal_modes import (
    DISABLE_IN_BAND_RESIZE,
    guard_pixel_mouse_latch,
    reset_in_band_resize,
)

#: An SGR mouse move at cell (40, 44) — the wire is 1-based, the event 0-based.
MOUSE_MOVE = "\x1b[<35;41;45M"

#: ``CSI 48;rows;cols;pxH;pxW t`` for a 44x133 cell frame measuring 1064x704
#: pixels: a 8x16 cell. Herdr-class terminals send this UNSOLICITED on resize.
IN_BAND_REPORT = "\x1b[48;44;133;704;1064t"

#: The reply to ``CSI ?2048$p`` that a terminal gives once the mode is reset:
#: supported, currently off. ``linux_driver.py:470-483`` answers it by
#: re-enabling the mode.
MODE_REPLY_SUPPORTED_BUT_RESET = "\x1b[?2048;2$y"

#: Where the divisor puts a (40, 44) pointer once it has latched.
COLLAPSED = (5, 2)

#: Where the pointer actually is.
TRUE_CELL = (40, 44)


def _positions(parser: XTermParser, data: str) -> list[tuple[int, int]]:
    """Feed ``data`` and return the (x, y) of every mouse event it yields."""
    return [
        (event.x, event.y)
        for event in list(parser.feed(data))
        if isinstance(event, events.MouseEvent)
    ]


def test_in_band_report_latches_pixel_mouse_coordinates() -> None:
    """The bug itself: one resize report and every later position collapses.

    Green before AND after the fix — it pins Textual's behaviour, which we do
    not change. What the fix changes is whether the report ever arrives.
    """
    parser = XTermParser()

    assert _positions(parser, MOUSE_MOVE) == [TRUE_CELL]

    resize_tokens = [
        token for token in list(parser.feed(IN_BAND_REPORT)) if isinstance(token, events.Resize)
    ]
    assert len(resize_tokens) == 1

    assert parser.mouse_pixels is True
    assert _positions(parser, MOUSE_MOVE) == [COLLAPSED]


def test_env_guard_alone_does_not_stop_a_delivered_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env-guard-alone approach does not defend against an UNSOLICITED report.

    With smooth scrolling off Textual never negotiates mode 2048 — but a
    Herdr-class terminal sends the report anyway, and the latch at
    ``_xterm_parser.py:271-283`` checks nothing before setting ``mouse_pixels``.
    This is why ``reset_in_band_resize`` exists.
    """
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)
    parser = XTermParser()

    assert _positions(parser, MOUSE_MOVE) == [TRUE_CELL]
    list(parser.feed(IN_BAND_REPORT))

    assert parser.mouse_pixels is True
    assert _positions(parser, MOUSE_MOVE) == [COLLAPSED]


def test_guard_suppresses_in_band_negotiation(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the guard set, the mode reply produces no negotiation token at all.

    The gate is ``_xterm_parser.py:321``. No token means the driver's
    re-enable branch never runs, which is the half of the fix the reset
    cannot provide.

    The CONTROL arm asserts the gate is open when we do not close it, so every
    other input to that gate has to be pinned or the developer's terminal
    decides the result. The gate reads ``constants.SMOOTH_SCROLL`` AND
    ``not IS_ITERM``, and ``IS_ITERM`` freezes from ``LC_TERMINAL``/
    ``TERM_PROGRAM`` at import — iTerm2 exports ``LC_TERMINAL`` to everything
    it spawns, so without this pin the control arm false-reds for any
    contributor running the suite from iTerm2.
    """
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", False)

    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)
    guarded = list(XTermParser().feed(MODE_REPLY_SUPPORTED_BUT_RESET))
    assert [t for t in guarded if isinstance(t, messages.InBandWindowResize)] == []

    # Control: the same bytes DO negotiate when the guard is absent, so the
    # assertion above is about the guard and not about malformed input.
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", True)
    unguarded = list(XTermParser().feed(MODE_REPLY_SUPPORTED_BUT_RESET))
    assert [t for t in unguarded if isinstance(t, messages.InBandWindowResize)] != []


def test_reply_two_would_re_enable_the_mode() -> None:
    """A bare reset produces exactly the reply that makes Textual undo it.

    ``supported=True, enabled=False`` is the combination
    ``linux_driver.py:480-482`` answers with ``?2048h`` followed by ``?1016h``.
    """
    message = messages.InBandWindowResize.from_setting_parameter(2)
    assert (message.supported, message.enabled) == (True, False)


def test_cell_coordinates_stay_stable_when_no_report_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed world: no report, no latch, and the pointer stays where it is."""
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", False)
    parser = XTermParser()

    assert _positions(parser, MOUSE_MOVE) == [TRUE_CELL]
    assert _positions(parser, MOUSE_MOVE) == [TRUE_CELL]
    assert parser.mouse_pixels is False


def test_guard_sets_the_textual_switch() -> None:
    """A pure-dict call: the guard installs the variable and says it did."""
    env: dict[str, str] = {}
    assert guard_pixel_mouse_latch(env) is True
    assert env == {"TEXTUAL_SMOOTH_SCROLL": "0"}


def test_guard_respects_an_explicit_user_setting() -> None:
    """A user who set the variable by hand keeps their value."""
    env = {"TEXTUAL_SMOOTH_SCROLL": "1"}
    assert guard_pixel_mouse_latch(env) is False
    assert env == {"TEXTUAL_SMOOTH_SCROLL": "1"}


class _FakeStream:
    """Minimal stand-in for ``sys.__stderr__`` with a scriptable ``isatty``."""

    def __init__(self, *, tty: bool, raise_on_write: BaseException | None = None) -> None:
        self._tty = tty
        self._raise_on_write = raise_on_write
        self.written = ""
        self.flushes = 0

    def isatty(self) -> bool:
        return self._tty

    def write(self, data: str) -> int:
        if self._raise_on_write is not None:
            raise self._raise_on_write
        self.written += data
        return len(data)

    def flush(self) -> None:
        self.flushes += 1


def test_reset_writes_only_to_a_tty() -> None:
    """Nothing is written to a pipe; a tty gets exactly the reset; errors are swallowed."""
    piped = _FakeStream(tty=False)
    assert reset_in_band_resize(piped) is False  # type: ignore[arg-type]
    assert piped.written == ""

    tty = _FakeStream(tty=True)
    assert reset_in_band_resize(tty) is True  # type: ignore[arg-type]
    assert tty.written == DISABLE_IN_BAND_RESIZE == "\x1b[?2048l"
    assert tty.flushes == 1

    # A closed or detached stderr must never be what stops the app booting.
    broken = _FakeStream(tty=True, raise_on_write=OSError("closed"))
    assert reset_in_band_resize(broken) is False  # type: ignore[arg-type]


def test_reset_honours_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """``LOCAL_OPERATOR_NO_MODE_RESET`` suppresses the write entirely."""
    monkeypatch.setenv("LOCAL_OPERATOR_NO_MODE_RESET", "1")
    tty = _FakeStream(tty=True)
    assert reset_in_band_resize(tty) is False  # type: ignore[arg-type]
    assert tty.written == ""


_REPO_ROOT = Path(__file__).resolve().parents[3]

#: What ``run_tui`` does, reduced to the part this test is about: the two calls,
#: in wire order, and then a Textual app that boots far enough for the driver to
#: run its startup negotiation before exiting itself.
_BOOT_CHILD = """
import sys
sys.path.insert(0, {repo!r})
from local_operator.tui.terminal_modes import (
    guard_pixel_mouse_latch,
    reset_in_band_resize,
)

reset_in_band_resize()
guard_pixel_mouse_latch()

from textual.app import App


class _Boot(App):
    def on_mount(self) -> None:
        self.set_timer(0.1, self.exit)


_Boot().run()
"""


def _capture_boot_bytes(child_source: str, timeout: float = 30.0) -> bytes:
    """Run ``child_source`` on a real pty and return every byte it wrote.

    A pty is not optional here: the driver only negotiates when stdin is a
    terminal, and ``reset_in_band_resize`` deliberately writes nothing when it
    is not. Ordering between our write and Textual's query is a property of the
    wire, so it is asserted on captured bytes and nowhere else.

    ``fcntl``/``termios`` are imported HERE rather than at module scope: they do
    not exist on Windows, and a module-level import would fail collection before
    this test's ``skipif`` could skip it, erroring the whole file. Repo
    precedent for function-level POSIX imports: ``test_teams.py:1652``.
    """
    import fcntl
    import struct
    import termios

    master_fd, slave_fd = os.openpty()
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 45, 160, 0, 0))

    env = {
        "TERM": "xterm-256color",
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }

    pid = os.fork()
    if pid == 0:  # pragma: no cover - replaced by execve
        try:
            os.setsid()
            fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)
            os.dup2(slave_fd, 0)
            os.dup2(slave_fd, 1)
            os.dup2(slave_fd, 2)
            if slave_fd > 2:
                os.close(slave_fd)
            os.close(master_fd)
            os.execve(sys.executable, [sys.executable, "-c", child_source], env)
        except BaseException:
            os._exit(127)

    os.close(slave_fd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.25)
            if not readable:
                continue
            try:
                data = os.read(master_fd, 65536)
            except OSError:
                # EIO: the slave side closed, i.e. the child is gone.
                break
            if not data:
                break
            chunks.append(data)
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        os.close(master_fd)

    return b"".join(chunks)


@pytest.mark.skipif(sys.platform == "win32", reason="pty semantics are POSIX-only")
def test_startup_writes_the_reset_before_the_in_band_query() -> None:
    """On a real pty the reset precedes the query, and 1016 is never enabled.

    This is the only test that can fail for an ORDERING mistake. The two calls
    could both be present and still be useless if they landed after
    ``linux_driver.py:299`` had already asked the terminal about mode 2048.
    """
    data = _capture_boot_bytes(_BOOT_CHILD.format(repo=str(_REPO_ROOT)))

    query = b"\x1b[?2048$p"
    assert query in data, (
        "the driver never queried mode 2048, so this run proves nothing about "
        f"ordering; captured {len(data)} bytes: {data[:400]!r}"
    )

    reset = DISABLE_IN_BAND_RESIZE.encode()
    assert reset in data, f"the reset never reached the wire; captured: {data[:400]!r}"

    reset_at = data.index(reset)
    query_at = data.index(query)
    assert reset_at < query_at, (
        f"the reset landed at byte {reset_at}, after the driver's query at {query_at}: "
        "the terminal was asked before it was told"
    )

    # NOT a discriminating assertion: on a bare pty nothing answers the
    # `?2048$p` query, so the re-enable branch (linux_driver.py:470-483) never
    # runs and this holds with or without the guard. It is kept as a standing
    # guard against a future Textual putting `?1016h` on the wire
    # unconditionally at startup. The discriminating proof that the guard
    # closes the negotiation is `test_guard_suppresses_in_band_negotiation`.
    assert b"\x1b[?1016h" not in data


def test_importing_the_tui_package_does_not_pull_textual() -> None:
    """The guard is inert unless it runs before ``textual.constants`` is imported.

    ``SMOOTH_SCROLL`` is read once, at ``textual.constants`` import time, so a
    module-scope Textual import ANYWHERE in the ``local_operator.tui`` import
    graph would freeze the constant before ``run_tui`` ever calls the guard —
    silently disarming the fix while this suite stayed green (test 10's child
    is hermetic, and the guard's presence is indistinguishable on a bare pty).
    A subprocess is required: this suite has already imported Textual, so the
    question can only be asked in a fresh interpreter.

    Mirrors ``test_cli_resume_guard.py::test_startup_import_weight_unchanged``,
    which guards the same invariant for the ``cli`` half.
    """
    import subprocess

    code = (
        "import sys, local_operator.tui; "
        "bad = [m for m in sys.modules if m.startswith('textual')]; "
        "print('LEAKED:' + ','.join(bad) if bad else 'CLEAN')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT
    )
    assert out.stdout.strip() == "CLEAN", out.stdout
