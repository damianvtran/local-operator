"""The mouse-coordinate collapse after a resize, and the two halves that stop it.

These tests drive the REAL installed ``XTermParser`` rather than a model of it.
The behaviour under test is Textual's, not ours: a fake parser would assert that
our understanding is self-consistent, which is exactly the thing that was wrong
before this fix existed.

THE FIX HAS TWO HALVES AND EACH HAS A TEST PROVING THE OTHER DOES NOT COVER IT.
``test_env_guard_alone_does_not_stop_a_delivered_report`` kills the
"just set ``TEXTUAL_SMOOTH_SCROLL=0``" approach: a report that reaches the
parser latches the divisor regardless of what we negotiated, and mode 2048 is
sticky per-VT, so a VT left dirty by an earlier app delivers one to a process
that never asked — which is why the mode reset is load-bearing.
``test_reply_two_would_re_enable_the_mode`` pins the converse: a bare reset
leaves the mode supported-but-disabled, and that is precisely the reply
Textual's driver answers by turning the mode back on, so the environment guard
is load-bearing too.

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

OUT OF SCOPE, deliberately: nothing DIRECTLY here asserts behaviour under a
terminal that genuinely honours mode 1016 and sends true pixel coordinates. That
arm is modelled instead in ``test_pixel_mouse_gate.py`` (the compliant-VT pty
arm), because it needs a pty that switches scale on observing our write — and it
is now a supported arm rather than a cost: the fix clears ``?1016l`` alongside
``?2048l``, so a compliant VT is told to report cells and the pointer stays true
there as well. The three configurations and why the pair of resets collapses
them to one are recorded in ``terminal_modes``.

All references pinned to textual 8.2.8; ``_xterm_parser.py`` is
``textual/_xterm_parser.py``, and ``drivers/linux_driver.py`` is
textual 8.2.8's ``textual/drivers/linux_driver.py`` (a bare ``linux_driver.py``
resolves to nothing, so the prefix is what makes it greppable).
"""

from __future__ import annotations

import os
import select
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from textual import constants, events, messages
from textual._xterm_parser import XTermParser

from local_operator.tui.terminal_modes import (
    DISABLE_IN_BAND_RESIZE,
    DISABLE_PIXEL_SCALE_MODES,
    guard_pixel_mouse_latch,
    pixel_mouse_negotiation_open,
    reset_in_band_resize,
)

#: An SGR mouse move at cell (40, 44) — the wire is 1-based, the event 0-based.
MOUSE_MOVE = "\x1b[<35;41;45M"

#: ``CSI 48;rows;cols;pxH;pxW t`` for a 44x133 cell frame measuring 1064x704
#: pixels: a 8x16 cell. A terminal sends this on resize once mode 2048 is set —
#: by this process, or by an earlier one, since the mode is sticky per-VT.
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
    """The env-guard-alone approach does not defend against a DELIVERED report.

    With smooth scrolling off Textual never negotiates mode 2048 — but the latch
    at ``_xterm_parser.py:271-283`` checks nothing before setting
    ``mouse_pixels``, so it does not matter who asked. Mode 2048 is sticky
    per-VT: an earlier app can leave it set and the reports arrive anyway. This
    is why ``reset_in_band_resize`` exists.
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


@pytest.mark.parametrize(
    ("smooth_scroll", "is_iterm"),
    [
        pytest.param(True, False, id="smooth-scrolling-on"),
        pytest.param(True, True, id="smooth-scrolling-on-iterm"),
        pytest.param(False, False, id="smooth-scrolling-off"),
        pytest.param(False, True, id="smooth-scrolling-off-iterm"),
    ],
)
def test_the_negotiation_mirror_agrees_with_textuals_own_gate(
    monkeypatch: pytest.MonkeyPatch, smooth_scroll: bool, is_iterm: bool
) -> None:
    """The predicate the gate installs on, checked against the REAL branch.

    Both halves of Textual's mode-report gate are asserted here through the
    parser rather than through a table that restates the expression, because a
    restatement cannot disagree with itself (review round 1, MINOR-1). The
    iTerm rows are the ones that matter: a Textual release that drops the
    ``not IS_ITERM`` clause, or adds a second route to ``_enable_mouse_pixels``
    (one call site today, ``linux_driver.py:482``), would leave our copy
    silently wrong in the direction this fix exists to prevent — installing the
    gate, and clearing 1016, while ``?1016h`` is on the wire because we were the
    ones who asked for it.
    """
    monkeypatch.setattr(constants, "SMOOTH_SCROLL", smooth_scroll)
    monkeypatch.setattr("textual._xterm_parser.IS_ITERM", is_iterm)

    negotiated = [
        token
        for token in XTermParser().feed(MODE_REPLY_SUPPORTED_BUT_RESET)
        if isinstance(token, messages.InBandWindowResize)
    ]
    assert pixel_mouse_negotiation_open() is bool(negotiated), (
        "our mirror of Textual's mode-report gate disagrees with what the parser "
        f"does for SMOOTH_SCROLL={smooth_scroll}, IS_ITERM={is_iterm}"
    )


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
    """A user who set the variable to a value Textual honours keeps their value."""
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


def test_reset_writes_only_to_a_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is written to a pipe; a tty gets exactly the reset; errors are swallowed.

    The kill switch is cleared rather than scrubbed suite-wide: it is a switch
    naming no machine resource, so it belongs in this file's own setup the way
    ``test_herdr_reporter.py`` clears ``LOCAL_OPERATOR_NO_HERDR``. A developer
    who exports it would otherwise see this test alone fail.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_NO_MODE_RESET", raising=False)

    piped = _FakeStream(tty=False)
    assert reset_in_band_resize(piped) is False  # type: ignore[arg-type]
    assert piped.written == ""

    tty = _FakeStream(tty=True)
    assert reset_in_band_resize(tty) is True  # type: ignore[arg-type]
    # Both modes, as ONE write, in the driver's own re-enable order inverted:
    # `?2048l` for the report mode, `?1016l` for the pixel-mouse mode a
    # co-tenant sets with it. Asserted as literal bytes because the wire is the
    # contract — a constant that drifted would keep this green otherwise.
    assert tty.written == DISABLE_PIXEL_SCALE_MODES == "\x1b[?2048l\x1b[?1016l"
    assert DISABLE_IN_BAND_RESIZE == "\x1b[?2048l"
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
#:
#: Deliberately the CALLS and not ``run_tui`` (review round 3, MINOR 1): the
#: wiring is asserted separately, by
#: ``test_run_tui_calls_the_guard_before_importing_textual`` below, which calls
#: the real ``run_tui`` and goes red if either call is removed from it. Keeping
#: this child hermetic — its own calls, no session, no app — is what makes it a
#: wire-ordering test rather than a boot test.
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
    """On a real pty the resets precede the query, and 1016 is never enabled.

    This is the only test that can fail for an ORDERING mistake on the wire.
    The two calls could both be present and still be useless if they landed
    after ``drivers/linux_driver.py:299`` had already asked the terminal about
    mode 2048 — and the same is true of the ``?1016l`` half, which is why the
    PAIR (contiguous, one write) is what is located here rather than the 2048
    sequence alone.

    Scope, stated because it was overstated once (review round 3, MINOR 1):
    the child above re-implements the two calls, so this test pins the calls
    and their wire order — NOT the wiring. Deleting both calls from ``run_tui``
    leaves this file green; ``test_run_tui_calls_the_guard_before_importing
    _textual`` below is what fails.
    """
    data = _capture_boot_bytes(_BOOT_CHILD.format(repo=str(_REPO_ROOT)))

    query = b"\x1b[?2048$p"
    assert query in data, (
        "the driver never queried mode 2048, so this run proves nothing about "
        f"ordering; captured {len(data)} bytes: {data[:400]!r}"
    )

    pair = DISABLE_PIXEL_SCALE_MODES.encode()
    assert pair in data, f"the reset pair never reached the wire; captured: {data[:400]!r}"

    resets_at = data.index(pair)
    query_at = data.index(query)
    assert resets_at < query_at, (
        f"the resets landed at byte {resets_at}, after the driver's query at {query_at}: "
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


#: The env→constant hop, in the only place it can be asked: ``SMOOTH_SCROLL`` is
#: a ``Final`` frozen when ``textual.constants`` is imported. Production order
#: is mirrored exactly — guard first, Textual's import second. ``inherited``
#: arrives as JSON so ``None`` (absent) and ``""`` (present-but-empty) stay
#: distinguishable, the distinction the whole finding is about.
_HOP_CHILD = """
import json, os, sys

from local_operator.tui.terminal_modes import guard_pixel_mouse_latch

inherited = json.loads(sys.argv[1])
if inherited is not None:
    os.environ["TEXTUAL_SMOOTH_SCROLL"] = inherited

wrote = guard_pixel_mouse_latch()

import textual.constants as constants

print(json.dumps({"wrote": wrote, "smooth_scroll": constants.SMOOTH_SCROLL}))
"""

#: The five states an inherited ``TEXTUAL_SMOOTH_SCROLL`` can be in, and what
#: each must leave behind: ``(inherited value or None, the guard wrote?,
#: Textual's SMOOTH_SCROLL)``.
#:
#: The last two are the shapes a user writes when they believe this is a
#: boolean switch, and they are the ones that used to silently disarm the fix
#: (review round 3, MINOR 2): Textual's ``_get_environ_int`` returns its
#: default of 1 for both, i.e. smooth scrolling ON, so a guard that deferred on
#: mere presence handed the latch straight back.
_SMOOTH_SCROLL_SHAPES = [
    pytest.param(None, True, False, id="unset"),
    pytest.param("0", False, False, id="zero-int"),
    pytest.param("1", False, True, id="one-int"),
    pytest.param("", True, False, id="empty-string"),
    pytest.param("true", True, False, id="non-integer-word"),
]


@pytest.mark.parametrize(("inherited", "guard_wrote", "smooth_scroll"), _SMOOTH_SCROLL_SHAPES)
def test_guard_defers_only_to_a_value_textual_honours(
    inherited: str | None, guard_wrote: bool, smooth_scroll: bool
) -> None:
    """The guard's own contract over all five shapes.

    ``guard_wrote`` is the return value and ``smooth_scroll`` the value the
    environment must be left holding (``"1"`` only where the user's integer was
    honoured and asked for it). A presence check fails the last two rows, which
    is the point: this test bites if the guard ever goes back to
    ``if _SMOOTH_SCROLL_ENV in env``.
    """
    env: dict[str, str] = {} if inherited is None else {"TEXTUAL_SMOOTH_SCROLL": inherited}

    assert guard_pixel_mouse_latch(env) is guard_wrote

    expected_value = "1" if smooth_scroll else "0"
    assert env["TEXTUAL_SMOOTH_SCROLL"] == expected_value


@pytest.mark.parametrize(("inherited", "guard_wrote", "smooth_scroll"), _SMOOTH_SCROLL_SHAPES)
def test_the_guard_leaves_a_value_textual_actually_reads(
    inherited: str | None, guard_wrote: bool, smooth_scroll: bool, tmp_path: Path
) -> None:
    """The same five shapes through Textual's parse — the env-to-constant hop.

    Nothing checked this hop before: the guard's unit test asserts the string it
    writes, while ``constants.SMOOTH_SCROLL`` is a ``Final`` frozen at import,
    so this suite — which imported Textual long ago — cannot observe what
    Textual makes of it. A child per shape, guard first and ``textual.constants``
    second exactly as production orders them, is the only honest way to ask.
    """
    import json
    import subprocess

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / ".local-operator"),
    }
    out = subprocess.run(
        [sys.executable, "-c", _HOP_CHILD, json.dumps(inherited)],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr

    assert json.loads(out.stdout) == {"wrote": guard_wrote, "smooth_scroll": smooth_scroll}


#: The wiring itself, which the pty child above cannot see: it re-implements the
#: two calls, so deleting them from ``run_tui`` leaves this file green (review
#: round 3, MINOR 1). This child calls the REAL ``run_tui`` and stubs the lazy
#: ``local_operator.tui.app`` import to raise, so its failure lands immediately
#: after the three calls and nothing has to boot.
#:
#: The recorders replace the module attributes ``run_tui`` resolves at call
#: time, which is what makes this a test of the production path rather than of
#: the child: remove a call from ``run_tui`` and no recorder fires.
#:
#: ``guard`` and ``gate`` are WRAPPERS around the real functions rather than
#: recorders that return True: both decide for themselves what to do from the
#: frozen constants, so a stub's return value would make every arm below assert
#: the same thing.
_WIRING_CHILD = """
import asyncio, json, os, sys, types

import local_operator.tui as tui
from local_operator.tui.terminal_modes import pixel_mouse_gate_installed

calls = []
textual_loaded_at_call = None


def _record(name):
    global textual_loaded_at_call
    calls.append(name)
    if textual_loaded_at_call is None:
        textual_loaded_at_call = any(
            module == "textual" or module.startswith("textual.")
            for module in sys.modules
        )


def _recorder(name):
    def _call(*_args, **_kwargs):
        _record(name)
        return True

    return _call


_real_guard = tui.guard_pixel_mouse_latch


def _guard(*args, **kwargs):
    _record("guard")
    return _real_guard(*args, **kwargs)


# The REAL gate, wrapped rather than stubbed: its whole point is that it decides
# for itself whether it applies, and a stub returning True would make every
# shape below assert the same thing. The guard is wrapped for the same reason.
_real_gate = tui.install_pixel_mouse_gate


def _gate(*args, **kwargs):
    _record("gate")
    return _real_gate(*args, **kwargs)


tui.reset_in_band_resize = _recorder("reset")
tui.guard_pixel_mouse_latch = _guard
tui.install_pixel_mouse_gate = _gate


inherited = sys.argv[1]
if inherited:
    os.environ["TEXTUAL_SMOOTH_SCROLL"] = inherited


class _NoApp(types.ModuleType):
    def __getattr__(self, name):
        raise ImportError("stubbed by the wiring test: " + name)


sys.modules["local_operator.tui.app"] = _NoApp("local_operator.tui.app")

try:
    asyncio.run(tui.run_tui(lambda: None))
except ImportError:
    pass

print(
    json.dumps(
        {
            "calls": calls,
            "installed": pixel_mouse_gate_installed(),
            "textual_loaded_at_call": textual_loaded_at_call,
        }
    )
)
"""


def _run_wiring_child(tmp_path: Path, inherited: str) -> dict[str, Any]:
    """Run the real ``run_tui`` with ``TEXTUAL_SMOOTH_SCROLL`` as given.

    ``inherited`` is ``""`` for absent: the child needs to distinguish absent
    from an empty string, which is one of the shapes the guard deliberately
    treats as absent, so it is passed as an argv value rather than through the
    environment. The child's environment is built for it and carries no
    ``TEXTUAL_SMOOTH_SCROLL`` of its own, so the three shapes here are the three
    shapes under test — a developer whose shell exports ``0`` cannot make the
    absent row pass for the wrong reason.
    """
    import json
    import subprocess

    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / ".local-operator"),
    }
    out = subprocess.run(
        [sys.executable, "-c", _WIRING_CHILD, inherited],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_run_tui_calls_the_guard_before_importing_textual(tmp_path: Path) -> None:
    """The wiring, which the pty ordering test cannot see (review round 3, MINOR 1).

    ``_capture_boot_bytes``'s child re-implements the two calls, so both could be
    deleted from ``run_tui`` with the whole file green — a plausible refactor
    (moving them below the lazy import) would silently restore the reported bug.
    This calls the REAL ``run_tui`` and stubs only the ``local_operator.tui.app``
    import to raise, so the failure lands immediately after the calls: with any
    of them missing, or with the calls moved below the import, ``calls`` comes
    back short and this goes red.

    ``textual_loaded_at_call`` is the guard's precondition, asserted at the
    moment the first call ran rather than inferred afterwards: the guard is
    inert if ``textual.constants`` is already in ``sys.modules``.
    """
    result = _run_wiring_child(tmp_path, "")

    assert result["calls"] == ["reset", "guard", "gate"], result
    assert result["installed"] is True, result
    assert result["textual_loaded_at_call"] is False, result


def test_run_tui_installs_the_gate_for_an_inherited_zero(tmp_path: Path) -> None:
    """The recommended configuration must not lose the gate.

    ``TEXTUAL_SMOOTH_SCROLL=0`` is what the operator's own runtime exports, what
    a ``replace_self`` re-entry inherits, and what our documentation tells users
    to set. The guard DEFERS to it — it is an integer Textual honours — so this
    is the row that fails if the install is keyed on the guard's return value
    instead of on the negotiation, which is exactly what it used to do.
    """
    result = _run_wiring_child(tmp_path, "0")

    assert result["calls"] == ["reset", "guard", "gate"], result
    assert result["installed"] is True, result
    assert result["textual_loaded_at_call"] is False, result


def test_run_tui_installs_nothing_while_textual_negotiates_pixel_mouse(
    tmp_path: Path,
) -> None:
    """``TEXTUAL_SMOOTH_SCROLL=1`` on a non-iTerm terminal keeps upstream.

    That user's textual still negotiates pixel mouse, so a delivered report IS a
    statement about their coordinates and the divisor is correct: the gate must
    install nothing, and the re-clean must not fire (it is keyed on the gate
    being in force). The call is still MADE — the refusal lives inside it, so no
    caller can install the gate in a configuration where upstream behaviour is
    the right answer — which is why the sequence assertion below stays at three
    calls and the meaning is carried by ``installed``.
    """
    result = _run_wiring_child(tmp_path, "1")

    assert result["calls"] == ["reset", "guard", "gate"], result
    assert result["installed"] is False, result
    assert result["textual_loaded_at_call"] is False, result
